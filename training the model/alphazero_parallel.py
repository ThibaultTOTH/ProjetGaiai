"""
Parallel Batched AlphaZero Engine for Gaia Project Deep RL (SOTA Architecture 2).

Architecture:
- Central GPU Batch Server / Learner (Main Process on CUDA with RTX 5070).
- Multiple parallel CPU Worker Processes (4-10 workers, 1 per CPU core).
- Dynamic Batching of MCTS leaf evaluations across all workers into a single GPU forward pass.
- 100% binary and checkpoint-compatible with single-thread AlphaZeroTrainer (DualGaiaAgent).
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import time
import math
import random
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from config import AppConfig
from models import DualGaiaAgent
from mcts import MultiPlayerMCTS
from environment import make_gaia_env, NativeGaiaEnv, compute_competitive_value
from league import LeagueManager
from alphazero_trainer import AlphaZeroTrainer, AlphaZeroReplayBuffer, EpochMetrics


class MCTSRemoteAgentProxy:
    """Lightweight client proxy running inside CPU worker processes.

    Instead of loading heavy PyTorch weights in every worker process,
    this proxy routes root and leaf evaluation queries through an IPC pipe
    to the central GPU Batch Server, which evaluates them in large batched FP16 passes.
    """

    def __init__(self, worker_id: int, req_queue: Any, resp_pipe: Any, model_config: Any):
        self.worker_id = worker_id
        self.req_queue = req_queue
        self.resp_pipe = resp_pipe
        self.config = model_config
        self.device = torch.device("cpu")
        self.model_name: str = "CurrentPolicy"

    def parameters(self):
        return iter([])

    def evaluate_root(
        self,
        obs_np: np.ndarray,
        mask_np: np.ndarray,
        actor: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sends root prior evaluation request to central GPU server."""
        obs = np.asarray(obs_np, dtype=np.float32)
        mask = np.asarray(mask_np, dtype=bool)
        self.req_queue.put((self.worker_id, "root", obs, mask, actor, self.model_name))
        priors, raw_logits = self.resp_pipe.recv()
        return priors, raw_logits

    def evaluate_leaf_batch(
        self,
        batch_obs: Any,
        batch_mask: Any,
        leaf_actors: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sends candidate leaf evaluations to central GPU server for batching."""
        if isinstance(batch_obs, torch.Tensor):
            obs_np = batch_obs.detach().cpu().numpy()
        else:
            obs_np = np.asarray(batch_obs, dtype=np.float32)

        if isinstance(batch_mask, torch.Tensor):
            mask_np = batch_mask.detach().cpu().numpy()
        elif batch_mask is not None:
            mask_np = np.asarray(batch_mask, dtype=bool)
        else:
            mask_np = None

        if obs_np.ndim == 1:
            obs_np = obs_np[np.newaxis, :]
        if mask_np is not None and mask_np.ndim == 1:
            mask_np = mask_np[np.newaxis, :]

        self.req_queue.put((self.worker_id, "leaf_batch", obs_np, mask_np, leaf_actors, self.model_name))
        vals, priors = self.resp_pipe.recv()
        return vals, priors

    def predict_score_with_uncertainty(self, obs: Any, num_passes: int = 1) -> Tuple[float, float]:
        vals, _ = self.evaluate_leaf_batch(obs, None, None)
        return float(vals[0]), 0.0

    def evaluate_leaf(self, obs: Any, mask: Optional[Any] = None, leaf_actor: int = 0) -> Tuple[float, np.ndarray]:
        vals, priors = self.evaluate_leaf_batch(obs, mask, [leaf_actor])
        return float(vals[0]), priors[0]

    def predict_opponent_action(
        self,
        obs: Any,
        mask: Optional[Any] = None,
    ) -> torch.Tensor:
        """Sends opponent action prediction request to central GPU server."""
        obs_np = obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor) else np.asarray(obs, dtype=np.float32)
        mask_np = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else (np.asarray(mask, dtype=bool) if mask is not None else None)
        self.req_queue.put((self.worker_id, "opp_act", obs_np, mask_np, None))
        opp_probs_np = self.resp_pipe.recv()
        return torch.from_numpy(opp_probs_np)


def alpha_zero_worker_process(
    worker_id: int,
    app_cfg: AppConfig,
    req_queue: Any,
    resp_pipe: Any,
    cmd_pipe: Any,
    stop_event: Any,
):
    """Isolated CPU Worker process running Native Gaia Project environment and MCTS."""
    # Seed per worker
    base_seed = int(time.time() * 1000) % 100000 + worker_id * 1000
    np.random.seed(base_seed)
    random.seed(base_seed)
    torch.manual_seed(base_seed)

    env = make_gaia_env(players=app_cfg.model.num_players, seed=base_seed)
    proxy = MCTSRemoteAgentProxy(worker_id, req_queue, resp_pipe, app_cfg.model)
    mcts = MultiPlayerMCTS(proxy, app_cfg.mcts, device=None)

    while not stop_event.is_set():
        if cmd_pipe.poll(0.01):
            msg = cmd_pipe.recv()
            if msg == "STOP":
                break
            elif isinstance(msg, dict) and msg.get("cmd") == "PLAY_GAME":
                game_start_time = time.time()
                num_sims = msg.get("num_simulations", app_cfg.alphazero.num_simulations)
                temp_high = msg.get("temp_high", app_cfg.alphazero.temperature_high)
                temp_low = msg.get("temp_low", app_cfg.alphazero.temperature_low)
                temp_thresh = msg.get("temp_thresh", app_cfg.alphazero.temperature_threshold_move)
                game_seed = msg.get("seed", base_seed + random.randint(1, 1000000))
                current_epoch_num = msg.get("epoch", 1)

                try:
                    # Reset environment
                    obs, mask = env.reset(seed=game_seed)
                    num_players = getattr(env, "num_players", 4)
                    participants = msg.get("participants", ["CurrentPolicy"] * num_players)
                    factions_list = []
                    if hasattr(env, "set_player_faction"):
                        factions = msg.get("factions", None)
                        if factions is None or len(factions) < num_players:
                            factions = np.random.choice(14, num_players, replace=False)
                        for seat, faction_id in enumerate(factions):
                            env.set_player_faction(seat, int(faction_id))
                        factions_list = [int(f) for f in factions]

                    history = []
                    move_count = 0
                    total_game_steps = 0
                    MAX_STEPS = 3000

                    while not env.terminated and total_game_steps < MAX_STEPS:
                        if stop_event.is_set():
                            break

                        total_game_steps += 1
                        curr_player = env.current_player
                        p_name = participants[curr_player] if curr_player < len(participants) else "CurrentPolicy"
                        curr_obs = env._get_obs() if hasattr(env, "_get_obs") else env.observe().values
                        curr_mask = env.get_action_mask()

                        legal_indices = np.where(curr_mask)[0]
                        if len(legal_indices) == 0:
                            env.terminated = True
                            break

                        if p_name == "CurrentPolicy":
                            proxy.model_name = "CurrentPolicy"
                            temp = temp_high if move_count < temp_thresh else temp_low
                            action, probs, _ = mcts.search(
                                env,
                                num_simulations=num_sims,
                                temperature=temp,
                                add_noise=True,
                                root_obs=curr_obs,
                                root_mask=curr_mask,
                                epoch=current_epoch_num,
                            )
                            history.append((curr_player, curr_obs, curr_mask, probs))
                        elif p_name == "RandomBaseline":
                            action = int(np.random.choice(legal_indices))
                        else:
                            # Historical snapshot: full competitive tactical MCTS search (full 128 sims budget)
                            proxy.model_name = p_name
                            snap_sims = num_sims
                            action, _, _ = mcts.search(
                                env,
                                num_simulations=snap_sims,
                                temperature=0.1,
                                add_noise=False,
                                root_obs=curr_obs,
                                root_mask=curr_mask,
                                epoch=current_epoch_num,
                            )
                            proxy.model_name = "CurrentPolicy"

                        move_count += 1
                        step_res = env.step(action)
                        if hasattr(step_res, "info") and "error" in step_res.info:
                            alt_actions = np.random.permutation(legal_indices)
                            for alt_act in alt_actions:
                                if alt_act == action:
                                    continue
                                step_res = env.step(int(alt_act))
                                if not (hasattr(step_res, "info") and "error" in step_res.info):
                                    break

                    if hasattr(env, "get_all_vps"):
                        raw_vps = env.get_all_vps().tolist()
                    else:
                        raw_vps = [float(p.get("vp", 0.0)) for p in getattr(env, "players_state", [{"vp": 0.0}] * num_players)]
                    # Determine CurrentPolicy score and win status
                    cp_seats = [i for i, name in enumerate(participants) if name == "CurrentPolicy"]
                    max_vp = max(raw_vps) if raw_vps else 0.0
                    is_unique_win = (raw_vps.count(max_vp) == 1)
                    winner_seat = int(np.argmax(raw_vps)) if is_unique_win else -1

                    is_league_match = (len(cp_seats) == 1 and len(participants) > 1)
                    if is_league_match:
                        # 1v3 League match: track exact CurrentPolicy seat and whether it won the match
                        cp_seat = cp_seats[0]
                        logged_vp = float(raw_vps[cp_seat])
                        logged_won = bool(winner_seat == cp_seat)
                    elif len(cp_seats) > 0:
                        # Pure self-play: track average VP across seats
                        logged_vp = float(np.mean([raw_vps[s] for s in cp_seats]))
                        logged_won = bool(winner_seat in cp_seats)
                    else:
                        logged_vp = float(raw_vps[0]) if raw_vps else 50.0
                        logged_won = False

                    final_history = [
                        (obs_s, mask_s, probs_s, compute_competitive_value(raw_vps, p, epoch=current_epoch_num), p)
                        for p, obs_s, mask_s, probs_s in history
                    ]

                    game_duration = max(0.1, time.time() - game_start_time)

                    # Push completed game to central server
                    req_queue.put((
                        worker_id,
                        "game_done",
                        final_history,
                        logged_vp,
                        logged_won,
                        move_count,
                        participants,
                        raw_vps,
                        is_league_match,
                        factions_list,
                        game_duration,
                    ))

                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    print(f"\n[Worker {worker_id} Exception]: {e}\n{tb}", file=sys.stderr, flush=True)
                    req_queue.put((worker_id, "game_error", str(e), tb))


class ParallelAlphaZeroTrainer(AlphaZeroTrainer):
    """High-Throughput Parallel Batched AlphaZero Trainer.

    Spawns multiple CPU worker processes playing games simultaneously,
    coalescing leaf neural network evaluations into large batched forward passes
    on the dedicated GPU (e.g. RTX 5070 with AMP FP16).
    """

    def __init__(
        self,
        config: AppConfig,
        agent: Optional[DualGaiaAgent] = None,
        num_workers: Optional[int] = None,
    ):
        super().__init__(config, agent)

        # Determine optimal worker count
        cpu_count = os.cpu_count() or 4
        if num_workers is not None:
            self.num_workers = max(1, num_workers)
        else:
            # Leave 2 threads for GPU batching loop, OS, and learner
            self.num_workers = max(2, min(cpu_count - 2, 8))

        self.ctx = mp.get_context("spawn")
        self.req_queue = self.ctx.Queue(maxsize=256)
        self.resp_pipes_parent = []
        self.cmd_pipes_parent = []
        self.workers = []
        self.stop_event = self.ctx.Event()
        self._workers_running = False

    def _sample_game_participants(self) -> List[str]:
        """Samples participants for a 4-player game (CurrentPolicy vs League Snapshots)."""
        num_players = 4
        if (
            getattr(self.config.league, "enabled", True)
            and self.league_manager is not None
            and len(self.league_manager.historical_names) > 0
            and random.random() < 0.50  # 50% clean 1v3 league eval games, 50% pure self-play
        ):
            # True 1v3 League evaluation match: 1 CurrentPolicy seat vs 3 past snapshots / opponents
            opps = self.league_manager.sample_opponent_names(num_opponents=num_players - 1)
            clean_opps = []
            for opp in opps:
                if opp == "CurrentPolicy" and len(self.league_manager.historical_names) > 0:
                    clean_opps.append(str(np.random.choice(self.league_manager.historical_names)))
                else:
                    clean_opps.append(opp)
            participants = ["CurrentPolicy"] + clean_opps
            random.shuffle(participants)
            return participants
        return ["CurrentPolicy"] * num_players

    def _sample_game_factions(self) -> List[int]:
        """Samples 4 factions with stratified round-robin coverage across all 14 factions."""
        if not hasattr(self, "_faction_deck") or len(self._faction_deck) < 4:
            new_deck = list(range(14))
            random.shuffle(new_deck)
            if hasattr(self, "_faction_deck") and self._faction_deck:
                self._faction_deck.extend(new_deck)
            else:
                self._faction_deck = new_deck
        sampled = self._faction_deck[:4]
        self._faction_deck = self._faction_deck[4:]
        return sampled

    def _render_progress_bar(
        self,
        epoch: int,
        max_epochs: int,
        done_games: int,
        target_games: int,
        moves: int,
        elapsed: float,
        active_workers: int,
    ):
        """Displays in-place terminal progress bar and updates live status file."""
        pct = (done_games / target_games) * 100.0 if target_games > 0 else 0.0
        bar_len = 16
        filled = int(round(bar_len * (pct / 100.0)))
        filled = min(bar_len, max(0, filled))
        bar = "█" * filled + "░" * (bar_len - filled)
        speed = moves / max(1.0, elapsed)
        m_el = int(elapsed // 60)
        s_el = int(elapsed % 60)
        dur_str = f"{m_el}m{s_el:02d}s"

        line = f"\r  ⏳ [Ép {epoch:04d}/{max_epochs}] [{bar}] {pct:>3.0f}% ({done_games}/{target_games} p.) | {moves:>4d} cps | {speed:>4.1f} st/s | {dur_str} | {active_workers} w."
        sys.stdout.write(line)
        sys.stdout.flush()

        try:
            status_file = os.path.join(getattr(self.config.training, "runs_dir", "runs"), "current_status.txt")
            with open(status_file, "w", encoding="utf-8") as sf:
                sf.write(line.strip() + "\n")
        except Exception:
            pass

    def start_workers(self):
        """Spawns background CPU worker processes."""
        if self._workers_running:
            return

        self.stop_event.clear()
        self.resp_pipes_parent = []
        self.cmd_pipes_parent = []
        self.workers = []

        for wid in range(self.num_workers):
            p_resp, c_resp = self.ctx.Pipe(duplex=True)
            p_cmd, c_cmd = self.ctx.Pipe(duplex=True)

            w = self.ctx.Process(
                target=alpha_zero_worker_process,
                args=(
                    wid,
                    self.config,
                    self.req_queue,
                    c_resp,
                    c_cmd,
                    self.stop_event,
                ),
                daemon=True,
            )
            w.start()
            self.workers.append(w)
            self.resp_pipes_parent.append(p_resp)
            self.cmd_pipes_parent.append(p_cmd)

        self._workers_running = True
        print(f"  * [Parallel MCTS Engine] Initialized {self.num_workers} CPU worker processes with dynamic GPU batching.")

    def stop_workers(self):
        """Stops all background worker processes safely."""
        if not self._workers_running:
            return

        self.stop_event.set()
        for p in self.cmd_pipes_parent:
            try:
                p.send("STOP")
            except Exception:
                pass

        for w in self.workers:
            w.join(timeout=1.0)
            if w.is_alive():
                w.terminate()

        self.workers = []
        self.resp_pipes_parent = []
        self.cmd_pipes_parent = []
        self._workers_running = False

    def run_training_loop(self, env: Any, max_epochs: int, callback=None):
        """Runs the parallel batched self-play and training loop."""
        self._is_running = True
        self._stop_event.clear()
        self._pause_event.clear()

        self.start_workers()

        start_epoch = getattr(self, "current_epoch", 0) + 1
        target_games_per_epoch = int(getattr(self.az_config, "games_per_epoch", 8))

        # Continuous streaming: track worker busy state across epochs to eliminate straggler idle time
        worker_busy = [False] * self.num_workers
        total_games_dispatched = 0

        for epoch in range(start_epoch, max_epochs + 1):
            if self._stop_event.is_set():
                break

            self.current_epoch = epoch
            self.update_learning_rate()
            start_time = time.time()
            self.agent.eval()

            epoch_real_scores = []
            epoch_winner_scores = []
            epoch_wins = 0
            epoch_games = 0
            epoch_league_wins = 0
            epoch_league_games = 0
            epoch_moves = 0
            worker_error_count = 0
            last_heartbeat_time = time.time()

            # 1. Dispatch games to any currently IDLE workers
            for wid in range(self.num_workers):
                if not worker_busy[wid]:
                    parts = self._sample_game_participants()
                    self.cmd_pipes_parent[wid].send({
                        "cmd": "PLAY_GAME",
                        "num_simulations": self.az_config.num_simulations,
                        "temp_high": self.az_config.temperature_high,
                        "temp_low": self.az_config.temperature_low,
                        "temp_thresh": self.az_config.temperature_threshold_move,
                        "seed": int(time.time() * 1000) % 1000000 + epoch * 1000 + wid * 19 + total_games_dispatched,
                        "epoch": epoch,
                        "participants": parts,
                        "factions": self._sample_game_factions(),
                    })
                    worker_busy[wid] = True
                    total_games_dispatched += 1

            # 2. Central GPU Batch Server Loop
            idle_stall_count = 0
            while epoch_games < target_games_per_epoch:
                if self._stop_event.is_set():
                    break

                # Live in-place progress bar updated dynamically
                now = time.time()
                if now - last_heartbeat_time > 0.8:
                    last_heartbeat_time = now
                    active_w = sum(1 for b in worker_busy if b)
                    self._render_progress_bar(
                        epoch, max_epochs, epoch_games, target_games_per_epoch, epoch_moves, now - start_time, active_w
                    )

                # Collect batched evaluation requests from workers
                eval_requests = []
                try:
                    first_req = self.req_queue.get(timeout=0.05)
                    eval_requests.append(first_req)
                    idle_stall_count = 0
                    # Non-blocking burst drain of all waiting worker requests
                    max_batch_requests = min(128, max(8, self.num_workers * 4))
                    while len(eval_requests) < max_batch_requests:
                        try:
                            eval_requests.append(self.req_queue.get_nowait())
                        except Exception:
                            break
                except Exception:
                    idle_stall_count += 1
                    active_w = sum(1 for b in worker_busy if b)
                    if idle_stall_count > 60 and epoch_games > 0 and active_w == 0:
                        print(f"  ⚡ [Epoch Watchdog] {epoch_games}/{target_games_per_epoch} games finished, all workers idle. Advancing to SGD training.", flush=True)
                        break
                    continue

                # Process batch of requests
                leaf_batch_reqs = []
                root_reqs = []
                opp_reqs = []
                league_act_reqs = []

                for req in eval_requests:
                    wid, req_type = req[0], req[1]
                    if req_type == "leaf_batch":
                        leaf_batch_reqs.append(req)
                    elif req_type == "root":
                        root_reqs.append(req)
                    elif req_type == "opp_act":
                        opp_reqs.append(req)
                    elif req_type == "league_act":
                        league_act_reqs.append(req)
                    elif req_type == "game_done":
                        _, _, history, logged_vp, logged_won, moves, participants, raw_vps = req[:8]
                        is_league_match = req[8] if len(req) > 8 else (participants.count("CurrentPolicy") == 1)
                        factions = req[9] if len(req) > 9 else []
                        game_duration = req[10] if len(req) > 10 else max(1.0, time.time() - start_time)

                        worker_busy[wid] = False

                        # Ingest completed game into buffer (updates sliding FIFO + preserves elite reservoir)
                        self.replay_buffer.add_game(history, raw_vps, participants)
                        epoch_real_scores.append(logged_vp)
                        winner_score = float(np.max(raw_vps)) if len(raw_vps) > 0 else float(logged_vp)
                        epoch_winner_scores.append(winner_score)
                        winner_seat = int(np.argmax(raw_vps)) if len(raw_vps) > 0 else 0
                        if is_league_match:
                            epoch_league_games += 1
                            if logged_won:
                                epoch_league_wins += 1
                        if logged_won:
                            epoch_wins += 1
                        epoch_games += 1
                        epoch_moves += moves

                        # Update League ratings with actual match participants
                        if getattr(self.config.league, "enabled", True) and self.league_manager is not None:
                            self.league_manager.update_match_results(participants, raw_vps)

                        # Faction names mapping
                        FACTION_NAMES = [
                            "Terrans", "Lantids", "Xenos", "Gleens", "Taklons", "Ambas",
                            "Hadsch Hallas", "Ivits", "Geodens", "Bal T'aks", "Firaks",
                            "Bescods", "Nevlas", "Itars"
                        ]
                        winner_faction = FACTION_NAMES[factions[winner_seat]] if factions and 0 <= factions[winner_seat] < 14 else f"Joueur {winner_seat}"
                        m_dur = int(game_duration // 60)
                        s_dur = int(game_duration % 60)
                        dur_str = f"{m_dur}m{s_dur:02d}s"

                        score_strs = []
                        for s_idx, vp in enumerate(raw_vps):
                            f_id = factions[s_idx] if s_idx < len(factions) else -1
                            f_name = FACTION_NAMES[f_id] if 0 <= f_id < len(FACTION_NAMES) else f"Siège {s_idx}"
                            star = " 👑" if s_idx == winner_seat else ""
                            score_strs.append(f"{f_name}: {vp:.0f} VP{star}")
                        scores_formatted = ", ".join(score_strs)

                        if is_league_match:
                            pol_status = f"CurrentPolicy 1v3 -> {'VICTOIRE 🏆' if logged_won else 'Défaite'} ({logged_vp:.0f} VP)"
                        else:
                            pol_status = f"CurrentPolicy (Moy: {logged_vp:.0f} VP, {'Vainqueur' if logged_won else 'Top: ' + str(round(winner_score)) + ' VP'})"

                        # Log game details to dedicated audit file runs/games.log (NOT polluting terminal/az_run.log)
                        try:
                            runs_dir = getattr(self.config.training, "runs_dir", "runs")
                            os.makedirs(runs_dir, exist_ok=True)
                            games_log = os.path.join(runs_dir, "games.log")
                            with open(games_log, "a", encoding="utf-8") as gf:
                                gf.write(
                                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Ep {epoch:04d} G{epoch_games:02d}/{target_games_per_epoch:02d} "
                                    f"| W#{wid} in {dur_str} ({moves} coups) | Winner: {winner_faction} ({winner_score:.0f} VP) "
                                    f"| {pol_status} | [{scores_formatted}]\n"
                                )
                        except Exception:
                            pass

                        # Update in-place progress bar immediately upon game completion
                        active_w = sum(1 for b in worker_busy if b)
                        self._render_progress_bar(
                            epoch, max_epochs, epoch_games, target_games_per_epoch, epoch_moves, time.time() - start_time, active_w
                        )

                        # TRUE CONTINUOUS STREAMING DISPATCH:
                        # Immediately give this worker another game so it NEVER idles!
                        next_parts = self._sample_game_participants()
                        self.cmd_pipes_parent[wid].send({
                            "cmd": "PLAY_GAME",
                            "num_simulations": self.az_config.num_simulations,
                            "temp_high": self.az_config.temperature_high,
                            "temp_low": self.az_config.temperature_low,
                            "temp_thresh": self.az_config.temperature_threshold_move,
                            "seed": int(time.time() * 1000) % 1000000 + epoch * 1000 + wid * 13 + total_games_dispatched,
                            "epoch": epoch,
                            "participants": next_parts,
                            "factions": self._sample_game_factions(),
                        })
                        worker_busy[wid] = True
                        total_games_dispatched += 1

                    elif req_type == "game_error":
                        err_msg = req[2] if len(req) > 2 else "Unknown error"
                        tb_msg = req[3] if len(req) > 3 else ""
                        print(f"  [!] [Worker {wid} Exception]: {err_msg}\n{tb_msg}", flush=True)
                        worker_error_count += 1
                        worker_busy[wid] = False
                        if worker_error_count > 5:
                            raise RuntimeError(f"Parallel Worker repeatedly failed: {err_msg}\n{tb_msg}")
                        # Re-dispatch game immediately so worker doesn't stay idle
                        retry_parts = self._sample_game_participants()
                        self.cmd_pipes_parent[wid].send({
                            "cmd": "PLAY_GAME",
                            "num_simulations": self.az_config.num_simulations,
                            "temp_high": self.az_config.temperature_high,
                            "temp_low": self.az_config.temperature_low,
                            "temp_thresh": self.az_config.temperature_threshold_move,
                            "seed": int(time.time() * 1000) % 1000000 + epoch * 1000 + wid * 17 + total_games_dispatched,
                            "epoch": epoch,
                            "participants": retry_parts,
                            "factions": self._sample_game_factions(),
                        })
                        worker_busy[wid] = True
                        total_games_dispatched += 1

                # A. Handle Batched Leaf Evaluations on GPU in Batched Forward Passes
                if leaf_batch_reqs:
                    grouped_leaf = {}
                    for req in leaf_batch_reqs:
                        m_name = req[5] if len(req) > 5 and req[5] is not None else "CurrentPolicy"
                        grouped_leaf.setdefault(m_name, []).append(req)

                    for m_name, reqs in grouped_leaf.items():
                        all_obs = []
                        all_masks = []
                        all_actors = []
                        slice_map = []
                        curr_idx = 0

                        for req in reqs:
                            wid, _, obs_np, mask_np, actors_list = req[:5]
                            n_items = len(obs_np)
                            all_obs.append(obs_np)
                            if mask_np is None:
                                mask_np = np.ones((n_items, getattr(self.config.model, "action_dim", 3130)), dtype=bool)
                            all_masks.append(mask_np)
                            if actors_list:
                                all_actors.extend(actors_list)
                            else:
                                all_actors.extend([0] * n_items)
                            slice_map.append((wid, curr_idx, curr_idx + n_items))
                            curr_idx += n_items

                        stacked_obs = np.concatenate(all_obs, axis=0)
                        stacked_masks = np.concatenate(all_masks, axis=0)

                        obs_t = torch.from_numpy(stacked_obs).float().to(self.device)
                        mask_t = torch.from_numpy(stacked_masks).bool().to(self.device)

                        with torch.no_grad():
                            if m_name == "CurrentPolicy" or self.league_manager is None or m_name not in self.league_manager.members:
                                vals, priors = self.agent.evaluate_leaf_batch(obs_t, mask_t, leaf_actors=all_actors)
                            else:
                                member = self.league_manager.members[m_name]
                                if member.agent is not None:
                                    vals, priors = member.agent.evaluate_leaf_batch(obs_t, mask_t, leaf_actors=all_actors)
                                elif member.policy_net is not None:
                                    device_type = "cuda" if obs_t.is_cuda else "cpu"
                                    with torch.amp.autocast(device_type=device_type, enabled=obs_t.is_cuda):
                                        feat = self.agent.shared_backbone(obs_t)
                                        vals_t = torch.tanh(self.agent.score_net.head(feat).squeeze(-1))
                                        if vals_t.dim() == 0:
                                            vals_t = vals_t.unsqueeze(0)
                                        logits = member.policy_net(obs_t, mask_t)
                                        neg_val = -6e4 if logits.dtype == torch.float16 else -1e9
                                        logits = torch.where(mask_t, logits, torch.tensor(neg_val, dtype=logits.dtype, device=logits.device))
                                        priors_t = F.softmax(logits, dim=-1)
                                    vals = vals_t.float().cpu().numpy()
                                    priors = priors_t.float().cpu().numpy()
                                else:
                                    vals, priors = self.agent.evaluate_leaf_batch(obs_t, mask_t, leaf_actors=all_actors)

                        for wid, start_i, end_i in slice_map:
                            self.resp_pipes_parent[wid].send((vals[start_i:end_i], priors[start_i:end_i]))

                # B. Handle Root Evaluations
                if root_reqs:
                    grouped_root = {}
                    for req in root_reqs:
                        m_name = req[5] if len(req) > 5 and req[5] is not None else "CurrentPolicy"
                        grouped_root.setdefault(m_name, []).append(req)

                    for m_name, r_reqs in grouped_root.items():
                        with torch.no_grad():
                            if m_name == "CurrentPolicy" or self.league_manager is None or m_name not in self.league_manager.members:
                                eval_agent = self.agent
                                eval_member = None
                            else:
                                eval_member = self.league_manager.members[m_name]
                                eval_agent = eval_member.agent if eval_member.agent is not None else None

                            if eval_agent is not None:
                                if len(r_reqs) == 1:
                                    wid, _, obs_np, mask_np, actor = r_reqs[0][:5]
                                    obs_t = torch.from_numpy(obs_np).float().to(self.device)
                                    mask_t = torch.from_numpy(mask_np).bool().to(self.device)
                                    priors, raw_logits = eval_agent.evaluate_root(obs_t, mask_t, actor=actor)
                                    self.resp_pipes_parent[wid].send((priors, raw_logits))
                                else:
                                    stacked_obs = np.stack([r[2] for r in r_reqs])
                                    stacked_masks = np.stack([r[3] for r in r_reqs])
                                    actors_list = [r[4] for r in r_reqs]
                                    obs_t = torch.from_numpy(stacked_obs).float().to(self.device)
                                    mask_t = torch.from_numpy(stacked_masks).bool().to(self.device)
                                    device_type = "cuda" if obs_t.is_cuda else "cpu"
                                    with torch.amp.autocast(device_type=device_type, enabled=obs_t.is_cuda):
                                        feats = eval_agent.shared_backbone(obs_t)
                                        logits = eval_agent.action_net.policy_head(feats)
                                        if getattr(eval_agent.config, "use_opponent_modeling", False):
                                            for i_a, a in enumerate(actors_list):
                                                if a != 0:
                                                    logits[i_a : i_a + 1] = eval_agent.action_net.opponent_head(feats[i_a : i_a + 1])
                                        if mask_t is not None:
                                            neg_val = -6e4 if logits.dtype == torch.float16 else -1e9
                                            logits = torch.where(mask_t, logits, torch.tensor(neg_val, dtype=logits.dtype, device=logits.device))
                                        priors = F.softmax(logits, dim=-1)
                                    priors_np = priors.float().detach().cpu().numpy()
                                    logits_np = logits.float().detach().cpu().numpy()
                                    for i_r, req in enumerate(r_reqs):
                                        self.resp_pipes_parent[req[0]].send((priors_np[i_r], logits_np[i_r]))
                            elif eval_member is not None and eval_member.policy_net is not None:
                                for req in r_reqs:
                                    wid, _, obs_np, mask_np, actor = req[:5]
                                    obs_t = torch.from_numpy(obs_np).float().to(self.device).unsqueeze(0)
                                    mask_t = torch.from_numpy(mask_np).bool().to(self.device).unsqueeze(0)
                                    device_type = "cuda" if obs_t.is_cuda else "cpu"
                                    with torch.amp.autocast(device_type=device_type, enabled=obs_t.is_cuda):
                                        logits = eval_member.policy_net(obs_t, mask_t)
                                        neg_val = -6e4 if logits.dtype == torch.float16 else -1e9
                                        logits = torch.where(mask_t, logits, torch.tensor(neg_val, dtype=logits.dtype, device=logits.device))
                                        priors = F.softmax(logits, dim=-1)
                                    priors_np = priors.squeeze(0).float().detach().cpu().numpy()
                                    logits_np = logits.squeeze(0).float().detach().cpu().numpy()
                                    self.resp_pipes_parent[wid].send((priors_np, logits_np))
                            else:
                                for req in r_reqs:
                                    wid, _, obs_np, mask_np, actor = req[:5]
                                    obs_t = torch.from_numpy(obs_np).float().to(self.device)
                                    mask_t = torch.from_numpy(mask_np).bool().to(self.device)
                                    priors, raw_logits = self.agent.evaluate_root(obs_t, mask_t, actor=actor)
                                    self.resp_pipes_parent[wid].send((priors, raw_logits))

                # C. Handle Opponent Actions
                if opp_reqs:
                    with torch.no_grad():
                        for req in opp_reqs:
                            wid, _, obs_np, mask_np, _ = req
                            obs_t = torch.from_numpy(obs_np).float().to(self.device)
                            mask_t = torch.from_numpy(mask_np).bool().to(self.device) if mask_np is not None else None
                            opp_probs = self.agent.predict_opponent_action(obs_t, mask_t)
                            self.resp_pipes_parent[wid].send(opp_probs.detach().cpu().numpy())

                # D. Handle League Snapshot Actions
                if league_act_reqs:
                    with torch.no_grad():
                        for req in league_act_reqs:
                            wid, _, p_name, obs_np, mask_np, _ = req
                            snap_member = self.league_manager.members.get(p_name) if self.league_manager else None
                            if snap_member is not None and snap_member.policy_net is not None:
                                obs_t = torch.from_numpy(obs_np).float().to(self.device).unsqueeze(0)
                                mask_t = torch.from_numpy(mask_np).bool().to(self.device).unsqueeze(0)
                                device_type = "cuda" if obs_t.is_cuda else "cpu"
                                with torch.amp.autocast(device_type=device_type, enabled=obs_t.is_cuda):
                                    logits = snap_member.policy_net(obs_t, mask_t)
                                    neg_val = -6e4 if logits.dtype == torch.float16 else -1e9
                                    logits = torch.where(mask_t, logits, neg_val)
                                    probs = F.softmax(logits, dim=-1).squeeze(0)
                                legal = np.where(mask_np)[0]
                                p_legal = probs[legal].float().cpu().numpy()
                                p_sum = float(p_legal.sum())
                                if p_sum > 1e-6 and not np.isnan(p_sum):
                                    p_legal = p_legal / p_sum
                                    act = int(np.random.choice(legal, p=p_legal))
                                elif len(legal) > 0:
                                    act = int(np.random.choice(legal))
                                else:
                                    act = 1412
                            else:
                                legal = np.where(mask_np)[0]
                                act = int(np.random.choice(legal)) if len(legal) > 0 else 1412
                            self.resp_pipes_parent[wid].send(act)

            if self._stop_event.is_set():
                break

            # Clear in-place progress bar cleanly before training and metric display
            sys.stdout.write("\r" + " " * 110 + "\r")
            sys.stdout.flush()

            # 3. Model Training Phase on Replay Buffer (Standard GPU SGD)
            self.agent.train()
            total_p_loss = 0.0
            total_v_loss = 0.0
            total_entropy = 0.0
            total_pred_score = 0.0

            # Démarre l'entraînement dès que le buffer contient au moins 64 transitions
            if len(self.replay_buffer) >= 64:
                steps = self.az_config.training_steps_per_epoch
            else:
                steps = 0
            for _ in range(steps):
                if self._stop_event.is_set():
                    break
                sample_size = min(len(self.replay_buffer), self.az_config.batch_size)
                batch = self.replay_buffer.sample(
                    sample_size,
                    optimism_power=getattr(self.az_config, "optimism_power", 1.5),
                    elite_ratio=getattr(self.az_config, "elite_reservoir_fraction", 0.25),
                )
                p_loss, v_loss, ent, pred_val = self.train_on_batch(batch)
                total_p_loss += p_loss
                total_v_loss += v_loss
                total_entropy += ent
                total_pred_score += pred_val

            duration = max(1e-4, time.time() - start_time)
            speed = epoch_moves / duration
            if epoch_league_games > 0:
                win_r = float(epoch_league_wins) / float(epoch_league_games)
            else:
                win_r = float(epoch_wins) / float(max(1, epoch_games))

            # 4. League Snapshotting
            if (
                getattr(self.config.league, "enabled", True)
                and self.league_manager is not None
                and epoch % getattr(self.config.league, "snapshot_interval_epochs", 10) == 0
            ):
                snap_name = self.league_manager.add_snapshot(self.agent, epoch)
                print(f"[LeagueManager] Snapshot saved: {snap_name} (Current Elo: {self.league_manager.members['CurrentPolicy'].elo:.1f})")

            curr_elo = self.league_manager.members["CurrentPolicy"].elo if self.league_manager else 1200.0
            curr_league_size = len(self.league_manager.members) if self.league_manager else 1

            if steps > 0:
                p_loss_avg = total_p_loss / steps
                v_loss_avg = total_v_loss / steps
                ent_avg = total_entropy / steps
                pred_sc_avg = total_pred_score / steps
            else:
                p_loss_avg, v_loss_avg, ent_avg, pred_sc_avg = 0.0, 0.0, 0.0, 50.0

            avg_real_vp = float(np.mean(epoch_real_scores)) if epoch_real_scores else 50.0
            avg_winner_vp = float(np.mean(epoch_winner_scores)) if epoch_winner_scores else avg_real_vp
            max_vp = float(np.max(epoch_winner_scores)) if epoch_winner_scores else avg_real_vp
            elite_count = len(getattr(self.replay_buffer, "elite_buffer", []))
            alpha_sched = 2.0 + 1.0 * min(1.0, max(0.0, float(epoch) / 100.0))

            # VRAM tracking
            if self.device.type == "cuda":
                vram_mb = int(torch.cuda.memory_reserved(self.device) / (1024 * 1024))
            else:
                vram_mb = 0

            metrics = EpochMetrics(
                epoch=epoch,
                policy_loss=p_loss_avg,
                value_loss=v_loss_avg,
                avg_predicted_score=pred_sc_avg,
                avg_real_score=avg_real_vp,
                entropy=ent_avg,
                duration=duration,
                win_rate=win_r,
                steps_per_sec=speed,
                league_elo=curr_elo,
                league_size=curr_league_size,
                rgsc_puzzles=0,
                avg_winner_score=avg_winner_vp,
                max_score=max_vp,
                elite_size=elite_count,
                target_ambition=alpha_sched,
            )

            if callback:
                callback(metrics)
            else:
                pol_sign = "+" if p_loss_avg >= 0 else ""
                print(
                    f"[{epoch:04d}/{max_epochs}] {int(round(speed)):>2} st/s   "
                    f"P={pol_sign}{p_loss_avg:.3f} V={v_loss_avg:.3f}   "
                    f"Win: {avg_winner_vp:>5.1f} ({max_vp:>5.1f})   "
                    f"Moy: {avg_real_vp:>5.1f}   "
                    f"α: {alpha_sched:.2f}   "
                    f"Elite: {elite_count:<3d}   "
                    f"{vram_mb} MB",
                    flush=True
                )

            # Auto-save checkpoints
            ckpt_dir = getattr(self.config.training, "checkpoint_dir", "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            self.save_checkpoint(os.path.join(ckpt_dir, "gaia_latest.pt"))
            ckpt_interval = getattr(self.az_config, "checkpoint_interval", getattr(self.config.training, "save_checkpoint_interval", 50))
            if epoch % ckpt_interval == 0:
                self.save_checkpoint(os.path.join(ckpt_dir, f"az_checkpoint_{epoch}.pt"))

        self.stop_workers()
        self._is_running = False
