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
from environment import make_gaia_env, NativeGaiaEnv
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
        self.req_queue.put((self.worker_id, "root", obs, mask, actor))
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

        self.req_queue.put((self.worker_id, "leaf_batch", obs_np, mask_np, leaf_actors))
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
                num_sims = msg.get("num_simulations", app_cfg.alphazero.num_simulations)
                temp_high = msg.get("temp_high", app_cfg.alphazero.temperature_high)
                temp_low = msg.get("temp_low", app_cfg.alphazero.temperature_low)
                temp_thresh = msg.get("temp_thresh", app_cfg.alphazero.temperature_threshold_move)
                game_seed = msg.get("seed", base_seed + random.randint(1, 1000000))

                try:
                    # Reset environment
                    obs, mask = env.reset(seed=game_seed)
                    num_players = getattr(env, "num_players", 4)
                    if hasattr(env, "set_player_faction"):
                        factions = np.random.choice(14, num_players, replace=False)
                        for seat, faction_id in enumerate(factions):
                            env.set_player_faction(seat, int(faction_id))

                    history = []
                    move_count = 0
                    total_game_steps = 0
                    MAX_STEPS = 1200

                    while not env.terminated and total_game_steps < MAX_STEPS:
                        if stop_event.is_set():
                            break

                        total_game_steps += 1
                        curr_player = env.current_player
                        curr_obs = env._get_obs() if hasattr(env, "_get_obs") else env.observe().values
                        curr_mask = env.get_action_mask()

                        legal_indices = np.where(curr_mask)[0]
                        if len(legal_indices) == 0:
                            env.terminated = True
                            break

                        temp = temp_high if move_count < temp_thresh else temp_low

                        # Search using MCTS proxy
                        action, probs, _ = mcts.search(
                            env,
                            num_simulations=num_sims,
                            temperature=temp,
                            add_noise=True,
                        )

                        history.append((curr_player, curr_obs, curr_mask, probs))
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

                    raw_vps = [float(p.get("vp", 0.0)) for p in getattr(env, "players_state", [{"vp": 0.0}] * num_players)]
                    p0_vp = raw_vps[0] if len(raw_vps) > 0 else 50.0
                    p0_won = bool(len(raw_vps) >= 2 and p0_vp > max([v for i, v in enumerate(raw_vps) if i != 0] + [0.0]))

                    final_history = [
                        (obs_s, mask_s, probs_s, float(raw_vps[p]))
                        for p, obs_s, mask_s, probs_s in history
                    ]

                    # Push completed game to central server
                    req_queue.put((
                        worker_id,
                        "game_done",
                        final_history,
                        p0_vp,
                        p0_won,
                        move_count,
                        ["CurrentPolicy"] * num_players,
                        raw_vps,
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
        target_games_per_epoch = max(self.num_workers, getattr(self.az_config, "games_per_epoch", 8))

        for epoch in range(start_epoch, max_epochs + 1):
            if self._stop_event.is_set():
                break

            self.current_epoch = epoch
            start_time = time.time()
            self.agent.eval()

            epoch_real_scores = []
            epoch_wins = 0
            epoch_games = 0
            epoch_moves = 0
            worker_error_count = 0

            # 1. Dispatch games to all idle workers
            games_dispatched = 0
            games_completed = 0
            for wid in range(self.num_workers):
                if games_dispatched < target_games_per_epoch:
                    self.cmd_pipes_parent[wid].send({
                        "cmd": "PLAY_GAME",
                        "num_simulations": self.az_config.num_simulations,
                        "temp_high": self.az_config.temperature_high,
                        "temp_low": self.az_config.temperature_low,
                        "temp_thresh": self.az_config.temperature_threshold_move,
                        "seed": int(time.time() * 1000) % 1000000 + epoch * 100 + wid,
                    })
                    games_dispatched += 1

            # 2. Central GPU Batch Server Loop
            while games_completed < target_games_per_epoch:
                if self._stop_event.is_set():
                    break

                # Collect batched evaluation requests from workers
                eval_requests = []
                # Non-blocking fetch with up to 1ms wait to coalesce multi-worker leaves
                try:
                    first_req = self.req_queue.get(timeout=0.005)
                    eval_requests.append(first_req)
                    while len(eval_requests) < 64:
                        try:
                            req = self.req_queue.get_nowait()
                            eval_requests.append(req)
                        except Exception:
                            break
                except Exception:
                    continue

                # Process batch of requests
                leaf_batch_reqs = []
                root_reqs = []
                opp_reqs = []

                for req in eval_requests:
                    wid, req_type = req[0], req[1]
                    if req_type == "leaf_batch":
                        leaf_batch_reqs.append(req)
                    elif req_type == "root":
                        root_reqs.append(req)
                    elif req_type == "opp_act":
                        opp_reqs.append(req)
                    elif req_type == "game_done":
                        _, _, history, p0_vp, p0_won, moves, participants, raw_vps = req
                        for step_data in history:
                            self.replay_buffer.add(*step_data)
                        epoch_real_scores.append(p0_vp)
                        if p0_won:
                            epoch_wins += 1
                        epoch_games += 1
                        epoch_moves += moves
                        games_completed += 1

                        # Update League ratings
                        if getattr(self.config.league, "enabled", True) and self.league_manager is not None:
                            self.league_manager.update_match_results(participants, raw_vps)

                        # Launch next game if budget remains
                        if games_dispatched < target_games_per_epoch:
                            self.cmd_pipes_parent[wid].send({
                                "cmd": "PLAY_GAME",
                                "num_simulations": self.az_config.num_simulations,
                                "temp_high": self.az_config.temperature_high,
                                "temp_low": self.az_config.temperature_low,
                                "temp_thresh": self.az_config.temperature_threshold_move,
                                "seed": int(time.time() * 1000) % 1000000 + epoch * 100 + games_dispatched,
                            })
                            games_dispatched += 1

                    elif req_type == "game_error":
                        err_msg = req[2] if len(req) > 2 else "Unknown error"
                        tb_msg = req[3] if len(req) > 3 else ""
                        print(f"  [!] [Worker {wid} Exception]: {err_msg}\n{tb_msg}", flush=True)
                        worker_error_count += 1
                        if worker_error_count > 5:
                            raise RuntimeError(f"Parallel Worker repeatedly failed: {err_msg}\n{tb_msg}")
                        # Re-dispatch game so epoch completes real games instead of skipping!
                        if games_dispatched < target_games_per_epoch:
                            self.cmd_pipes_parent[wid].send({
                                "cmd": "PLAY_GAME",
                                "num_simulations": self.az_config.num_simulations,
                                "temp_high": self.az_config.temperature_high,
                                "temp_low": self.az_config.temperature_low,
                                "temp_thresh": self.az_config.temperature_threshold_move,
                                "seed": int(time.time() * 1000) % 1000000 + epoch * 100 + games_dispatched,
                            })
                            games_dispatched += 1

                # A. Handle Batched Leaf Evaluations on GPU in ONE Forward Pass
                if leaf_batch_reqs:
                    all_obs = []
                    all_masks = []
                    all_actors = []
                    slice_map = []
                    curr_idx = 0

                    for req in leaf_batch_reqs:
                        wid, _, obs_np, mask_np, actors_list = req
                        n_items = len(obs_np)
                        all_obs.append(obs_np)
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
                        vals, priors = self.agent.evaluate_leaf_batch(obs_t, mask_t, leaf_actors=all_actors)

                    for wid, start_i, end_i in slice_map:
                        self.resp_pipes_parent[wid].send((vals[start_i:end_i], priors[start_i:end_i]))

                # B. Handle Root Evaluations
                if root_reqs:
                    for req in root_reqs:
                        wid, _, obs_np, mask_np, actor = req
                        obs_t = torch.from_numpy(obs_np).float().to(self.device)
                        mask_t = torch.from_numpy(mask_np).bool().to(self.device)
                        priors, raw_logits = self.agent.evaluate_root(obs_t, mask_t, actor=actor)
                        self.resp_pipes_parent[wid].send((priors, raw_logits))

                # C. Handle Opponent Actions
                if opp_reqs:
                    for req in opp_reqs:
                        wid, _, obs_np, mask_np, _ = req
                        obs_t = torch.from_numpy(obs_np).float().to(self.device)
                        mask_t = torch.from_numpy(mask_np).bool().to(self.device) if mask_np is not None else None
                        opp_probs = self.agent.predict_opponent_action(obs_t, mask_t)
                        self.resp_pipes_parent[wid].send(opp_probs.cpu().numpy())

            if self._stop_event.is_set():
                break

            # 3. Model Training Phase on Replay Buffer (Standard GPU SGD)
            self.agent.train()
            total_p_loss = 0.0
            total_v_loss = 0.0
            total_entropy = 0.0
            total_pred_score = 0.0

            steps = self.az_config.training_steps_per_epoch if len(self.replay_buffer) >= 16 else 0
            for _ in range(steps):
                if self._stop_event.is_set():
                    break
                batch = self.replay_buffer.sample(
                    self.az_config.batch_size,
                    optimism_power=getattr(self.az_config, "optimism_power", 1.0),
                )
                p_loss, v_loss, ent, pred_val = self.train_on_batch(batch)
                total_p_loss += p_loss
                total_v_loss += v_loss
                total_entropy += ent
                total_pred_score += pred_val

            duration = max(1e-4, time.time() - start_time)
            speed = epoch_moves / duration
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
            )

            if callback:
                callback(metrics)
            else:
                pol_sign = "+" if p_loss_avg >= 0 else ""
                print(
                    f"[{epoch:04d}/{max_epochs}] {int(round(speed)):>2} st/s       "
                    f"P={pol_sign}{p_loss_avg:.3f} V={v_loss_avg:.3f}          "
                    f"{avg_real_vp:>5.1f} VP    {int(round(win_r * 100)):>2}%          "
                    f"ALPHAZERO  {vram_mb} MB"
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
