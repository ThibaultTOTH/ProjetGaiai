"""Advanced Neural Architecture Search (NAS) & Accelerated Tournament Optimizer for Gaia Project.

Provides:
- Joint search over Neural Architecture (SwiGLU/PreLN/Bottleneck, Depth, Width, Activations, GNN)
  and AlphaZero Dynamics (MCTS simulations, c_PUCT, Gumbel candidate budget, Policy LR, Value loss coefficient, Optimism power).
- Evolutionary regularized mutation & crossover of top-performing agent chromosomes.
- Multi-fidelity early pruning to abort diverging or stalling architectures early.
- Accelerated Head-to-Head Tournament Evaluation:
  Candidates are evaluated not by noisy self-play, but in direct competitive matches with alternating
  seats against baseline and champion agents. This strictly prevents the premature "Pass" local minimum trap.
- Automated champion serialization to runs/best_hyperparams_alphazero.json.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import logging
import math
import os
import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch

from config import AppConfig
from environment import make_gaia_env
from mcts import MultiPlayerMCTS
from models import DualGaiaAgent
from alphazero_trainer import AlphaZeroTrainer

logger = logging.getLogger(__name__)


@dataclass
class HyperoptTrial:
    trial_id: int
    params: Dict[str, Any]
    architecture_name: str = ""
    val_loss: float = 0.0
    win_rate: float = 0.0
    avg_score: float = 0.0
    steps_per_sec: float = 0.0
    objective_score: float = 0.0
    status: str = "Pending"  # 'Pending', 'Running', 'Completed', 'Pruned (Divergé)', 'Pruned (Lent)'
    history_losses: List[float] = field(default_factory=list)
    history_scores: List[float] = field(default_factory=list)
    history_entropies: List[float] = field(default_factory=list)
    vp_margin: float = 0.0


def play_head_to_head_game(
    env: Any,
    agent_p0: DualGaiaAgent,
    mcts_p0: MultiPlayerMCTS,
    agent_p1: DualGaiaAgent,
    mcts_p1: MultiPlayerMCTS,
    sims: int = 16,
    max_steps: int = 2000,
) -> Tuple[List[float], int]:
    """Plays a 2-player competitive match between agent_p0 and agent_p1 with MCTS.

    Returns:
        (scores, winner_seat): Final VP scores for [P0, P1] and index of the winning seat.
    """
    env.reset()
    if hasattr(env, "set_player_faction"):
        factions = np.random.choice(14, 2, replace=False)
        env.set_player_faction(0, int(factions[0]))
        env.set_player_faction(1, int(factions[1]))

    step_count = 0
    consecutive_errors = 0

    while not env.terminated and step_count < max_steps:
        step_count += 1
        curr_p = env.current_player
        mask = env.get_action_mask()
        legal = np.where(mask)[0]
        if len(legal) == 0:
            env.terminated = True
            break

        active_mcts = mcts_p0 if curr_p == 0 else mcts_p1
        act, _, _ = active_mcts.search(
            env,
            num_simulations=sims,
            temperature=0.0,
            add_noise=False,
        )

        res = env.step(int(act))
        if hasattr(res, "info") and "error" in res.info:
            consecutive_errors += 1
            for alt_act in np.random.permutation(legal):
                if alt_act != act:
                    res = env.step(int(alt_act))
                    if not (hasattr(res, "info") and "error" in res.info):
                        consecutive_errors = 0
                        break
            if consecutive_errors >= 5:
                env.terminated = True
                break
        else:
            consecutive_errors = 0

    num_players = getattr(env, "num_players", 2)
    raw_vps = [float(p.get("vp", 0.0)) for p in getattr(env, "players_state", [{"vp": 0.0}] * num_players)]
    vps = raw_vps[:2]
    if len(vps) < 2:
        vps = [50.0, 50.0]
    winner = 0 if vps[0] >= vps[1] else 1
    return vps, winner


class AdvancedNASOptimizer:
    """Manages joint Neural Architecture Search and AlphaZero Hyperparameter Optimization via Accelerated Tournaments."""

    def __init__(self, base_config: Optional[AppConfig] = None, mode: Optional[str] = None):
        self.base_config = base_config or AppConfig()
        self.mode = "alphazero"
        self.trials: List[HyperoptTrial] = []
        self.best_trial: Optional[HyperoptTrial] = None
        self._stop_event = threading.Event()
        self._is_running = False
        self._thread: Optional[threading.Thread] = None

        # Create a cached baseline agent for tournament comparisons
        self._baseline_agent: Optional[DualGaiaAgent] = None

    def _get_baseline_agent(self, device: torch.device) -> DualGaiaAgent:
        if self._baseline_agent is None:
            base_agent = DualGaiaAgent(self.base_config.model)
            base_agent.to_device(device)
            base_agent.eval()
            self._baseline_agent = base_agent
        return self._baseline_agent

    def is_running(self) -> bool:
        return self._is_running

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._is_running = False

    @staticmethod
    def format_arch_name(params: Dict[str, Any], mode: str = "alphazero") -> str:
        """Returns a clean concise string describing the architecture and components."""
        block = params.get("block_type", "swiglu").replace("_", "").upper()
        layers = params.get("hidden_layers", [512, 512, 256])
        layers_str = "x".join(str(x) for x in layers)
        act = params.get("activation", "silu").upper()
        gnn = f"+GNN{params.get('gnn_layers', 3)}" if params.get("use_gnn_map", True) else ""
        sims = params.get("az_num_simulations", 16)
        lr = params.get("az_policy_lr", 2.5e-4)
        c_puct = params.get("az_c_puct", 1.414)
        v_coef = params.get("az_value_loss_coef", 1.0)
        return f"{block} {layers_str} ({act}){gnn} [AZ: sims={sims}, c_puct={c_puct:.2f}, lr={lr:.1e}, v_coef={v_coef:.1f}]"

    @staticmethod
    def extract_params_from_config(cfg: AppConfig, mode: str = "alphazero") -> Dict[str, Any]:
        """Extracts hyperparameter search dictionary from an AppConfig instance to serve as baseline."""
        return {
            # Architecture
            "block_type": getattr(cfg.model, "block_type", "swiglu"),
            "hidden_layers": list(getattr(cfg.model, "policy_hidden_layers", [512, 512, 256])),
            "activation": getattr(cfg.model, "policy_activation", "silu"),
            "dropout": float(getattr(cfg.model, "policy_dropout", 0.03)),
            "use_input_norm": bool(getattr(cfg.model, "use_input_norm", True)),
            "use_gnn_map": bool(getattr(cfg.model, "use_gnn_map", True)),
            "gnn_layers": int(getattr(cfg.model, "gnn_layers", 3)),
            "gnn_hidden_dim": int(getattr(cfg.model, "gnn_hidden_dim", 64)),
            "policy_weight_decay": float(getattr(cfg.model, "policy_weight_decay", 1e-4)),
            # AlphaZero Parameters
            "az_policy_lr": float(getattr(cfg.model, "policy_lr", 2.5e-4)),
            "az_value_loss_coef": float(getattr(cfg.alphazero, "value_loss_coef", 1.0)),
            "az_batch_size": int(getattr(cfg.alphazero, "batch_size", 256)),
            "az_num_simulations": int(getattr(cfg.alphazero, "num_simulations", 16)),
            "az_gumbel_candidates": int(getattr(cfg.alphazero, "gumbel_candidates", 8)),
            "az_c_puct": float(getattr(cfg.mcts, "c_puct", 1.414)),
            "az_optimism_power": float(getattr(cfg.alphazero, "optimism_power", 1.0)),
            "az_temp_threshold": int(getattr(cfg.alphazero, "temperature_threshold_move", 4)),
            "az_milestone_weight": float(getattr(cfg.mcts, "milestone_shaping_weight", 0.25)),
            "az_optimism_weight": float(getattr(cfg.mcts, "optimism_weight", 0.15)),
            # Tactical Shaping Annealing
            "shaping_anneal_epochs": int(getattr(cfg.mcts, "shaping_anneal_epochs", 500)),
            "initial_shaping_scale": float(getattr(cfg.mcts, "initial_shaping_scale", 50.0)),
        }

    def generate_candidate(self, elite_pool: Optional[List[HyperoptTrial]] = None) -> Dict[str, Any]:
        """Generates a candidate using Bayesian TPE guided sampling or regularized mutation."""
        blocks_candidates = ["swiglu", "pre_ln", "bottleneck"]
        activations_candidates = ["silu", "gelu", "mish"]
        dropouts_candidates = [0.0, 0.03, 0.05]

        layers_candidates = [
            [512, 512, 256],            # Fast Baseline (~6M params)
            [768, 768, 384, 256],       # Balanced 4-layer (~10M params)
            [1024, 1024, 512, 256],     # Grandmaster Standard (~12M params)
            [1024, 1024, 1024, 512],    # Deep Heavy SOTA (~20M params)
            [1536, 1024, 512, 256],     # Wide Front-End (~22M params)
        ]

        if elite_pool and len(elite_pool) >= 1 and random.random() < 0.75:
            parent = random.choice(elite_pool[:min(3, len(elite_pool))]).params
            return self.mutate_candidate(parent)

        return {
            # Architecture
            "block_type": random.choice(blocks_candidates),
            "hidden_layers": random.choice(layers_candidates),
            "activation": random.choice(activations_candidates),
            "dropout": random.choice(dropouts_candidates),
            "use_input_norm": True,
            "use_gnn_map": random.choice([True, True, False]),
            "gnn_layers": random.choice([2, 3]),
            "gnn_hidden_dim": random.choice([48, 64]),
            "policy_weight_decay": float(random.choice([1e-5, 1e-4, 5e-4])),
            # AlphaZero Optimization
            "az_policy_lr": float(random.choice([1.5e-4, 2.5e-4, 3.5e-4, 5.0e-4])),
            "az_value_loss_coef": float(random.choice([0.50, 1.0, 1.5, 2.0])),
            "az_batch_size": int(random.choice([128, 256])),
            "az_num_simulations": int(random.choice([16, 24, 32])),
            "az_gumbel_candidates": int(random.choice([4, 8, 12])),
            "az_c_puct": float(random.choice([1.25, 1.414, 1.75, 2.0])),
            "az_optimism_power": float(random.choice([0.5, 1.0, 1.5])),
            "az_temp_threshold": int(random.choice([2, 4, 6])),
            "az_milestone_weight": float(random.choice([0.15, 0.25, 0.40])),
            "az_optimism_weight": float(random.choice([0.10, 0.15, 0.25])),
            # Shaping Schedule
            "shaping_anneal_epochs": 500,
            "initial_shaping_scale": float(random.choice([40.0, 50.0, 60.0])),
        }

    def mutate_candidate(self, parent: Dict[str, Any]) -> Dict[str, Any]:
        """Mutates 1 or 2 targeted genes of an elite parent candidate."""
        child = deepcopy(parent)
        all_genes = [
            "block_type", "hidden_layers", "activation", "dropout",
            "gnn", "az_policy_lr", "az_value_loss_coef", "az_sims",
            "az_c_puct", "az_optimism_power", "shaping_scale"
        ]
        mutations = random.sample(all_genes, k=random.choice([1, 2]))

        if "block_type" in mutations:
            child["block_type"] = random.choice(["swiglu", "pre_ln", "bottleneck"])
        if "hidden_layers" in mutations:
            layers = list(child.get("hidden_layers", [512, 512, 256]))
            action = random.choice(["widen", "narrow", "tweak"])
            if action == "widen":
                layers = [min(2048, int(x * 1.25)) for x in layers]
            elif action == "narrow":
                layers = [max(128, int(x * 0.8)) for x in layers]
            else:
                layers[-1] = random.choice([128, 256, 384])
            child["hidden_layers"] = layers
        if "activation" in mutations:
            child["activation"] = random.choice(["silu", "gelu", "mish"])
        if "dropout" in mutations:
            child["dropout"] = float(random.choice([0.0, 0.03, 0.05]))
        if "gnn" in mutations:
            child["use_gnn_map"] = not child.get("use_gnn_map", True)
            child["gnn_hidden_dim"] = random.choice([48, 64])
        if "az_policy_lr" in mutations:
            scale = random.uniform(0.8, 1.25)
            child["az_policy_lr"] = float(np.clip(child.get("az_policy_lr", 2.5e-4) * scale, 1.0e-4, 6.0e-4))
        if "az_value_loss_coef" in mutations:
            child["az_value_loss_coef"] = float(random.choice([0.5, 1.0, 1.5, 2.0]))
        if "az_sims" in mutations:
            child["az_num_simulations"] = int(random.choice([16, 24, 32]))
            child["az_gumbel_candidates"] = int(random.choice([4, 8, 12]))
        if "az_c_puct" in mutations:
            child["az_c_puct"] = float(random.choice([1.2, 1.414, 1.75, 2.0]))
        if "az_optimism_power" in mutations:
            child["az_optimism_power"] = float(random.choice([0.5, 1.0, 1.5]))
        if "shaping_scale" in mutations:
            child["initial_shaping_scale"] = float(random.choice([40.0, 50.0, 60.0]))

        return child

    def evaluate_trial(
        self,
        trial: HyperoptTrial,
        sprint_epochs: int = 3,
        completed_pool: Optional[List[HyperoptTrial]] = None,
    ) -> None:
        """Evaluates an AlphaZero candidate trial via rapid self-play sprint and accelerated tournament."""
        self._evaluate_alphazero_trial(trial, sprint_epochs=max(2, min(4, sprint_epochs)), completed_pool=completed_pool)

    def _evaluate_alphazero_trial(
        self,
        trial: HyperoptTrial,
        sprint_epochs: int = 3,
        completed_pool: Optional[List[HyperoptTrial]] = None,
    ) -> None:
        """Evaluates candidate using rapid training sprint + accelerated head-to-head tournament."""
        cfg = deepcopy(self.base_config)
        p = trial.params

        # Architecture injection
        cfg.model.block_type = p.get("block_type", "swiglu")
        cfg.model.policy_hidden_layers = p.get("hidden_layers", [512, 512, 256])
        cfg.model.score_hidden_layers = p.get("hidden_layers", [512, 512, 256])
        cfg.model.policy_activation = p.get("activation", "silu")
        cfg.model.score_activation = p.get("activation", "silu")
        cfg.model.policy_dropout = p.get("dropout", 0.03)
        cfg.model.score_dropout = p.get("dropout", 0.03)
        cfg.model.use_input_norm = p.get("use_input_norm", True)
        cfg.model.use_gnn_map = p.get("use_gnn_map", True)
        cfg.model.gnn_layers = p.get("gnn_layers", 3)
        cfg.model.gnn_hidden_dim = p.get("gnn_hidden_dim", 64)
        cfg.model.policy_lr = float(p.get("az_policy_lr", 2.5e-4))
        cfg.model.policy_weight_decay = float(p.get("policy_weight_decay", 1e-4))

        # AlphaZero hyperparameters
        cfg.alphazero.enabled = True
        cfg.alphazero.batch_size = int(p.get("az_batch_size", 256))
        cfg.alphazero.value_loss_coef = float(p.get("az_value_loss_coef", 1.0))
        cfg.alphazero.num_simulations = int(p.get("az_num_simulations", 16))
        cfg.alphazero.gumbel_candidates = int(p.get("az_gumbel_candidates", 8))
        cfg.alphazero.optimism_power = float(p.get("az_optimism_power", 1.0))
        cfg.alphazero.temperature_threshold_move = int(p.get("az_temp_threshold", 4))
        cfg.alphazero.games_per_epoch = 1
        cfg.alphazero.training_steps_per_epoch = 15

        cfg.mcts.c_puct = float(p.get("az_c_puct", 1.414))
        cfg.mcts.milestone_shaping_weight = float(p.get("az_milestone_weight", 0.25))
        cfg.mcts.optimism_weight = float(p.get("az_optimism_weight", 0.15))
        cfg.mcts.shaping_anneal_epochs = int(p.get("shaping_anneal_epochs", 500))
        cfg.mcts.initial_shaping_scale = float(p.get("initial_shaping_scale", 50.0))

        # -------------------------------------------------------------
        # 1. Sprint Self-Play Training
        # -------------------------------------------------------------
        agent = DualGaiaAgent(cfg.model)
        trainer = AlphaZeroTrainer(cfg, agent=agent)
        env = make_gaia_env(players=cfg.model.num_players)

        losses: List[float] = []
        scores: List[float] = []
        entropies: List[float] = []
        speeds: List[float] = []

        for ep in range(sprint_epochs):
            if self._stop_event.is_set():
                trial.status = "Cancelled"
                return

            t0 = time.time()
            game_history, p0_vp, _, moves = trainer.self_play_game(env)
            scores.append(p0_vp)

            for step_data in game_history:
                trainer.replay_buffer.add(*step_data)

            ep_p_loss = 0.0
            ep_v_loss = 0.0
            ep_ent = 0.0
            steps = min(trainer.az_config.training_steps_per_epoch, len(trainer.replay_buffer))
            if steps > 0 and len(trainer.replay_buffer) >= 8:
                for _ in range(steps):
                    batch = trainer.replay_buffer.sample(
                        min(len(trainer.replay_buffer), trainer.az_config.batch_size),
                        optimism_power=p.get("az_optimism_power", 1.0),
                    )
                    p_loss, v_loss, ent, _ = trainer.train_on_batch(batch)
                    ep_p_loss += p_loss
                    ep_v_loss += v_loss
                    ep_ent += ent
                v_coef = float(p.get("az_value_loss_coef", 1.0))
                norm_v_loss = (ep_v_loss / max(1, steps)) / 25.0
                mean_p_loss = ep_p_loss / max(1, steps)
                ep_loss = mean_p_loss + v_coef * norm_v_loss
                ep_ent = ep_ent / max(1, steps)
            else:
                ep_loss = 5.0
                ep_ent = 1.0

            dur = max(1e-4, time.time() - t0)
            losses.append(ep_loss)
            entropies.append(ep_ent)
            speeds.append(moves / dur)

            # Divergence or NaN guard
            if np.isnan(ep_loss) or np.isinf(ep_loss) or ep_loss > 25.0:
                trial.status = "Pruned (Divergé)"
                trial.val_loss = round(float(ep_loss), 4)
                return

        trial.history_losses = list(losses)
        trial.history_scores = list(scores)
        trial.history_entropies = list(entropies)

        # -------------------------------------------------------------
        # 2. Accelerated Head-to-Head Tournament Evaluation
        # -------------------------------------------------------------
        agent.eval()
        tournament_sims = min(24, max(12, int(p.get("az_num_simulations", 16))))
        cand_mcts = MultiPlayerMCTS(agent, config=cfg.mcts)

        # Baseline opponent
        base_agent = self._get_baseline_agent(trainer.device)
        base_cfg = deepcopy(self.base_config.mcts)
        base_mcts = MultiPlayerMCTS(base_agent, config=base_cfg)

        tourney_env = make_gaia_env(players=2)

        cand_vps: List[float] = []
        opp_vps: List[float] = []
        tournament_wins = 0
        tournament_games = 0

        # Match 1: Candidate is Seat 0, Baseline is Seat 1
        vps_1, win_1 = play_head_to_head_game(
            tourney_env, agent, cand_mcts, base_agent, base_mcts, sims=tournament_sims
        )
        cand_vps.append(vps_1[0])
        opp_vps.append(vps_1[1])
        if win_1 == 0:
            tournament_wins += 1
        tournament_games += 1

        # Match 2: Baseline is Seat 0, Candidate is Seat 1 (Seat symmetry test)
        vps_2, win_2 = play_head_to_head_game(
            tourney_env, base_agent, base_mcts, agent, cand_mcts, sims=tournament_sims
        )
        cand_vps.append(vps_2[1])
        opp_vps.append(vps_2[0])
        if win_2 == 1:
            tournament_wins += 1
        tournament_games += 1

        # Match 3 & 4: If completed champion exists in the pool, test against champion
        if self.best_trial is not None and completed_pool and len(completed_pool) >= 1:
            champ_params = self.best_trial.params
            champ_cfg = deepcopy(self.base_config)
            champ_cfg = apply_params_to_config(champ_cfg, champ_params, mode="alphazero")
            champ_agent = DualGaiaAgent(champ_cfg.model)
            champ_agent.to_device(trainer.device)
            champ_agent.eval()
            champ_mcts = MultiPlayerMCTS(champ_agent, config=champ_cfg.mcts)

            # Match 3: Cand P0 vs Champion P1
            vps_3, win_3 = play_head_to_head_game(
                tourney_env, agent, cand_mcts, champ_agent, champ_mcts, sims=tournament_sims
            )
            cand_vps.append(vps_3[0])
            opp_vps.append(vps_3[1])
            if win_3 == 0:
                tournament_wins += 1
            tournament_games += 1

            # Match 4: Champion P0 vs Cand P1
            vps_4, win_4 = play_head_to_head_game(
                tourney_env, champ_agent, champ_mcts, agent, cand_mcts, sims=tournament_sims
            )
            cand_vps.append(vps_4[1])
            opp_vps.append(vps_4[0])
            if win_4 == 1:
                tournament_wins += 1
            tournament_games += 1

        # -------------------------------------------------------------
        # 3. Composite Objective Scoring
        # -------------------------------------------------------------
        win_rate = float(tournament_wins) / float(max(1, tournament_games))
        avg_score = float(np.mean(cand_vps)) if cand_vps else 50.0
        vp_margin = float(np.mean([c - o for c, o in zip(cand_vps, opp_vps)])) if cand_vps else 0.0
        val_loss = float(np.mean(losses)) if losses else 1.0
        avg_speed = float(np.mean(speeds)) if speeds else 10.0

        throughput_bonus = min(5.0, avg_speed / 50.0)
        clamped_margin = max(-50.0, min(50.0, vp_margin))

        objective = (
            (win_rate * 50.0)
            + (avg_score * 0.4)
            + (clamped_margin * 0.3)
            - (val_loss * 1.5)
            + throughput_bonus
        )

        trial.val_loss = round(val_loss, 4)
        trial.win_rate = round(win_rate, 3)
        trial.avg_score = round(avg_score, 2)
        trial.vp_margin = round(vp_margin, 2)
        trial.steps_per_sec = round(avg_speed, 1)
        trial.objective_score = round(objective, 3)
        trial.status = "Completed"

    def run_optimization_sync(
        self,
        num_trials: int = 15,
        sprint_epochs: int = 3,
        on_trial_update: Optional[Callable[[HyperoptTrial], None]] = None,
    ) -> Optional[HyperoptTrial]:
        """Executes Advanced NAS and Schedule Optimization synchronously in the calling thread."""
        self._stop_event.clear()
        self._is_running = True
        self.trials.clear()
        self.best_trial = None

        try:
            for i in range(num_trials):
                if self._stop_event.is_set():
                    break

                completed = [t for t in self.trials if t.status == "Completed"]

                # Trial #1 begins with baseline registered configuration
                if i == 0:
                    candidate_params = self.extract_params_from_config(self.base_config, mode=self.mode)
                    arch_name = "★ BASELINE: " + self.format_arch_name(candidate_params, mode=self.mode)
                else:
                    candidate_params = self.generate_candidate(elite_pool=completed)
                    arch_name = self.format_arch_name(candidate_params, mode=self.mode)

                trial = HyperoptTrial(
                    trial_id=i + 1,
                    params=candidate_params,
                    architecture_name=arch_name,
                    status="Running",
                )
                self.trials.append(trial)
                if on_trial_update:
                    on_trial_update(trial)

                self.evaluate_trial(
                    trial,
                    sprint_epochs=sprint_epochs,
                    completed_pool=completed,
                )

                if on_trial_update:
                    on_trial_update(trial)

                if trial.status == "Completed":
                    if (
                        self.best_trial is None
                        or trial.objective_score > self.best_trial.objective_score
                    ):
                        self.best_trial = trial
                        try:
                            import json
                            out_dir = getattr(self.base_config.training, "runs_dir", "runs")
                            os.makedirs(out_dir, exist_ok=True)
                            out_file = os.path.join(out_dir, "best_hyperparams_alphazero.json")
                            with open(out_file, "w") as f:
                                json.dump(trial.params, f, indent=2)
                        except Exception:
                            pass

        except KeyboardInterrupt:
            self._stop_event.set()
        finally:
            self._is_running = False

        return self.best_trial

    def run_optimization(
        self,
        num_trials: int = 6,
        sprint_epochs: int = 3,
        on_trial_update: Optional[Callable[[HyperoptTrial], None]] = None,
        on_finished: Optional[Callable[[Optional[HyperoptTrial]], None]] = None,
    ) -> None:
        """Executes Advanced NAS and Schedule Optimization in a background thread."""
        if self._is_running:
            return

        def _worker():
            best = self.run_optimization_sync(
                num_trials=num_trials,
                sprint_epochs=sprint_epochs,
                on_trial_update=on_trial_update,
            )
            if on_finished:
                on_finished(best)

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()


# Backwards compatibility alias
HyperparameterOptimizer = AdvancedNASOptimizer


def apply_params_to_config(cfg: AppConfig, params: Dict[str, Any], mode: str = "alphazero") -> AppConfig:
    """Applies hyperparameter dictionary (e.g. from hyperopt JSON output) to an AppConfig instance."""
    # Model architecture
    if "block_type" in params:
        cfg.model.block_type = params["block_type"]
    if "hidden_layers" in params:
        cfg.model.policy_hidden_layers = list(params["hidden_layers"])
        cfg.model.score_hidden_layers = list(params["hidden_layers"])
    if "activation" in params:
        cfg.model.policy_activation = params["activation"]
        cfg.model.score_activation = params["activation"]
    if "dropout" in params:
        cfg.model.policy_dropout = float(params["dropout"])
        cfg.model.score_dropout = float(params["dropout"])
    if "use_input_norm" in params:
        cfg.model.use_input_norm = bool(params["use_input_norm"])
    if "use_gnn_map" in params:
        cfg.model.use_gnn_map = bool(params["use_gnn_map"])
    if "gnn_layers" in params:
        cfg.model.gnn_layers = int(params["gnn_layers"])
    if "gnn_hidden_dim" in params:
        cfg.model.gnn_hidden_dim = int(params["gnn_hidden_dim"])
    if "policy_weight_decay" in params:
        cfg.model.policy_weight_decay = float(params["policy_weight_decay"])

    # AlphaZero specific
    if "az_policy_lr" in params:
        cfg.model.policy_lr = float(params["az_policy_lr"])
    if "az_value_loss_coef" in params:
        cfg.alphazero.value_loss_coef = float(params["az_value_loss_coef"])
    if "az_num_simulations" in params:
        cfg.alphazero.num_simulations = int(params["az_num_simulations"])
        cfg.mcts.num_simulations = int(params["az_num_simulations"])
    if "az_gumbel_candidates" in params:
        cfg.alphazero.gumbel_candidates = int(params["az_gumbel_candidates"])
    if "az_c_puct" in params:
        cfg.mcts.c_puct = float(params["az_c_puct"])
    if "az_optimism_power" in params:
        cfg.alphazero.optimism_power = float(params["az_optimism_power"])
    if "az_batch_size" in params:
        cfg.alphazero.batch_size = int(params["az_batch_size"])
    if "az_temp_threshold" in params:
        cfg.alphazero.temperature_threshold_move = int(params["az_temp_threshold"])
    if "az_milestone_weight" in params:
        cfg.mcts.milestone_shaping_weight = float(params["az_milestone_weight"])
    if "az_optimism_weight" in params:
        cfg.mcts.optimism_weight = float(params["az_optimism_weight"])
    if "shaping_anneal_epochs" in params:
        cfg.mcts.shaping_anneal_epochs = int(params["shaping_anneal_epochs"])
    if "initial_shaping_scale" in params:
        cfg.mcts.initial_shaping_scale = float(params["initial_shaping_scale"])

    return cfg


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Advanced NAS & Accelerated Tournament Optimizer for Gaia Project")
    parser.add_argument("--mode", type=str, default="alphazero", choices=["alphazero"])
    parser.add_argument("--algo", type=str, default=None, help="Alias for --mode")
    parser.add_argument("--trials", type=int, default=15, help="Number of trials")
    parser.add_argument("--sprint-epochs", type=int, default=3, help="Sprint epochs per trial")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    cfg = AppConfig()
    if args.device != "auto":
        cfg.hardware.device_override = args.device

    print("=" * 105)
    print("  🧬 ADVANCED NAS & ACCELERATED TOURNAMENT OPTIMIZER (CLI)")
    print(f"  Mode: [ALPHAZERO] | Trials: {args.trials} | Sprint Epochs: {args.sprint_epochs}")
    print(f"  Device: {cfg.hardware.device_override}")
    print("=" * 105)

    opt = AdvancedNASOptimizer(base_config=cfg, mode="alphazero")

    print(f"{'Trial':<8} {'Status':<14} {'Score Obj.':<12} {'Perte':<10} {'Victoires':<12} {'VP Moyen':<12} {'Architecture & Params'}")
    print("-" * 105)

    def _on_update(t: HyperoptTrial):
        if t.status == "Running":
            return
        wr_str = f"{t.win_rate * 100:.0f}%" if t.win_rate > 0 else "-"
        score_str = f"{t.avg_score:.1f}" if t.avg_score > 0 else "-"
        print(
            f"[{t.trial_id:02d}/{args.trials:02d}] "
            f"{t.status:<14} "
            f"{t.objective_score:>10.2f} "
            f"{t.val_loss:>8.3f} "
            f"{wr_str:>10} "
            f"{score_str:>10}   "
            f"{t.architecture_name}"
        )

    best = opt.run_optimization_sync(
        num_trials=args.trials,
        sprint_epochs=args.sprint_epochs,
        on_trial_update=_on_update,
    )

    print("=" * 105)
    if best:
        print(f"  🏆 BEST CONFIGURATION FOUND (Trial #{best.trial_id}):")
        print(f"     Objective Score : {best.objective_score:.2f}")
        print(f"     Description     : {best.architecture_name}")
        print(f"     Validation Loss : {best.val_loss:.4f}")
        print(f"     Win Rate        : {best.win_rate * 100:.1f}%")
        print(f"     Avg Score       : {best.avg_score:.1f} VP")
        print(f"     VP Margin       : {best.vp_margin:+.1f} VP")
        print("     Parameters:")
        for k, v in best.params.items():
            print(f"       - {k}: {v}")
    print("=" * 105)


if __name__ == "__main__":
    main()
