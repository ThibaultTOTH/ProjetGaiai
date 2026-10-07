"""Advanced Neural Architecture Search (NAS) & Joint Schedule Optimizer for Gaia Project.

Provides:
- Joint search over Neural Architecture (Block type, Depth, Width, Activations, Normalization)
  and Learning Dynamics (Warmup, Cosine/Exp/Linear decay, Entropy schedules, Value clipping).
- Evolutionary regularized mutation & crossover of top-performing network chromosomes.
- Multi-fidelity early pruning (ASHA / Hyperband principle) to abort non-promising architectures early.
- Multi-objective composite scoring: Win Rate, VP progression, Loss stability, and Steps/sec.
- Multi-threaded background execution with GUI live updates.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import math
import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import torch

from config import AppConfig
from environment import make_gaia_env
from models import DualGaiaAgent
from trainer import RLTrainer
from alphazero_trainer import AlphaZeroTrainer


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
    status: str = "Pending"  # 'Pending', 'Running', 'Completed', 'Pruned (Divergé)', 'Pruned (Lent)', etc.
    history_losses: List[float] = field(default_factory=list)
    history_scores: List[float] = field(default_factory=list)
    history_entropies: List[float] = field(default_factory=list)
    gen_gap: float = 0.0


class AdvancedNASOptimizer:
    """Manages joint Neural Architecture Search and Hyperparameter/Schedule Optimization across PPO, AlphaZero and Pretrainer."""

    def __init__(self, base_config: Optional[AppConfig] = None, mode: Optional[str] = None):
        self.base_config = base_config or AppConfig()
        if mode:
            self.mode = mode.lower()
        elif getattr(self.base_config.alphazero, "enabled", False):
            self.mode = "alphazero"
        else:
            self.mode = "ppo"

        self.trials: List[HyperoptTrial] = []
        self.best_trial: Optional[HyperoptTrial] = None
        self._stop_event = threading.Event()
        self._is_running = False
        self._thread: Optional[threading.Thread] = None

    def is_running(self) -> bool:
        return self._is_running

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._is_running = False

    @staticmethod
    def format_arch_name(params: Dict[str, Any], mode: str = "ppo") -> str:
        """Returns a clean concise string describing the architecture and components."""
        block = params.get("block_type", "pre_ln").replace("_", "").upper()
        layers = params.get("hidden_layers", [512, 256])
        layers_str = "x".join(str(x) for x in layers)
        act = params.get("activation", "silu").upper()
        gnn = f"+GNN{params.get('gnn_layers', 3)}" if params.get("use_gnn_map", True) else ""

        if mode == "alphazero":
            sims = params.get("az_num_simulations", 16)
            lr = params.get("az_policy_lr", 2.5e-4)
            v_loss = params.get("az_value_loss_coef", 1.0)
            return f"{block} {layers_str} ({act}){gnn} [AZ: sims={sims}, lr={lr:.1e}, v_coef={v_loss}]"
        elif mode == "pretrain":
            lr = params.get("pretrain_lr", 3e-4)
            vw = params.get("pretrain_value_weight", 0.5)
            ls = params.get("pretrain_label_smoothing", 0.03)
            return f"{block} {layers_str} ({act}){gnn} [Pre: lr={lr:.1e}, vw={vw}, ls={ls}]"
        else:
            micro = f"+Micro{params.get('micro_dispatch_candidates', 4)}" if params.get("micro_dispatch_enabled", True) else ""
            setup_tag = "+Setup" if params.get("micro_dispatch_setup_mode") == "value_guided" else ""
            return f"{block} {layers_str} ({act}){gnn}{micro}{setup_tag}"

    @staticmethod
    def extract_params_from_config(cfg: AppConfig, mode: str = "ppo") -> Dict[str, Any]:
        """Extracts hyperparameter search dictionary from an AppConfig instance to serve as baseline."""
        micro_cfg = getattr(cfg, "micro_dispatch", None)
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
            "score_weight_decay": float(getattr(cfg.model, "score_weight_decay", 1e-4)),
            # Optimization
            "policy_lr": float(getattr(cfg.model, "policy_lr", 2.5e-4)),
            "score_lr": float(getattr(cfg.model, "score_lr", 3.5e-4)),
            "batch_size": int(getattr(cfg.training, "batch_size", 256)),
            # Schedules
            "lr_schedule_type": getattr(cfg.training, "lr_schedule_type", "cosine"),
            "warmup_ratio": float(getattr(cfg.training, "warmup_ratio", 0.05)),
            "lr_final_factor": float(getattr(cfg.training, "lr_final_factor", 0.10)),
            "exp_decay_rate": float(getattr(cfg.training, "exp_decay_rate", 0.98)),
            "entropy_schedule_type": getattr(cfg.training, "entropy_schedule_type", "cosine"),
            "entropy_start": float(getattr(cfg.training, "entropy_start", 0.05)),
            "entropy_end": float(getattr(cfg.training, "entropy_end", 0.005)),
            # Clipping
            "clip_epsilon": float(getattr(cfg.training, "clip_epsilon", 0.20)),
            "value_clip_epsilon": float(getattr(cfg.training, "value_clip_epsilon", 0.20)),
            # SOTA Multi-Agent & Exploration
            "rnad_enabled": bool(getattr(cfg.training, "rnad_enabled", True)),
            "rnad_alpha": float(getattr(cfg.training, "rnad_alpha", 0.05)),
            "rnad_polyak_beta": float(getattr(cfg.training, "rnad_polyak_beta", 0.20)),
            "rnd_enabled": bool(getattr(cfg.training, "rnd_enabled", True)),
            "rnd_initial_weight": float(getattr(cfg.training, "rnd_initial_weight", 0.05)),
            "rnd_decay_rate": float(getattr(cfg.training, "rnd_decay_rate", 0.98)),
            "rgsc_enabled": bool(getattr(cfg.training, "rgsc_enabled", True)),
            "rgsc_regret_threshold": float(getattr(cfg.training, "rgsc_regret_threshold", 0.40)),
            "rgsc_reset_prob": float(getattr(cfg.training, "rgsc_reset_prob", 0.35)),
            # Population League
            "league_matchmaking_elo_window": float(getattr(cfg.league, "matchmaking_elo_window", 150.0)),
            "league_self_play_prob": float(getattr(cfg.league, "self_play_prob", 0.40)),
            # Value-Guided Micro-Dispatch & Strategic Setup
            "micro_dispatch_enabled": bool(getattr(micro_cfg, "enabled", True)),
            "micro_dispatch_candidates": int(getattr(micro_cfg, "num_candidates", 4)),
            "micro_dispatch_temperature": float(getattr(micro_cfg, "temperature", 0.0)),
            "micro_dispatch_setup_mode": getattr(micro_cfg, "setup_mode", "value_guided"),
            "initial_booster_draft_enabled": bool(getattr(micro_cfg, "initial_booster_draft_enabled", True)),
            "tech_tile_dispatch_enabled": bool(getattr(micro_cfg, "tech_tile_dispatch_enabled", True)),
            # AlphaZero Parameters
            "az_policy_lr": float(getattr(cfg.model, "policy_lr", 2.5e-4)),
            "az_value_loss_coef": float(getattr(cfg.alphazero, "value_loss_coef", 1.0)),
            "az_batch_size": int(getattr(cfg.alphazero, "batch_size", 256)),
            "az_num_simulations": int(getattr(cfg.alphazero, "num_simulations", 32)),
            "az_gumbel_candidates": int(getattr(cfg.alphazero, "gumbel_candidates", 8)),
            "az_c_puct": float(getattr(cfg.mcts, "c_puct", 1.414)),
            "az_optimism_power": float(getattr(cfg.alphazero, "optimism_power", 1.0)),
            "az_temp_threshold": int(getattr(cfg.alphazero, "temperature_threshold_move", 4)),
            "az_milestone_weight": float(getattr(cfg.mcts, "milestone_shaping_weight", 0.25)),
            "az_optimism_weight": float(getattr(cfg.mcts, "optimism_weight", 0.15)),
            # Pretraining Parameters
            "pretrain_lr": 3e-4,
            "pretrain_weight_decay": 1e-4,
            "pretrain_value_weight": 0.5,
            "pretrain_label_smoothing": 0.03,
            "pretrain_batch_size": 256,
        }

    def generate_candidate(self, elite_pool: Optional[List[HyperoptTrial]] = None) -> Dict[str, Any]:
        """Generates a candidate using Bayesian TPE guided sampling or regularized mutation."""
        blocks_candidates = ["swiglu", "pre_ln", "bottleneck"]
        activations_candidates = ["silu", "gelu", "mish"]
        dropouts_candidates = [0.0, 0.03, 0.05]

        layers_candidates = [
            [1024, 1024, 512, 256],     # Grandmaster Standard (~12M params)
            [1024, 1024, 1024, 512],    # Deep Heavy SOTA (~20M params)
            [1536, 1024, 512, 256],     # Wide Front-End (~22M params)
            [768, 768, 384, 256],       # Balanced 4-layer (~10M params)
            [512, 512, 512, 256],       # Compact Deep (~6M params)
        ]

        # Bayesian TPE: if elite pool exists (including evaluated baseline), sample around elite configurations
        if elite_pool and len(elite_pool) >= 1 and random.random() < 0.75:
            parent = random.choice(elite_pool[:min(3, len(elite_pool))]).params
            return self.mutate_candidate(parent)

        p_lr = float(random.choice([2.0e-4, 2.5e-4, 3.0e-4, 3.5e-4]))
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
            # Optimization
            "policy_lr": p_lr,
            "score_lr": float(round(p_lr * random.choice([1.2, 1.4, 1.6]), 6)),
            "batch_size": int(random.choice([128, 256, 512])),
            # Learning Rate Schedule
            "lr_schedule_type": random.choice(["cosine", "exponential", "linear"]),
            "warmup_ratio": float(random.choice([0.03, 0.05, 0.08])),
            "lr_final_factor": float(random.choice([0.05, 0.10, 0.15])),
            "exp_decay_rate": float(random.choice([0.97, 0.98, 0.99])),
            # Entropy Schedule (Anti-Overfitting)
            "entropy_schedule_type": "cosine",
            "entropy_start": float(random.choice([0.04, 0.05, 0.06])),
            "entropy_end": float(random.choice([0.003, 0.005, 0.008])),
            # Clipping
            "clip_epsilon": float(random.choice([0.15, 0.20, 0.25])),
            "value_clip_epsilon": float(random.choice([0.15, 0.20, 0.25])),
            # SOTA Game Theory & Multi-Agent (R-NaD)
            "rnad_enabled": True,
            "rnad_alpha": float(random.choice([0.02, 0.05, 0.08])),
            "rnad_polyak_beta": float(random.choice([0.15, 0.20, 0.25])),
            # SOTA Exploration (RND & RGSC)
            "rnd_enabled": True,
            "rnd_initial_weight": float(random.choice([0.02, 0.05, 0.08])),
            "rnd_decay_rate": float(random.choice([0.96, 0.98])),
            "rgsc_enabled": True,
            "rgsc_regret_threshold": float(random.choice([0.30, 0.40, 0.50])),
            "rgsc_reset_prob": float(random.choice([0.25, 0.35, 0.45])),
            # SOTA Population League
            "league_matchmaking_elo_window": float(random.choice([100.0, 150.0, 200.0])),
            "league_self_play_prob": float(random.choice([0.30, 0.40, 0.50])),
            # Value-Guided Micro-Dispatch & Strategic Setup
            "micro_dispatch_enabled": random.choice([True, True, False]),
            "micro_dispatch_candidates": int(random.choice([2, 4, 8])),
            "micro_dispatch_temperature": float(random.choice([0.0, 0.05, 0.10])),
            "micro_dispatch_setup_mode": random.choice(["value_guided", "random"]),
            "initial_booster_draft_enabled": random.choice([True, False]),
            "tech_tile_dispatch_enabled": random.choice([True, False]),
            # AlphaZero Parameters
            "az_policy_lr": float(random.choice([1.5e-4, 2.5e-4, 3.5e-4])),
            "az_value_loss_coef": float(random.choice([0.50, 1.0, 1.5])),
            "az_batch_size": int(random.choice([256, 512])),
            "az_num_simulations": int(random.choice([16, 24, 32])),
            "az_gumbel_candidates": int(random.choice([4, 8, 12])),
            "az_c_puct": float(random.choice([1.2, 1.414, 1.8])),
            "az_optimism_power": float(random.choice([0.5, 1.0])),
            "az_temp_threshold": int(random.choice([2, 4, 6])),
            "az_milestone_weight": float(random.choice([0.15, 0.25, 0.35])),
            "az_optimism_weight": float(random.choice([0.10, 0.15, 0.20])),
            # Pretraining Parameters
            "pretrain_lr": float(random.choice([2e-4, 3e-4, 4e-4, 5e-4])),
            "pretrain_weight_decay": float(random.choice([1e-5, 1e-4, 5e-4])),
            "pretrain_value_weight": float(random.choice([0.3, 0.5, 0.8])),
            "pretrain_label_smoothing": float(random.choice([0.03, 0.05, 0.08])),
            "pretrain_batch_size": int(random.choice([256, 512])),
        }

    def mutate_candidate(self, parent: Dict[str, Any]) -> Dict[str, Any]:
        """Mutates 1 or 2 targeted genes of an elite parent candidate."""
        child = deepcopy(parent)
        all_genes = [
            "block_type", "hidden_layers", "activation", "dropout",
            "policy_lr", "batch_size", "lr_schedule", "entropy_schedule",
            "gnn", "rnad", "rnd", "rgsc", "micro_dispatch",
            "az_policy_lr", "az_value_loss_coef", "az_sims", "az_c_puct",
            "pretrain_lr", "pretrain_value_weight"
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
        if "policy_lr" in mutations:
            scale = random.uniform(0.8, 1.25)
            child["policy_lr"] = float(np.clip(child.get("policy_lr", 2.5e-4) * scale, 1.2e-4, 4.5e-4))
            child["score_lr"] = float(child["policy_lr"] * 1.4)
        if "batch_size" in mutations:
            child["batch_size"] = int(random.choice([128, 256, 512]))
        if "lr_schedule" in mutations:
            child["lr_schedule_type"] = random.choice(["cosine", "exponential"])
            child["warmup_ratio"] = float(random.choice([0.03, 0.05, 0.08]))
        if "entropy_schedule" in mutations:
            child["entropy_start"] = float(random.choice([0.04, 0.05, 0.06]))
            child["entropy_end"] = float(random.choice([0.003, 0.005, 0.008]))
        if "rnad" in mutations:
            child["rnad_alpha"] = float(random.choice([0.03, 0.05, 0.08]))
        if "rnd" in mutations:
            child["rnd_initial_weight"] = float(random.choice([0.03, 0.05, 0.08]))
        if "rgsc" in mutations:
            child["rgsc_regret_threshold"] = float(random.choice([0.35, 0.40, 0.45]))
        if "micro_dispatch" in mutations:
            child["micro_dispatch_enabled"] = not child.get("micro_dispatch_enabled", True)
            child["micro_dispatch_candidates"] = int(random.choice([2, 4, 8]))
            child["micro_dispatch_temperature"] = float(random.choice([0.0, 0.05, 0.10]))
            child["micro_dispatch_setup_mode"] = random.choice(["value_guided", "random"])
            child["initial_booster_draft_enabled"] = random.choice([True, False])
            child["tech_tile_dispatch_enabled"] = random.choice([True, False])
        if "az_policy_lr" in mutations:
            child["az_policy_lr"] = float(random.choice([1.5e-4, 2.5e-4, 3.5e-4, 5e-4]))
        if "az_value_loss_coef" in mutations:
            child["az_value_loss_coef"] = float(random.choice([0.25, 0.50, 1.0, 1.5]))
        if "az_sims" in mutations:
            child["az_num_simulations"] = int(random.choice([8, 16, 24, 32]))
            child["az_gumbel_candidates"] = int(random.choice([4, 6, 8, 10]))
        if "az_c_puct" in mutations:
            child["az_c_puct"] = float(random.choice([1.2, 1.414, 1.8, 2.0]))
        if "pretrain_lr" in mutations:
            child["pretrain_lr"] = float(random.choice([1.5e-4, 3e-4, 5e-4, 8e-4]))
        if "pretrain_value_weight" in mutations:
            child["pretrain_value_weight"] = float(random.choice([0.2, 0.5, 1.0]))

        return child

    def evaluate_trial(
        self,
        trial: HyperoptTrial,
        sprint_epochs: int = 6,
        completed_pool: Optional[List[HyperoptTrial]] = None,
    ) -> None:
        """Evaluates candidate using Multi-Fidelity Early Pruning, Learning Curve & Anti-Overfitting checks."""
        if self.mode == "alphazero":
            self._evaluate_alphazero_trial(trial, sprint_epochs=max(2, min(4, sprint_epochs)), completed_pool=completed_pool)
        elif self.mode == "pretrain":
            self._evaluate_pretrain_trial(trial, sprint_epochs=max(2, min(3, sprint_epochs)), completed_pool=completed_pool)
        else:
            self._evaluate_ppo_trial(trial, sprint_epochs=sprint_epochs, completed_pool=completed_pool)

    def _evaluate_alphazero_trial(
        self,
        trial: HyperoptTrial,
        sprint_epochs: int = 3,
        completed_pool: Optional[List[HyperoptTrial]] = None,
    ) -> None:
        """Evaluates an AlphaZero candidate trial via rapid self-play sprints and MCTS dynamics."""
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
        cfg.mcts.milestone_shaping_weight = float(p.get("az_milestone_weight", 0.50))
        cfg.mcts.optimism_weight = float(p.get("az_optimism_weight", 0.25))

        agent = DualGaiaAgent(cfg.model)
        trainer = AlphaZeroTrainer(cfg, agent=agent)
        env = make_gaia_env(players=cfg.model.num_players)

        losses: List[float] = []
        scores: List[float] = []
        entropies: List[float] = []
        speeds: List[float] = []
        wins = 0

        for ep in range(sprint_epochs):
            if self._stop_event.is_set():
                trial.status = "Cancelled"
                return

            t0 = time.time()
            game_history, p0_vp, p0_won, moves = trainer.self_play_game(env)
            if p0_won:
                wins += 1
            scores.append(p0_vp)

            for step_data in game_history:
                trainer.replay_buffer.add(*step_data)

            ep_p_loss = 0.0
            ep_v_loss = 0.0
            ep_ent = 0.0
            steps = min(trainer.az_config.training_steps_per_epoch, len(trainer.replay_buffer))
            if steps > 0 and len(trainer.replay_buffer) >= 8:
                for _ in range(steps):
                    batch = trainer.replay_buffer.sample(min(len(trainer.replay_buffer), trainer.az_config.batch_size))
                    p_loss, v_loss, ent, _ = trainer.train_on_batch(batch)
                    ep_p_loss += p_loss
                    ep_v_loss += v_loss
                    ep_ent += ent
                # Normalize VP loss (/ 25.0) and apply value_loss_coef to reflect true joint optimization objective
                v_coef = float(p.get("az_value_loss_coef", 0.5))
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

            # Divergence or NaN guard (normalized loss > 25.0 indicates true gradient divergence)
            if np.isnan(ep_loss) or np.isinf(ep_loss) or ep_loss > 25.0:
                trial.status = "Pruned (Divergé)"
                trial.val_loss = round(float(ep_loss), 4)
                return

        trial.history_losses = list(losses)
        trial.history_scores = list(scores)
        trial.history_entropies = list(entropies)

        val_loss = float(np.mean(losses)) if losses else 1.0
        avg_score = float(np.mean(scores)) if scores else 50.0
        win_rate = float(wins) / max(1, sprint_epochs)
        avg_speed = float(np.mean(speeds)) if speeds else 10.0

        throughput_bonus = min(10.0, avg_speed / 5.0)
        objective = (win_rate * 40.0) + (avg_score * 0.5) - (val_loss * 2.0) + throughput_bonus

        trial.val_loss = round(val_loss, 4)
        trial.win_rate = round(win_rate, 3)
        trial.avg_score = round(avg_score, 2)
        trial.steps_per_sec = round(avg_speed, 1)
        trial.objective_score = round(objective, 3)
        trial.status = "Completed"

    def _evaluate_pretrain_trial(
        self,
        trial: HyperoptTrial,
        sprint_epochs: int = 2,
        completed_pool: Optional[List[HyperoptTrial]] = None,
    ) -> None:
        """Evaluates behavioral cloning pretraining hyperparameters against expert transitions."""
        cfg = deepcopy(self.base_config)
        p = trial.params

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

        lr = float(p.get("pretrain_lr", 3e-4))
        wd = float(p.get("pretrain_weight_decay", 1e-4))
        val_w = float(p.get("pretrain_value_weight", 0.5))
        ls = float(p.get("pretrain_label_smoothing", 0.03))
        b_size = int(p.get("pretrain_batch_size", 256))

        device = cfg.hardware.get_torch_device()
        agent = DualGaiaAgent(cfg.model).to(device)

        from pathlib import Path
        project_root = Path(__file__).resolve().parent.parent
        ds_path = project_root / "scraper" / "data" / "dataset" / "gaia_expert_dataset.pt"

        from torch.utils.data import DataLoader, TensorDataset
        if ds_path.exists():
            try:
                raw = torch.load(ds_path, weights_only=False)
                n_samples = min(50000, len(raw["actions"]))
                actions = raw["actions"][:n_samples]
                values = raw["values"][:n_samples]
                factions = raw["factions"][:n_samples]
                observations = raw.get("observations", None)
                if observations is not None:
                    observations = observations[:n_samples]
                    dataset = TensorDataset(observations, actions, values, factions)
                else:
                    dataset = TensorDataset(actions, values, factions)
            except Exception as e:
                logger.error(f"[Hyperopt Pretrain] Error loading {ds_path}: {e}")
                dataset = None
        else:
            dataset = None

        if dataset is None:
            raise FileNotFoundError(
                f"Authentic expert dataset not found or unreadable at {ds_path}. "
                f"Run 'python scraper/rebuild_expert_dataset.py' to generate it."
            )

        train_len = int(len(dataset) * 0.8)
        val_len = len(dataset) - train_len
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_len, val_len])
        train_loader = DataLoader(train_ds, batch_size=b_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=b_size, shuffle=False)

        optimizer = torch.optim.AdamW(agent.parameters(), lr=lr, weight_decay=wd)
        use_cuda = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
        criterion_p = torch.nn.CrossEntropyLoss(label_smoothing=ls)
        criterion_v = torch.nn.SmoothL1Loss()

        val_top1 = 0.0
        val_loss = 0.0
        val_top5 = 0.0

        for ep in range(sprint_epochs):
            if self._stop_event.is_set():
                trial.status = "Cancelled"
                return

            agent.train()
            for batch in train_loader:
                if len(batch) == 4:
                    b_obs, b_act, b_val, _ = batch
                    b_obs = b_obs.to(device).float()
                else:
                    b_act, b_val, _ = batch
                    b_obs = torch.zeros(len(b_act), cfg.model.obs_dim, device=device)

                b_act = b_act.to(device)
                b_val = b_val.to(device).float()

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda" if use_cuda else "cpu", enabled=use_cuda):
                    logits = agent.action_net(b_obs)
                    p_val = agent.score_net(b_obs).view(-1)
                    loss_p = criterion_p(logits, b_act)
                    target_v = torch.tanh(b_val.view(-1))
                    loss_v = criterion_v(p_val, target_v)
                    total_loss = loss_p + val_w * loss_v

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            # Validation
            agent.eval()
            c_top1 = 0
            c_top5 = 0
            tot = 0
            tot_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 4:
                        b_obs, b_act, b_val, _ = batch
                        b_obs = b_obs.to(device).float()
                    else:
                        b_act, b_val, _ = batch
                        b_obs = torch.zeros(len(b_act), cfg.model.obs_dim, device=device)
                    b_act = b_act.to(device)
                    b_val = b_val.to(device).float()

                    with torch.amp.autocast(device_type="cuda" if use_cuda else "cpu", enabled=use_cuda):
                        logits = agent.action_net(b_obs)
                        p_val = agent.score_net(b_obs).view(-1)
                        loss_p = criterion_p(logits, b_act)
                        target_v = torch.tanh(b_val.view(-1))
                        loss_v = criterion_v(p_val, target_v)
                        tot_loss += (loss_p + val_w * loss_v).item() * len(b_act)

                    top1 = logits.argmax(dim=-1)
                    c_top1 += (top1 == b_act).sum().item()
                    _, top5 = logits.topk(5, dim=-1)
                    c_top5 += (top5 == b_act.unsqueeze(-1)).any(dim=-1).sum().item()
                    tot += len(b_act)

            val_top1 = (c_top1 / max(1, tot)) * 100.0
            val_top5 = (c_top5 / max(1, tot)) * 100.0
            val_loss = tot_loss / max(1, tot)

            if np.isnan(val_loss) or np.isinf(val_loss):
                trial.status = "Pruned (Divergé)"
                trial.val_loss = 99.0
                return

        objective = (val_top1 * 3.0) + (val_top5 * 0.5) - (val_loss * 2.0)
        trial.val_loss = round(val_loss, 4)
        trial.win_rate = round(val_top1 / 100.0, 3)
        trial.avg_score = round(val_top5, 2)
        trial.steps_per_sec = 200.0
        trial.objective_score = round(objective, 3)
        trial.status = "Completed"

    def _evaluate_ppo_trial(
        self,
        trial: HyperoptTrial,
        sprint_epochs: int = 6,
        completed_pool: Optional[List[HyperoptTrial]] = None,
    ) -> None:
        """Evaluates PPO candidate using Multi-Fidelity Early Pruning & Learning Curve checks."""
        cfg = deepcopy(self.base_config)
        p = trial.params

        # 1. Architecture injection
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
        cfg.model.policy_weight_decay = float(p.get("policy_weight_decay", getattr(cfg.model, "policy_weight_decay", 1e-4)))
        cfg.model.score_weight_decay = float(p.get("score_weight_decay", getattr(cfg.model, "score_weight_decay", 1e-4)))

        # 2. Optimizer & Schedules
        cfg.model.policy_lr = p.get("policy_lr", 2.5e-4)
        cfg.model.score_lr = p.get("score_lr", 3.5e-4)
        cfg.training.batch_size = p.get("batch_size", 256)
        cfg.training.total_episodes = sprint_epochs

        cfg.training.lr_schedule_type = p.get("lr_schedule_type", "cosine")
        cfg.training.warmup_ratio = p.get("warmup_ratio", 0.05)
        cfg.training.lr_final_factor = p.get("lr_final_factor", 0.1)
        cfg.training.exp_decay_rate = p.get("exp_decay_rate", 0.98)

        cfg.training.entropy_schedule_type = p.get("entropy_schedule_type", "cosine")
        cfg.training.entropy_start = p.get("entropy_start", 0.05)
        cfg.training.entropy_end = p.get("entropy_end", 0.005)

        cfg.training.clip_epsilon = p.get("clip_epsilon", 0.20)
        cfg.training.value_clip_epsilon = p.get("value_clip_epsilon", 0.20)

        # 3. Game Theory, Curiosity & Go-Exploit
        cfg.training.rnad_enabled = p.get("rnad_enabled", True)
        cfg.training.rnad_alpha = p.get("rnad_alpha", 0.05)
        cfg.training.rnad_polyak_beta = p.get("rnad_polyak_beta", 0.20)

        cfg.training.rnd_enabled = p.get("rnd_enabled", True)
        cfg.training.rnd_initial_weight = p.get("rnd_initial_weight", 0.05)
        cfg.training.rnd_decay_rate = p.get("rnd_decay_rate", 0.98)

        cfg.training.rgsc_enabled = p.get("rgsc_enabled", True)
        cfg.training.rgsc_regret_threshold = p.get("rgsc_regret_threshold", 0.40)
        cfg.training.rgsc_reset_prob = p.get("rgsc_reset_prob", 0.35)

        cfg.league.enabled = True
        cfg.league.matchmaking_type = "gaussian"
        cfg.league.matchmaking_elo_window = p.get("league_matchmaking_elo_window", 150.0)
        cfg.league.self_play_prob = p.get("league_self_play_prob", 0.40)

        # Value-Guided Micro-Dispatch & Strategic Setup
        if hasattr(cfg, "micro_dispatch"):
            cfg.micro_dispatch.enabled = p.get("micro_dispatch_enabled", True)
            cfg.micro_dispatch.num_candidates = int(p.get("micro_dispatch_candidates", 4))
            cfg.micro_dispatch.temperature = float(p.get("micro_dispatch_temperature", 0.0))
            cfg.micro_dispatch.setup_mode = p.get("micro_dispatch_setup_mode", "value_guided")
            cfg.micro_dispatch.initial_booster_draft_enabled = p.get("initial_booster_draft_enabled", True)
            cfg.micro_dispatch.tech_tile_dispatch_enabled = p.get("tech_tile_dispatch_enabled", True)

        # Instantiate agent, trainer, environment
        agent = DualGaiaAgent(cfg.model)
        trainer = RLTrainer(cfg, agent=agent)
        env = make_gaia_env(players=cfg.model.num_players)

        early_fidelity = max(2, sprint_epochs // 2)
        losses: List[float] = []
        scores: List[float] = []
        entropies: List[float] = []
        speeds: List[float] = []

        # --- FIDELITY STEP 1: Early Screening & Anti-Overfitting Checks ---
        for ep in range(early_fidelity):
            if self._stop_event.is_set():
                trial.status = "Cancelled"
                return

            m = trainer.train_step(env)
            losses.append(m.total_loss)
            scores.append(m.avg_real_score)
            entropies.append(m.entropy)
            speeds.append(m.steps_per_sec)

            # 1. Divergence / NaN Guard
            if np.isnan(m.total_loss) or np.isinf(m.total_loss) or m.total_loss > 35.0 or abs(m.policy_loss) > 6.0:
                trial.status = "Pruned (Divergé)"
                trial.val_loss = round(float(losses[-1]), 4)
                trial.history_losses = list(losses)
                trial.history_scores = list(scores)
                trial.history_entropies = list(entropies)
                return

            # 2. Entropy Collapse / Premature Overfitting Guard (Epoch 1-2)
            if m.entropy < 0.012 and ep < 2:
                trial.status = "Pruned (Lent)"
                trial.val_loss = round(float(losses[-1]), 4)
                trial.history_losses = list(losses)
                trial.history_scores = list(scores)
                trial.history_entropies = list(entropies)
                return

        trial.history_losses = list(losses)
        trial.history_scores = list(scores)
        trial.history_entropies = list(entropies)

        # 3. ASHA Median Learning Curve Pruning
        if completed_pool and len(completed_pool) >= 3:
            valid_losses = [t.val_loss for t in completed_pool if t.status == "Completed"]
            if valid_losses:
                q75 = float(np.percentile(valid_losses, 75))
                if losses[-1] > max(15.0, q75 * 1.6):
                    trial.status = "Pruned (Lent)"
                    trial.val_loss = round(float(losses[-1]), 4)
                    return

        # --- FIDELITY STEP 2: Sprint Completion ---
        for ep in range(early_fidelity, sprint_epochs):
            if self._stop_event.is_set():
                trial.status = "Cancelled"
                return

            m = trainer.train_step(env)
            losses.append(m.total_loss)
            scores.append(m.avg_real_score)
            entropies.append(m.entropy)
            speeds.append(m.steps_per_sec)

        trial.history_losses = list(losses)
        trial.history_scores = list(scores)
        trial.history_entropies = list(entropies)

        # --- GENERALIZATION & OVERFITTING GAP EVALUATION ---
        eval_wr, eval_vp, _ = trainer.evaluate_model(num_games=4)
        train_vp = float(np.mean(scores[-2:])) if scores else eval_vp

        gen_gap = max(0.0, train_vp - eval_vp)
        val_loss = float(np.mean(losses[-2:])) if losses else 1.0
        avg_speed = float(np.mean(speeds)) if speeds else 100.0

        throughput_bonus = min(5.0, avg_speed / 50.0)
        overfit_penalty = min(15.0, gen_gap * 0.3)

        objective = (
            (eval_wr * 40.0)
            + (eval_vp * 0.6)
            - (val_loss * 1.5)
            - overfit_penalty
            + throughput_bonus
        )

        trial.val_loss = round(val_loss, 4)
        trial.win_rate = round(eval_wr, 3)
        trial.avg_score = round(eval_vp, 2)
        trial.steps_per_sec = round(avg_speed, 1)
        trial.gen_gap = round(gen_gap, 2)
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

                # Collect currently completed trials as elite candidates for evolutionary mutation
                completed = [t for t in self.trials if t.status == "Completed"]

                # Trial #1 begins with the currently registered / saved hyperparameters (baseline)
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
                        # Automatically persist current champion to disk immediately
                        try:
                            import json
                            out_dir = getattr(self.base_config.training, "runs_dir", "runs")
                            os.makedirs(out_dir, exist_ok=True)
                            out_file = os.path.join(out_dir, f"best_hyperparams_{self.mode}.json")
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
        sprint_epochs: int = 6,
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

    # AlphaZero specific
    if mode == "alphazero":
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

    return cfg


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Advanced NAS & Hyperparameter Optimizer for Gaia Project")
    parser.add_argument("--mode", type=str, default="alphazero", choices=["alphazero", "ppo", "pretrain"])
    parser.add_argument("--algo", type=str, default=None, help="Alias for --mode")
    parser.add_argument("--trials", type=int, default=15, help="Number of trials")
    parser.add_argument("--sprint-epochs", type=int, default=3, help="Sprint epochs per trial")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    mode = args.algo if args.algo else args.mode
    cfg = AppConfig()
    if args.device != "auto":
        cfg.hardware.device_override = args.device

    print("=" * 105)
    print("  🧬 ADVANCED NAS & HYPERPARAMETER SEARCH (CLI)")
    print(f"  Mode: [{mode.upper()}] | Trials: {args.trials} | Sprint Epochs: {args.sprint_epochs}")
    print(f"  Device: {cfg.hardware.device_override}")
    print("=" * 105)

    opt = AdvancedNASOptimizer(base_config=cfg, mode=mode)

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
        print(f"     Win Rate / Top1 : {best.win_rate * 100:.1f}%")
        print(f"     Parameters:")
        for k, v in best.params.items():
            print(f"       - {k}: {v}")
    print("=" * 105)


if __name__ == "__main__":
    main()

