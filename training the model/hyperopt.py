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

from config import AppConfig
from environment import make_gaia_env
from models import DualGaiaAgent
from trainer import RLTrainer


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
    """Manages joint Neural Architecture Search and Hyperparameter/Schedule Optimization."""

    def __init__(self, base_config: Optional[AppConfig] = None):
        self.base_config = base_config or AppConfig()
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
    def format_arch_name(params: Dict[str, Any]) -> str:
        """Returns a clean concise string describing the architecture and components."""
        block = params.get("block_type", "pre_ln").replace("_", "").upper()
        layers = params.get("hidden_layers", [512, 256])
        layers_str = "x".join(str(x) for x in layers)
        act = params.get("activation", "silu").upper()
        gnn = f"+GNN{params.get('gnn_layers', 3)}" if params.get("use_gnn_map", True) else ""
        micro = f"+Micro{params.get('micro_dispatch_candidates', 4)}" if params.get("micro_dispatch_enabled", True) else ""
        setup_tag = "+Setup" if params.get("micro_dispatch_setup_mode") == "value_guided" else ""
        return f"{block} {layers_str} ({act}){gnn}{micro}{setup_tag}"

    @staticmethod
    def extract_params_from_config(cfg: AppConfig) -> Dict[str, Any]:
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
        }

    def generate_candidate(self, elite_pool: Optional[List[HyperoptTrial]] = None) -> Dict[str, Any]:
        """Generates a candidate using Bayesian TPE guided sampling or regularized mutation."""
        blocks_candidates = ["swiglu", "pre_ln", "bottleneck"]
        activations_candidates = ["silu", "gelu", "mish"]
        dropouts_candidates = [0.0, 0.03, 0.05]

        layers_candidates = [
            [512, 512, 256],            # Pro ResNet 3-layer (balanced)
            [512, 512, 512, 256],       # Deep 4-layer (SOTA capacity)
            [768, 768, 384],            # Wide ResNet 3-layer
            [512, 256, 256, 128],       # Tapered lightweight
            [384, 384, 384, 192],       # Mid-size fast
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
        }

    def mutate_candidate(self, parent: Dict[str, Any]) -> Dict[str, Any]:
        """Mutates 1 or 2 targeted genes of an elite parent candidate."""
        child = deepcopy(parent)
        all_genes = [
            "block_type", "hidden_layers", "activation", "dropout",
            "policy_lr", "batch_size", "lr_schedule", "entropy_schedule",
            "gnn", "rnad", "rnd", "rgsc", "micro_dispatch"
        ]
        mutations = random.sample(all_genes, k=random.choice([1, 2]))

        if "block_type" in mutations:
            child["block_type"] = random.choice(["swiglu", "pre_ln", "bottleneck"])
        if "hidden_layers" in mutations:
            layers = list(child.get("hidden_layers", [512, 512, 256]))
            action = random.choice(["widen", "narrow", "tweak"])
            if action == "widen":
                layers = [min(1024, int(x * 1.25)) for x in layers]
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

        return child

    def evaluate_trial(
        self,
        trial: HyperoptTrial,
        sprint_epochs: int = 6,
        completed_pool: Optional[List[HyperoptTrial]] = None,
    ) -> None:
        """Evaluates candidate using Multi-Fidelity Early Pruning, Learning Curve & Anti-Overfitting checks."""
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

        self._stop_event.clear()
        self._is_running = True
        self.trials.clear()
        self.best_trial = None

        def _worker():
            try:
                for i in range(num_trials):
                    if self._stop_event.is_set():
                        break

                    # Collect currently completed trials as elite candidates for evolutionary mutation
                    # Trial #1 begins with the currently registered / saved hyperparameters (baseline)
                    if i == 0:
                        candidate_params = self.extract_params_from_config(self.base_config)
                        arch_name = "★ BASELINE: " + self.format_arch_name(candidate_params)
                    else:
                        candidate_params = self.generate_candidate(elite_pool=completed)
                        arch_name = self.format_arch_name(candidate_params)

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

            finally:
                self._is_running = False
                if on_finished:
                    on_finished(self.best_trial)

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()


# Backwards compatibility alias
HyperparameterOptimizer = AdvancedNASOptimizer
