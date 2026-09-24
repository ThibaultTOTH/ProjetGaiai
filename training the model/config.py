"""Configuration module for Gaia Project Deep RL training.

Handles:
- CUDA / RTX 5070 GPU detection & hardware acceleration settings.
- CPU fallback for development/testing machines without a dedicated GPU.
- Model architecture hyperparameters (Score Predictor & Action Optimizer).
- Reinforcement learning training hyperparameters (PPO / Actor-Critic).
- Serialization to/from JSON/YAML.
"""

from dataclasses import asdict, dataclass, field
import json
import os
from typing import Any, Dict, List, Optional
import torch


@dataclass
class HardwareConfig:
    """Hardware settings with RTX 5070 12GB optimizations and CPU fallback."""

    device_override: str = "auto"  # 'auto', 'cuda', 'cuda:0', 'cpu'
    use_mixed_precision: bool = True  # FP16 / BF16 automatic mixed precision
    enable_tf32: bool = True  # Accelerate matrix multiplies on Ampere/Ada/Blackwell
    enable_cudnn_benchmark: bool = True
    num_workers: int = 4
    pin_memory: bool = True

    def get_torch_device(self) -> torch.device:
        """Determines the torch device, applying user override or auto-detection."""
        if self.device_override == "cpu":
            return torch.device("cpu")
        if self.device_override.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(self.device_override)
            print(f"[HardwareConfig] âš ï¸ WARNING: Device '{self.device_override}' demandÃ© mais torch.cuda.is_available() est False. Bascule automatique sur CPU.")
            return torch.device("cpu")

        # Auto-detect
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def configure_cuda(self) -> Dict[str, Any]:
        """Applies Blackwell/Ada/Ampere optimizations if CUDA is active."""
        info = {
            "cuda_available": torch.cuda.is_available(),
            "device_name": "CPU",
            "vram_total_gb": 0.0,
            "vram_free_gb": 0.0,
            "mixed_precision": False,
            "tf32_active": False,
        }

        if torch.cuda.is_available():
            device = self.get_torch_device()
            if device.type == "cuda":
                dev_idx = device.index if device.index is not None else 0
                props = torch.cuda.get_device_properties(dev_idx)
                total_vram = props.total_memory / (1024**3)
                free_vram = (
                    torch.cuda.mem_get_info(dev_idx)[0] / (1024**3)
                    if hasattr(torch.cuda, "mem_get_info")
                    else total_vram
                )

                info["device_name"] = props.name
                info["vram_total_gb"] = round(total_vram, 2)
                info["vram_free_gb"] = round(free_vram, 2)
                info["mixed_precision"] = self.use_mixed_precision

                if self.enable_tf32:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    info["tf32_active"] = True

                if self.enable_cudnn_benchmark:
                    torch.backends.cudnn.benchmark = True

        return info


@dataclass
class ModelConfig:
    """Hyperparameters for Score Predictor and Action Optimizer networks."""

    num_players: int = 4  # Standard competitive Gaia Project table (4 players 1v1v1v1)
    obs_dim: int = 2476  # Exhaustive observation: Global 88 + 4 Players 388 + Map 2000
    action_dim: int = 3130  # Discrete action space size (Gaia Project + Lost Fleet)

    # Score Predictor (Value Network) - Large Pro ResNet
    score_hidden_layers: List[int] = field(
        default_factory=lambda: [512, 512, 512, 256]
    )
    score_dropout: float = 0.05
    score_activation: str = "silu"  # 'silu', 'gelu', 'mish', 'relu'
    score_loss_type: str = "huber"  # 'huber', 'mse', 'smooth_l1'
    score_lr: float = 5e-5  # Réduit pour empêcher l'overfitting précoce de la Baseline
    score_weight_decay: float = 1e-4

    # Action Optimizer (Policy Network) - Large Pro ResNet
    policy_hidden_layers: List[int] = field(
        default_factory=lambda: [512, 512, 512, 256]
    )
    policy_dropout: float = 0.05
    policy_activation: str = "silu"
    policy_lr: float = 3e-4
    policy_weight_decay: float = 1e-4

    # Architecture block type & normalization
    block_type: str = "pre_ln"  # 'pre_ln', 'bottleneck', 'swiglu'
    use_input_norm: bool = True

    # Spatial GNN Map Encoder (Hexagonal message passing on 200 board hexes)
    use_gnn_map: bool = True
    gnn_hidden_dim: int = 64
    gnn_layers: int = 3
    gnn_dropout: float = 0.05

    # Shared or separate backbones
    use_shared_backbone: bool = False


@dataclass
class TrainingConfig:
    """PPO / Actor-Critic training hyperparameters (Unbridled for High-Capacity Models & GPU)."""

    total_episodes: int = 2000
    rollout_steps_per_epoch: int = 1024  # High capacity rollout buffer
    batch_size: int = 256  # Satures RTX 5070 Tensor Cores with TF32/AMP
    train_epochs_per_rollout: int = 4

    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # Generalized Advantage Estimation lambda
    clip_epsilon: float = 0.2  # PPO clip ratio
    entropy_coef: float = 0.04  # Fallback entropy coef
    value_loss_coef: float = 0.5  # Weight of value loss in joint training
    max_grad_norm: float = 0.5  # Gradient clipping

    # Learning rate schedule parameters
    lr_schedule_type: str = "cosine"  # 'cosine', 'exponential', 'linear', 'constant'
    warmup_ratio: float = 0.05  # Ratio of total epochs for linear warmup
    lr_final_factor: float = 0.1  # Ratio of initial LR at final epoch
    exp_decay_rate: float = 0.98  # Per-epoch multiplier for exponential decay
    min_lr: float = 1e-5

    # Entropy decay schedule parameters
    entropy_schedule_type: str = "cosine"  # 'cosine', 'exponential', 'linear', 'constant'
    entropy_start: float = 0.15  # Forcer une énorme exploration au début (Espace d'action 3130)
    entropy_end: float = 0.01  # Exploitation floor entropy

    # PPO Value loss clipping (calibrated for 0..150 VP board game score range)
    value_clip_epsilon: float = 10.0

    # Dynamic Annealed Reward Shaping (Potential-based milestone exploration)
    shaping_enabled: bool = True
    shaping_initial_weight: float = 1.0
    shaping_decay_rate: float = 0.998  # Ralenti drastiquement pour s'adapter à 2000 époques
    shaping_min_weight: float = 0.0

    # Random Network Distillation (RND) Intrinsic Curiosity
    rnd_enabled: bool = True
    rnd_initial_weight: float = 0.05
    rnd_decay_rate: float = 0.998
    rnd_learning_rate: float = 1e-4

    # Opponent Modeling Auxiliary Task
    use_opponent_modeling: bool = True
    opponent_loss_coef: float = 0.25

    # Regularized Nash Dynamics (R-NaD) for Multi-Player 1v1v1v1 Convergence
    rnad_enabled: bool = True
    rnad_alpha: float = 0.05
    rnad_ref_update_interval: int = 10
    rnad_polyak_beta: float = 0.20

    # Regret-Guided Search Control (RGSC / Go-Exploit)
    rgsc_enabled: bool = True
    rgsc_regret_threshold: float = 0.40
    rgsc_buffer_capacity: int = 200
    rgsc_reset_prob: float = 0.35

    # Checkpoint and evaluation
    eval_interval_episodes: int = 50
    eval_games_count: int = 10
    save_checkpoint_interval: int = 100
    checkpoint_dir: str = "checkpoints"
    runs_dir: str = "runs"


@dataclass
class LeagueConfig:
    """Configuration for Population-based League Training against historical versions."""

    enabled: bool = True
    snapshot_interval_epochs: int = 10  # Automatically snapshot current policy into league
    max_snapshots: int = 20  # Maximum historical versions kept in pool
    self_play_prob: float = 0.40  # Probability an opponent is current policy clone
    historical_prob: float = 0.50  # Probability an opponent is a sampled past snapshot
    random_prob: float = 0.10  # Probability an opponent is a random legal player
    matchmaking_type: str = "gaussian"  # 'gaussian' (Elo-window matchmaking) or 'uniform'
    matchmaking_elo_window: float = 150.0  # Standard deviation for Gaussian Elo matchmaking
    league_dir: str = "league_checkpoints"


@dataclass
class MCTSConfig:
    """Configuration for Multi-Player Monte-Carlo Tree Search (AlphaZero PUCT & Gumbel GAZ)."""

    enabled: bool = False  # Off by default during basic PPO, activated for eval/tournament
    algorithm: str = "gumbel"  # 'gumbel' (TSS GAZ 2026, 8-16 sims) or 'puct' (AlphaZero standard)
    num_simulations: int = 16  # 16 is optimal for Gumbel GAZ, 50-100 for PUCT
    gumbel_candidates: int = 4  # Number of candidate actions retained in Sequential Halving
    c_puct: float = 1.414  # Exploration constant for PUCT fallback
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    temperature: float = 1.0

    # Epistemic Uncertainty Guidance (MC-Dropout) & Resource-Efficient Search Budget
    use_epistemic_uncertainty: bool = True
    mc_dropout_passes: int = 4
    uncertainty_scale: float = 0.50
    adaptive_budget_enabled: bool = True
    entropy_threshold: float = 0.15
    min_simulations: int = 2


@dataclass
class AsyncConfig:
    """Configuration for Asynchronous Distributed Actor-Learner training (APPO / IMPALA)."""

    enabled: bool = False
    num_actors: int = 4
    queue_max_size: int = 16
    sync_weights_interval: int = 10
    staleness_threshold: int = 3


@dataclass
class MicroDispatchConfig:
    """Configuration for Value-Guided Micro-Action Dispatch (Hex, Track, Booster & Tech Selection)."""

    enabled: bool = True
    num_candidates: int = 4  # Number of candidate targets evaluated by ScorePredictorNet
    temperature: float = 0.0  # 0.0 = greedy argmax, > 0.0 = Boltzmann exploration
    value_weight: float = 1.0  # Scaling factor for value guidance
    setup_mode: str = "value_guided"  # 'value_guided', 'random', 'deterministic'
    initial_booster_draft_enabled: bool = True  # Reverse-order draft for starting round booster
    tech_tile_dispatch_enabled: bool = True  # Strategic selection of tech tiles on Lab/Academy/Claim
    federation_dispatch_enabled: bool = True  # Strategic selection of federation token on FormFederation


@dataclass
class AppConfig:
    """Root configuration aggregator."""

    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    league: LeagueConfig = field(default_factory=LeagueConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    async_dist: AsyncConfig = field(default_factory=AsyncConfig)
    micro_dispatch: MicroDispatchConfig = field(default_factory=MicroDispatchConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_to_file(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "AppConfig":
        if not os.path.exists(filepath):
            return cls()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = cls()
        if "hardware" in data:
            cfg.hardware = HardwareConfig(**data["hardware"])
        if "model" in data:
            cfg.model = ModelConfig(**data["model"])
        if "training" in data:
            cfg.training = TrainingConfig(**data["training"])
        if "league" in data:
            cfg.league = LeagueConfig(**data["league"])
        if "mcts" in data:
            cfg.mcts = MCTSConfig(**data["mcts"])
        if "async_dist" in data:
            cfg.async_dist = AsyncConfig(**data["async_dist"])
        if "micro_dispatch" in data:
            import inspect
            sig_params = inspect.signature(MicroDispatchConfig).parameters
            filtered = {k: v for k, v in data["micro_dispatch"].items() if k in sig_params}
            cfg.micro_dispatch = MicroDispatchConfig(**filtered)
        return cfg


def get_training_preset(name: str = "grandmaster") -> AppConfig:
    """Returns a production-ready, scientifically calibrated configuration preset.

    Supported presets:
    - 'fast': Rapid validation / debug run (50 epochs, lightweight MLP, ~3-5 min).
    - 'pro': Competitive tournament level (1,000 epochs, SwiGLU + HexGNN, PPO + R-NaD + RND, ~1.5h).
    - 'grandmaster': Absolute SOTA Top 1% (2,500 epochs, HexGNN 3-layer + SwiGLU 512x512x256,
                    RGSC Go-Exploit + R-NaD + RND + Gaussian League ZPD + Gumbel GAZ MCTS + AMP, ~4-6h).
    """
    preset = name.lower().strip()
    cfg = AppConfig()

    if preset in ("fast", "debug", "test"):
        cfg.training.total_episodes = 50
        cfg.training.rollout_steps_per_epoch = 512
        cfg.training.batch_size = 128
        cfg.model.block_type = "pre_ln"
        cfg.model.policy_hidden_layers = [256, 128]
        cfg.model.score_hidden_layers = [256, 128]
        cfg.model.use_gnn_map = False
        cfg.training.shaping_decay_rate = 0.90
        cfg.league.snapshot_interval_epochs = 10
        cfg.training.eval_interval_episodes = 25
        cfg.training.save_checkpoint_interval = 25
        cfg.micro_dispatch.enabled = False

    elif preset in ("pro", "competitive"):
        cfg.training.total_episodes = 1000
        cfg.training.rollout_steps_per_epoch = 1024
        cfg.training.batch_size = 256
        cfg.model.block_type = "swiglu"
        cfg.model.policy_hidden_layers = [512, 512, 256]
        cfg.model.score_hidden_layers = [512, 512, 256]
        cfg.model.use_gnn_map = True
        cfg.model.gnn_layers = 3
        cfg.model.gnn_hidden_dim = 64
        cfg.training.shaping_enabled = True
        cfg.training.shaping_decay_rate = 0.96
        cfg.training.rnd_enabled = True
        cfg.training.rnad_enabled = True
        cfg.training.rgsc_enabled = True
        cfg.training.use_opponent_modeling = True
        cfg.league.enabled = True
        cfg.league.matchmaking_type = "gaussian"
        cfg.league.matchmaking_elo_window = 150.0
        cfg.mcts.enabled = True
        cfg.mcts.algorithm = "gumbel"
        cfg.mcts.num_simulations = 16
        cfg.micro_dispatch.enabled = True
        cfg.micro_dispatch.num_candidates = 2
        cfg.micro_dispatch.temperature = 0.0
    elif preset in ("double_descent", "overparameterized", "wide"):
        cfg.training.total_episodes = 1200
        cfg.training.rollout_steps_per_epoch = 1024
        cfg.training.batch_size = 256
        cfg.model.block_type = "swiglu"
        # Deep Over-Parameterized Regime (~16M parameters, safely past the interpolation peak)
        cfg.model.policy_hidden_layers = [1024, 1024, 512, 256]
        cfg.model.score_hidden_layers = [1024, 1024, 512, 256]
        cfg.model.policy_dropout = 0.03
        cfg.model.score_dropout = 0.03
        cfg.model.policy_weight_decay = 2e-4  # Essential L2 regularization for flat minima
        cfg.model.score_weight_decay = 2e-4
        cfg.model.use_gnn_map = True
        cfg.model.gnn_layers = 3
        cfg.model.gnn_hidden_dim = 64
        cfg.training.lr_schedule_type = "cosine"
        cfg.training.warmup_ratio = 0.05
        cfg.training.lr_final_factor = 0.05
        cfg.training.entropy_schedule_type = "cosine"
        cfg.training.entropy_start = 0.05
        cfg.training.entropy_end = 0.003
        cfg.training.shaping_enabled = True
        cfg.training.shaping_decay_rate = 0.96
        cfg.training.rnd_enabled = True
        cfg.training.rnad_enabled = True
        cfg.training.rnad_alpha = 0.05
        cfg.training.rnad_polyak_beta = 0.20
        cfg.training.rgsc_enabled = True
        cfg.league.enabled = True
        cfg.league.matchmaking_type = "gaussian"
        cfg.league.matchmaking_elo_window = 150.0
        cfg.mcts.enabled = True
        cfg.mcts.algorithm = "gumbel"
        cfg.mcts.num_simulations = 16
        cfg.micro_dispatch.enabled = True
        cfg.micro_dispatch.num_candidates = 4
        cfg.micro_dispatch.temperature = 0.05
        cfg.hardware.use_mixed_precision = True
        cfg.hardware.enable_tf32 = True

    else:  # 'grandmaster', 'top1%', or default
        cfg.training.total_episodes = 2500
        cfg.training.rollout_steps_per_epoch = 1024
        cfg.training.batch_size = 256
        cfg.model.block_type = "swiglu"
        cfg.model.policy_hidden_layers = [512, 512, 512, 256]
        cfg.model.score_hidden_layers = [512, 512, 512, 256]
        cfg.model.policy_dropout = 0.05
        cfg.model.score_dropout = 0.05
        cfg.model.use_gnn_map = True
        cfg.model.gnn_layers = 3
        cfg.model.gnn_hidden_dim = 64
        cfg.training.lr_schedule_type = "cosine"
        cfg.training.warmup_ratio = 0.05
        cfg.training.lr_final_factor = 0.10
        cfg.training.entropy_schedule_type = "cosine"
        cfg.training.entropy_start = 0.05
        cfg.training.entropy_end = 0.005
        cfg.training.shaping_enabled = True
        cfg.training.shaping_initial_weight = 1.0
        cfg.training.shaping_decay_rate = 0.96
        cfg.training.rnd_enabled = True
        cfg.training.rnd_initial_weight = 0.05
        cfg.training.rnad_enabled = True
        cfg.training.rnad_alpha = 0.05
        cfg.training.rnad_polyak_beta = 0.20
        cfg.training.rnad_ref_update_interval = 10
        cfg.training.rgsc_enabled = True
        cfg.training.rgsc_regret_threshold = 0.40
        cfg.training.rgsc_buffer_capacity = 200
        cfg.training.rgsc_reset_prob = 0.35
        cfg.training.use_opponent_modeling = True
        cfg.training.opponent_loss_coef = 0.25
        cfg.league.enabled = True
        cfg.league.matchmaking_type = "gaussian"
        cfg.league.matchmaking_elo_window = 150.0
        cfg.mcts.enabled = True
        cfg.mcts.algorithm = "gumbel"
        cfg.mcts.num_simulations = 16
        cfg.mcts.gumbel_candidates = 4
        cfg.mcts.use_epistemic_uncertainty = True
        cfg.mcts.mc_dropout_passes = 4
        cfg.mcts.uncertainty_scale = 0.50
        cfg.mcts.adaptive_budget_enabled = True
        cfg.mcts.entropy_threshold = 0.15
        cfg.mcts.min_simulations = 2
        cfg.hardware.use_mixed_precision = True
        cfg.hardware.enable_tf32 = True
        cfg.micro_dispatch.enabled = True
        cfg.micro_dispatch.num_candidates = 4
        cfg.micro_dispatch.temperature = 0.05

    return cfg


