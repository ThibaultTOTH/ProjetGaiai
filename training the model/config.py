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
                    # Modern PyTorch 2.9+ syntax with backward compatibility
                    try:
                        torch.backends.cuda.matmul.fp32_precision = "tf32"
                    except Exception:
                        torch.backends.cuda.matmul.allow_tf32 = True
                    try:
                        torch.backends.cudnn.conv.fp32_precision = "tf32"
                    except Exception:
                        try:
                            torch.backends.cudnn.allow_tf32 = True
                        except Exception:
                            pass
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
    
    # Fine-tuning mode: freeze backbone
    finetune_mode: bool = False
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

    gamma: float = 1.0  # Board game, finite horizon
    gae_lambda: float = 0.98  # Generalized Advantage Estimation lambda
    clip_epsilon: float = 0.2  # PPO clip ratio
    entropy_coef: float = 0.001  # Fallback entropy coef
    value_loss_coef: float = 0.5  # Weight of value loss in joint training
    max_grad_norm: float = 0.5  # Gradient clipping

    # Learning rate schedule parameters
    lr_schedule_type: str = "cosine"  # 'cosine', 'exponential', 'linear', 'constant'
    warmup_ratio: float = 0.05  # Ratio of total epochs for linear warmup
    lr_final_factor: float = 0.1  # Ratio of initial LR at final epoch
    exp_decay_rate: float = 0.98  # Per-epoch multiplier for exponential decay
    min_lr: float = 1e-5

    # Entropy decay schedule parameters
    entropy_schedule_type: str = "cosine"
    entropy_start: float = 0.005  # Massive reduction due to ln(3130) max entropy
    entropy_end: float = 0.0001

    # PPO Value loss clipping (calibrated for 0..150 VP board game score range)
    value_clip_epsilon: float = 10.0

    # Dynamic Annealed Reward Shaping (Potential-based milestone exploration)
    shaping_enabled: bool = True
    shaping_initial_weight: float = 50.0
    shaping_decay_rate: float = 0.998  # Ralenti drastiquement pour s'adapter à 2000 époques
    shaping_min_weight: float = 0.0

    # Random Network Distillation (RND) Intrinsic Curiosity
    rnd_enabled: bool = True
    rnd_initial_weight: float = 0.05
    rnd_decay_rate: float = 0.998
    rnd_learning_rate: float = 1e-4

    # Opponent Modeling Auxiliary Task
    use_opponent_modeling: bool = False
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

    # Epistemic Uncertainty Guidance (MC-Dropout) - Désactivé par défaut en self-play pour vitesse maximale (x5)
    use_epistemic_uncertainty: bool = False
    mc_dropout_passes: int = 1
    uncertainty_scale: float = 0.50
    adaptive_budget_enabled: bool = True
    entropy_threshold: float = 0.15
    min_simulations: int = 2
    optimism_weight: float = 0.25  # Pousse le modèle vers l'optimisme (recherche de branches à fort VP)


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
class AlphaZeroConfig:
    enabled: bool = False
    num_simulations: int = 50
    games_per_epoch: int = 10
    training_steps_per_epoch: int = 100
    replay_buffer_size: int = 100_000
    batch_size: int = 256
    value_loss_coef: float = 1.0
    temperature_threshold_move: int = 30
    temperature_high: float = 1.0
    temperature_low: float = 0.1
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    optimism_power: float = 1.0  # Échantillonnage priorisé optimiste des parties à haut score VP
    checkpoint_interval: int = 50

@dataclass
class MuZeroConfig:
    enabled: bool = False
    hidden_dim: int = 256
    num_dynamics_blocks: int = 4
    unroll_steps: int = 5
    num_simulations: int = 50
    games_per_epoch: int = 10
    training_steps_per_epoch: int = 200
    replay_buffer_size: int = 50_000
    batch_size: int = 128
    value_loss_coef: float = 0.25
    reward_loss_coef: float = 1.0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    temperature_threshold_move: int = 30
    checkpoint_interval: int = 50

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
    alphazero: AlphaZeroConfig = field(default_factory=AlphaZeroConfig)
    muzero: MuZeroConfig = field(default_factory=MuZeroConfig)

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
        if "alphazero" in data:
            cfg.alphazero = AlphaZeroConfig(**data["alphazero"])
        if "muzero" in data:
            cfg.muzero = MuZeroConfig(**data["muzero"])
        return cfg


def get_training_preset(name: str = "pretrain") -> AppConfig:
    """Returns a production-ready, scientifically calibrated configuration preset.

    Supported presets:
    - 'pretrain' (or 'fondation', 'phase1'):
        Phase 1 Foundation Model (~150 VP). AlphaZero + MCTS (50 simulations),
        Over-parameterized SwiGLU [1024, 1024, 512, 256] with HexGNN, unfreezed backbone,
        Value-Guided Micro-Dispatch, FP16/TF32 enabled. Clean, parasite-free training.
    - 'finetune' (or 'grandmaster', 'phase2'):
        Phase 2 Grandmaster Model (230+ VP). AlphaZero with frozen shared backbone,
        deepened MCTS (100 simulations), multi-agent PBT League with Gaussian matchmaking,
        RGSC Go-Exploit crisis puzzles, surgical deterministic micro-dispatch.
    - 'fast': Rapid validation / debug run (20 epochs, lightweight).
    """
    preset = name.lower().strip()
    cfg = AppConfig()

    if preset in ("finetune", "phase2", "grandmaster", "gm"):
        # 👑 PHASE 2 : Fine-Tuning Grand Maître (Cible 220 - 240+ VP)
        cfg.alphazero.enabled = True
        cfg.alphazero.num_simulations = 32
        cfg.alphazero.gumbel_candidates = 4
        cfg.alphazero.games_per_epoch = 4
        cfg.alphazero.training_steps_per_epoch = 100
        cfg.alphazero.temperature_high = 0.5
        cfg.alphazero.temperature_low = 0.05
        cfg.alphazero.dirichlet_eps = 0.15

        cfg.mcts.enabled = True
        cfg.mcts.algorithm = "gumbel"
        cfg.mcts.num_simulations = 32
        cfg.mcts.gumbel_candidates = 4
        cfg.mcts.use_epistemic_uncertainty = False

        cfg.model.block_type = "swiglu"
        cfg.model.policy_activation = "silu"
        cfg.model.score_activation = "silu"
        cfg.model.policy_dropout = 0.05
        cfg.model.score_dropout = 0.05
        cfg.model.policy_hidden_layers = [1024, 1024, 512, 256]
        cfg.model.score_hidden_layers = [1024, 1024, 512, 256]
        cfg.model.policy_weight_decay = 2e-4
        cfg.model.score_weight_decay = 2e-4
        cfg.model.use_gnn_map = True
        cfg.model.gnn_layers = 3
        cfg.model.gnn_hidden_dim = 64
        cfg.model.finetune_mode = True  # GÈLE LE SHARED BACKBONE !

        cfg.model.policy_lr = 3e-5
        cfg.model.score_lr = 1e-4
        cfg.training.batch_size = 256
        cfg.training.total_episodes = 10000

        # Puzzles de crise tactiques
        cfg.training.rgsc_enabled = True
        cfg.training.rgsc_regret_threshold = 0.30
        cfg.training.rgsc_buffer_capacity = 200
        cfg.training.rgsc_reset_prob = 0.25

        # Self-Play en Ligue PBT
        cfg.league.enabled = True
        cfg.league.matchmaking_type = "gaussian"
        cfg.league.matchmaking_elo_window = 120.0
        cfg.league.snapshot_interval_epochs = 50
        cfg.league.max_snapshots = 15

        # Pas de bruit parasite
        cfg.training.shaping_enabled = False
        cfg.training.rnd_enabled = False
        cfg.training.rnad_enabled = False
        cfg.training.use_opponent_modeling = False

        # Micro-Dispatch déterministe
        cfg.micro_dispatch.enabled = True
        cfg.micro_dispatch.num_candidates = 4
        cfg.micro_dispatch.temperature = 0.0

        # Hardware RTX
        cfg.hardware.use_mixed_precision = True
        cfg.hardware.enable_tf32 = True

    elif preset in ("fast", "debug", "test"):
        # Helper rapide pour les tests unitaires
        cfg.training.total_episodes = 20
        cfg.training.batch_size = 64
        cfg.alphazero.enabled = True
        cfg.alphazero.num_simulations = 8
        cfg.mcts.num_simulations = 8
        cfg.model.policy_hidden_layers = [256, 128]
        cfg.model.score_hidden_layers = [256, 128]

    else:
        # 🚀 PHASE 1 : Pré-entraînement Fondation (Cible ~150 VP) - DÉFAUT
        cfg.alphazero.enabled = True
        cfg.alphazero.num_simulations = 16
        cfg.alphazero.gumbel_candidates = 4
        cfg.alphazero.games_per_epoch = 4
        cfg.alphazero.training_steps_per_epoch = 50
        cfg.alphazero.temperature_high = 1.0
        cfg.alphazero.temperature_low = 0.1
        cfg.alphazero.dirichlet_eps = 0.25

        cfg.mcts.enabled = True
        cfg.mcts.algorithm = "gumbel"
        cfg.mcts.num_simulations = 16
        cfg.mcts.gumbel_candidates = 4
        cfg.mcts.use_epistemic_uncertainty = False

        cfg.model.block_type = "swiglu"
        cfg.model.policy_activation = "silu"
        cfg.model.score_activation = "silu"
        cfg.model.policy_dropout = 0.02
        cfg.model.score_dropout = 0.02
        cfg.model.policy_hidden_layers = [1024, 1024, 512, 256]
        cfg.model.score_hidden_layers = [1024, 1024, 512, 256]
        cfg.model.policy_weight_decay = 2e-4
        cfg.model.score_weight_decay = 2e-4
        cfg.model.use_gnn_map = True
        cfg.model.gnn_layers = 3
        cfg.model.gnn_hidden_dim = 64
        cfg.model.finetune_mode = False  # Dégelé

        cfg.model.policy_lr = 1e-4
        cfg.model.score_lr = 3e-4
        cfg.training.batch_size = 256
        cfg.training.total_episodes = 2500

        # Pas de bruit parasite
        cfg.training.shaping_enabled = False
        cfg.training.rnd_enabled = False
        cfg.training.rnad_enabled = False
        cfg.training.rgsc_enabled = False
        cfg.training.use_opponent_modeling = False
        cfg.league.enabled = False

        # Micro-Dispatch guidé par la valeur
        cfg.micro_dispatch.enabled = True
        cfg.micro_dispatch.setup_mode = "value_guided"
        cfg.micro_dispatch.initial_booster_draft_enabled = True
        cfg.micro_dispatch.tech_tile_dispatch_enabled = True
        cfg.micro_dispatch.num_candidates = 4
        cfg.micro_dispatch.temperature = 0.05

        # Hardware RTX
        cfg.hardware.use_mixed_precision = True
        cfg.hardware.enable_tf32 = True

    return cfg


