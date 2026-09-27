"""Neural Network Architectures for Gaia Project RL:
1. ScorePredictorNet: Value Network (predicts expected final score / VP).
2. ActionOptimizerNet: Policy Network (selects optimal actions using legal action masking).
3. DualGaiaAgent: Combined agent orchestrating both networks for Self-Play and PPO.
"""

import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from config import ModelConfig


def get_activation(name: str) -> nn.Module:
    """Returns the requested PyTorch activation module."""
    name_lower = name.lower()
    if name_lower == "relu":
        return nn.ReLU(inplace=True)
    if name_lower in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    if name_lower == "mish":
        return nn.Mish(inplace=True)
    if name_lower == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    return nn.GELU()


class PreLNResBlock(nn.Module):
    """Pre-LayerNorm Residual Block: x + Linear(Act(Linear(LN(x)))). Superior gradient flow."""

    def __init__(self, hidden_dim: int, dropout: float = 0.05, activation: str = "silu"):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = get_activation(activation)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.ln1(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.ln2(out)
        out = self.fc2(out)
        return residual + out


class BottleneckResBlock(nn.Module):
    """Bottleneck Residual Block: downprojects to hidden_dim//2, applies transform, projects back up."""

    def __init__(self, hidden_dim: int, bottleneck_ratio: int = 2, dropout: float = 0.05, activation: str = "silu"):
        super().__init__()
        mid_dim = max(64, hidden_dim // bottleneck_ratio)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, mid_dim)
        self.act = get_activation(activation)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ln2 = nn.LayerNorm(mid_dim)
        self.up = nn.Linear(mid_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.ln1(x)
        out = self.down(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.ln2(out)
        out = self.up(out)
        return residual + out


class SwiGLUBlock(nn.Module):
    """Swish-Gated Linear Unit Block (standard in modern high-capacity transformer/dense architectures)."""

    def __init__(self, hidden_dim: int, dropout: float = 0.05, activation: str = "silu"):
        super().__init__()
        mid_dim = int(hidden_dim * 1.5)
        self.ln = nn.LayerNorm(hidden_dim)
        self.w_gate = nn.Linear(hidden_dim, mid_dim, bias=False)
        self.w_up = nn.Linear(hidden_dim, mid_dim, bias=False)
        self.w_down = nn.Linear(mid_dim, hidden_dim, bias=False)
        self.act = get_activation(activation)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        normed = self.ln(x)
        gate = self.act(self.w_gate(normed))
        up = self.w_up(normed)
        out = self.w_down(gate * up)
        out = self.drop(out)
        return residual + out


# Alias for backwards compatibility
ResidualBlock = PreLNResBlock


def build_block(
    block_type: str,
    in_dim: int,
    out_dim: int,
    dropout: float = 0.05,
    activation: str = "silu",
) -> nn.Module:
    """Factory creating residual blocks or projection transitions."""
    if in_dim == out_dim:
        bt = block_type.lower()
        if bt in ("bottleneck", "bottle"):
            return BottleneckResBlock(in_dim, dropout=dropout, activation=activation)
        if bt in ("swiglu", "glu"):
            return SwiGLUBlock(in_dim, dropout=dropout, activation=activation)
        return PreLNResBlock(in_dim, dropout=dropout, activation=activation)

    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, out_dim),
        get_activation(activation),
        nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
    )


def compute_normalized_adjacency(adj_table: np.ndarray) -> torch.Tensor:
    """Converts a (200, 6) neighbor index array into a normalized (200, 200) spectral adjacency matrix.

    A_tilde = A + I_200
    D_tilde_ii = sum_j A_tilde_ij
    A_hat = D_tilde^(-1/2) @ A_tilde @ D_tilde^(-1/2)
    """
    n = 200
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(min(n, adj_table.shape[0])):
        for d in range(min(6, adj_table.shape[1])):
            nb = int(adj_table[i, d])
            if nb < n:
                A[i, nb] = 1.0
                A[nb, i] = 1.0

    A_tilde = A + np.eye(n, dtype=np.float32)
    d = np.sum(A_tilde, axis=1)
    d_inv_sqrt = np.power(d, -0.5, where=d > 0)
    d_inv_sqrt[d <= 0] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)

    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    return torch.from_numpy(A_hat.astype(np.float32))


class HexGraphConv(nn.Module):
    """Spectral Hexagonal Graph Convolution: X_next = LN(Act(Dropout(A_hat @ Linear(X)))) + Residual."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.05, activation: str = "silu"):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.act = get_activation(activation)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.has_res = (in_dim == out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # x: (B, 200, in_dim), adj_norm: (200, 200)
        support = self.fc(x)
        out = torch.matmul(adj_norm, support)
        out = self.ln(out)
        out = self.act(out)
        out = self.drop(out)
        if self.has_res:
            return x + out
        return out


class HexGNNEncoder(nn.Module):
    """Encodes the 200-hex galaxy board (2000 features = 200 hexes x 10 features) into a compact spatial vector."""

    def __init__(
        self,
        in_features: int = 10,
        hidden_dim: int = 64,
        out_dim: int = 256,
        layers: int = 3,
        dropout: float = 0.05,
        activation: str = "silu",
        adj_norm: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        if adj_norm is None:
            from environment import build_canonical_hex_adjacency
            adj_norm = compute_normalized_adjacency(build_canonical_hex_adjacency())
        self.register_buffer("adj_norm", adj_norm)

        convs = []
        convs.append(HexGraphConv(in_features, hidden_dim, dropout=dropout, activation=activation))
        for _ in range(layers - 1):
            convs.append(HexGraphConv(hidden_dim, hidden_dim, dropout=dropout, activation=activation))
        self.convs = nn.ModuleList(convs)

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            get_activation(activation),
        )

    def forward(self, map_flat: torch.Tensor) -> torch.Tensor:
        # map_flat: (B, 2000) -> reshape to (B, 200, in_features)
        batch_size = map_flat.size(0)
        x = map_flat.view(batch_size, 200, self.in_features)
        adj = self.adj_norm
        for conv in self.convs:
            x = conv(x, adj)
        mean_p = x.mean(dim=1)
        max_p = x.max(dim=1).values
        pool = torch.cat([mean_p, max_p], dim=-1)
        return self.proj(pool)


class DualStreamBackbone(nn.Module):
    """Dual-Stream Encoder: Spatial GNN for map + Pre-LN MLP for scalar features -> Fusion trunk."""

    def __init__(
        self,
        scalar_dim: int = 476,
        map_dim: int = 2000,
        obs_dim: Optional[int] = None,
        gnn_hidden_dim: int = 64,
        gnn_layers: int = 3,
        map_out_dim: int = 256,
        scalar_out_dim: int = 256,
        trunk_layers: Optional[List[int]] = None,
        block_type: str = "pre_ln",
        dropout: float = 0.05,
        activation: str = "silu",
        use_input_norm: bool = True,
        use_gnn_map: bool = True,
    ):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.map_dim = map_dim
        self.obs_dim = obs_dim if obs_dim is not None else (scalar_dim + map_dim)
        self.use_gnn_map = use_gnn_map

        if trunk_layers is None:
            trunk_layers = [512, 512, 256]

        if use_gnn_map:
            self.map_encoder = HexGNNEncoder(
                in_features=10,
                hidden_dim=gnn_hidden_dim,
                out_dim=map_out_dim,
                layers=gnn_layers,
                dropout=dropout,
                activation=activation,
            )
            self.scalar_norm = nn.LayerNorm(scalar_dim) if use_input_norm else nn.Identity()
            self.scalar_proj = nn.Sequential(
                nn.Linear(scalar_dim, scalar_out_dim),
                nn.LayerNorm(scalar_out_dim),
                get_activation(activation),
            )
            fusion_dim = map_out_dim + scalar_out_dim  # 256 + 256 = 512
            blocks = []
            curr_dim = fusion_dim
            for out_dim in trunk_layers:
                blocks.append(build_block(block_type, curr_dim, out_dim, dropout, activation))
                curr_dim = out_dim
            self.trunk = nn.Sequential(*blocks)
            self.output_dim = trunk_layers[-1] if trunk_layers else fusion_dim
        else:
            effective_obs_dim = self.obs_dim
            self.flat_norm = nn.LayerNorm(effective_obs_dim) if use_input_norm else nn.Identity()
            first_dim = trunk_layers[0] if trunk_layers else 512
            self.flat_proj = nn.Sequential(
                nn.Linear(effective_obs_dim, first_dim),
                nn.LayerNorm(first_dim),
                get_activation(activation),
            )
            blocks = []
            curr_dim = first_dim
            for out_dim in (trunk_layers[1:] if len(trunk_layers) > 1 else [first_dim]):
                blocks.append(build_block(block_type, curr_dim, out_dim, dropout, activation))
                curr_dim = out_dim
            self.trunk = nn.Sequential(*blocks)
            self.output_dim = curr_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        if self.use_gnn_map and obs.size(1) >= (self.scalar_dim + self.map_dim):
            scalar_x = obs[:, :self.scalar_dim]
            map_x = obs[:, self.scalar_dim:self.scalar_dim + self.map_dim]

            e_scalar = self.scalar_proj(self.scalar_norm(scalar_x))
            e_map = self.map_encoder(map_x)
            fused = torch.cat([e_scalar, e_map], dim=-1)
            return self.trunk(fused)
        else:
            x = self.flat_norm(obs)
            x = self.flat_proj(x)
            return self.trunk(x)


class ScorePredictorNet(nn.Module):
    """Model 1: Predicts expected final score / Victory Points from state observation."""

    def __init__(
        self,
        obs_dim: int = 2476,
        hidden_layers: Optional[List[int]] = None,
        dropout: float = 0.05,
        activation: str = "silu",
        block_type: str = "pre_ln",
        use_input_norm: bool = True,
        use_gnn_map: bool = True,
        gnn_hidden_dim: int = 64,
        gnn_layers: int = 3,
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [512, 512, 512, 256]

        self.obs_dim = obs_dim
        self.block_type = block_type
        self.use_input_norm = use_input_norm
        self.use_gnn_map = use_gnn_map and (obs_dim == 2476)

        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = DualStreamBackbone(
                scalar_dim=476,
                map_dim=2000,
                obs_dim=self.obs_dim,
                gnn_hidden_dim=gnn_hidden_dim,
                gnn_layers=gnn_layers,
                map_out_dim=256,
                scalar_out_dim=256,
                trunk_layers=hidden_layers,
                block_type=block_type,
                dropout=dropout,
                activation=activation,
                use_input_norm=use_input_norm,
                use_gnn_map=self.use_gnn_map,
            )

        self.head = nn.Sequential(
            nn.Linear(self.backbone.output_dim, 128),
            get_activation(activation),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        gain = math.sqrt(2.0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        if hasattr(self, "head") and len(self.head) > 0 and isinstance(self.head[-1], nn.Linear):
            nn.init.orthogonal_(self.head[-1].weight, gain=1.0)
            if self.head[-1].bias is not None:
                nn.init.constant_(self.head[-1].bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(obs)
        score = self.head(feat)
        return score.squeeze(-1)

    @torch.no_grad()
    def predict_score(self, obs: torch.Tensor) -> float:
        self.eval()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        val = self.forward(obs)
        return float(val.item())

    @torch.no_grad()
    def predict_score_with_uncertainty(
        self, obs: torch.Tensor, num_passes: int = 4
    ) -> Tuple[float, float]:
        """Estimates expected score and epistemic uncertainty sigma_V via parallel MC-Dropout passes.

        Dropout layers are set to train mode while LayerNorms remain frozen in eval mode.
        Returns (mean_score, std_score).
        """
        if num_passes <= 1:
            val = self.predict_score(obs)
            return val, 0.0

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        def _set_dropout_train(module: nn.Module):
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
                module.train()

        self.eval()
        self.apply(_set_dropout_train)
        try:
            repeated = obs.repeat(num_passes, 1)
            scores = self.forward(repeated)  # shape (num_passes,)
            mean_val = float(scores.mean().item())
            std_val = float(scores.std().item())
        finally:
            self.eval()

        return mean_val, std_val


class ActionOptimizerNet(nn.Module):
    """Model 2: Action Optimizer (Policy Network) with strict legal action masking."""

    def __init__(
        self,
        obs_dim: int = 2476,
        action_dim: int = 3130,
        hidden_layers: Optional[List[int]] = None,
        dropout: float = 0.05,
        activation: str = "silu",
        block_type: str = "pre_ln",
        use_input_norm: bool = True,
        use_gnn_map: bool = True,
        gnn_hidden_dim: int = 64,
        gnn_layers: int = 3,
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [512, 512, 512, 256]

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.block_type = block_type
        self.use_input_norm = use_input_norm
        self.use_gnn_map = use_gnn_map and (obs_dim == 2476)

        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = DualStreamBackbone(
                scalar_dim=476,
                map_dim=2000,
                obs_dim=self.obs_dim,
                gnn_hidden_dim=gnn_hidden_dim,
                gnn_layers=gnn_layers,
                map_out_dim=256,
                scalar_out_dim=256,
                trunk_layers=hidden_layers,
                block_type=block_type,
                dropout=dropout,
                activation=activation,
                use_input_norm=use_input_norm,
                use_gnn_map=self.use_gnn_map,
            )

        self.policy_head = nn.Sequential(
            nn.Linear(self.backbone.output_dim, 128),
            get_activation(activation),
            nn.Linear(128, action_dim),
        )

        self.opponent_head = nn.Sequential(
            nn.Linear(self.backbone.output_dim, 128),
            get_activation(activation),
            nn.Linear(128, action_dim),
        )

        self._init_weights()

    def _init_weights(self):
        gain = math.sqrt(2.0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        if hasattr(self, "policy_head") and len(self.policy_head) > 0 and isinstance(self.policy_head[-1], nn.Linear):
            nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        if hasattr(self, "opponent_head") and len(self.opponent_head) > 0 and isinstance(self.opponent_head[-1], nn.Linear):
            nn.init.orthogonal_(self.opponent_head[-1].weight, gain=0.01)

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        return_opponent: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        feat = self.backbone(obs)
        logits = self.policy_head(feat)

        if action_mask is not None:
            if action_mask.dim() == 1 and logits.dim() == 2:
                action_mask = action_mask.unsqueeze(0)
            neg_val = torch.tensor(-1e4, dtype=logits.dtype, device=logits.device)
            logits = torch.where(action_mask, logits, neg_val)

        if return_opponent:
            opp_logits = self.opponent_head(feat)
            return logits, opp_logits

        return logits

    def predict_opponent(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns predicted probability distribution over the 16 actions for the next opponent."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        feat = self.backbone(obs)
        opp_logits = self.opponent_head(feat)
        return F.softmax(opp_logits, dim=-1)

    def get_action(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[int, float, torch.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if action_mask is not None and action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)

        logits = self.forward(obs, action_mask)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs=probs)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return int(action.item()), float(log_prob.item()), probs.squeeze(0)


class DualGaiaAgent(nn.Module):
    """Combined agent managing both ScorePredictorNet and ActionOptimizerNet."""

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        self._build_modules()

    def _build_modules(self) -> None:
        block_t = getattr(self.config, "block_type", "swiglu")
        use_in_norm = getattr(self.config, "use_input_norm", True)
        use_gnn = getattr(self.config, "use_gnn_map", True)
        gnn_h = getattr(self.config, "gnn_hidden_dim", 64)
        gnn_l = getattr(self.config, "gnn_layers", 3)
        pol_layers = getattr(self.config, "policy_hidden_layers", None) or [1024, 1024, 512, 256]
        sc_layers = getattr(self.config, "score_hidden_layers", None) or pol_layers

        # Create Foundation Model (Shared Backbone)
        self.shared_backbone = DualStreamBackbone(
            scalar_dim=476,
            map_dim=2000,
            obs_dim=self.config.obs_dim,
            gnn_hidden_dim=gnn_h,
            gnn_layers=gnn_l,
            map_out_dim=256,
            scalar_out_dim=256,
            trunk_layers=pol_layers,
            block_type=block_t,
            dropout=self.config.policy_dropout,
            activation=self.config.policy_activation,
            use_input_norm=use_in_norm,
            use_gnn_map=use_gnn and (self.config.obs_dim == 2476),
        )

        self.score_net = ScorePredictorNet(
            obs_dim=self.config.obs_dim,
            hidden_layers=sc_layers,
            dropout=self.config.score_dropout,
            activation=self.config.score_activation,
            block_type=block_t,
            use_input_norm=use_in_norm,
            use_gnn_map=use_gnn,
            gnn_hidden_dim=gnn_h,
            gnn_layers=gnn_l,
            backbone=self.shared_backbone,
        )
        self.action_net = ActionOptimizerNet(
            obs_dim=self.config.obs_dim,
            action_dim=self.config.action_dim,
            hidden_layers=pol_layers,
            dropout=self.config.policy_dropout,
            activation=self.config.policy_activation,
            block_type=block_t,
            use_input_norm=use_in_norm,
            use_gnn_map=use_gnn,
            gnn_hidden_dim=gnn_h,
            gnn_layers=gnn_l,
            backbone=self.shared_backbone,
        )

        if getattr(self.config, "finetune_mode", False):
            for param in self.shared_backbone.parameters():
                param.requires_grad = False

    def _adapt_architecture_from_state_dict(
        self, state_dict: dict, config_obj: Optional[Any] = None
    ) -> bool:
        """Dynamically inspects checkpoint state_dict and adjusts agent architecture if needed.
        Returns True if the architecture was modified and modules were re-instantiated.
        """
        rebuild_needed = False

        # 1. Adapt from checkpoint config object if provided
        if config_obj is not None:
            model_cfg = getattr(config_obj, "model", config_obj)
            if hasattr(model_cfg, "block_type") and model_cfg.block_type != getattr(self.config, "block_type", "swiglu"):
                self.config.block_type = model_cfg.block_type
                rebuild_needed = True
            if hasattr(model_cfg, "policy_hidden_layers") and model_cfg.policy_hidden_layers != getattr(self.config, "policy_hidden_layers", None):
                self.config.policy_hidden_layers = list(model_cfg.policy_hidden_layers)
                self.config.score_hidden_layers = list(getattr(model_cfg, "score_hidden_layers", model_cfg.policy_hidden_layers))
                rebuild_needed = True
            if hasattr(model_cfg, "use_gnn_map") and model_cfg.use_gnn_map != getattr(self.config, "use_gnn_map", True):
                self.config.use_gnn_map = model_cfg.use_gnn_map
                rebuild_needed = True

        # 2. Inspect state_dict keys directly (guarantees 100% precision even if config_obj is absent)
        trunk_prefix = None
        for cand in ("shared_backbone.trunk.", "backbone.trunk.", "trunk."):
            if any(k.startswith(cand) for k in state_dict):
                trunk_prefix = cand
                break

        detected_block_type = None
        if any("w_gate" in k or "w_up" in k or "w_down" in k for k in state_dict):
            detected_block_type = "swiglu"
        elif any("down.weight" in k and "up.weight" in k for k in state_dict):
            detected_block_type = "bottleneck"
        elif trunk_prefix and any(k.startswith(trunk_prefix) and "ln1" in k for k in state_dict):
            detected_block_type = "pre_ln"

        if detected_block_type is not None and detected_block_type != getattr(self.config, "block_type", "swiglu"):
            self.config.block_type = detected_block_type
            rebuild_needed = True

        if trunk_prefix is not None:
            block_indices = set()
            for k in state_dict:
                if k.startswith(trunk_prefix):
                    remainder = k[len(trunk_prefix):]
                    parts = remainder.split(".")
                    if parts and parts[0].isdigit():
                        block_indices.add(int(parts[0]))

            if block_indices:
                max_idx = max(block_indices)
                detected_layers = []
                for i in range(max_idx + 1):
                    proj_key = f"{trunk_prefix}{i}.1.weight"
                    swiglu_key = f"{trunk_prefix}{i}.w_gate.weight"
                    preln_key = f"{trunk_prefix}{i}.fc1.weight"
                    bottle_key = f"{trunk_prefix}{i}.down.weight"

                    if proj_key in state_dict:
                        detected_layers.append(int(state_dict[proj_key].shape[0]))
                    elif swiglu_key in state_dict:
                        detected_layers.append(int(state_dict[swiglu_key].shape[1]))
                    elif preln_key in state_dict:
                        detected_layers.append(int(state_dict[preln_key].shape[0]))
                    elif bottle_key in state_dict:
                        detected_layers.append(int(state_dict[bottle_key].shape[1]))
                    else:
                        break

                if len(detected_layers) == (max_idx + 1):
                    if detected_layers != getattr(self.config, "policy_hidden_layers", None):
                        self.config.policy_hidden_layers = detected_layers
                        self.config.score_hidden_layers = list(detected_layers)
                        rebuild_needed = True

        gnn_in_state = any("map_encoder" in k for k in state_dict)
        if gnn_in_state != getattr(self.config, "use_gnn_map", True):
            self.config.use_gnn_map = gnn_in_state
            rebuild_needed = True

        if rebuild_needed:
            device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
            self._build_modules()
            self.to(device)
            return True

        return False

    def to_device(self, device: torch.device) -> "DualGaiaAgent":
        self.to(device)
        return self

    def act_and_evaluate(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[int, float, float, torch.Tensor]:
        """Returns (action, log_prob, predicted_score, action_probabilities)."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if action_mask is not None and action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)

        with torch.no_grad():
            score_val = self.score_net(obs)
            logits = self.action_net(obs, action_mask)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs=probs)

            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

        return (
            int(action.item()),
            float(log_prob.item()),
            float(score_val.item()),
            probs.squeeze(0),
        )

    def act_policy(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[int, float]:
        """Fast-path policy evaluation: evaluates ONLY action_net for maximum rollout throughput."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if action_mask is not None and action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)

        with torch.no_grad():
            logits = self.action_net(obs, action_mask)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs=probs)

            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

        return int(action.item()), float(log_prob.item())

    def act(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> int:
        """Ultra-fast policy evaluation: returns only action integer without log_prob or value."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if action_mask is not None and action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)

        with torch.no_grad():
            logits = self.action_net(obs, action_mask)
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = Categorical(probs=probs).sample()

        return int(action.item())

    def predict_opponent_action(
        self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predicts probability distribution over the 16 actions for the next opponent, optionally masked."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        with torch.no_grad():
            opp_probs = self.action_net.predict_opponent(obs)
            if action_mask is not None:
                if action_mask.dim() == 1:
                    action_mask = action_mask.unsqueeze(0)
                opp_probs = torch.where(action_mask, opp_probs, torch.zeros_like(opp_probs))
                sums = opp_probs.sum(dim=-1, keepdim=True)
                opp_probs = torch.where(
                    sums > 1e-6,
                    opp_probs / sums,
                    action_mask.float() / action_mask.sum().clamp(min=1.0),
                )
        return opp_probs.squeeze(0)

    @torch.no_grad()
    def evaluate_leaf(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        leaf_actor: int = 0,
    ) -> Tuple[float, np.ndarray]:
        """Ultra-fast joint evaluation of both value and policy prior with a SINGLE shared backbone forward pass."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        feat = self.shared_backbone(obs)
        val = self.score_net.head(feat).squeeze(-1)
        neg_val = torch.tensor(-1e4, dtype=feat.dtype, device=feat.device)
        if leaf_actor == 0 or not getattr(self.config, "use_opponent_modeling", False):
            logits = self.action_net.policy_head(feat)
            if action_mask is not None:
                if action_mask.dim() == 1 and logits.dim() == 2:
                    action_mask = action_mask.unsqueeze(0)
                logits = torch.where(action_mask, logits, neg_val)
            priors = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        else:
            opp_logits = self.action_net.opponent_head(feat)
            if action_mask is not None:
                if action_mask.dim() == 1 and opp_logits.dim() == 2:
                    action_mask = action_mask.unsqueeze(0)
                opp_logits = torch.where(action_mask, opp_logits, neg_val)
            priors = F.softmax(opp_logits, dim=-1).squeeze(0).cpu().numpy()
        return float(val.item()), priors

    def predict_score_with_uncertainty(
        self, obs: torch.Tensor, num_passes: int = 4
    ) -> Tuple[float, float]:
        """Estimates expected score and epistemic uncertainty sigma_V via MC-Dropout."""
        return self.score_net.predict_score_with_uncertainty(obs, num_passes=num_passes)

    def clone_policy_net(self) -> ActionOptimizerNet:
        """Creates an independent frozen copy of the action policy network for league play."""
        import copy
        cloned = copy.deepcopy(self.action_net)
        cloned.eval()
        for p in cloned.parameters():
            p.requires_grad = False
        return cloned

    def save_checkpoint(self, path: str, extra_meta: Optional[dict] = None) -> None:
        """Saves weights and metadata to file."""
        import os

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            "score_net_state": self.score_net.state_dict(),
            "action_net_state": self.action_net.state_dict(),
            "agent_state_dict": self.state_dict(),
            "config": self.config,
            "meta": extra_meta or {},
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, device: Optional[torch.device] = None) -> dict:
        """Loads weights from checkpoint file supporting DualGaiaAgent, AlphaZero, and state_dict formats."""
        target_device = device or (next(self.parameters()).device if list(self.parameters()) else torch.device("cpu"))
        try:
            checkpoint = torch.load(path, map_location=target_device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=target_device)

        meta = checkpoint.get("meta", {}) if isinstance(checkpoint, dict) else {}
        if isinstance(checkpoint, dict) and "epoch" in checkpoint and "epoch" not in meta:
            meta["epoch"] = checkpoint["epoch"]

        config_obj = checkpoint.get("config", None) if isinstance(checkpoint, dict) else None

        if isinstance(checkpoint, dict) and "agent_state_dict" in checkpoint:
            state_dict = checkpoint["agent_state_dict"]
            self._adapt_architecture_from_state_dict(state_dict, config_obj)
            self.load_state_dict(state_dict)
        elif isinstance(checkpoint, dict) and "score_net_state" in checkpoint and "action_net_state" in checkpoint:
            score_state = checkpoint["score_net_state"]
            action_state = checkpoint["action_net_state"]
            self._adapt_architecture_from_state_dict(score_state, config_obj)
            self.score_net.load_state_dict(score_state)
            self.action_net.load_state_dict(action_state)
        else:
            state_dict = checkpoint if isinstance(checkpoint, dict) else {}
            self._adapt_architecture_from_state_dict(state_dict, config_obj)
            try:
                self.load_state_dict(state_dict)
            except Exception:
                raise KeyError(
                    f"Unrecognized checkpoint format in {path}. Keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}"
                )

        if target_device is not None:
            self.to_device(target_device)
        return meta
