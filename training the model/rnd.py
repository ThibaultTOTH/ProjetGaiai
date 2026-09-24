"""Random Network Distillation (RND) for Intrinsic Curiosity & Novelty Exploration.

Implements:
1. RunningMeanStd: Online Welford-based mean and variance tracking for reward normalization.
2. RNDTargetNet: Frozen random neural network mapping states to fixed embeddings.
3. RNDPredictorNet: Trainable neural network trained via MSE to predict target embeddings.
4. RNDModel: High-level orchestrator computing normalized intrinsic rewards and distillation losses.
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningMeanStd:
    """Tracks running mean and variance of intrinsic rewards using Welford's algorithm."""

    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        """Updates internal statistics with a new batch of data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 0 else 1

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Scales reward by standard deviation so that scale remains stable across epochs."""
        std = np.sqrt(self.var + 1e-8)
        return x / std


class RNDTargetNet(nn.Module):
    """Frozen random neural network mapping states to fixed embeddings."""

    def __init__(self, in_dim: int = 2476, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )
        # Initialize orthogonally and freeze forever
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=float(np.sqrt(2.0)))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDPredictorNet(nn.Module):
    """Trainable neural network trained to distill the target network."""

    def __init__(self, in_dim: int = 2476, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=float(np.sqrt(2.0)))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDModel(nn.Module):
    """Orchestrates Random Network Distillation for novelty discovery."""

    def __init__(
        self,
        in_dim: int = 2476,
        out_dim: int = 256,
        lr: float = 1e-4,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device or torch.device("cpu")

        self.target = RNDTargetNet(in_dim, out_dim).to(self.device)
        self.predictor = RNDPredictorNet(in_dim, out_dim).to(self.device)

        self.optimizer = torch.optim.AdamW(self.predictor.parameters(), lr=lr, weight_decay=1e-5)
        self.normalizer = RunningMeanStd()

    def compute_raw_error(self, obs: torch.Tensor) -> torch.Tensor:
        """Computes raw squared error per sample in batch: ||f_pred(s) - f_target(s)||^2."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        with torch.no_grad():
            target_feat = self.target(obs)
            pred_feat = self.predictor(obs)
            err = torch.norm(pred_feat - target_feat, dim=-1) ** 2
        return err

    def compute_intrinsic_reward(self, obs: torch.Tensor) -> float:
        """Computes normalized intrinsic reward for a single state observation."""
        if obs.device != self.device:
            obs = obs.to(self.device)
        err = self.compute_raw_error(obs)
        raw_val = float(err.squeeze().item())
        self.normalizer.update(np.array([raw_val]))
        norm_val = float(self.normalizer.normalize(raw_val))
        return norm_val

    def compute_loss(self, obs: torch.Tensor) -> torch.Tensor:
        """Computes distillation MSE loss on a batch of observations."""
        if obs.device != self.device:
            obs = obs.to(self.device)
        with torch.no_grad():
            target_feat = self.target(obs)
        pred_feat = self.predictor(obs)
        loss = F.mse_loss(pred_feat, target_feat)
        return loss

    def train_step(self, obs: torch.Tensor) -> float:
        """Performs a single gradient descent step on the predictor network."""
        self.predictor.train()
        loss = self.compute_loss(obs)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.item())
