"""Experience Replay & Rollout Buffer with Multi-Agent Generalized Advantage Estimation (GAE)
for training ScorePredictorNet (Value) and ActionOptimizerNet (Policy).
"""

from typing import Generator, List, NamedTuple, Optional
import numpy as np
import torch


class BatchData(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor
    action_masks: torch.Tensor
    opponent_actions: torch.Tensor
    has_opponents: torch.Tensor


class RolloutBuffer:
    """Stores transitions from self-play rollouts and computes player-separated GAE advantages."""

    def __init__(self, obs_dim: int = 42, action_dim: int = 16, capacity: int = 2048):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.capacity = capacity

        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=bool)
        self.values = np.zeros((capacity,), dtype=np.float32)
        self.log_probs = np.zeros((capacity,), dtype=np.float32)
        self.action_masks = np.zeros((capacity, action_dim), dtype=bool)
        self.player_ids = np.zeros((capacity,), dtype=np.int32)
        self.opponent_actions = np.zeros((capacity,), dtype=np.int64)
        self.has_opponents = np.zeros((capacity,), dtype=bool)

        self.advantages = np.zeros((capacity,), dtype=np.float32)
        self.returns = np.zeros((capacity,), dtype=np.float32)

        self.ptr = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
        action_mask: np.ndarray,
        player_id: int = 0,
        opponent_action: Optional[int] = None,
    ) -> None:
        """Appends one transition to the buffer."""
        idx = self.ptr
        self.observations[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.action_masks[idx] = action_mask
        self.player_ids[idx] = player_id
        if opponent_action is not None and 0 <= opponent_action < self.action_dim:
            self.opponent_actions[idx] = opponent_action
            self.has_opponents[idx] = True
        else:
            self.opponent_actions[idx] = 0
            self.has_opponents[idx] = False
        self.ptr += 1
        if self.ptr >= self.capacity:
            self.full = True
            self.ptr = 0
        return idx

    def set_opponent_action(self, idx: int, opponent_action: int) -> None:
        """Sets the opponent action observed after transition idx was stored."""
        if 0 <= idx < self.capacity and 0 <= opponent_action < self.action_dim:
            self.opponent_actions[idx] = opponent_action
            self.has_opponents[idx] = True

    def size(self) -> int:
        return self.capacity if self.full else self.ptr

    def clear(self) -> None:
        self.ptr = 0
        self.full = False

    def compute_returns_and_advantages(
        self, last_val: float = 0.0, done: bool = False, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> None:
        """Computes Generalized Advantage Estimation (GAE) per player trajectory."""
        size = self.size()
        if size == 0:
            return

        # Separate trajectories by player seat so advantages are causally accurate
        for p in np.unique(self.player_ids[:size]):
            p_indices = [t for t in range(size) if self.player_ids[t] == p]
            if not p_indices:
                continue

            last_advantage = 0.0
            for pos in reversed(range(len(p_indices))):
                t = p_indices[pos]
                if pos == len(p_indices) - 1:
                    next_non_terminal = 1.0 - float(done)
                    next_val = last_val
                else:
                    t_next = p_indices[pos + 1]
                    next_non_terminal = 1.0 - float(self.dones[t])
                    next_val = self.values[t_next]

                delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
                last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
                self.advantages[t] = last_advantage

        self.returns[:size] = self.advantages[:size] + self.values[:size]

        # Normalize advantages across batch for training stability
        adv_slice = self.advantages[:size]
        adv_std = float(np.std(adv_slice))
        if adv_std > 1e-6:
            adv_mean = float(np.mean(adv_slice))
            self.advantages[:size] = (adv_slice - adv_mean) / (adv_std + 1e-8)

    def get_batches(
        self, batch_size: int, device: torch.device
    ) -> Generator[BatchData, None, None]:
        """Yields randomized mini-batches placed directly on the specified device."""
        size = self.size()
        indices = np.random.permutation(size)
        non_blocking = device.type == "cuda"

        for start_idx in range(0, size, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]

            yield BatchData(
                observations=torch.from_numpy(self.observations[batch_indices]).to(
                    device, non_blocking=non_blocking
                ),
                actions=torch.from_numpy(self.actions[batch_indices]).to(
                    device, non_blocking=non_blocking
                ),
                log_probs=torch.from_numpy(self.log_probs[batch_indices]).to(
                    device, non_blocking=non_blocking
                ),
                advantages=torch.from_numpy(self.advantages[batch_indices]).to(
                    device, non_blocking=non_blocking
                ),
                returns=torch.from_numpy(self.returns[batch_indices]).to(
                    device, non_blocking=non_blocking
                ),
                values=torch.from_numpy(self.values[batch_indices]).to(
                    device, non_blocking=non_blocking
                ),
                action_masks=torch.from_numpy(self.action_masks[batch_indices]).to(
                    device, non_blocking=non_blocking
                ),
                opponent_actions=torch.from_numpy(self.opponent_actions[batch_indices]).to(
                    device, non_blocking=non_blocking
                ),
                has_opponents=torch.from_numpy(self.has_opponents[batch_indices]).to(
                    device, non_blocking=non_blocking
                ),
            )
