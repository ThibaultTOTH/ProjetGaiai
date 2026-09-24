"""Value-Guided Micro-Action Dispatcher for Gaia Project Deep RL.

Selects fine-grained spatial and technological targets (Initial Mines, Initial Boosters,
Hex placements, Tech Tiles, Research Tracks, Federation Tokens, and Round Boosters)
guided by the ScorePredictorNet (Value Network), while retaining the compact 16-action PPO policy head.
"""

from typing import Any, List, Optional
import numpy as np
import torch
import torch.nn.functional as F

from config import MicroDispatchConfig


class ValueGuidedMicroDispatcher:
    """Dispatches macro-actions and setup phases to optimal micro-targets.

    Leverages the ScorePredictorNet to evaluate candidate future states in batch on GPU/CPU.
    """

    def __init__(
        self,
        config: Optional[MicroDispatchConfig] = None,
        agent: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or MicroDispatchConfig()
        self.agent = agent
        self.device = device or (
            next(agent.parameters()).device if agent is not None else torch.device("cpu")
        )

    def get_candidates(self, env: Any, action: int) -> List[int]:
        """Queries environment or deduces eligible micro-targets for the given macro action."""
        if hasattr(env, "get_micro_candidates"):
            return env.get_micro_candidates(action)
        return [0]

    def select_best_target(
        self,
        env: Any,
        action: int,
        candidates: List[int],
        deterministic: bool = False,
    ) -> Optional[int]:
        """Evaluates candidate next states using ScorePredictorNet and selects the optimal target."""
        if not candidates or len(candidates) <= 1 or self.agent is None:
            return candidates[0] if candidates else None

        k = min(getattr(self.config, "num_candidates", 4), len(candidates))
        eval_cands = candidates[:k]

        next_obs_list = []
        valid_cands = []

        for cand in eval_cands:
            try:
                sim_env = env.clone()
                res = sim_env.step(action, target=cand)
                next_obs_list.append(res.obs)
                valid_cands.append(cand)
            except Exception:
                continue

        if not valid_cands:
            return candidates[0]

        if len(valid_cands) == 1:
            return valid_cands[0]

        # Batched evaluation through ScorePredictorNet
        obs_tensor = torch.from_numpy(np.stack(next_obs_list)).float().to(self.device)
        with torch.no_grad():
            pred_scores = self.agent.score_net(obs_tensor).squeeze(-1).cpu().numpy()

        if pred_scores.ndim == 0:
            pred_scores = np.array([float(pred_scores)])

        temp = getattr(self.config, "temperature", 0.0)
        if temp <= 1e-4 or deterministic:
            best_idx = int(np.argmax(pred_scores))
        else:
            # Boltzmann softmax sampling
            shifted = (pred_scores - np.max(pred_scores)) / max(1e-4, temp)
            exp_s = np.exp(shifted)
            probs = exp_s / np.sum(exp_s)
            best_idx = int(np.random.choice(len(valid_cands), p=probs))

        return valid_cands[best_idx]

    def select_initial_mine(
        self,
        env: Any,
        seat: int,
        candidates: List[int],
        deterministic: bool = False,
    ) -> int:
        """Selects optimal starting planet hex for initial mine placement guided by value."""
        if not candidates or len(candidates) <= 1 or self.agent is None:
            return candidates[0] if candidates else 0

        k = min(getattr(self.config, "num_candidates", 4), len(candidates))
        eval_cands = candidates[:k]

        next_obs_list = []
        valid_cands = []

        for cand in eval_cands:
            try:
                sim_env = env.clone()
                if hasattr(sim_env, "place_initial_mine"):
                    sim_env.place_initial_mine(seat, cand)
                next_obs_list.append(sim_env._get_obs())
                valid_cands.append(cand)
            except Exception:
                continue

        if not valid_cands:
            return candidates[0]
        if len(valid_cands) == 1:
            return valid_cands[0]

        obs_tensor = torch.from_numpy(np.stack(next_obs_list)).float().to(self.device)
        with torch.no_grad():
            pred_scores = self.agent.score_net(obs_tensor).squeeze(-1).cpu().numpy()

        if pred_scores.ndim == 0:
            pred_scores = np.array([float(pred_scores)])

        temp = getattr(self.config, "temperature", 0.0)
        if temp <= 1e-4 or deterministic:
            best_idx = int(np.argmax(pred_scores))
        else:
            shifted = (pred_scores - np.max(pred_scores)) / max(1e-4, temp)
            exp_s = np.exp(shifted)
            probs = exp_s / np.sum(exp_s)
            best_idx = int(np.random.choice(len(valid_cands), p=probs))

        return valid_cands[best_idx]

    def select_initial_booster(
        self,
        env: Any,
        seat: int,
        candidates: List[int],
        deterministic: bool = False,
    ) -> int:
        """Selects optimal starting round booster during pre-game booster draft."""
        if not candidates or len(candidates) <= 1 or self.agent is None:
            return candidates[0] if candidates else 1

        k = min(getattr(self.config, "num_candidates", 4), len(candidates))
        eval_cands = candidates[:k]

        next_obs_list = []
        valid_cands = []

        for cand in eval_cands:
            try:
                sim_env = env.clone()
                if hasattr(sim_env, "assign_initial_booster"):
                    sim_env.assign_initial_booster(seat, cand)
                next_obs_list.append(sim_env._get_obs())
                valid_cands.append(cand)
            except Exception:
                continue

        if not valid_cands:
            return candidates[0]
        if len(valid_cands) == 1:
            return valid_cands[0]

        obs_tensor = torch.from_numpy(np.stack(next_obs_list)).float().to(self.device)
        with torch.no_grad():
            pred_scores = self.agent.score_net(obs_tensor).squeeze(-1).cpu().numpy()

        if pred_scores.ndim == 0:
            pred_scores = np.array([float(pred_scores)])

        temp = getattr(self.config, "temperature", 0.0)
        if temp <= 1e-4 or deterministic:
            best_idx = int(np.argmax(pred_scores))
        else:
            shifted = (pred_scores - np.max(pred_scores)) / max(1e-4, temp)
            exp_s = np.exp(shifted)
            probs = exp_s / np.sum(exp_s)
            best_idx = int(np.random.choice(len(valid_cands), p=probs))

        return valid_cands[best_idx]

    def step(
        self,
        env: Any,
        action: int,
        deterministic: bool = False,
    ) -> Any:
        """Applies action to environment with value-guided micro-dispatch if enabled."""
        if not getattr(self.config, "enabled", True) or self.agent is None:
            return env.step(action)

        # Optional feature toggles
        if action == 11 and not getattr(self.config, "tech_tile_dispatch_enabled", True):
            return env.step(action)
        if action == 9 and not getattr(self.config, "federation_dispatch_enabled", True):
            return env.step(action)

        candidates = self.get_candidates(env, action)
        if len(candidates) <= 1:
            return env.step(action)

        best_target = self.select_best_target(
            env, action, candidates, deterministic=deterministic
        )
        return env.step(action, target=best_target)
