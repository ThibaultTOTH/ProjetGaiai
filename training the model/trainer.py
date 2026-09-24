"""Reinforcement Learning Trainer for Gaia Project.

Implements:
- Self-play rollout collection with Multi-Agent GAE advantages.
- PPO policy optimization for ActionOptimizerNet with legal action masking.
- Value regression training for ScorePredictorNet.
- Composite reward function (Delta VP + Development Shaping + Margin Bonus).
- CUDA / RTX 5070 Mixed Precision (AMP FP16/BF16) & TF32 acceleration with CPU fallback.
- Multi-threaded asynchronous execution for seamless GUI integration.
"""

import copy
from dataclasses import dataclass
import math
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from buffer import RolloutBuffer
from config import AppConfig
from environment import make_gaia_env
from league import LeagueManager, LeagueMember
from models import DualGaiaAgent
from rnd import RNDModel
from micro_dispatch import ValueGuidedMicroDispatcher


@dataclass
class TrainingMetrics:
    epoch: int
    episodes: int
    policy_loss: float
    value_loss: float
    total_loss: float
    entropy: float
    avg_predicted_score: float
    avg_real_score: float
    win_rate: float
    steps_per_sec: float
    vram_allocated_mb: float
    device_name: str
    league_elo: float = 1200.0
    league_size: int = 2
    rnad_kl: float = 0.0


class PrioritizedStateBuffer:
    """Prioritized State Buffer for Regret-Guided Search Control (RGSC / Go-Exploit).

    Stores mid-game game environments where the agent suffered high counterfactual regret
    (ΔV = V(s_t) - V(s_{t+1}) >= threshold). Allows resetting games directly into mid-game
    crisis puzzles (Rounds 2-5) to master critical decision points instead of over-fitting
    to early openings.
    """

    def __init__(self, capacity: int = 200, regret_threshold: float = 0.40):
        self.capacity = capacity
        self.regret_threshold = regret_threshold
        self.states: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.states)

    def add(self, env: Any, regret: float, round_num: int = 0) -> bool:
        """Stores a cloned copy of env if regret exceeds threshold."""
        if regret < self.regret_threshold:
            return False

        try:
            cloned = env.clone()
        except Exception:
            return False

        if len(self.states) >= self.capacity:
            # Evict state with lowest regret
            min_idx = int(np.argmin([s["regret"] for s in self.states]))
            self.states.pop(min_idx)

        self.states.append({
            "env": cloned,
            "regret": float(regret),
            "round": round_num,
            "added_at": time.time(),
        })
        return True

    def sample(self) -> Optional[Any]:
        """Samples a crisis puzzle clone with probability proportional to regret."""
        if not self.states:
            return None
        regrets = np.array([s["regret"] for s in self.states], dtype=np.float64)
        total = float(regrets.sum())
        if total <= 1e-8:
            idx = int(np.random.randint(0, len(self.states)))
        else:
            probs = regrets / total
            idx = int(np.random.choice(len(self.states), p=probs))
        try:
            return self.states[idx]["env"].clone()
        except Exception:
            return None

    def clear(self) -> None:
        self.states.clear()


class RLTrainer:
    """Orchestrates Self-Play, Multi-Agent GAE, PPO, and Score Predictor training."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        agent: Optional[DualGaiaAgent] = None,
    ):
        self.config = config or AppConfig()
        self.device = self.config.hardware.get_torch_device()
        self.hw_info = self.config.hardware.configure_cuda()

        self.agent = agent or DualGaiaAgent(self.config.model)
        self.agent.to_device(self.device)

        if self.device.type == "cpu":
            cpu_count = os.cpu_count() or 4
            optimal_threads = min(6, max(2, cpu_count // 2 if cpu_count > 4 else cpu_count))
            torch.set_num_threads(optimal_threads)

        self.buffer = RolloutBuffer(
            obs_dim=self.config.model.obs_dim,
            action_dim=self.config.model.action_dim,
            capacity=self.config.training.rollout_steps_per_epoch,
        )

        self.policy_optimizer = torch.optim.AdamW(
            self.agent.action_net.parameters(),
            lr=self.config.model.policy_lr,
            weight_decay=self.config.model.policy_weight_decay,
        )
        self.value_optimizer = torch.optim.AdamW(
            self.agent.score_net.parameters(),
            lr=self.config.model.score_lr,
            weight_decay=self.config.model.score_weight_decay,
        )

        self.use_amp = (
            self.device.type == "cuda"
            and self.config.hardware.use_mixed_precision
        )
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._is_running = False

        self.current_epoch = 0
        self.total_episodes_played = 0
        self._last_win_rate = 0.5
        self.best_elo = 1200.0

        self.league_manager = LeagueManager(self.config.league, device=self.device)
        if self.config.league.enabled:
            loaded = self.league_manager.load_existing_checkpoints(
                obs_dim=self.config.model.obs_dim,
                action_dim=self.config.model.action_dim,
            )
            if loaded > 0:
                print(f"[LeagueManager] Initialized with {loaded} historical snapshot(s).")

        self.rnd = None
        if getattr(self.config.training, "rnd_enabled", True):
            self.rnd = RNDModel(
                in_dim=self.config.model.obs_dim,
                out_dim=256,
                lr=getattr(self.config.training, "rnd_learning_rate", 1e-4),
                device=self.device,
            )

        self.state_buffer = None
        if getattr(self.config.training, "rgsc_enabled", True):
            self.state_buffer = PrioritizedStateBuffer(
                capacity=getattr(self.config.training, "rgsc_buffer_capacity", 200),
                regret_threshold=getattr(self.config.training, "rgsc_regret_threshold", 0.40),
            )

        self.ref_policy_net = None
        self.last_rnad_kl = 0.0
        if getattr(self.config.training, "rnad_enabled", True):
            self.ref_policy_net = copy.deepcopy(self.agent.action_net)
            self.ref_policy_net.eval()
            for p in self.ref_policy_net.parameters():
                p.requires_grad = False
            self.ref_policy_net.to(self.device)

        micro_cfg = getattr(self.config, "micro_dispatch", None)
        self.dispatcher = ValueGuidedMicroDispatcher(
            config=micro_cfg,
            agent=self.agent,
            device=self.device,
        )

    def is_running(self) -> bool:
        return self._is_running

    def is_paused(self) -> bool:
        return self._pause_event.is_set()

    def pause(self) -> None:
        self._pause_event.set()

    def resume(self) -> None:
        self._pause_event.clear()

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._is_running = False

    def collect_rollout(
        self, env: Any
    ) -> Tuple[int, float, float, int, float]:
        """Plays matches with League opponents until buffer is filled with Seat 0 transitions.

        Returns (episodes_completed, avg_pred_score, avg_real_score,
        steps_collected, rollout_win_rate).
        """
        self.agent.eval()
        self.buffer.clear()

        steps = 0
        episodes = 0
        predicted_scores: List[float] = []
        final_scores: List[float] = []
        p0_wins = 0
        target_steps = self.config.training.rollout_steps_per_epoch

        def _get_env_obs_and_mask(e: Any) -> Tuple[np.ndarray, np.ndarray]:
            if hasattr(e, "_get_obs") and hasattr(e, "get_action_mask"):
                return e._get_obs(), e.get_action_mask()
            return _reset_env_instance(e)

        def _reset_env_instance(e: Any) -> Tuple[np.ndarray, np.ndarray]:
            setup_m = getattr(getattr(self.config, "micro_dispatch", None), "setup_mode", "deterministic")
            disp = self.dispatcher if setup_m == "value_guided" else None
            try:
                return e.reset(setup_mode=setup_m, dispatcher=disp)
            except TypeError:
                return e.reset()

        if (
            getattr(self.config.training, "rgsc_enabled", False)
            and self.state_buffer is not None
            and len(self.state_buffer) > 0
            and np.random.rand() < getattr(self.config.training, "rgsc_reset_prob", 0.35)
        ):
            sampled_env = self.state_buffer.sample()
            if sampled_env is not None:
                env = sampled_env
                obs, mask = _get_env_obs_and_mask(env)
            else:
                obs, mask = _reset_env_instance(env)
        else:
            obs, mask = _reset_env_instance(env)
        last_p0_idx = -1

        # Match setup: sample opponents for seats 1..N-1
        num_players = getattr(env, "num_players", 4)
        if self.config.league.enabled:
            opponents = self.league_manager.sample_opponents(
                self.agent, num_opponents=max(1, num_players - 1)
            )
            participants = ["CurrentPolicy"] + [name for name, _ in opponents]
        else:
            opponents = []
            participants = ["CurrentPolicy"] * num_players

        while steps < target_steps and not self._stop_event.is_set():
            actor = getattr(env, "current_player", 0)

            if actor == 0:
                obs_tensor = torch.from_numpy(obs).float().to(self.device)
                mask_tensor = torch.from_numpy(mask).bool().to(self.device)

                action, log_prob = self.agent.act_policy(
                    obs_tensor, mask_tensor, deterministic=False
                )

                step_res = self.dispatcher.step(env, action)
                reward = float(step_res.reward)

                # Regret tracking for RGSC (Go-Exploit)
                if (
                    getattr(self.config.training, "rgsc_enabled", False)
                    and self.state_buffer is not None
                    and not step_res.done
                    and getattr(env, "round", 1) >= 2
                    and steps % 3 == 0
                ):
                    with torch.no_grad():
                        curr_val = float(self.agent.score_net(obs_tensor).item())
                        next_obs_t = torch.from_numpy(step_res.obs).float().to(self.device)
                        next_val = float(self.agent.score_net(next_obs_t).item())
                    regret = max(0.0, curr_val - next_val)
                    if regret >= getattr(self.config.training, "rgsc_regret_threshold", 0.40):
                        self.state_buffer.add(env, regret=regret, round_num=getattr(env, "round", 1))

                # Dynamic Annealed Reward Shaping (Milestone exploration on 3130 flat action space)
                shaping_w = self.compute_scheduled_shaping_weight()
                if shaping_w > 1e-4:
                    milestone_bonus = 0.0
                    if 0 <= action < 200:         # BuildMine
                        milestone_bonus = 0.3
                    elif 400 <= action < 600:       # UpgradeTradingStation
                        milestone_bonus = 0.4
                    elif 600 <= action < 800:     # UpgradeResearchLab
                        milestone_bonus = 0.5
                    elif 800 <= action < 1000:    # UpgradePlanetaryInstitute
                        milestone_bonus = 0.5
                    elif 1000 <= action < 1200:   # UpgradeAcademy
                        milestone_bonus = 0.5
                    elif 1400 <= action < 1406:   # FormFederation
                        milestone_bonus = 1.0
                    elif 1406 <= action < 1412:   # AdvanceResearch
                        milestone_bonus = 0.5
                    elif 1444 <= action < 1498:   # ClaimTechTile
                        milestone_bonus = 0.4
                    elif 1498 <= action < 2308:   # ClaimAdvTechTile
                        milestone_bonus = 0.6
                    elif 2308 <= action < 3108:   # ExploreSpaceship
                        milestone_bonus = 0.5
                    elif 200 <= action < 400:     # StartGaiaProject
                        milestone_bonus = 0.3
                    reward += shaping_w * milestone_bonus

                # RND Intrinsic Curiosity Reward (safely bounded in [0, 1])
                rnd_w = self.compute_scheduled_rnd_weight()
                if rnd_w > 1e-4 and self.rnd is not None:
                    int_reward = self.rnd.compute_intrinsic_reward(obs_tensor)
                    clamped_int = float(np.clip(int_reward, 0.0, 1.0))
                    reward += rnd_w * clamped_int

                last_p0_idx = self.buffer.ptr
                self.buffer.add(
                    obs=obs,
                    action=action,
                    reward=reward,
                    done=step_res.done,
                    value=0.0,
                    log_prob=log_prob,
                    action_mask=mask,
                    player_id=0,
                )
                steps += 1
            else:
                # Opponent action
                if self.config.league.enabled and opponents:
                    opp_idx = (actor - 1) % len(opponents)
                    _, act_fn = opponents[opp_idx]
                    action = act_fn(obs, mask)
                else:
                    obs_tensor = torch.from_numpy(obs).float().to(self.device)
                    mask_tensor = torch.from_numpy(mask).bool().to(self.device)
                    action = self.agent.act(
                        obs_tensor, mask_tensor, deterministic=False
                    )

                # Record opponent action for opponent modeling auxiliary task
                if last_p0_idx >= 0 and self.buffer.size() > 0:
                    self.buffer.set_opponent_action(last_p0_idx, action)
                    last_p0_idx = -1

                step_res = env.step(action)

            obs = step_res.obs
            mask = step_res.action_mask

            if step_res.done:
                episodes += 1
                # Mark last seat 0 transition as done to prevent cross-game advantage bleeding
                if last_p0_idx >= 0 and self.buffer.size() > 0:
                    self.buffer.dones[last_p0_idx] = True
                    last_p0_idx = -1

                vps = step_res.info.get(
                    "player_vp",
                    [p["vp"] for p in getattr(env, "players_state", [{"vp": 0}, {"vp": 0}])]
                )
                if len(vps) >= 2 and vps[0] > max(vps[1:]):
                    p0_wins += 1
                final_scores.extend(vps)

                if self.config.league.enabled:
                    self.league_manager.update_match_results(participants, vps)

                if (
                    getattr(self.config.training, "rgsc_enabled", False)
                    and self.state_buffer is not None
                    and len(self.state_buffer) > 0
                    and np.random.rand() < getattr(self.config.training, "rgsc_reset_prob", 0.35)
                ):
                    sampled_env = self.state_buffer.sample()
                    if sampled_env is not None:
                        env = sampled_env
                        obs, mask = _get_env_obs_and_mask(env)
                    else:
                        obs, mask = _reset_env_instance(env)
                else:
                    obs, mask = _reset_env_instance(env)

                if self.config.league.enabled:
                    opponents = self.league_manager.sample_opponents(
                        self.agent, num_opponents=max(1, num_players - 1)
                    )
                    participants = ["CurrentPolicy"] + [name for name, _ in opponents]

        buf_size = self.buffer.size()
        avg_pred = 0.0
        if buf_size > 0:
            with torch.no_grad():
                buf_obs_t = torch.from_numpy(self.buffer.observations[:buf_size]).float().to(self.device)
                if self.use_amp and self.device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        batched_vals = self.agent.score_net(buf_obs_t).squeeze(-1)
                else:
                    batched_vals = self.agent.score_net(buf_obs_t).squeeze(-1)
                self.buffer.values[:buf_size] = batched_vals.cpu().numpy()
                avg_pred = float(batched_vals.mean().item())

        last_val = 0.0
        if not step_res.done:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).float().to(self.device)
                last_val = self.agent.score_net.predict_score(obs_t)

        self.buffer.compute_returns_and_advantages(
            last_val=last_val,
            done=step_res.done,
            gamma=self.config.training.gamma,
            gae_lambda=self.config.training.gae_lambda,
        )

        avg_real = float(np.mean(final_scores)) if final_scores else avg_pred
        rollout_wr = float(p0_wins) / float(max(1, episodes)) if episodes > 0 else 0.5

        return episodes, avg_pred, avg_real, steps, rollout_wr

    def compute_scheduled_lr(self, base_lr: float) -> float:
        """Calculates current learning rate based on warmup and decay schedule."""
        total = max(1, self.config.training.total_episodes)
        epoch = self.current_epoch
        warmup_epochs = max(1, int(total * getattr(self.config.training, "warmup_ratio", 0.05)))
        min_factor = getattr(self.config.training, "lr_final_factor", 0.1)
        sched_type = getattr(self.config.training, "lr_schedule_type", "cosine").lower()

        if epoch < warmup_epochs:
            # Linear warmup from 10% to 100% of base_lr
            warmup_frac = float(epoch + 1) / float(warmup_epochs)
            return base_lr * (0.1 + 0.9 * warmup_frac)

        # Post-warmup progress in [0.0, 1.0]
        progress = float(epoch - warmup_epochs) / float(max(1, total - warmup_epochs))
        progress = min(1.0, max(0.0, progress))

        if sched_type == "cosine":
            factor = min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress))
        elif sched_type == "linear":
            factor = 1.0 - (1.0 - min_factor) * progress
        elif sched_type == "exponential":
            rate = getattr(self.config.training, "exp_decay_rate", 0.98)
            factor = max(min_factor, rate ** (epoch - warmup_epochs))
        else:  # 'constant'
            factor = 1.0

        return max(getattr(self.config.training, "min_lr", 1e-5), base_lr * factor)

    def compute_scheduled_entropy(self) -> float:
        """Calculates current entropy coefficient based on schedule."""
        total = max(1, self.config.training.total_episodes)
        epoch = self.current_epoch
        sched_type = getattr(self.config.training, "entropy_schedule_type", "cosine").lower()
        start = getattr(self.config.training, "entropy_start", 0.05)
        end = getattr(self.config.training, "entropy_end", 0.005)

        progress = min(1.0, max(0.0, float(epoch) / float(total)))

        if sched_type == "cosine":
            return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))
        elif sched_type == "linear":
            return start - (start - end) * progress
        elif sched_type == "exponential":
            rate = getattr(self.config.training, "exp_decay_rate", 0.98)
            return max(end, start * (rate ** epoch))
        return start

    def compute_scheduled_shaping_weight(self) -> float:
        """Calculates annealed reward shaping weight via exponential decay."""
        if not getattr(self.config.training, "shaping_enabled", True):
            return 0.0
        w0 = getattr(self.config.training, "shaping_initial_weight", 1.0)
        decay = getattr(self.config.training, "shaping_decay_rate", 0.96)
        min_w = getattr(self.config.training, "shaping_min_weight", 0.0)
        epoch = self.current_epoch
        weight = max(min_w, w0 * (decay ** epoch))
        return float(weight)

    def compute_scheduled_rnd_weight(self) -> float:
        """Calculates annealed RND intrinsic exploration weight."""
        if not getattr(self.config.training, "rnd_enabled", True) or self.rnd is None:
            return 0.0
        w0 = getattr(self.config.training, "rnd_initial_weight", 0.05)
        decay = getattr(self.config.training, "rnd_decay_rate", 0.98)
        epoch = self.current_epoch
        return float(w0 * (decay ** epoch))

    def train_epoch(self) -> Tuple[float, float, float, float]:
        """Runs PPO mini-batch updates on the collected rollout buffer with dynamic schedules and value clipping."""
        self.agent.train()

        # 1. Dynamically update learning rates for this epoch
        curr_policy_lr = self.compute_scheduled_lr(self.config.model.policy_lr)
        curr_score_lr = self.compute_scheduled_lr(self.config.model.score_lr)
        for g in self.policy_optimizer.param_groups:
            g["lr"] = curr_policy_lr
        for g in self.value_optimizer.param_groups:
            g["lr"] = curr_score_lr

        # 2. Dynamically update entropy coefficient
        entropy_coef = self.compute_scheduled_entropy()

        total_p_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        total_rnad_kl = 0.0
        num_batches = 0

        clip_eps = self.config.training.clip_epsilon
        val_clip_eps = getattr(self.config.training, "value_clip_epsilon", 0.2)
        epochs = self.config.training.train_epochs_per_rollout
        batch_size = self.config.training.batch_size

        for _ in range(epochs):
            for batch in self.buffer.get_batches(batch_size, self.device):
                # Mini-batch advantage normalization for numerical conditioning
                advantages = batch.advantages
                adv_std = advantages.std()
                if adv_std > 1e-6:
                    advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

                # 1. Score Predictor (Value Network) Update with PPO Value Clipping
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        values_pred = self.agent.score_net(batch.observations)
                        v_loss_unclipped = F.smooth_l1_loss(values_pred, batch.returns, reduction="none")
                        v_clipped = batch.values + torch.clamp(values_pred - batch.values, -val_clip_eps, val_clip_eps)
                        v_loss_clipped = F.smooth_l1_loss(v_clipped, batch.returns, reduction="none")
                        v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    self.value_optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(v_loss).backward()
                    self.scaler.unscale_(self.value_optimizer)
                    nn.utils.clip_grad_norm_(
                        self.agent.score_net.parameters(),
                        self.config.training.max_grad_norm,
                    )
                    self.scaler.step(self.value_optimizer)
                else:
                    values_pred = self.agent.score_net(batch.observations)
                    v_loss_unclipped = F.smooth_l1_loss(values_pred, batch.returns, reduction="none")
                    v_clipped = batch.values + torch.clamp(values_pred - batch.values, -val_clip_eps, val_clip_eps)
                    v_loss_clipped = F.smooth_l1_loss(v_clipped, batch.returns, reduction="none")
                    v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    self.value_optimizer.zero_grad(set_to_none=True)
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.agent.score_net.parameters(),
                        self.config.training.max_grad_norm,
                    )
                    self.value_optimizer.step()

                # 2. Action Optimizer (Policy Network) Update
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        logits, opp_logits = self.agent.action_net(
                            batch.observations, batch.action_masks, return_opponent=True
                        )
                        probs = F.softmax(logits, dim=-1)
                        dist = Categorical(probs=probs)
                        new_log_probs = dist.log_prob(batch.actions)
                        entropy = dist.entropy().mean()

                        ratios = torch.exp(new_log_probs - batch.log_probs)
                        surr1 = ratios * advantages
                        surr2 = (
                            torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps)
                            * advantages
                        )
                        p_loss = (
                            -torch.min(surr1, surr2).mean()
                            - entropy_coef * entropy
                        )
                        if getattr(self.config.training, "use_opponent_modeling", True) and batch.has_opponents.any():
                            opp_loss = F.cross_entropy(
                                opp_logits[batch.has_opponents],
                                batch.opponent_actions[batch.has_opponents],
                            )
                            opp_coef = getattr(self.config.training, "opponent_loss_coef", 0.25)
                            p_loss = p_loss + opp_coef * opp_loss

                        # Regularized Nash Dynamics (R-NaD) Relative Entropy Penalty
                        kl_div = torch.tensor(0.0, device=self.device)
                        if getattr(self.config.training, "rnad_enabled", True) and self.ref_policy_net is not None:
                            with torch.no_grad():
                                ref_logits = self.ref_policy_net(batch.observations, batch.action_masks)
                            p_log_probs = F.log_softmax(logits, dim=-1)
                            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                            diff = torch.where(batch.action_masks, p_log_probs - ref_log_probs, torch.zeros_like(p_log_probs))
                            kl_div = (probs * diff).sum(dim=-1).mean()
                            rnad_alpha = getattr(self.config.training, "rnad_alpha", 0.05)
                            p_loss = p_loss + rnad_alpha * kl_div

                    self.policy_optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(p_loss).backward()
                    self.scaler.unscale_(self.policy_optimizer)
                    nn.utils.clip_grad_norm_(
                        self.agent.action_net.parameters(),
                        self.config.training.max_grad_norm,
                    )
                    self.scaler.step(self.policy_optimizer)
                    self.scaler.update()
                else:
                    logits, opp_logits = self.agent.action_net(
                        batch.observations, batch.action_masks, return_opponent=True
                    )
                    probs = F.softmax(logits, dim=-1)
                    dist = Categorical(probs=probs)
                    new_log_probs = dist.log_prob(batch.actions)
                    entropy = dist.entropy().mean()

                    ratios = torch.exp(new_log_probs - batch.log_probs)
                    surr1 = ratios * advantages
                    surr2 = (
                        torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps)
                        * advantages
                    )
                    p_loss = (
                        -torch.min(surr1, surr2).mean() - entropy_coef * entropy
                    )
                    if getattr(self.config.training, "use_opponent_modeling", True) and batch.has_opponents.any():
                        opp_loss = F.cross_entropy(
                            opp_logits[batch.has_opponents],
                            batch.opponent_actions[batch.has_opponents],
                        )
                        opp_coef = getattr(self.config.training, "opponent_loss_coef", 0.25)
                        p_loss = p_loss + opp_coef * opp_loss

                    # Regularized Nash Dynamics (R-NaD) Relative Entropy Penalty
                    kl_div = torch.tensor(0.0, device=self.device)
                    if getattr(self.config.training, "rnad_enabled", True) and self.ref_policy_net is not None:
                        with torch.no_grad():
                            ref_logits = self.ref_policy_net(batch.observations, batch.action_masks)
                        p_log_probs = F.log_softmax(logits, dim=-1)
                        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                        diff = torch.where(batch.action_masks, p_log_probs - ref_log_probs, torch.zeros_like(p_log_probs))
                        kl_div = (probs * diff).sum(dim=-1).mean()
                        rnad_alpha = getattr(self.config.training, "rnad_alpha", 0.05)
                        p_loss = p_loss + rnad_alpha * kl_div

                    self.policy_optimizer.zero_grad(set_to_none=True)
                    p_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.agent.action_net.parameters(),
                        self.config.training.max_grad_norm,
                    )
                    self.policy_optimizer.step()

                # 3. RND Distillation Predictor Training
                if self.rnd is not None:
                    self.rnd.train_step(batch.observations)

                total_p_loss += float(p_loss.item())
                total_v_loss += float(v_loss.item())
                total_entropy += float(entropy.item())
                total_rnad_kl += float(kl_div.item())
                num_batches += 1

        avg_p = total_p_loss / max(1, num_batches)
        avg_v = total_v_loss / max(1, num_batches)
        avg_e = total_entropy / max(1, num_batches)
        self.last_rnad_kl = total_rnad_kl / max(1, num_batches)
        total_l = avg_p + self.config.training.value_loss_coef * avg_v

        return avg_p, avg_v, avg_e, total_l

    def evaluate_model(
        self,
        num_games: int = 5,
        use_mcts: bool = False,
        mcts_simulations: int = 25,
    ) -> Tuple[float, float, List[Dict[str, Any]]]:
        """Evaluates trained agent against random legal players, optionally using MCTS lookahead search."""
        env = make_gaia_env(players=self.config.model.num_players)
        wins = 0
        scores: List[float] = []
        logs: List[Dict[str, Any]] = []

        mcts_engine = None
        if use_mcts:
            from mcts import MultiPlayerMCTS
            mcts_engine = MultiPlayerMCTS(self.agent, device=self.device)

        self.agent.eval()
        setup_m = getattr(getattr(self.config, "micro_dispatch", None), "setup_mode", "deterministic")
        disp = self.dispatcher if setup_m == "value_guided" else None
        for g in range(num_games):
            try:
                obs, mask = env.reset(setup_mode=setup_m, dispatcher=disp)
            except TypeError:
                obs, mask = env.reset()
            game_history = []
            steps = 0
            while not getattr(env, "terminated", False) and steps < 120:
                steps += 1
                curr_p = getattr(env, "current_player", 0)
                obs_t = torch.from_numpy(obs).float().to(self.device)
                mask_t = torch.from_numpy(mask).bool().to(self.device)

                if curr_p == 0:
                    if mcts_engine is not None:
                        action, _, _ = mcts_engine.search(
                            env, num_simulations=mcts_simulations, temperature=0.0
                        )
                        with torch.no_grad():
                            pred_vp = float(self.agent.score_net(obs_t).item())
                    else:
                        action, _, pred_vp, probs = self.agent.act_and_evaluate(
                            obs_t, mask_t, deterministic=True
                        )
                else:
                    legal_indices = np.where(mask)[0]
                    action = int(np.random.choice(legal_indices))
                    pred_vp = 0.0

                action_names = getattr(env, "ACTION_NAMES", [])
                action_name = action_names[action] if action < len(action_names) else f"Action_{action}"
                game_history.append({
                    "player": curr_p,
                    "action_name": action_name,
                    "pred_vp": pred_vp,
                })

                if curr_p == 0:
                    res = self.dispatcher.step(env, action, deterministic=True)
                else:
                    res = env.step(action)
                obs = res.obs
                mask = res.action_mask

            final_vps = res.info.get("player_vp", [p["vp"] for p in getattr(env, "players_state", [{"vp": 0}, {"vp": 0}])])
            if len(final_vps) >= 2 and final_vps[0] > max(final_vps[1:]):
                wins += 1
            scores.append(final_vps[0] if final_vps else 0.0)
            logs.append({"game": g + 1, "vps": final_vps, "steps": len(game_history)})

        win_rate = float(wins) / float(max(1, num_games))
        avg_score = float(np.mean(scores)) if scores else 0.0
        return win_rate, avg_score, logs

    def update_rnad_reference(self) -> None:
        """Applies Polyak moving average update to R-NaD reference policy net:
        θ_ref <- (1 - β) * θ_ref + β * θ
        """
        if (
            getattr(self.config.training, "rnad_enabled", True)
            and self.ref_policy_net is not None
        ):
            beta = getattr(self.config.training, "rnad_polyak_beta", 0.20)
            with torch.no_grad():
                for ref_p, p in zip(self.ref_policy_net.parameters(), self.agent.action_net.parameters()):
                    ref_p.data.mul_(1.0 - beta).add_(p.data, alpha=beta)

    def train_step(self, env: Any) -> TrainingMetrics:
        """Runs one full training iteration (Rollout -> PPO Updates -> Telemetry)."""
        t0 = time.time()
        episodes, avg_pred, avg_real, steps, rollout_wr = self.collect_rollout(env)
        p_loss, v_loss, entropy, total_loss = self.train_epoch()
        elapsed = max(1e-4, time.time() - t0)
        steps_per_sec = steps / elapsed

        self.current_epoch += 1
        self.total_episodes_played += episodes

        # R-NaD Reference Policy Periodic Polyak Update
        if (
            getattr(self.config.training, "rnad_enabled", True)
            and self.ref_policy_net is not None
            and self.current_epoch % getattr(self.config.training, "rnad_ref_update_interval", 10) == 0
        ):
            self.update_rnad_reference()

        vram_mb = 0.0
        if self.device.type == "cuda":
            vram_mb = torch.cuda.memory_allocated(self.device) / (1024**2)

        # Dynamic win rate tracking
        if episodes > 0:
            self._last_win_rate = 0.8 * self._last_win_rate + 0.2 * rollout_wr
        if self.current_epoch % self.config.training.eval_interval_episodes == 0:
            val_wr, _, _ = self.evaluate_model(num_games=4)
            self._last_win_rate = val_wr

        # Snapshot policy into League pool at configured intervals
        if (
            self.config.league.enabled
            and self.current_epoch % self.config.league.snapshot_interval_epochs == 0
        ):
            self.league_manager.add_snapshot(self.agent, self.current_epoch)

        curr_elo = (
            self.league_manager.members["CurrentPolicy"].elo
            if "CurrentPolicy" in self.league_manager.members
            else 1200.0
        )
        l_size = len(self.league_manager.members)

        return TrainingMetrics(
            epoch=self.current_epoch,
            episodes=self.total_episodes_played,
            policy_loss=p_loss,
            value_loss=v_loss,
            total_loss=total_loss,
            entropy=entropy,
            avg_predicted_score=avg_pred,
            avg_real_score=avg_real,
            win_rate=self._last_win_rate,
            steps_per_sec=steps_per_sec,
            vram_allocated_mb=vram_mb,
            device_name=(
                self.hw_info.get("device_name", "CPU")
                if self.device.type == "cuda"
                else "CPU"
            ),
            league_elo=curr_elo,
            league_size=l_size,
            rnad_kl=getattr(self, "last_rnad_kl", 0.0),
        )

    def start_background_training(
        self,
        on_metrics: Optional[Callable[[TrainingMetrics], None]] = None,
        on_finished: Optional[Callable[[], None]] = None,
        max_epochs: int = 500,
    ) -> None:
        """Spawns background training worker thread."""
        if self._is_running:
            return

        self._stop_event.clear()
        self._pause_event.clear()
        self._is_running = True

        def _worker():
            env = make_gaia_env(players=self.config.model.num_players)
            try:
                while (
                    not self._stop_event.is_set()
                    and self.current_epoch < max_epochs
                ):
                    if self._pause_event.is_set():
                        time.sleep(0.1)
                        continue

                    metrics = self.train_step(env)
                    if on_metrics is not None:
                        on_metrics(metrics)

                    if (
                        self.current_epoch
                        % self.config.training.save_checkpoint_interval
                        == 0
                    ):
                        self.save_current_checkpoint(is_milestone=True)

            except Exception as e:
                print(f"[Trainer Error]: {e}")
            finally:
                self._is_running = False
                if on_finished is not None:
                    on_finished()

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def save_current_checkpoint(self, is_milestone: bool = True) -> str:
        """Saves current policy weights, latest snapshot, and peak Elo checkpoint."""
        ckpt_dir = self.config.training.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        curr_elo = (
            self.league_manager.members["CurrentPolicy"].elo
            if "CurrentPolicy" in self.league_manager.members
            else 1200.0
        )
        meta = {
            "epoch": self.current_epoch,
            "episodes": self.total_episodes_played,
            "elo": curr_elo,
            "win_rate": self._last_win_rate,
            "saved_at": time.time(),
        }

        # 1. Always update gaia_latest.pt for seamless auto-resume
        latest_path = os.path.join(ckpt_dir, "gaia_latest.pt")
        self.agent.save_checkpoint(latest_path, meta)

        # 2. If new peak Elo reached, save gaia_best_elo.pt
        if curr_elo > self.best_elo:
            self.best_elo = curr_elo
            best_path = os.path.join(ckpt_dir, "gaia_best_elo.pt")
            self.agent.save_checkpoint(best_path, meta)

        # 3. If milestone epoch, save versioned checkpoint
        milestone_path = os.path.join(ckpt_dir, f"gaia_epoch_{self.current_epoch}.pt")
        if is_milestone:
            self.agent.save_checkpoint(milestone_path, meta)

        return latest_path

    def resume_from_checkpoint(self, checkpoint_path: str) -> int:
        """Loads model weights and training state from checkpoint file. Returns resumed epoch."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        meta = self.agent.load_checkpoint(checkpoint_path, device=self.device)
        resumed_epoch = int(meta.get("epoch", 0))
        self.current_epoch = resumed_epoch
        self.total_episodes_played = int(meta.get("episodes", resumed_epoch * 10))
        if "elo" in meta and "CurrentPolicy" in self.league_manager.members:
            res_elo = float(meta["elo"])
            self.league_manager.members["CurrentPolicy"].elo = res_elo
            self.best_elo = max(self.best_elo, res_elo)
        print(f"[RLTrainer] Resumed successfully from {checkpoint_path} at epoch {resumed_epoch} (Elo: {self.best_elo:.1f}).")
        return resumed_epoch

