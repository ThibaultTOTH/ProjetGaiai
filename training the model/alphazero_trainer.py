import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import random
import threading

from config import AppConfig
from environment import compute_competitive_value
from models import DualGaiaAgent
from mcts import MultiPlayerMCTS
from league import LeagueManager, LeagueMember
from buffer import PrioritizedStateBuffer

class AlphaZeroReplayBuffer:
    """Stores (observation, action_mask, mcts_policy, mcts_value) tuples from self-play.

    Features:
    1. Sliding Buffer: Standard rolling FIFO buffer for recent transitions (capacity: 50,000+).
    2. Elite Reservoir Replay: Dedicated permanent pool of decisive moves from the top 0.1%
       highest-scoring winning games (capacity: 25,000).
    3. Hybrid Balanced Sampling: Blends fresh self-play exploration (75%) with permanent
       mastery of breakthrough high-VP winning trajectories (25%).
    """
    def __init__(
        self,
        capacity: int = 50_000,
        elite_capacity: int = 25_000,
        expert_dataset_path: Optional[str] = None,
    ):
        self.capacity = capacity
        self.elite_capacity = elite_capacity
        self.buffer = []
        self.position = 0
        self.elite_buffer = []
        self.winning_vps: List[float] = []
        self.expert_obs = None
        self.expert_masks = None
        self.expert_acts = None
        self.expert_vals = None
        self.num_expert_samples = 0

        # Load Grandmaster Expert Anchor Dataset if present
        if expert_dataset_path and os.path.exists(expert_dataset_path):
            try:
                print(f"  📖 [Replay Buffer] Loading Grandmaster Expert Anchor from {expert_dataset_path}...", flush=True)
                d = torch.load(expert_dataset_path, map_location="cpu", weights_only=False)
                self.expert_obs = d["observations"]
                self.expert_masks = d["action_masks"]
                self.expert_acts = d["actions"]
                self.expert_vals = d["values"]
                self.num_expert_samples = len(self.expert_acts)
                print(
                    f"  ✨ [Replay Buffer] Loaded {self.num_expert_samples:,} Grandmaster transitions "
                    f"(Avg GM score: {d.get('avg_gm_score', 178):.1f} VP)!",
                    flush=True,
                )
            except Exception as e:
                print(f"  ⚠️ [Replay Buffer Warning] Could not load expert dataset: {e}", flush=True)

    def get_elite_threshold(self) -> float:
        """Computes dynamic top 0.01% (99.99th percentile) winning VP threshold.
        
        Requires a strict minimum bar of 100.0 VP so mediocre games (<100 VP)
        never enter the Elite Reservoir and pollute training data.
        """
        MIN_ELITE_BAR = 100.0
        if not self.winning_vps:
            return MIN_ELITE_BAR
        if len(self.winning_vps) < 20:
            return max(MIN_ELITE_BAR, float(np.max(self.winning_vps)))
        p_elite = float(np.percentile(self.winning_vps, 99.99))
        return max(MIN_ELITE_BAR, p_elite)

    def add(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        mcts_policy: np.ndarray,
        value_target: np.ndarray,
        value_weight: Optional[np.ndarray] = None,
    ):
        """Appends a single transition to the sliding FIFO buffer."""
        w = value_weight if value_weight is not None else np.ones(4, dtype=np.float32)
        data = (obs, action_mask, mcts_policy, value_target, w)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def add_game(
        self,
        history: List[Any],
        raw_vps: List[float],
        participants: List[str],
    ) -> None:
        """Ingests all transitions of a completed game, and preserves decisive moves of top 0.1% winners."""
        if not history:
            return

        winner_seat = int(np.argmax(raw_vps)) if raw_vps else 0
        winner_vp = float(raw_vps[winner_seat]) if raw_vps else 0.0
        self.winning_vps.append(winner_vp)
        if len(self.winning_vps) > 5000:
            self.winning_vps.pop(0)

        threshold = self.get_elite_threshold()
        is_elite_game = (
            winner_vp >= threshold
            and winner_seat < len(participants)
            and participants[winner_seat] == "CurrentPolicy"
        )

        n_elite_added = 0
        for step_data in history:
            obs = step_data[0]
            mask = step_data[1]
            probs = step_data[2]
            z_val = step_data[3]
            seat = step_data[4] if len(step_data) > 4 else None
            w = np.ones(4, dtype=np.float32)

            # 1. Add to sliding FIFO buffer
            self.add(obs, mask, probs, z_val, w)

            # 2. If game is top 0.1% winner, preserve decisive moves of the winner in Elite Reservoir!
            if is_elite_game and seat == winner_seat:
                elite_entry = (obs, mask, probs, z_val, w, winner_vp)
                if len(self.elite_buffer) < self.elite_capacity:
                    self.elite_buffer.append(elite_entry)
                else:
                    min_idx = int(np.argmin([x[5] for x in self.elite_buffer]))
                    if winner_vp > self.elite_buffer[min_idx][5]:
                        self.elite_buffer[min_idx] = elite_entry
                n_elite_added += 1

        if is_elite_game and n_elite_added > 0:
            print(
                f"  * [Elite Reservoir] Stored {n_elite_added} decisive moves of top 0.01% winner "
                f"({winner_vp:.1f} VP >= {threshold:.1f} VP threshold). Total elite pool: {len(self.elite_buffer)}",
                flush=True,
            )

    def sample(
        self,
        batch_size: int,
        optimism_power: float = 1.0,
        elite_ratio: float = 0.25,
        expert_ratio: float = 0.25,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(self.buffer) == 0 and len(self.elite_buffer) == 0 and self.num_expert_samples == 0:
            raise ValueError("Cannot sample from empty replay buffer")

        batch = []

        # If self-play buffer is still empty or minimal, sample 100% from Grandmaster expert dataset
        if len(self.buffer) < 32 and self.num_expert_samples > 0:
            exp_indices = np.random.choice(self.num_expert_samples, size=batch_size, replace=False)
            for idx in exp_indices:
                obs_np = self.expert_obs[idx].float().numpy()
                mask_np = self.expert_masks[idx].numpy()
                act_idx = int(self.expert_acts[idx].item())
                probs_np = np.zeros(3130, dtype=np.float32)
                if 0 <= act_idx < 3130:
                    probs_np[act_idx] = 1.0
                vals_np = np.zeros(4, dtype=np.float32)
                vals_np[0] = float(self.expert_vals[idx].item())
                weight_np = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                batch.append((obs_np, mask_np, probs_np, vals_np, weight_np))
        else:
            # Balanced Tri-Hybrid Sampling: Self-Play (50-75%), Elite Reservoir (optional), Grandmaster Anchor (25%)
            n_expert = int(batch_size * expert_ratio) if self.num_expert_samples > 0 else 0
            n_remaining = batch_size - n_expert
            n_elite = int(n_remaining * elite_ratio) if len(self.elite_buffer) >= 32 else 0
            n_recent = n_remaining - n_elite

            # 1. Sample from Grandmaster Expert Dataset
            if n_expert > 0:
                exp_indices = np.random.choice(self.num_expert_samples, size=n_expert, replace=False)
                for idx in exp_indices:
                    obs_np = self.expert_obs[idx].float().numpy()
                    mask_np = self.expert_masks[idx].numpy()
                    act_idx = int(self.expert_acts[idx].item())
                    probs_np = np.zeros(3130, dtype=np.float32)
                    if 0 <= act_idx < 3130:
                        probs_np[act_idx] = 1.0
                    vals_np = np.zeros(4, dtype=np.float32)
                    vals_np[0] = float(self.expert_vals[idx].item())
                    weight_np = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                    batch.append((obs_np, mask_np, probs_np, vals_np, weight_np))

            # 2. Sample from Elite Reservoir
            if n_elite > 0:
                if len(self.elite_buffer) < n_elite:
                    elite_indices = np.random.choice(len(self.elite_buffer), size=n_elite, replace=True)
                else:
                    elite_indices = np.random.choice(len(self.elite_buffer), size=n_elite, replace=False)
                for idx in elite_indices:
                    x = self.elite_buffer[idx]
                    batch.append((x[0], x[1], x[2], x[3], x[4]))

            # 3. Sample from Sliding Buffer
            if n_recent > 0 and len(self.buffer) > 0:
                if len(self.buffer) < n_recent:
                    indices = np.random.choice(len(self.buffer), size=n_recent, replace=True)
                    batch.extend([self.buffer[i] for i in indices])
                else:
                    vps = np.array(
                        [float(np.max(x[3])) if hasattr(x[3], '__iter__') else float(x[3]) for x in self.buffer],
                        dtype=np.float32,
                    )
                    v_min = float(np.min(vps))
                    v_max = float(np.max(vps))
                    if v_max > v_min and optimism_power > 0.0:
                        normalized = (vps - v_min) / (v_max - v_min)
                        weights = (0.2 + 0.8 * normalized) ** optimism_power
                        probs = weights / weights.sum()
                        indices = np.random.choice(len(self.buffer), size=n_recent, replace=False, p=probs)
                        batch.extend([self.buffer[i] for i in indices])
                    else:
                        batch.extend(random.sample(self.buffer, n_recent))
            elif n_recent > 0 and self.num_expert_samples > 0:
                # Fallback to expert if buffer has insufficient items
                extra_exp = np.random.choice(self.num_expert_samples, size=n_recent, replace=True)
                for idx in extra_exp:
                    obs_np = self.expert_obs[idx].float().numpy()
                    mask_np = self.expert_masks[idx].numpy()
                    act_idx = int(self.expert_acts[idx].item())
                    probs_np = np.zeros(3130, dtype=np.float32)
                    if 0 <= act_idx < 3130:
                        probs_np[act_idx] = 1.0
                    vals_np = np.zeros(4, dtype=np.float32)
                    vals_np[0] = float(self.expert_vals[idx].item())
                    weight_np = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                    batch.append((obs_np, mask_np, probs_np, vals_np, weight_np))

        obs, action_mask, mcts_policy, value_target, value_weight = zip(*batch)
        return (
            np.stack(obs),
            np.stack(action_mask),
            np.stack(mcts_policy),
            np.array(value_target, dtype=np.float32),
            np.array(value_weight, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer) + len(self.elite_buffer)

@dataclass
class EpochMetrics:
    epoch: int
    policy_loss: float
    value_loss: float
    avg_predicted_score: float
    avg_real_score: float
    entropy: float
    duration: float
    win_rate: float = 0.5
    steps_per_sec: float = 0.0
    league_elo: float = 1200.0
    league_size: int = 2
    rgsc_puzzles: int = 0
    avg_winner_score: float = 0.0
    max_score: float = 0.0
    elite_size: int = 0
    target_ambition: float = 80.0


class AlphaZeroTrainer:
    def __init__(self, config: AppConfig, agent: DualGaiaAgent = None):
        self.config = config
        self.az_config = config.alphazero
        
        self.device = config.hardware.get_torch_device()
        self.hw_info = config.hardware.configure_cuda()
        if agent is None:
            self.agent = DualGaiaAgent(config.model).to(self.device)
        else:
            self.agent = agent.to(self.device)
            
        self.mcts = MultiPlayerMCTS(self.agent, config.mcts, self.device)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        expert_ds_path = os.path.join(project_root, "scraper", "data", "dataset", "gaia_certified_expert_dataset.pt")
        self.replay_buffer = AlphaZeroReplayBuffer(
            self.az_config.replay_buffer_size,
            expert_dataset_path=expert_ds_path,
        )

        # Population-Based League Training (Fictitious Self-Play)
        self.league_manager = None
        if getattr(self.config.league, "enabled", True):
            self.league_manager = LeagueManager(self.config.league, device=self.device)
            loaded = self.league_manager.load_existing_checkpoints(
                obs_dim=self.config.model.obs_dim,
                action_dim=self.config.model.action_dim,
                model_config=self.config.model,
            )
            if loaded > 0:
                print(f"[LeagueManager] Initialized with {loaded} historical snapshot(s).")

        # Regret-Guided Search Control (RGSC / Jeu sur Problème / Go-Exploit)
        self.state_buffer = None
        if getattr(self.config.training, "rgsc_enabled", True):
            self.state_buffer = PrioritizedStateBuffer(
                capacity=getattr(self.config.training, "rgsc_buffer_capacity", 200),
                regret_threshold=getattr(self.config.training, "rgsc_regret_threshold", 0.30),
            )
        
        parameters = list(self.agent.parameters())
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=config.model.policy_lr,
            weight_decay=config.model.policy_weight_decay
        )
        
        self.use_amp = config.hardware.use_mixed_precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.use_amp and self.device.type == "cuda"))
        
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._is_running = False
        self.current_epoch = 0

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
        self._is_running = False

    def compute_scheduled_lr(self, base_lr: float) -> float:
        """Calculates scheduled learning rate with warmup and cosine/exponential/linear decay."""
        cfg = self.config.training
        total_episodes = max(1, getattr(cfg, "total_episodes", 100))
        warmup_epochs = max(1, int(getattr(cfg, "warmup_ratio", 0.05) * total_episodes))
        lr_final_factor = getattr(cfg, "lr_final_factor", 0.10)
        min_lr = getattr(cfg, "min_lr", 1e-6)
        schedule_type = getattr(cfg, "lr_schedule_type", "cosine").lower()

        if self.current_epoch < warmup_epochs:
            alpha = float(self.current_epoch) / float(warmup_epochs)
            return float(min_lr + alpha * (base_lr - min_lr))

        progress = float(self.current_epoch - warmup_epochs) / float(max(1, total_episodes - warmup_epochs))
        progress = max(0.0, min(1.0, progress))

        if schedule_type == "cosine":
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return float(base_lr * (lr_final_factor + (1.0 - lr_final_factor) * decay))
        elif schedule_type == "linear":
            return float(base_lr * (1.0 - progress * (1.0 - lr_final_factor)))
        elif schedule_type == "exponential":
            rate = getattr(cfg, "exp_decay_rate", 0.98)
            return float(max(min_lr, base_lr * (rate ** (self.current_epoch - warmup_epochs))))
        return base_lr

    def update_learning_rate(self) -> float:
        """Applies scheduled learning rate to the optimizer."""
        base_lr = getattr(self.config.model, "policy_lr", 2.5e-4)
        new_lr = self.compute_scheduled_lr(base_lr)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr

    def self_play_game(self, env: Any) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]], float, bool, int]:
        rgsc_active = (
            getattr(self.config.training, "rgsc_enabled", True)
            and self.current_epoch >= getattr(self.config.training, "rgsc_start_epoch", 500)
        )
        # 1. Reset or Jump into a Mid-Game Crisis Puzzle (RGSC / Jeu sur problème)
        if (
            rgsc_active
            and self.state_buffer is not None
            and len(self.state_buffer) > 0
            and np.random.rand() < getattr(self.config.training, "rgsc_reset_prob", 0.30)
        ):
            sampled_env = self.state_buffer.sample()
            if sampled_env is not None:
                env = sampled_env
            else:
                env.reset()
        else:
            env.reset()

        if hasattr(env, "set_player_faction"):
            factions = np.random.choice(14, getattr(env, "num_players", 4), replace=False)
            for seat, faction_id in enumerate(factions):
                env.set_player_faction(seat, int(faction_id))

        # 2. Opponent Matchmaking from League Pool (Population / Historical)
        num_players = getattr(env, "num_players", 4)
        if getattr(self.config.league, "enabled", True) and self.league_manager is not None:
            opponents = self.league_manager.sample_opponents(
                self.agent, num_opponents=max(1, num_players - 1)
            )
            players = [("CurrentPolicy", None)] + opponents
            random.shuffle(players)
            participants = [name for name, _ in players]
        else:
            players = [("CurrentPolicy", None)] * num_players
            participants = ["CurrentPolicy"] * num_players

        history = []
        move_count = 0
        total_game_steps = 0
        consecutive_step_errors = 0
        # Increased to 1200 to accommodate Free Actions and Leech interrupts in a 4-player game.
        MAX_TOTAL_STEPS = 3000
        
        while not env.terminated and total_game_steps < MAX_TOTAL_STEPS:
            if self._stop_event.is_set():
                break
            while self._pause_event.is_set():
                if self._stop_event.is_set():
                    break
                time.sleep(0.2)

            total_game_steps += 1
            current_player = env.current_player
            obs = env._get_obs() if hasattr(env, '_get_obs') else env.observe().values
            mask = env.get_action_mask()
            
            legal_indices = np.where(mask)[0]
            if len(legal_indices) == 0:
                env.terminated = True
                break

            p_name, act_fn = players[current_player]
            if p_name == "CurrentPolicy":
                temp = self.az_config.temperature_high if move_count < self.az_config.temperature_threshold_move else self.az_config.temperature_low
                
                # Clone state for regret tracking if in round >= 2
                need_rgsc_clone = (
                    rgsc_active
                    and self.state_buffer is not None
                    and getattr(env, "round", 1) >= 2
                )
                try:
                    env_before_step = env.clone() if need_rgsc_clone else None
                except Exception:
                    env_before_step = None

                action, probs, search_meta = self.mcts.search(
                    env, 
                    num_simulations=self.az_config.num_simulations, 
                    temperature=temp,
                    add_noise=True,
                    epoch=self.current_epoch,
                )
                
                history.append((current_player, obs, mask, probs))
                move_count += 1
            else:
                # Opponents played by League Member action function
                action = act_fn(obs, mask)
                search_meta = {}
                    
            step_res = env.step(action)

            # Regret tracking for RGSC (Jeu sur problème / Crisis Puzzle Caching)
            if p_name == "CurrentPolicy" and env_before_step is not None and not env.terminated:
                try:
                    regret = float(search_meta.get("regret", 0.0))
                    thresh = getattr(self.config.training, "rgsc_regret_threshold", 0.30)
                    if regret >= thresh:
                        self.state_buffer.add(env_before_step, regret=regret, round_num=getattr(env_before_step, "round", 2))
                except Exception:
                    pass

            if hasattr(step_res, "info") and "error" in step_res.info:
                consecutive_step_errors += 1
                fallback_success = False
                alt_actions = np.random.permutation(legal_indices)
                for alt_act in alt_actions:
                    if alt_act == action:
                        continue
                    step_res = env.step(int(alt_act))
                    if not (hasattr(step_res, "info") and "error" in step_res.info):
                        fallback_success = True
                        consecutive_step_errors = 0
                        break
                if not fallback_success and consecutive_step_errors >= 5:
                    env.terminated = True
                    break
            else:
                consecutive_step_errors = 0
            
        raw_vps = [float(p.get("vp", 0.0)) for p in getattr(env, "players_state", [{"vp": 0.0}] * num_players)]
        
        # We need a primary perspective to report p0_vp/p0_won for metrics. We'll pick the first CurrentPolicy found.
        cp_indices = [i for i, (name, _) in enumerate(players) if name == "CurrentPolicy"]
        primary_idx = cp_indices[0] if cp_indices else 0
        p0_vp = raw_vps[primary_idx] if len(raw_vps) > primary_idx else 50.0
        p0_won = bool(len(raw_vps) >= 2 and p0_vp > max([v for i, v in enumerate(raw_vps) if i != primary_idx] + [0.0]))
        
        # Update multi-player Elo in the League
        if (
            getattr(self.config.league, "enabled", True)
            and self.league_manager is not None
            and len(participants) == len(raw_vps)
        ):
            self.league_manager.update_match_results(participants, raw_vps)

        # Store bounded 4-player competitive value vector Z in [-1, +1]
        final_history = [
            (
                obs,
                mask,
                probs,
                np.array([
                    compute_competitive_value(raw_vps, (p + rel_s) % 4, epoch=getattr(self, "current_epoch", 0))
                    for rel_s in range(4)
                ], dtype=np.float32),
                p,
            )
            for p, obs, mask, probs in history
        ]
        return final_history, p0_vp, p0_won, move_count

    def train_on_batch(self, batch: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[float, float, float, float]:
        obs_b, mask_b, policy_b, value_b, weight_b = batch
        
        obs_t = torch.from_numpy(obs_b).float().to(self.device)
        mask_t = torch.from_numpy(mask_b).bool().to(self.device)
        policy_t = torch.from_numpy(policy_b).float().to(self.device)
        value_t = torch.from_numpy(value_b).float().to(self.device)
        weight_t = torch.from_numpy(weight_b).float().to(self.device)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        device_type = "cuda" if (self.use_amp and self.device.type == "cuda") else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=(self.use_amp and self.device.type == "cuda")):
            pred_values = self.agent.score_net(obs_t)
            logits = self.agent.action_net(obs_t, mask_t)
            
            # Smooth L1 loss on bounded 4-player values weighted across active seats
            elem_loss = F.smooth_l1_loss(pred_values, value_t, reduction="none")
            value_loss = (elem_loss * weight_t).sum() / (weight_t.sum() + 1e-8)
            
            # Safe Cross-Entropy / Policy Loss:
            # 1. Normalize target policy over legal actions only
            policy_t_masked = torch.where(mask_t, policy_t, torch.zeros_like(policy_t))
            policy_sum = policy_t_masked.sum(dim=-1, keepdim=True)
            policy_t_masked = torch.where(policy_sum > 0, policy_t_masked / policy_sum, torch.zeros_like(policy_t_masked))
            
            # 2. Compute log_softmax
            log_probs = F.log_softmax(logits, dim=-1)
            
            # 3. Only evaluate cross-entropy where mask is True and target prob > 0 (prevents 0.0 * -inf = NaN)
            safe_terms = torch.where(mask_t & (policy_t_masked > 0), policy_t_masked * log_probs, torch.zeros_like(log_probs))
            policy_loss = -safe_terms.sum(dim=-1).mean()
            
            # 4. Guarantee finite scalars
            policy_loss = torch.nan_to_num(policy_loss, nan=0.0, posinf=10.0, neginf=-10.0)
            value_loss = torch.nan_to_num(value_loss, nan=0.0, posinf=10.0, neginf=-10.0)
            
            loss = policy_loss + self.az_config.value_loss_coef * value_loss
            
            # Safe entropy calculation over legal actions
            probs = torch.where(mask_t, torch.exp(log_probs), torch.zeros_like(log_probs))
            safe_lp = torch.where(mask_t, log_probs, torch.zeros_like(log_probs))
            entropy = -(probs * safe_lp).sum(dim=-1).mean()
            entropy = torch.nan_to_num(entropy, nan=0.0)

        self.scaler.scale(loss).backward()
        
        if self.config.training.max_grad_norm > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.training.max_grad_norm)
            
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return policy_loss.item(), value_loss.item(), entropy.item(), pred_values.mean().item()

    def run_training_loop(self, env: Any, max_epochs: int, callback=None):
        self._is_running = True
        self._stop_event.clear()
        self._pause_event.clear()
        
        start_epoch = getattr(self, "current_epoch", 0) + 1
        for epoch in range(start_epoch, max_epochs + 1):
            if self._stop_event.is_set():
                break
            while self._pause_event.is_set():
                if self._stop_event.is_set():
                    break
                time.sleep(0.2)
                
            self.current_epoch = epoch
            start_time = time.time()
            self.agent.eval()
            
            epoch_real_scores = []
            epoch_wins = 0
            epoch_games = 0
            epoch_moves = 0
            
            for _ in range(self.az_config.games_per_epoch):
                if self._stop_event.is_set():
                    break
                game_history, p0_vp, p0_won, moves = self.self_play_game(env)
                for step_data in game_history:
                    self.replay_buffer.add(*step_data)
                
                epoch_real_scores.append(p0_vp)
                if p0_won:
                    epoch_wins += 1
                epoch_games += 1
                epoch_moves += moves
            
            if self._stop_event.is_set():
                break
                
            self.agent.train()
            total_p_loss = 0.0
            total_v_loss = 0.0
            total_entropy = 0.0
            total_pred_score = 0.0
            
            if len(self.replay_buffer) >= 16:
                steps = self.az_config.training_steps_per_epoch
            else:
                steps = 0
                
            for _ in range(steps):
                if self._stop_event.is_set():
                    break
                batch = self.replay_buffer.sample(
                    self.az_config.batch_size,
                    optimism_power=getattr(self.az_config, "optimism_power", 1.0)
                )
                p_loss, v_loss, ent, pred_val = self.train_on_batch(batch)
                total_p_loss += p_loss
                total_v_loss += v_loss
                total_entropy += ent
                total_pred_score += pred_val
                
            duration = max(1e-4, time.time() - start_time)
            speed = epoch_moves / duration
            win_r = float(epoch_wins) / float(max(1, epoch_games))
            
            # Snapshot policy into League pool
            if (
                getattr(self.config.league, "enabled", True)
                and self.league_manager is not None
                and epoch % getattr(self.config.league, "snapshot_interval_epochs", 10) == 0
            ):
                snap_name = self.league_manager.add_snapshot(self.agent, epoch)
                print(f"[LeagueManager] Snapshot saved: {snap_name} (Current Elo: {self.league_manager.members['CurrentPolicy'].elo:.1f})")

            curr_elo = self.league_manager.members["CurrentPolicy"].elo if self.league_manager else 1200.0
            curr_league_size = len(self.league_manager.members) if self.league_manager else 1
            puzzles_count = len(self.state_buffer) if self.state_buffer else 0

            if steps > 0:
                metrics = EpochMetrics(
                    epoch=epoch,
                    policy_loss=total_p_loss / steps,
                    value_loss=total_v_loss / steps,
                    avg_predicted_score=total_pred_score / steps,
                    avg_real_score=float(np.mean(epoch_real_scores)) if epoch_real_scores else 0.0,
                    entropy=total_entropy / steps,
                    duration=duration,
                    win_rate=win_r,
                    steps_per_sec=speed,
                    league_elo=curr_elo,
                    league_size=curr_league_size,
                    rgsc_puzzles=puzzles_count,
                )
                
                if callback:
                    callback(metrics)
                    
            if epoch % self.az_config.checkpoint_interval == 0:
                self.save_checkpoint(os.path.join(self.config.training.checkpoint_dir, f"az_checkpoint_{epoch}.pt"))
                
        self._is_running = False

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        save_dict = {
            'epoch': self.current_epoch,
            'agent_state_dict': self.agent.state_dict(),
            'score_net_state': self.agent.score_net.state_dict(),
            'action_net_state': self.agent.action_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if hasattr(self, 'scaler') and self.scaler else None,
            'config': self.config,
            'meta': {'epoch': self.current_epoch, 'algorithm': 'AlphaZero'}
        }
        if hasattr(self, 'replay_buffer') and self.replay_buffer is not None:
            save_dict['winning_vps'] = getattr(self.replay_buffer, 'winning_vps', [])
            save_dict['elite_buffer'] = getattr(self.replay_buffer, 'elite_buffer', [])
        torch.save(save_dict, path)

    def save_current_checkpoint(self, path: Optional[str] = None, is_milestone: bool = False, **kwargs) -> str:
        ckpt_dir = self.config.training.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        if path is None:
            path = os.path.join(ckpt_dir, f"az_checkpoint_{self.current_epoch}.pt")
            if not is_milestone:
                latest_path = os.path.join(ckpt_dir, "gaia_latest.pt")
                self.save_checkpoint(latest_path)
        self.save_checkpoint(path)
        return path

    def load_checkpoint(self, path: str) -> int:
        if not os.path.exists(path):
            return 0
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)

        meta = self.agent.load_checkpoint(path, device=self.device)

        # After agent is loaded (and potentially rebuilt with new parameters), rebind optimizer if needed
        current_param_ids = {id(p) for p in self.agent.parameters()}
        opt_param_ids = {id(p) for group in self.optimizer.param_groups for p in group['params']}
        if current_param_ids != opt_param_ids:
            lr = getattr(self.config.model, "policy_lr", 1e-4)
            wd = getattr(self.config.model, "policy_weight_decay", 1e-4)
            self.optimizer = torch.optim.AdamW(self.agent.parameters(), lr=lr, weight_decay=wd)
            optimizer_rebuilt = True
        else:
            optimizer_rebuilt = False

        if not optimizer_rebuilt and isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception:
                pass
        if isinstance(checkpoint, dict) and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] and self.use_amp:
            try:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            except Exception:
                pass
        if isinstance(checkpoint, dict) and hasattr(self, 'replay_buffer') and self.replay_buffer is not None:
            if 'winning_vps' in checkpoint and checkpoint['winning_vps']:
                self.replay_buffer.winning_vps = [float(v) for v in checkpoint['winning_vps']]
            if 'elite_buffer' in checkpoint and checkpoint['elite_buffer']:
                # Filter out any entries < 100 VP to ensure elite pool purity
                self.replay_buffer.elite_buffer = [
                    e for e in checkpoint['elite_buffer'] if len(e) > 4 and e[4] >= 100.0
                ]
            
        epoch = 0
        if isinstance(meta, dict) and 'epoch' in meta:
            epoch = int(meta['epoch'])
        if epoch == 0:
            if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
                epoch = int(checkpoint['epoch'])
            elif isinstance(checkpoint, dict) and 'meta' in checkpoint and 'epoch' in checkpoint['meta']:
                epoch = int(checkpoint['meta']['epoch'])
            else:
                try:
                    filename = os.path.basename(path)
                    for part in filename.replace('.', '_').split('_'):
                        if part.isdigit():
                            epoch = int(part)
                except Exception:
                    pass
        self.current_epoch = epoch
        return epoch

    def resume_from_checkpoint(self, checkpoint_path: str) -> int:
        return self.load_checkpoint(checkpoint_path)

    def get_league_leaderboard(self) -> List[Dict[str, Any]]:
        """Returns leaderboard of current and historical league policies."""
        if self.league_manager is not None:
            return self.league_manager.get_leaderboard()
        return []

