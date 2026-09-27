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
from models import DualGaiaAgent
from mcts import MultiPlayerMCTS

class AlphaZeroReplayBuffer:
    """Stores (observation, action_mask, mcts_policy, mcts_value) tuples from self-play."""
    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, obs: np.ndarray, action_mask: np.ndarray, mcts_policy: np.ndarray, value_target: float):
        data = (obs, action_mask, mcts_policy, value_target)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, optimism_power: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(self.buffer) < batch_size:
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=True)
            batch = [self.buffer[i] for i in indices]
        else:
            vps = np.array([x[3] for x in self.buffer], dtype=np.float32)
            v_min = float(np.min(vps))
            v_max = float(np.max(vps))
            if v_max > v_min and optimism_power > 0.0:
                normalized = (vps - v_min) / (v_max - v_min)
                weights = (0.2 + 0.8 * normalized) ** optimism_power
                probs = weights / weights.sum()
                indices = np.random.choice(len(self.buffer), size=batch_size, replace=False, p=probs)
                batch = [self.buffer[i] for i in indices]
            else:
                batch = random.sample(self.buffer, batch_size)

        obs, action_mask, mcts_policy, value_target = zip(*batch)
        return (
            np.stack(obs),
            np.stack(action_mask),
            np.stack(mcts_policy),
            np.array(value_target, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)

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
        self.replay_buffer = AlphaZeroReplayBuffer(self.az_config.replay_buffer_size)
        
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

    def self_play_game(self, env: Any) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]], float, bool, int]:
        env.reset()
        history = []
        move_count = 0
        total_game_steps = 0
        consecutive_step_errors = 0
        MAX_TOTAL_STEPS = 400
        
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

            if current_player == 0:
                temp = self.az_config.temperature_high if move_count < self.az_config.temperature_threshold_move else self.az_config.temperature_low
                
                action, probs, _ = self.mcts.search(
                    env, 
                    num_simulations=self.az_config.num_simulations, 
                    temperature=temp,
                    add_noise=True
                )
                
                history.append((obs, mask, probs, 0.0))
                move_count += 1
            else:
                obs_t = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
                mask_t = torch.from_numpy(mask).bool().to(self.device).unsqueeze(0)
                with torch.no_grad():
                    # In AlphaZero self-play, opponents play using the current policy network
                    logits = self.agent.action_net(obs_t, mask_t)
                    probs_t = F.softmax(logits, dim=-1)
                    probs = probs_t.squeeze(0).cpu().numpy()
                
                p_legal = probs[legal_indices]
                p_sum = p_legal.sum()
                if p_sum > 0:
                    p_legal = p_legal / p_sum
                    action = np.random.choice(legal_indices, p=p_legal)
                else:
                    action = int(np.random.choice(legal_indices))
                    
            step_res = env.step(action)
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
            
        raw_vps = [float(p.get("vp", 0.0)) for p in getattr(env, "players_state", [{"vp": 0.0}] * 4)]
        p0_vp = raw_vps[0] if len(raw_vps) > 0 else 50.0
        p0_won = bool(len(raw_vps) >= 2 and p0_vp > max(raw_vps[1:]))
        
        # Store true Victory Points (0 - 250+ VP) for natural calibration with MCTS and GUI
        final_history = [(obs, mask, probs, p0_vp) for obs, mask, probs, _ in history]
        return final_history, p0_vp, p0_won, move_count

    def train_on_batch(self, batch: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[float, float, float, float]:
        obs_b, mask_b, policy_b, value_b = batch
        
        obs_t = torch.from_numpy(obs_b).float().to(self.device)
        mask_t = torch.from_numpy(mask_b).bool().to(self.device)
        policy_t = torch.from_numpy(policy_b).float().to(self.device)
        value_t = torch.from_numpy(value_b).float().to(self.device)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        device_type = "cuda" if (self.use_amp and self.device.type == "cuda") else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=(self.use_amp and self.device.type == "cuda")):
            pred_values = self.agent.score_net(obs_t).view(-1)
            logits = self.agent.action_net(obs_t, mask_t)
            
            # Huber Smooth L1 loss on VP prevents gradient explosion
            value_loss = F.smooth_l1_loss(pred_values, value_t.view(-1))
            
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
            value_loss = torch.nan_to_num(value_loss, nan=0.0, posinf=100.0, neginf=-100.0)
            
            # Normalize VP scale (variance ~25 VP) so value gradients do not overwhelm policy gradients
            norm_value_loss = value_loss / 25.0
            loss = policy_loss + self.az_config.value_loss_coef * norm_value_loss
            
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
                )
                
                if callback:
                    callback(metrics)
                    
            if epoch % self.az_config.checkpoint_interval == 0:
                self.save_checkpoint(os.path.join(self.config.training.checkpoint_dir, f"az_checkpoint_{epoch}.pt"))
                
        self._is_running = False

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            'epoch': self.current_epoch,
            'agent_state_dict': self.agent.state_dict(),
            'score_net_state': self.agent.score_net.state_dict(),
            'action_net_state': self.agent.action_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if hasattr(self, 'scaler') and self.scaler else None,
            'config': self.config,
            'meta': {'epoch': self.current_epoch, 'algorithm': 'AlphaZero'}
        }, path)

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
