import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import numpy as np
import random
import threading
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from config import MuZeroConfig, MCTSConfig, AppConfig
from models import DualStreamBackbone, PreLNResBlock, build_block

class MuZeroNetwork(nn.Module):
    """MuZero: Representation + Dynamics + Prediction networks."""
    
    def __init__(self, obs_dim=2476, action_dim=3130, hidden_dim=256, num_blocks=4, backbone: Optional[nn.Module] = None):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Representation function h()
        if backbone is not None:
            self.representation = backbone
        else:
            self.representation = DualStreamBackbone(obs_dim=obs_dim, map_out_dim=hidden_dim, trunk_layers=[512, 512, hidden_dim])
            
        # Dynamics function g()
        self.action_embedding = nn.Embedding(action_dim, 128)
        
        dynamics_blocks = []
        dynamics_dim = hidden_dim + 128
        
        # Initial projection to hidden_dim
        dynamics_blocks.append(nn.Linear(dynamics_dim, hidden_dim))
        dynamics_blocks.append(nn.SiLU())
        
        for _ in range(num_blocks):
            dynamics_blocks.append(PreLNResBlock(hidden_dim, dropout=0.0))
            
        self.dynamics_trunk = nn.Sequential(*dynamics_blocks)
        
        # Norm after dynamics to prevent drift
        self.dynamics_norm = nn.LayerNorm(hidden_dim)
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )
        
        # Prediction function f()
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.SiLU(),
            nn.Linear(512, action_dim)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )
        
    def initial_inference(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h(obs) -> hidden, then f(hidden) -> (policy, value)"""
        hidden_state = self.representation(obs)
        policy_logits = self.policy_head(hidden_state)
        value = self.value_head(hidden_state)
        return hidden_state, policy_logits, value
        
    def recurrent_inference(self, hidden_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """g(hidden, action) -> (next_hidden, reward), then f(next_hidden) -> (policy, value)"""
        action_emb = self.action_embedding(action)
        
        # Concat along feature dim
        x = torch.cat([hidden_state, action_emb], dim=-1)
        
        next_hidden = self.dynamics_trunk(x)
        next_hidden = self.dynamics_norm(next_hidden)
        
        reward = self.reward_head(next_hidden)
        policy_logits = self.policy_head(next_hidden)
        value = self.value_head(next_hidden)
        
        return next_hidden, reward, policy_logits, value


class Node:
    def __init__(self, prior: float, hidden_state: Optional[torch.Tensor] = None, reward: float = 0.0):
        self.prior = prior
        self.hidden_state = hidden_state
        self.reward = reward
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}
        
    def expanded(self) -> bool:
        return len(self.children) > 0
        
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MuZeroMCTS:
    """MCTS that operates entirely in latent space using learned dynamics."""
    
    def __init__(self, network: MuZeroNetwork, config: MCTSConfig, device: torch.device):
        self.network = network
        self.config = config
        self.device = device
        self.discount = 0.99
        
    @torch.no_grad()
    def search(self, obs: Any, action_mask: Any, num_simulations: int = 50, temperature: float = 1.0, add_noise: bool = False) -> Tuple[int, np.ndarray, dict]:
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        elif not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            
        if isinstance(action_mask, np.ndarray):
            action_mask = torch.from_numpy(action_mask).bool().to(self.device)
        elif not isinstance(action_mask, torch.Tensor):
            action_mask = torch.tensor(action_mask, dtype=torch.bool, device=self.device)
            
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            
        hidden_state, policy_logits, value = self.network.initial_inference(obs)
        
        policy_logits = policy_logits.squeeze(0)
        
        # Mask illegal actions
        policy_logits[~action_mask] = -float('inf')
        
        policy_probs = F.softmax(policy_logits, dim=-1)
        
        if add_noise:
            legal_actions = torch.where(action_mask)[0]
            num_legal = len(legal_actions)
            if num_legal > 0:
                noise = torch.distributions.Dirichlet(torch.full((num_legal,), self.config.dirichlet_alpha)).sample().to(self.device)
                eps = self.config.dirichlet_eps
                policy_probs[legal_actions] = (1 - eps) * policy_probs[legal_actions] + eps * noise
                
        root = Node(prior=1.0, hidden_state=hidden_state.squeeze(0))
        
        legal_actions = torch.where(action_mask)[0].cpu().numpy()
        for a in legal_actions:
            root.children[a] = Node(prior=policy_probs[a].item(), hidden_state=None)
            
        for _ in range(num_simulations):
            node = root
            search_path = [node]
            actions_history = []
            
            # Select
            while node.expanded():
                best_score = -float('inf')
                best_action = -1
                best_child = None
                
                for a, child in node.children.items():
                    q = child.value()
                    u = self.config.c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
                    score = q + u
                    if score > best_score:
                        best_score = score
                        best_action = a
                        best_child = child
                        
                actions_history.append(best_action)
                node = best_child
                search_path.append(node)
                
            parent = search_path[-2]
            action = actions_history[-1]
            
            action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)
            parent_hidden_tensor = parent.hidden_state.unsqueeze(0)
            
            next_hidden, reward, policy_logits, value = self.network.recurrent_inference(parent_hidden_tensor, action_tensor)
            
            node.hidden_state = next_hidden.squeeze(0)
            node.reward = reward.item()
            
            policy_logits = policy_logits.squeeze(0)
            policy_probs = F.softmax(policy_logits, dim=-1)
            
            # Use top 50 actions for simplicity in latent space
            topk_probs, topk_actions = torch.topk(policy_probs, min(50, len(policy_probs)))
            for prob, a in zip(topk_probs, topk_actions):
                node.children[a.item()] = Node(prior=prob.item(), hidden_state=None)
                
            value_eval = value.item()
            
            # Backpropagate
            for i, p in enumerate(reversed(search_path)):
                p.visit_count += 1
                
                if i == 0:
                    p.value_sum += value_eval
                else:
                    v = search_path[-i].reward + self.discount * value_eval
                    p.value_sum += v
                    value_eval = v

        counts = torch.zeros(self.network.action_dim, device=self.device)
        for a, child in root.children.items():
            counts[a] = child.visit_count
            
        if temperature == 0:
            best_action = torch.argmax(counts).item()
            action_probs = torch.zeros_like(counts)
            action_probs[best_action] = 1.0
        else:
            counts_t = counts ** (1.0 / temperature)
            if counts_t.sum() > 0:
                action_probs = counts_t / counts_t.sum()
                best_action = torch.multinomial(action_probs, 1).item()
            else:
                best_action = random.choice(legal_actions)
                action_probs = torch.zeros_like(counts)
                action_probs[best_action] = 1.0
            
        meta = {
            "root_value": root.value(),
            "visit_counts": counts.cpu().numpy()
        }
            
        return best_action, action_probs.cpu().numpy() if isinstance(action_probs, torch.Tensor) else action_probs, meta


class MuZeroReplayBuffer:
    """Stores game trajectories for MuZero training with K-step unrolling."""
    
    def __init__(self, capacity=50_000, unroll_steps=5):
        self.capacity = capacity
        self.unroll_steps = unroll_steps
        self.buffer = []
        self.position = 0
        
    def add_game(self, game_history: List[dict]):
        if len(self.buffer) < self.capacity:
            self.buffer.append(game_history)
        else:
            self.buffer[self.position] = game_history
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_obs = []
        batch_actions = []
        batch_target_values = []
        batch_target_policies = []
        batch_target_rewards = []
        batch_action_masks = []
        
        for _ in range(batch_size):
            game = random.choice(self.buffer)
            # Choose a random start position
            start_pos = random.randint(0, len(game) - 1)
            
            # Unroll K steps
            actions = []
            values = []
            policies = []
            rewards = []
            
            batch_obs.append(game[start_pos]["obs"])
            batch_action_masks.append(game[start_pos]["action_mask"])
            
            for i in range(self.unroll_steps):
                pos = start_pos + i
                if pos < len(game):
                    actions.append(game[pos]["action"])
                    values.append(game[pos]["mcts_value"])
                    policies.append(game[pos]["mcts_policy"])
                    rewards.append(game[pos]["reward"])
                else:
                    # Pad if end of game reached
                    actions.append(0) 
                    values.append(0.0)
                    pol0 = game[0]["mcts_policy"]
                    p_pad = np.zeros_like(pol0) if isinstance(pol0, np.ndarray) else torch.zeros_like(pol0).cpu().numpy()
                    policies.append(p_pad)
                    rewards.append(0.0)
                    
            batch_actions.append(actions)
            batch_target_values.append(values)
            batch_target_policies.append(policies)
            batch_target_rewards.append(rewards)
            
        return (
            torch.as_tensor(np.array(batch_obs), dtype=torch.float32),
            torch.as_tensor(np.array(batch_actions), dtype=torch.long),
            torch.as_tensor(np.array(batch_target_values), dtype=torch.float32),
            torch.as_tensor(np.array(batch_target_policies), dtype=torch.float32),
            torch.as_tensor(np.array(batch_target_rewards), dtype=torch.float32),
            torch.as_tensor(np.array(batch_action_masks), dtype=torch.bool)
        )


class MuZeroTrainer:
    def __init__(self, config: AppConfig, network: Optional[MuZeroNetwork] = None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if network is None:
            self.network = MuZeroNetwork(
                hidden_dim=config.muzero.hidden_dim, 
                num_blocks=config.muzero.num_dynamics_blocks
            ).to(self.device)
        else:
            self.network = network.to(self.device)
            
        self.mcts = MuZeroMCTS(self.network, config.mcts, self.device)
        self.replay_buffer = MuZeroReplayBuffer(
            capacity=config.muzero.replay_buffer_size,
            unroll_steps=config.muzero.unroll_steps
        )
        
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), 
            lr=config.muzero.lr, 
            weight_decay=config.muzero.weight_decay
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.config.hardware.use_mixed_precision and self.device.type == "cuda"))
        
        self._stop_event = threading.Event()
        self._is_running = False
        
    def is_running(self):
        return self._is_running
        
    def stop(self):
        self._stop_event.set()
        
    def self_play_game(self, env) -> List[dict]:
        obs = env.reset()
        game_history = []
        done = False
        step = 0
        
        while not done and not self._stop_event.is_set():
            temp = self.config.alphazero.temperature_high if step < self.config.muzero.temperature_threshold_move else self.config.alphazero.temperature_low
            
            action_mask_np = env.get_action_mask()
            action_mask = torch.tensor(action_mask_np, dtype=torch.bool, device=self.device)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            
            action, policy_probs, meta = self.mcts.search(
                obs_tensor, 
                action_mask, 
                num_simulations=self.config.muzero.num_simulations,
                temperature=temp,
                add_noise=True
            )
            
            step_res = env.step(action)
            if hasattr(step_res, "obs"):
                next_obs = step_res.obs
                reward = float(step_res.reward)
                done = bool(step_res.done)
                info = getattr(step_res, "info", {})
            else:
                next_obs, reward, done, info = step_res[:4]
            
            m_pol = policy_probs.cpu().numpy() if isinstance(policy_probs, torch.Tensor) else policy_probs
            game_history.append({
                "obs": obs,
                "action": action,
                "reward": reward,
                "mcts_policy": m_pol,
                "mcts_value": meta.get("root_value", 0.0),
                "action_mask": action_mask_np
            })
            
            obs = next_obs
            done = done or getattr(env, "terminated", False)
            step += 1
            
        return game_history
        
    def train_on_batch(self, batch) -> Dict[str, float]:
        obs, actions, target_values, target_policies, target_rewards, action_masks = [b.to(self.device) for b in batch]
        
        batch_size = obs.size(0)
        unroll_steps = actions.size(1)
        
        self.optimizer.zero_grad()
        
        total_loss = 0
        policy_loss = 0
        value_loss = 0
        reward_loss = 0
        
        device_type = "cuda" if (self.config.hardware.use_mixed_precision and self.device.type == "cuda") else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=(self.config.hardware.use_mixed_precision and self.device.type == "cuda")):
            # Step 0
            hidden_state, policy_logits, value = self.network.initial_inference(obs)
            
            # Loss for step 0 (no reward)
            target_policy = target_policies[:, 0, :]
            target_value = target_values[:, 0].unsqueeze(-1)
            
            p_loss_0 = F.cross_entropy(policy_logits, target_policy)
            v_loss_0 = F.mse_loss(value, target_value)
            
            loss = p_loss_0 + self.config.muzero.value_loss_coef * v_loss_0
            total_loss += loss
            
            policy_loss += p_loss_0.item()
            value_loss += v_loss_0.item()
            
            # Unroll
            for k in range(unroll_steps):
                action_k = actions[:, k]
                
                next_hidden, reward, next_policy_logits, next_value = self.network.recurrent_inference(hidden_state, action_k)
                
                # Scale gradient by 0.5 according to MuZero paper
                next_hidden.register_hook(lambda grad: grad * 0.5)
                
                target_policy_k = target_policies[:, k, :]
                target_value_k = target_values[:, k].unsqueeze(-1)
                target_reward_k = target_rewards[:, k].unsqueeze(-1)
                
                p_loss_k = F.cross_entropy(next_policy_logits, target_policy_k)
                v_loss_k = F.mse_loss(next_value, target_value_k)
                r_loss_k = F.mse_loss(reward, target_reward_k)
                
                loss_k = p_loss_k + self.config.muzero.value_loss_coef * v_loss_k + self.config.muzero.reward_loss_coef * r_loss_k
                
                # Scale by 1/K
                total_loss += loss_k / unroll_steps
                
                policy_loss += p_loss_k.item()
                value_loss += v_loss_k.item()
                reward_loss += r_loss_k.item()
                
                hidden_state = next_hidden
                
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss / (unroll_steps + 1),
            "value_loss": value_loss / (unroll_steps + 1),
            "reward_loss": reward_loss / unroll_steps
        }
        
    def run_training_loop(self, env, max_epochs: int, callback=None):
        self._is_running = True
        self._stop_event.clear()
        
        for epoch in range(max_epochs):
            if self._stop_event.is_set():
                break
                
            # Self-play phase
            self.network.eval()
            for _ in range(self.config.muzero.games_per_epoch):
                if self._stop_event.is_set():
                    break
                game = self.self_play_game(env)
                self.replay_buffer.add_game(game)
                
            if len(self.replay_buffer.buffer) < self.config.muzero.batch_size:
                continue
                
            # Training phase
            self.network.train()
            metrics = {"total_loss": 0, "policy_loss": 0, "value_loss": 0, "reward_loss": 0}
            
            for _ in range(self.config.muzero.training_steps_per_epoch):
                if self._stop_event.is_set():
                    break
                batch = self.replay_buffer.sample(self.config.muzero.batch_size)
                step_metrics = self.train_on_batch(batch)
                
                for k, v in step_metrics.items():
                    metrics[k] += v
                    
            # Average metrics
            for k in metrics:
                metrics[k] /= self.config.muzero.training_steps_per_epoch
                
            if callback:
                callback(epoch, metrics)
                
        self._is_running = False

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            'epoch': getattr(self, 'current_epoch', 0),
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'meta': {'algorithm': 'MuZero'}
        }, path)

    def save_current_checkpoint(self, path: Optional[str] = None, is_milestone: bool = False, **kwargs) -> str:
        ckpt_dir = self.config.training.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        if path is None:
            path = os.path.join(ckpt_dir, f"muzero_checkpoint_{getattr(self, 'current_epoch', 0)}.pt")
            if not is_milestone:
                latest_path = os.path.join(ckpt_dir, "gaia_latest.pt")
                self.save_checkpoint(latest_path)
        self.save_checkpoint(path)
        return path

    def resume_from_checkpoint(self, path: str) -> int:
        if not os.path.exists(path):
            return 0
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict) and 'network_state_dict' in ckpt:
            self.network.load_state_dict(ckpt['network_state_dict'])
        elif isinstance(ckpt, dict):
            try:
                self.network.load_state_dict(ckpt)
            except Exception:
                pass
        epoch = int(ckpt.get('epoch', 0)) if isinstance(ckpt, dict) else 0
        self.current_epoch = epoch
        return epoch
