import os
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from buffer import BatchData, RolloutBuffer
from config import AppConfig, AsyncConfig
from environment import make_gaia_env
from models import DualGaiaAgent
from trainer import RLTrainer, TrainingMetrics


def actor_worker_fn(
    worker_id: int,
    app_cfg: AppConfig,
    shared_state_dict: Dict[str, torch.Tensor],
    rollout_queue: mp.Queue,
    stop_event: mp.Event,
    weights_version: mp.Value,
):
    """Background Actor process continuously playing matches and pushing rollout chunks."""
    import os
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    np.random.seed(worker_id * 1000 + int(time.time()) % 10000)
    torch.manual_seed(worker_id * 1000 + int(time.time()) % 10000)

    env = make_gaia_env(players=app_cfg.model.num_players, seed=worker_id * 100 + 42)
    local_agent = DualGaiaAgent(app_cfg.model).to(torch.device("cpu"))
    local_agent.eval()

    # Load initial weights
    local_agent.load_state_dict({k: v.clone() for k, v in shared_state_dict.items()})
    local_version = int(weights_version.value)

    obs, mask = env.reset()
    target_chunk = 64  # Small chunks for smooth pipelined queue feeding

    while not stop_event.is_set():
        # Check for weight updates
        if weights_version.value > local_version:
            local_agent.load_state_dict({k: v.clone() for k, v in shared_state_dict.items()})
            local_version = int(weights_version.value)

        chunk_obs = []
        chunk_actions = []
        chunk_rewards = []
        chunk_dones = []
        chunk_values = []
        chunk_log_probs = []
        chunk_masks = []
        chunk_opp_actions = []
        chunk_has_opp = []

        last_p0_chunk_idx = -1

        for _ in range(target_chunk):
            actor = getattr(env, "current_player", 0)
            if actor == 0:
                obs_t = torch.from_numpy(obs).float()
                mask_t = torch.from_numpy(mask).bool()
                act, logp, val, _ = local_agent.act_and_evaluate(obs_t, mask_t, deterministic=False)

                step_res = env.step(act)
                rew = float(step_res.reward)

                last_p0_chunk_idx = len(chunk_obs)
                chunk_obs.append(obs.copy())
                chunk_actions.append(act)
                chunk_rewards.append(rew)
                chunk_dones.append(step_res.done)
                chunk_values.append(val)
                chunk_log_probs.append(logp)
                chunk_masks.append(mask.copy())
                chunk_opp_actions.append(0)
                chunk_has_opp.append(False)
            else:
                # Opponent action: sample using local agent as baseline or random
                legal_acts = np.where(mask)[0]
                opp_act = int(np.random.choice(legal_acts)) if len(legal_acts) > 0 else 0
                step_res = env.step(opp_act)

                if last_p0_chunk_idx >= 0 and last_p0_chunk_idx < len(chunk_opp_actions):
                    chunk_opp_actions[last_p0_chunk_idx] = opp_act
                    chunk_has_opp[last_p0_chunk_idx] = True
                    last_p0_chunk_idx = -1

            obs = step_res.obs
            mask = step_res.action_mask
            if step_res.done:
                obs, mask = env.reset()
                last_p0_chunk_idx = -1

        if len(chunk_obs) > 0:
            payload = {
                "observations": np.array(chunk_obs, dtype=np.float32),
                "actions": np.array(chunk_actions, dtype=np.int64),
                "rewards": np.array(chunk_rewards, dtype=np.float32),
                "dones": np.array(chunk_dones, dtype=bool),
                "values": np.array(chunk_values, dtype=np.float32),
                "log_probs": np.array(chunk_log_probs, dtype=np.float32),
                "action_masks": np.array(chunk_masks, dtype=bool),
                "opponent_actions": np.array(chunk_opp_actions, dtype=np.int64),
                "has_opponents": np.array(chunk_has_opp, dtype=bool),
                "version": local_version,
            }
            try:
                rollout_queue.put(payload, timeout=2.0)
            except Exception:
                pass


class AsyncRLTrainer:
    """Asynchronous Distributed Actor-Learner Trainer (APPO / IMPALA architecture)."""

    def __init__(self, config: Optional[AppConfig] = None):
        if mp.get_start_method(allow_none=True) != "spawn":
            try:
                mp.set_start_method("spawn", force=True)
            except (RuntimeError, ValueError):
                pass

        self.config = config or AppConfig()
        self.device = self.config.hardware.get_torch_device()
        self.trainer = RLTrainer(self.config)

        self.num_actors = getattr(self.config.async_dist, "num_actors", 2)
        self.queue_max_size = getattr(self.config.async_dist, "queue_max_size", 16)
        self.staleness_threshold = getattr(self.config.async_dist, "staleness_threshold", 3)

        self.shared_state_dict: Dict[str, torch.Tensor] = {}
        for k, v in self.trainer.agent.state_dict().items():
            t = v.cpu().clone()
            t.share_memory_()
            self.shared_state_dict[k] = t

        self.weights_version = mp.Value("i", 1)
        self.stop_event = mp.Event()
        self.rollout_queue = mp.Queue(maxsize=self.queue_max_size)
        self.workers: List[mp.Process] = []
        self._is_running = False

    def start_actors(self) -> None:
        """Spawns background actor processes."""
        if self._is_running:
            return
        self.stop_event.clear()
        self.workers = []
        for wid in range(self.num_actors):
            p = mp.Process(
                target=actor_worker_fn,
                args=(
                    wid,
                    self.config,
                    self.shared_state_dict,
                    self.rollout_queue,
                    self.stop_event,
                    self.weights_version,
                ),
                daemon=True,
            )
            p.start()
            self.workers.append(p)
        self._is_running = True

    def stop_actors(self) -> None:
        """Terminates all background actor processes gracefully."""
        self.stop_event.set()
        # Drain queue
        while not self.rollout_queue.empty():
            try:
                self.rollout_queue.get_nowait()
            except Exception:
                break
        for p in self.workers:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()
        self.workers = []
        self._is_running = False

    def sync_weights_to_shared(self) -> None:
        """Copies current central GPU weights to CPU shared tensors."""
        curr_state = self.trainer.agent.state_dict()
        with torch.no_grad():
            for k, v in curr_state.items():
                if k in self.shared_state_dict:
                    self.shared_state_dict[k].copy_(v.cpu())
        self.weights_version.value += 1

    def train_step_async(self, timeout: float = 10.0) -> Optional[TrainingMetrics]:
        """Consumes trajectories from the async queue, trains PPO + Opponent + RND on GPU."""
        if not self._is_running:
            self.start_actors()

        start_time = time.time()
        collected_steps = 0
        target_steps = self.config.training.rollout_steps_per_epoch
        self.trainer.buffer.clear()

        while collected_steps < target_steps and (time.time() - start_time) < timeout:
            try:
                payload = self.rollout_queue.get(timeout=1.0)
            except Exception:
                continue

            # Staleness guard (APPO): discard if too old
            if (self.weights_version.value - payload.get("version", 1)) > self.staleness_threshold:
                continue

            n = len(payload["actions"])
            for i in range(n):
                if self.trainer.buffer.size() >= self.trainer.buffer.capacity:
                    break
                self.trainer.buffer.add(
                    obs=payload["observations"][i],
                    action=int(payload["actions"][i]),
                    reward=float(payload["rewards"][i]),
                    done=bool(payload["dones"][i]),
                    value=float(payload["values"][i]),
                    log_prob=float(payload["log_probs"][i]),
                    action_mask=payload["action_masks"][i],
                    player_id=0,
                    opponent_action=int(payload["opponent_actions"][i]) if payload["has_opponents"][i] else None,
                )
                collected_steps += 1

        if self.trainer.buffer.size() == 0:
            return None

        # Compute GAE
        self.trainer.buffer.compute_returns_and_advantages(
            gamma=self.config.training.gamma,
            gae_lambda=self.config.training.gae_lambda,
        )

        # Train on GPU
        p_loss, v_loss, entropy, total_loss = self.trainer.train_epoch()
        self.trainer.current_epoch += 1

        # Synchronize new weights to actors
        self.sync_weights_to_shared()

        elapsed = max(1e-4, time.time() - start_time)
        steps_per_sec = float(collected_steps) / elapsed

        vram_mb = 0.0
        if self.device.type == "cuda":
            vram_mb = torch.cuda.memory_allocated(self.device) / (1024**2)

        curr_elo = (
            self.trainer.league_manager.members["CurrentPolicy"].elo
            if hasattr(self.trainer, "league_manager") and "CurrentPolicy" in self.trainer.league_manager.members
            else 1200.0
        )

        return TrainingMetrics(
            epoch=self.trainer.current_epoch,
            episodes=collected_steps // 20,
            policy_loss=p_loss,
            value_loss=v_loss,
            total_loss=total_loss,
            entropy=entropy,
            avg_predicted_score=75.0,
            avg_real_score=75.0,
            win_rate=self.trainer._last_win_rate,
            steps_per_sec=steps_per_sec,
            vram_allocated_mb=vram_mb,
            device_name=str(self.device),
            league_elo=curr_elo,
        )

    @property
    def hw_info(self) -> Dict[str, Any]:
        return self.trainer.hw_info

    @property
    def current_epoch(self) -> int:
        return self.trainer.current_epoch

    @property
    def agent(self):
        return self.trainer.agent

    @property
    def league_manager(self):
        return self.trainer.league_manager

    @property
    def state_buffer(self):
        return getattr(self.trainer, "state_buffer", None)

    @property
    def buffer(self):
        return self.trainer.buffer

    @property
    def rnd(self):
        return getattr(self.trainer, "rnd", None)

    @property
    def best_elo(self) -> float:
        return getattr(self.trainer, "best_elo", 1200.0)

    def resume_from_checkpoint(self, path: str) -> int:
        res = self.trainer.resume_from_checkpoint(path)
        self.sync_weights_to_shared()
        return res

    def save_current_checkpoint(self, is_milestone: bool = False) -> str:
        return self.trainer.save_current_checkpoint(is_milestone=is_milestone)

    def is_running(self) -> bool:
        return getattr(self, "_bg_running", False) or self._is_running

    def is_paused(self) -> bool:
        return self.trainer.is_paused()

    def pause(self) -> None:
        self.trainer.pause()

    def resume(self) -> None:
        self.trainer.resume()

    def start_background_training(
        self,
        on_metrics: Optional[Callable[[TrainingMetrics], None]] = None,
        on_finished: Optional[Callable[[], None]] = None,
        max_epochs: int = 500,
    ) -> None:
        """Spawns background training worker thread for AsyncRLTrainer."""
        if getattr(self, "_bg_running", False):
            return

        self._bg_running = True
        self.start_actors()
        self.trainer._stop_event.clear()
        self.trainer._pause_event.clear()

        def _worker():
            try:
                while (
                    not self.trainer._stop_event.is_set()
                    and self.current_epoch < max_epochs
                ):
                    if self.trainer._pause_event.is_set():
                        time.sleep(0.1)
                        continue

                    metrics = self.train_step()
                    if on_metrics is not None and metrics is not None and metrics.steps_per_sec > 0:
                        on_metrics(metrics)

                    if (
                        self.current_epoch
                        % self.config.training.save_checkpoint_interval
                        == 0
                    ):
                        self.save_current_checkpoint(is_milestone=True)

            except Exception as e:
                print(f"[AsyncTrainer Error]: {e}")
            finally:
                self.stop_actors()
                self._bg_running = False
                if on_finished is not None:
                    on_finished()

        self._bg_thread = threading.Thread(target=_worker, daemon=True)
        self._bg_thread.start()

    def stop(self) -> None:
        self.stop_actors()
        self.trainer.stop()
        self._bg_running = False

    def train_step(self, env: Any = None) -> TrainingMetrics:
        """Drop-in interface matching RLTrainer.train_step for seamless integration."""
        for _ in range(10):
            m = self.train_step_async(timeout=3.0)
            if m is not None:
                return m
            time.sleep(0.05)
        return TrainingMetrics(
            epoch=self.trainer.current_epoch,
            episodes=0,
            policy_loss=0.0,
            value_loss=0.0,
            total_loss=0.0,
            entropy=0.0,
            avg_predicted_score=0.0,
            avg_real_score=0.0,
            win_rate=0.5,
            steps_per_sec=0.0,
            vram_allocated_mb=0.0,
            device_name=str(self.device),
        )


