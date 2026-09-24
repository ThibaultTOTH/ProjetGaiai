"""Population-Based League Training Module for Gaia Project Deep RL.

Maintains a pool of historical model snapshots, current policy clones,
and exploratory baseline agents. Coordinates opponent sampling and multi-player
Elo rating updates (Fictitious Self-Play / AlphaStar-style League).
"""

import copy
import glob
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from config import LeagueConfig, ModelConfig
from models import ActionOptimizerNet, DualGaiaAgent


@dataclass
class LeagueMember:
    """Represents a competitive policy within the league pool."""

    name: str
    policy_net: Optional[ActionOptimizerNet] = None
    elo: float = 1200.0
    games: int = 0
    wins: int = 0
    epoch: int = 0
    filepath: Optional[str] = None
    is_current: bool = False
    is_random: bool = False

    @property
    def win_rate(self) -> float:
        return float(self.wins) / float(self.games) if self.games > 0 else 0.0


class LeagueManager:
    """Orchestrates population-based league training and matchmaking."""

    def __init__(
        self,
        config: Optional[LeagueConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or LeagueConfig()
        self.device = device or torch.device("cpu")
        self.members: Dict[str, LeagueMember] = {}
        self.historical_names: List[str] = []

        # Register permanent virtual members
        self.members["CurrentPolicy"] = LeagueMember(
            name="CurrentPolicy",
            elo=1200.0,
            is_current=True,
        )
        self.members["RandomBaseline"] = LeagueMember(
            name="RandomBaseline",
            elo=900.0,
            is_random=True,
        )

        os.makedirs(self.config.league_dir, exist_ok=True)

    def add_snapshot(
        self,
        agent: DualGaiaAgent,
        epoch: int,
        name: Optional[str] = None,
    ) -> str:
        """Saves a frozen snapshot of the agent's policy into the league pool."""
        snapshot_name = name or f"Snapshot_Epoch_{epoch:04d}"
        if snapshot_name in self.members:
            return snapshot_name

        cloned_policy = agent.clone_policy_net().to(self.device)
        cloned_policy.eval()

        # Inherit current policy's Elo as starting baseline for new snapshot
        current_elo = self.members["CurrentPolicy"].elo

        filepath = os.path.join(self.config.league_dir, f"{snapshot_name}.pt")
        try:
            torch.save(
                {
                    "action_net_state": cloned_policy.state_dict(),
                    "epoch": epoch,
                    "name": snapshot_name,
                    "created_at": time.time(),
                },
                filepath,
            )
        except Exception as e:
            print(f"[LeagueManager] Warning: could not write snapshot to disk: {e}")
            filepath = None

        member = LeagueMember(
            name=snapshot_name,
            policy_net=cloned_policy,
            elo=current_elo,
            epoch=epoch,
            filepath=filepath,
        )

        self.members[snapshot_name] = member
        self.historical_names.append(snapshot_name)

        # Evict oldest snapshots if pool exceeds capacity (keep milestone epochs like 10, 50, 100)
        if len(self.historical_names) > self.config.max_snapshots:
            to_remove = None
            for old_name in self.historical_names:
                m = self.members[old_name]
                if m.epoch % 50 != 0:  # Preserve major milestones
                    to_remove = old_name
                    break
            if to_remove is None:
                to_remove = self.historical_names[0]

            self.historical_names.remove(to_remove)
            self.members.pop(to_remove, None)

        return snapshot_name

    def sample_opponents(
        self,
        current_agent: DualGaiaAgent,
        num_opponents: int = 3,
    ) -> List[Tuple[str, Callable[[np.ndarray, np.ndarray], int]]]:
        """Samples opponents for opponent seats based on the configured league distribution.

        Returns list of (member_name, action_fn).
        """
        opponents = []

        for _ in range(num_opponents):
            r = np.random.rand()
            has_history = len(self.historical_names) > 0

            # 1. Current Policy self-play
            if r < self.config.self_play_prob or not has_history:
                member_name = "CurrentPolicy"

                def _act_current(obs: np.ndarray, mask: np.ndarray) -> int:
                    obs_t = torch.from_numpy(obs).float().to(self.device)
                    mask_t = torch.from_numpy(mask).bool().to(self.device)
                    return current_agent.act(obs_t, mask_t, deterministic=False)

                opponents.append((member_name, _act_current))

            # 2. Historical Snapshot from the League Pool (Gaussian Elo Matchmaking or Uniform)
            elif r < self.config.self_play_prob + self.config.historical_prob:
                matchmaking = getattr(self.config, "matchmaking_type", "gaussian").lower()
                if matchmaking == "gaussian" and len(self.historical_names) > 1:
                    curr_elo = self.members.get(
                        "CurrentPolicy",
                        LeagueMember(name="CurrentPolicy", policy_net=None, elo=1200.0),
                    ).elo
                    elo_window = max(10.0, getattr(self.config, "matchmaking_elo_window", 150.0))
                    hist_elos = np.array([self.members[name].elo for name in self.historical_names], dtype=np.float64)
                    diffs = hist_elos - curr_elo
                    weights = np.exp(-0.5 * (diffs / elo_window) ** 2)
                    weight_sum = weights.sum()
                    if weight_sum > 1e-8:
                        probs = weights / weight_sum
                        hist_name = str(np.random.choice(self.historical_names, p=probs))
                    else:
                        hist_name = str(np.random.choice(self.historical_names))
                else:
                    hist_name = str(np.random.choice(self.historical_names))
                hist_member = self.members[hist_name]
                policy = hist_member.policy_net

                def _act_hist(obs: np.ndarray, mask: np.ndarray, net=policy) -> int:
                    if net is None:
                        legal = np.where(mask)[0]
                        return int(np.random.choice(legal)) if len(legal) > 0 else 15

                    obs_t = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
                    mask_t = torch.from_numpy(mask).bool().to(self.device).unsqueeze(0)
                    with torch.no_grad():
                        logits = net(obs_t, mask_t)
                        probs = F.softmax(logits, dim=-1)
                        dist = Categorical(probs=probs)
                        action = dist.sample()
                    return int(action.item())

                opponents.append((hist_name, _act_hist))

            # 3. Random / Exploratory Baseline
            else:
                member_name = "RandomBaseline"

                def _act_random(obs: np.ndarray, mask: np.ndarray) -> int:
                    legal = np.where(mask)[0]
                    return int(np.random.choice(legal)) if len(legal) > 0 else 15

                opponents.append((member_name, _act_random))

        return opponents

    def update_match_results(
        self,
        participants: List[str],
        vps: List[float],
        k_factor: float = 24.0,
    ) -> None:
        """Applies multi-player Bradley-Terry Elo rating updates to all participants."""
        n = len(participants)
        if n < 2 or len(vps) != n:
            return

        # Find match winner(s)
        max_vp = max(vps)
        for i, name in enumerate(participants):
            if name in self.members:
                m = self.members[name]
                m.games += 1
                if vps[i] == max_vp and vps.count(max_vp) == 1:
                    m.wins += 1

        # Pairwise Elo updates across all participant pairs
        elo_deltas = [0.0] * n
        for i in range(n):
            for j in range(i + 1, n):
                p_i = participants[i]
                p_j = participants[j]

                elo_i = self.members[p_i].elo if p_i in self.members else 1200.0
                elo_j = self.members[p_j].elo if p_j in self.members else 1200.0

                # Expected scores
                e_i = 1.0 / (1.0 + 10.0 ** ((elo_j - elo_i) / 400.0))
                e_j = 1.0 - e_i

                # Actual match scores
                if vps[i] > vps[j]:
                    s_i, s_j = 1.0, 0.0
                elif vps[j] > vps[i]:
                    s_i, s_j = 0.0, 1.0
                else:
                    s_i, s_j = 0.5, 0.5

                pair_k = k_factor / max(1.0, float(n - 1))
                elo_deltas[i] += pair_k * (s_i - e_i)
                elo_deltas[j] += pair_k * (s_j - e_j)

        for i, name in enumerate(participants):
            if name in self.members:
                self.members[name].elo = max(100.0, self.members[name].elo + elo_deltas[i])

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Returns sorted list of league members by Elo rating."""
        leaderboard = []
        for m in sorted(self.members.values(), key=lambda x: x.elo, reverse=True):
            leaderboard.append({
                "name": m.name,
                "elo": round(m.elo, 1),
                "games": m.games,
                "wins": m.wins,
                "win_rate": f"{m.win_rate * 100:.1f}%",
                "epoch": m.epoch,
                "is_current": m.is_current,
                "is_random": m.is_random,
            })
        return leaderboard

    def load_existing_checkpoints(
        self,
        obs_dim: int = 42,
        action_dim: int = 16,
    ) -> int:
        """Scans disk checkpoints and loads compatible models into the league pool."""
        pattern = os.path.join(self.config.league_dir, "*.pt")
        files = sorted(glob.glob(pattern))
        loaded = 0

        for fpath in files:
            try:
                try:
                    ckpt = torch.load(fpath, map_location=self.device, weights_only=False)
                except TypeError:
                    ckpt = torch.load(fpath, map_location=self.device)

                if "action_net_state" not in ckpt:
                    continue

                state_dict = ckpt["action_net_state"]
                # Verify dimension compatibility
                first_weight = state_dict.get("input_proj.0.weight")
                if first_weight is not None and first_weight.shape[1] != obs_dim:
                    continue

                policy = ActionOptimizerNet(obs_dim=obs_dim, action_dim=action_dim).to(self.device)
                policy.load_state_dict(state_dict, strict=False)
                policy.eval()

                name = ckpt.get("name", os.path.splitext(os.path.basename(fpath))[0])
                epoch = int(ckpt.get("epoch", 0))

                member = LeagueMember(
                    name=name,
                    policy_net=policy,
                    elo=1200.0,
                    epoch=epoch,
                    filepath=fpath,
                )
                self.members[name] = member
                if name not in self.historical_names:
                    self.historical_names.append(name)
                loaded += 1
            except Exception as e:
                print(f"[LeagueManager] Skipped loading {fpath}: {e}")

        return loaded
