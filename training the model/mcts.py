"""Multi-Player Monte-Carlo Tree Search (MCTS) Engine for Gaia Project.

Implements:
- Vectorized PUCT / Max^n search for 4-player non-zero-sum Eurogame dynamics.
- Zero-overhead state cloning using native Rust gaiapi C-ABI or FastGaiaSimEnv.
- Prior policy guidance via ActionOptimizerNet with legal action masking.
- Leaf value estimation via ScorePredictorNet with relative table margin normalization.
- Dirichlet noise injection for exploratory self-play rollouts.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from config import MCTSConfig
from models import DualGaiaAgent


def compute_milestone_bonus(action: int) -> float:
    """Calculates tactical milestone bonus for productive strategic actions in Gaia Project."""
    if 0 <= action < 200:         # BuildMine
        return 0.3
    elif 400 <= action < 600:       # UpgradeTradingStation
        return 0.4
    elif 600 <= action < 800:     # UpgradeResearchLab
        return 0.5
    elif 800 <= action < 1000:    # UpgradePlanetaryInstitute
        return 0.6
    elif 1000 <= action < 1200:   # UpgradeAcademy
        return 0.6
    elif 1400 <= action < 1406:   # FormFederation
        return 1.2
    elif 1406 <= action < 1412:   # AdvanceResearch
        return 0.5
    elif 1444 <= action < 1498:   # ClaimTechTile
        return 0.4
    elif 1498 <= action < 2308:   # ClaimAdvTechTile
        return 0.7
    elif 2308 <= action < 3108:   # ExploreSpaceship
        return 0.5
    elif 200 <= action < 400:     # StartGaiaProject
        return 0.3
    return 0.0


class MCTSNode:
    """A high-performance node in the Multi-Player MCTS search tree with __slots__."""

    __slots__ = (
        "player",
        "action_mask",
        "prior",
        "parent",
        "action_from_parent",
        "visit_count",
        "v0",
        "v1",
        "v2",
        "v3",
        "total_uncertainty",
        "children",
        "is_expanded",
    )

    def __init__(
        self,
        player: int,
        action_mask: Optional[np.ndarray],
        prior: float = 1.0,
        parent: Optional["MCTSNode"] = None,
        action_from_parent: Optional[int] = None,
    ):
        self.player = player  # Active player making a move at this node
        self.action_mask = action_mask
        self.prior = float(prior)
        self.parent = parent
        self.action_from_parent = action_from_parent

        self.visit_count: int = 0
        self.v0: float = 0.0
        self.v1: float = 0.0
        self.v2: float = 0.0
        self.v3: float = 0.0
        self.total_uncertainty: float = 0.0
        self.children: Dict[int, "MCTSNode"] = {}
        self.is_expanded: bool = False

    def add_value(self, values: Any):
        self.v0 += float(values[0])
        self.v1 += float(values[1])
        self.v2 += float(values[2])
        self.v3 += float(values[3])

    def apply_virtual_loss(self, player: int, loss: float):
        if player == 0:
            self.v0 -= loss
        elif player == 1:
            self.v1 -= loss
        elif player == 2:
            self.v2 -= loss
        else:
            self.v3 -= loss

    def revert_virtual_loss(self, player: int, loss: float):
        if player == 0:
            self.v0 += loss
        elif player == 1:
            self.v1 += loss
        elif player == 2:
            self.v2 += loss
        else:
            self.v3 += loss

    def get_q(self, p: int) -> float:
        if self.visit_count == 0:
            return 0.0
        inv = 1.0 / self.visit_count
        if p == 0:
            return self.v0 * inv
        if p == 1:
            return self.v1 * inv
        if p == 2:
            return self.v2 * inv
        return self.v3 * inv

    @property
    def total_value(self) -> np.ndarray:
        return np.array([self.v0, self.v1, self.v2, self.v3], dtype=np.float32)

    @total_value.setter
    def total_value(self, arr: Any):
        self.v0 = float(arr[0])
        self.v1 = float(arr[1])
        self.v2 = float(arr[2])
        self.v3 = float(arr[3])

    @property
    def q_values(self) -> np.ndarray:
        """Returns mean value vector Q for all 4 players."""
        if self.visit_count == 0:
            return np.zeros(4, dtype=np.float32)
        inv = 1.0 / float(self.visit_count)
        return np.array([self.v0 * inv, self.v1 * inv, self.v2 * inv, self.v3 * inv], dtype=np.float32)

    def best_child(self, c_puct: float = 1.414, uncertainty_scale: float = 0.0) -> Tuple[int, "MCTSNode"]:
        """Selects child maximizing PUCT score with zero-allocation fast arithmetic."""
        best_action = -1
        best_score = -float("inf")
        sqrt_total = math.sqrt(max(1, self.visit_count))
        p = self.player

        for action, child in self.children.items():
            cnt = child.visit_count
            if cnt > 0:
                q = child.get_q(p)
                if uncertainty_scale > 0.0:
                    unc = child.total_uncertainty / cnt
                    c_eff = c_puct * (1.0 + uncertainty_scale * min(2.0, unc))
                else:
                    c_eff = c_puct
                u = c_eff * child.prior * (sqrt_total / (1.0 + cnt))
                score = q + u
            else:
                score = c_puct * child.prior * sqrt_total

            if score > best_score:
                best_score = score
                best_action = action

        if best_action == -1 and self.children:
            best_action = next(iter(self.children.keys()))

        return best_action, self.children[best_action]


class MultiPlayerMCTS:
    """Orchestrates AlphaZero / Max^n MCTS search on top of Gaia Project environments."""

    def __init__(
        self,
        agent: DualGaiaAgent,
        config: Optional[MCTSConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.agent = agent
        self.config = config if config is not None else MCTSConfig()
        if device is not None:
            self.device = device
        elif hasattr(agent, "parameters"):
            try:
                self.device = next(agent.parameters()).device
            except (StopIteration, Exception):
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

    def search(
        self,
        env: Any,
        num_simulations: Optional[int] = None,
        temperature: Optional[float] = None,
        add_noise: bool = False,
        root_obs: Optional[np.ndarray] = None,
        root_mask: Optional[np.ndarray] = None,
    ) -> Tuple[int, np.ndarray, Dict[str, Any]]:
        """Performs MCTS search using configured algorithm (Gumbel GAZ or PUCT)."""
        algo = getattr(self.config, "algorithm", "gumbel").lower()
        sims = num_simulations or self.config.num_simulations
        temp = temperature if temperature is not None else self.config.temperature

        if algo == "gumbel":
            return self._search_gumbel(env, num_simulations=sims, temperature=temp, add_noise=add_noise, root_obs=root_obs, root_mask=root_mask)
        return self._search_puct(env, num_simulations=sims, temperature=temp, add_noise=add_noise, root_obs=root_obs, root_mask=root_mask)

    def _search_gumbel(
        self,
        env: Any,
        num_simulations: int,
        temperature: float,
        add_noise: bool = False,
        root_obs: Optional[np.ndarray] = None,
        root_mask: Optional[np.ndarray] = None,
    ) -> Tuple[int, np.ndarray, Dict[str, Any]]:
        """Gumbel AlphaZero (Two-Stage Sequential Halving / TSS GAZ 2026).

        Guarantees policy improvement with minimal budget (8-16 simulations) by:
        1. Perturbing root logits with Gumbel noise.
        2. Selecting top-k candidate actions.
        3. Progressively allocating search budget through Sequential Halving.
        """
        root_actor = getattr(env, "current_player", 0)
        root_mask = root_mask if root_mask is not None else env.get_action_mask()
        action_dim = len(root_mask)
        legal_indices = np.where(root_mask)[0]

        if len(legal_indices) <= 1:
            act = int(legal_indices[0]) if len(legal_indices) == 1 else 0
            probs = np.zeros(action_dim, dtype=np.float32)
            if len(legal_indices) == 1:
                probs[act] = 1.0
            return act, probs, {
                "algorithm": "gumbel_gaz",
                "root_visits": 0,
                "selected_action": act,
                "root_entropy": 0.0,
                "entropy_gated": True,
            }

        # 1. Root policy evaluation
        if hasattr(self.agent, "evaluate_root"):
            root_obs_raw = root_obs if root_obs is not None else (env._get_obs() if hasattr(env, "_get_obs") else env.observe().values)
            priors, raw_logits = self.agent.evaluate_root(root_obs_raw, root_mask, root_actor)
        else:
            dev = self.device if self.device is not None else torch.device("cpu")
            root_obs_raw = root_obs if root_obs is not None else (env._get_obs() if hasattr(env, "_get_obs") else env.observe().values)
            obs_tensor = torch.from_numpy(root_obs_raw).float().to(dev).unsqueeze(0)
            mask_tensor = torch.from_numpy(root_mask).bool().to(dev).unsqueeze(0)

            with torch.no_grad():
                use_opp_model = getattr(self.agent.config, "use_opponent_modeling", False)
                if not use_opp_model or root_actor == 0:
                    logits_tensor = self.agent.action_net(obs_tensor, mask_tensor)
                else:
                    opp_probs = self.agent.predict_opponent_action(obs_tensor, mask_tensor)
                    logits_tensor = torch.log(opp_probs.clamp(min=1e-8)).unsqueeze(0)
                priors = F.softmax(logits_tensor, dim=-1).squeeze(0).cpu().numpy()
                raw_logits = logits_tensor.squeeze(0).cpu().numpy()

        # Root exploration noise (Dirichlet) for training self-play
        if add_noise and len(legal_indices) > 1:
            dir_alpha = float(getattr(self.config, "dirichlet_alpha", 0.30))
            dir_eps = float(getattr(self.config, "dirichlet_eps", 0.25))
            d_noise = np.random.dirichlet([dir_alpha] * len(legal_indices))
            priors[legal_indices] = (1.0 - dir_eps) * priors[legal_indices] + dir_eps * d_noise
            # Align raw logits with noisy priors to ensure Gumbel perturbation stays coherent
            raw_logits[legal_indices] = np.log(np.maximum(1e-8, priors[legal_indices]))

        # Root prior entropy & resource-efficient search budget
        p_legal = priors[legal_indices]
        p_norm = p_legal / (np.sum(p_legal) + 1e-12)
        root_entropy = -float(np.sum(p_norm * np.log(p_norm + 1e-12)))

        is_entropy_gated = bool(
            not add_noise  # Never gate budget during self-play training!
            and getattr(self.config, "adaptive_budget_enabled", False)
            and (root_entropy < getattr(self.config, "entropy_threshold", 0.15))
        )
        effective_sims = (
            min(num_simulations, getattr(self.config, "min_simulations", 2))
            if is_entropy_gated
            else num_simulations
        )

        # 2. Gumbel noise perturbation
        u = np.random.uniform(1e-6, 1.0 - 1e-6, size=len(legal_indices))
        gumbel_noise = -np.log(-np.log(u))
        perturbed_logits = np.copy(raw_logits)
        for i, a_idx in enumerate(legal_indices):
            perturbed_logits[a_idx] += gumbel_noise[i]

        # 3. Two-Stage: Top-k Candidates Selection
        k = min(getattr(self.config, "gumbel_candidates", 4), len(legal_indices))
        if is_entropy_gated:
            k = min(k, max(1, effective_sims))
        sorted_legal = sorted(legal_indices, key=lambda a: perturbed_logits[a], reverse=True)
        candidates = list(sorted_legal[:k])

        # Initialize root node with children
        root = MCTSNode(player=root_actor, action_mask=root_mask)
        for action_idx in legal_indices:
            root.children[int(action_idx)] = MCTSNode(
                player=root_actor,
                action_mask=root_mask,
                prior=float(priors[action_idx]),
                parent=root,
                action_from_parent=int(action_idx),
            )
        root.is_expanded = True

        # 4. Sequential Halving
        total_budget = max(len(candidates), effective_sims)
        num_phases = max(1, int(math.ceil(math.log2(max(2, len(candidates))))))
        c_puct = self.config.c_puct
        use_unc = getattr(self.config, "use_epistemic_uncertainty", True)
        unc_scale = getattr(self.config, "uncertainty_scale", 0.50) if use_unc else 0.0
        dropout_passes = getattr(self.config, "mc_dropout_passes", 4)

        if hasattr(env, "get_all_vps"):
            root_vps = env.get_all_vps()
        else:
            root_vps = np.array(
                [float(p.get("vp", 0.0)) for p in getattr(env, "players_state", [{"vp": 0.0}] * 4)],
                dtype=np.float32,
            )

        active_set = list(candidates)
        for phase in range(num_phases):
            if len(active_set) <= 1:
                break
            sims_per_cand = max(1, total_budget // (num_phases * len(active_set)))

            for sim_round in range(sims_per_cand):
                rollout_items = []
                non_terminal_indices = []
                obs_list = []
                mask_list = []
                actors_list = []

                for cand_idx, cand_act in enumerate(active_set):
                    cand_milestone = compute_milestone_bonus(cand_act)
                    sim_env = env.clone()
                    node = root.children[cand_act]
                    search_path = [root, node]
                    if hasattr(sim_env, "step_fast"):
                        sim_env.step_fast(cand_act)
                    else:
                        sim_env.step(cand_act)

                    # Traverse downward if node is already expanded
                    depth = 0
                    while node.is_expanded and not sim_env.terminated and node.children and depth < 50:
                        act, node = node.best_child(c_puct=c_puct, uncertainty_scale=unc_scale)
                        search_path.append(node)
                        if hasattr(sim_env, "step_fast"):
                            sim_env.step_fast(act)
                        else:
                            sim_env.step(act)
                        depth += 1

                    item = {
                        "sim_env": sim_env,
                        "node": node,
                        "search_path": search_path,
                        "cand_milestone": cand_milestone,
                    }
                    rollout_items.append(item)

                    if not sim_env.terminated:
                        curr_obs = sim_env._get_obs()
                        curr_mask = sim_env.get_action_mask()
                        leaf_actor = sim_env.current_player
                        item["curr_obs"] = curr_obs
                        item["curr_mask"] = curr_mask
                        item["leaf_actor"] = leaf_actor
                        non_terminal_indices.append(cand_idx)
                        obs_list.append(curr_obs)
                        mask_list.append(curr_mask)
                        actors_list.append(leaf_actor)

                # Batched Neural Evaluation for non-terminal leaves in one GPU forward pass
                if non_terminal_indices:
                    stacked_obs = np.stack(obs_list)
                    stacked_masks = np.stack(mask_list)

                    if use_unc:
                        dev = self.device if self.device is not None else torch.device("cpu")
                        batch_obs_t = torch.from_numpy(stacked_obs).float().to(dev)
                        batch_mask_t = torch.from_numpy(stacked_masks).bool().to(dev)
                        pred_scores = []
                        leaf_uncs = []
                        leaf_priors_list = []
                        with torch.no_grad():
                            for i_nt in range(len(non_terminal_indices)):
                                s_obs = batch_obs_t[i_nt : i_nt + 1]
                                s_mask = batch_mask_t[i_nt : i_nt + 1]
                                s_actor = actors_list[i_nt]
                                p_s, l_u = self.agent.predict_score_with_uncertainty(s_obs, num_passes=dropout_passes)
                                if s_actor == root_actor or not getattr(self.agent.config, "use_opponent_modeling", False):
                                    l_p = F.softmax(self.agent.action_net(s_obs, s_mask), dim=-1).squeeze(0).cpu().numpy()
                                else:
                                    l_p = self.agent.predict_opponent_action(s_obs, s_mask).cpu().numpy()
                                pred_scores.append(p_s)
                                leaf_uncs.append(l_u)
                                leaf_priors_list.append(l_p)
                    else:
                        if hasattr(self.agent, "evaluate_leaf_batch"):
                            pred_scores, leaf_priors_list = self.agent.evaluate_leaf_batch(
                                stacked_obs, stacked_masks, leaf_actors=actors_list
                            )
                        else:
                            dev = self.device if self.device is not None else torch.device("cpu")
                            batch_obs_t = torch.from_numpy(stacked_obs).float().to(dev)
                            batch_mask_t = torch.from_numpy(stacked_masks).bool().to(dev)
                            pred_scores = []
                            leaf_priors_list = []
                            for i_nt in range(len(non_terminal_indices)):
                                p_s, l_p = self.agent.evaluate_leaf(
                                    batch_obs_t[i_nt : i_nt + 1],
                                    batch_mask_t[i_nt : i_nt + 1],
                                    leaf_actor=actors_list[i_nt],
                                )
                                pred_scores.append(p_s)
                                leaf_priors_list.append(l_p)
                        leaf_uncs = [0.0] * len(non_terminal_indices)

                    for i_nt, nt_idx in enumerate(non_terminal_indices):
                        rollout_items[nt_idx]["pred_score"] = float(pred_scores[i_nt])
                        rollout_items[nt_idx]["leaf_priors"] = leaf_priors_list[i_nt]
                        rollout_items[nt_idx]["leaf_unc"] = float(leaf_uncs[i_nt])

                # Process results and backpropagate
                for item in rollout_items:
                    sim_env = item["sim_env"]
                    node = item["node"]
                    search_path = item["search_path"]

                    if sim_env.terminated:
                        if hasattr(sim_env, "get_all_vps"):
                            raw_vps = sim_env.get_all_vps()
                        else:
                            raw_vps = np.array(
                                [float(p.get("vp", 0.0)) for p in getattr(sim_env, "players_state", [{"vp": 0.0}] * 4)],
                                dtype=np.float32,
                            )
                        mean_vp = float(np.mean(raw_vps)) if len(raw_vps) > 0 else 50.0
                        margin = (raw_vps - mean_vp) / 20.0
                        ambition = (raw_vps - 90.0) / 40.0
                        value_vector = 0.6 * margin + 0.4 * ambition
                        leaf_unc = 0.0
                    else:
                        curr_mask = item["curr_mask"]
                        leaf_actor = item["leaf_actor"]
                        pred_score = item["pred_score"]
                        leaf_priors = item["leaf_priors"]
                        leaf_unc = item["leaf_unc"]
                        cand_milestone = item["cand_milestone"]

                        if np.any(curr_mask):
                            node.player = leaf_actor
                            node.action_mask = curr_mask
                            for a_idx in np.where(curr_mask)[0]:
                                node.children[int(a_idx)] = MCTSNode(
                                    player=leaf_actor,
                                    action_mask=curr_mask,
                                    prior=float(leaf_priors[a_idx]),
                                    parent=node,
                                    action_from_parent=int(a_idx),
                                )
                            node.is_expanded = True

                        if hasattr(sim_env, "get_all_vps"):
                            leaf_vps = sim_env.get_all_vps()
                        else:
                            leaf_vps = np.array(
                                [float(p.get("vp", 0.0)) for p in getattr(sim_env, "players_state", [{"vp": 0.0}] * 4)],
                                dtype=np.float32,
                            )
                        delta_vps = leaf_vps - root_vps

                        milestone_w = getattr(self.config, "milestone_shaping_weight", 0.50)
                        cand_bonus = milestone_w * cand_milestone if leaf_actor == root_actor else 0.0

                        optimism_weight = getattr(self.config, "optimism_weight", 0.35)
                        ego_optimism = optimism_weight * max(0.0, pred_score - 70.0)
                        leaf_projected = pred_score + delta_vps[leaf_actor] + cand_bonus + ego_optimism

                        raw_vps = np.zeros(4, dtype=np.float32)
                        raw_vps[leaf_actor] = leaf_projected
                        for i in range(4):
                            if i != leaf_actor:
                                raw_vps[i] = pred_score + (root_vps[i] - root_vps[leaf_actor]) + (delta_vps[i] - delta_vps[leaf_actor])

                        mean_vp = float(np.mean(raw_vps)) if len(raw_vps) > 0 else 50.0
                        margin = (raw_vps - mean_vp) / 25.0
                        ambition = (raw_vps - 90.0) / 40.0
                        value_vector = 0.6 * margin + 0.4 * ambition

                    # Backpropagate
                    for n in reversed(search_path):
                        n.visit_count += 1
                        n.add_value(value_vector)
                        n.total_uncertainty += leaf_unc

            # Eliminate worst half of candidates with epistemic exploration term
            active_set.sort(
                key=lambda a: float(
                    perturbed_logits[a]
                    + 2.0 * root.children[a].get_q(root_actor)
                    + (unc_scale * (root.children[a].total_uncertainty / max(1, root.children[a].visit_count)) if unc_scale > 0 else 0.0)
                ),
                reverse=True,
            )
            survivors_count = max(1, len(active_set) // 2)
            active_set = active_set[:survivors_count]

        # 5. Formulate final policy decision
        q_scores = np.zeros(action_dim, dtype=np.float32)
        for a in legal_indices:
            child = root.children.get(a)
            if child is not None and child.visit_count > 0:
                q_val = child.get_q(root_actor)
                unc_val = (child.total_uncertainty / child.visit_count) if unc_scale > 0 else 0.0
                q_scores[a] = raw_logits[a] + 2.0 * q_val + unc_scale * unc_val
            else:
                q_scores[a] = raw_logits[a]

        exp_s = np.exp(q_scores[legal_indices] - np.max(q_scores[legal_indices]))
        probs = np.zeros(action_dim, dtype=np.float32)
        probs[legal_indices] = exp_s / np.sum(exp_s)

        # Temperature-aware action selection in self-play
        if temperature > 0.05 and len(legal_indices) > 1:
            temp_logits = (q_scores[legal_indices] - np.max(q_scores[legal_indices])) / max(0.05, float(temperature))
            exp_t = np.exp(np.clip(temp_logits, -50.0, 50.0))
            sample_probs = exp_t / np.sum(exp_t)
            selected_action = int(np.random.choice(legal_indices, p=sample_probs))
        else:
            selected_action = int(active_set[0])

        total_visits = sum(c.visit_count for c in root.children.values())
        meta = {
            "algorithm": "gumbel_gaz",
            "root_visits": int(total_visits),
            "simulations": int(total_visits),
            "root_q_values": root.q_values.tolist(),
            "child_visits": {int(a): int(c.visit_count) for a, c in root.children.items()},
            "candidates": candidates,
            "candidate_actions": candidates,
            "completed_scores": q_scores.tolist(),
            "selected_action": selected_action,
            "root_entropy": float(root_entropy),
            "entropy_gated": bool(is_entropy_gated),
            "epistemic_uncertainty": {
                int(a): float(c.total_uncertainty / max(1, c.visit_count))
                for a, c in root.children.items()
            },
        }

        return selected_action, probs, meta

    def _search_puct(
        self,
        env: Any,
        num_simulations: int,
        temperature: float,
        add_noise: bool = False,
        root_obs: Optional[np.ndarray] = None,
        root_mask: Optional[np.ndarray] = None,
    ) -> Tuple[int, np.ndarray, Dict[str, Any]]:
        """AlphaZero PUCT / Max^n search (Classic multi-player baseline)."""
        sims = num_simulations
        temp = temperature
        c_puct = self.config.c_puct
        use_unc = getattr(self.config, "use_epistemic_uncertainty", True)
        unc_scale = getattr(self.config, "uncertainty_scale", 0.50) if use_unc else 0.0
        dropout_passes = getattr(self.config, "mc_dropout_passes", 4)

        # 1. Initialize Root Node
        root_actor = getattr(env, "current_player", 0)
        root_mask = root_mask if root_mask is not None else env.get_action_mask()
        action_dim = len(root_mask)
        legal_indices = np.where(root_mask)[0]

        if len(legal_indices) <= 1:
            act = int(legal_indices[0]) if len(legal_indices) == 1 else 0
            probs = np.zeros(action_dim, dtype=np.float32)
            probs[act] = 1.0
            return act, probs, {
                "algorithm": "puct",
                "root_visits": 0,
                "selected_action": act,
                "root_entropy": 0.0,
                "entropy_gated": True,
            }

        root = MCTSNode(player=root_actor, action_mask=root_mask)

        # Evaluate root priors
        if hasattr(self.agent, "evaluate_root"):
            root_obs_raw = root_obs if root_obs is not None else (env._get_obs() if hasattr(env, "_get_obs") else env.observe().values)
            priors, _ = self.agent.evaluate_root(root_obs_raw, root_mask, root_actor)
        else:
            dev = self.device if self.device is not None else torch.device("cpu")
            root_obs_raw = root_obs if root_obs is not None else (env._get_obs() if hasattr(env, "_get_obs") else env.observe().values)
            obs_tensor = torch.from_numpy(root_obs_raw).float().to(dev).unsqueeze(0)
            mask_tensor = torch.from_numpy(root_mask).bool().to(dev).unsqueeze(0)

            with torch.no_grad():
                use_opp_model = getattr(self.agent.config, "use_opponent_modeling", False)
                if not use_opp_model or root_actor == 0:
                    logits = self.agent.action_net(obs_tensor, mask_tensor)
                    priors = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                else:
                    priors = self.agent.predict_opponent_action(obs_tensor, mask_tensor).cpu().numpy()

        p_legal = priors[legal_indices]
        p_norm = p_legal / (np.sum(p_legal) + 1e-12)
        root_entropy = -float(np.sum(p_norm * np.log(p_norm + 1e-12)))

        is_entropy_gated = bool(
            getattr(self.config, "adaptive_budget_enabled", True)
            and (root_entropy < getattr(self.config, "entropy_threshold", 0.15))
        )
        if is_entropy_gated:
            sims = min(sims, getattr(self.config, "min_simulations", 2))

        if add_noise and len(legal_indices) > 1:
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(legal_indices))
            eps = self.config.dirichlet_eps
            for idx, noise_val in zip(legal_indices, noise):
                priors[idx] = (1.0 - eps) * priors[idx] + eps * noise_val

        # Expand root children
        for action_idx in legal_indices:
            root.children[int(action_idx)] = MCTSNode(
                player=root_actor,
                action_mask=root_mask,
                prior=float(priors[action_idx]),
                parent=root,
                action_from_parent=int(action_idx),
            )
        root.is_expanded = True

        if hasattr(env, "get_all_vps"):
            root_vps = env.get_all_vps()
        else:
            root_vps = np.array(
                [float(p.get("vp", 0.0)) for p in getattr(env, "players_state", [{"vp": 0.0}] * 4)],
                dtype=np.float32,
            )

        # 2. Run MCTS Simulations with Batched Virtual Loss
        chunk_size = getattr(self.config, "mcts_batch_size", 16)
        virtual_loss = 3.0
        sims_done = 0

        while sims_done < sims:
            current_chunk = min(chunk_size, sims - sims_done)
            sims_done += current_chunk

            rollout_items = []
            non_terminal_indices = []
            obs_list = []
            mask_list = []
            actors_list = []

            for c_idx in range(current_chunk):
                sim_env = env.clone()
                node = root
                search_path = [node]

                # --- SELECT with Virtual Loss ---
                while node.is_expanded and not sim_env.terminated and node.children:
                    action, node = node.best_child(c_puct=c_puct, uncertainty_scale=unc_scale)
                    search_path.append(node)
                    # Temporary virtual loss to diversify parallel exploration paths
                    node.visit_count += int(virtual_loss)
                    node.apply_virtual_loss(node.player, virtual_loss)
                    if hasattr(sim_env, "step_fast"):
                        sim_env.step_fast(action)
                    else:
                        sim_env.step(action)

                item = {
                    "sim_env": sim_env,
                    "node": node,
                    "search_path": search_path,
                }
                rollout_items.append(item)

                if not sim_env.terminated:
                    curr_obs = sim_env._get_obs()
                    curr_mask = sim_env.get_action_mask()
                    leaf_actor = sim_env.current_player
                    item["curr_obs"] = curr_obs
                    item["curr_mask"] = curr_mask
                    item["leaf_actor"] = leaf_actor
                    non_terminal_indices.append(c_idx)
                    obs_list.append(curr_obs)
                    mask_list.append(curr_mask)
                    actors_list.append(leaf_actor)

            # Revert virtual loss on search paths
            for item in rollout_items:
                for n in item["search_path"]:
                    if n != root:
                        n.visit_count -= int(virtual_loss)
                        n.revert_virtual_loss(n.player, virtual_loss)

            # Batched GPU inference for non-terminal leaves
            if non_terminal_indices:
                stacked_obs = np.stack(obs_list)
                stacked_masks = np.stack(mask_list)

                if use_unc:
                    dev = self.device if self.device is not None else torch.device("cpu")
                    batch_obs_t = torch.from_numpy(stacked_obs).float().to(dev)
                    batch_mask_t = torch.from_numpy(stacked_masks).bool().to(dev)
                    pred_scores = []
                    leaf_uncs = []
                    leaf_priors_list = []
                    with torch.no_grad():
                        for i_nt in range(len(non_terminal_indices)):
                            s_obs = batch_obs_t[i_nt : i_nt + 1]
                            s_mask = batch_mask_t[i_nt : i_nt + 1]
                            s_actor = actors_list[i_nt]
                            p_s, l_u = self.agent.predict_score_with_uncertainty(s_obs, num_passes=dropout_passes)
                            if s_actor == root_actor or not getattr(self.agent.config, "use_opponent_modeling", False):
                                l_p = F.softmax(self.agent.action_net(s_obs, s_mask), dim=-1).squeeze(0).cpu().numpy()
                            else:
                                l_p = self.agent.predict_opponent_action(s_obs, s_mask).cpu().numpy()
                            pred_scores.append(p_s)
                            leaf_uncs.append(l_u)
                            leaf_priors_list.append(l_p)
                else:
                    if hasattr(self.agent, "evaluate_leaf_batch"):
                        pred_scores, leaf_priors_list = self.agent.evaluate_leaf_batch(
                            stacked_obs, stacked_masks, leaf_actors=actors_list
                        )
                    else:
                        dev = self.device if self.device is not None else torch.device("cpu")
                        batch_obs_t = torch.from_numpy(stacked_obs).float().to(dev)
                        batch_mask_t = torch.from_numpy(stacked_masks).bool().to(dev)
                        pred_scores = []
                        leaf_priors_list = []
                        for i_nt in range(len(non_terminal_indices)):
                            p_s, l_p = self.agent.evaluate_leaf(
                                batch_obs_t[i_nt : i_nt + 1],
                                batch_mask_t[i_nt : i_nt + 1],
                                leaf_actor=actors_list[i_nt],
                            )
                            pred_scores.append(p_s)
                            leaf_priors_list.append(l_p)
                    leaf_uncs = [0.0] * len(non_terminal_indices)

                for i_nt, nt_idx in enumerate(non_terminal_indices):
                    rollout_items[nt_idx]["pred_score"] = float(pred_scores[i_nt])
                    rollout_items[nt_idx]["leaf_priors"] = leaf_priors_list[i_nt]
                    rollout_items[nt_idx]["leaf_unc"] = float(leaf_uncs[i_nt])

            # Expand & Backpropagate
            for item in rollout_items:
                sim_env = item["sim_env"]
                node = item["node"]
                search_path = item["search_path"]

                if sim_env.terminated:
                    if hasattr(sim_env, "get_all_vps"):
                        raw_vps = sim_env.get_all_vps()
                    else:
                        raw_vps = np.array(
                            [float(p.get("vp", 0.0)) for p in getattr(sim_env, "players_state", [{"vp": 0.0}] * 4)],
                            dtype=np.float32,
                        )
                    mean_vp = float(np.mean(raw_vps)) if len(raw_vps) > 0 else 50.0
                    value_vector = (raw_vps - mean_vp) / 20.0
                    leaf_unc = 0.0
                else:
                    curr_mask = item["curr_mask"]
                    leaf_actor = item["leaf_actor"]
                    pred_score = item["pred_score"]
                    leaf_priors = item["leaf_priors"]
                    leaf_unc = item["leaf_unc"]

                    if np.any(curr_mask):
                        node.player = leaf_actor
                        node.action_mask = curr_mask
                        for a_idx in np.where(curr_mask)[0]:
                            node.children[int(a_idx)] = MCTSNode(
                                player=leaf_actor,
                                action_mask=curr_mask,
                                prior=float(leaf_priors[a_idx]),
                                parent=node,
                                action_from_parent=int(a_idx),
                            )
                        node.is_expanded = True

                    if hasattr(sim_env, "get_all_vps"):
                        leaf_vps = sim_env.get_all_vps()
                    else:
                        leaf_vps = np.array(
                            [float(p.get("vp", 0.0)) for p in getattr(sim_env, "players_state", [{"vp": 0.0}] * 4)],
                            dtype=np.float32,
                        )
                    delta_vps = leaf_vps - root_vps

                    first_action = search_path[1].action_from_parent if len(search_path) > 1 and search_path[1].action_from_parent is not None else -1
                    cand_milestone = compute_milestone_bonus(first_action) if first_action >= 0 else 0.0
                    milestone_w = getattr(self.config, "milestone_shaping_weight", 0.50)
                    cand_bonus = milestone_w * cand_milestone if leaf_actor == root_actor else 0.0

                    optimism_weight = getattr(self.config, "optimism_weight", 0.25)
                    ego_optimism = optimism_weight * max(0.0, pred_score - 70.0)
                    leaf_projected = pred_score + delta_vps[leaf_actor] + cand_bonus + ego_optimism

                    raw_vps = np.zeros(4, dtype=np.float32)
                    raw_vps[leaf_actor] = leaf_projected
                    for i in range(4):
                        if i != leaf_actor:
                            raw_vps[i] = pred_score + (root_vps[i] - root_vps[leaf_actor]) + (delta_vps[i] - delta_vps[leaf_actor])

                    mean_vp = float(np.mean(raw_vps)) if len(raw_vps) > 0 else 50.0
                    value_vector = (raw_vps - mean_vp) / 25.0

                for n in reversed(search_path):
                    n.visit_count += 1
                    n.add_value(value_vector)
                    n.total_uncertainty += leaf_unc

        # 3. Formulate Action Decision from Visit Counts
        visit_counts = np.zeros(action_dim, dtype=np.float32)
        for action, child in root.children.items():
            visit_counts[action] = float(child.visit_count)

        total_root_visits = np.sum(visit_counts)
        if total_root_visits < 1e-6:
            probs = priors
            selected_action = int(np.argmax(priors))
        elif temp <= 1e-3:
            selected_action = int(np.argmax(visit_counts))
            probs = np.zeros(action_dim, dtype=np.float32)
            probs[selected_action] = 1.0
        else:
            exp_counts = np.power(visit_counts, 1.0 / max(1e-2, temp))
            probs = exp_counts / np.sum(exp_counts)
            selected_action = int(np.random.choice(action_dim, p=probs))

        meta = {
            "algorithm": "puct",
            "root_visits": int(total_root_visits),
            "root_q_values": root.q_values.tolist(),
            "child_visits": {int(a): int(c.visit_count) for a, c in root.children.items()},
            "selected_action": selected_action,
            "root_entropy": float(root_entropy),
            "entropy_gated": bool(is_entropy_gated),
            "epistemic_uncertainty": {
                int(a): float(c.total_uncertainty / max(1, c.visit_count))
                for a, c in root.children.items()
            },
        }

        return selected_action, probs, meta


# Backwards compatibility alias
GumbelAlphaZeroMCTS = MultiPlayerMCTS
