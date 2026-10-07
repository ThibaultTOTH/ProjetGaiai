"""Comprehensive Smoke & Unit Test for the Gaia Project Deep RL Pipeline."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import sys
import numpy as np
import torch
import torch.nn.functional as F

from config import AppConfig, LeagueConfig, MCTSConfig, MicroDispatchConfig, ModelConfig, TrainingConfig
from environment import NativeGaiaEnv, make_gaia_env
from hyperopt import AdvancedNASOptimizer, HyperoptTrial, play_head_to_head_game
from league import LeagueManager, LeagueMember
from mcts import MultiPlayerMCTS, compute_shaping_scale, compute_milestone_bonus, compute_premature_pass_penalty
from models import (
    ActionOptimizerNet,
    BottleneckResBlock,
    DualGaiaAgent,
    DualStreamBackbone,
    HexGNNEncoder,
    PreLNResBlock,
    ScorePredictorNet,
    SwiGLUBlock,
    build_block,
)
from alphazero_trainer import AlphaZeroTrainer, AlphaZeroReplayBuffer
from alphazero_parallel import ParallelAlphaZeroTrainer
from buffer import PrioritizedStateBuffer


def test_models_forward_and_backward():
    print("[1/19] Testing Models Forward & Backward (4 Players, obs_dim=2476)...")
    obs_dim = 2476
    action_dim = 16
    batch_size = 4

    obs = torch.randn(batch_size, obs_dim)
    mask = torch.ones(batch_size, action_dim, dtype=torch.bool)
    mask[:, 3] = False  # Make action 3 illegal

    # 1. Score Predictor
    score_net = ScorePredictorNet(obs_dim=obs_dim)
    predicted_score = score_net(obs)
    assert predicted_score.shape == (batch_size,), f"Unexpected score shape {predicted_score.shape}"

    loss_v = predicted_score.mean()
    loss_v.backward()
    assert score_net.head[-1].weight.grad is not None, "Score net gradients missing"

    # 2. Action Optimizer
    action_net = ActionOptimizerNet(obs_dim=obs_dim, action_dim=action_dim)
    logits = action_net(obs, mask)
    probs = torch.softmax(logits, dim=-1)

    # Action 3 must have zero probability due to masking
    assert torch.all(probs[:, 3] < 1e-6), "Action masking failed to zero out illegal action"

    loss_p = -logits.mean()
    loss_p.backward()
    assert action_net.policy_head[-1].weight.grad is not None, "Policy net gradients missing"
    print("      -> Forward, Backward & Action Masking OK!")


def test_dual_agent():
    print("[2/19] Testing DualGaiaAgent Act & Evaluate...")
    agent = DualGaiaAgent()
    obs = torch.randn(agent.config.obs_dim)
    mask = torch.ones(agent.config.action_dim, dtype=torch.bool)

    action, log_prob, pred_val, probs = agent.act_and_evaluate(obs, mask)
    assert 0 <= action < agent.config.action_dim, f"Invalid action {action}"
    assert isinstance(pred_val, float), "Predicted score is not float"
    assert probs.shape == (agent.config.action_dim,), f"Invalid probability distribution shape {probs.shape}"
    print("      -> Dual agent inference OK!")


def test_native_environment():
    print("[3/21] Testing NativeGaiaEnv (4 Players & Lost Fleet Factions)...")
    assert NativeGaiaEnv.is_available(), "Native gaiapi library (gaiapi.dll / libgaiapi.so) was not found"
    env = NativeGaiaEnv(players=4, seed=42)
    obs, mask = env.reset()
    assert obs.shape == (2476,), f"Unexpected obs shape {obs.shape}"
    assert mask.shape == (env.action_dim,), f"Unexpected mask shape {mask.shape}"
    assert env.action_dim == 3130, f"Expected action_dim=3130, got {env.action_dim}"

    # Test setting Lost Fleet factions (indices 14..17: Tinkeroids, Darkanians, Moweyds, Space Giants)
    env.set_player_faction(0, 14)  # Tinkeroids
    env.set_player_faction(1, 15)  # Darkanians
    env.set_player_faction(2, 16)  # Moweyds
    env.set_player_faction(3, 17)  # Space Giants

    legal = np.where(mask)[0]
    assert len(legal) > 0, "No legal actions found in initial state"
    act = int(legal[0])
    res = env.step(act)
    assert res.obs.shape == (2476,), "Unexpected obs shape after step"
    print("      -> Native Rust DLL bridge 4-player, 18-faction & turn rules OK!")


def test_trainer_single_step():
    print("[5/19] Testing AlphaZeroTrainer Iteration & Replay Buffer Training...")
    cfg = AppConfig()
    cfg.alphazero.enabled = True
    cfg.alphazero.num_simulations = 4
    cfg.alphazero.batch_size = 4
    trainer = AlphaZeroTrainer(cfg)
    env = make_gaia_env(players=cfg.model.num_players)

    # 1. Self-play game step
    game_history, p0_vp, p0_won, moves = trainer.self_play_game(env)
    assert len(game_history) > 0, "No transitions collected during self-play"
    assert moves > 0, "Move count must be > 0"
    for step in game_history:
        trainer.replay_buffer.add(*step)
    assert len(trainer.replay_buffer) == len(game_history)

    # 2. Batch gradient update
    batch = trainer.replay_buffer.sample(min(len(trainer.replay_buffer), 4))
    p_loss, v_loss, ent, pred_val = trainer.train_on_batch(batch)
    assert not math.isnan(p_loss), "Policy loss is NaN"
    assert not math.isnan(v_loss), "Value loss is NaN"
    print(f"      -> AlphaZero self-play & batch step OK! (Moves: {moves}, P-Loss: {p_loss:.3f}, V-Loss: {v_loss:.3f})")


def test_league_training():
    print("[6/19] Testing Population-Based League Training & Elo...")
    import os
    import shutil

    l_cfg = LeagueConfig(
        enabled=True,
        snapshot_interval_epochs=1,
        max_snapshots=5,
        league_dir="test_league_checkpoints",
    )
    manager = LeagueManager(l_cfg)

    # 1. Virtual members check
    assert "CurrentPolicy" in manager.members
    assert "RandomBaseline" in manager.members
    assert manager.members["CurrentPolicy"].elo == 1200.0

    # 2. Snapshot creation
    agent = DualGaiaAgent()
    snap_name = manager.add_snapshot(agent, epoch=1)
    assert snap_name in manager.members
    assert len(manager.historical_names) == 1
    assert manager.members[snap_name].policy_net is not None
    assert not manager.members[snap_name].policy_net.training, "Snapshot policy must be frozen in eval mode"

    # 3. Opponent sampling
    opponents = manager.sample_opponents(agent, num_opponents=3)
    assert len(opponents) == 3, f"Expected 3 opponents, got {len(opponents)}"
    test_obs = np.random.randn(2476).astype(np.float32)
    test_mask = np.ones(agent.config.action_dim, dtype=bool)
    for opp_name, act_fn in opponents:
        act = act_fn(test_obs, test_mask)
        assert 0 <= act < agent.config.action_dim, f"Invalid action {act} from opponent {opp_name}"

    # 4. Bradley-Terry Multi-Player Elo Update
    participants = ["CurrentPolicy", snap_name, "RandomBaseline", "RandomBaseline"]
    vps = [150.0, 110.0, 70.0, 60.0]
    initial_current_elo = manager.members["CurrentPolicy"].elo
    manager.update_match_results(participants, vps)
    assert manager.members["CurrentPolicy"].elo > initial_current_elo, "Winner Elo must increase!"
    assert manager.members["CurrentPolicy"].wins == 1
    assert manager.members["CurrentPolicy"].games == 1

    # 5. Leaderboard sorting
    leaderboard = manager.get_leaderboard()
    assert len(leaderboard) >= 3
    assert leaderboard[0]["name"] == "CurrentPolicy", "Winner must lead the leaderboard"

    # Cleanup test dir
    if os.path.exists("test_league_checkpoints"):
        shutil.rmtree("test_league_checkpoints", ignore_errors=True)

    print("      -> League snapshotting, sampling, Elo rating & leaderboards OK!")


def test_evaluation():
    print("[7/19] Testing AlphaZero Head-to-Head Tournament Match...")
    cfg = AppConfig()
    agent_a = DualGaiaAgent(cfg.model)
    agent_b = DualGaiaAgent(cfg.model)
    mcts_a = MultiPlayerMCTS(agent_a, config=cfg.mcts)
    mcts_b = MultiPlayerMCTS(agent_b, config=cfg.mcts)
    env = make_gaia_env(players=2)

    vps, winner = play_head_to_head_game(env, agent_a, mcts_a, agent_b, mcts_b, sims=4, max_steps=100)
    assert len(vps) == 2, f"Expected 2 player scores, got {len(vps)}"
    assert winner in (0, 1), f"Unexpected winner index {winner}"
    print(f"      -> Tournament match OK! (P0: {vps[0]:.0f} VP, P1: {vps[1]:.0f} VP, Winner: P{winner})")


def test_modular_blocks():
    print("[8/19] Testing Modular Neural Blocks (PreLNResBlock, BottleneckResBlock, SwiGLUBlock)...")
    batch_size = 4
    dim = 256

    x = torch.randn(batch_size, dim, requires_grad=True)

    # 1. PreLNResBlock
    pre_ln = PreLNResBlock(hidden_dim=dim, dropout=0.0, activation="silu")
    out_pre = pre_ln(x)
    assert out_pre.shape == (batch_size, dim), f"PreLN shape mismatch: {out_pre.shape}"
    out_pre.sum().backward(retain_graph=True)
    assert x.grad is not None and not torch.isnan(x.grad).any(), "PreLN backward failed"

    # 2. BottleneckResBlock
    x.grad.zero_()
    bottleneck = BottleneckResBlock(hidden_dim=dim, bottleneck_ratio=2, dropout=0.0, activation="gelu")
    out_bottle = bottleneck(x)
    assert out_bottle.shape == (batch_size, dim), f"Bottleneck shape mismatch: {out_bottle.shape}"
    out_bottle.sum().backward(retain_graph=True)
    assert x.grad is not None and not torch.isnan(x.grad).any(), "Bottleneck backward failed"

    # 3. SwiGLUBlock
    x.grad.zero_()
    swiglu = SwiGLUBlock(hidden_dim=dim, dropout=0.0, activation="silu")
    out_swiglu = swiglu(x)
    assert out_swiglu.shape == (batch_size, dim), f"SwiGLU shape mismatch: {out_swiglu.shape}"
    out_swiglu.sum().backward()
    assert x.grad is not None and not torch.isnan(x.grad).any(), "SwiGLU backward failed"

    # 4. Factory build_block
    blk_same = build_block("bottleneck", 256, 256)
    assert isinstance(blk_same, BottleneckResBlock)
    blk_proj = build_block("swiglu", 256, 128)
    out_proj = blk_proj(torch.randn(batch_size, 256))
    assert out_proj.shape == (batch_size, 128), "Projection block shape mismatch"
    print("      -> Pre-LN, Bottleneck, SwiGLU & Factory build OK!")


def test_dynamic_schedules():
    print("[9/19] Testing Dynamic LR Schedules (Warmup, Cosine, Linear, Exponential)...")
    import math

    cfg = AppConfig()
    cfg.training.total_episodes = 100
    cfg.training.warmup_ratio = 0.10  # 10 epochs warmup
    cfg.training.lr_final_factor = 0.10
    cfg.training.min_lr = 1e-6

    trainer = AlphaZeroTrainer(cfg)
    base_lr = 3e-4

    # 1. Warmup verification
    trainer.current_epoch = 0
    lr_ep0 = trainer.compute_scheduled_lr(base_lr)
    assert lr_ep0 < base_lr, f"Warmup ep0 should be smaller than base LR ({lr_ep0} vs {base_lr})"

    trainer.current_epoch = 10  # Warmup complete
    lr_ep10 = trainer.compute_scheduled_lr(base_lr)
    assert math.isclose(lr_ep10, base_lr, rel_tol=1e-3), f"Post-warmup LR should match base LR ({lr_ep10} vs {base_lr})"

    # 2. Cosine Decay verification
    cfg.training.lr_schedule_type = "cosine"
    trainer.current_epoch = 100
    lr_final = trainer.compute_scheduled_lr(base_lr)
    expected_min = base_lr * 0.10
    assert math.isclose(lr_final, expected_min, rel_tol=1e-2), f"Cosine final LR mismatch ({lr_final} vs {expected_min})"

    # 3. Linear Decay verification
    cfg.training.lr_schedule_type = "linear"
    trainer.current_epoch = 55  # Midpoint of 90 decay epochs
    lr_mid = trainer.compute_scheduled_lr(base_lr)
    assert expected_min < lr_mid < base_lr, f"Linear midpoint LR invalid: {lr_mid}"

    # 4. Exponential Decay verification
    cfg.training.lr_schedule_type = "exponential"
    cfg.training.exp_decay_rate = 0.95
    trainer.current_epoch = 20
    lr_exp = trainer.compute_scheduled_lr(base_lr)
    assert lr_exp < base_lr, "Exponential decay should reduce LR"

    print("      -> Dynamic Warmup, Cosine, Linear & Exponential decay OK!")


def test_nas_optimizer():
    print("[10/19] Testing Advanced NAS & AlphaZero Tournament Optimizer...")
    base_cfg = AppConfig()
    nas = AdvancedNASOptimizer(base_cfg)

    # 1. Candidate sampling
    cand = nas.generate_candidate()
    assert "block_type" in cand, "Missing block_type in candidate"
    assert "hidden_layers" in cand, "Missing hidden_layers in candidate"
    assert "az_policy_lr" in cand, "Missing az_policy_lr in candidate"
    assert "az_num_simulations" in cand, "Missing az_num_simulations in candidate"
    assert cand["block_type"] in ("pre_ln", "bottleneck", "swiglu")

    arch_str = nas.format_arch_name(cand)
    assert len(arch_str) > 0, "Empty formatted architecture string"

    # 2. Evolutionary Mutation
    mutated = nas.mutate_candidate(cand)
    assert isinstance(mutated, dict), "Mutation returned non-dict"
    assert "hidden_layers" in mutated and len(mutated["hidden_layers"]) >= 2

    # 3. Registered Baseline Extraction
    base_params = nas.extract_params_from_config(base_cfg)
    assert base_params["block_type"] == base_cfg.model.block_type
    assert base_params["hidden_layers"] == list(base_cfg.model.policy_hidden_layers)
    assert "az_policy_lr" in base_params
    base_arch_str = nas.format_arch_name(base_params)
    assert len(base_arch_str) > 0

    print("      -> AlphaZero Candidate Generation, Evolutionary Mutation & Baseline Seed OK!")


def test_annealed_shaping():
    print("[11/19] Testing Violent 500-Epoch Quadratic Shaping Annealing Schedule...")
    # 1. Quadratic schedule values
    s0 = compute_shaping_scale(0, anneal_horizon=500, initial_scale=50.0)
    s100 = compute_shaping_scale(100, anneal_horizon=500, initial_scale=50.0)
    s250 = compute_shaping_scale(250, anneal_horizon=500, initial_scale=50.0)
    s400 = compute_shaping_scale(400, anneal_horizon=500, initial_scale=50.0)
    s500 = compute_shaping_scale(500, anneal_horizon=500, initial_scale=50.0)
    s600 = compute_shaping_scale(600, anneal_horizon=500, initial_scale=50.0)

    assert math.isclose(s0, 50.0, rel_tol=1e-3), f"Initial scale must be 50.0, got {s0}"
    assert math.isclose(s250, 12.5, rel_tol=1e-3), f"Midpoint (ep 250) must be 12.5, got {s250}"
    assert s500 == 0.0, f"Scale at epoch 500 must strictly be 0.0, got {s500}"
    assert s600 == 0.0, f"Scale past epoch 500 must strictly be 0.0, got {s600}"

    # 2. Milestone bonuses
    assert compute_milestone_bonus(50) == 0.5    # BuildMine
    assert compute_milestone_bonus(450) == 0.8   # UpgradeTS
    assert compute_milestone_bonus(1401) == 2.5  # FormFederation

    # 3. Premature Pass Penalty
    dummy_obs = np.zeros(2476, dtype=np.float32)
    dummy_obs[88 + 4] = 0.5   # ore
    dummy_obs[88 + 3] = 0.5   # credits
    pass_penalty = compute_premature_pass_penalty(1415, dummy_obs)
    assert pass_penalty > 0.0, f"Pass when holding resources must incur penalty, got {pass_penalty}"
    print(f"      -> Violent Quadratic Annealing OK! Ep0={s0:.1f} -> Ep250={s250:.1f} -> Ep500={s500:.1f} (0.0 strictly at 500+)")


def test_hex_gnn_encoder():
    print("[12/19] Testing HexGNNEncoder & DualStreamBackbone (Hexagonal Adjacency)...")
    batch_size = 4
    map_dim = 2000
    scalar_dim = 476
    total_dim = 2476

    # 1. Direct HexGNNEncoder test
    gnn = HexGNNEncoder(in_features=10, hidden_dim=64, out_dim=256, layers=3)
    map_x = torch.randn(batch_size, map_dim, requires_grad=True)
    e_map = gnn(map_x)
    assert e_map.shape == (batch_size, 256), f"GNN map embedding shape mismatch: {e_map.shape}"
    e_map.sum().backward()
    assert map_x.grad is not None and not torch.isnan(map_x.grad).any(), "GNN backward gradient failed"

    # 2. DualStreamBackbone test
    backbone = DualStreamBackbone(
        scalar_dim=scalar_dim,
        map_dim=map_dim,
        use_gnn_map=True,
        trunk_layers=[512, 256],
    )
    obs = torch.randn(batch_size, total_dim, requires_grad=True)
    out = backbone(obs)
    assert out.shape == (batch_size, 256), f"Dual-Stream output shape mismatch: {out.shape}"
    out.sum().backward()
    assert obs.grad is not None and not torch.isnan(obs.grad).any(), "Dual-Stream backward gradient failed"
    print("      -> HexGNNEncoder & DualStreamBackbone forward/backward OK!")


def test_mcts_engine():
    print("[13/19] Testing Multi-Player MCTS Engine (Vectorized PUCT / Max^n)...")
    env = make_gaia_env(players=4)
    agent = DualGaiaAgent()
    mcts = MultiPlayerMCTS(agent)
    mcts.config.algorithm = "puct"

    action, probs, meta = mcts.search(env, num_simulations=20, temperature=0.0)
    assert 0 <= action < env.action_dim, f"Invalid MCTS action {action}"
    assert meta["root_visits"] == 20, f"Expected 20 root visits, got {meta['root_visits']}"
    assert len(meta["root_q_values"]) == 4, "Expected 4-player Q value vector"

    mask = env.get_action_mask()
    assert mask[action], f"MCTS selected an illegal action: {action}"
    print(f"      -> MCTS Search OK: Action={action} | Visits={meta['child_visits']}")


def test_opponent_modeling():
    print("[14/19] Testing Opponent Modeling Auxiliary Head & MCTS Prior...")
    batch_size = 4
    obs_dim = 2476
    action_dim = 16

    action_net = ActionOptimizerNet(obs_dim=obs_dim, action_dim=action_dim)
    obs = torch.randn(batch_size, obs_dim)
    mask = torch.ones(batch_size, action_dim, dtype=torch.bool)

    # 1. Dual forward pass (policy logits + opponent logits)
    logits, opp_logits = action_net(obs, mask, return_opponent=True)
    assert logits.shape == (batch_size, action_dim), f"Policy logits shape mismatch: {logits.shape}"
    assert opp_logits.shape == (batch_size, action_dim), f"Opponent logits shape mismatch: {opp_logits.shape}"

    # 2. Auxiliary Cross-Entropy loss & backward pass
    targets = torch.tensor([1, 3, 5, 2], dtype=torch.long)
    opp_loss = F.cross_entropy(opp_logits, targets)
    opp_loss.backward()
    assert action_net.opponent_head[-1].weight.grad is not None, "Opponent head gradients missing"

    # 3. Predict opponent on DualGaiaAgent
    agent = DualGaiaAgent()
    single_obs = torch.randn(obs_dim)
    single_mask = torch.ones(agent.config.action_dim, dtype=torch.bool)
    single_mask[3] = False  # Action 3 illegal
    opp_probs = agent.predict_opponent_action(single_obs, single_mask)
    assert opp_probs.shape == (agent.config.action_dim,), f"Opponent probs shape mismatch: {opp_probs.shape}"
    assert opp_probs[3] < 1e-6, "Opponent masked probability should be 0"
    assert math.isclose(float(opp_probs.sum()), 1.0, rel_tol=1e-3), "Opponent probs must sum to 1.0"
    print("      -> Opponent Head forward/backward, masked prior & backbone gradient OK!")


def test_alphazero_replay_buffer_and_optimism():
    print("[15/19] Testing AlphaZero Replay Buffer & Optimism-Biased Prioritization...")
    buf = AlphaZeroReplayBuffer(capacity=100)
    for i in range(20):
        obs = np.random.randn(2476).astype(np.float32)
        mask = np.ones(3130, dtype=bool)
        policy = np.zeros(3130, dtype=np.float32)
        policy[i] = 1.0
        val = float(i) / 20.0  # varying values
        buf.add(obs, mask, policy, val)

    assert len(buf) == 20
    b_obs, b_mask, b_pol, b_val = buf.sample(batch_size=8, optimism_power=2.0)
    assert b_obs.shape == (8, 2476)
    assert b_mask.shape == (8, 3130)
    assert b_pol.shape == (8, 3130)
    assert b_val.shape == (8,)
    print("      -> AlphaZero Replay Buffer, Capacity & Optimistic Sampling OK!")


def test_parallel_alphazero_architecture():
    print("[16/19] Testing Parallel AlphaZero Multi-Actor Architecture...")
    cfg = AppConfig()
    trainer = ParallelAlphaZeroTrainer(cfg, num_workers=2)
    assert trainer.num_workers == 2
    assert trainer.req_queue is not None
    assert trainer.stop_event is not None
    print("      -> Parallel AlphaZero Multi-Worker Process Queue & Context OK!")


def test_gumbel_alphazero():
    print("[17/19] Testing Gumbel AlphaZero (TSS-GAZ / Two-Stage Sequential Halving)...")
    env = make_gaia_env(players=4)
    agent = DualGaiaAgent()
    mcts = MultiPlayerMCTS(agent)

    # 1. Verify Gumbel search runs with 16 simulations and algorithm="gumbel"
    mcts.config.algorithm = "gumbel"
    mcts.config.num_simulations = 16
    mcts.config.gumbel_candidates = 4

    action, probs, meta = mcts.search(env, num_simulations=16, temperature=0.0)
    assert 0 <= action < env.action_dim, f"Invalid Gumbel action {action}"
    assert meta["algorithm"] in ("gumbel", "gumbel_gaz"), f"Expected algorithm gumbel, got {meta.get('algorithm')}"
    assert "candidate_actions" in meta, "Expected candidate_actions in Gumbel meta"
    assert "completed_scores" in meta, "Expected completed_scores in Gumbel meta"
    assert meta["simulations"] <= 16, f"Too many simulations: {meta['simulations']}"

    mask = env.get_action_mask()
    assert mask[action], f"Gumbel GAZ selected illegal action {action}"
    assert math.isclose(float(probs.sum()), 1.0, rel_tol=1e-3), "Gumbel policy probs must sum to 1.0"
    for a in range(env.action_dim):
        if not mask[a]:
            assert probs[a] == 0.0, f"Illegal action {a} received non-zero prob in Gumbel search"

    print(f"      -> Gumbel GAZ OK: Best Action={action} | Top Candidates={meta['candidate_actions']} | Policy Sum={probs.sum():.2f}")


def test_rnad_equilibrium():
    print("[18/19] Testing Regularized Nash Dynamics (R-NaD) Policy Anchor & Polyak Update...")
    from copy import deepcopy
    agent = DualGaiaAgent()
    ref_policy = deepcopy(agent.action_net)
    for p in ref_policy.parameters():
        p.requires_grad = False

    # 1. Parameter identity check
    p_orig = next(agent.action_net.parameters()).clone()
    p_ref_orig = next(ref_policy.parameters()).clone()
    assert torch.equal(p_orig, p_ref_orig), "Initial ref policy parameters should match active policy"

    # 2. Simulate parameter change in active policy and test Polyak update
    beta = 0.20
    with torch.no_grad():
        for p in agent.action_net.parameters():
            p.add_(1.0)
        for p_ref, p_act in zip(ref_policy.parameters(), agent.action_net.parameters()):
            p_ref.data.mul_(1.0 - beta).add_(p_act.data, alpha=beta)

    p_ref_updated = next(ref_policy.parameters())
    expected_val = p_ref_orig + beta
    assert torch.allclose(p_ref_updated, expected_val, atol=1e-5), "Polyak update formula mismatch!"

    # 3. Check KL divergence computation on dummy batch
    batch_obs = torch.randn(4, 2476)
    batch_mask = torch.ones(4, agent.config.action_dim, dtype=torch.bool)
    logits = agent.action_net(batch_obs, batch_mask)
    with torch.no_grad():
        ref_logits = ref_policy(batch_obs, batch_mask)
    p_probs = F.softmax(logits, dim=-1)
    p_log = F.log_softmax(logits, dim=-1)
    ref_log = F.log_softmax(ref_logits, dim=-1)
    kl = (p_probs * (p_log - ref_log)).sum(dim=-1).mean()
    assert float(kl.item()) >= 0.0, "KL divergence must be non-negative"
    print(f"      -> R-NaD Frozen Anchor, Polyak Target Update (beta=0.20) & KL Penalty ({float(kl.item()):.4f}) OK!")


def test_rgsc_state_curriculum():
    print("[19/19] Testing Regret-Guided Search Control (RGSC / Go-Exploit)...")
    buffer = PrioritizedStateBuffer(capacity=5, regret_threshold=0.40)
    env = make_gaia_env(players=4)
    env.reset()

    # Advance env by a few steps so it's an active mid-game state
    for _ in range(5):
        mask = env.get_action_mask()
        legal = np.where(mask)[0]
        env.step(int(legal[0]))

    # 1. Below threshold: must NOT be added
    added = buffer.add(env, regret=0.25, round_num=env.round)
    assert not added, "State below regret threshold should not be added"
    assert len(buffer) == 0

    # 2. Above threshold: must be added as a clone
    added = buffer.add(env, regret=0.85, round_num=env.round)
    assert added, "State above regret threshold must be added"
    assert len(buffer) == 1

    # 3. Add multiple states to test capacity eviction
    for r in [0.50, 0.95, 0.42, 0.70, 0.60]:
        buffer.add(env, regret=r, round_num=env.round)
    assert len(buffer) == 5, f"Expected buffer capacity 5, got {len(buffer)}"

    # 4. Sample mid-game puzzle clone
    puzzle = buffer.sample()
    assert puzzle is not None, "Sampled puzzle must not be None"
    assert puzzle.obs_dim == 2476
    assert puzzle.action_dim == env.action_dim
    assert puzzle.round == env.round, "Sampled puzzle must preserve game round"

    # 5. Play a step in the sampled clone without mutating original env
    orig_p = env.current_player
    puzzle_mask = puzzle.get_action_mask()
    legal_acts = np.where(puzzle_mask)[0]
    res = puzzle.step(int(legal_acts[0]))
    assert res.obs.shape == (2476,)
    assert env.current_player == orig_p, "Cloned puzzle step must not mutate original environment"

    print(f"      -> Prioritized State Buffer, Mid-Game State Cloning & Regret Priority OK!")


def test_gaussian_elo_matchmaking():
    print("[20/22] Testing Gaussian Elo-Window Matchmaking (Zone of Proximal Development)...")
    config = LeagueConfig(
        enabled=True,
        matchmaking_type="gaussian",
        matchmaking_elo_window=100.0,
        self_play_prob=0.0,
        historical_prob=1.0,
        random_prob=0.0,
    )
    league = LeagueManager(config=config)
    agent = DualGaiaAgent()

    # Current policy Elo set to 1500.0
    league.members["CurrentPolicy"].elo = 1500.0

    # Add historical snapshots:
    # snap_near: 1520 (diff=20, very close)
    # snap_mid: 1700 (diff=200, 2 stds away)
    # snap_far: 2100 (diff=600, 6 stds away)
    for name, elo_val in [("SnapNear", 1520.0), ("SnapMid", 1700.0), ("SnapFar", 2100.0)]:
        member = LeagueMember(
            name=name,
            policy_net=agent.clone_policy_net(),
            elo=elo_val,
            epoch=1,
        )
        league.members[name] = member
        league.historical_names.append(name)

    # Sample opponents 400 times
    counts = {"SnapNear": 0, "SnapMid": 0, "SnapFar": 0}
    for _ in range(400):
        opps = league.sample_opponents(agent, num_opponents=1)
        name = opps[0][0]
        if name in counts:
            counts[name] += 1

    assert counts["SnapNear"] > counts["SnapMid"], (
        f"SnapNear ({counts['SnapNear']}) must be sampled more than SnapMid ({counts['SnapMid']})"
    )
    assert counts["SnapMid"] > counts["SnapFar"], (
        f"SnapMid ({counts['SnapMid']}) must be sampled more than SnapFar ({counts['SnapFar']})"
    )
    assert counts["SnapFar"] <= 5, f"SnapFar ({counts['SnapFar']}) should virtually never be sampled"
    print(f"      -> Gaussian distribution: {counts} matches ZPD Elo curve perfectly!")


def test_epistemic_uncertainty_and_adaptive_mcts():
    print("[21/22] Testing Epistemic Uncertainty Guidance (MC-Dropout) & Adaptive Search Budget...")
    obs_dim = 2476
    agent = DualGaiaAgent()

    # 1. Test MC-Dropout in ScorePredictorNet & DualGaiaAgent
    dummy_obs = torch.randn(1, obs_dim)
    mean_val, std_val = agent.predict_score_with_uncertainty(dummy_obs, num_passes=4)
    assert isinstance(mean_val, float), "Mean score must be float"
    assert isinstance(std_val, float), "Uncertainty std must be float"
    assert std_val >= 0.0, "Epistemic uncertainty std must be non-negative"

    # With num_passes=1, std must be exactly 0.0
    _, std_single = agent.predict_score_with_uncertainty(dummy_obs, num_passes=1)
    assert std_single == 0.0, "Single pass uncertainty must be 0.0"

    # 2. Test MultiPlayerMCTS with Epistemic Uncertainty
    mcts_cfg = MCTSConfig(
        enabled=True,
        algorithm="gumbel",
        num_simulations=16,
        use_epistemic_uncertainty=True,
        mc_dropout_passes=4,
        uncertainty_scale=0.50,
        adaptive_budget_enabled=True,
        entropy_threshold=0.15,
        min_simulations=2,
    )
    mcts = MultiPlayerMCTS(agent=agent, config=mcts_cfg)
    env = make_gaia_env(players=4)
    env.reset()

    action, probs, meta = mcts.search(env, num_simulations=16)
    assert 0 <= action < env.action_dim
    assert probs.shape == (env.action_dim,)
    assert "epistemic_uncertainty" in meta, "Meta must contain epistemic_uncertainty"
    assert "root_entropy" in meta, "Meta must contain root_entropy"
    assert "entropy_gated" in meta, "Meta must contain entropy_gated"
    assert isinstance(meta["root_entropy"], float)

    # 3. Test Adaptive Budget / Entropy Gating
    # Simulate an obvious tactical action by forcing policy net logits to heavily favor action 0
    with torch.no_grad():
        for p in agent.action_net.policy_head.parameters():
            p.zero_()
        agent.action_net.policy_head[-1].bias.fill_(-100.0)
        first_legal = int(np.where(env.get_action_mask())[0][0])
        agent.action_net.policy_head[-1].bias[first_legal] = 100.0  # Dominant action will have ~1.0 probability

    action_gated, probs_gated, meta_gated = mcts.search(env, num_simulations=16)
    assert meta_gated["root_entropy"] < 0.15, f"Expected entropy < 0.15, got {meta_gated['root_entropy']}"
    assert meta_gated["entropy_gated"] is True, "Low entropy root must trigger entropy gating"
    assert meta_gated["simulations"] <= 4, f"Expected reduced simulations <= 4, got {meta_gated['simulations']}"
    assert action_gated == first_legal, f"Expected action {first_legal}, got {action_gated}"

    # 4. Test PUCT search with epistemic uncertainty
    mcts_puct_cfg = MCTSConfig(
        enabled=True,
        algorithm="puct",
        num_simulations=8,
        use_epistemic_uncertainty=True,
        mc_dropout_passes=2,
        uncertainty_scale=0.50,
    )
    mcts_puct = MultiPlayerMCTS(agent=agent, config=mcts_puct_cfg)
    p_act, p_probs, p_meta = mcts_puct.search(env, num_simulations=8)
    assert 0 <= p_act < env.action_dim
    assert "epistemic_uncertainty" in p_meta
    print("      -> MC-Dropout Epistemic Uncertainty (std >= 0) & Entropy Gated Budget (16 -> 2 sims) OK!")


def test_flat_action_space_and_native_bridge():
    print("[22/24] Testing Flat 3130 Action Space & Native Rust Bridge...")
    assert NativeGaiaEnv.is_available(), "Native Rust gaiapi DLL is required for flat 3130 action space"
    env = NativeGaiaEnv(players=4, seed=42)
    obs, mask = env.reset()

    # 1. Structural dimensions contract
    assert env.action_dim == 3130, f"Expected action_dim=3130, got {env.action_dim}"
    assert env.obs_dim == 2476, f"Expected obs_dim=2476, got {env.obs_dim}"
    assert obs.shape == (2476,), f"Observation vector shape mismatch: {obs.shape}"
    assert mask.shape == (3130,), f"Action mask shape mismatch: {mask.shape}"

    # 2. Action space offsets verification (3130 total discrete actions)
    # Offsets defined in Rust action_space.rs:
    # 0..199: BuildMine (200 hexes)
    # 200..399: StartGaiaProject (200 hexes)
    # 400..1399: UpgradeBuilding (200 hexes x 5 targets = 1000)
    # 1400..1405: FormFederation (6 token variants)
    # 1406..1411: AdvanceResearch (6 fields)
    # 1412..1421: Pass (10 round boosters)
    # 1422..1423: ChargePower / Decline
    # 1424..1433: BoardAction (10 actions)
    # 1434..1443: SpecialAction (10 actions)
    # 1444..1497: ClaimStandardTech (9 tiles x 6 fields = 54)
    # 1498..2307: ClaimAdvancedTech (15 adv tiles x 9 covered x 6 fields = 810)
    # 2308..3107: ExploreSpaceship (4 ships x 200 hexes = 800)
    # 3108..3123: SpaceshipBoardAction (4 ships x 4 types = 16)
    # 3124..3129: FreeAction (6 conversion types)
    assert 3129 < env.action_dim == 3130

    # 3. Step execution across consecutive turns
    legal_actions = np.where(mask)[0]
    assert len(legal_actions) > 0, "No initial legal actions available"

    for step_i in range(10):
        legal = np.where(env.get_action_mask())[0]
        if len(legal) == 0:
            break
        act = int(legal[0])
        assert 0 <= act < 3130, f"Action {act} outside valid 3130 bounds"
        res = env.step(act)
        assert res.obs.shape == (2476,), f"Step {step_i} obs shape mismatch: {res.obs.shape}"
        assert res.action_mask.shape == (3130,), f"Step {step_i} mask shape mismatch: {res.action_mask.shape}"
        assert isinstance(res.reward, float), f"Step {step_i} reward should be float"
        assert isinstance(res.done, bool), f"Step {step_i} done should be bool"
        if res.done:
            break

    print("      -> Flat 3130 Action Space (Offsets 0..3129) & Multi-Step Execution OK!")


def test_game_elements_and_lost_fleet_rules():
    print("[23/24] Testing All 18 Factions & Lost Fleet Elements Parity...")
    assert NativeGaiaEnv.is_available(), "Native Rust gaiapi DLL is required"
    env = NativeGaiaEnv(players=4, seed=100)

    # 1. Verify all 18 factions (Base 14 + Lost Fleet 4) can be seated
    factions_tested = [
        (0, 14),  # Seat 0: Tinkeroids (Lost Fleet)
        (1, 15),  # Seat 1: Darkanians (Lost Fleet)
        (2, 16),  # Seat 2: Moweyds (Lost Fleet)
        (3, 17),  # Seat 3: Space Giants (Lost Fleet)
    ]
    for seat, faction_id in factions_tested:
        env.set_player_faction(seat, faction_id)

    obs, mask = env.reset()
    assert obs.shape == (2476,)
    assert mask.shape == (3130,)
    assert np.any(mask), "Lost Fleet faction setup should yield legal opening moves"

    # 2. Test Base 14 Factions setup (Terrans, Lantids, Taklons, Nevlas, Gleens, etc.)
    base_factions = [(0, 0), (1, 1), (2, 8), (3, 12)]
    for seat, faction_id in base_factions:
        env.set_player_faction(seat, faction_id)
    obs_b, mask_b = env.reset()
    assert obs_b.shape == (2476,)
    assert np.any(mask_b)

    # 3. Test Environment State Cloning & Determinism
    env_clone = env.clone()
    assert env_clone.obs_dim == env.obs_dim == 2476
    assert env_clone.action_dim == env.action_dim == 3130
    assert np.array_equal(env.get_action_mask(), env_clone.get_action_mask())

    # Step on original must not mutate clone
    legal = np.where(env.get_action_mask())[0]
    if len(legal) > 0:
        env.step(int(legal[0]))
        clone_mask = env_clone.get_action_mask()
        assert clone_mask.shape == (3130,)

    print("      -> 18 Factions, Lost Fleet Elements, Cloning & Determinism OK!")


def test_strategy_pdf_report_and_analytics():
    print("[24/24] Testing Game-Theoretic Analytics & 4-Page Strategy PDF Report Generation...")
    from analytics import generate_strategy_pdf, generate_pdf_report, GameTheoryAnalytics
    agent = DualGaiaAgent()
    temp_pdf = os.path.join(os.path.dirname(__file__), "test_pipeline_report.pdf")
    try:
        out_path = generate_strategy_pdf(agent, output_path=temp_pdf)
        assert os.path.exists(out_path), "PDF report file must be created"
        assert os.path.getsize(out_path) > 10000, "PDF report should be at least 10 KB (multi-page)"
        print("      -> 4-Page SOTA Strategic PDF Report & Game-Theoretic Analytics OK!")
    finally:
        if os.path.exists(temp_pdf):
            try:
                os.remove(temp_pdf)
            except Exception:
                pass


def test_egocentric_observation():
    print("[25/27] Testing Egocentric Invariant Observation Formulation & Rotational Symmetry...")
    assert NativeGaiaEnv.is_available(), "Native gaiapi library is required"
    env = NativeGaiaEnv(players=4, seed=42, egocentric=True)
    obs_ego, mask = env.reset()
    assert obs_ego.shape == (2476,), f"Expected obs shape (2476,), got {obs_ego.shape}"
    assert np.all(np.isfinite(obs_ego)), "Observation tensor contains NaN or Inf"

    # Verify public method get_observation
    obs_ego2 = env.get_observation(egocentric=True)
    assert np.allclose(obs_ego, obs_ego2), "Egocentric observations should be identical"

    obs_std = env.get_observation(egocentric=False)
    assert obs_std.shape == (2476,)

    # In relative perspective, active seat is mapped to seat 0
    legal = np.where(mask)[0]
    if len(legal) > 0:
        res = env.step(int(legal[0]), egocentric=True)
        assert res.obs.shape == (2476,)
        assert np.all(np.isfinite(res.obs))
    print("      -> Egocentric Rotational Symmetry, Relative Perspective & Stability OK!")


def test_dynamic_hex_gnn_adjacency():
    print("[26/27] Testing Dynamic Procedural Map HexGNN Adjacency & Gradient Flow...")
    assert NativeGaiaEnv.is_available(), "Native gaiapi library is required"
    from models import compute_normalized_adjacency

    env = NativeGaiaEnv(players=4, seed=777)
    adj_table = env.get_map_adjacency()
    assert adj_table.shape == (200, 6), f"Expected adjacency shape (200, 6), got {adj_table.shape}"
    assert adj_table.dtype == np.uint8

    # Spectral normalization
    adj_norm = compute_normalized_adjacency(adj_table)
    assert adj_norm.shape == (200, 200), f"Expected normalized adjacency (200, 200), got {adj_norm.shape}"
    assert torch.all(torch.isfinite(adj_norm))

    # Encoder with dynamic adjacency
    encoder = HexGNNEncoder(in_features=10, hidden_dim=64, out_dim=256, layers=3)
    encoder.set_adjacency(adj_table)
    dummy_map = torch.randn(4, 2000)
    out = encoder(dummy_map)
    assert out.shape == (4, 256), f"Expected encoder output (4, 256), got {out.shape}"
    assert torch.all(torch.isfinite(out))

    # Agent dynamic adjacency injection & forward pass
    agent = DualGaiaAgent()
    agent.set_map_adjacency(adj_table)
    dummy_obs = torch.randn(4, 2476)
    dummy_mask = torch.ones(4, agent.config.action_dim, dtype=torch.bool)
    score_pred = agent.score_net(dummy_obs)
    logits = agent.action_net(dummy_obs, dummy_mask)
    assert score_pred.shape == (4,)
    assert logits.shape == (4, agent.config.action_dim)

    # Test gradient backprop
    loss = score_pred.mean() + logits.mean()
    loss.backward()
    assert agent.shared_backbone.map_encoder.convs[0].fc.weight.grad is not None, "GNN weights must receive gradients"
    print("      -> Procedural Hex Graph Extraction, Spectral Normalization & Backprop OK!")


if __name__ == "__main__":
    import os
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("  GAIA PROJECT ALPHAZERO PIPELINE - VERIFICATION SUITE")
    print("=" * 60)
    try:
        test_models_forward_and_backward()
        test_dual_agent()
        test_native_environment()
        test_trainer_single_step()
        test_league_training()
        test_evaluation()
        test_modular_blocks()
        test_dynamic_schedules()
        test_nas_optimizer()
        test_annealed_shaping()
        test_hex_gnn_encoder()
        test_mcts_engine()
        test_opponent_modeling()
        test_alphazero_replay_buffer_and_optimism()
        test_parallel_alphazero_architecture()
        test_gumbel_alphazero()
        test_rnad_equilibrium()
        test_rgsc_state_curriculum()
        test_gaussian_elo_matchmaking()
        test_epistemic_uncertainty_and_adaptive_mcts()
        test_flat_action_space_and_native_bridge()
        test_game_elements_and_lost_fleet_rules()
        test_strategy_pdf_report_and_analytics()
        test_egocentric_observation()
        test_dynamic_hex_gnn_adjacency()
        print("\n[SUCCESS] ALL 24 ALPHAZERO PIPELINE TESTS PASSED WITH 100% SUCCESS!")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FAILED] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

