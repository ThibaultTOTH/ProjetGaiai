"""Comprehensive Test Suite for Asynchronous Batched MCTS & Tree Concurrency.

Tests:
1. Virtual Loss Algebra & Invariant Preservation.
2. PUCT Diversification under Virtual Loss.
3. Thread Safety & Race Condition Stress Test on MCTSNode.
4. BatchedNeuralEvaluator Queue Concurrency & Correctness.
5. Batched vs Serial Numerical Parity.
6. Async PUCT MCTS End-to-End Execution with Rust NativeGaiaEnv.
7. Gumbel GAZ MCTS Execution with Batched Leaf Evaluations.
8. Anti-Premature Pass & Milestone Bonus Behavioral Tests.
9. BatchedMultiGameMCTS Multi-Game Concurrent Search.
10. Throughput Benchmark & Deadlock Prevention.
"""

import math
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from config import AppConfig, MCTSConfig, ModelConfig
from environment import NativeGaiaEnv, make_gaia_env
from mcts import (
    BatchedMultiGameMCTS,
    BatchedNeuralEvaluator,
    EvaluationRequest,
    MCTSNode,
    MultiPlayerMCTS,
    compute_milestone_bonus,
    compute_premature_pass_penalty,
    compute_shaping_scale,
    compute_tier3_exploration_boost,
)
from models import DualGaiaAgent


# =====================================================================
# 1. Virtual Loss Algebra & Invariant Preservation
# =====================================================================

def test_virtual_loss_algebra():
    print("\n--- [Test 1] Virtual Loss Algebra & Invariant Preservation ---")
    mask = np.ones(3130, dtype=bool)
    node = MCTSNode(player=0, action_mask=mask)

    assert node.visit_count == 0
    assert node.virtual_loss == 0.0
    assert node.virtual_visits == 0
    assert np.allclose(node.total_value, [0.0, 0.0, 0.0, 0.0])
    assert node.get_q(0) == 0.0

    # 1. Apply virtual loss for Player 1
    loss_val = 3.0
    node.apply_virtual_loss(player=1, loss=loss_val, visits=1)
    assert node.visit_count == 1
    assert node.virtual_loss == loss_val
    assert node.virtual_visits == 1
    assert node.v1 == -loss_val
    assert node.get_q(1) == -3.0

    # 2. Revert virtual loss
    node.revert_virtual_loss(player=1, loss=loss_val, visits=1)
    assert node.visit_count == 0
    assert node.virtual_loss == 0.0
    assert node.virtual_visits == 0
    assert node.v1 == 0.0
    assert node.get_q(1) == 0.0

    # 3. Add real visit and multi-player values
    real_vals = [0.8, -0.4, 0.1, -0.5]
    node.add_real_visit_and_value(real_vals, uncertainty=0.25)
    assert node.visit_count == 1
    assert np.allclose(node.total_value, real_vals)
    assert math.isclose(node.get_q(0), 0.8, rel_tol=1e-5)
    assert math.isclose(node.get_q(1), -0.4, rel_tol=1e-5)
    assert math.isclose(node.total_uncertainty, 0.25, rel_tol=1e-5)
    print("  -> PASSED: Invariants strictly preserved, zero drift detected.")


# =====================================================================
# 2. PUCT Diversification under Virtual Loss
# =====================================================================

def test_puct_selection_diversification():
    print("\n--- [Test 2] PUCT Selection Diversification with Virtual Loss ---")
    mask = np.zeros(3130, dtype=bool)
    mask[10] = True
    mask[20] = True
    mask[30] = True

    parent = MCTSNode(player=0, action_mask=mask)
    parent.visit_count = 3

    # Children with equal priors
    c10 = MCTSNode(player=0, action_mask=mask, prior=0.333, parent=parent, action_from_parent=10)
    c20 = MCTSNode(player=0, action_mask=mask, prior=0.333, parent=parent, action_from_parent=20)
    c30 = MCTSNode(player=0, action_mask=mask, prior=0.333, parent=parent, action_from_parent=30)
    parent.children = {10: c10, 20: c20, 30: c30}
    parent.is_expanded = True

    # With no virtual loss and no visits, all scores equal -> first child 10 selected
    act1, _ = parent.best_child(c_puct=1.414)
    assert act1 == 10

    # Thread 1 enters branch 10 and applies virtual loss
    c10.apply_virtual_loss(player=0, loss=3.0, visits=1)
    # Thread 2 now selects from parent -> Must steer away from 10 to 20!
    act2, _ = parent.best_child(c_puct=1.414)
    assert act2 == 20, f"Expected 20, got {act2}"

    # Thread 2 enters branch 20 and applies virtual loss
    c20.apply_virtual_loss(player=0, loss=3.0, visits=1)
    # Thread 3 now selects from parent -> Must steer away from 10 and 20 to 30!
    act3, _ = parent.best_child(c_puct=1.414)
    assert act3 == 30, f"Expected 30, got {act3}"

    # Revert virtual loss on c10 and c20
    c10.revert_virtual_loss(player=0, loss=3.0, visits=1)
    c20.revert_virtual_loss(player=0, loss=3.0, visits=1)
    print("  -> PASSED: Virtual loss successfully diverts concurrent search threads across siblings.")


# =====================================================================
# 3. Thread Safety & Race Condition Stress Test on MCTSNode
# =====================================================================

def test_node_thread_safety():
    print("\n--- [Test 3] Thread Safety & Race Condition Stress Test ---")
    mask = np.ones(3130, dtype=bool)
    node = MCTSNode(player=0, action_mask=mask)

    NUM_THREADS = 16
    ITERS_PER_THREAD = 1000

    def worker_stress(t_id: int):
        for _ in range(ITERS_PER_THREAD):
            node.apply_virtual_loss(player=t_id % 4, loss=2.0, visits=1)
            # tiny computation
            _ = node.get_q(t_id % 4)
            node.revert_virtual_loss(player=t_id % 4, loss=2.0, visits=1)
            node.add_real_visit_and_value([0.1, 0.2, 0.3, 0.4], uncertainty=0.01)

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
        futures = [pool.submit(worker_stress, i) for i in range(NUM_THREADS)]
        for f in futures:
            f.result()

    expected_visits = NUM_THREADS * ITERS_PER_THREAD
    assert node.visit_count == expected_visits, f"Visit count mismatch: {node.visit_count} vs {expected_visits}"
    assert math.isclose(node.virtual_loss, 0.0, abs_tol=1e-5), "Virtual loss leaked!"
    assert math.isclose(node.virtual_visits, 0, abs_tol=1e-5), "Virtual visits leaked!"
    assert math.isclose(node.v0, expected_visits * 0.1, rel_tol=1e-4)
    print(f"  -> PASSED: {expected_visits} concurrent operations executed without race conditions or leaks.")


# =====================================================================
# 4. BatchedNeuralEvaluator Queue Concurrency & Correctness
# =====================================================================

def test_batched_evaluator_concurrency():
    print("\n--- [Test 4] BatchedNeuralEvaluator Queue Concurrency ---")
    cfg = ModelConfig(obs_dim=2476, action_dim=3130, use_gnn_map=False)
    agent = DualGaiaAgent(cfg)
    agent.eval()

    evaluator = BatchedNeuralEvaluator(
        agent=agent,
        device=torch.device("cpu"),
        max_batch_size=16,
        max_wait_ms=0.5,
    )

    TOTAL_REQUESTS = 64
    NUM_CLIENTS = 8
    reqs_per_client = TOTAL_REQUESTS // NUM_CLIENTS

    dummy_obs = np.random.randn(2476).astype(np.float32)
    dummy_mask = np.ones(3130, dtype=bool)

    results = []
    res_lock = threading.Lock()

    def client_worker(c_id: int):
        for _ in range(reqs_per_client):
            score, priors, unc = evaluator.request_eval(dummy_obs, dummy_mask, actor=c_id % 4)
            with res_lock:
                results.append((score, priors, unc))

    with ThreadPoolExecutor(max_workers=NUM_CLIENTS) as pool:
        futures = [pool.submit(client_worker, i) for i in range(NUM_CLIENTS)]
        for f in futures:
            f.result()

    assert len(results) == TOTAL_REQUESTS
    assert evaluator.total_evals == TOTAL_REQUESTS
    # Batching must have aggregated requests (total batches must be significantly less than 64)
    assert evaluator.total_batches < TOTAL_REQUESTS, f"Batching did not occur: batches={evaluator.total_batches}"

    for score, priors, unc in results:
        assert isinstance(score, float)
        assert -1.5 <= score <= 1.5
        assert priors.shape == (3130,)
        assert math.isclose(np.sum(priors), 1.0, abs_tol=1e-3)

    evaluator.close()
    print(f"  -> PASSED: {TOTAL_REQUESTS} evaluations batched into {evaluator.total_batches} passes cleanly.")


# =====================================================================
# 5. Batched vs Serial Numerical Parity
# =====================================================================

def test_batched_vs_serial_parity():
    print("\n--- [Test 5] Batched vs Serial Numerical Parity ---")
    cfg = ModelConfig(obs_dim=2476, action_dim=3130, use_gnn_map=False)
    agent = DualGaiaAgent(cfg)
    agent.eval()

    evaluator = BatchedNeuralEvaluator(
        agent=agent,
        device=torch.device("cpu"),
        max_batch_size=16,
        max_wait_ms=0.5,
    )

    obs = np.random.randn(2476).astype(np.float32)
    mask = np.ones(3130, dtype=bool)
    mask[50:100] = False

    # 1. Serial sync eval
    score_sync, priors_sync, _ = evaluator.evaluate_sync(obs, mask, actor=0)

    # 2. Async queued eval
    score_async, priors_async, _ = evaluator.request_eval(obs, mask, actor=0)

    assert math.isclose(score_sync, score_async, abs_tol=1e-5), f"Score mismatch: {score_sync} vs {score_async}"
    assert np.allclose(priors_sync, priors_async, atol=1e-5), "Priors mismatch between sync and async eval!"
    assert np.all(priors_sync[50:100] == 0.0), "Masked actions must have zero probability"

    evaluator.close()
    print("  -> PASSED: Sync and queued async evaluations match to < 1e-5 precision.")


# =====================================================================
# 6. Async PUCT MCTS End-to-End with NativeGaiaEnv
# =====================================================================

def test_async_puct_mcts_native_env():
    print("\n--- [Test 6] Async PUCT MCTS End-to-End with NativeGaiaEnv ---")
    cfg = ModelConfig(obs_dim=2476, action_dim=3130, use_gnn_map=False)
    agent = DualGaiaAgent(cfg)
    agent.eval()

    mcts_cfg = MCTSConfig(
        algorithm="puct",
        num_simulations=16,
        num_search_threads=4,
        virtual_loss=3.0,
        mcts_batch_size=16,
        use_async_batcher=True,
    )

    env = make_gaia_env(players=4, seed=42)
    obs, mask = env.reset(seed=42)

    mcts = MultiPlayerMCTS(agent, mcts_cfg, device=torch.device("cpu"))
    action, probs, meta = mcts.search(
        env,
        num_simulations=16,
        temperature=1.0,
        add_noise=True,
        root_obs=obs,
        root_mask=mask,
    )

    legal_indices = np.where(mask)[0]
    assert action in legal_indices, f"Selected action {action} is not legal!"
    assert probs.shape == (3130,)
    assert math.isclose(np.sum(probs), 1.0, abs_tol=1e-4)
    assert meta["algorithm"] == "puct_async"
    assert meta["root_visits"] == 16
    assert len(meta["child_visits"]) > 0

    if mcts.evaluator is not None:
        mcts.evaluator.close()
    print(f"  -> PASSED: Async PUCT completed 16 sims across 4 threads. Action={action}, Visits={meta['root_visits']}.")


# =====================================================================
# 7. Gumbel GAZ MCTS Execution with Batched Leaf Evaluations
# =====================================================================

def test_gumbel_gaz_mcts_batched():
    print("\n--- [Test 7] Gumbel GAZ MCTS with Batched Leaf Evaluations ---")
    cfg = ModelConfig(obs_dim=2476, action_dim=3130, use_gnn_map=False)
    agent = DualGaiaAgent(cfg)
    agent.eval()

    mcts_cfg = MCTSConfig(
        algorithm="gumbel",
        num_simulations=16,
        gumbel_candidates=4,
        mcts_batch_size=16,
        use_async_batcher=True,
    )

    env = make_gaia_env(players=4, seed=123)
    obs, mask = env.reset(seed=123)

    mcts = MultiPlayerMCTS(agent, mcts_cfg, device=torch.device("cpu"))
    action, probs, meta = mcts.search(
        env,
        num_simulations=16,
        temperature=0.0,  # Argmax
        add_noise=False,
        root_obs=obs,
        root_mask=mask,
    )

    legal_indices = np.where(mask)[0]
    assert action in legal_indices
    assert meta["algorithm"] == "gumbel_gaz"
    assert "candidate_actions" in meta
    assert len(meta["candidate_actions"]) <= 4
    assert meta["root_visits"] >= 4

    if mcts.evaluator is not None:
        mcts.evaluator.close()
    print(f"  -> PASSED: Gumbel GAZ completed Sequential Halving. Action={action}, Candidates={meta['candidates']}.")


# =====================================================================
# 8. Anti-Premature Pass & Milestone Bonus Behavioral Tests
# =====================================================================

def test_anti_pass_and_milestone_shaping():
    print("\n--- [Test 8] Anti-Premature Pass & Milestone Bonus Tests ---")

    # Milestone bonuses
    assert compute_milestone_bonus(50) == 0.5   # Mine
    assert compute_milestone_bonus(400) == 0.8  # Trading Station
    assert compute_milestone_bonus(401) == 1.0  # Lab
    assert compute_milestone_bonus(402) == 1.5  # PI
    assert compute_milestone_bonus(403) == 1.5  # Academy
    assert compute_milestone_bonus(1402) == 2.5 # Federation
    assert compute_milestone_bonus(1415) == 0.0 # Pass

    # Premature pass penalty
    dummy_obs = np.zeros(2476, dtype=np.float32)
    # Give player ample ore (base_idx=88 + 4) and credits (base_idx=88 + 3)
    dummy_obs[88 + 3] = 10.0 / 30.0  # 10 credits
    dummy_obs[88 + 4] = 6.0 / 15.0   # 6 ore
    dummy_obs[88 + 5] = 5.0 / 15.0   # 5 knowledge

    penalty_pass = compute_premature_pass_penalty(1415, dummy_obs)
    penalty_mine = compute_premature_pass_penalty(50, dummy_obs)
    assert penalty_pass > 2.0, f"Expected strong pass penalty, got {penalty_pass}"
    assert penalty_mine == 0.0

    # Shaping decay schedule
    assert compute_shaping_scale(epoch=0, anneal_horizon=500, initial_scale=2.5) == 2.5
    assert compute_shaping_scale(epoch=500, anneal_horizon=500, initial_scale=2.5) == 0.0
    assert compute_shaping_scale(epoch=250, anneal_horizon=500, initial_scale=2.5, mean_vp=145.0, cutoff_vp=140.0) == 0.0

    # Tier-3 boost
    assert compute_tier3_exploration_boost(epoch=0, initial_boost=3.5, horizon=500) == 3.5
    assert compute_tier3_exploration_boost(epoch=500, initial_boost=3.5, horizon=500) == 0.0
    print("  -> PASSED: Shaping and milestone dynamics verified.")


# =====================================================================
# 9. BatchedMultiGameMCTS Multi-Game Concurrent Search
# =====================================================================

def test_batched_multi_game_search():
    print("\n--- [Test 9] BatchedMultiGameMCTS Multi-Game Concurrent Search ---")
    cfg = ModelConfig(obs_dim=2476, action_dim=3130, use_gnn_map=False)
    agent = DualGaiaAgent(cfg)
    agent.eval()

    mcts_cfg = MCTSConfig(
        algorithm="puct",
        num_simulations=8,
        num_search_threads=2,
        mcts_batch_size=32,
    )

    multi_mcts = BatchedMultiGameMCTS(agent, mcts_cfg, device=torch.device("cpu"))

    envs = [make_gaia_env(players=4, seed=i * 100 + 7) for i in range(4)]
    for i, e in enumerate(envs):
        e.reset(seed=i * 100 + 7)

    results = multi_mcts.search_games(
        envs,
        num_simulations=8,
        temperature=1.0,
        add_noise=False,
    )

    assert len(results) == 4
    for i, (action, probs, meta) in enumerate(results):
        mask = envs[i].get_action_mask()
        assert action in np.where(mask)[0]
        assert meta["root_visits"] == 8
        assert probs.shape == (3130,)

    multi_mcts.close()
    print(f"  -> PASSED: 4 concurrent games searched simultaneously via shared BatchedNeuralEvaluator.")


# =====================================================================
# 10. Throughput Benchmark & Deadlock Prevention
# =====================================================================

def test_throughput_benchmark_and_deadlock():
    print("\n--- [Test 10] Throughput Benchmark & Deadlock Prevention ---")
    cfg = ModelConfig(obs_dim=2476, action_dim=3130, use_gnn_map=False)
    agent = DualGaiaAgent(cfg)
    agent.eval()

    mcts_cfg = MCTSConfig(
        algorithm="puct",
        num_simulations=32,
        num_search_threads=4,
        virtual_loss=3.0,
        mcts_batch_size=32,
        use_async_batcher=True,
    )

    env = make_gaia_env(players=4, seed=999)
    env.reset(seed=999)

    mcts = MultiPlayerMCTS(agent, mcts_cfg, device=torch.device("cpu"))

    start_t = time.perf_counter()
    action, probs, meta = mcts.search(env, num_simulations=32)
    elapsed = time.perf_counter() - start_t

    sims_per_sec = 32.0 / max(1e-4, elapsed)
    assert meta["root_visits"] == 32
    assert action in np.where(env.get_action_mask())[0]

    if mcts.evaluator is not None:
        mcts.evaluator.close()

    print(f"  -> PASSED: 32 simulations completed in {elapsed:.3f}s ({sims_per_sec:.1f} sims/sec) on CPU.")


if __name__ == "__main__":
    print("==================================================================")
    print("RUNNING MASSIVE ASYNC BATCHED MCTS TEST SUITE (10 TEST STAGES)")
    print("==================================================================")
    test_virtual_loss_algebra()
    test_puct_selection_diversification()
    test_node_thread_safety()
    test_batched_evaluator_concurrency()
    test_batched_vs_serial_parity()
    test_async_puct_mcts_native_env()
    test_gumbel_gaz_mcts_batched()
    test_anti_pass_and_milestone_shaping()
    test_batched_multi_game_search()
    test_throughput_benchmark_and_deadlock()
    print("\n==================================================================")
    print("ALL 10 TEST STAGES PASSED CLEANLY! ZERO DRIFT, ZERO DEADLOCKS!")
    print("==================================================================")
