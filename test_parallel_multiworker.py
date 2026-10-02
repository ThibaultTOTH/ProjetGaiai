import sys, os
sys.path.insert(0, os.path.abspath("training the model"))
os.environ["PYTHONIOENCODING"] = "utf-8"

import time
import torch
from config import AppConfig
from environment import make_gaia_env
from models import DualGaiaAgent
from alphazero_parallel import ParallelAlphaZeroTrainer

def run_test():
    print("[TEST] Initializing ParallelAlphaZeroTrainer with 4 workers...")
    cfg = AppConfig()
    cfg.alphazero.num_simulations = 8
    cfg.alphazero.games_per_epoch = 4
    cfg.alphazero.train_epochs_per_cycle = 1
    cfg.alphazero.batch_size = 16
    cfg.mcts.num_simulations = 8

    agent = DualGaiaAgent(cfg.model)
    trainer = ParallelAlphaZeroTrainer(cfg, agent, num_workers=4)

    env = make_gaia_env(players=4, seed=123)

    called_stats = []
    def test_cb(stats):
        called_stats.append(stats)
        ep = stats.epoch
        p_loss = stats.policy_loss
        v_loss = stats.value_loss
        speed = stats.steps_per_sec
        vp = stats.avg_real_score
        print(f"[TEST] Callback received epoch {ep}: speed={speed:.1f} st/s, loss_p={p_loss:.3f}, loss_v={v_loss:.3f}, avg_vp={vp:.1f}")

    print("[TEST] Running 1 epoch of parallel self-play & training...")
    start_t = time.time()
    trainer.run_training_loop(env, max_epochs=1, callback=test_cb)
    duration = time.time() - start_t

    print(f"[TEST] Finished in {duration:.2f}s!")
    assert len(called_stats) == 1, "Expected 1 epoch callback"
    buf_size = len(trainer.replay_buffer)
    print(f"[TEST] Replay buffer contains {buf_size} transitions.")
    assert buf_size > 0, f"Expected replay buffer > 0, got {buf_size}"

    print("[TEST] Shutting down workers...")
    trainer.stop_workers()
    print("[TEST] ALL PARALLEL MULTI-WORKER TESTS PASSED CLEANLY!")

if __name__ == "__main__":
    run_test()
