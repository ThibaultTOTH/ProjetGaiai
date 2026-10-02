#!/usr/bin/env python3
"""Production Deep RL Training Launcher for Gaia Project.

Supports:
- 1-Click scientifically calibrated presets ('fast', 'pro', 'grandmaster').
- Headless training on Linux / Cloud GPU clusters (AWS, GCP, RunPod, Lambda) or local terminal.
- Automatic resume from interruption (--resume auto / path).
- Live real-time terminal metrics with ANSI colors.
- Automatic checkpointing (gaia_latest.pt, gaia_best_elo.pt, versioned milestones).
- Post-training PDF strategic report generation.
"""

import argparse
import os
import signal
import sys
import time
from typing import Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch

from analytics import generate_strategy_pdf
from async_trainer import AsyncRLTrainer
from alphazero_trainer import AlphaZeroTrainer
from muzero import MuZeroTrainer
from config import AppConfig, get_training_preset
from environment import NativeGaiaEnv, make_gaia_env
from trainer import RLTrainer, TrainingMetrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SOTA Deep Reinforcement Learning Agent for Gaia Project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="grandmaster",
        help="Training preset ('pretrain', 'grandmaster', 'finetune', 'fast', 'debug')",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="auto",
        choices=["auto", "alphazero", "muzero", "ppo"],
        help="Training algorithm ('auto': auto-detect from preset, 'alphazero', 'muzero', 'ppo')",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override total epochs from preset",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override PPO mini-batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device target",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="auto",
        help="Checkpoint path to resume from ('auto' checks checkpoints/gaia_latest.pt or gaia_supervised_pretrained.pt)",
    )
    parser.add_argument(
        "--async-appo",
        action="store_true",
        help="Enable distributed asynchronous Actor-Learner training (APPO)",
    )
    parser.add_argument(
        "--actors",
        type=int,
        default=4,
        help="Number of CPU worker actors for asynchronous APPO mode",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        default=True,
        help="Generate PDF strategy analytics report on training completion",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Epoch interval between saved checkpoints",
    )
    parser.add_argument(
        "--hyperopt",
        action="store_true",
        help="Run Advanced Neural Architecture Search & Hyperparameter Optimization",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=15,
        help="Number of trials for hyperparameter search (default: 15)",
    )
    parser.add_argument(
        "--sprint-epochs",
        type=int,
        default=3,
        help="Number of sprint epochs per trial evaluation (default: 3)",
    )
    return parser.parse_args()


def print_banner(cfg: AppConfig, device: torch.device, hw_info: dict, preset_name: str, resumed_epoch: int = 0):
    print("=" * 78)
    print("  🌌 GAIA PROJECT DEEP RL — SOTA PRODUCTION TRAINER")
    print("=" * 78)
    cuda_avail = hw_info.get("cuda_available", False) and device.type == "cuda"
    if cuda_avail:
        dev_desc = f"🚀 GPU: {hw_info.get('device_name', 'CUDA')} ({hw_info.get('vram_total_gb', 0):.1f} GB VRAM)"
    else:
        dev_desc = "💻 Device: CPU (Optimized multi-threaded)"
    print(f"  {dev_desc}")
    if not cuda_avail and getattr(cfg.hardware, "device_override", "auto") != "cpu":
        print("  💡 INFO GPU : PyTorch tourne actuellement sur CPU. Sur votre machine dédiée avec GPU NVIDIA, activez CUDA via :")
        print("     pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    print(f"  ⚡ Mixed Precision: {'ON (AMP FP16/BF16)' if cfg.hardware.use_mixed_precision and cuda_avail else 'OFF'} | TF32: {'ON' if cfg.hardware.enable_tf32 and cuda_avail else 'OFF'}")
    print(f"  ⚙️ Preset: [{preset_name.upper()}] | Target Epochs: {cfg.training.total_episodes} | Batch: {cfg.training.batch_size}")
    print(f"  🧠 Network: {cfg.model.block_type.upper()} ({cfg.model.policy_activation}) Layers={cfg.model.policy_hidden_layers}")
    print(f"  🗺️ Map GNN: {'3-Layer HexGNN (200 Hexagons)' if cfg.model.use_gnn_map else 'Flat MLP'}")
    print(f"  🏆 League: {'Gaussian Elo ZPD (sigma=150)' if cfg.league.enabled and cfg.league.matchmaking_type == 'gaussian' else ('Enabled' if cfg.league.enabled else 'Disabled')}")
    print(f"  🛡️ SOTA Modules: R-NaD={'ON' if cfg.training.rnad_enabled else 'OFF'} | RGSC Go-Exploit={'ON' if cfg.training.rgsc_enabled else 'OFF'} | RND Curiosity={'ON' if cfg.training.rnd_enabled else 'OFF'}")
    print(f"  🌲 MCTS Inference: {cfg.mcts.algorithm.upper()} GAZ ({cfg.mcts.num_simulations} sims) | MC-Dropout Epistemic Guidance={'ON' if cfg.mcts.use_epistemic_uncertainty else 'OFF'}")
    native_env = NativeGaiaEnv.is_available()
    print(f"  🏎️ Game Engine: {'Rust Native C-ABI (DLL/so) [<2µs/step]' if native_env else 'REST / Fallback Engine'}")
    if resumed_epoch > 0:
        print(f"  🔄 Resumed From Epoch: {resumed_epoch}")
    print("-" * 78)


def main():
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # 1. Load Preset Config
    cfg = get_training_preset(args.preset)

    if args.epochs is not None:
        cfg.training.total_episodes = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
        cfg.alphazero.batch_size = args.batch_size
    if args.device != "auto":
        cfg.hardware.device_override = args.device
    if args.save_interval is not None:
        cfg.training.save_checkpoint_interval = args.save_interval
        cfg.alphazero.checkpoint_interval = args.save_interval

    algo = args.algo.lower()
    if algo == "auto":
        if getattr(cfg.alphazero, "enabled", False):
            algo = "alphazero"
        elif getattr(cfg.muzero, "enabled", False):
            algo = "muzero"
        else:
            algo = "ppo"

    # Handle Hyperparameter Search Mode (--hyperopt)
    if args.hyperopt:
        from hyperopt import AdvancedNASOptimizer, HyperoptTrial
        import json

        print("=" * 105)
        print("  🧬 ADVANCED NAS & HYPERPARAMETER SEARCH (GENETIC / MULTI-FIDELITY)")
        print(f"  Mode: [{algo.upper()}] | Target Trials: {args.trials} | Sprint Epochs: {args.sprint_epochs}")
        print(f"  Device: {cfg.hardware.device_override}")
        print("=" * 105)

        optimizer = AdvancedNASOptimizer(base_config=cfg, mode=algo)

        print(f"{'Trial':<8} {'Status':<14} {'Score Obj.':<12} {'Perte':<10} {'Victoires':<12} {'VP Moyen':<12} {'Architecture & Params'}")
        print("-" * 105)

        def _on_trial_update(t: HyperoptTrial):
            if t.status == "Running":
                return
            wr_str = f"{t.win_rate * 100:.0f}%" if t.win_rate > 0 else "-"
            score_str = f"{t.avg_score:.1f}" if t.avg_score > 0 else "-"
            print(
                f"[{t.trial_id:02d}/{args.trials:02d}] "
                f"{t.status:<14} "
                f"{t.objective_score:>10.2f} "
                f"{t.val_loss:>8.3f} "
                f"{wr_str:>10} "
                f"{score_str:>10}   "
                f"{t.architecture_name}"
            )

        best = optimizer.run_optimization_sync(
            num_trials=args.trials,
            sprint_epochs=args.sprint_epochs,
            on_trial_update=_on_trial_update,
        )

        print("=" * 105)
        if best:
            print(f"  🏆 BEST ARCHITECTURE & CONFIGURATION FOUND (Trial #{best.trial_id}):")
            print(f"     Objective Score : {best.objective_score:.2f}")
            print(f"     Description     : {best.architecture_name}")
            print(f"     Validation Loss : {best.val_loss:.4f}")
            print(f"     Win Rate / Top1 : {best.win_rate * 100:.1f}%")
            print(f"     Parameters:")
            for k, v in best.params.items():
                print(f"       - {k}: {v}")

            out_dir = getattr(cfg.training, "runs_dir", "runs")
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f"best_hyperparams_{algo}.json")
            with open(out_file, "w") as f:
                json.dump(best.params, f, indent=2)
            print(f"  💾 Best hyperparameters saved to: {out_file}")
        else:
            print("  [!] No trial completed successfully.")
        print("=" * 105)
        return

    if algo == "alphazero":
        trainer = AlphaZeroTrainer(cfg)
    elif algo == "muzero":
        trainer = MuZeroTrainer(cfg)
    elif args.async_appo:
        cfg.async_dist.enabled = True
        cfg.async_dist.num_actors = args.actors
        trainer = AsyncRLTrainer(cfg)
    else:
        trainer = RLTrainer(cfg)
    device = trainer.device

    # 2. Check for Resume
    resumed_epoch = 0
    if args.resume:
        resume_target = args.resume
        if resume_target.lower() == "auto":
            auto_path = os.path.join(cfg.training.checkpoint_dir, "gaia_latest.pt")
            if not os.path.exists(auto_path):
                auto_path = os.path.join(cfg.training.checkpoint_dir, "az_checkpoint_500.pt")
            if not os.path.exists(auto_path):
                auto_path = os.path.join(cfg.training.checkpoint_dir, "gaia_supervised_pretrained.pt")
            if os.path.exists(auto_path):
                resume_target = auto_path
            else:
                resume_target = None
        if resume_target and os.path.exists(resume_target):
            resumed_epoch = trainer.resume_from_checkpoint(resume_target)

    print_banner(cfg, device, trainer.hw_info, args.preset, resumed_epoch=resumed_epoch)

    # 3. Setup Graceful Shutdown on Ctrl+C (SIGINT)
    stop_requested = [False]

    def _sig_handler(sig, frame):
        if not stop_requested[0]:
            print("\n[!] Interrupt received (Ctrl+C). Saving current checkpoint before exiting...")
            stop_requested[0] = True
            trainer.stop()
        else:
            print("\n[!] Force exiting...")
            sys.exit(1)

    signal.signal(signal.SIGINT, _sig_handler)

    # 4. Training Loop
    target_epochs = cfg.training.total_episodes
    env = make_gaia_env(players=cfg.model.num_players)
    start_time = time.time()

    print(f"{'Époque':<10} {'Cadence':<12} {'Pertes (Pol. / Valeur)':<24} {'VP Moyen':<12} {'Victoires':<12} {'Algo':<10} {'VRAM'}")
    print("-" * 78)

    def _format_metric_row(ep, speed, p_loss, v_loss, avg_vp, win_r, vram):
        p_loss_str = f"P={p_loss:+.3f}"
        v_loss_str = f"V={v_loss:.3f}"
        losses_str = f"{p_loss_str} {v_loss_str}"
        wr_str = f"{win_r * 100.0:.0f}%"
        speed_str = f"{speed:.0f} st/s"
        print(
            f"[{ep:04d}/{target_epochs}] "
            f"{speed_str:<12} "
            f"{losses_str:<24} "
            f"{avg_vp:>5.1f} VP    "
            f"{wr_str:<12} "
            f"{algo.upper():<10} "
            f"{vram}"
        )

    try:
        if algo in ("alphazero", "muzero"):
            def _az_callback(m):
                vram = f"{torch.cuda.memory_allocated() / (1024**2):.0f} MB" if device.type == "cuda" else "CPU"
                _format_metric_row(m.epoch, m.steps_per_sec, m.policy_loss, m.value_loss, m.avg_real_score, m.win_rate, vram)
            trainer.run_training_loop(env, max_epochs=target_epochs, callback=_az_callback)
        else:
            while trainer.current_epoch < target_epochs and not stop_requested[0]:
                m: TrainingMetrics = trainer.train_step(env)
                vram = f"{m.vram_allocated_mb:.0f} MB" if device.type == "cuda" else "CPU"
                _format_metric_row(m.epoch, m.steps_per_sec, m.policy_loss, m.value_loss, m.avg_real_score, m.win_rate, vram)
                if m.epoch % cfg.training.save_checkpoint_interval == 0:
                    saved = trainer.save_current_checkpoint(is_milestone=True)
                    print(f"      💾 Checkpoint saved -> {saved}")

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
    finally:
        # Final checkpoint save
        final_ckpt = trainer.save_current_checkpoint(is_milestone=False)
        elapsed_min = (time.time() - start_time) / 60.0
        print("-" * 78)
        print(f"  🏁 Training Session Finished | Elapsed Time: {elapsed_min:.1f} minutes")
        print(f"  💾 Latest weights saved: {final_ckpt}")
        if os.path.exists(os.path.join(cfg.training.checkpoint_dir, "gaia_best_elo.pt")):
            print(f"  🏆 Best Elo weights: {os.path.join(cfg.training.checkpoint_dir, 'gaia_best_elo.pt')}")

        # Generate Strategy Report PDF if requested
        if args.report and trainer.current_epoch >= 5:
            report_path = os.path.join(cfg.training.runs_dir, f"gaia_strategy_report_ep{trainer.current_epoch}.pdf")
            print(f"  📄 Generating Strategic Analysis & Game Theory Report...")
            try:
                generate_strategy_pdf(trainer.agent, output_path=report_path, device=device)
                print(f"  [✓] Report PDF saved -> {report_path}")
            except Exception as e:
                print(f"  [!] Report generation notice: {e}")
        print("=" * 78)


if __name__ == "__main__":
    main()
