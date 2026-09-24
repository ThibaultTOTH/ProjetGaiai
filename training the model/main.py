"""Entry point for Gaia Project Deep RL Studio.

Usage:
  # Launch GUI studio (Default):
  python main.py

  # Headless CLI training mode:
  python main.py --cli --epochs 200 --device cuda:0
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
import tkinter as tk

from config import AppConfig
from environment import make_gaia_env
from gui import GaiaRLStudioGUI
from trainer import RLTrainer


def run_cli_training(args: argparse.Namespace) -> None:
    """Runs headless command-line training."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cfg = AppConfig()
    if args.device:
        cfg.hardware.device_override = args.device
    if args.epochs:
        cfg.training.total_episodes = args.epochs

    print("\n" + "=" * 60, flush=True)
    print("  GAIA PROJECT - DEEP REINFORCEMENT LEARNING TRAINER (CLI)", flush=True)
    print("=" * 60, flush=True)

    trainer = RLTrainer(cfg)
    device_name = (
        trainer.hw_info.get("device_name", "CPU")
        if trainer.device.type == "cuda"
        else "CPU"
    )
    print(f"Device: {device_name} ({trainer.device})", flush=True)
    print(f"Target Epochs: {cfg.training.total_episodes}", flush=True)
    print(f"Batch Size: {cfg.training.batch_size} | Rollout Steps: {cfg.training.rollout_steps_per_epoch}\n", flush=True)

    env = make_gaia_env()
    print(f"Environment: {type(env).__name__}\n", flush=True)
    for epoch in range(1, cfg.training.total_episodes + 1):
        metrics = trainer.train_step(env)
        print(
            f"Epoch {metrics.epoch:04d} | "
            f"Steps/s: {metrics.steps_per_sec:5.0f} | "
            f"P-Loss: {metrics.policy_loss:6.3f} | "
            f"V-Loss: {metrics.value_loss:6.3f} | "
            f"Pred VP: {metrics.avg_predicted_score:5.1f} | "
            f"WinRate: {metrics.win_rate * 100:3.0f}%",
            flush=True,
        )

    print("\n[Done] Training complete. Evaluating final model...", flush=True)
    win_rate, avg_score, _ = trainer.evaluate_model(num_games=10)
    print(f"Final Validation: Win Rate = {win_rate * 100:.1f}% | Avg Score = {avg_score:.1f} VP", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gaia Project Deep RL Studio")
    parser.add_argument("--cli", action="store_true", help="Run in headless terminal mode without Tkinter GUI")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train (CLI mode)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cuda:0', 'cpu')")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config JSON")
    parser.add_argument("--report", nargs="?", const="Gaia_Project_Strategy_Report.pdf", default=None, help="Generate PDF Strategy Report and exit")
    args = parser.parse_args()

    if args.report:
        from analytics import generate_strategy_pdf
        from models import DualGaiaAgent
        cfg = AppConfig.load_from_file(args.config) if args.config else AppConfig()
        if args.device:
            cfg.hardware.device_override = args.device
        dev = cfg.hardware.get_torch_device()
        agent = DualGaiaAgent(cfg.model)
        agent.to_device(dev)
        print(f"Generating Strategy PDF Report at '{args.report}' using {dev}...", flush=True)
        pdf_path = generate_strategy_pdf(agent, output_path=args.report, device=dev)
        print(f"[Done] PDF Report ready at: {pdf_path}", flush=True)
        return

    if args.cli:
        run_cli_training(args)
    else:
        root = tk.Tk()
        cfg = AppConfig.load_from_file(args.config) if args.config else AppConfig()
        app = GaiaRLStudioGUI(root, cfg)
        root.mainloop()


if __name__ == "__main__":
    main()
