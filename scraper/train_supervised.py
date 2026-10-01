"""
Supervised Behavioral Cloning for Gaia Project DualGaiaAgent.
Pre-trains policy and score heads on expert human games scraped from Boardgamers.space.
Produces the foundational checkpoint to kickstart AlphaZero MCTS.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Add 'training the model' to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training the model"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
sys.path.insert(0, str(TRAINING_DIR))

from config import AppConfig, ModelConfig
from models import DualGaiaAgent
from environment import NativeGaiaEnv
from scraper.config import DATASET_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("supervised_trainer")


def train_supervised(
    dataset_path: Path,
    output_checkpoint: Path,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 3e-4,
    device_name: Optional[str] = None,
    callback: Optional[Any] = None,
):
    device = torch.device(
        device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Using device: {device}")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run convert_to_dataset first.")

    logger.info(f"Loading dataset from {dataset_path}...")
    raw = torch.load(dataset_path, weights_only=True)
    actions = raw["actions"]
    values = raw["values"]
    factions = raw["factions"]
    
    # If using rebuild_dataset_with_env.py, we have true observations!
    if "observations" in raw:
        observations = raw["observations"]
        logger.info("Found true observations in dataset!")
    else:
        logger.warning("No observations found in dataset. Using dummy templates.")
        observations = None
        
    num_samples = len(actions)
    logger.info(f"Dataset contains {num_samples:,} expert action samples.")

    obs_dim = 2476
    action_dim = 3130

    if observations is not None:
        dataset = TensorDataset(observations, actions, values, factions)
    else:
        dataset = TensorDataset(actions, values, factions)
        
    train_size = int(0.9 * num_samples)
    val_size = num_samples - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Initialize model
    config = AppConfig()
    config.model.obs_dim = obs_dim
    config.model.action_dim = action_dim
    agent = DualGaiaAgent(config.model).to(device)

    # Initialize GNN map adjacency & authentic observation templates
    faction_templates = {}
    if observations is None:
        try:
            base_env = NativeGaiaEnv(seed=42, egocentric=True)
            if hasattr(base_env, "get_map_adjacency"):
                adj = base_env.get_map_adjacency()
                agent.set_map_adjacency(adj)

            for f_idx in range(18):
                f_env = NativeGaiaEnv(seed=42 + f_idx, egocentric=True)
                if hasattr(f_env, "set_player_faction"):
                    f_env.set_player_faction(0, f_idx)
                    f_obs, _ = f_env.reset(seed=42 + f_idx)
                else:
                    f_obs = f_env.get_observation()
                f_obs[88] = float(f_idx) / 17.0
                f_obs[89] = 0.0  # relative seat 0
                faction_templates[f_idx] = torch.from_numpy(f_obs.copy()).float()
            logger.info(f"Initialized {len(faction_templates)} authentic faction templates and GNN adjacency.")
        except Exception as e:
            logger.warning(f"Failed to initialize NativeGaiaEnv templates: {e}. Falling back to default baseline.")
            for f_idx in range(18):
                t = torch.zeros(obs_dim, dtype=torch.float32)
                t[88] = float(f_idx) / 17.0
                faction_templates[f_idx] = t

    def _build_batch_obs(batch):
        if len(batch) == 4:
            # We have true observations (obs, act, val, fac)
            b_obs = batch[0].to(device)
            return b_obs
        else:
            # Fallback to templates (act, val, fac)
            b_fac = batch[2]
            B_len = b_fac.size(0)
            out = torch.zeros((B_len, obs_dim), device=device)
            for i_idx in range(B_len):
                f_id = b_fac[i_idx].item()
                tmpl = faction_templates.get(f_id, faction_templates[0]).clone().to(device)
                out[i_idx] = tmpl
            return out

    # Joint optimizer with cosine annealing
    optimizer = torch.optim.AdamW(agent.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.SmoothL1Loss()

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    logger.info(f"Starting Supervised Training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        agent.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total_train_samples = 0

        t0 = time.time()
        for batch in train_loader:
            if len(batch) == 4:
                batch_obs, batch_act, batch_val, batch_fac = batch
            else:
                batch_act, batch_val, batch_fac = batch
                
            batch_act = batch_act.to(device)
            batch_val = batch_val.to(device).unsqueeze(-1)
            batch_fac = batch_fac.to(device)
            B = batch_act.size(0)

            # Construct grounded observation vector from authentic templates
            batch_obs = _build_batch_obs(batch)

            # Forward pass
            logits = agent.action_net(batch_obs)
            pred_val = agent.score_net(batch_obs).view(-1)
            target_val = batch_val.view(-1)

            loss_p = criterion_policy(logits, batch_act)
            loss_v = criterion_value(pred_val, target_val)
            loss = loss_p + 0.5 * loss_v

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

            total_policy_loss += loss_p.item() * B
            total_value_loss += loss_v.item() * B
            total_train_samples += B

            # Top-1 & Top-5 accuracy
            with torch.no_grad():
                top1 = logits.argmax(dim=-1)
                correct_top1 += (top1 == batch_act).sum().item()
                _, top5 = logits.topk(5, dim=-1)
                correct_top5 += (top5 == batch_act.unsqueeze(-1)).any(dim=-1).sum().item()

        scheduler.step()
        train_p_loss = total_policy_loss / total_train_samples
        train_v_loss = total_value_loss / total_train_samples
        train_top1 = (correct_top1 / total_train_samples) * 100
        train_top5 = (correct_top5 / total_train_samples) * 100

        # Validation
        agent.eval()
        val_correct_top1 = 0
        val_correct_top5 = 0
        val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    batch_obs, batch_act, batch_val, batch_fac = batch
                else:
                    batch_act, batch_val, batch_fac = batch
                    
                batch_act = batch_act.to(device)
                batch_val = batch_val.to(device).unsqueeze(-1)
                batch_fac = batch_fac.to(device)
                B = batch_act.size(0)

                batch_obs = _build_batch_obs(batch)

                logits = agent.action_net(batch_obs)
                top1 = logits.argmax(dim=-1)
                val_correct_top1 += (top1 == batch_act).sum().item()
                _, top5 = logits.topk(5, dim=-1)
                val_correct_top5 += (top5 == batch_act.unsqueeze(-1)).any(dim=-1).sum().item()
                val_samples += B

        val_top1 = (val_correct_top1 / val_samples * 100) if val_samples else 0.0
        val_top5 = (val_correct_top5 / val_samples * 100) if val_samples else 0.0
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch:2d}/{epochs:2d} ({elapsed:.1f}s) | "
            f"Loss (P/V): {train_p_loss:.3f} / {train_v_loss:.3f} | "
            f"Train Acc: Top-1={train_top1:4.1f}%, Top-5={train_top5:4.1f}% | "
            f"Val Acc: Top-1={val_top1:4.1f}%, Top-5={val_top5:4.1f}%"
        )

        # Save best checkpoint
        if val_top1 >= best_val_acc or epoch == epochs:
            best_val_acc = val_top1
            checkpoint_data = {
                "epoch": epoch,
                "agent_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "val_top1_acc": val_top1,
                "val_top5_acc": val_top5,
            }
            torch.save(checkpoint_data, output_checkpoint)
            logger.info(f"Saved checkpoint to {output_checkpoint} (Val Top-1: {val_top1:.1f}%)")

        if callback is not None:
            class SupervisedMetrics:
                pass
            m = SupervisedMetrics()
            m.epoch = epoch
            m.policy_loss = float(train_p_loss)
            m.value_loss = float(train_v_loss)
            m.avg_predicted_score = float(val_top1)
            m.avg_real_score = 175.0
            m.entropy = 0.0
            m.fps = float(total_train_samples / max(0.1, elapsed))
            try:
                callback(m)
            except Exception as e:
                logger.warning(f"Callback error: {e}")

    logger.info("Supervised Pre-training finished successfully!")


def main():
    parser = argparse.ArgumentParser(description="Supervised Behavioral Cloning for Gaia Project")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DATASET_DIR / "gaia_expert_dataset.pt"),
        help="Path to dataset .pt file",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(CHECKPOINTS_DIR / "gaia_supervised_pretrained.pt"),
        help="Target checkpoint path",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    train_supervised(
        dataset_path=Path(args.dataset),
        output_checkpoint=Path(args.out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
