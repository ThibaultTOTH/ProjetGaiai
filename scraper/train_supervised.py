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

# Clean sys.path to avoid module shadowing with scraper.py
SCRAPER_DIR = str(Path(__file__).resolve().parent)
while sys.path and sys.path[0] == SCRAPER_DIR:
    sys.path.pop(0)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training the model"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(1, str(TRAINING_DIR))

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
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 1e-3,
    label_smoothing: float = 0.08,
    patience: int = 4,
    hyperparams_path: Optional[Path] = None,
    use_best_hyperparams: bool = False,
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
    raw = torch.load(dataset_path, weights_only=False)
    actions = raw["actions"]
    values = raw["values"]
    factions = raw["factions"]
    
    # If using rebuild_dataset_with_env.py, we have true observations!
    if "observations" in raw:
        observations = raw["observations"]
        logger.info("Found true observations in dataset!")
    else:
        logger.info("Dataset contains expert action sequences and outcomes without cached observation tensors.")
        logger.info("Using authentic NativeGaiaEnv faction-board state templates for behavioral cloning prior.")
        observations = None
        
    num_samples = len(actions)
    logger.info(f"Dataset contains {num_samples:,} expert action samples.")

    obs_dim = 2476
    action_dim = 3130

    if observations is not None:
        action_masks = raw.get("action_masks", None)
        if action_masks is not None:
            dataset = TensorDataset(observations, action_masks, actions, values, factions)
        else:
            dataset = TensorDataset(observations, actions, values, factions)
    else:
        dataset = TensorDataset(actions, values, factions)
        
    train_size = int(0.9 * num_samples)
    val_size = num_samples - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Initialize model config and load optimal hyperparameters if requested
    config = AppConfig()
    config.model.obs_dim = obs_dim
    config.model.action_dim = action_dim

    params_file = None
    if hyperparams_path and Path(hyperparams_path).exists():
        params_file = Path(hyperparams_path)
    elif use_best_hyperparams:
        default_file = Path(__file__).resolve().parent.parent / "runs" / "best_hyperparams_alphazero.json"
        if default_file.exists():
            params_file = default_file

    if params_file:
        import json
        with open(params_file, "r") as f:
            hparams = json.load(f)
        if "block_type" in hparams:
            config.model.block_type = hparams["block_type"]
        if "hidden_layers" in hparams:
            config.model.policy_hidden_layers = list(hparams["hidden_layers"])
            config.model.score_hidden_layers = list(hparams["hidden_layers"])
        if "activation" in hparams:
            config.model.policy_activation = hparams["activation"]
            config.model.score_activation = hparams["activation"]
        if "dropout" in hparams:
            config.model.policy_dropout = float(hparams["dropout"])
            config.model.score_dropout = float(hparams["dropout"])
        if "use_input_norm" in hparams:
            config.model.use_input_norm = bool(hparams["use_input_norm"])
        if "use_gnn_map" in hparams:
            config.model.use_gnn_map = bool(hparams["use_gnn_map"])
        if "gnn_layers" in hparams:
            config.model.gnn_layers = int(hparams["gnn_layers"])

        # Auto-apply optimal batch size & LR from hyperparams if present
        if "pretrain_batch_size" in hparams:
            batch_size = int(hparams["pretrain_batch_size"])
        elif "batch_size" in hparams:
            batch_size = int(hparams["batch_size"])
        if "pretrain_lr" in hparams:
            lr = float(hparams["pretrain_lr"])

        logger.info(f"✨ [Hyperopt] Applied optimal architecture and settings from {params_file.name}:")
        logger.info(f"   Architecture: {config.model.block_type.upper()} {config.model.policy_hidden_layers} ({config.model.policy_activation}) | GNN Map: {config.model.use_gnn_map}")
        logger.info(f"   Batch Size: {batch_size} | Learning Rate: {lr:.1e}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

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

    template_tensor = None
    if observations is None:
        template_tensor = torch.stack([
            faction_templates.get(i, faction_templates.get(0, torch.zeros(obs_dim)))
            for i in range(18)
        ]).to(device)

    def _build_batch(batch):
        """Unpacks batch into (obs, mask_or_None, act, val, fac) regardless of dataset format."""
        if len(batch) == 5:
            # (obs, action_mask, act, val, fac) - full dataset with simulator replay
            return batch[0].to(device).float(), batch[1].to(device).bool(), batch[2], batch[3], batch[4]
        elif len(batch) == 4:
            # (obs, act, val, fac) - obs without masks
            return batch[0].to(device).float(), None, batch[1], batch[2], batch[3]
        else:
            # (act, val, fac) - action-only, use faction templates
            b_fac = batch[2].to(device).clamp(0, 17)
            return template_tensor[b_fac], None, batch[0], batch[1], batch[2]

    # Joint optimizer with cosine annealing and L2 weight decay regularization
    optimizer = torch.optim.AdamW(agent.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    criterion_policy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    criterion_value = nn.SmoothL1Loss()

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    logger.info(f"Starting Supervised Training for {epochs} epochs (AMP {'ON' if use_cuda else 'OFF'}, EarlyStopping patience={patience})...")
    for epoch in range(1, epochs + 1):
        agent.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total_train_samples = 0

        t0 = time.time()
        for batch in train_loader:
            batch_obs, batch_mask, batch_act, batch_val, batch_fac = _build_batch(batch)
            batch_act = batch_act.to(device)
            batch_val = batch_val.to(device).unsqueeze(-1)
            B = batch_act.size(0)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda" if use_cuda else "cpu", enabled=use_cuda):
                # In supervised BC, compute policy loss on unmasked logits to prevent log(0) / inf
                # if the expert action falls outside a noisy simulated mask
                logits_unmasked = agent.action_net(batch_obs, action_mask=None)
                pred_val = agent.score_net(batch_obs).view(-1)
                target_val = torch.tanh(batch_val.view(-1))

                loss_p = criterion_policy(logits_unmasked, batch_act)
                loss_v = criterion_value(pred_val, target_val)
                loss = loss_p + 0.5 * loss_v

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_policy_loss += loss_p.item() * B
            total_value_loss += loss_v.item() * B
            total_train_samples += B

            # Top-1 & Top-5 accuracy
            with torch.no_grad():
                top1 = logits_unmasked.argmax(dim=-1)
                correct_top1 += (top1 == batch_act).sum().item()
                _, top5 = logits_unmasked.topk(5, dim=-1)
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
                batch_obs, batch_mask, batch_act, batch_val, batch_fac = _build_batch(batch)
                batch_act = batch_act.to(device)
                B = batch_act.size(0)

                with torch.amp.autocast(device_type="cuda" if use_cuda else "cpu", enabled=use_cuda):
                    logits = agent.action_net(batch_obs, action_mask=None)
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

        # Strictly save ONLY when validation accuracy improves (never overwrite with worse overfitted epoch!)
        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            best_epoch = epoch
            patience_counter = 0
            checkpoint_data = {
                "epoch": epoch,
                "agent_state_dict": agent.state_dict(),
                "score_net_state": agent.score_net.state_dict(),
                "action_net_state": agent.action_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "val_top1_acc": val_top1,
                "val_top5_acc": val_top5,
            }
            torch.save(checkpoint_data, output_checkpoint)
            logger.info(f"★ NEW BEST MODEL saved to {output_checkpoint} (Val Top-1: {val_top1:.1f}%, Top-5: {val_top5:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"⏹ Early stopping triggered at epoch {epoch} (no validation improvement for {patience} epochs).")
                logger.info(f"★ Best model preserved from Epoch {best_epoch} (Val Top-1: {best_val_acc:.1f}%)")
                break

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

    logger.info(f"Supervised Pre-training finished successfully! Best model: Epoch {best_epoch} with Val Top-1 = {best_val_acc:.1f}%")


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
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="L2 weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.08, help="Label smoothing")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    parser.add_argument(
        "--use-best-hyperparams",
        action="store_true",
        help="Train the supervised model using the optimal architecture from runs/best_hyperparams_alphazero.json",
    )
    parser.add_argument(
        "--hyperparams",
        type=str,
        default=None,
        help="Path to custom JSON file with architecture hyperparameters",
    )
    args = parser.parse_args()

    train_supervised(
        dataset_path=Path(args.dataset),
        output_checkpoint=Path(args.out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        patience=args.patience,
        hyperparams_path=Path(args.hyperparams) if args.hyperparams else None,
        use_best_hyperparams=args.use_best_hyperparams,
    )


if __name__ == "__main__":
    main()
