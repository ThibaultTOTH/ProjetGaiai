#!/usr/bin/env python3
"""Benchmark MCTS: Compare High-Budget Deterministic AI vs Training Exploration AI.

Scientifically measures the true playing strength of an AlphaZero model by pitting:
- 1 'PRO' Player: High MCTS budget (e.g. 128 or 256 sims), Dirichlet noise OFF, temperature=0.0 (greedy argmax).
against:
- 3 'TRAINING' Players: Training budget (32 sims), Dirichlet noise ON (alpha=0.3, eps=0.25), temperature exploration.

Features:
- Seat Rotation: Rotates the PRO player across all 4 seats (0, 1, 2, 3) to eliminate turn-order bias.
- Real-time Board & Metrics: Shows final VP, rank, delta VP, factions and duration.
- Statistical Summary: Win rate %, Top-2 rate %, Average VP differential.
"""

import argparse
import os
import sys
import time
from typing import List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

# Add "training the model" to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.join(ROOT_DIR, "training the model")
if TRAINING_DIR not in sys.path:
    sys.path.insert(0, TRAINING_DIR)

import numpy as np
import torch

from config import get_training_preset, AppConfig
from environment import make_gaia_env, NativeGaiaEnv, format_flat_action
from models import DualGaiaAgent
from mcts import MultiPlayerMCTS

FACTION_NAMES = [
    "Terrans", "Lantids", "Hadsch Hallas", "Ivits", "Geodens", "Bal T'aks",
    "Xenos", "Gleens", "Taklons", "Ambas", "Firaks", "Bescods",
    "Nevlas", "Itars"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark AlphaZero MCTS: High-Budget Deterministic vs Training Exploration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/gaia_latest.pt",
        help="Path to neural network weights (.pt)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=8,
        help="Total number of tournament games to play (recommended: multiple of 4)",
    )
    parser.add_argument(
        "--pro-sims",
        type=int,
        default=128,
        help="MCTS simulations for the PRO agent (e.g. 128 or 256)",
    )
    parser.add_argument(
        "--train-sims",
        type=int,
        default=32,
        help="MCTS simulations for the TRAINING baseline agents (32 as in training)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device for inference",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="grandmaster",
        help="Configuration preset ('grandmaster', 'pretrain')",
    )
    parser.add_argument(
        "--use-best-hyperparams",
        action="store_true",
        default=True,
        help="Apply architecture from runs/best_hyperparams_alphazero.json if present",
    )
    return parser.parse_args()


def load_agent(checkpoint_path: str, cfg: AppConfig, device: torch.device) -> DualGaiaAgent:
    """Loads and initializes DualGaiaAgent with checkpoint weights."""
    best_params_path = os.path.join(ROOT_DIR, "runs", "best_hyperparams_alphazero.json")
    if os.path.exists(best_params_path):
        import json
        from hyperopt import apply_params_to_config
        with open(best_params_path, "r") as f:
            custom_params = json.load(f)
        cfg = apply_params_to_config(cfg, custom_params, mode="alphazero")

    agent = DualGaiaAgent(cfg.model).to(device)

    # Resolve checkpoint
    target_path = checkpoint_path
    if not os.path.isabs(target_path):
        target_path = os.path.join(ROOT_DIR, target_path)

    if not os.path.exists(target_path):
        # Fallbacks
        alt_targets = [
            os.path.join(ROOT_DIR, "checkpoints", "gaia_latest.pt"),
            os.path.join(ROOT_DIR, "checkpoints", "az_checkpoint_1500.pt"),
            os.path.join(ROOT_DIR, "checkpoints", "az_checkpoint_1000.pt"),
            os.path.join(ROOT_DIR, "checkpoints", "gaia_supervised_pretrained.pt"),
        ]
        for alt in alt_targets:
            if os.path.exists(alt):
                target_path = alt
                break

    if os.path.exists(target_path):
        print(f"  📦 Loading checkpoint: {os.path.relpath(target_path, ROOT_DIR)}")
        try:
            meta = agent.load_checkpoint(target_path, device=device)
            agent = agent.to(device)
            ep = meta.get("epoch", "?")
            param_count = sum(p.numel() for p in agent.parameters() if p.requires_grad)
            print(f"  [✓] Weights loaded successfully (Epoch: {ep} | Parameters: {param_count:,}).")
        except Exception as e:
            print(f"  [!] Checkpoint load error: {e}")
    else:
        print(f"  ⚠️ Warning: No checkpoint found at {checkpoint_path}. Testing with random agent.")

    agent.eval()
    return agent


def run_benchmark_game(
    game_idx: int,
    pro_seat: int,
    pro_sims: int,
    train_sims: int,
    mcts: MultiPlayerMCTS,
    device: torch.device,
) -> Tuple[float, List[float], int, List[int], float]:
    """Runs a single 4-player game where 1 seat is PRO and 3 seats are TRAINING."""
    env = make_gaia_env(players=4)
    obs, mask = env.reset(seed=int(time.time() * 1000) % 1000000 + game_idx * 17)

    # Select 4 random distinct factions
    faction_ids = list(np.random.choice(len(FACTION_NAMES), 4, replace=False))
    if hasattr(env, "set_player_faction"):
        for s, f_id in enumerate(faction_ids):
            env.set_player_faction(s, int(f_id))

    start_t = time.time()
    move_count = 0
    total_steps = 0
    MAX_STEPS = 1200
    consecutive_errors = 0

    while not env.terminated and total_steps < MAX_STEPS:
        total_steps += 1
        curr_p = env.current_player
        curr_mask = env.get_action_mask()
        curr_mask[1422:1424] = False
        legal = np.where(curr_mask)[0]
        if len(legal) == 0:
            env.terminated = True
            break

        is_pro = (curr_p == pro_seat)
        round_num = getattr(env, "round", 1)
        pro_curr_vp = float(env.get_all_vps()[pro_seat]) if hasattr(env, "get_all_vps") else 0.0

        curr_obs = env._get_obs() if hasattr(env, "_get_obs") else None

        if is_pro:
            # Deterministic, high budget, zero exploration noise
            action, _, _ = mcts.search(
                env,
                num_simulations=pro_sims,
                temperature=0.0,
                add_noise=False,
                root_obs=curr_obs,
                root_mask=curr_mask,
            )
        else:
            # Exploration mode (like self-play training)
            temp = 0.8 if move_count < 30 else 0.2
            action, _, _ = mcts.search(
                env,
                num_simulations=train_sims,
                temperature=temp,
                add_noise=True,
                root_obs=curr_obs,
                root_mask=curr_mask,
            )

        action_name = format_flat_action(int(action))
        player_tag = f"PRO (P{curr_p + 1})" if is_pro else f"TRAIN (P{curr_p + 1})"
        sys.stderr.write(
            f"\r  ⚡ [Match #{game_idx:02d}] Coup {total_steps:03d} | R{round_num}/6 | {player_tag} | {action_name[:32]:<32} | PRO: {pro_curr_vp:.0f} VP ... "
        )
        sys.stderr.flush()

        step_res = env.step(action)
        move_count += 1

        # Fallback if illegal step
        if hasattr(step_res, "info") and "error" in step_res.info:
            consecutive_errors += 1
            sys.stderr.write(
                f"\n  ⚠️ [Rejet moteur] {player_tag} action {action} ({action_name}) invalide: {step_res.info.get('error', '')}\n"
            )
            sys.stderr.flush()
            fallback_ok = False
            alt_actions = np.random.permutation(legal)
            for alt in alt_actions:
                if alt != action:
                    step_res = env.step(int(alt))
                    if not (hasattr(step_res, "info") and "error" in step_res.info):
                        fallback_ok = True
                        consecutive_errors = 0
                        break
            if not fallback_ok and consecutive_errors >= 3:
                # Engine deadlock on illegal action mask, end game cleanly
                env.terminated = True
                break
        else:
            consecutive_errors = 0

    # Clear heartbeat line
    sys.stderr.write("\r" + " " * 95 + "\r")
    sys.stderr.flush()

    duration = time.time() - start_t

    if hasattr(env, "get_all_vps"):
        vps = [float(v) for v in env.get_all_vps()]
    else:
        vps = [float(p.get("vp", 0.0)) for p in getattr(env, "players_state", [{"vp": 0.0}] * 4)]

    pro_vp = vps[pro_seat]
    # Determine ranks (1st = best score)
    sorted_seats = sorted(range(4), key=lambda s: vps[s], reverse=True)
    pro_rank = sorted_seats.index(pro_seat) + 1

    return pro_vp, vps, pro_rank, faction_ids, duration


def main():
    args = parse_args()
    print("=" * 78)
    print("  🌌 GAIA PROJECT MCTS BENCHMARK — PRO EVAL vs TRAINING BASELINE")
    print("=" * 78)

    # Device selection
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        dev = torch.device("cuda")
        print(f"  🚀 Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        dev = torch.device("cpu")
        print("  💻 Device: CPU")

    cfg = get_training_preset(args.preset)
    agent = load_agent(args.checkpoint, cfg, dev)

    mcts = MultiPlayerMCTS(agent, cfg.mcts, device=dev)

    print(f"  ⚔️ PRO Player        : {args.pro_sims} simulations | Bruit: OFF (0.00) | Temp: 0.0 (Greedy Argmax)")
    print(f"  🛡️ TRAINING Players   : {args.train_sims} simulations | Bruit: ON (Dirichlet 0.25) | Temp: 0.8/0.2")
    print(f"  🔄 Matchs Prévus     : {args.games} parties (avec rotation cyclique des sièges 0, 1, 2, 3)")
    print("-" * 78)
    print(f"{'Match':<7} {'Siège PRO':<11} {'Faction PRO':<14} {'Score PRO':<12} {'Rang':<8} {'Scores Autres (TRAIN)':<24} {'Δ VP':<8} {'Durée'}")
    print("-" * 78)

    pro_wins = 0
    pro_top2 = 0
    pro_vps = []
    train_vps = []
    deltas = []

    for g in range(1, args.games + 1):
        pro_seat = (g - 1) % 4
        pro_vp, all_vps, pro_rank, factions, dur = run_benchmark_game(
            game_idx=g,
            pro_seat=pro_seat,
            pro_sims=args.pro_sims,
            train_sims=args.train_sims,
            mcts=mcts,
            device=dev,
        )

        other_vps = [v for s, v in enumerate(all_vps) if s != pro_seat]
        avg_other = float(np.mean(other_vps)) if other_vps else 0.0
        delta = pro_vp - avg_other

        pro_vps.append(pro_vp)
        train_vps.extend(other_vps)
        deltas.append(delta)

        if pro_rank == 1:
            pro_wins += 1
            rank_str = "1er 🏆"
        elif pro_rank == 2:
            rank_str = "2e 🥈"
        elif pro_rank == 3:
            rank_str = "3e 🥉"
        else:
            rank_str = "4e"

        if pro_rank <= 2:
            pro_top2 += 1

        faction_pro = FACTION_NAMES[factions[pro_seat]]
        other_str = " / ".join([f"{v:.0f}" for v in other_vps])
        delta_str = f"{delta:+.1f}"

        print(
            f"#{g:02d}     "
            f"P{pro_seat + 1:<10} "
            f"{faction_pro:<14} "
            f"{pro_vp:>5.1f} VP    "
            f"{rank_str:<8} "
            f"[{other_str}]           "
            f"{delta_str:<8} "
            f"{dur:.1f}s",
            flush=True,
        )

    print("=" * 78)
    print("  📊 BILAN STATISTIQUE DU BENCHMARK")
    print("=" * 78)
    win_rate = (pro_wins / max(1, args.games)) * 100.0
    top2_rate = (pro_top2 / max(1, args.games)) * 100.0
    mean_pro_vp = float(np.mean(pro_vps)) if pro_vps else 0.0
    mean_train_vp = float(np.mean(train_vps)) if train_vps else 0.0
    mean_delta = float(np.mean(deltas)) if deltas else 0.0

    print(f"  🏆 Taux de Victoire PRO (1ère place) : {win_rate:.1f}% (Attendu au hasard : 25.0%)")
    print(f"  🥈 Taux de Top-2 PRO (1er ou 2e)     : {top2_rate:.1f}% (Attendu au hasard : 50.0%)")
    print(f"  📈 Score Moyen PRO (128 sims)       : {mean_pro_vp:.1f} VP (Max: {max(pro_vps):.1f} VP)")
    print(f"  📉 Score Moyen TRAIN (32 sims bruit): {mean_train_vp:.1f} VP")
    print(f"  ⚡ Différentiel Moyen Net (Δ VP)    : {mean_delta:+.1f} VP par partie")
    print("=" * 78)


if __name__ == "__main__":
    main()
