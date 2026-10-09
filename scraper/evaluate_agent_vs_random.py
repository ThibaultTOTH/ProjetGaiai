#!/usr/bin/env python3
"""
Scientific Benchmark: DualGaiaAgent (Supervised / AlphaZero) vs Random Bots.
Evaluates win rate, victory points, action distribution, and game length
over N complete games in NativeGaiaEnv.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn.functional as F

SCRAPER_DIR = str(Path(__file__).resolve().parent)
while sys.path and sys.path[0] == SCRAPER_DIR:
    sys.path.pop(0)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training the model"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

from environment import NativeGaiaEnv
from models import DualGaiaAgent
from config import AppConfig


def classify_action(action_id: int) -> str:
    """Classifies Gaia Project action ID into game category."""
    if 0 <= action_id < 200:
        return "Build Mine"
    elif 200 <= action_id < 400:
        return "Start Gaia Project"
    elif 400 <= action_id < 1400:
        return "Upgrade Building"
    elif 1400 <= action_id < 1406:
        return "Form Federation"
    elif 1406 <= action_id < 1412:
        return "Advance Research"
    elif 1412 <= action_id < 1422:
        return "Pass & Booster"
    elif 1422 <= action_id < 1436:
        return "Special Action"
    elif 1436 <= action_id < 1500:
        return "Power/QIC Action"
    else:
        return "Other Action"


def play_game(
    env: NativeGaiaEnv,
    agent: DualGaiaAgent,
    device: torch.device,
    agent_seat: int = 0,
    temperature: float = 0.1,
    max_steps: int = 1500,
    agent_is_random: bool = False,
) -> Dict[str, Any]:
    """Plays 1 full 4-player game. Returns stats dictionary."""
    obs, mask = env.reset()
    step_count = 0
    agent_actions_taken = []
    agent_action_categories = {}

    while not env.terminated and step_count < max_steps:
        step_count += 1
        active_player = env.current_player
        legal_actions = np.where(mask)[0]

        if len(legal_actions) == 0:
            break

        if active_player == agent_seat and not agent_is_random:
            # Agent move
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                mask_t = torch.from_numpy(mask).bool().unsqueeze(0).to(device)
                logits = agent.action_net(obs_t, mask_t).squeeze(0)

                if temperature <= 0.01:
                    chosen_action = int(torch.argmax(logits).item())
                else:
                    scaled_logits = logits / temperature
                    # Force illegal actions to -inf
                    scaled_logits = torch.where(mask_t.squeeze(0), scaled_logits, torch.tensor(-1e9, device=device))
                    probs = F.softmax(scaled_logits, dim=-1)
                    chosen_action = int(torch.multinomial(probs, 1).item())

            agent_actions_taken.append(chosen_action)
            cat = classify_action(chosen_action)
            agent_action_categories[cat] = agent_action_categories.get(cat, 0) + 1
        else:
            # Random bot move
            chosen_action = int(np.random.choice(legal_actions))

        res = env.step(chosen_action)
        obs = res.obs
        mask = res.action_mask

    all_vps = env.get_all_vps()
    agent_vp = float(all_vps[agent_seat])
    opp_vps = [float(all_vps[i]) for i in range(len(all_vps)) if i != agent_seat]

    # Rank: 1 = 1st place, 4 = 4th place
    ranks = np.argsort(-all_vps)
    agent_rank = int(np.where(ranks == agent_seat)[0][0]) + 1
    won = (agent_rank == 1)

    return {
        "agent_vp": agent_vp,
        "opp_vps": opp_vps,
        "all_vps": all_vps.tolist(),
        "agent_rank": agent_rank,
        "won": won,
        "rounds": env.round,
        "total_steps": step_count,
        "agent_steps": len(agent_actions_taken),
        "categories": agent_action_categories,
    }


def evaluate(
    checkpoint_path: Path,
    num_games: int = 10,
    device_name: str = "auto",
    temperature: float = 0.1,
    run_control: bool = True,
):
    print("=" * 76)
    print("  🌌 BENCHMARK: DualGaiaAgent vs Random Opponents")
    print("=" * 76)

    device = torch.device(
        "cuda" if (device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available())) else "cpu"
    )
    print(f"  Device: {device}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"  Loading Checkpoint: {checkpoint_path.name}...")
    agent = DualGaiaAgent().to(device)
    meta = agent.load_checkpoint(str(checkpoint_path), device=device)
    agent.eval()
    print(f"  Agent Architecture: {agent.config.block_type.upper()} {agent.config.policy_hidden_layers} (GNN={agent.config.use_gnn_map})")

    env = NativeGaiaEnv(players=4, max_rounds=6, seed=42)

    # 1. Evaluate Pretrained Agent
    print(f"\n--- [1/2] Running {num_games} Games with Pretrained Agent (Player 0) ---")
    agent_results = []
    t0 = time.time()

    for g in range(1, num_games + 1):
        env_seed = 1000 + g * 37
        env.reset(seed=env_seed)
        res = play_game(env, agent, device, agent_seat=0, temperature=temperature, agent_is_random=False)
        agent_results.append(res)
        print(
            f"  Game {g:2d}/{num_games:2d} | Agent: {res['agent_vp']:5.1f} VP (Rank #{res['agent_rank']}) | "
            f"Opps: [{', '.join(f'{v:4.1f}' for v in res['opp_vps'])}] | "
            f"Rounds: {res['rounds']} | Steps: {res['agent_steps']}"
        )

    t_elapsed = time.time() - t0

    # 2. Control Group (Random Bot as Player 0)
    control_results = []
    if run_control:
        print(f"\n--- [2/2] Running {num_games} Control Games (Random Bot as Player 0) ---")
        for g in range(1, num_games + 1):
            env_seed = 5000 + g * 37
            env.reset(seed=env_seed)
            res = play_game(env, agent, device, agent_seat=0, temperature=temperature, agent_is_random=True)
            control_results.append(res)
            print(
                f"  Control {g:2d}/{num_games:2d} | Random P0: {res['agent_vp']:5.1f} VP (Rank #{res['agent_rank']}) | "
                f"Opps: [{', '.join(f'{v:4.1f}' for v in res['opp_vps'])}] | "
                f"Rounds: {res['rounds']}"
            )

    # Summary Statistics
    agent_vps = [r["agent_vp"] for r in agent_results]
    agent_wins = sum(1 for r in agent_results if r["won"])
    agent_ranks = [r["agent_rank"] for r in agent_results]

    all_opp_vps = [v for r in agent_results for v in r["opp_vps"]]

    # Category Breakdown
    cat_counts = {}
    for r in agent_results:
        for c, cnt in r["categories"].items():
            cat_counts[c] = cat_counts.get(c, 0) + cnt

    total_agent_moves = sum(cat_counts.values()) or 1

    print("\n" + "=" * 76)
    print("  📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 76)
    print(f"  🏆 Agent Win Rate:        {agent_wins}/{num_games} ({agent_wins / num_games * 100:.1f}%)")
    print(f"  🎯 Agent Average VP:      {np.mean(agent_vps):.1f} ± {np.std(agent_vps):.1f} VP (Min: {min(agent_vps):.1f}, Max: {max(agent_vps):.1f})")
    print(f"  🎲 Opponents Average VP:  {np.mean(all_opp_vps):.1f} ± {np.std(all_opp_vps):.1f} VP")
    print(f"  🥇 Agent Average Rank:    {np.mean(agent_ranks):.2f} / 4.0 (1.0 = Always 1st)")
    print(f"  ⚡ Total Time:            {t_elapsed:.1f}s ({t_elapsed / num_games:.2f}s per game)")

    if run_control:
        ctrl_vps = [r["agent_vp"] for r in control_results]
        ctrl_wins = sum(1 for r in control_results if r["won"])
        ctrl_ranks = [r["agent_rank"] for r in control_results]
        print(f"\n  📋 Control Baseline (Random Bot P0):")
        print(f"     - Win Rate:       {ctrl_wins}/{num_games} ({ctrl_wins / num_games * 100:.1f}%)")
        print(f"     - Average VP:     {np.mean(ctrl_vps):.1f} ± {np.std(ctrl_vps):.1f} VP")
        print(f"     - Average Rank:   {np.mean(ctrl_ranks):.2f} / 4.0")
        vp_gain = np.mean(agent_vps) - np.mean(ctrl_vps)
        print(f"  🚀 Agent VP Advantage over Random: +{vp_gain:.1f} VP")

    print("\n  🛠️ Agent Action Diversity Distribution:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        pct = (cnt / total_agent_moves) * 100
        bar = "█" * int(pct / 2.5)
        print(f"     {cat:<24} : {cnt:4d} ({pct:5.1f}%) {bar}")
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gaia Project agent vs Random Bots")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINTS_DIR / "gaia_supervised_pretrained.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument("--games", type=int, default=10, help="Number of games to evaluate")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto/cuda/cpu)")
    parser.add_argument("--temp", type=float, default=0.1, help="Softmax sampling temperature (0 for argmax)")
    parser.add_argument("--no-control", action="store_true", help="Skip random bot control games")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=Path(args.checkpoint),
        num_games=args.games,
        device_name=args.device,
        temperature=args.temp,
        run_control=not args.no_control,
    )


if __name__ == "__main__":
    main()
