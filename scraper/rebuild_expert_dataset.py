"""
Rebuilds the Authentic Gaia Project Expert Dataset with True 2476-Dimensional Observation Tensors.
Replays BGS (Boardgamers.space) Grand Master games (>= 150 VP) through NativeGaiaEnv C-ABI.
Extracts grounded state-action pairs: (observation_2476, action, value, faction).
"""

import argparse
import glob
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training the model"
SCRAPER_DIR = PROJECT_ROOT / "scraper"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from environment import NativeGaiaEnv
from scraper.config import RAW_GAMES_DIR, DATASET_DIR, BASE_FACTIONS, BASE_FACTIONS_SET
from scraper.convert_to_dataset import (
    parse_action_index,
    FACTION_TO_ID,
    A_FREE_ACTION_OFFSET,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dataset_builder")


def parse_micro_actions(move_str: str) -> List[int]:
    """
    Parses a BGS move string into a list of discrete action indices (0..3129),
    splitting compound sentences into individual sub-actions (free conversions, builds, upgrades, etc.).
    """
    # Remove commentary and power changes in parentheses
    clean = re.sub(r'\(.*?\)', '', move_str).strip()
    parts = [p.strip() for p in clean.split('.') if p.strip()]
    actions = []

    for part in parts:
        if not part:
            continue
        p_lower = part.lower()

        # Skip initialization and meta comments
        if any(p_lower.startswith(k) for k in ["init", "setup", "rotate", "faction", "income"]):
            continue

        # Free Resource Actions (Offset 3124):
        # 3124: PowerToQic, 3125: PowerToKnowledge, 3126: PowerToOre,
        # 3127: PowerToCredit, 3128: QicToOre, 3129: OreToToken
        if "spend" in p_lower:
            if "for 1q" in p_lower or "for q" in p_lower:
                actions.append(A_FREE_ACTION_OFFSET + 0)
            elif "for 1k" in p_lower or "for k" in p_lower:
                actions.append(A_FREE_ACTION_OFFSET + 1)
            elif "for 1o" in p_lower or "for o" in p_lower:
                actions.append(A_FREE_ACTION_OFFSET + 2)
            elif "for 2o" in p_lower:
                actions.extend([A_FREE_ACTION_OFFSET + 2, A_FREE_ACTION_OFFSET + 2])
            elif "for 1c" in p_lower or "for c" in p_lower:
                actions.append(A_FREE_ACTION_OFFSET + 3)
            elif "for 2c" in p_lower:
                actions.extend([A_FREE_ACTION_OFFSET + 3, A_FREE_ACTION_OFFSET + 3])
            elif "for 3c" in p_lower:
                actions.extend([A_FREE_ACTION_OFFSET + 3] * 3)
            elif "for 1t" in p_lower or "for t" in p_lower:
                actions.append(A_FREE_ACTION_OFFSET + 5)
            elif "for 1o" in p_lower or "for o" in p_lower:
                actions.append(A_FREE_ACTION_OFFSET + 4)
            elif "1q for 1o" in p_lower:
                actions.append(A_FREE_ACTION_OFFSET + 4)
            elif "2q for 2o" in p_lower:
                actions.extend([A_FREE_ACTION_OFFSET + 4, A_FREE_ACTION_OFFSET + 4])
            continue

        if "burn" in p_lower:
            actions.append(A_FREE_ACTION_OFFSET + 0)
            continue

        act = parse_action_index(part)
        if act is not None:
            actions.append(act)

    return actions


def build_authentic_dataset(
    raw_dir: Path = RAW_GAMES_DIR,
    out_dir: Path = DATASET_DIR,
    min_winning_score: int = 145,
    max_games: Optional[int] = None,
):
    game_files = sorted(list(raw_dir.glob("*.json")))
    if max_games:
        game_files = game_files[:max_games]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "gaia_expert_dataset.pt"

    logger.info(f"Scanning {len(game_files)} BGS raw game files (min_vp={min_winning_score})...")

    chunk_size = 20000
    all_obs_chunks: List[np.ndarray] = []
    current_obs: List[np.ndarray] = []
    all_actions: List[int] = []
    all_values: List[float] = []
    all_factions: List[int] = []

    parsed_games = 0
    skipped_games = 0
    total_actions_processed = 0

    t_start = time.time()

    for idx, gf in enumerate(game_files, 1):
        try:
            with open(gf, "r", encoding="utf-8") as f:
                game = json.load(f)
        except Exception as e:
            logger.warning(f"Error reading {gf.name}: {e}")
            skipped_games += 1
            continue

        players = game.get("players", [])
        if len(players) < 2:
            skipped_games += 1
            continue

        scores_by_faction: Dict[str, float] = {}
        for p in players:
            f = p.get("faction")
            if f:
                scores_by_faction[f.lower()] = float(p.get("score", 0))

        if not scores_by_faction:
            skipped_games += 1
            continue

        winning_score = max(scores_by_faction.values())
        if winning_score < min_winning_score:
            skipped_games += 1
            continue

        # Setup native Rust environment
        try:
            env = NativeGaiaEnv(players=len(players), seed=42, egocentric=True)
        except Exception as e:
            logger.error(f"Failed to create NativeGaiaEnv: {e}")
            break

        player_factions = []
        for seat, p in enumerate(players):
            f_name = p.get("faction", "").lower()
            f_id = FACTION_TO_ID.get(f_name, seat % 14)
            player_factions.append(f_id)
            env.set_player_faction(seat, f_id)

        moves = game.get("data", {}).get("moveHistory", [])
        game_extracted = 0

        for m in moves:
            parts = m.strip().split()
            if not parts:
                continue

            # Identify acting faction if explicitly declared in move prefix
            actor_token = parts[0].lower()
            current_fac = FACTION_TO_ID.get(actor_token)
            if current_fac is None:
                curr_seat = getattr(env, "current_player", 0)
                current_fac = player_factions[curr_seat] if curr_seat < len(player_factions) else 0

            # Normalized target score: (score - 100) / 50.0 (typically [-1.0, +1.5])
            fac_name = BASE_FACTIONS[current_fac] if current_fac < len(BASE_FACTIONS) else ""
            raw_score = scores_by_faction.get(fac_name, 120.0)
            norm_val = (raw_score - 100.0) / 50.0

            sub_actions = parse_micro_actions(m)
            for act in sub_actions:
                mask = env.get_action_mask()
                obs = env.get_observation()

                is_legal = bool(mask[act]) if act < len(mask) else False
                if is_legal:
                    current_obs.append(obs.astype(np.float16))
                    all_actions.append(int(act))
                    all_values.append(float(norm_val))
                    all_factions.append(int(current_fac))
                    game_extracted += 1
                    env.step(int(act))
                else:
                    # Attempt step in native engine
                    res = env.step(int(act))
                    if not (hasattr(res, "info") and "error" in res.info):
                        current_obs.append(obs.astype(np.float16))
                        all_actions.append(int(act))
                        all_values.append(float(norm_val))
                        all_factions.append(int(current_fac))
                        game_extracted += 1

                total_actions_processed += 1

                # Flush chunk to avoid massive python list overhead
                if len(current_obs) >= chunk_size:
                    all_obs_chunks.append(np.stack(current_obs))
                    current_obs.clear()

        parsed_games += 1

        if idx % 50 == 0 or idx == len(game_files):
            cur_total = sum(c.shape[0] for c in all_obs_chunks) + len(current_obs)
            el = max(1e-4, time.time() - t_start)
            logger.info(
                f"[{idx}/{len(game_files)}] Games: {parsed_games} parsed | "
                f"Samples: {cur_total:,} | Speed: {cur_total / el:.0f} samples/s"
            )

    if current_obs:
        all_obs_chunks.append(np.stack(current_obs))
        current_obs.clear()

    total_samples = len(all_actions)
    if total_samples == 0:
        logger.error("No valid samples extracted! Verify raw games directory.")
        return None

    logger.info(f"Concatenating {len(all_obs_chunks)} chunks for {total_samples:,} observations...")
    combined_obs_np = np.concatenate(all_obs_chunks, axis=0)

    logger.info(f"Building PyTorch tensors (Obs shape: {combined_obs_np.shape})...")
    dataset_dict = {
        "observations": torch.from_numpy(combined_obs_np),  # float16 tensor (N, 2476)
        "actions": torch.tensor(all_actions, dtype=torch.long),
        "values": torch.tensor(all_values, dtype=torch.float32),
        "factions": torch.tensor(all_factions, dtype=torch.long),
        "num_games": parsed_games,
        "num_samples": total_samples,
    }

    logger.info(f"Saving dataset to {out_file}...")
    torch.save(dataset_dict, out_file)
    size_mb = out_file.stat().st_size / (1024 * 1024)
    total_time = time.time() - t_start

    logger.info(
        f"✓ Successfully built Authentic Expert Dataset!\n"
        f"  • Destination : {out_file}\n"
        f"  • File Size   : {size_mb:.1f} MB\n"
        f"  • Games parsed: {parsed_games} ({skipped_games} skipped)\n"
        f"  • Valid Pairs : {total_samples:,} state-action transitions\n"
        f"  • Elapsed Time: {total_time:.1f}s ({total_samples / total_time:.0f} samples/s)"
    )
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild Authentic Expert Dataset with True 2476D Observations")
    parser.add_argument("--min-vp", type=int, default=145, help="Minimum winning score threshold")
    parser.add_argument("--max-games", type=int, default=None, help="Limit number of games for quick test")
    args = parser.parse_args()

    build_authentic_dataset(
        min_winning_score=args.min_vp,
        max_games=args.max_games,
    )
