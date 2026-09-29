"""
Converts raw scraped BGS Gaia Project games into PyTorch Supervised Training Datasets.
Extracts expert move sequences, observations, and outcome targets for Behavioral Cloning.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import (
    BASE_FACTIONS,
    BASE_FACTIONS_SET,
    RAW_GAMES_DIR,
    DATASET_DIR,
    METADATA_FILE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("converter")

# Map faction names to index 0..13
FACTION_TO_ID = {f: idx for idx, f in enumerate(BASE_FACTIONS)}

# Research fields
RESEARCH_FIELDS = {
    "terra": 0,
    "nav": 1,
    "int": 2,
    "gaia": 3,
    "eco": 4,
    "sci": 5,
}

# Offsets matching action_space.rs
A_BUILD_MINE_OFFSET = 0
A_START_GAIA_OFFSET = 200
A_UPGRADE_OFFSET = 400
A_FEDERATION_OFFSET = 1400
A_ADVANCE_RESEARCH_OFFSET = 1406
A_PASS_OFFSET = 1412
A_CHARGE_POWER_OFFSET = 1422
A_BOARD_ACTION_OFFSET = 1424
A_SPECIAL_ACTION_OFFSET = 1434
A_CLAIM_TECH_OFFSET = 1444
A_CLAIM_ADV_TECH_OFFSET = 1498
A_EXPLORE_SPACESHIP_OFFSET = 2308
A_SPACESHIP_BOARD_OFFSET = 3108
A_FREE_ACTION_OFFSET = 3124
ACTION_DIM = 3130


def parse_action_index(move_str: str) -> Optional[int]:
    """
    Parses a BGS move string into our discrete action space index (0..3129).
    Returns None if the move is an automatic transition (income, map rotate, setup).
    """
    m = re.sub(r'\(.*?\)', '', move_str).strip()
    parts = m.split()
    if not parts:
        return None

    # Skip engine setup / automatic income
    if parts[0] in ["init", "setup"] or "income" in m or "rotate" in m:
        return None
    if len(parts) >= 2 and parts[1] == "faction":
        return None

    # 1. Pass: "pass booster<N>"
    if "pass" in m:
        booster_match = re.search(r'booster(\d+)', m)
        booster_num = int(booster_match.group(1)) if booster_match else 0
        booster_idx = max(0, min(9, booster_num - 1))
        return A_PASS_OFFSET + booster_idx

    # 2. Charge power / Decline: "charge <N>pw" or "decline"
    if "charge" in m:
        return A_CHARGE_POWER_OFFSET + 1
    if "decline" in m:
        return A_CHARGE_POWER_OFFSET + 0

    # 3. Advance Research: "up <field>"
    if "up " in m:
        for fname, fid in RESEARCH_FIELDS.items():
            if f"up {fname}" in m:
                return A_ADVANCE_RESEARCH_OFFSET + fid
        return A_ADVANCE_RESEARCH_OFFSET + 0

    # 4. Board Actions: "action power<N>" or "action qic<N>"
    if "action power" in m:
        p_match = re.search(r'action power(\d+)', m)
        p_idx = int(p_match.group(1)) - 1 if p_match else 0
        return A_BOARD_ACTION_OFFSET + max(0, min(6, p_idx))
    if "action qic" in m:
        q_match = re.search(r'action qic(\d+)', m)
        q_idx = int(q_match.group(1)) - 1 if q_match else 0
        return A_BOARD_ACTION_OFFSET + 7 + max(0, min(2, q_idx))

    # 5. Form Federation: "federation fed<N>"
    if "federation" in m:
        fed_match = re.search(r'fed(\d+)', m)
        fed_idx = int(fed_match.group(1)) - 1 if fed_match else 0
        return A_FEDERATION_OFFSET + max(0, min(5, fed_idx))

    # 6. Special action: "special ..."
    if "special" in m:
        return A_SPECIAL_ACTION_OFFSET + 0

    # 7. Start Gaia project: "build gf <coord>"
    if "build gf" in m:
        return A_START_GAIA_OFFSET + 0

    # 8. Upgrade building: "build ts/lab/PI/ac1/ac2"
    if "build ts" in m:
        return A_UPGRADE_OFFSET + 0
    if "build lab" in m:
        return A_UPGRADE_OFFSET + 1
    if "build PI" in m:
        return A_UPGRADE_OFFSET + 2
    if "build ac1" in m:
        return A_UPGRADE_OFFSET + 3
    if "build ac2" in m:
        return A_UPGRADE_OFFSET + 4

    # 9. Build Mine: "build m <coord>"
    if "build m" in m:
        return A_BUILD_MINE_OFFSET + 0

    # 10. Free action: "burn" or "spend"
    if "burn" in m or "spend" in m:
        return A_FREE_ACTION_OFFSET + 0

    return None


class GaiaExpertDataset(Dataset):
    """PyTorch Dataset holding parsed expert human game trajectories."""

    def __init__(self, pt_file: Path):
        self.data = torch.load(pt_file, weights_only=True)
        self.actions = self.data["actions"]
        self.values = self.data["values"]
        self.factions = self.data["factions"]

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.actions[idx], self.values[idx], self.factions[idx]


def build_expert_dataset(
    raw_dir: Path = RAW_GAMES_DIR,
    out_dir: Path = DATASET_DIR,
    min_winning_score: int = 140,
) -> Path:
    """Parses all valid scraped games and builds a compact dataset file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    game_files = list(raw_dir.glob("*.json"))
    logger.info(f"Processing {len(game_files)} game files from {raw_dir}...")

    all_actions = []
    all_values = []
    all_factions = []

    parsed_games = 0
    skipped_games = 0

    for gf in game_files:
        try:
            with open(gf, "r", encoding="utf-8") as f:
                game = json.load(f)
        except Exception as e:
            logger.warning(f"Error reading {gf}: {e}")
            skipped_games += 1
            continue

        players = game.get("players", [])
        scores_by_faction = {}
        for p in players:
            f = p.get("faction")
            if f:
                scores_by_faction[f] = p.get("score", 0)

        max_score = max(scores_by_faction.values()) if scores_by_faction else 0
        if max_score < min_winning_score:
            skipped_games += 1
            continue

        moves = game.get("data", {}).get("moveHistory", [])
        for m in moves:
            parts = m.split()
            if not parts:
                continue

            actor_str = parts[0].lower()
            faction_id = FACTION_TO_ID.get(actor_str)

            action_idx = parse_action_index(m)
            if action_idx is not None and faction_id is not None:
                final_score = scores_by_faction.get(actor_str, 100)
                # Normalize VP to roughly [-1, 1] range: (score - 100) / 50
                norm_value = (float(final_score) - 100.0) / 50.0

                all_actions.append(action_idx)
                all_values.append(norm_value)
                all_factions.append(faction_id)

        parsed_games += 1

    logger.info(
        f"Parsed {parsed_games} games ({skipped_games} skipped). "
        f"Total expert moves extracted: {len(all_actions):,}"
    )

    dataset_dict = {
        "actions": torch.tensor(all_actions, dtype=torch.long),
        "values": torch.tensor(all_values, dtype=torch.float32),
        "factions": torch.tensor(all_factions, dtype=torch.long),
        "num_games": parsed_games,
        "num_samples": len(all_actions),
    }

    out_file = out_dir / "gaia_expert_dataset.pt"
    torch.save(dataset_dict, out_file)
    logger.info(f"Saved dataset to {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Convert scraped games to training dataset")
    parser.add_argument("--min-vp", type=int, default=140, help="Minimum winning score threshold")
    parser.add_argument("--raw-dir", type=str, default=str(RAW_GAMES_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DATASET_DIR))
    args = parser.parse_args()

    build_expert_dataset(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        min_winning_score=args.min_vp,
    )


if __name__ == "__main__":
    main()
