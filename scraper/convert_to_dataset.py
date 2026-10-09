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

TECH_TILE_MAP = {
    "terra": 0, "o3": 0,
    "nav": 1, "k1": 1,
    "int": 2, "q1": 2,
    "gaia": 3, "c4": 3,
    "eco": 4, "o1": 4,
    "sci": 5, "k1o1": 5,
    "free1": 6, "vp": 6,
    "free2": 7, "pw": 7,
    "free3": 8, "charge": 8,
}

# Offsets matching action_space.rs
A_BUILD_MINE_OFFSET = 0
A_START_GAIA_OFFSET = 200
A_UPGRADE_TS_OFFSET = 400
A_UPGRADE_LAB_OFFSET = 600
A_UPGRADE_PI_OFFSET = 800
A_UPGRADE_AC1_OFFSET = 1000
A_UPGRADE_AC2_OFFSET = 1200
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


def parse_tech_action(move_str: str) -> Optional[int]:
    """Extracts a ClaimTechTile discrete action index (1444..1497) from a move string."""
    m = re.sub(r'\(.*?\)', '', move_str).strip()
    match = re.search(r'tech\s+([a-zA-Z0-9]+)', m)
    if not match:
        return None
    tile_str = match.group(1).lower()
    tech_idx = TECH_TILE_MAP.get(tile_str, 0)

    field_match = re.search(r'up\s+([a-zA-Z]+)', m)
    field_str = field_match.group(1).lower() if field_match else "terra"
    field_idx = RESEARCH_FIELDS.get(field_str, 0)

    return A_CLAIM_TECH_OFFSET + tech_idx * 6 + field_idx


def parse_hex_coord(coord_str: str) -> Optional[int]:
    """
    Parses a BGS hex coordinate like '4A6', '10B2', '1A0' into 0..189 spatial hex index.
    Sectors: 1..10 (19 hexes each)
    Ring 'A': 0..11 (outer ring, 12 hexes)
    Ring 'B': 0..5 (inner ring, 6 hexes)
    Ring 'C': 0 (center hex)
    """
    match = re.search(r'(\d{1,2})([ABCabc])(\d{1,2})', coord_str)
    if not match:
        return None
    sector = int(match.group(1))
    ring = match.group(2).upper()
    sub_idx = int(match.group(3))

    if not (1 <= sector <= 10):
        return None

    sector_base = (sector - 1) * 19
    if ring == 'A':
        if not (0 <= sub_idx <= 11):
            return None
        return sector_base + sub_idx
    elif ring == 'B':
        if not (0 <= sub_idx <= 5):
            return None
        return sector_base + 12 + sub_idx
    elif ring == 'C':
        return sector_base + 18
    return None


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

    # 1. Pass / Booster selection: "pass booster<N>" or "booster booster<N>"
    if "pass" in m or "booster" in m:
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
        if "swap-pi" in m or "ambas" in m:
            return A_SPECIAL_ACTION_OFFSET + 0
        elif "down-lab" in m or "downgrade" in m:
            return A_SPECIAL_ACTION_OFFSET + 1
        elif "lowest" in m or "bescods" in m:
            return A_SPECIAL_ACTION_OFFSET + 2
        elif "space-station" in m or "sp " in m or "station" in m:
            return A_SPECIAL_ACTION_OFFSET + 3
        elif "terraform" in m or "step" in m:
            return A_SPECIAL_ACTION_OFFSET + 4
        elif "tech9" in m or "charge" in m:
            return A_SPECIAL_ACTION_OFFSET + 5
        elif "adv3" in m or "qic" in m:
            return A_SPECIAL_ACTION_OFFSET + 6
        elif "3o" in m or "adv11" in m:
            return A_SPECIAL_ACTION_OFFSET + 7
        elif "3k" in m or "adv13" in m:
            return A_SPECIAL_ACTION_OFFSET + 8
        elif "range" in m:
            return A_SPECIAL_ACTION_OFFSET + 9
        return A_SPECIAL_ACTION_OFFSET + 0

    # 7. Start Gaia project: "build gf <coord>"
    if "build gf" in m:
        coord_m = re.search(r'build gf\s+([0-9]{1,2}[A-Za-z][0-9]{1,2})', m)
        hex_idx = parse_hex_coord(coord_m.group(1)) if coord_m else None
        return A_START_GAIA_OFFSET + (hex_idx if hex_idx is not None else 0)

    # 8. Upgrade building: "build ts/lab/PI/ac1/ac2 <coord>"
    if "build ts" in m:
        coord_m = re.search(r'build ts\s+([0-9]{1,2}[A-Za-z][0-9]{1,2})', m)
        hex_idx = parse_hex_coord(coord_m.group(1)) if coord_m else None
        return A_UPGRADE_TS_OFFSET + (hex_idx if hex_idx is not None else 0)
    if "build lab" in m:
        coord_m = re.search(r'build lab\s+([0-9]{1,2}[A-Za-z][0-9]{1,2})', m)
        hex_idx = parse_hex_coord(coord_m.group(1)) if coord_m else None
        return A_UPGRADE_LAB_OFFSET + (hex_idx if hex_idx is not None else 0)
    if "build PI" in m or "build pi" in m:
        coord_m = re.search(r'build [Pp][Ii]\s+([0-9]{1,2}[A-Za-z][0-9]{1,2})', m)
        hex_idx = parse_hex_coord(coord_m.group(1)) if coord_m else None
        return A_UPGRADE_PI_OFFSET + (hex_idx if hex_idx is not None else 0)
    if "build ac1" in m:
        coord_m = re.search(r'build ac1\s+([0-9]{1,2}[A-Za-z][0-9]{1,2})', m)
        hex_idx = parse_hex_coord(coord_m.group(1)) if coord_m else None
        return A_UPGRADE_AC1_OFFSET + (hex_idx if hex_idx is not None else 0)
    if "build ac2" in m:
        coord_m = re.search(r'build ac2\s+([0-9]{1,2}[A-Za-z][0-9]{1,2})', m)
        hex_idx = parse_hex_coord(coord_m.group(1)) if coord_m else None
        return A_UPGRADE_AC2_OFFSET + (hex_idx if hex_idx is not None else 0)

    # 9. Build Mine: "build m <coord>"
    if "build m" in m:
        coord_m = re.search(r'build m\s+([0-9]{1,2}[A-Za-z][0-9]{1,2})', m)
        hex_idx = parse_hex_coord(coord_m.group(1)) if coord_m else None
        return A_BUILD_MINE_OFFSET + (hex_idx if hex_idx is not None else 0)

    # 10. Free action: "burn" or "spend"
    if "burn" in m or "spend" in m:
        return A_FREE_ACTION_OFFSET + 0

    # 11. Standalone tech claim: "tech ..."
    if "tech " in m and not any(k in m for k in ["build lab", "build ac"]):
        tech_act = parse_tech_action(m)
        if tech_act is not None:
            return tech_act

    return None


def parse_move_actions(move_str: str) -> List[int]:
    """Parses all actions from a move string, including compound moves (e.g. build lab + tech claim)."""
    actions = []
    primary = parse_action_index(move_str)
    if primary is not None:
        actions.append(primary)

    # If the move included building a Lab/Academy AND claiming a tech tile:
    if ("build lab" in move_str or "build ac" in move_str) and "tech " in move_str:
        tech_act = parse_tech_action(move_str)
        if tech_act is not None:
            actions.append(tech_act)

    return actions


class GaiaExpertDataset(Dataset):
    """PyTorch Dataset holding parsed expert human game trajectories with full observation tensors."""

    def __init__(self, pt_file: Path):
        self.data = torch.load(pt_file, weights_only=False)
        self.actions = self.data["actions"]
        self.values = self.data["values"]
        self.factions = self.data["factions"]
        self.has_obs = "observations" in self.data
        if self.has_obs:
            self.observations = self.data["observations"]
            self.action_masks = self.data["action_masks"]
        else:
            logger.warning(
                "Dataset has no observation tensors. Regenerate with the Rust "
                "simulator to enable supervised training."
            )

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int):
        if self.has_obs:
            return (
                self.observations[idx],
                self.action_masks[idx],
                self.actions[idx],
                self.values[idx],
                self.factions[idx],
            )
        return self.actions[idx], self.values[idx], self.factions[idx]


def build_expert_dataset(
    raw_dir: Path = RAW_GAMES_DIR,
    out_dir: Path = DATASET_DIR,
    min_winning_score: int = 140,
) -> Path:
    """Parses all valid scraped games and builds a compact dataset file.

    IMPORTANT: Replays each game through the Rust simulator to capture
    observation tensors at every step. Without obs tensors, the neural
    network has no input for training.
    """
    import sys
    training_dir = Path(__file__).resolve().parent.parent / "training the model"
    if str(training_dir) not in sys.path:
        sys.path.insert(0, str(training_dir))

    from environment import NativeGaiaEnv, compute_competitive_value

    out_dir.mkdir(parents=True, exist_ok=True)
    game_files = list(raw_dir.glob("*.json"))
    logger.info(f"Processing {len(game_files)} game files from {raw_dir}...")

    all_obs = []
    all_action_masks = []
    all_actions = []
    all_values = []
    all_factions = []

    parsed_games = 0
    skipped_games = 0
    replay_failures = 0

    # Check if simulator is available for replay-based obs extraction
    has_simulator = NativeGaiaEnv.is_available()
    if not has_simulator:
        logger.warning(
            "Rust simulator NOT available. Falling back to action-only dataset "
            "(no observation tensors — supervised training will NOT work until "
            "the simulator is compiled)."
        )

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
            faction_name = p.get("faction")
            if faction_name:
                scores_by_faction[faction_name] = p.get("score", 0)

        max_score = max(scores_by_faction.values()) if scores_by_faction else 0
        if max_score < min_winning_score:
            skipped_games += 1
            continue

        # Parse all moves into action indices first
        moves = game.get("data", {}).get("moveHistory", [])
        game_actions = []  # list of (faction_name, action_index)
        for m in moves:
            parts = m.split()
            if not parts:
                continue
            actor_str = parts[0].lower()
            if actor_str not in FACTION_TO_ID:
                continue
            action_indices = parse_move_actions(m)
            for action_idx in action_indices:
                game_actions.append((actor_str, action_idx))

        if not game_actions:
            skipped_games += 1
            continue

        # Compute competitive value targets for each player
        num_players = len(scores_by_faction)
        faction_names = list(scores_by_faction.keys())
        raw_vps = [float(scores_by_faction[fn]) for fn in faction_names]

        # If simulator is available, replay the game to get observations
        if has_simulator:
            try:
                env = NativeGaiaEnv(
                    players=min(4, max(2, num_players)),
                    seed=hash(gf.name) % 1000000,
                    egocentric=True,
                )
                obs, mask = env.reset()

                # Map factions to seats
                faction_to_seat = {}
                for seat_idx, fn in enumerate(faction_names[:env.num_players]):
                    faction_id = FACTION_TO_ID.get(fn, seat_idx)
                    env.set_player_faction(seat_idx, faction_id)
                    faction_to_seat[fn] = seat_idx
                obs = env._get_obs()

                for actor_faction, action_idx in game_actions:
                    if env.terminated:
                        break

                    seat = faction_to_seat.get(actor_faction, 0)
                    faction_id = FACTION_TO_ID.get(actor_faction, 0)

                    # Capture observation and mask BEFORE the action
                    current_obs = env._get_obs().copy()
                    current_mask = env.get_action_mask().copy()

                    # Compute competitive value for this player
                    value = compute_competitive_value(raw_vps, seat)

                    all_obs.append(current_obs)
                    all_action_masks.append(current_mask)
                    all_actions.append(action_idx)
                    all_values.append(value)
                    all_factions.append(faction_id)

                    # Execute action in simulator
                    step_res = env.step(action_idx)

                del env  # Free native memory

            except Exception as e:
                replay_failures += 1
                if replay_failures <= 5:
                    logger.warning(f"Replay failed for {gf.name}: {e}")
                # Fallback: store without obs (action-only)
                for actor_faction, action_idx in game_actions:
                    faction_id = FACTION_TO_ID.get(actor_faction, 0)
                    seat = faction_names.index(actor_faction) if actor_faction in faction_names else 0
                    value = compute_competitive_value(raw_vps, seat)
                    all_actions.append(action_idx)
                    all_values.append(value)
                    all_factions.append(faction_id)
        else:
            # No simulator: store action-only data with competitive values
            for actor_faction, action_idx in game_actions:
                faction_id = FACTION_TO_ID.get(actor_faction, 0)
                seat = faction_names.index(actor_faction) if actor_faction in faction_names else 0
                value = compute_competitive_value(raw_vps, seat)
                all_actions.append(action_idx)
                all_values.append(value)
                all_factions.append(faction_id)

        parsed_games += 1

    logger.info(
        f"Parsed {parsed_games} games ({skipped_games} skipped, {replay_failures} replay failures). "
        f"Total expert moves extracted: {len(all_actions):,}"
    )
    if all_obs:
        logger.info(f"Observation tensors captured: {len(all_obs):,} (obs_dim={all_obs[0].shape[0]})")
    else:
        logger.warning("NO observation tensors captured. Compile the Rust engine to enable full dataset generation.")

    dataset_dict = {
        "actions": torch.tensor(all_actions, dtype=torch.long),
        "values": torch.tensor(all_values, dtype=torch.float32),
        "factions": torch.tensor(all_factions, dtype=torch.long),
        "num_games": parsed_games,
        "num_samples": len(all_actions),
    }

    # Add observation tensors and masks if captured via simulator replay
    if all_obs:
        dataset_dict["observations"] = torch.from_numpy(np.stack(all_obs))
        dataset_dict["action_masks"] = torch.from_numpy(np.stack(all_action_masks).astype(np.bool_))

    out_file = out_dir / "gaia_expert_dataset.pt"
    torch.save(dataset_dict, out_file)
    logger.info(f"Saved dataset to {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Convert scraped games to training dataset")
    parser.add_argument("--min-vp", type=int, default=150, help="Minimum winning score threshold")
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
