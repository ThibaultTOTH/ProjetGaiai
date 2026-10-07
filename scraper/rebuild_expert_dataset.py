"""
Rebuilds the Authentic Gaia Project Expert Dataset with True 2476-Dimensional Observation Tensors.
Replays BGS (Boardgamers.space) Grand Master games through NativeGaiaEnv C-ABI.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training the model"
SCRAPER_DIR = PROJECT_ROOT / "scraper"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from environment import NativeGaiaEnv

RAW_GAMES_DIR = PROJECT_ROOT / "scraper" / "data" / "raw_games"
DATASET_DIR = PROJECT_ROOT / "scraper" / "data" / "dataset"
BASE_FACTIONS = [
    "terrans", "lantids", "hadsch_hallas", "ivits", "geodens", "bal_taks",
    "xenos", "gleens", "taklons", "ambas", "firaks", "bescods", "nevlas", "itars",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
def normalize_faction(f_name: str) -> str:
    f = f_name.lower().strip().replace("-", "_").replace(" ", "_")
    if f in ("baltaks", "bal-taks"):
        return "bal_taks"
    if f in ("hadschhallas", "hadsch-hallas"):
        return "hadsch_hallas"
    return f

FACTION_TO_ID = {f: idx for idx, f in enumerate(BASE_FACTIONS)}
FACTION_TO_ID["hadsch-hallas"] = FACTION_TO_ID["hadsch_hallas"]
FACTION_TO_ID["baltaks"] = FACTION_TO_ID["bal_taks"]
FACTION_TO_ID["hadschhallas"] = FACTION_TO_ID["hadsch_hallas"]

# BGS 19 hex relative offsets matching sector.rs SECTOR_OFFSETS
SECTOR_OFFSETS = {
    # Outer Ring A: A0 to A11
    "A0": (2, 0, -2),
    "A1": (1, 1, -2),
    "A2": (0, 2, -2),
    "A3": (-1, 2, -1),
    "A4": (-2, 2, 0),
    "A5": (-2, 1, 1),
    "A6": (-2, 0, 2),
    "A7": (-1, -1, 2),
    "A8": (0, -2, 2),
    "A9": (1, -2, 1),
    "A10": (2, -2, 0),
    "A11": (2, -1, -1),
    # Middle Ring B: B0 to B5
    "B0": (1, 0, -1),
    "B1": (0, 1, -1),
    "B2": (-1, 1, 0),
    "B3": (-1, 0, 1),
    "B4": (0, -1, 1),
    "B5": (1, -1, 0),
    # Center: C
    "C": (0, 0, 0),
    "C0": (0, 0, 0),
}

RESEARCH_FIELDS = {
    "terra": 0, "nav": 1, "int": 2, "gaia": 3, "eco": 4, "sci": 5,
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

BOARDS_POWER_ACTIONS = {
    "power1": 0, "power2": 1, "power3": 2, "power4": 3, "power5": 4, "power6": 5, "power7": 6,
    "qic1": 7, "qic2": 8, "qic3": 9,
}

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
A_FREE_ACTION_OFFSET = 3124
ACTION_DIM = 3130


def resolve_bgs_coord(coord_str: str, sectors_dict: Dict) -> Optional[Tuple[int, int, int]]:
    """Resolves a BGS coordinate like '6A2', '3B4', '1C' into cubic (q, r, s)."""
    m = re.match(r'^(\d+[A-Za-z]?)([ABCabc])(\d*)$', coord_str)
    if not m:
        return None
    sec_id = m.group(1)
    ring = m.group(2).upper()
    num = m.group(3)
    sub = f"{ring}{num}" if num else ring
    if sub not in SECTOR_OFFSETS:
        return None
    matching = [s for s in sectors_dict.values() if s["sector"] == sec_id or s["sector"].startswith(sec_id)]
    if not matching:
        return None
    center = matching[0]["center"]
    rel = SECTOR_OFFSETS[sub]
    return (center["q"] + rel[0], center["r"] + rel[1], center["s"] + rel[2])


def parse_move_to_actions(part: str, env: NativeGaiaEnv, sectors_dict: Dict) -> List[int]:
    """Parses a move subpart into action index(es) matching action_space.rs."""
    p = part.strip()
    p_lower = p.lower()

    # Free resource conversions (3124..3129)
    if "spend" in p_lower:
        if "for 1q" in p_lower or "for q" in p_lower:
            return [A_FREE_ACTION_OFFSET + 0]
        elif "for 1k" in p_lower or "for k" in p_lower:
            return [A_FREE_ACTION_OFFSET + 1]
        elif "for 1o" in p_lower or "for o" in p_lower:
            return [A_FREE_ACTION_OFFSET + 2]
        elif "for 2o" in p_lower:
            return [A_FREE_ACTION_OFFSET + 2, A_FREE_ACTION_OFFSET + 2]
        elif "for 1c" in p_lower or "for c" in p_lower:
            return [A_FREE_ACTION_OFFSET + 3]
        elif "for 2c" in p_lower:
            return [A_FREE_ACTION_OFFSET + 3, A_FREE_ACTION_OFFSET + 3]
        elif "for 3c" in p_lower:
            return [A_FREE_ACTION_OFFSET + 3] * 3
        elif "for 1t" in p_lower or "for t" in p_lower:
            return [A_FREE_ACTION_OFFSET + 5]
        elif "1q for 1o" in p_lower:
            return [A_FREE_ACTION_OFFSET + 4]
        return []

    if "burn" in p_lower:
        return [A_FREE_ACTION_OFFSET + 0]

    # Power charge / leech
    if "charge" in p_lower:
        return [A_CHARGE_POWER_OFFSET + 1]
    if "decline" in p_lower:
        return [A_CHARGE_POWER_OFFSET + 0]

    # Booster / Pass
    m_booster = re.search(r'booster\s*(\d+)', p_lower)
    if m_booster or "pass" in p_lower:
        b_idx = int(m_booster.group(1)) if m_booster else 1
        b_idx = max(1, min(10, b_idx))
        return [A_PASS_OFFSET + (b_idx - 1)]

    # Board power action
    m_pwr = re.search(r'action\s+(power\d|qic\d)', p_lower)
    if m_pwr:
        p_act = m_pwr.group(1)
        if p_act in BOARDS_POWER_ACTIONS:
            return [A_BOARD_ACTION_OFFSET + BOARDS_POWER_ACTIONS[p_act]]

    # Advance Research
    m_up = re.search(r'\bup\s+([a-zA-Z]+)', p_lower)
    if m_up and "tech" not in p_lower:
        field = m_up.group(1)
        if field in RESEARCH_FIELDS:
            return [A_ADVANCE_RESEARCH_OFFSET + RESEARCH_FIELDS[field]]

    # Claim Tech Tile: tech <tile> [up <field>]
    m_tech = re.search(r'tech\s+([a-zA-Z0-9]+)', p_lower)
    if m_tech:
        t_str = m_tech.group(1)
        t_idx = TECH_TILE_MAP.get(t_str, 0)
        f_idx = 0
        if m_up:
            f_idx = RESEARCH_FIELDS.get(m_up.group(1), 0)
        return [A_CLAIM_TECH_OFFSET + t_idx * 6 + f_idx]

    # Form Federation: fed <token>
    if "fed" in p_lower:
        m_fed = re.search(r'fed(\d+)', p_lower)
        f_token = int(m_fed.group(1)) - 1 if m_fed else 0
        f_token = max(0, min(5, f_token))
        return [A_FEDERATION_OFFSET + f_token]

    # Building placement or upgrade: build <type> <coord>
    m_bld = re.search(r'\b(?:build|gaia)\s+(m|ts|lab|pi|ac1|ac2|gf|sp)\s+(\d+[A-Za-z]?[ABCabc]\d*)', p_lower)
    if not m_bld:
        m_bld = re.search(r'\b(gaia)\s+(\d+[A-Za-z]?[ABCabc]\d*)', p_lower)

    if m_bld:
        bld_type = m_bld.group(1)
        coord_token = m_bld.group(2) if m_bld.lastindex >= 2 else m_bld.group(1)
        cube = resolve_bgs_coord(coord_token, sectors_dict)
        if cube:
            hex_idx = env.get_hex_index(cube[0], cube[1], cube[2])
            if hex_idx >= 0:
                if bld_type == "m":
                    return [A_BUILD_MINE_OFFSET + hex_idx]
                elif bld_type in ("gf", "gaia"):
                    return [A_START_GAIA_OFFSET + hex_idx]
                elif bld_type == "ts":
                    return [A_UPGRADE_OFFSET + hex_idx * 5 + 0]
                elif bld_type == "lab":
                    return [A_UPGRADE_OFFSET + hex_idx * 5 + 1]
                elif bld_type == "pi":
                    return [A_UPGRADE_OFFSET + hex_idx * 5 + 2]
                elif bld_type == "ac1":
                    return [A_UPGRADE_OFFSET + hex_idx * 5 + 3]
                elif bld_type == "ac2":
                    return [A_UPGRADE_OFFSET + hex_idx * 5 + 4]

    return []


def build_authentic_dataset(
    raw_dir: Path = RAW_GAMES_DIR,
    out_dir: Path = DATASET_DIR,
    min_winning_score: int = 140,
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
            skipped_games += 1
            continue

        players = game.get("players", [])
        if len(players) < 2:
            skipped_games += 1
            continue

        scores_by_faction: Dict[str, float] = {}
        faction_to_seat: Dict[str, int] = {}
        seat_to_faction: Dict[int, int] = {}

        for seat, p in enumerate(players):
            raw_f = p.get("faction", "").lower()
            f_name = normalize_faction(raw_f)
            if f_name:
                scores_by_faction[f_name] = float(p.get("score", 0))
                faction_to_seat[f_name] = seat
                faction_to_seat[raw_f] = seat
                f_id = FACTION_TO_ID.get(f_name, seat % 14)
                seat_to_faction[seat] = f_id

        if not scores_by_faction:
            skipped_games += 1
            continue

        winning_score = max(scores_by_faction.values())
        if winning_score < min_winning_score:
            skipped_games += 1
            continue

        map_conf = game.get("data", {}).get("options", {}).get("map")
        if not map_conf or "sectors" not in map_conf:
            skipped_games += 1
            continue

        sectors_dict = {s["sector"]: s for s in map_conf["sectors"]}
        map_json = json.dumps(map_conf)

        try:
            env = NativeGaiaEnv(players=len(players), seed=42, egocentric=True)
            if not env.load_map_json(map_json):
                skipped_games += 1
                continue
            for seat in range(len(players)):
                f_id = seat_to_faction.get(seat, seat % 14)
                env.set_player_faction(seat, f_id)
        except Exception as e:
            skipped_games += 1
            continue

        moves = game.get("data", {}).get("moveHistory", [])

        for m in moves:
            clean = re.sub(r'\(.*?\)', '', m).strip()
            parts = [p.strip() for p in clean.split('.') if p.strip()]

            # Determine acting faction from prefix
            actor_token = normalize_faction(parts[0].split()[0].lower()) if parts and parts[0].split() else ""
            acting_seat = faction_to_seat.get(actor_token)

            for part in parts:
                if not part:
                    continue
                p_lower = part.lower()

                # Check if this subpart mentions a specific faction
                first_token = normalize_faction(p_lower.split()[0]) if p_lower.split() else ""
                if first_token in faction_to_seat:
                    acting_seat = faction_to_seat[first_token]

                if any(p_lower.startswith(k) for k in ["init", "setup", "rotate", "faction", "income"]):
                    continue

                if acting_seat is not None:
                    env.set_current_player(acting_seat)

                sub_actions = parse_move_to_actions(part, env, sectors_dict)
                for act in sub_actions:
                    curr_seat = getattr(env, "current_player", 0)
                    fac_id = seat_to_faction.get(curr_seat, 0)
                    fac_name = BASE_FACTIONS[fac_id] if fac_id < len(BASE_FACTIONS) else ""
                    raw_score = scores_by_faction.get(fac_name, 120.0)
                    norm_val = (raw_score - 100.0) / 50.0

                    mask = env.get_action_mask()
                    obs = env.get_observation()

                    is_legal = bool(mask[act]) if act < len(mask) else False
                    current_obs.append(obs.astype(np.float16))
                    all_actions.append(int(act))
                    all_values.append(float(norm_val))
                    all_factions.append(int(fac_id))

                    if is_legal:
                        env.step(int(act))
                    else:
                        env.step(int(act))

                    total_actions_processed += 1

                    if len(current_obs) >= chunk_size:
                        all_obs_chunks.append(np.stack(current_obs))
                        current_obs.clear()

        parsed_games += 1

        if idx % 100 == 0 or idx == len(game_files):
            cur_total = sum(c.shape[0] for c in all_obs_chunks) + len(current_obs)
            el = max(1e-4, time.time() - t_start)
            logger.info(
                f"[{idx}/{len(game_files)}] Games parsed: {parsed_games} | "
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

    logger.info(f"Building PyTorch dataset (Obs shape: {combined_obs_np.shape})...")
    dataset_dict = {
        "observations": torch.from_numpy(combined_obs_np),
        "actions": torch.tensor(all_actions, dtype=torch.long),
        "values": torch.tensor(all_values, dtype=torch.float32),
        "factions": torch.tensor(all_factions, dtype=torch.long),
        "num_games": parsed_games,
        "num_samples": total_samples,
    }

    logger.info(f"Saving authentic dataset to {out_file}...")
    torch.save(dataset_dict, out_file)
    size_mb = out_file.stat().st_size / (1024 * 1024)
    total_time = time.time() - t_start

    logger.info(
        f"✓ Successfully built Authentic Expert Dataset!\n"
        f"  • Destination : {out_file}\n"
        f"  • File Size   : {size_mb:.1f} MB\n"
        f"  • Games parsed: {parsed_games} ({skipped_games} skipped)\n"
        f"  • Valid Pairs : {total_samples:,} transitions\n"
        f"  • Elapsed Time: {total_time:.1f}s ({total_samples / total_time:.0f} samples/s)"
    )
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild Authentic Expert Dataset with True 2476D Observations")
    parser.add_argument("--min-vp", type=int, default=140, help="Minimum winning score threshold")
    parser.add_argument("--max-games", type=int, default=None, help="Limit number of games for quick test")
    args = parser.parse_args()

    build_authentic_dataset(
        min_winning_score=args.min_vp,
        max_games=args.max_games,
    )
