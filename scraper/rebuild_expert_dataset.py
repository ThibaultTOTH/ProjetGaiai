"""
Rebuilds the Authentic Gaia Project Expert Dataset with True 2476-Dimensional Observation Tensors.
Replays BGS (Boardgamers.space) Grand Master games through NativeGaiaEnv C-ABI in parallel.
Extracts strictly mask-legal transitions ONLY from Grandmaster players (score >= min_gm_vp)
in competitive, non-griefed matches (min table score >= min_table_vp).
"""

import argparse
import concurrent.futures
import glob
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training the model"
SCRAPER_DIR = PROJECT_ROOT / "scraper"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from environment import NativeGaiaEnv, compute_competitive_value

RAW_GAMES_DIR = PROJECT_ROOT / "scraper" / "data" / "raw_games"
DATASET_DIR = PROJECT_ROOT / "scraper" / "data" / "dataset"
BASE_FACTIONS = [
    "terrans", "lantids", "hadsch_hallas", "ivits", "geodens", "bal_taks",
    "xenos", "gleens", "taklons", "ambas", "firaks", "bescods", "nevlas", "itars",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dataset_builder")


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


def parse_compound_move(parts: List[str], env: NativeGaiaEnv, sectors_dict: Dict) -> List[int]:
    """Parses subparts of a move into flat discrete actions matching action_space.rs."""
    actions = []
    i = 0
    while i < len(parts):
        part = parts[i]
        p_lower = part.strip().lower()

        if any(p_lower.startswith(k) for k in ["init", "setup", "rotate", "faction", "income"]):
            i += 1
            continue

        if "burn" in p_lower or "build sp" in p_lower:
            i += 1
            continue

        # Free resource conversions (3124..3129)
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
            elif "1q for 1o" in p_lower:
                actions.append(A_FREE_ACTION_OFFSET + 4)
            i += 1
            continue

        # Power charge / leech
        if "charge" in p_lower:
            actions.append(A_CHARGE_POWER_OFFSET + 1)
            i += 1
            continue
        if "decline" in p_lower:
            actions.append(A_CHARGE_POWER_OFFSET + 0)
            i += 1
            continue

        # Booster / Pass
        if "pass" in p_lower:
            m_pass_take = re.search(r'pass\s+booster(\d+)', p_lower)
            if m_pass_take:
                b_idx = int(m_pass_take.group(1))
                actions.append(A_PASS_OFFSET + (b_idx - 1))
            elif "returning" in p_lower or p_lower.strip() == "pass":
                actions.append(A_PASS_OFFSET)
            else:
                m_booster = re.search(r'booster\s*(\d+)', p_lower)
                b_idx = int(m_booster.group(1)) if m_booster else 1
                actions.append(A_PASS_OFFSET + (b_idx - 1))
            i += 1
            continue
        elif "booster" in p_lower:
            m_booster = re.search(r'booster\s*(\d+)', p_lower)
            if m_booster:
                b_idx = int(m_booster.group(1))
                actions.append(A_PASS_OFFSET + (b_idx - 1))
            i += 1
            continue

        # Board power action
        m_pwr = re.search(r'action\s+(power\d|qic\d)', p_lower)
        if m_pwr:
            p_act = m_pwr.group(1)
            if p_act in BOARDS_POWER_ACTIONS:
                actions.append(A_BOARD_ACTION_OFFSET + BOARDS_POWER_ACTIONS[p_act])
            i += 1
            continue

        # Special Actions
        if "space-station" in p_lower:
            actions.append(A_SPECIAL_ACTION_OFFSET + 3)
            i += 1
            continue
        if "ambas-swap" in p_lower or "action ambas" in p_lower or "swap-pi" in p_lower:
            actions.append(A_SPECIAL_ACTION_OFFSET + 0)
            i += 1
            continue
        if "downgrade" in p_lower or "action firaks" in p_lower:
            actions.append(A_SPECIAL_ACTION_OFFSET + 1)
            i += 1
            continue

        # Claim Tech Tile (may be followed by 'up <field>')
        m_tech = re.search(r'tech\s+([a-zA-Z0-9]+)', p_lower)
        if m_tech:
            t_str = m_tech.group(1)
            t_idx = TECH_TILE_MAP.get(t_str, 0)
            f_idx = 0
            if i + 1 < len(parts):
                next_p = parts[i + 1].strip().lower()
                m_next_up = re.search(r'\bup\s+([a-zA-Z]+)', next_p)
                if m_next_up:
                    f_idx = RESEARCH_FIELDS.get(m_next_up.group(1), 0)
                    i += 1
            actions.append(A_CLAIM_TECH_OFFSET + t_idx * 6 + f_idx)
            i += 1
            continue

        # Advance Research standalone
        m_up = re.search(r'\bup\s+([a-zA-Z]+)', p_lower)
        if m_up and "tech" not in p_lower:
            field = m_up.group(1)
            if field in RESEARCH_FIELDS:
                actions.append(A_ADVANCE_RESEARCH_OFFSET + RESEARCH_FIELDS[field])
            i += 1
            continue

        # Form Federation
        if "fed" in p_lower:
            m_fed = re.search(r'fed(\d+)', p_lower)
            f_token = int(m_fed.group(1)) - 1 if m_fed else 0
            f_token = max(0, min(5, f_token))
            actions.append(A_FEDERATION_OFFSET + f_token)
            i += 1
            continue

        # Build Mine
        m_mine = re.search(r'build\s+m\s+(\d+[A-Za-z]?[ABCabc]\d*)', p_lower)
        if m_mine:
            cube = resolve_bgs_coord(m_mine.group(1), sectors_dict)
            if cube:
                h_idx = env.get_hex_index(cube[0], cube[1], cube[2])
                if h_idx >= 0:
                    actions.append(A_BUILD_MINE_OFFSET + h_idx)
            i += 1
            continue

        # Build / Upgrade structure (ts, lab, pi, ac)
        m_upg = re.search(r'build\s+(ts|lab|pi|ac)\s+(\d+[A-Za-z]?[ABCabc]\d*)', p_lower)
        if m_upg:
            b_name = m_upg.group(1)
            cube = resolve_bgs_coord(m_upg.group(2), sectors_dict)
            if cube:
                h_idx = env.get_hex_index(cube[0], cube[1], cube[2])
                if h_idx >= 0:
                    upg_type = 0
                    if b_name == "ts": upg_type = 0
                    elif b_name == "lab": upg_type = 1
                    elif b_name == "pi": upg_type = 2
                    elif b_name == "ac": upg_type = 3
                    actions.append(A_UPGRADE_OFFSET + h_idx * 5 + upg_type)
            i += 1
            continue

        # Start Gaia Project
        m_gaia = re.search(r'gaia\s+(\d+[A-Za-z]?[ABCabc]\d*)', p_lower)
        if m_gaia:
            cube = resolve_bgs_coord(m_gaia.group(1), sectors_dict)
            if cube:
                h_idx = env.get_hex_index(cube[0], cube[1], cube[2])
                if h_idx >= 0:
                    actions.append(A_START_GAIA_OFFSET + h_idx)
            i += 1
            continue

        i += 1

    return actions


def process_single_game(args_tuple: Tuple[str, int, int]) -> Optional[Dict]:
    """Replays a single game end-to-end and returns extracted legal Grandmaster transitions."""
    fpath, min_gm_vp, min_table_vp = args_tuple
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            game = json.load(f)
    except Exception:
        return None

    data = game.get("data", {})
    players = data.get("players", [])
    if not (3 <= len(players) <= 4):
        return None

    scores_by_seat = []
    faction_to_seat = {}
    seat_to_faction = {}

    for idx, p in enumerate(players):
        f_name = normalize_faction(p.get("faction", ""))
        vp = p.get("data", {}).get("victoryPoints", 0)
        scores_by_seat.append(int(vp))
        faction_to_seat[f_name] = idx
        fac_id = FACTION_TO_ID.get(f_name, idx % 14)
        seat_to_faction[idx] = fac_id

    # Filter 1: Check table quality (no AFKs or low-skill tables that corrupt expansion)
    if min(scores_by_seat) < min_table_vp:
        return None

    # Filter 2: Check presence of at least one Grandmaster player (>= min_gm_vp)
    gm_seats = {idx for idx, vp in enumerate(scores_by_seat) if vp >= min_gm_vp}
    if not gm_seats:
        return None

    map_conf = data.get("options", {}).get("map")
    if not map_conf or "sectors" not in map_conf:
        return None

    sectors_dict = {s["sector"]: s for s in map_conf["sectors"]}
    map_json = json.dumps(map_conf)

    try:
        env = NativeGaiaEnv(players=len(players), seed=42, egocentric=True)
        if not env.load_map_json(map_json):
            return None
        for seat in range(len(players)):
            env.set_player_faction(seat, seat_to_faction.get(seat, seat % 14))

        if hasattr(env, "set_turn_order"):
            turn_order = data.get("turnOrderAfterSetupAuction") or data.get("turnOrder") or list(range(len(players)))
            env.set_turn_order(turn_order)
    except Exception:
        return None

    moves = data.get("moveHistory", [])
    if not moves:
        return None

    # Setup Phase handling (Initial Mines + Booster selection)
    setup_mines_placed = 0
    setup_boosters_set = 0
    normal_start_idx = 0

    for m_idx, raw_m in enumerate(moves):
        clean = re.sub(r'\(.*?\)', '', raw_m).strip()
        m_bld = re.search(r'([a-zA-Z_-]+)\s+build\s+(m|pi)\s+(\d+[A-Za-z]?[ABCabc]\d*)', clean, re.I)
        m_bst = re.search(r'([a-zA-Z_-]+)\s+booster\s+booster(\d+)', clean, re.I)

        if m_bld and setup_boosters_set < len(players):
            f_tok = normalize_faction(m_bld.group(1))
            seat = faction_to_seat.get(f_tok)
            bld_type = 0 if m_bld.group(2).lower() == 'm' else 2  # 0=Mine, 2=PI (Ivits)
            cube = resolve_bgs_coord(m_bld.group(3), sectors_dict)
            if cube and seat is not None:
                env.set_hex_building(cube[0], cube[1], cube[2], bld_type, seat)
                setup_mines_placed += 1
            continue

        if m_bst and setup_boosters_set < len(players):
            f_tok = normalize_faction(m_bst.group(1))
            seat = faction_to_seat.get(f_tok)
            b_id = int(m_bst.group(2))
            if seat is not None:
                env.set_player_booster(seat, b_id)
                setup_boosters_set += 1
            if setup_boosters_set == len(players):
                normal_start_idx = m_idx + 1
            continue

    if normal_start_idx == 0:
        normal_start_idx = min(len(moves), len(players) * 3)

    env.set_round(1)

    raw_vps_by_seat = [float(vp) for vp in scores_by_seat]

    game_obs = []
    game_masks = []
    game_actions = []
    game_values = []
    game_factions = []
    illegal_filtered = 0
    micro_filtered = 0

    # Normal gameplay loop (Rounds 1..6)
    for m_idx in range(normal_start_idx, len(moves)):
        raw_m = moves[m_idx]
        clean = re.sub(r'\(.*?\)', '', raw_m).strip()
        parts = [p.strip() for p in clean.split('.') if p.strip()]
        if not parts:
            continue

        actor_token = normalize_faction(parts[0].split()[0].lower()) if parts and parts[0].split() else ""
        acting_seat = faction_to_seat.get(actor_token)

        actions = parse_compound_move(parts, env, sectors_dict)
        for act in actions:
            if acting_seat is not None and acting_seat != env.current_player:
                env.set_current_player(acting_seat)

            mask = env.get_action_mask()
            is_legal = bool(act < len(mask) and mask[act])

            if is_legal:
                # Discard trivial micro-burns/spends
                if act >= A_FREE_ACTION_OFFSET:
                    micro_filtered += 1
                    env.step(int(act))
                else:
                    # STRICT GRANDMASTER FILTER:
                    # ONLY record transitions if the acting player scored >= min_gm_vp!
                    if acting_seat in gm_seats:
                        obs = env.get_observation()
                        curr_seat = getattr(env, "current_player", 0)
                        fac_id = seat_to_faction.get(curr_seat, 0)
                        z_val = compute_competitive_value(raw_vps_by_seat, curr_seat)

                        game_obs.append(obs.astype(np.float16))
                        game_masks.append(np.array(mask, dtype=np.bool_))
                        game_actions.append(int(act))
                        game_values.append(float(z_val))
                        game_factions.append(int(fac_id))

                    env.step(int(act))
            else:
                illegal_filtered += 1

    if not game_actions:
        return None

    return {
        "obs": np.stack(game_obs),
        "masks": np.stack(game_masks),
        "actions": game_actions,
        "values": game_values,
        "factions": game_factions,
        "illegal_filtered": illegal_filtered,
        "micro_filtered": micro_filtered,
        "gm_scores": [scores_by_seat[s] for s in gm_seats],
    }


def build_authentic_dataset(
    raw_dir: Path = RAW_GAMES_DIR,
    out_file: Path = DATASET_DIR / "gaia_expert_dataset.pt",
    min_gm_vp: int = 175,
    min_table_vp: int = 110,
    max_games: Optional[int] = None,
    workers: int = 8,
):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    game_files = sorted(glob.glob(str(raw_dir / "*.json")))
    if not game_files:
        logger.error(f"No game files found in {raw_dir}!")
        return None

    if max_games:
        game_files = game_files[:max_games]

    logger.info(
        f"Rebuilding Grandmaster Dataset from {len(game_files)} games\n"
        f"  • Filter: GM Player Score >= {min_gm_vp} VP\n"
        f"  • Filter: Table Quality (All Players >= {min_table_vp} VP)\n"
        f"  • Parallel Workers: {workers}"
    )
    t_start = time.time()

    all_obs_chunks = []
    all_masks_chunks = []
    all_actions = []
    all_values = []
    all_factions = []
    all_gm_scores = []
    parsed_games = 0
    skipped_games = 0
    total_illegal_filtered = 0

    args_list = [(gf, min_gm_vp, min_table_vp) for gf in game_files]

    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for idx, res in enumerate(executor.map(process_single_game, args_list), 1):
                if res is None:
                    skipped_games += 1
                else:
                    parsed_games += 1
                    all_obs_chunks.append(res["obs"])
                    all_masks_chunks.append(res["masks"])
                    all_actions.extend(res["actions"])
                    all_values.extend(res["values"])
                    all_factions.extend(res["factions"])
                    all_gm_scores.extend(res.get("gm_scores", []))
                    total_illegal_filtered += res["illegal_filtered"]

                if idx % 50 == 0 or idx == len(game_files):
                    cur_samples = len(all_actions)
                    el = max(1e-4, time.time() - t_start)
                    logger.info(
                        f"[{idx}/{len(game_files)}] Clean GM Games: {parsed_games} ({skipped_games} skipped) | "
                        f"Certified GM Samples: {cur_samples:,} | Speed: {cur_samples / el:.0f} samples/s"
                    )
    else:
        for idx, arg in enumerate(args_list, 1):
            res = process_single_game(arg)
            if res is None:
                skipped_games += 1
            else:
                parsed_games += 1
                all_obs_chunks.append(res["obs"])
                all_masks_chunks.append(res["masks"])
                all_actions.extend(res["actions"])
                all_values.extend(res["values"])
                all_factions.extend(res["factions"])
                all_gm_scores.extend(res.get("gm_scores", []))
                total_illegal_filtered += res["illegal_filtered"]

            if idx % 50 == 0 or idx == len(game_files):
                cur_samples = len(all_actions)
                el = max(1e-4, time.time() - t_start)
                logger.info(
                    f"[{idx}/{len(game_files)}] Clean GM Games: {parsed_games} ({skipped_games} skipped) | "
                    f"Certified GM Samples: {cur_samples:,} | Speed: {cur_samples / el:.0f} samples/s"
                )

    total_samples = len(all_actions)
    if total_samples == 0:
        logger.error("No valid samples extracted! Verify raw games and filters.")
        return None

    logger.info(f"Concatenating {len(all_obs_chunks)} chunks for {total_samples:,} observations & masks...")
    combined_obs_np = np.concatenate(all_obs_chunks, axis=0)
    combined_masks_np = np.concatenate(all_masks_chunks, axis=0)

    logger.info(f"Building PyTorch dataset (Obs shape: {combined_obs_np.shape}, Masks shape: {combined_masks_np.shape})...")
    dataset_dict = {
        "observations": torch.from_numpy(combined_obs_np),
        "action_masks": torch.from_numpy(combined_masks_np),
        "actions": torch.tensor(all_actions, dtype=torch.long),
        "values": torch.tensor(all_values, dtype=torch.float32),
        "factions": torch.tensor(all_factions, dtype=torch.long),
        "num_games": parsed_games,
        "num_samples": total_samples,
        "avg_gm_score": float(np.mean(all_gm_scores)) if all_gm_scores else 0.0,
    }

    logger.info(f"Saving certified Grandmaster dataset to {out_file}...")
    torch.save(dataset_dict, out_file)
    size_mb = out_file.stat().st_size / (1024 * 1024)
    total_time = time.time() - t_start

    avg_gm = np.mean(all_gm_scores) if all_gm_scores else 0.0
    logger.info(
        f"\n============================================================\n"
        f"  🏆 Authentic Grandmaster Dataset Successfully Built!\n"
        f"============================================================\n"
        f"  • File Path       : {out_file}\n"
        f"  • File Size       : {size_mb:.1f} MB\n"
        f"  • GM Games Parsed : {parsed_games} ({skipped_games} filtered/skipped)\n"
        f"  • GM Transitions  : {total_samples:,} strictly certified legal transitions\n"
        f"  • Average GM Score: {avg_gm:.1f} VP (Peak: {max(all_gm_scores) if all_gm_scores else 0} VP)\n"
        f"  • Total Time      : {total_time:.1f}s ({total_samples / total_time:.0f} samples/s)\n"
        f"============================================================"
    )
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild Authentic Grandmaster Dataset with True 2476D Observations")
    parser.add_argument("--min-gm-vp", type=int, default=175, help="Minimum individual GM score threshold (default: 175)")
    parser.add_argument("--min-table-vp", type=int, default=110, help="Minimum table score threshold (default: 110)")
    parser.add_argument("--max-games", type=int, default=None, help="Limit number of games for testing")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes (default: 8)")
    args = parser.parse_args()

    build_authentic_dataset(
        min_gm_vp=args.min_gm_vp,
        min_table_vp=args.min_table_vp,
        max_games=args.max_games,
        workers=args.workers,
    )
