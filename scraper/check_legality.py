import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training the model"
SCRAPER_DIR = PROJECT_ROOT / "scraper"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

from environment import NativeGaiaEnv
from scraper.rebuild_expert_dataset import (
    parse_move_to_actions,
    normalize_faction,
    FACTION_TO_ID,
    BASE_FACTIONS,
    SECTOR_OFFSETS
)

# Test first 5 games
raw_files = sorted(list((SCRAPER_DIR / "data" / "raw").glob("bgs_game_*.json")))[:5]

total_actions = 0
legal_actions = 0
illegal_samples = []

for gf in raw_files:
    with open(gf, "r", encoding="utf-8") as f:
        game = json.load(f)

    players = game.get("players", [])
    map_conf = game.get("data", {}).get("options", {}).get("map")
    if not map_conf or "sectors" not in map_conf:
        continue

    sectors_dict = {s["sector"]: s for s in map_conf["sectors"]}
    env = NativeGaiaEnv(players=len(players), seed=42, egocentric=True)
    env.load_map_json(json.dumps(map_conf))

    faction_to_seat = {}
    for seat, p in enumerate(players):
        raw_f = p.get("faction", "").lower()
        f_name = normalize_faction(raw_f)
        if f_name:
            faction_to_seat[f_name] = seat
            faction_to_seat[raw_f] = seat
            f_id = FACTION_TO_ID.get(f_name, seat % 14)
            env.set_player_faction(seat, f_id)

    moves = game.get("data", {}).get("moveHistory", [])
    for m in moves:
        clean = re.sub(r'\(.*?\)', '', m).strip()
        parts = [p.strip() for p in clean.split('.') if p.strip()]
        actor_token = normalize_faction(parts[0].split()[0].lower()) if parts and parts[0].split() else ""
        acting_seat = faction_to_seat.get(actor_token)

        for part in parts:
            if not part:
                continue
            p_lower = part.lower()
            first_token = normalize_faction(p_lower.split()[0]) if p_lower.split() else ""
            if first_token in faction_to_seat:
                acting_seat = faction_to_seat[first_token]

            if any(p_lower.startswith(k) for k in ["init", "setup", "rotate", "faction", "income"]):
                continue

            if acting_seat is not None:
                env.set_current_player(acting_seat)

            sub_actions = parse_move_to_actions(part, env, sectors_dict)
            for act in sub_actions:
                total_actions += 1
                mask = env.get_action_mask()
                is_leg = bool(mask[act]) if act < len(mask) else False
                if is_leg:
                    legal_actions += 1
                    env.step(int(act))
                else:
                    if len(illegal_samples) < 15:
                        illegal_samples.append((part, act, acting_seat, np.where(mask)[0][:10].tolist()))

print(f"Total Actions parsed: {total_actions}")
print(f"Legal in Rust engine: {legal_actions} ({legal_actions/max(1,total_actions)*100:.1f}%)")
print(f"Illegal samples (first 10):")
for p, act, s, legal_subset in illegal_samples[:10]:
    print(f"  Part: '{p}' -> act {act} (seat {s}) | Legal subset: {legal_subset}")
