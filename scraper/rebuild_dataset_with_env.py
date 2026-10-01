import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "training the model"))
from environment import NativeGaiaEnv
from scraper.config import RAW_GAMES_DIR, DATASET_DIR, BASE_FACTIONS
from scraper.convert_to_dataset import parse_move_actions, FACTION_TO_ID

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rebuilder")

A_FREE_ACTION_OFFSET = 3124
FREE_ACTION_COUNT = 6
FREE_ACTIONS = list(range(A_FREE_ACTION_OFFSET, A_FREE_ACTION_OFFSET + FREE_ACTION_COUNT))

def search_implicit_free_actions(history: List[int], target_action: int, max_depth: int = 3, seed: int = 42) -> Optional[List[int]]:
    """
    BFS to find a sequence of free actions (max length `max_depth`) that makes `target_action` legal.
    Because NativeGaiaEnv doesn't support state cloning from Python yet, we replay the history from scratch
    for each node. The Rust engine is fast enough to handle this for depth 3.
    """
    queue = [ [] ]
    
    while queue:
        free_path = queue.pop(0)
        
        # Test this path
        env = NativeGaiaEnv(seed=seed, egocentric=True)
        # Note: BGS games have their own setup, we assume NativeGaiaEnv(seed=seed) matches the initial state
        # (This might be a problem if BGS games have random setups not defined by a single seed, 
        # but for now we try our best).
        
        try:
            # Replay history
            for act in history:
                env.step(act)
            
            # Apply free path
            path_legal = True
            for f_act in free_path:
                mask = env.get_action_mask()
                if not mask[f_act]:
                    path_legal = False
                    break
                env.step(f_act)
                
            if not path_legal:
                continue
                
            # Check if target is now legal
            mask = env.get_action_mask()
            if mask[target_action]:
                return free_path
                
        except Exception:
            continue
            
        # Expand
        if len(free_path) < max_depth:
            for f_act in FREE_ACTIONS:
                queue.append(free_path + [f_act])
                
    return None

def rebuild_dataset(min_winning_score: int = 140):
    game_files = list(RAW_GAMES_DIR.glob("*.json"))
    logger.info(f"Processing {len(game_files)} games with environment validation...")

    all_obs = []
    all_actions = []
    all_values = []
    all_factions = []

    parsed_games = 0
    skipped_games = 0

    for gf in game_files:
        with open(gf, "r", encoding="utf-8") as f:
            game = json.load(f)

        players = game.get("players", [])
        scores_by_faction = {p.get("faction"): p.get("score", 0) for p in players if p.get("faction")}
        max_score = max(scores_by_faction.values()) if scores_by_faction else 0
        
        if max_score < min_winning_score:
            skipped_games += 1
            continue
            
        # Determine seed (simplified, ideally extract from game)
        seed = 42 
        
        moves = game.get("data", {}).get("moveHistory", [])
        history = []
        game_obs = []
        game_acts = []
        game_vals = []
        game_facs = []
        
        valid_game = True
        for m in moves:
            parts = m.split()
            if not parts:
                continue

            actor_str = parts[0].lower()
            faction_id = FACTION_TO_ID.get(actor_str)
            if faction_id is None:
                continue

            action_indices = parse_move_actions(m)
            
            for act in action_indices:
                # Find path
                free_path = search_implicit_free_actions(history, act, max_depth=3, seed=seed)
                
                if free_path is None:
                    # Unrecoverable desync (e.g., auto-leech divergence or missing setup)
                    logger.debug(f"Failed to resolve {m} in {gf.name}. Skipping game.")
                    valid_game = False
                    break
                    
                # Replay successful path to collect observations
                env = NativeGaiaEnv(seed=seed, egocentric=True)
                for h_act in history:
                    env.step(h_act)
                    
                final_score = scores_by_faction.get(actor_str, 100)
                norm_value = (float(final_score) - 100.0) / 50.0
                
                for f_act in free_path:
                    game_obs.append(env.get_observation().copy())
                    game_acts.append(f_act)
                    game_vals.append(norm_value)
                    game_facs.append(faction_id)
                    env.step(f_act)
                    history.append(f_act)
                    
                game_obs.append(env.get_observation().copy())
                game_acts.append(act)
                game_vals.append(norm_value)
                game_facs.append(faction_id)
                env.step(act)
                history.append(act)
                
            if not valid_game:
                break
                
        if valid_game and game_acts:
            all_obs.extend(game_obs)
            all_actions.extend(game_acts)
            all_values.extend(game_vals)
            all_factions.extend(game_facs)
            parsed_games += 1
        else:
            skipped_games += 1

    logger.info(f"Parsed {parsed_games} games ({skipped_games} skipped). Extracted {len(all_actions)} moves.")
    if all_actions:
        dataset_dict = {
            "observations": torch.tensor(np.array(all_obs), dtype=torch.float32),
            "actions": torch.tensor(all_actions, dtype=torch.long),
            "values": torch.tensor(all_values, dtype=torch.float32),
            "factions": torch.tensor(all_factions, dtype=torch.long),
            "num_games": parsed_games,
            "num_samples": len(all_actions),
        }
        out_file = DATASET_DIR / "gaia_expert_dataset_v2.pt"
        torch.save(dataset_dict, out_file)
        logger.info(f"Saved exact-state dataset to {out_file}")

if __name__ == "__main__":
    rebuild_dataset()
