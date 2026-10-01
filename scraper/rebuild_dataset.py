import json
import logging
from pathlib import Path
import re
import sys
import torch
from collections import deque

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "training the model"))

from environment import NativeGaiaEnv
from scraper.config import RAW_GAMES_DIR, DATASET_DIR, BASE_FACTIONS
from scraper.convert_to_dataset import parse_move_actions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rebuilder")

FACTION_TO_ID = {f: idx for idx, f in enumerate(BASE_FACTIONS)}

def find_legal_path(env: NativeGaiaEnv, target_action_id: int, actor_id: int, depth: int = 4):
    """
    BFS to find a sequence of free actions that makes the target_action_id legal.
    (Because BGS logs often omit free actions like converting power to ore).
    """
    legal_actions = env.get_legal_actions()
    if legal_actions[target_action_id]:
        return []
    
    # BFS queue: (env_state, path)
    # NativeGaiaEnv has clone() if it's a Rust object?
    # We can't easily clone the Rust env without FFI support.
    # Let's see if we have get_state / set_state.
    pass
