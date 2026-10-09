"""
Configuration for Boardgamers.space (BGS) Gaia Project Scraper.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set

# Base directory for the scraper
SCRAPER_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRAPER_DIR / "data"
RAW_GAMES_DIR = DATA_DIR / "raw_games"
DATASET_DIR = DATA_DIR / "dataset"
METADATA_FILE = DATA_DIR / "metadata.json"

# All 14 base factions in Gaia Project
BASE_FACTIONS: List[str] = [
    "terrans",
    "lantids",
    "xenos",
    "gleens",
    "taklons",
    "ambas",
    "hadsch-hallas",
    "ivits",
    "geodens",
    "baltaks",
    "firaks",
    "nevlas",
    "itars",
    "bescods",
]

BASE_FACTIONS_SET: Set[str] = set(BASE_FACTIONS)

# Lost Fleet factions (expansion)
LOST_FLEET_FACTIONS: Set[str] = {
    "moweyds",
    "tinkeroids",
    "darkanians",
    "space-giants",
}


@dataclass
class ScraperConfig:
    # API endpoints
    base_url: str = "https://www.boardgamers.space"
    game_boardgame: str = "gaia-project"
    
    # Target criteria
    target_per_faction: int = 1000      # Target number of games per faction
    min_winning_score: int = 150       # Minimum winning score (tournament level >= 150 VP)
    min_player_score: int = 60         # Minimum score for each participant (filters ragequits)
    required_num_players: int = 4      # Standard tournament setup: 4 players
    base_game_only: bool = True        # Exclude Lost Fleet factions
    
    # Scraping & rate limiting
    page_size: int = 50                # Max items per listing page
    max_skip: int = 10000              # Max skip supported by BGS API
    request_delay: float = 0.4         # Seconds to sleep between requests (polite & fast)
    max_retries: int = 5               # Retry attempts on 429 or network glitch
    backoff_factor: float = 2.0        # Exponential backoff multiplier
    user_agent: str = "GaiaProjectRLAgentScraper/1.0 (Research/Academic project; thibaulttoth@psl.eu)"
    
    # Storage
    raw_dir: Path = RAW_GAMES_DIR
    metadata_path: Path = METADATA_FILE
    dataset_dir: Path = DATASET_DIR


DEFAULT_CONFIG = ScraperConfig()
