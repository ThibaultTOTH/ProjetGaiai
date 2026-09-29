"""
Gaia Project Scraper Package.
Automated collection and dataset generation of expert human games from Boardgamers.space.
"""

from .config import ScraperConfig, BASE_FACTIONS, DEFAULT_CONFIG
from .api_client import BGSApiClient
from .scraper import GaiaGameScraper

__all__ = [
    "ScraperConfig",
    "BASE_FACTIONS",
    "DEFAULT_CONFIG",
    "BGSApiClient",
    "GaiaGameScraper",
]
