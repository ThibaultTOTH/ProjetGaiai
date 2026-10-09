"""
Automated Game Scraper for Gaia Project on Boardgamers.space.
Balances faction representation and selects high-level human expert games.
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from rich.console import Console
    from rich.table import Table
    from rich.box import ASCII
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from .api_client import BGSApiClient
from .config import (
    BASE_FACTIONS,
    BASE_FACTIONS_SET,
    LOST_FLEET_FACTIONS,
    ScraperConfig,
    DEFAULT_CONFIG,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("scraper")


class GaiaGameScraper:
    """Orchestrates balanced scraping of high-level expert games from BGS."""

    def __init__(self, config: Optional[ScraperConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.client = BGSApiClient(self.config)

        # Ensure directories exist
        self.config.raw_dir.mkdir(parents=True, exist_ok=True)
        self.config.dataset_dir.mkdir(parents=True, exist_ok=True)

        # In-memory index of scraped games: {game_id: summary_dict}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        # Faction appearance counts: {faction_name: count}
        self.faction_counts: Counter = Counter()

        self._load_existing_metadata()

    def _load_existing_metadata(self) -> None:
        """Loads metadata.json if it exists to resume scraping without duplicates."""
        if self.config.metadata_path.exists():
            try:
                with open(self.config.metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.metadata = data.get("games", {})
                    # Recompute faction counts from verified metadata
                    self.faction_counts.clear()
                    for gid, g_info in self.metadata.items():
                        for f in g_info.get("factions", []):
                            if f in BASE_FACTIONS_SET:
                                self.faction_counts[f] += 1
                logger.info(
                    f"Resumed from {self.config.metadata_path}: {len(self.metadata)} games already indexed."
                )
            except Exception as e:
                logger.warning(f"Could not load metadata from {self.config.metadata_path}: {e}")

    def _save_metadata(self) -> None:
        """Atomically saves the current metadata index to disk."""
        tmp_path = self.config.metadata_path.with_suffix(".tmp")
        payload = {
            "total_games": len(self.metadata),
            "faction_counts": dict(self.faction_counts),
            "games": self.metadata,
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp_path.replace(self.config.metadata_path)

    def is_game_valid_and_needed(self, listing_game: Dict[str, Any]) -> bool:
        """
        Evaluates a game from the listing endpoint before requesting full gameplay.
        Filters for:
        1. Not cancelled.
        2. Expected player count (e.g. 4 players).
        3. All players have picked standard factions (base game only if configured).
        4. High skill: maximum score >= min_winning_score.
        5. Faction quota: game must contain at least one faction that hasn't reached target.
        """
        if listing_game.get("cancelled", False):
            return False

        options = listing_game.get("options", {})
        nb_players = options.get("setup", {}).get("nbPlayers")
        if nb_players != self.config.required_num_players:
            return False

        players = listing_game.get("players", [])
        if len(players) != self.config.required_num_players:
            return False

        # Check players, factions and scores
        factions = []
        scores = []
        for p in players:
            faction = p.get("faction")
            if not faction:
                return False
            if self.config.base_game_only and faction not in BASE_FACTIONS_SET:
                return False
            factions.append(faction)
            scores.append(p.get("score", 0))

        # Check skill level: winning score must be high
        max_score = max(scores) if scores else 0
        if max_score < self.config.min_winning_score:
            return False

        # Faction balancing: check if any faction in this game still needs games
        needs_download = False
        for f in factions:
            if self.faction_counts[f] < self.config.target_per_faction:
                needs_download = True
                break

        return needs_download

    def all_factions_satisfied(self) -> bool:
        """Checks if all target factions have reached the desired game count."""
        for f in BASE_FACTIONS:
            if self.faction_counts[f] < self.config.target_per_faction:
                return False
        return True

    def display_status(self) -> None:
        """Prints a clean summary table of current scraping progress."""
        if HAS_RICH:
            console = Console(force_terminal=True, legacy_windows=False)
            table = Table(title="Gaia Project Scraper - Faction Balance Dashboard", box=ASCII)
            table.add_column("Faction", justify="left", style="cyan")
            table.add_column("Scraped", justify="right", style="green")
            table.add_column("Target", justify="right", style="white")
            table.add_column("Progress", justify="right", style="yellow")
            table.add_column("Bar", justify="left")

            for f in sorted(BASE_FACTIONS):
                count = self.faction_counts[f]
                target = self.config.target_per_faction
                pct = min(100.0, (count / target) * 100) if target > 0 else 100.0
                bar_len = int(pct / 5)
                bar = "=" * bar_len + "-" * (20 - bar_len)
                table.add_row(f, str(count), str(target), f"{pct:5.1f}%", f"[{bar}]")

            console.print(table)
            console.print(
                f"[bold]Total distinct games scraped:[/] {len(self.metadata)} | "
                f"[bold]Min required score:[/] {self.config.min_winning_score} VP"
            )
        else:
            print("\n--- Faction Balance Progress ---")
            for f in sorted(BASE_FACTIONS):
                count = self.faction_counts[f]
                target = self.config.target_per_faction
                pct = min(100.0, (count / target) * 100) if target > 0 else 100.0
                print(f"  {f:15s}: {count:4d}/{target:4d} ({pct:5.1f}%)")
            print(f"Total games: {len(self.metadata)}\n")

    def run(self, max_pages: Optional[int] = None) -> None:
        """
        Executes the scraping loop:
        1. Paginates through ended games.
        2. Filters games using balanced criteria.
        3. Downloads full gameplays and saves to disk.
        4. Updates live metadata index.
        """
        logger.info(
            f"Starting scraper: target={self.config.target_per_faction} per faction, "
            f"min_vp={self.config.min_winning_score}, players={self.config.required_num_players}"
        )
        self.display_status()

        skip = 0
        pages_processed = 0
        games_downloaded_this_session = 0

        try:
            while skip <= self.config.max_skip:
                if self.all_factions_satisfied():
                    logger.info("All factions reached target quotas! Scraping complete.")
                    break

                if max_pages is not None and pages_processed >= max_pages:
                    logger.info(f"Reached max pages limit ({max_pages}). Stopping.")
                    break

                logger.info(f"Fetching listing page: skip={skip}, count={self.config.page_size}...")
                try:
                    listing = self.client.get_ended_games(skip=skip, count=self.config.page_size)
                except Exception as e:
                    logger.error(f"Failed to fetch listing at skip={skip}: {e}")
                    time.sleep(3.0)
                    skip += self.config.page_size
                    pages_processed += 1
                    continue

                if not listing:
                    logger.info("No more games returned by API. Reached end of listing.")
                    break

                # Process candidates
                for g in listing:
                    game_id = g.get("_id")
                    if not game_id:
                        continue

                    # Skip if already downloaded
                    if game_id in self.metadata:
                        continue

                    if not self.is_game_valid_and_needed(g):
                        continue

                    # Fetch full gameplay data
                    try:
                        logger.info(f"Downloading gameplay: {game_id}...")
                        gameplay = self.client.get_gameplay(game_id)
                    except Exception as e:
                        logger.warning(f"Could not download gameplay {game_id}: {e}")
                        continue

                    data = gameplay.get("data", {})
                    move_history = data.get("moveHistory", [])
                    if len(move_history) < 30:
                        # Skip incomplete or abnormally short games
                        continue

                    # Save raw JSON
                    game_path = self.config.raw_dir / f"{game_id}.json"
                    with open(game_path, "w", encoding="utf-8") as f:
                        json.dump(gameplay, f, ensure_ascii=False)

                    # Extract metadata summary
                    players_info = []
                    factions_in_game = []
                    scores_in_game = []
                    for p in gameplay.get("players", []):
                        f_name = p.get("faction")
                        score = p.get("score", 0)
                        elo = p.get("elo", {}).get("initial") if isinstance(p.get("elo"), dict) else None
                        players_info.append({
                            "name": p.get("name"),
                            "faction": f_name,
                            "score": score,
                            "elo": elo,
                            "ranking": p.get("ranking"),
                        })
                        if f_name:
                            factions_in_game.append(f_name)
                            scores_in_game.append(score)

                    self.metadata[game_id] = {
                        "id": game_id,
                        "endedAt": gameplay.get("endedAt"),
                        "num_moves": len(move_history),
                        "max_score": max(scores_in_game) if scores_in_game else 0,
                        "factions": factions_in_game,
                        "players": players_info,
                    }

                    for f in factions_in_game:
                        if f in BASE_FACTIONS_SET:
                            self.faction_counts[f] += 1

                    games_downloaded_this_session += 1

                    # Save checkpoint every 10 downloads
                    if games_downloaded_this_session % 10 == 0:
                        self._save_metadata()
                        self.display_status()

                skip += self.config.page_size
                pages_processed += 1

        except KeyboardInterrupt:
            logger.info("Scraping interrupted by user. Saving metadata...")
        finally:
            self._save_metadata()
            self.display_status()
            logger.info(
                f"Session completed: {games_downloaded_this_session} new games downloaded. "
                f"Total games in library: {len(self.metadata)}"
            )


def main():
    parser = argparse.ArgumentParser(description="Gaia Project Expert Game Scraper")
    parser.add_argument(
        "--target-per-faction",
        type=int,
        default=DEFAULT_CONFIG.target_per_faction,
        help="Target number of games per faction (default: 1000)",
    )
    parser.add_argument(
        "--min-vp",
        type=int,
        default=DEFAULT_CONFIG.min_winning_score,
        help="Minimum winning score in VP (default: 140)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum listing pages to scrape in this run",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_CONFIG.request_delay,
        help="Polite delay in seconds between HTTP requests (default: 0.6)",
    )
    parser.add_argument(
        "--players",
        type=int,
        default=DEFAULT_CONFIG.required_num_players,
        help="Required number of players (default: 4)",
    )
    args = parser.parse_args()

    config = ScraperConfig(
        target_per_faction=args.target_per_faction,
        min_winning_score=args.min_vp,
        request_delay=args.delay,
        required_num_players=args.players,
    )

    scraper = GaiaGameScraper(config)
    scraper.run(max_pages=args.max_pages)


if __name__ == "__main__":
    main()
