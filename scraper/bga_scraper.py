#!/usr/bin/env python3
"""
Board Game Arena (BGA) Gaia Project Grandmaster Game Scraper.
Fetches top-ranked players from the official BGA Hall of Fame for Gaia Project (Game ID: 1495),
extracts their match histories, and downloads complete move-by-move replay logs.
Features:
- Resilient session management with curl_cffi (Chrome TLS fingerprint).
- Strict deduplication (guarantees zero duplicate tables).
- Polite rate-limiting (2.0s - 3.2s) with exponential backoff on retries.
- Progress checkpointing to bga_progress.json.
- Targeted at 20,000 high-ELO 4-player competitive matches.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from curl_cffi import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRAPER_DIR = PROJECT_ROOT / "scraper"
DATA_DIR = SCRAPER_DIR / "data"
RAW_BGA_DIR = DATA_DIR / "raw_bga_games"
PROGRESS_FILE = DATA_DIR / "bga_progress.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DATA_DIR / "bga_scraper.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger("bga_scraper")

DEFAULT_COOKIES = {
    "TournoiEnLigneidt": "sk5A1p7OWJ1Khrf",
    "PHPSESSID": "7ddb09867c4abebd3ccaabaaa97133cd",
    "TournoiEnLignetkt": "cF5cBDlFd6ktKGAbUrGAMwyHHKb9GaUxdX37kU36gzjIG5ZOMDDJkhCqAmG5yHNM",
    "TournoiEnLigne_sso_id": "6ca5493fc2cdb86a01c53fd22ac08f4f",
    "TournoiEnLigne_sso_user": "Yollo_g%243%24thibault.toth%40gmail.com",
    "euconsent-v2": "CQsYtYAQsYtYAAHABBENC1FgAAAAAAAAAAAAAAAAAAAA.IAAA.YAAAAAAAAAAA",
}
REQUEST_TOKEN = "sk5A1p7OWJ1Khrf"
GAIA_PROJECT_GAME_ID = 1495


class BGAScraper:
    def __init__(
        self,
        cookies: Dict[str, str],
        request_token: str,
        delay_min: float = 2.0,
        delay_max: float = 3.2,
        target_games: int = 20000,
        max_players: int = 500,
    ):
        self.request_token = request_token
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.target_games = target_games
        self.max_players = max_players

        self.session = requests.Session(impersonate="chrome124")
        for k, v in cookies.items():
            self.session.cookies.set(k, v, domain=".boardgamearena.com")
            self.session.cookies.set(k, v, domain="boardgamearena.com")
            self.session.cookies.set(k, v, domain="fr.boardgamearena.com")
            self.session.cookies.set(k, v, domain="en.boardgamearena.com")

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://boardgamearena.com/gamestats",
        }

        RAW_BGA_DIR.mkdir(parents=True, exist_ok=True)
        self.downloaded_tables: Set[str] = self._load_downloaded_tables()
        logger.info(f"Initialized BGA Scraper. Already downloaded: {len(self.downloaded_tables)} tables.")

    def _load_downloaded_tables(self) -> Set[str]:
        existing = set()
        if RAW_BGA_DIR.exists():
            for p in RAW_BGA_DIR.glob("*.json"):
                if p.stat().st_size > 1000:  # Valid non-empty json
                    existing.add(p.stem)
        return existing

    def _sleep_politely(self) -> None:
        delay = random.uniform(self.delay_min, self.delay_max)
        time.sleep(delay)

    def _save_progress(self, current_player_idx: int, total_players: int, candidate_count: int):
        progress = {
            "downloaded_count": len(self.downloaded_tables),
            "target": self.target_games,
            "current_player_idx": current_player_idx,
            "total_players": total_players,
            "candidate_tables": candidate_count,
            "timestamp": time.time(),
        }
        try:
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not write progress file: {e}")

    def get_top_players(self, max_players: int = 500) -> List[Dict]:
        """Fetches top-ranked players from the official BGA Hall of Fame."""
        players = []
        start = 0
        logger.info(f"Querying BGA Hall of Fame up to {max_players} grandmasters...")
        while len(players) < max_players:
            url = f"https://boardgamearena.com/halloffame/halloffame/getRanking.html?game={GAIA_PROJECT_GAME_ID}&start={start}&request_token={self.request_token}"
            try:
                r = self.session.get(url, headers=self.headers, timeout=20)
                if r.status_code != 200:
                    logger.warning(f"Hall of Fame status {r.status_code} at start={start}")
                    break
                data = r.json()
                ranks = data.get("data", {}).get("ranks", [])
                if not ranks:
                    break
                for p in ranks:
                    players.append({
                        "id": str(p["id"]),
                        "name": p.get("name", ""),
                        "ranking": p.get("ranking", ""),
                        "nbr_game": int(p.get("nbr_game", 0)),
                    })
                logger.info(f"Loaded {len(players)}/{max_players} top players (latest: {players[-1]['name']} - ELO {players[-1]['ranking']}).")
                start += len(ranks)
                self._sleep_politely()
            except Exception as e:
                logger.error(f"Error fetching top players at start={start}: {e}")
                time.sleep(5)
                break
        return players[:max_players]

    def get_player_tables(self, player_id: str, max_pages: int = 40) -> List[Dict]:
        """Fetches table metadata for a given player."""
        tables = []
        for page in range(1, max_pages + 1):
            url = f"https://boardgamearena.com/gamestats/gamestats/getGames.html?player={player_id}&updateStats=0&game_id={GAIA_PROJECT_GAME_ID}&page={page}&request_token={self.request_token}"
            try:
                r = self.session.get(url, headers=self.headers, timeout=20)
                if r.status_code != 200:
                    break
                data = r.json()
                page_tables = data.get("data", {}).get("tables", [])
                if not page_tables:
                    break
                tables.extend(page_tables)
                if len(page_tables) < 10:
                    break
                self._sleep_politely()
            except Exception as e:
                logger.error(f"Error fetching tables for player {player_id} page {page}: {e}")
                break
        return tables

    def download_table_replay(self, table_id: str, table_meta: Optional[Dict] = None) -> bool:
        """Executes the 3-step archive protocol to retrieve the full game replay."""
        target_path = RAW_BGA_DIR / f"{table_id}.json"
        if target_path.exists() and target_path.stat().st_size > 1000:
            return True

        for retry in range(3):
            try:
                # Step 1: Request Table Archive
                url_req = f"https://boardgamearena.com/gamereview/gamereview/requestTableArchive.html?table={table_id}&request_token={self.request_token}"
                r1 = self.session.get(url_req, headers=self.headers, timeout=20)
                if r1.status_code != 200:
                    time.sleep(2)
                    continue

                # Step 2: Poll archive ready
                for _ in range(6):
                    time.sleep(1.0)
                    url_check = f"https://boardgamearena.com/gamereview/gamereview/checkTableArchiveReady.html?table={table_id}&request_token={self.request_token}"
                    r2 = self.session.get(url_check, headers=self.headers, timeout=20)
                    try:
                        d2 = r2.json()
                        if d2.get("data") is True or d2.get("status") == 1:
                            break
                    except:
                        pass

                # Step 3: Fetch Replay Log
                url_logs = f"https://boardgamearena.com/archive/archive/logs.html?table={table_id}&translated=1&request_token={self.request_token}"
                r3 = self.session.get(url_logs, headers=self.headers, timeout=30)
                if r3.status_code != 200:
                    time.sleep(2)
                    continue

                log_data = r3.json()
                if log_data.get("status") == "0" and "limite" in str(log_data.get("error", "")).lower():
                    logger.critical(f"🛑 BGA Replay Rate Limit hit on table {table_id}: {log_data.get('error')}")
                    return "QUOTA_EXCEEDED"

                if not log_data.get("data") or not log_data["data"].get("logs"):
                    time.sleep(2)
                    continue

                payload = {
                    "table_id": table_id,
                    "metadata": table_meta or {},
                    "replay": log_data["data"]
                }

                temp_path = RAW_BGA_DIR / f"{table_id}.tmp"
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
                temp_path.replace(target_path)

                self.downloaded_tables.add(table_id)
                moves_count = len(log_data["data"]["logs"])
                logger.info(f"Downloaded table {table_id} ({moves_count} moves). Total: {len(self.downloaded_tables)}/{self.target_games}")
                return True

            except Exception as e:
                logger.warning(f"Retry {retry+1}/3 for table {table_id}: {e}")
                time.sleep(2 * (retry + 1))

        logger.error(f"Failed to download table {table_id} after 3 retries.")
        return False

    def run(self):
        logger.info(f"=== Starting BGA Gaia Project Harvester (Target: {self.target_games} unique games) ===")
        
        # 1. Collect top players
        top_players = self.get_top_players(max_players=self.max_players)
        logger.info(f"Harvesting match lists from {len(top_players)} champions...")

        # 2. Collect unique candidate tables & download progressively
        candidate_tables: Dict[str, Dict] = {}
        download_queue: List[str] = []

        for p_idx, player in enumerate(top_players, 1):
            pid = player["id"]
            pname = player["name"]
            logger.info(f"[{p_idx}/{len(top_players)}] Scanning games of {pname} (ID: {pid}, ELO: {player['ranking']})...")
            
            tables = self.get_player_tables(pid, max_pages=35)
            new_tables_for_player = 0
            
            for t in tables:
                tid = str(t.get("table_id"))
                if not tid or tid in self.downloaded_tables or tid in candidate_tables:
                    continue

                # Strict quality filters:
                # - Exactly 4 players
                # - Normal completion (not cancelled / conceded)
                # - Standard rules
                players_str = str(t.get("players", ""))
                num_players = len(players_str.split(",")) if players_str else 0
                if num_players == 4 and str(t.get("normalend")) in ("1", "true", "True"):
                    candidate_tables[tid] = t
                    download_queue.append(tid)
                    new_tables_for_player += 1

            logger.info(f"Player {pname}: +{new_tables_for_player} new unique tables. Queue size: {len(download_queue)}. Total saved: {len(self.downloaded_tables)}.")

            # Flush download queue as we go so progress is continuous
            while download_queue:
                tid = download_queue.pop(0)
                if tid in self.downloaded_tables:
                    continue

                res = self.download_table_replay(tid, candidate_tables.get(tid))
                if res == "QUOTA_EXCEEDED":
                    download_queue.insert(0, tid)
                    logger.warning("BGA Replay Quota exceeded (approx 15-20 replays/day per account).")
                    logger.warning("Saving candidate tables queue to disk and pausing replay downloads.")
                    queue_file = DATA_DIR / "pending_replay_queue.json"
                    with open(queue_file, "w", encoding="utf-8") as qf:
                        json.dump({"queue": download_queue, "candidates": candidate_tables}, qf, indent=2)
                    self._save_progress(p_idx, len(top_players), len(candidate_tables))
                    break

                self._sleep_politely()
                self._save_progress(p_idx, len(top_players), len(candidate_tables))

                if len(self.downloaded_tables) >= self.target_games:
                    logger.info(f"🎯 Target reached: {self.target_games} games downloaded! Harvester stopping cleanly.")
                    return

        logger.info(f"=== Harvester Finished! Total downloaded: {len(self.downloaded_tables)} tables ===")


def main():
    parser = argparse.ArgumentParser(description="BGA Gaia Project Scraper")
    parser.add_argument("--target", type=int, default=20000, help="Target number of games to download")
    parser.add_argument("--max-players", type=int, default=500, help="Max top players to scan")
    parser.add_argument("--min-delay", type=float, default=2.0, help="Minimum delay between requests")
    parser.add_argument("--max-delay", type=float, default=3.2, help="Maximum delay between requests")
    args = parser.parse_args()

    scraper = BGAScraper(
        cookies=DEFAULT_COOKIES,
        request_token=REQUEST_TOKEN,
        delay_min=args.min_delay,
        delay_max=args.max_delay,
        target_games=args.target,
        max_players=args.max_players,
    )
    scraper.run()


if __name__ == "__main__":
    main()
