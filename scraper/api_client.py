"""
API client for Boardgamers.space (BGS) with rate limiting and retry handling.
"""

import logging
import time
from typing import Any, Dict, List, Optional
import requests

from .config import ScraperConfig, DEFAULT_CONFIG

logger = logging.getLogger("scraper.api")


class BGSApiClient:
    """Robust HTTP client for Boardgamers.space REST API."""

    def __init__(self, config: Optional[ScraperConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.config.user_agent,
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
        })
        self._last_request_time = 0.0

    def _wait_rate_limit(self) -> None:
        """Enforces a polite delay between successive outgoing requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.config.request_delay:
            time.sleep(self.config.request_delay - elapsed)
        self._last_request_time = time.time()

    def _request(self, method: str, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Executes HTTP request with exponential backoff on rate limits or network issues."""
        retries = 0
        backoff = 1.0

        while retries <= self.config.max_retries:
            self._wait_rate_limit()
            try:
                resp = self.session.request(method, url, params=params, timeout=15)
                
                # Check for rate limiting
                if resp.status_code == 429:
                    retries += 1
                    retry_after = resp.headers.get("Retry-After")
                    sleep_time = float(retry_after) if retry_after else (backoff * self.config.backoff_factor)
                    logger.warning(
                        f"Rate limited (HTTP 429) on {url}. Retrying in {sleep_time:.1f}s (attempt {retries}/{self.config.max_retries})..."
                    )
                    time.sleep(sleep_time)
                    backoff *= self.config.backoff_factor
                    continue

                if resp.status_code >= 500:
                    retries += 1
                    logger.warning(
                        f"Server error {resp.status_code} on {url}. Retrying in {backoff:.1f}s..."
                    )
                    time.sleep(backoff)
                    backoff *= self.config.backoff_factor
                    continue

                resp.raise_for_status()
                return resp.json()

            except (requests.ConnectionError, requests.Timeout) as e:
                retries += 1
                if retries > self.config.max_retries:
                    raise
                logger.warning(
                    f"Network error ({e.__class__.__name__}) on {url}. Retrying in {backoff:.1f}s (attempt {retries}/{self.config.max_retries})..."
                )
                time.sleep(backoff)
                backoff *= self.config.backoff_factor

        raise RuntimeError(f"Max retries exceeded for {url}")

    def get_ended_games(self, skip: int = 0, count: int = 50) -> List[Dict[str, Any]]:
        """
        Fetches a page of ended games.
        Endpoint: /api/game/status/ended?count={count}&skip={skip}&boardgame={boardgame}
        """
        url = f"{self.config.base_url}/api/game/status/ended"
        params = {
            "count": count,
            "skip": skip,
            "boardgame": self.config.game_boardgame,
        }
        res = self._request("GET", url, params=params)
        return res if isinstance(res, list) else []

    def get_gameplay(self, game_id: str) -> Dict[str, Any]:
        """
        Fetches the complete gameplay data for a game, including move history and final state.
        Endpoint: /api/gameplay/{game_id}
        """
        url = f"{self.config.base_url}/api/gameplay/{game_id}"
        return self._request("GET", url)

    def get_rankings(self, count: int = 50, offset: int = 0) -> Dict[str, Any]:
        """
        Fetches player leaderboard.
        Endpoint: /api/boardgame/{boardgame}/elo/page?count={count}&offset={offset}
        """
        url = f"{self.config.base_url}/api/boardgame/{self.config.game_boardgame}/elo/page"
        params = {"count": count, "offset": offset}
        return self._request("GET", url, params=params)
