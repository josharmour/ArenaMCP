"""17lands draft statistics with CSV download and caching."""

import csv
import io
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Cache configuration
CACHE_DIR = Path.home() / ".arenamcp" / "cache" / "17lands"
CACHE_MAX_AGE_HOURS = 24
SEVENTEEN_LANDS_URL = "https://www.17lands.com/card_ratings/data"


@dataclass
class DraftStats:
    """17lands draft statistics for a card."""

    name: str
    set_code: str
    gih_wr: Optional[float]  # Games in Hand Win Rate (0.0-1.0)
    alsa: Optional[float]  # Average Last Seen At
    iwd: Optional[float]  # Improvement When Drawn
    games_in_hand: int  # Sample size for GIH WR


class DraftStatsCache:
    """17lands draft statistics with CSV download and caching.

    Downloads and caches 17lands card ratings by set, providing
    GIH WR, ALSA, and IWD metrics for draft card evaluation.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the cache.

        Args:
            cache_dir: Directory for cache files. Defaults to ~/.arenamcp/cache/17lands/
        """
        self._cache_dir = cache_dir or CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache: {set_code: {card_name_lower: DraftStats}}
        self._stats_cache: dict[str, dict[str, DraftStats]] = {}

    def _get_cache_path(self, set_code: str) -> Path:
        """Get path to the cached CSV file for a set."""
        return self._cache_dir / f"{set_code.upper()}_PremierDraft.csv"

    def _is_cache_stale(self, set_code: str) -> bool:
        """Check if the cache file is older than CACHE_MAX_AGE_HOURS."""
        cache_path = self._get_cache_path(set_code)
        if not cache_path.exists():
            return True

        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - mtime
        return age > timedelta(hours=CACHE_MAX_AGE_HOURS)

    def _parse_percentage(self, value: str) -> Optional[float]:
        """Parse a percentage string like '55.2%' to 0.552 float.

        Returns None for 'NA', empty strings, or unparseable values.
        """
        if not value or value.strip().upper() in ("NA", "N/A", ""):
            return None

        try:
            # Remove % sign and convert
            value = value.strip().rstrip("%")
            return float(value) / 100.0
        except (ValueError, TypeError):
            return None

    def _parse_float(self, value: str) -> Optional[float]:
        """Parse a float string, returning None for 'NA' or invalid."""
        if not value or value.strip().upper() in ("NA", "N/A", ""):
            return None

        try:
            return float(value.strip())
        except (ValueError, TypeError):
            return None

    def _parse_int(self, value: str) -> int:
        """Parse an int string, returning 0 for invalid."""
        if not value or value.strip().upper() in ("NA", "N/A", ""):
            return 0

        try:
            # Handle numbers with commas like "1,234"
            return int(value.strip().replace(",", ""))
        except (ValueError, TypeError):
            return 0

    def _download_set_data(self, set_code: str) -> str:
        """Download card ratings CSV from 17lands.

        Args:
            set_code: Set code like 'DSK', 'BLB', etc.

        Returns:
            Raw CSV content as string.

        Raises:
            requests.RequestException: If download fails.
        """
        url = f"{SEVENTEEN_LANDS_URL}?expansion={set_code.upper()}&format=PremierDraft"
        logger.info(f"Downloading 17lands data from {url}...")

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Save to cache
        cache_path = self._get_cache_path(set_code)
        cache_path.write_text(response.text, encoding="utf-8")
        logger.info(f"Cached 17lands data to {cache_path}")

        return response.text

    def _parse_csv(self, csv_content: str, set_code: str) -> dict[str, DraftStats]:
        """Parse 17lands CSV into DraftStats dict keyed by lowercase card name."""
        result: dict[str, DraftStats] = {}

        reader = csv.DictReader(io.StringIO(csv_content))

        for row in reader:
            name = row.get("Name", "").strip()
            if not name:
                continue

            stats = DraftStats(
                name=name,
                set_code=set_code.upper(),
                gih_wr=self._parse_percentage(row.get("GIH WR", "")),
                alsa=self._parse_float(row.get("ALSA", "")),
                iwd=self._parse_percentage(row.get("IWD", "")),
                games_in_hand=self._parse_int(row.get("# GIH", "")),
            )

            result[name.lower()] = stats

        logger.info(f"Parsed {len(result)} cards from 17lands data for {set_code}")
        return result

    def _load_set(self, set_code: str) -> dict[str, DraftStats]:
        """Load set data from cache or download if needed."""
        set_code = set_code.upper()

        # Check in-memory cache first
        if set_code in self._stats_cache:
            return self._stats_cache[set_code]

        # Check file cache
        cache_path = self._get_cache_path(set_code)
        if cache_path.exists() and not self._is_cache_stale(set_code):
            logger.info(f"Loading cached 17lands data from {cache_path}")
            csv_content = cache_path.read_text(encoding="utf-8")
        else:
            # Download fresh data
            csv_content = self._download_set_data(set_code)

        # Parse and cache in memory
        self._stats_cache[set_code] = self._parse_csv(csv_content, set_code)
        return self._stats_cache[set_code]

    def load_set(self, set_code: str) -> None:
        """Pre-load a set's data into memory.

        Args:
            set_code: Set code like 'DSK', 'BLB', etc.
        """
        self._load_set(set_code)

    def get_draft_rating(
        self, card_name: str, set_code: str
    ) -> Optional[DraftStats]:
        """Get draft statistics for a card.

        Args:
            card_name: Card name (case-insensitive)
            set_code: Set code like 'DSK', 'BLB', etc.

        Returns:
            DraftStats with GIH WR, ALSA, IWD metrics, or None if not found.
        """
        try:
            set_data = self._load_set(set_code)
        except requests.RequestException as e:
            logger.warning(f"Failed to load 17lands data for {set_code}: {e}")
            return None

        return set_data.get(card_name.lower())
