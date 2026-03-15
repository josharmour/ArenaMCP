"""Unified card database interface with fallback chain.

Provides a common protocol for card lookups across multiple data sources
(MTGA local DB, MTGJSON, Scryfall) and a FallbackCardDatabase that tries
each source in order until a result is found.
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class CardInfo:
    """Unified card data returned by all card database implementations.

    This is the common currency for card lookups across all sources.
    Fields that a source cannot provide are left as their defaults.
    """

    name: str
    oracle_text: str = ""
    type_line: str = ""
    mana_cost: str = ""
    cmc: float = 0.0
    colors: list[str] = field(default_factory=list)
    arena_id: int = 0
    scryfall_uri: str = ""
    source: str = ""  # Which database provided this result


@runtime_checkable
class CardDatabase(Protocol):
    """Protocol for card database implementations.

    Any class implementing get_card_by_arena_id and get_card_by_name
    with the correct signatures satisfies this protocol.
    """

    def get_card_by_arena_id(self, arena_id: int) -> Optional[CardInfo]:
        """Look up a card by MTGA arena_id / grp_id.

        Args:
            arena_id: The MTGA arena ID of the card.

        Returns:
            CardInfo if found, None otherwise.
        """
        ...

    def get_card_by_name(self, name: str) -> Optional[CardInfo]:
        """Look up a card by name.

        Args:
            name: The card name to search for.

        Returns:
            CardInfo if found, None otherwise.
        """
        ...


class ScryfallAdapter:
    """Adapts ScryfallCache to the CardDatabase protocol."""

    def __init__(self, scryfall_cache: Any) -> None:
        self._cache = scryfall_cache

    def get_card_by_arena_id(self, arena_id: int) -> Optional[CardInfo]:
        card = self._cache.get_card_by_arena_id(arena_id)
        if card is None:
            return None
        return CardInfo(
            name=card.name or "",
            oracle_text=card.oracle_text or "",
            type_line=card.type_line or "",
            mana_cost=card.mana_cost or "",
            cmc=card.cmc or 0.0,
            colors=card.colors or [],
            arena_id=card.arena_id or 0,
            scryfall_uri=card.scryfall_uri or "",
            source="scryfall",
        )

    def get_card_by_name(self, name: str) -> Optional[CardInfo]:
        card = self._cache.get_card_by_name(name)
        if card is None:
            return None
        return CardInfo(
            name=card.name or "",
            oracle_text=card.oracle_text or "",
            type_line=card.type_line or "",
            mana_cost=card.mana_cost or "",
            cmc=card.cmc or 0.0,
            colors=card.colors or [],
            arena_id=card.arena_id or 0,
            scryfall_uri=card.scryfall_uri or "",
            source="scryfall",
        )


class MTGJSONAdapter:
    """Adapts MTGJSONDatabase to the CardDatabase protocol."""

    def __init__(self, mtgjson_db: Any) -> None:
        self._db = mtgjson_db

    def get_card_by_arena_id(self, arena_id: int) -> Optional[CardInfo]:
        if not self._db.available:
            return None
        # MTGJSONDatabase uses get_card() for arena_id lookups
        card = self._db.get_card(arena_id)
        if card is None:
            return None
        return CardInfo(
            name=card.name or "",
            oracle_text=card.oracle_text or "",
            type_line=card.type_line or "",
            mana_cost=card.mana_cost or "",
            cmc=card.cmc or 0.0,
            colors=card.colors or [],
            arena_id=card.arena_id or 0,
            source="mtgjson",
        )

    def get_card_by_name(self, name: str) -> Optional[CardInfo]:
        if not self._db.available:
            return None
        card = self._db.get_card_by_name(name)
        if card is None:
            return None
        return CardInfo(
            name=card.name or "",
            oracle_text=card.oracle_text or "",
            type_line=card.type_line or "",
            mana_cost=card.mana_cost or "",
            cmc=card.cmc or 0.0,
            colors=card.colors or [],
            arena_id=card.arena_id or 0,
            source="mtgjson",
        )


class MTGADatabaseAdapter:
    """Adapts MTGADatabase to the CardDatabase protocol.

    Also exposes MTGA-specific features (ability text, batch lookups)
    that don't fit the generic protocol.
    """

    def __init__(self, mtga_db: Any) -> None:
        self._db = mtga_db

    @property
    def available(self) -> bool:
        """Check if the underlying MTGA database is available."""
        return self._db.available

    def get_card_by_arena_id(self, arena_id: int) -> Optional[CardInfo]:
        if not self._db.available:
            return None
        card = self._db.get_card(arena_id)
        if card is None:
            return None
        return CardInfo(
            name=card.name or "",
            oracle_text=card.oracle_text or "",
            type_line=card.types or "",
            mana_cost="",  # MTGA DB doesn't store mana cost
            cmc=0.0,
            colors=card.colors.split(",") if card.colors else [],
            arena_id=card.grp_id,
            source="mtgadb",
        )

    def get_card_by_name(self, name: str) -> Optional[CardInfo]:
        """MTGA local DB does not support name-based lookups."""
        return None

    def get_ability_text(self, ability_id: int) -> Optional[str]:
        """Look up text for an ability ID (MTGA-specific)."""
        return self._db.get_ability_text(ability_id)

    def get_cards_batch(self, grp_ids: list[int]) -> dict[int, Any]:
        """Batch lookup (MTGA-specific). Returns raw MTGACard objects."""
        return self._db.get_cards_batch(grp_ids)

    @property
    def raw(self) -> Any:
        """Access the underlying MTGADatabase for MTGA-specific operations."""
        return self._db


class NullCardDatabase:
    """No-op card database used when all initializations fail."""

    def get_card_by_arena_id(self, arena_id: int) -> Optional[CardInfo]:
        return None

    def get_card_by_name(self, name: str) -> Optional[CardInfo]:
        return None


class FallbackCardDatabase:
    """Card database that tries multiple sources in order.

    The default fallback chain is:
    1. MTGJSON (most complete oracle text, updated daily)
    2. MTGA local DB (has newest cards, tokens, digital-only)
    3. Scryfall (API fallback, rate-limited)

    For arena_id lookups, sources are tried in order. If a source returns
    a card without oracle_text, additional sources are consulted to enrich it.
    Name-based lookups follow the same pattern.
    """

    def __init__(self, sources: Optional[list[CardDatabase]] = None) -> None:
        """Initialize with a list of CardDatabase sources.

        Args:
            sources: Ordered list of card databases to try. If None,
                     must be set later via set_sources().
        """
        self._sources: list[CardDatabase] = sources or []
        # Keep typed references for source-specific features
        self._mtga_adapter: Optional[MTGADatabaseAdapter] = None
        for src in self._sources:
            if isinstance(src, MTGADatabaseAdapter):
                self._mtga_adapter = src
                break

    @property
    def sources(self) -> list[CardDatabase]:
        """Return the list of configured sources."""
        return list(self._sources)

    def get_card_by_arena_id(self, arena_id: int) -> Optional[CardInfo]:
        """Look up a card by arena_id, trying each source in order.

        If a source returns a result without oracle_text, continues
        checking remaining sources and merges in the oracle_text.
        """
        best: Optional[CardInfo] = None

        for source in self._sources:
            try:
                result = source.get_card_by_arena_id(arena_id)
            except Exception as exc:
                logger.debug(
                    "Card lookup failed in %s for arena_id %d: %s",
                    type(source).__name__, arena_id, exc,
                )
                continue

            if result is None:
                continue

            if best is None:
                best = result

            # If we already have oracle_text, we're done
            if best.oracle_text:
                return best

            # Merge oracle_text from this source into best
            if result.oracle_text and not best.oracle_text:
                best.oracle_text = result.oracle_text
                # Also pull in other fields if they were missing
                if not best.type_line and result.type_line:
                    best.type_line = result.type_line
                if not best.mana_cost and result.mana_cost:
                    best.mana_cost = result.mana_cost
                if not best.cmc and result.cmc:
                    best.cmc = result.cmc
                if not best.colors and result.colors:
                    best.colors = result.colors
                if not best.scryfall_uri and result.scryfall_uri:
                    best.scryfall_uri = result.scryfall_uri

        # If we have a name but no oracle_text, try name-based lookup as last resort
        if best and not best.oracle_text and best.name and not best.name.startswith("Unknown"):
            name_result = self.get_card_by_name(best.name)
            if name_result and name_result.oracle_text:
                best.oracle_text = name_result.oracle_text
                if not best.type_line and name_result.type_line:
                    best.type_line = name_result.type_line
                if not best.mana_cost and name_result.mana_cost:
                    best.mana_cost = name_result.mana_cost
                if not best.cmc and name_result.cmc:
                    best.cmc = name_result.cmc
                if not best.colors and name_result.colors:
                    best.colors = name_result.colors

        return best

    def get_card_by_name(self, name: str) -> Optional[CardInfo]:
        """Look up a card by name, trying each source in order."""
        for source in self._sources:
            try:
                result = source.get_card_by_name(name)
            except Exception as exc:
                logger.debug(
                    "Card name lookup failed in %s for '%s': %s",
                    type(source).__name__, name, exc,
                )
                continue

            if result is not None:
                return result

        return None

    def get_ability_text(self, ability_id: int) -> Optional[str]:
        """Look up ability text (delegates to MTGA adapter if available)."""
        if self._mtga_adapter:
            return self._mtga_adapter.get_ability_text(ability_id)
        return None

    def get_raw_mtgadb(self) -> Optional[Any]:
        """Get the underlying MTGADatabase for MTGA-specific operations.

        Returns None if MTGA database is not available.
        """
        if self._mtga_adapter:
            return self._mtga_adapter.raw
        return None


def create_card_database(
    scryfall_cache: Optional[Any] = None,
    mtgjson_db: Optional[Any] = None,
    mtga_db: Optional[Any] = None,
) -> FallbackCardDatabase:
    """Create a FallbackCardDatabase with the standard source order.

    The default order is: MTGJSON -> MTGA local DB -> Scryfall.
    Sources that are None are skipped.

    Args:
        scryfall_cache: A ScryfallCache instance (or _NullScryfallCache).
        mtgjson_db: A MTGJSONDatabase instance.
        mtga_db: A MTGADatabase instance.

    Returns:
        Configured FallbackCardDatabase.
    """
    sources: list[CardDatabase] = []

    if mtgjson_db is not None:
        sources.append(MTGJSONAdapter(mtgjson_db))

    if mtga_db is not None:
        sources.append(MTGADatabaseAdapter(mtga_db))

    if scryfall_cache is not None:
        sources.append(ScryfallAdapter(scryfall_cache))

    return FallbackCardDatabase(sources)


# ---------------------------------------------------------------------------
# Module-level lazy singleton
# ---------------------------------------------------------------------------
_card_db: Optional[FallbackCardDatabase] = None
_card_db_lock = threading.Lock()


def get_card_database() -> FallbackCardDatabase:
    """Get or create the global FallbackCardDatabase singleton.

    Lazily initializes all three card sources on first call:
      - MTGJSON (via get_mtgjson())
      - MTGA local DB (via MTGADatabase())
      - Scryfall (via ScryfallCache(), with graceful fallback)

    Thread-safe.
    """
    global _card_db
    if _card_db is not None:
        return _card_db

    with _card_db_lock:
        if _card_db is not None:
            return _card_db

        logger.info("Initializing unified card database...")

        # -- MTGJSON --
        mtgjson_db = None
        try:
            from arenamcp.mtgjson import get_mtgjson
            mtgjson_db = get_mtgjson()
        except Exception as e:
            logger.warning(f"MTGJSON init failed: {e}")

        # -- MTGA local DB --
        mtga_db = None
        try:
            from arenamcp.mtgadb import MTGADatabase
            mtga_db = MTGADatabase()
        except Exception as e:
            logger.warning(f"MTGA database init failed: {e}")

        # -- Scryfall --
        scryfall_cache = None
        try:
            from arenamcp.scryfall import ScryfallCache
            scryfall_cache = ScryfallCache()
        except Exception as e:
            logger.warning(f"Scryfall init failed: {e}")

        _card_db = create_card_database(
            scryfall_cache=scryfall_cache,
            mtgjson_db=mtgjson_db,
            mtga_db=mtga_db,
        )
        logger.info(
            "Unified card database ready with %d sources: %s",
            len(_card_db.sources),
            [type(s).__name__ for s in _card_db.sources],
        )
        return _card_db


def set_card_database(db: FallbackCardDatabase) -> None:
    """Replace the global card database singleton (for testing)."""
    global _card_db
    _card_db = db


def reset_card_database() -> None:
    """Reset the global card database singleton (for testing)."""
    global _card_db
    _card_db = None
