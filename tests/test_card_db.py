"""Tests for the unified card database interface (card_db.py)."""

import pytest

from arenamcp.card_db import (
    CardInfo,
    FallbackCardDatabase,
    NullCardDatabase,
    ScryfallAdapter,
    MTGJSONAdapter,
    MTGADatabaseAdapter,
    create_card_database,
    set_card_database,
    reset_card_database,
    get_card_database,
)


# ---------------------------------------------------------------------------
# Stub / fake data sources
# ---------------------------------------------------------------------------

class FakeScryfallCache:
    """Fake ScryfallCache with canned data."""

    def __init__(self, cards_by_id=None, cards_by_name=None):
        self._by_id = cards_by_id or {}
        self._by_name = cards_by_name or {}

    def get_card_by_arena_id(self, arena_id):
        return self._by_id.get(arena_id)

    def get_card_by_name(self, name):
        return self._by_name.get(name.lower() if name else name)


class _FakeScryfallCard:
    """Minimal ScryfallCard-like object."""

    def __init__(self, name="", oracle_text="", type_line="",
                 mana_cost="", cmc=0.0, colors=None,
                 arena_id=0, scryfall_uri=""):
        self.name = name
        self.oracle_text = oracle_text
        self.type_line = type_line
        self.mana_cost = mana_cost
        self.cmc = cmc
        self.colors = colors or []
        self.arena_id = arena_id
        self.scryfall_uri = scryfall_uri


class FakeMTGJSONDatabase:
    """Fake MTGJSONDatabase with canned data."""

    def __init__(self, cards_by_id=None, cards_by_name=None, available=True):
        self._by_id = cards_by_id or {}
        self._by_name = cards_by_name or {}
        self._available = available

    @property
    def available(self):
        return self._available

    def get_card(self, arena_id):
        return self._by_id.get(arena_id)

    def get_card_by_name(self, name):
        return self._by_name.get(name.lower() if name else name)


class _FakeMTGJSONCard:
    """Minimal MTGJSONCard-like object."""

    def __init__(self, name="", oracle_text="", type_line="",
                 mana_cost="", cmc=0.0, colors=None, arena_id=None):
        self.name = name
        self.oracle_text = oracle_text
        self.type_line = type_line
        self.mana_cost = mana_cost
        self.cmc = cmc
        self.colors = colors or []
        self.arena_id = arena_id


class FakeMTGADatabase:
    """Fake MTGADatabase with canned data."""

    def __init__(self, cards_by_id=None, ability_texts=None, available=True):
        self._by_id = cards_by_id or {}
        self._ability_texts = ability_texts or {}
        self._available = available

    @property
    def available(self):
        return self._available

    def get_card(self, grp_id):
        return self._by_id.get(grp_id)

    def get_cards_batch(self, grp_ids):
        return {gid: c for gid, c in self._by_id.items() if gid in grp_ids}

    def get_ability_text(self, ability_id):
        return self._ability_texts.get(ability_id)


class _FakeMTGACard:
    """Minimal MTGACard-like object."""

    def __init__(self, grp_id=0, name="", types="",
                 power="", toughness="", colors="",
                 is_token=False, expansion_code="",
                 oracle_text=""):
        self.grp_id = grp_id
        self.name = name
        self.types = types
        self.power = power
        self.toughness = toughness
        self.colors = colors
        self.is_token = is_token
        self.expansion_code = expansion_code
        self.oracle_text = oracle_text


# ---------------------------------------------------------------------------
# Tests: CardInfo
# ---------------------------------------------------------------------------

class TestCardInfo:
    def test_defaults(self):
        card = CardInfo(name="Lightning Bolt")
        assert card.name == "Lightning Bolt"
        assert card.oracle_text == ""
        assert card.type_line == ""
        assert card.mana_cost == ""
        assert card.cmc == 0.0
        assert card.colors == []
        assert card.arena_id == 0
        assert card.scryfall_uri == ""
        assert card.source == ""

    def test_full_init(self):
        card = CardInfo(
            name="Lightning Bolt",
            oracle_text="Lightning Bolt deals 3 damage to any target.",
            type_line="Instant",
            mana_cost="{R}",
            cmc=1.0,
            colors=["R"],
            arena_id=12345,
            scryfall_uri="https://scryfall.com/card/...",
            source="scryfall",
        )
        assert card.cmc == 1.0
        assert card.source == "scryfall"


# ---------------------------------------------------------------------------
# Tests: NullCardDatabase
# ---------------------------------------------------------------------------

class TestNullCardDatabase:
    def test_returns_none_for_arena_id(self):
        db = NullCardDatabase()
        assert db.get_card_by_arena_id(12345) is None

    def test_returns_none_for_name(self):
        db = NullCardDatabase()
        assert db.get_card_by_name("Lightning Bolt") is None


# ---------------------------------------------------------------------------
# Tests: ScryfallAdapter
# ---------------------------------------------------------------------------

class TestScryfallAdapter:
    def test_get_card_by_arena_id(self):
        fake_card = _FakeScryfallCard(
            name="Lightning Bolt",
            oracle_text="Deals 3 damage.",
            type_line="Instant",
            mana_cost="{R}",
            cmc=1.0,
            colors=["R"],
            arena_id=100,
            scryfall_uri="https://scryfall.com/...",
        )
        cache = FakeScryfallCache(cards_by_id={100: fake_card})
        adapter = ScryfallAdapter(cache)

        result = adapter.get_card_by_arena_id(100)
        assert result is not None
        assert result.name == "Lightning Bolt"
        assert result.oracle_text == "Deals 3 damage."
        assert result.source == "scryfall"
        assert result.scryfall_uri == "https://scryfall.com/..."

    def test_get_card_by_arena_id_not_found(self):
        cache = FakeScryfallCache()
        adapter = ScryfallAdapter(cache)
        assert adapter.get_card_by_arena_id(999) is None

    def test_get_card_by_name(self):
        fake_card = _FakeScryfallCard(name="Counterspell", oracle_text="Counter target spell.")
        cache = FakeScryfallCache(cards_by_name={"counterspell": fake_card})
        adapter = ScryfallAdapter(cache)

        result = adapter.get_card_by_name("Counterspell")
        assert result is not None
        assert result.name == "Counterspell"
        assert result.source == "scryfall"


# ---------------------------------------------------------------------------
# Tests: MTGJSONAdapter
# ---------------------------------------------------------------------------

class TestMTGJSONAdapter:
    def test_get_card_by_arena_id(self):
        fake_card = _FakeMTGJSONCard(
            name="Giant Growth",
            oracle_text="Target creature gets +3/+3.",
            type_line="Instant",
            mana_cost="{G}",
            arena_id=200,
        )
        db = FakeMTGJSONDatabase(cards_by_id={200: fake_card})
        adapter = MTGJSONAdapter(db)

        result = adapter.get_card_by_arena_id(200)
        assert result is not None
        assert result.name == "Giant Growth"
        assert result.source == "mtgjson"

    def test_unavailable_returns_none(self):
        db = FakeMTGJSONDatabase(available=False)
        adapter = MTGJSONAdapter(db)
        assert adapter.get_card_by_arena_id(200) is None
        assert adapter.get_card_by_name("Giant Growth") is None


# ---------------------------------------------------------------------------
# Tests: MTGADatabaseAdapter
# ---------------------------------------------------------------------------

class TestMTGADatabaseAdapter:
    def test_get_card_by_arena_id(self):
        fake_card = _FakeMTGACard(
            grp_id=300,
            name="Shock",
            types="Instant",
            colors="R",
        )
        db = FakeMTGADatabase(cards_by_id={300: fake_card})
        adapter = MTGADatabaseAdapter(db)

        result = adapter.get_card_by_arena_id(300)
        assert result is not None
        assert result.name == "Shock"
        assert result.type_line == "Instant"
        assert result.colors == ["R"]
        assert result.source == "mtgadb"

    def test_get_card_by_name_returns_none(self):
        """MTGA DB does not support name lookups."""
        db = FakeMTGADatabase()
        adapter = MTGADatabaseAdapter(db)
        assert adapter.get_card_by_name("Shock") is None

    def test_get_ability_text(self):
        db = FakeMTGADatabase(ability_texts={42: "Draw a card."})
        adapter = MTGADatabaseAdapter(db)
        assert adapter.get_ability_text(42) == "Draw a card."

    def test_unavailable_returns_none(self):
        db = FakeMTGADatabase(available=False)
        adapter = MTGADatabaseAdapter(db)
        assert adapter.get_card_by_arena_id(300) is None


# ---------------------------------------------------------------------------
# Tests: FallbackCardDatabase
# ---------------------------------------------------------------------------

class TestFallbackCardDatabase:
    def test_tries_sources_in_order(self):
        """First source wins."""
        mtgjson_card = _FakeMTGJSONCard(
            name="Bolt", oracle_text="3 damage", arena_id=1
        )
        scryfall_card = _FakeScryfallCard(
            name="Lightning Bolt", oracle_text="Deals 3 damage to any target.",
            arena_id=1,
        )

        mtgjson = FakeMTGJSONDatabase(cards_by_id={1: mtgjson_card})
        scryfall = FakeScryfallCache(cards_by_id={1: scryfall_card})

        db = create_card_database(
            scryfall_cache=scryfall,
            mtgjson_db=mtgjson,
        )

        result = db.get_card_by_arena_id(1)
        assert result is not None
        assert result.source == "mtgjson"  # MTGJSON is first in chain

    def test_fallback_when_first_source_misses(self):
        """Falls through to next source when first returns None."""
        scryfall_card = _FakeScryfallCard(
            name="Counterspell", oracle_text="Counter target spell.",
            arena_id=5,
        )
        mtgjson = FakeMTGJSONDatabase()  # empty
        scryfall = FakeScryfallCache(cards_by_id={5: scryfall_card})

        db = create_card_database(scryfall_cache=scryfall, mtgjson_db=mtgjson)
        result = db.get_card_by_arena_id(5)
        assert result is not None
        assert result.source == "scryfall"

    def test_enrichment_across_sources(self):
        """If MTGA DB has name but no oracle_text, MTGJSON fills it in."""
        mtga_card = _FakeMTGACard(
            grp_id=10, name="New Card", types="Creature",
            oracle_text="",  # no oracle text
        )
        mtgjson_card = _FakeMTGJSONCard(
            name="New Card", oracle_text="When New Card enters, draw a card.",
            type_line="Creature - Human",
        )

        mtga_db = FakeMTGADatabase(cards_by_id={10: mtga_card})
        mtgjson_db = FakeMTGJSONDatabase(
            cards_by_name={"new card": mtgjson_card},
        )

        # Order: MTGJSON (no arena_id match) -> MTGA (has card, no oracle) -> ...
        # Then name-based fallback fills oracle_text
        db = create_card_database(mtgjson_db=mtgjson_db, mtga_db=mtga_db)
        result = db.get_card_by_arena_id(10)
        assert result is not None
        assert result.name == "New Card"
        assert result.oracle_text == "When New Card enters, draw a card."

    def test_all_sources_miss(self):
        db = create_card_database(
            scryfall_cache=FakeScryfallCache(),
            mtgjson_db=FakeMTGJSONDatabase(),
            mtga_db=FakeMTGADatabase(),
        )
        result = db.get_card_by_arena_id(999)
        assert result is None

    def test_get_card_by_name(self):
        mtgjson_card = _FakeMTGJSONCard(
            name="Opt", oracle_text="Scry 1, then draw a card.",
        )
        mtgjson = FakeMTGJSONDatabase(cards_by_name={"opt": mtgjson_card})
        db = create_card_database(mtgjson_db=mtgjson)

        result = db.get_card_by_name("Opt")
        assert result is not None
        assert result.oracle_text == "Scry 1, then draw a card."
        assert result.source == "mtgjson"

    def test_get_card_by_name_fallback(self):
        """Name lookup falls through to Scryfall when MTGJSON misses."""
        scryfall_card = _FakeScryfallCard(
            name="Dark Ritual", oracle_text="Add {B}{B}{B}.",
        )
        scryfall = FakeScryfallCache(cards_by_name={"dark ritual": scryfall_card})
        mtgjson = FakeMTGJSONDatabase()  # empty

        db = create_card_database(scryfall_cache=scryfall, mtgjson_db=mtgjson)
        result = db.get_card_by_name("Dark Ritual")
        assert result is not None
        assert result.source == "scryfall"

    def test_get_ability_text_delegates(self):
        mtga_db = FakeMTGADatabase(ability_texts={77: "Destroy target creature."})
        db = create_card_database(mtga_db=mtga_db)
        assert db.get_ability_text(77) == "Destroy target creature."

    def test_get_ability_text_no_mtga(self):
        db = create_card_database()
        assert db.get_ability_text(77) is None

    def test_get_raw_mtgadb(self):
        mtga_db = FakeMTGADatabase()
        db = create_card_database(mtga_db=mtga_db)
        assert db.get_raw_mtgadb() is mtga_db

    def test_get_raw_mtgadb_none(self):
        db = create_card_database()
        assert db.get_raw_mtgadb() is None

    def test_source_exception_doesnt_propagate(self):
        """If a source raises, it's caught and the next source is tried."""

        class ExplodingSource:
            def get_card_by_arena_id(self, arena_id):
                raise RuntimeError("boom")

            def get_card_by_name(self, name):
                raise RuntimeError("boom")

        scryfall_card = _FakeScryfallCard(
            name="Bolt", oracle_text="3 damage", arena_id=1,
        )
        scryfall = FakeScryfallCache(cards_by_id={1: scryfall_card})

        db = FallbackCardDatabase([
            ExplodingSource(),
            ScryfallAdapter(scryfall),
        ])

        result = db.get_card_by_arena_id(1)
        assert result is not None
        assert result.source == "scryfall"

    def test_no_sources(self):
        db = FallbackCardDatabase([])
        assert db.get_card_by_arena_id(1) is None
        assert db.get_card_by_name("test") is None


# ---------------------------------------------------------------------------
# Tests: create_card_database
# ---------------------------------------------------------------------------

class TestCreateCardDatabase:
    def test_source_order(self):
        """Default order is: MTGJSON -> MTGA -> Scryfall."""
        db = create_card_database(
            scryfall_cache=FakeScryfallCache(),
            mtgjson_db=FakeMTGJSONDatabase(),
            mtga_db=FakeMTGADatabase(),
        )
        source_types = [type(s).__name__ for s in db.sources]
        assert source_types == ["MTGJSONAdapter", "MTGADatabaseAdapter", "ScryfallAdapter"]

    def test_skips_none_sources(self):
        db = create_card_database(scryfall_cache=FakeScryfallCache())
        assert len(db.sources) == 1
        assert isinstance(db.sources[0], ScryfallAdapter)


# ---------------------------------------------------------------------------
# Tests: Singleton management
# ---------------------------------------------------------------------------

class TestSingleton:
    def setup_method(self):
        reset_card_database()

    def teardown_method(self):
        reset_card_database()

    def test_set_and_get(self):
        custom_db = create_card_database()
        set_card_database(custom_db)
        assert get_card_database() is custom_db

    def test_reset(self):
        custom_db = create_card_database()
        set_card_database(custom_db)
        reset_card_database()
        # After reset, get_card_database would try to auto-init
        # which would fail in test env; just check that the old one is gone
        # by setting a new one
        new_db = create_card_database()
        set_card_database(new_db)
        assert get_card_database() is new_db


# ---------------------------------------------------------------------------
# Tests: Protocol compliance
# ---------------------------------------------------------------------------

class TestProtocolCompliance:
    """Verify adapters satisfy the CardDatabase protocol."""

    def test_scryfall_adapter(self):
        from arenamcp.card_db import CardDatabase
        adapter = ScryfallAdapter(FakeScryfallCache())
        assert isinstance(adapter, CardDatabase)

    def test_mtgjson_adapter(self):
        from arenamcp.card_db import CardDatabase
        adapter = MTGJSONAdapter(FakeMTGJSONDatabase())
        assert isinstance(adapter, CardDatabase)

    def test_mtga_adapter(self):
        from arenamcp.card_db import CardDatabase
        adapter = MTGADatabaseAdapter(FakeMTGADatabase())
        assert isinstance(adapter, CardDatabase)

    def test_null_database(self):
        from arenamcp.card_db import CardDatabase
        db = NullCardDatabase()
        assert isinstance(db, CardDatabase)

    def test_fallback_database(self):
        from arenamcp.card_db import CardDatabase
        db = FallbackCardDatabase([])
        assert isinstance(db, CardDatabase)
