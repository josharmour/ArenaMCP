"""Tests for RE-derived battlefield permanent positioning.

Validates that get_permanent_coord() and its helpers produce coordinates
consistent with the decompiled MTGA BattlefieldLayout_MP.GenerateData()
four-region layout: Creatures, Planeswalkers/Sagas, Lands, Artifacts.

Cards are sorted by instance_id (age proxy) within each row, and
attached cards (auras) are positioned on their parent rather than
occupying independent row space.
"""

import pytest

from arenamcp.screen_mapper import ScreenMapper, ScreenCoord


@pytest.fixture
def mapper():
    return ScreenMapper()


# ---------------------------------------------------------------------------
# Helpers to build card dicts
# ---------------------------------------------------------------------------

def _card(
    name: str,
    instance_id: int,
    owner_seat_id: int = 1,
    card_types: list[str] | None = None,
    subtypes: list[str] | None = None,
    type_line: str = "",
    parent_instance_id: int | None = None,
) -> dict:
    d = {
        "name": name,
        "instance_id": instance_id,
        "owner_seat_id": owner_seat_id,
        "card_types": card_types or [],
        "subtypes": subtypes or [],
        "type_line": type_line,
    }
    if parent_instance_id is not None:
        d["parent_instance_id"] = parent_instance_id
    return d


def _creature(name, iid, owner=1, **kw):
    return _card(name, iid, owner, card_types=["CardType_Creature"], **kw)


def _land(name, iid, owner=1, **kw):
    return _card(name, iid, owner, card_types=["CardType_Land"], **kw)


def _artifact(name, iid, owner=1, **kw):
    return _card(name, iid, owner, card_types=["CardType_Artifact"], **kw)


def _enchantment(name, iid, owner=1, **kw):
    return _card(name, iid, owner, card_types=["CardType_Enchantment"], **kw)


def _planeswalker(name, iid, owner=1, **kw):
    return _card(name, iid, owner, card_types=["CardType_Planeswalker"], **kw)


def _saga(name, iid, owner=1, **kw):
    return _card(
        name, iid, owner,
        card_types=["CardType_Enchantment"],
        subtypes=["Saga"],
        **kw,
    )


def _aura(name, iid, owner=1, parent_instance_id=None, **kw):
    return _card(
        name, iid, owner,
        card_types=["CardType_Enchantment"],
        type_line="Enchantment - Aura",
        parent_instance_id=parent_instance_id,
        **kw,
    )


# ===================================================================
# Region classification
# ===================================================================

class TestClassifyRegion:
    def test_creature(self, mapper):
        c = _creature("Bear", 1)
        assert mapper._classify_region(c) == "creature"

    def test_land(self, mapper):
        c = _land("Forest", 1)
        assert mapper._classify_region(c) == "land"

    def test_artifact(self, mapper):
        c = _artifact("Sol Ring", 1)
        assert mapper._classify_region(c) == "artifact"

    def test_enchantment_goes_to_artifact_row(self, mapper):
        c = _enchantment("Propaganda", 1)
        assert mapper._classify_region(c) == "artifact"

    def test_planeswalker(self, mapper):
        c = _planeswalker("Jace", 1)
        assert mapper._classify_region(c) == "planeswalker"

    def test_saga_goes_to_planeswalker_row(self, mapper):
        c = _saga("History of Benalia", 1)
        assert mapper._classify_region(c) == "planeswalker"

    def test_class_subtype_goes_to_planeswalker_row(self, mapper):
        c = _card("Ranger Class", 1, subtypes=["Class"],
                   card_types=["CardType_Enchantment"])
        assert mapper._classify_region(c) == "planeswalker"

    def test_case_subtype_goes_to_planeswalker_row(self, mapper):
        c = _card("Case of the Shattered Pact", 1, subtypes=["Case"],
                   card_types=["CardType_Enchantment"])
        assert mapper._classify_region(c) == "planeswalker"

    def test_room_subtype_goes_to_planeswalker_row(self, mapper):
        c = _card("Spooky Room", 1, subtypes=["Room"],
                   card_types=["CardType_Enchantment"])
        assert mapper._classify_region(c) == "planeswalker"

    def test_battle_type(self, mapper):
        c = _card("Invasion of Zendikar", 1,
                   card_types=["CardType_Battle"])
        assert mapper._classify_region(c) == "planeswalker"

    def test_creature_takes_priority_over_other_types(self, mapper):
        """A creature-enchantment (e.g., Bestow) is classified as creature."""
        c = _card("Nighthowler", 1,
                   card_types=["CardType_Creature", "CardType_Enchantment"])
        assert mapper._classify_region(c) == "creature"

    def test_type_line_fallback_creature(self, mapper):
        c = _card("Unknown Bear", 1, type_line="Creature - Bear")
        assert mapper._classify_region(c) == "creature"

    def test_type_line_fallback_land(self, mapper):
        c = _card("Unknown Land", 1, type_line="Land")
        assert mapper._classify_region(c) == "land"

    def test_type_line_fallback_planeswalker(self, mapper):
        c = _card("Unknown PW", 1, type_line="Legendary Planeswalker - Jace")
        assert mapper._classify_region(c) == "planeswalker"


# ===================================================================
# Attachment detection
# ===================================================================

class TestIsAttached:
    def test_aura_with_parent_id(self, mapper):
        c = _aura("Pacifism", 10, parent_instance_id=5)
        assert mapper._is_attached(c) is True

    def test_aura_without_parent_id(self, mapper):
        """Aura type_line fallback when parent_instance_id not yet set."""
        c = _aura("Pacifism", 10, parent_instance_id=None)
        assert mapper._is_attached(c) is True

    def test_non_aura_with_parent_id(self, mapper):
        """Equipment attached via parentId."""
        c = _artifact("Sword of F&I", 10, parent_instance_id=5)
        assert mapper._is_attached(c) is True

    def test_free_creature(self, mapper):
        c = _creature("Bear", 10)
        assert mapper._is_attached(c) is False

    def test_free_land(self, mapper):
        c = _land("Forest", 10)
        assert mapper._is_attached(c) is False


# ===================================================================
# Row X positioning
# ===================================================================

class TestBattlefieldRowPositions:
    def test_single_card_centered(self, mapper):
        xs = mapper._battlefield_row_positions(1)
        assert len(xs) == 1
        assert xs[0] == pytest.approx(0.50, abs=0.01)

    def test_empty_row(self, mapper):
        assert mapper._battlefield_row_positions(0) == []

    def test_positions_left_to_right(self, mapper):
        for n in [2, 3, 5, 8, 12]:
            xs = mapper._battlefield_row_positions(n)
            assert len(xs) == n
            for i in range(len(xs) - 1):
                assert xs[i] < xs[i + 1], (
                    f"n={n}: x[{i}]={xs[i]:.4f} >= x[{i+1}]={xs[i+1]:.4f}"
                )

    def test_positions_symmetric(self, mapper):
        for n in [3, 5, 7]:
            xs = mapper._battlefield_row_positions(n)
            for i in range(n // 2):
                assert xs[i] + xs[n - 1 - i] == pytest.approx(1.0, abs=0.01), (
                    f"n={n}: symmetry broken at i={i}"
                )

    def test_positions_within_bounds(self, mapper):
        for n in [1, 2, 5, 10, 15, 20]:
            xs = mapper._battlefield_row_positions(n)
            for i, x in enumerate(xs):
                assert 0.05 <= x <= 0.95, (
                    f"n={n}, i={i}: x={x:.4f} out of bounds"
                )

    def test_more_cards_wider_span(self, mapper):
        xs_3 = mapper._battlefield_row_positions(3)
        xs_8 = mapper._battlefield_row_positions(8)
        span_3 = xs_3[-1] - xs_3[0]
        span_8 = xs_8[-1] - xs_8[0]
        assert span_8 > span_3


# ===================================================================
# Sorting by age (instance_id)
# ===================================================================

class TestSortByAge:
    def test_sorted_by_instance_id(self, mapper):
        cards = [
            _creature("C", 30),
            _creature("A", 10),
            _creature("B", 20),
        ]
        result = mapper._sort_by_age(cards)
        assert [c["name"] for c in result] == ["A", "B", "C"]


# ===================================================================
# Full get_permanent_coord tests
# ===================================================================

class TestGetPermanentCoord:

    def test_single_creature_centered(self, mapper):
        bf = [_creature("Bear", 100)]
        coord = mapper.get_permanent_coord("Bear", 100, bf, 1, 1)
        assert coord is not None
        assert coord.x == pytest.approx(0.50, abs=0.01)
        assert coord.y == pytest.approx(0.58, abs=0.02)

    def test_opponent_creature_higher(self, mapper):
        bf = [_creature("Goblin", 200, owner=2)]
        coord = mapper.get_permanent_coord("Goblin", 200, bf, 2, 1)
        assert coord is not None
        assert coord.y == pytest.approx(0.38, abs=0.02)

    def test_land_in_land_row(self, mapper):
        bf = [_land("Forest", 100)]
        coord = mapper.get_permanent_coord("Forest", 100, bf, 1, 1)
        assert coord is not None
        assert coord.y == pytest.approx(0.78, abs=0.02)

    def test_opponent_land_row(self, mapper):
        bf = [_land("Island", 100, owner=2)]
        coord = mapper.get_permanent_coord("Island", 100, bf, 2, 1)
        assert coord is not None
        assert coord.y == pytest.approx(0.14, abs=0.02)

    def test_artifact_in_artifact_row(self, mapper):
        bf = [_artifact("Sol Ring", 100)]
        coord = mapper.get_permanent_coord("Sol Ring", 100, bf, 1, 1)
        assert coord is not None
        assert coord.y == pytest.approx(0.72, abs=0.02)

    def test_planeswalker_in_planeswalker_row(self, mapper):
        bf = [_planeswalker("Jace", 100)]
        coord = mapper.get_permanent_coord("Jace", 100, bf, 1, 1)
        assert coord is not None
        assert coord.y == pytest.approx(0.66, abs=0.02)

    def test_saga_in_planeswalker_row(self, mapper):
        bf = [_saga("History of Benalia", 100)]
        coord = mapper.get_permanent_coord("History of Benalia", 100, bf, 1, 1)
        assert coord is not None
        assert coord.y == pytest.approx(0.66, abs=0.02)

    def test_creature_row_ordered_by_age(self, mapper):
        """Oldest creature (lowest instance_id) should be leftmost."""
        bf = [
            _creature("Bear C", 300),
            _creature("Bear A", 100),
            _creature("Bear B", 200),
        ]
        c_a = mapper.get_permanent_coord("Bear A", 100, bf, 1, 1)
        c_b = mapper.get_permanent_coord("Bear B", 200, bf, 1, 1)
        c_c = mapper.get_permanent_coord("Bear C", 300, bf, 1, 1)
        assert c_a.x < c_b.x < c_c.x

    def test_creatures_and_lands_separate_rows(self, mapper):
        """Creatures and lands must be in distinct Y rows."""
        bf = [
            _creature("Bear", 100),
            _land("Forest", 200),
        ]
        creature_coord = mapper.get_permanent_coord("Bear", 100, bf, 1, 1)
        land_coord = mapper.get_permanent_coord("Forest", 200, bf, 1, 1)
        # Creature row should be closer to center (lower y) than land row
        assert creature_coord.y < land_coord.y

    def test_four_regions_distinct_y(self, mapper):
        """All four region types should get distinct Y values."""
        bf = [
            _creature("Bear", 100),
            _planeswalker("Jace", 200),
            _artifact("Ring", 300),
            _land("Forest", 400),
        ]
        ys = set()
        for name, iid in [("Bear", 100), ("Jace", 200), ("Ring", 300), ("Forest", 400)]:
            coord = mapper.get_permanent_coord(name, iid, bf, 1, 1)
            assert coord is not None
            ys.add(round(coord.y, 2))
        assert len(ys) == 4, f"Expected 4 distinct Y values, got {ys}"

    def test_aura_excluded_from_row_spacing(self, mapper):
        """An aura with parent_instance_id should not affect row positions."""
        bf = [
            _creature("Bear", 100),
            _creature("Wolf", 200),
            _aura("Pacifism", 300, parent_instance_id=100),
        ]
        # Bear + Wolf = 2 creatures. Aura should not add a 3rd position.
        c_bear = mapper.get_permanent_coord("Bear", 100, bf, 1, 1)
        c_wolf = mapper.get_permanent_coord("Wolf", 200, bf, 1, 1)
        assert c_bear is not None
        assert c_wolf is not None

        # Aura should resolve to its parent's position (Bear)
        c_aura = mapper.get_permanent_coord("Pacifism", 300, bf, 1, 1)
        assert c_aura is not None
        assert c_aura.x == pytest.approx(c_bear.x, abs=0.01)
        assert c_aura.y == pytest.approx(c_bear.y, abs=0.01)

    def test_card_not_found(self, mapper):
        bf = [_creature("Bear", 100)]
        result = mapper.get_permanent_coord("Nonexistent", None, bf, 1, 1)
        assert result is None

    def test_partial_name_match(self, mapper):
        bf = [_creature("Grizzly Bears", 100)]
        coord = mapper.get_permanent_coord("Grizzly", None, bf, 1, 1)
        assert coord is not None
        assert "Grizzly Bears" in coord.description or "Grizzly" in coord.description

    def test_instance_id_match_preferred(self, mapper):
        """When two cards have the same name, instance_id should disambiguate."""
        bf = [
            _creature("Bear", 100),
            _creature("Bear", 200),
        ]
        c1 = mapper.get_permanent_coord("Bear", 100, bf, 1, 1)
        c2 = mapper.get_permanent_coord("Bear", 200, bf, 1, 1)
        # Card with instance_id=100 is older, so it should be leftward
        assert c1.x < c2.x

    def test_mixed_board_many_cards(self, mapper):
        """Stress test: 15 permanents across all four regions."""
        bf = []
        for i in range(5):
            bf.append(_creature(f"Creature_{i}", 100 + i))
        for i in range(4):
            bf.append(_land(f"Land_{i}", 200 + i))
        for i in range(3):
            bf.append(_artifact(f"Artifact_{i}", 300 + i))
        for i in range(3):
            bf.append(_planeswalker(f"PW_{i}", 400 + i))

        # Every card should be locatable
        for card in bf:
            coord = mapper.get_permanent_coord(
                card["name"], card["instance_id"], bf, 1, 1
            )
            assert coord is not None, f"{card['name']} not found"
            assert 0.05 <= coord.x <= 0.95
            assert 0.10 <= coord.y <= 0.90

    def test_opponent_rows_mirrored(self, mapper):
        """Opponent's rows should be in reversed vertical order."""
        bf = [
            _creature("Opp Bear", 100, owner=2),
            _land("Opp Forest", 200, owner=2),
        ]
        creature = mapper.get_permanent_coord("Opp Bear", 100, bf, 2, 1)
        land = mapper.get_permanent_coord("Opp Forest", 200, bf, 2, 1)
        # Opponent creature row is closer to center (higher y) than land row
        assert creature.y > land.y

    def test_enchantment_in_artifact_row(self, mapper):
        """Non-saga enchantments go to the artifact region."""
        bf = [_enchantment("Propaganda", 100)]
        coord = mapper.get_permanent_coord("Propaganda", 100, bf, 1, 1)
        assert coord is not None
        assert coord.y == pytest.approx(0.72, abs=0.02)

    def test_creature_enchantment_in_creature_row(self, mapper):
        """Creature-enchantment (bestow) should be in creature row."""
        bf = [_card("Nighthowler", 100,
                     card_types=["CardType_Creature", "CardType_Enchantment"])]
        coord = mapper.get_permanent_coord("Nighthowler", 100, bf, 1, 1)
        assert coord is not None
        assert coord.y == pytest.approx(0.58, abs=0.02)

    def test_empty_battlefield(self, mapper):
        coord = mapper.get_permanent_coord("Bear", 100, [], 1, 1)
        assert coord is None


# ===================================================================
# Y-ordering invariant: per side, rows must maintain relative order
# ===================================================================

class TestYRowOrdering:
    """Verify that Y coordinates follow the MTGA visual layout order."""

    def test_local_rows_top_to_bottom(self, mapper):
        """Local player: creature < planeswalker < artifact < land (in Y)."""
        assert (
            mapper._ROW_Y[(True, "creature")]
            < mapper._ROW_Y[(True, "planeswalker")]
            < mapper._ROW_Y[(True, "artifact")]
            < mapper._ROW_Y[(True, "land")]
        )

    def test_opponent_rows_top_to_bottom(self, mapper):
        """Opponent: land < artifact < planeswalker < creature (in Y)."""
        assert (
            mapper._ROW_Y[(False, "land")]
            < mapper._ROW_Y[(False, "artifact")]
            < mapper._ROW_Y[(False, "planeswalker")]
            < mapper._ROW_Y[(False, "creature")]
        )

    def test_local_creatures_below_opponent_creatures(self, mapper):
        """Local creature row should be below (higher Y) than opponent's."""
        assert (
            mapper._ROW_Y[(True, "creature")]
            > mapper._ROW_Y[(False, "creature")]
        )
