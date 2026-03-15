"""Tests for arc-based hand card positioning.

Validates that _hand_arc_positions() produces coordinates consistent with
the decompiled MTGA CardLayout_Hand.cs layout: cards on a circular arc
where middle cards are highest and edge cards droop.
"""

import math

import pytest

from arenamcp.screen_mapper import ScreenMapper


@pytest.fixture
def mapper():
    return ScreenMapper()


# ---- Ordering ----

@pytest.mark.parametrize("hand_size", [2, 3, 5, 7, 10, 13])
def test_positions_ordered_left_to_right(mapper, hand_size):
    """X coordinates must strictly increase with card index."""
    positions = mapper._hand_arc_positions(hand_size)
    xs = [p[0] for p in positions]
    for i in range(len(xs) - 1):
        assert xs[i] < xs[i + 1], (
            f"hand_size={hand_size}: x[{i}]={xs[i]:.4f} >= x[{i+1}]={xs[i+1]:.4f}"
        )


# ---- Arc shape: middle cards highest ----

@pytest.mark.parametrize("hand_size", [3, 5, 7, 10, 13])
def test_middle_cards_have_lowest_y(mapper, hand_size):
    """Middle cards should have the lowest y value (highest on screen)."""
    positions = mapper._hand_arc_positions(hand_size)
    ys = [p[1] for p in positions]
    mid = hand_size // 2
    # The middle card should be <= every other card's y value (or within
    # floating-point tolerance for even-sized hands where two cards share
    # near-equal peak position).
    min_y = min(ys)
    assert ys[mid] == pytest.approx(min_y, abs=0.002), (
        f"hand_size={hand_size}: middle card y={ys[mid]:.4f} but min y={min_y:.4f}"
    )


@pytest.mark.parametrize("hand_size", [3, 5, 7])
def test_edge_cards_droop_below_center(mapper, hand_size):
    """Edge cards (first and last) should have higher y (lower on screen) than middle."""
    positions = mapper._hand_arc_positions(hand_size)
    ys = [p[1] for p in positions]
    mid = hand_size // 2
    assert ys[0] > ys[mid], f"Left edge y={ys[0]:.4f} should be > middle y={ys[mid]:.4f}"
    assert ys[-1] > ys[mid], f"Right edge y={ys[-1]:.4f} should be > middle y={ys[mid]:.4f}"


# ---- Single card centering ----

def test_single_card_centered(mapper):
    """A single card should be centered at x=0.5."""
    positions = mapper._hand_arc_positions(1)
    assert len(positions) == 1
    x, y = positions[0]
    assert x == pytest.approx(0.5, abs=0.01)
    assert 0.88 <= y <= 0.98


# ---- Larger hands fan wider ----

def test_larger_hands_fan_wider(mapper):
    """With more cards, the hand should span a wider x range."""
    pos_3 = mapper._hand_arc_positions(3)
    pos_7 = mapper._hand_arc_positions(7)
    pos_13 = mapper._hand_arc_positions(13)

    span_3 = pos_3[-1][0] - pos_3[0][0]
    span_7 = pos_7[-1][0] - pos_7[0][0]
    span_13 = pos_13[-1][0] - pos_13[0][0]

    assert span_7 > span_3, f"7-card span ({span_7:.3f}) should exceed 3-card span ({span_3:.3f})"
    assert span_13 > span_7, f"13-card span ({span_13:.3f}) should exceed 7-card span ({span_7:.3f})"


# ---- Bounds checking ----

@pytest.mark.parametrize("hand_size", [1, 2, 3, 5, 7, 10, 13])
def test_positions_within_valid_range(mapper, hand_size):
    """All positions must stay within clamped screen bounds."""
    positions = mapper._hand_arc_positions(hand_size)
    for i, (x, y) in enumerate(positions):
        assert 0.15 <= x <= 0.85, (
            f"hand_size={hand_size}, card {i}: x={x:.4f} out of [0.15, 0.85]"
        )
        assert 0.88 <= y <= 0.98, (
            f"hand_size={hand_size}, card {i}: y={y:.4f} out of [0.88, 0.98]"
        )


# ---- Count correctness ----

@pytest.mark.parametrize("hand_size", [0, 1, 2, 3, 5, 7, 10, 13])
def test_correct_number_of_positions(mapper, hand_size):
    """Should return exactly hand_size positions."""
    positions = mapper._hand_arc_positions(hand_size)
    assert len(positions) == hand_size


# ---- Empty hand ----

def test_empty_hand_returns_empty(mapper):
    positions = mapper._hand_arc_positions(0)
    assert positions == []


# ---- Symmetry for odd hand sizes ----

@pytest.mark.parametrize("hand_size", [3, 5, 7])
def test_arc_is_symmetric(mapper, hand_size):
    """The arc should be symmetric: first card mirrors last, etc."""
    positions = mapper._hand_arc_positions(hand_size)
    mid = hand_size // 2
    for k in range(mid):
        left_x = positions[k][0]
        right_x = positions[hand_size - 1 - k][0]
        left_y = positions[k][1]
        right_y = positions[hand_size - 1 - k][1]

        # X should be symmetric around 0.5
        assert left_x + right_x == pytest.approx(1.0, abs=0.01), (
            f"hand_size={hand_size}, k={k}: x symmetry broken "
            f"({left_x:.4f} + {right_x:.4f} != 1.0)"
        )
        # Y should be equal for symmetric cards
        assert left_y == pytest.approx(right_y, abs=0.001), (
            f"hand_size={hand_size}, k={k}: y symmetry broken "
            f"({left_y:.4f} != {right_y:.4f})"
        )


# ---- Integration with get_card_in_hand_coord ----

def test_get_card_in_hand_coord_uses_arc(mapper):
    """The public method should return arc-based coordinates."""
    hand = [
        {"name": "Lightning Bolt", "mana_cost": "{R}", "card_types": ["CardType_Instant"], "type_line": "Instant"},
        {"name": "Counterspell", "mana_cost": "{U}{U}", "card_types": ["CardType_Instant"], "type_line": "Instant"},
        {"name": "Dark Ritual", "mana_cost": "{B}", "card_types": ["CardType_Instant"], "type_line": "Instant"},
    ]
    # MTGA sort: by first frame color (R=4, U=2, B=3), then CMC, then name
    # Sorted order: Counterspell (U=2, cmc=2), Dark Ritual (B=3, cmc=1), Lightning Bolt (R=4, cmc=1)
    # Counterspell is index 0 (leftmost), Dark Ritual index 1 (middle), Lightning Bolt index 2
    coord = mapper.get_card_in_hand_coord("Dark Ritual", hand, {})
    assert coord is not None
    # Dark Ritual is index 1 of 3 -> middle card, should be near x=0.5
    assert coord.x == pytest.approx(0.5, abs=0.01)
    assert 0.88 <= coord.y <= 0.98


def test_get_card_in_hand_coord_not_found(mapper):
    """Missing card should return None."""
    hand = [{"name": "Lightning Bolt"}]
    result = mapper.get_card_in_hand_coord("Ancestral Recall", hand, {})
    assert result is None


def test_get_card_in_hand_coord_empty_hand(mapper):
    """Empty hand should return None."""
    result = mapper.get_card_in_hand_coord("Bolt", [], {})
    assert result is None


# ---- Delta angle saturation ----

def test_small_hands_use_max_delta(mapper):
    """With few cards, deltaAngle should be MaxDeltaAngle (4.5),
    so cards cluster toward center rather than spanning the full arc."""
    pos_2 = mapper._hand_arc_positions(2)
    span_2 = pos_2[-1][0] - pos_2[0][0]
    # 2 cards with delta=4.5 should span much less than the full 0.60 width
    assert span_2 < 0.15, f"2-card span ({span_2:.3f}) should be narrow"


def test_large_hands_fill_arc(mapper):
    """With many cards (FitAngle / (n-1) < MaxDeltaAngle), cards fill the arc."""
    pos_13 = mapper._hand_arc_positions(13)
    span_13 = pos_13[-1][0] - pos_13[0][0]
    # 13 cards should span most of the 0.60 available width
    assert span_13 > 0.45, f"13-card span ({span_13:.3f}) should be wide"
