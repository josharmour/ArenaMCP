"""Tests for resolution-aware button coordinate placement.

Validates that ButtonCoordinates computes correct positions across
different aspect ratios and that the ScreenMapper integration works
correctly with the aspect-ratio-aware button system.

Based on RE analysis of MTGA's UI layout:
- Primary prompt buttons (Pass/Done/Resolve) in bottom-right via
  _promptButtonsAnchorPoint (BattleFieldStaticElementsLayout).
- Mulligan buttons in centred MulliganBrowser overlay.
- Scry buttons in centred browser overlay.
- Pillar-boxing for ultra-wide, letter-boxing for 4:3.
"""

import pytest

from arenamcp.screen_mapper import ButtonCoordinates, FixedCoordinates, ScreenCoord, ScreenMapper


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """FixedCoordinates alias must work identically to ButtonCoordinates."""

    def test_alias_is_same_class(self):
        assert FixedCoordinates is ButtonCoordinates

    def test_get_without_aspect_returns_coord(self):
        """Old-style call without aspect argument must still work."""
        coord = FixedCoordinates.get("pass")
        assert coord is not None
        assert isinstance(coord, ScreenCoord)

    def test_get_returns_none_for_unknown(self):
        assert FixedCoordinates.get("nonexistent_button") is None


# ---------------------------------------------------------------------------
# 16:9 baseline positions
# ---------------------------------------------------------------------------

class TestBaseline16x9:
    """Button positions at the native 16:9 aspect ratio."""

    ASPECT_16_9 = 16.0 / 9.0

    @pytest.mark.parametrize("name", [
        "pass", "pass_turn", "resolve", "done",
        "next", "attack", "block", "no_attacks", "no_blocks",
    ])
    def test_primary_buttons_share_position(self, name):
        """All primary prompt buttons map to the same location."""
        coord = ButtonCoordinates.get(name, aspect=self.ASPECT_16_9)
        assert coord is not None
        assert coord.x == pytest.approx(0.78, abs=0.001)
        assert coord.y == pytest.approx(0.85, abs=0.001)

    def test_keep_button_position(self):
        coord = ButtonCoordinates.get("keep", aspect=self.ASPECT_16_9)
        assert coord is not None
        assert coord.x == pytest.approx(0.58, abs=0.001)
        assert coord.y == pytest.approx(0.68, abs=0.001)

    def test_mulligan_button_position(self):
        coord = ButtonCoordinates.get("mulligan", aspect=self.ASPECT_16_9)
        assert coord is not None
        assert coord.x == pytest.approx(0.42, abs=0.001)
        assert coord.y == pytest.approx(0.68, abs=0.001)

    def test_keep_right_of_mulligan(self):
        """Keep button should be to the right of Mulligan (RE: browser layout)."""
        keep = ButtonCoordinates.get("keep", aspect=self.ASPECT_16_9)
        mull = ButtonCoordinates.get("mulligan", aspect=self.ASPECT_16_9)
        assert keep.x > mull.x

    def test_keep_mulligan_symmetric_about_center(self):
        """Keep and Mulligan should be roughly symmetric around x=0.5."""
        keep = ButtonCoordinates.get("keep", aspect=self.ASPECT_16_9)
        mull = ButtonCoordinates.get("mulligan", aspect=self.ASPECT_16_9)
        center = (keep.x + mull.x) / 2.0
        assert center == pytest.approx(0.50, abs=0.01)

    def test_scry_top_position(self):
        coord = ButtonCoordinates.get("scry_top", aspect=self.ASPECT_16_9)
        assert coord is not None
        assert coord.x == pytest.approx(0.56, abs=0.001)
        assert coord.y == pytest.approx(0.55, abs=0.001)

    def test_scry_bottom_position(self):
        coord = ButtonCoordinates.get("scry_bottom", aspect=self.ASPECT_16_9)
        assert coord is not None
        assert coord.x == pytest.approx(0.44, abs=0.001)
        assert coord.y == pytest.approx(0.55, abs=0.001)

    def test_scry_top_right_of_bottom(self):
        """Scry to Top should be right of Scry to Bottom."""
        top = ButtonCoordinates.get("scry_top", aspect=self.ASPECT_16_9)
        bot = ButtonCoordinates.get("scry_bottom", aspect=self.ASPECT_16_9)
        assert top.x > bot.x


# ---------------------------------------------------------------------------
# Viewport inset / aspect ratio compensation
# ---------------------------------------------------------------------------

class TestViewportInset:
    """Test the pillar-box / letter-box viewport calculation."""

    def test_16_9_no_inset(self):
        """16:9 should produce identity viewport."""
        left, top, w, h = ButtonCoordinates._viewport_inset(16.0 / 9.0)
        assert left == pytest.approx(0.0, abs=0.01)
        assert top == pytest.approx(0.0, abs=0.01)
        assert w == pytest.approx(1.0, abs=0.01)
        assert h == pytest.approx(1.0, abs=0.01)

    def test_21_9_pillarbox(self):
        """21:9 ultra-wide should produce pillar-boxing (bars on sides)."""
        aspect = 21.0 / 9.0
        left, top, w, h = ButtonCoordinates._viewport_inset(aspect)
        # The viewport width should be less than 1.0
        assert w < 1.0
        # No vertical bars
        assert top == pytest.approx(0.0, abs=0.001)
        assert h == pytest.approx(1.0, abs=0.001)
        # Symmetric left bar
        assert left > 0.0
        assert left == pytest.approx((1.0 - w) / 2.0, abs=0.001)

    def test_4_3_letterbox(self):
        """4:3 should produce letter-boxing (bars on top/bottom)."""
        aspect = 4.0 / 3.0
        left, top, w, h = ButtonCoordinates._viewport_inset(aspect)
        # No horizontal bars
        assert left == pytest.approx(0.0, abs=0.001)
        assert w == pytest.approx(1.0, abs=0.001)
        # The viewport height should be less than 1.0
        assert h < 1.0
        # Symmetric top bar
        assert top > 0.0
        assert top == pytest.approx((1.0 - h) / 2.0, abs=0.001)

    def test_16_10_letterbox(self):
        """16:10 should produce slight letter-boxing."""
        aspect = 16.0 / 10.0
        left, top, w, h = ButtonCoordinates._viewport_inset(aspect)
        assert left == pytest.approx(0.0, abs=0.001)
        assert w == pytest.approx(1.0, abs=0.001)
        assert h < 1.0
        assert top > 0.0


class TestAspectRatioCompensation:
    """Coordinates shift correctly for non-16:9 aspect ratios."""

    def test_ultrawide_pass_x_shifts_inward(self):
        """On ultra-wide, the pass button x should be < 0.78 in window coords
        because the playable viewport is narrower than the full window."""
        coord_16_9 = ButtonCoordinates.get("pass", aspect=16.0 / 9.0)
        coord_21_9 = ButtonCoordinates.get("pass", aspect=21.0 / 9.0)
        # The 21:9 window-relative x should be smaller (shifted towards center)
        assert coord_21_9.x < coord_16_9.x
        # But still in the right half
        assert coord_21_9.x > 0.5

    def test_ultrawide_pass_y_unchanged(self):
        """On ultra-wide, pillar-boxing doesn't affect Y."""
        coord_16_9 = ButtonCoordinates.get("pass", aspect=16.0 / 9.0)
        coord_21_9 = ButtonCoordinates.get("pass", aspect=21.0 / 9.0)
        assert coord_21_9.y == pytest.approx(coord_16_9.y, abs=0.001)

    def test_4_3_pass_y_shifts_inward(self):
        """On 4:3, the pass button y should differ from 16:9 due to letter-boxing."""
        coord_16_9 = ButtonCoordinates.get("pass", aspect=16.0 / 9.0)
        coord_4_3 = ButtonCoordinates.get("pass", aspect=4.0 / 3.0)
        # Letter-boxing compresses vertical range, so button moves towards center
        assert coord_4_3.y != pytest.approx(coord_16_9.y, abs=0.01)

    def test_4_3_pass_x_unchanged(self):
        """On 4:3, letter-boxing doesn't affect X."""
        coord_16_9 = ButtonCoordinates.get("pass", aspect=16.0 / 9.0)
        coord_4_3 = ButtonCoordinates.get("pass", aspect=4.0 / 3.0)
        assert coord_4_3.x == pytest.approx(coord_16_9.x, abs=0.001)

    def test_mulligan_buttons_still_symmetric_on_ultrawide(self):
        """Keep/Mulligan should remain symmetric even on ultra-wide."""
        keep = ButtonCoordinates.get("keep", aspect=21.0 / 9.0)
        mull = ButtonCoordinates.get("mulligan", aspect=21.0 / 9.0)
        # They should be symmetric about the window center
        mid = (keep.x + mull.x) / 2.0
        assert mid == pytest.approx(0.5, abs=0.01)

    def test_all_buttons_in_bounds(self):
        """All buttons should stay within [0, 1] at any supported aspect ratio."""
        aspects = [4.0 / 3.0, 16.0 / 10.0, 16.0 / 9.0, 21.0 / 9.0, 32.0 / 9.0]
        names = ["pass", "resolve", "done", "keep", "mulligan", "scry_top", "scry_bottom"]
        for aspect in aspects:
            for name in names:
                coord = ButtonCoordinates.get(name, aspect=aspect)
                assert coord is not None, f"{name} at {aspect}"
                assert 0.0 <= coord.x <= 1.0, (
                    f"{name} at {aspect}: x={coord.x:.4f} out of [0, 1]"
                )
                assert 0.0 <= coord.y <= 1.0, (
                    f"{name} at {aspect}: y={coord.y:.4f} out of [0, 1]"
                )


# ---------------------------------------------------------------------------
# ScreenMapper integration
# ---------------------------------------------------------------------------

class TestScreenMapperButtonIntegration:
    """ScreenMapper.get_button_coord uses aspect ratio from window rect."""

    def test_no_window_defaults_to_16_9(self):
        """Without a window rect, should use 16:9 defaults."""
        mapper = ScreenMapper()
        coord = mapper.get_button_coord("pass")
        baseline = ButtonCoordinates.get("pass", aspect=16.0 / 9.0)
        assert coord.x == pytest.approx(baseline.x, abs=0.001)
        assert coord.y == pytest.approx(baseline.y, abs=0.001)

    def test_with_16_9_window(self):
        """With a 16:9 window rect, coordinates match baseline."""
        mapper = ScreenMapper()
        mapper._window_rect = (0, 0, 1920, 1080)
        coord = mapper.get_button_coord("pass")
        assert coord.x == pytest.approx(0.78, abs=0.001)
        assert coord.y == pytest.approx(0.85, abs=0.001)

    def test_with_ultrawide_window(self):
        """With a 21:9 window, coordinates are adjusted for pillar-boxing."""
        mapper = ScreenMapper()
        mapper._window_rect = (0, 0, 2560, 1080)  # ~21:9
        coord = mapper.get_button_coord("pass")
        # Should be shifted inward compared to 16:9
        assert coord.x < 0.78
        assert coord.x > 0.5

    def test_with_4_3_window(self):
        """With a 4:3 window, coordinates are adjusted for letter-boxing."""
        mapper = ScreenMapper()
        mapper._window_rect = (0, 0, 1024, 768)  # 4:3
        coord = mapper.get_button_coord("pass")
        # X should be the same as 16:9 (letter-boxing doesn't affect X)
        assert coord.x == pytest.approx(0.78, abs=0.001)

    def test_unknown_button_returns_none(self):
        mapper = ScreenMapper()
        assert mapper.get_button_coord("nonexistent") is None

    def test_current_aspect_with_no_window(self):
        """_current_aspect returns 16:9 default when no window."""
        mapper = ScreenMapper()
        assert mapper._current_aspect() == pytest.approx(16.0 / 9.0, abs=0.01)

    def test_current_aspect_with_window(self):
        """_current_aspect returns actual aspect from window rect."""
        mapper = ScreenMapper()
        mapper._window_rect = (0, 0, 2560, 1080)
        assert mapper._current_aspect() == pytest.approx(2560.0 / 1080.0, abs=0.01)


# ---------------------------------------------------------------------------
# Button description strings
# ---------------------------------------------------------------------------

class TestButtonDescriptions:
    """Returned ScreenCoords should have meaningful descriptions."""

    @pytest.mark.parametrize("name,expected_substr", [
        ("pass", "Pass"),
        ("resolve", "Resolve"),
        ("done", "Done"),
        ("keep", "Keep"),
        ("mulligan", "Mulligan"),
        ("scry_top", "Scry"),
        ("scry_bottom", "Scry"),
    ])
    def test_description_contains_button_name(self, name, expected_substr):
        coord = ButtonCoordinates.get(name)
        assert coord is not None
        assert expected_substr in coord.description


# ---------------------------------------------------------------------------
# to_window_coord round-trip consistency
# ---------------------------------------------------------------------------

class TestCoordConversion:
    """_to_window_coord should be consistent with _viewport_inset."""

    @pytest.mark.parametrize("aspect", [4/3, 16/10, 16/9, 21/9, 32/9])
    def test_origin_maps_to_viewport_topleft(self, aspect):
        """Viewport (0,0) should map to the top-left of the playable area."""
        wx, wy = ButtonCoordinates._to_window_coord(0.0, 0.0, aspect)
        left, top, _, _ = ButtonCoordinates._viewport_inset(aspect)
        assert wx == pytest.approx(left, abs=0.001)
        assert wy == pytest.approx(top, abs=0.001)

    @pytest.mark.parametrize("aspect", [4/3, 16/10, 16/9, 21/9, 32/9])
    def test_one_one_maps_to_viewport_bottomright(self, aspect):
        """Viewport (1,1) should map to the bottom-right of the playable area."""
        wx, wy = ButtonCoordinates._to_window_coord(1.0, 1.0, aspect)
        left, top, w, h = ButtonCoordinates._viewport_inset(aspect)
        assert wx == pytest.approx(left + w, abs=0.001)
        assert wy == pytest.approx(top + h, abs=0.001)

    @pytest.mark.parametrize("aspect", [4/3, 16/10, 16/9, 21/9, 32/9])
    def test_center_maps_to_window_center(self, aspect):
        """Viewport (0.5, 0.5) should always map to window center (0.5, 0.5)."""
        wx, wy = ButtonCoordinates._to_window_coord(0.5, 0.5, aspect)
        assert wx == pytest.approx(0.5, abs=0.001)
        assert wy == pytest.approx(0.5, abs=0.001)
