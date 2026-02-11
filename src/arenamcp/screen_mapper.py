"""Hybrid UI Detection for Autopilot Mode.

Maps game actions to screen coordinates using fixed normalized coordinates
for known buttons, positional heuristics for cards, and vision/LLM fallback
for dynamic elements.

All coordinates are normalized to the MTGA window (0.0-1.0 range) and
converted to absolute pixel coordinates at click time.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ScreenCoord:
    """A normalized screen coordinate within the MTGA window.

    x, y are in range 0.0-1.0, relative to the MTGA window.
    """
    x: float
    y: float
    description: str = ""

    def to_absolute(self, window_rect: tuple[int, int, int, int]) -> tuple[int, int]:
        """Convert normalized coords to absolute screen pixels.

        Args:
            window_rect: (left, top, width, height) of the MTGA window.

        Returns:
            (abs_x, abs_y) in screen pixels.
        """
        left, top, width, height = window_rect
        return (
            int(left + self.x * width),
            int(top + self.y * height),
        )


class FixedCoordinates:
    """Known fixed button positions in MTGA (normalized 0.0-1.0).

    These are calibrated for standard 16:9 MTGA resolution.
    May need adjustment via tools/calibrate_screen.py.
    """

    # Bottom-right action buttons (Pass/Resolve/Done all share same position)
    PASS_TURN = ScreenCoord(0.78, 0.85, "Pass Turn button")
    RESOLVE = ScreenCoord(0.78, 0.85, "Resolve button")
    DONE = ScreenCoord(0.78, 0.85, "Done button")

    # Mulligan buttons
    KEEP = ScreenCoord(0.58, 0.68, "Keep button")
    MULLIGAN = ScreenCoord(0.42, 0.68, "Mulligan button")

    # Scry options
    SCRY_TOP = ScreenCoord(0.56, 0.55, "Scry to Top")
    SCRY_BOTTOM = ScreenCoord(0.44, 0.55, "Scry to Bottom")

    # Zone centers (approximate)
    HAND_CENTER = ScreenCoord(0.50, 0.92, "Hand zone center")
    BATTLEFIELD_YOUR = ScreenCoord(0.50, 0.65, "Your battlefield center")
    BATTLEFIELD_OPP = ScreenCoord(0.50, 0.35, "Opponent battlefield center")

    # Button name -> coordinate lookup
    _BUTTONS = {
        "pass": PASS_TURN,
        "pass_turn": PASS_TURN,
        "resolve": RESOLVE,
        "done": DONE,
        "keep": KEEP,
        "mulligan": MULLIGAN,
        "scry_top": SCRY_TOP,
        "scry_bottom": SCRY_BOTTOM,
    }

    @classmethod
    def get(cls, name: str) -> Optional[ScreenCoord]:
        """Look up a fixed coordinate by button name."""
        return cls._BUTTONS.get(name.lower())


class ScreenMapper:
    """Maps game actions to screen coordinates in the MTGA window."""

    def __init__(self):
        """Initialize the screen mapper."""
        self._window_rect: Optional[tuple[int, int, int, int]] = None
        self._hwnd: Optional[int] = None

    def get_mtga_window(self) -> Optional[tuple[int, int, int, int]]:
        """Find the MTGA window and return its client-area rectangle.

        Uses ctypes user32 (FindWindowW + GetClientRect + ClientToScreen)
        instead of pygetwindow for reliable detection that avoids title bar
        offset bugs and works without extra pip dependencies.

        Returns:
            (left, top, width, height) of the client area, or None if not found.
        """
        try:
            from arenamcp.input_controller import find_mtga_hwnd, get_client_rect

            hwnd = find_mtga_hwnd()
            if not hwnd:
                logger.warning("MTGA window not found")
                return None

            self._hwnd = hwnd
            rect = get_client_rect(hwnd)
            if not rect:
                logger.warning("Failed to get MTGA client rect")
                return None

            self._window_rect = rect
            return rect
        except Exception as e:
            logger.error(f"Failed to get MTGA window: {e}")
            return None

    @property
    def window_rect(self) -> Optional[tuple[int, int, int, int]]:
        """Cached window rectangle, refreshed on demand."""
        if self._window_rect is None:
            self.get_mtga_window()
        return self._window_rect

    def refresh_window(self) -> Optional[tuple[int, int, int, int]]:
        """Force refresh window position."""
        self._window_rect = None
        return self.get_mtga_window()

    def get_button_coord(self, name: str) -> Optional[ScreenCoord]:
        """Get the screen coordinate for a known button.

        Args:
            name: Button name (e.g., "pass", "keep", "resolve").

        Returns:
            ScreenCoord or None if button not found.
        """
        return FixedCoordinates.get(name)

    def get_card_in_hand_coord(
        self,
        card_name: str,
        hand_cards: list[dict[str, Any]],
        game_state: dict[str, Any],
    ) -> Optional[ScreenCoord]:
        """Calculate the screen position of a card in hand.

        Uses positional heuristic based on hand size and card index.
        MTGA spreads cards evenly across the bottom of the screen.

        Args:
            card_name: Name of the card to find.
            hand_cards: List of card dicts in hand.
            game_state: Full game state for additional context.

        Returns:
            ScreenCoord for the card, or None if not found.
        """
        if not hand_cards:
            return None

        card_index = self._find_card_index(card_name, hand_cards)

        if card_index is None:
            hand_names = [c.get("name", "???") for c in hand_cards]
            logger.warning(
                f"Card '{card_name}' not found in hand: {hand_names}"
            )
            return None

        hand_size = len(hand_cards)

        # Hand spans roughly x=0.25 to x=0.75, y=0.92
        # Cards are evenly distributed
        hand_left = 0.25
        hand_right = 0.75
        hand_y = 0.92

        if hand_size == 1:
            x = 0.50
        else:
            spacing = (hand_right - hand_left) / (hand_size - 1)
            x = hand_left + card_index * spacing

        return ScreenCoord(x, hand_y, f"Hand card: {card_name}")

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize card name for matching.

        Handles: curly apostrophes, numbered duplicates (Swamp #2),
        leading/trailing whitespace, case.
        """
        n = name.lower().strip()
        # Normalize Unicode apostrophes/quotes to ASCII
        n = n.replace("\u2019", "'").replace("\u2018", "'")
        n = n.replace("\u201c", '"').replace("\u201d", '"')
        # Strip trailing "#N" duplicate markers (e.g., "Swamp #2" -> "swamp")
        n = re.sub(r'\s*#\d+$', '', n)
        return n

    def _find_card_index(
        self, card_name: str, hand_cards: list[dict[str, Any]]
    ) -> Optional[int]:
        """Find a card's index in hand using progressively fuzzier matching.

        Match order:
        1. Exact (case-insensitive)
        2. Normalized (strip punctuation variants, #N suffixes)
        3. Partial (card_name substring of hand name, or vice versa)
        4. Word overlap (share significant words)
        """
        target = card_name.lower().strip()
        target_norm = self._normalize_name(card_name)

        # 1. Exact match
        for i, card in enumerate(hand_cards):
            if card.get("name", "").lower().strip() == target:
                return i

        # 2. Normalized match
        for i, card in enumerate(hand_cards):
            if self._normalize_name(card.get("name", "")) == target_norm:
                return i

        # 3. Partial / substring match (either direction)
        for i, card in enumerate(hand_cards):
            hand_name = card.get("name", "").lower().strip()
            if target in hand_name or hand_name in target:
                return i

        # 4. Word overlap (handles "Adherent's Heirloom" vs "Adherent's Heirloom (ART)")
        target_words = set(target_norm.split()) - {"the", "of", "a", "an"}
        if target_words:
            best_score = 0
            best_idx = None
            for i, card in enumerate(hand_cards):
                card_words = set(self._normalize_name(card.get("name", "")).split()) - {"the", "of", "a", "an"}
                overlap = len(target_words & card_words)
                if overlap > best_score and overlap >= min(2, len(target_words)):
                    best_score = overlap
                    best_idx = i
            if best_idx is not None:
                return best_idx

        return None

    @staticmethod
    def _is_creature(card: dict[str, Any]) -> bool:
        """Check if a battlefield card is a creature."""
        # Check card_types list first (from game state)
        card_types = card.get("card_types", [])
        if card_types:
            return any("Creature" in ct for ct in card_types)
        # Fallback: check type_line string
        type_line = card.get("type_line", "")
        return "creature" in type_line.lower()

    @staticmethod
    def _is_land(card: dict[str, Any]) -> bool:
        """Check if a battlefield card is a land."""
        card_types = card.get("card_types", [])
        if card_types:
            return any("Land" in ct for ct in card_types)
        type_line = card.get("type_line", "")
        return "land" in type_line.lower()

    def get_permanent_coord(
        self,
        card_name: str,
        instance_id: Optional[int],
        battlefield: list[dict[str, Any]],
        owner_seat: int,
        local_seat: int,
    ) -> Optional[ScreenCoord]:
        """Calculate the screen position of a permanent on the battlefield.

        MTGA uses a row-based layout with separate rows for creatures and
        non-creatures (lands/enchantments/artifacts):

            Opponent lands:      y ≈ 0.22  (top)
            Opponent creatures:  y ≈ 0.38
            --- center line ---
            Your creatures:      y ≈ 0.58
            Your lands:          y ≈ 0.75  (bottom)

        Within each row, cards are centered and evenly spaced.

        Args:
            card_name: Name of the permanent.
            instance_id: Instance ID for disambiguation.
            battlefield: List of all battlefield card dicts.
            owner_seat: Seat ID of the permanent's owner.
            local_seat: Seat ID of the local player.

        Returns:
            ScreenCoord for the permanent, or None if not found.
        """
        is_yours = owner_seat == local_seat

        # Filter battlefield to the right owner
        owner_cards = [c for c in battlefield if c.get("owner_seat_id") == owner_seat]

        # Split into creatures vs non-creatures (lands/enchantments/artifacts)
        creatures = [c for c in owner_cards if self._is_creature(c)]
        non_creatures = [c for c in owner_cards if not self._is_creature(c)]

        # Find the specific card and determine which row it's in
        target_card = None
        card_row = None  # "creature" or "non_creature"
        card_idx_in_row = None

        # Search by instance_id first (exact match)
        if instance_id:
            for i, card in enumerate(creatures):
                if card.get("instance_id") == instance_id:
                    target_card = card
                    card_row = "creature"
                    card_idx_in_row = i
                    break
            if target_card is None:
                for i, card in enumerate(non_creatures):
                    if card.get("instance_id") == instance_id:
                        target_card = card
                        card_row = "non_creature"
                        card_idx_in_row = i
                        break

        # Search by exact name match
        if target_card is None:
            for i, card in enumerate(creatures):
                if card.get("name", "").lower() == card_name.lower():
                    target_card = card
                    card_row = "creature"
                    card_idx_in_row = i
                    break
            if target_card is None:
                for i, card in enumerate(non_creatures):
                    if card.get("name", "").lower() == card_name.lower():
                        target_card = card
                        card_row = "non_creature"
                        card_idx_in_row = i
                        break

        # Partial name match fallback
        if target_card is None:
            for i, card in enumerate(creatures):
                if card_name.lower() in card.get("name", "").lower():
                    target_card = card
                    card_row = "creature"
                    card_idx_in_row = i
                    break
            if target_card is None:
                for i, card in enumerate(non_creatures):
                    if card_name.lower() in card.get("name", "").lower():
                        target_card = card
                        card_row = "non_creature"
                        card_idx_in_row = i
                        break

        if target_card is None:
            logger.warning(f"Permanent '{card_name}' not found on battlefield")
            return None

        # Determine Y coordinate based on owner and card type
        # MTGA layout (16:9):
        #   Opp non-creatures (lands):  y ≈ 0.22
        #   Opp creatures:              y ≈ 0.38
        #   Your creatures:             y ≈ 0.58
        #   Your non-creatures (lands): y ≈ 0.75
        if is_yours:
            y = 0.58 if card_row == "creature" else 0.75
        else:
            y = 0.38 if card_row == "creature" else 0.22

        # Determine X coordinate — centered, evenly spaced in the row
        row_cards = creatures if card_row == "creature" else non_creatures
        num_in_row = len(row_cards)

        bf_left = 0.20
        bf_right = 0.80

        if num_in_row == 1:
            x = 0.50
        else:
            # Evenly space across the row, centered
            total_width = bf_right - bf_left
            spacing = total_width / num_in_row
            # Center the group: offset = (total_width - (n-1)*spacing) / 2
            x = bf_left + (spacing / 2) + card_idx_in_row * spacing

        logger.info(
            f"Mapped '{card_name}' -> ({x:.2f}, {y:.2f}) "
            f"[{card_row}, idx={card_idx_in_row}/{num_in_row}, "
            f"{'yours' if is_yours else 'opponent'}]"
        )

        return ScreenCoord(x, y, f"Permanent: {card_name}")

    def get_card_coord_via_vision(
        self, card_name: str, screenshot_bytes: bytes
    ) -> Optional[ScreenCoord]:
        """Use vision LLM to locate a card on screen.

        Fallback for when positional heuristics fail.

        Args:
            card_name: Name of the card to find.
            screenshot_bytes: Screenshot of the MTGA window as PNG bytes.

        Returns:
            ScreenCoord or None if vision detection fails.
        """
        # TODO: Implement vision-based card detection
        # This would send the screenshot to a vision LLM (GPT-4V, Gemini Pro Vision)
        # with a prompt like "Find the card named X and return its center coordinates"
        logger.warning(f"Vision fallback not yet implemented for '{card_name}'")
        return None

    def get_option_coord(
        self, option_index: int, total_options: int, context: str = ""
    ) -> Optional[ScreenCoord]:
        """Calculate position for modal/select UI options.

        MTGA presents options in a centered vertical list or horizontal row.

        Args:
            option_index: 0-based index of the option to click.
            total_options: Total number of options presented.
            context: Optional context hint (e.g., "modal", "scry").

        Returns:
            ScreenCoord for the option.
        """
        if total_options <= 0:
            return None

        # Options are typically centered, stacked vertically
        # Roughly y=0.40 to y=0.60, centered at x=0.50
        option_top = 0.40
        option_bottom = 0.60

        if total_options == 1:
            y = 0.50
        else:
            spacing = (option_bottom - option_top) / (total_options - 1)
            y = option_top + option_index * spacing

        return ScreenCoord(0.50, y, f"Option {option_index + 1}/{total_options}")

    def get_draft_card_coord(
        self, card_index: int, pack_size: int
    ) -> Optional[ScreenCoord]:
        """Calculate position of a card in a draft pack.

        Draft packs are displayed in a grid, typically 3 rows of 5.

        Args:
            card_index: 0-based index in the pack.
            pack_size: Number of cards in the pack.

        Returns:
            ScreenCoord for the card.
        """
        if pack_size <= 0:
            return None

        # Draft grid: roughly 5 columns, up to 3 rows
        cols = min(pack_size, 5)
        rows = (pack_size + cols - 1) // cols

        row = card_index // cols
        col = card_index % cols

        # Grid spans x=0.15 to x=0.85, y=0.20 to y=0.70
        grid_left = 0.15
        grid_right = 0.85
        grid_top = 0.20
        grid_bottom = 0.70

        if cols == 1:
            x = 0.50
        else:
            x = grid_left + col * (grid_right - grid_left) / (cols - 1)

        if rows == 1:
            y = 0.45
        else:
            y = grid_top + row * (grid_bottom - grid_top) / (rows - 1)

        return ScreenCoord(x, y, f"Draft card {card_index + 1}/{pack_size}")
