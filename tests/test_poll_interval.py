"""Tests for urgency-aware polling intervals (Workstream 5)."""

import pytest

from arenamcp.standalone import StandaloneCoach


@pytest.fixture
def coach():
    """Create a bare StandaloneCoach without full initialization."""
    c = StandaloneCoach.__new__(StandaloneCoach)
    return c


class TestGetPollInterval:
    def test_idle_no_match(self, coach):
        """No active match → idle polling."""
        state = {"turn": {"turn_number": 0}}
        assert coach._get_poll_interval(state) == coach._POLL_IDLE

    def test_empty_state_is_idle(self, coach):
        """Empty state dict → idle polling."""
        assert coach._get_poll_interval({}) == coach._POLL_IDLE

    def test_pending_decision_is_urgent(self, coach):
        """Pending decision → urgent polling."""
        state = {
            "turn": {"turn_number": 3, "phase": "Phase_Main1"},
            "pending_decision": "Select Targets",
        }
        assert coach._get_poll_interval(state) == coach._POLL_URGENT

    def test_stack_items_is_urgent(self, coach):
        """Items on the stack → urgent polling."""
        state = {
            "turn": {"turn_number": 3, "phase": "Phase_Main1"},
            "stack": [{"name": "Lightning Bolt"}],
        }
        assert coach._get_poll_interval(state) == coach._POLL_URGENT

    def test_combat_phase_is_active(self, coach):
        """Combat phase → active polling."""
        state = {
            "turn": {"turn_number": 5, "phase": "Phase_Combat"},
        }
        assert coach._get_poll_interval(state) == coach._POLL_ACTIVE

    def test_our_priority_is_active(self, coach):
        """Our turn with priority → active polling."""
        state = {
            "turn": {"turn_number": 3, "phase": "Phase_Main1", "priority_player": 1},
            "local_seat_id": 1,
        }
        assert coach._get_poll_interval(state) == coach._POLL_ACTIVE

    def test_opponent_turn_is_normal(self, coach):
        """Opponent's turn, no stack → normal polling."""
        state = {
            "turn": {"turn_number": 4, "phase": "Phase_Main1", "priority_player": 2},
            "local_seat_id": 1,
        }
        assert coach._get_poll_interval(state) == coach._POLL_NORMAL

    def test_pending_decision_overrides_combat(self, coach):
        """Pending decision during combat → urgent (highest priority)."""
        state = {
            "turn": {"turn_number": 5, "phase": "Phase_Combat"},
            "pending_decision": "Declare Blockers",
        }
        assert coach._get_poll_interval(state) == coach._POLL_URGENT

    def test_interval_ordering(self, coach):
        """Verify urgency levels are in correct order."""
        assert coach._POLL_URGENT < coach._POLL_ACTIVE < coach._POLL_NORMAL < coach._POLL_IDLE
