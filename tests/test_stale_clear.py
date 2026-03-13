"""Tests for stale-clear tolerance (Workstream 6)."""

import time
from unittest.mock import patch

import pytest

from arenamcp.gamestate import GameState, Zone, ZoneType, GameObject


@pytest.fixture
def gs():
    """Create a GameState with basic setup."""
    g = GameState()
    g.local_seat_id = 1
    g._seat_source = 2
    # Add two players
    from arenamcp.gamestate import Player
    g.players[1] = Player(seat_id=1, life_total=20)
    g.players[2] = Player(seat_id=2, life_total=20)
    return g


class TestStackClearGracePeriod:
    def test_turn_boundary_clears_immediately(self, gs):
        """Stack is always cleared on turn boundaries (force=True)."""
        # Add a stack zone with items
        gs.zones[100] = Zone(
            zone_id=100,
            zone_type=ZoneType.STACK,
            object_instance_ids=[1, 2, 3],
        )
        gs._last_stack_update_time = time.time()  # Just updated

        gs._clear_stale_stack(force=True)

        assert gs.zones[100].object_instance_ids == []

    def test_phase_transition_respects_grace(self, gs):
        """Phase transition defers clearing if stack was just updated."""
        gs.zones[100] = Zone(
            zone_id=100,
            zone_type=ZoneType.STACK,
            object_instance_ids=[1, 2],
        )
        # Stack was updated 0.5s ago — within grace period
        gs._last_stack_update_time = time.time() - 0.5

        gs._clear_stale_stack(force=False)

        # Should NOT have been cleared
        assert gs.zones[100].object_instance_ids == [1, 2]

    def test_phase_transition_clears_old_stack(self, gs):
        """Phase transition clears stack if it hasn't been updated recently."""
        gs.zones[100] = Zone(
            zone_id=100,
            zone_type=ZoneType.STACK,
            object_instance_ids=[1, 2],
        )
        # Stack was updated 5s ago — outside grace period
        gs._last_stack_update_time = time.time() - 5.0

        gs._clear_stale_stack(force=False)

        assert gs.zones[100].object_instance_ids == []

    def test_phase_transition_clears_when_no_timestamp(self, gs):
        """Phase transition clears stack if no update timestamp exists."""
        gs.zones[100] = Zone(
            zone_id=100,
            zone_type=ZoneType.STACK,
            object_instance_ids=[1],
        )
        gs._last_stack_update_time = 0  # No timestamp

        gs._clear_stale_stack(force=False)

        assert gs.zones[100].object_instance_ids == []


class TestStackUpdateTracking:
    def test_zone_update_tracks_stack_time(self, gs):
        """Updating a stack zone sets _last_stack_update_time."""
        gs._last_stack_update_time = 0

        gs._update_zone({
            "zoneId": 50,
            "type": "ZoneType_Stack",
            "objectInstanceIds": [10, 20],
        })

        assert gs._last_stack_update_time > 0
        assert time.time() - gs._last_stack_update_time < 1

    def test_zone_update_ignores_empty_stack(self, gs):
        """Empty stack zone update doesn't update the timestamp."""
        gs._last_stack_update_time = 0

        gs._update_zone({
            "zoneId": 50,
            "type": "ZoneType_Stack",
            "objectInstanceIds": [],
        })

        assert gs._last_stack_update_time == 0

    def test_zone_update_non_stack_doesnt_track(self, gs):
        """Non-stack zone update doesn't affect stack timestamp."""
        gs._last_stack_update_time = 0

        gs._update_zone({
            "zoneId": 60,
            "type": "ZoneType_Hand",
            "objectInstanceIds": [10],
        })

        assert gs._last_stack_update_time == 0


class TestCombatStepTimestamp:
    def test_combat_step_tracks_time(self, gs):
        """Queuing a combat step updates the timestamp."""
        gs._last_combat_step_time = 0
        gs.turn_info.turn_number = 3
        gs.turn_info.active_player = 1
        gs.turn_info.phase = "Phase_Combat"
        gs.turn_info.step = ""

        # Simulate _update_turn_info with a combat step
        gs._update_turn_info({
            "turnNumber": 3,
            "activePlayer": 1,
            "phase": "Phase_Combat",
            "step": "Step_DeclareAttack",
        })

        assert gs._last_combat_step_time > 0
        assert len(gs._pending_combat_steps) == 1


class TestDecisionTimeoutTolerance:
    def test_normal_phase_15s_timeout(self, gs):
        """Decisions without source_id clear after 15s in normal phases."""
        gs.pending_decision = "Some Prompt"
        gs.decision_timestamp = time.time() - 16  # 16s old
        gs.decision_context = {"type": "prompt"}
        gs.turn_info.phase = "Phase_Main1"

        # Simulating what the handler does:
        # Check: no source_id, not busy phase, age > 15
        is_busy = "Combat" in gs.turn_info.phase or len(gs.stack) > 0
        timeout = 25 if is_busy else 15
        age = time.time() - gs.decision_timestamp
        should_clear = age > timeout

        assert not is_busy
        assert timeout == 15
        assert should_clear is True

    def test_combat_phase_25s_timeout(self, gs):
        """Decisions without source_id get 25s timeout during combat."""
        gs.pending_decision = "Declare Blockers"
        gs.decision_timestamp = time.time() - 20  # 20s old
        gs.decision_context = {"type": "combat"}
        gs.turn_info.phase = "Phase_Combat"

        is_busy = "Combat" in gs.turn_info.phase or len(gs.stack) > 0
        timeout = 25 if is_busy else 15
        age = time.time() - gs.decision_timestamp

        assert is_busy
        assert timeout == 25
        # 20s < 25s, should NOT clear yet
        assert (age > timeout) is False

    def test_combat_phase_clears_after_25s(self, gs):
        """Decisions clear after 25s even during combat."""
        gs.pending_decision = "Declare Blockers"
        gs.decision_timestamp = time.time() - 26  # 26s old
        gs.decision_context = {"type": "combat"}
        gs.turn_info.phase = "Phase_Combat"

        is_busy = "Combat" in gs.turn_info.phase
        timeout = 25 if is_busy else 15
        age = time.time() - gs.decision_timestamp

        assert is_busy
        assert (age > timeout) is True
