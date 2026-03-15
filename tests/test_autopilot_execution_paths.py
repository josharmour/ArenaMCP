"""Tests for autopilot execution path tracking (Phases 3 & 4).

Validates:
- ExecutionPath constants exist and have expected values
- _log_execution_path tracks stats correctly
- Vision fallback is only used when deterministic fails
- Instance ID helpers work for combat decision contexts
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field

from arenamcp.autopilot import AutopilotConfig, AutopilotEngine, ExecutionPath
from arenamcp.action_planner import ActionPlan, ActionType, GameAction
from arenamcp.input_controller import ClickResult


# ---- ExecutionPath constants ----

class TestExecutionPathConstants:
    """ExecutionPath constants exist and have expected string values."""

    def test_gre_aware_constant(self):
        assert ExecutionPath.GRE_AWARE == "gre-aware"

    def test_deterministic_geometry_constant(self):
        assert ExecutionPath.DETERMINISTIC_GEOMETRY == "deterministic-geometry"

    def test_vision_fallback_constant(self):
        assert ExecutionPath.VISION_FALLBACK == "vision-fallback"

    def test_all_constants_are_unique(self):
        values = [
            ExecutionPath.GRE_AWARE,
            ExecutionPath.DETERMINISTIC_GEOMETRY,
            ExecutionPath.VISION_FALLBACK,
        ]
        assert len(set(values)) == len(values)


# ---- Fixtures ----

@pytest.fixture
def mock_mapper():
    mapper = MagicMock()
    mapper.window_rect = (0, 0, 1920, 1080)
    mapper.refresh_window.return_value = (0, 0, 1920, 1080)
    return mapper


@pytest.fixture
def mock_controller():
    controller = MagicMock()
    controller.click.return_value = ClickResult(True, 100, 100, "test")
    controller.drag_card_from_hand.return_value = ClickResult(True, 100, 100, "drag")
    controller.click_card_in_hand.return_value = ClickResult(True, 100, 100, "click_hand")
    controller.focus_mtga_window.return_value = None
    controller.wait.return_value = None
    controller.press_key.return_value = None
    return controller


@pytest.fixture
def mock_planner():
    return MagicMock()


@pytest.fixture
def engine(mock_planner, mock_mapper, mock_controller):
    """Create an AutopilotEngine with mocked dependencies."""
    config = AutopilotConfig(
        enable_vision_fallback=True,
        prefer_deterministic=True,
        dry_run=True,
    )
    return AutopilotEngine(
        planner=mock_planner,
        mapper=mock_mapper,
        controller=mock_controller,
        get_game_state=lambda: {},
        config=config,
    )


# ---- _log_execution_path ----

class TestLogExecutionPath:
    """_log_execution_path tracks stats correctly."""

    def test_tracks_single_path(self, engine):
        engine._log_execution_path(ExecutionPath.DETERMINISTIC_GEOMETRY, "test action")
        assert engine.path_stats[ExecutionPath.DETERMINISTIC_GEOMETRY] == 1

    def test_tracks_multiple_paths(self, engine):
        engine._log_execution_path(ExecutionPath.DETERMINISTIC_GEOMETRY, "action 1")
        engine._log_execution_path(ExecutionPath.DETERMINISTIC_GEOMETRY, "action 2")
        engine._log_execution_path(ExecutionPath.VISION_FALLBACK, "action 3")
        assert engine.path_stats[ExecutionPath.DETERMINISTIC_GEOMETRY] == 2
        assert engine.path_stats[ExecutionPath.VISION_FALLBACK] == 1

    def test_tracks_gre_aware(self, engine):
        engine._log_execution_path(ExecutionPath.GRE_AWARE, "gre action")
        assert engine.path_stats[ExecutionPath.GRE_AWARE] == 1

    def test_path_stats_returns_copy(self, engine):
        """path_stats should return a copy, not the internal dict."""
        engine._log_execution_path(ExecutionPath.DETERMINISTIC_GEOMETRY, "test")
        stats = engine.path_stats
        stats[ExecutionPath.DETERMINISTIC_GEOMETRY] = 999
        assert engine.path_stats[ExecutionPath.DETERMINISTIC_GEOMETRY] == 1

    def test_empty_stats_initially(self, engine):
        assert engine.path_stats == {}


# ---- AutopilotConfig.prefer_deterministic ----

class TestAutopilotConfigPreferDeterministic:
    """AutopilotConfig has prefer_deterministic with correct default."""

    def test_default_is_true(self):
        config = AutopilotConfig()
        assert config.prefer_deterministic is True

    def test_can_set_false(self):
        config = AutopilotConfig(prefer_deterministic=False)
        assert config.prefer_deterministic is False


class TestVisionPrefetch:
    """Vision prefetch should stay off the hot path in deterministic mode."""

    def test_process_trigger_skips_prefetch_when_deterministic(self, engine, mock_planner):
        engine._scan_layout_if_needed = MagicMock()
        mock_planner.plan_actions.return_value = ActionPlan(actions=[])

        game_state = {
            "turn": {"turn_number": 3, "phase": "Phase_Main1", "active_player": 1},
            "players": [{"is_local": True, "seat_id": 1}],
            "pending_decision": "Choose Action",
            "legal_actions": ["Pass"],
        }

        engine.process_trigger(game_state, "decision_required")

        engine._scan_layout_if_needed.assert_not_called()

    def test_process_trigger_prefetches_when_vision_heavy_mode(self, mock_planner, mock_mapper, mock_controller):
        config = AutopilotConfig(
            enable_vision_fallback=True,
            prefer_deterministic=False,
            dry_run=True,
        )
        engine = AutopilotEngine(
            planner=mock_planner,
            mapper=mock_mapper,
            controller=mock_controller,
            get_game_state=lambda: {},
            config=config,
        )
        engine._scan_layout_if_needed = MagicMock()
        mock_planner.plan_actions.return_value = ActionPlan(actions=[])

        game_state = {
            "turn": {"turn_number": 3, "phase": "Phase_Main1", "active_player": 1},
            "players": [{"is_local": True, "seat_id": 1}],
            "pending_decision": "Choose Action",
            "legal_actions": ["Pass"],
        }

        engine.process_trigger(game_state, "decision_required")

        engine._scan_layout_if_needed.assert_called_once_with(game_state)


# ---- Button clicks use deterministic geometry ----

class TestButtonClicksDeterministic:
    """Button clicks (pass, resolve, done) always use deterministic-geometry."""

    def test_pass_priority_logs_deterministic(self, engine, mock_mapper):
        from arenamcp.screen_mapper import ScreenCoord
        mock_mapper.get_button_coord.return_value = ScreenCoord(0.5, 0.9, "pass")
        engine._exec_pass_priority()
        assert engine.path_stats.get(ExecutionPath.DETERMINISTIC_GEOMETRY, 0) >= 1

    def test_resolve_logs_deterministic(self, engine, mock_mapper):
        from arenamcp.screen_mapper import ScreenCoord
        mock_mapper.get_button_coord.return_value = ScreenCoord(0.5, 0.9, "resolve")
        engine._exec_resolve()
        assert engine.path_stats.get(ExecutionPath.DETERMINISTIC_GEOMETRY, 0) >= 1

    def test_click_button_logs_deterministic(self, engine, mock_mapper):
        from arenamcp.screen_mapper import ScreenCoord
        mock_mapper.get_button_coord.return_value = ScreenCoord(0.5, 0.9, "done")
        action = GameAction(action_type=ActionType.CLICK_BUTTON, card_name="done")
        engine._exec_click_button(action)
        assert engine.path_stats.get(ExecutionPath.DETERMINISTIC_GEOMETRY, 0) >= 1


# ---- Play card uses deterministic first, vision fallback second ----

class TestPlayCardExecutionPath:
    """Play card prefers deterministic arc-based lookup, falls back to vision."""

    def test_deterministic_hand_lookup_used(self, engine, mock_mapper):
        from arenamcp.screen_mapper import ScreenCoord
        mock_mapper.get_card_in_hand_coord.return_value = ScreenCoord(0.5, 0.9, "Lightning Bolt")
        action = GameAction(action_type=ActionType.CAST_SPELL, card_name="Lightning Bolt")
        game_state = {"hand": [{"name": "Lightning Bolt"}]}
        engine._exec_play_card(action, game_state)
        assert engine.path_stats.get(ExecutionPath.DETERMINISTIC_GEOMETRY, 0) >= 1
        assert engine.path_stats.get(ExecutionPath.VISION_FALLBACK, 0) == 0

    def test_vision_fallback_when_deterministic_fails(self, engine, mock_mapper):
        from arenamcp.screen_mapper import ScreenCoord
        mock_mapper.get_card_in_hand_coord.return_value = None
        # Mock vision path
        with patch.object(engine, '_get_vision_coord', return_value=ScreenCoord(0.5, 0.9, "Lightning Bolt")):
            action = GameAction(action_type=ActionType.CAST_SPELL, card_name="Lightning Bolt")
            game_state = {"hand": [{"name": "Lightning Bolt"}]}
            engine._exec_play_card(action, game_state)
            assert engine.path_stats.get(ExecutionPath.VISION_FALLBACK, 0) >= 1
            assert engine.path_stats.get(ExecutionPath.DETERMINISTIC_GEOMETRY, 0) == 0

    def test_no_vision_when_disabled(self, engine, mock_mapper):
        engine._config.enable_vision_fallback = False
        mock_mapper.get_card_in_hand_coord.return_value = None
        action = GameAction(action_type=ActionType.CAST_SPELL, card_name="Lightning Bolt")
        game_state = {"hand": [{"name": "Lightning Bolt"}]}
        result = engine._exec_play_card(action, game_state)
        assert not result.success
        assert engine.path_stats.get(ExecutionPath.VISION_FALLBACK, 0) == 0


# ---- GRE-aware logging in _execute_action ----

class TestGREAwareLogging:
    """_execute_action logs gre-aware when action has gre_action_ref."""

    def test_gre_action_ref_logged(self, engine, mock_mapper):
        from arenamcp.screen_mapper import ScreenCoord
        mock_mapper.get_button_coord.return_value = ScreenCoord(0.5, 0.9, "pass")
        action = GameAction(action_type=ActionType.PASS_PRIORITY, card_name="pass")
        # Simulate Phase 1 adding gre_action_ref via setattr (since field may not exist yet)
        action.gre_action_ref = {"type": "ActionType_Pass", "instance": 42}
        engine._execute_action(action, {})
        assert engine.path_stats.get(ExecutionPath.GRE_AWARE, 0) >= 1

    def test_no_gre_ref_no_gre_aware_log(self, engine, mock_mapper):
        from arenamcp.screen_mapper import ScreenCoord
        mock_mapper.get_button_coord.return_value = ScreenCoord(0.5, 0.9, "pass")
        action = GameAction(action_type=ActionType.PASS_PRIORITY, card_name="pass")
        engine._execute_action(action, {})
        assert engine.path_stats.get(ExecutionPath.GRE_AWARE, 0) == 0


# ---- Instance ID helpers ----

class TestInstanceIdHelpers:
    """Instance ID lookup helpers for combat."""

    def test_find_instance_id_by_name(self, engine):
        battlefield = [
            {"name": "Grizzly Bears", "owner_seat_id": 1, "instance_id": 100},
            {"name": "Lightning Bolt", "owner_seat_id": 1, "instance_id": 101},
            {"name": "Grizzly Bears", "owner_seat_id": 2, "instance_id": 200},
        ]
        assert engine._find_instance_id("Grizzly Bears", battlefield, 1) == 100
        assert engine._find_instance_id("Lightning Bolt", battlefield, 1) == 101
        assert engine._find_instance_id("Grizzly Bears", battlefield, 2) == 200
        assert engine._find_instance_id("Nonexistent", battlefield, 1) is None

    def test_find_instance_id_case_insensitive(self, engine):
        battlefield = [
            {"name": "Grizzly Bears", "owner_seat_id": 1, "instance_id": 100},
        ]
        assert engine._find_instance_id("grizzly bears", battlefield, 1) == 100

    def test_build_attacker_id_map_with_context(self, engine):
        game_state = {
            "decision_context": {
                "type": "declare_attackers",
                "legal_attackers": ["Grizzly Bears", "Elvish Mystic"],
                "legal_attacker_ids": [100, 101],
            }
        }
        result = engine._build_attacker_id_map(game_state)
        assert result == {"Grizzly Bears": 100, "Elvish Mystic": 101}

    def test_build_attacker_id_map_no_ids(self, engine):
        game_state = {
            "decision_context": {
                "type": "declare_attackers",
                "legal_attackers": ["Grizzly Bears"],
            }
        }
        result = engine._build_attacker_id_map(game_state)
        assert result == {}

    def test_build_attacker_id_map_wrong_type(self, engine):
        game_state = {
            "decision_context": {
                "type": "declare_blockers",
                "legal_blockers": ["Grizzly Bears"],
                "legal_blocker_ids": [100],
            }
        }
        result = engine._build_attacker_id_map(game_state)
        assert result == {}

    def test_build_blocker_id_map_with_context(self, engine):
        game_state = {
            "decision_context": {
                "type": "declare_blockers",
                "legal_blockers": ["Wall of Omens", "Elvish Mystic"],
                "legal_blocker_ids": [200, 201],
            }
        }
        result = engine._build_blocker_id_map(game_state)
        assert result == {"Wall of Omens": 200, "Elvish Mystic": 201}

    def test_build_blocker_id_map_no_ids(self, engine):
        game_state = {
            "decision_context": {
                "type": "declare_blockers",
                "legal_blockers": ["Wall of Omens"],
            }
        }
        result = engine._build_blocker_id_map(game_state)
        assert result == {}

    def test_build_blocker_id_map_mismatched_lengths(self, engine):
        """If names and ids have different lengths, return empty."""
        game_state = {
            "decision_context": {
                "type": "declare_blockers",
                "legal_blockers": ["Wall of Omens", "Elvish Mystic"],
                "legal_blocker_ids": [200],  # mismatched
            }
        }
        result = engine._build_blocker_id_map(game_state)
        assert result == {}

    def test_build_attacker_id_map_no_context(self, engine):
        """No decision context at all."""
        game_state = {}
        assert engine._build_attacker_id_map(game_state) == {}

    def test_build_blocker_id_map_no_context(self, engine):
        """No decision context at all."""
        game_state = {}
        assert engine._build_blocker_id_map(game_state) == {}


# ---- Declare attackers uses instance_id ----

class TestDeclareAttackersInstanceId:
    """Declare attackers uses instance_id for coordinate lookup when available."""

    def test_uses_instance_id_from_decision_context(self, engine, mock_mapper, mock_controller):
        from arenamcp.screen_mapper import ScreenCoord
        mock_mapper.get_permanent_coord.return_value = ScreenCoord(0.3, 0.5, "Grizzly Bears")
        mock_mapper.get_button_coord.return_value = ScreenCoord(0.5, 0.9, "done")

        action = GameAction(
            action_type=ActionType.DECLARE_ATTACKERS,
            attacker_names=["Grizzly Bears"],
        )
        game_state = {
            "battlefield": [
                {"name": "Grizzly Bears", "owner_seat_id": 1, "instance_id": 100},
            ],
            "players": [
                {"is_local": True, "seat_id": 1},
                {"is_local": False, "seat_id": 2},
            ],
            "decision_context": {
                "type": "declare_attackers",
                "legal_attackers": ["Grizzly Bears"],
                "legal_attacker_ids": [100],
            },
        }

        engine._exec_declare_attackers(action, game_state)

        # Verify get_permanent_coord was called with instance_id=100
        call_args = mock_mapper.get_permanent_coord.call_args
        assert call_args[0][1] == 100  # instance_id parameter


# ---- Select target uses instance_id ----

class TestSelectTargetInstanceId:
    """Select target uses instance_id for coordinate lookup."""

    def test_finds_instance_id_from_battlefield(self, engine, mock_mapper, mock_controller):
        from arenamcp.screen_mapper import ScreenCoord
        mock_mapper.get_permanent_coord.return_value = ScreenCoord(0.3, 0.5, "Goblin Guide")

        action = GameAction(
            action_type=ActionType.SELECT_TARGET,
            target_names=["Goblin Guide"],
        )
        game_state = {
            "battlefield": [
                {"name": "Goblin Guide", "owner_seat_id": 2, "instance_id": 300},
            ],
            "players": [
                {"is_local": True, "seat_id": 1},
                {"is_local": False, "seat_id": 2},
            ],
        }

        engine._exec_select_target(action, game_state)

        # Verify get_permanent_coord was called with instance_id=300
        call_args = mock_mapper.get_permanent_coord.call_args
        assert call_args[0][1] == 300  # instance_id parameter
