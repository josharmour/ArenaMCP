"""Regression tests for _is_planner_action_stale_vs_bridge.

The original detector caught two stale shapes:
  - PLAY_LAND/CAST_SPELL against ActionsAvailable with no matching Play/Cast
    entries (issues #136 #137 #139 #140).
  - DECLARE_ATTACKERS/BLOCKERS against a non-combat request class (window
    changed mid-LLM-call).

After the v2.3.0+ release the SelectN bug-cluster (#189 et al.) showed a
third shape: PLAY_LAND/CAST_SPELL against an entirely different request
type (SelectN/Search/SelectTargets/PayCosts/CastingTimeOption). A new
decision window opened between plan-generation and submission, displacing
the planned step. These tests pin that this shape is now stale-detected
and silent-skipped instead of flipping autopilot to MANUAL REQUIRED.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import importlib
import sys


def _engine_with_method():
    try:
        autopilot = importlib.import_module("arenamcp.autopilot")
    except ImportError as e:
        if "PIL" in str(e):
            sys.modules["PIL"] = MagicMock()
            sys.modules["PIL.ImageGrab"] = MagicMock()
            autopilot = importlib.import_module("arenamcp.autopilot")
        else:
            raise

    engine = autopilot.AutopilotEngine.__new__(autopilot.AutopilotEngine)
    return engine, autopilot


def _action(action_type, **kw):
    _, autopilot = _engine_with_method()
    return autopilot.GameAction(action_type=action_type, **kw)


def _make_play_land(card="Escape Tunnel"):
    _, ap = _engine_with_method()
    return ap.GameAction(action_type=ap.ActionType.PLAY_LAND, card_name=card)


def _make_cast_spell(card="Lightning Bolt"):
    _, ap = _engine_with_method()
    return ap.GameAction(action_type=ap.ActionType.CAST_SPELL, card_name=card)


def test_play_land_against_select_n_is_stale():
    """Shape #2: planner picked play_land but bridge moved to a SelectN window."""
    engine, _ = _engine_with_method()
    state = {
        "_bridge_request_type": "SelectN",
        "_bridge_request_class": "SelectNRequest",
        "_bridge_actions": [],
    }
    assert engine._is_planner_action_stale_vs_bridge(_make_play_land(), state) is True


def test_cast_spell_against_search_is_stale():
    engine, _ = _engine_with_method()
    state = {
        "_bridge_request_type": "Search",
        "_bridge_request_class": "SearchRequest",
        "_bridge_actions": [],
    }
    assert engine._is_planner_action_stale_vs_bridge(_make_cast_spell(), state) is True


def test_play_land_against_select_targets_is_stale():
    engine, _ = _engine_with_method()
    state = {
        "_bridge_request_type": "SelectTargets",
        "_bridge_request_class": "SelectTargetsRequest",
    }
    assert engine._is_planner_action_stale_vs_bridge(_make_play_land(), state) is True


def test_play_land_against_pay_costs_is_stale():
    engine, _ = _engine_with_method()
    state = {
        "_bridge_request_type": "PayCosts",
        "_bridge_request_class": "PayCostsRequest",
    }
    assert engine._is_planner_action_stale_vs_bridge(_make_play_land(), state) is True


def test_play_land_against_actions_available_with_play_match_is_not_stale():
    """Bridge IS ActionsAvailable AND offers a Play entry — not stale."""
    engine, _ = _engine_with_method()
    state = {
        "_bridge_request_type": "ActionsAvailable",
        "_bridge_request_class": "ActionsAvailableRequest",
        "_bridge_actions": [
            {"actionType": "Play", "grpId": 12345},
            {"actionType": "Cast", "grpId": 67890},
        ],
    }
    assert engine._is_planner_action_stale_vs_bridge(_make_play_land(), state) is False


def test_play_land_against_actions_available_without_play_is_stale():
    """Shape #1 — bridge IS ActionsAvailable but no Play entries (lands_played already used)."""
    engine, _ = _engine_with_method()
    state = {
        "_bridge_request_type": "ActionsAvailable",
        "_bridge_request_class": "ActionsAvailableRequest",
        "_bridge_actions": [{"actionType": "Cast", "grpId": 67890}],
    }
    assert engine._is_planner_action_stale_vs_bridge(_make_play_land(), state) is True


def test_play_land_with_no_bridge_request_at_all_is_not_stale():
    """No bridge request info → don't second-guess; let the normal path run."""
    engine, _ = _engine_with_method()
    state = {
        "_bridge_request_type": None,
        "_bridge_request_class": None,
        "_bridge_actions": None,
    }
    assert engine._is_planner_action_stale_vs_bridge(_make_play_land(), state) is False


def test_declare_attackers_against_main_phase_request_is_stale():
    """Shape #3 — combat action against non-combat bridge state."""
    engine, ap = _engine_with_method()
    action = ap.GameAction(
        action_type=ap.ActionType.DECLARE_ATTACKERS,
        attacker_names=["Goblin Guide"],
    )
    state = {
        "_bridge_request_type": "ActionsAvailable",
        "_bridge_request_class": "ActionsAvailableRequest",
    }
    assert engine._is_planner_action_stale_vs_bridge(action, state) is True


def test_declare_attackers_against_combat_request_is_not_stale():
    engine, ap = _engine_with_method()
    action = ap.GameAction(
        action_type=ap.ActionType.DECLARE_ATTACKERS,
        attacker_names=["Goblin Guide"],
    )
    state = {
        "_bridge_request_type": "DeclareAttackersReq",
        "_bridge_request_class": "DeclareAttackersRequest",
    }
    assert engine._is_planner_action_stale_vs_bridge(action, state) is False


def test_select_n_action_against_select_n_request_is_not_stale():
    """A real SelectN action against SelectNRequest is the right pairing
    and must pass through to the normal _try_gre_bridge_select_n path."""
    engine, ap = _engine_with_method()
    action = ap.GameAction(
        action_type=ap.ActionType.SELECT_N,
        select_card_names=["Forest"],
    )
    state = {
        "_bridge_request_type": "SelectN",
        "_bridge_request_class": "SelectNRequest",
    }
    assert engine._is_planner_action_stale_vs_bridge(action, state) is False


def test_search_library_against_no_bridge_request_is_stale():
    """The race scenario from #187: planner picked search_library but the
    SearchRequest resolved itself before the autopilot's submit attempt,
    so the bridge poll returns no pending request."""
    engine, ap = _engine_with_method()
    action = ap.GameAction(
        action_type=ap.ActionType.SEARCH_LIBRARY,
        select_card_names=["Forest"],
    )
    state = {
        "_bridge_request_type": None,
        "_bridge_request_class": None,
    }
    assert engine._is_planner_action_stale_vs_bridge(action, state) is True


def test_search_library_against_search_request_is_not_stale():
    engine, ap = _engine_with_method()
    action = ap.GameAction(
        action_type=ap.ActionType.SEARCH_LIBRARY,
        select_card_names=["Forest"],
    )
    state = {
        "_bridge_request_type": "Search",
        "_bridge_request_class": "SearchRequest",
    }
    assert engine._is_planner_action_stale_vs_bridge(action, state) is False


def test_select_n_against_actions_available_is_stale():
    """Window changed: planner picked SELECT_N but bridge moved on to a
    fresh ActionsAvailable window."""
    engine, ap = _engine_with_method()
    action = ap.GameAction(
        action_type=ap.ActionType.SELECT_N, select_card_names=["Forest"]
    )
    state = {
        "_bridge_request_type": "ActionsAvailable",
        "_bridge_request_class": "ActionsAvailableRequest",
    }
    assert engine._is_planner_action_stale_vs_bridge(action, state) is True
