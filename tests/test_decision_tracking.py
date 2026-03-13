"""Tests for GRE decision tracking — ActionsAvailableReq and PayCostsReq.

Verifies fixes for:
- ActionsAvailableReq setting pending_decision (was missing)
- PayCostsReq extracting source from manaCost[].objectId (was using nonexistent sourceId)
- Auto-clear logic not prematurely clearing actions_available/pay_costs decisions
"""
import copy
import time

from arenamcp.gamestate import GameState, create_game_state_handler


def _make_gs() -> GameState:
    gs = GameState()
    gs.set_local_seat_id(1, source=3)
    return gs


def _process(gs: GameState, messages: list[dict]) -> None:
    """Feed GRE messages through the handler as if from a GreToClientEvent."""
    handler = create_game_state_handler(gs)
    # Wrap messages in the GreToClientEvent envelope
    payload = {
        "greToClientEvent": {
            "greToClientMessages": messages,
        }
    }
    handler(payload)


# ---------------------------------------------------------------------------
# ActionsAvailableReq
# ---------------------------------------------------------------------------

def test_actions_available_sets_pending_decision():
    """ActionsAvailableReq should set pending_decision so the coach knows the game needs input."""
    gs = _make_gs()
    _process(gs, [{
        "type": "GREMessageType_ActionsAvailableReq",
        "actionsAvailableReq": {
            "actions": [
                {"actionType": "ActionType_Cast", "grpId": 12345, "instanceId": 100},
                {"actionType": "ActionType_Play", "grpId": 67890, "instanceId": 101},
                {"actionType": "ActionType_Pass"},
            ]
        }
    }])
    assert gs.pending_decision == "Priority"
    assert gs.decision_context is not None
    assert gs.decision_context["type"] == "actions_available"
    assert gs.decision_context["num_actions"] == 3


def test_actions_available_pass_only():
    """When only Pass is available, label as 'Priority (Pass Only)'."""
    gs = _make_gs()
    _process(gs, [{
        "type": "GREMessageType_ActionsAvailableReq",
        "actionsAvailableReq": {
            "actions": [{"actionType": "ActionType_Pass"}]
        }
    }])
    assert gs.pending_decision == "Priority (Pass Only)"


def test_actions_available_attack_decision():
    """Attack group actions should produce 'Declare Attackers' decision."""
    gs = _make_gs()
    _process(gs, [{
        "type": "GREMessageType_ActionsAvailableReq",
        "actionsAvailableReq": {
            "actions": [
                {"actionType": "ActionType_AttackWithGroup"},
                {"actionType": "ActionType_Pass"},
            ]
        }
    }])
    assert gs.pending_decision == "Declare Attackers"


def test_actions_available_block_decision():
    """Block group actions should produce 'Declare Blockers' decision."""
    gs = _make_gs()
    _process(gs, [{
        "type": "GREMessageType_ActionsAvailableReq",
        "actionsAvailableReq": {
            "actions": [
                {"actionType": "ActionType_BlockWithGroup"},
                {"actionType": "ActionType_Pass"},
            ]
        }
    }])
    assert gs.pending_decision == "Declare Blockers"


def test_actions_available_populates_legal_actions():
    """ActionsAvailableReq should still populate legal_actions and legal_actions_raw."""
    gs = _make_gs()
    _process(gs, [{
        "type": "GREMessageType_ActionsAvailableReq",
        "actionsAvailableReq": {
            "actions": [
                {"actionType": "ActionType_Cast", "grpId": 12345, "instanceId": 100},
                {"actionType": "ActionType_Pass"},
            ]
        }
    }])
    assert len(gs.legal_actions) == 2
    assert len(gs.legal_actions_raw) == 2
    assert "Pass" in gs.legal_actions


def test_actions_available_replaces_prior_decision():
    """A new ActionsAvailableReq should replace any prior pending_decision."""
    gs = _make_gs()
    # Set up a prior decision
    gs.pending_decision = "Select Targets"
    gs.decision_context = {"type": "target_selection", "source_id": 42}
    gs.decision_timestamp = time.time()

    _process(gs, [{
        "type": "GREMessageType_ActionsAvailableReq",
        "actionsAvailableReq": {
            "actions": [{"actionType": "ActionType_Pass"}]
        }
    }])
    assert gs.pending_decision == "Priority (Pass Only)"
    assert gs.decision_context["type"] == "actions_available"


# ---------------------------------------------------------------------------
# PayCostsReq
# ---------------------------------------------------------------------------

def test_pay_costs_extracts_source_from_mana_cost():
    """PayCostsReq should get source from manaCost[].objectId, not nonexistent sourceId."""
    gs = _make_gs()
    # Add a game object so the source can be resolved
    from arenamcp.gamestate import GameObject
    obj = GameObject(instance_id=200, grp_id=54321, owner_seat_id=1, zone_id=1)
    gs.game_objects[200] = obj

    _process(gs, [{
        "type": "GREMessageType_PayCostsReq",
        "payCostsReq": {
            "manaCost": [
                {"color": ["ManaColor_Green"], "count": 1, "objectId": 200},
                {"color": ["ManaColor_Any"], "count": 2, "objectId": 200},
            ],
            "autoTapActionsReq": {
                "autoTapSolutions": [{"autoTapActions": [{"instanceId": 300}]}]
            }
        }
    }])
    assert gs.pending_decision == "Pay Costs"
    ctx = gs.decision_context
    assert ctx["type"] == "pay_costs"
    assert ctx["source_id"] == 200
    assert ctx["has_autotap"] is True
    assert "G" in ctx["mana_cost"]


def test_pay_costs_no_source_uses_none():
    """PayCostsReq with empty manaCost should set source_id=None (longer timeout)."""
    gs = _make_gs()
    _process(gs, [{
        "type": "GREMessageType_PayCostsReq",
        "payCostsReq": {
            "manaCost": [],
        }
    }])
    assert gs.pending_decision == "Pay Costs"
    assert gs.decision_context["source_id"] is None


def test_pay_costs_mana_string_format():
    """Mana cost string should be human-readable."""
    gs = _make_gs()
    _process(gs, [{
        "type": "GREMessageType_PayCostsReq",
        "payCostsReq": {
            "manaCost": [
                {"color": ["ManaColor_White", "ManaColor_White"], "count": 1},
                {"color": ["ManaColor_Any"], "count": 3},
            ],
        }
    }])
    mana = gs.decision_context["mana_cost"]
    assert "WW" in mana
    assert "3" in mana


# ---------------------------------------------------------------------------
# Auto-clear logic
# ---------------------------------------------------------------------------

def test_actions_available_not_cleared_by_game_state_message():
    """GameStateMessage should NOT auto-clear actions_available decisions."""
    gs = _make_gs()
    _process(gs, [{
        "type": "GREMessageType_ActionsAvailableReq",
        "actionsAvailableReq": {
            "actions": [{"actionType": "ActionType_Pass"}]
        }
    }])
    assert gs.pending_decision is not None

    # Simulate a GameStateMessage arriving (e.g., board update)
    # Backdate the timestamp so it would normally be cleared
    gs.decision_timestamp = time.time() - 30  # 30 seconds old
    _process(gs, [{
        "type": "GREMessageType_GameStateMessage",
        "gameStateMessage": {
            "type": "GameStateType_Diff",
            "turnInfo": {"turnNumber": 5, "activePlayer": 1, "priorityPlayer": 1},
            "players": [{"seatId": 1, "lifeTotal": 18}],
        }
    }])
    # Should still be set — actions_available is not auto-cleared by GSM
    assert gs.pending_decision is not None


def test_pay_costs_not_cleared_by_game_state_message():
    """GameStateMessage should NOT auto-clear pay_costs decisions."""
    gs = _make_gs()
    _process(gs, [{
        "type": "GREMessageType_PayCostsReq",
        "payCostsReq": {
            "manaCost": [{"color": ["ManaColor_Green"], "count": 1}],
        }
    }])
    assert gs.pending_decision == "Pay Costs"

    gs.decision_timestamp = time.time() - 30
    _process(gs, [{
        "type": "GREMessageType_GameStateMessage",
        "gameStateMessage": {
            "type": "GameStateType_Diff",
            "turnInfo": {"turnNumber": 5, "activePlayer": 1, "priorityPlayer": 1},
            "players": [{"seatId": 1, "lifeTotal": 18}],
        }
    }])
    assert gs.pending_decision == "Pay Costs"


def test_target_selection_still_auto_clears():
    """Other decision types (e.g., target_selection) should still auto-clear normally."""
    gs = _make_gs()
    gs.pending_decision = "Select Targets"
    gs.decision_timestamp = time.time() - 30  # well past timeout
    gs.decision_context = {"type": "target_selection", "source_id": None}

    _process(gs, [{
        "type": "GREMessageType_GameStateMessage",
        "gameStateMessage": {
            "type": "GameStateType_Diff",
            "turnInfo": {"turnNumber": 5, "activePlayer": 1, "priorityPlayer": 1},
            "players": [{"seatId": 1, "lifeTotal": 18}],
        }
    }])
    # Should be cleared — target_selection is not exempt
    assert gs.pending_decision is None
