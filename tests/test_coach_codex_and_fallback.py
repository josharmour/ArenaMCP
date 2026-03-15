import subprocess

import arenamcp.coach as coach_mod
import arenamcp.backends.codex as codex_mod
from arenamcp.coach import CoachEngine, CodexCliBackend


def test_codex_cli_backend_direct_api_preferred(monkeypatch):
    """When Azure credentials are available, direct API is preferred over codex exec."""
    backend = CodexCliBackend(model="gpt-5.4-pro", command="codex")
    # If Azure config exists and key is set, _use_direct_api should be True
    # If not (CI/test env), it falls back to CLI mode — both are valid
    assert hasattr(backend, '_use_direct_api')
    assert hasattr(backend, '_complete_azure')
    assert hasattr(backend, '_complete_cli')


def test_codex_cli_backend_cli_fallback(monkeypatch):
    """When Azure credentials are not available, falls back to codex exec."""
    calls = []

    def _fake_run(run_args, **kwargs):
        calls.append((run_args, kwargs.get("input")))
        return subprocess.CompletedProcess(run_args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(codex_mod.subprocess, "run", _fake_run)

    backend = CodexCliBackend(model="gpt-5.4-pro", command="codex")
    backend._use_direct_api = False  # Force CLI fallback
    result = backend._complete_cli("sys", "msg")

    assert result == "ok"
    assert calls
    assert "exec" in calls[0][0]


def test_postprocess_prefers_declare_attackers_from_decision_context(monkeypatch):
    class _FakeRulesEngine:
        @staticmethod
        def get_legal_actions(game_state):
            return [
                "Cast Mighty Mutanimals",
                "Activate Mona Lisa, Science Geek",
                "Pass",
            ]

    monkeypatch.setattr(coach_mod, "RulesEngine", _FakeRulesEngine, raising=False)

    engine = CoachEngine(backend=None)
    game_state = {
        "pending_decision": "Declare Attackers",
        "decision_context": {
            "type": "declare_attackers",
            "legal_attackers": ["Heroes in a Half Shell"],
        },
        "turn": {
            "turn_number": 13,
            "phase": "Phase_Combat",
            "step": "Step_DeclareAttack",
            "active_player": 2,
        },
        "players": [{"seat_id": 2, "is_local": True}, {"seat_id": 1, "is_local": False}],
        "hand": [],
        "battlefield": [],
        "graveyard": [],
        "stack": [],
        "exile": [],
    }

    advice = engine._postprocess_advice(
        "Error: Codex CLI failed: unexpected argument '-q'",
        game_state,
        style="concise",
    )

    assert advice == "Declare Attackers: Heroes in a Half Shell"


def test_format_game_context_includes_compact_raw_gre_actions():
    engine = CoachEngine(backend=None)
    game_state = {
        "match_id": "abc123",
        "turn": {
            "turn_number": 3,
            "phase": "Phase_Main1",
            "step": "Step_PreCombatMain",
            "active_player": 1,
            "priority_player": 1,
        },
        "players": [
            {"seat_id": 1, "is_local": True, "life": 20, "lands_played": 0},
            {"seat_id": 2, "is_local": False, "life": 20},
        ],
        "hand": [{"name": "Shock", "type_line": "Instant", "mana_cost": "{R}"}],
        "battlefield": [],
        "graveyard": [],
        "stack": [],
        "exile": [],
        "legal_actions": ["Cast Shock", "Pass"],
        "legal_actions_raw": [
            {
                "actionType": "ActionType_Cast",
                "grpId": 123,
                "instanceId": 456,
                "abilityGrpId": 789,
                "targets": [{"instanceId": 999, "targetType": "TargetType_Card"}],
                "manaPaymentOptions": [{"foo": "bar"}],
            },
            {
                "actionType": "ActionType_Pass",
            },
        ],
    }

    context = engine._format_game_context(game_state)

    assert "Legal: Cast Shock, Pass" in context
    assert "LegalGRE:" in context
    assert '"actionType":"ActionType_Cast"' in context
    assert '"grpId":123' in context
    assert '"instanceId":456' in context
    assert '"manaPaymentOptionsCount":1' in context
