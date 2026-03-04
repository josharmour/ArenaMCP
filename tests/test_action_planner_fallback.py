from arenamcp.action_planner import ActionPlanner, ActionType


class _StubBackend:
    def __init__(self, response: str):
        self._response = response

    def complete(self, system_prompt: str, user_message: str) -> str:
        return self._response


def test_planner_falls_back_from_markdown_text_to_legal_action() -> None:
    planner = ActionPlanner(_StubBackend("## Coach's Advice: Pass Priority"))
    plan = planner.plan_actions(
        game_state={"turn": {"turn_number": 3}},
        trigger="priority_gained",
        legal_actions=["Pass", "Cast Shock"],
    )
    assert len(plan.actions) == 1
    assert plan.actions[0].action_type == ActionType.PASS_PRIORITY


def test_planner_falls_back_from_backend_error_to_preferred_legal_action() -> None:
    planner = ActionPlanner(
        _StubBackend("Error: Codex CLI failed: error: unexpected argument '-q' found")
    )
    plan = planner.plan_actions(
        game_state={"turn": {"turn_number": 4}},
        trigger="new_turn",
        legal_actions=["Pass", "Cast Lightning Strike", "Play Land: Mountain"],
    )
    assert len(plan.actions) == 1
    assert plan.actions[0].action_type == ActionType.PLAY_LAND
    assert plan.actions[0].card_name == "Mountain"


def test_planner_keeps_valid_json_response() -> None:
    planner = ActionPlanner(
        _StubBackend(
            """{
  "actions": [{"action_type":"cast_spell","card_name":"Shock"}],
  "overall_strategy":"Use mana efficiently"
}"""
        )
    )
    plan = planner.plan_actions(
        game_state={"turn": {"turn_number": 2}},
        trigger="new_turn",
        legal_actions=["Cast Shock", "Pass"],
    )
    assert len(plan.actions) == 1
    assert plan.actions[0].action_type == ActionType.CAST_SPELL
    assert plan.actions[0].card_name == "Shock"
