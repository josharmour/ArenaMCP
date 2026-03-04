from arenamcp.standalone import StandaloneCoach


def test_standalone_initializes_advice_history():
    coach = StandaloneCoach(register_hotkeys=False)
    assert hasattr(coach, "_advice_history")
    assert isinstance(coach._advice_history, list)
