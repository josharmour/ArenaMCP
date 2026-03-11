from arenamcp.coach import get_models_for_provider


def test_codex_cli_models_match_arena_mcp_configuration():
    assert get_models_for_provider("codex-cli") == [
        ("GPT-5.4 Pro", "gpt-5.4-pro"),
        ("GPT-5.3 Codex", "gpt-5.3-codex"),
    ]
