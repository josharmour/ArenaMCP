import launcher


def test_get_mtga_log_path_prefers_env(monkeypatch):
    monkeypatch.setenv("MTGA_LOG_PATH", r"C:\custom\Player.log")

    path = launcher.get_mtga_log_path()

    assert str(path) == r"C:\custom\Player.log"


def test_trim_player_log_if_needed_skips_when_mtga_running(monkeypatch, tmp_path):
    log_path = tmp_path / "Player.log"
    log_path.write_text("x" * 1024, encoding="utf-8")

    monkeypatch.setattr(launcher, "is_mtga_running", lambda: True)

    result = launcher.trim_player_log_if_needed(log_path=log_path, max_mb=1, keep_mb=1)

    assert result == "skipped: MTGA is running"
    assert log_path.read_text(encoding="utf-8") == "x" * 1024


def test_trim_player_log_if_needed_skips_when_below_threshold(monkeypatch, tmp_path):
    log_path = tmp_path / "Player.log"
    log_path.write_text("small log\n", encoding="utf-8")

    monkeypatch.setattr(launcher, "is_mtga_running", lambda: False)

    result = launcher.trim_player_log_if_needed(log_path=log_path, max_mb=1, keep_mb=1)

    assert result.startswith("skipped:")
    assert log_path.read_text(encoding="utf-8") == "small log\n"


def test_trim_player_log_if_needed_keeps_tail(monkeypatch, tmp_path):
    log_path = tmp_path / "Player.log"
    lines = [f"line-{i:06d}\n" for i in range(120000)]
    original = "".join(lines)
    log_path.write_text(original, encoding="utf-8")

    monkeypatch.setattr(launcher, "is_mtga_running", lambda: False)

    result = launcher.trim_player_log_if_needed(log_path=log_path, max_mb=1, keep_mb=1)

    trimmed = log_path.read_text(encoding="utf-8")
    assert result.startswith("trimmed:")
    assert len(trimmed.encode("utf-8")) <= 1024 * 1024
    assert trimmed.endswith(lines[-1])
    assert "line-000000" not in trimmed
    assert "line-119999" in trimmed
