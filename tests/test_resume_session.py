"""Tests for session-aware resume logic (Workstream 1)."""

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from arenamcp.gamestate import (
    save_match_state,
    load_match_state,
    validate_log_identity,
    GameState,
    MATCH_STATE_PATH,
)


@pytest.fixture
def tmp_log(tmp_path):
    """Create a temporary log file with some content."""
    log_file = tmp_path / "Player.log"
    log_file.write_text("some log content here\n" * 100)
    return log_file


@pytest.fixture
def game_state_with_match():
    """Create a GameState with an active match."""
    gs = GameState()
    gs.match_id = "test-match-123"
    gs.local_seat_id = 1
    gs.turn_info.turn_number = 3
    gs.turn_info.phase = "Phase_Main1"
    return gs


@pytest.fixture(autouse=True)
def clean_match_state():
    """Remove saved match state before/after each test."""
    if MATCH_STATE_PATH.exists():
        MATCH_STATE_PATH.unlink()
    yield
    if MATCH_STATE_PATH.exists():
        MATCH_STATE_PATH.unlink()


class TestSaveMatchState:
    def test_saves_log_identity(self, game_state_with_match, tmp_log):
        save_match_state(
            game_state_with_match,
            log_offset=5000,
            log_path=str(tmp_log),
        )

        with open(MATCH_STATE_PATH) as f:
            state = json.load(f)

        assert "log_identity" in state
        assert state["log_identity"]["path"] == str(tmp_log)
        assert state["log_identity"]["size"] == tmp_log.stat().st_size
        assert state["log_identity"]["mtime"] == pytest.approx(
            tmp_log.stat().st_mtime, abs=1
        )

    def test_saves_without_log_path(self, game_state_with_match):
        save_match_state(game_state_with_match, log_offset=5000)

        with open(MATCH_STATE_PATH) as f:
            state = json.load(f)

        assert "log_identity" not in state

    def test_saves_without_nonexistent_log(self, game_state_with_match, tmp_path):
        missing = tmp_path / "missing.log"
        save_match_state(
            game_state_with_match,
            log_offset=5000,
            log_path=str(missing),
        )

        with open(MATCH_STATE_PATH) as f:
            state = json.load(f)

        assert "log_identity" not in state


class TestValidateLogIdentity:
    def test_same_session_resumes(self, tmp_log):
        """Saved offset with same file and matching metadata resumes."""
        stat = tmp_log.stat()
        saved = {
            "log_offset": 1000,
            "log_identity": {
                "path": str(tmp_log),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            },
        }

        result = validate_log_identity(saved, str(tmp_log))
        assert result == "resume_same_session"

    def test_smaller_file_rejects(self, tmp_log):
        """Saved offset with smaller/new file does not resume."""
        saved = {
            "log_offset": 50000,
            "log_identity": {
                "path": str(tmp_log),
                "size": 100000,  # saved size larger than current
                "mtime": tmp_log.stat().st_mtime,
            },
        }

        result = validate_log_identity(saved, str(tmp_log))
        assert result == "fresh_log_after_restart"

    def test_changed_path_rejects(self, tmp_log, tmp_path):
        """Saved offset with different path does not resume."""
        other_log = tmp_path / "other.log"
        other_log.write_text("different log")

        saved = {
            "log_offset": 1000,
            "log_identity": {
                "path": str(tmp_log),
                "size": tmp_log.stat().st_size,
                "mtime": tmp_log.stat().st_mtime,
            },
        }

        result = validate_log_identity(saved, str(other_log))
        assert result == "resume_invalid_path"

    def test_missing_log_rejects(self, tmp_path):
        """Missing log path skips resume cleanly."""
        saved = {
            "log_offset": 1000,
            "log_identity": {
                "path": str(tmp_path / "gone.log"),
                "size": 5000,
                "mtime": time.time(),
            },
        }

        result = validate_log_identity(saved, str(tmp_path / "gone.log"))
        assert result == "resume_invalid_path"

    def test_no_identity_allows_resume(self):
        """Legacy saved state without log_identity allows resume."""
        saved = {"log_offset": 1000}
        result = validate_log_identity(saved, "/some/path")
        assert result == "resume_no_identity"

    def test_no_current_path_rejects(self):
        """No current log path provided rejects resume."""
        saved = {
            "log_identity": {"path": "/old/path", "size": 100, "mtime": 0},
        }
        result = validate_log_identity(saved, None)
        assert result == "resume_invalid_path"

    def test_large_mtime_gap_with_growth_is_ambiguous(self, tmp_log):
        """File grew but mtime jumped significantly → appendlog ambiguous."""
        stat = tmp_log.stat()
        saved = {
            "log_offset": 100,
            "log_identity": {
                "path": str(tmp_log),
                "size": stat.st_size - 500,  # file grew
                "mtime": stat.st_mtime - 600,  # mtime jumped > 300s
            },
        }

        result = validate_log_identity(saved, str(tmp_log))
        assert result == "resume_append_mode_ambiguous"

    def test_file_grew_normally_resumes(self, tmp_log):
        """File grew and mtime advanced normally → same session."""
        stat = tmp_log.stat()
        saved = {
            "log_offset": 100,
            "log_identity": {
                "path": str(tmp_log),
                "size": stat.st_size - 100,  # file grew slightly
                "mtime": stat.st_mtime - 10,  # mtime advanced ~10s
            },
        }

        result = validate_log_identity(saved, str(tmp_log))
        assert result == "resume_same_session"


class TestLoadMatchState:
    def test_returns_none_when_no_file(self):
        assert load_match_state() is None

    def test_returns_none_for_old_state(self, game_state_with_match):
        save_match_state(game_state_with_match, log_offset=100)

        # Tamper with timestamp to make it old
        with open(MATCH_STATE_PATH) as f:
            state = json.load(f)
        state["timestamp"] = time.time() - 3600
        with open(MATCH_STATE_PATH, "w") as f:
            json.dump(state, f)

        assert load_match_state() is None

    def test_returns_none_for_ended_state(self, game_state_with_match):
        save_match_state(game_state_with_match, log_offset=100)

        with open(MATCH_STATE_PATH) as f:
            state = json.load(f)
        state["status"] = "ended"
        with open(MATCH_STATE_PATH, "w") as f:
            json.dump(state, f)

        assert load_match_state() is None

    def test_returns_valid_state(self, game_state_with_match):
        save_match_state(game_state_with_match, log_offset=100)
        state = load_match_state()
        assert state is not None
        assert state["match_id"] == "test-match-123"
        assert state["log_offset"] == 100
