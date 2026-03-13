"""Tests for watcher startup modes (Workstream 2)."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from arenamcp.watcher import MTGALogWatcher


@pytest.fixture
def tmp_log(tmp_path):
    """Create a small fresh log file."""
    log_file = tmp_path / "Player.log"
    log_file.write_text("[UnityCrossThreadLogger]Session started\n")
    return log_file


@pytest.fixture
def large_log(tmp_path):
    """Create a large log file (simulates existing session)."""
    log_file = tmp_path / "Player.log"
    # Write >100KB of content
    log_file.write_text("x" * 200_000 + "\n")
    return log_file


class TestFreshLogDetection:
    def test_small_recent_file_is_fresh(self, tmp_log):
        callback = MagicMock()
        watcher = MTGALogWatcher(callback=callback, log_path=str(tmp_log))
        assert watcher._is_fresh_log() is True

    def test_large_file_is_not_fresh(self, large_log):
        callback = MagicMock()
        watcher = MTGALogWatcher(callback=callback, log_path=str(large_log))
        assert watcher._is_fresh_log() is False

    def test_missing_file_is_not_fresh(self, tmp_path):
        missing = tmp_path / "missing.log"
        callback = MagicMock()
        watcher = MTGALogWatcher(callback=callback, log_path=str(missing))
        assert watcher._is_fresh_log() is False

    def test_old_small_file_is_not_fresh(self, tmp_log):
        """A small file that's old (>60s) is not fresh."""
        import os
        # Set mtime to 120 seconds ago
        old_time = time.time() - 120
        os.utime(tmp_log, (old_time, old_time))

        callback = MagicMock()
        watcher = MTGALogWatcher(callback=callback, log_path=str(tmp_log))
        assert watcher._is_fresh_log() is False


class TestStartupModes:
    @patch("arenamcp.watcher.Observer")
    def test_fresh_log_skips_backfill_scan(self, mock_observer, tmp_log):
        """Fresh tiny log starts from position 0 without scanning for match start."""
        callback = MagicMock()
        watcher = MTGALogWatcher(callback=callback, log_path=str(tmp_log))

        with patch.object(watcher, "find_last_match_start") as mock_scan:
            watcher.start()
            # Should NOT call the expensive find_last_match_start
            mock_scan.assert_not_called()
            # Callback should have been called with log content
            assert callback.called

        watcher.stop()

    @patch("arenamcp.watcher.Observer")
    def test_resume_offset_used_directly(self, mock_observer, tmp_log):
        """Resume offset skips backfill entirely."""
        callback = MagicMock()
        watcher = MTGALogWatcher(
            callback=callback,
            log_path=str(tmp_log),
            resume_offset=10,
        )

        with patch.object(watcher, "find_last_match_start") as mock_scan:
            watcher.start()
            mock_scan.assert_not_called()

        watcher.stop()

    @patch("arenamcp.watcher.Observer")
    def test_large_log_uses_backfill(self, mock_observer, large_log):
        """Large log triggers backfill scan for match start."""
        callback = MagicMock()
        watcher = MTGALogWatcher(callback=callback, log_path=str(large_log))

        with patch.object(watcher, "find_last_match_start", return_value=150_000) as mock_scan:
            watcher.start()
            mock_scan.assert_called_once()

        watcher.stop()

    @patch("arenamcp.watcher.Observer")
    def test_no_match_found_starts_live(self, mock_observer, large_log):
        """When no match start found in large log, starts from end (live mode)."""
        callback = MagicMock()
        watcher = MTGALogWatcher(callback=callback, log_path=str(large_log))

        file_size = large_log.stat().st_size
        # find_last_match_start returns file_size when no match found
        with patch.object(watcher, "find_last_match_start", return_value=file_size):
            watcher.start()

        # Callback should still be invoked (may be with empty string)
        watcher.stop()

    @patch("arenamcp.watcher.Observer")
    def test_backfill_disabled_skips_all(self, mock_observer, large_log):
        """When backfill is disabled, no scanning happens."""
        callback = MagicMock()
        watcher = MTGALogWatcher(
            callback=callback,
            log_path=str(large_log),
            backfill=False,
        )

        with patch.object(watcher, "find_last_match_start") as mock_scan:
            watcher.start()
            mock_scan.assert_not_called()

        watcher.stop()
