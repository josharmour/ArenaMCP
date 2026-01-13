"""MTGA log file watcher using watchdog.

This module provides real-time monitoring of the MTGA Player.log file,
delivering new content via callback as it's written.
"""

import os
import logging
import re
from pathlib import Path
from typing import Callable, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent

logger = logging.getLogger(__name__)

# Pattern to identify match start in logs (MatchCreated event or match game room)
MATCH_START_PATTERN = re.compile(
    r'\[UnityCrossThreadLogger\].*(?:MatchCreated|Event_Join|"matchId":|MatchGameRoomStateChangedEvent)'
)

# Default MTGA log path on Windows
# Use LOCALAPPDATA approach which is more reliable than APPDATA/../LocalLow
_local_appdata = os.environ.get("LOCALAPPDATA", "")
if _local_appdata:
    # LOCALAPPDATA is C:\Users\<user>\AppData\Local, we need LocalLow sibling
    DEFAULT_LOG_PATH = os.path.join(
        os.path.dirname(_local_appdata),
        "LocalLow",
        "Wizards Of The Coast",
        "MTGA",
        "Player.log"
    )
else:
    # Fallback to hardcoded path for joshu user
    DEFAULT_LOG_PATH = r"C:\Users\joshu\AppData\LocalLow\Wizards Of The Coast\MTGA\Player.log"


class MTGALogHandler(FileSystemEventHandler):
    """FileSystemEventHandler that tracks file position for incremental reads."""

    def __init__(self, log_path: str, callback: Callable[[str], None]) -> None:
        """Initialize the handler.

        Args:
            log_path: Path to the MTGA Player.log file.
            callback: Function called with new log content as it's written.
        """
        super().__init__()
        self.log_path = Path(log_path).resolve()
        self.callback = callback
        self.file_position: int = 0

        # Initialize position to end of file if it exists
        if self.log_path.exists():
            try:
                self.file_position = self.log_path.stat().st_size
                logger.debug(f"Initialized file position to {self.file_position}")
            except OSError as e:
                logger.warning(f"Could not get file size: {e}")
                self.file_position = 0

    def on_modified(self, event: FileModifiedEvent) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return

        # Check if this is our target file
        event_path = Path(event.src_path).resolve()
        if event_path != self.log_path:
            return

        self._read_new_content()

    def on_created(self, event: FileCreatedEvent) -> None:
        """Handle file creation events (log truncation on MTGA restart)."""
        if event.is_directory:
            return

        event_path = Path(event.src_path).resolve()
        if event_path != self.log_path:
            return

        # Reset position when file is recreated
        logger.info("Log file recreated, resetting position to 0")
        self.file_position = 0
        self._read_new_content()

    def _read_new_content(self) -> None:
        """Read new content from the log file and invoke callback."""
        try:
            with open(self.log_path, 'r', encoding='utf-8', errors='replace') as f:
                # Check if file was truncated (position beyond file size)
                f.seek(0, 2)  # Seek to end
                file_size = f.tell()

                if file_size < self.file_position:
                    # File was truncated, reset to beginning
                    logger.info(f"File truncated (size {file_size} < position {self.file_position}), resetting")
                    self.file_position = 0

                # Seek to our tracked position and read new content
                f.seek(self.file_position)
                new_content = f.read()

                if new_content:
                    self.file_position = f.tell()
                    logger.debug(f"Read {len(new_content)} chars, new position: {self.file_position}")
                    self.callback(new_content)

        except FileNotFoundError:
            logger.debug("Log file not found (MTGA may not be running)")
        except PermissionError as e:
            # Windows file locking - retry is handled by watchdog's next event
            logger.debug(f"Permission error reading log: {e}")
        except OSError as e:
            logger.warning(f"Error reading log file: {e}")

    def read_from_position(self, start_position: int) -> None:
        """Read content from a specific position and invoke callback.

        Used for backfilling existing log content on startup.

        Args:
            start_position: Byte position to start reading from.
        """
        try:
            with open(self.log_path, 'r', encoding='utf-8', errors='replace') as f:
                f.seek(start_position)
                content = f.read()
                self.file_position = f.tell()

                if content:
                    logger.info(f"Backfill: read {len(content)} chars from position {start_position}")
                    self.callback(content)

        except FileNotFoundError:
            logger.debug("Log file not found for backfill")
        except OSError as e:
            logger.warning(f"Error during backfill read: {e}")


class MTGALogWatcher:
    """Watches the MTGA Player.log file for changes and delivers new content via callback."""

    def __init__(
        self,
        callback: Callable[[str], None],
        log_path: Optional[str] = None,
        backfill: bool = True
    ) -> None:
        """Initialize the log watcher.

        Args:
            callback: Function called with new log content as chunks of text.
            log_path: Path to Player.log. Defaults to MTGA_LOG_PATH env var
                     or standard Windows location.
            backfill: If True, parse existing log content from the last match
                     start on first call to start(). Enables catching up on
                     in-progress games. Defaults to True.
        """
        # Resolve log path
        if log_path is None:
            log_path = os.environ.get("MTGA_LOG_PATH", DEFAULT_LOG_PATH)

        self.log_path = Path(log_path).resolve()
        self.callback = callback
        self._backfill_enabled = backfill
        self._observer: Optional[Observer] = None
        self._handler: Optional[MTGALogHandler] = None

        logger.info(f"MTGALogWatcher initialized for: {self.log_path}")

    def find_last_match_start(self) -> int:
        """Find the byte position of the last match start in the log file.

        Scans the log file to find the most recent MatchCreated or similar
        match start indicator. This allows catching up on in-progress games.

        Returns:
            Byte position to start reading from, or 0 if no match found.
        """
        if not self.log_path.exists():
            return 0

        try:
            # Read as bytes to get accurate byte positions for seek()
            with open(self.log_path, 'rb') as f:
                content_bytes = f.read()

            # Decode for regex matching, tracking byte positions
            content = content_bytes.decode('utf-8', errors='replace')

            # Find all match start positions (character positions)
            last_char_pos = 0
            for match in MATCH_START_PATTERN.finditer(content):
                # Find the start of this line (search backward for newline)
                line_start = content.rfind('\n', 0, match.start())
                last_char_pos = line_start + 1 if line_start != -1 else 0

            if last_char_pos > 0:
                # Convert character position to byte position
                last_byte_pos = len(content[:last_char_pos].encode('utf-8'))
                logger.info(f"Found last match start at byte position {last_byte_pos}")
                return last_byte_pos
            else:
                logger.info("No match start found in log, will read from beginning")
                return 0

        except OSError as e:
            logger.warning(f"Error scanning log for match start: {e}")
            return 0

    def start(self) -> None:
        """Start watching the log file.

        Creates a watchdog Observer that monitors the directory containing
        the log file for modifications. If backfill is enabled, first processes
        existing log content from the last match start.
        """
        if self._observer is not None:
            logger.warning("Watcher already started")
            return

        # Ensure parent directory exists
        watch_dir = self.log_path.parent
        if not watch_dir.exists():
            logger.warning(f"Watch directory does not exist: {watch_dir}")
            # Still set up the watcher - it will detect when directory is created

        self._handler = MTGALogHandler(str(self.log_path), self.callback)

        # Backfill existing content if enabled
        if self._backfill_enabled and self.log_path.exists():
            start_pos = self.find_last_match_start()
            self._handler.read_from_position(start_pos)

        self._observer = Observer()

        # Watch the parent directory (watchdog requires watching directories)
        self._observer.schedule(self._handler, str(watch_dir), recursive=False)
        self._observer.start()

        logger.info(f"Started watching: {watch_dir}")

    def stop(self) -> None:
        """Stop watching the log file and clean up resources."""
        if self._observer is None:
            logger.debug("Watcher not running")
            return

        self._observer.stop()
        self._observer.join(timeout=5.0)
        self._observer = None
        self._handler = None

        logger.info("Watcher stopped")

    def __enter__(self) -> "MTGALogWatcher":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
