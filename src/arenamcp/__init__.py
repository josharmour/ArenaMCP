"""ArenaMCP: Real-time MCP server for MTGA game analysis."""

from typing import Optional

from arenamcp.watcher import MTGALogWatcher
from arenamcp.parser import LogParser
from arenamcp.gamestate import GameState, create_game_state_handler
from arenamcp.scryfall import ScryfallCache, ScryfallCard
from arenamcp.draftstats import DraftStatsCache, DraftStats
from arenamcp.server import mcp, start_watching, stop_watching
from arenamcp.voice import VoiceInput
from arenamcp.tts import VoiceOutput, KokoroTTS
from arenamcp.coach import (
    CoachEngine,
    GameStateTrigger,
    create_backend,
    ClaudeBackend,
    GeminiBackend,
    OllamaBackend,
)

__version__ = "0.1.0"


def create_log_pipeline(
    log_path: Optional[str] = None,
    backfill: bool = True
) -> tuple[MTGALogWatcher, LogParser]:
    """Create a connected watcher -> parser pipeline.

    Convenience factory that wires the log watcher to feed the parser.

    Args:
        log_path: Path to MTGA Player.log. Defaults to MTGA_LOG_PATH env var
                 or standard Windows location.
        backfill: If True, parse existing log content from the last match
                 start when the watcher starts. Enables catching up on
                 in-progress games. Defaults to True.

    Returns:
        Tuple of (watcher, parser). Start the watcher to begin processing.
        Register handlers on the parser before starting the watcher.

    Example:
        watcher, parser = create_log_pipeline()
        parser.register_handler('GreToClientEvent', handle_game_event)
        with watcher:
            # Events flow through pipeline
            time.sleep(10)
    """
    parser = LogParser()
    watcher = MTGALogWatcher(
        callback=parser.process_chunk,
        log_path=log_path,
        backfill=backfill
    )
    return watcher, parser


__all__ = [
    "__version__",
    "MTGALogWatcher",
    "LogParser",
    "create_log_pipeline",
    "GameState",
    "create_game_state_handler",
    "ScryfallCache",
    "ScryfallCard",
    "DraftStatsCache",
    "DraftStats",
    "mcp",
    "start_watching",
    "stop_watching",
    "VoiceInput",
    "VoiceOutput",
    "KokoroTTS",
    "CoachEngine",
    "GameStateTrigger",
    "create_backend",
    "ClaudeBackend",
    "GeminiBackend",
    "OllamaBackend",
]
