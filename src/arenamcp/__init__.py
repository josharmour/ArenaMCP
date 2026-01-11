"""ArenaMCP: Real-time MCP server for MTGA game analysis."""

from typing import Optional

from arenamcp.watcher import MTGALogWatcher
from arenamcp.parser import LogParser
from arenamcp.gamestate import GameState, create_game_state_handler

__version__ = "0.1.0"


def create_log_pipeline(
    log_path: Optional[str] = None
) -> tuple[MTGALogWatcher, LogParser]:
    """Create a connected watcher -> parser pipeline.

    Convenience factory that wires the log watcher to feed the parser.

    Args:
        log_path: Path to MTGA Player.log. Defaults to MTGA_LOG_PATH env var
                 or standard Windows location.

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
    watcher = MTGALogWatcher(callback=parser.process_chunk, log_path=log_path)
    return watcher, parser


__all__ = [
    "__version__",
    "MTGALogWatcher",
    "LogParser",
    "create_log_pipeline",
    "GameState",
    "create_game_state_handler",
]
