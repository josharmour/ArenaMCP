"""FastMCP server exposing MTGA game state and card information.

This module provides the MCP server that bridges live MTGA games to Claude,
implementing the Calculator + Coach pattern: deterministic code tracks state
while the LLM provides strategic analysis.
"""

import logging
import threading
import time
from collections import deque
from typing import Any, Literal, Optional

from mcp.server.fastmcp import FastMCP

from arenamcp.coach import CoachEngine, GameStateTrigger, create_backend

from arenamcp.gamestate import GameState, ZoneType, create_game_state_handler
from arenamcp.parser import LogParser
from arenamcp.scryfall import ScryfallCache
from arenamcp.draftstats import DraftStatsCache
from arenamcp.watcher import MTGALogWatcher
from arenamcp.voice import VoiceInput
from arenamcp.tts import VoiceOutput

logger = logging.getLogger(__name__)

# Initialize FastMCP server with STDIO transport
mcp = FastMCP("mtga")

# Module-level state instances
game_state: GameState = GameState()
parser: LogParser = LogParser()
watcher: Optional[MTGALogWatcher] = None

# Lazy-loaded caches (avoid blocking startup with bulk downloads)
_scryfall: Optional[ScryfallCache] = None
_draft_stats: Optional[DraftStatsCache] = None

# Lazy-loaded voice components
_voice_input: Optional[VoiceInput] = None
_voice_output: Optional[VoiceOutput] = None

# Advice queue for proactive coaching (Phase 9 integration)
_pending_advice: deque[dict[str, Any]] = deque(maxlen=10)

# Background coaching state
_coaching_thread: Optional[threading.Thread] = None
_coaching_enabled: bool = False
_coaching_backend: Optional[str] = None
_coaching_auto_speak: bool = False


def _get_scryfall() -> ScryfallCache:
    """Get or initialize the Scryfall cache (lazy loading)."""
    global _scryfall
    if _scryfall is None:
        logger.info("Initializing Scryfall cache...")
        _scryfall = ScryfallCache()
    return _scryfall


def _get_draft_stats() -> DraftStatsCache:
    """Get or initialize the draft stats cache (lazy loading)."""
    global _draft_stats
    if _draft_stats is None:
        logger.info("Initializing 17lands draft stats cache...")
        _draft_stats = DraftStatsCache()
    return _draft_stats


def _get_voice_input(mode: Literal["ptt", "vox"] = "ptt") -> VoiceInput:
    """Get or initialize voice input (lazy loading).

    Args:
        mode: Voice input mode - 'ptt' for push-to-talk, 'vox' for voice activation.

    Returns:
        VoiceInput instance configured for the requested mode.
    """
    global _voice_input
    # Recreate if mode changed
    if _voice_input is None or _voice_input.mode != mode:
        logger.info(f"Initializing voice input in {mode} mode...")
        _voice_input = VoiceInput(mode=mode)
    return _voice_input


def _get_voice_output() -> VoiceOutput:
    """Get or initialize voice output (lazy loading)."""
    global _voice_output
    if _voice_output is None:
        logger.info("Initializing voice output (TTS)...")
        _voice_output = VoiceOutput()
    return _voice_output


def _background_coaching_loop(
    coach: CoachEngine,
    trigger_detector: GameStateTrigger,
    auto_speak: bool
) -> None:
    """Background loop that monitors game state and generates proactive advice.

    Args:
        coach: CoachEngine instance to generate advice
        trigger_detector: GameStateTrigger to detect state changes
        auto_speak: Whether to automatically speak advice via TTS
    """
    global _coaching_enabled

    logger.info("Background coaching loop started")
    prev_state: dict[str, Any] = {}

    while _coaching_enabled:
        try:
            # Get current game state
            curr_state = get_game_state()

            # Check for triggers (skip on first iteration when prev_state empty)
            if prev_state:
                triggers = trigger_detector.check_triggers(prev_state, curr_state)

                for trigger in triggers:
                    logger.debug(f"Trigger fired: {trigger}")
                    try:
                        advice = coach.get_advice(curr_state, trigger=trigger)
                        queue_advice(advice, trigger)

                        if auto_speak:
                            try:
                                voice = _get_voice_output()
                                voice.speak(advice, blocking=True)
                            except Exception as e:
                                logger.error(f"TTS error: {e}")
                    except Exception as e:
                        logger.error(f"Error getting advice for {trigger}: {e}")

            prev_state = curr_state

        except Exception as e:
            logger.error(f"Error in coaching loop: {e}")

        # Poll interval
        time.sleep(1.5)

    logger.info("Background coaching loop stopped")


def start_background_coaching(
    backend: str = "claude",
    auto_speak: bool = False
) -> None:
    """Start background game monitoring with proactive coaching.

    Creates a daemon thread that polls game state and generates advice
    when triggers fire.

    Args:
        backend: LLM backend to use ("claude", "gemini", "ollama")
        auto_speak: If True, automatically speak advice via TTS
    """
    global _coaching_thread, _coaching_enabled, _coaching_backend, _coaching_auto_speak

    if _coaching_enabled:
        raise RuntimeError("Background coaching already running")

    # Create coach components
    llm_backend = create_backend(backend)
    coach = CoachEngine(backend=llm_backend)
    trigger_detector = GameStateTrigger()

    # Store config
    _coaching_backend = backend
    _coaching_auto_speak = auto_speak
    _coaching_enabled = True

    # Start daemon thread
    _coaching_thread = threading.Thread(
        target=_background_coaching_loop,
        args=(coach, trigger_detector, auto_speak),
        daemon=True,
        name="coaching-loop"
    )
    _coaching_thread.start()
    logger.info(f"Started background coaching with {backend} backend, auto_speak={auto_speak}")


def stop_background_coaching() -> None:
    """Stop background coaching if running."""
    global _coaching_thread, _coaching_enabled, _coaching_backend, _coaching_auto_speak

    if not _coaching_enabled:
        raise RuntimeError("Background coaching not running")

    _coaching_enabled = False

    if _coaching_thread is not None:
        _coaching_thread.join(timeout=5.0)
        _coaching_thread = None

    _coaching_backend = None
    _coaching_auto_speak = False
    logger.info("Stopped background coaching")


def start_watching() -> None:
    """Start watching the MTGA log file for game events.

    Creates and starts the watcher if not already running.
    Watcher feeds log chunks to the parser, which updates game_state.
    """
    global watcher
    if watcher is not None:
        logger.debug("Watcher already running")
        return

    # Wire up the event handler
    handler = create_game_state_handler(game_state)
    parser.register_handler("GreToClientEvent", handler)

    # Create and start the watcher
    watcher = MTGALogWatcher(callback=parser.process_chunk)
    watcher.start()
    logger.info("Started MTGA log watcher")


def stop_watching() -> None:
    """Stop watching the MTGA log file."""
    global watcher
    if watcher is None:
        return
    watcher.stop()
    watcher = None
    logger.info("Stopped MTGA log watcher")


def enrich_with_oracle_text(grp_id: int) -> dict[str, Any]:
    """Look up card data from Scryfall and return enriched dict.

    Args:
        grp_id: MTGA arena_id for the card

    Returns:
        Dict with name, oracle_text, type_line, mana_cost if found,
        or minimal dict with just grp_id if lookup fails (graceful degradation).
    """
    if grp_id == 0:
        return {"grp_id": 0, "name": "Unknown", "oracle_text": "", "type_line": "", "mana_cost": ""}

    scryfall = _get_scryfall()
    card = scryfall.get_card_by_arena_id(grp_id)

    if card is None:
        return {
            "grp_id": grp_id,
            "name": f"Unknown (ID: {grp_id})",
            "oracle_text": "",
            "type_line": "",
            "mana_cost": "",
        }

    return {
        "grp_id": grp_id,
        "name": card.name,
        "oracle_text": card.oracle_text,
        "type_line": card.type_line,
        "mana_cost": card.mana_cost,
    }


def _serialize_game_object(obj) -> dict[str, Any]:
    """Serialize a GameObject with oracle text enrichment."""
    enriched = enrich_with_oracle_text(obj.grp_id)
    return {
        "instance_id": obj.instance_id,
        "grp_id": obj.grp_id,
        "name": enriched["name"],
        "oracle_text": enriched["oracle_text"],
        "type_line": enriched["type_line"],
        "mana_cost": enriched["mana_cost"],
        "owner_seat_id": obj.owner_seat_id,
        "controller_seat_id": obj.controller_seat_id,
        "power": obj.power,
        "toughness": obj.toughness,
        "is_tapped": obj.is_tapped,
    }


# MCP Tools

@mcp.tool()
def get_game_state() -> dict[str, Any]:
    """Get the complete current game state snapshot.

    Returns the full board state including turn info, player life totals,
    and all cards in each zone (battlefield, hand, graveyard, stack, exile).
    Each card includes oracle text for strategic analysis.

    Use this to understand the current game situation before providing advice.
    Call periodically during a game to track state changes.

    Returns:
        Dict with structure:
        - turn: {turn_number, active_player, priority_player, phase, step}
        - players: [{seat_id, life_total, mana_pool, is_local}]
        - battlefield: [card objects with oracle text]
        - hand: [card objects - local player only]
        - graveyard: [card objects]
        - stack: [card objects]
        - exile: [card objects]
    """
    # Auto-start watcher if not running
    if watcher is None:
        start_watching()

    # Serialize turn info
    turn = {
        "turn_number": game_state.turn_info.turn_number,
        "active_player": game_state.turn_info.active_player,
        "priority_player": game_state.turn_info.priority_player,
        "phase": game_state.turn_info.phase,
        "step": game_state.turn_info.step,
    }

    # Serialize players
    players = []
    for player in game_state.players.values():
        players.append({
            "seat_id": player.seat_id,
            "life_total": player.life_total,
            "mana_pool": player.mana_pool,
            "is_local": player.seat_id == game_state.local_seat_id,
        })

    # Serialize zones with oracle text enrichment
    battlefield = [_serialize_game_object(obj) for obj in game_state.battlefield]
    hand = [_serialize_game_object(obj) for obj in game_state.hand]
    graveyard = [_serialize_game_object(obj) for obj in game_state.graveyard]
    stack = [_serialize_game_object(obj) for obj in game_state.stack]
    exile = [_serialize_game_object(obj) for obj in game_state.get_objects_in_zone(ZoneType.EXILE)]

    return {
        "turn": turn,
        "players": players,
        "battlefield": battlefield,
        "hand": hand,
        "graveyard": graveyard,
        "stack": stack,
        "exile": exile,
    }


@mcp.tool()
def get_card_info(arena_id: int) -> dict[str, Any]:
    """Look up detailed card information by MTGA arena ID.

    Use this to get oracle text, mana cost, and other card details
    when you need to understand what a specific card does.

    Args:
        arena_id: The MTGA arena ID (grp_id) of the card

    Returns:
        Dict with name, oracle_text, type_line, mana_cost, cmc, colors, scryfall_uri
        or {"error": "Card not found"} if the card isn't in Scryfall data.
    """
    scryfall = _get_scryfall()
    card = scryfall.get_card_by_arena_id(arena_id)

    if card is None:
        return {"error": f"Card not found for arena_id {arena_id}"}

    return {
        "name": card.name,
        "oracle_text": card.oracle_text,
        "type_line": card.type_line,
        "mana_cost": card.mana_cost,
        "cmc": card.cmc,
        "colors": card.colors,
        "scryfall_uri": card.scryfall_uri,
    }


@mcp.tool()
def get_opponent_played_cards() -> list[dict[str, Any]]:
    """Get all cards the opponent has revealed this game.

    Tracks cards as they move from library to other zones (hand, battlefield,
    graveyard, etc.), building a picture of opponent's deck composition.

    Use this to:
    - Understand what cards opponent has access to
    - Predict what they might play based on revealed cards
    - Identify deck archetype from card patterns

    Returns:
        List of card dicts with grp_id, name, oracle_text, type_line, mana_cost.
        Empty list if no opponent or no cards revealed yet.
    """
    # Auto-start watcher if not running
    if watcher is None:
        start_watching()

    grp_ids = game_state.get_opponent_played_cards()

    cards = []
    for grp_id in grp_ids:
        enriched = enrich_with_oracle_text(grp_id)
        cards.append(enriched)

    return cards


@mcp.tool()
def get_draft_rating(card_name: str, set_code: str) -> dict[str, Any]:
    """Get 17lands draft statistics for a card.

    Provides win rate and pick order data from 17lands.com to help evaluate
    cards during draft. Data is from Premier Draft format.

    Args:
        card_name: The card name (case-insensitive)
        set_code: The set code (e.g., 'DSK', 'BLB', 'MKM')

    Returns:
        Dict with:
        - name: Card name
        - set_code: Set code
        - gih_wr: Games in Hand Win Rate (0.0-1.0, e.g., 0.55 = 55%)
        - alsa: Average Last Seen At (pick position)
        - iwd: Improvement When Drawn
        - games_in_hand: Sample size for statistics

        or {"error": "Card not found"} if not found.
    """
    draft_stats = _get_draft_stats()
    stats = draft_stats.get_draft_rating(card_name, set_code)

    if stats is None:
        return {"error": f"Card '{card_name}' not found in {set_code} draft data"}

    return {
        "name": stats.name,
        "set_code": stats.set_code,
        "gih_wr": stats.gih_wr,
        "alsa": stats.alsa,
        "iwd": stats.iwd,
        "games_in_hand": stats.games_in_hand,
    }


@mcp.tool()
def listen_for_voice(
    mode: Literal["ptt", "vox"] = "ptt",
    timeout: Optional[float] = None,
) -> dict[str, Any]:
    """Listen for voice input and return transcription.

    Blocks until voice is captured and transcribed. In PTT mode, waits for
    the user to press and release F4. In VOX mode, waits for voice activity.

    Use this to get spoken commands or questions from the user during gameplay.

    Args:
        mode: Voice input mode. 'ptt' (default) for push-to-talk with F4 key,
             'vox' for voice-activated detection.
        timeout: Maximum time to wait in seconds. None (default) waits forever.

    Returns:
        Dict with:
        - transcription: The transcribed text from speech
        - mode: The voice mode used ('ptt' or 'vox')

        or {"error": message} if timeout or failure occurs.
    """
    try:
        voice_input = _get_voice_input(mode)
        voice_input.start()
        transcription = voice_input.wait_for_speech(timeout=timeout)
        voice_input.stop()

        if not transcription:
            return {"error": "No speech detected or timeout", "mode": mode}

        return {"transcription": transcription, "mode": mode}
    except Exception as e:
        logger.exception("Voice input error")
        return {"error": str(e), "mode": mode}


@mcp.tool()
def speak_advice(text: str) -> dict[str, Any]:
    """Speak text using text-to-speech synthesis.

    Synthesizes the provided text and plays it through the audio output.
    Blocks until playback is complete.

    Use this to give spoken coaching advice to the player during games.

    Args:
        text: The text to synthesize and speak.

    Returns:
        Dict with:
        - spoken: True if speech completed successfully
        - text: The text that was spoken

        or {"error": message} if TTS fails (e.g., missing model files).
    """
    try:
        voice_output = _get_voice_output()
        voice_output.speak(text, blocking=True)
        return {"spoken": True, "text": text}
    except FileNotFoundError as e:
        # Missing Kokoro model files - provide helpful error
        return {"error": f"TTS model not found: {e}", "spoken": False}
    except Exception as e:
        logger.exception("TTS error")
        return {"error": str(e), "spoken": False}


def queue_advice(advice: str, trigger: str) -> None:
    """Queue proactive coaching advice for later retrieval.

    Internal function called by background monitoring (Phase 9) when
    game state triggers fire. Advice is queued for retrieval by
    get_pending_advice().

    Args:
        advice: The coaching advice text.
        trigger: Description of what triggered this advice.
    """
    _pending_advice.append({
        "advice": advice,
        "trigger": trigger,
        "timestamp": time.time(),
    })
    logger.debug(f"Queued advice: {trigger}")


@mcp.tool()
def get_pending_advice() -> dict[str, Any]:
    """Get all pending proactive coaching advice.

    Returns and clears all advice that has been queued by the background
    game state monitor. Each advice item includes the trigger that caused it.

    Use this to poll for proactive coaching suggestions during gameplay.

    Returns:
        Dict with:
        - advice_items: List of advice dicts, each with:
          - advice: The coaching advice text
          - trigger: What triggered this advice
          - timestamp: When the advice was generated (Unix timestamp)
        - count: Number of advice items returned
    """
    items = []
    while _pending_advice:
        try:
            items.append(_pending_advice.popleft())
        except IndexError:
            break  # Queue emptied by another thread

    return {"advice_items": items, "count": len(items)}


@mcp.tool()
def clear_pending_advice() -> dict[str, Any]:
    """Clear all pending proactive coaching advice.

    Removes all queued advice without returning it. Use this to reset
    the advice queue, for example when starting a new game.

    Returns:
        Dict with:
        - cleared: True
        - count: Number of items that were cleared
    """
    count = len(_pending_advice)
    _pending_advice.clear()
    return {"cleared": True, "count": count}


# Entry point for running as module
if __name__ == "__main__":
    mcp.run()
