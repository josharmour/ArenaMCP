"""FastMCP server exposing MTGA game state and card information.

This module provides the MCP server that bridges live MTGA games to Claude,
implementing the Calculator + Coach pattern: deterministic code tracks state
while the LLM provides strategic analysis.
"""

import logging
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from arenamcp.gamestate import GameState, ZoneType, create_game_state_handler
from arenamcp.parser import LogParser
from arenamcp.scryfall import ScryfallCache
from arenamcp.draftstats import DraftStatsCache
from arenamcp.watcher import MTGALogWatcher

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


# Entry point for running as module
if __name__ == "__main__":
    mcp.run()
