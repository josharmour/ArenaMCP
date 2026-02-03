#!/usr/bin/env python3
"""Log Replay Tool for debugging MTGA game state tracking.

This tool allows you to:
1. Load a Player.log file (or segment saved during a bug report)
2. Step through game state messages one-by-one
3. See what the game state parser sees at each step
4. Debug issues like incorrect card names, missing abilities, etc.

Usage:
    python log_replay.py <log_file> [--step] [--from-turn N] [--to-turn N]
    
Examples:
    # Replay entire log file
    python log_replay.py Player.log
    
    # Step through interactively
    python log_replay.py Player.log --step
    
    # Replay specific turns 
    python log_replay.py Player.log --from-turn 5 --to-turn 8
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Generator

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arenamcp.gamestate import GameState, create_game_state_handler
from arenamcp.parser import LogParser


def extract_gre_events(log_content: str) -> Generator[dict, None, None]:
    """Extract all GreToClientEvent JSON blocks from log content.
    
    Yields:
        Parsed JSON dicts for each game state event.
    """
    # Pattern to match GreToClientEvent lines with JSON
    # The log format is: [Timestamp] [UnityCrossThreadLogger]<= GreToClientEvent {...}
    pattern = r'<= GreToClientEvent\s*(\{.+?\})\s*(?=\n\[|\Z)'
    
    # Multiline search - JSON can span multiple lines
    # Alternative approach: look for lines containing greToClientEvent
    
    lines = log_content.split('\n')
    buffer = []
    in_json = False
    brace_count = 0
    
    for line in lines:
        if '<= GreToClientEvent' in line or 'greToClientEvent' in line.lower():
            # Start of a new event
            if buffer:
                # Try to parse previous buffer
                try:
                    json_str = ''.join(buffer)
                    parsed = json.loads(json_str)
                    yield parsed
                except json.JSONDecodeError:
                    pass
            
            # Find the JSON start
            json_start = line.find('{')
            if json_start >= 0:
                buffer = [line[json_start:]]
                brace_count = line[json_start:].count('{') - line[json_start:].count('}')
                in_json = brace_count > 0
        elif in_json:
            buffer.append(line)
            brace_count += line.count('{') - line.count('}')
            if brace_count <= 0:
                # End of JSON block
                try:
                    json_str = ''.join(buffer)
                    parsed = json.loads(json_str)
                    yield parsed
                except json.JSONDecodeError:
                    pass
                buffer = []
                in_json = False
                brace_count = 0


def format_game_state_summary(game_state: GameState) -> str:
    """Create a readable summary of current game state."""
    lines = []
    
    # Turn info
    turn_info = game_state.turn_info
    turn_num = turn_info.get("turn_number", "?")
    phase = turn_info.get("phase", "?")
    step = turn_info.get("step", "")
    active = turn_info.get("active_player", "?")
    priority = turn_info.get("priority_player", "?")
    
    lines.append(f"=== Turn {turn_num} | Phase: {phase} | Step: {step} ===")
    lines.append(f"Active Player: {active} | Priority: {priority} | Local: {game_state.local_seat_id}")
    
    # Players
    for seat_id, player in game_state.players.items():
        marker = " (YOU)" if seat_id == game_state.local_seat_id else " (OPP)"
        lines.append(f"  Seat {seat_id}{marker}: {player.life_total} life")
    
    # Battlefield summary
    battlefield_objs = []
    for zone in game_state.zones.values():
        if zone.zone_type.name == "BATTLEFIELD":
            for inst_id in zone.object_ids:
                if inst_id in game_state.game_objects:
                    obj = game_state.game_objects[inst_id]
                    battlefield_objs.append(obj)
    
    if battlefield_objs:
        lines.append(f"\nBattlefield ({len(battlefield_objs)} permanents):")
        for obj in battlefield_objs[:10]:  # Limit display
            owner = "You" if obj.owner_seat_id == game_state.local_seat_id else "Opp"
            tapped = " [T]" if obj.is_tapped else ""
            pt = f" {obj.power}/{obj.toughness}" if obj.power is not None else ""
            lines.append(f"  [{owner}] grp={obj.grp_id}{pt}{tapped} (inst={obj.instance_id})")
        if len(battlefield_objs) > 10:
            lines.append(f"  ... and {len(battlefield_objs) - 10} more")
    
    # Hand summary (local player only)
    hand_objs = []
    for zone in game_state.zones.values():
        if zone.zone_type.name == "HAND" and zone.owner_seat_id == game_state.local_seat_id:
            for inst_id in zone.object_ids:
                if inst_id in game_state.game_objects:
                    hand_objs.append(game_state.game_objects[inst_id])
    
    if hand_objs:
        lines.append(f"\nHand ({len(hand_objs)} cards):")
        for obj in hand_objs:
            lines.append(f"  grp={obj.grp_id} (inst={obj.instance_id})")
    
    # Pending decision
    if game_state.pending_decision:
        lines.append(f"\n⚠️ PENDING DECISION: {game_state.pending_decision}")
    
    return '\n'.join(lines)


def replay_log(log_path: Path, step_mode: bool = False, from_turn: int = 0, to_turn: int = 999):
    """Replay a log file through the game state parser.
    
    Args:
        log_path: Path to Player.log or saved log segment
        step_mode: If True, pause after each event
        from_turn: Start displaying from this turn
        to_turn: Stop after this turn
    """
    print(f"Loading log file: {log_path}")
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        log_content = f.read()
    
    print(f"Log size: {len(log_content):,} bytes")
    
    # Initialize game state
    game_state = GameState()
    handler = create_game_state_handler(game_state)
    
    event_count = 0
    displayed_count = 0
    current_turn = 0
    
    for event in extract_gre_events(log_content):
        event_count += 1
        
        # Process through handler
        handler(event)
        
        # Get current turn
        new_turn = game_state.turn_info.get("turn_number", 0)
        if new_turn != current_turn:
            current_turn = new_turn
            print(f"\n{'='*60}")
            print(f"TURN {current_turn}")
            print(f"{'='*60}")
        
        # Filter by turn range
        if current_turn < from_turn:
            continue
        if current_turn > to_turn:
            print(f"\n[Stopping at turn {to_turn}]")
            break
        
        displayed_count += 1
        
        # Display state summary
        print(f"\n--- Event {event_count} ---")
        print(format_game_state_summary(game_state))
        
        if step_mode:
            user_input = input("\n[Enter to continue, 'q' to quit, 's' for snapshot] ").strip().lower()
            if user_input == 'q':
                break
            elif user_input == 's':
                # Show full snapshot
                snapshot = game_state.get_snapshot()
                print(json.dumps(snapshot, indent=2, default=str))
    
    print(f"\n{'='*60}")
    print(f"Replay complete: {event_count} events processed, {displayed_count} displayed")
    print(f"Final state: Turn {current_turn}")


def main():
    parser = argparse.ArgumentParser(description="Replay MTGA log files for debugging")
    parser.add_argument("log_file", type=Path, help="Path to Player.log or log segment")
    parser.add_argument("--step", action="store_true", help="Step through events interactively")
    parser.add_argument("--from-turn", type=int, default=0, help="Start from this turn")
    parser.add_argument("--to-turn", type=int, default=999, help="Stop after this turn")
    
    args = parser.parse_args()
    
    if not args.log_file.exists():
        print(f"Error: File not found: {args.log_file}")
        sys.exit(1)
    
    replay_log(args.log_file, args.step, args.from_turn, args.to_turn)


if __name__ == "__main__":
    main()
