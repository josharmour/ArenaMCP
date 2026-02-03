#!/usr/bin/env python3
"""Post-Match Validator - Compare our parser output against Arena's raw data.

This tool replays a Player.log file twice:
1. Once through our GameState parser (what we interpret)
2. Once extracting raw Arena data (ground truth)

Then it compares key fields to find discrepancies in our parsing.

Usage:
    python validate_match.py <Player.log> [--match-id ID] [--verbose]

Example:
    python validate_match.py "%LOCALAPPDATA%Low\\Wizards Of The Coast\\MTGA\\Player.log"
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Generator, Tuple
from dataclasses import dataclass

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arenamcp.gamestate import GameState, create_game_state_handler


@dataclass
class Discrepancy:
    """A difference between Arena's data and our parsing."""
    turn: int
    phase: str
    field: str
    arena_value: str
    parsed_value: str
    severity: str = "warning"  # info, warning, error


def extract_events(log_content: str) -> Generator[dict, None, None]:
    """Extract GreToClientEvent JSON blocks from log content."""
    lines = log_content.split('\n')
    buffer = []
    in_json = False
    brace_count = 0
    
    for line in lines:
        # Look for GRE event markers
        if 'GreToClientEvent' in line or 'greToClientEvent' in line:
            if buffer:
                try:
                    yield json.loads(''.join(buffer))
                except json.JSONDecodeError:
                    pass
            
            json_start = line.find('{')
            if json_start >= 0:
                buffer = [line[json_start:]]
                brace_count = buffer[0].count('{') - buffer[0].count('}')
                in_json = brace_count > 0
        elif in_json:
            buffer.append(line)
            brace_count += line.count('{') - line.count('}')
            if brace_count <= 0:
                try:
                    yield json.loads(''.join(buffer))
                except json.JSONDecodeError:
                    pass
                buffer = []
                in_json = False


def extract_arena_state(event: dict) -> dict:
    """Extract key state fields directly from Arena's raw message."""
    result = {
        "turn": 0,
        "phase": "",
        "step": "",
        "active_player": 0,
        "life_totals": {},
        "battlefield_objects": [],
        "hand_objects": [],
    }
    
    messages = event.get("greToClientEvent", {}).get("greToClientMessages", [])
    for msg in messages:
        gsm = msg.get("gameStateMessage", {})
        if not gsm:
            continue
            
        # Turn info
        turn_info = gsm.get("turnInfo", {})
        if turn_info.get("turnNumber"):
            result["turn"] = turn_info.get("turnNumber", 0)
        if turn_info.get("phase"):
            result["phase"] = turn_info.get("phase", "")
        if turn_info.get("step"):
            result["step"] = turn_info.get("step", "")
        if turn_info.get("activePlayer"):
            result["active_player"] = turn_info.get("activePlayer", 0)
        
        # Players/life
        for player in gsm.get("players", []):
            seat = player.get("systemSeatNumber") or player.get("controllerSeatId")
            life = player.get("lifeTotal")
            if seat and life is not None:
                result["life_totals"][seat] = life
        
        # Zones
        for zone in gsm.get("zones", []):
            zone_type = zone.get("type", "")
            obj_ids = zone.get("objectInstanceIds", [])
            
            if zone_type == "ZoneType_Battlefield":
                result["battlefield_objects"].extend(obj_ids)
            elif zone_type == "ZoneType_Hand":
                result["hand_objects"].extend(obj_ids)
    
    return result


def compare_states(arena: dict, parsed: dict, turn: int, phase: str) -> list[Discrepancy]:
    """Compare Arena state vs our parsed state."""
    discrepancies = []
    
    # Compare turn number
    parsed_turn = parsed.get("turn", {}).get("turn_number", 0)
    if arena["turn"] and parsed_turn != arena["turn"]:
        discrepancies.append(Discrepancy(
            turn=turn, phase=phase,
            field="turn_number",
            arena_value=str(arena["turn"]),
            parsed_value=str(parsed_turn),
            severity="error"
        ))
    
    # Compare phase
    parsed_phase = parsed.get("turn", {}).get("phase", "")
    if arena["phase"] and parsed_phase != arena["phase"]:
        discrepancies.append(Discrepancy(
            turn=turn, phase=phase,
            field="phase",
            arena_value=arena["phase"],
            parsed_value=parsed_phase,
            severity="warning"
        ))
    
    # Compare life totals
    for seat, arena_life in arena["life_totals"].items():
        parsed_life = None
        for p in parsed.get("players", []):
            if p.get("seat_id") == seat:
                parsed_life = p.get("life_total")
                break
        
        if parsed_life is not None and parsed_life != arena_life:
            discrepancies.append(Discrepancy(
                turn=turn, phase=phase,
                field=f"life_seat_{seat}",
                arena_value=str(arena_life),
                parsed_value=str(parsed_life),
                severity="error"
            ))
    
    # Compare battlefield count
    parsed_bf = parsed.get("battlefield", [])
    if arena["battlefield_objects"]:
        if len(parsed_bf) != len(set(arena["battlefield_objects"])):
            discrepancies.append(Discrepancy(
                turn=turn, phase=phase,
                field="battlefield_count",
                arena_value=str(len(set(arena["battlefield_objects"]))),
                parsed_value=str(len(parsed_bf)),
                severity="warning"
            ))
    
    return discrepancies


def validate_log(log_path: Path, verbose: bool = False) -> Tuple[int, list[Discrepancy]]:
    """Validate our parsing against a Player.log file.
    
    Returns:
        Tuple of (event_count, list of discrepancies)
    """
    print(f"Loading: {log_path}")
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        log_content = f.read()
    
    print(f"Log size: {len(log_content):,} bytes")
    
    # Initialize our parser
    game_state = GameState()
    handler = create_game_state_handler(game_state)
    
    all_discrepancies = []
    event_count = 0
    current_turn = 0
    current_phase = ""
    
    print("\nProcessing events...")
    
    for event in extract_events(log_content):
        event_count += 1
        
        # Extract Arena's raw state BEFORE we process
        arena_state = extract_arena_state(event)
        
        # Update turn tracking from Arena
        if arena_state["turn"]:
            current_turn = arena_state["turn"]
        if arena_state["phase"]:
            current_phase = arena_state["phase"]
        
        # Process through our handler
        handler(event)
        
        # Get our parsed snapshot
        try:
            parsed = game_state.get_snapshot()
        except Exception as e:
            print(f"  [ERROR] Snapshot failed at event {event_count}: {e}")
            continue
        
        # Compare
        discrepancies = compare_states(arena_state, parsed, current_turn, current_phase)
        all_discrepancies.extend(discrepancies)
        
        if verbose and discrepancies:
            for d in discrepancies:
                print(f"  [T{d.turn}] {d.field}: Arena={d.arena_value}, Parsed={d.parsed_value}")
    
    return event_count, all_discrepancies


def main():
    parser = argparse.ArgumentParser(description="Validate game state parsing against Arena logs")
    parser.add_argument("log_file", type=Path, help="Path to Player.log")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show each discrepancy as found")
    
    args = parser.parse_args()
    
    if not args.log_file.exists():
        # Try expanding environment variables
        expanded = Path(str(args.log_file).replace("%LOCALAPPDATA%", 
                        str(Path.home() / "AppData" / "Local")))
        if expanded.exists():
            args.log_file = expanded
        else:
            print(f"Error: File not found: {args.log_file}")
            sys.exit(1)
    
    event_count, discrepancies = validate_log(args.log_file, args.verbose)
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Events processed: {event_count}")
    print(f"Total discrepancies: {len(discrepancies)}")
    
    if discrepancies:
        errors = [d for d in discrepancies if d.severity == "error"]
        warnings = [d for d in discrepancies if d.severity == "warning"]
        
        print(f"  - Errors: {len(errors)}")
        print(f"  - Warnings: {len(warnings)}")
        
        # Group by field
        by_field = {}
        for d in discrepancies:
            by_field[d.field] = by_field.get(d.field, 0) + 1
        
        print("\nBy field:")
        for field, count in sorted(by_field.items(), key=lambda x: -x[1]):
            print(f"  {field}: {count}")
        
        # Show sample errors
        if errors:
            print("\nSample errors:")
            for d in errors[:5]:
                print(f"  Turn {d.turn}: {d.field} - Arena={d.arena_value}, Parsed={d.parsed_value}")
    else:
        print("\n✅ No discrepancies found! Parsing matches Arena data.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
