#!/usr/bin/env python3
"""Replay a match recording through the advisor to see what it would have said.

Takes an advisor recording (match_manual_*.json) and replays each game state
through the current advisor, showing what advice would be given.

This helps identify:
1. What the advisor would have said at each decision point
2. Whether the advice has improved since the recording
3. Patterns where the advisor gives suboptimal advice

Usage:
    python replay_through_advisor.py <recording.json> [--backend gemini|claude]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arenamcp.mtgadb import MTGADatabase


def load_recording(path: Path) -> dict:
    """Load a match recording."""
    with open(path) as f:
        return json.load(f)


def get_turn_frames(recording: dict) -> dict:
    """Group frames by turn number, return first frame of each main phase."""
    frames = recording.get('frames', [])
    turns = {}

    for frame in frames:
        snap = frame.get('parsed_snapshot', {})
        turn_info = snap.get('turn_info', {})
        turn = turn_info.get('turn_number', 0)
        phase = turn_info.get('phase', '')
        active = turn_info.get('active_player', 0)

        if turn == 0:
            continue

        # Get first frame of each turn's main phase
        key = (turn, phase)
        if key not in turns and 'Main1' in phase:
            turns[key] = frame

    return turns


def get_advisor_suggestion(game_state: dict, trigger: str, backend: str) -> str:
    """Get advisor suggestion for a game state."""
    try:
        from arenamcp.coach import CoachEngine, create_backend

        backend_obj = create_backend(backend)
        coach = CoachEngine(backend=backend_obj)

        advice = coach.get_advice(game_state, trigger=trigger, style="concise")
        return advice
    except Exception as e:
        return f"[Error: {e}]"


def format_board_state(snap: dict, card_db: Optional[MTGADatabase] = None) -> str:
    """Format a readable board state summary."""
    lines = []

    turn_info = snap.get('turn_info', {})
    players = snap.get('players', [])
    zones = snap.get('zones', {})
    local_seat = snap.get('local_seat_id', 1)

    lines.append(f"Turn {turn_info.get('turn_number', '?')} - {turn_info.get('phase', '?')}")
    active = turn_info.get('active_player', 0)
    lines.append(f"{'YOUR TURN' if active == local_seat else 'OPPONENT TURN'}")

    # Life totals
    for p in players:
        seat = p.get('seat_id')
        life = p.get('life_total', 20)
        label = 'You' if p.get('is_local') else 'Opp'
        lines.append(f"  {label}: {life} life")

    # Battlefield
    battlefield = zones.get('battlefield', [])
    your_cards = [c for c in battlefield if c.get('controller_seat_id') == local_seat]
    opp_cards = [c for c in battlefield if c.get('controller_seat_id') != local_seat]

    if your_cards:
        lines.append(f"\nYour board ({len(your_cards)} permanents):")
        for card in your_cards[:5]:
            name = "Unknown"
            if card_db and card.get('grp_id'):
                try:
                    c = card_db.get_card(card['grp_id'])
                    if c:
                        name = c.name
                except:
                    pass
            pt = f" {card.get('power', '?')}/{card.get('toughness', '?')}" if card.get('power') is not None else ""
            tapped = " (tapped)" if card.get('is_tapped') else ""
            lines.append(f"  - {name}{pt}{tapped}")
        if len(your_cards) > 5:
            lines.append(f"  ... and {len(your_cards) - 5} more")

    # Hand
    hand = zones.get('my_hand', [])
    if hand:
        lines.append(f"\nYour hand ({len(hand)} cards):")
        for card in hand[:5]:
            name = "Unknown"
            if card_db and card.get('grp_id'):
                try:
                    c = card_db.get_card(card['grp_id'])
                    if c:
                        name = c.name
                except:
                    pass
            lines.append(f"  - {name}")
        if len(hand) > 5:
            lines.append(f"  ... and {len(hand) - 5} more")

    return '\n'.join(lines)


def replay_recording(recording_path: Path, backend: str = "gemini",
                     skip_advisor: bool = False) -> list[dict]:
    """Replay a recording through the advisor."""

    print(f"Loading recording: {recording_path.name}")
    recording = load_recording(recording_path)

    print(f"Match ID: {recording.get('match_id')}")
    print(f"Total frames: {len(recording.get('frames', []))}")

    # Initialize card database
    card_db = None
    try:
        card_db = MTGADatabase()
    except:
        pass

    # Get key frames (first main phase of each turn)
    turn_frames = get_turn_frames(recording)
    print(f"Found {len(turn_frames)} main phase decision points")

    results = []

    for (turn, phase), frame in sorted(turn_frames.items()):
        snap = frame.get('parsed_snapshot', {})
        turn_info = snap.get('turn_info', {})
        local_seat = snap.get('local_seat_id', 1)
        active = turn_info.get('active_player', 0)

        # Only analyze turns where it's our priority
        if active != local_seat:
            continue

        print(f"\n{'='*60}")
        print(f"Turn {turn} - {phase}")
        print('='*60)

        # Show board state
        board = format_board_state(snap, card_db)
        print(board)

        # Get advisor suggestion
        advice = None
        if not skip_advisor:
            print("\nGetting advisor suggestion...")
            advice = get_advisor_suggestion(snap, "priority", backend)
            print(f"\nAdvisor says: {advice}")

        results.append({
            "turn": turn,
            "phase": phase,
            "frame": frame.get('frame_number'),
            "board_summary": board,
            "advisor_advice": advice,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Replay recording through advisor")
    parser.add_argument("recording", type=Path, help="Path to match recording JSON")
    parser.add_argument("--backend", default="gemini", help="LLM backend")
    parser.add_argument("--skip-advisor", action="store_true", help="Skip advisor calls")
    parser.add_argument("--output", "-o", type=Path, help="Save results to JSON")

    args = parser.parse_args()

    if not args.recording.exists():
        print(f"Error: Recording not found: {args.recording}")
        return 1

    results = replay_recording(args.recording, args.backend, args.skip_advisor)

    print(f"\n{'='*60}")
    print(f"REPLAY COMPLETE")
    print(f"Analyzed {len(results)} decision points")
    print('='*60)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
