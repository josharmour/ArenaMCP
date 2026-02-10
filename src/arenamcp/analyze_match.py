"""
Post-Match Advice Analyzer

Analyzes recorded matches to compare given advice against re-generated optimal advice.
Usage: python -m arenamcp.analyze_match <recording_file.json> [--regenerate]
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Coach import only needed if regenerating
# from arenamcp.coach import MTGCoach


def load_recording(path: Path) -> dict:
    """Load a match recording JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def load_advice_history(bug_report_path: Path) -> list:
    """Load advice history from a bug report."""
    with open(bug_report_path, 'r') as f:
        data = json.load(f)
    return data.get("advice_history", [])


def find_advice_for_frame(advice_history: list, frame_timestamp: str) -> Optional[dict]:
    """Find advice given closest to a frame's timestamp."""
    frame_time = datetime.fromisoformat(frame_timestamp.replace('Z', '+00:00'))
    
    closest = None
    min_delta = float('inf')
    
    for advice in advice_history:
        advice_time = datetime.fromisoformat(advice["timestamp"])
        delta = abs((frame_time - advice_time).total_seconds())
        if delta < min_delta and delta < 5.0:  # Within 5 seconds
            min_delta = delta
            closest = advice
    
    return closest


def format_state_summary(parsed_snapshot: dict) -> str:
    """Create a brief summary of the game state."""
    turn = parsed_snapshot.get("turn_info", {})
    players = parsed_snapshot.get("players", [])
    zones = parsed_snapshot.get("zones", {})
    
    turn_num = turn.get("turn_number", "?")
    phase = turn.get("phase", "").replace("Phase_", "")
    step = turn.get("step", "").replace("Step_", "")
    active = turn.get("active_player", 0)
    
    local_seat = parsed_snapshot.get("local_seat_id", 1)
    is_your_turn = active == local_seat
    
    # Life totals
    life_parts = []
    for p in players:
        seat = p.get("seat_id")
        life = p.get("life_total", 20)
        label = "You" if p.get("is_local") else "Opp"
        life_parts.append(f"{label}:{life}")
    
    # Board counts
    battlefield = zones.get("battlefield", [])
    your_cards = len([c for c in battlefield if c.get("controller_seat_id") == local_seat])
    opp_cards = len([c for c in battlefield if c.get("controller_seat_id") != local_seat])
    
    # Hand count
    hand = zones.get("my_hand", [])
    hand_count = len(hand)
    
    return (
        f"T{turn_num} {phase}{':' + step if step else ''} "
        f"({'YOUR' if is_your_turn else 'OPP'} turn) | "
        f"{' | '.join(life_parts)} | "
        f"Board: You={your_cards} Opp={opp_cards} | Hand={hand_count}"
    )


def analyze_frame(frame: dict, advice_history: list, coach: Optional[MTGCoach] = None) -> dict:
    """Analyze a single frame and optionally regenerate advice."""
    parsed = frame.get("parsed_snapshot", {})
    frame_num = frame.get("frame_number", 0)
    timestamp = frame.get("timestamp", "")
    
    # Find actual advice given near this frame
    actual_advice = find_advice_for_frame(advice_history, timestamp)
    
    result = {
        "frame": frame_num,
        "timestamp": timestamp,
        "state_summary": format_state_summary(parsed),
        "actual_advice": actual_advice.get("advice") if actual_advice else None,
        "actual_trigger": actual_advice.get("trigger") if actual_advice else None,
        "regenerated_advice": None,
    }
    
    # Regenerate advice if coach provided
    if coach and actual_advice:
        try:
            # Build game state dict from parsed snapshot
            game_state = {
                "turn": parsed.get("turn_info", {}),
                "players": parsed.get("players", []),
                "battlefield": parsed.get("zones", {}).get("battlefield", []),
                "hand": parsed.get("zones", {}).get("my_hand", []),
                "graveyard": parsed.get("zones", {}).get("graveyard", []),
                "stack": parsed.get("zones", {}).get("stack", []),
                "exile": parsed.get("zones", {}).get("exile", []),
            }
            result["regenerated_advice"] = coach.get_advice(
                game_state, 
                trigger=actual_advice.get("trigger", "analysis"),
                style="concise"
            )
        except Exception as e:
            result["regenerated_advice"] = f"[Error: {e}]"
    
    return result


def find_key_frames(recording: dict) -> list[int]:
    """Find frames where advice was likely triggered (turn/phase changes)."""
    frames = recording.get("frames", [])
    key_frames = []
    
    prev_turn = 0
    prev_phase = ""
    
    for i, frame in enumerate(frames):
        parsed = frame.get("parsed_snapshot", {})
        turn_info = parsed.get("turn_info", {})
        turn = turn_info.get("turn_number", 0)
        phase = turn_info.get("phase", "")
        
        # Key frame if turn or phase changed
        if turn != prev_turn or phase != prev_phase:
            if turn > 0:  # Skip pre-game
                key_frames.append(i)
        
        prev_turn = turn
        prev_phase = phase
    
    return key_frames


def generate_analysis_report(recording_path: Path, bug_report_path: Optional[Path] = None, 
                            regenerate: bool = False) -> str:
    """Generate a full analysis report for a match."""
    recording = load_recording(recording_path)
    
    # Try to find matching bug report if not provided
    if bug_report_path is None:
        bug_dir = Path.home() / ".arenamcp" / "bug_reports"
        if bug_dir.exists():
            # Find most recent bug report
            reports = sorted(bug_dir.glob("bug_*.json"), reverse=True)
            if reports:
                bug_report_path = reports[0]
    
    advice_history = []
    if bug_report_path and bug_report_path.exists():
        advice_history = load_advice_history(bug_report_path)
    
    # Initialize coach if regenerating
    coach = None
    if regenerate:
        from arenamcp.coach import create_coach
        coach = create_coach(backend="claude-code", model="sonnet")
    
    # Build report
    lines = [
        "=" * 60,
        "POST-MATCH ADVICE ANALYSIS",
        f"Recording: {recording_path.name}",
        f"Match ID: {recording.get('match_id', 'Unknown')}",
        f"Total Frames: {len(recording.get('frames', []))}",
        f"Advice History Entries: {len(advice_history)}",
        "=" * 60,
        "",
    ]
    
    # Find key frames
    key_frames = find_key_frames(recording)
    lines.append(f"Key Frames (turn/phase changes): {len(key_frames)}")
    lines.append("")
    
    # Analyze frames where advice was given
    advice_frames = []
    for advice in advice_history:
        timestamp = advice.get("timestamp", "")
        trigger = advice.get("trigger", "")
        advice_text = advice.get("advice", "")
        context = advice.get("game_context", "")
        
        lines.append("-" * 40)
        lines.append(f"[{trigger.upper()}] {timestamp}")
        lines.append(f"ADVICE: {advice_text}")
        
        # Extract key info from context
        if context:
            # Find turn/phase info
            for line in context.split('\n'):
                if any(x in line for x in ["Turn ", "Phase:", "LAND DROP:", "YOUR MANA:"]):
                    lines.append(f"  {line.strip()}")
        
        lines.append("")
    
    lines.append("=" * 60)
    lines.append("END OF ANALYSIS")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze match recording and advice")
    parser.add_argument("recording", type=Path, help="Path to match recording JSON")
    parser.add_argument("--bug-report", type=Path, help="Path to bug report with advice history")
    parser.add_argument("--regenerate", action="store_true", help="Re-run LLM for comparison")
    parser.add_argument("--output", type=Path, help="Save report to file")
    
    args = parser.parse_args()
    
    if not args.recording.exists():
        print(f"Error: Recording not found: {args.recording}")
        return 1
    
    report = generate_analysis_report(
        args.recording, 
        args.bug_report,
        args.regenerate
    )
    
    if args.output:
        args.output.write_text(report)
        print(f"Report saved to: {args.output}")
    else:
        print(report)
    
    return 0


if __name__ == "__main__":
    exit(main())
