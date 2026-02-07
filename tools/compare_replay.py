#!/usr/bin/env python3
"""Compare Arena replay files against our parser's recorded game states.

This tool identifies discrepancies between:
- Arena's ground truth (.rply files)
- Our parser's understanding (match recordings)

Checks: life totals, battlefield objects, zone contents, turn/phase tracking.

Usage:
    python compare_replay.py <replay.rply> <recording.json>
    python compare_replay.py <replay.rply>  # Auto-finds matching recording
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class Discrepancy:
    """A difference between replay and recording."""
    turn: int
    phase: str
    category: str  # life, battlefield, hand, graveyard, exile, turn_info
    field: str
    replay_value: str
    recording_value: str
    severity: str = "warning"  # info, warning, error

    def __str__(self):
        return f"[T{self.turn}] {self.category}.{self.field}: replay={self.replay_value} vs recording={self.recording_value}"


@dataclass
class ComparisonResult:
    """Result of comparing replay to recording."""
    replay_file: str
    recording_file: str
    total_replay_states: int = 0
    total_recording_states: int = 0
    discrepancies: list = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(d.severity == "error" for d in self.discrepancies)

    @property
    def error_count(self) -> int:
        return sum(1 for d in self.discrepancies if d.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for d in self.discrepancies if d.severity == "warning")


def parse_replay_v2(replay_path: Path) -> tuple[dict, list[dict]]:
    """Parse Version2 format Arena replay file.

    Returns:
        Tuple of (metadata dict, list of server messages)
    """
    with open(replay_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if not lines or not lines[0].strip().startswith("#Version"):
        raise ValueError(f"Not a valid Arena replay file: {replay_path}")

    # Line 2 is metadata
    metadata = json.loads(lines[1].strip()) if len(lines) > 1 else {}

    # Parse IN messages (from server)
    messages = []
    for line in lines[2:]:
        line = line.strip()
        if line.startswith("IN-"):
            colon_idx = line.find(':')
            if colon_idx > 0:
                try:
                    msg = json.loads(line[colon_idx+1:])
                    messages.append(msg)
                except json.JSONDecodeError:
                    pass

    return metadata, messages


def extract_replay_states(messages: list[dict]) -> list[dict]:
    """Extract game state snapshots from replay messages.

    Accumulates state across messages (Arena sends incremental updates).
    Implements blanket untap at turn boundaries (MTGA never sends isTapped=false;
    untapping is implicit). Uses untap prevention tracking for creatures that
    MTGA explicitly keeps tapped during the untap step (e.g., Blossombind effects).

    Returns list of state dicts with: turn, phase, step, active_player,
    life_totals, battlefield_ids, hand_ids, graveyard_ids, exile_ids,
    tapped_instance_ids
    """
    states = []

    # Accumulated state tracking (Arena sends incremental updates)
    cum_life_totals = {}
    cum_objects = {}  # instance_id -> {grp_id, owner, zone_id, is_tapped, card_types}
    cum_zones = {}    # zone_id -> {type, owner, object_ids}
    last_turn = 0
    untap_prevention = set()  # instance_ids MTGA kept tapped during untap step
    in_untap_step = False

    for msg in messages:
        if 'gameStateMessage' not in msg:
            continue

        gsm = msg['gameStateMessage']
        ti = gsm.get('turnInfo', {})
        turn = ti.get('turnNumber', 0)

        if turn == 0:
            continue

        # BLANKET UNTAP: When turn changes, untap all permanents for the new active player.
        # MTGA never sends isTapped=false; untapping is implicit at the start of each turn.
        # Skip permanents in untap_prevention (those MTGA explicitly kept tapped last turn).
        new_active = ti.get('activePlayer', 0)
        if turn != last_turn and new_active != 0:
            in_untap_step = True
            # Find battlefield zone ids
            bf_inst_ids = set()
            for z_info in cum_zones.values():
                if z_info['type'] == 'ZoneType_Battlefield':
                    bf_inst_ids.update(z_info['object_ids'])
            # Clean up prevention set
            untap_prevention &= bf_inst_ids
            # Untap permanents controlled by new active player
            for inst_id, obj in cum_objects.items():
                if obj['owner'] == new_active and obj.get('is_tapped') and inst_id in bf_inst_ids:
                    if inst_id not in untap_prevention:
                        obj['is_tapped'] = False
            last_turn = turn

        # Update accumulated life totals
        for p in gsm.get('players', []):
            seat = p.get('systemSeatNumber') or p.get('controllerSeatId', 0)
            life = p.get('lifeTotal')
            if seat and life is not None:
                cum_life_totals[seat] = life

        # Update accumulated objects
        for obj in gsm.get('gameObjects', []):
            inst_id = obj.get('instanceId')
            if inst_id:
                existing = cum_objects.get(inst_id, {})
                grp_id = obj.get('grpId', existing.get('grp_id', 0))
                owner = obj.get('ownerSeatId', existing.get('owner', 0))
                zone_id = obj.get('zoneId', existing.get('zone_id', 0))
                is_tapped = existing.get('is_tapped', False)
                card_types = existing.get('card_types', [])

                if 'isTapped' in obj:
                    is_tapped = obj['isTapped']
                    # Track untap prevention during untap step diffs
                    if in_untap_step:
                        if is_tapped:
                            untap_prevention.add(inst_id)
                        else:
                            untap_prevention.discard(inst_id)

                if 'cardTypes' in obj:
                    card_types = list(obj['cardTypes'])

                cum_objects[inst_id] = {
                    'grp_id': grp_id, 'owner': owner, 'zone_id': zone_id,
                    'is_tapped': is_tapped, 'card_types': card_types,
                }

        # Clear untap step flag after processing objects in this message
        in_untap_step = False

        # Update accumulated zones
        for zone in gsm.get('zones', []):
            zone_id = zone.get('zoneId')
            if zone_id:
                zone_type = zone.get('type', '')
                owner = zone.get('ownerSeatId', 0)
                obj_ids = zone.get('objectInstanceIds', [])
                cum_zones[zone_id] = {'type': zone_type, 'owner': owner, 'object_ids': set(obj_ids)}

        # Build state snapshot from accumulated data
        state = {
            'turn': turn,
            'phase': ti.get('phase', ''),
            'step': ti.get('step', ''),
            'active_player': ti.get('activePlayer', 0),
            'priority_player': ti.get('priorityPlayer', 0),
            'life_totals': dict(cum_life_totals),
            'battlefield_grp_ids': {},
            'battlefield_inst_ids': set(),
            'tapped_instance_ids': set(),
            'hand_grp_ids': {},
            'hand_inst_ids': set(),
            'graveyard_grp_ids': {},
            'graveyard_inst_ids': set(),
            'exile_inst_ids': set(),
        }

        # Build zone contents from accumulated data
        for zone_id, zone_info in cum_zones.items():
            zone_type = zone_info['type']
            owner = zone_info['owner']
            obj_ids = zone_info['object_ids']

            if zone_type == 'ZoneType_Battlefield':
                state['battlefield_inst_ids'].update(obj_ids)
                for inst_id in obj_ids:
                    if inst_id in cum_objects:
                        obj = cum_objects[inst_id]
                        ctrl_seat = obj['owner']  # Use owner as controller for now
                        if ctrl_seat not in state['battlefield_grp_ids']:
                            state['battlefield_grp_ids'][ctrl_seat] = set()
                        if obj['grp_id']:
                            state['battlefield_grp_ids'][ctrl_seat].add(obj['grp_id'])
                        if obj.get('is_tapped'):
                            state['tapped_instance_ids'].add(inst_id)

            elif zone_type == 'ZoneType_Hand':
                state['hand_inst_ids'].update(obj_ids)
                if owner not in state['hand_grp_ids']:
                    state['hand_grp_ids'][owner] = set()
                for inst_id in obj_ids:
                    if inst_id in cum_objects:
                        obj = cum_objects[inst_id]
                        if obj['grp_id']:
                            state['hand_grp_ids'][owner].add(obj['grp_id'])

            elif zone_type == 'ZoneType_Graveyard':
                state['graveyard_inst_ids'].update(obj_ids)
                if owner not in state['graveyard_grp_ids']:
                    state['graveyard_grp_ids'][owner] = set()
                for inst_id in obj_ids:
                    if inst_id in cum_objects:
                        obj = cum_objects[inst_id]
                        if obj['grp_id']:
                            state['graveyard_grp_ids'][owner].add(obj['grp_id'])

            elif zone_type == 'ZoneType_Exile':
                state['exile_inst_ids'].update(obj_ids)

        states.append(state)

    return states


def extract_recording_states(recording: dict) -> list[dict]:
    """Extract game state snapshots from our recording format."""
    states = []

    for frame in recording.get('frames', []):
        parsed = frame.get('parsed_snapshot') or frame.get('our_game_state', {})
        ti = parsed.get('turn_info', {})
        turn = ti.get('turn_number', 0)

        if turn == 0:
            continue

        state = {
            'turn': turn,
            'phase': ti.get('phase', ''),
            'step': ti.get('step', ''),
            'active_player': ti.get('active_player', 0),
            'priority_player': ti.get('priority_player', 0),
            'life_totals': {},
            'battlefield_grp_ids': {},
            'battlefield_count': 0,
            'hand_grp_ids': {},
            'hand_count': 0,
            'graveyard_grp_ids': {},
            'graveyard_count': 0,
            'frame_number': frame.get('frame_number', 0),
        }

        # Extract life totals
        for p in parsed.get('players', []):
            seat = p.get('seat_id', 0)
            life = p.get('life_total')
            if seat and life is not None:
                state['life_totals'][seat] = life

        # Extract zone contents
        zones = parsed.get('zones', {})

        # Battlefield
        battlefield = zones.get('battlefield', [])
        state['battlefield_count'] = len(battlefield)
        for card in battlefield:
            owner = card.get('controller_seat_id') or card.get('owner_seat_id', 0)
            grp_id = card.get('grp_id', 0)
            if owner not in state['battlefield_grp_ids']:
                state['battlefield_grp_ids'][owner] = set()
            if grp_id:
                state['battlefield_grp_ids'][owner].add(grp_id)

        # Hand
        hand = zones.get('my_hand', [])
        state['hand_count'] = len(hand)
        for card in hand:
            grp_id = card.get('grp_id', 0)
            # Hand is always local player's
            local_seat = parsed.get('local_seat_id', 1)
            if local_seat not in state['hand_grp_ids']:
                state['hand_grp_ids'][local_seat] = set()
            if grp_id:
                state['hand_grp_ids'][local_seat].add(grp_id)

        # Graveyard
        graveyard = zones.get('graveyard', [])
        state['graveyard_count'] = len(graveyard)
        for card in graveyard:
            owner = card.get('owner_seat_id', 0)
            grp_id = card.get('grp_id', 0)
            if owner not in state['graveyard_grp_ids']:
                state['graveyard_grp_ids'][owner] = set()
            if grp_id:
                state['graveyard_grp_ids'][owner].add(grp_id)

        states.append(state)

    return states


def compare_states_at_turn(replay_states: list[dict], recording_states: list[dict],
                           turn: int) -> list[Discrepancy]:
    """Compare replay and recording states at a specific turn boundary."""
    discrepancies = []

    # Get first state for this turn from each source
    replay_state = None
    for s in replay_states:
        if s['turn'] == turn and s.get('life_totals'):
            replay_state = s
            break

    rec_state = None
    for s in recording_states:
        if s['turn'] == turn and s.get('life_totals'):
            rec_state = s
            break

    if not replay_state or not rec_state:
        return discrepancies

    phase = replay_state.get('phase', '')

    # Compare life totals
    for seat, replay_life in replay_state['life_totals'].items():
        rec_life = rec_state['life_totals'].get(seat)
        if rec_life is not None and rec_life != replay_life:
            discrepancies.append(Discrepancy(
                turn=turn,
                phase=phase,
                category="life",
                field=f"seat_{seat}",
                replay_value=str(replay_life),
                recording_value=str(rec_life),
                severity="error"
            ))

    # Compare battlefield by grp_ids per seat
    for seat in set(replay_state['battlefield_grp_ids'].keys()) | set(rec_state['battlefield_grp_ids'].keys()):
        replay_bf = replay_state['battlefield_grp_ids'].get(seat, set())
        rec_bf = rec_state['battlefield_grp_ids'].get(seat, set())

        if replay_bf != rec_bf:
            missing_in_rec = replay_bf - rec_bf
            extra_in_rec = rec_bf - replay_bf

            if missing_in_rec:
                discrepancies.append(Discrepancy(
                    turn=turn,
                    phase=phase,
                    category="battlefield",
                    field=f"seat_{seat}_missing",
                    replay_value=f"grp_ids: {sorted(missing_in_rec)}",
                    recording_value="not tracked",
                    severity="warning"
                ))

            if extra_in_rec:
                discrepancies.append(Discrepancy(
                    turn=turn,
                    phase=phase,
                    category="battlefield",
                    field=f"seat_{seat}_extra",
                    replay_value="not present",
                    recording_value=f"grp_ids: {sorted(extra_in_rec)}",
                    severity="warning"
                ))

    # Compare hand (local player only usually visible)
    for seat in set(replay_state['hand_grp_ids'].keys()) | set(rec_state['hand_grp_ids'].keys()):
        replay_hand = replay_state['hand_grp_ids'].get(seat, set())
        rec_hand = rec_state['hand_grp_ids'].get(seat, set())

        if replay_hand and rec_hand and replay_hand != rec_hand:
            missing = replay_hand - rec_hand
            extra = rec_hand - replay_hand

            if missing:
                discrepancies.append(Discrepancy(
                    turn=turn,
                    phase=phase,
                    category="hand",
                    field=f"seat_{seat}_missing",
                    replay_value=f"grp_ids: {sorted(missing)}",
                    recording_value="not tracked",
                    severity="warning"
                ))

    # Compare graveyard
    for seat in set(replay_state['graveyard_grp_ids'].keys()) | set(rec_state['graveyard_grp_ids'].keys()):
        replay_gy = replay_state['graveyard_grp_ids'].get(seat, set())
        rec_gy = rec_state['graveyard_grp_ids'].get(seat, set())

        if replay_gy and rec_gy and replay_gy != rec_gy:
            missing = replay_gy - rec_gy
            if missing:
                discrepancies.append(Discrepancy(
                    turn=turn,
                    phase=phase,
                    category="graveyard",
                    field=f"seat_{seat}_missing",
                    replay_value=f"grp_ids: {sorted(missing)}",
                    recording_value="not tracked",
                    severity="warning"
                ))

    return discrepancies


def compare_replay_to_recording(replay_path: Path, recording_path: Path) -> ComparisonResult:
    """Full comparison of replay against recording."""
    result = ComparisonResult(
        replay_file=str(replay_path),
        recording_file=str(recording_path)
    )

    # Parse replay
    metadata, messages = parse_replay_v2(replay_path)
    replay_states = extract_replay_states(messages)
    result.total_replay_states = len(replay_states)

    # Parse recording
    with open(recording_path) as f:
        recording = json.load(f)
    recording_states = extract_recording_states(recording)
    result.total_recording_states = len(recording_states)

    # Find all turns
    replay_turns = set(s['turn'] for s in replay_states if s['turn'] > 0)
    rec_turns = set(s['turn'] for s in recording_states if s['turn'] > 0)
    all_turns = sorted(replay_turns | rec_turns)

    # Compare at each turn boundary
    for turn in all_turns:
        discrepancies = compare_states_at_turn(replay_states, recording_states, turn)
        result.discrepancies.extend(discrepancies)

    return result


def format_report(result: ComparisonResult) -> str:
    """Format comparison result as readable report."""
    lines = [
        "=" * 70,
        "REPLAY VS RECORDING COMPARISON REPORT",
        "=" * 70,
        f"Replay: {Path(result.replay_file).name}",
        f"Recording: {Path(result.recording_file).name}",
        f"",
        f"Replay game states: {result.total_replay_states}",
        f"Recording game states: {result.total_recording_states}",
        f"",
        f"Total discrepancies: {len(result.discrepancies)}",
        f"  Errors: {result.error_count}",
        f"  Warnings: {result.warning_count}",
        "=" * 70,
    ]

    if result.discrepancies:
        lines.append("")
        lines.append("DISCREPANCIES BY CATEGORY:")
        lines.append("-" * 40)

        # Group by category
        by_category = {}
        for d in result.discrepancies:
            if d.category not in by_category:
                by_category[d.category] = []
            by_category[d.category].append(d)

        for category, discs in sorted(by_category.items()):
            lines.append(f"\n{category.upper()} ({len(discs)} issues):")
            for d in discs:
                severity_marker = "ERROR" if d.severity == "error" else "WARN"
                lines.append(f"  [{severity_marker}] Turn {d.turn}: {d.field}")
                lines.append(f"    Replay:    {d.replay_value}")
                lines.append(f"    Recording: {d.recording_value}")
    else:
        lines.append("")
        lines.append("SUCCESS: No discrepancies found!")
        lines.append("Our parser matches Arena's ground truth.")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def find_matching_recording(replay_path: Path) -> Optional[Path]:
    """Try to find a recording that matches the replay timestamp."""
    recording_dir = Path.home() / ".arenamcp" / "match_validations"
    if not recording_dir.exists():
        return None

    # Get replay modification time
    replay_time = datetime.fromtimestamp(replay_path.stat().st_mtime)

    # Find recordings within 1 hour of replay
    best_match = None
    best_delta = float('inf')

    for rec_path in recording_dir.glob("match_*.json"):
        try:
            rec_time = datetime.fromtimestamp(rec_path.stat().st_mtime)
            delta = abs((rec_time - replay_time).total_seconds())
            if delta < best_delta and delta < 3600:  # Within 1 hour
                best_delta = delta
                best_match = rec_path
        except:
            continue

    return best_match


def live_compare_replay(replay_path: Path) -> ComparisonResult:
    """Compare replay against our parser by processing it live.

    This validates that our current parser code produces correct results.
    """
    # Import here to avoid circular deps
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from arenamcp.gamestate import GameState

    result = ComparisonResult(
        replay_file=str(replay_path),
        recording_file="[LIVE PARSER]"
    )

    # Parse replay
    metadata, messages = parse_replay_v2(replay_path)
    replay_states = extract_replay_states(messages)
    result.total_replay_states = len(replay_states)

    # Process through our GameState parser
    gs = GameState()
    parser_states = []

    for msg in messages:
        if 'gameStateMessage' not in msg:
            continue

        gsm = msg['gameStateMessage']
        ti = gsm.get('turnInfo', {})
        turn = ti.get('turnNumber', 0)

        if turn == 0:
            continue

        # Process through our parser
        gs.update_from_message(gsm)

        # Capture state
        state = {
            'turn': turn,
            'phase': ti.get('phase', ''),
            'life_totals': {seat: p.life_total for seat, p in gs.players.items()},
            'battlefield_grp_ids': {},
            'hand_grp_ids': {},
            'graveyard_grp_ids': {},
        }

        # Extract battlefield by seat
        for obj in gs.battlefield:
            seat = obj.controller_seat_id or obj.owner_seat_id
            if seat not in state['battlefield_grp_ids']:
                state['battlefield_grp_ids'][seat] = set()
            if obj.grp_id:
                state['battlefield_grp_ids'][seat].add(obj.grp_id)

        # Extract hand (local player only)
        for obj in gs.hand:
            if gs.local_seat_id not in state['hand_grp_ids']:
                state['hand_grp_ids'][gs.local_seat_id] = set()
            if obj.grp_id:
                state['hand_grp_ids'][gs.local_seat_id].add(obj.grp_id)

        # Extract graveyard
        for obj in gs.graveyard:
            seat = obj.owner_seat_id
            if seat not in state['graveyard_grp_ids']:
                state['graveyard_grp_ids'][seat] = set()
            if obj.grp_id:
                state['graveyard_grp_ids'][seat].add(obj.grp_id)

        parser_states.append(state)

    result.total_recording_states = len(parser_states)

    # Compare at turn boundaries
    replay_turns = set(s['turn'] for s in replay_states if s['turn'] > 0)
    parser_turns = set(s['turn'] for s in parser_states if s['turn'] > 0)
    all_turns = sorted(replay_turns | parser_turns)

    for turn in all_turns:
        # Get last state for this turn from each source
        replay_state = None
        for s in reversed(replay_states):
            if s['turn'] == turn and s.get('life_totals'):
                replay_state = s
                break

        parser_state = None
        for s in reversed(parser_states):
            if s['turn'] == turn and s.get('life_totals'):
                parser_state = s
                break

        if not replay_state or not parser_state:
            continue

        phase = replay_state.get('phase', '')

        # Compare life totals
        for seat, replay_life in replay_state['life_totals'].items():
            parser_life = parser_state['life_totals'].get(seat)
            if parser_life is not None and parser_life != replay_life:
                result.discrepancies.append(Discrepancy(
                    turn=turn, phase=phase, category="life",
                    field=f"seat_{seat}",
                    replay_value=str(replay_life),
                    recording_value=str(parser_life),
                    severity="error"
                ))

        # Compare battlefield
        for seat in set(replay_state['battlefield_grp_ids'].keys()) | set(parser_state['battlefield_grp_ids'].keys()):
            replay_bf = replay_state['battlefield_grp_ids'].get(seat, set())
            parser_bf = parser_state['battlefield_grp_ids'].get(seat, set())

            if replay_bf != parser_bf:
                missing = replay_bf - parser_bf
                extra = parser_bf - replay_bf

                if missing:
                    result.discrepancies.append(Discrepancy(
                        turn=turn, phase=phase, category="battlefield",
                        field=f"seat_{seat}_missing",
                        replay_value=f"grp_ids: {sorted(missing)}",
                        recording_value="not in parser",
                        severity="warning"
                    ))
                if extra:
                    result.discrepancies.append(Discrepancy(
                        turn=turn, phase=phase, category="battlefield",
                        field=f"seat_{seat}_extra",
                        replay_value="not in replay",
                        recording_value=f"grp_ids: {sorted(extra)}",
                        severity="warning"
                    ))

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compare Arena replay to parser recording"
    )
    parser.add_argument("replay", type=Path, help="Path to Arena .rply file")
    parser.add_argument("recording", type=Path, nargs="?", help="Path to recording JSON (auto-detected if omitted)")
    parser.add_argument("--live", action="store_true", help="Compare replay against live parser (validates current code)")
    parser.add_argument("--output", "-o", type=Path, help="Save report to file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if not args.replay.exists():
        print(f"Error: Replay file not found: {args.replay}")
        return 1

    # Live mode: compare replay against current parser code
    if args.live:
        result = live_compare_replay(args.replay)
    else:
        # Find or validate recording
        recording_path = args.recording
        if recording_path is None:
            recording_path = find_matching_recording(args.replay)
            if recording_path is None:
                print("Error: Could not find matching recording. Please specify path or use --live.")
                return 1
            print(f"Auto-detected recording: {recording_path.name}")

        if not recording_path.exists():
            print(f"Error: Recording file not found: {recording_path}")
            return 1

        # Run comparison against recording
        result = compare_replay_to_recording(args.replay, recording_path)

    # Output
    if args.json:
        output = {
            "replay_file": result.replay_file,
            "recording_file": result.recording_file,
            "replay_states": result.total_replay_states,
            "recording_states": result.total_recording_states,
            "error_count": result.error_count,
            "warning_count": result.warning_count,
            "discrepancies": [
                {
                    "turn": d.turn,
                    "phase": d.phase,
                    "category": d.category,
                    "field": d.field,
                    "replay_value": d.replay_value,
                    "recording_value": d.recording_value,
                    "severity": d.severity
                }
                for d in result.discrepancies
            ]
        }
        report = json.dumps(output, indent=2)
    else:
        report = format_report(result)

    if args.output:
        args.output.write_text(report)
        print(f"Report saved to: {args.output}")
    else:
        print(report)

    return 1 if result.has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
