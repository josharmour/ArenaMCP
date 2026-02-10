#!/usr/bin/env python3
"""Analyze replay to find divergences between advisor suggestions and player actions.

This script:
1. Parses the Arena replay to extract what the player actually did
2. Runs the advisor at each decision point to get recommended actions
3. Compares them to identify where you diverged from advice

Usage:
    python analyze_replay_divergence.py <replay.rply> [--backend gemini|claude|ollama]
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arenamcp.gamestate import GameState
from arenamcp.mtgadb import MTGADatabase


@dataclass
class PlayerAction:
    """A player action extracted from replay."""
    frame: int
    action_type: str  # "play_card", "attack", "block", "mulligan", "select_target", etc.
    grp_id: Optional[int] = None
    instance_id: Optional[int] = None
    description: str = ""
    game_state_before: Optional[dict] = None


@dataclass
class DecisionPoint:
    """A point where player had to make a decision."""
    frame: int
    turn: int
    phase: str
    decision_type: str  # "mulligan", "action", "attackers", "blockers", "target"
    game_state: dict
    available_actions: list = None
    player_chose: Optional[PlayerAction] = None
    advisor_suggests: Optional[str] = None


def parse_replay_v2(replay_path: Path) -> tuple[dict, list[dict], list[dict]]:
    """Parse Version2 replay file.

    Returns:
        Tuple of (metadata, in_messages, out_messages)
    """
    with open(replay_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if not lines or not lines[0].strip().startswith("#Version"):
        raise ValueError(f"Not a Version2 replay: {replay_path}")

    metadata = json.loads(lines[1].strip()) if len(lines) > 1 else {}

    in_msgs = []
    out_msgs = []

    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue

        if line.startswith("IN-"):
            colon_idx = line.find(':')
            if colon_idx > 0:
                try:
                    msg = json.loads(line[colon_idx+1:])
                    in_msgs.append(msg)
                except:
                    pass

        elif line.startswith("OUT-"):
            colon_idx = line.find(':')
            if colon_idx > 0:
                try:
                    msg = json.loads(line[colon_idx+1:])
                    out_msgs.append(msg)
                except:
                    pass

    return metadata, in_msgs, out_msgs


def extract_player_actions(out_msgs: list[dict]) -> list[PlayerAction]:
    """Extract meaningful player actions from OUT messages."""
    actions = []

    for i, msg in enumerate(out_msgs):
        msg_type = msg.get('type', '')

        # Skip UI messages (hover, etc.)
        if msg_type == 'ClientMessageType_UIMessage':
            continue

        action = None

        if msg_type == 'ClientMessageType_MulliganResp':
            resp = msg.get('mulliganResp', {})
            decision = resp.get('decision', '')
            action = PlayerAction(
                frame=i,
                action_type='mulligan',
                description=f"Mulligan: {decision.replace('MulliganOption_', '')}"
            )

        elif msg_type == 'ClientMessageType_PerformActionResp':
            resp = msg.get('performActionResp', {})
            for act in resp.get('actions', []):
                act_type = act.get('actionType', '')
                grp_id = act.get('grpId', 0)
                inst_id = act.get('instanceId', 0)
                action = PlayerAction(
                    frame=i,
                    action_type='play_card',
                    grp_id=grp_id,
                    instance_id=inst_id,
                    description=f"Play card grp={grp_id} (ActionType: {act_type.replace('ActionType_', '')})"
                )

        elif msg_type == 'ClientMessageType_DeclareAttackersResp':
            resp = msg.get('declareAttackersResp', {})
            if resp.get('autoDeclare'):
                target = resp.get('autoDeclareDamageRecipient', {})
                target_type = target.get('type', '')
                action = PlayerAction(
                    frame=i,
                    action_type='attack',
                    description=f"Attack: Auto-declare attackers targeting {target_type.replace('DamageRecType_', '')}"
                )

        elif msg_type == 'ClientMessageType_ChooseStartingPlayerResp':
            resp = msg.get('chooseStartingPlayerResp', {})
            seat = resp.get('systemSeatId', 0)
            action = PlayerAction(
                frame=i,
                action_type='choose_starter',
                description=f"Choose to {'play' if seat == 2 else 'draw'}"
            )

        elif msg_type == 'ClientMessageType_EffectCostResp':
            resp = msg.get('effectCostResp', {})
            cost_type = resp.get('effectCostType', '')
            selection = resp.get('costSelection', {})
            ids = selection.get('ids', [])
            action = PlayerAction(
                frame=i,
                action_type='select_target',
                description=f"Target selection: {cost_type.replace('EffectCostType_', '')} - IDs: {ids}"
            )

        if action:
            actions.append(action)

    return actions


def find_decision_points(in_msgs: list[dict], out_msgs: list[dict], gs: GameState) -> list[DecisionPoint]:
    """Find all decision points in the game."""
    decision_points = []
    player_actions = extract_player_actions(out_msgs)
    action_idx = 0

    # Track which request messages correspond to decision points
    request_types = {
        'GREMessageType_MulliganReq': 'mulligan',
        'GREMessageType_ActionsAvailableReq': 'action',
        'GREMessageType_DeclareAttackersReq': 'attackers',
        'GREMessageType_DeclareBlockersReq': 'blockers',
        'GREMessageType_SelectTargetsReq': 'target',
        'GREMessageType_SelectNReq': 'select',
        'GREMessageType_PromptReq': 'prompt',
    }

    for i, msg in enumerate(in_msgs):
        msg_type = msg.get('type', '')

        # Update game state
        if 'gameStateMessage' in msg:
            gs.update_from_message(msg['gameStateMessage'])

        # Check if this is a decision request
        if msg_type in request_types:
            decision_type = request_types[msg_type]

            # Get current turn/phase
            turn = gs.turn_info.turn_number
            phase = gs.turn_info.phase

            # Find the player's response to this decision
            player_action = None
            if action_idx < len(player_actions):
                player_action = player_actions[action_idx]
                action_idx += 1

            # Get game state snapshot
            game_state = gs.get_snapshot()

            decision_points.append(DecisionPoint(
                frame=i,
                turn=turn,
                phase=phase,
                decision_type=decision_type,
                game_state=game_state,
                player_chose=player_action,
            ))

    return decision_points


def get_advisor_suggestion(game_state: dict, decision_type: str, backend: str = "gemini") -> Optional[str]:
    """Get advisor suggestion for a decision point."""
    try:
        from arenamcp.coach import CoachEngine, create_backend

        backend_obj = create_backend(backend)
        coach = CoachEngine(backend=backend_obj)

        # Map decision type to trigger
        trigger_map = {
            'mulligan': 'decision_required',
            'action': 'priority',
            'attackers': 'combat_attackers',
            'blockers': 'combat_blockers',
            'target': 'decision_required',
            'select': 'decision_required',
            'prompt': 'decision_required',
        }
        trigger = trigger_map.get(decision_type, 'decision_required')

        advice = coach.get_advice(game_state, trigger=trigger, style="concise")
        return advice
    except Exception as e:
        return f"[Error getting advice: {e}]"


def analyze_divergences(replay_path: Path, backend: str = "gemini", skip_advisor: bool = False) -> dict:
    """Analyze replay for divergences between player actions and advisor suggestions."""
    print(f"Analyzing replay: {replay_path.name}")

    # Parse replay
    metadata, in_msgs, out_msgs = parse_replay_v2(replay_path)
    print(f"Loaded {len(in_msgs)} server messages, {len(out_msgs)} player messages")

    # Extract player info
    local_player = metadata.get('Local', {}).get('ScreenName', 'Unknown')
    opponent = metadata.get('Opponent', {}).get('ScreenName', 'Unknown')
    print(f"Players: {local_player} vs {opponent}")

    # Initialize card database for name resolution
    card_db = None
    try:
        card_db = MTGADatabase()
        print(f"Card database loaded")
    except Exception as e:
        print(f"Warning: Could not load card database: {e}")

    # Initialize game state
    gs = GameState()

    # Find decision points
    decision_points = find_decision_points(in_msgs, out_msgs, gs)
    print(f"Found {len(decision_points)} decision points")

    # Get advisor suggestions for each decision point
    if not skip_advisor:
        print("\nGetting advisor suggestions (this may take a moment)...")
        for i, dp in enumerate(decision_points):
            print(f"  [{i+1}/{len(decision_points)}] Turn {dp.turn} {dp.phase}: {dp.decision_type}")
            dp.advisor_suggests = get_advisor_suggestion(dp.game_state, dp.decision_type, backend)
    else:
        print("\nSkipping advisor suggestions (--skip-advisor)")

    # Build report
    report = {
        "replay_file": str(replay_path),
        "local_player": local_player,
        "opponent": opponent,
        "total_decisions": len(decision_points),
        "decisions": [],
    }

    for dp in decision_points:
        grp_id = dp.player_chose.grp_id if dp.player_chose else None
        card_name = None
        if grp_id and card_db:
            try:
                card = card_db.get_card(grp_id)
                if card:
                    card_name = card.name
            except:
                pass

        player_action = dp.player_chose.description if dp.player_chose else None
        if card_name and grp_id:
            # Replace grp_id with card name in description
            player_action = player_action.replace(f"grp={grp_id}", f"{card_name}")

        report["decisions"].append({
            "frame": dp.frame,
            "turn": dp.turn,
            "phase": dp.phase,
            "type": dp.decision_type,
            "player_action": player_action,
            "card_name": card_name,
            "grp_id": grp_id,
            "advisor_suggestion": dp.advisor_suggests,
        })

    return report


def print_report(report: dict):
    """Print human-readable report."""
    print("\n" + "=" * 70)
    print("REPLAY DIVERGENCE ANALYSIS")
    print("=" * 70)
    print(f"Replay: {Path(report['replay_file']).name}")
    print(f"Players: {report['local_player']} vs {report['opponent']}")
    print(f"Total decisions: {report['total_decisions']}")
    print("=" * 70)

    for dec in report['decisions']:
        print(f"\n[Turn {dec['turn']}] {dec['phase']} - {dec['type'].upper()}")
        if dec['player_action']:
            print(f"  You did:  {dec['player_action']}")
        else:
            print(f"  You did:  (unknown)")
        if dec['advisor_suggestion']:
            advice = dec['advisor_suggestion']
            # Truncate long advice
            if len(advice) > 100:
                advice = advice[:100] + "..."
            print(f"  Advisor:  {advice}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze replay divergences")
    parser.add_argument("replay", type=Path, help="Path to Arena .rply file")
    parser.add_argument("--backend", default="gemini", help="LLM backend (gemini/claude/ollama)")
    parser.add_argument("--skip-advisor", action="store_true", help="Skip getting advisor suggestions")
    parser.add_argument("--output", "-o", type=Path, help="Save report to JSON file")

    args = parser.parse_args()

    if not args.replay.exists():
        print(f"Error: Replay file not found: {args.replay}")
        return 1

    report = analyze_divergences(args.replay, args.backend, args.skip_advisor)
    print_report(report)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
