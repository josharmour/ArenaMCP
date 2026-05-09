"""Game state reconstructor for replay-based eval.

Walks a replay's IN messages in order, maintaining a current view of the
match (players, zones, gameObjects, turnInfo, annotations). The state at
the moment a request arrives is what the coach would have seen at decision
time.

We only re-implement the slice of GreInterface's state machine needed to
build prompts:
  - Player list  (life totals, hand size, system seat IDs)
  - Zones        (id -> {type, ownerSeatId})
  - Game objects (instanceId -> object dict)
  - turnInfo     (turn number, phase, active/priority player)

Diff handling:
  - 'Full' replaces state
  - 'Diff' merges:
      * players, turnInfo (replaced if present)
      * gameObjects (added by id; removed via update.diffDeletedInstanceIds)
      * zones (replaced/upserted by id; this also reflects card movements
        because zones list their objects)
      * annotations / persistentAnnotations: tracked but not heavily used

This mirrors the behavior we observed in HeadlessClient.cs but without
requiring the GreInterface assembly. Edge cases (e.g. ObjectIdChanged
annotations renaming an instance) are best-effort.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

from .reader import ReplayMessage


@dataclass
class GameStateSnapshot:
    """A flat snapshot of relevant game state at one moment."""

    msg_id: Optional[int] = None
    game_state_id: Optional[int] = None
    turn_number: int = 0
    phase: str = ""
    active_player: int = 0
    priority_player: int = 0
    players: dict[int, dict] = field(default_factory=dict)   # systemSeatNumber -> player obj
    zones: dict[int, dict] = field(default_factory=dict)     # zoneId -> zone obj
    game_objects: dict[int, dict] = field(default_factory=dict)  # instanceId -> obj
    annotations: dict[int, dict] = field(default_factory=dict)
    persistent_annotations: dict[int, dict] = field(default_factory=dict)
    actions: list[dict] = field(default_factory=list)        # legal actions per state
    local_seat_id: int = 0   # which seat is the player whose perspective we want

    # ---- helpers used by prompt builder ------------------------------------

    def hand(self, seat_id: int) -> list[dict]:
        """Return all gameObjects in the seat's hand zone."""
        hand_zones = [z["zoneId"] for z in self.zones.values()
                      if z.get("type") == "ZoneType_Hand" and z.get("ownerSeatId") == seat_id]
        return [obj for obj in self.game_objects.values()
                if obj.get("zoneId") in hand_zones]

    def battlefield(self, seat_id: int) -> list[dict]:
        """Cards on a player's battlefield."""
        bf_zones = [z["zoneId"] for z in self.zones.values()
                    if z.get("type") == "ZoneType_Battlefield"]
        return [obj for obj in self.game_objects.values()
                if obj.get("zoneId") in bf_zones and obj.get("controllerSeatId") == seat_id]

    def graveyard(self, seat_id: int) -> list[dict]:
        gy_zones = [z["zoneId"] for z in self.zones.values()
                    if z.get("type") == "ZoneType_Graveyard" and z.get("ownerSeatId") == seat_id]
        return [obj for obj in self.game_objects.values()
                if obj.get("zoneId") in gy_zones]

    def life(self, seat_id: int) -> int:
        p = self.players.get(seat_id)
        return int(p.get("lifeTotal", 0)) if p else 0

    def hand_size(self, seat_id: int) -> int:
        # Prefer the zone-listed objects; fall back to player.handSize if the
        # opponent's hand is hidden (Visibility_Private).
        h = self.hand(seat_id)
        if h:
            return len(h)
        # Server reports opponent hand size on their player record.
        p = self.players.get(seat_id) or {}
        return int(p.get("handSize", 0))


def _apply_state_message(state: GameStateSnapshot, gsm: dict) -> None:
    """Mutate ``state`` in place using one gameStateMessage payload.

    Field names follow the proto-3 JSON naming used in `.rply` lines.
    """
    state.game_state_id = gsm.get("gameStateId")

    # Full replaces; Diff merges. Both upsert though — we treat Full as
    # "clear-then-merge" via the same path, after wiping containers.
    if gsm.get("type") == "GameStateType_Full":
        state.players.clear()
        state.zones.clear()
        state.game_objects.clear()
        state.annotations.clear()
        state.persistent_annotations.clear()
        state.actions = []

    # Players (always replace by seat id when present)
    for p in gsm.get("players") or []:
        seat = p.get("systemSeatNumber")
        if seat is not None:
            state.players[int(seat)] = p

    # turnInfo replaces wholesale
    ti = gsm.get("turnInfo")
    if ti:
        state.turn_number = int(ti.get("turnNumber") or state.turn_number)
        state.phase = str(ti.get("phase") or state.phase)
        state.active_player = int(ti.get("activePlayer") or state.active_player)
        state.priority_player = int(ti.get("priorityPlayer") or state.priority_player)

    # zones upsert by zoneId
    for z in gsm.get("zones") or []:
        zid = z.get("zoneId")
        if zid is not None:
            state.zones[int(zid)] = z

    # gameObjects upsert by instanceId
    for o in gsm.get("gameObjects") or []:
        iid = o.get("instanceId")
        if iid is not None:
            state.game_objects[int(iid)] = o

    # Diff deletions: the proto includes a diff*DeletedInstanceIds repeated
    # field that lists instances no longer in the game (e.g., destroyed
    # creatures). Strip them.
    for iid in (gsm.get("diffDeletedInstanceIds") or []):
        state.game_objects.pop(int(iid), None)

    # Annotations are usually transient; we still index them for prompts that
    # want to show "creature took +1/+1 this turn" etc.
    for ann in gsm.get("annotations") or []:
        aid = ann.get("id")
        if aid is not None:
            state.annotations[int(aid)] = ann
    for paid in (gsm.get("diffDeletedPersistentAnnotationIds") or []):
        state.persistent_annotations.pop(int(paid), None)
    for ann in gsm.get("persistentAnnotations") or []:
        aid = ann.get("id")
        if aid is not None:
            state.persistent_annotations[int(aid)] = ann

    # actions: list of legal actions per seat
    if "actions" in gsm:
        state.actions = list(gsm.get("actions") or [])


def walk_states(messages: list[ReplayMessage], local_seat_id: int = 2) -> Iterator[tuple[ReplayMessage, GameStateSnapshot]]:
    """Yield (message, snapshot_after_processing_it) for every IN message.

    The snapshot reflects ALL `GameStateMessage`s up to and including the
    yielded message. For a request message, this is the state the coach
    would face when answering.

    `local_seat_id` defaults to 2 because the player's seat in
    `mtgacoach_*.rply` is consistently 2 (the recording side).
    """
    state = GameStateSnapshot(local_seat_id=local_seat_id)
    for m in messages:
        if m.direction != "IN":
            continue
        if m.msg_type == "GREMessageType_GameStateMessage":
            gsm = m.payload.get("gameStateMessage") or {}
            _apply_state_message(state, gsm)
        elif m.msg_type == "GREMessageType_QueuedGameStateMessage":
            # This wraps multiple gameStateMessages (sent in a burst).
            qg = m.payload.get("queuedGameStateMessage") or {}
            for gsm in qg.get("gameStateMessages") or []:
                _apply_state_message(state, gsm)
        # Update msg_id on snapshot so callers know "as of which proto msgId"
        state.msg_id = m.msg_id
        yield m, state


def snapshot_at_decision(messages: list[ReplayMessage], request: ReplayMessage,
                         local_seat_id: int = 2) -> GameStateSnapshot:
    """Return the state immediately before ``request`` resolves.

    We walk every IN message up to and including the request itself.
    Requests don't carry state updates of their own, but we include them
    in the iteration so the snapshot's ``msg_id`` reflects the request.
    """
    state = GameStateSnapshot(local_seat_id=local_seat_id)
    for m in messages:
        if m.direction != "IN":
            continue
        if m.msg_type == "GREMessageType_GameStateMessage":
            gsm = m.payload.get("gameStateMessage") or {}
            _apply_state_message(state, gsm)
        elif m.msg_type == "GREMessageType_QueuedGameStateMessage":
            qg = m.payload.get("queuedGameStateMessage") or {}
            for gsm in qg.get("gameStateMessages") or []:
                _apply_state_message(state, gsm)
        if m is request:
            state.msg_id = m.msg_id
            return state
    state.msg_id = request.msg_id
    return state


def main():
    """CLI: dump the snapshot at each ActionsAvailable decision."""
    import argparse
    from pathlib import Path
    from .reader import parse_replay_path
    from .decisions import extract_decisions

    p = argparse.ArgumentParser(description="Dump game-state snapshots")
    p.add_argument("path", type=Path)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--seat", type=int, default=2,
                   help="Local player's seat id (the one we're scoring; default 2)")
    args = p.parse_args()

    meta, messages = parse_replay_path(args.path)
    decisions = extract_decisions(messages)
    aa = [d for d in decisions if d.ground_truth and d.ground_truth.kind == "ActionsAvailable"]
    print(f"=== {args.path.name} === local={meta.local_screen_name} opp={meta.opponent_screen_name}")
    print(f"ActionsAvailable decisions: {len(aa)} (showing first {args.limit})\n")
    for d in aa[: args.limit]:
        snap = snapshot_at_decision(messages, d.request, local_seat_id=args.seat)
        you_life = snap.life(args.seat)
        opp_seat = 1 if args.seat == 2 else 2
        opp_life = snap.life(opp_seat)
        you_hand = snap.hand_size(args.seat)
        you_bf = len(snap.battlefield(args.seat))
        opp_bf = len(snap.battlefield(opp_seat))
        print(f"[{d.index:3d}] T{snap.turn_number} {snap.phase[6:]:>10}  "
              f"you {you_life}/{you_hand}h/{you_bf}bf  opp {opp_life}/{opp_bf}bf  "
              f"-> {d.ground_truth.summary}")


if __name__ == "__main__":
    main()
