import threading
import time

from arenamcp.gamestate import GameState


def test_published_snapshot_is_frame_consistent_under_concurrency() -> None:
    """Readers should never observe mixed-frame turn/player data.

    Writer updates turn and life in a single message; readers consume the
    published immutable snapshot and assert the mapping remains consistent.
    """
    gs = GameState()
    gs.set_local_seat_id(1, source=3)

    mismatches: list[tuple[int, int]] = []
    stop = threading.Event()

    def writer() -> None:
        for i in range(1, 300):
            msg = {
                "type": "GameStateType_Diff",
                "turnInfo": {
                    "turnNumber": i,
                    "activePlayer": 1,
                    "priorityPlayer": 1,
                },
                "players": [
                    {"seatId": 1, "lifeTotal": 1000 + i, "landsPlayedThisTurn": 0},
                ],
            }
            gs.update_from_message(msg)
            # Keep overlap high so readers race with writer publication cadence.
            time.sleep(0.0005)
        stop.set()

    def reader() -> None:
        while not stop.is_set():
            snap = gs.get_published_snapshot()
            turn = int(snap.get("turn_info", {}).get("turn_number", 0) or 0)
            if turn <= 0:
                continue
            players = snap.get("players", [])
            you = next((p for p in players if p.get("seat_id") == 1), None)
            if not you:
                continue
            life = int(you.get("life_total", 0) or 0)
            expected = 1000 + turn
            if life != expected:
                mismatches.append((turn, life))
                break

    wt = threading.Thread(target=writer, daemon=True)
    rt = threading.Thread(target=reader, daemon=True)
    wt.start()
    rt.start()
    wt.join(timeout=5)
    rt.join(timeout=1)

    assert not mismatches, f"Observed mixed-frame snapshot(s): {mismatches[:5]}"
