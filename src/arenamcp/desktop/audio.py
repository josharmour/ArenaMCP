from __future__ import annotations

import threading
from pathlib import Path

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("This module is not executable")

try:
    import winsound
except ImportError:  # pragma: no cover - non-Windows
    winsound = None  # type: ignore[assignment]


class AudioPlayback:
    _lock = threading.RLock()

    @classmethod
    def play_file(cls, path: str) -> bool:
        if winsound is None or not path:
            return False

        full_path = Path(path).resolve()
        if not full_path.exists():
            return False

        with cls._lock:
            cls._stop_unlocked()
            winsound.PlaySound(
                str(full_path),
                winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
            )
        return True

    @classmethod
    def stop(cls) -> None:
        if winsound is None:
            return
        with cls._lock:
            cls._stop_unlocked()

    @classmethod
    def _stop_unlocked(cls) -> None:
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except RuntimeError:
            pass
