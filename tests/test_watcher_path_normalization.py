import importlib
import sys
import types


def _load_watcher(monkeypatch):
    """Import watcher with lightweight watchdog stubs."""
    watchdog = types.ModuleType("watchdog")
    observers = types.ModuleType("watchdog.observers")
    events = types.ModuleType("watchdog.events")

    class _Observer:
        def schedule(self, *args, **kwargs):
            return None

        def start(self):
            return None

        def stop(self):
            return None

        def join(self, timeout=None):
            return None

    class _FileSystemEventHandler:
        pass

    class _FileModifiedEvent:
        pass

    class _FileCreatedEvent:
        pass

    observers.Observer = _Observer
    events.FileSystemEventHandler = _FileSystemEventHandler
    events.FileModifiedEvent = _FileModifiedEvent
    events.FileCreatedEvent = _FileCreatedEvent
    watchdog.observers = observers
    watchdog.events = events

    monkeypatch.setitem(sys.modules, "watchdog", watchdog)
    monkeypatch.setitem(sys.modules, "watchdog.observers", observers)
    monkeypatch.setitem(sys.modules, "watchdog.events", events)

    import arenamcp.watcher as watcher

    return importlib.reload(watcher)


def test_normalize_windows_path_in_wsl(monkeypatch):
    watcher = _load_watcher(monkeypatch)
    monkeypatch.setattr(watcher, "_is_wsl", lambda: True)

    path = watcher._normalize_log_path(
        r"C:\Users\Alice\AppData\LocalLow\Wizards Of The Coast\MTGA\Player.log"
    )

    assert str(path) == "/mnt/c/Users/Alice/AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log"


def test_default_log_path_prefers_wsl_candidate(monkeypatch):
    watcher = _load_watcher(monkeypatch)
    monkeypatch.setattr(watcher, "_is_wsl", lambda: True)
    monkeypatch.setattr(watcher.glob, "glob", lambda pattern: [
        "/mnt/c/Users/Bob/AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log"
    ])
    monkeypatch.setenv("LOCALAPPDATA", "")
    monkeypatch.setenv("USERPROFILE", "")

    assert watcher._default_log_path() == (
        "/mnt/c/Users/Bob/AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log"
    )
