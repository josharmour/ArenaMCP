import subprocess
from types import SimpleNamespace

import pytest

import arenamcp.standalone as standalone
from arenamcp.standalone import StandaloneCoach


def test_standalone_initializes_advice_history():
    coach = StandaloneCoach(backend="codex-cli", register_hotkeys=False)
    assert hasattr(coach, "_advice_history")
    assert isinstance(coach._advice_history, list)


def test_standalone_replaces_invalid_saved_codex_model(monkeypatch):
    class _FakeSettings:
        def __init__(self):
            self.data = {
                "backend": "codex-cli",
                "model": "gpt-5-mini",
                "voice_mode": "ptt",
                "advice_frequency": "every_priority",
            }

        def get(self, key, default=None):
            return self.data.get(key, default)

        def set(self, key, value, save=True):
            self.data[key] = value

    fake_settings = _FakeSettings()
    monkeypatch.setattr(standalone, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(
        "arenamcp.coach.get_models_for_provider",
        lambda provider: [
            ("GPT-5.4 Pro", "gpt-5.4-pro"),
            ("GPT-5.3 Codex", "gpt-5.3-codex"),
        ],
    )

    coach = StandaloneCoach(backend="codex-cli", register_hotkeys=False)

    assert coach.model_name == "gpt-5.4-pro"
    assert fake_settings.data["model"] == "gpt-5.4-pro"


def test_probe_sounddevice_import_timeout(monkeypatch):
    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="python -c import sounddevice", timeout=8)

    monkeypatch.setattr(standalone.subprocess, "run", _raise_timeout)
    ok, reason = standalone._probe_sounddevice_import(timeout_seconds=8.0)
    assert ok is False
    assert "timeout" in reason


def test_probe_sounddevice_import_nonzero_exit(monkeypatch):
    result = subprocess.CompletedProcess(
        args=["python", "-c", "import sounddevice"],
        returncode=1,
        stdout="",
        stderr="Traceback...\nImportError: bad audio setup",
    )
    monkeypatch.setattr(standalone.subprocess, "run", lambda *a, **k: result)
    ok, reason = standalone._probe_sounddevice_import(timeout_seconds=8.0)
    assert ok is False
    assert reason == "ImportError: bad audio setup"


def test_probe_sounddevice_import_success(monkeypatch):
    result = subprocess.CompletedProcess(
        args=["python", "-c", "import sounddevice"],
        returncode=0,
        stdout="",
        stderr="",
    )
    monkeypatch.setattr(standalone.subprocess, "run", lambda *a, **k: result)
    ok, reason = standalone._probe_sounddevice_import(timeout_seconds=8.0)
    assert ok is True
    assert reason == "ok"


def test_init_voice_disables_voice_when_probe_fails(monkeypatch):
    class _UI:
        def __init__(self):
            self.status_calls = []
            self.error_calls = []

        def status(self, key, value):
            self.status_calls.append((key, value))

        def error(self, message):
            self.error_calls.append(message)

    monkeypatch.setattr(
        standalone,
        "_probe_sounddevice_import",
        lambda timeout_seconds=8.0: (False, "timeout after 8s"),
    )

    coach = StandaloneCoach.__new__(StandaloneCoach)
    coach._backend_name = "proxy"
    coach._voice_mode = "ptt"
    coach._voice_output = None
    coach._voice_input = None
    coach.ui = _UI()

    coach._init_voice()

    assert coach._voice_output is None
    assert coach._voice_input is None
    assert coach.ui.status_calls[-1] == ("VOICE", "Audio init failed - voice disabled")
    assert coach.ui.error_calls[-1].startswith("Audio driver issue:")


def test_save_bug_report_can_skip_ui_announcement(monkeypatch, tmp_path):
    class _UI:
        def __init__(self):
            self.log_calls = []
            self.error_calls = []

        def log(self, message):
            self.log_calls.append(message)

        def error(self, message):
            self.error_calls.append(message)

    class _Settings:
        def __init__(self):
            self._data = {}

        def set(self, key, value, save=True):
            self._data[key] = value

    monkeypatch.setattr(standalone, "LOG_DIR", tmp_path)
    monkeypatch.setattr(standalone, "LOG_FILE", tmp_path / "standalone.log")
    monkeypatch.setattr(standalone, "copy_to_clipboard", lambda text: True)

    coach = StandaloneCoach.__new__(StandaloneCoach)
    coach.ui = _UI()
    coach.settings = _Settings()
    coach._backend_name = "codex-cli"
    coach._model_name = "gpt-5.4-pro"
    coach._voice_mode = "ptt"
    coach.advice_style = "concise"
    coach._advice_frequency = "every_priority"
    coach.draft_mode = False
    coach.set_code = None
    coach._auto_speak = False
    coach._mcp = None
    coach._voice_output = None
    coach._voice_input = None
    coach._coach = None
    coach._advice_history = []
    coach._recent_errors = []
    coach._start_time = standalone.datetime.now()

    bug_path = coach.save_bug_report("Copy Debug (F7)", announce=False)

    assert bug_path is not None
    assert bug_path.exists()
    assert coach.ui.log_calls == []
    assert coach.ui.error_calls == []


def test_init_mcp_starts_background_cache_warmup(monkeypatch):
    thread_events = []

    class _FakeThread:
        def __init__(self, target=None, daemon=None, name=None):
            self.target = target
            thread_events.append(("init", target, daemon, name))

        def start(self):
            thread_events.append(("start", self.target))

    class _FakeMCPClient:
        pass

    monkeypatch.setattr(standalone, "MCPClient", _FakeMCPClient)
    monkeypatch.setattr(standalone.threading, "Thread", _FakeThread)

    coach = StandaloneCoach.__new__(StandaloneCoach)
    coach._card_cache_warm_started = False

    coach._init_mcp()

    assert isinstance(coach._mcp, _FakeMCPClient)
    assert thread_events[0] == ("init", coach._warm_local_card_caches, True, "card-cache-warm")
    assert thread_events[1] == ("start", coach._warm_local_card_caches)


def test_has_explicit_game_end_evidence_true_for_persistent_result(monkeypatch):
    fake_gs = SimpleNamespace(
        game_ended_event=SimpleNamespace(is_set=lambda: False),
        last_game_result="win",
        _pre_reset_snapshot=None,
    )
    fake_server = SimpleNamespace(game_state=fake_gs)
    monkeypatch.setattr(standalone.importlib, "import_module", lambda name: fake_server)

    coach = StandaloneCoach.__new__(StandaloneCoach)

    assert coach._has_explicit_game_end_evidence() is True


def test_has_explicit_game_end_evidence_false_without_signal(monkeypatch):
    fake_gs = SimpleNamespace(
        game_ended_event=SimpleNamespace(is_set=lambda: False),
        last_game_result=None,
        _pre_reset_snapshot=None,
    )
    fake_server = SimpleNamespace(game_state=fake_gs)
    monkeypatch.setattr(standalone.importlib, "import_module", lambda name: fake_server)

    coach = StandaloneCoach.__new__(StandaloneCoach)

    assert coach._has_explicit_game_end_evidence() is False


def test_tui_adapter_safe_call_noops_when_app_not_running():
    tui_mod = pytest.importorskip("arenamcp.tui")

    class _FakeApp:
        is_running = False
        _thread_id = 0

        def call_from_thread(self, method, *args, **kwargs):
            raise AssertionError("call_from_thread should not be used when app is stopped")

    adapter = tui_mod.TUIAdapter(_FakeApp())

    adapter.log("hello")
    adapter.status("ANALYSIS", "done")


def test_tui_adapter_safe_call_swallows_app_not_running_runtimeerror():
    tui_mod = pytest.importorskip("arenamcp.tui")

    class _FakeApp:
        is_running = True
        _thread_id = 0

        def call_from_thread(self, method, *args, **kwargs):
            raise RuntimeError("App is not running")

        def write_log(self, message):
            raise AssertionError("write_log should not be called directly in this test")

    adapter = tui_mod.TUIAdapter(_FakeApp())

    adapter.log("hello")
