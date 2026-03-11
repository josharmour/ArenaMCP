import subprocess

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
