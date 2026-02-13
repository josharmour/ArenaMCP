#!/usr/bin/env python3
"""
ArenaMCP Interactive Setup Wizard

Guides users through environment setup: venv creation, dependency installation,
LLM backend selection (Ollama, Proxy, Cloud API, Claude/Gemini CLI), model
selection, language configuration, and settings persistence.

Runs with system Python (no venv needed). Uses only stdlib modules.
"""

import json
import os
import shutil
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / "venv"
IS_WIN = sys.platform == "win32"
PIP_PATH = VENV_DIR / ("Scripts" if IS_WIN else "bin") / ("pip.exe" if IS_WIN else "pip")
PYTHON_PATH = VENV_DIR / ("Scripts" if IS_WIN else "bin") / ("python.exe" if IS_WIN else "python")
ENV_FILE = ROOT / ".env"
SETTINGS_DIR = Path.home() / ".arenamcp"
SETTINGS_FILE = SETTINGS_DIR / "settings.json"
MTGA_LOG_DEFAULT = (
    Path(os.environ.get("APPDATA", "")) / "LocalLow" / "Wizards Of The Coast" / "MTGA" / "Player.log"
    if IS_WIN else Path.home() / ".wine" / "MTGA" / "Player.log"  # unlikely but placeholder
)

PROXY_DEFAULT_URL = "http://127.0.0.1:8080/v1"
PROXY_DEFAULT_KEY = "your-api-key-1"
OLLAMA_DEFAULT_URL = "http://localhost:11434/v1"

# Supported languages for TTS (Kokoro) and STT (Whisper)
LANGUAGES = [
    ("en", "English"),
    ("de", "German / Deutsch"),
    ("es", "Spanish / Espanol"),
    ("fr", "French / Francais"),
    ("it", "Italian / Italiano"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("pt", "Portuguese / Portugues"),
    ("zh", "Chinese"),
    ("hi", "Hindi"),
    ("nl", "Dutch / Nederlands (STT only, TTS falls back to English)"),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def print_header(step_num: int, title: str) -> None:
    """Print a colored section header."""
    bar = "\u2500" * (50 - len(title) - 1)
    print(f"\n[{step_num}] {title.upper()} {bar}")


def prompt_choice(options: list[str], prompt_text: str = "Choice") -> int:
    """Show a numbered menu and return the 1-based selection."""
    while True:
        try:
            raw = input(f"\n    {prompt_text} [{'/'.join(str(i+1) for i in range(len(options)))}]: ").strip()
            idx = int(raw)
            if 1 <= idx <= len(options):
                return idx
        except (ValueError, EOFError):
            pass
        print(f"    Please enter a number between 1 and {len(options)}.")


def prompt_input(label: str, default: str = "") -> str:
    """Prompt for text input with an optional default."""
    suffix = f" [{default}]" if default else ""
    try:
        raw = input(f"    {label}{suffix}: ").strip()
    except EOFError:
        raw = ""
    return raw or default


def prompt_yn(label: str, default: bool = False) -> bool:
    """Yes/No prompt."""
    hint = "[y/N]" if not default else "[Y/n]"
    try:
        raw = input(f"    {label} {hint}: ").strip().lower()
    except EOFError:
        raw = ""
    if not raw:
        return default
    return raw.startswith("y")


def ok(msg: str) -> None:
    print(f"    \u2713 {msg}")


def fail(msg: str) -> None:
    print(f"    \u2717 {msg}")


def info(msg: str) -> None:
    print(f"    {msg}")


def run_pip(args: list[str], capture: bool = False) -> subprocess.CompletedProcess:
    """Run pip inside the venv."""
    cmd = [str(PIP_PATH)] + args
    if capture:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    return subprocess.run(cmd, cwd=str(ROOT))


def check_url(url: str, timeout: int = 3) -> bool:
    """Return True if a GET to url succeeds."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False


def fetch_models_from_url(base_url: str, api_key: str = "", timeout: int = 5) -> list[str]:
    """Fetch model IDs from an OpenAI-compatible /models endpoint."""
    try:
        url = f"{base_url.rstrip('/')}/models"
        req = urllib.request.Request(url)
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            return [m.get("id", "?") for m in body.get("data", [])]
    except Exception:
        return []


def fetch_ollama_models() -> list[str]:
    """List locally available Ollama models via CLI."""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10,
        )
        lines = [l for l in result.stdout.strip().splitlines() if l and not l.startswith("NAME")]
        return [line.split()[0] for line in lines]
    except Exception:
        return []


def read_env(path: Path) -> dict[str, str]:
    """Parse a .env file into a dict, preserving order via dict."""
    data: dict[str, str] = {}
    if not path.exists():
        return data
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                data[key.strip()] = value.strip()
    return data


def write_env(path: Path, data: dict[str, str]) -> None:
    """Write a dict as a .env file with a header comment."""
    lines = [
        "# ArenaMCP Configuration",
        "# Generated by setup_wizard.py",
        "",
    ]
    for key, value in data.items():
        lines.append(f"{key}={value}")
    lines.append("")  # trailing newline
    with open(path, "w") as f:
        f.write("\n".join(lines))


def load_settings() -> dict:
    """Load existing settings.json or return empty dict."""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_settings(data: dict) -> None:
    """Write settings dict to ~/.arenamcp/settings.json."""
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── Detection ────────────────────────────────────────────────────────────────

def detect_backends() -> dict[str, dict]:
    """Auto-detect which LLM backends are available.

    Returns dict of backend_name -> {available: bool, details: str, models: list}
    """
    backends = {}

    # 1. Ollama
    ollama_bin = shutil.which("ollama")
    ollama_running = check_url("http://localhost:11434/v1/models")
    ollama_models = []
    if ollama_bin or ollama_running:
        ollama_models = fetch_ollama_models()
        # Also try the API if CLI didn't work
        if not ollama_models and ollama_running:
            ollama_models = fetch_models_from_url(OLLAMA_DEFAULT_URL)
    backends["ollama"] = {
        "available": bool(ollama_bin or ollama_running),
        "running": ollama_running,
        "details": f"{len(ollama_models)} model(s)" if ollama_models else ("installed but no models" if ollama_bin else "not installed"),
        "models": ollama_models,
    }

    # 2. Proxy (cli-proxy-api or compatible)
    proxy_url = os.environ.get("PROXY_BASE_URL", PROXY_DEFAULT_URL)
    proxy_key = os.environ.get("PROXY_API_KEY", PROXY_DEFAULT_KEY)
    proxy_running = check_url(f"{proxy_url}/models")
    proxy_models = fetch_models_from_url(proxy_url, proxy_key) if proxy_running else []
    backends["proxy"] = {
        "available": proxy_running,
        "running": proxy_running,
        "details": f"{len(proxy_models)} model(s)" if proxy_models else "not running",
        "models": proxy_models,
        "url": proxy_url,
    }

    # 3. Claude CLI
    claude_bin = shutil.which("claude")
    backends["claude-code"] = {
        "available": bool(claude_bin),
        "running": True if claude_bin else False,
        "details": "found" if claude_bin else "not installed",
        "models": [],  # Claude CLI handles its own models
    }

    # 4. Gemini CLI
    gemini_bin = shutil.which("gemini")
    backends["gemini-cli"] = {
        "available": bool(gemini_bin),
        "running": True if gemini_bin else False,
        "details": "found" if gemini_bin else "not installed",
        "models": [],  # Gemini CLI handles its own models
    }

    return backends


# ── Steps ────────────────────────────────────────────────────────────────────

def step_check_python() -> bool:
    """Step 1: Verify Python version."""
    print_header(1, "Check Python")
    v = sys.version_info
    if v < (3, 10):
        fail(f"Python {v.major}.{v.minor}.{v.micro} — version 3.10+ required")
        info("Please install Python 3.10+ from https://python.org")
        return False
    ok(f"Python {v.major}.{v.minor}.{v.micro}")
    return True


def step_virtual_environment() -> bool:
    """Step 2: Create or reuse venv, upgrade pip."""
    print_header(2, "Virtual Environment")

    if VENV_DIR.exists() and PIP_PATH.exists():
        ok("venv/ exists")
    else:
        info("Creating venv...")
        result = subprocess.run(
            [sys.executable, "-m", "venv", str(VENV_DIR)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            fail("Failed to create virtual environment")
            if result.stderr:
                info(result.stderr.strip())
            return False
        ok("venv/ created")

    # Activate internally for subprocess calls
    if IS_WIN:
        scripts = str(VENV_DIR / "Scripts")
    else:
        scripts = str(VENV_DIR / "bin")
    os.environ["VIRTUAL_ENV"] = str(VENV_DIR)
    os.environ["PATH"] = scripts + os.pathsep + os.environ.get("PATH", "")

    info("Upgrading pip...")
    result = run_pip(["install", "--upgrade", "pip"], capture=True)
    if result.returncode == 0:
        ok("pip upgraded")
    else:
        # Non-fatal — pip may already be current
        info("pip upgrade skipped (may already be up to date)")

    return True


def step_install_dependencies() -> bool:
    """Step 3: Install packages from pyproject.toml and extras."""
    print_header(3, "Install Dependencies")

    info("Installing core + voice + LLM packages...")
    result = run_pip(["install", "-e", ".[full]"])
    if result.returncode != 0:
        fail("Some packages from pyproject.toml failed")
        info("Trying base install only...")
        run_pip(["install", "-e", "."])

    # Install extras from requirements.txt not covered by pyproject.toml
    extras = [
        "textual", "openai", "websocket-client", "scipy", "Pillow",
        "networkx", "beautifulsoup4", "pyedhrec", "lxml",
        "pyautogui", "pydirectinput-rgx",
    ]
    info("Installing additional packages...")
    result = run_pip(["install"] + extras)
    if result.returncode != 0:
        fail("Some additional packages failed (non-fatal)")
    else:
        ok("All packages installed")

    return True


def step_detect_and_choose_backend(settings: dict) -> tuple[str, str]:
    """Step 4: Auto-detect backends, let user choose. Returns (backend, model)."""
    print_header(4, "LLM Backend")

    info("Scanning for available backends...\n")
    backends = detect_backends()

    # Display detection results
    options = []
    option_keys = []

    for key, label, desc in [
        ("ollama", "Ollama (local, free)", "Run models on your GPU. No internet needed."),
        ("proxy", "CLI Proxy", "Routes to Claude/Gemini/GPT via local proxy server."),
        ("claude-code", "Claude CLI", "Uses the 'claude' command directly."),
        ("gemini-cli", "Gemini CLI", "Uses the 'gemini' command directly."),
    ]:
        be = backends[key]
        status = "\u2713" if be["available"] else "\u2717"
        detail = be["details"]
        tag = f" [{detail}]" if detail else ""
        options.append(f"{label}{tag}")
        option_keys.append(key)
        print(f"    {status} [{len(options)}] {label} — {desc}{tag}")

    print()
    choice = prompt_choice(options, "Select backend")
    backend = option_keys[choice - 1]
    be = backends[backend]

    # ── Backend-specific setup ──

    model = ""

    if backend == "ollama":
        if not be["available"]:
            fail("Ollama not found on PATH")
            info("Install from https://ollama.ai then re-run this wizard.")
            if not prompt_yn("Continue anyway?"):
                return backend, model

        if be["models"]:
            ok(f"{len(be['models'])} model(s) found:")
            for i, m in enumerate(be["models"][:10], 1):
                info(f"  [{i}] {m}")
            print()
            info("Enter the number of a model above, or type a model name.")
            raw = prompt_input("Model", be["models"][0])
            # If user typed a number, map to model name
            try:
                idx = int(raw)
                if 1 <= idx <= len(be["models"]):
                    model = be["models"][idx - 1]
                else:
                    model = raw
            except ValueError:
                model = raw
        else:
            fail("No models pulled yet")
            if prompt_yn("Pull llama3.2 (recommended)?", default=True):
                info("Pulling llama3.2 — this may take a while...")
                pull = subprocess.run(["ollama", "pull", "llama3.2"])
                if pull.returncode == 0:
                    ok("llama3.2 ready")
                    model = "llama3.2"
                else:
                    fail("Pull failed — retry manually: ollama pull llama3.2")
            if not model:
                model = prompt_input("Model name", "llama3.2")

        # Save Ollama URL to settings
        ollama_url = prompt_input("Ollama API URL", OLLAMA_DEFAULT_URL)
        settings["ollama_url"] = ollama_url

    elif backend == "proxy":
        if not be["running"]:
            fail("Proxy not reachable at " + PROXY_DEFAULT_URL)
            info("Start cli-proxy-api or enter a custom URL.\n")

        url = prompt_input("Proxy URL", be.get("url", PROXY_DEFAULT_URL))
        key = prompt_input("API key (leave empty if none)", PROXY_DEFAULT_KEY)
        settings["proxy_url"] = url
        settings["proxy_api_key"] = key

        # Write to .env too for backward compatibility
        env = read_env(ENV_FILE)
        env["PROXY_BASE_URL"] = url
        env["PROXY_API_KEY"] = key
        write_env(ENV_FILE, env)

        # Re-fetch models with possibly updated URL/key
        models = fetch_models_from_url(url, key)
        if models:
            ok(f"{len(models)} model(s) available:")
            for m in models[:10]:
                info(f"  {m}")
            model = prompt_input("Model", models[0])
        else:
            model = prompt_input("Model name", "claude-sonnet-4-5-20250929")

    elif backend in ("claude-code", "gemini-cli"):
        if not be["available"]:
            cmd = "claude" if backend == "claude-code" else "gemini"
            fail(f"'{cmd}' not found on PATH")
            info(f"Install the {cmd} CLI first.")
            if not prompt_yn("Continue anyway?"):
                return backend, model
        ok(f"{backend} ready")
        model = prompt_input("Model (leave empty for default)", "")

    return backend, model


def step_language(settings: dict) -> str:
    """Step 5: Choose spoken language for TTS and STT."""
    print_header(5, "Language")

    info("Choose the language for voice output (TTS) and input (STT).\n")
    for i, (code, name) in enumerate(LANGUAGES, 1):
        current = " (current)" if code == settings.get("language", "en") else ""
        print(f"    [{i:2d}] {code:5s}  {name}{current}")

    print()
    choice = prompt_choice([name for _, name in LANGUAGES], "Language")
    lang_code = LANGUAGES[choice - 1][0]
    lang_name = LANGUAGES[choice - 1][1]
    ok(f"Language: {lang_name} ({lang_code})")
    return lang_code


def step_voice_mode(settings: dict) -> str:
    """Step 6: Choose voice input mode."""
    print_header(6, "Voice Input")

    info("How do you want to talk to the coach?\n")
    print(textwrap.dedent("""\
        [1] Push-to-Talk (recommended)
            Hold F4 to speak, release to send

        [2] Voice Activation
            Auto-detects when you start talking

        [3] Disabled
            No voice input (keyboard only)
    """))

    choice = prompt_choice(["Push-to-Talk", "Voice Activation", "Disabled"])
    modes = ["ptt", "vox", "none"]
    return modes[choice - 1]


def step_verify(settings: dict) -> None:
    """Step 7: Quick connectivity and path checks."""
    print_header(7, "Verify")

    backend = settings.get("backend", "proxy")

    # Test backend
    if backend == "ollama":
        ollama_url = settings.get("ollama_url", OLLAMA_DEFAULT_URL)
        base = ollama_url.replace("/v1", "")
        info("Testing Ollama connection...")
        if check_url(f"{ollama_url}/models"):
            ok(f"Ollama API responding ({ollama_url})")
        elif check_url(f"{base}/api/tags"):
            ok(f"Ollama responding ({base})")
        else:
            fail("Ollama not responding (start it with: ollama serve)")
    elif backend == "proxy":
        url = settings.get("proxy_url") or PROXY_DEFAULT_URL
        info("Testing proxy connection...")
        if check_url(f"{url}/models"):
            ok(f"Proxy responding ({url})")
        else:
            fail("Proxy not reachable (you can start it later)")
    elif backend == "claude-code":
        info("Checking claude CLI...")
        if shutil.which("claude"):
            ok("claude CLI found")
        else:
            fail("claude CLI not found on PATH")
    elif backend == "gemini-cli":
        info("Checking gemini CLI...")
        if shutil.which("gemini"):
            ok("gemini CLI found")
        else:
            fail("gemini CLI not found on PATH")

    # Test MTGA log path
    info("Checking MTGA log path...")
    if MTGA_LOG_DEFAULT.exists():
        ok(f"Player.log found at {MTGA_LOG_DEFAULT}")
    else:
        fail(f"Player.log not found at {MTGA_LOG_DEFAULT}")
        info("This is normal if MTGA hasn't been run yet.")
        info("The coach will find it automatically when MTGA starts.")

    # Success banner
    print()
    print("    " + "\u2550" * 44)
    print("    Setup complete! Configuration saved to:")
    print(f"      {SETTINGS_FILE}")
    print()
    print("    Run the coach with:")
    print("      coach.bat")
    if backend == "ollama":
        info("")
        info(f"  Or: python -m arenamcp.standalone --backend ollama --model {settings.get('model', 'llama3.2')}")
    print("    " + "\u2550" * 44)
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    print()
    print("=" * 52)
    print("  ArenaMCP Setup Wizard")
    print("=" * 52)

    # Load existing settings as starting point
    settings = load_settings()

    # Step 1: Python
    if not step_check_python():
        return 1

    # Step 2: Venv
    if not step_virtual_environment():
        return 1

    # Step 3: Dependencies
    if not step_install_dependencies():
        return 1

    # Step 4: Backend + model
    backend, model = step_detect_and_choose_backend(settings)
    settings["backend"] = backend
    if model:
        settings["model"] = model

    # Step 5: Language
    lang = step_language(settings)
    settings["language"] = lang

    # Step 6: Voice mode
    voice_mode = step_voice_mode(settings)
    settings["voice_mode"] = voice_mode

    # Also write voice mode to .env for backward compat
    if voice_mode != "none":
        env = read_env(ENV_FILE)
        env["VOICE_MODE"] = voice_mode
        write_env(ENV_FILE, env)

    # Save everything to settings.json
    info("\nSaving configuration...")
    save_settings(settings)
    ok(f"Settings saved to {SETTINGS_FILE}")

    # Step 7: Verify
    step_verify(settings)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n    Setup cancelled.")
        sys.exit(1)
