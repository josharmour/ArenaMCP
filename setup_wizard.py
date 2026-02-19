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

# -- Constants ----------------------------------------------------------------

GITHUB_REPO = "https://github.com/josharmour/ArenaMCP.git"
IS_WIN = sys.platform == "win32"
SETTINGS_DIR = Path.home() / ".arenamcp"
SETTINGS_FILE = SETTINGS_DIR / "settings.json"

# ROOT is resolved at runtime -- see _resolve_root().
# These are set by _init_paths() after ROOT is known.
ROOT: Path = Path(".")
VENV_DIR: Path = Path(".")
PIP_PATH: Path = Path(".")
PYTHON_PATH: Path = Path(".")
ENV_FILE: Path = Path(".")


def _is_repo_dir(p: Path) -> bool:
    """Return True if *p* looks like the ArenaMCP repo root."""
    return (p / "pyproject.toml").exists() and (p / "src" / "arenamcp").is_dir()


def _running_as_exe() -> bool:
    """True when bundled by PyInstaller."""
    return getattr(sys, "frozen", False)


def _find_system_python() -> str:
    """Return the path to a real Python interpreter.

    When running as a PyInstaller exe, sys.executable is the .exe itself,
    which cannot be used to create venvs or run ``-m pip``.  This function
    finds the actual system Python.
    """
    if not _running_as_exe():
        return sys.executable

    # Try common names in order of preference
    for name in ("python", "python3", "py"):
        found = shutil.which(name)
        if found:
            # Verify it's a real interpreter, not us
            try:
                result = subprocess.run(
                    [found, "--version"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0 and "Python" in result.stdout:
                    return found
            except Exception:
                continue
    # Last resort
    return "python"


def _resolve_root() -> Path:
    """Figure out where the ArenaMCP repo lives.

    Priority:
    1. If the script/exe sits inside the repo, use that.
    2. If an existing clone is recorded in settings, reuse it.
    3. Ask the user to point at an existing clone or pick a directory
       for a fresh ``git clone``.
    """
    # When running as PyInstaller exe, __file__ is inside a temp dir.
    # Use the directory containing the exe instead.
    if _running_as_exe():
        exe_dir = Path(sys.executable).resolve().parent
    else:
        exe_dir = Path(__file__).resolve().parent

    # 1. Already inside the repo?
    if _is_repo_dir(exe_dir):
        return exe_dir

    # 2. Previous install recorded in settings?
    settings = load_settings()
    saved = settings.get("install_dir")
    if saved:
        saved_path = Path(saved)
        if _is_repo_dir(saved_path):
            return saved_path

    # 3. Interactive -- ask the user
    print()
    print("    The setup wizard needs to know where ArenaMCP is (or should be).")
    print()
    print("    [1] I already have a git clone -- let me type the path")
    print("    [2] Clone fresh into a folder I choose")
    print()
    choice = prompt_choice(["Existing clone", "Clone fresh"])

    if choice == 1:
        while True:
            raw = prompt_input("Path to ArenaMCP folder")
            p = Path(raw).expanduser().resolve()
            if _is_repo_dir(p):
                return p
            print(f"    '{p}' doesn't look like the ArenaMCP repo (no pyproject.toml + src/arenamcp).")
            if not prompt_yn("Try another path?", default=True):
                sys.exit(1)
    else:
        default_parent = Path.home()
        parent = Path(prompt_input("Parent folder for clone", str(default_parent))).expanduser().resolve()
        parent.mkdir(parents=True, exist_ok=True)
        dest = parent / "ArenaMCP"
        if dest.exists() and _is_repo_dir(dest):
            print(f"    Found existing clone at {dest}")
            return dest
        if not shutil.which("git"):
            print()
            print("    ERROR: git is not installed (or not on your PATH).")
            print()
            if IS_WIN:
                print("    Install it from https://git-scm.com/download/win")
                print("    or run:  winget install Git.Git")
            else:
                print("    Install it with your package manager, e.g.:")
                print("      sudo apt install git   # Debian/Ubuntu")
                print("      brew install git        # macOS")
            print()
            print("    After installing, restart this wizard.")
            sys.exit(1)
        print(f"    Cloning {GITHUB_REPO} into {dest} ...")
        result = subprocess.run(
            ["git", "clone", GITHUB_REPO, str(dest)],
            timeout=120,
        )
        if result.returncode != 0:
            print("    ERROR: git clone failed.")
            sys.exit(1)
        return dest


def _init_paths(root: Path) -> None:
    """Set the module-level path constants from the resolved ROOT."""
    global ROOT, VENV_DIR, PIP_PATH, PYTHON_PATH, ENV_FILE
    ROOT = root
    VENV_DIR = ROOT / "venv"
    PIP_PATH = VENV_DIR / ("Scripts" if IS_WIN else "bin") / ("pip.exe" if IS_WIN else "pip")
    PYTHON_PATH = VENV_DIR / ("Scripts" if IS_WIN else "bin") / ("python.exe" if IS_WIN else "python")
    ENV_FILE = ROOT / ".env"
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


# -- Helpers ------------------------------------------------------------------

def print_header(step_num: int, title: str) -> None:
    """Print a colored section header."""
    bar = "-" * (50 - len(title) - 1)
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
    print(f"    [OK] {msg}")


def fail(msg: str) -> None:
    print(f"    [!!] {msg}")


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


# -- Detection ----------------------------------------------------------------

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

    # 5. Codex CLI
    codex_bin = shutil.which("codex")
    backends["codex-cli"] = {
        "available": bool(codex_bin),
        "running": True if codex_bin else False,
        "details": "found" if codex_bin else "not installed",
        "models": [],  # Codex CLI handles its own models
    }

    return backends


# -- Steps --------------------------------------------------------------------

def step_check_python() -> bool:
    """Step 1: Verify Python version."""
    print_header(1, "Check Python")

    if _running_as_exe():
        # We can't trust sys.version_info (it's the bundled Python).
        # Find and verify the system Python instead.
        python = _find_system_python()
        try:
            result = subprocess.run(
                [python, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"],
                capture_output=True, text=True, timeout=5,
            )
            ver_str = result.stdout.strip()
            parts = tuple(int(x) for x in ver_str.split("."))
            if parts < (3, 10):
                fail(f"Python {ver_str} -- version 3.10+ required")
                info("Please install Python 3.10+ from https://python.org")
                return False
            ok(f"Python {ver_str} (system: {python})")
        except Exception as exc:
            fail(f"Could not find a system Python: {exc}")
            info("Please install Python 3.10+ from https://python.org")
            info("Make sure 'python' is on your PATH.")
            return False
    else:
        v = sys.version_info
        if v < (3, 10):
            fail(f"Python {v.major}.{v.minor}.{v.micro} -- version 3.10+ required")
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
        python = _find_system_python()
        info(f"Creating venv (using {python})...")
        result = subprocess.run(
            [python, "-m", "venv", str(VENV_DIR)],
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
        # Non-fatal -- pip may already be current
        info("pip upgrade skipped (may already be up to date)")

    return True


def step_update_code() -> bool:
    """Step 3: Pull latest code from git before installing dependencies."""
    print_header(3, "Update Code")

    git_bin = shutil.which("git")
    if not git_bin:
        info("git not found on PATH -- skipping auto-update.")
        return True

    # Check if we're inside a git repo
    check = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    if check.returncode != 0:
        info("Not a git repository -- skipping auto-update.")
        return True

    # Show current version
    info("Checking for updates...")
    result = subprocess.run(
        ["git", "ls-remote", "--tags", "origin"],
        capture_output=True, text=True, timeout=10, cwd=str(ROOT),
    )
    if result.returncode != 0:
        info("Could not reach remote -- skipping update (will install from local code).")
        return True

    # Parse remote tags to find latest version
    best = (0, 0, 0)
    best_str = ""
    for line in result.stdout.splitlines():
        parts = line.split("refs/tags/")
        if len(parts) != 2:
            continue
        tag = parts[1].strip()
        if tag.endswith("^{}"):
            continue
        ver_str = tag.lstrip("v")
        try:
            ver_tuple = tuple(int(x) for x in ver_str.split("."))
            if ver_tuple > best:
                best = ver_tuple
                best_str = ver_str
        except (ValueError, TypeError):
            continue

    # Read local version from pyproject.toml (no package import needed)
    local_ver = "0.0.0"
    pyproject = ROOT / "pyproject.toml"
    if pyproject.exists():
        for line in pyproject.read_text().splitlines():
            line = line.strip()
            if line.startswith("version"):
                # version = "0.2.0"
                local_ver = line.split("=", 1)[1].strip().strip('"').strip("'")
                break

    local_tuple = tuple(int(x) for x in local_ver.split("."))

    if best > local_tuple and best_str:
        info(f"Update available: v{local_ver} -> v{best_str}")
        if prompt_yn("Pull latest code before installing?", default=True):
            info("Running git pull --ff-only ...")
            pull = subprocess.run(
                ["git", "pull", "--ff-only", "origin", "master"],
                capture_output=True, text=True, timeout=60, cwd=str(ROOT),
            )
            if pull.returncode == 0:
                ok(f"Updated to latest code")
                # Show summary line
                summary = pull.stdout.strip().splitlines()[-1] if pull.stdout.strip() else ""
                if summary:
                    info(summary)
            else:
                stderr = pull.stderr.strip()
                fail("git pull failed -- installing from current code")
                if "not possible to fast-forward" in stderr or "divergent" in stderr:
                    info("Local branch has diverged. Run 'git pull' manually to resolve.")
                elif "uncommitted changes" in stderr or "dirty" in stderr:
                    info("You have uncommitted local changes. Commit or stash them first.")
                else:
                    info(stderr[:200] if stderr else "Unknown error")
        else:
            info("Skipping update.")
    elif best_str:
        ok(f"Already up to date (v{local_ver})")
    else:
        info("No remote tags found -- skipping version check.")

    return True


def step_install_dependencies() -> bool:
    """Step 4: Install packages from pyproject.toml and extras."""
    print_header(4, "Install Dependencies")

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
    """Step 5: Auto-detect backends, let user choose. Returns (backend, model)."""
    print_header(5, "LLM Backend")

    info("Scanning for available backends...\n")
    backends = detect_backends()

    # Show three categories
    ollama_be = backends["ollama"]
    ollama_tag = f" [{ollama_be['details']}]" if ollama_be["available"] else ""
    cli_available = []
    for key, cmd_name in [("claude-code", "Claude"), ("gemini-cli", "Gemini"), ("codex-cli", "Codex")]:
        if key in backends and backends[key]["available"]:
            cli_available.append(cmd_name)
    cli_tag = f" [{', '.join(cli_available)} detected]" if cli_available else ""

    print(f"    [1] Ollama (Local or Cloud){ollama_tag}")
    print(f"        Run models on your GPU, no internet needed")
    print(f"    [2] CLI Subscription{cli_tag}")
    print(f"        Claude CLI / Gemini CLI / Codex CLI")
    print(f"    [3] API Endpoint (Advanced)")
    print(f"        Any OpenAI-compatible endpoint (URL + API key)")

    print()
    choice = prompt_choice(["Ollama (Local or Cloud)", "CLI Subscription", "API Endpoint"], "Select category")

    model = ""

    # -- Category 1: Ollama --
    if choice == 1:
        backend = "ollama"
        be = ollama_be

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
            info("Browse available models at: https://ollama.com/library")
            model_name = prompt_input("Model to pull (e.g. llama3.2, mistral, phi3)", "llama3.2")
            if prompt_yn(f"Pull {model_name} now?", default=True):
                info(f"Pulling {model_name} -- this may take a while...")
                pull = subprocess.run(["ollama", "pull", model_name])
                if pull.returncode == 0:
                    ok(f"{model_name} ready")
                    model = model_name
                else:
                    fail(f"Pull failed -- retry manually: ollama pull {model_name}")
            if not model:
                model = model_name

        print()
        info("Ollama can connect to a local or remote server:")
        info("  Local:  http://localhost:11434/v1  (default)")
        info("  Cloud:  http://your-server:11434/v1")
        info("  Tunnel: https://my-ollama.example.com/v1")
        ollama_url = prompt_input("Ollama API URL", OLLAMA_DEFAULT_URL)
        settings["ollama_url"] = ollama_url

    # -- Category 2: CLI Subscription --
    elif choice == 2:
        cli_options = []
        cli_keys = []
        for key, label, cmd_name in [
            ("claude-code", "Claude CLI", "claude"),
            ("gemini-cli", "Gemini CLI", "gemini"),
            ("codex-cli", "Codex CLI", "codex"),
        ]:
            be = backends.get(key, {"available": False, "details": "not installed"})
            status = "[OK]" if be.get("available") else "[!!]"
            detail = be.get("details", "not installed")
            cli_options.append(f"{label} [{detail}]")
            cli_keys.append(key)
            print(f"    {status} [{len(cli_options)}] {label} -- {detail}")

        print()
        sub_choice = prompt_choice(cli_options, "Select CLI")
        backend = cli_keys[sub_choice - 1]
        be = backends.get(backend, {"available": False})

        if not be.get("available"):
            cmd_map = {"claude-code": "claude", "gemini-cli": "gemini", "codex-cli": "codex"}
            cmd = cmd_map.get(backend, backend)
            fail(f"'{cmd}' not found on PATH")
            info(f"Install the {cmd} CLI first.")
            if not prompt_yn("Continue anyway?"):
                return backend, model
        else:
            ok(f"{backend} ready")
        model = prompt_input("Model (leave empty for default)", "")

    # -- Category 3: API Endpoint --
    else:
        backend = "api"
        url = prompt_input("Base URL", "https://api.openai.com/v1")
        key = prompt_input("API key (leave empty if none)", "")
        settings["api_url"] = url
        settings["api_key"] = key

        # Try to list models from the endpoint
        models = fetch_models_from_url(url, key)
        if models:
            ok(f"{len(models)} model(s) available:")
            for i, m in enumerate(models[:10], 1):
                info(f"  [{i}] {m}")
            print()
            raw = prompt_input("Model", models[0])
            try:
                idx = int(raw)
                if 1 <= idx <= len(models):
                    model = models[idx - 1]
                else:
                    model = raw
            except ValueError:
                model = raw
        else:
            info("Could not fetch models from endpoint (you can enter one manually).")
            model = prompt_input("Model name", "gpt-4o")

    return backend, model


def step_language(settings: dict) -> str:
    """Step 6: Choose spoken language for TTS and STT."""
    print_header(6, "Language")

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
    """Step 7: Choose voice input mode."""
    print_header(7, "Voice Input")

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
    """Step 8: Quick connectivity and path checks."""
    print_header(8, "Verify")

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
    elif backend == "codex-cli":
        info("Checking codex CLI...")
        if shutil.which("codex"):
            ok("codex CLI found")
        else:
            fail("codex CLI not found on PATH")
    elif backend == "api":
        api_url = settings.get("api_url") or "https://api.openai.com/v1"
        info("Testing API endpoint...")
        api_key = settings.get("api_key") or ""
        if check_url(f"{api_url.rstrip('/')}/models"):
            ok(f"API responding ({api_url})")
        else:
            fail(f"API not reachable at {api_url} (check URL and key)")

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
    print("    " + "=" * 44)
    print("    Setup complete! Configuration saved to:")
    print(f"      {SETTINGS_FILE}")
    print()
    print("    Run the coach with:")
    print("      coach.bat")
    if backend == "ollama":
        info("")
        info(f"  Or: python -m arenamcp.standalone --backend ollama --model {settings.get('model', 'llama3.2')}")
    print("    " + "=" * 44)
    print()


def step_desktop_shortcut() -> None:
    """Step 9: Optionally create a desktop shortcut."""
    print_header(9, "Desktop Shortcut")

    if not IS_WIN:
        info("Desktop shortcut creation is only supported on Windows.")
        return

    if not prompt_yn("Create a desktop shortcut?", default=True):
        info("Skipping shortcut.")
        return

    desktop = Path.home() / "Desktop"
    if not desktop.exists():
        fail("Desktop folder not found -- skipping.")
        return

    shortcut_path = desktop / "ArenaMCP Coach.bat"
    content = f'@echo off\ncd /d "{ROOT}"\ncall coach.bat %*\n'

    try:
        with open(shortcut_path, "w") as f:
            f.write(content)
        ok(f"Shortcut created: {shortcut_path}")
    except Exception as exc:
        fail(f"Failed to create shortcut: {exc}")


# -- Main ---------------------------------------------------------------------

def main() -> int:
    print()
    print("=" * 52)
    print("  ArenaMCP Setup Wizard")
    print("=" * 52)

    # Step 0: Resolve where the repo lives (handles exe-from-Downloads, etc.)
    root = _resolve_root()
    _init_paths(root)
    ok(f"Repo: {ROOT}")

    # Persist install dir so re-runs find it automatically
    settings = load_settings()
    settings["install_dir"] = str(ROOT)
    save_settings(settings)

    # -- Existing repo detected -> offer quick update --
    has_venv = VENV_DIR.exists() and PIP_PATH.exists()
    has_settings = bool(settings.get("backend"))

    if has_venv or has_settings:
        print()
        print("    Existing installation detected.")
        if has_venv:
            print(f"      venv:    {VENV_DIR}")
        if has_settings:
            print(f"      backend: {settings.get('backend')}/{settings.get('model', 'default')}")
        print()
        print("    [1] Quick update (pull code + reinstall deps)")
        print("    [2] Full setup  (reconfigure everything)")
        print()
        mode = prompt_choice(["Quick update", "Full setup"])

        if mode == 1:
            # Quick update: python check -> venv -> pull -> deps -> done
            if not step_check_python():
                return 1
            if not step_virtual_environment():
                return 1
            if not step_update_code():
                return 1
            if not step_install_dependencies():
                return 1
            print()
            print("    " + "=" * 44)
            print("    Update complete! Run the coach with:")
            print("      coach.bat")
            print("    " + "=" * 44)
            print()
            return 0

    # -- Full setup --

    # Step 1: Python
    if not step_check_python():
        return 1

    # Step 2: Venv
    if not step_virtual_environment():
        return 1

    # Step 3: Update code from git (before installing deps)
    if not step_update_code():
        return 1

    # Step 4: Dependencies
    if not step_install_dependencies():
        return 1

    # Step 5: Backend + model
    backend, model = step_detect_and_choose_backend(settings)
    settings["backend"] = backend
    if model:
        settings["model"] = model

    # Step 6: Language
    lang = step_language(settings)
    settings["language"] = lang

    # Step 7: Voice mode
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

    # Step 8: Verify
    step_verify(settings)

    # Step 9: Desktop shortcut
    step_desktop_shortcut()

    return 0


def _pause() -> None:
    """Wait for Enter so the console window doesn't vanish."""
    print()
    try:
        input("    Press Enter to exit...")
    except EOFError:
        pass


if __name__ == "__main__":
    try:
        code = main()
        _pause()
        sys.exit(code)
    except KeyboardInterrupt:
        print("\n\n    Setup cancelled.")
        _pause()
        sys.exit(1)
    except Exception as exc:
        print(f"\n    FATAL ERROR: {exc}")
        import traceback
        traceback.print_exc()
        _pause()
        sys.exit(1)
