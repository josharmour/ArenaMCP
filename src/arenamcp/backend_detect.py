"""Lightweight backend detection using only stdlib.

Used by the TUI to detect newly-installed backends on launch.
"""

import shutil
import urllib.request


def detect_backends_quick() -> dict[str, bool]:
    """Check which LLM backends are available right now.

    Returns a dict of backend_name -> is_available.
    HTTP checks use a 2-second timeout so this never blocks for long.
    """
    results: dict[str, bool] = {}

    # Ollama: binary on PATH or HTTP server responding
    ollama_bin = shutil.which("ollama") is not None
    ollama_http = False
    if not ollama_bin:
        try:
            req = urllib.request.Request("http://localhost:11434/", method="GET")
            with urllib.request.urlopen(req, timeout=2):
                ollama_http = True
        except Exception:
            pass
    results["ollama"] = ollama_bin or ollama_http

    # CLI-based backends
    results["claude-code"] = shutil.which("claude") is not None
    results["gemini-cli"] = shutil.which("gemini") is not None
    results["codex-cli"] = shutil.which("codex") is not None

    return results
