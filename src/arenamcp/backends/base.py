"""Base protocol and shared utilities for LLM backends."""

import logging
import os
import subprocess
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


class LLMBackend(Protocol):
    """Protocol for LLM backends that can provide coaching advice."""

    def complete(self, system_prompt: str, user_message: str) -> str:
        """Get a completion from the LLM."""
        ...

    def list_models(self) -> list[str]:
        """List available models (optional)."""
        return []


def _kill_proc_tree(proc: subprocess.Popen) -> None:
    """Kill a subprocess and all its children, suppressing console windows.

    On Windows, ``terminate()`` only kills the top-level process.  If the
    CLI is a ``.cmd``/``.ps1`` wrapper, its child (Node.js / Python) survives
    and may flash a console window.  ``taskkill /T /F`` kills the whole tree.
    """
    pid = proc.pid
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            return
        except Exception:
            pass  # fall through to generic terminate
    try:
        proc.terminate()
        proc.wait(timeout=2.0)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _resolve_command(command: str) -> list[str]:
    """Resolve a CLI command name to an executable path, handling Windows shims.

    Returns a list of args suitable for subprocess.Popen / subprocess.run.
    Handles .ps1 and .cmd shims on Windows.
    """
    import shutil

    cmd = command
    if cmd and os.path.isabs(cmd):
        resolved = cmd
    else:
        resolved = shutil.which(cmd) or ""

    # Try common Windows shim names if direct resolution fails
    if not resolved and os.name == "nt":
        resolved = (
            shutil.which(f"{cmd}.ps1")
            or shutil.which(f"{cmd}.cmd")
            or ""
        )

    if resolved and resolved.lower().endswith(".ps1"):
        return [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            resolved,
        ]

    return [resolved or cmd]
