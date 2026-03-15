"""Gemini CLI backend for LLM coaching."""

import logging
import os
import queue
import subprocess
import threading
import time
from typing import Any, Optional

from arenamcp.backends.base import _kill_proc_tree

logger = logging.getLogger(__name__)


class GeminiCliBackend:
    """LLM backend using Gemini CLI (subscription session, no API key)."""

    def __init__(
        self,
        model: Optional[str] = None,
        command: Optional[str] = None,
        persistent: Optional[bool] = None,
        max_turns: Optional[int] = None,
        timeout_s: Optional[float] = None,
        progress_callback: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.command = command or os.environ.get("GEMINI_CLI_CMD", "gemini")
        # Default to non-persistent: Gemini CLI rejects --prompt-interactive
        # when stdin is a pipe (not a TTY), which is always the case when
        # spawned as a subprocess.  One-shot mode (-p) works reliably.
        self.persistent = bool(
            int(os.environ.get("GEMINI_CLI_PERSISTENT", "0"))
            if persistent is None
            else persistent
        )
        self.max_turns = int(os.environ.get("GEMINI_CLI_MAX_TURNS", max_turns or 200))
        self.timeout_s = float(
            os.environ.get("GEMINI_CLI_TIMEOUT_S", timeout_s or 20.0)
        )
        self._proc: Optional[subprocess.Popen[str]] = None
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None
        self._turns = 0
        self._persistent_failed = False
        self._initial_system_prompt: str = ""
        self.progress_callback = progress_callback

    def complete(self, system_prompt: str, user_message: str) -> str:
        """Get completion via Gemini CLI (non-interactive)."""
        if self.persistent and not self._persistent_failed:
            return self._complete_persistent(system_prompt, user_message)
        return self._complete_one_shot(system_prompt, user_message)

    def _build_args(self) -> list[str]:
        import shutil

        cmd = self.command
        if cmd and os.path.isabs(cmd):
            resolved = cmd
        else:
            resolved = shutil.which(cmd) or ""

        # Try common Windows shim names if direct resolution fails
        if not resolved and os.name == "nt":
            resolved = (
                shutil.which(f"{cmd}.ps1")
                or shutil.which(f"{cmd}.cmd")
                or shutil.which("gemini.ps1")
                or shutil.which("gemini.cmd")
                or ""
            )

        if resolved.lower().endswith(".ps1"):
            return [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                resolved,
            ]

        return [resolved or cmd]

    def _start_process(self, system_prompt: str) -> None:
        args = self._build_args() + [
            "--prompt-interactive",
            system_prompt,
            "--output-format",
            "text",
            "--raw-output",
            "--accept-raw-output-risk",
            "--approval-mode",
            "plan",
        ]
        if self.model:
            args += ["--model", self.model]

        # Strip API-key env vars so the CLI always uses subscription auth.
        # GOOGLE_API_KEY / GEMINI_API_KEY cause the CLI to switch from
        # subscription to API-key billing, which can produce billing errors.
        env = {k: v for k, v in os.environ.items()
               if k not in ("GOOGLE_API_KEY", "GEMINI_API_KEY")}

        try:
            # OPTIMIZATION: Use CREATE_NO_WINDOW to hide the console window on Windows
            # and ensure UTF-8 encoding for reliable IPC.
            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NO_WINDOW

            self._proc = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                creationflags=creationflags,
                env=env,
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "Gemini CLI not found. Set GEMINI_CLI_CMD to the full path of gemini.ps1 or gemini.cmd."
            )

        # Track the base system prompt so we don't re-send it every call
        self._initial_system_prompt = system_prompt

        self._queue = queue.Queue()
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def _reader_loop(self) -> None:
        if not self._proc or self._proc.stdout is None:
            return
        for line in self._proc.stdout:
            if line:
                self._queue.put(line)

    def _ensure_process(self, system_prompt: str) -> None:
        if self._proc is None or self._proc.poll() is not None:
            self._start_process(system_prompt)

    def _restart_process(self, system_prompt: str) -> None:
        self.close()
        self._turns = 0
        self._start_process(system_prompt)

    def _complete_persistent(self, system_prompt: str, user_message: str) -> str:
        import uuid

        fallback_to_one_shot = False

        if not self._lock.acquire(timeout=self.timeout_s + 1):
            logger.warning(
                "[GEMINI-CLI] Lock busy (previous call still in progress), skipping"
            )
            return "Error: Backend busy (previous call still in progress)"
        try:
            if self._turns >= self.max_turns:
                logger.info(
                    f"[GEMINI-CLI] Max turns reached ({self.max_turns}); restarting session"
                )
                self._restart_process(system_prompt)

            try:
                self._ensure_process(system_prompt)
            except FileNotFoundError as e:
                return str(e)

            marker = f"<<END-{uuid.uuid4()}>>"

            # Only send dynamic additions, not the full system prompt —
            # the base prompt was already set via --prompt-interactive at startup.
            # This cuts ~3000 chars (~750 tokens) from every call.
            dynamic_context = ""
            base = getattr(self, "_initial_system_prompt", "")
            if system_prompt and system_prompt != base:
                if base and system_prompt.startswith(base):
                    # Extract only the new suffix (deck strategy, rules, etc.)
                    dynamic_context = system_prompt[len(base):]
                else:
                    # System prompt changed completely — include it all
                    dynamic_context = f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\n"

            combined_parts = []
            if dynamic_context:
                combined_parts.append(f"ADDITIONAL CONTEXT:{dynamic_context}")
            combined_parts.append(user_message)
            combined_parts.append(
                f"\nEnd your response with this exact marker on its own line:\n{marker}"
            )
            combined = "\n\n".join(combined_parts)

            try:
                assert self._proc is not None and self._proc.stdin is not None
                self._proc.stdin.write(combined + "\n")
                self._proc.stdin.flush()
            except Exception as e:
                logger.warning(
                    f"[GEMINI-CLI] Persistent write failed, falling back: {e}"
                )
                self._persistent_failed = True
                self.close()
                fallback_to_one_shot = True

            if not fallback_to_one_shot:
                deadline = time.time() + self.timeout_s
                buffer: list[str] = []

                # Emit initial progress
                if self.progress_callback:
                    self.progress_callback("Thinking...")

                while time.time() < deadline:
                    try:
                        chunk = self._queue.get(timeout=0.05)
                    except queue.Empty:
                        # Update elapsed time in progress
                        if self.progress_callback:
                            elapsed = time.time() - (deadline - self.timeout_s)
                            self.progress_callback(f"Thinking... ({elapsed:.0f}s)")
                        continue
                    buffer.append(chunk)
                    if self.progress_callback and len(buffer) == 1:
                        self.progress_callback("Generating response...")
                    if marker in chunk or marker in "".join(buffer[-5:]):
                        break

                if self.progress_callback:
                    self.progress_callback("")

                text = "".join(buffer)
                if marker in text:
                    text = text.split(marker, 1)[0]
                text = text.strip()
                # Detect Gemini CLI errors that get returned as "responses"
                # (e.g., --prompt-interactive rejected when stdin is a pipe)
                is_cli_error = (
                    not text
                    or text.startswith("Error:")
                    or "--prompt-interactive" in text
                    or "cannot be used when input is piped" in text
                )
                if is_cli_error:
                    # Fallback to one-shot if persistent didn't yield output
                    if text:
                        logger.warning(f"[GEMINI-CLI] Persistent mode returned error: {text[:200]}")
                    self._persistent_failed = True
                    fallback_to_one_shot = True
                else:
                    self._turns += 1
                    return text
        finally:
            self._lock.release()

        # One-shot fallback runs OUTSIDE the lock so it doesn't block
        # other callers (e.g. advice calls blocked by deck analysis)
        if fallback_to_one_shot:
            return self._complete_one_shot(system_prompt, user_message)

        return ""

    def _complete_one_shot(self, system_prompt: str, user_message: str) -> str:
        combined = (
            f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\nUSER MESSAGE:\n{user_message}"
        )
        args = self._build_args() + [
            "-p",
            combined,
            "--output-format",
            "text",
            "--raw-output",
            "--accept-raw-output-risk",
            "--approval-mode",
            "plan",
        ]
        if self.model:
            args += ["--model", self.model]

        if self.progress_callback:
            self.progress_callback("Thinking (one-shot)...")

        # Strip API-key env vars so the CLI uses subscription auth.
        env = {k: v for k, v in os.environ.items()
               if k not in ("GOOGLE_API_KEY", "GEMINI_API_KEY")}

        try:
            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NO_WINDOW

            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout_s,
                env=env,
                creationflags=creationflags,
            )
        except FileNotFoundError:
            if self.progress_callback:
                self.progress_callback("")
            return (
                "Error: Gemini CLI not found. "
                "Set GEMINI_CLI_CMD to the full path of gemini.ps1 or gemini.cmd."
            )
        except subprocess.TimeoutExpired:
            if self.progress_callback:
                self.progress_callback("")
            return "Error: Gemini CLI request timed out"
        except Exception as e:
            if self.progress_callback:
                self.progress_callback("")
            return f"Error running Gemini CLI: {e}"

        if self.progress_callback:
            self.progress_callback("")

        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            return f"Error: Gemini CLI failed: {err}"

        return (result.stdout or "").strip()

    def close(self) -> None:
        if self._proc is None:
            return
        _kill_proc_tree(self._proc)
        self._proc = None
