"""Claude Code CLI backend for LLM coaching."""

import json
import logging
import os
import queue
import subprocess
import threading
import time
from typing import Any, Optional

from arenamcp.backends.base import _kill_proc_tree

logger = logging.getLogger(__name__)


class ClaudeCodeBackend:
    """LLM backend using Claude Code CLI (subscription session, no API key).

    Uses a persistent subprocess with stream-json I/O for low latency.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        command: Optional[str] = None,
        max_turns: Optional[int] = None,
        timeout_s: Optional[float] = None,
        add_dirs: Optional[list[str]] = None,
        tools: Optional[list[str]] = None,
        permission_mode: Optional[str] = None,
        progress_callback: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.command = command or os.environ.get("CLAUDE_CODE_CMD", "claude")
        self.max_turns = int(os.environ.get("CLAUDE_CODE_MAX_TURNS", max_turns or 200))
        self.timeout_s = float(
            os.environ.get("CLAUDE_CODE_TIMEOUT_S", timeout_s or 12.0)
        )
        self.add_dirs = add_dirs or []
        self.tools = tools or []
        self.permission_mode = permission_mode or "dontAsk"
        self.progress_callback = progress_callback

        self._base_system_prompt = os.environ.get(
            "CLAUDE_CODE_SYSTEM_PROMPT",
            "You are an MTG coach. Follow the instructions in the user message. Do not use tools.",
        )

        self._proc: Optional[subprocess.Popen[str]] = None
        self._queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._lock = threading.Lock()
        self._init_event = threading.Event()
        self._turns = 0
        self._session_id: Optional[str] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None

    def _start_process(self) -> None:
        args = [
            self.command,
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--replay-user-messages",
            "--verbose",
            "--permission-mode",
            self.permission_mode,
            "--no-session-persistence",
        ]
        if self.tools:
            args += ["--tools", ",".join(self.tools)]
        else:
            args += ["--tools", ""]
        for d in self.add_dirs:
            args += ["--add-dir", d]
        if self.model:
            args += ["--model", self.model]
        if self._base_system_prompt:
            args += ["--system-prompt", self._base_system_prompt]

        # Strip API-key env vars so the CLI always uses subscription auth.
        # If ANTHROPIC_API_KEY is set (e.g. from other dev work), the CLI
        # silently switches from subscription to API-key billing, which can
        # produce "Credit balance is too low" errors.
        env = {k: v for k, v in os.environ.items()
               if k not in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY")}

        try:
            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NO_WINDOW

            self._proc = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=env,
                creationflags=creationflags,
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Claude Code CLI not found at '{self.command}'. "
                "Install it or set CLAUDE_CODE_CMD to the correct path."
            )

        self._init_event.clear()
        self._queue = queue.Queue()

        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            args=(self._proc.stdout, False),
            daemon=True,
        )
        self._reader_thread.start()

        self._stderr_thread = threading.Thread(
            target=self._reader_loop,
            args=(self._proc.stderr, True),
            daemon=True,
        )
        self._stderr_thread.start()

        if not self._init_event.wait(timeout=5.0):
            rc = self._proc.poll() if self._proc else None
            if rc is not None:
                logger.warning(
                    f"Claude Code CLI exited during init (code {rc}). "
                    "Check 'claude --version' and auth status."
                )
            else:
                logger.warning(
                    "Claude Code CLI did not emit init event in 5s "
                    "(process still running — may be doing first-time setup)"
                )

    def _reader_loop(self, stream, is_stderr: bool) -> None:
        if stream is None:
            return
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                if is_stderr:
                    logger.warning(f"[CLAUDE-CLI][stderr] {line}")
                else:
                    logger.debug(f"[CLAUDE-CLI] Non-JSON line: {line}")
                continue

            if data.get("type") == "system" and data.get("subtype") == "init":
                self._session_id = data.get("session_id")
                self._init_event.set()

            # Log error events so they appear in bug reports
            if data.get("type") == "error":
                logger.warning(f"[CLAUDE-CLI] Error event: {data}")

            self._queue.put(data)

    def _ensure_process(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            self._start_process()

    def _restart_process(self) -> None:
        self.close()
        self._turns = 0
        self._start_process()

    def _extract_assistant_text(self, message: dict[str, Any]) -> str:
        content = message.get("content", [])
        parts: list[str] = []
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "text" and part.get("text"):
                    parts.append(part["text"])
        return "".join(parts)

    def complete(self, system_prompt: str, user_message: str) -> str:
        """Get completion via Claude Code CLI."""
        if not self._lock.acquire(timeout=self.timeout_s + 1):
            logger.warning(
                "[CLAUDE-CLI] Lock busy (another call in progress), skipping"
            )
            return "Error: Backend busy (previous call still in progress)"
        try:
            if self._turns >= self.max_turns:
                logger.info(
                    f"[CLAUDE-CLI] Max turns reached ({self.max_turns}); restarting session"
                )
                self._restart_process()

            self._ensure_process()

            # Embed dynamic system prompt in the user message to avoid restarts.
            combined = (
                "SYSTEM INSTRUCTIONS:\n"
                f"{system_prompt}\n\n"
                "USER MESSAGE:\n"
                f"{user_message}"
            )

            payload = {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": combined}],
                },
            }

            try:
                assert self._proc is not None and self._proc.stdin is not None
                self._proc.stdin.write(json.dumps(payload) + "\n")
                self._proc.stdin.flush()
            except Exception as e:
                logger.error(f"[CLAUDE-CLI] Failed to write to subprocess: {e}")
                self._restart_process()
                return "Error: Claude Code CLI write failed"

            assistant_text = ""
            result_text = ""
            deadline = time.time() + self.timeout_s

            # Emit initial progress
            if self.progress_callback:
                self.progress_callback("Thinking...")

            while time.time() < deadline:
                # Check if process died
                if self._proc and self._proc.poll() is not None:
                    rc = self._proc.returncode
                    logger.warning(f"[CLAUDE-CLI] Process exited with code {rc} during response wait")
                    return f"Error: Claude Code CLI exited (code {rc})"

                try:
                    data = self._queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                msg_type = data.get("type")
                if msg_type == "assistant":
                    message = data.get("message", {})
                    assistant_text = (
                        self._extract_assistant_text(message) or assistant_text
                    )
                    # Report subtask progress from tool_use blocks
                    if self.progress_callback:
                        content = message.get("content", [])
                        if isinstance(content, list):
                            for part in content:
                                part_type = part.get("type", "")
                                if part_type == "tool_use":
                                    tool_name = part.get("name", "tool")
                                    self.progress_callback(f"Using {tool_name}...")
                                elif part_type == "text" and part.get("text"):
                                    # Trim to first 60 chars for status display
                                    snippet = part["text"][:60].replace("\n", " ")
                                    self.progress_callback(f"Responding: {snippet}...")
                elif msg_type == "result":
                    result_text = data.get("result") or result_text
                    if self.progress_callback:
                        self.progress_callback("")
                    break
                elif msg_type == "error":
                    if self.progress_callback:
                        self.progress_callback("")
                    return f"Error from Claude CLI: {data}"

            if self.progress_callback:
                self.progress_callback("")

            if not assistant_text:
                assistant_text = result_text

            self._turns += 1
            return assistant_text or "Error: No response from Claude Code"
        finally:
            self._lock.release()

    def close(self) -> None:
        if self._proc is None:
            return
        _kill_proc_tree(self._proc)
        self._proc = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
