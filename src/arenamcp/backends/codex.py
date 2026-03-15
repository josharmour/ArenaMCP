"""Codex CLI / Azure OpenAI backend for LLM coaching."""

import json
import logging
import os
import subprocess
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CodexCliBackend:
    """LLM backend using Codex's Azure OpenAI Responses API directly.

    Instead of spawning ``codex exec`` (which runs a full agent loop and
    takes 20-30s), this backend reads the Azure endpoint and API key from
    ``~/.codex/config.toml`` and calls the Responses API directly for
    sub-3-second latency.

    Falls back to the ``codex exec`` subprocess if Azure credentials are
    not available.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        command: Optional[str] = None,
        timeout_s: Optional[float] = None,
        progress_callback: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.command = command or os.environ.get("CODEX_CLI_CMD", "codex")
        self.timeout_s = float(
            os.environ.get("CODEX_CLI_TIMEOUT_S", timeout_s or 5.0)
        )
        self.progress_callback = progress_callback

        # Try to load Azure credentials (env var first, then codex config)
        self._azure_base_url: Optional[str] = None
        self._azure_api_key: Optional[str] = None
        self._azure_api_version: str = "2025-04-01-preview"
        self._use_direct_api = False
        self._load_azure_config()

        # Direct env var override (simpler than codex config parsing)
        if not self._use_direct_api:
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            base_url = os.environ.get(
                "AZURE_OPENAI_BASE_URL",
                "https://bmfllm-resource.cognitiveservices.azure.com/openai",
            )
            if api_key:
                self._azure_api_key = api_key
                self._azure_base_url = base_url
                self._use_direct_api = True
                if not self.model:
                    self.model = "gpt-5.4"
                logger.info(
                    f"CodexCliBackend: using Azure API via env var "
                    f"({base_url}, model={self.model})"
                )

    def _load_azure_config(self) -> None:
        """Load Azure OpenAI credentials from ~/.codex/config.toml."""
        try:
            config_path = os.path.join(os.path.expanduser("~"), ".codex", "config.toml")
            if not os.path.exists(config_path):
                return

            # Simple TOML parsing for the fields we need
            with open(config_path, "r") as f:
                text = f.read()
            import re

            # Get model if not set
            if not self.model:
                m = re.search(r'^model\s*=\s*"([^"]+)"', text, re.MULTILINE)
                if m:
                    self.model = m.group(1)

            # Get Azure base_url
            m = re.search(r'base_url\s*=\s*"([^"]+)"', text)
            if m:
                self._azure_base_url = m.group(1)

            # Get api-version from query_params
            m = re.search(r'"api-version"\s*=\s*"([^"]+)"', text)
            if m:
                self._azure_api_version = m.group(1)

            # Get env key name for API key
            m = re.search(r'env_key\s*=\s*"([^"]+)"', text)
            if m:
                env_key = m.group(1)
                self._azure_api_key = os.environ.get(env_key)

            if self._azure_base_url and self._azure_api_key:
                self._use_direct_api = True
                logger.info(
                    f"CodexCliBackend: using direct Azure API "
                    f"({self._azure_base_url}, model={self.model})"
                )
            else:
                logger.info("CodexCliBackend: Azure credentials not found, using codex exec fallback")
        except Exception as e:
            logger.debug(f"Failed to load codex Azure config: {e}")

        # Fire a warmup request in background to avoid cold-start timeout
        if self._use_direct_api:
            self._warmup_azure()

    def _warmup_azure(self) -> None:
        """Send a minimal request to warm up the Azure deployment."""
        def _do_warmup():
            try:
                import urllib.request
                url = (
                    f"{self._azure_base_url}/responses"
                    f"?api-version={self._azure_api_version}"
                )
                body = json.dumps({
                    "model": self.model or "gpt-5.4",
                    "input": "hi",
                    "max_output_tokens": 16,
                    "reasoning": {"effort": "medium"},
                }).encode()
                req = urllib.request.Request(url, data=body, headers={
                    "Content-Type": "application/json",
                    "api-key": self._azure_api_key,
                })
                urllib.request.urlopen(req, timeout=10)
                logger.info("CodexCliBackend: Azure warmup complete")
            except Exception as e:
                logger.debug(f"CodexCliBackend: Azure warmup failed (non-fatal): {e}")

        threading.Thread(target=_do_warmup, daemon=True).start()

    def complete(self, system_prompt: str, user_message: str) -> str:
        """Get completion via Azure Responses API (or codex exec fallback)."""
        if self._use_direct_api:
            return self._complete_azure(system_prompt, user_message)
        return self._complete_cli(system_prompt, user_message)

    def _complete_azure(self, system_prompt: str, user_message: str) -> str:
        """Call Azure OpenAI Responses API directly (~2-3s latency)."""
        import urllib.request

        url = (
            f"{self._azure_base_url}/responses"
            f"?api-version={self._azure_api_version}"
        )

        body = json.dumps({
            "model": self.model or "gpt-5.4",
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_output_tokens": 400,
            "reasoning": {"effort": "medium"},
        }).encode()

        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json",
            "api-key": self._azure_api_key,
        })

        if self.progress_callback:
            self.progress_callback("Thinking (codex)...")

        try:
            start = time.perf_counter()
            resp = urllib.request.urlopen(req, timeout=self.timeout_s)
            data = json.loads(resp.read())
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"Azure Responses API call took {elapsed:.0f}ms")

            # Extract text from Responses API format
            text = ""
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if c.get("type") == "output_text":
                            text += c["text"]

            if self.progress_callback:
                self.progress_callback("")
            return text.strip() or "Error: Empty response from Azure API"
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode()[:200]
            except Exception:
                pass
            logger.error(f"Azure API error {e.code}: {err_body}")
            if self.progress_callback:
                self.progress_callback("")
            return f"Error: Azure API {e.code}: {err_body}"
        except Exception as e:
            logger.error(f"Azure API call failed: {e}")
            if self.progress_callback:
                self.progress_callback("")
            return f"Error: Azure API failed: {e}"

    def _complete_cli(self, system_prompt: str, user_message: str) -> str:
        """Fallback: Get completion via codex exec subprocess."""
        combined = (
            f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\nUSER MESSAGE:\n{user_message}"
        )
        args = self._build_args() + ["exec"]
        if self.model:
            args += ["--model", self.model]

        if self.progress_callback:
            self.progress_callback("Thinking (codex)...")

        env = {k: v for k, v in os.environ.items()
               if k not in ("OPENAI_API_KEY",)}

        try:
            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NO_WINDOW

            result = subprocess.run(
                args + ["-", "--ephemeral", "--skip-git-repo-check"],
                input=combined,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout_s,
                creationflags=creationflags,
                env=env,
            )
        except FileNotFoundError:
            if self.progress_callback:
                self.progress_callback("")
            return "Error: Codex CLI not found."
        except subprocess.TimeoutExpired:
            if self.progress_callback:
                self.progress_callback("")
            return f"Error: Codex CLI timed out after {self.timeout_s}s"
        except Exception as e:
            if self.progress_callback:
                self.progress_callback("")
            return f"Error running Codex CLI: {e}"

        if self.progress_callback:
            self.progress_callback("")

        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            return f"Error: Codex CLI failed: {err}"

        return (result.stdout or "").strip()

    def _build_args(self) -> list[str]:
        import shutil

        cmd = self.command
        if cmd and os.path.isabs(cmd):
            resolved = cmd
        else:
            resolved = shutil.which(cmd) or ""

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

    def list_models(self) -> list[str]:
        return []

    def close(self) -> None:
        pass
