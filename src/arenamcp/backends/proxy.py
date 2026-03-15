"""Proxy / OpenAI-compatible API backend for LLM coaching."""

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class ProxyBackend:
    """LLM backend using CLI Proxy API (OpenAI-compatible endpoint).

    Routes requests through a local cli-proxy-api server that load-balances
    across multiple OAuth providers (Antigravity, Claude, Codex, etc.).
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        enable_thinking: bool = False,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize Proxy backend.

        Args:
            model: Model name as exposed by the proxy (default: claude-sonnet-4-5-20250929)
            enable_thinking: If True, enable extended thinking for models that support it.
                             Used by background win-plan workers for deeper analysis.
            base_url: Override the proxy endpoint URL. Falls back to PROXY_BASE_URL
                     env var, then settings, then http://127.0.0.1:8080/v1.
            api_key: Override the API key. Falls back to PROXY_API_KEY env var,
                    then settings, then default placeholder.
        """
        self.model = model or "claude-sonnet-4-5-20250929"
        self.enable_thinking = enable_thinking
        self._base_url = base_url
        self._api_key = api_key
        self._client = None

        # Fire-and-forget warmup for Ollama to pre-load the model
        self._ollama_warmup(base_url, api_key)

    def _get_client(self):
        """Lazy init of OpenAI client pointed at proxy."""
        if self._client is None:
            try:
                from openai import OpenAI

                # Resolve URL: explicit param > env var > settings > default
                if self._base_url:
                    url = self._base_url
                elif os.environ.get("PROXY_BASE_URL"):
                    url = os.environ["PROXY_BASE_URL"]
                else:
                    try:
                        from arenamcp.settings import get_settings
                        url = get_settings().get("proxy_url") or "http://127.0.0.1:8080/v1"
                    except Exception:
                        url = "http://127.0.0.1:8080/v1"

                # Resolve key: explicit param > env var > settings > default
                if self._api_key:
                    key = self._api_key
                elif os.environ.get("PROXY_API_KEY"):
                    key = os.environ["PROXY_API_KEY"]
                else:
                    try:
                        from arenamcp.settings import get_settings
                        key = get_settings().get("proxy_api_key") or "your-api-key-1"
                    except Exception:
                        key = "your-api-key-1"

                self._client = OpenAI(base_url=url, api_key=key)
            except ImportError:
                raise ImportError("openai package required: pip install openai")
        return self._client

    def _ollama_warmup(self, base_url: Optional[str], api_key: Optional[str]) -> None:
        """Send a minimal warmup request to Ollama in a background thread to pre-load the model."""
        # Determine the effective URL/key to check if this is an Ollama endpoint
        url = base_url or os.environ.get("PROXY_BASE_URL", "")
        key = api_key or os.environ.get("PROXY_API_KEY", "")
        if not url:
            try:
                from arenamcp.settings import get_settings
                url = get_settings().get("proxy_url") or ""
            except Exception:
                pass
        if not key:
            try:
                from arenamcp.settings import get_settings
                key = get_settings().get("proxy_api_key") or ""
            except Exception:
                pass

        is_ollama = ("localhost:11434" in url or "127.0.0.1:11434" in url or
                     key == "ollama")
        if not is_ollama:
            return

        def _warmup():
            try:
                client = self._get_client()
                client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=1,
                )
                logger.info(f"[PROXY] Ollama warmup complete for model {self.model}")
            except Exception as e:
                logger.debug(f"[PROXY] Ollama warmup failed (non-fatal): {e}")

        t = threading.Thread(target=_warmup, daemon=True)
        t.start()

    def complete(self, system_prompt: str, user_message: str, max_tokens: int = 400) -> str:
        """Get completion from proxy."""
        import time

        try:
            client = self._get_client()

            # Build request params optimized for low-latency real-time advice
            params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "max_completion_tokens": max_tokens,
                "temperature": 0.3,
            }

            # Configure thinking/reasoning based on enable_thinking flag.
            # NOTE: Must use extra_body for non-standard params (OpenAI SDK rejects unknown kwargs)
            model_lower = self.model.lower()
            extra = {}
            if self.enable_thinking:
                # Enable extended thinking for deeper analysis (win plans)
                if "claude" in model_lower:
                    extra["thinking"] = {"type": "enabled", "budget_tokens": 8000}
                    params["max_completion_tokens"] = max_tokens + 8000
                elif "gemini" in model_lower:
                    extra["thinking_config"] = {"thinking_budget": 4096}
                # Other models: no thinking config needed
            else:
                # Disable thinking for low-latency real-time advice
                if "claude" in model_lower:
                    extra["thinking"] = {"type": "disabled"}
                if "gemini" in model_lower:
                    extra["thinking_config"] = {"thinking_budget": 0}
            if extra:
                params["extra_body"] = extra

            request_start = time.perf_counter()

            # Try streaming first for lower perceived latency
            try:
                stream = client.chat.completions.create(**params, stream=True)
                chunks: list[str] = []
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        chunks.append(chunk.choices[0].delta.content)
                content = "".join(chunks)
                request_time = (time.perf_counter() - request_start) * 1000
                logger.info(
                    f"[PROXY] API (streamed): {request_time:.0f}ms, model: {self.model}"
                )
                return content
            except Exception as stream_err:
                logger.debug(f"[PROXY] Streaming failed, falling back to non-streaming: {stream_err}")

            # Fallback: non-streaming request
            request_start = time.perf_counter()
            response = client.chat.completions.create(**params)
            request_time = (time.perf_counter() - request_start) * 1000

            content = response.choices[0].message.content
            usage = getattr(response, 'usage', None)
            tokens_info = ""
            if usage:
                tokens_info = f", in={usage.prompt_tokens}, out={usage.completion_tokens}"
            logger.info(
                f"[PROXY] API: {request_time:.0f}ms, model: {self.model}{tokens_info}"
            )
            return content
        except Exception as e:
            logger.error(f"Proxy API error: {e}")
            return f"Error getting advice from proxy: {e}"

    def complete_with_image(self, system_prompt: str, user_message: str, image_bytes: bytes) -> str:
        """Get completion with an image via the OpenAI multimodal message format."""
        import base64
        import time

        try:
            client = self._get_client()
            b64 = base64.b64encode(image_bytes).decode("utf-8")

            params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": user_message},
                    ]},
                ],
                "max_completion_tokens": 600,
                "temperature": 0.3,
            }

            model_lower = self.model.lower()
            extra = {}
            if "claude" in model_lower:
                extra["thinking"] = {"type": "disabled"}
            if "gemini" in model_lower:
                extra["thinking_config"] = {"thinking_budget": 0}
            if extra:
                params["extra_body"] = extra

            request_start = time.perf_counter()
            response = client.chat.completions.create(**params)
            request_time = (time.perf_counter() - request_start) * 1000

            content = response.choices[0].message.content
            logger.info(f"[PROXY] Vision API: {request_time:.0f}ms, model: {self.model}")
            return content
        except Exception as e:
            logger.error(f"Proxy vision API error: {e}")
            return f"Error getting vision analysis from proxy: {e}"

    def list_models(self) -> list[str]:
        """List available models from the proxy."""
        try:
            client = self._get_client()
            models = client.models.list()
            return [m.id for m in models.data]
        except Exception as e:
            logger.error(f"Failed to list proxy models: {e}")
            return []
