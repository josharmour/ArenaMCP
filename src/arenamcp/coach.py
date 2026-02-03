"""Coach engine with pluggable LLM backends for MTG game coaching.

This module provides the CoachEngine for getting strategic advice from LLMs,
with support for Claude (Anthropic), Gemini (Google), and local models via Ollama.
"""

import logging
import os
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


# LLM Backend Protocol and Implementations

class LLMBackend(Protocol):
    """Protocol for LLM backends that can provide coaching advice."""

    def complete(self, system_prompt: str, user_message: str) -> str:
        """Get a completion from the LLM."""
        ...
        
    def list_models(self) -> list[str]:
        """List available models (optional)."""
        return []


class ClaudeBackend:
    """LLM backend using Anthropic's Claude API."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        """Initialize Claude backend with lazy client creation.

        Args:
            model: The Claude model to use (default: claude-sonnet-4-20250514)
        """
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")
        return self._client

    def complete(self, system_prompt: str, user_message: str) -> str:
        """Get completion from Claude API."""
        import time

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return "Error: ANTHROPIC_API_KEY environment variable not set"

        try:
            client_start = time.perf_counter()
            client = self._get_client()
            client_time = (time.perf_counter() - client_start) * 1000

            request_start = time.perf_counter()
            response = client.messages.create(
                model=self.model,
                max_tokens=500,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}]
            )
            request_time = (time.perf_counter() - request_start) * 1000

            logger.debug(f"[CLAUDE] client init: {client_time:.1f}ms, API request: {request_time:.0f}ms, model: {self.model}")
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return f"Error getting advice from Claude: {e}"


class GeminiBackend:
    """LLM backend using Google's Gemini API (new google.genai SDK)."""

    def __init__(self, model: str = "gemini-2.0-flash-lite"):
        """Initialize Gemini backend with lazy client creation.

        Args:
            model: The Gemini model to use (default: gemini-2.0-flash-lite)
        """
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy initialization of Gemini client."""
        if self._client is None:
            try:
                from google import genai
                api_key = os.environ.get("GOOGLE_API_KEY")
                self._client = genai.Client(api_key=api_key)
            except ImportError:
                raise ImportError("google-genai package required: pip install google-genai")
        return self._client

    def complete(self, system_prompt: str, user_message: str) -> str:
        """Get completion from Gemini API."""
        import time

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return "Error: GOOGLE_API_KEY environment variable not set"

        try:
            from google import genai
            from google.genai import types

            client_start = time.perf_counter()
            client = self._get_client()
            client_time = (time.perf_counter() - client_start) * 1000

            request_start = time.perf_counter()
            response = client.models.generate_content(
                model=self.model,
                contents=user_message,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=1000,
                    safety_settings=[
                        types.SafetySetting(
                            category="HARM_CATEGORY_HARASSMENT",
                            threshold="BLOCK_NONE",
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_HATE_SPEECH",
                            threshold="BLOCK_NONE",
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            threshold="BLOCK_NONE",
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_DANGEROUS_CONTENT",
                            threshold="BLOCK_NONE",
                        ),
                    ],
                    # Disable thinking for now as it causes massive verbosity leakage in standard Flash models
                    # thinking_config=types.ThinkingConfig(include_thoughts=True) if "thinking" in self.model else None
                )
            )
            request_time = (time.perf_counter() - request_start) * 1000
            
            # Log finish reason and safety ratings for debugging truncation
            finish_reason = "UNKNOWN"
            safety_ratings = "UNKNOWN"
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                finish_reason = getattr(candidate, 'finish_reason', 'UNKNOWN')
                safety_ratings = getattr(candidate, 'safety_ratings', 'UNKNOWN')
            
            # CHANGED TO INFO so it appears in standard logs
            logger.info(f"[GEMINI] client: {client_time:.1f}ms, API: {request_time:.0f}ms, model: {self.model}, reason: {finish_reason}, safety: {safety_ratings}")
            
            # Extract text, handling potential thought blocks if present
            text = ""
            if hasattr(response, 'candidates') and response.candidates:
                # Iterate through parts to find the actual text response, ignoring "thought" parts if distinct
                # Note: The google-genai SDK maps parts. If there are thoughts, they might be in a separate part.
                for part in response.candidates[0].content.parts:
                    # In some versions, thoughts are just text parts. In others, they might be distinct.
                    # We'll just grab everything for now, but if thinking is enabled, we might need to parse.
                    # Assuming .text returns the concatenated text which is what we usually want.
                    if hasattr(part, 'text') and part.text:
                         text += part.text
            
            if not text:
                 text = response.text # Fallback
                 
            if not text or text.strip() == "...":
                logger.warning(f"Empty/Ellipsis response from Gemini. Reason: {finish_reason}")
                return "I didn't catch that. Please try again."

            return text
        except ImportError as e:
            return f"Error: {e}"
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"Error getting advice from Gemini: {e}"

    def complete_with_audio(
        self,
        system_prompt: str,
        user_message: str,
        audio_data: "np.ndarray",
        sample_rate: int = 16000
    ) -> str:
        """Get completion from Gemini API with audio input.

        Args:
            system_prompt: The system prompt setting up the assistant role
            user_message: The text context/question
            audio_data: Raw audio as numpy float32 array
            sample_rate: Audio sample rate (default: 16000)

        Returns:
            The LLM's response text, or error message string on failure
        """
        import io
        import time
        import wave
        import numpy as np

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return "Error: GOOGLE_API_KEY environment variable not set"

        try:
            from google import genai
            from google.genai import types

            # Convert numpy audio to WAV bytes
            wav_start = time.perf_counter()
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            wav_bytes = wav_buffer.getvalue()
            wav_time = (time.perf_counter() - wav_start) * 1000

            client_start = time.perf_counter()
            client = self._get_client()
            client_time = (time.perf_counter() - client_start) * 1000

            # Create audio part
            audio_part = types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")

            request_start = time.perf_counter()
            response = client.models.generate_content(
                model=self.model,
                contents=[user_message, audio_part],
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=1000,
                    safety_settings=[
                        types.SafetySetting(
                            category="HARM_CATEGORY_HARASSMENT",
                            threshold="BLOCK_NONE",
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_HATE_SPEECH",
                            threshold="BLOCK_NONE",
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            threshold="BLOCK_NONE",
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_DANGEROUS_CONTENT",
                            threshold="BLOCK_NONE",
                        ),
                    ]
                )
            )
            request_time = (time.perf_counter() - request_start) * 1000
            
            # Log finish reason
            finish_reason = "UNKNOWN"
            safety_ratings = "UNKNOWN"
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                finish_reason = getattr(candidate, 'finish_reason', 'UNKNOWN')
                safety_ratings = getattr(candidate, 'safety_ratings', 'UNKNOWN')

            logger.info(f"[GEMINI+AUDIO] wav: {wav_time:.1f}ms, client: {client_time:.1f}ms, API: {request_time:.0f}ms, reason: {finish_reason}, safety: {safety_ratings}")
            
            text = response.text
            if not text or text.strip() == "...":
                 logger.warning(f"Empty/Ellipsis response from Gemini. Reason: {finish_reason}")
                 return "I didn't catch that."
                 
            return text
        except ImportError as e:
            return f"Error: {e}"
        except Exception as e:
            logger.error(f"Gemini audio API error: {e}")
            return f"Error getting advice from Gemini with audio: {e}"

    def complete_with_image(
        self,
        system_prompt: str,
        user_message: str,
        image_data: bytes,
        mime_type: str = "image/png"
    ) -> str:
        """Get completion from Gemini API with image input."""
        import time
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return "Error: GOOGLE_API_KEY environment variable not set"

        try:
            client = self._get_client()
            
            # Create image part
            image_part = types.Part.from_bytes(data=image_data, mime_type=mime_type)

            response = client.models.generate_content(
                model=self.model,
                contents=[user_message, image_part],
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=1000,
                    safety_settings=[
                         types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                         types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                         types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                         types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                    ]
                )
            )

            text = response.text or ""
            return text
        except Exception as e:
            logger.error(f"Gemini Image API error: {e}")
            return f"Error analyzing image: {e}"


class OllamaBackend:
    """LLM backend using local Ollama server."""

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        """Initialize Ollama backend.

        Args:
            model: The Ollama model to use (default: llama3.2)
            base_url: Ollama server URL (default: localhost:11434)
        """
        self.model = model
        self.base_url = base_url

    def complete(self, system_prompt: str, user_message: str) -> str:
        """Get completion from local Ollama server."""
        import requests

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": user_message,
                    "system": system_prompt,
                    "stream": False,
                },
                timeout=60,
            )

            if response.status_code == 404:
                return f"Error: Model '{self.model}' not found. Run: ollama pull {self.model}"

            response.raise_for_status()
            return response.json().get("response", "No response from Ollama")

        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Is it running? Start with: ollama serve"
        except requests.exceptions.Timeout:
            return "Error: Ollama request timed out"
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return f"Error getting advice from Ollama: {e}"


class AzureBackend:
    """LLM backend using Azure OpenAI."""

    def __init__(self, model: str = "gpt-4o"):
        """Initialize Azure backend.

        Args:
           model: Deployment name (default: gpt-4o)
        """
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy init of Azure client."""
        if self._client is None:
            try:
                from openai import AzureOpenAI
                self._client = AzureOpenAI(
                    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
                    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
                )
            except ImportError:
                raise ImportError("openai package required: pip install openai")
        return self._client

    def complete(self, system_prompt: str, user_message: str) -> str:
        """Get completion from Azure OpenAI."""
        import time

        if not os.environ.get("AZURE_OPENAI_API_KEY"):
            return "Error: AZURE_OPENAI_API_KEY not set"

        try:
            client = self._get_client()

            client_start = time.perf_counter()
            response = client.chat.completions.create(
                model=self.model, # This is the deployment name
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=800
            )
            request_time = (time.perf_counter() - client_start) * 1000

            content = response.choices[0].message.content
            logger.info(f"[AZURE] API: {request_time:.0f}ms, model: {self.model}")

            return content
            return content
        except Exception as e:
            # Try to handle "DeploymentNotFound" gracefully
            err_str = str(e)
            is_404 = "404" in err_str or "DeploymentNotFound" in err_str

            if is_404:
                # Try to list available models to help the user
                try:
                    available = self.list_models()
                    help_msg = f"\n\nAvailable models on your Azure endpoint:\n" + "\n".join([f"- {m}" for m in available[:10]])
                    if len(available) > 10:
                        help_msg += f"\n...and {len(available)-10} more."
                except Exception:
                    help_msg = ""

                return f"Error: Deployment '{self.model}' not found.{help_msg}"

            logger.error(f"Azure API error: {e}")
            return f"Error getting advice from Azure: {e}"

    def list_models(self) -> list[str]:
        """List available models (deployments) from Azure endpoint."""
        try:
            client = self._get_client()
            # client.models.list() gives base models.
            # We want deployments. The SDK doesn't support listing deployments directly easily.
            # We will try to filter models that look like deployments or return base models
            # and let the user try them.
            models = client.models.list()
            return [m.id for m in models.data]
        except Exception as e:
            logger.error(f"Failed to list Azure models: {e}")
            # Fallback to standard known models
            return ["gpt-4o", "gpt-4-turbo", "gpt-35-turbo"]


class GPTRealtimeBackendWrapper:
    """LLM backend wrapper for GPT-Realtime (voice-to-voice).

    This provides the standard LLMBackend interface for GPT-Realtime,
    enabling it to be used as a drop-in replacement for other backends.
    """

    def __init__(self, model: str = "gpt-realtime", voice: str = "alloy"):
        """Initialize GPT-Realtime backend.

        Args:
            model: Deployment name (default: gpt-realtime)
            voice: Voice for TTS output (default: alloy)
        """
        self.model = model
        self._voice = voice
        self._backend = None

    def _get_backend(self):
        """Lazy init of realtime backend."""
        if self._backend is None:
            try:
                from arenamcp.realtime import GPTRealtimeBackend
                self._backend = GPTRealtimeBackend(
                    model=self.model,
                    voice=self._voice
                )
            except ImportError as e:
                raise ImportError(f"GPT-Realtime dependencies required: {e}")
        return self._backend

    def complete(self, system_prompt: str, user_message: str) -> str:
        """Get completion from GPT-Realtime (text mode)."""
        backend = self._get_backend()
        return backend.complete(system_prompt, user_message)

    def complete_with_audio(
        self,
        system_prompt: str,
        user_message: str,
        audio_data: "np.ndarray",
        sample_rate: int = 16000
    ) -> str:
        """Get completion from GPT-Realtime with audio input.

        Args:
            system_prompt: System instructions
            user_message: Text context
            audio_data: Audio as numpy float32 array
            sample_rate: Audio sample rate

        Returns:
            Response text
        """
        backend = self._get_backend()
        return backend.complete_with_audio(system_prompt, user_message, audio_data, sample_rate)

    def get_last_audio_response(self) -> Optional[bytes]:
        """Get audio from last response (PCM16 at 24kHz)."""
        if self._backend:
            return self._backend.get_last_audio_response()
        return None

    def disconnect(self) -> None:
        """Close the realtime connection."""
        if self._backend:
            self._backend.disconnect()


def create_backend(backend_type: str, model: Optional[str] = None) -> LLMBackend:
    """Factory function to create LLM backends by name.

    Args:
        backend_type: One of "claude", "gemini", "ollama", "azure", "gpt-realtime"
        model: Optional model override (uses backend default if not specified)

    Returns:
        Configured LLMBackend instance

    Raises:
        ValueError: If backend_type is not recognized
    """
    backend_type = backend_type.lower()

    if backend_type == "claude":
        return ClaudeBackend(model=model) if model else ClaudeBackend()
    elif backend_type == "gemini":
        return GeminiBackend(model=model) if model else GeminiBackend()
    elif backend_type == "ollama":
        return OllamaBackend(model=model) if model else OllamaBackend()
    elif backend_type == "azure":
        return AzureBackend(model=model) if model else AzureBackend()
    elif backend_type in ("gpt-realtime", "realtime"):
        return GPTRealtimeBackendWrapper(model=model) if model else GPTRealtimeBackendWrapper()
    elif backend_type == "gemini-live":
        # For text-based coaching/quick advice while in Live mode,
        # use the standard Gemini REST backend.
        # The streaming Live client is handled separately in standalone.py for voice.
        # Fallback to a standard model if the live model isn't supported for text generation
        text_model = model
        if "live" in (text_model or ""):
             text_model = "gemini-2.0-flash-lite" # Live models might not support generateContent text-only? 
             # Actually gemini-2.0-flash-exp supports both. 
             # But safely default to standard flash for text advice.
             text_model = "gemini-2.0-flash" 
        
        return GeminiBackend(model=text_model)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}. Use 'claude', 'gemini', 'ollama', 'azure', or 'gpt-realtime'")


# Default MTG coach system prompt
DEFAULT_SYSTEM_PROMPT = """You are an expert MTG coach providing real-time advice during Arena games.

Keep responses concise (2-3 sentences max) since they'll be spoken aloud.
Focus ONLY on the final strategic recommendation.
Do NOT show your thinking process, "reasoning", or "corrections".
Do NOT use internal monologue tags like [plan] or [thought].
Do NOT second-guess yourself in the text (e.g., "Wait, I need to check...").
Be authoritative and decisive. Start your response immediately with the command.

CRITICAL GAME RULES:
- DEFAULT: You can only play ONE LAND per turn unless a card grants additional land drops.
- Check the LAND DROP status to see if a land can still be played this turn.
- Cards marked [INSTANT] can be cast anytime you have priority
- Cards marked [SORCERY SPEED] can ONLY be cast during YOUR Main phase with empty stack
- During opponent's turn or combat: ONLY suggest instants or abilities
- If it's not your Main phase, do NOT suggest casting creatures or sorceries

CRITICAL MANA RULES:
- Trust the [CAN CAST] and [NEED X MORE] tags implicitly.
- Do NOT perform your own mana math checks in the text.
- If a card says [CAN CAST], you can recommend it.
- If no castable plays exist, say "pass" or suggest holding mana.

CRITICAL MATH RULES:
- When suggesting removal, check the creature's TOUGHNESS (second number, e.g., 4/5 has 5 toughness).
- -2/-2 or 2 damage ONLY kills toughness 2 or less (unless damaged).
- Do NOT suggest removal that won't kill the target unless it enables a profitable attack.

Analyze: phase (critical for timing!), board state, life totals, cards in hand, mana available.
Output directly as the coach. No preamble, no meta-commentary."""

CONCISE_SYSTEM_PROMPT = """You are an expert MTG coach.
Your goal is to give a concise, sequential plan for the entire turn.
Chain your actions clearly. Cover Main 1, Combat, and Main 2 if relevant.

Examples:
"Play Forest. Cast Llanowar Elves. Pass."
"Attack with all. If they block, cast Giant Growth."
"Main 1: Cast Removal on their flyer. Combat: Swing with 5/5."
"No attacks. Hold mana for Counterspell."

Style: Military/Pro player. Imperative. No fluff.
Do NOT explain "why". Just say "what".
Keep it under 30 words total.
"""


# Words that tend to be overused by LLMs in coaching contexts
OVERUSE_CANDIDATES = {
    "consider", "considering", "important", "crucial", "critical",
    "definitely", "absolutely", "certainly", "essentially", "basically",
    "potentially", "priority", "prioritize", "focus", "key",
}

# Threshold for blacklisting (uses in window)
OVERUSE_THRESHOLD = 3
OVERUSE_WINDOW_SECONDS = 120


class WordUsageTracker:
    """Tracks word usage over time to detect overused words."""

    def __init__(self, threshold: int = OVERUSE_THRESHOLD, window_seconds: float = OVERUSE_WINDOW_SECONDS):
        self._threshold = threshold
        self._window = window_seconds
        self._usage: list[tuple[float, str]] = []  # (timestamp, word)

    def record(self, text: str, exclude_words: Optional[set[str]] = None) -> None:
        """Record words from a response.

        Args:
            text: The response text to analyze
            exclude_words: Set of words to ignore (e.g., card names)
        """
        import time
        import re
        now = time.time()

        exclude = exclude_words or set()

        # Extract words, lowercase
        words = re.findall(r'\b[a-z]+\b', text.lower())

        # Only track candidate words that aren't excluded
        for word in words:
            if word in OVERUSE_CANDIDATES and word not in exclude:
                self._usage.append((now, word))

        # Prune old entries
        cutoff = now - self._window
        self._usage = [(t, w) for t, w in self._usage if t > cutoff]

    def get_blacklisted(self, exclude_words: Optional[set[str]] = None) -> list[str]:
        """Get words that have been overused in the current window.

        Args:
            exclude_words: Set of words to never blacklist (e.g., card names)
        """
        import time
        from collections import Counter

        exclude = exclude_words or set()
        now = time.time()
        cutoff = now - self._window

        # Count words in window
        recent_words = [w for t, w in self._usage if t > cutoff]
        counts = Counter(recent_words)

        # Return words over threshold, excluding protected words
        return [word for word, count in counts.items()
                if count >= self._threshold and word not in exclude]


class CoachEngine:
    """Engine for getting MTG coaching advice from an LLM backend."""

    def __init__(
        self,
        backend: Optional[LLMBackend] = None,
        system_prompt: Optional[str] = None
    ):
        """Initialize the coach engine.

        Args:
            backend: LLM backend to use (default: ClaudeBackend)
            system_prompt: Custom system prompt (default: MTG coach persona)
        """
        self._backend = backend if backend is not None else ClaudeBackend()
        self._system_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        self._word_tracker = WordUsageTracker()


    def _remove_reminder_text(self, text: str) -> str:
        """Remove reminder text (text in parentheses) from oracle text."""
        import re
        # Handle nested parens if possible, but simple greedy match usually works for MTG
        # Use simple non-greedy match for multiple parens
        return re.sub(r'\(.*?\)', '', text)

    def _format_game_context(self, game_state: dict[str, Any], question: str = "") -> str:
        """Format the game state into a concise context for the LLM."""
        
        # Determine local player seat and active turn
        players = game_state.get("players", [])
        local_player = next((p for p in players if p.get("is_local")), None)
        local_seat = local_player.get("seat_id") if local_player else 1
        
        turn = game_state.get("turn", {})
        active_seat = turn.get("active_player", 0)
        priority_seat = turn.get("priority_player", 0)
        is_my_turn = (active_seat == local_seat)
        has_priority = (priority_seat == local_seat)
        
        phase = turn.get("phase", "Unknown")
        turn_num = turn.get("turn_number", 0)

        # Import RulesEngine here to avoid circular imports given the flat structure
        try:
            from arenamcp.rules_engine import RulesEngine
            valid_moves = RulesEngine.get_legal_actions(game_state)
            valid_moves_str = "\n".join([f"- {m}" for m in valid_moves])
        except Exception as e:
            # Use global logger
            logger.error(f"RulesEngine error: {e}")
            valid_moves_str = "Error calculating legal moves."

        lines = []
        lines.append("=== GAME STATE ===")
        lines.append("VALID MOVES:")
        lines.append(valid_moves_str)
        lines.append("") # Add a blank line for separation

        # Get player info and identify local player
        players = game_state.get("players", [])
        local_player = None
        opponent_player = None
        for p in players:
            if p.get("is_local"):
                local_player = p
            else:
                opponent_player = p

        local_seat = local_player.get("seat_id") if local_player else None
        opp_seat = opponent_player.get("seat_id") if opponent_player else None

        # EXPLICIT seat identification at the top
        lines.append("=== GAME STATE ===")
        if local_seat is not None:
            lines.append(f"YOU ARE SEAT {local_seat}")
        else:
            lines.append("WARNING: Local player seat unknown")

        # Turn info with explicit seat references
        turn = game_state.get("turn", {})
        turn_num = turn.get("turn_number", "?")
        phase = turn.get("phase", "").replace("Phase_", "")
        step = turn.get("step", "").replace("Step_", "")
        active = turn.get("active_player", 0)
        priority = turn.get("priority_player", 0)

        active_label = "YOUR" if active == local_seat else "OPPONENT'S"
        priority_label = "You" if priority == local_seat else "Opponent"

        # Determine if sorcery-speed spells can be cast
        # Only during your own main phase (Main1 or Main2) with priority
        is_main_phase = "Main" in phase
        is_your_turn = active == local_seat
        stack = game_state.get("stack", [])
        stack_empty = len(stack) == 0
        can_cast_sorcery = is_your_turn and is_main_phase and stack_empty and priority == local_seat

        # Decision Check
        pending_decision = game_state.get("pending_decision")
        if pending_decision:
             lines.append(f"!!! GAME INTERFACE PROMPT: {pending_decision} !!!")
             # Special instruction for mulligans
             if pending_decision == "Mulligan":
                 lines.append("ACTION: The user must decide whether to KEEP or MULLIGAN.")
                 lines.append("Evaluate the hand based on: land count, color sources matching spells, curve, and synergy.")
                 if turn_num <= 1:
                     lines.append("Recommend 'KEEP' or 'MULLIGAN' explicitly.")

        lines.append(f"Turn {turn_num} - {active_label} TURN")
        lines.append(f"Phase: {phase}" + (f" ({step})" if step else ""))
        lines.append(f"Priority: {priority_label}")

        # CRITICAL timing indicator with specific reason
        is_blocking = step == "DeclareBlock" and active != local_seat

        if can_cast_sorcery:
            lines.append(">>> CAN CAST: Creatures, Sorceries, Instants <<<")
        elif is_blocking:
            lines.append(">>> ACTION: MUST BLOCK (or Pass) - Choose blockers now! <<<")
        else:
            # Be specific about WHY sorceries can't be cast
            if not is_your_turn:
                reason = "opponent's turn"
            elif not is_main_phase:
                reason = "not main phase"
            elif not stack_empty:
                reason = "stack not empty"
            elif priority != local_seat:
                reason = "opponent has priority"
            else:
                reason = "unknown"
            lines.append(f">>> CAN CAST: ONLY Instants ({reason}) <<<")

        # Life totals with seat IDs
        lines.append("")
        lines.append("LIFE TOTALS:")
        if local_player:
            lines.append(f"  You (Seat {local_seat}): {local_player.get('life_total', '?')} life")
        if opponent_player:
            lines.append(f"  Opponent (Seat {opp_seat}): {opponent_player.get('life_total', '?')} life")

        # Battlefield - grouped by owner with explicit seat labels
        battlefield = game_state.get("battlefield", [])
        your_cards = [c for c in battlefield if c.get("owner_seat_id") == local_seat]
        opp_cards = [c for c in battlefield if c.get("owner_seat_id") != local_seat]

        # Count available mana by color
        mana_pool = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0, "C": 0, "Any": 0}
        total_mana = 0
        
        for card in your_cards:
            type_line = card.get("type_line", "").lower()
            if "land" in type_line and not card.get("is_tapped"):
                total_mana += 1
                name = card.get("name", "")
                oracle = card.get("oracle_text", "")
                
                # Simple heuristic for basic lands and common duals
                produced = False
                if "Plains" in name or "{W}" in oracle:
                    mana_pool["W"] += 1
                    produced = True
                if "Island" in name or "{U}" in oracle:
                    mana_pool["U"] += 1
                    produced = True
                if "Swamp" in name or "{B}" in oracle:
                    mana_pool["B"] += 1
                    produced = True
                if "Mountain" in name or "{R}" in oracle:
                    mana_pool["R"] += 1
                    produced = True
                if "Forest" in name or "{G}" in oracle:
                    mana_pool["G"] += 1
                    produced = True
                if "{C}" in oracle:
                    mana_pool["C"] += 1
                    produced = True
                
                # "Add one mana of any color"
                if "any color" in oracle.lower():
                    mana_pool["Any"] += 1
                    produced = True
                
                # If we couldn't identify color (e.g. unknown land), assume colorless for safety
                # or maybe Any if we want to be generous? Let's treat valid land as generic source at least.
                pass 

        logger.info(f"Estimated Mana: {mana_pool} (Total: {total_mana})")
        logger.info(f"Estimated Mana: {mana_pool} (Total: {total_mana})")
        lines.append(f"YOUR MANA: {total_mana} available (W:{mana_pool['W']} U:{mana_pool['U']} B:{mana_pool['B']} R:{mana_pool['R']} G:{mana_pool['G']} Any:{mana_pool['Any']})")

        # Lands played status - make it VERY clear if a land can be played
        lands_played = local_player.get("lands_played", 0) if local_player else 0
        if lands_played == 0:
            lines.append("LAND DROP: AVAILABLE (you can play 1 land this turn)")
        else:
            lines.append(f"LAND DROP: USED (already played {lands_played} land this turn - CANNOT play more)")

        if battlefield:
            # ... (Existing battlefield rendering code) ...
            lines.append("")
            lines.append(f"YOUR BATTLEFIELD (Seat {local_seat}):")
            if your_cards:
                 for card in your_cards:
                    name = card.get("name", "Unknown")
                    pt = ""
                    if card.get("power") is not None:
                        pt = f" ({card['power']}/{card['toughness']})"
                    
                    # Check Summoning Sickness
                    is_tapped = card.get("is_tapped")
                    tapped_str = " [TAPPED]" if is_tapped else ""
                    
                    turn_entered = card.get("turn_entered_battlefield", -1)
                    # Safe integer conversion for turn comparison
                    try:
                        current_turn_int = int(str(turn_num).replace("?","0"))
                    except:
                        current_turn_int = 0
                        
                    oracle_text = self._remove_reminder_text(card.get("oracle_text", "")).lower()
                    has_haste = "haste" in oracle_text
                    is_creature = "creature" in card.get("type_line", "").lower()
                    
                    sick_str = ""
                    if is_creature and turn_entered == current_turn_int and not has_haste:
                        sick_str = " [SUMMONING SICK - CANNOT ATTACK]"
                    elif is_creature and turn_entered == -1 and not has_haste and not is_tapped:
                        # Fallback: If unknown entry time but appears active...
                        # We can't safely assume sickness for -1.
                        pass

                    lines.append(f"  - {name}{pt}{tapped_str}{sick_str}")
                    

            else:
                lines.append("  (empty)")

            lines.append(f"OPPONENT'S BATTLEFIELD (Seat {opp_seat}):")
            if opp_cards:
                for card in opp_cards:
                    name = card.get("name", "Unknown")
                    pt = ""
                    if card.get("power") is not None:
                        pt = f" ({card['power']}/{card['toughness']})"
                    tapped = " [TAPPED]" if card.get("is_tapped") else ""
                    lines.append(f"  - {name}{pt}{tapped}")
            else:
                lines.append("  (empty)")

            # COMBAT ANALYSIS - help LLM understand attacking
            # Only show attack options during YOUR turn
            if ("Combat" in phase or "Main" in phase) and is_your_turn:
                # Calculate your potential attackers (untapped creatures WITHOUT summoning sickness)
                turn_num = turn.get("turn_number", 0)
                
                your_creatures = [c for c in your_cards
                                  if c.get("power") is not None
                                  and "land" not in c.get("type_line", "").lower()]
                
                valid_attackers = []
                for c in your_creatures:
                    if c.get("is_tapped"):
                        continue
                        
                    turn_entered = c.get("turn_entered_battlefield", -1)
                    oracle_text = self._remove_reminder_text(c.get("oracle_text", "")).lower()
                    has_haste = "haste" in oracle_text
                    
                    if (turn_entered == turn_num) and not has_haste:
                        continue # Summoning sick
                        
                    valid_attackers.append(c)
                    
                your_attack_power = sum(c.get("power", 0) for c in valid_attackers)
                your_untapped_creatures = valid_attackers # Update for below usage (count)

                # Calculate opponent's potential blockers (untapped creatures)
                opp_creatures = [c for c in opp_cards
                                 if c.get("power") is not None
                                 and "land" not in c.get("type_line", "").lower()]
                opp_untapped_creatures = [c for c in opp_creatures if not c.get("is_tapped")]
                opp_block_toughness = sum(c.get("toughness", 0) for c in opp_untapped_creatures)
                opp_block_count = len(opp_untapped_creatures)

                opp_life = opponent_player.get("life_total", 20) if opponent_player else 20

                lines.append("")
                lines.append("COMBAT ANALYSIS:")
                if valid_attackers:
                    attacker_names = [c.get("name", "?") for c in valid_attackers]
                    lines.append(f"  CAN ATTACK: {', '.join(attacker_names)} (total power: {your_attack_power})")
                else:
                    # Explain WHY no attackers
                    tapped_creatures = [c for c in your_creatures if c.get("is_tapped")]
                    sick_creatures = [c for c in your_creatures
                                      if not c.get("is_tapped")
                                      and c.get("turn_entered_battlefield", -1) == turn_num
                                      and "haste" not in self._remove_reminder_text(c.get("oracle_text", "")).lower()]
                    if tapped_creatures and not sick_creatures:
                        lines.append(f"  NO ATTACKERS: All {len(tapped_creatures)} creatures are tapped")
                    elif sick_creatures and not tapped_creatures:
                        lines.append(f"  NO ATTACKERS: All {len(sick_creatures)} creatures have summoning sickness")
                    elif tapped_creatures and sick_creatures:
                        lines.append(f"  NO ATTACKERS: {len(tapped_creatures)} tapped, {len(sick_creatures)} summoning sick")
                    else:
                        lines.append(f"  NO ATTACKERS: No creatures on battlefield")
                lines.append(f"  Opponent's untapped blockers: {opp_block_count} (total toughness: {opp_block_toughness})")
                if opp_block_count > 0:
                    lines.append(f"  WARNING: Opponent CAN block! Damage will be reduced or prevented.")
                    # Check if lethal through blockers
                    if your_attack_power > opp_block_toughness + opp_life:
                        lines.append(f"  Lethal possible IF you have trample or evasion.")
                    else:
                        lines.append(f"  NOT lethal if opponent blocks optimally.")
                else:
                    if your_attack_power >= opp_life:
                        lines.append(f"  LETHAL! No blockers, attack for the win!")
                    else:
                        lines.append(f"  No blockers available.")
        else:
            lines.append("")
            lines.append("BATTLEFIELD: Empty")

        # Hand - these are YOUR cards, with castability and timing info
        hand = game_state.get("hand", [])
        lines.append("")
        lines.append(f"YOUR HAND (Seat {local_seat}):")

        # Pre-compute opponent creatures for removal analysis
        opp_creatures = [c for c in opp_cards
                         if c.get("power") is not None
                         and "land" not in c.get("type_line", "").lower()]

        if hand:
            import re
            
            # Pre-calculate potential land mana
            # If we haven't played land, calculate if we have one in hand that enters untapped
            can_play_land = (lands_played == 0) and is_your_turn and is_main_phase and stack_empty
            possible_extra_mana = 0
            possible_extra_colors = set()
            
            if can_play_land:
                for c in hand:
                    c_type = c.get("type_line", "").lower()
                    c_oracle = c.get("oracle_text", "").lower()
                    if "land" in c_type and "enters tapped" not in c_oracle:
                         possible_extra_mana = 1
                         # Heuristic for colors
                         c_name = c.get("name", "")
                         if "Plains" in c_name or "{W}" in c_oracle: possible_extra_colors.add("W")
                         if "Island" in c_name or "{U}" in c_oracle: possible_extra_colors.add("U")
                         if "Swamp" in c_name or "{B}" in c_oracle: possible_extra_colors.add("B")
                         if "Mountain" in c_name or "{R}" in c_oracle: possible_extra_colors.add("R")
                         if "Forest" in c_name or "{G}" in c_oracle: possible_extra_colors.add("G")
                         if "any color" in c_oracle: possible_extra_colors.add("Any")
            
            for card in hand:
                name = card.get("name", "Unknown")
                cost = card.get("mana_cost", "")
                type_line = card.get("type_line", "").lower()
                oracle_text = card.get("oracle_text", "").lower()

                # Determine timing
                is_instant_speed = "instant" in type_line or "flash" in oracle_text
                timing = "[INSTANT]" if is_instant_speed else "[SORCERY SPEED]"

                # Parse Requirements
                reqs = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0, "C": 0}
                cmc = 0
                
                if cost:
                    # Generic
                    generic = re.findall(r'\{(\d+)\}', cost)
                    cmc += sum(int(g) for g in generic)
                    
                    # Colored pips
                    for color in ["W", "U", "B", "R", "G", "C"]:
                        count = len(re.findall(rf"\{{{color}\}}", cost))
                        reqs[color] += count
                        cmc += count
                        
                    # Hybrid (treat as OR? For now, add to CMC but don't enforce specific color strictness to avoid complexity)
                    # Ideally: {R/W} needs R or W.
                    hybrid = re.findall(r'\{([^}]+)/([^}]+)\}', cost)
                    cmc += len(hybrid) 
                    # Note: We are under-constraining hybrid mana here, which is safer than over-constraining.

                # Check Castability
                
                # Logic:
                # 1. Standard check
                # 2. If fail, check with potential land mana
                
                def check_mana(base_pool, base_total, extra_any=0, extra_colors=None):
                    if extra_colors is None: extra_colors = set()
                    
                    # Total check
                    sim_total = base_total + extra_any
                    if sim_total < cmc: return False, ["mana"]
                    
                    # Color check
                    needed_any_generic = 0
                    for color, amount in reqs.items():
                        if amount > 0:
                            have = base_pool.get(color, 0) + (1 if color in extra_colors else 0)
                            # Note: current `base_pool` logic counts 'Any' separately in coach code?
                            # Lines 837 define pool: W,U,B,R,G,C,Any.
                            # Line 1069 checks pool.get(color).
                            
                            # If we have 'First Land' logic, we assume we use that land for ONE color.
                            # But wait, `extra_colors` is a Set of possible colors. 
                            # If we use it for Red, we can't use it for Green.
                            # Simplified: We treat the land as adding 1 to the 'Any' pool 
                            # AND adding 1 to specific color pools for availability check?
                            # No, we can only add 1 total.
                            
                            # Strict check:
                            if have >= amount:
                                continue
                            else:
                                if color in extra_colors:
                                    # Use the potential land for this
                                    # But we can only use it once. 
                                    # This is getting complex.
                                    pass
                                
                                deficit = amount - have
                                needed_any_generic += deficit

                    # Can 'Any' pool cover deficit?
                    # pool['Any'] + 1 (if land is generic source)
                    available_any = base_pool["Any"]
                    if "Any" in extra_colors:
                         # Land produces Any
                         available_any += 1
                    elif extra_colors:
                         # Land produces specific, but if we didn't use it for specific above...
                         # This logic is too complex for inline.
                         # Fallback: Treat potential land as +1 Generic for total, 
                         # and +1 Specific Color if we are missing exactly one pip?
                         pass
                    
                    # Revert to simple check compatible with existing code structure
                    return False, []

                # --- SIMPLIFIED CHECK ---
                
                # 1. Standard Check (Existing logic re-implemented)
                missing_std = []
                needed_any_std = 0
                for color, amount in reqs.items():
                    have = mana_pool.get(color, 0)
                    if have < amount:
                        needed_any_std += (amount - have)
                
                has_total = total_mana >= cmc
                has_colors = needed_any_std <= mana_pool["Any"]
                
                castable = ""
                reasons = []
                
                if has_total and has_colors:
                    castable = "[CAN CAST]"
                else:
                    # Capture reasons for standard failure
                    if not has_total: reasons.append(f"{cmc - total_mana} mana")
                    # Color reasons
                    # logic...
                    reason_str = ", ".join(reasons) if reasons else "colors" 
                    if not reasons: reason_str = "colors" # Fallback
                    
                    castable = f"[NEED {reason_str}]"
                    
                    # 2. Potential Land Check
                    if possible_extra_mana > 0 and not (has_total and has_colors):
                         # Try simulating +1 mana
                         # Assume land can fix our biggest color problem or just add generic
                         
                         # Check Total
                         has_total_with_land = (total_mana + 1) >= cmc
                         
                         # Check Colors
                         # Reducing needed_any_std by 1?
                         # Only if the land provides the needed color or Any
                         
                         covered_color_deficit = False
                         if needed_any_std <= mana_pool["Any"]:
                             # Colors were already fine, just lacked total
                             covered_color_deficit = True
                         else:
                             # We lacked specific colors (covered by Any deficit)
                             # If deficit is 1, and land provides it
                             needed_remaining = needed_any_std - mana_pool["Any"]
                             if needed_remaining <= 1:
                                 # We need 1 more pip.
                                 # Does land provide relevant colors?
                                 # We need to know WHICH colors are missing.
                                 # Iterate again
                                 missing_cols = []
                                 for color, amount in reqs.items():
                                     if mana_pool.get(color,0) < amount:
                                         missing_cols.append(color)
                                 
                                 # If land provides ANY of the missing colors, or Any
                                 if "Any" in possible_extra_colors:
                                     covered_color_deficit = True
                                 else:
                                     # Intersection
                                     if any(c in possible_extra_colors for c in missing_cols):
                                         covered_color_deficit = True
                         
                         if has_total_with_land and covered_color_deficit:
                             castable = "[CAN CAST WITH LAND DROP]"
                
                # Special Override for Lands
                if "land" in type_line:
                    if can_play_land:
                        castable = "[LAND DROP AVAILABLE]"
                    else:
                        castable = "[HOLD]"
                
                if castable == "": # Should not happen if logic is tight
                     castable = f"[NEED {cmc - total_mana} mana]" if total_mana < cmc else "[NEED COLORS]"

                # Removal analysis - pre-calculate what this spell can kill
                removal_info = ""
                oracle_lower = oracle_text.lower()

                # Check for damage-based removal
                damage_match = re.search(r'deals?\s+(\d+)\s+damage\s+to\s+(?:target\s+)?(?:any\s+target|creature|player)', oracle_lower)

                # Check for -X/-X effects
                minus_match = re.search(r'gets?\s+(-\d+)/(-\d+)', oracle_lower)

                # Check for destroy effects
                is_destroy = "destroy target creature" in oracle_lower or "destroy target permanent" in oracle_lower

                # Check for exile effects
                is_exile = "exile target creature" in oracle_lower or "exile target permanent" in oracle_lower

                if damage_match or minus_match or is_destroy or is_exile:
                    kills = []
                    wont_kill = []

                    for creature in opp_creatures:
                        c_name = creature.get("name", "Unknown")
                        c_tough = creature.get("toughness", 0)

                        if is_destroy or is_exile:
                            # Unconditional removal (ignoring indestructible/hexproof for now)
                            kills.append(c_name)
                        elif damage_match:
                            damage = int(damage_match.group(1))
                            if damage >= c_tough:
                                kills.append(c_name)
                            else:
                                wont_kill.append(f"{c_name}(T={c_tough})")
                        elif minus_match:
                            toughness_reduction = abs(int(minus_match.group(2)))
                            if toughness_reduction >= c_tough:
                                kills.append(c_name)
                            else:
                                wont_kill.append(f"{c_name}(T={c_tough})")

                    if kills:
                        removal_info = f" [KILLS: {', '.join(kills)}]"
                    if wont_kill:
                        removal_info += f" [WON'T KILL: {', '.join(wont_kill)}]"

                lines.append(f"  - {name} {cost} {timing} {castable}{removal_info}")
                if card.get("oracle_text"):
                    lines.append(f"      {card['oracle_text']}")
        else:
            lines.append("  (empty)")

        # Stack with owner labels
        stack = game_state.get("stack", [])
        if stack:
            lines.append("")
            lines.append("STACK:")
            for card in stack:
                name = card.get("name", "Unknown")
                owner_seat = card.get("owner_seat_id")
                owner_label = "Your" if owner_seat == local_seat else "Opp's"
                lines.append(f"  - {owner_label} {name}")

        # Graveyards with counts per player
        graveyard = game_state.get("graveyard", [])
        if graveyard:
            your_gy = [c for c in graveyard if c.get("owner_seat_id") == local_seat]
            opp_gy = [c for c in graveyard if c.get("owner_seat_id") != local_seat]
            lines.append("")
            lines.append(f"GRAVEYARDS: Your={len(your_gy)}, Opponent={len(opp_gy)}")

        return "\n".join(lines)

    def _extract_card_name_words(self, game_state: dict[str, Any]) -> set[str]:
        """Extract all words from card names in the current game state.

        These words are excluded from overuse tracking since they're card names.
        """
        import re
        card_words: set[str] = set()

        # Collect card names from all zones
        for zone in ["battlefield", "hand", "graveyard", "stack", "exile"]:
            for card in game_state.get(zone, []):
                name = card.get("name", "")
                # Extract words from card name
                words = re.findall(r'\b[a-z]+\b', name.lower())
                card_words.update(words)

        return card_words

    def get_advice(
        self,
        game_state: dict[str, Any],
        question: Optional[str] = None,
        trigger: Optional[str] = None,
        style: str = "concise"
    ) -> str:
        """Get coaching advice for the current game state.

        Args:
            game_state: Dict from get_game_state() MCP tool
            question: Optional user question to answer
            trigger: Optional trigger name (e.g., "combat_attackers", "low_life")
            style: Advice style ("concise" or "verbose")

        Returns:
            Advice string from the LLM
        """
        import time
        total_start = time.perf_counter()

        # Build context
        context_start = time.perf_counter()
        context = self._format_game_context(game_state)
        context_time = (time.perf_counter() - context_start) * 1000

        # Get card name words to exclude from overuse tracking
        card_words = self._extract_card_name_words(game_state)

        # Check for overused words to avoid (excluding card names)
        blacklisted = self._word_tracker.get_blacklisted(exclude_words=card_words)

        # Build dynamic system prompt
        system_prompt = self._system_prompt
        
        # Adjust for style
        if style == "verbose":
            system_prompt = system_prompt.replace(
                "Keep responses concise (2-3 sentences)",
                "Provide detailed strategic reasoning (4-5 sentences)"
            )
            # Remove "Be direct... no 'consider'" constraint for verbose mode to allow more nuance
            system_prompt = system_prompt.replace(
                "Be direct and specific - tell the player exactly what to do, not what to \"consider\".",
                "Explain the 'why' behind your advice, discussing alternatives if relevant."
            )
        
        if blacklisted:
            avoid_list = ", ".join(blacklisted)
            system_prompt += f"\n\nIMPORTANT: Avoid using these overused words: {avoid_list}. Use different phrasing."
            logger.debug(f"Blacklisted words: {blacklisted}")

        # Build user message
        if question:
            user_message = f"{context}\n\nThe player asks: {question}"
        if trigger:
            trigger_descriptions = {
                "new_turn": "A new turn has started.",
                "priority_gained": "You just gained priority.",
                "combat_attackers": "You're declaring attackers.",
                "combat_blockers": "Opponent attacked, declare blockers.",
                "low_life": "Your life total is dangerously low!",
                "opponent_low_life": "Opponent's life is low - chance to win!",
                "stack_spell": "Something was just cast - do you want to respond?",
                "user_request": "Give quick strategic advice.",
                "decision_required": "Game is waiting for a decision (e.g. Mulligan).",
            }
            trigger_desc = trigger_descriptions.get(trigger, f"Trigger: {trigger}")
            user_message = f"{context}\n\n{trigger_desc} What should the player do?"
        else:
            user_message = f"{context}\n\nWhat's the best play right now?"

        prompt_len = len(system_prompt) + len(user_message)
        logger.info(f"[TIMING] Context built in {context_time:.1f}ms, prompt size: {prompt_len} chars")

        # Select system prompt based on style
        # Priority: explicit arg > object property > default
        selected_style = style if style else getattr(self, "advice_style", "concise")
        style_key = selected_style.lower()
        
        # Define style prompts (lazy loaded or defined here)
        prompts = {
            "concise": CONCISE_SYSTEM_PROMPT,
            "normal": DEFAULT_SYSTEM_PROMPT,
            "explain": DEFAULT_SYSTEM_PROMPT.replace("Keep responses concise (2-3 sentences max)", "Explain your reasoning clearly but briefly.") + "\nInclude a short explanation of WHY this is the best line.",
            "pirate": "You are a ruthless pirate captain coaching a swabby! Speak like a pirate! Yarr! Keep it short!",
        }
        
        effective_system_prompt = prompts.get(style_key, CONCISE_SYSTEM_PROMPT)
        
        # Get response and track word usage (excluding card names)
        api_start = time.perf_counter()
        response = self._backend.complete(effective_system_prompt, user_message)
        api_time = (time.perf_counter() - api_start) * 1000

        self._word_tracker.record(response, exclude_words=card_words)

        total_time = (time.perf_counter() - total_start) * 1000
        logger.info(f"[TIMING] API call: {api_time:.0f}ms, total: {total_time:.0f}ms, response: {len(response)} chars")

        return response

    def complete_with_image(self, system_prompt: str, user_message: str, image_data: bytes) -> str:
        """Call complete_with_image on backend if supported."""
        if hasattr(self._backend, 'complete_with_image'):
            return self._backend.complete_with_image(system_prompt, user_message, image_data)
        logger.error(f"Backend {type(self._backend).__name__} does not support complete_with_image")
        return "Image analysis not supported by current backend."


class GameStateTrigger:
    """Detects trigger conditions by comparing game states."""

    def __init__(self, life_threshold: int = 5):
        """Initialize trigger detector.

        Args:
            life_threshold: Life total below which "low_life" triggers (default: 5)
        """
        self.life_threshold = life_threshold

    def _get_local_player(self, state: dict[str, Any]) -> Optional[dict]:
        """Get the local player dict from game state."""
        for p in state.get("players", []):
            if p.get("is_local"):
                return p
        return None

    def _get_opponent_player(self, state: dict[str, Any]) -> Optional[dict]:
        """Get the opponent player dict from game state."""
        for p in state.get("players", []):
            if not p.get("is_local"):
                return p
        return None

    def _has_castable_instants(self, state: dict[str, Any]) -> bool:
        """Check if player has any instant-speed cards they can cast.

        Returns True if hand contains instants or flash cards that can be
        cast with the current available mana.
        """
        import re

        # Count untapped lands for mana
        local_seat = None
        for p in state.get("players", []):
            if p.get("is_local"):
                local_seat = p.get("seat_id")
                break

        if local_seat is None:
            return False

        battlefield = state.get("battlefield", [])
        untapped_lands = sum(
            1 for c in battlefield
            if c.get("owner_seat_id") == local_seat
            and "land" in c.get("type_line", "").lower()
            and not c.get("is_tapped")
        )

        # Check hand for castable instants/flash
        hand = state.get("hand", [])
        for card in hand:
            type_line = card.get("type_line", "").lower()
            oracle_text = card.get("oracle_text", "").lower()

            # Check if instant speed
            is_instant_speed = "instant" in type_line or "flash" in oracle_text
            if not is_instant_speed:
                continue

            # Calculate CMC
            cost = card.get("mana_cost", "")
            cmc = 0
            if cost:
                generic = re.findall(r'\{(\d+)\}', cost)
                cmc += sum(int(g) for g in generic)
                colored = re.findall(r'\{[WUBRGC]\}', cost)
                cmc += len(colored)
                hybrid = re.findall(r'\{[^}]+/[^}]+\}', cost)
                cmc += len(hybrid)

            if untapped_lands >= cmc:
                return True

        return False

    def check_triggers(
        self,
        prev_state: dict[str, Any],
        curr_state: dict[str, Any]
    ) -> list[str]:
        """Compare two game states and return triggered condition names.

        Args:
            prev_state: Previous game state dict
            curr_state: Current game state dict

        Returns:
            List of trigger names that fired (may be empty)
        """
        triggers = []


        prev_turn = prev_state.get("turn", {})
        curr_turn = curr_state.get("turn", {})

        # Retrieve phase and step early (fix scoping issues)
        curr_phase = curr_turn.get("phase", "")
        curr_step = curr_turn.get("step", "")

        # Get local player info first (needed for turn detection)
        prev_local = self._get_local_player(prev_state)
        curr_local = self._get_local_player(curr_state)
        local_seat = curr_local.get("seat_id") if curr_local else None

        # New turn detection - only on YOUR turn
        prev_turn_num = prev_turn.get("turn_number", 0)
        curr_turn_num = curr_turn.get("turn_number", 0)
        curr_active = curr_turn.get("active_player", 0)
        if curr_turn_num > prev_turn_num:
            triggers.append("new_turn")

        # Check if it's your turn or opponent's turn
        is_your_turn = curr_active == local_seat

        # Priority gained - trigger when priority shifts to you
        prev_priority = prev_turn.get("priority_player", 0)
        curr_priority = curr_turn.get("priority_player", 0)
        if local_seat and curr_priority == local_seat and prev_priority != local_seat:
            # Always trigger on your turn
            # On opponent's turn, trigger if:
            #   1. You have castable instants
            #   2. There's something on the stack to consider
            #   3. We're in a significant phase (combat, main)
            has_options = self._has_castable_instants(curr_state)
            has_stack = len(curr_state.get("stack", [])) > 0
            # Retrieve phase and step early
            curr_phase = curr_turn.get("phase", "")
            curr_step = curr_turn.get("step", "")

            if is_your_turn or has_options or has_stack or (any(p in curr_phase for p in ["Main", "Combat", "Beginning"])):
                triggers.append("priority_gained")
        
        # Check explicit pending decisions (like Mulligan)
        pending_decision = curr_state.get("pending_decision")
        if pending_decision:
             # Just trigger it every time if it's there?
             # Or only if it wasn't there before?
             # For now, let's trigger it continuously but the coach loop has a dampener
             # Actually, best to trigger only on change or if it's new
             prev_decision = prev_state.get("pending_decision")
             if pending_decision != prev_decision:
                 logger.info(f"Triggering decision: {pending_decision}")
                 triggers.append("decision_required")

        # Combat phase detection - use pending steps to catch fast combat phases
        pending_steps = curr_turn.get("pending_combat_steps", [])

        for step_info in pending_steps:
            step = step_info.get("step", "")
            step_active = step_info.get("active_player", 0)
            step_is_your_turn = step_active == local_seat

            logger.debug(f"Processing pending combat step: {step}, active={step_active}, is_your_turn={step_is_your_turn}")

            if "DeclareAttack" in step and step_is_your_turn:
                if "combat_attackers" not in triggers:
                    logger.info(f"Combat attackers trigger from pending: {step}")
                    triggers.append("combat_attackers")
            elif "DeclareBlock" in step and not step_is_your_turn:
                if "combat_blockers" not in triggers:
                    logger.info(f"Combat blockers trigger from pending: {step}")
                    triggers.append("combat_blockers")

        # Also check current step (in case we're still in combat)
        # curr_phase and curr_step are already defined above

        if "Combat" in curr_phase:
            prev_step = prev_turn.get("step", "")
            # Only trigger on STEP CHANGE to avoid spamming every polling cycle
            if curr_step != prev_step:
                if "DeclareAttack" in curr_step and is_your_turn and "combat_attackers" not in triggers:
                    logger.info(f"Combat attackers trigger: step={curr_step}")
                    triggers.append("combat_attackers")
                elif "DeclareBlock" in curr_step and not is_your_turn and "combat_blockers" not in triggers:
                    logger.info(f"Combat blockers trigger: step={curr_step}")
                    triggers.append("combat_blockers")

        # Low life detection - always important
        if curr_local:
            curr_life = curr_local.get("life_total", 20)
            prev_life = prev_local.get("life_total", 20) if prev_local else 20
            if curr_life < self.life_threshold and prev_life >= self.life_threshold:
                triggers.append("low_life")

        # Opponent low life detection - always important
        prev_opp = self._get_opponent_player(prev_state)
        curr_opp = self._get_opponent_player(curr_state)
        if curr_opp:
            curr_opp_life = curr_opp.get("life_total", 20)
            prev_opp_life = prev_opp.get("life_total", 20) if prev_opp else 20
            if curr_opp_life < self.life_threshold and prev_opp_life >= self.life_threshold:
                triggers.append("opponent_low_life")

        # Stack spell detection - always trigger so coach can advise (even if just "let it resolve")
        prev_stack = prev_state.get("stack", [])
        curr_stack = curr_state.get("stack", [])
        if len(curr_stack) > len(prev_stack):
            triggers.append("stack_spell")

        return triggers
