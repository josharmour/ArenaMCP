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
            # OPTIMIZATION: Increased from 500 to 1000 for complex game states
            response = client.messages.create(
                model=self.model,
                max_tokens=1000,
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

    def __init__(self, model: str = "gemini-2.5-flash"):
        """Initialize Gemini backend with lazy client creation.

        Args:
            model: The Gemini model to use (default: gemini-2.5-flash)
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

    def complete(self, system_prompt: str, user_message: str, max_tokens: int = 1024) -> str:
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
                    max_output_tokens=max_tokens,
                    temperature=0.0,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
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
                candidate = response.candidates[0]
                parts = getattr(candidate.content, 'parts', None) if candidate.content else None
                if parts:
                    for part in parts:
                        if hasattr(part, 'text') and part.text:
                            text += part.text

            if not text:
                try:
                    text = response.text  # Fallback
                except (AttributeError, ValueError):
                    text = ""
                 
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
                    max_output_tokens=1024,
                    temperature=0.0,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
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

    # Models known to support response_modalities=["AUDIO"] via generate_content.
    # NOTE: As of 2026-02, no standard Gemini model supports this — only the
    # dedicated TTS model (gemini-2.5-flash-preview-tts) and Live API do.
    # Keep this set for future models; the text+GeminiTTS fallback works well.
    AUDIO_CAPABLE_MODELS: set[str] = set()

    def complete_audio(
        self,
        system_prompt: str,
        user_message: str,
        voice: str = "Kore"
    ) -> "tuple[np.ndarray, int] | None":
        """Get audio completion from Gemini API (model generates speech directly).

        Uses the same model as text path but with response_modalities=["AUDIO"]
        so the model reasons about the game state AND speaks the advice in one call.

        Args:
            system_prompt: The system prompt setting up the assistant role
            user_message: The game context and trigger description
            voice: Gemini voice name (default: Kore)

        Returns:
            Tuple of (samples as float32 numpy array, sample_rate) on success,
            None on any error.
        """
        import time

        # Only attempt audio on models known to support it
        if self.model not in self.AUDIO_CAPABLE_MODELS:
            logger.debug(f"Model {self.model} not in audio-capable list, skipping complete_audio")
            return None

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY not set for audio generation")
            return None

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
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice,
                            )
                        )
                    ),
                    temperature=0.0,
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
                )
            )
            request_time = (time.perf_counter() - request_start) * 1000

            # Extract audio data from response (robust parsing)
            from arenamcp.tts import _extract_audio_from_response, _decode_audio_bytes

            audio_data = _extract_audio_from_response(response)
            if audio_data is None:
                logger.error(f"[GEMINI+AUDIO_OUT] No audio data in response (model={self.model})")
                return None

            samples, sample_rate = _decode_audio_bytes(audio_data)

            logger.info(f"[GEMINI+AUDIO_OUT] API: {request_time:.0f}ms, client: {client_time:.1f}ms, model: {self.model}, voice: {voice}, samples: {len(samples)}, rate: {sample_rate}")

            return (samples, sample_rate)

        except Exception as e:
            logger.error(f"Gemini audio output error: {e}")
            return None

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
                    max_output_tokens=1024,
                    temperature=0.0,
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

    def complete_with_image(self, system_prompt: str, user_message: str, image_data: bytes) -> str:
        """Get completion from Ollama with image input (vision models like gemma3n, llava, etc.)."""
        import requests
        import base64

        try:
            # Encode image as base64
            img_b64 = base64.b64encode(image_data).decode('utf-8')

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": user_message,
                    "system": system_prompt,
                    "images": [img_b64],  # Ollama expects list of base64 images
                    "stream": False,
                },
                timeout=120,  # Vision requests may take longer
            )

            if response.status_code == 404:
                return f"Error: Model '{self.model}' not found. Run: ollama pull {self.model}"

            response.raise_for_status()
            return response.json().get("response", "No response from Ollama")

        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Is it running? Start with: ollama serve"
        except requests.exceptions.Timeout:
            return "Error: Ollama vision request timed out"
        except Exception as e:
            logger.error(f"Ollama vision error: {e}")
            return f"Error getting vision advice from Ollama: {e}"


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
                max_completion_tokens=800  # GPT-5 requires this instead of max_tokens
            )
            request_time = (time.perf_counter() - client_start) * 1000

            content = response.choices[0].message.content
            logger.info(f"[AZURE] API: {request_time:.0f}ms, model: {self.model}")

            return content
        except Exception as e:
            # Try to handle "DeploymentNotFound" gracefully
            err_str = str(e)
            is_404 = "404" in err_str or "DeploymentNotFound" in err_str

            if is_404:
                # Try to list available models to help the user
                try:
                    available = self.list_models()
                    # Prioritize GPT-5 models for visibility
                    gpt5_models = [m for m in available if "gpt-5" in m.lower() or "5.1" in m or "5.2" in m]
                    other_gpt = [m for m in available if "gpt" in m.lower() and m not in gpt5_models][:15]
                    
                    if gpt5_models:
                        help_msg = f"\n\nGPT-5 models found:\n" + "\n".join([f"- {m}" for m in gpt5_models[:10]])
                        if other_gpt:
                            help_msg += f"\n\nOther GPT models:\n" + "\n".join([f"- {m}" for m in other_gpt])
                    else:
                        help_msg = f"\n\nNo GPT-5 deployments found. Available GPT models:\n" + "\n".join([f"- {m}" for m in other_gpt])
                        help_msg += f"\n\n(Total {len(available)} models available)"
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
             text_model = "gemini-2.5-flash"
        
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
- Cards tagged [CAN CAST] are the ONLY cards you can suggest casting.
- Cards tagged [NEED X mana] CANNOT be cast - do NOT suggest them!
- Do NOT perform your own mana calculations - trust the tags completely.
- If ALL cards show [NEED X mana], say "pass" - you cannot cast anything.

CRITICAL MATH RULES:
- When suggesting removal, check the creature's TOUGHNESS (second number, e.g., 4/5 has 5 toughness).
- -2/-2 or 2 damage ONLY kills toughness 2 or less (unless damaged).
- Do NOT suggest removal that won't kill the target unless it enables a profitable attack.

CRITICAL BLOCKING RULES:
- Creatures tagged [FLYING] can ONLY be blocked by creatures with [FLYING] or [REACH].
- Do NOT suggest blocking a [FLYING] creature with a ground creature (no [FLYING]/[REACH]).
- If enemy attackers have [FLYING] and you have no flyers/reach, you CANNOT block them.

Analyze: phase (critical for timing!), board state, life totals, cards in hand, mana available.
Output directly as the coach. No preamble, no meta-commentary."""

CONCISE_SYSTEM_PROMPT = """You are an expert MTG coach giving real-time spoken advice.
Give ONE action for the CURRENT phase only. You will be re-consulted as the turn progresses.

PHASE GUIDE:
- Main phase: Suggest ONE play (land OR spell). You'll advise again after it resolves.
- Combat/DeclareAttack: Say who to attack with (or "don't attack").
- Combat/DeclareBlock: Say how to block (or "don't block, take the damage").
- Opponent's turn: React to what's happening (instants/abilities only).
- Stack: Say whether to respond or let it resolve.

After your ONE action, you may add a brief reason or hint at the next step.

Examples:
"Play Mountain. Sets up Geological Appraiser next turn."
"Cast Etali's Favor on Laelia — triggers discover for the cascade chain."
"Attack with Laelia, the Blade Reforged. She exiles and grows."
"Don't block. Take the 3 damage, you're at 20."
"Let it resolve. Nothing worth countering."
"Pass priority."

RULES:
- Only suggest cards tagged [CAN CAST] or [OK]. Cards tagged [NEED X mana] CANNOT be cast!
- Use exact FULL card names from the game state. Never abbreviate.
- Only suggest lands shown in HAND. If no land in hand, don't suggest playing one.
- Say "pass priority" not just "pass" to avoid sounding like a card name.
- This is spoken aloud — keep it natural and under 30 words.
"""

# PHASE 2: Decision-specific prompt guidance
DECISION_PROMPTS = {
    "scry": """
SCRY DECISION: Decide whether to keep the card on top or put it on bottom.
- KEEP if: It's a land and you need mana, OR it's a threat you can cast soon
- BOTTOM if: It's redundant/dead right now, or you need to dig for answers
Evaluate based on: current mana, hand quality, board state urgency.
Answer: "Keep" or "Bottom" with brief reason (1 sentence).
""",
    "surveil": """
SURVEIL DECISION: Decide whether to keep cards on top or put in graveyard.
- KEEP if: You want to draw them next (lands if ramping, threats if you have mana)
- GRAVEYARD if: Enables graveyard synergies OR you want to dig deeper
Answer: "Keep [card names]" or "Graveyard [card names]" with brief reason.
""",
    "discard": """
DISCARD DECISION: Choose which card(s) to discard.
Priority (discard FIRST):
1. Excess lands if you have 4+ in hand
2. Highest CMC card you can't cast this turn or next
3. Redundant copies of cards already in play
4. KEEP: Removal, counters, win conditions
Answer: "Discard [card name]" with brief reason (1 sentence).
""",
    "target_selection": """
TARGET SELECTION: Choose the best target for this spell/ability.
Evaluate each potential target:
- Which target solves the biggest immediate threat?
- Which target advances your win condition?
- Consider opponent's likely responses (do they have protection?)
Answer: "Target [card name]" with brief tactical reason.
""",
    "modal_choice": """
MODAL SPELL: Choose which mode to use.
Compare each mode's impact:
- Which mode answers the most pressing threat?
- Which mode creates the best advantage?
- Consider mana efficiency and follow-up plays
Answer: "Choose mode [X]" with brief reason (1 sentence).
""",
}

DECK_ANALYSIS_PROMPT = """Analyze this Magic: The Gathering deck list. Provide a brief strategic summary:
1. ARCHETYPE: One-line description (e.g. "Mono-Red Aggro", "Dimir Control")
2. WIN CONDITION: How does this deck win?
3. KEY CARDS: 3-5 most important cards and why
4. PLAY PATTERN: Ideal curve and sequencing (e.g. "Play threats T1-T3, hold up removal T4+")
5. WATCH OUT: Key weaknesses or cards to play around

Keep the entire analysis under 300 characters. Be specific to THIS deck, not generic advice."""


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
        self._deck_strategy: Optional[str] = None
        self._deck_strategy_pending = False
    
    def clear_deck_strategy(self) -> None:
        """Reset deck strategy for a new match."""
        self._deck_strategy = None
        self._deck_strategy_pending = False

    def analyze_deck(self, deck_cards: list[tuple[str, str]]) -> Optional[str]:
        """Analyze a deck list and store the strategy summary.

        Args:
            deck_cards: List of (card_name, card_type) tuples

        Returns:
            Strategy string, or None on failure
        """
        import time
        start = time.perf_counter()
        self._deck_strategy_pending = True

        try:
            # Group duplicates compactly: "4x Mountain (Basic Land)"
            from collections import Counter
            card_counts = Counter(deck_cards)
            deck_lines = []
            for (name, card_type), count in card_counts.most_common():
                type_short = card_type.split("—")[0].strip() if card_type else "Unknown"
                deck_lines.append(f"{count}x {name} ({type_short})")

            deck_text = "\n".join(deck_lines)
            user_message = f"DECK LIST ({len(deck_cards)} cards):\n{deck_text}"

            # Deck analysis needs far more tokens than the default 150 for game advice.
            # Thinking models (gemini-2.5-flash) consume tokens on internal reasoning,
            # so max_output_tokens must be high enough for thinking + visible output.
            try:
                strategy = self._backend.complete(DECK_ANALYSIS_PROMPT, user_message, max_tokens=2048)
            except TypeError:
                # Backend doesn't support max_tokens parameter
                strategy = self._backend.complete(DECK_ANALYSIS_PROMPT, user_message)

            # Truncate if too long
            if len(strategy) > 400:
                strategy = strategy[:397] + "..."

            self._deck_strategy = strategy
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"Deck analysis complete: {elapsed:.0f}ms, {len(strategy)} chars")
            return strategy
        except Exception as e:
            logger.error(f"Deck analysis failed: {e}")
            return None
        finally:
            self._deck_strategy_pending = False

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate for logging: ~4 chars per token.
        
        OPTIMIZATION: Added for prompt size monitoring.
        """
        return len(text) // 4


    def _remove_reminder_text(self, text: str) -> str:
        """Remove reminder text (text in parentheses) from oracle text."""
        import re
        # Handle nested parens if possible, but simple greedy match usually works for MTG
        # Use simple non-greedy match for multiple parens
        return re.sub(r'\(.*?\)', '', text)

    def _format_game_context(self, game_state: dict[str, Any], question: str = "") -> str:
        """Format the game state into a COMPACT context for the LLM.
        
        OPTIMIZATION: Heavily compressed to reduce token usage while maintaining accuracy.
        - Uses symbols (T=tapped, FLY=flying, SS=summoning sick)
        - Only shows oracle text for relevant cards (not basic lands)
        - Terse removal analysis (kill range, not individual targets)
        - Consolidated combat/blocking info
        - Removed redundant rule explanations (LLM knows MTG rules)
        """
        
        # Determine local player seat and active turn
        players = game_state.get("players", [])
        local_player = next((p for p in players if p.get("is_local")), None)
        local_seat = local_player.get("seat_id") if local_player else 1
        
        turn = game_state.get("turn", {})
        active_seat = turn.get("active_player", 0)
        priority_seat = turn.get("priority_player", 0)
        is_my_turn = (active_seat == local_seat)
        has_priority = (priority_seat == local_seat)
        
        phase = turn.get("phase", "Unknown").replace("Phase_", "")
        step = turn.get("step", "").replace("Step_", "")
        turn_num = turn.get("turn_number", 0)

        # Get legal moves (still useful for complex decisions)
        try:
            from arenamcp.rules_engine import RulesEngine
            valid_moves = RulesEngine.get_legal_actions(game_state)
            # OPTIMIZATION: Join inline instead of list, limit to 8 most important
            valid_moves_str = ", ".join(valid_moves[:8])
            if len(valid_moves) > 8:
                valid_moves_str += f"... (+{len(valid_moves)-8})"
        except Exception as e:
            logger.error(f"RulesEngine error: {e}")
            valid_moves_str = "Error"

        lines = []
        lines.append("=== GAME ===")
        lines.append(f"Legal: {valid_moves_str}")

        # Get player info
        opponent_player = None
        for p in players:
            if not p.get("is_local"):
                opponent_player = p
                break
        
        opp_seat = opponent_player.get("seat_id") if opponent_player else None

        # OPTIMIZATION: Compact turn info - one line
        active_label = "YOUR" if active_seat == local_seat else "OPP"
        priority_label = "You" if priority_seat == local_seat else "Opp"
        
        # Timing context
        is_main_phase = "Main" in phase
        is_your_turn = active_seat == local_seat
        stack = game_state.get("stack", [])
        stack_empty = len(stack) == 0
        can_cast_sorcery = is_your_turn and is_main_phase and stack_empty and has_priority
        is_blocking = "DeclareBlock" in step and not is_your_turn
        
        # Decision Check
        pending_decision = game_state.get("pending_decision")
        decision_context = game_state.get("decision_context")
        
        if pending_decision:
            # PHASE 1+2: Enhanced decision display with context
            if decision_context:
                dec_type = decision_context.get("type", "unknown")
                
                if dec_type == "discard":
                    count = decision_context.get("count", 1)
                    lines.append(f"!!! DECISION: DISCARD {count} card(s) !!!")
                    lines.append("Choose: excess lands > high CMC uncastables > redundant copies")
                    
                elif dec_type == "scry":
                    count = decision_context.get("count", 1)
                    lines.append(f"!!! DECISION: SCRY {count} !!!")
                    lines.append("Keep: needed lands/threats | Bottom: dead cards")
                    
                elif dec_type == "surveil":
                    count = decision_context.get("count", 1)
                    lines.append(f"!!! DECISION: SURVEIL {count} !!!")
                    lines.append("Keep: want to draw | Graveyard: synergy or digging")
                    
                elif dec_type == "target_selection":
                    source = decision_context.get("source_card", "spell")
                    lines.append(f"!!! DECISION: TARGET for {source} !!!")
                    lines.append("Choose: biggest threat or best value target")
                    
                elif dec_type == "modal_choice":
                    num_opts = decision_context.get("num_options", "?")
                    lines.append(f"!!! DECISION: CHOOSE MODE ({num_opts} options) !!!")
                    lines.append("Evaluate: which mode solves current problem best")
                    
                else:
                    # Fallback for other decision types
                    lines.append(f"!!! DECISION: {pending_decision} !!!")
            else:
                # No context available - generic display
                lines.append(f"!!! DECISION: {pending_decision} !!!")
                
            # Special handling for Mulligan (legacy)
            if pending_decision == "Mulligan":
                my_hand = game_state.get("zones", {}).get("my_hand", [])
                if not my_hand:
                    lines.append("Waiting for hand...")
                else:
                    lines.append("Evaluate: lands, colors, curve → KEEP or MULLIGAN")
        
        # OPTIMIZATION: Single line for turn/phase/priority
        phase_str = f"{phase}/{step}" if step else phase
        lines.append(f"T{turn_num} {active_label} | {phase_str} | Pri:{priority_label}")
        
        # OPTIMIZATION: Compact timing rules - single line
        if can_cast_sorcery:
            lines.append("Timing: ALL SPELLS")
        elif is_blocking:
            lines.append("ACTION: DECLARE BLOCKERS")
        else:
            lines.append("Timing: INSTANTS ONLY")
        
        # OPTIMIZATION: Compact life totals - single line
        your_life = local_player.get('life_total', '?') if local_player else '?'
        opp_life = opponent_player.get('life_total', '?') if opponent_player else '?'
        lines.append(f"Life: You={your_life} Opp={opp_life}")

        # Battlefield - grouped by owner
        battlefield = game_state.get("battlefield", [])
        your_cards = [c for c in battlefield if c.get("owner_seat_id") == local_seat]
        opp_cards = [c for c in battlefield if c.get("owner_seat_id") != local_seat]

        # OPTIMIZATION: Compact mana calculation
        mana_pool = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0, "C": 0, "Any": 0}
        total_mana = 0
        
        for card in your_cards:
            type_line = card.get("type_line", "").lower()
            oracle = card.get("oracle_text", "")
            is_creature = "creature" in type_line
            # Check for casting sickness (unless it has haste)
            has_haste = "haste" in self._remove_reminder_text(oracle).lower()
            is_summoning_sick = is_creature and card.get("turn_entered_battlefield") == turn_num and not has_haste
            
            # Count mana sources (Lands AND Creatures)
            # Logic: Untapped AND (Land OR (Creature AND "add {" in oracle AND not summoning sick))
            # Relaxed check: Just looking for "Add " or "{T}: Add" is surprisingly robust for dorks
            has_mana_ability = "add {" in oracle.lower() or "add one mana" in oracle.lower()
            
            if not card.get("is_tapped"):
                if "land" in type_line or (is_creature and has_mana_ability and not is_summoning_sick):
                    total_mana += 1
                    name = card.get("name", "")
                    
                    # Color detection (simplified)
                    if "Plains" in name or "{W}" in oracle: mana_pool["W"] += 1
                    if "Island" in name or "{U}" in oracle: mana_pool["U"] += 1
                    if "Swamp" in name or "{B}" in oracle: mana_pool["B"] += 1
                    if "Mountain" in name or "{R}" in oracle: mana_pool["R"] += 1
                    if "Forest" in name or "{G}" in oracle: mana_pool["G"] += 1
                    if "{C}" in oracle: mana_pool["C"] += 1
                    if "any color" in oracle.lower(): mana_pool["Any"] += 1

        logger.info(f"Mana: {mana_pool} (Total: {total_mana})")
        
        # OPTIMIZATION: Compact mana display - only show colors you have
        mana_colors = [f"{c}:{mana_pool[c]}" for c in "WUBRGC" if mana_pool[c] > 0]
        if mana_pool["Any"] > 0:
            mana_colors.append(f"Any:{mana_pool['Any']}")
        mana_str = " ".join(mana_colors) if mana_colors else "0"
        lines.append(f"Mana: {total_mana} ({mana_str})")

        # OPTIMIZATION: Compact land drop status
        lands_played = local_player.get("lands_played", 0) if local_player else 0
        if is_your_turn and lands_played == 0:
            lines.append("Land: AVAILABLE")
        elif is_your_turn:
            lines.append(f"Land: USED ({lands_played})")
        else:
            lines.append("Land: N/A (opp turn)")

        # OPTIMIZATION: Compact battlefield display with symbols
        # T=tapped, FLY=flying, RCH=reach, SS=summoning sick, ATK=attacking, BLK=blocking
        if battlefield:
            lines.append("")
            lines.append(f"YOUR BOARD:")
            if your_cards:
                for card in your_cards:
                    name = card.get("name", "Unknown")
                    type_line = card.get("type_line", "").lower()
                    
                    # P/T for creatures
                    pt = f" {card['power']}/{card['toughness']}" if card.get("power") is not None else ""
                    
                    # Status flags (compact symbols)
                    flags = []
                    if card.get("is_tapped"): flags.append("T")
                    
                    oracle_text = self._remove_reminder_text(card.get("oracle_text", "")).lower()
                    is_creature = "creature" in type_line
                    
                    # Keywords that matter for combat
                    if "flying" in oracle_text: flags.append("FLY")
                    if "reach" in oracle_text: flags.append("RCH")
                    if "haste" in oracle_text: flags.append("HST")
                    if "vigilance" in oracle_text: flags.append("VIG")
                    if "trample" in oracle_text: flags.append("TRM")
                    if "first strike" in oracle_text: flags.append("FS")
                    if "deathtouch" in oracle_text: flags.append("DTH")
                    
                    # Summoning sickness check
                    if is_creature and card.get("turn_entered_battlefield") == turn_num and "haste" not in oracle_text:
                        flags.append("SS")
                    
                    if card.get("is_attacking"): flags.append("ATK")
                    if card.get("is_blocking"): flags.append("BLK")
                    
                    flag_str = f" [{','.join(flags)}]" if flags else ""
                    lines.append(f"  {name}{pt}{flag_str}")

            else:
                lines.append("  (empty)")

            lines.append(f"OPP BOARD:")
            if opp_cards:
                for card in opp_cards:
                    name = card.get("name", "Unknown")
                    type_line = card.get("type_line", "").lower()
                    pt = f" {card['power']}/{card['toughness']}" if card.get("power") is not None else ""
                    
                    flags = []
                    if card.get("is_tapped"): flags.append("T")
                    
                    oracle_text = self._remove_reminder_text(card.get("oracle_text", "")).lower()
                    if "flying" in oracle_text: flags.append("FLY")
                    if "reach" in oracle_text: flags.append("RCH")
                    if "vigilance" in oracle_text: flags.append("VIG")
                    if "trample" in oracle_text: flags.append("TRM")
                    if "first strike" in oracle_text: flags.append("FS")
                    if "deathtouch" in oracle_text: flags.append("DTH")
                    
                    if card.get("is_attacking"): flags.append("ATK")
                    if card.get("is_blocking"): flags.append("BLK")
                    
                    flag_str = f" [{','.join(flags)}]" if flags else ""
                    lines.append(f"  {name}{pt}{flag_str}")
            else:
                lines.append("  (empty)")

            # OPTIMIZATION: Compact combat analysis (was 70+ lines, now ~20)
            if ("Combat" in phase or "Main" in phase) and is_your_turn:
                your_creatures = [c for c in your_cards
                                  if c.get("power") is not None
                                  and "land" not in c.get("type_line", "").lower()]
                
                valid_attackers = [c for c in your_creatures
                                   if not c.get("is_tapped")
                                   and not (c.get("turn_entered_battlefield") == turn_num
                                           and "haste" not in self._remove_reminder_text(c.get("oracle_text", "")).lower())]
                
                your_attack_power = sum(c.get("power", 0) for c in valid_attackers)
                
                opp_creatures = [c for c in opp_cards
                                 if c.get("power") is not None
                                 and "land" not in c.get("type_line", "").lower()]
                opp_blockers = [c for c in opp_creatures if not c.get("is_tapped")]
                opp_block_count = len(opp_blockers)
                opp_life = opponent_player.get("life_total", 20) if opponent_player else 20
                
                # Single line combat summary
                if valid_attackers:
                    lethal = "LETHAL" if (opp_block_count == 0 and your_attack_power >= opp_life) else f"{opp_block_count}blk"
                    lines.append(f"Atk: {len(valid_attackers)}cr/{your_attack_power}pwr vs {lethal}")
                else:
                    lines.append(f"Atk: None (T/SS)")
            
            # OPTIMIZATION: Compact blocking analysis (was 50+ lines, now ~10)
            elif "Combat" in phase and not is_your_turn:
                attacking = [c for c in opp_cards if c.get("is_attacking")]
                flying_atk = [c for c in attacking if "flying" in self._remove_reminder_text(c.get("oracle_text", "")).lower()]
                ground_atk = [c for c in attacking if c not in flying_atk]
                
                your_creatures = [c for c in your_cards
                                  if c.get("power") is not None
                                  and not c.get("is_tapped")
                                  and "land" not in c.get("type_line", "").lower()]
                
                flyer_blockers = [c for c in your_creatures
                                  if any(kw in self._remove_reminder_text(c.get("oracle_text", "")).lower()
                                        for kw in ["flying", "reach"])]
                
                # Single line blocking summary
                if attacking:
                    fly_dmg = sum(c.get("power", 0) for c in flying_atk)
                    gnd_dmg = sum(c.get("power", 0) for c in ground_atk)
                    lines.append(f"Blk: {fly_dmg}fly/{gnd_dmg}gnd dmg | {len(flyer_blockers)}FLY-blk avail")
                    if flying_atk and not flyer_blockers:
                        lines.append(f"⚠️ {fly_dmg} UNBLOCKABLE!")
        else:
            lines.append("")
            lines.append("BOARD: Empty")

        # OPTIMIZATION: Compact hand display
        hand = game_state.get("hand", [])
        lines.append("")
        lines.append(f"HAND:")

        # Pre-compute opponent creatures for terse removal analysis
        opp_creatures = [c for c in opp_cards
                         if c.get("power") is not None
                         and "land" not in c.get("type_line", "").lower()]

        if hand:
            import re
            
            # OPTIMIZATION: Simplified mana checking - just need to know if castable
            can_play_land = (lands_played == 0) and is_your_turn and is_main_phase and stack_empty
            
            for card in hand:
                name = card.get("name", "Unknown")
                cost = card.get("mana_cost", "")
                type_line = card.get("type_line", "").lower()
                oracle_text = card.get("oracle_text", "")
                oracle_lower = oracle_text.lower()

                # OPTIMIZATION: Simplified timing - just need instant vs sorcery
                is_instant = "instant" in type_line or "flash" in oracle_lower
                timing = "I" if is_instant else "S"

                # OPTIMIZATION: Simplified CMC calculation
                cmc = 0
                reqs = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0, "C": 0}
                if cost:
                    generic = re.findall(r'\{(\d+)\}', cost)
                    cmc += sum(int(g) for g in generic)
                    for color in "WUBRGC":
                        count = len(re.findall(rf"\{{{color}\}}", cost))
                        reqs[color] += count
                        cmc += count
                    hybrid = re.findall(r'\{[^}]+/[^}]+\}', cost)
                    cmc += len(hybrid)

                # OPTIMIZATION: Simple castability check
                castable = ""
                if "land" in type_line:
                    castable = "LAND" if can_play_land else "HOLD"
                elif total_mana >= cmc:
                    # Color check (simplified - just check if we have any colored pips we can't pay)
                    color_ok = all(mana_pool.get(c, 0) + mana_pool.get("Any", 0) >= reqs[c]
                                   for c in "WUBRGC" if reqs[c] > 0)
                    castable = "OK" if color_ok else f"NEED:{cmc - total_mana}"
                else:
                    castable = f"NEED:{cmc - total_mana}"

                # OPTIMIZATION: Terse removal analysis - show kill RANGE, not every target
                removal_info = ""
                damage_match = re.search(r'deals?\s+(\d+)\s+damage', oracle_lower)
                minus_match = re.search(r'gets?\s+(-\d+)/(-\d+)', oracle_lower)
                is_destroy = "destroy target creature" in oracle_lower
                is_exile = "exile target creature" in oracle_lower

                if damage_match or minus_match or is_destroy or is_exile:
                    if is_destroy or is_exile:
                        removal_info = " [RM:any]"
                    elif damage_match:
                        dmg = int(damage_match.group(1))
                        removal_info = f" [RM:<={dmg}T]"
                    elif minus_match:
                        tough_reduction = abs(int(minus_match.group(2)))
                        removal_info = f" [RM:<={tough_reduction}T]"

                # OPTIMIZATION: Only show oracle text for non-basic, non-land cards with relevant text
                is_basic_land = "land" in type_line and ("basic" in type_line or
                                                         name in ["Plains", "Island", "Swamp", "Mountain", "Forest"])
                show_oracle = oracle_text and not is_basic_land and len(oracle_text) < 150

                # OPTIMIZATION: Compact card display
                lines.append(f"  {name} {cost} [{timing},{castable}]{removal_info}")
                if show_oracle:
                    # OPTIMIZATION: Remove reminder text to save tokens
                    oracle_compact = self._remove_reminder_text(oracle_text)
                    if len(oracle_compact) > 100:
                        oracle_compact = oracle_compact[:97] + "..."
                    lines.append(f"    {oracle_compact}")
        else:
            lines.append("  (empty)")

        # OPTIMIZATION: Compact stack display
        stack = game_state.get("stack", [])
        if stack:
            stack_items = []
            for card in stack:
                name = card.get("name", "Unknown")
                owner = "Y" if card.get("owner_seat_id") == local_seat else "O"
                stack_items.append(f"{owner}:{name}")
            lines.append(f"Stack: {' > '.join(stack_items)}")

        # OPTIMIZATION: Compact graveyard counts
        graveyard = game_state.get("graveyard", [])
        if graveyard:
            your_gy = len([c for c in graveyard if c.get("owner_seat_id") == local_seat])
            opp_gy = len([c for c in graveyard if c.get("owner_seat_id") != local_seat])
            if your_gy > 0 or opp_gy > 0:
                lines.append(f"GY: Y={your_gy} O={opp_gy}")

        # Command zone (Commander/Brawl)
        command = game_state.get("command", [])
        if command:
            cmd_names = [c.get("name", "Unknown") for c in command]
            lines.append(f"CMD: {', '.join(cmd_names)}")

        return "\n".join(lines)

    def _extract_card_name_words(self, game_state: dict[str, Any]) -> set[str]:
        """Extract all words from card names in the current game state.

        These words are excluded from overuse tracking since they're card names.
        """
        import re
        card_words: set[str] = set()

        # Collect card names from all zones
        for zone in ["battlefield", "hand", "graveyard", "stack", "exile", "command"]:
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

        # PHASE 2: Inject decision-specific guidance when a decision is pending
        decision_context = game_state.get("decision_context")
        if decision_context:
            dec_type = decision_context.get("type", "unknown")
            decision_guidance = DECISION_PROMPTS.get(dec_type)
            if decision_guidance:
                system_prompt += f"\n\n{decision_guidance}"
                logger.debug(f"Injected decision prompt for type: {dec_type}")

        # Build user message
        if question:
            user_message = f"{context}\n\nThe player asks: {question}"
        elif trigger:
            trigger_descriptions = {
                "new_turn": "Your turn just started (Main 1). What is the ONE best play right now?",
                "land_played": "A land was just played. What is the ONE next play?",
                "spell_resolved": "A spell just resolved. What is the ONE next play?",
                "priority_gained": "You have priority. Respond or pass?",
                "combat_attackers": "Combat: Declare attackers. Which creatures should attack?",
                "combat_blockers": "Combat: Opponent is attacking. How should you block?",
                "low_life": "Your life is dangerously low! What's the survival plan?",
                "opponent_low_life": "Opponent's life is low — can you finish them?",
                "stack_spell": "Something was just cast. Respond or let it resolve?",
                "stack_spell_yours": "Your spell is on the stack. Pass priority or hold?",
                "stack_spell_opponent": "Opponent just cast something. Respond or let it resolve?",
                "user_request": "Give quick strategic advice for this moment.",
                "decision_required": "Decision required (scry, discard, target, mulligan, etc). What should the player choose?",
                "threat_detected": "ALERT: A dangerous card just hit the battlefield!",
            }
            trigger_desc = trigger_descriptions.get(trigger, f"Trigger: {trigger}")
            user_message = f"{context}\n\n{trigger_desc}"
        else:
            user_message = f"{context}\n\nWhat's the best play right now?"

        # OPTIMIZATION: Log prompt size with token estimate
        prompt_chars = len(system_prompt) + len(user_message)
        prompt_tokens_est = self._estimate_tokens(system_prompt + user_message)
        context_lines = context.count('\n') + 1
        logger.info(f"[PROMPT] {context_lines} lines, {prompt_chars} chars, ~{prompt_tokens_est} tokens | context: {context_time:.1f}ms")

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

        # Inject deck strategy if available — instruct model to reference it
        if self._deck_strategy:
            effective_system_prompt += (
                f"\n\nDECK STRATEGY:\n{self._deck_strategy}"
                "\n\nConnect your advice to this strategy — briefly explain WHY a play "
                "matters for the deck's game plan (e.g. 'Cast X to trigger cascade into combo pieces')."
            )

        # Re-inject blacklisted words and decision guidance into effective prompt
        if blacklisted:
            avoid_list = ", ".join(blacklisted)
            effective_system_prompt += f"\n\nIMPORTANT: Avoid using these overused words: {avoid_list}. Use different phrasing."

        if decision_context:
            dec_type = decision_context.get("type", "unknown")
            decision_guidance = DECISION_PROMPTS.get(dec_type)
            if decision_guidance:
                effective_system_prompt += f"\n\n{decision_guidance}"

        # Get response and track word usage (excluding card names)
        api_start = time.perf_counter()
        response = self._backend.complete(effective_system_prompt, user_message)
        api_time = (time.perf_counter() - api_start) * 1000

        # POST-PROCESSING: Validate and fix common LLM issues (especially for smaller models)
        response = self._postprocess_advice(response, game_state)

        self._word_tracker.record(response, exclude_words=card_words)

        total_time = (time.perf_counter() - total_start) * 1000
        logger.info(f"[TIMING] API call: {api_time:.0f}ms, total: {total_time:.0f}ms, response: {len(response)} chars")

        return response

    def get_advice_audio(
        self,
        game_state: dict[str, Any],
        trigger: Optional[str] = None,
        style: str = "concise",
        voice: str = "Kore"
    ) -> "tuple[np.ndarray, int] | None":
        """Get coaching advice as direct audio from the LLM backend.

        Uses the same prompt-building logic as get_advice() but calls
        complete_audio() to get audio output directly from the model.
        No postprocessing or word tracking (can't modify audio).

        Args:
            game_state: Dict from get_game_state() MCP tool
            trigger: Optional trigger name (e.g., "new_turn", "combat_attackers")
            style: Advice style ("concise" or "verbose")
            voice: Gemini voice name (default: Kore)

        Returns:
            Tuple of (samples as float32 numpy array, sample_rate) on success,
            None on any error (caller should fall back to text + TTS).
        """
        import time
        total_start = time.perf_counter()

        # Build context (same as get_advice)
        context = self._format_game_context(game_state)

        # Get card name words to exclude from overuse tracking
        card_words = self._extract_card_name_words(game_state)

        # Check for overused words
        blacklisted = self._word_tracker.get_blacklisted(exclude_words=card_words)

        # Build user message (same logic as get_advice)
        if trigger:
            trigger_descriptions = {
                "new_turn": "Your turn just started (Main 1). What is the ONE best play right now?",
                "land_played": "A land was just played. What is the ONE next play?",
                "spell_resolved": "A spell just resolved. What is the ONE next play?",
                "priority_gained": "You have priority. Respond or pass?",
                "combat_attackers": "Combat: Declare attackers. Which creatures should attack?",
                "combat_blockers": "Combat: Opponent is attacking. How should you block?",
                "low_life": "Your life is dangerously low! What's the survival plan?",
                "opponent_low_life": "Opponent's life is low — can you finish them?",
                "stack_spell": "Something was just cast. Respond or let it resolve?",
                "stack_spell_yours": "Your spell is on the stack. Pass priority or hold?",
                "stack_spell_opponent": "Opponent just cast something. Respond or let it resolve?",
                "user_request": "Give quick strategic advice for this moment.",
                "decision_required": "Decision required (scry, discard, target, mulligan, etc). What should the player choose?",
                "threat_detected": "ALERT: A dangerous card just hit the battlefield!",
            }
            trigger_desc = trigger_descriptions.get(trigger, f"Trigger: {trigger}")
            user_message = f"{context}\n\n{trigger_desc}"
        else:
            user_message = f"{context}\n\nWhat's the best play right now?"

        # Select system prompt based on style (same as get_advice)
        style_key = style.lower()
        prompts = {
            "concise": CONCISE_SYSTEM_PROMPT,
            "normal": DEFAULT_SYSTEM_PROMPT,
            "explain": DEFAULT_SYSTEM_PROMPT.replace("Keep responses concise (2-3 sentences max)", "Explain your reasoning clearly but briefly.") + "\nInclude a short explanation of WHY this is the best line.",
            "pirate": "You are a ruthless pirate captain coaching a swabby! Speak like a pirate! Yarr! Keep it short!",
        }
        effective_system_prompt = prompts.get(style_key, CONCISE_SYSTEM_PROMPT)

        # Inject deck strategy if available — instruct model to reference it
        if self._deck_strategy:
            effective_system_prompt += (
                f"\n\nDECK STRATEGY:\n{self._deck_strategy}"
                "\n\nConnect your advice to this strategy — briefly explain WHY a play "
                "matters for the deck's game plan (e.g. 'Cast X to trigger cascade into combo pieces')."
            )

        # Inject blacklisted words
        if blacklisted:
            avoid_list = ", ".join(blacklisted)
            effective_system_prompt += f"\n\nIMPORTANT: Avoid using these overused words: {avoid_list}. Use different phrasing."

        # Inject decision guidance
        decision_context = game_state.get("decision_context")
        if decision_context:
            dec_type = decision_context.get("type", "unknown")
            decision_guidance = DECISION_PROMPTS.get(dec_type)
            if decision_guidance:
                effective_system_prompt += f"\n\n{decision_guidance}"

        # Log prompt size
        prompt_chars = len(effective_system_prompt) + len(user_message)
        prompt_tokens_est = self._estimate_tokens(effective_system_prompt + user_message)
        logger.info(f"[PROMPT+AUDIO] {prompt_chars} chars, ~{prompt_tokens_est} tokens")

        # Call the audio backend
        result = self._backend.complete_audio(effective_system_prompt, user_message, voice=voice)

        total_time = (time.perf_counter() - total_start) * 1000
        if result is not None:
            logger.info(f"[TIMING+AUDIO] total: {total_time:.0f}ms, samples: {len(result[0])}")
        else:
            logger.warning(f"[TIMING+AUDIO] total: {total_time:.0f}ms, result: None (will fallback)")

        return result

    def _postprocess_advice(self, advice: str, game_state: dict[str, Any]) -> str:
        """Post-process LLM advice to fix common issues with smaller models.
        
        1. Remove 'Play [Land]' suggestions when no land is in hand
        2. Fix typos in card names using fuzzy matching
        """
        import re
        
        # Get cards in hand
        hand_cards = game_state.get("hand", [])
        hand_names = {c.get("name", "").lower() for c in hand_cards}
        
        # Get all card names in game state for fuzzy matching
        all_cards = []
        for zone in ["hand", "battlefield", "graveyard", "stack", "exile"]:
            all_cards.extend(game_state.get(zone, []))
        all_card_names = {c.get("name", "") for c in all_cards if c.get("name")}
        
        # Check for land names in hand
        land_types = {"forest", "island", "swamp", "mountain", "plains"}
        lands_in_hand = {name for name in hand_names if any(lt in name for lt in land_types)}
        
        # 1. Remove "Play [Land]" if no land in hand
        if not lands_in_hand:
            # Remove patterns like "Play Forest.", "Play Island,", "Play a land."
            advice = re.sub(r"Play\s+(Forest|Island|Swamp|Mountain|Plains|a land)[.,]?\s*", "", advice, flags=re.IGNORECASE)
            # Clean up any resulting double spaces or leading/trailing spaces
            advice = re.sub(r"\s+", " ", advice).strip()
        
        # 2. Fix typos in card names using simple fuzzy matching
        # Common typos seen from Gemma 3N:
        typo_fixes = {
            "brerak out": "Break Out",
            "braimble familiar": "Bramble Familiar",
            "llanowar eves": "Llanowar Elves",
            "llanowar elfs": "Llanowar Elves",
            "craterhood behemoth": "Craterhoof Behemoth",
            "creterhoof behemoth": "Craterhoof Behemoth",
            "crterhoof behemoth": "Craterhoof Behemoth",
            "baadgermole cub": "Badgermole Cub",
            "badgremole cub": "Badgermole Cub",
        }
        
        advice_lower = advice.lower()
        for typo, correct in typo_fixes.items():
            if typo in advice_lower:
                # Case-insensitive replacement
                pattern = re.compile(re.escape(typo), re.IGNORECASE)
                advice = pattern.sub(correct, advice)
        
        # Also try to match against actual card names in game state
        # Split advice into words and check for near-matches
        for card_name in all_card_names:
            if len(card_name) < 4:
                continue  # Skip short names to avoid false matches
            # Check if card name appears with typos (simple Levenshtein-like check)
            card_words = card_name.lower().split()
            for word in card_words:
                if len(word) < 4:
                    continue
                # Look for similar words in advice
                advice_words = advice.lower().split()
                for i, advice_word in enumerate(advice_words):
                    if len(advice_word) >= 4 and self._is_similar(word, advice_word):
                        # Replace the typo with correct spelling
                        # Find the actual word in original advice and replace
                        original_words = advice.split()
                        if i < len(original_words):
                            # Only replace if first letter matches (to avoid false positives)
                            if original_words[i][0].lower() == word[0].lower():
                                original_words[i] = word.capitalize() if original_words[i][0].isupper() else word
                                advice = " ".join(original_words)
        
        # 3. Remove Cast suggestions for cards that cost more mana than available
        # Calculate available mana (lands on battlefield + land drop potential)
        battlefield = game_state.get("battlefield", [])
        local_seat = None
        for p in game_state.get("players", []):
            if p.get("is_local"):
                local_seat = p.get("seat_id")
                break
        
        # Count untapped lands we control
        untapped_lands = 0
        for card in battlefield:
            if card.get("controller_seat_id") == local_seat or card.get("owner_seat_id") == local_seat:
                type_line = card.get("type_line", "").lower()
                if "land" in type_line and not card.get("is_tapped"):
                    untapped_lands += 1
        
        # Check if we have a land in hand (potential +1 mana)
        has_land_in_hand = lands_in_hand  # already computed above
        potential_mana = untapped_lands + (1 if has_land_in_hand else 0)
        
        # Check each card in hand for mana cost violations
        for card in hand_cards:
            card_name = card.get("name", "")
            mana_cost = card.get("mana_cost", "")
            if not card_name or not mana_cost:
                continue
            
            # Parse CMC from mana cost (simple heuristic)
            cmc = 0
            import re as re_inner
            # Count {X} symbols
            symbols = re_inner.findall(r'\{([^}]+)\}', mana_cost)
            for sym in symbols:
                if sym.isdigit():
                    cmc += int(sym)
                elif sym in ['W', 'U', 'B', 'R', 'G', 'C']:
                    cmc += 1
                elif '/' in sym:  # Hybrid like {R/G}
                    cmc += 1
            
            # If this card costs more than we can have, remove Cast suggestions for it
            if cmc > potential_mana:
                # Remove "Cast [Card Name]" from advice
                pattern = re.compile(rf"Cast\s+{re.escape(card_name)}[.,]?\s*", re.IGNORECASE)
                if pattern.search(advice):
                    advice = pattern.sub("", advice)
                    logger.debug(f"Removed uncastable suggestion: {card_name} (needs {cmc}, have {potential_mana})")
        
        # Clean up double spaces
        advice = re.sub(r"\s+", " ", advice).strip()
        
        return advice
    
    def _is_similar(self, a: str, b: str, threshold: float = 0.7) -> bool:
        """Check if two strings are similar using simple character overlap."""
        if a == b:
            return True
        if abs(len(a) - len(b)) > 3:
            return False
        # Count matching characters
        matches = sum(1 for c1, c2 in zip(a.lower(), b.lower()) if c1 == c2)
        similarity = matches / max(len(a), len(b))
        return similarity >= threshold

    def complete_with_image(self, system_prompt: str, user_message: str, image_data: bytes) -> str:
        """Call complete_with_image on backend if supported."""
        if hasattr(self._backend, 'complete_with_image'):
            return self._backend.complete_with_image(system_prompt, user_message, image_data)
        logger.error(f"Backend {type(self._backend).__name__} does not support complete_with_image")
        return "Image analysis not supported by current backend."


class GameStateTrigger:
    """Detects trigger conditions by comparing game states."""

    # Tier list of dangerous cards that warrant immediate warning
    # Format: card_name -> brief description of the threat
    THREAT_CARDS = {
        # Board wipes
        "Wrath of God": "Board wipe! Destroys all creatures.",
        "Damnation": "Board wipe! Destroys all creatures.",
        "Farewell": "Exiles ALL permanents of chosen types!",
        "Sunfall": "Exiles all creatures, makes a big token.",
        "Depopulate": "Board wipe, draws if you have multicolor.",
        "Temporary Lockdown": "Exiles all permanents MV 2 or less!",
        "Meticulous Archive": "Can find board wipes or removal.",
        
        # Combo pieces / Must-answer threats
        "Sheoldred, the Apocalypse": "Drains 2 on your draws, heals on theirs!",
        "Atraxa, Grand Unifier": "Draws 10+ cards on ETB, lifelink flyer.",
        "Raffine, Scheming Seer": "Grows attackers and filters cards.",
        "The Wandering Emperor": "Flash! Can exile or make blockers anytime.",
        "Teferi, Time Raveler": "Shuts off your instant-speed plays!",
        "Narset, Parter of Veils": "You can only draw 1 card per turn!",
        "Omnath, Locus of Creation": "Massive value engine, gains life.",
        "Vorinclex, Voice of Hunger": "Doubles their counters, halves yours.",
        
        # Powerful planeswalkers
        "Oko, Thief of Crowns": "Elks your best creatures!",
        "Karn, the Great Creator": "Shuts off artifacts, grabs from sideboard.",
        "Wrenn and Six": "Recurring lands and pinging creatures.",
        
        # Lock pieces
        "Drannith Magistrate": "You can't cast from graveyard/exile!",
        "Archon of Emeria": "Only 1 spell per turn, lands ETB tapped.",
        "Thalia, Guardian of Thraben": "Noncreature spells cost 1 more.",
        "Authority of the Consuls": "Your creatures ETB tapped.",
        "High Noon": "Only 1 spell per turn for everyone.",
        
        # Removal magnets
        "Questing Beast": "Can't be chumped, damages walkers!",
        "Elder Gargaroth": "Massive value every combat.",
        "Cruelty of Gix": "3-mode saga, steals creatures!",
    }
    
    def __init__(self, life_threshold: int = 5):
        """Initialize trigger detector.

        Args:
            life_threshold: Life total below which "low_life" triggers (default: 5)
        """
        self.life_threshold = life_threshold
        # Track threats we've already warned about (by instance_id)
        self._seen_threats: set[int] = set()

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

        # FIRST CONNECTION: If prev_state has no turn info but curr_state does,
        # we just connected mid-game. Fire a trigger to give immediate advice.
        prev_turn_num = prev_turn.get("turn_number", 0)
        curr_turn_num = curr_turn.get("turn_number", 0)
        curr_active = curr_turn.get("active_player", 0)
        
        if prev_turn_num == 0 and curr_turn_num > 0:
            # Just connected to an active game
            is_your_turn = curr_active == local_seat
            if is_your_turn:
                logger.info(f"First connection mid-game, triggering new_turn (turn {curr_turn_num})")
                triggers.append("new_turn")
            # Also check for pending decision on first connection
            pending = curr_state.get("pending_decision")
            if pending:
                logger.info(f"First connection with pending decision: {pending}")
                triggers.append("decision_required")

        # New turn detection
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

            logger.debug(f"Processing pending combat step: {step}, active={step_active}, step_is_your_turn={step_is_your_turn}, current_is_your_turn={is_your_turn}")

            # Double-check both the step's active player AND current turn state
            # This prevents stale pending steps from firing triggers after turn changes
            if "DeclareAttack" in step and step_is_your_turn and is_your_turn:
                if "combat_attackers" not in triggers:
                    logger.info(f"Combat attackers trigger from pending: {step}")
                    triggers.append("combat_attackers")
            elif "DeclareBlock" in step and not step_is_your_turn and not is_your_turn:
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

        # Stack spell detection - differentiate between your spells and opponent's
        prev_stack = prev_state.get("stack", [])
        curr_stack = curr_state.get("stack", [])
        if len(curr_stack) > len(prev_stack):
            # Check who owns the newest spell on the stack
            newest_spell = curr_stack[-1] if curr_stack else None
            if newest_spell:
                spell_owner = newest_spell.get("owner_seat_id")
                if spell_owner == local_seat:
                    triggers.append("stack_spell_yours")
                else:
                    triggers.append("stack_spell_opponent")

        # Land played detection - only on your turn, only in main phases
        if is_your_turn and "Main" in curr_phase:
            prev_battlefield = prev_state.get("battlefield", [])
            curr_battlefield = curr_state.get("battlefield", [])

            # Count YOUR lands before and after
            prev_land_count = sum(1 for obj in prev_battlefield
                                  if obj.get("owner_seat_id") == local_seat
                                  and "land" in obj.get("type_line", "").lower())
            curr_land_count = sum(1 for obj in curr_battlefield
                                  if obj.get("owner_seat_id") == local_seat
                                  and "land" in obj.get("type_line", "").lower())

            if curr_land_count > prev_land_count:
                logger.info(f"Land played trigger: {prev_land_count} -> {curr_land_count}")
                triggers.append("land_played")

        # Spell resolved detection - your spell left the stack on your turn
        if is_your_turn and len(curr_stack) < len(prev_stack):
            # Check if a spell you owned just resolved
            prev_your_spells = [s for s in prev_stack if s.get("owner_seat_id") == local_seat]
            curr_your_spells = [s for s in curr_stack if s.get("owner_seat_id") == local_seat]
            if len(curr_your_spells) < len(prev_your_spells):
                # Your spell resolved - what's next?
                logger.info("Spell resolved trigger: your spell left the stack")
                triggers.append("spell_resolved")

        # THREAT DETECTION - warn about dangerous opponent cards
        opp_seat = curr_opp.get("seat_id") if curr_opp else None
        if opp_seat:
            curr_battlefield = curr_state.get("battlefield", [])
            for card in curr_battlefield:
                # Only check opponent's permanents
                controller = card.get("controller_seat_id") or card.get("owner_seat_id")
                if controller != opp_seat:
                    continue
                
                instance_id = card.get("instance_id")
                card_name = card.get("name", "")
                
                # Check if this is a threat card we haven't warned about
                if card_name in self.THREAT_CARDS and instance_id not in self._seen_threats:
                    self._seen_threats.add(instance_id)
                    # Store threat info for the standalone coach to retrieve
                    self._last_threat = {
                        "name": card_name,
                        "warning": self.THREAT_CARDS[card_name]
                    }
                    logger.info(f"Threat detected: {card_name} - {self.THREAT_CARDS[card_name]}")
                    triggers.append("threat_detected")

        return triggers
