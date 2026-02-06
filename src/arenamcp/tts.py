"""Text-to-speech synthesis using Kokoro ONNX.

This module provides TTS using the kokoro-onnx library for offline,
low-latency speech synthesis with high-quality neural voices.

NOTE: Model files must be downloaded manually (~300MB total):
- kokoro-v1.0.onnx: https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
- voices-v1.0.bin: https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

Place files in ~/.cache/kokoro/ or specify paths explicitly.
"""

import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import logging

logger = logging.getLogger(__name__)


# Default model cache location
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "kokoro"
MODEL_FILE = "kokoro-v1.0.onnx"
VOICES_FILE = "voices-v1.0.bin"

# Model download URLs for error messages
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"


def _extract_audio_from_response(response) -> bytes | None:
    """Extract audio bytes from a Gemini API response.

    Searches through response candidates and parts to find inline_data
    with audio content. Returns the raw bytes, or None if not found.
    """
    try:
        if not hasattr(response, 'candidates') or not response.candidates:
            return None
        for candidate in response.candidates:
            if not hasattr(candidate, 'content') or not candidate.content:
                continue
            if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
                continue
            for part in candidate.content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    if hasattr(part.inline_data, 'data') and part.inline_data.data:
                        return part.inline_data.data
    except Exception as e:
        logger.error(f"Error extracting audio from response: {e}")
    return None


def _decode_audio_bytes(audio_data: bytes, default_rate: int = 24000) -> tuple[np.ndarray, int]:
    """Decode audio bytes (WAV or raw PCM) to numpy float32 array.

    Tries WAV format first. If that fails (not RIFF header), falls back
    to interpreting as raw 16-bit PCM at the default sample rate.

    Returns:
        Tuple of (samples as float32, sample_rate).
    """
    import io
    import wave

    # Try WAV first
    if audio_data[:4] == b'RIFF':
        try:
            with wave.open(io.BytesIO(audio_data), "rb") as wf:
                sample_rate = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                raw_frames = wf.readframes(wf.getnframes())

            if sampwidth == 2:
                samples = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 4:
                samples = np.frombuffer(raw_frames, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                samples = np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)

            return samples, sample_rate
        except Exception as e:
            logger.warning(f"WAV decode failed despite RIFF header: {e}, trying raw PCM")

    # Fallback: raw 16-bit PCM at default sample rate
    logger.info(f"Decoding as raw PCM16 ({len(audio_data)} bytes, {default_rate}Hz)")
    samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, default_rate


class KokoroTTS:
    """Text-to-speech synthesizer using Kokoro ONNX.

    Uses lazy model loading to avoid startup delay. The model is only
    loaded when the first synthesis is requested.

    Example:
        tts = KokoroTTS()
        # First call loads model (~300MB)
        samples, sample_rate = tts.synthesize("Hello, how can I help?")
        sd.play(samples, sample_rate)
        sd.wait()

    Note:
        Model files must be downloaded manually. See module docstring
        for download URLs. Default location: ~/.cache/kokoro/
    """

    # Kokoro outputs at fixed 24kHz sample rate
    SAMPLE_RATE = 24000

    def __init__(
        self,
        model_path: Optional[str] = None,
        voices_path: Optional[str] = None,
        voice: str = "am_adam",
        speed: float = 1.0,
        lang: str = "en-us",
    ) -> None:
        """Initialize the TTS synthesizer.

        Args:
            model_path: Path to kokoro-v1.0.onnx file. Defaults to
                       ~/.cache/kokoro/kokoro-v1.0.onnx
            voices_path: Path to voices-v1.0.bin file. Defaults to
                        ~/.cache/kokoro/voices-v1.0.bin
            voice: Voice ID to use. Default 'am_adam' (American Male Adam).
            speed: Speech speed multiplier. Default 1.0.
            lang: Language code. Default 'en-us'.
        """
        # Resolve model paths
        if model_path is None:
            self._model_path = DEFAULT_CACHE_DIR / MODEL_FILE
        else:
            self._model_path = Path(model_path)

        if voices_path is None:
            self._voices_path = DEFAULT_CACHE_DIR / VOICES_FILE
        else:
            self._voices_path = Path(voices_path)

        self._voice = voice
        self._speed = speed
        self._lang = lang

        # Lazy-loaded model
        self._kokoro: Optional[object] = None
        self._load_lock = threading.Lock()

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the Kokoro model on first use.

        Raises:
            FileNotFoundError: If model files are not found, with
                             download instructions.
            ImportError: If kokoro-onnx is not installed.
        """
        if self._kokoro is not None:
            return

        with self._load_lock:
            # Double-check after acquiring lock
            if self._kokoro is not None:
                return

            # Check model files exist
            missing_files = []
            if not self._model_path.exists():
                missing_files.append(
                    f"Model file not found: {self._model_path}\n"
                    f"  Download from: {MODEL_URL}"
                )
            if not self._voices_path.exists():
                missing_files.append(
                    f"Voices file not found: {self._voices_path}\n"
                    f"  Download from: {VOICES_URL}"
                )

            if missing_files:
                # Create cache directory hint
                cache_hint = (
                    f"\nCreate directory and download files:\n"
                    f"  mkdir -p {DEFAULT_CACHE_DIR}\n"
                    f"  cd {DEFAULT_CACHE_DIR}\n"
                    f"  curl -LO {MODEL_URL}\n"
                    f"  curl -LO {VOICES_URL}"
                )
                raise FileNotFoundError(
                    "Kokoro TTS model files not found:\n\n"
                    + "\n\n".join(missing_files)
                    + cache_hint
                )

            # Import and load kokoro
            try:
                from kokoro_onnx import Kokoro
            except ImportError as e:
                raise ImportError(
                    "kokoro-onnx not installed. Run: pip install kokoro-onnx"
                ) from e

            self._kokoro = Kokoro(
                str(self._model_path),
                str(self._voices_path),
            )

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text to audio samples.

        Args:
            text: Text to synthesize. Should be reasonable length
                 (sentences, not paragraphs) for best quality.

        Returns:
            Tuple of (samples, sample_rate) where samples is a numpy
            array of float32 audio data and sample_rate is always 24000.

        Raises:
            FileNotFoundError: If model files are not found.
            ImportError: If kokoro-onnx is not installed.
        """
        self._ensure_model_loaded()

        if not text or not text.strip():
            return np.array([], dtype=np.float32), self.SAMPLE_RATE

        # Generate audio using Kokoro
        samples, sample_rate = self._kokoro.create(
            text,
            voice=self._voice,
            speed=self._speed,
            lang=self._lang,
        )

        return samples, sample_rate


class AzureTTS:
    """Text-to-speech using Azure OpenAI Audio API."""

    def __init__(self, voice: str = "alloy", speed: float = 1.0):
        self._voice = voice.replace("azure/", "")
        self._speed = speed
        self._client = None
        
    def _get_client(self):
        if self._client is None:
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
            )
        return self._client

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text using Azure OpenAI TTS."""
        import io
        import soundfile as sf
        
        if not text or not text.strip():
            return np.array([], dtype=np.float32), 24000

        try:
            client = self._get_client()
            response = client.audio.speech.create(
                model="tts-1",
                voice=self._voice,
                input=text,
                speed=self._speed,
                response_format="flac"  # Use FLAC for lossless decoding support in soundfile
            )
            
            # Read from memory
            # openai returns binary content in response.content
            data = io.BytesIO(response.content)
            data.seek(0)
            
            samples, sample_rate = sf.read(data)
            return samples.astype(np.float32), sample_rate
            
        except Exception as e:
            logger.error(f"Azure TTS failed: {e}")
            return np.array([], dtype=np.float32), 24000


class GeminiTTS:
    """Text-to-speech using Gemini's native TTS model."""

    def __init__(self, voice: str = "Kore", speed: float = 1.0):
        self._voice = voice.replace("gemini/", "")
        self._speed = speed
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                logger.error("GOOGLE_API_KEY not set, Gemini TTS unavailable")
                return None
            self._client = genai.Client(api_key=api_key)
        return self._client

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text using Gemini TTS."""
        if not text or not text.strip():
            return np.array([], dtype=np.float32), 24000

        try:
            client = self._get_client()
            if client is None:
                return np.array([], dtype=np.float32), 24000

            from google.genai import types

            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=self._voice,
                            )
                        )
                    ),
                ),
            )

            # Extract audio data from response (robust parsing)
            audio_data = _extract_audio_from_response(response)
            if audio_data is None:
                logger.error("Gemini TTS: no audio data in response")
                return np.array([], dtype=np.float32), 24000

            samples, sample_rate = _decode_audio_bytes(audio_data)
            return samples, sample_rate

        except Exception as e:
            logger.error(f"Gemini TTS failed: {e}")
            return np.array([], dtype=np.float32), 24000


class VoiceOutput:
    """Unified voice output interface for TTS playback.

    Provides speak(), speak_async(), and stop() methods for easy
    text-to-speech with audio playback. Counterpart to VoiceInput.

    Example:
        output = VoiceOutput()
        output.speak("Hello, I'm your coach!")  # Blocks until done

        output.speak_async("This plays in background")
        # ... do other work ...
        output.stop()  # Interrupt if needed

    Note:
        First call loads TTS model (~300MB). Model files must be
        downloaded manually - see KokoroTTS docstring.
    """

    # Available Kokoro voices (name, description)
    VOICES = [
        ("af_heart", "American Female - Heart (Grade A)"),
        ("af_bella", "American Female - Bella"),
        ("af_nicole", "American Female - Nicole"),
        ("af_sarah", "American Female - Sarah"),
        ("af_sky", "American Female - Sky"),
        ("am_adam", "American Male - Adam"),
        ("am_michael", "American Male - Michael"),
        ("bf_emma", "British Female - Emma"),
        ("bf_isabella", "British Female - Isabella"),
        ("bm_george", "British Male - George"),
        ("bm_lewis", "British Male - Lewis"),
        ("azure/alloy", "Azure - Alloy (Neutral)"),
        ("azure/echo", "Azure - Echo (Male)"),
        ("azure/fable", "Azure - Fable (British-ish)"),
        ("azure/onyx", "Azure - Onyx (Deep Male)"),
        ("azure/nova", "Azure - Nova (Female)"),
        ("azure/shimmer", "Azure - Shimmer (Female)"),
        ("gemini/Kore", "Gemini - Kore (Female)"),
        ("gemini/Puck", "Gemini - Puck (Male)"),
        ("gemini/Charon", "Gemini - Charon (Male)"),
        ("gemini/Fenrir", "Gemini - Fenrir (Male)"),
        ("gemini/Aoede", "Gemini - Aoede (Female)"),
    ]

    def __init__(
        self,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> None:
        """Initialize VoiceOutput.

        Args:
            voice: Kokoro voice ID. If None, loads from settings
                  (default: 'am_adam' - American Male Adam).
            speed: Speech speed multiplier. If None, loads from settings
                  (default: 1.0).
        """
        # Load from settings if not specified
        from arenamcp.settings import get_settings
        settings = get_settings()

        self._voice = voice if voice is not None else settings.get("voice", "am_adam")
        self._speed = speed if speed is not None else settings.get("voice_speed", 1.0)
        self._device_index = settings.get("device_index", None)
        self._voice_index = 0  # Index into VOICES list
        self._muted = settings.get("muted", False)
        self._settings = settings

        # Find initial voice index
        for i, (vid, _) in enumerate(self.VOICES):
            if vid == self._voice:
                self._voice_index = i
                break

        # Lazy-initialized TTS
        self._tts_engine = None  # Generic engine holder

        # Playback state
        self._lock = threading.Lock()
        self._speak_lock = threading.Lock()  # Prevents overlapping speech
        self._is_speaking = False
        self._stop_requested = False
        self._stream: Optional[sd.OutputStream] = None
        self._playback_thread: Optional[threading.Thread] = None

    def _ensure_tts(self) -> None:
        """Initialize TTS on first use."""
        if self._tts_engine is None:
            if self._voice.startswith("gemini/"):
                self._tts_engine = GeminiTTS(voice=self._voice, speed=self._speed)
            elif self._voice.startswith("azure/"):
                self._tts_engine = AzureTTS(voice=self._voice, speed=self._speed)
            else:
                self._tts_engine = KokoroTTS(voice=self._voice, speed=self._speed)

    @property
    def muted(self) -> bool:
        """Check if output is muted."""
        return self._muted

    @property
    def current_voice(self) -> tuple[str, str]:
        """Get current voice (id, description)."""
        return self.VOICES[self._voice_index]

    def toggle_mute(self) -> bool:
        """Toggle mute state.

        Returns:
            New mute state (True if now muted).
        """
        self._muted = not self._muted
        if self._muted:
            self.stop()  # Stop any current playback

        # Persist setting
        self._settings.set("muted", self._muted)
        return self._muted

    def next_voice(self, step: int = 1) -> tuple[str, str]:
        """Cycle to the next voice.

        Args:
            step: Number of voices to skip (default 1).

        Returns:
            Tuple of (voice_id, description) for the new voice.
        """
        self._voice_index = (self._voice_index + step) % len(self.VOICES)
        voice_id, description = self.VOICES[self._voice_index]
        self._voice = voice_id

        # Recreate TTS with new voice
        if self._tts_engine is not None:
             # Force re-init with new voice
            self._tts_engine = None
            self._ensure_tts()

        # Persist setting
        self._settings.set("voice", voice_id)
        return (voice_id, description)

    def set_voice(self, voice_id: str) -> None:
        """Set the current voice directly by ID.
        
        Args:
            voice_id: The ID of the voice to set (e.g. 'am_adam', 'azure/alloy')
        """
        # Find index
        found = False
        for i, (vid, _) in enumerate(self.VOICES):
            if vid == voice_id:
                self._voice_index = i
                self._voice = voice_id
                found = True
                break
        
        if not found:
            logger.warning(f"Voice {voice_id} not found, ignoring.")
            return

        # Recreate TTS engine
        if self._tts_engine is not None:
            self._tts_engine = None
            self._ensure_tts()
            
        self._settings.set("voice", voice_id)

    @property
    def is_speaking(self) -> bool:
        """Check if audio is currently playing.

        Returns:
            True if speak() or speak_async() is actively playing audio.
        """
        with self._lock:
            return self._is_speaking

    def _clean_text(self, text: str) -> str:
        """Remove markdown and special characters that TTS shouldn't pronounce."""
        # Remove asterisks (bold/italic)
        text = text.replace("**", "").replace("*", "")
        # Remove hash (headers)
        text = text.replace("##", "").replace("#", "")
        # Remove backticks (code)
        text = text.replace("```", "").replace("`", "")
        # Remove ellipsis to prevent "dot dot dot" or "d d d"
        text = text.replace("...", " ")
        return text

    def speak(self, text: str, blocking: bool = True) -> None:
        """Synthesize and play text as speech.

        Args:
            text: Text to speak.
            blocking: If True (default), block until audio finishes.
                     If False, same as speak_async().

        Raises:
            FileNotFoundError: If TTS model files not found.
            sd.PortAudioError: If no audio output device available.
        """
        # Skip if muted
        if self._muted:
            return
            
        text = self._clean_text(text)

        if not blocking:
            self.speak_async(text)
            return

        self._ensure_tts()

        if not text or not text.strip():
            return

        # Use speak lock to prevent overlapping speech from multiple threads
        with self._speak_lock:
            # Stop any existing playback
            self.stop()

            # Synthesize
            samples, sample_rate = self._tts_engine.synthesize(text)
            if len(samples) == 0:
                return

            with self._lock:
                self._is_speaking = True
                self._stop_requested = False

            try:
                # Play with sounddevice (blocking)
                logger.info(f"Speaking (device={self._device_index}): {text[:50]}...")
                sd.play(samples, sample_rate, device=self._device_index)
                sd.wait()
            except sd.PortAudioError as e:
                # No audio device - fail gracefully
                logger.error(f"Audio playback failed: {e}")
            finally:
                with self._lock:
                    self._is_speaking = False

    def speak_async(self, text: str) -> None:
        """Synthesize and play text without blocking.

        Returns immediately. Use is_speaking property to check status
        or stop() to interrupt.

        Args:
            text: Text to speak.

        Raises:
            FileNotFoundError: If TTS model files not found.
        """
        self._ensure_tts()

        if not text or not text.strip():
            return

        # Stop any existing playback
        self.stop()

        # Synthesize
        samples, sample_rate = self._tts_engine.synthesize(text)
        if len(samples) == 0:
            return

        def _playback_worker():
            with self._lock:
                self._is_speaking = True
                self._stop_requested = False

            try:
                # Use callback-based playback for async
                idx = [0]  # Mutable container for closure
                finished = threading.Event()

                def callback(outdata, frames, time_info, status):
                    with self._lock:
                        if self._stop_requested:
                            outdata.fill(0)
                            raise sd.CallbackStop()

                    start = idx[0]
                    end = min(start + frames, len(samples))
                    out_frames = end - start

                    if out_frames > 0:
                        outdata[:out_frames, 0] = samples[start:end]
                    if out_frames < frames:
                        outdata[out_frames:] = 0
                        raise sd.CallbackStop()

                    idx[0] = end

                def finished_callback():
                    finished.set()

                with sd.OutputStream(
                    samplerate=sample_rate,
                    channels=1,
                    callback=callback,
                    finished_callback=finished_callback,
                    device=self._device_index,
                ):
                    logger.info(f"Speaking async (device={self._device_index}): {text[:50]}...")
                    finished.wait()

            except sd.PortAudioError as e:
                # No audio device - fail gracefully
                logger.error(f"Audio playback failed: {e}")
            finally:
                with self._lock:
                    self._is_speaking = False
                    self._playback_thread = None

        # Start playback in background thread
        with self._lock:
            self._playback_thread = threading.Thread(
                target=_playback_worker,
                daemon=True,
            )
            self._playback_thread.start()

    def stop(self) -> None:
        """Stop any ongoing playback.

        Safe to call even if nothing is playing.
        """
        with self._lock:
            self._stop_requested = True
            thread = self._playback_thread

        # Stop sounddevice blocking playback
        sd.stop()

        # Wait for async thread to finish
        if thread is not None:
            thread.join(timeout=1.0)

        with self._lock:
            self._is_speaking = False

    def play_audio_chunk(self, audio_data: bytes, sample_rate: int = 24000) -> None:
        """Queue raw audio chunk for playback (streaming mode).
        
        Args:
            audio_data: Raw PCM audio bytes (int16)
            sample_rate: Sample rate (default 24000 for Gemini/Kokoro)
        """
        if self._muted:
            return

        if not hasattr(self, "_stream_queue"):
            self._stream_queue = __import__("queue").Queue()
            self._stream_thread = None
            
        self._stream_queue.put((audio_data, sample_rate))
        
        # Start streaming worker if not running
        with self._lock:
            if self._stream_thread is None or not self._stream_thread.is_alive():
                self._stream_thread = threading.Thread(
                    target=self._stream_worker, 
                    daemon=True,
                    name="audio-stream-worker"
                )
                self._stream_thread.start()

    def _stream_worker(self):
        """Worker to play streamed audio chunks continuously."""
        import queue
        try:
            # Create a localized output stream 
            # We assume constant sample rate for the session (usually 24000)
            # If sample rate changes, we'd need to restart the stream, but for Gemini it's stable.
            
            # Peek first item to get sample rate
            try:
                first = self._stream_queue.get(timeout=0.1)
            except queue.Empty:
                return # Nothing to play

            _, rate = first
            # Put it back or process it? Processing it is better.
            
            with sd.OutputStream(
                samplerate=rate,
                channels=1,
                dtype='int16',
                device=self._device_index
            ) as stream:
                # Play first chunk
                stream.write(np.frombuffer(first[0], dtype=np.int16))
                
                # Keep playing from queue
                while True:
                    try:
                        # Wait briefly for next chunk
                        item = self._stream_queue.get(timeout=1.0) 
                        data, new_rate = item
                        
                        if new_rate != rate:
                            logger.warning("Sample rate changed in stream, restarting stream...")
                            # In a robust impl we would restart stream, here we just warn/try
                            
                        stream.write(np.frombuffer(data, dtype=np.int16))
                        
                    except queue.Empty:
                        # Silence logic or exit? 
                        # If queue is empty for 1s, assume stream pause and exit to save resources
                        break
                        
        except Exception as e:
            logger.error(f"Stream playback error: {e}")
        finally:
            with self._lock:
                self._stream_thread = None
