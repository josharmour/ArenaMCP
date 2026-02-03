"""Gemini Live backend for voice-to-voice coaching.

This module integrates Google's Gemini Multimodal Live API using the google-genai SDK.
It adapts the Google Live API to match the interface expected by StandaloneCoach
(similar to GPTRealtimeClient).
"""

import logging
import os
import threading
import queue
import time
import base64
import json
import asyncio
from typing import Optional, Callable, Any

import numpy as np

logger = logging.getLogger(__name__)

class GeminiLiveClient:
    """Client for Google Gemini Live API (Multimodal Live)."""

    def __init__(self, config: Any = None):
        """Initialize Gemini Live client. 
        
        Args:
           config: Optional configuration object (duck-typed with .instructions, .model, etc)
        """
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        self.model = getattr(config, "deployment", "gemini-2.0-flash-live-001") if config else "gemini-2.0-flash-live-001"
        self.instructions = getattr(config, "instructions", "You are a helpful assistant.") if config else ""
        
        self._connected = False
        self._session = None
        self._stop_event = threading.Event()
        self._send_queue = queue.Queue()
        
        # Threading
        self._recv_thread: Optional[threading.Thread] = None
        self._send_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._on_transcript: Optional[Callable[[str], None]] = None
        self._on_audio: Optional[Callable[[bytes], None]] = None
        self._on_response_done: Optional[Callable[[str, bytes], None]] = None
        self._on_error: Optional[Callable[[str], None]] = None

        # Buffers for current turn
        self._audio_buffer = []
        self._transcript_buffer = ""

    def set_callbacks(
        self,
        on_transcript: Optional[Callable[[str], None]] = None,
        on_audio: Optional[Callable[[bytes], None]] = None,
        on_response_done: Optional[Callable[[str, bytes], None]] = None,
        on_error: Optional[Callable[[str], None]] = None
    ) -> None:
        self._on_transcript = on_transcript
        self._on_audio = on_audio
        self._on_response_done = on_response_done
        self._on_error = on_error

    def connect(self) -> bool:
        """Establish connection to Gemini Live."""
        if self._connected:
            return True

        if not self.api_key:
            logger.error("GOOGLE_API_KEY not set")
            return False

        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key, http_options={"api_version": "v1alpha"})
            
            # Start the session loop in a thread
            self._stop_event.clear()
            self._recv_thread = threading.Thread(
                target=self._session_loop,
                daemon=True,
                name="GeminiLiveLoop"
            )
            self._recv_thread.start()
            
            self._connected = True
            logger.info(f"Gemini Live connected (model: {self.model})")
            return True

        except ImportError:
            logger.error("google-genai package required: pip install google-genai")
            if self._on_error:
                self._on_error("google-genai package missing")
            return False
        except Exception as e:
            logger.error(f"Gemini Live connection failed: {e}")
            if self._on_error:
                self._on_error(str(e))
            return False

    def disconnect(self) -> None:
        """Close connection."""
        self._connected = False
        self._stop_event.set()
        # Wake up send loop if needed (by putting dummy item or session close)
        self._send_queue.put(None)
        
        if self._recv_thread and self._recv_thread.is_alive():
            self._recv_thread.join(timeout=2.0)
        
        logger.info("Gemini Live disconnected")

    def send_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> None:
        """Send audio chunk."""
        if not self._connected:
            return

        # Resample to 16kHz if needed (Gemini supports 16k or 24k)
        # Assuming input is float32, we convert to PCM16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # Decode to base64 string as raw bytes in json might be problematic for the queue/sdk bridge
        b64_data = base64.b64encode(audio_bytes).decode('utf-8')
        # Queue for sending
        self._send_queue.put({"data": b64_data, "mime_type": "audio/pcm;rate=16000"})

    def send_text(self, text: str) -> None:
        """Send text message."""
        if not self._connected:
            logger.warning(f"Attempted to send text to Gemini but NOT CONNECTED: {text[:50]}...")
            return
        logger.info(f"Sending text to Gemini: {text}")
        self._send_queue.put({"text": text})

    def update_instructions(self, instructions: str) -> None:
        """Update system instructions."""
        self.instructions = instructions
        # Note: In current SDK, system instruction is usually set at connect time.
        # But we can try to send it as a system message if supported, or restart session.
        # For now, we'll just log it.
        logger.info("Updating instructions (Gemini Live session might need restart to fully apply)")
        # Ideally we'd send a config update frame if supported

    def clear_audio_buffer(self) -> None:
        """Clear local audio/transcript buffers."""
        self._audio_buffer = []
        self._transcript_buffer = ""
        # Also try to clear send queue of pending text?
        # while not self._send_queue.empty():
        #    try: self._send_queue.get_nowait()
        #    except: break

    def interrupt(self) -> None:
        """Interrupt current generation/playback."""
        self.clear_audio_buffer()
        # For Gemini Live, sending meaningful text might help reset context, but silence is often enough.
        # We mainly rely on client-side audio stop.
        logger.info("Interrupted Gemini Live client")

    def _session_loop(self) -> None:
        """Main loop managing the websocket session."""
        try:
            # Connect using the context manager
            # config requires types.LiveConnectConfig
            from google.genai import types
            
            config = types.LiveConnectConfig(
                response_modalities=["AUDIO"], # We want audio back, strictly.
                system_instruction=types.Content(parts=[types.Part(text=self.instructions)]),
            )
            
            # Run everything in the asyncio loop
            async def main():
                try:
                    # For Live API, we must use a supported model.
                    # 'gemini-2.0-flash-exp' is the primary one supporting websocket live API currently.
                    # We will try to use the configured model, but if it fails, the user needs to update config.
                    logger.info(f"Connecting to Gemini Live with model: {self.model}")
                    async with self.client.aio.live.connect(model=self.model, config=config) as session:
                        self._session = session
                        
                        async def send_task():
                            while not self._stop_event.is_set():
                                try:
                                    # Non-blocking check of queue using run_in_executor
                                    item = await asyncio.get_event_loop().run_in_executor(None, self._send_queue.get)
                                    if item is None:
                                        break
                                        
                                    try:
                                        if "text" in item:
                                            # Create textual content for the turn
                                            await session.send(input=item["text"], end_of_turn=True)
                                        elif "data" in item:
                                            # Send audio content
                                            await session.send(input={"data": item["data"], "mime_type": item["mime_type"]}, end_of_turn=False)
                                    except Exception as send_err:
                                        err_str = str(send_err)
                                        
                                        # FATAL errors: Break loop to trigger reconnect
                                        if "1011" in err_str or "keepalive ping timeout" in err_str:
                                            logger.error(f"FATAL connection error: {err_str}")
                                            if self._on_error:
                                                self._on_error(f"FATAL: {err_str}")
                                            break # Exit send loop, which should cancel recv loop and exit session
                                            
                                        # Non-fatal errors (e.g. active response spam)
                                        logger.warning(f"Failed to send item to Gemini: {send_err}")
                                        if self._on_error:
                                            if "active response" in err_str:
                                                logger.warning("Attempted to speak over active response")
                                            else:
                                                self._on_error(f"Send Error: {send_err}")

                                except Exception as e:
                                    logger.error(f"Send task queue error: {e}")
                                    break

                        async def recv_task():
                            try:
                                async for response in session.receive():
                                    if self._stop_event.is_set():
                                        break
                                    
                                    # Process response
                                    server_content = response.server_content
                                    if server_content:
                                        # logger.debug(f"Received server_content: {server_content}")
                                        
                                        # Audio
                                        if server_content.model_turn:
                                            for part in server_content.model_turn.parts:
                                                if part.inline_data:
                                                    # Audio data
                                                    logger.info(f"Received audio chunk: {len(part.inline_data.data)} bytes")
                                                    audio_bytes = part.inline_data.data
                                                    self._audio_buffer.append(audio_bytes)
                                                    if self._on_audio:
                                                        self._on_audio(audio_bytes)
                                                if part.text:
                                                    # Transcript
                                                    logger.info(f"Received transcript chunk: {part.text[:50]}...")
                                                    self._transcript_buffer += part.text
                                                    if self._on_transcript:
                                                        self._on_transcript(part.text)
                                                        
                                        # Turn complete?
                                        if server_content.turn_complete:
                                            logger.info("Gemini Live turn complete")
                                            full_audio = b"".join(self._audio_buffer)
                                            full_text = self._transcript_buffer
                                            if self._on_response_done:
                                                self._on_response_done(full_text, full_audio)
                                            # Reset buffers
                                            self._audio_buffer = []
                                            self._transcript_buffer = ""
                            except Exception as e:
                                logger.error(f"Receive task error: {e}")

                        t1 = asyncio.create_task(send_task())
                        t2 = asyncio.create_task(recv_task())
                        
                        # Wait for either to finish (e.g. send triggers error, or recv gets disconnect)
                        done, pending = await asyncio.wait(
                            [t1, t2], 
                            return_when=asyncio.FIRST_COMPLETED
                        )
                        
                        # Cancel remaining (if send crashed, cancel recv, etc)
                        for t in pending:
                            t.cancel()
                            try:
                                await t
                            except asyncio.CancelledError:
                                pass
                                
                        # Check for exceptions in finished tasks
                        for t in done:
                            if t.exception():
                                raise t.exception()
                        
                except Exception as e:
                    logger.error(f"Connection error in async loop: {e}")
                    if self._on_error:
                        self._on_error(f"Live Connection Failed: {e}")
                    raise e

            asyncio.run(main())
                
        except Exception as e:
            logger.error(f"Gemini Live session error: {e}")
            if self._on_error:
                self._on_error(f"Session error: {e}")
            self._connected = False
