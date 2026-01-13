"""Audio capture infrastructure for voice input.

This module provides non-blocking audio recording using sounddevice for
voice input capture. It uses a callback-based InputStream pattern that
will integrate with PTT (Push-to-Talk) and VOX (Voice Activation) modes.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import sounddevice as sd


@dataclass
class AudioConfig:
    """Configuration for audio recording.

    Attributes:
        sample_rate: Sample rate in Hz. Default 16000 (Whisper's native rate).
        channels: Number of audio channels. Default 1 (mono for voice).
        dtype: Numpy dtype for audio samples. Default 'float32'.
        device: Audio input device index or name. None uses system default.
    """

    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    device: Optional[int | str] = None


class AudioRecorder:
    """Non-blocking audio recorder using sounddevice.

    Uses a callback-based InputStream pattern for PTT/VOX integration.
    Thread-safe buffer management allows concurrent recording and access.

    Example:
        recorder = AudioRecorder()
        recorder.start_recording()
        time.sleep(1.0)
        audio = recorder.stop_recording()
        # audio is numpy array of shape (samples,) at 16kHz float32
    """

    def __init__(self, config: Optional[AudioConfig] = None) -> None:
        """Initialize the audio recorder.

        Args:
            config: Audio configuration. If None, uses default AudioConfig.
        """
        self.config = config or AudioConfig()
