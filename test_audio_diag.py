
import sounddevice as sd
import numpy as np
import time

def test_audio():
    print("=== Audio Diagnostic Test ===")
    
    # 1. Check Default Device
    try:
        defaults = sd.default.device
        input_idx, output_idx = defaults
        print(f"Default Input Index: {input_idx}")
        print(f"Default Output Index: {output_idx}")
        
        dev_info = sd.query_devices(output_idx)
        print(f"Target Output Device: {dev_info['name']}")
        print(f"Host API: {dev_info['hostapi']}")
    except Exception as e:
        print(f"Error querying defaults: {e}")
        return

    # 2. Generate Tone (440Hz Sine Wave, 2 seconds)
    print("\nGenerating audio test tone...")
    fs = 24000
    duration = 2.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    # Generate int16 audio to match application format
    audio = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    # 3. Play using OutputStream (simulating the app)
    print("\nAttempting playback via OutputStream (device=None)...")
    try:
        stream = sd.OutputStream(
            samplerate=fs,
            channels=1,
            dtype='int16',
            device=None  # Use system default
        )
        with stream:
            stream.write(audio)
        print("Playback 1 complete. Did you hear it?")
    except Exception as e:
        print(f"Playback 1 Failed: {e}")

    time.sleep(1)

    # 4. Play using Explicit Device Index
    print(f"\nAttempting playback via Explicit Device Index ({output_idx})...")
    try:
        stream = sd.OutputStream(
            samplerate=fs,
            channels=1,
            dtype='int16',
            device=output_idx
        )
        with stream:
            stream.write(audio)
        print("Playback 2 complete. Did you hear it?")
    except Exception as e:
        print(f"Playback 2 Failed: {e}")

if __name__ == "__main__":
    test_audio()
