---
phase: 06-voice-output
plan: 01
subsystem: voice
tags: [kokoro, tts, onnx, sounddevice, audio]

requires:
  - phase: 05-voice-input
    provides: VoiceInput, WhisperTranscriber patterns
provides:
  - KokoroTTS class with lazy model loading
  - VoiceOutput unified API with speak/speak_async/stop
affects: [07-coach-engine, voice-coaching]

tech-stack:
  added: [kokoro-onnx]
  patterns: [lazy-model-loading, thread-safe-tts, async-playback]

key-files:
  created: [src/arenamcp/tts.py]
  modified: [src/arenamcp/__init__.py]

key-decisions:
  - "af_heart voice as default - highest quality Grade A American English"
  - "24kHz fixed sample rate - Kokoro native output"
  - "Manual model download required - 300MB total, one-time setup"

patterns-established:
  - "VoiceOutput mirrors VoiceInput API pattern"
  - "TTS lazy loading matches WhisperTranscriber pattern"

issues-created: []

duration: 2min
completed: 2026-01-13
---

# Phase 6 Plan 1: Voice Output Summary

**Kokoro TTS integration with VoiceOutput API providing speak/speak_async/stop for coach audio responses**

## Performance

- **Duration:** 2 min
- **Started:** 2026-01-13T01:05:06Z
- **Completed:** 2026-01-13T01:07:15Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- KokoroTTS class with lazy model loading matching WhisperTranscriber pattern
- VoiceOutput unified API with speak(), speak_async(), stop(), is_speaking
- Thread-safe model loading and async playback
- Clear error messages with download URLs when model files missing
- Package exports updated

## Task Commits

Each task was committed atomically:

1. **Task 1: Create KokoroTTS module** - `b08af89` (feat)
2. **Task 2: Create VoiceOutput class** - `a935b49` (feat)

**Plan metadata:** (this commit)

## Files Created/Modified

- `src/arenamcp/tts.py` - KokoroTTS and VoiceOutput classes (380 lines)
- `src/arenamcp/__init__.py` - Added VoiceOutput, KokoroTTS exports

## Decisions Made

- **af_heart voice default** - Highest quality Grade A American English female voice per kokoro-onnx benchmarks
- **Manual model download** - 300MB total is too large for automatic download; provide clear instructions
- **24kHz sample rate** - Fixed by Kokoro; no resampling needed

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## Next Phase Readiness

Plan 06-01 complete. Ready for 06-02-PLAN.md (if exists) or Phase 7: Coach Engine.

---
*Phase: 06-voice-output*
*Completed: 2026-01-13*
