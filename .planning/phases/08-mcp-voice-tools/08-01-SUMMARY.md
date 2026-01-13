---
phase: 08-mcp-voice-tools
plan: 01
subsystem: api
tags: [mcp, voice, tts, stt, fastmcp]

# Dependency graph
requires:
  - phase: 05-voice-input
    provides: VoiceInput class with PTT/VOX modes
  - phase: 06-voice-output
    provides: VoiceOutput class with TTS synthesis
  - phase: 07-coach-engine
    provides: CoachEngine for LLM coaching (future integration)
provides:
  - listen_for_voice MCP tool for voice capture
  - speak_advice MCP tool for TTS playback
  - get_pending_advice MCP tool for proactive coaching queue
  - clear_pending_advice MCP tool for queue reset
  - queue_advice internal function for Phase 9 integration
affects: [09-integration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Lazy-loaded voice components (_voice_input, _voice_output)
    - Thread-safe deque for advice queue

key-files:
  created: []
  modified:
    - src/arenamcp/server.py

key-decisions:
  - "Combined both tasks in single implementation (cohesive change)"

patterns-established:
  - "Voice tools return error dicts on failure (graceful degradation)"
  - "Advice queue uses deque with maxlen for bounded memory"

issues-created: []

# Metrics
duration: 3min
completed: 2026-01-12
---

# Phase 8 Plan 1: MCP Voice Tools Summary

**Four new MCP tools exposing voice I/O (PTT/VOX capture, TTS playback) and proactive advice queue for any client**

## Performance

- **Duration:** 3 min
- **Started:** 2026-01-12T05:25:00Z
- **Completed:** 2026-01-12T05:28:00Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Added listen_for_voice(mode, timeout) tool for voice capture with transcription
- Added speak_advice(text) tool for TTS synthesis and playback
- Added get_pending_advice() tool to drain proactive advice queue
- Added clear_pending_advice() tool to reset queue
- Added queue_advice() internal function for Phase 9 background loop

## Task Commits

Tasks 1 and 2 were implemented together (cohesive change):

1. **Task 1+2: Voice tools and advice queue** - `f723b85` (feat)

**Plan metadata:** (this commit)

## Files Created/Modified

- `src/arenamcp/server.py` - Added 4 MCP tools, lazy loaders, advice queue infrastructure (+171 lines)

## Decisions Made

- Combined both tasks into single commit (tools and queue are cohesive)
- Used deque(maxlen=10) for advice queue to bound memory
- Recreate VoiceInput when mode changes (PTT vs VOX have different configs)

## Deviations from Plan

- Tasks combined into single commit rather than separate commits (plan expected 2 commits, we made 1)
- Rationale: Both tasks modify same file with cohesive functionality

## Issues Encountered

None

## Next Phase Readiness

Phase 8 complete. Ready for Phase 9: Integration.

- Voice MCP tools ready for any client
- queue_advice() ready for background loop to push proactive triggers
- All voice I/O encapsulated in simple tool calls
- Total: 8 MCP tools now available

---
*Phase: 08-mcp-voice-tools*
*Completed: 2026-01-12*
