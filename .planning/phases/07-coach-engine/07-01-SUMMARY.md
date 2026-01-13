---
phase: 07-coach-engine
plan: 01
subsystem: ai
tags: [anthropic, claude, gemini, ollama, llm, coaching]

# Dependency graph
requires:
  - phase: 04-mcp-server
    provides: get_game_state() serialization pattern
provides:
  - LLMBackend protocol for pluggable AI backends
  - CoachEngine for game advice generation
  - GameStateTrigger for proactive coaching detection
affects: [08-mcp-voice-tools, 09-integration]

# Tech tracking
tech-stack:
  added: []
  patterns: [protocol-based backends, lazy initialization, factory pattern]

key-files:
  created: [src/arenamcp/coach.py]
  modified: [src/arenamcp/__init__.py]

key-decisions:
  - "Use Protocol for LLMBackend interface (duck typing over ABC)"
  - "Lazy client initialization to avoid API key requirement at import"
  - "Return error strings instead of raising exceptions for resilient coaching"
  - "Default to Claude claude-sonnet-4-20250514 for quality, Gemini flash for speed"

patterns-established:
  - "LLMBackend protocol: complete(system_prompt, user_message) -> str"
  - "Factory pattern: create_backend(type, model) for backend instantiation"

issues-created: []

# Metrics
duration: 4min
completed: 2026-01-12
---

# Phase 7 Plan 1: Coach Engine Summary

**Backend-agnostic CoachEngine with Claude/Gemini/Ollama support and GameStateTrigger for proactive advice detection**

## Performance

- **Duration:** 4 min
- **Started:** 2026-01-12T15:30:00Z
- **Completed:** 2026-01-12T15:34:00Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments

- LLMBackend protocol with Claude, Gemini, and Ollama implementations
- CoachEngine formats game state context and generates strategic advice
- GameStateTrigger detects: new_turn, priority_gained, combat_attackers, combat_blockers, low_life, opponent_low_life, stack_spell
- Factory function create_backend() for easy backend selection
- Lazy initialization - no API keys required until actual use

## Task Commits

Each task was committed atomically:

1. **Task 1: Create LLMBackend protocol and implementations** - `65884d8` (feat)
2. **Task 3: Create GameStateTrigger and update exports** - `68b3606` (feat)

_Note: Tasks 1-3 code was created together; Task 2 (CoachEngine) included in Task 1 commit._

## Files Created/Modified

- `src/arenamcp/coach.py` - New file with LLMBackend protocol, Claude/Gemini/Ollama backends, CoachEngine, GameStateTrigger
- `src/arenamcp/__init__.py` - Added coach exports to package namespace

## Decisions Made

- **Protocol over ABC**: Used typing.Protocol for LLMBackend to enable duck typing flexibility
- **Lazy initialization**: Clients created on first use, not at import (no API key errors during import)
- **Error strings over exceptions**: Backends return error messages as strings for resilient coaching flow
- **Model defaults**: Claude uses claude-sonnet-4-20250514 (quality), Gemini uses gemini-1.5-flash (speed), Ollama uses llama3.2 (common local model)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

Phase 7 complete. Ready for Phase 8: MCP Voice Tools.

- CoachEngine can be instantiated with any backend
- GameStateTrigger ready to detect state changes for proactive advice
- Package exports make all components easily accessible

---
*Phase: 07-coach-engine*
*Completed: 2026-01-12*
