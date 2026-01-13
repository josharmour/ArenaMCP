# Roadmap: MTGA MCP Server

## Overview

Build a real-time MCP server that parses MTGA Player.log to expose live game state to Claude. Start with reliable log watching and parsing, build game state tracking, integrate external card/draft data, then wrap it all in FastMCP tools.

## Domain Expertise

None

## Milestones

- [v1.0 MVP](milestones/v1.0-ROADMAP.md) (Phases 1-4) — SHIPPED 2026-01-11
- 🚧 **v1.1 Voice Coaching** — Phases 5-9 (in progress)

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

<details>
<summary>v1.0 MVP (Phases 1-4) — SHIPPED 2026-01-11</summary>

- [x] **Phase 1: Foundation** - Log watcher and parser for real-time event extraction
- [x] **Phase 2: Game State** - State manager tracking zones, turns, and game objects
- [x] **Phase 3: External Data** - Scryfall card data and 17lands draft statistics
- [x] **Phase 4: MCP Server** - FastMCP tools exposing game state and card info

</details>

### 🚧 v1.1 Voice Coaching (In Progress)

**Milestone Goal:** Add voice coaching with global PTT/VOX support so ANY MCP client automatically gets voice input/output capabilities. The server should be self-contained - connecting a client starts the voice hotkey listener.

#### Phase 5: Voice Input

**Goal**: Global PTT (F4 hotkey) and VOX (voice activation) capture with Whisper STT transcription
**Depends on**: Phase 4 (MCP Server complete)
**Research**: Likely (keyboard library for global hotkeys, faster-whisper setup)
**Research topics**: keyboard library Windows compatibility, faster-whisper model selection, audio capture with sounddevice
**Plans**: TBD

Plans:
- [x] 05-01: Audio capture with sounddevice
- [x] 05-02: PTT with global F4 hotkey
- [x] 05-03: Whisper STT transcription

#### Phase 6: Voice Output

**Goal**: Kokoro TTS integration for speaking responses aloud
**Depends on**: Phase 5
**Research**: Likely (kokoro-onnx model setup, audio output configuration)
**Research topics**: Kokoro model download/paths, sounddevice playback, voice selection
**Plans**: TBD

Plans:
- [ ] 06-01: TBD

#### Phase 7: Coach Engine

**Goal**: Claude API integration with game context and proactive advice triggers (priority pass, combat, new turn, low life)
**Depends on**: Phase 6
**Research**: Unlikely (internal patterns, anthropic SDK already known)
**Plans**: TBD

Plans:
- [ ] 07-01: TBD

#### Phase 8: MCP Voice Tools

**Goal**: Expose listen_for_voice() (blocking), speak_advice(), get_pending_advice() as MCP tools for any client
**Depends on**: Phase 7
**Research**: Unlikely (existing FastMCP patterns from v1.0)
**Plans**: TBD

Plans:
- [ ] 08-01: TBD

#### Phase 9: Integration

**Goal**: Background voice loop on MCP server startup, end-to-end testing with Claude Code
**Depends on**: Phase 8
**Research**: Unlikely (wiring existing components)
**Plans**: TBD

Plans:
- [ ] 09-01: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Foundation | v1.0 | 2/2 | Complete | 2026-01-11 |
| 2. Game State | v1.0 | 1/1 | Complete | 2026-01-11 |
| 3. External Data | v1.0 | 2/2 | Complete | 2026-01-11 |
| 4. MCP Server | v1.0 | 1/1 | Complete | 2026-01-11 |
| 5. Voice Input | v1.1 | 3/3 | Complete | 2026-01-12 |
| 6. Voice Output | v1.1 | 0/? | Not started | - |
| 7. Coach Engine | v1.1 | 0/? | Not started | - |
| 8. MCP Voice Tools | v1.1 | 0/? | Not started | - |
| 9. Integration | v1.1 | 0/? | Not started | - |

**v1.0 Complete:** All 4 phases delivered.
**v1.1 In Progress:** 5 phases planned.
