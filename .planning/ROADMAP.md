# Roadmap: MTGA MCP Server

## Overview

Build a real-time MCP server that parses MTGA Player.log to expose live game state to Claude. Start with reliable log watching and parsing, build game state tracking, integrate external card/draft data, then wrap it all in FastMCP tools.

## Domain Expertise

None

## Milestones

- [v1.0 MVP](milestones/v1.0-ROADMAP.md) (Phases 1-4) — SHIPPED 2026-01-11
- [v1.1 Voice Coaching](milestones/v1.1-ROADMAP.md) (Phases 5-9) — SHIPPED 2026-01-12

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

<details>
<summary>v1.1 Voice Coaching (Phases 5-9) — SHIPPED 2026-01-12</summary>

- [x] **Phase 5: Voice Input** - Global PTT (F4 hotkey) and VOX capture with Whisper STT
- [x] **Phase 6: Voice Output** - Kokoro TTS integration for spoken responses
- [x] **Phase 7: Coach Engine** - Multi-backend LLM coaching (Claude/Gemini/Ollama) with triggers
- [x] **Phase 8: MCP Voice Tools** - listen_for_voice, speak_advice, get_pending_advice tools
- [x] **Phase 9: Integration** - Background coaching loop with end-to-end testing

</details>

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
| 6. Voice Output | v1.1 | 1/1 | Complete | 2026-01-12 |
| 7. Coach Engine | v1.1 | 1/1 | Complete | 2026-01-12 |
| 8. MCP Voice Tools | v1.1 | 1/1 | Complete | 2026-01-12 |
| 9. Integration | v1.1 | 1/1 | Complete | 2026-01-12 |

**v1.0 Complete:** All 4 phases delivered.
**v1.1 Complete:** All 5 phases delivered.
