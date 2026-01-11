# MTGA MCP Server

## Current State

**Version:** v1.0 MVP (shipped 2026-01-11)
**Status:** Complete and ready for integration testing
**Codebase:** 1,911 lines of Python across 7 modules

## What This Is

A real-time MCP server that bridges MTGA (Magic: The Gathering Arena) game logs to Claude for conversational game analysis and coaching. While existing MTG MCP servers handle card lookup and deck building, this one connects to **live MTGA games** by parsing the Player.log file in real-time — enabling Claude to serve as an interactive coach during actual gameplay.

## Core Value

**Real-time game state from live MTGA games.** No existing solution exposes actual in-progress game data to LLMs. If everything else fails, the server must reliably parse the log and expose current board state.

## Requirements

### Validated

- Log watcher that tails Player.log in real-time using watchdog, handling Windows file locking and log truncation on MTGA restart — v1.0
- Log parser that accumulates multi-line JSON blocks and routes events (GreToClientEvent, GameStateMessage, MatchCreated, etc.) — v1.0
- Game state manager maintaining current snapshot, history, and tracking cards played by opponent — v1.0
- `get_game_state()` tool returning complete board state: battlefield, hands, life totals, stack, zones, turn/phase info — v1.0
- `get_card_info(arena_id)` tool for card oracle text and rulings via Scryfall — v1.0
- `get_opponent_played_cards()` tool for tracking revealed opponent cards — v1.0
- `get_draft_rating(card_name, set_code)` tool for 17lands draft statistics — v1.0
- Scryfall integration with bulk data download and `arena_id` index, API fallback for new cards — v1.0
- 17lands JSON API integration for draft statistics (GIH WR, ALSA, IWD) — v1.0
- MCP server using FastMCP with STDIO transport for Claude Code integration — v1.0
- Card serialization including oracle text for LLM context (Calculator + Coach pattern) — v1.0

### Active

(None — v1.0 complete. Define v1.1 scope to add new requirements.)

### Out of Scope

- Opponent's hand contents — not available in logs, would require cheating
- Opponent's library contents — not available in logs
- Face-down card identities — not available in logs
- Remote/network deployment — local STDIO transport only
- Web UI or overlay — this is a headless MCP server for Claude
- Game simulation or AI opponent — this tracks real games, not simulated ones
- Sideboard tracking — known MTGA logging gap

## Context

**Technical environment:**
- MTGA Player.log at `%AppData%\LocalLow\Wizards Of The Coast\MTGA\Player.log`
- Requires "Detailed Logs (Plugin Support)" enabled in MTGA settings
- Log format: `[UnityCrossThreadLogger]` prefixed lines with multi-line JSON payloads
- Critical message type: `GreToClientEvent` containing `GameStateMessage` objects
- Card IDs: `grpId` in logs maps to Scryfall's `arena_id`

**Reference implementations:**
- mtga-pro-tracker — battle-tested log parsing patterns
- 17lands mtga-log-client — proven event handling
- Existing MTG MCP servers (artillect, pato, ericraio) — deck building patterns, but no live game support

**Key insight from research:**
The "Calculator + Coach" pattern from Go AI: use deterministic game state tracking for accuracy (the calculator) while the LLM provides strategic commentary and explanations (the coach). This avoids LLM state-tracking weaknesses while leveraging explanation strengths.

## Constraints

- **Language**: Python — best log parsing libraries, Scryfall SDK (scrython), rapid prototyping
- **Framework**: FastMCP — purpose-built for MCP servers
- **Transport**: STDIO — lowest latency for local use, simplest configuration
- **Card data**: Scryfall bulk + API fallback — free, comprehensive, has `arena_id` mapping
- **Draft stats**: 17lands JSON API — only available source
- **Platform**: Windows primary — MTGA's home platform

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| STDIO transport over HTTP | Microseconds vs milliseconds latency, no network config | Good |
| Tools over Resources for game state | MCP Resources don't support server-push; polling via tools works | Good |
| Calculator + Coach pattern | LLMs struggle with state tracking; let code track, LLM explain | Good |
| Scryfall bulk data with API fallback | Rate limits (10/sec) make API-only impractical for game flow | Good |
| Include oracle text in serialization | LLM needs card text for analysis, not just names | Good |
| Upsert pattern for state updates | Handles both full and incremental GameStateMessage uniformly | Good |
| Lazy-load external caches | Avoids blocking server startup with bulk data downloads | Good |
| JSON parsing for 17lands (not CSV) | API actually returns JSON despite documentation | Good |

---
*Last updated: 2026-01-11 after v1.0 milestone*
