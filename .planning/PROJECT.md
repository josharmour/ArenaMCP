# MTGA MCP Server

## What This Is

A real-time MCP server that bridges MTGA (Magic: The Gathering Arena) game logs to Claude for conversational game analysis and coaching. While existing MTG MCP servers handle card lookup and deck building, this one connects to **live MTGA games** by parsing the Player.log file in real-time — enabling Claude to serve as an interactive coach during actual gameplay.

## Core Value

**Real-time game state from live MTGA games.** No existing solution exposes actual in-progress game data to LLMs. If everything else fails, the server must reliably parse the log and expose current board state.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Log watcher that tails Player.log in real-time using watchdog, handling Windows file locking and log truncation on MTGA restart
- [ ] Log parser that accumulates multi-line JSON blocks and routes events (GreToClientEvent, GameStateMessage, MatchCreated, etc.)
- [ ] Game state manager maintaining current snapshot, history, and tracking cards played by opponent
- [ ] `get_game_state()` tool returning complete board state: battlefield, hands, life totals, stack, zones, turn/phase info
- [ ] `get_card_info(arena_id)` tool for card oracle text and rulings via Scryfall
- [ ] `get_opponent_played_cards()` tool for tracking revealed opponent cards
- [ ] `get_draft_rating(card_name, set_code)` tool for 17lands draft statistics
- [ ] Scryfall integration with bulk data download and `arena_id` index, API fallback for new cards
- [ ] 17lands CSV integration for draft statistics (GIH WR, ALSA, IWD)
- [ ] MCP server using FastMCP with STDIO transport for Claude Code integration
- [ ] Card serialization including oracle text for LLM context (Calculator + Coach pattern)

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
- **Draft stats**: 17lands CSV files — only available source (no API)
- **Platform**: Windows primary — MTGA's home platform

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| STDIO transport over HTTP | Microseconds vs milliseconds latency, no network config | — Pending |
| Tools over Resources for game state | MCP Resources don't support server-push; polling via tools works | — Pending |
| Calculator + Coach pattern | LLMs struggle with state tracking; let code track, LLM explain | — Pending |
| Scryfall bulk data with API fallback | Rate limits (10/sec) make API-only impractical for game flow | — Pending |
| Include oracle text in serialization | LLM needs card text for analysis, not just names | — Pending |

---
*Last updated: 2025-01-11 after initialization*
