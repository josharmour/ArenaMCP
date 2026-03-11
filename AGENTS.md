# ArenaMCP Agent Guide

This file is for coding agents working in this repository. It is intentionally practical and source-aligned.

For a deeper architecture walkthrough, see `docs/AGENT_ARCHITECTURE.md`.

## 1) What This App Actually Is

ArenaMCP has two runtime modes:

1. `python -m arenamcp.standalone` (primary path)
- Runs a local Textual TUI (default) or CLI fallback.
- Uses an in-process MCP client wrapper (`MCPClient` in `standalone.py`) that imports/calls `arenamcp.server` functions directly.
- Polls game state and asks a local LLM backend for coaching.

2. `python -m arenamcp.server` (MCP server path)
- Runs FastMCP over STDIO (`mcp = FastMCP("mtga")`, `mcp.run()`).
- Exposes MTGA tools for external MCP clients.

In both modes, live game state comes from the same pipeline:
`watcher.py -> parser.py -> gamestate.py`.

## 2) Core Runtime Pipeline (Must Understand Before Editing)

1. `MTGALogWatcher` tails `Player.log` and emits chunks.
2. `LogParser` accumulates multiline JSON and emits typed events.
3. `create_game_state_handler(game_state)` handles `GreToClientEvent` payloads.
4. `GameState.update_from_message()` applies GRE full/diff state updates.
5. `server.get_game_state()` serializes and enriches cards with oracle text.
6. `StandaloneCoach._coaching_loop()` polls and evaluates triggers.
7. `CoachEngine.get_advice()` calls backend and returns concise action text.

Important reliability behavior:
- Watchdog misses are mitigated by explicit `poll_log()` calls in the coaching loop.
- Match recovery uses persisted `~/.arenamcp/last_match.json` with log offset.
- Game end is signaled cross-thread using `game_state.game_ended_event`.

## 3) Major Modules and Responsibilities

- `src/arenamcp/standalone.py`
  - Main app orchestrator and polling loop.
  - Backend/model switching (F11/F12), fallback, hotkeys, voice integration.
  - Draft/sealed auto-detection and post-match analysis dispatch.

- `src/arenamcp/tui.py`
  - Textual UI wrapper around `StandaloneCoach`.
  - Button/keyboard routing to coach methods.
  - Provider/model cycling calls `coach.get_available_providers()` and `coach.get_models_for_provider()`.

- `src/arenamcp/server.py`
  - Shared state singletons (`game_state`, `draft_state`, `parser`, `watcher`).
  - MCP tool surface (state, card info, draft, voice, coaching control, deck/meta tools).
  - Enrichment chain: MTGJSON -> MTGA DB -> Scryfall fallback.

- `src/arenamcp/gamestate.py`
  - Canonical in-memory game model (zones, objects, players, turn info).
  - Sticky diff-update semantics for GRE partial updates.
  - Local seat detection/support, pending decision context, combat-step queue.
  - Match state persistence and game-end snapshot lifecycle.

- `src/arenamcp/parser.py`
  - Brace-depth JSON assembler for MTGA multiline logs.
  - Event hinting when event labels and JSON are split across lines/chunks.

- `src/arenamcp/watcher.py`
  - Watchdog file monitor with file position tracking and backfill strategy.
  - Backfill scans last chunk for match-start markers to avoid giant log reads.

- `src/arenamcp/coach.py`
  - LLM backends: Claude Code CLI, Gemini CLI, Codex CLI, OpenAI-compatible Proxy.
  - Provider/model enumeration API used by TUI and standalone.
  - Trigger detection (`GameStateTrigger`) and advice generation (`CoachEngine`).

- `src/arenamcp/backend_detect.py`
  - Fast backend presence checks + health validation.
  - Auto-select backend policy and query failure classification.

- `src/arenamcp/settings.py`
  - Persistent settings in `~/.arenamcp/settings.json`.

## 4) MCP Tools (Server Surface)

Primary tools in `server.py` include:
- `get_game_state`
- `get_card_info`
- `get_opponent_played_cards`
- `get_draft_rating`
- `get_draft_pack`
- `reset_game_state`
- `listen_for_voice`
- `speak_advice`
- `get_pending_advice`, `clear_pending_advice`
- `start_coaching`, `stop_coaching`, `get_coaching_status`
- `start_draft_helper_tool`, `stop_draft_helper_tool`, `get_draft_helper_status`
- `build_deck`, `get_metagame`, `get_commander_info`

Standalone mode calls these directly via `MCPClient` (in-process import), not over STDIO.

## 5) Backends, Providers, and Models

- Provider list for UI cycling comes from `coach.get_available_providers()`.
- Model list for F12/model button comes from `coach.get_models_for_provider(provider)`.
- CLI providers use static model lists (`_CLI_MODELS` in `coach.py`).
- Proxy/Ollama/API providers query `/models` dynamically with fallbacks.

Current codex static models include:
- `gpt-5.4-pro`
- `gpt-5.3-codex`

## 6) Build, Run, and Test

- Install: `.\install.bat` or `pip install -e ".[full,dev]"`
- Standalone: `.\run.bat --backend auto`
- TUI default: `python -m arenamcp.standalone`
- Server: `python -m arenamcp.server`
- Tests: `pytest`
- Build exe: `python build_windows.py --clean --zip`

## 7) Editing Rules for Agents

- Prefer minimal changes in hot paths:
  - `watcher.py`, `parser.py`, `gamestate.py`, `standalone.py`, `server.py`.
- Preserve import-time behavior in `server.py` and `standalone.py` unless explicitly changing startup semantics.
- Keep fallback/recovery behavior intact:
  - `poll_log()` backup path
  - match offset persistence
  - game-end event signaling
  - backend fallback to Ollama
- When changing parser/state logic, add/update deterministic tests in `tests/`.
- Avoid introducing network/device coupling in tests.

## 8) Known Sharp Edges

- Backend availability can differ between quick detect (`detect_backends_quick`) and full validation (`validate_backend`); debug both when provider switching behaves unexpectedly.
- Keep release notes and installer labels synchronized with package/runtime versions (`pyproject.toml`, `arenamcp.__version__`, and batch script banners).
- TUI and standalone have separate provider/model cache invalidation paths; test both F11/F12 and sidebar buttons.

## 9) Commit/PR Expectations

- Conventional Commits: `feat:`, `fix:`, `docs:`, etc.
- Include repro steps and impacted config keys/env vars.
- For UI changes, include screenshots.

## 10) Security and Config Hygiene

- Never commit secrets.
- Keep credentials in `.env` or `~/.arenamcp/endpoints.json`.
- If adding config keys, update `.env.example` and relevant docs.
