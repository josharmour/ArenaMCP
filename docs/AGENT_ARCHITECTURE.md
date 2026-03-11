# ArenaMCP Architecture for Agents

This document is a source-level architecture guide for contributors and coding agents.

## 1) Execution Modes

### Standalone Mode (default user flow)
Entry: `python -m arenamcp.standalone` (usually through `coach.bat` -> `launcher.py`).

- Starts Textual UI by default (`tui.run_tui`).
- Creates `StandaloneCoach`.
- `StandaloneCoach` creates `MCPClient`, which imports/calls `arenamcp.server` in-process.
- No STDIO transport is used in this mode, even though tools live in FastMCP-decorated functions.

### MCP Server Mode
Entry: `python -m arenamcp.server`.

- Initializes `FastMCP("mtga")`.
- Exposes tools over STDIO via `mcp.run()`.
- External MCP clients can call tool functions directly.

## 2) Data Flow: MTGA Log -> Advice

1. `watcher.MTGALogWatcher`
- Watches `Player.log` with watchdog.
- Tracks byte offset and reads only appended content.
- Supports backfill/resume from match offset.

2. `parser.LogParser`
- Accepts text chunks from watcher.
- Buffers partial lines/chunks.
- Accumulates multiline JSON blocks with brace-depth accounting.
- Emits typed events to handlers.

3. `gamestate.create_game_state_handler`
- Consumes `GreToClientEvent` messages.
- Extracts GRE `GameStateMessage` payloads.
- Calls `GameState.update_from_message`.

4. `gamestate.GameState`
- Maintains canonical zones/objects/players/turn state.
- Tracks pending decisions, legal actions, combat steps, seat IDs.
- Emits game-end signal (`game_ended_event`) and pre-reset snapshot.

5. `server.get_game_state`
- Ensures watcher is running.
- Ensures local seat is resolved.
- Serializes zones and enriches cards with oracle/type/mana text.
- Returns LLM-facing snapshot used by coaching loop.

6. `standalone.StandaloneCoach._coaching_loop`
- Polls `poll_log` and `get_game_state` continuously.
- Uses `GameStateTrigger` to detect advice events.
- Calls `CoachEngine.get_advice` and routes TTS/UI.

## 3) Important State Machines

### Match Recovery and Resume
- Saved in `~/.arenamcp/last_match.json` by `save_match_state`.
- Includes `match_id`, `local_seat_id`, and `log_offset`.
- Loaded by `server.start_watching` and used as watcher `resume_offset`.
- Stale states (> 30 min) or non-active status are ignored.

### Game-End Signaling
- Parser thread sees `IntermissionReq`/end signals.
- `GameState.prepare_for_game_end`:
  - infers result if needed,
  - snapshots full state,
  - sets thread event.
- Coaching loop consumes once via `consume_game_end` for post-match analysis.

### Draft/Sealed Auto-Direction
In coaching loop:
- Poll `get_draft_pack`.
- If active draft:
  - call `evaluate_draft_pack_for_standalone`, speak recommendation.
- If sealed pool active:
  - call `get_sealed_pool`, speak recommendation.
- On draft completion:
  - call `analyze_draft_pool` to suggest build direction.

## 4) LLM Backend Architecture

Implemented in `coach.py`:
- `ClaudeCodeBackend` (persistent stream-json subprocess)
- `GeminiCliBackend` (persistent interactive path + one-shot fallback)
- `CodexCliBackend` (one-shot subprocess)
- `ProxyBackend` (OpenAI-compatible HTTP endpoint)

Factory:
- `create_backend(backend_type, model, progress_callback)`

Provider/model listing for UI:
- `get_available_providers()`
- `get_models_for_provider(provider)`

Model list behavior:
- CLI providers use static model tables (`_CLI_MODELS`).
- proxy/api/ollama are queried dynamically via `/models` with fallbacks.

## 5) TUI + Hotkey Control Surface

`src/arenamcp/tui.py`:
- Provider button -> `_cycle_provider` -> `_verify_and_switch`.
- Model button -> `_cycle_model` -> `coach.set_backend(provider, model)`.
- Syncs sidebar labels/status from coach actual backend/model.
- Polls game-state snapshot every 500ms for right-pane rendering.

`src/arenamcp/standalone.py` hotkeys:
- F11 provider cycle
- F12 model cycle
- F5 mute, F6 voice, F7 bug report, F8 swap seat, F9 restart

## 6) Server Tool Surface (Current)

Core gameplay:
- `get_game_state`, `get_card_info`, `get_opponent_played_cards`

Draft/sealed:
- `get_draft_rating`, `get_draft_pack`, `start_draft_helper_tool`, `stop_draft_helper_tool`, `get_draft_helper_status`
- Internal standalone helpers: `evaluate_draft_pack_for_standalone`, `get_sealed_pool`, `analyze_draft_pool`

Coaching controls:
- `start_coaching`, `stop_coaching`, `get_coaching_status`
- `get_pending_advice`, `clear_pending_advice`

Voice:
- `listen_for_voice`, `speak_advice`

Deck/meta:
- `build_deck`, `get_metagame`, `get_commander_info`

## 7) Config Layers and Precedence

Persistent settings (`settings.py`):
- `~/.arenamcp/settings.json`

Advanced endpoints (`backend_detect.py`):
- `~/.arenamcp/endpoints.json`

Environment:
- `.env` loaded early by standalone module.

Practical precedence used in code paths:
- Explicit CLI args > settings > env/defaults.
- Some endpoints use `endpoints.json` first, then settings/env fallback.

## 8) Testing Strategy Anchors

High-value tests:
- `tests/test_integration.py`: watcher/parser end-to-end.
- `tests/test_state_replay.py`: sticky state and replay invariants.
- `tests/test_model_benchmark.py`: model benchmark plumbing.

When editing critical layers:
- Parser changes: add chunk-boundary and malformed-json tests.
- Game state changes: add GRE diff/full replay fixture tests.
- Model/provider changes: verify F11/F12 and TUI sidebar cycles.

## 9) Agent Checklist Before/After Changes

Before:
1. Identify mode(s) affected: standalone, server, or both.
2. Confirm whether changes touch watcher/parser/gamestate hot path.
3. Check if settings/model/provider surfaces are impacted.

After:
1. Run targeted tests (or at least `python -m py_compile` on touched modules).
2. Verify no encoding regressions in large prompt strings/comments.
3. Confirm provider/model UI cycle behavior still works.
4. If changing logs/state flow, validate restart/recovery semantics.

## 10) Operational Caveats

- Quick detection (`detect_backends_quick`) and strict validation (`validate_backend`) are intentionally different checks; investigate both when provider switching fails.
- Keep package/runtime/installer version labels synchronized during releases (`pyproject.toml`, `arenamcp.__version__`, and batch script banners).

Treat source behavior as truth when documenting or debugging.