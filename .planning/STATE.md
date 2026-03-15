# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-11)

**Core value:** Real-time game state from live MTGA games
**Current focus:** v1.1 Voice Coaching complete

## Current Position

Phase: 9 of 9 (Integration)
Plan: 1 of 1 in current phase
Status: Complete
Last activity: 2026-01-12 — Completed 09-01-PLAN.md

Progress: [██████████] 100% of v1.1 (15/15 plans)

## Performance Metrics

**Velocity (v1.0):**
- Total plans completed: 6
- Average duration: 4 min
- Total execution time: 0.4 hours

**Velocity (v1.1):**
- Total plans completed: 9
- Includes bug fixes and testing during integration phase

**By Phase:**

| Phase | Plans | Status |
|-------|-------|--------|
| 1. Foundation | 2 | Complete |
| 2. Game State | 1 | Complete |
| 3. External Data | 2 | Complete |
| 4. MCP Server | 1 | Complete |
| 5. Voice Input | 3 | Complete |
| 6. Voice Output | 1 | Complete |
| 7. Coach Engine | 1 | Complete |
| 8. MCP Voice Tools | 1 | Complete |
| 9. Integration | 1 | Complete |

## Accumulated Context

### Decisions

All v1.0 decisions documented in PROJECT.md Key Decisions table.

v1.1 Decisions:
- **Model parameter**: Added optional `model` parameter to `start_coaching()` for custom LLM models
- **Manual reset tool**: Added `reset_game_state()` MCP tool for when automatic detection fails
- **Multiple inference points**: Local player inferred from hand zones at update time and on-demand
- **New game detection**: Turn number reset triggers automatic state reset

### Deferred Issues

None.

### Blockers/Concerns

None.

### Roadmap Evolution

- v1.0 MVP shipped 2026-01-11
- v1.1 Voice Coaching shipped 2026-01-12

## Session Continuity

Last session: 2026-01-12
Stopped at: Completed 09-01-PLAN.md (Phase 9 complete, v1.1 milestone complete)
Resume file: None

## 2026-03-14 Research Audit

This section consolidates the current status of the research-driven work across:

- `mtga-re-investigation-report.md`
- `game-state-tracking-improvement-plan.md`
- `game-state-tracking-task-list.md`
- `OPTION_C_SUMMARY.md`
- `DIVERGENCE_SUMMARY.md`

Audit method:

- Read the tracking/research Markdown files in the repo root and `docs/`
- Verified implementation status against `src/arenamcp/` and `tests/`
- Verified reverse-engineering claims against checked-in ILSpy output under `re-output/`

### Verified Research Conclusions

- MTGA gameplay is fundamentally GRE action-ID driven, not coordinate driven. The checked-in decompilation confirms `ActionsAvailableWorkflow` submits GRE `Action` objects and the protobuf model exposes `actionType`, `grpId`, and `instanceId`.
- ArenaMCP's core live-state architecture remains correct: `watcher.py -> parser.py -> gamestate.py`.
- MTGA `Player.log` lifecycle and command-line logging flags (`-nolog`, `-appendlog`, `-logfile`) materially affect resume correctness and diagnostics.
- Replay-based advisor tuning and divergence review are valid improvement loops, but they are separate from the live runtime unless explicitly integrated.

### Implemented Since The Original Plans

- Session-aware resume now exists.
  - `src/arenamcp/gamestate.py` saves `log_identity` with `path`, `size`, and `mtime`.
  - `src/arenamcp/server.py` validates resume state before reusing the saved offset.
  - Tests exist in `tests/test_resume_session.py`.

- Startup backfill cost was reduced.
  - `src/arenamcp/watcher.py` detects fresh small logs and skips expensive backfill scanning.
  - Tests exist in `tests/test_watcher_startup.py`.

- Parser robustness work landed.
  - `src/arenamcp/parser.py` now uses string-aware brace tracking and recovery guards.
  - Tests exist in `tests/test_parser_robustness.py`.

- GRE decision tracking was expanded substantially.
  - `src/arenamcp/gamestate.py` now captures `decision_context`, `decision_timestamp`, `legal_actions`, and `legal_actions_raw`.
  - Rich handling exists for `ActionsAvailableReq`, `PayCostsReq`, combat requests, `SelectTargetsReq`, `SelectNReq`, modal/group requests, and other GRE decision messages.
  - Tests exist in `tests/test_decision_tracking.py`.

- Decision-aware coaching improvements are implemented.
  - `src/arenamcp/coach.py` includes compact `LegalGRE` prompt context plus decision-specific guidance for discard, scry, surveil, targeting, modal choices, combat, pay-costs, search, distribution, and related GRE decisions.

- Urgency-aware polling is implemented.
  - `src/arenamcp/standalone.py` uses shorter poll intervals during pending decisions, stack windows, and combat.

- RE-driven autopilot groundwork is implemented.
  - `src/arenamcp/action_planner.py` carries `gre_action_ref` on `GameAction`.
  - `src/arenamcp/gre_action_matcher.py` matches structured actions to raw GRE legal actions.
  - `src/arenamcp/screen_mapper.py` uses arc-based hand geometry derived from `CardLayout_Hand.cs`.
  - `src/arenamcp/autopilot.py` explicitly prefers deterministic geometry and tracks execution paths: `gre-aware`, `deterministic-geometry`, `vision-fallback`.
  - Tests exist in `tests/test_hand_arc_geometry.py` and `tests/test_autopilot_execution_paths.py`.

- Replay-based advisor tuning is implemented as tooling.
  - `src/arenamcp/arena_replay.py`
  - `src/arenamcp/replay_converter.py`
  - `src/arenamcp/advisor_tuner.py`
  - `tools/tune_advisor.py`

### Partial Or Still Missing

- Direct GRE submission is still not implemented.
  - ArenaMCP preserves and reasons over raw GRE action identity, but execution still clicks the UI.
  - There is no live `PerformActionResp` send path in `src/arenamcp/`.

- Battlefield layout remains heuristic.
  - Hand layout got the RE treatment.
  - Battlefield permanent positioning in `src/arenamcp/screen_mapper.py` is still an approximation and has not been refactored around decompiled battlefield layout generation.

- Button placement is still mostly fixed coordinates.
  - `src/arenamcp/screen_mapper.py` uses `FixedCoordinates` for pass/resolve/done/keep/mulligan.
  - The RE notes about `UpdateBasicThreeButtonStates` and runtime button positioning have not yet been translated into a dynamic mapper.

- Divergence tracking is not integrated into the live runtime.
  - `src/arenamcp/divergence_tracker.py`, `src/arenamcp/action_detector.py`, and `tools/review_divergences.py` exist.
  - `DIVERGENCE_SUMMARY.md` explicitly marks integration into `standalone.py` as pending.
  - `F7` in `standalone.py` still produces bug reports, not divergence flags.

- Runtime log-health checks are only partially integrated.
  - `src/arenamcp/watcher.py` has `check_log_health()`.
  - `src/arenamcp/diagnose.py` performs log health checks in the standalone diagnosis flow.
  - The live coaching loop does not currently call the watcher health check during normal runtime.

- Resume validation is improved but not fully fingerprinted.
  - The task plan suggested an optional fingerprint/session marker.
  - Current resume identity uses `path`, `size`, and `mtime`, which is good but not the strongest possible session discriminator.

- Automatic prompt updates from divergence review are still future work.
  - `DIVERGENCE_WORKFLOW.md` still marks that as a future enhancement.

### High-Confidence Next Steps

- Integrate divergence tracking into `src/arenamcp/standalone.py` without colliding with the current `F7` debug-report workflow.
- Call watcher log-health checks from the live polling loop and surface warnings in the TUI.
- Replace fixed button coordinates with RE-derived, resolution-aware button placement.
- Continue RE work on battlefield layout so deterministic permanent targeting is less heuristic.
- Build an offline `GREActionRef -> PerformActionResp` serializer before attempting any live GRE transport/injection work.

### Practical Status Summary

- The game-state tracking plan is mostly implemented.
- The RE-driven autopilot plan is partially implemented: raw GRE awareness and hand geometry are in place, but battlefield layout and direct GRE execution are not.
- Replay tuning tooling is implemented.
- Divergence review tooling is implemented but still not part of the live coach workflow.
