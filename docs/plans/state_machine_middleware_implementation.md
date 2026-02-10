# State Machine Middleware Implementation Plan
**Objective:** Transition `ArenaMCP` from raw delta-log parsing to a robust "State Anchor" middleware that provides stable, complete game state snapshots to the LangGraph agent system.

**Strategy:** Decompose the implementation into three parallel workstreams executable by specialized sub-agents.

---

## 🏗️ Architecture Pattern: The State Anchor

The core concept is to treat the `GameState` class not effectively as a cache, but as a persistent database that "merges" incoming delta updates (`GreToClientEvent`) while preserving "Sticky State" (fields that are not re-sent in every frame, like `turn_entered_battlefield`).

**Data Flow:**
`LogParser` -> `Delta JSON` -> **State Anchor (Merge Logic)** -> `Snapshot JSON` -> `LangGraph Agent`

---

## 🚀 Workstream 1: State Core Hardening
**Role:** Backend Systems Agent
**Focus:** `src/arenamcp/gamestate.py`

### Task 1.1: Implement Sticky State Merger
**Context:** Current updates overwrite objects completely, losing derived data like `turn_entered_battlefield` (critical for Summoning Sickness).
**Requirement:**
1.  Modify `GameState._update_game_object`.
2.  Before creating a new `GameObject`, check if `instance_id` exists.
3.  If exists, extract **Sticky Fields**:
    -   `turn_entered_battlefield`
    -   `is_local` (inferred)
    -   `known_identity` (metadata)
4.  Pass these into the new `GameObject` constructor.
5.  **Deliverable:** Updated `gamestate.py` with valid merger logic.

### Task 1.2: Snapshot Serialization
**Context:** The LangGraph agent needs a clean, JSON-serializable dictionary, not a Python object with circular references or methods.
**Requirement:**
1.  Add `to_dict()` method to `GameObject`, `Zone`, `Player`, and `GameState`.
2.  Ensure output is "LLM-Optimized" (remove noise, keep semantic keys).
    -   *Example:* Convert `ZoneType.BATTLEFIELD` enum to string "Battlefield".
    -   *Example:* Flatten `mana_pool` to a simple string string if empty.
3.  **Deliverable:** `GameState.get_snapshot() -> dict` method.

---

## 🧪 Workstream 2: Replay & Verification
**Role:** QA / Test Agent
**Focus:** `tests/` and `scripts/`

### Task 2.1: Deterministic Replay Harness
**Context:** We cannot verify state stability by playing live games manually. We need to replay partial logs and assert the final state.
**Requirement:**
1.  Create `src/arenamcp/tests/test_state_replay.py`.
2.  Function `replay_log_segment(log_content: str) -> GameState`.
3.  Implement a test case `test_summoning_sickness_persistence`:
    -   Ingest Frame 1 (Enter Battlefield).
    -   Ingest Frame 2 (Tap/Update).
    -   **Assert** `turn_entered_battlefield` is preserved in Frame 2.
4.  **Deliverable:** A pytest-compatible operational test file.

### Task 2.2: Golden State Verification
**Context:** Ensure complex board states (e.g., 20 creatures) are correctly parsed.
**Requirement:**
1.  Select a "Gold Sample" log file (complex game).
2.  Run the Replay Harness.
3.  Dump the final `get_snapshot()` to `tests/fixtures/gold_state.json`.
4.  **Deliverable:** Validated golden file for regression testing.

---

## 🔌 Workstream 3: LangGraph Integration
**Role:** Integration Agent
**Focus:** `src/arenamcp/coach.py` / `src/arenamcp/agent/`

### Task 3.1: Context Provider Update
**Context:** The current prompt injection uses ad-hoc string formatting that may miss fields.
**Requirement:**
1.  Modify `Coach.get_context()` (or equivalent).
2.  Replace manual state looping with `game_state.get_snapshot()`.
3.  Format the JSON snapshot into a readable YAML or Markdown block for the LLM prompt.
    -   *Pattern:*
        ```yaml
        Current State:
          Turn: 4 (Phase: Main 1)
          My Hand:
            - Card: Giant Growth (Mana: 1G)
          Battlefield:
            - Card: Grizzly Bears (Status: Summoning Sick)
        ```
4.  **Deliverable:** Updated `_format_game_context` method in `coach.py`.

---

## Execution Sequence

1.  **Start Task 2.1 (QA)**: Create the failing test case first (TDD). It should fail because `turn_entered` resets today.
2.  **Start Task 1.1 (Backend)**: Implement the fix.
3.  **Verify**: QA Agent runs Task 2.1 again. It should pass.
4.  **Parallel**: Task 1.2 (Serialization) and Task 3.1 (Integration) can proceed once the core logic is stable.
