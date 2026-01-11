---
phase: 03-external-data
plan: 02
subsystem: external-data
tags: [17lands, draft, statistics, cache]

# Dependency graph
requires:
  - phase: 03-external-data
    provides: Cache infrastructure pattern from ScryfallCache
provides:
  - DraftStatsCache class for 17lands draft statistics
  - get_draft_rating() method for card evaluation
  - GIH WR, ALSA, IWD metrics access
affects: [mcp-server, draft-assistant]

# Tech tracking
tech-stack:
  added: []
  patterns: [json-api-cache, lazy-loading]

key-files:
  created: [src/arenamcp/draftstats.py]
  modified: [src/arenamcp/__init__.py]

key-decisions:
  - "Use JSON parsing instead of CSV (17lands API returns JSON, not CSV as plan stated)"
  - "Cache files as .json with 24-hour expiration"

patterns-established:
  - "Lazy set loading: only fetch data when first requested"

issues-created: []

# Metrics
duration: 4min
completed: 2026-01-11
---

# Phase 03-02 Summary: 17lands Draft Statistics

**DraftStatsCache with lazy-loaded JSON from 17lands card_ratings API, providing GIH WR, ALSA, and IWD metrics for draft card evaluation.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-01-11T21:21:25Z
- **Completed:** 2026-01-11T21:25:15Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- DraftStatsCache class with JSON download from 17lands API
- Lazy loading: only fetches set data when first requested
- get_draft_rating(card_name, set_code) returns DraftStats dataclass
- Cache stored in ~/.arenamcp/cache/17lands/{SET}_PremierDraft.json
- 24-hour cache expiration with automatic refresh

## Task Commits

1. **Task 1: Create DraftStatsCache with download and parsing** - `67bd32f` (feat)
2. **Task 2: Implement get_draft_rating with lazy loading** - `400e8f9` (feat)

## Files Created/Modified

- `src/arenamcp/draftstats.py` (created) - DraftStatsCache and DraftStats classes
- `src/arenamcp/__init__.py` (modified) - Export new classes

## Decisions Made

- **JSON not CSV**: Plan stated 17lands provides CSV data, but the API actually returns JSON. Adapted parser accordingly.
- **Field mapping**: Mapped 17lands JSON fields to DraftStats:
  - `ever_drawn_win_rate` → `gih_wr`
  - `avg_seen` → `alsa`
  - `drawn_improvement_win_rate` → `iwd`
  - `ever_drawn_game_count` → `games_in_hand`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed data format assumption**
- **Found during:** Task 1 (CSV parsing)
- **Issue:** Plan assumed 17lands returns CSV data, but API returns JSON
- **Fix:** Rewrote parser to use json.load() instead of csv.DictReader
- **Files modified:** src/arenamcp/draftstats.py
- **Verification:** Successfully parsed 281 cards from DSK set with valid GIH WR data
- **Committed in:** 400e8f9 (combined with Task 2)

---

**Total deviations:** 1 auto-fixed (data format assumption)
**Impact on plan:** Minor - JSON parsing is simpler than CSV. No scope creep.

## Issues Encountered

None.

## Verification Results

- [x] `from arenamcp import DraftStatsCache` imports without error
- [x] Cache directory exists at ~/.arenamcp/cache/17lands/
- [x] get_draft_rating() returns stats for known cards (tested DSK: Acrobatic Cheerleader)
- [x] get_draft_rating() returns None for invalid card/set combos
- [x] All 15 existing tests pass

## Next Phase Readiness

- External data layer complete (Scryfall + 17lands)
- Ready for Phase 4: MCP Server implementation
- All data sources available for MCP tools

---
*Phase: 03-external-data*
*Completed: 2026-01-11*
