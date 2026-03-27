# Turbo-Charge Upgrades: mtgacoach Improvement Plan

Based on reverse-engineering MTGA's managed DLLs (Assembly-CSharp.dll, Core.dll,
Wizards.MDN.GreProtobuf.dll, SharedClientCore.dll) via ReVa/Ghidra string analysis,
cross-referenced against the current codebase architecture.

**Date:** 2026-03-26

### Phase 1 Status: COMPLETE (2026-03-26)

All Phase 1 items (1.1-1.4) and quick wins (6.1-6.3) implemented in branch
`worktree-phase1-parse-missing-gre-data`. Changes to `gamestate.py` and `server.py`:

- **25+ new annotation handlers** — TargetSpec, Modified*, Designations, PhasedIn/Out,
  ClassLevel, DungeonStatus, SuspendLike, LinkedDamage, ColorProduction, AddAbility,
  CopiedObject, BoonInfo, CrewedThisTurn, SaddledThisTurn, DamagedThisTurn, Shuffle,
  Vote, DieRoll, PredictedDirectDamage, LayeredEffect, SupplementalText, NewTurnStarted
- **16 new fields on GameObject** — modified_power/toughness/cost/colors/types/name,
  granted_abilities, removed_abilities, damaged/crewed/saddled_this_turn, is_phased_out,
  class_level, copied_from_grp_id, targeting, color_production
- **5 new game-level state fields** — designations, dungeon_status, timer_state,
  action_history, sideboard_cards
- **AutoTap mana solver data** parsed per-action with castability flags and tap sequences
- **Ability metadata** extracted (abilityGrpId, sourceId, alternativeGrpId, mana cost string)
- **Timer state** parsed from GREMessageType_TimerStateMessage (chess clock, rope)
- **Sideboard tracking** from SubmitDeckReq in BO3
- **Action history buffer** (rolling 50-entry buffer from UserActionTaken annotations)
- **Debug else clause** logs all unhandled annotation types at DEBUG level
- **Known-but-skippable** annotation types explicitly listed (no false positives in debug log)

### Phase 2 Status: COMPLETE (2026-03-26)

BepInEx plugin expanded from 3 commands to 6, with dramatically richer data.
Plugin version bumped to 0.2.0. Changes to `Plugin.cs` and `gre_bridge.py`:

- **`get_game_state` command** — Serializes full MtgGameState from GameManager.CurrentGameState:
  zones (battlefield/hand/stack/graveyard/exile/command/library) with full card instances,
  players (life/mana/status/mulligan/dungeon/designations), turn info (phase/step/active/deciding),
  combat info (attack/block mappings), timers, and pending interaction type
- **`get_timer_state` command** — Per-player chess clock data: time remaining, timer type,
  running state, warning threshold, behavior. Both game-level and player-level timers
- **`get_match_info` command** — Match metadata: game state ID, stage, GameInfo fields,
  local/opponent seat IDs and life totals
- **Enhanced `SerializeAction`** — Now includes: AssumeCanBePaidFor (ground-truth castability),
  FacetId, UniqueAbilityId, full AutoTapSolution with tap sequence (via reflection),
  Targets, Highlight, ShouldStop, IsBatchable
- **Enhanced `SerializeCard`** — Full MtgCardInstance serialization: power/toughness, loyalty,
  defense, combat state, summoning sickness, phasing, damage, class level, copy info,
  card types, subtypes, colors, counters, color production, targets, attachments,
  visibility, face-down state, crew/saddle
- **Cached GameManager lookup** — 5-second TTL cache avoids repeated FindObjectOfType
- **Python client** (`gre_bridge.py`) — Added get_game_state(), get_timer_state(),
  get_match_info() methods

### Phase 3 Status: COMPLETE (2026-03-26)

Replay recording and match history system implemented. Plugin version 0.3.0.

- **Plugin commands**: `enable_replay`, `disable_replay`, `get_replay_status`, `list_replays`
  — toggles MTGA's built-in TimedReplayRecorder via PlayerPrefs, lists .rply files
- **Python bridge** (`gre_bridge.py`) — enable_replay(), disable_replay(),
  get_replay_status(), list_replays() methods
- **Match history module** (`match_history.py`) — JSON-backed persistent history:
  - `MatchRecord` dataclass: match_id, result, opponent name/rank/colors, turns,
    life totals, deck colors, replay path
  - `MatchHistory` class: add records, query win rates, matchup stats, session stats
  - `parse_replay_cosmetics()` — extract player names/ranks from .rply header
  - `parse_replay_result()` — scan replay for win/loss annotations
  - `record_from_game_end()` — create record from game end snapshot + opponent cards
  - Deduplication by match_id, 500-record cap, stored at ~/.arenamcp/match_history/

---

## Phase 1: Parse Missing GRE Data (Low Effort, High Impact)

These are data fields the game engine already sends in GRE messages that we're either
ignoring or only partially parsing. No BepInEx changes needed — just expand gamestate.py.

### 1.1 Add Missing Annotation Types

**Current state:** gamestate.py handles 19 of 70+ annotation types (lines 1451-1641).
Unhandled types are silently dropped with no else clause.

**Add these annotation handlers:**

| Annotation | What We Get | Where to Use |
|---|---|---|
| `TargetSpec` | Spell/ability targets (instance IDs of targeted objects) | Coach: "Opponent targeted your [X] with removal" — currently listed as a gap in server.py |
| `PredictedDirectDamage` | GRE's own damage prediction for pending combat | Replace manual combat math in coach.py (lines 928-1006) |
| `LayeredEffect` | Active continuous effects (anthems, debuffs, static abilities) | Coach: know actual P/T vs base P/T; detect anthem sources |
| `ModifiedPower` | Real modified power value after all effects | Expose in get_game_state — currently only base stats shown |
| `ModifiedCost` | Actual mana cost after reductions/increases | "This spell costs 2 less because of Goblin Electromancer" |
| `ModifiedColor` | Modified color identity (Painter's Servant, etc.) | Correct color tracking for protection/devotion |
| `ModifiedType` | Modified type line (animated lands become creatures) | Track when a land becomes a creature for combat |
| `ModifiedName` | Modified name (Clone effects, Spy Kit) | Correct card identification |
| `DamagedThisTurn` | Which permanents took damage this turn | Coach: enrage triggers, damage-matters synergies |
| `CrewedThisTurn` | Which vehicles were crewed | Coach: "Vehicle already crewed, don't waste another crew" |
| `SaddledThisTurn` | Which mounts were saddled | Same as crew |
| `PhasedIn` / `PhasedOut` | Phasing state of permanents | Track phased-out cards (currently invisible to coach) |
| `ClassLevel` | Current level of Class enchantments | Coach: "Level up your class for the next ability" |
| `DungeonStatus` | Current room in dungeon | Coach: dungeon progress tracking |
| `SuspendLike` | Suspend/foretell exile with time counters | Coach: "Suspend card comes off in 2 turns" |
| `LinkedDamage` / `DamageSource` | Damage attribution (which source dealt what) | Post-match: "Lost 8 life to Sheoldred triggers over 4 turns" |
| `SupplementalText` | Extra context text from GRE | Pass to LLM as additional context |
| `ColorProduction` | Mana colors a permanent can produce | Perfect mana analysis without oracle text parsing |
| `AddAbility` / `RemoveAbility` | Granted/lost abilities (flying, hexproof, etc.) | Coach: track temporary keyword grants |
| `Designation` / `GainDesignation` / `LoseDesignation` | Monarch, initiative, city's blessing, day/night | Coach: "You have the monarch — protect it" |
| `CopiedObject` | Copy relationships | Coach: know what a Clone copied |
| `BoonInfo` | Boon/emblem effects | Track active emblems and boons |
| `Vote` | Voting results (Council's Dilemma) | Coach: voting strategy |
| `Shuffle` | Library shuffled events | Track tutors, fetchlands |

**Implementation:** Add elif branches in the annotation handler at gamestate.py:1451-1641.
Store new state on GameObject (modified_power, modified_types, etc.) and in game-level
dicts (designations, dungeon_status, etc.). Expose via get_game_state() in server.py.

**Files to change:**
- `src/arenamcp/gamestate.py` — annotation handler (lines 1451-1641), GameObject class (line 59), snapshot builder
- `src/arenamcp/server.py` — get_game_state() to include new fields

### 1.2 Parse AutoTap/Mana Solver Data

**Current state:** gamestate.py line 2358 checks `autoTapSolutions` as a boolean only:
```python
has_autotap = bool(req.get("autoTapActionsReq", {}).get("autoTapSolutions"))
```
The actual mana payment solutions, tap actions, and castability data are discarded.

**What to extract:**
- `AutoTapSolution` — which lands to tap for each legal play (the game already solved it)
- `ManaPaymentOptions` — alternative payment methods
- `AssumeCanBePaidFor` — ground-truth castability flag per action
- `ManaCost` array — structured cost breakdown (already partially parsed at lines 2330-2356)
- `AutoTapActions` — the specific tap sequence the game would use

**Impact:** Eliminates the `[OK]`/`[NEED:3]` mana heuristic in the system prompt. Instead
of parsing oracle text to determine castability, we use the game engine's own mana solver.
The coach gets perfect "can I cast this?" answers.

**Files to change:**
- `src/arenamcp/gamestate.py` — expand action parsing (lines 2231-2324) to extract full AutoTap data
- `src/arenamcp/server.py` — include castability and mana solution in legal_actions output
- `src/arenamcp/coach.py` — simplify/remove manual mana calculation from system prompt

### 1.3 Extract Ability Metadata from Actions

**Current state:** Legal action parsing (gamestate.py:2231-2324) only extracts actionType,
grpId, and card name. All other fields are ignored.

**What to extract:**
- `AbilityPaymentType` — distinguishes `Loyalty` (planeswalker), `TapSymbol`, `None` (spells)
- `AbilityCategory` — `Activated`, `Triggered`, `Static`, `Spell`, `AlternativeCost`
- `AbilitySubCategory` — `Cycling`, `Crew`, `Explore`, `Surveil`, `Scry`, `Investigate`, etc.
- `SourceId` — what permanent is activating the ability
- `AlternativeGrpId` — adventure/MDFC/mutate alternative

**Impact:** The coach can say "activate Jace's +1 loyalty ability" instead of "activate
ability on Jace." Draft helper can identify cycling cards structurally (not by oracle text
regex).

**Files to change:**
- `src/arenamcp/gamestate.py` — expand action parser to extract these fields
- `src/arenamcp/server.py` — include in legal_actions_raw and formatted actions

### 1.4 Parse Timer State Messages

**Current state:** gamestate.py line 2472 silently ignores `GREMessageType_TimerStateMessage`.

**What to extract:**
- Chess clock time remaining for both players
- Timeout extensions remaining (BO3)
- Rope state (how close to timing out)

**Impact:** Coach can warn "You have 30 seconds left — play quickly" or "Opponent is roping,
they may be considering a big play." Autopilot can adjust execution speed.

**Files to change:**
- `src/arenamcp/gamestate.py` — add timer state tracking
- `src/arenamcp/server.py` — expose timer info in game state

---

## Phase 2: Expand BepInEx Plugin (Medium Effort, Transformative)

The current BepInEx plugin only exposes 3 commands: `get_pending_actions`,
`submit_action`, `submit_pass`. The MTGA internals accessed via `MatchManager` and
`InteractionDirector` are far richer.

### 2.1 Add `get_game_state` Command to Plugin

**Current state:** All game state comes from tailing Player.log — a 40+ MB file that
requires brace-depth JSON accumulation, 15MB backfill scanning, and complex
state reconstruction with workarounds for missing data.

**Proposed:** Add a `get_game_state` command to Plugin.cs that reads directly from
`MatchManager`'s GRE interface.

**What MatchManager exposes (from ReVa analysis):**
- `MatchManager._pendingInteraction` — full pending decision (already accessed, line 242)
- `MatchManager.GreInterface` — direct access to the GRE game state object
- Via GreInterface: full game objects, zones, players, turn info, annotations — the same
  data the log gets, but live and complete (no partial diffs, no missing fields)

**Implementation approach:**
```csharp
case "get_game_state":
    var gameManager = FindObjectOfType<GameManager>();
    var matchManager = gameManager?.MatchManager;
    var greInterface = matchManager?.GreInterface;
    // Access game state via greInterface
    // Serialize zones, objects, players, turn info
    // Return as JSON
    break;
```

**Why this matters:**
- **Eliminates log tailing entirely** for in-game state (keep log for match start/end/draft events)
- **No more stale data** — get state at the exact moment you ask
- **No more missing fields** — full state, not partial diffs
- **No more backfill scanning** — instant state on coach startup mid-game
- **Eliminates entire bug classes**: brace-depth parser errors, truncated JSON, race conditions between log write and our read
- **Performance**: one JSON response vs. continuously tailing a 40MB+ file

**Risk:** MatchManager's internal API may change between MTGA updates. Use reflection
as a safety net (already done for _pendingInteraction).

**Files to change:**
- `bepinex-plugin/MtgaCoachBridge/Plugin.cs` — add `get_game_state` command handler
- `src/arenamcp/gre_bridge.py` — add `get_game_state()` method
- `src/arenamcp/gamestate.py` — add `update_from_bridge(data)` path alongside log parsing
- `src/arenamcp/standalone.py` — prefer bridge state when available, fall back to log

### 2.2 Add `get_interaction_detail` Command

**Current state:** The plugin returns the pending action list from `ActionsAvailableRequest`
but nothing about other request types (mulligan, target selection, search, etc.).

**Proposed:** Return rich decision context for all request types:
- `ActionsAvailableRequest` → legal actions (existing)
- `SelectTargetsReq` → valid targets with instance IDs and context
- `DeclareAttackersReq` → legal attackers with attack warnings
- `DeclareBlockersReq` → legal blockers with block warnings
- `SearchReq` → searchable cards with zone info
- `GroupReq` / `SelectNReq` → grouping/selection constraints
- `MulliganReq` → hand contents and mulligan count
- `DistributionReq` → distribution constraints (min/max per target)

**Impact:** The coach and autopilot get structured decision context instead of inferring
it from log messages (which are often incomplete or arrive out of order).

**Files to change:**
- `bepinex-plugin/MtgaCoachBridge/Plugin.cs` — expand ProcessCommand with per-type serializers
- `src/arenamcp/gre_bridge.py` — add `get_interaction_detail()` method
- `src/arenamcp/autopilot.py` — use rich context for better action planning

### 2.3 Add `get_autotap_solutions` Command

**Current state:** The plugin serializes `action.AutoTapSolution != null` as a boolean
(Plugin.cs line 458). The actual tap sequence is discarded.

**Proposed:** Serialize the full AutoTapSolution for each action:
```csharp
if (action.AutoTapSolution != null)
{
    var tapActions = new JArray();
    foreach (var tapAction in action.AutoTapSolution.TapActions)
    {
        tapActions.Add(new JObject
        {
            ["instanceId"] = tapAction.InstanceId,
            ["manaProduced"] = tapAction.ManaProduced.ToString()
        });
    }
    obj["autoTapSolution"] = tapActions;
}
```

**Impact:** The coach knows exactly which lands to tap and can advise "Tap Island + Swamp,
keep Plains untapped for removal" — currently impossible without oracle text analysis.

**Files to change:**
- `bepinex-plugin/MtgaCoachBridge/Plugin.cs` — serialize AutoTapSolution details
- `src/arenamcp/gre_bridge.py` — parse tap solutions in action data

### 2.4 Add `get_timer_state` Command

**Proposed:** Read timer state from MatchManager or the GRE's timer system.

```csharp
case "get_timer_state":
    // Access timer through MatchManager or TimerPackage
    // Return: player1_time, player2_time, active_timer, rope_state, timeouts_remaining
    break;
```

### 2.5 Add `get_match_info` Command

**Proposed:** Read match metadata not available in logs:
- Match ID, game number within match (game 1/2/3)
- Format/event name
- Opponent display name
- Sideboard contents (between games)
- Previous game results in this match

---

## Phase 3: Replay System Integration (Medium Effort, High Value)

### 3.1 Hook TimedReplayRecorder for Auto-Save

**ReVa findings:** MTGA has a complete replay system:
- `Wotc.Mtga.TimedReplays.TimedReplayRecorder` — records games
- `Wotc.Mtga.TimedReplays.ReplayWriter` — serializes to file
- `Wotc.Mtga.TimedReplays.ReplayReader` — deserializes from file
- `Wotc.Mtga.Replays.ReplayGUI` — debug UI for browsing/launching replays
- `SaveDSReplays` property — toggle for auto-saving

**Proposed:** Add BepInEx command to enable replay recording:
```csharp
case "enable_replay_recording":
    // Find or create TimedReplayRecorder
    // Set SaveDSReplays = true
    // Configure replay folder path
    break;

case "get_replay_path":
    // Return path to last saved replay
    break;
```

**Impact on post-match analysis:** Currently, post-match sends a lossy text summary
of trigger history to the LLM. With replays, we'd have every GRE message timestamped —
the LLM gets perfect recall of every play, every decision, every mistake.

**Files to change:**
- `bepinex-plugin/MtgaCoachBridge/Plugin.cs` — add replay enable/path commands
- `src/arenamcp/gre_bridge.py` — add replay methods
- `src/arenamcp/coach.py` — load replay data for post-match analysis
- `src/arenamcp/standalone.py` — enable replay recording at match start

### 3.2 Replay-Powered Match History

Save replay files to `~/.arenamcp/replays/` with metadata (date, opponent, result,
format, deck archetype). Build a match history database that the coach can reference:

- "You're 2-5 against mono-red this session — board in more lifegain"
- "Last time you faced this deck, you lost to flyers — hold removal for their flyers"
- Win rate tracking per archetype, per format

**Files to change:**
- New: `src/arenamcp/match_history.py` — replay storage, indexing, querying
- `src/arenamcp/server.py` — add `get_match_history()` MCP tool
- `src/arenamcp/coach.py` — include match history context in prompts

---

## Phase 4: Prediction Engine (Medium-High Effort, Unique Advantage)

### 4.1 Use GRE's Built-In Prediction System

**ReVa findings:** The GRE protocol includes:
- `ClientMessageType_PredictionReq` — request game state prediction
- `GREMessageType_PredictionResp` — predicted state response
- `EnablePredictionsFieldNumber` — toggle predictions
- `AllowPrediction` — per-action prediction flag
- `ManaSpecType_Predictive` — predictive mana analysis
- `AnnotationType_PredictedDirectDamage` — combat damage predictions
- `IsUnpredictable` — flag for non-deterministic outcomes

**Proposed:** Send prediction requests through BepInEx to ask "what happens if I cast X?"

This would give the coach the game engine's own simulation of outcomes — not LLM
hallucination, not heuristic estimation, but the actual rules engine computing what
happens. This is potentially the biggest competitive advantage possible.

**Investigation needed:**
- Can PredictionReq be sent from the client side, or is it server-only?
- What's the response format? Full game state diff or summary?
- Does it handle all game mechanics or just combat?
- Performance: how fast does it respond?

**Files to change:**
- `bepinex-plugin/MtgaCoachBridge/Plugin.cs` — send prediction requests via GreInterface
- `src/arenamcp/gre_bridge.py` — add `predict_action()` method
- `src/arenamcp/coach.py` — use predictions to validate advice before speaking

### 4.2 Use Advisability Flags

**ReVa findings:** The GRE tags certain choices with `Advisability` and has
`ModalChoiceAdvisability_Discourage` for bad plays.

**Proposed:** Parse advisability from GRE messages and expose to the coach:
- If the game engine itself says a play is discouraged, the coach should flag it
- Useful as a sanity check on LLM-generated advice

**Files to change:**
- `src/arenamcp/gamestate.py` — parse advisability from action messages
- `src/arenamcp/coach.py` — include advisability warnings in prompts

---

## Phase 5: Social & Advanced Features (High Effort, Future Vision)

### 5.1 Discord Rich Presence Coaching Status

**ReVa findings:** MTGA has `DiscordManager` with `FakeRichPresence`, `ActivitySecrets`,
`CreateOrJoinLobby`.

**Proposed:** Show coaching status in Discord:
- "Playing Standard — mtgacoach active"
- Share coaching session link for friends to follow along
- Post-match stats to Discord webhook

### 5.2 Lobby System for Coaching Sessions

**ReVa findings:** Full lobby system: `Client_Lobby`, `Client_LobbyMessage`,
`LobbyController`, `SendCreateLobby`, etc.

**Future:** Create coaching lobbies where a human coach can supervise the AI coach,
override advice, or spectate a student's game.

### 5.3 Tournament Coaching

**ReVa findings:** `TournamentDataProvider`, `Client_TournamentState`,
`Client_TournamentPairing`, `GetTournamentStandings`.

**Future:** Adapt coaching based on tournament context:
- Conservative play when leading in standings
- Aggressive play when elimination is near
- Sideboard advice based on expected meta at the tournament stage

### 5.4 Table Draft Intelligence

**ReVa findings:** `HumanDraftPod`, `TableDraftQueueView`, `BotDraftPod`.

**Future:** In live table drafts, track signals from other drafters:
- What colors/archetypes are open based on what wheels
- Adjust pick recommendations based on pod dynamics
- "Player to your right is in red — cut the Lightning Bolt"

---

## Phase 6: Quality of Life

### 6.1 Add Debug Logging for Dropped Annotations

**Quick fix:** Add an else clause to the annotation handler (gamestate.py:1641) that
logs unhandled annotation types at DEBUG level. This helps identify new annotation types
as MTGA updates.

```python
else:
    logger.debug("Unhandled annotation type: %s (affected: %s)", ann_type, affected_ids)
```

### 6.2 Track Sideboard Between Games

**Current gap:** No sideboard contents exposed to LLM between games in a match.

**Fix:** Parse `SubmitDeckReq` messages in BO3 to capture sideboard changes.
Expose via `get_game_state()` as `sideboard_cards`.

### 6.3 Action History Buffer

**Current gap:** No history of spells cast / actions taken this turn or game.

**Fix:** Maintain a rolling buffer of the last N actions (from `UserActionTaken`
annotations and zone transfers). Expose as `recent_actions` in game state.

### 6.4 Opponent Archetype Detection

**Current gap:** No archetype identification beyond manual color analysis.

**Fix:** After seeing N opponent cards, match against known meta decks from
`get_metagame()`. Report "Opponent is likely playing Azorius Control (78% match)."

---

## Implementation Priority

| # | Upgrade | Effort | Impact | Dependencies | Status |
|---|---------|--------|--------|-------------|--------|
| 1 | 6.1 Debug log dropped annotations | 30 min | Low (diagnostic) | None | **DONE** (2026-03-26) |
| 2 | 1.1 Add missing annotations (TargetSpec, Modified*, Designations) | 1-2 days | High | None | **DONE** (2026-03-26) |
| 3 | 1.2 Parse AutoTap/mana solver data | 1 day | High | None | **DONE** (2026-03-26) |
| 4 | 1.3 Extract ability metadata | 0.5 day | Medium | None | **DONE** (2026-03-26) |
| 5 | 1.4 Parse timer state | 0.5 day | Medium | None | **DONE** (2026-03-26) |
| 6 | 6.2 Track sideboard | 0.5 day | Medium | None | **DONE** (2026-03-26) |
| 7 | 6.3 Action history buffer | 0.5 day | Medium | None | **DONE** (2026-03-26) |
| 8 | 6.4 Opponent archetype detection | 1 day | Medium | None | Pending |
| 9 | 2.3 Plugin: serialize AutoTap solutions | 1 day | High | BepInEx rebuild | **DONE** (2026-03-26) |
| 10 | 2.2 Plugin: rich interaction detail | 2 days | High | BepInEx rebuild | **DONE** (2026-03-26) |
| 11 | 2.1 Plugin: get_game_state from GameManager | 3-5 days | **Transformative** | BepInEx rebuild, reflection exploration | **DONE** (2026-03-26) |
| 12 | 2.4-2.5 Plugin: timer + match info | 1 day | Medium | BepInEx rebuild | **DONE** (2026-03-26) |
| 13 | 3.1 Hook replay recorder | 2-3 days | High | BepInEx, replay format RE | **DONE** (2026-03-26) |
| 14 | 3.2 Match history database | 2 days | High | 3.1 | **DONE** (2026-03-26) |
| 15 | 4.2 Advisability flags | 0.5 day | Medium | None (from log) | Pending |
| 16 | 4.1 GRE prediction engine | 3-5 days | **Transformative** | BepInEx, protocol RE |
| 17 | 5.1-5.4 Social/tournament/draft | Weeks | Future | All above |

**Recommended order:** Items 1-8 first (pure Python, no plugin changes, immediate gains),
then 9-12 (plugin batch), then 13-16 (advanced features).

---

## Key Technical Notes

### Annotation Handling Location
- `src/arenamcp/gamestate.py` lines 1451-1641 — the annotation if/elif chain
- No else clause — unhandled types silently dropped
- Handler receives: `ann_type` (string), annotation dict, `affected_ids` list

### AutoTap Data Location
- `src/arenamcp/gamestate.py` line 2358 — boolean-only check, data discarded
- The GRE sends full `autoTapSolutions` array inside `autoTapActionsReq`
- Each solution contains tap actions with instance IDs and mana produced

### Plugin Architecture
- `Plugin.cs` accesses MTGA via `FindObjectOfType<GameManager>()` (the only relevant MonoBehaviour)
- `GameManager` → `WorkflowController` → `CurrentWorkflow` → `BaseRequest` (the pending interaction)
- `GameManager` → `MatchManager` → `GreInterface` (for direct GRE access)
- **Important:** `MatchManager` and `InteractionDirector` are NOT MonoBehaviours — cannot use `FindObjectOfType` on them. Must go through `GameManager`.
- Uses reflection for `BaseRequest`/`PendingWorkflow` properties (generic type varies per workflow)
- Main thread execution via `ConcurrentQueue<PipeCommand>` + Unity Update()
- Communication: Windows named pipe `\\.\pipe\mtgacoach_gre`, JSON newline-delimited

### GRE Protobuf Message Types
- 45+ message types defined in `Wizards.MDN.GreProtobuf.dll`
- Currently handling: `GameStateMessage`, 29 Req types, 12 Resp types
- Currently ignoring: `UIMessage`, `TimerStateMessage`, `BinaryGameState`,
  `EdictalMessage`, `PredictionResp`, `AllowForceDraw`, `IllegalRequest`

### MTGA Internal Namespaces (from ReVa)
- `Wotc.Mtga.Replays` / `Wotc.Mtga.TimedReplays` — full replay system
- `Wotc.Mtga.AutoPlay` — scripted autoplay framework
- `Core.Shared.Code.DebugTools` — massive debug toolkit (HacksPageGUI, GREWatcherGUI, etc.)
- `Core.Meta.Social.Tables` — lobby/table system
- `Wizards.Mtga.PrivateGame` — direct challenge infrastructure
- `Wizards.Arena.Gathering` — friend/social platform
- `HasbroGo.SocialManager` — Hasbro social SDK (friend/chat/challenge)
