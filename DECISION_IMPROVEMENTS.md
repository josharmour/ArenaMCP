# Decision Advice Improvements

## Problem
Advisor was silent during micro-decisions (discard, scry, modal spells) even after suggesting to cast those spells.

## Solution: 3-Phase Implementation

### Phase 1: Rich Decision Context Capture ✅
**File:** `gamestate.py`

Added new fields:
- `decision_context` - dict with decision type, options, source card
- `decision_timestamp` - when decision was detected

Enhanced capture for:
- **SelectNReq** (discard/scry): Detects type, count, min/max
- **SelectTargetsReq**: Captures source card from stack
- **GroupOptionReq**: Captures modal spell options

Example context:
```python
{
    "type": "discard",
    "count": 1,
    "min": 1,
    "max": 1,
}
```

### Phase 2: Decision-Specific Prompts ✅
**File:** `coach.py`

Added `DECISION_PROMPTS` dictionary with specialized guidance:

**Discard:**
- Priority order: excess lands > high CMC uncastables > redundant copies
- Keep: removal, counters, win conditions

**Scry:**
- Keep: needed lands/threats you can cast
- Bottom: dead cards, redundant pieces

**Surveil:**
- Keep: want to draw next
- Graveyard: enables synergy or digging deeper

**Target Selection:**
- Evaluate: biggest threat vs best value
- Consider opponent's likely responses

**Modal Spells:**
- Compare each mode's immediate impact
- Which advances win condition?

System prompt is dynamically enhanced when decisions are detected.

### Phase 3: Prevent Premature Clearing ✅
**File:** `gamestate.py`

**Before:** Any response message cleared decisions → advisor lost context before it could respond

**After:** Only clear on actual decision response types:
- `SelectTargetsResp`
- `SelectNResp`
- `GroupOptionResp`
- `SubmitDeckResp` (mulligan)
- `PromptResp`

UI events (hover, mouseover) no longer clear pending decisions.

## Expected Behavior

### Example 1: Discard
```
Turn 5: You cast Sephiroth ability
[Trigger: SelectNReq detected]
Context shows:
  !!! DECISION: DISCARD 1 card(s) !!!
  Choose: excess lands > high CMC uncastables > redundant copies

Advisor: "Discard the Mountain - you have 5 lands already."
```

### Example 2: Scry
```
You cast Opt
[Trigger: SelectNReq with scry context]
Context shows:
  !!! DECISION: SCRY 1 !!!
  Keep: needed lands/threats | Bottom: dead cards

Advisor: "Bottom it - you need threats, not another land."
```

### Example 3: Modal Spell
```
You cast Charm spell
[Trigger: GroupOptionReq]
Context shows:
  !!! DECISION: CHOOSE MODE (3 options) !!!
  Evaluate: which mode solves current problem best

Advisor: "Choose mode 1 - removes their flyer, opens lethal."
```

## Testing
Run actual Arena games and cast:
- Cards with discard (Liliana, Sephiroth)
- Scry spells (Opt, Preordain)
- Modal spells (Charms, Commands)
- Removal with targets

Verify advisor gives specific guidance for each decision type.

## Files Modified
1. `src/arenamcp/gamestate.py` - Context capture + clearing logic
2. `src/arenamcp/coach.py` - Decision prompts + context display
