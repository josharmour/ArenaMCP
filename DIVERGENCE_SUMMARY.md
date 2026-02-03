# Divergence Detection System - Implementation Summary

## What Was Built

A complete system for detecting when you disagree with the advisor and using those disagreements to improve prompt quality.

### Core Components

**1. divergence_tracker.py** - Real-time divergence tracking
- Records when advice is given
- Detects when player action differs from advice
- F7 hotkey flagging support
- Generates post-game reports

**2. action_detector.py** - Player action detection
- Parses Arena response messages
- Extracts what player actually did
- Handles: discard, scry, targets, combat, mulligans
- Converts actions to human-readable descriptions

**3. review_divergences.py** - Interactive review CLI
- Step through each divergence
- Mark: Advisor correct / Player correct / Both valid
- Add notes explaining why
- Flag decisions for prompt updates
- Track improvement metrics

### Two Workflows

**Mode 1: Manual Play + Replay Analysis**
```
Play WITHOUT advisor → Replay analyzed post-game → 
Advisor says what it WOULD have said → Review differences →
Update prompts
```

**Mode 2: Live Advisor + Real-Time Detection**
```
Play WITH advisor → Divergence detected automatically →
Hit F7 to flag → Post-game review of ONLY flagged decisions →
Update prompts
```

## How It Works

### During Match (Mode 2)

```python
# Advisor gives advice
record_advice_given(trigger="decision_required", 
                   game_state=current_state,
                   advice="Discard the Dragon")

# Player acts differently
action = detect_player_action(response_msg, game_state)
# action = "Selected: Mountain"

# System detects divergence
divergence = check_player_action(action)
if divergence:
    play_tone()  # 🔔
    show_notification("Press F7 to flag")

# User presses F7
flag_current_decision()  # Marked for review

# Match ends
report = end_match_tracking()
# → divergence_reports/match_001_divergences.json saved
```

### Post-Game Review

```bash
python tools/review_divergences.py divergence_reports/match_001_divergences.json
```

Interactive prompts:
```
Advisor said: "Discard Dragon"
You did: "Selected Mountain"

Who was correct? [a]dvisor / [p]layer / [b]oth: p
Why was advisor wrong?: Should prioritize excess lands over win-cons
Update prompt? [y/n]: y

✓ Flagged for prompt update
```

### Metrics & Improvement

Track over time:
- Divergence rate: % of decisions where you disagree
- Advisor accuracy: When you disagree, how often was advisor right?
- Goal: <10% divergence rate, >70% advisor accuracy

## Integration Points

### Add to Standalone Coach

```python
from arenamcp.divergence_tracker import (
    start_match_tracking,
    record_advice_given,
    check_player_action,
    flag_current_decision,
    end_match_tracking,
)
from arenamcp.action_detector import detect_player_action

# Match lifecycle
start_match_tracking(match_id)  # On match start
record_advice_given(...)        # When advice given
check_player_action(...)        # When player acts
flag_current_decision()         # F7 pressed
end_match_tracking()            # Match ends
```

### F7 Hotkey Setup

```python
import keyboard
keyboard.add_hotkey('f7', flag_current_decision)
```

Or integrate with existing debug log F7 handler.

## Example Output

### Divergence Report (JSON)
```json
{
  "match_id": "match_20260203_1200",
  "total_divergences": 5,
  "user_flagged": 2,
  "auto_detected": 3,
  "divergences": [
    {
      "frame": 8,
      "trigger": "decision_required",
      "advice_given": "Discard the Dragon",
      "action_taken": "Selected: Mountain",
      "flagged_by_user": true,
      "timestamp": "2026-02-03T12:15:00"
    }
  ]
}
```

### Review Summary
```
REVIEW SUMMARY
==============
Reviewed: 5/5
Advisor correct: 1
Player correct: 3
Prompt updates needed: 2

PROMPT UPDATE SUGGESTIONS
==========================
decision_required:
  Fix: Prioritize excess lands over win-cons
  
combat_attackers:
  Fix: Consider opponent's instant-speed removal
```

## Benefits

**Fast Feedback Loop**
- Flag bad advice in real-time
- Review only flagged decisions (not all 50+)
- Immediate improvement cycle

**Data-Driven**
- Objective metrics (divergence rate)
- Track improvement quantitatively
- Prioritize common failure patterns

**Expert Knowledge Transfer**
- Your expertise teaches the advisor
- Notes explain WHY decisions are wrong
- Systematic prompt improvement

**Flexible**
- Works with or without Arena replays
- Mode 1 (manual) or Mode 2 (live)
- Remote review possible (export reports)

## Next Steps

1. **Integrate into standalone.py**
   - Add tracker initialization
   - Hook action detection
   - Set up F7 hotkey

2. **Play test matches**
   - Enable tracking
   - Flag 5-10 bad decisions
   - Run interactive review

3. **Update prompts**
   - Based on review feedback
   - Re-test same scenarios
   - Measure divergence rate decrease

4. **Iterate**
   - Play more matches
   - Track metrics over time
   - Goal: <10% divergence rate

## Files Added

1. `src/arenamcp/divergence_tracker.py` (300 lines)
2. `src/arenamcp/action_detector.py` (240 lines)
3. `tools/review_divergences.py` (220 lines)
4. `DIVERGENCE_WORKFLOW.md` (complete guide)
5. `DIVERGENCE_SUMMARY.md` (this file)

**Total:** ~760 lines + comprehensive documentation

## Status

✅ Core system implemented  
✅ Action detection works  
✅ Interactive review tool ready  
✅ Documentation complete  

⏳ Pending: Integration into standalone.py (your choice when/how)

## Ready to Use

The system is fully functional. To start using:

1. Import divergence_tracker into your coach integration
2. Add calls at appropriate lifecycle points
3. Set up F7 hotkey
4. Play a match and flag bad advice
5. Run review tool after match
6. Update prompts based on feedback

**Estimated integration time:** 30-60 minutes

---

**Implementation Time:** ~2 hours  
**Benefits:** Systematic advisor improvement via expert feedback  
**Next Action:** Integrate into standalone coach and test!
