

# Divergence Detection & Advisor Tuning Workflow

Two complementary workflows for improving the advisor based on disagreements between advisor suggestions and your actual decisions.

## Overview

**Mode 1: Manual Play (Replay Analysis)**
- Play WITHOUT advisor
- Make your own decisions
- Post-game: Replay analyzed by advisor
- Review where you disagreed

**Mode 2: Live Advisor (Real-Time Detection)**
- Play WITH advisor active
- When you disagree: Hit F7 to flag
- Post-game: Review flagged decisions
- Approve/reject improvements

## Mode 1: Manual Play + Replay Analysis

### Workflow

```
1. Play match WITHOUT advisor running
   → Make all decisions yourself
   → Arena records replay (if .autoplay enabled)

2. After match: Analyze replay
   → python tools/tune_advisor.py Replay0.rply
   → Advisor processes replay, generates what it WOULD have said

3. Compare decisions:
   Frame 5: You discarded Mountain
           Advisor would say: "Discard Dragon"
           → DIVERGENCE DETECTED

4. Review divergences:
   → python tools/review_divergences.py reports/match_001_divergences.json
   → Mark each: Advisor correct / Player correct / Both valid

5. Update prompts based on review:
   → System highlights which prompts need fixing
   → You manually update coach.py
   → Re-analyze to verify improvement
```

### Advantages
- Learn from your own gameplay expertise
- No bias from advisor's suggestions
- Clean test data (pure human decisions)

### Use Cases
- You're a skilled player who wants to teach the advisor
- Testing new deck/strategy the advisor doesn't understand yet
- Building validation corpus

## Mode 2: Live Advisor + Divergence Detection

### Workflow

```
1. Play match WITH advisor active
   → Advisor gives real-time suggestions
   → You see: "Discard Dragon"

2. You disagree and do something else:
   → You discard Mountain instead
   → System DETECTS divergence automatically
   → 🔔 Tone plays (optional)
   → Notification: "Divergence detected - F7 to flag"

3. Hit F7 to flag for review:
   → Decision marked as "needs review"
   → Continue playing normally

4. After match: Bug report generated
   → Only flagged decisions included
   → python tools/review_divergences.py reports/match_001_divergences.json

5. Review and approve/reject:
   → "Who was correct? [a]dvisor / [p]layer / [b]oth"
   → Add notes explaining why
   → Mark for prompt update if needed

6. System updates prompts automatically (future enhancement)
```

### Advantages
- Flag bad advice in real-time
- Don't review all 50+ decisions, only flagged ones
- Fast feedback loop

### Use Cases
- Active play with advisor assistance
- Catching specific bad advice
- Real-time learning

## Setup

### Enable Divergence Tracking (Mode 2)

**In `standalone.py` or your coach integration:**

```python
from arenamcp.divergence_tracker import (
    start_match_tracking,
    record_advice_given,
    check_player_action,
    flag_current_decision,
    end_match_tracking,
)
from arenamcp.action_detector import detect_player_action

# At match start
match_id = f"match_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
start_match_tracking(match_id)

# When advisor gives advice
def on_advice_given(trigger, game_state, advice):
    record_advice_given(trigger, game_state, advice)
    # Show advice to user...

# When player acts (detect from response messages)
def on_player_action(response_msg, game_state):
    action = detect_player_action(response_msg, game_state)
    if action:
        divergence = check_player_action(action)
        if divergence:
            # DIVERGENCE DETECTED!
            play_notification_tone()  # Optional
            show_notification("Divergence detected - Press F7 to flag")

# F7 hotkey handler
def on_f7_pressed():
    if flag_current_decision():
        show_notification("Decision flagged for review")

# At match end
report = end_match_tracking(save_report=True)
print(f"Match ended: {report['total_divergences']} divergences")
print(f"  {report['user_flagged']} flagged by you")
print(f"Review: python tools/review_divergences.py reports/{match_id}_divergences.json")
```

### Configure F7 Hotkey

**Option A: Python keyboard library (cross-platform)**
```python
import keyboard

keyboard.add_hotkey('f7', on_f7_pressed)
```

**Option B: Windows-specific (using win32api)**
```python
import win32api
import win32con

def check_hotkey():
    if win32api.GetAsyncKeyState(win32con.VK_F7):
        on_f7_pressed()
```

**Option C: Existing debug log hotkey**
Since F7 already triggers debug logs, integrate:
```python
def on_debug_log_triggered():
    # Existing debug log code...
    
    # NEW: Also flag decision for review
    flag_current_decision()
```

## Interactive Review

### Run Review Tool

```bash
# Review latest divergence report
python tools/review_divergences.py divergence_reports/match_20260203_1200_divergences.json
```

### Example Review Session

```
==============================================================
DIVERGENCE REVIEW: match_20260203_1200
==============================================================
Total divergences: 5
User flagged: 2
Auto detected: 3

==============================================================
Divergence 1/5
==============================================================
Frame: 8
Trigger: decision_required
⚠️  YOU FLAGGED THIS (F7)

📢 Advisor said:
   "Discard the Dragon - it costs 6 mana and you only have 3 lands"

👤 You did:
   Selected: Mountain

❓ Who was correct?
   [a] Advisor was correct (I should have followed the advice)
   [p] Player was correct (advisor should change)
   [b] Both valid (situational)
   [s] Skip (unclear/not sure)
   [q] Quit review

Your choice: p

✓ Marked: Player was correct
Why was advisor wrong? (REQUIRED): Should prioritize discarding excess lands over win conditions. I have 5 lands already.
Update prompt based on this? [y/n]: y

==============================================================
Divergence 2/5
==============================================================
...

==============================================================
REVIEW SUMMARY
==============================================================
Reviewed: 5/5
Advisor correct: 1
Player correct: 3
Prompt updates needed: 2

==============================================================
PROMPT UPDATE SUGGESTIONS
==============================================================

decision_required:
  Issue: Discard the Dragon - it costs 6 mana and you only...
  Fix: Should prioritize discarding excess lands over win conditions

combat_attackers:
  Issue: Attack with all creatures
  Fix: Should consider opponent's instant-speed removal

✓ Saved reviewed report to divergence_reports/match_20260203_1200_divergences.json
```

## Prompt Update Workflow

### Manual Update (Current)

1. Review suggestions from review tool
2. Edit `src/arenamcp/coach.py`
3. Find relevant `DECISION_PROMPTS` or system prompt
4. Update based on feedback

Example:
```python
# Before
DECISION_PROMPTS["discard"] = """
Choose which card(s) to discard.
Discard highest CMC card you can't cast.
"""

# After (based on review feedback)
DECISION_PROMPTS["discard"] = """
Choose which card(s) to discard.
Priority order:
1. Excess lands if you have 4+ already
2. Highest CMC card you can't cast
3. Redundant copies
Keep: removal, counters, win conditions
"""
```

5. Test updated prompts:
   ```bash
   # Re-analyze same matches with new prompts
   python tools/tune_advisor.py --latest 5
   # Check if "Player correct" count decreases
   ```

### Automated Update (Future Enhancement)

```python
# System auto-generates prompt improvements
python tools/auto_tune_prompts.py divergence_reports/*.json --apply

# Reviews all feedback
# Identifies common patterns
# Generates prompt diffs
# Applies with user confirmation
```

## Metrics & Tracking

### Measure Improvement Over Time

```bash
# Track divergence rate over multiple matches
python tools/divergence_stats.py divergence_reports/

Output:
  Week 1 (10 matches):
    Total decisions: 142
    Divergences: 28 (19.7%)
    Player correct: 18 (64% of divergences)
    
  Week 2 (10 matches):
    Total decisions: 156
    Divergences: 15 (9.6%) ↓
    Player correct: 6 (40% of divergences) ↓
    
  Improvement: 10.1% reduction in divergence rate
              24% reduction in advisor errors
```

### Goal Metrics

- **Divergence rate**: <10% (you disagree with advisor <10% of the time)
- **Advisor accuracy when divergence**: >70% (when you disagree, advisor is right 70%+ of the time)
- **User flags**: <5 per match (only truly bad advice gets flagged)

## Notification Options

### Visual Notification

```python
def show_notification(message):
    # Option 1: Console (TUI)
    print(f"\n⚠️  {message}\n")
    
    # Option 2: Windows toast notification
    from win10toast import ToastNotifier
    toaster = ToastNotifier()
    toaster.show_toast("ArenaMCP", message, duration=3)
    
    # Option 3: Overlay on screen
    # (Would need overlay library like tkinter or pygame)
```

### Audio Notification

```python
def play_notification_tone():
    # Option 1: Simple beep
    import winsound
    winsound.Beep(1000, 200)  # 1000 Hz for 200ms
    
    # Option 2: Custom sound file
    import pygame
    pygame.mixer.init()
    pygame.mixer.music.load("divergence_detected.wav")
    pygame.mixer.music.play()
    
    # Option 3: Text-to-speech
    from arenamcp.tts import VoiceOutput
    voice = VoiceOutput("elevenlabs")
    voice.speak("Divergence detected")
```

## Integration Examples

### Add to Standalone Coach

```python
# In standalone.py

from arenamcp.divergence_tracker import (
    start_match_tracking, record_advice_given, 
    check_player_action, end_match_tracking
)

class StandaloneCoach:
    def __init__(self):
        # ... existing init ...
        self._current_match_id = None
        self._tracking_enabled = True
    
    def on_match_start(self):
        if self._tracking_enabled:
            self._current_match_id = f"match_{datetime.now():%Y%m%d_%H%M%S}"
            start_match_tracking(self._current_match_id)
    
    def on_advice_given(self, trigger, state, advice):
        if self._tracking_enabled:
            record_advice_given(trigger, state, advice)
        # ... show advice to user ...
    
    def on_log_message(self, msg):
        # ... existing processing ...
        
        # Check if player took action
        if "Resp" in msg_type:
            action = detect_player_action(msg, current_state)
            if action and self._tracking_enabled:
                divergence = check_player_action(action)
                if divergence:
                    self.on_divergence_detected(divergence)
    
    def on_divergence_detected(self, divergence):
        # Play tone
        winsound.Beep(1000, 200)
        # Show notification
        print(f"\n⚠️  DIVERGENCE: Press F7 to flag for review\n")
    
    def on_match_end(self):
        if self._tracking_enabled and self._current_match_id:
            report = end_match_tracking()
            if report['total_divergences'] > 0:
                print(f"\n{'='*60}")
                print(f"Match Report: {report['total_divergences']} divergences detected")
                print(f"  {report['user_flagged']} flagged by you")
                print(f"\nReview: python tools/review_divergences.py")
                print(f"        divergence_reports/{self._current_match_id}_divergences.json")
                print(f"{'='*60}\n")
```

### Add to GPT-Realtime Voice Coach

```python
# In voice coach integration

def on_advice_generated(advice_text):
    # Record for divergence tracking
    record_advice_given(current_trigger, current_state, advice_text)
    
    # Speak advice
    speak(advice_text)

def on_user_speaks(transcription):
    # Check if user is reporting a divergence
    if "that was wrong" in transcription.lower():
        flag_current_decision()
        speak("Flagged for review. What should I have said?")
    
    # Or detect implicit divergence from action
    action = parse_user_action(transcription)
    if action:
        divergence = check_player_action(action)
        if divergence:
            speak("I notice you did something different. Want to flag this?")
```

## Files Structure

```
ArenaMCP/
├── src/arenamcp/
│   ├── divergence_tracker.py    # Core tracking system
│   ├── action_detector.py       # Detect player actions from logs
│   └── ...
│
├── tools/
│   ├── review_divergences.py    # Interactive review CLI
│   ├── divergence_stats.py      # Metrics/analytics (future)
│   └── ...
│
├── divergence_reports/          # Generated reports (gitignored)
│   ├── match_20260203_1200_divergences.json
│   └── ...
│
└── DIVERGENCE_WORKFLOW.md       # This file
```

## Troubleshooting

### "No divergences detected" but I disagreed many times

- Check action detection is working (add logging)
- Verify `detect_player_action()` is being called
- Action matching might be too lenient (adjust `_actions_match()` threshold)

### F7 not working

- Check hotkey library is installed: `pip install keyboard`
- On Linux: may need sudo
- Alternative: use in-game command instead of hotkey

### Too many false positive divergences

- Adjust matching threshold in `DivergenceTracker._actions_match()`
- Current: 50% word overlap
- Increase to 70% for stricter matching

### Can't find divergence reports

- Default location: `./divergence_reports/`
- Check `DivergenceTracker` initialization
- Verify `save_report=True` in `end_match_tracking()`

## Future Enhancements

- [ ] Automated prompt updates based on review patterns
- [ ] Machine learning to predict which divergences matter
- [ ] Integration with Telegram for remote review
- [ ] Voice-based review ("advisor was wrong because...")
- [ ] A/B testing different prompts on same replays
- [ ] Crowd-sourced review (multiple users rate same decision)
- [ ] Visual overlay showing divergence in real-time

## Summary

**Mode 1 (Manual Play):** Best for building validation corpus  
**Mode 2 (Live Detection):** Best for fast iteration and catching bad advice

Both modes feed into the same review workflow, allowing you to:
1. Identify advisor mistakes
2. Provide expert feedback
3. Update prompts systematically
4. Measure improvement quantitatively

Start with Mode 2 (live detection) for fastest results!
