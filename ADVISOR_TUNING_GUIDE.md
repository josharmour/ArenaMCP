

# Advisor Tuning Guide

Complete workflow for analyzing and improving advisor performance using Arena's replay system.

## Overview

This system lets you:
1. **Play matches** with Arena's native replay recording enabled
2. **Analyze replays** to see what advice was given vs what should be given
3. **Get improvement suggestions** for prompts and decision handling
4. **Iterate and improve** the advisor based on real game data

## Setup (One-Time)

### 1. Enable Arena Replay Recording

Create an empty file named `.autoplay` at this location:

**Windows:**
```
%APPDATA%\..\LocalLow\Wizards Of The Coast\MTGA\ArenaAutoplayConfigs\.autoplay
```

Full path (PowerShell):
```powershell
mkdir "$env:APPDATA\..\LocalLow\Wizards Of The Coast\MTGA\ArenaAutoplayConfigs" -Force
New-Item -ItemType File -Path "$env:APPDATA\..\LocalLow\Wizards Of The Coast\MTGA\ArenaAutoplayConfigs\.autoplay" -Force
```

**Android:**
```
/storage/emulated/0/Android/data/com.wizards.mtga/files/ArenaAutoplayConfigs/.autoplay
```

**iOS (theoretical):**
```
/var/mobile/Containers/Data/Application/<guid>/Documents/ArenaAutoplayConfigs/.autoplay
```

### 2. Verify Setup

1. Launch MTGA
2. During a match, hold **Alt** key (PC) or **three-finger tap** (Android)
3. You should see a debug panel appear
4. Look for "Record" button - this confirms replay is enabled

## Usage Workflow

### Step 1: Play Matches with Recording

1. Start a match in Arena
2. Hold Alt to open debug panel
3. Click **Record** button before game starts
4. Play normally (advisor gives advice as usual)
5. After match, replay is saved to:
   ```
   %LOCALAPPDATA%\..\LocalLow\Wizards Of The Coast\MTGA\Replays\Replay0.rply
   ```

### Step 2: Analyze the Replay

Run the tuning tool on your replay file:

```bash
# Analyze specific replay
python tools/tune_advisor.py path/to/Replay0.rply

# Analyze latest 3 replays
python tools/tune_advisor.py --latest 3

# Analyze all replays
python tools/tune_advisor.py --all
```

### Step 3: Review the Report

The tool outputs:

**Console Summary:**
```
ADVISOR TUNING REPORT
=========================================
Decisions Analyzed: 12
  ✓ Correct advice:   8
  ✗ Incorrect advice: 3
  ø Missing advice:   1
  ↑ Improved:         4

IMPROVEMENT SUGGESTIONS
=========================================
1. Ensure decision_required trigger fires correctly for discard decisions
2. Mana calculation tags may be incorrect for hybrid costs
3. Combat math needs adjustment for first strike interactions
```

**JSON Report:**
Saved to `<replay_name>_analysis.json` with full details:
- Every decision point frame-by-frame
- What advice was given vs optimal
- Specific prompt improvement suggestions

### Step 4: Iterate on Prompts

Based on suggestions, update prompts in:
- `src/arenamcp/coach.py` → `DECISION_PROMPTS`
- `src/arenamcp/coach.py` → `DEFAULT_SYSTEM_PROMPT`

Then re-analyze the same replay to verify improvements:
```bash
python tools/tune_advisor.py Replay0.rply
```

Compare the new report to see if "Improved" count increases.

## Advanced Usage

### Repair Corrupted Replays

Arena's replay recording sometimes has timing issues causing "Timeout Exceeded" errors. Fix with:

```bash
python tools/tune_advisor.py Replay0.rply --repair
```

This removes `onHover` messages which cause most corruption issues.

### Compare Old vs New Advice

If you have logs of advice that was given during a match:

```bash
python -m arenamcp.advisor_tuner recording.json --advice-log advice.jsonl
```

Format of `advice.jsonl`:
```json
{"frame_number": 0, "advice": "Mulligan - only 1 land, no keepable hand"}
{"frame_number": 5, "advice": "Play Forest, cast Llanowar Elves"}
```

### Remote Testing

Since replay files are portable:

1. Play matches on your gaming PC (recording enabled)
2. Upload `.rply` files to cloud/Dropbox
3. Download on work laptop
4. Run tuning analysis remotely
5. Get improvement suggestions via Telegram (future enhancement)

### Automated Testing

Process multiple replays in batch:

```bash
# Analyze all replays and aggregate results
for f in Replays/*.rply; do
    python tools/tune_advisor.py "$f" --output reports/
done

# Aggregate all reports
python tools/aggregate_reports.py reports/*.json
```

## Understanding the Analysis

### Decision Categories

**✓ Correct Advice:** Current prompts produce same advice as was given
- No changes needed for this scenario

**✗ Incorrect Advice:** Current prompts produce different advice
- Review `optimal_advice` in report
- Check if new advice is actually better
- Update prompts if new advice is superior

**ø Missing Advice:** Advisor didn't respond when it should have
- Check trigger detection
- Verify decision_context is being captured
- May indicate a bug in decision detection

**↑ Improved:** Newer prompts produce better advice than was given
- Validates recent optimizations
- Track these to measure improvement over time

### Common Issues & Fixes

**Issue:** "Missing advice for discard decisions"
**Fix:** Check that `SelectNReq` messages with discard context are triggering correctly in `gamestate.py`

**Issue:** "Combat math incorrect"
**Fix:** Review combat analysis in `coach.py` → `_format_game_context`

**Issue:** "Mana calculations wrong"
**Fix:** Verify mana pool calculation and castability logic in prompt context

**Issue:** "Advice too verbose"
**Fix:** Reduce `max_tokens` or tighten `CONCISE_SYSTEM_PROMPT`

## Files & Locations

### Arena Replay Files
```
Windows: %LOCALAPPDATA%\..\LocalLow\Wizards Of The Coast\MTGA\Replays\
Format: Replay0.rply, Replay1.rply, etc.
```

### Converted Recordings
```
Location: Same as replay file
Format: <replay_name>_converted.json
```

### Analysis Reports
```
Location: Same as replay file (or --output dir)
Format: <replay_name>_analysis.json
```

## Troubleshooting

### Debug panel doesn't appear when holding Alt
- Verify `.autoplay` file exists in correct location
- Restart MTGA after creating the file
- File must be named exactly `.autoplay` (no extension)

### Replay files not being created
- Click "Record" button in debug panel BEFORE match starts
- Check Replays folder exists
- Verify disk space available

### "Timeout Exceeded" error when analyzing
- Run with `--repair` flag to remove hover messages
- Some replays are unrecoverable if too corrupted

### Analysis shows "0 decisions"
- Replay may be from tutorial/bot match (limited decision points)
- Check converted recording has frames with triggers
- Verify game state parsing is working (check logs)

## Tips for Best Results

1. **Record diverse scenarios:**
   - Aggro, midrange, control matchups
   - Games with discard, scry, modal spells
   - Games where you made mistakes (best learning data!)

2. **Analyze fresh replays:**
   - Replay immediately after playing for freshest memory
   - Note specific decisions you want to check

3. **Track improvements:**
   - Keep a spreadsheet of "Correct/Incorrect/Missing" over time
   - Measure accuracy % trending upward

4. **Focus on patterns:**
   - If same type of decision fails repeatedly, prioritize fixing it
   - One good prompt fix can improve hundreds of future decisions

5. **Iterate quickly:**
   - Make small prompt changes
   - Re-analyze same replay to verify improvement
   - Don't change too many things at once

## Integration with Development

### Before Committing Prompt Changes

```bash
# Test against your validation corpus
python tools/tune_advisor.py --all --output validation/

# Check that "Improved" count is > "Degraded" count
# Commit if overall improvement
```

### Continuous Improvement Loop

```
Play matches → Record replays → Analyze → Identify issues → 
Update prompts → Re-analyze → Verify improvement → Commit → Repeat
```

## Future Enhancements

- [ ] Automated Telegram notifications with analysis results
- [ ] Web dashboard for replay analysis visualization
- [ ] A/B testing different prompt versions on same replay
- [ ] Prompt optimization using LLM meta-analysis
- [ ] Integration with Arena's autoplay AI (if we can reverse engineer it)

## Credits

Arena replay format discovered and documented by community members. 
Debug panel has existed since ~2020 for WotC's internal tournament use.
