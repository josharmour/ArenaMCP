# Before/After Comparison - Bug Fixes

## Visual Examples of Changes

### Example 1: Dual Land Mana Display

**Scenario:** Turn 3, you have Temple of Mystery (untapped) on the battlefield.

**BEFORE:**
```
Mana: 1 (U:1 G:1)
HAND:
  Llanowar Elves {G} [S,OK]
```
❌ **Problem:** Looks like you have 2 mana (U:1 AND G:1), but you only have 1 mana total!

**AFTER:**
```
Mana: 1 (sources: {U/G})
HAND:
  Llanowar Elves {G} [S,OK]
```
✅ **Fixed:** Clear that you have 1 source producing U OR G (not both).

---

### Example 2: Multiple Dual Lands

**Scenario:** Temple of Mystery (untapped), Breeding Pool (untapped)

**BEFORE:**
```
Mana: 2 (U:2 G:2)
```
❌ **Problem:** Looks like you have 4 mana total (2U + 2G)! Actually only 2 mana.

**AFTER:**
```
Mana: 2 (sources: {U/G} {U/G})
```
✅ **Fixed:** Clear that you have 2 sources, each producing U or G.

---

### Example 3: Mixed Mana Sources

**Scenario:** Plains, Island, Breeding Pool (all untapped)

**BEFORE:**
```
Mana: 3 (W:1 U:2 G:1)
```
❌ **Problem:** Looks like W:1 + U:2 + G:1 = 4 mana, but only 3 lands!

**AFTER:**
```
Mana: 3 (sources: W U {U/G})
```
✅ **Fixed:** Clear you have 3 sources: Plains (W), Island (U), Breeding Pool (U or G).

---

## System Prompt Changes

### Change 1: Resource Efficiency

**BEFORE:**
```
CRITICAL MANA RULES:
- Cards tagged [CAN CAST] are the ONLY cards you can suggest casting.
- Cards tagged [NEED X mana] CANNOT be cast - do NOT suggest them!
- Do NOT perform your own mana calculations - trust the tags completely.
```

**AFTER:**
```
CRITICAL MANA RULES:
- Cards tagged [OK] or [CAN CAST] are castable RIGHT NOW with available mana - no additional mana needed!
- Cards tagged [NEED X] CANNOT be cast - do NOT suggest them!
- Do NOT perform your own mana calculations - trust the tags completely.
- If a card shows [OK], you already have enough mana. Don't suggest paying extra life/resources for more mana.
- RESOURCE EFFICIENCY: Don't waste life or mana. If you can cast a spell with current mana, don't pay extra.
- The "sources:" display shows what mana EACH source can produce (e.g., "{U/G}" means one source producing U OR G, not both).
```

✅ **Added:** Explicit guidance about resource efficiency and not wasting life unnecessarily.

---

### Change 2: Legal Actions Enforcement

**BEFORE:**
```
CRITICAL GAME RULES:
- The "Legal:" line lists ALL valid actions. ONLY suggest actions listed there.
- Creatures tagged [SS] have SUMMONING SICKNESS — they CANNOT attack or use tap abilities this turn.
```

**AFTER:**
```
CRITICAL GAME RULES:
- The "Legal:" line lists ALL valid actions. ONLY suggest actions listed there.
- NEVER suggest actions not in the Legal: line. If you want to cast a spell, it MUST appear as "Cast [card name]" in Legal:.
- Do NOT hallucinate actions like "flash in" or "hold up" unless they are explicitly legal actions.
- Creatures tagged [SS] have SUMMONING SICKNESS — they CANNOT attack or use tap abilities this turn.
```

✅ **Added:** Stronger constraints against hallucinating illegal actions like "flash in" non-instant creatures.

---

## Advice Examples

### Bug Scenario: Breeding Pool Decision

**Game State:**
- Turn 3, YOUR Main1
- Board: Temple of Mystery (untapped)
- Hand: Llanowar Elves {G} [S,OK], Breeding Pool [S,LAND]
- Mana: 1 (sources: {U/G})
- Life: 20

**BEFORE (Bad Advice):**
> "Play Breeding Pool. Pay the 2 life so it enters untapped. We need the mana immediately to cast Llanowar Elves and start ramping."

❌ **Problems:**
1. Llanowar Elves is already [S,OK] - castable with current mana!
2. Paying 2 life is unnecessary
3. Wasted 10% of starting life for no benefit

**AFTER (Good Advice):**
> "Play Breeding Pool, let it enter tapped. You already have enough mana to cast Llanowar Elves with Temple of Mystery."

✅ **Correct:**
1. Recognizes Llanowar Elves is already castable
2. Preserves life resource
3. Still plays the land (land drops are free and should be used)

---

### Bug Scenario: Opponent's Turn

**Game State:**
- Turn 4, OPPONENT'S Main1
- Legal: Wait (Opponent has priority)
- Timing: INSTANTS ONLY
- Hand: Badgermole Cub {1}{G} [S,OK], Leyline Weaver {1}{R/G} [S,OK]

**BEFORE (Bad Advice):**
> "Opponent is on Izzet colors with a slow start. We need to race them before they stabilize; plan to flash in Badgermole Cub or Leyline Weaver if they pass, or just untap and slam Tyvar to pressure them. Wait for their move."

❌ **Problems:**
1. Badgermole Cub and Leyline Weaver are NOT instants and don't have flash
2. "flash in" is not a legal action (not in Legal: list)
3. Hallucinating abilities the cards don't have

**AFTER (Good Advice):**
> "Wait. Opponent is setting up their control game with Gran-Gran. Plan to untap and cast Tyvar or a mana dork to ramp into threats."

✅ **Correct:**
1. Only suggests "Wait" which is the only legal action
2. No hallucinated flash abilities
3. Plans for next turn using sorcery-speed spells correctly
