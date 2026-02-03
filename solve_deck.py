import logging
import sys
# Ensure src is in path
sys.path.append('src')

from arenamcp.scryfall import ScryfallCache
from arenamcp.draftstats import DraftStatsCache
from arenamcp.sealed_eval import analyze_sealed_pool, format_sealed_detailed

# logging.basicConfig(level=logging.INFO)

raw_pool_text = """
1 Evershrike's Gift (ECL) 15
3 Swamp (ECL) 271
1 Encumbered Reejerey (ECL) 14
1 Riverguard's Reflexes (ECL) 33
2 Timid Shieldbearer (ECL) 39
1 Wanderbrine Preacher (ECL) 41
1 Moonlit Lamenter (ECL) 26
1 Pyrrhic Strike (ECL) 30
1 Reluctant Dounguard (ECL) 31
1 Morningtide's Light (ECL) 27
2 Shore Lurker (ECL) 34
1 Slumbering Walker (ECL) 35
1 Sun-Dappled Celebrant (ECL) 37
2 Summit Sentinel (ECL) 73
3 Island (ECL) 270
1 Wild Unraveling (ECL) 84
1 Glamermite (ECL) 50
1 Silvergill Peddler (ECL) 70
1 Wanderwine Distracter (ECL) 82
1 Lofty Dreams (ECL) 58
3 Bile-Vial Boggart (ECL) 87
4 Mountain (ECL) 272
2 Boggart Prankster (ECL) 93
2 Bogslither's Embrace (ECL) 94
1 Iron-Shield Elf (ECL) 108
1 Boggart Mischief (ECL) 92
1 Heirloom Auntie (ECL) 107
1 Moonglove Extractor (ECL) 109
1 Mornsong Aria (ECL) 111
1 Dawnhand Eulogist (ECL) 99
1 Graveshifter (ECL) 104
1 Blighted Blackthorn (ECL) 90
2 Dose of Dawnglow (ECL) 100
2 Soulbright Seeker (ECL) 157
4 Plains (ECL) 269
1 Explosive Prodigy (ECL) 136
1 Hexing Squelcher (ECL) 145
2 Reckless Ransacking (ECL) 152
1 Warren Torchmaster (ECL) 163
1 Elder Auntie (ECL) 133
1 Enraged Flamecaster (ECL) 135
2 End-Blaze Epiphany (ECL) 134
1 Blossoming Defense (ECL) 167
2 Forest (ECL) 273
1 Virulent Emissary (ECL) 202
1 Bristlebane Battler (ECL) 168
1 Great Forest Druid (ECL) 178
1 Shimmerwilds Growth (ECL) 194
1 Formidable Speaker (ECL) 176
2 Tend the Sprigs (ECL) 197
1 Unforgiving Aim (ECL) 200
2 Mistmeadow Council (ECL) 183
1 Wildvine Pummeler (ECL) 203
1 Prideful Feastling (ECL) 238
1 Eclipsed Flamekin (ECL) 219
2 Flaring Cinder (ECL) 225
1 Overgrown Tomb (ECL) 266
1 Gangly Stompling (ECL) 226
1 Feisty Spikeling (ECL) 223
1 Wary Farmer (ECL) 251
2 Chitinous Graspling (ECL) 211
1 Glister Bairn (ECL) 227
2 Stalactite Dagger (ECL) 261
1 Changeling Wayfinder (ECL) 1
1 Firdoch Core (ECL) 255
1 Gathering Stone (ECL) 257
1 Rooftop Percher (ECL) 2
"""

import re

def parse_pool(text):
    names = []
    # Pattern to match: Count Name (Set) Num
    # e.g. "1 Evershrike's Gift (ECL) 15" -> "Evershrike's Gift" (1 time)
    # The name can contain spaces, apostrophes, etc.
    # We can split by line, then for each line:
    # 1. Take the first part as count
    # 2. Find the index of the first '(' which starts the Set code
    # 3. Everything between count and '(' is the name
    
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line: continue
        
        parts = line.split(' ', 1)
        if len(parts) < 2: continue
        
        try:
            count = int(parts[0])
        except ValueError:
            continue
            
        rest = parts[1]
        # Find last occurrence of (SET) or just look for (ECL)
        # Actually simplest is to find the index of " (ECL)" or generally " (" 
        # But some cards might have parens in name? Not common in this list.
        # Let's split by " (" and take everything before the last one if multiple?
        # Or specifically split by " (ECL)" since we know the set.
        
        if " (" in rest:
            name = rest.rsplit(" (", 1)[0]
        else:
            name = rest
            
        # Add 'count' times
        for _ in range(count):
            names.append(name)
            
    return names

raw_names = parse_pool(raw_pool_text)


def solve():
    print("Initializing caches...")
    scryfall = ScryfallCache()
    draft = DraftStatsCache()
    
    set_code = "ecl" 
    try:
        draft.load_set(set_code)
    except Exception as e:
        print(f"Error loading draft stats: {e}")
    
    # 1. Parse and Enrich
    all_cards = []
    print("Enriching card data...")
    for name in raw_names:
        if name in ["Plains", "Island", "Swamp", "Mountain", "Forest"]:
            continue # Basic lands handled separately
            
        sc_card = scryfall.get_card_by_name(name)
        if not sc_card:
            continue
            
        stats = draft.get_draft_rating(sc_card.name, set_code)
        wr = stats.gih_wr if (stats and stats.gih_wr is not None) else 0.0
        
        # Determine strict colors (including mana cost)
        # We assume if it has W or B in mana cost/colors or is Colorless/Artifact, it's a candidate
        
        is_white = "W" in sc_card.colors
        is_black = "B" in sc_card.colors
        is_red = "R" in sc_card.colors
        is_green = "G" in sc_card.colors
        is_blue = "U" in sc_card.colors
        
        # Check hybrid/gold nature
        # A card like Feisty Spikeling (R/W) is White-playable.
        # A card like Wary Farmer (G/W) is White-playable.
        
        # Candidate Logic:
        # 1. Main Colors: White OR Black (excludes cards that require U, R, G unless hybrid/splash)
        # 2. Splash: Red high WR cards.
        
        colors = set(sc_card.colors)
        
        is_main_col = False
        # Pure Colorless/Artifacts are always candidates
        if not colors: 
            is_main_col = True
        # If it has W or B
        elif "W" in colors or "B" in colors:
            # Must NOT require another color strictly (unless we have fixing)
            # But hybrids allow mono-color payment.
            # Scryfall 'colors' field is the color identity essentially? No, it's the colors of the card.
            # Hybrids have both colors.
            # We need to assume standard W/B deck can play {W/G} or {W/R} hybrids using W mana.
            
            # Exclude if it STRICTLY requires U or G or R (and isn't splash or hybrid)
            # Actually, parsing mana_cost is complex.
            # Simpler heuristic:
            # If it has W or B, it's 'on color' for selection.
            # We will filter out "splash candidates" separately.
            is_main_col = True
            
        card_data = {
            "name": sc_card.name,
            "wr": wr,
            "colors": sc_card.colors,
            "mana_cost": sc_card.mana_cost,
            "type_line": sc_card.type_line,
            "is_land": "Land" in sc_card.type_line
        }
        all_cards.append(card_data)

    # 2. Select Candidates
    # Target: 23 Non-lands
    deck_cards = []
    
    # Priority 1: High WR Red Splash (>60% or specific bombs)
    splash_targets = ["End-Blaze Epiphany"] 
    # Logic: if name in splash_targets OR (is_red and wr > 0.60 and not (is_white or is_black))
    
    # Priority 2: On-color (W or B or Colorless) sorted by WR
    candidates = []
    
    for c in all_cards:
        name = c["name"]
        colors = set(c["colors"])
        
        # Lands
        if c["is_land"]:
            # Special lands? Overgrown Tomb is good. Firdoch Core?
            if name == "Overgrown Tomb": deck_cards.append(c)
            # Firdoch Core is a fixer, maybe include?
            if name == "Firdoch Core": deck_cards.append(c)
            # Gathering Stone is artifact but fixes?
            if name == "Gathering Stone" and "Artifact" in c["type_line"]: deck_cards.append(c)
            continue

        # Splash Red
        if "R" in colors and not ("W" in colors or "B" in colors):
            if name in splash_targets or c["wr"] > 0.60:
                print(f"Adding Splash: {name} ({c['wr']:.1%})")
                deck_cards.append(c)
            continue
            
        # Exclude purely off-color
        # If it has G or U and NOT (W or B), skip
        if ("G" in colors or "U" in colors) and not ("W" in colors or "B" in colors):
            continue
            
        # Main Color Candidates (W, B, Colorless, or Hybrids like W/G, W/R, B/G, B/U, etc.)
        # Note: B/G hybrid is playable in B deck.
        candidates.append(c)

    # Sort candidates by WR
    candidates.sort(key=lambda x: x["wr"], reverse=True)
    
    # Fill remaining spots up to ~23 spells
    slots_needed = 23 - len([x for x in deck_cards if not x.get("is_land", False)])
    
    # Take top N
    selected_candidates = candidates[:slots_needed]
    deck_cards.extend(selected_candidates)
    
    # 3. Print List
    print(f"\n=== GENERATED DECK ({len(deck_cards)} non-basic cards) ===")
    
    creatures = [c for c in deck_cards if "Creature" in c["type_line"]]
    spells = [c for c in deck_cards if "Creature" not in c["type_line"] and not c.get("is_land", False)]
    lands = [c for c in deck_cards if c.get("is_land", False)]
    
    print("\n-- Creatures --")
    for c in sorted(creatures, key=lambda x: x["mana_cost"]):
        print(f"1 {c['name']}")
        
    print("\n-- Spells --")
    for c in sorted(spells, key=lambda x: x["mana_cost"]):
        print(f"1 {c['name']}")

    print("\n-- Special Lands/Artifacts --")
    for c in sorted(lands, key=lambda x: x["name"]):
        print(f"1 {c['name']}")
        
    # Basic Land Math
    # Total cards so far (non-basics)
    count = len(deck_cards)
    needed_basics = 40 - count
    print(f"\n-- Basic Lands ({needed_basics} needed) --")
    print(f"{max(1, int(needed_basics * 0.4))} Plains")
    print(f"{max(1, int(needed_basics * 0.4))} Swamp")
    print(f"{max(1, int(needed_basics * 0.2))} Mountain (for splash)")

solve()

