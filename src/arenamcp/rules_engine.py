
import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class RulesEngine:
    """
    A deterministic rules engine to calculate legal game actions.
    Serves as a 'Grounding' layer for the AI.
    """

    @staticmethod
    def _count_available_mana(game_state: Dict[str, Any], local_seat: int) -> int:
        """Count total available mana from untapped lands and mana creatures."""
        battlefield = game_state.get("battlefield", [])
        total = 0
        turn_num = game_state.get("turn", {}).get("turn_number", 0)
        for card in battlefield:
            if card.get("owner_seat_id") != local_seat:
                continue
            if card.get("is_tapped"):
                continue
            type_line = card.get("type_line", "").lower()
            oracle = card.get("oracle_text", "")
            is_land = "land" in type_line
            is_creature = "creature" in type_line
            has_mana_ability = bool(re.search(r'\{T\}.*[Aa]dd\s+\{', oracle))
            entered = card.get("turn_entered_battlefield", -1)
            has_haste = "haste" in oracle.lower()
            is_sick = is_creature and (entered == turn_num) and not has_haste
            if is_land or (is_creature and has_mana_ability and not is_sick):
                total += 1
        return total

    @staticmethod
    def _parse_cmc(mana_cost: str) -> int:
        """Parse converted mana cost from a mana cost string like '{3}{R}{R}'."""
        if not mana_cost:
            return 0
        cmc = 0
        generic = re.findall(r'\{(\d+)\}', mana_cost)
        cmc += sum(int(g) for g in generic)
        for color in "WUBRGC":
            cmc += len(re.findall(rf"\{{{color}\}}", mana_cost))
        # Hybrid mana symbols like {U/R} count as 1 each
        hybrid = re.findall(r'\{[^}]+/[^}]+\}', mana_cost)
        cmc += len(hybrid)
        return cmc

    @staticmethod
    def _disambiguate_names(names: List[str]) -> List[str]:
        """Add #1, #2 suffixes to duplicate names in a list."""
        from collections import Counter
        counts = Counter(names)
        seen = {}
        result = []
        for name in names:
            if counts[name] > 1:
                seen[name] = seen.get(name, 0) + 1
                result.append(f"{name} #{seen[name]}")
            else:
                result.append(name)
        return result

    @staticmethod
    def get_legal_actions(game_state: Dict[str, Any]) -> List[str]:
        actions = []

        turn = game_state.get("turn", {})
        phase = turn.get("phase", "")

        players = game_state.get("players", [])
        local_player = next((p for p in players if p.get("is_local")), None)
        if not local_player:
            return ["Wait (Game State Syncing)"]

        local_seat = local_player.get("seat_id")
        is_active_player = (turn.get("active_player") == local_seat)
        has_priority = (turn.get("priority_player") == local_seat)

        if not has_priority:
             # Exception: We can declare blockers if it's the DeclareBlock step and we are defender
             step = turn.get("step", "")
             is_blocking_step = (step == "Step_DeclareBlock") and (not is_active_player)

             if not is_blocking_step:
                 return ["Wait (Opponent has priority)"]

        # Calculate available mana
        available_mana = RulesEngine._count_available_mana(game_state, local_seat)

        # 1. LAND DROPS
        # Legal if: Main Phase, Stack Empty, Lands Played < 1, Active Player
        stack = game_state.get("stack", [])
        is_stack_empty = len(stack) == 0
        is_main_phase = "Main" in phase
        
        if is_active_player and is_main_phase and is_stack_empty:
             if local_player.get("lands_played", 0) < 1:
                # Check hand for lands
                hand = game_state.get("hand", [])
                for card in hand:
                    if "Land" in card.get("type_line", ""):
                        actions.append(f"Play Land: {card.get('name')}")
                        # We only need to list one land action generally, or all specific ones?
                        # Let's list specific logic.

        # 2. CASTING
        # Sorcery Speed: Main Phase, Stack Empty, Active Player
        # Instant Speed: Anytime we have priority
        hand = game_state.get("hand", [])
        for card in hand:
            type_line = card.get("type_line", "")
            name = card.get("name", "")

            # Skip lands handled above
            if "Land" in type_line:
                continue

            is_instant_speed = "Instant" in type_line or "Flash" in card.get("oracle_text", "")

            can_cast_timing = False
            if is_instant_speed:
                can_cast_timing = True
            elif is_active_player and is_main_phase and is_stack_empty:
                can_cast_timing = True

            # Mana check: ensure player can afford the spell
            cmc = RulesEngine._parse_cmc(card.get("mana_cost", ""))
            can_afford = available_mana >= cmc

            if can_cast_timing and can_afford:
                actions.append(f"Cast {name}")

        # 3. ATTACKING
        # Legal if: Combat Phase (specifically Declare Attackers step?), Active Player, Creatures Untapped + !Sick
        # In Arena, we usually get priority *before* attackers are declared (Beginning of Combat) 
        # or *during* declare attackers (if we hold priority, but usually it's a game step).
        # Actually, asking "Who should attack" happens at 'Phase_Combat_Beginning' or 'Phase_Main1' (planning).
        
        battlefield = game_state.get("battlefield", [])
        my_creatures = [c for c in battlefield if c.get("owner_seat_id") == local_seat and "Creature" in c.get("type_line", "")]
        
        if is_active_player and ("Main" in phase or "Combat" in phase):
             potential_attackers = []
             turn_num = turn.get("turn_number", 0)
             
             # Safe turn parse
             try:
                 current_turn_int = int(str(turn_num).replace("?", "0"))
             except:
                 current_turn_int = 0

             for c in my_creatures:
                 # Check Sickness
                 entered = c.get("turn_entered_battlefield", -1)
                 has_haste = "haste" in c.get("oracle_text", "").lower()
                 is_tapped = c.get("is_tapped", False)

                 is_sick = (entered == current_turn_int) and not has_haste

                 if not is_sick and not is_tapped:
                     potential_attackers.append(c.get("name"))

             if potential_attackers:
                 actions.append(f"Declare Attackers: {', '.join(RulesEngine._disambiguate_names(potential_attackers))}")
        
        # 4. BLOCKING
        # Legal if: Combat Phase, Defending Player
        if not is_active_player and "Combat" in phase:
             untapped_blockers = [c.get("name") for c in my_creatures if not c.get("is_tapped")]
             if untapped_blockers:
                 actions.append(f"Block with: {', '.join(RulesEngine._disambiguate_names(untapped_blockers))}")

        # 5. ABILITIES
        # Activated abilities on battlefield
        ability_names = []
        for c in my_creatures: # And lands/artifacts
            if ": " in c.get("oracle_text", ""): # Crude check for activated ability
                 if not c.get("is_tapped"): # Assuming tap cost? Dangerous assumption.
                     ability_names.append(c.get("name"))
        for aname in RulesEngine._disambiguate_names(ability_names):
            actions.append(f"Activate {aname}")

        return actions

