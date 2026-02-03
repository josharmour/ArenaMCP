
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class RulesEngine:
    """
    A deterministic rules engine to calculate legal game actions.
    Serves as a 'Grounding' layer for the AI.
    """

    @staticmethod
    def get_legal_actions(game_state: Dict[str, Any]) -> List[str]:
        actions = []
        
        turn = game_state.get("turn", {})
        phase = turn.get("phase", "")
        is_my_turn = (turn.get("active_player") == turn.get("priority_player")) # Simplified approximation
        # Better is_my_turn check: active_player is me.
        # We need to know 'local_seat_id'.
        
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
        
        # Parse Mana
        # Note: This is an estimation. 
        # For a true engine, we'd need complex mana fixing logic.
        # We will assume the 'mana_pool' passed in gamestate is accurate OR use a simple counter.
        # Currently the game_state parsing logic is:
        # YOUR MANA: X available.
        # We'll rely on the simple heuristics used in coach.py for now, or just logic checks.
        
        # For now, let's focus on PHASING and SICKNESS which were the bugs.
        
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
                
            # Mana Check (Simplified: relies on client heuristics [CAN CAST] tag presence in prompts? 
            # No, we don't have tags here. We can only do coarse filtering or pass-through).
            # Grounding Step 1: Just Timing.
            
            if can_cast_timing:
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
                 actions.append(f"Declare Attackers: {', '.join(potential_attackers)}")
        
        # 4. BLOCKING
        # Legal if: Combat Phase, Defending Player
        if not is_active_player and "Combat" in phase:
             untapped_blockers = [c.get("name") for c in my_creatures if not c.get("is_tapped")]
             if untapped_blockers:
                 actions.append(f"Block with: {', '.join(untapped_blockers)}")

        # 5. ABILITIES
        # Activated abilities on battlefield
        for c in my_creatures: # And lands/artifacts
            if ": " in c.get("oracle_text", ""): # Crude check for activated ability
                 if not c.get("is_tapped"): # Assuming tap cost? Dangerous assumption.
                     actions.append(f"Activate {c.get('name')}")

        return actions

