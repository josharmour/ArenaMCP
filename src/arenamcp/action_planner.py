"""Structured LLM Action Planning for Autopilot Mode.

Converts game state + trigger into structured JSON action commands
via a separate LLM call with a constrained schema prompt.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions the autopilot can execute in MTGA."""
    PLAY_LAND = "play_land"
    CAST_SPELL = "cast_spell"
    DECLARE_ATTACKERS = "declare_attackers"
    DECLARE_BLOCKERS = "declare_blockers"
    SELECT_TARGET = "select_target"
    SELECT_N = "select_n"
    MODAL_CHOICE = "modal_choice"
    MULLIGAN_KEEP = "mulligan_keep"
    MULLIGAN_MULL = "mulligan_mull"
    PASS_PRIORITY = "pass_priority"
    RESOLVE = "resolve"
    DRAFT_PICK = "draft_pick"
    CLICK_BUTTON = "click_button"
    ACTIVATE_ABILITY = "activate_ability"
    ORDER_BLOCKERS = "order_blockers"


@dataclass
class GameAction:
    """A single structured action to execute in MTGA."""
    action_type: ActionType
    card_name: str = ""
    target_names: list[str] = field(default_factory=list)
    attacker_names: list[str] = field(default_factory=list)
    blocker_assignments: dict[str, str] = field(default_factory=dict)
    modal_index: int = 0
    select_card_names: list[str] = field(default_factory=list)
    scry_position: str = ""  # "top" or "bottom"
    reasoning: str = ""
    confidence: float = 1.0

    def __str__(self) -> str:
        parts = [self.action_type.value]
        if self.card_name:
            parts.append(self.card_name)
        if self.target_names:
            parts.append(f"-> {', '.join(self.target_names)}")
        if self.attacker_names:
            parts.append(f"attackers: {', '.join(self.attacker_names)}")
        if self.blocker_assignments:
            assigns = [f"{b}->{a}" for b, a in self.blocker_assignments.items()]
            parts.append(f"blocks: {', '.join(assigns)}")
        if self.scry_position:
            parts.append(f"scry {self.scry_position}")
        return " | ".join(parts)


@dataclass
class ActionPlan:
    """A complete plan of actions to execute."""
    actions: list[GameAction] = field(default_factory=list)
    overall_strategy: str = ""
    trigger: str = ""
    turn_number: int = 0

    def __str__(self) -> str:
        lines = [f"Plan ({self.trigger}, turn {self.turn_number}): {self.overall_strategy}"]
        for i, action in enumerate(self.actions, 1):
            lines.append(f"  {i}. {action}")
        return "\n".join(lines)


# JSON schema embedded in the system prompt for constrained output
ACTION_SCHEMA = """{
  "actions": [{
    "action_type": "play_land|cast_spell|declare_attackers|declare_blockers|select_target|select_n|modal_choice|mulligan_keep|mulligan_mull|pass_priority|resolve|draft_pick|click_button|activate_ability|order_blockers",
    "card_name": "string (card name, empty if not applicable)",
    "target_names": ["string (target card/player names)"],
    "attacker_names": ["string (creature names to attack with)"],
    "blocker_assignments": {"blocker_name": "attacker_name"},
    "modal_index": 0,
    "select_card_names": ["string (cards to select for scry/discard/etc)"],
    "scry_position": "top|bottom (only for scry decisions)",
    "reasoning": "string (brief explanation)"
  }],
  "overall_strategy": "string (1-sentence strategy summary)"
}"""


AUTOPILOT_SYSTEM_PROMPT = """You are an expert MTG Arena autopilot. Given the current game state and trigger,
output a JSON action plan that the autopilot will execute by clicking in the MTGA client.

CRITICAL RULES:
- ONLY suggest actions that appear in the "Legal:" line. Never hallucinate actions.
- Creatures tagged [SS] have SUMMONING SICKNESS — they CANNOT attack or use tap abilities.
- Output ONLY valid JSON matching the schema below. No markdown, no commentary outside JSON.
- Each action in the array will be executed sequentially.
- For "pass_priority" or "resolve", card_name can be empty.
- For "declare_attackers", list creature names in attacker_names.
- For "declare_blockers", map each blocker to the attacker it should block.
- For "select_n" (scry, discard, etc.), list cards in select_card_names.
- For "modal_choice", set modal_index (0-based).
- Be decisive. Pick the best line of play.

JSON SCHEMA:
""" + ACTION_SCHEMA


class ActionPlanner:
    """Converts game state + trigger into structured JSON action commands via LLM."""

    def __init__(self, backend: Any, timeout: float = 15.0):
        """Initialize the action planner.

        Args:
            backend: An LLMBackend instance (same interface as CoachEngine uses).
            timeout: Maximum seconds to wait for LLM response.
        """
        self._backend = backend
        self._timeout = timeout

    def plan_actions(
        self,
        game_state: dict[str, Any],
        trigger: str,
        legal_actions: Optional[list[str]] = None,
        decision_context: Optional[dict[str, Any]] = None,
    ) -> ActionPlan:
        """Plan actions for the current game state.

        Args:
            game_state: Full game state dict from get_game_state().
            trigger: The trigger that caused this planning (e.g. "new_turn").
            legal_actions: Optional pre-computed legal actions list.
            decision_context: Optional decision context from game state.

        Returns:
            ActionPlan with structured actions to execute.
        """
        start = time.perf_counter()

        # Build the prompt
        system_prompt = AUTOPILOT_SYSTEM_PROMPT
        user_message = self._build_action_prompt(
            game_state, trigger, legal_actions, decision_context
        )

        # Call LLM
        try:
            response = self._backend.complete(system_prompt, user_message)
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"Action planning took {elapsed:.0f}ms")
        except Exception as e:
            logger.error(f"Action planning LLM call failed: {e}")
            return ActionPlan(trigger=trigger)

        # Parse response
        plan = self._parse_response(response)
        plan.trigger = trigger
        plan.turn_number = game_state.get("turn", {}).get("turn_number", 0)

        logger.info(f"Planned {len(plan.actions)} actions: {plan.overall_strategy}")
        return plan

    def _build_action_prompt(
        self,
        game_state: dict[str, Any],
        trigger: str,
        legal_actions: Optional[list[str]] = None,
        decision_context: Optional[dict[str, Any]] = None,
    ) -> str:
        """Build the user message with formatted game context.

        Reuses the compact format from CoachEngine._format_game_context().
        """
        # Import and use CoachEngine's formatter for consistency
        try:
            from arenamcp.coach import CoachEngine
            # Create a temporary instance just for formatting
            formatter = CoachEngine.__new__(CoachEngine)
            context = formatter._format_game_context(game_state)
        except Exception as e:
            logger.warning(f"Failed to use CoachEngine formatter: {e}")
            context = self._fallback_format(game_state)

        # Build trigger description
        trigger_descriptions = {
            "new_turn": "Your turn started (Main Phase 1). Plan your plays.",
            "opponent_turn": "Opponent's turn. Plan responses if you have instants.",
            "combat_attackers": "Declare attackers phase. Choose which creatures attack.",
            "combat_blockers": "Opponent is attacking. Assign blockers.",
            "priority_gained": "You have priority. Respond or pass.",
            "spell_resolved": "A spell resolved. What's next?",
            "decision_required": "A game decision is pending. Make your choice.",
            "mulligan": "Mulligan decision. Keep or mulligan?",
            "land_played": "Land played. What's the next play?",
        }
        trigger_desc = trigger_descriptions.get(trigger, f"Trigger: {trigger}")

        parts = [
            f"TRIGGER: {trigger_desc}",
            "",
            context,
        ]

        if legal_actions:
            parts.append(f"\nLegal: {', '.join(legal_actions)}")

        if decision_context:
            parts.append(f"\nDecision: {json.dumps(decision_context, indent=2)}")

        parts.append("\nRespond with ONLY a JSON action plan matching the schema.")

        return "\n".join(parts)

    def _fallback_format(self, game_state: dict[str, Any]) -> str:
        """Fallback game state formatter if CoachEngine is unavailable."""
        parts = []

        # Turn info
        turn = game_state.get("turn", {})
        parts.append(
            f"Turn {turn.get('turn_number', '?')} | "
            f"Phase: {turn.get('phase', '?')} | "
            f"Step: {turn.get('step', '')} | "
            f"Active: Seat {turn.get('active_player', '?')}"
        )

        # Players
        for p in game_state.get("players", []):
            marker = "(YOU)" if p.get("is_local") else "(OPP)"
            parts.append(
                f"Seat {p.get('seat_id', '?')} {marker}: "
                f"Life={p.get('life', '?')}"
            )

        # Hand
        hand = game_state.get("hand", [])
        if hand:
            card_names = [c.get("name", "?") for c in hand]
            parts.append(f"Hand: {', '.join(card_names)}")

        # Battlefield
        battlefield = game_state.get("battlefield", [])
        if battlefield:
            bf_names = [c.get("name", "?") for c in battlefield]
            parts.append(f"Battlefield: {', '.join(bf_names)}")

        return "\n".join(parts)

    def _parse_response(self, response: str) -> ActionPlan:
        """Parse LLM response into an ActionPlan.

        Handles markdown fences, trailing commas, missing fields, and
        other common LLM output quirks.
        """
        plan = ActionPlan()

        # Extract JSON from markdown fences if present
        json_str = response.strip()
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", json_str, re.DOTALL)
        if fence_match:
            json_str = fence_match.group(1).strip()

        # Remove trailing commas before } or ]
        json_str = re.sub(r",\s*([\]}])", r"\1", json_str)

        # Try to parse
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse action plan JSON: {e}")
            logger.debug(f"Raw response: {response[:500]}")
            return plan

        # Extract overall strategy
        plan.overall_strategy = data.get("overall_strategy", "")

        # Parse actions
        for action_data in data.get("actions", []):
            action = self._parse_action(action_data)
            if action:
                plan.actions.append(action)

        return plan

    def _parse_action(self, data: dict[str, Any]) -> Optional[GameAction]:
        """Parse a single action dict into a GameAction."""
        try:
            action_type_str = data.get("action_type", "")
            try:
                action_type = ActionType(action_type_str)
            except ValueError:
                logger.warning(f"Unknown action type: {action_type_str}")
                return None

            return GameAction(
                action_type=action_type,
                card_name=data.get("card_name", ""),
                target_names=data.get("target_names", []),
                attacker_names=data.get("attacker_names", []),
                blocker_assignments=data.get("blocker_assignments", {}),
                modal_index=data.get("modal_index", 0),
                select_card_names=data.get("select_card_names", []),
                scry_position=data.get("scry_position", ""),
                reasoning=data.get("reasoning", ""),
                confidence=data.get("confidence", 1.0),
            )
        except Exception as e:
            logger.error(f"Failed to parse action: {e}, data={data}")
            return None
