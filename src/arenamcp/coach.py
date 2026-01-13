"""Coach engine with pluggable LLM backends for MTG game coaching.

This module provides the CoachEngine for getting strategic advice from LLMs,
with support for Claude (Anthropic), Gemini (Google), and local models via Ollama.
"""

import logging
import os
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


# LLM Backend Protocol and Implementations

class LLMBackend(Protocol):
    """Protocol for LLM backends that can provide coaching advice."""

    def complete(self, system_prompt: str, user_message: str) -> str:
        """Get a completion from the LLM.

        Args:
            system_prompt: The system prompt setting up the assistant role
            user_message: The user's message/question

        Returns:
            The LLM's response text, or error message string on failure
        """
        ...


class ClaudeBackend:
    """LLM backend using Anthropic's Claude API."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        """Initialize Claude backend with lazy client creation.

        Args:
            model: The Claude model to use (default: claude-sonnet-4-20250514)
        """
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")
        return self._client

    def complete(self, system_prompt: str, user_message: str) -> str:
        """Get completion from Claude API."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return "Error: ANTHROPIC_API_KEY environment variable not set"

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=500,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return f"Error getting advice from Claude: {e}"


class GeminiBackend:
    """LLM backend using Google's Gemini API."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        """Initialize Gemini backend with lazy client creation.

        Args:
            model: The Gemini model to use (default: gemini-2.0-flash)
        """
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy initialization of Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                api_key = os.environ.get("GOOGLE_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                self._client = genai
            except ImportError:
                raise ImportError("google-generativeai package required: pip install google-generativeai")
        return self._client

    def complete(self, system_prompt: str, user_message: str) -> str:
        """Get completion from Gemini API."""
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return "Error: GOOGLE_API_KEY environment variable not set"

        try:
            genai = self._get_client()
            model = genai.GenerativeModel(
                self.model,
                system_instruction=system_prompt
            )
            response = model.generate_content(user_message)
            return response.text
        except ImportError as e:
            return f"Error: {e}"
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"Error getting advice from Gemini: {e}"


class OllamaBackend:
    """LLM backend using local Ollama server."""

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        """Initialize Ollama backend.

        Args:
            model: The Ollama model to use (default: llama3.2)
            base_url: Ollama server URL (default: localhost:11434)
        """
        self.model = model
        self.base_url = base_url

    def complete(self, system_prompt: str, user_message: str) -> str:
        """Get completion from local Ollama server."""
        import requests

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": user_message,
                    "system": system_prompt,
                    "stream": False,
                },
                timeout=60,
            )

            if response.status_code == 404:
                return f"Error: Model '{self.model}' not found. Run: ollama pull {self.model}"

            response.raise_for_status()
            return response.json().get("response", "No response from Ollama")

        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Is it running? Start with: ollama serve"
        except requests.exceptions.Timeout:
            return "Error: Ollama request timed out"
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return f"Error getting advice from Ollama: {e}"


def create_backend(backend_type: str, model: Optional[str] = None) -> LLMBackend:
    """Factory function to create LLM backends by name.

    Args:
        backend_type: One of "claude", "gemini", "ollama"
        model: Optional model override (uses backend default if not specified)

    Returns:
        Configured LLMBackend instance

    Raises:
        ValueError: If backend_type is not recognized
    """
    backend_type = backend_type.lower()

    if backend_type == "claude":
        return ClaudeBackend(model=model) if model else ClaudeBackend()
    elif backend_type == "gemini":
        return GeminiBackend(model=model) if model else GeminiBackend()
    elif backend_type == "ollama":
        return OllamaBackend(model=model) if model else OllamaBackend()
    else:
        raise ValueError(f"Unknown backend type: {backend_type}. Use 'claude', 'gemini', or 'ollama'")


# Default MTG coach system prompt
DEFAULT_SYSTEM_PROMPT = """You are an expert MTG coach providing real-time advice during Arena games.

Keep responses concise (2-3 sentences) since they'll be spoken aloud.
Focus on the most important strategic consideration.
If asked a question, answer it directly.
If triggered proactively, offer the key insight for the current situation.

Consider: board state, life totals, cards in hand, phase/priority, and potential plays."""


class CoachEngine:
    """Engine for getting MTG coaching advice from an LLM backend."""

    def __init__(
        self,
        backend: Optional[LLMBackend] = None,
        system_prompt: Optional[str] = None
    ):
        """Initialize the coach engine.

        Args:
            backend: LLM backend to use (default: ClaudeBackend)
            system_prompt: Custom system prompt (default: MTG coach persona)
        """
        self._backend = backend if backend is not None else ClaudeBackend()
        self._system_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT

    def _format_game_context(self, game_state: dict[str, Any]) -> str:
        """Format game state dict into readable context for LLM.

        Args:
            game_state: Dict from get_game_state() MCP tool

        Returns:
            Formatted string summarizing the game state
        """
        lines = ["Current Game State:"]

        # Life totals prominently
        players = game_state.get("players", [])
        for p in players:
            role = "You" if p.get("is_local") else "Opponent"
            lines.append(f"  {role}: {p.get('life_total', '?')} life")

        # Turn info
        turn = game_state.get("turn", {})
        turn_num = turn.get("turn_number", "?")
        phase = turn.get("phase", "").replace("Phase_", "")
        step = turn.get("step", "").replace("Step_", "")
        priority = turn.get("priority_player", 0)

        # Determine who has priority
        local_id = None
        for p in players:
            if p.get("is_local"):
                local_id = p.get("seat_id")
                break

        priority_holder = "You" if priority == local_id else "Opponent"

        lines.append(f"  Turn {turn_num}, {phase}" + (f" ({step})" if step else ""))
        lines.append(f"  Priority: {priority_holder}")

        # Battlefield
        battlefield = game_state.get("battlefield", [])
        if battlefield:
            lines.append("Battlefield:")
            for card in battlefield:
                owner = "Your" if card.get("owner_seat_id") == local_id else "Opp's"
                name = card.get("name", "Unknown")
                pt = ""
                if card.get("power") is not None:
                    pt = f" ({card['power']}/{card['toughness']})"
                tapped = " (tapped)" if card.get("is_tapped") else ""
                lines.append(f"  {owner} {name}{pt}{tapped}")
        else:
            lines.append("Battlefield: Empty")

        # Hand
        hand = game_state.get("hand", [])
        if hand:
            lines.append("Your Hand:")
            for card in hand:
                name = card.get("name", "Unknown")
                cost = card.get("mana_cost", "")
                lines.append(f"  {name} {cost}")
        else:
            lines.append("Your Hand: Empty")

        # Stack (if anything on it)
        stack = game_state.get("stack", [])
        if stack:
            lines.append("Stack:")
            for card in stack:
                name = card.get("name", "Unknown")
                lines.append(f"  {name}")

        # Graveyards (brief)
        graveyard = game_state.get("graveyard", [])
        if graveyard:
            gy_count = len(graveyard)
            lines.append(f"Graveyards: {gy_count} card(s)")

        return "\n".join(lines)

    def get_advice(
        self,
        game_state: dict[str, Any],
        question: Optional[str] = None,
        trigger: Optional[str] = None
    ) -> str:
        """Get coaching advice for the current game state.

        Args:
            game_state: Dict from get_game_state() MCP tool
            question: Optional user question to answer
            trigger: Optional trigger name (e.g., "combat_attackers", "low_life")

        Returns:
            Advice string from the LLM
        """
        # Build context
        context = self._format_game_context(game_state)

        # Build user message
        if question:
            user_message = f"{context}\n\nThe player asks: {question}"
        elif trigger:
            trigger_descriptions = {
                "new_turn": "A new turn has started.",
                "priority_gained": "You just gained priority.",
                "combat_attackers": "You're in combat, declaring attackers.",
                "combat_blockers": "Opponent attacked, you need to declare blockers.",
                "low_life": "Your life total is getting dangerously low!",
                "opponent_low_life": "Opponent's life is low - this could be your chance to win!",
                "stack_spell": "Something was just cast - consider if you want to respond.",
            }
            trigger_desc = trigger_descriptions.get(trigger, f"Trigger: {trigger}")
            user_message = f"{context}\n\n{trigger_desc} What should the player consider?"
        else:
            user_message = f"{context}\n\nWhat's the most important strategic consideration right now?"

        return self._backend.complete(self._system_prompt, user_message)


class GameStateTrigger:
    """Detects trigger conditions by comparing game states."""

    def __init__(self, life_threshold: int = 5):
        """Initialize trigger detector.

        Args:
            life_threshold: Life total below which "low_life" triggers (default: 5)
        """
        self.life_threshold = life_threshold

    def _get_local_player(self, state: dict[str, Any]) -> Optional[dict]:
        """Get the local player dict from game state."""
        for p in state.get("players", []):
            if p.get("is_local"):
                return p
        return None

    def _get_opponent_player(self, state: dict[str, Any]) -> Optional[dict]:
        """Get the opponent player dict from game state."""
        for p in state.get("players", []):
            if not p.get("is_local"):
                return p
        return None

    def check_triggers(
        self,
        prev_state: dict[str, Any],
        curr_state: dict[str, Any]
    ) -> list[str]:
        """Compare two game states and return triggered condition names.

        Args:
            prev_state: Previous game state dict
            curr_state: Current game state dict

        Returns:
            List of trigger names that fired (may be empty)
        """
        triggers = []

        prev_turn = prev_state.get("turn", {})
        curr_turn = curr_state.get("turn", {})

        # New turn detection
        prev_turn_num = prev_turn.get("turn_number", 0)
        curr_turn_num = curr_turn.get("turn_number", 0)
        if curr_turn_num > prev_turn_num:
            triggers.append("new_turn")

        # Get local player info
        prev_local = self._get_local_player(prev_state)
        curr_local = self._get_local_player(curr_state)
        local_seat = curr_local.get("seat_id") if curr_local else None

        # Priority gained
        prev_priority = prev_turn.get("priority_player", 0)
        curr_priority = curr_turn.get("priority_player", 0)
        if local_seat and curr_priority == local_seat and prev_priority != local_seat:
            triggers.append("priority_gained")

        # Combat phase detection
        curr_phase = curr_turn.get("phase", "")
        curr_step = curr_turn.get("step", "")
        curr_active = curr_turn.get("active_player", 0)

        if "Combat" in curr_phase:
            if "DeclareAttackers" in curr_step and curr_active == local_seat:
                triggers.append("combat_attackers")
            elif "DeclareBlockers" in curr_step and curr_active != local_seat:
                triggers.append("combat_blockers")

        # Low life detection
        if curr_local:
            curr_life = curr_local.get("life_total", 20)
            prev_life = prev_local.get("life_total", 20) if prev_local else 20
            if curr_life < self.life_threshold and prev_life >= self.life_threshold:
                triggers.append("low_life")

        # Opponent low life detection
        prev_opp = self._get_opponent_player(prev_state)
        curr_opp = self._get_opponent_player(curr_state)
        if curr_opp:
            curr_opp_life = curr_opp.get("life_total", 20)
            prev_opp_life = prev_opp.get("life_total", 20) if prev_opp else 20
            if curr_opp_life < self.life_threshold and prev_opp_life >= self.life_threshold:
                triggers.append("opponent_low_life")

        # Stack spell detection
        prev_stack = prev_state.get("stack", [])
        curr_stack = curr_state.get("stack", [])
        if len(curr_stack) > len(prev_stack):
            triggers.append("stack_spell")

        return triggers
