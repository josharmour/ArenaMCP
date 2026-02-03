"""Real-time divergence detection between advisor suggestions and player actions.

Tracks when the player does something different than what the advisor recommended,
allowing for post-game review and prompt optimization.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Divergence:
    """A single instance where player action differed from advisor suggestion."""
    timestamp: datetime
    frame_number: int
    trigger: str
    game_state: dict
    
    # What happened
    advice_given: str
    action_taken: str
    
    # Flagging
    flagged_by_user: bool = False  # User hit F7 to mark for review
    auto_detected: bool = True
    
    # Analysis
    reviewed: bool = False
    advisor_was_correct: Optional[bool] = None
    review_notes: str = ""


class DivergenceTracker:
    """Tracks divergences between advisor advice and player actions during live play."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize tracker.
        
        Args:
            output_dir: Directory to save divergence reports (default: ./divergence_reports)
        """
        self.output_dir = output_dir or Path("divergence_reports")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self._current_match_id: Optional[str] = None
        self._divergences: list[Divergence] = []
        self._last_advice: Optional[dict] = None
        self._frame_number = 0
        
    def start_match(self, match_id: str):
        """Start tracking a new match.
        
        Args:
            match_id: Unique identifier for this match
        """
        self._current_match_id = match_id
        self._divergences = []
        self._last_advice = None
        self._frame_number = 0
        logger.info(f"Started divergence tracking for match {match_id}")
    
    def record_advice(self, trigger: str, game_state: dict, advice: str):
        """Record advice that was given.
        
        Args:
            trigger: What triggered the advice
            game_state: Current game state snapshot
            advice: The advice text that was given
        """
        self._last_advice = {
            "timestamp": datetime.now(),
            "frame_number": self._frame_number,
            "trigger": trigger,
            "game_state": game_state,
            "advice": advice,
        }
        self._frame_number += 1
        logger.debug(f"Recorded advice: {advice[:50]}...")
    
    def check_action(self, action_description: str) -> Optional[Divergence]:
        """Check if player action matches last advice.
        
        Args:
            action_description: What the player actually did
            
        Returns:
            Divergence object if action differs from advice, None otherwise
        """
        if not self._last_advice:
            return None
        
        advice = self._last_advice["advice"]
        
        # Simple text comparison (could be enhanced with semantic similarity)
        # Check if action keywords are in advice
        if self._actions_match(advice, action_description):
            logger.debug(f"Action matches advice: {action_description}")
            return None
        
        # Divergence detected!
        logger.info(f"DIVERGENCE: Advice was '{advice[:50]}...', action was '{action_description}'")
        
        divergence = Divergence(
            timestamp=self._last_advice["timestamp"],
            frame_number=self._last_advice["frame_number"],
            trigger=self._last_advice["trigger"],
            game_state=self._last_advice["game_state"],
            advice_given=advice,
            action_taken=action_description,
            auto_detected=True,
        )
        
        self._divergences.append(divergence)
        return divergence
    
    def flag_current_decision(self):
        """Mark the most recent decision for review (user pressed F7).
        
        Returns:
            True if flagged successfully
        """
        if not self._last_advice:
            logger.warning("No recent advice to flag")
            return False
        
        # Check if already tracked as divergence
        if self._divergences and self._divergences[-1].frame_number == self._last_advice["frame_number"]:
            # Already tracked, just mark as user-flagged
            self._divergences[-1].flagged_by_user = True
            logger.info(f"Flagged existing divergence at frame {self._last_advice['frame_number']}")
        else:
            # Create new divergence entry
            divergence = Divergence(
                timestamp=self._last_advice["timestamp"],
                frame_number=self._last_advice["frame_number"],
                trigger=self._last_advice["trigger"],
                game_state=self._last_advice["game_state"],
                advice_given=self._last_advice["advice"],
                action_taken="[User flagged - action unknown]",
                flagged_by_user=True,
                auto_detected=False,
            )
            self._divergences.append(divergence)
            logger.info(f"Created new flagged divergence at frame {self._last_advice['frame_number']}")
        
        return True
    
    def end_match(self, save_report: bool = True) -> dict:
        """End match tracking and generate report.
        
        Args:
            save_report: Whether to save report to disk
            
        Returns:
            Report dict with all divergences
        """
        if not self._current_match_id:
            logger.warning("No active match to end")
            return {}
        
        report = {
            "match_id": self._current_match_id,
            "timestamp": datetime.now().isoformat(),
            "total_divergences": len(self._divergences),
            "user_flagged": len([d for d in self._divergences if d.flagged_by_user]),
            "auto_detected": len([d for d in self._divergences if d.auto_detected and not d.flagged_by_user]),
            "divergences": [
                {
                    "frame": d.frame_number,
                    "trigger": d.trigger,
                    "advice_given": d.advice_given,
                    "action_taken": d.action_taken,
                    "flagged_by_user": d.flagged_by_user,
                    "timestamp": d.timestamp.isoformat(),
                }
                for d in self._divergences
            ],
        }
        
        if save_report:
            report_path = self.output_dir / f"{self._current_match_id}_divergences.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved divergence report to {report_path}")
        
        logger.info(f"Match ended: {len(self._divergences)} divergences ({report['user_flagged']} flagged)")
        
        # Reset for next match
        self._current_match_id = None
        self._divergences = []
        self._last_advice = None
        
        return report
    
    def _actions_match(self, advice: str, action: str) -> bool:
        """Check if action matches advice.
        
        This is a simple heuristic - could be enhanced with:
        - Semantic similarity (embeddings)
        - Entity extraction (card names)
        - Parse structured advice format
        
        Args:
            advice: Advisor's suggestion
            action: Player's actual action
            
        Returns:
            True if they match
        """
        advice_lower = advice.lower()
        action_lower = action.lower()
        
        # Extract key verbs/nouns
        advice_words = set(advice_lower.split())
        action_words = set(action_lower.split())
        
        # Check for overlap
        overlap = advice_words & action_words
        
        # If >50% of action words are in advice, consider it a match
        if len(action_words) > 0:
            match_ratio = len(overlap) / len(action_words)
            return match_ratio > 0.5
        
        return False


# Global instance for easy access
_tracker: Optional[DivergenceTracker] = None


def get_tracker() -> DivergenceTracker:
    """Get the global divergence tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = DivergenceTracker()
    return _tracker


def start_match_tracking(match_id: str):
    """Start tracking divergences for a new match."""
    get_tracker().start_match(match_id)


def record_advice_given(trigger: str, game_state: dict, advice: str):
    """Record that advisor gave specific advice."""
    get_tracker().record_advice(trigger, game_state, advice)


def check_player_action(action: str) -> Optional[Divergence]:
    """Check if player action differs from last advice."""
    return get_tracker().check_action(action)


def flag_current_decision():
    """Flag current decision for review (F7 hotkey)."""
    return get_tracker().flag_current_decision()


def end_match_tracking(save_report: bool = True) -> dict:
    """End match and generate divergence report."""
    return get_tracker().end_match(save_report)
