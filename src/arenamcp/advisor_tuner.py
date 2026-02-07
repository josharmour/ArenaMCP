"""Post-game advisor analysis and tuning tool.

Analyzes recorded matches to compare advisor performance:
1. What advice was actually given during the game
2. What you actually did (from replay)
3. What advice SHOULD be given (re-run with current prompts)
4. Generate optimization suggestions

Usage:
    python -m arenamcp.advisor_tuner analyze recording.json --advice-log advice.jsonl
    python -m arenamcp.advisor_tuner replay arena_recording.rply --with-advice
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from arenamcp.match_validator import MatchRecording, MatchFrame
from arenamcp.coach import CoachEngine, create_backend
from arenamcp.arena_replay import ArenaReplay
from arenamcp.replay_converter import ReplayConverter

logger = logging.getLogger(__name__)


@dataclass
class AdviceComparison:
    """Comparison of advice at a single decision point."""
    frame_number: int
    timestamp: datetime
    game_state: dict
    trigger: str
    
    # What happened
    advice_given: Optional[str] = None  # What advisor said during game
    action_taken: Optional[str] = None  # What player actually did
    
    # Re-evaluation
    optimal_advice: Optional[str] = None  # What advisor says NOW with current prompts
    
    # Analysis
    was_correct: Optional[bool] = None
    improvement_notes: list[str] = field(default_factory=list)
    prompt_suggestions: list[str] = field(default_factory=list)


class AdvisorTuner:
    """Analyzes advisor performance and suggests improvements."""
    
    def __init__(self, backend_type: str = "claude", model: Optional[str] = None):
        """Initialize tuner with LLM backend.
        
        Args:
            backend_type: LLM backend to use for re-evaluation
            model: Optional model override
        """
        self.backend = create_backend(backend_type, model=model)
        self.coach = CoachEngine(backend=self.backend)
        
    def analyze_recording(self, recording: MatchRecording, 
                         advice_log: Optional[Path] = None) -> list[AdviceComparison]:
        """Analyze a match recording.
        
        Args:
            recording: MatchRecording to analyze
            advice_log: Optional log of advice that was given during the game
            
        Returns:
            List of AdviceComparison objects for each decision point
        """
        logger.info(f"Analyzing {len(recording.frames)} frames...")
        
        # Load advice log if provided
        given_advice = {}
        if advice_log and advice_log.exists():
            given_advice = self._load_advice_log(advice_log)
        
        comparisons = []

        for frame in recording.frames:
            # Extract trigger from raw_message (may be stored as _trigger by converter)
            trigger = frame.raw_message.get("_trigger") if frame.raw_message else None

            if not trigger:
                continue  # Skip non-decision frames

            # Use parsed_snapshot for game state
            game_state = frame.parsed_snapshot or {}

            logger.info(f"Frame {frame.frame_number}: {trigger}")

            # Get what advice was given (if available)
            advice_given = given_advice.get(frame.frame_number)

            # Re-run advisor with current prompts to get optimal advice
            try:
                optimal_advice = self.coach.get_advice(
                    game_state,
                    trigger=trigger,
                    style="concise"
                )
            except Exception as e:
                logger.error(f"Failed to generate optimal advice: {e}")
                optimal_advice = None

            # Create comparison
            comparison = AdviceComparison(
                frame_number=frame.frame_number,
                timestamp=frame.timestamp,
                game_state=game_state,
                trigger=trigger,
                advice_given=advice_given,
                action_taken=self._extract_action_taken(frame),
                optimal_advice=optimal_advice,
            )
            
            # Analyze differences
            self._analyze_comparison(comparison)
            
            comparisons.append(comparison)
        
        logger.info(f"Analyzed {len(comparisons)} decision points")
        return comparisons
    
    def generate_report(self, comparisons: list[AdviceComparison], 
                       output_path: Optional[Path] = None) -> dict:
        """Generate analysis report with improvement suggestions.
        
        Args:
            comparisons: List of AdviceComparison objects
            output_path: Optional path to save report JSON
            
        Returns:
            Report dict with stats and suggestions
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_decisions": len(comparisons),
            "decisions_analyzed": len([c for c in comparisons if c.optimal_advice]),
            "summary": {
                "correct_advice": 0,
                "incorrect_advice": 0,
                "missing_advice": 0,
                "improved_advice": 0,
            },
            "decisions": [],
            "improvement_suggestions": [],
        }
        
        for comp in comparisons:
            # Categorize
            if comp.advice_given is None:
                report["summary"]["missing_advice"] += 1
            elif comp.was_correct:
                report["summary"]["correct_advice"] += 1
            else:
                report["summary"]["incorrect_advice"] += 1
            
            if comp.optimal_advice and comp.advice_given and comp.optimal_advice != comp.advice_given:
                report["summary"]["improved_advice"] += 1
            
            # Add to detailed list
            report["decisions"].append({
                "frame": comp.frame_number,
                "trigger": comp.trigger,
                "advice_given": comp.advice_given,
                "action_taken": comp.action_taken,
                "optimal_advice": comp.optimal_advice,
                "was_correct": comp.was_correct,
                "notes": comp.improvement_notes,
                "prompt_suggestions": comp.prompt_suggestions,
            })
            
            # Collect unique suggestions
            for suggestion in comp.prompt_suggestions:
                if suggestion not in report["improvement_suggestions"]:
                    report["improvement_suggestions"].append(suggestion)
        
        # Save if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved report to {output_path}")
        
        return report
    
    def print_report(self, report: dict):
        """Print human-readable report to console."""
        print("\n" + "="*60)
        print("ADVISOR TUNING REPORT")
        print("="*60)
        
        summary = report["summary"]
        total = report["total_decisions"]
        
        print(f"\nDecisions Analyzed: {total}")
        print(f"  ✓ Correct advice:   {summary['correct_advice']}")
        print(f"  ✗ Incorrect advice: {summary['incorrect_advice']}")
        print(f"  ø Missing advice:   {summary['missing_advice']}")
        print(f"  ↑ Improved:         {summary['improved_advice']}")
        
        if report["improvement_suggestions"]:
            print(f"\n{'='*60}")
            print("IMPROVEMENT SUGGESTIONS")
            print("="*60)
            for i, suggestion in enumerate(report["improvement_suggestions"], 1):
                print(f"\n{i}. {suggestion}")
        
        print(f"\n{'='*60}")
        print("DECISION DETAILS")
        print("="*60)
        
        for dec in report["decisions"]:
            if not dec["was_correct"] or dec["prompt_suggestions"]:
                print(f"\nFrame {dec['frame']}: {dec['trigger']}")
                if dec["advice_given"]:
                    print(f"  Given:   \"{dec['advice_given']}\"")
                if dec["action_taken"]:
                    print(f"  Taken:   {dec['action_taken']}")
                if dec["optimal_advice"]:
                    print(f"  Optimal: \"{dec['optimal_advice']}\"")
                if dec["notes"]:
                    for note in dec["notes"]:
                        print(f"    → {note}")
    
    def _load_advice_log(self, log_path: Path) -> dict[int, str]:
        """Load advice log (JSONL format: {frame_number, advice})."""
        advice_map = {}
        with open(log_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                advice_map[entry["frame_number"]] = entry["advice"]
        return advice_map
    
    def _extract_action_taken(self, frame: MatchFrame) -> Optional[str]:
        """Extract what action the player actually took from the frame.
        
        This would require parsing the RESPONSE message that followed
        the request. For now, returns None (future enhancement).
        """
        # TODO: Parse response messages to see what player chose
        return None
    
    def _analyze_comparison(self, comp: AdviceComparison):
        """Analyze a single advice comparison and populate suggestions.
        
        Args:
            comp: AdviceComparison to analyze (modified in-place)
        """
        # If no advice was given but should have been
        if comp.trigger and not comp.advice_given:
            comp.was_correct = False
            comp.improvement_notes.append("Missing advice - advisor didn't respond to trigger")
            comp.prompt_suggestions.append(
                f"Ensure {comp.trigger} trigger fires correctly and decision context is captured"
            )
            return
        
        # If we have both old and new advice, compare
        if comp.advice_given and comp.optimal_advice:
            # Simple text comparison (could be enhanced with semantic similarity)
            if comp.advice_given.strip().lower() == comp.optimal_advice.strip().lower():
                comp.was_correct = True
                comp.improvement_notes.append("Current prompts match original advice")
            else:
                comp.was_correct = False
                comp.improvement_notes.append("Current prompts produce different advice")
                
                # Heuristic checks for common issues
                if len(comp.optimal_advice) < len(comp.advice_given) * 0.5:
                    comp.prompt_suggestions.append(
                        "New advice is much shorter - check if context is being lost"
                    )
                elif "need" in comp.advice_given.lower() and "need" not in comp.optimal_advice.lower():
                    comp.prompt_suggestions.append(
                        "Mana/resource checking may have changed - verify castability tags"
                    )
                elif comp.trigger == "decision_required":
                    comp.prompt_suggestions.append(
                        "Decision-specific prompt may need tuning for this scenario"
                    )


def analyze_arena_replay(replay_path: Path, output_dir: Optional[Path] = None) -> dict:
    """Analyze an Arena replay file end-to-end.
    
    Args:
        replay_path: Path to Arena .rply file
        output_dir: Optional output directory for reports
        
    Returns:
        Analysis report dict
    """
    if output_dir is None:
        output_dir = replay_path.parent
    
    logger.info(f"Analyzing Arena replay: {replay_path.name}")
    
    # Convert Arena replay to our format
    converter = ReplayConverter()
    recording = converter.convert(replay_path)
    
    if not recording:
        logger.error("Failed to convert replay")
        return {}
    
    # Analyze with tuner
    tuner = AdvisorTuner()
    comparisons = tuner.analyze_recording(recording)
    
    # Generate report
    report_path = output_dir / f"{replay_path.stem}_analysis.json"
    report = tuner.generate_report(comparisons, report_path)
    
    # Print to console
    tuner.print_report(report)
    
    return report


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze advisor performance and suggest improvements")
    parser.add_argument("replay", type=Path, help="Arena .rply file or converted .json recording")
    parser.add_argument("--advice-log", type=Path, help="Optional advice log file (JSONL)")
    parser.add_argument("--output", type=Path, help="Output directory for reports")
    parser.add_argument("--backend", default="claude", help="LLM backend (claude/gemini/ollama)")
    parser.add_argument("--model", help="Model override")
    
    args = parser.parse_args()
    
    if args.replay.suffix == ".rply":
        # Arena replay file
        analyze_arena_replay(args.replay, args.output)
    else:
        # Our converted format
        recording = MatchRecording.load(args.replay)
        tuner = AdvisorTuner(backend_type=args.backend, model=args.model)
        comparisons = tuner.analyze_recording(recording, args.advice_log)
        report = tuner.generate_report(comparisons, args.output)
        tuner.print_report(report)
