"""Interactive review tool for divergence reports.

Allows user to review decisions where they disagreed with the advisor,
mark which was correct, and auto-update prompts based on patterns.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arenamcp.divergence_tracker import Divergence


def review_divergence(divergence: dict, index: int, total: int) -> dict:
    """Interactively review a single divergence.
    
    Args:
        divergence: Divergence data dict
        index: Current divergence number
        total: Total number of divergences
        
    Returns:
        Updated divergence with review data
    """
    print(f"\n{'='*60}")
    print(f"Divergence {index}/{total}")
    print(f"{'='*60}")
    print(f"Frame: {divergence['frame']}")
    print(f"Trigger: {divergence['trigger']}")
    
    if divergence.get('flagged_by_user'):
        print("⚠️  YOU FLAGGED THIS (F7)")
    
    print(f"\n📢 Advisor said:")
    print(f"   \"{divergence['advice_given']}\"")
    
    print(f"\n👤 You did:")
    print(f"   {divergence['action_taken']}")
    
    print(f"\n❓ Who was correct?")
    print("   [a] Advisor was correct (I should have followed the advice)")
    print("   [p] Player was correct (advisor should change)")
    print("   [b] Both valid (situational)")
    print("   [s] Skip (unclear/not sure)")
    print("   [q] Quit review")
    
    while True:
        choice = input("\nYour choice: ").strip().lower()
        
        if choice == 'q':
            return None  # Signal quit
        
        if choice == 's':
            divergence['reviewed'] = False
            return divergence
        
        if choice in ['a', 'p', 'b']:
            divergence['reviewed'] = True
            
            if choice == 'a':
                divergence['advisor_was_correct'] = True
                print("\n✓ Marked: Advisor was correct")
                notes = input("Why was advisor correct? (optional): ").strip()
                if notes:
                    divergence['review_notes'] = notes
                    
            elif choice == 'p':
                divergence['advisor_was_correct'] = False
                print("\n✓ Marked: Player was correct")
                notes = input("Why was advisor wrong? (REQUIRED): ").strip()
                while not notes:
                    notes = input("Please explain why (helps improve prompts): ").strip()
                divergence['review_notes'] = notes
                
                # Ask if prompt should be updated
                update = input("Update prompt based on this? [y/n]: ").strip().lower()
                divergence['should_update_prompt'] = (update == 'y')
                
            elif choice == 'b':
                divergence['advisor_was_correct'] = None  # Both valid
                print("\n✓ Marked: Both valid")
                notes = input("When is each option better? (optional): ").strip()
                if notes:
                    divergence['review_notes'] = notes
            
            return divergence
        
        print("Invalid choice. Please enter a, p, b, s, or q.")


def review_report(report_path: Path, auto_save: bool = True) -> dict:
    """Interactively review all divergences in a report.
    
    Args:
        report_path: Path to divergence report JSON
        auto_save: Whether to auto-save after each review
        
    Returns:
        Updated report dict
    """
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    divergences = report.get('divergences', [])
    
    if not divergences:
        print("No divergences to review.")
        return report
    
    print(f"\n{'='*60}")
    print(f"DIVERGENCE REVIEW: {report['match_id']}")
    print(f"{'='*60}")
    print(f"Total divergences: {len(divergences)}")
    print(f"User flagged: {report.get('user_flagged', 0)}")
    print(f"Auto detected: {report.get('auto_detected', 0)}")
    
    reviewed_count = 0
    advisor_correct = 0
    player_correct = 0
    prompt_updates_needed = 0
    
    for i, div in enumerate(divergences, 1):
        if div.get('reviewed'):
            # Skip already reviewed
            continue
        
        updated_div = review_divergence(div, i, len(divergences))
        
        if updated_div is None:
            # User quit
            print("\nExiting review...")
            break
        
        # Update in report
        divergences[i-1] = updated_div
        
        # Track stats
        if updated_div.get('reviewed'):
            reviewed_count += 1
            if updated_div.get('advisor_was_correct') == True:
                advisor_correct += 1
            elif updated_div.get('advisor_was_correct') == False:
                player_correct += 1
            if updated_div.get('should_update_prompt'):
                prompt_updates_needed += 1
        
        # Auto-save progress
        if auto_save:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
    
    # Final summary
    print(f"\n{'='*60}")
    print("REVIEW SUMMARY")
    print(f"{'='*60}")
    print(f"Reviewed: {reviewed_count}/{len(divergences)}")
    print(f"Advisor correct: {advisor_correct}")
    print(f"Player correct: {player_correct}")
    print(f"Prompt updates needed: {prompt_updates_needed}")
    
    # Save final version
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Saved reviewed report to {report_path}")
    
    # Generate prompt suggestions
    if prompt_updates_needed > 0:
        print(f"\n{'='*60}")
        print("PROMPT UPDATE SUGGESTIONS")
        print(f"{'='*60}")
        
        for div in divergences:
            if div.get('should_update_prompt'):
                print(f"\n{div['trigger']}:")
                print(f"  Issue: {div['advice_given'][:60]}...")
                print(f"  Fix: {div['review_notes']}")
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Review divergence reports and mark correct advice"
    )
    parser.add_argument("report", type=Path, help="Divergence report JSON file")
    parser.add_argument("--no-auto-save", action="store_true", 
                       help="Don't auto-save after each review")
    
    args = parser.parse_args()
    
    if not args.report.exists():
        print(f"Error: Report not found: {args.report}")
        return 1
    
    review_report(args.report, auto_save=not args.no_auto_save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
