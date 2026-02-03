"""Advisor Tuning Workflow Tool

End-to-end workflow for tuning the advisor using Arena replay files:

1. Enable Arena replay recording (create .autoplay file)
2. Play matches normally
3. Run this tool to analyze and get improvement suggestions

Usage:
    # Analyze single replay
    python tune_advisor.py path/to/Replay0.rply
    
    # Analyze all replays in Arena's replay folder
    python tune_advisor.py --all
    
    # Analyze latest N replays
    python tune_advisor.py --latest 5
"""

import sys
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arenamcp.arena_replay import find_replay_files, repair_replay_file
from arenamcp.advisor_tuner import analyze_arena_replay


def main():
    parser = argparse.ArgumentParser(
        description="Tune advisor using Arena replay files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze specific replay
  python tune_advisor.py Replay0.rply
  
  # Analyze latest 3 replays
  python tune_advisor.py --latest 3
  
  # Analyze all replays in Arena's folder
  python tune_advisor.py --all
  
  # Repair corrupted replay first
  python tune_advisor.py Replay0.rply --repair

First time setup:
  1. Create empty file at:
     %APPDATA%\\..\\LocalLow\\Wizards Of The Coast\\MTGA\\ArenaAutoplayConfigs\\.autoplay
  2. Launch Arena, hold Alt during matches to see debug panel
  3. Enable "Record" before match starts
  4. After match, run this tool on the .rply file
        """
    )
    
    parser.add_argument("replay", nargs="?", type=Path, help="Replay file to analyze")
    parser.add_argument("--all", action="store_true", help="Analyze all replays in Arena folder")
    parser.add_argument("--latest", type=int, metavar="N", help="Analyze latest N replays")
    parser.add_argument("--repair", action="store_true", help="Repair corrupted replay file first")
    parser.add_argument("--output", type=Path, help="Output directory for reports")
    parser.add_argument("--backend", default="claude", help="LLM backend (claude/gemini/ollama)")
    parser.add_argument("--model", help="Model override")
    
    args = parser.parse_args()
    
    # Determine which replays to process
    replay_files = []
    
    if args.all:
        print("Finding all Arena replay files...")
        replay_files = find_replay_files()
        if not replay_files:
            print("No replay files found. Make sure Arena replay recording is enabled.")
            print("See --help for setup instructions.")
            return 1
        print(f"Found {len(replay_files)} replay files")
        
    elif args.latest:
        print(f"Finding latest {args.latest} replay files...")
        all_replays = find_replay_files()
        replay_files = all_replays[:args.latest]
        if not replay_files:
            print("No replay files found.")
            return 1
        print(f"Will analyze {len(replay_files)} replays")
        
    elif args.replay:
        replay_files = [args.replay]
        
    else:
        parser.print_help()
        return 1
    
    # Process each replay
    for i, replay_path in enumerate(replay_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing replay {i}/{len(replay_files)}: {replay_path.name}")
        print(f"{'='*60}")
        
        # Repair if needed
        if args.repair:
            print("Repairing replay file...")
            replay_path = repair_replay_file(replay_path)
        
        # Analyze
        try:
            report = analyze_arena_replay(
                replay_path, 
                output_dir=args.output
            )
            
            if not report:
                print(f"✗ Failed to analyze {replay_path.name}")
                continue
                
            # Summary
            summary = report.get("summary", {})
            print(f"\n✓ Analysis complete:")
            print(f"  Total decisions: {report.get('total_decisions', 0)}")
            print(f"  Correct: {summary.get('correct_advice', 0)}")
            print(f"  Incorrect: {summary.get('incorrect_advice', 0)}")
            print(f"  Missing: {summary.get('missing_advice', 0)}")
            
            if report.get("improvement_suggestions"):
                print(f"\n  {len(report['improvement_suggestions'])} improvement suggestions generated")
                
        except Exception as e:
            print(f"✗ Error analyzing {replay_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Processed {len(replay_files)} replays")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
