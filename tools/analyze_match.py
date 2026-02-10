"""Quick script to analyze a match recording."""
import json
import sys

filepath = sys.argv[1] if len(sys.argv) > 1 else r'C:\Users\joshu\.arenamcp\match_validations\match_manual_20260202_192338_20260202_192338.json'

with open(filepath) as f:
    d = json.load(f)

print(f"Match: {d['match_id']}")
print(f"Frames: {len(d['frames'])}")
print()

for fr in d['frames']:
    snap = fr.get('parsed_snapshot', {}) or {}
    turn_info = snap.get('turn_info', {})
    
    print(f"Frame {fr['frame_number']:3d}: Arena[T{fr['arena_turn']:2d} {fr['arena_phase']:20s} active={fr['arena_active_player']}] "
          f"| Parsed[T{turn_info.get('turn_number', 0):2d} {turn_info.get('phase', ''):20s} active={turn_info.get('active_player', 0)}]")
