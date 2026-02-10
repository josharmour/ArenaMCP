
import json
import os

LOG_PATH = r"C:\Users\joshu\AppData\LocalLow\Wizards Of The Coast\MTGA\Player.log"

def extract_json_context():
    found_lines = []
    with open(LOG_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if "systemSeatId" in line:
            # Capture this line, and maybe context if it's a huge block structure
            # But typically these logs are one-line-per-entry or structured JSON.
            # Let's just grab the line itself if it looks like JSON.
            found_lines.append(line.strip())

    print(f"Found {len(found_lines)} occurrences of systemSeatId. Showing last 1:")
    if found_lines:
        last = found_lines[-1]
        print(last)

if __name__ == "__main__":
    extract_json_context()
