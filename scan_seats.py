
import json
import os
import re

LOG_PATH = r"C:\Users\joshu\AppData\LocalLow\Wizards Of The Coast\MTGA\Player.log"

def scan_log():
    try:
        if not os.path.exists(LOG_PATH):
            print("Log file not found")
            return

        print(f"Scanning {LOG_PATH}...")
        
        # We only care about the last ~50MB if it's huge, but let's try to find the LAST occurrence of relevant keys.
        # Efficient ref: seek to end and read backwards? Or just stream forward. 
        # Given standard log rotation, streaming forward is fine unless it's GBs.
        
        relevant_keys = ["systemSeatId", "MatchGameRoomStateChangedEvent", "GreToClientEvent", "AuthenticateResponse"]
        
        last_system_seat = None
        last_match_start = None
        
        with open(LOG_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if "systemSeatId" in line:
                    last_system_seat = line.strip()
                if "MatchGameRoomStateChangedEvent" in line:
                    last_match_start = line.strip()
                    
        print("\n--- FINDINGS ---")
        if last_system_seat:
            print(f"LAST SYSTEM SEAT MARKER:\n{last_system_seat[:500]}...") # Truncate
        else:
            print("No 'systemSeatId' found.")
            
        if last_match_start:
            print(f"LAST MATCH START MARKER:\n{last_match_start[:500]}...")
            
    except Exception as e:
        print(f"Error scanning log: {e}")

if __name__ == "__main__":
    scan_log()
