
import json
import re

TXT_PATH = r"z:\ArenaMCP\last_match_event.txt" # We will re-use this to dump a generic GRE message

# Logic:
# 1. Scan log for the last GreToClientEvent containing "gameObjects"
# 2. Parse it to find Hand zones
# 3. Check which seat has identifiable cards

def infer_seat_from_hand():
    # We'll read the big log again, scanning backwards for GreToClientEvent
    log_path = r"C:\Users\joshu\AppData\LocalLow\Wizards Of The Coast\MTGA\Player.log"
    
    print("Reading log backwards for GameObjects...")
    
    chunk_size = 100000 # 100KB
    with open(log_path, 'rb') as f:
        f.seek(0, 2)
        file_size = f.tell()
        current_pos = file_size
        
        buffer = b""
        found_json = False
        
        while current_pos > 0 and not found_json:
            to_read = min(chunk_size, current_pos)
            current_pos -= to_read
            f.seek(current_pos)
            chunk = f.read(to_read)
            buffer = chunk + buffer
            
            # Try to find "GreToClientEvent"
            lines = buffer.decode('utf-8', errors='ignore').splitlines()
            for line in reversed(lines):
                if "GreToClientEvent" in line and "gameObjects" in line:
                    # Found a candidate!
                    print("Found GRE message with GameObjects.")
                    # It's likely truncated in this line-based approach if it's pretty-printed
                    # But Unity logs often put the whole JSON on one line.
                    
                    try:
                        start = line.find('{')
                        if start != -1:
                            data = json.loads(line[start:])
                            analyze_gre(data)
                            return
                    except:
                        continue

def analyze_gre(data):
    gre = data.get("greToClientEvent", {})
    messages = gre.get("greToClientMessages", [])
    
    for msg in messages:
        if msg.get("type") == "GREMessageType_GameStateMessage":
             game_state = msg.get("gameStateMessage", {})
             objects = game_state.get("gameObjects", [])
             
             seat_cards = {} # seat_id -> count of identified cards
             
             for obj in objects:
                 # Check if it's a card in hand
                 # 31 = ZoneType_Hand? Need to verify Zone IDs. 
                 # Actually, usually zoneId is referencing a specific zone object.
                 # Let's look at "visibility" or distinct grpIds.
                 
                 grp_id = obj.get("grpId")
                 owner_seat = obj.get("ownerSeatId")
                 
                 if grp_id and owner_seat:
                     if owner_seat not in seat_cards:
                        seat_cards[owner_seat] = 0
                     seat_cards[owner_seat] += 1
                     
             print("\nCard ownership analysis (cards with visible GrpId):")
             for seat, count in seat_cards.items():
                 print(f"Seat {seat}: {count} visible items")
                 
             # Heuristic: The seat with the MOST visible items (especially in Hand/Library) is likely us.
             # Actually, battlefield cards are visible to both.
             # We need to filter for ZoneType_Hand.
             # But ZoneType is usually not directly on the object, it's on the Zone.
             
             # Simpler: ConnectResp often has systemSeatIds explicitly.
             
if __name__ == "__main__":
    infer_seat_from_hand()
