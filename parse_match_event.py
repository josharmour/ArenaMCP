
import json
import os

LOG_PATH = r"C:\Users\joshu\AppData\LocalLow\Wizards Of The Coast\MTGA\Player.log"
OUT_PATH = r"z:\ArenaMCP\match_event_structure.json"

def capture_match_event():
    last_match_event = ""
    with open(LOG_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if "MatchGameRoomStateChangedEvent" in line:
                last_match_event = line.strip()
    
    if last_match_event:
        # The line typically starts with "[UnityCrossThreadLogger]...: { JSON }"
        # We need to find the first '{'
        start_idx = last_match_event.find('{')
        if start_idx != -1:
            json_text = last_match_event[start_idx:]
            try:
                data = json.loads(json_text)
                with open(OUT_PATH, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"Saved match event to {OUT_PATH}")
                
                # Print relevant partials
                payload = data.get("Payload", {})
                if isinstance(payload, str):
                    try: 
                        payload = json.loads(payload) 
                    except: 
                        pass
                
                print("Payload extracted. Check file for details.")
                
            except json.JSONDecodeError:
                print("Failed to decode JSON from line")
                with open(OUT_PATH, "w") as f:
                    f.write(json_text)
    else:
        print("No MatchGameRoomStateChangedEvent found.")

if __name__ == "__main__":
    capture_match_event()
