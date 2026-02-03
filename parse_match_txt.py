
import json

TXT_PATH = r"z:\ArenaMCP\last_match_event.txt"

def parse():
    try:
        with open(TXT_PATH, "r", encoding="utf-16-le", errors="ignore") as f:
            content = f.read()
            
        print(f"Read {len(content)} chars")
        # Find JSON start
        start = content.find('{')
        if start != -1:
            json_str = content[start:]
            data = json.loads(json_str)
            
            # Navigate to player info
            # Structure usually: MatchGameRoomStateChangedEvent -> gameRoomInfo -> gameRoomConfig -> reservedPlayers
            print("Top level keys:", str(list(data.keys())))
            
            payload = data.get("matchGameRoomStateChangedEvent", {}).get("gameRoomInfo", {}).get("gameRoomConfig", {})
            if not payload:
                # deeper nested?
                 print("Structure might be nested differently. Dumping partial.")
                 print(json.dumps(data, indent=2)[:500])
                 return

            print("Found GameRoomConfig. Players:")
            players = payload.get("reservedPlayers", [])
            for p in players:
                print(f"UserId: {p.get('userId')} | Name: {p.get('playerName')} | Team: {p.get('teamId')} | SystemSeat: {p.get('systemSeatId')}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parse()
