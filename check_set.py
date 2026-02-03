import logging
import sys
from arenamcp.scryfall import ScryfallCache

# Configure logging
logging.basicConfig(level=logging.INFO)

def check_set():
    cache = ScryfallCache()
    card_name = "Feisty Spikeling"
    print(f"Searching for {card_name}...")
    
    # We need to access the internal method or just modify the class to return the raw dict
    # But ScryfallCache.get_card_by_name returns ScryfallCard dataclass which might not have 'set'.
    # Let's check ScryfallCache source again.
    # It returns ScryfallCard. ScryfallCard definition:
    # name, oracle_text, type_line, mana_cost, cmc, colors, arena_id, scryfall_uri.
    # No 'set' field!
    
    # We can use the _fetch_from_api directly if we hack it, or just use requests directly here.
    # Since we are running a script, let's just use requests to avoid modifying the codebase yet.
    import requests
    url = "https://api.scryfall.com/cards/named"
    params = {"fuzzy": card_name}
    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        data = resp.json()
        print(f"Set: {data.get('set')}")
        print(f"Set Name: {data.get('set_name')}")
        print(f"Card Data: {data}")
    else:
        print(f"Error: {resp.status_code}")

if __name__ == "__main__":
    check_set()
