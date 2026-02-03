import requests

def check_set():
    card_name = "Feisty Spikeling"
    import requests
    url = "https://api.scryfall.com/cards/named"
    params = {"fuzzy": card_name}
    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        data = resp.json()
        print(f"SET_CODE_IS:{data.get('set')}")
    else:
        print(f"Error: {resp.status_code}")

if __name__ == "__main__":
    check_set()
