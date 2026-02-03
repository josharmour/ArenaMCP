
import os
from google import genai
from google.genai import types
import dotenv

dotenv.load_dotenv("z:/ArenaMCP/.env")
api_key = os.environ.get("GOOGLE_API_KEY")

print(f"Checking models with key: {api_key[:5]}...")

client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

try:
    print("\nListing models:")
    # Pager for list_models if needed, but usually returns an iterable
    print("Inspecting first model attributes:")
    for m in client.models.list():
        # print(f"Model: {m.name}")
        # supported_actions seems to be the one
        actions = getattr(m, 'supported_actions', [])
        print(f"{m.name}: {actions}")
except Exception as e:
    print(f"Error listing models: {e}")

print("\nChecking specifically for gemini-2.0-flash-exp...")
try:
    m = client.models.get(model="gemini-2.0-flash-exp")
    print(f"Found: {m.name}")
    print(f"Methods: {m.supported_generation_methods}")
except Exception as e:
    print(f"gemini-2.0-flash-exp lookup failed: {e}")
