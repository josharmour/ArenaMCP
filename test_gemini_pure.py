
import os
import asyncio
import logging
import traceback
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_gemini")

# Load .env
from pathlib import Path
env_path = Path(".env")
if not env_path.exists():
    env_path = Path(__file__).parent.parent.parent / ".env"
    
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if strip_line := line.strip():
                if not strip_line.startswith("#") and "=" in strip_line:
                    key, val = strip_line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

API_KEY = os.environ.get("GOOGLE_API_KEY")
MODEL = "gemini-2.0-flash-exp"

async def test_connect():
    if not API_KEY:
        print("GOOGLE_API_KEY not set")
        return

    print(f"Testing connection to {MODEL}...")
    
    client = genai.Client(api_key=API_KEY, http_options={"api_version": "v1alpha"})
    
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=types.Content(parts=[types.Part(text="Say hello.")]),
    )

    try:
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            print("Connected successfully!")
            
            # Send a simple text message
            print("Sending text: Hello")
            await session.send(input="Hello", end_of_turn=True)
            
            print("Waiting for response...")
            async for response in session.receive():
                if response.server_content:
                    print(f"Received content: {response.server_content}")
                    if response.server_content.turn_complete:
                        print("Turn complete.")
                        break
            
            print("Test passed.")
            
    except Exception as e:
        print(f"Connection failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(test_connect())
    except KeyboardInterrupt:
        pass
