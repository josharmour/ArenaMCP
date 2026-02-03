
import os
import logging
from google import genai
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Configure logging
# logging.basicConfig(level=logging.INFO)

def list_gemini_models():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not set")
        return

    print("--- Gemini Models ---")
    try:
        client = genai.Client(api_key=api_key)
        # Pager object, iterate to get models
        for m in client.models.list():
            print(f"Model: {m.name}")
            print(f"  DisplayName: {m.display_name}")
            print(f"  SupportedMethods: {m.supported_generation_methods}")
            print("-" * 20)
    except Exception as e:
        print(f"Error listing Gemini models: {e}")

def list_azure_models():
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    if not api_key or not endpoint:
        print("\nAZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set")
        return

    print("\n--- Azure Models (Deployments) ---")
    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        
        # Note: client.models.list() lists *base models* available for deployment, 
        # NOT the deployments themselves usually. 
        # But let's see what it returns.
        # To list deployments specifically often requires Management Client or specific endpoint logic.
        # But often for standard users, they just need to know what they called their deployments.
        # We'll list what we can.
        
        for m in client.models.list():
            print(f"ID: {m.id}, Created: {m.created}, Object: {m.object}")

    except Exception as e:
        print(f"Error listing Azure models: {e}")

if __name__ == "__main__":
    list_gemini_models()
    # list_azure_models()
