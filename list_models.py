import os
import json
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("Modelos disponibles:")
for m in client.models.list():
    print(f"- {m.name} | Supported: {m.supported_actions}")
