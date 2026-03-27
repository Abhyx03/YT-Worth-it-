import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "Gemini_API.env")
api_key = os.getenv("GEMINI_API_KEY")
print(f"Using key: {api_key[:8]}...")

models = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]

for model in models:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": "Say hello in one word"}]}]}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print(f"✓ {model} WORKS")
        break
    else:
        data = response.json()
        msg = data.get("error", {}).get("message", "")[:80]
        print(f"✗ {model} — {response.status_code} — {msg}")
