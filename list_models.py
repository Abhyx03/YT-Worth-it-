from pathlib import Path
from dotenv import load_dotenv
import os
from google import genai

load_dotenv(dotenv_path=Path(__file__).parent / "Gemini_API.env")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

for m in client.models.list():
    print(m.name)
