from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
if not key:
    raise SystemExit("GEMINI_API_KEY is missing from .env")

from google import genai
client = genai.Client(api_key=key)
r = client.models.generate_content(model=model, contents="Reply exactly with: Gemini observer OK")
print(r.text)
