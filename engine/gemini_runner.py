# engine/gemini_runner.py
import os
import json
import requests
from config import Config

class GeminiLLM:
    def __init__(self, model_name=None, api_key=None):
        self.model_name = model_name or "gemini-1.5-flash"
        self.api_key = api_key or Config.GEMINI_API_KEY


    def generate(self, prompt: str, max_tokens: int = Config.MAX_TOKENS, temperature: float = Config.TEMPERATURE) -> str:
        """
        Calls the Gemini API and returns the generated text.
        """
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}

        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topK": 50,
                "topP": 0.95
            }
        }

        try:
            resp = requests.post(self.api_url, headers=headers, params=params, json=payload)
            resp.raise_for_status()
            data = resp.json()

            # Extract text from Gemini response
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            return f"❌ Gemini API call failed: {e}"


# 🔬 CLI Test
if __name__ == "__main__":
    gemini = GeminiLLM()
    test_prompt = (
        "A 46-year-old male with 3-month-old insurance had knee surgery in Pune. "
        "Is this covered under policy clauses? Provide decision and reasoning."
    )
    print("🧾 Response:", gemini.generate(test_prompt))
