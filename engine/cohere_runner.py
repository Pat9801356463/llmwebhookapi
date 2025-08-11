# engine/cohere_runner.py
import os
from typing import Optional
import cohere
from config import Config

class CohereLLM:
    """
    Minimal Cohere wrapper for text generation.
    Primary usage: fallback when Gemini API fails or daily limit is exceeded.
    """

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None, timeout: int = 15):
        self.model_name = model_name or getattr(Config, "COHERE_MODEL", "command-xlarge-nightly")
        self.api_key = api_key or getattr(Config, "COHERE_API_KEY", "")
        self.timeout = timeout

        if not self.api_key:
            print("⚠️ COHERE_API_KEY is not set. Set Config.COHERE_API_KEY to call Cohere API.")

        try:
            self.client = cohere.Client(self.api_key)
        except Exception as e:
            print(f"❌ Failed to initialize Cohere client: {e}")
            self.client = None

    def generate(self, prompt: str, max_tokens: int = None, temperature: float = None) -> str:
        """
        Generate text from Cohere. Returns the text or a failure string starting with ❌.
        """
        if not self.client:
            return "❌ Cohere client not initialized (missing API key?)"

        max_tokens = int(max_tokens or getattr(Config, "MAX_TOKENS", 300))
        temperature = float(temperature if temperature is not None else getattr(Config, "TEMPERATURE", 0.3))

        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                k=50,
                p=0.95,
                stop_sequences=[],
                return_likelihoods='NONE'
            )
            if response and hasattr(response, "generations") and response.generations:
                return response.generations[0].text.strip()
            return "❌ Cohere returned empty response"
        except Exception as e:
            return f"❌ Cohere API call failed: {e}"

    def health_check(self) -> dict:
        """Simple health helper (no API call if no key)."""
        if not self.api_key:
            return {"ok": False, "reason": "no_api_key"}
        return {"ok": True, "model": self.model_name}
        

# 🔬 CLI Test
if __name__ == "__main__":
    cohere_llm = CohereLLM()
    test_prompt = (
        "A 46-year-old male with 3-month-old insurance had knee surgery in Pune. "
        "Is this covered under policy clauses? Provide decision and reasoning in JSON only."
    )
    print("🧾 Response:", cohere_llm.generate(test_prompt))
