# engine/gemini_runner.py
import os
import json
import requests
from typing import Optional
from config import Config

class GeminiLLM:
    """
    Minimal Gemini wrapper for the Google Generative Language API (v1beta).
    Uses API key auth (query param) by default and calls the model endpoint:
    https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent

    Note: The exact response format may vary by API version; this class attempts
    to extract text from common response shapes.
    """
    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None, timeout: int = 15):
        self.model_name = model_name or getattr(Config, "GEMINI_MODEL", "gemini-1.5-flash")
        self.api_key = api_key or getattr(Config, "GEMINI_API_KEY", "")
        self.timeout = timeout

        # Primary URL (v1beta generateContent)
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"

        if not self.api_key:
            # Warning only — user code should ensure key is set in env/.env
            print("⚠️ GEMINI_API_KEY is not set. Set Config.GEMINI_API_KEY to call Gemini API.")

    def _extract_text(self, data: dict) -> str:
        """
        Try to extract generated text from common Gemini response formats.
        Returns empty string if nothing found.
        """
        # Common: { "candidates": [ { "content": { "parts": [{"text": "..."}] } } ] }
        try:
            if "candidates" in data and isinstance(data["candidates"], list):
                c = data["candidates"][0]
                # some shapes: c["content"]["parts"][0]["text"]
                if isinstance(c, dict):
                    # nested content->parts->text
                    content = c.get("content") or c.get("message") or {}
                    if isinstance(content, dict):
                        parts = content.get("parts") or content.get("text") or []
                        if parts and isinstance(parts, list):
                            part0 = parts[0]
                            if isinstance(part0, dict) and "text" in part0:
                                return part0["text"].strip()
                            if isinstance(part0, str):
                                return part0.strip()

                    # sometimes text directly:
                    if "text" in c:
                        return c["text"].strip()

            # Another possible shape: { "output": [ { "content": [ {"text": "..."} ] } ] }
            if "output" in data and isinstance(data["output"], list):
                out = data["output"][0]
                content = out.get("content") or []
                if content and isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            return item["text"].strip()

            # Another shape: { "candidates":[ {"message": {"content":[ {"text":"..."}]}} ] }
            if "candidates" in data:
                for c in data["candidates"]:
                    if isinstance(c, dict):
                        msg = c.get("message") or c.get("content") or {}
                        if isinstance(msg, dict):
                            cont = msg.get("content") or []
                            if cont and isinstance(cont, list):
                                for itm in cont:
                                    if isinstance(itm, dict) and "text" in itm:
                                        return itm["text"].strip()

            # As a fallback check top-level fields
            for key in ("response", "generated_text", "text"):
                if key in data and isinstance(data[key], str):
                    return data[key].strip()

        except Exception:
            # If parsing fails, just continue to return empty
            pass

        return ""

    def generate(self, prompt: str, max_tokens: int = None, temperature: float = None) -> str:
        """
        Generate text from Gemini. Returns the text or a failure string starting with ❌.
        """
        max_tokens = int(max_tokens or getattr(Config, "MAX_TOKENS", 1024))
        temperature = float(temperature if temperature is not None else getattr(Config, "TEMPERATURE", 0.3))

        headers = {"Content-Type": "application/json"}
        params = {}
        if self.api_key:
            params["key"] = self.api_key

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
            resp = requests.post(self.api_url, headers=headers, params=params, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            text = self._extract_text(data)
            if text:
                return text
            # If nothing parsed, return a helpful debug string (without leaking API key)
            return f"❌ Gemini returned unexpected JSON shape. Raw keys: {list(data.keys())}"

        except requests.HTTPError as http_err:
            # Include status code and brief message
            try:
                err_body = resp.text
            except Exception:
                err_body = "<no-body>"
            return f"❌ Gemini API HTTPError: {http_err} - body: {err_body[:1000]}"

        except Exception as e:
            return f"❌ Gemini API call failed: {e}"

    def health_check(self) -> dict:
        """Simple health helper (doesn't call API if no key)."""
        if not self.api_key:
            return {"ok": False, "reason": "no_api_key"}
        return {"ok": True, "model": self.model_name, "api_url": self.api_url}


# 🔬 CLI Test
if __name__ == "__main__":
    gemini = GeminiLLM()
    test_prompt = (
        "A 46-year-old male with 3-month-old insurance had knee surgery in Pune. "
        "Is this covered under policy clauses? Provide decision and reasoning in JSON only."
    )
    print("🧾 Response:", gemini.generate(test_prompt))
