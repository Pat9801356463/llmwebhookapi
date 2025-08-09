import os
import torch
from config import Config
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

class LocalLLM:
    def __init__(self, model_path: str = Config.LOCAL_MODEL_PATH, use_gpu: bool = Config.USE_GPU):
        print(f"🚀 Loading local model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        if use_gpu and torch.cuda.is_available():
            self.model = self.model.to("cuda")
            print("⚡ Using GPU")
        else:
            print("🧠 Using CPU")

        self.generator = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if use_gpu and torch.cuda.is_available() else -1
        )

    def generate(self, prompt: str, max_tokens: int = Config.MAX_TOKENS, temperature: float = Config.TEMPERATURE) -> str:
        print("📨 Prompt to LLM:", prompt[:200])
        output = self.generator(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
        )
        return output[0]['generated_text'].strip()

# 🔬 CLI Test
if __name__ == "__main__":
    llm = LocalLLM()
    test_prompt = (
        "A 46-year-old male with 3-month-old insurance had knee surgery in Pune. "
        "Is this covered under policy clauses? Provide decision and reasoning."
    )
    print("🧾 Response:", llm.generate(test_prompt))

