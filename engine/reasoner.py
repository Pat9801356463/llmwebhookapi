from engine.query_parser import parse_query
from engine.retriever import retrieve_clauses
from config import Config
import json

# Dynamically load LLM based on mode
if getattr(Config, "LLM_MODE", "local").lower() == "gemini":
    from engine.gemini_runner import GeminiLLM
    llm = GeminiLLM()
else:
    from engine.llm_local_runner import LocalLLM
    llm = LocalLLM()


def build_prompt(parsed: dict, matched_clauses: list) -> str:
    user_info = {
        "age": parsed.get("age", "unknown"),
        "procedure": parsed.get("procedure", "unknown"),
        "location": parsed.get("location", "unknown"),
        "policy_duration": parsed.get("policy_duration", "unknown")
    }

    # Include all matched clauses in full
    clause_summary = [
        {
            "doc_type": c.get("doc_type", ""),
            "source": c.get("source", ""),
            "text_snippet": c.get("text", "")[:1000]  # Limit per clause to avoid token overflow
        }
        for c in matched_clauses
    ]

    prompt = f"""
You are an insurance claim decision AI.

## Inputs:
- Query details: {json.dumps(user_info, ensure_ascii=False)}
- Relevant policy clauses: {json.dumps(clause_summary, ensure_ascii=False)}

## Rules:
1. Only respond in **valid JSON**.
2. JSON keys must be:
   decision (string: "approved" or "rejected"),
   payout_amount (number, 0 if rejected),
   justification (string),
   matched_clauses (array of provided clauses),
   query_details (copy of input query details).
3. If the policy has a waiting period (e.g., "30 days") and the policy duration is greater than that period, 
   this waiting period should NOT cause rejection — coverage should be approved unless another exclusion applies.
4. Approve if there is **no explicit exclusion** in the retrieved clauses that matches the procedure or condition.
5. Reject only if there is **any explicit exclusion** that clearly applies.
6. Always justify by referencing clause text directly and explain how the decision was made.

## Output:
Return only valid JSON — no extra text, no markdown.
"""
    return prompt


def run_llm_reasoning(parsed: dict, matched_clauses: list) -> dict:
    prompt = build_prompt(parsed, matched_clauses)
    raw_output = llm.generate(prompt)

    try:
        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1
        json_block = raw_output[start:end]
        return json.loads(json_block)
    except Exception as e:
        return {
            "decision": "unknown",
            "payout_amount": None,
            "justification": f"Failed to parse LLM response: {e}\nRaw Output: {raw_output}",
            "matched_clauses": matched_clauses,
            "query_details": parsed
        }


def reason_over_query(raw_query: str) -> dict:
    parsed = parse_query(raw_query)
    matched_clauses = retrieve_clauses(parsed, top_k=5)

    if not matched_clauses:
        return {
            "decision": "unknown",
            "payout_amount": None,
            "justification": "No relevant clauses found.",
            "matched_clauses": [],
            "query_details": parsed
        }

    result = run_llm_reasoning(parsed, matched_clauses)
    return result


# Alias for Flask app
decide = reason_over_query

# 🧪 CLI Test
if __name__ == "__main__":
    query = "46M, knee surgery in Pune, 3-month policy"
    from pprint import pprint
    pprint(reason_over_query(query))
