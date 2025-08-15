# engine/reasoner.py
import json
from config import Config
from engine.query_parser import parse_query
from engine.pinecone_handler import retrieve_top_chunks  # Pinecone-backed retrieval

# === Dynamically select LLM client ===
if getattr(Config, "LLM_MODE", "gemini").lower() == "gemini":
    from engine.gemini_runner import GeminiLLM
    llm = GeminiLLM()
else:
    from engine.llm_local_runner import LocalLLM
    llm = LocalLLM()


def build_prompt(raw_query: str, parsed: dict, matched_clauses: list) -> str:
    """Build the insurance claim reasoning prompt for the LLM."""
    user_info = {
        "raw_query": raw_query,  # include original text for extra grounding
        "age": parsed.get("age", "unknown"),
        "procedure": parsed.get("procedure", "unknown"),
        "location": parsed.get("location", "unknown"),
        "policy_duration": parsed.get("policy_duration", "unknown"),
    }

    clause_summary = [
        {
            "doc_type": c.get("doc_type", ""),
            "source": c.get("source", ""),
            "text_snippet": c.get("text", "")[:1200],
        }
        for c in matched_clauses
    ]

    # Strong guardrails to avoid "insufficient information" and bias toward approval unless clear exclusion
    prompt = f"""
You are an insurance claim decision AI.

## Inputs:
- Query details: {json.dumps(user_info, ensure_ascii=False)}
- Relevant policy clauses: {json.dumps(clause_summary, ensure_ascii=False)}

## Decision Policy (IMPORTANT):
1) Only respond in **valid JSON** (no markdown, no commentary).
2) JSON keys must be exactly:
   decision (string: "approved" or "rejected"),
   payout_amount (number; use 0 if rejected or unknown),
   justification (string),
   matched_clauses (array of provided clause objects),
   query_details (copy of input query details).
3) **Never** answer that information is "insufficient". If details are unclear, base your decision solely on the retrieved clauses and rules below.
4) If there is **any explicit exclusion** in the retrieved clauses that clearly matches the user's procedure/condition/timing → decision="rejected".
5) Otherwise, if there's a **waiting period**, and the policy duration in query_details is **greater than or equal to** that waiting period, the waiting period does **not** block approval.
6) If no exclusion obviously applies → decision="approved".
7) justification must quote or paraphrase the relevant clause text and explain the reasoning briefly.

## Output:
Return **only** a single JSON object following the keys above.
"""
    return prompt


def run_llm_reasoning(raw_query: str, parsed: dict, matched_clauses: list) -> dict:
    """Run the LLM with the built prompt and parse JSON output."""
    prompt = build_prompt(raw_query, parsed, matched_clauses)
    raw_output = llm.generate(prompt)

    # Try to extract a single JSON object robustly
    try:
        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1
        json_block = raw_output[start:end]
        result = json.loads(json_block)

        if "payout_amount" in result and "amount" not in result:
            result["amount"] = result["payout_amount"]

        result.setdefault("query_details", parsed)
        result.setdefault("matched_clauses", matched_clauses)
        return result
    except Exception as e:
        return {
            "decision": "approved",  # bias to approval when LLM glitches but we had clauses
            "payout_amount": 0,
            "amount": 0,
            "justification": f"Parser fallback. Could not parse LLM JSON cleanly: {e}. Raw Output trimmed: {raw_output[:400]}",
            "matched_clauses": matched_clauses,
            "query_details": parsed,
        }


def reason_over_query(raw_query: str, doc_id: str, top_k: int = 5) -> dict:
    """
    Full reasoning pipeline: parse -> retrieve -> reason.
    Gracefully handles empty retrieval.
    """
    parsed = parse_query(raw_query)

    matched_texts = retrieve_top_chunks(raw_query, doc_id, top_k=top_k) or []
    if not matched_texts:
        return {
            "decision": "approved",  # prefer approval when no exclusion evidence is found
            "payout_amount": 0,
            "amount": 0,
            "justification": "No explicit exclusion retrieved; by default approve unless a clear exclusion is found.",
            "matched_clauses": [],
            "query_details": parsed,
        }

    matched_clauses = [
        {"text": t, "source": doc_id, "doc_type": "policy"}
        for t in matched_texts if t
    ]

    if not matched_clauses:
        return {
            "decision": "approved",
            "payout_amount": 0,
            "amount": 0,
            "justification": "No valid clause text retrieved; defaulting to approval in absence of exclusions.",
            "matched_clauses": [],
            "query_details": parsed,
        }

    return run_llm_reasoning(raw_query, parsed, matched_clauses)


# Alias
decide = reason_over_query

if __name__ == "__main__":
    sample_query = "46M, knee surgery in Pune, 3-month policy"
    sample_doc_id = "testdoc12345678"
    from pprint import pprint
    pprint(reason_over_query(sample_query, sample_doc_id))
