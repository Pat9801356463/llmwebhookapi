import json
from typing import Dict, Any

def format_decision_response(reasoning_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format the reasoning result into a structured API/UI response.
    Ensures safe access to fields and trims long clause text.
    """
    return {
        "query_details": reasoning_result.get("parsed", {}),
        "decision": reasoning_result.get("decision", "unknown"),
        "payout_amount": reasoning_result.get("amount"),
        "justification": reasoning_result.get("justification", "No explanation provided."),
        "matched_clauses": [
            {
                "source": clause.get("source", "unknown"),
                "doc_type": clause.get("doc_type", "unknown"),
                "text_snippet": (
                    (clause.get("text") or "No text available")[:300] + "..."
                    if clause.get("text") and len(clause.get("text")) > 300
                    else (clause.get("text") or "No text available")
                )
            }
            for clause in reasoning_result.get("matched_clauses", [])
        ]
    }

def format_pretty_print(response_dict: Dict[str, Any]) -> str:
    """
    Returns a readable, indented JSON string version of the response (for logs/UI).
    """
    return json.dumps(response_dict, indent=2, ensure_ascii=False)

# Backward compatibility alias
format_response = format_decision_response

# 🧪 Example usage
if __name__ == "__main__":
    sample_reasoning_result = {
        "parsed": {"age": 46, "condition": "knee surgery", "location": "Pune"},
        "decision": "approve",
        "amount": 120000,
        "justification": "Covered under standard orthopedic surgery terms.",
        "matched_clauses": [
            {
                "source": "policy_doc_1",
                "doc_type": "policy",
                "text": "Knee surgeries are covered up to ₹1,50,000 in the 3-month plan..."
            }
        ]
    }

    formatted = format_decision_response(sample_reasoning_result)
    print("✅ Final Output:\n")
    print(format_pretty_print(formatted))
