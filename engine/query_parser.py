import re
from typing import Dict

# Regex patterns for key fields
AGE_PATTERN = re.compile(r'(\d{1,3})\s*[-]?\s*(year\s*old|yo|yr|y/o|M|F)?', re.IGNORECASE)
PROCEDURE_PATTERN = re.compile(r"(surgery|treatment|operation|therapy|scan|transplant|procedure|hospitalization)", re.IGNORECASE)
LOCATION_PATTERN = re.compile(r"in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", re.IGNORECASE)
DURATION_PATTERN = re.compile(r"(\d+)\s*[-]?\s*(month|year|day)s?", re.IGNORECASE)

def parse_query(query: str) -> Dict[str, str]:
    """Extracts age, procedure, location, and policy duration from query"""
    result = {
        "age": None,
        "gender": None,
        "procedure": None,
        "location": None,
        "policy_duration": None
    }

    # --- Age & Gender
    age_match = AGE_PATTERN.search(query)
    if age_match:
        result["age"] = int(age_match.group(1))
        result["gender"] = age_match.group(2).upper() if age_match.group(2) else None

    # --- Procedure
    procedure_match = PROCEDURE_PATTERN.search(query)
    if procedure_match:
        result["procedure"] = procedure_match.group(0).lower()

    # --- Location
    location_match = LOCATION_PATTERN.search(query)
    if location_match:
        result["location"] = location_match.group(1)

    # --- Policy Duration
    duration_match = DURATION_PATTERN.search(query)
    if duration_match:
        number = int(duration_match.group(1))
        unit = duration_match.group(2).lower()
        result["policy_duration"] = f"{number} {unit}{'s' if number > 1 else ''}"

    return result

# 🧪 Example usage
if __name__ == "__main__":
    sample = "46M, knee surgery in Pune, 3-month policy"
    parsed = parse_query(sample)
    print("🧠 Parsed Query:\n", parsed)

