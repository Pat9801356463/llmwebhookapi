# engine/query_parser.py
import re
from typing import Dict, Optional

# Regex patterns for key fields
AGE_PATTERN = re.compile(r"(\d{1,3})\s*[-]?\s*(year\s*old|yo|yr|y/o|M|F)?", re.IGNORECASE)
PROCEDURE_PATTERN = re.compile(
    r"(surgery|treatment|operation|therapy|scan|transplant|procedure|hospitalization)",
    re.IGNORECASE,
)
LOCATION_PATTERN = re.compile(r"in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", re.IGNORECASE)
DURATION_PATTERN = re.compile(r"(\d+)\s*[-]?\s*(month|year|day)s?", re.IGNORECASE)


def safe_upper(s: Optional[str]) -> Optional[str]:
    """Uppercase a string if not None."""
    return s.upper() if s else None


def parse_query(query: str) -> Dict[str, Optional[str]]:
    """
    Extracts structured details from a natural-language insurance query.
    Detects: age, gender, procedure, location, and policy duration.
    """
    result = {
        "age": None,
        "gender": None,
        "procedure": None,
        "location": None,
        "policy_duration": None,
    }

    # --- Age & Gender ---
    age_match = AGE_PATTERN.search(query)
    if age_match:
        result["age"] = int(age_match.group(1))
        gender_candidate = age_match.group(2)
        if gender_candidate:
            gender_candidate = gender_candidate.strip().upper()
            if gender_candidate in {"M", "F"}:
                result["gender"] = gender_candidate
            else:
                result["gender"] = None

    # --- Procedure ---
    procedure_match = PROCEDURE_PATTERN.search(query)
    if procedure_match:
        result["procedure"] = procedure_match.group(0).lower()

    # --- Location ---
    location_match = LOCATION_PATTERN.search(query)
    if location_match:
        result["location"] = location_match.group(1)

    # --- Policy Duration ---
    duration_match = DURATION_PATTERN.search(query)
    if duration_match:
        number = int(duration_match.group(1))
        unit = duration_match.group(2).lower()
        result["policy_duration"] = f"{number} {unit}{'s' if number > 1 else ''}"

    return result


if __name__ == "__main__":
    samples = [
        "46M, knee surgery in Pune, 3-month policy",
        "female 30 yo, MRI scan in New Delhi for 1 year",
        "65 year old male requires hospitalization in Mumbai for 2 years",
    ]
    for s in samples:
        print(f"Query: {s}")
        print("Parsed:", parse_query(s))
        print("-" * 50)
