import re

# Regex patterns that match harmful or illegal request phrasing.
# Evaluated before Claude sees any input — deterministic, < 1 ms per check.
_BLOCKED_PATTERNS = [
    r"\b(how to make|synthesize|manufacture)\b.{0,30}\b(bomb|weapon|explosive|poison)\b",
    r"\b(hack|exploit|attack)\b.{0,20}\b(system|server|database|account)\b",
]


def check_content_policy(message: str) -> tuple[bool, str]:
    """
    Scan message against blocked pattern list.

    Returns (True, "") if safe, or (False, reason) if blocked.
    Called by GuardrailChecker before PII redaction and injection detection.
    """
    msg_lower = message.lower()
    for pattern in _BLOCKED_PATTERNS:
        if re.search(pattern, msg_lower):
            return False, "Request blocked by content policy."
    return True, ""
