import re

_BLOCKED_PATTERNS = [
    r"\b(how to make|synthesize|manufacture)\b.{0,30}\b(bomb|weapon|explosive|poison)\b",
    r"\b(hack|exploit|attack)\b.{0,20}\b(system|server|database|account)\b",
]


def check_content_policy(message: str) -> tuple[bool, str]:
    """Check message against blocked patterns. Returns (passed, reason)."""
    msg_lower = message.lower()
    for pattern in _BLOCKED_PATTERNS:
        if re.search(pattern, msg_lower):
            return False, "Request blocked by content policy."
    return True, ""
