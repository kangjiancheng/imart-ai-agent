import re

_INJECTION_PATTERNS = [
    r"ignore (all |previous |prior |above )instructions",
    r"disregard (your |all |the )instructions",
    r"you are now (a |an )?(different|new|evil|unrestricted)",
    r"pretend (you are|to be) (a |an )?(different|new)",
    r"forget (everything|all instructions|your instructions)",
]


def check_injection(message: str) -> tuple[bool, str]:
    """Detect prompt injection attempts. Returns (passed, reason)."""
    msg_lower = message.lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, msg_lower):
            return False, "Prompt injection detected."
    return True, ""
