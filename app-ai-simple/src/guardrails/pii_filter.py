import re

_PII_PATTERNS = [
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
    (r"\b\d{3}[-.\ s]?\d{3}[-.\ s]?\d{4}\b", "[PHONE]"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
    (r"\b(?:\d[ -]*?){13,16}\b", "[CARD]"),
]


def redact_pii(message: str) -> str:
    """Replace PII (emails, phones, SSNs, credit cards) with placeholder tokens."""
    for pattern, replacement in _PII_PATTERNS:
        message = re.sub(pattern, replacement, message)
    return message
