# src/guardrails/pii_filter.py
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS PII?
#
# PII = Personally Identifiable Information — data that could identify a real person:
#   - Email addresses         → replaced with [EMAIL]
#   - Phone numbers           → replaced with [PHONE]
#   - Social Security Numbers → replaced with [SSN]
#   - Credit card numbers     → replaced with [CARD]
#
# WHY REDACT INSTEAD OF BLOCK?
#
# Unlike content_policy.py (which BLOCKS), PII redaction CLEANS and CONTINUES.
# Reason: the user's question may still be valid — they just accidentally included
# personal data. We scrub it and let Claude answer the underlying question.
#
# Example:
#   Input:  "My email is alice@example.com — can you help me reset my account?"
#   Output: "My email is [EMAIL] — can you help me reset my account?"
#   Effect: Claude answers the account question, but the email is never sent to the API.
#
# COMPLIANCE:
# In production, every redaction should be logged with a timestamp for GDPR audit trails.
# ─────────────────────────────────────────────────────────────────────────────

import re
# re = Python's built-in regular expression module (standard library, no install needed)


# ── PII pattern list ─────────────────────────────────────────────────────────
# Each item is a TUPLE of (regex_pattern, replacement_token).
# A tuple = an immutable (unchangeable) ordered pair. Written as (a, b).
# In JS/TS: similar to a readonly [string, string] pair.
_PII_PATTERNS = [
    # ── Email addresses ───────────────────────────────────────────────────────
    # Pattern breakdown: user.name+tag@sub.domain.com
    #   [A-Za-z0-9._%+-]+  = one or more valid email username characters
    #   @                   = literal @ symbol
    #   [A-Za-z0-9.-]+     = domain name (letters, digits, dots, hyphens)
    #   \.                  = literal dot (backslash escapes the dot — unescaped . = "any char")
    #   [A-Z|a-z]{2,}      = top-level domain (com, org, io…), at least 2 chars
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),

    # ── US phone numbers ──────────────────────────────────────────────────────
    # Matches formats: 555-123-4567  555.123.4567  555 123 4567  5551234567
    #   \d{3}       = exactly 3 digits (\d = any digit 0-9, same as [0-9])
    #   [-.\ s]?    = optional separator: hyphen, dot, or space (? = zero or one)
    (r"\b\d{3}[-.\ s]?\d{3}[-.\ s]?\d{4}\b", "[PHONE]"),

    # ── US Social Security Numbers ────────────────────────────────────────────
    # Only matches the standard dashed format: 123-45-6789
    #   \d{3}-\d{2}-\d{4} = exactly 3 digits, dash, 2 digits, dash, 4 digits
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),

    # ── Credit card numbers ───────────────────────────────────────────────────
    # Matches 13–16 digit sequences with optional spaces or hyphens between groups.
    #   (?:...)     = non-capturing group — group without creating a numbered reference
    #   \d[ -]*?    = one digit followed by zero or more optional spaces or hyphens
    #   {13,16}     = the above repeated 13 to 16 times total
    (r"\b(?:\d[ -]*?){13,16}\b", "[CARD]"),
]


def redact_pii(message: str) -> str:
    """
    Scan the message for PII patterns and replace each match with a safe token.

    WHAT IT DOES:
      Runs re.sub() for each pattern, replacing every match with the placeholder token.
      Patterns run sequentially — each one runs on the already-cleaned result, so
      multiple PII types in one message are all caught.

    WHAT IT DOES NOT DO:
      - Does NOT block the request (never returns False or raises an error)
      - Does NOT log to a database (add that in production for GDPR compliance)
      - Does NOT catch all PII — names, addresses, DOB need NLP/NER models to detect

    EFFECT ON THE AGENT:
      The returned string (with [EMAIL], [PHONE], etc.) is what Claude receives.
      The personal data is never sent to the Anthropic API or written to logs.

    RETURN TYPE — str:
      Returns the sanitized message string. Identical to input if no PII was found.
    """
    for pattern, replacement in _PII_PATTERNS:
        # PYTHON CONCEPT — tuple unpacking in a for loop:
        # Each item in _PII_PATTERNS is a (pattern_string, replacement_string) tuple.
        # `for pattern, replacement in ...` unpacks each tuple into two named variables
        # in one step. Equivalent to:
        #   for item in _PII_PATTERNS:
        #       pattern = item[0]
        #       replacement = item[1]
        # In JS/TS: for (const [pattern, replacement] of PII_PATTERNS)

        message = re.sub(pattern, replacement, message)
        # re.sub(pattern, replacement, string):
        #   = find ALL matches of `pattern` in `string`
        #   = replace each match with `replacement`
        #   = return the resulting new string
        #
        # Example:
        #   re.sub(email_pattern, "[EMAIL]", "reach me at alice@test.com thanks")
        #   → "reach me at [EMAIL] thanks"
        #
        # We reassign `message` each iteration so the next pattern runs on the
        # already-cleaned version — important when a message contains multiple PII types.

    return message
    # Returns the fully sanitized message string.
    # If no PII patterns matched, this is identical to the original input.
