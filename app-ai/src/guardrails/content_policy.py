# src/guardrails/content_policy.py
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS THIS FILE?
#
# This is Check 1 of 3 in the guardrail pipeline.
# It detects harmful or illegal requests and blocks them BEFORE Claude sees them.
#
# HOW IT WORKS — regex pattern matching:
# Each pattern is a "regular expression" (regex) — a mini-language for describing
# text patterns. Python's `re` module does the matching.
#
# Example pattern breakdown:
#   r"\b(how to make|synthesize)\b.{0,30}\b(bomb|weapon)\b"
#
#   r"..."      = raw string — backslashes are literal, not escape characters
#   \b          = word boundary — the match must start/end at a word edge
#                 (prevents "hacksaw" from matching "hack")
#   (a|b|c)     = alternation — match any of these options (like OR in code)
#   .{0,30}     = any character (.), repeated 0 to 30 times — allows words in between
#   \b(bomb|weapon)\b = must end at a word boundary with one of these words
#
# WHY REGEX INSTEAD OF AI?
# Guardrails must be FAST (< 100ms) and DETERMINISTIC (same input = same output).
# AI calls are slow and non-deterministic. Regex runs in microseconds.
# ─────────────────────────────────────────────────────────────────────────────

import re
# `re` = Python's built-in regular expression module.
# No installation needed — it's part of Python's standard library.
# In JS/TS: similar to using RegExp objects or the `re2` npm package.


# ── Module-level constant: the blocklist of harmful patterns ─────────────────
# Naming convention: leading underscore (_) = "private to this module".
# Other files can still import it, but the underscore signals "don't use this directly".
# In JS/TS: this would be a non-exported const.
_BLOCKED_PATTERNS = [
    # Pattern 1: Detects requests for making dangerous physical items.
    # Matches phrases like:
    #   "how to make a bomb"
    #   "synthesize poison at home"
    #   "manufacture explosive devices"
    r"\b(how to make|synthesize|manufacture)\b.{0,30}\b(bomb|weapon|explosive|poison)\b",

    # Pattern 2: Detects requests to attack computer systems.
    # Matches phrases like:
    #   "how to hack a server"
    #   "exploit a database"
    #   "attack an account"
    r"\b(hack|exploit|attack)\b.{0,20}\b(system|server|database|account)\b",
]
# NOTE: These are intentionally minimal. Production systems add many more patterns
# and combine regex with an LLM-based classifier for nuanced cases.


def check_content_policy(message: str) -> tuple[bool, str]:
    """
    Scan the user's message against the blocked pattern list.

    RETURN TYPE — tuple[bool, str]:
    This function returns TWO values packed into a "tuple" (an immutable pair).
    In Python, you can return multiple values by separating them with a comma:
      return True, ""          → returns the tuple (True, "")
      return False, "reason"   → returns the tuple (False, "reason")

    The caller unpacks them in one line:
      passed, reason = check_content_policy(message)
    In JS/TS: similar to returning { passed: bool, reason: string }

    RETURN VALUES:
      (True, "")              → message is safe, let it through
      (False, "reason string") → message is blocked, tell the user why

    EFFECT ON THE AGENT:
      If this returns False, the checker.py returns immediately with a 422 HTTP error.
      Claude never sees the message. No LLM call is made.
    """
    msg_lower = message.lower()
    # .lower() = convert to lowercase so the patterns are case-insensitive.
    # Without this, "How To Make A Bomb" would NOT match "how to make" (different case).
    # We do this once and reuse msg_lower instead of converting inside every loop iteration.

    for pattern in _BLOCKED_PATTERNS:
        # Iterate over each regex pattern in the list.
        # `for pattern in _BLOCKED_PATTERNS` = loop that assigns each list item to `pattern`

        if re.search(pattern, msg_lower):
            # re.search(pattern, string) = scan through `string` looking for ANY location
            # where the `pattern` matches. Returns a Match object if found, None if not.
            # `if re.search(...)` = True when a match is found (Match objects are truthy).
            # In JS/TS: similar to /pattern/.test(string)

            return False, "Request blocked by content policy."
            # Early return — stop checking other patterns, the message is already blocked.
            # Returning here means the `for` loop exits immediately.

    # If we reach here, no pattern matched → the message is safe.
    return True, ""
    # Empty string for reason = "no reason needed, it passed"
