# src/guardrails/checker.py
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS THIS FILE?
#
# The "guardrails" are a safety gate that runs BEFORE Claude sees any user input.
# Think of it as a security guard at the door — it inspects every message and
# either lets it through (possibly after cleaning it up) or rejects it entirely.
#
# The checker orchestrates THREE checks in a fixed order:
#   1. Content policy  — block harmful requests (violence, illegal instructions)
#   2. PII redaction   — scrub personal data (emails, phone numbers, credit cards)
#   3. Injection check — detect "ignore all previous instructions" attacks
#
# WHY MUST GUARDRAILS RUN BEFORE THE LLM?
#   - Once Claude sees harmful input, it might act on it even if it refuses.
#   - Prompt injection can override Claude's system prompt.
#   - PII sent to any API = compliance risk (GDPR, CCPA).
#   - The guardrails cannot be bypassed by any tool call or agent action.
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass
# @dataclass = auto-generates __init__, __repr__, __eq__ for the class
# So `GuardrailResult(passed=True, sanitized_message="Hello")` works automatically.

from src.guardrails.content_policy import check_content_policy
from src.guardrails.pii_filter import redact_pii
from src.guardrails.injection_detector import check_injection
# Each import brings in one pure function from its own file.
# Keeping them separate makes them easy to test and swap out independently.


@dataclass
class GuardrailResult:
    """
    The return value from GuardrailChecker.check().
    Carries the verdict (passed/blocked) and the cleaned-up message.

    FIELDS:
      passed           = True if the message passed all checks, False if blocked
      sanitized_message = the message after PII redaction (may differ from original)
      reason           = why the message was blocked (empty string if passed)

    USAGE IN THE ROUTER (src/routers/agent.py):
      result = guardrails.check(request.message)
      if not result.passed:
          raise HTTPException(422, result.reason)   # "Prompt injection detected."
      request.message = result.sanitized_message    # use the cleaned version
    """
    passed: bool
    sanitized_message: str
    reason: str = ""   # default = empty string (only set when passed=False)


class GuardrailChecker:
    """
    Orchestrates the three guardrail checks in sequence.
    Instantiated once as a module-level singleton in the router.
    """

    def check(self, message: str) -> GuardrailResult:
        """
        Run all three safety checks on the user message.

        `message: str`  = type hint: the parameter must be a string
        `-> GuardrailResult` = type hint: this method returns a GuardrailResult object

        CONTROL FLOW — early return pattern:
          If check 1 fails → return immediately (no point running checks 2 and 3).
          Check 2 never blocks — it always continues (only scrubs data).
          If check 3 fails → return immediately.
          If all checks pass → return the (possibly sanitized) result.
        """
        # Start with an optimistic "everything passed" result.
        # We'll flip passed=False if any check fails.
        result = GuardrailResult(passed=True, sanitized_message=message)

        # ── CHECK 1: Content policy ───────────────────────────────────────────
        # Blocks harmful requests like "how to make a bomb" before doing anything else.
        # Early exit = if this fails, we don't waste time on checks 2 and 3.
        passed, reason = check_content_policy(result.sanitized_message)
        # PYTHON CONCEPT — tuple unpacking:
        # check_content_policy() returns a tuple: (True, "") or (False, "reason string")
        # `passed, reason = ...` unpacks the two values into two variables in one line.
        # In JS/TS: const [passed, reason] = checkContentPolicy(message)

        if not passed:
            # `not passed` = True when passed=False (i.e., the check failed)
            result.passed = False
            result.reason = reason
            return result   # ← early return: stop here, don't run more checks

        # ── CHECK 2: PII redaction ────────────────────────────────────────────
        # Replace personal data with tokens like [EMAIL], [PHONE], [SSN].
        # This check NEVER blocks the request — it only cleans the message.
        # The cleaned version is what Claude will see.
        result.sanitized_message = redact_pii(result.sanitized_message)
        # Note: we update sanitized_message IN PLACE so check 3 runs on the clean version.

        # ── CHECK 3: Prompt injection detection ──────────────────────────────
        # Detects "ignore all previous instructions" style attacks.
        # These try to override Claude's system prompt and make it behave differently.
        passed, reason = check_injection(result.sanitized_message)

        if not passed:
            result.passed = False
            result.reason = reason
            return result   # ← early return: block the injected message

        # All checks passed — return the result with the sanitized message.
        # The caller (router) will replace request.message with result.sanitized_message.
        return result
