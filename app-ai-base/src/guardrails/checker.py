from dataclasses import dataclass

from src.guardrails.content_policy import check_content_policy
from src.guardrails.pii_filter import redact_pii
from src.guardrails.injection_detector import check_injection


@dataclass
class GuardrailResult:
    """
    Return value from GuardrailChecker.check().

    passed           — True if all checks passed; False if blocked.
    sanitized_message — message after PII redaction (may differ from input).
    reason           — human-readable block reason (empty string if passed).
    """
    passed: bool
    sanitized_message: str
    reason: str = ""


class GuardrailChecker:
    """
    Orchestrates three safety checks in sequence before Claude sees any input:
      1. Content policy  — blocks harmful requests (violence, illegal instructions).
      2. PII redaction   — replaces personal data with tokens ([EMAIL], [PHONE], etc.).
      3. Injection check — detects prompt-override attacks.

    Instantiated once as a module-level singleton in the router.
    All checks are regex-based and complete in < 1 ms.
    """

    def check(self, message: str) -> GuardrailResult:
        """Run all three checks. Returns early on first failure."""
        result = GuardrailResult(passed=True, sanitized_message=message)

        passed, reason = check_content_policy(result.sanitized_message)
        print(f"Content policy check: passed={passed}, reason='{reason}'")
        if not passed:
            result.passed = False
            result.reason = reason
            return result

        result.sanitized_message = redact_pii(result.sanitized_message)

        passed, reason = check_injection(result.sanitized_message)
        if not passed:
            result.passed = False
            result.reason = reason
            return result

        return result
