from dataclasses import dataclass

from src.guardrails.content_policy import check_content_policy
from src.guardrails.pii_filter import redact_pii
from src.guardrails.injection_detector import check_injection


@dataclass
class GuardrailResult:
    """Result from GuardrailChecker.check()."""
    passed: bool
    sanitized_message: str
    reason: str = ""


class GuardrailChecker:
    """Orchestrates three guardrail checks: content policy → PII redaction → injection detection."""

    def check(self, message: str) -> GuardrailResult:
        """Run all safety checks. Returns early on first failure."""
        result = GuardrailResult(passed=True, sanitized_message=message)

        passed, reason = check_content_policy(result.sanitized_message)
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
