from dataclasses import dataclass
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage


@dataclass
class TokenBudget:
    """Tracks token usage and trims history when the context window runs low."""

    MODEL_CONTEXT_LIMIT = 200_000
    RESPONSE_RESERVE = 4_096
    SAFETY_MARGIN = 2_000
    USABLE_LIMIT = MODEL_CONTEXT_LIMIT - RESPONSE_RESERVE - SAFETY_MARGIN  # 193,904

    used: int = 0

    def consume(self, tokens: int) -> None:
        self.used += tokens

    @property
    def remaining(self) -> int:
        return self.USABLE_LIMIT - self.used

    @property
    def is_exhausted(self) -> bool:
        return self.remaining <= 0

    def trim_history(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Drop oldest conversation turns to stay within the token budget.

        Always preserves the SystemMessage. Drops in whole turn blocks to avoid
        leaving orphaned ToolMessages (which would cause Anthropic API errors).
        """
        system = [m for m in messages if isinstance(m, SystemMessage)]
        history = [m for m in messages if not isinstance(m, SystemMessage)]

        while history and self._estimate(system + history) > self.remaining:
            # Find the next HumanMessage to identify a safe cut point
            cut = 2
            for idx in range(1, len(history)):
                if not isinstance(history[idx], ToolMessage) and hasattr(history[idx], "type") and history[idx].type == "human":
                    cut = idx
                    break
            history = history[cut:]

        return system + history

    @staticmethod
    def _estimate(messages: list[BaseMessage]) -> int:
        """Rough token estimate using 1 token ≈ 4 characters."""
        return sum(
            len(m.content) // 4
            if isinstance(m.content, str) else 0
            for m in messages
        )
