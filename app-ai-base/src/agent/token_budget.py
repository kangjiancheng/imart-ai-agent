from dataclasses import dataclass

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage


@dataclass
class TokenBudget:
    """
    Tracks token usage per agent request and trims conversation history
    when the context window budget runs low.

    Claude's context limit is 200,000 tokens. We reserve 4,096 for the
    response and 2,000 as a safety margin, leaving 193,904 usable input tokens.
    trim_history() drops the oldest user/assistant turn blocks when approaching
    the limit, always preserving the SystemMessage.
    """

    MODEL_CONTEXT_LIMIT = 200_000
    RESPONSE_RESERVE    = 4_096
    SAFETY_MARGIN       = 2_000
    USABLE_LIMIT = MODEL_CONTEXT_LIMIT - RESPONSE_RESERVE - SAFETY_MARGIN  # 193,904

    used: int = 0

    def consume(self, tokens: int) -> None:
        """Accumulate token usage from response.usage_metadata["total_tokens"]."""
        self.used += tokens

    @property
    def remaining(self) -> int:
        """Tokens still available for input."""
        return self.USABLE_LIMIT - self.used

    @property
    def is_exhausted(self) -> bool:
        """True when 0 or fewer tokens remain."""
        return self.remaining <= 0

    def trim_history(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """
        Drop the oldest conversation turn block until the list fits the budget.

        Drops from the start of the non-system history in blocks (one full
        HumanMessage + its associated AIMessage/ToolMessage chain) to avoid
        leaving orphaned ToolMessages that would cause a 400 API error.
        SystemMessage is always preserved.
        """
        system  = [m for m in messages if isinstance(m, SystemMessage)]
        history = [m for m in messages if not isinstance(m, SystemMessage)]

        while history and self._estimate(system + history) > self.remaining:
            cut = 2
            for idx in range(1, len(history)):
                if not isinstance(history[idx], ToolMessage) and hasattr(history[idx], "type") and history[idx].type == "human":
                    cut = idx
                    break
            history = history[cut:]

        return system + history

    @staticmethod
    def _estimate(messages: list[BaseMessage]) -> int:
        """Rough token estimate: 1 token ≈ 4 characters."""
        return sum(
            len(m.content) // 4
            if isinstance(m.content, str) else 0
            for m in messages
        )
