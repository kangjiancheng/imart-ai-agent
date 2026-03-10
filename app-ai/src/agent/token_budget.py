# src/agent/token_budget.py
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS A "TOKEN BUDGET"?
#
# Claude can only process a limited amount of text per conversation.
# That limit is called the "context window" = 200,000 tokens for claude-sonnet-4-6.
#
# A "token" ≈ 1 word or subword. "Hello" = 1 token. "Unbelievable" = 3 tokens.
# A rough rule: 1 token ≈ 4 English characters.
#
# PROBLEM: In a ReAct loop, the message list GROWS every iteration:
#   - Start:        SystemMessage + HumanMessage
#   - After iter 1: + AIMessage(tool_call) + ToolMessage(result)
#   - After iter 2: + AIMessage(tool_call) + ToolMessage(result)
#   - ... keeps growing until we hit 200,000 tokens → API error
#
# SOLUTION: Track how many tokens we've used, and TRIM old messages when running low.
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass
# @dataclass = a decorator that auto-generates __init__, __repr__, __eq__ for a class.
# Instead of writing def __init__(self, used=0): self.used = used
# we just write:  used: int = 0  inside the class body.
# It's a shortcut that reduces boilerplate — similar to TypeScript's class field syntax.

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
# BaseMessage = the parent type for all LangChain message objects
# SystemMessage = the system prompt — we NEVER trim this, it always stays


@dataclass
class TokenBudget:
    """
    Tracks token usage for one agent request and trims history when the budget runs low.

    PYTHON CONCEPT — @dataclass:
    This decorator auto-generates an __init__ method.
    `TokenBudget()` creates an instance with used=0.
    `TokenBudget(used=5000)` creates one with used=5000.
    """

    # ── Class-level constants ─────────────────────────────────────────────────
    # These are defined at CLASS level (not inside __init__), so they are SHARED
    # across all instances and never change. In Python, all-caps = "treat as constant".
    MODEL_CONTEXT_LIMIT = 200_000    # claude-sonnet-4-6 can process up to 200,000 tokens
    # Python allows _ in numbers for readability: 200_000 == 200000

    RESPONSE_RESERVE    = 4_096
    # We reserve 4,096 tokens for Claude's OUTPUT (response).
    # If we fill the entire 200k with input, Claude has no room to write an answer.

    SAFETY_MARGIN       = 2_000
    # An extra buffer because our token COUNT is only an estimate (1 token ≈ 4 chars).
    # The real count may be slightly higher — this gap prevents going over the limit.

    USABLE_LIMIT = MODEL_CONTEXT_LIMIT - RESPONSE_RESERVE - SAFETY_MARGIN
    # = 200,000 - 4,096 - 2,000 = 193,904
    # This is how many tokens our INPUT (messages) can safely use.
    # This expression is evaluated ONCE when the class is defined, not per-instance.

    # ── Per-request mutable field ─────────────────────────────────────────────
    used: int = 0
    # The only field that changes per request. Starts at 0, grows as we consume tokens.
    # @dataclass generates: def __init__(self, used: int = 0): self.used = used

    def consume(self, tokens: int) -> None:
        """
        Add `tokens` to the running total.
        Called after each planner.ainvoke() using the actual token count
        returned in response.usage_metadata["total_tokens"].

        `-> None` = this function returns nothing (just updates state).
        """
        self.used += tokens
        # += is shorthand for: self.used = self.used + tokens

    @property
    def remaining(self) -> int:
        """
        How many tokens we can still use.

        PYTHON CONCEPT — @property:
        `@property` lets you call this like an ATTRIBUTE, not a method.
        Instead of: budget.remaining()   ← method call with ()
        You write:  budget.remaining     ← looks like a plain attribute
        It feels cleaner and is a common Python pattern.
        """
        return self.USABLE_LIMIT - self.used

    @property
    def is_exhausted(self) -> bool:
        """Returns True if we have 0 or fewer tokens remaining."""
        return self.remaining <= 0

    def trim_history(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """
        Drop the oldest conversation turns until the message list fits within budget.

        HOW IT WORKS:
          1. Split messages into: system (never drop) + history (can drop)
          2. While the total is over budget, drop the OLDEST pair (user + assistant)
          3. Reassemble and return

        WHY DROP IN PAIRS?
          A valid conversation must alternate: user → assistant → user → assistant.
          If we drop just one turn, we'd have two consecutive user turns → API error.
          So we always drop 2 at a time (one user + one assistant).

        Example:
          Before trim:  [System, Human_1, AI_1, Human_2, AI_2, Human_3]
          After trim:   [System, Human_2, AI_2, Human_3]   ← oldest pair dropped
        """
        # List comprehension = a compact way to build a new list from an existing one.
        # [m for m in messages if isinstance(m, SystemMessage)]
        # = "for each m in messages, include it IF it is a SystemMessage"
        # In JS/TS this would be: messages.filter(m => m instanceof SystemMessage)
        system  = [m for m in messages if isinstance(m, SystemMessage)]
        history = [m for m in messages if not isinstance(m, SystemMessage)]

        # `while` = keep looping as long as this condition is True
        while history and self._estimate(system + history) > self.remaining:
            # `history and ...` = short-circuit: if history is empty, skip the estimate
            #
            # ReAct loop message pattern is NOT simple user/assistant pairs.
            # It can be: [HumanMessage, AIMessage(tool_calls), ToolMessage, ...]
            # We must drop a safe "block" to avoid leaving orphaned ToolMessages.
            #
            # Strategy: find the first HumanMessage after index 0, drop everything
            # before it (one full "user turn + all its tool iterations").
            # Fallback: drop 2 at a time (classic user+assistant pairs).
            cut = 2  # default: drop first 2
            for idx in range(1, len(history)):
                # Find the next HumanMessage — it marks the start of a new turn.
                if not isinstance(history[idx], ToolMessage) and hasattr(history[idx], "type") and history[idx].type == "human":
                    cut = idx
                    break
            history = history[cut:]
            # history[cut:] = "slice from index cut to the end" = drop the first `cut` items
            # In JS/TS: history.slice(cut)

        return system + history
        # + on lists = concatenate: [SystemMessage] + [Human, AI, ...] → one combined list

    @staticmethod
    def _estimate(messages: list[BaseMessage]) -> int:
        """
        Rough token count estimate using the 1-token-per-4-chars rule.

        PYTHON CONCEPT — @staticmethod:
        A @staticmethod belongs to the class but doesn't receive `self`.
        It's just a utility function that lives in the class namespace.
        Call it as: TokenBudget._estimate(messages)

        PYTHON CONCEPT — generator expression inside sum():
        sum(
            len(m.content) // 4        ← value to add for each m
            if isinstance(m.content, str) else 0    ← skip non-string content
            for m in messages          ← loop over every message
        )
        This is equivalent to:
          total = 0
          for m in messages:
              if isinstance(m.content, str):
                  total += len(m.content) // 4
          return total

        `//` = integer division: 20 // 4 = 5 (drops the remainder)
        """
        return sum(
            len(m.content) // 4
            if isinstance(m.content, str) else 0
            for m in messages
        )
