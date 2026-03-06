from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.agent.token_budget import TokenBudget


def test_usable_limit_is_193904():
    # Matches the architecture document's computed constant
    budget = TokenBudget()
    assert budget.USABLE_LIMIT == 193_904


def test_trim_history_keeps_system_message():
    budget = TokenBudget(used=193_000)  # nearly exhausted
    messages = [
        SystemMessage(content="System prompt"),
        HumanMessage(content="First user message"),
        AIMessage(content="First assistant reply"),
        HumanMessage(content="Second user message"),
    ]
    trimmed = budget.trim_history(messages)
    # System message must always be kept
    assert any(isinstance(m, SystemMessage) for m in trimmed)


def test_is_exhausted_when_over_limit():
    budget = TokenBudget()
    budget.consume(200_000)
    assert budget.is_exhausted is True


def test_consume_accumulates():
    budget = TokenBudget()
    budget.consume(1_000)
    budget.consume(500)
    assert budget.used == 1_500


def test_remaining_decreases_with_consumption():
    budget = TokenBudget()
    initial_remaining = budget.remaining
    budget.consume(10_000)
    assert budget.remaining == initial_remaining - 10_000
