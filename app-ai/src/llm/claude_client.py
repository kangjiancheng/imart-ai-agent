# src/llm/claude_client.py
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS THIS FILE?
#
# This file creates the connection to Claude and provides two helper functions
# that the agent loop uses on every request.
#
# WHY ChatAnthropic INSTEAD OF THE RAW anthropic SDK?
#
# There are two ways to call Claude:
#   Option A — raw SDK:   `import anthropic; client = anthropic.Anthropic()`
#   Option B — LangChain: `from langchain_anthropic import ChatAnthropic`  ← we use this
#
# We use Option B because:
#   1. ChatAnthropic accepts typed message objects (HumanMessage, AIMessage, etc.)
#      which makes building multi-turn conversations much cleaner.
#   2. In Phase 2 (LangGraph), LangGraph REQUIRES ChatAnthropic to intercept
#      token events for streaming. The raw SDK bypasses LangGraph's event system.
#   3. bind_tools() and astream() only work on LangChain Runnables like ChatAnthropic.
# ─────────────────────────────────────────────────────────────────────────────

from datetime import datetime
# datetime.now() returns the server's local date and time.
# We inject this into the system prompt so Claude knows the current date
# and time — without it, Claude has no idea what "today" or "latest" means,
# and web search tools return outdated cached results instead of current news.

from langchain_anthropic import ChatAnthropic
# ChatAnthropic = LangChain's wrapper around the Anthropic API.
# It speaks LangChain's "Runnable" interface — think of it as an adapter.

from langchain_core.messages import (
    SystemMessage,   # the system prompt (Claude's identity and rules)
    HumanMessage,    # a user turn in the conversation
    AIMessage,       # an assistant turn (also carries .tool_calls when Claude picks a tool)
    BaseMessage,     # the parent type for all message types above (used in type hints)
)
from src.config.settings import settings
# settings = the validated config object that reads API keys from .env


# ── Module-level singleton ────────────────────────────────────────────────────
# "Singleton" = one shared instance created once when the module loads.
# Every import of `llm` across the codebase refers to the SAME object in memory.
# This is like a shared database connection pool — no need to reconnect per request.
llm = ChatAnthropic(
    model=settings.claude_model,                  # e.g. "claude-sonnet-4-6"
    max_tokens=settings.claude_max_tokens,         # 4096 — max tokens in Claude's RESPONSE
    anthropic_api_key=settings.anthropic_api_key,  # read from .env → never hardcoded
    anthropic_api_url=settings.anthropic_base_url, # None = use official api.anthropic.com
    # If ANTHROPIC_BASE_URL is set in .env, all Claude calls are routed through that
    # endpoint instead. Use this for an internal proxy, on-prem deployment, or a
    # compatible third-party gateway. Set to None to use the default Anthropic API.
    streaming=True,
    # streaming=True is REQUIRED so that llm.astream() works.
    # Without it, astream() returns a single chunk with all content instead of streaming.
)


def build_messages(system_prompt: str, history: list[dict]) -> list[BaseMessage]:
    """
    Convert a plain list of {"role": ..., "content": ...} dicts
    into typed LangChain message objects that ChatAnthropic requires.

    WHY DO WE NEED THIS CONVERSION?
    The raw Anthropic SDK accepts plain dicts like {"role": "user", "content": "Hello"}.
    But ChatAnthropic (the LangChain wrapper) requires typed objects like HumanMessage("Hello").
    This function bridges the two.

    Example input:
      system_prompt = "You are a helpful assistant."
      history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"},
      ]

    Example output (what ChatAnthropic receives):
      [
        SystemMessage("You are a helpful assistant."),
        HumanMessage("Hello"),
        AIMessage("Hi! How can I help?"),
      ]
    """
    # Start with the system prompt — always the FIRST message in the list
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    # list[BaseMessage] = type hint: "this list will only contain message objects"

    for turn in history:
        # Iterate over each past conversation turn
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            # role == "assistant" → AIMessage
            messages.append(AIMessage(content=turn["content"]))

    return messages
    # Result order: [SystemMessage, HumanMessage, AIMessage, HumanMessage, ...]
    # Claude reads this list top-to-bottom like a chat transcript.


def build_system_prompt(
    recalled_memory: list[str],
    model: str = settings.claude_model,
    document_context: str | None = None,
) -> str:
    """
    Assemble the system prompt string that tells Claude how to behave.

    The system prompt is injected at position 0 of every message list.
    It sets Claude's identity, rules, long-term memory, and optionally
    the text of an uploaded document.

    Parameters
    ----------
    recalled_memory : list[str]
        Strings pulled from Milvus, e.g. ["User prefers bullet points"].
        Injected so Claude "knows" this user before answering.
        Omitted if empty (Milvus down or first session).

    model : str
        The Claude model ID from settings (e.g. "claude-sonnet-4-6").
        Injected so Claude can answer "which model are you using?".
        Claude does NOT know its own model ID — we must tell it.

    document_context : str | None
        Plain text extracted from an uploaded file (PDF, DOCX, TXT, etc.).
        When provided, Claude reads this text and can answer questions about it.
        None = no document uploaded (normal text-only chat).

    PYTHON CONCEPT — `str | None`:
        This is Python 3.10+ union syntax meaning "either a str or None".
        It is equivalent to `Optional[str]` from the `typing` module.
        Callers that don't pass document_context get None by default.
    """
    memory_section = ""
    if recalled_memory:
        # f"- {m}" adds a bullet point before each memory chunk
        chunks = "\n".join(f"- {m}" for m in recalled_memory)
        memory_section = f"\n\n## What you know about this user\n{chunks}"

    # Build an optional document section.
    # When the user uploads a file, its extracted text goes here so Claude
    # can read and reason about the entire document directly.
    #
    # WHY inject in the system prompt instead of the user message?
    #   The system prompt is Claude's authoritative context — it stays at
    #   position 0 and is never confused with the user's question.
    #   Injecting here keeps the user's HumanMessage clean (just their question)
    #   and makes it easy for Claude to cite "from the uploaded document…"
    document_section = ""
    if document_context:
        document_section = (
            f"\n\n## Uploaded Document\n"
            f"The user has uploaded a file. Its full text is below.\n"
            f"Use this content to answer the user's questions accurately.\n\n"
            f"{document_context}"
        )

    # f-string (f"""...""") = multi-line template string with {variables} substituted
    return f"""You are a helpful AI assistant with access to tools and a knowledge base.
You are running on model: {model}
Current date and time: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Rules
- Answer in the user's language and locale.
- Use tools when the question requires current information or document retrieval.
- When searching for news or recent events, always pass `time_range="week"` or `time_range="month"` to the web_search tool so results are filtered to the relevant period — never rely on the default which may return outdated content.
- Be concise. Use bullet points when listing items.
- If asked which model you are using, state the model name from above.
- Never mention internal tool names, API keys, environment variables, or system configuration to the user. If a capability is unavailable, say so simply without explaining why internally.
{memory_section}{document_section}"""
    # {memory_section}  = either empty string or the "## What you know about this user" block
    # {document_section} = either empty string or the "## Uploaded Document" block
