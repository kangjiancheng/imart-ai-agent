from datetime import datetime

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage,
)
from src.config.settings import settings

# Module-level singleton — one shared ChatAnthropic connection reused per request.
# streaming=True is required for llm.astream() to yield tokens incrementally.
llm = ChatAnthropic(
    model=settings.claude_model,
    max_tokens=settings.claude_max_tokens,
    anthropic_api_key=settings.anthropic_api_key,
    anthropic_api_url=settings.anthropic_base_url,
    streaming=True,
)

def build_llm(
    api_key: str,
    model: str | None = None,
    base_url: str | None = None,
) -> ChatAnthropic:
    """
    Build a per-request ChatAnthropic client using caller-supplied credentials.

    Used when client headers (X-Ai-Api-Key / X-Ai-Model / X-Ai-Base-Url) override
    the server-configured singleton.  Falls back to settings values for any
    parameter not explicitly provided.
    """
    return ChatAnthropic(
        model=model or settings.claude_model,
        max_tokens=settings.claude_max_tokens,
        anthropic_api_key=api_key,
        anthropic_api_url=base_url or settings.anthropic_base_url,
        streaming=True,
    )


def build_messages(system_prompt: str, history: list[dict]) -> list[BaseMessage]:
    """
    Convert plain {"role", "content"} dicts into typed LangChain message objects.

    ChatAnthropic requires typed objects (HumanMessage, AIMessage, etc.)
    rather than raw dicts. Prepends SystemMessage at index 0.
    """
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    return messages


def build_system_prompt(
    recalled_memory: list[str],
    model: str = settings.claude_model,
    document_context: str | None = None,
) -> str:
    """
    Assemble the system prompt injected at position 0 of every message list.

    Parameters
    ----------
    recalled_memory : list[str]
        Personal facts pulled from Milvus user_memory collection.
        Injected so Claude has cross-session context about this user.
    model : str
        Claude model ID — injected so Claude can answer "which model are you?".
    document_context : str | None
        Plain text from an uploaded file. When provided, injected under
        "## Uploaded Document" so Claude can read and reason about it.
    """
    memory_section = ""
    if recalled_memory:
        chunks = "\n".join(f"- {m}" for m in recalled_memory)
        memory_section = f"\n\n## What you know about this user\n{chunks}"

    document_section = ""
    if document_context:
        document_section = (
            f"\n\n## Uploaded Document\n"
            f"The user has uploaded a file. Its full text is below.\n"
            f"Use this content to answer the user's questions accurately.\n\n"
            f"{document_context}"
        )

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
