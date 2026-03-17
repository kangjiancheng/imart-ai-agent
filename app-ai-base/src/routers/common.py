from fastapi import APIRouter
from pydantic import BaseModel

from src.config.settings import settings

router = APIRouter()


@router.get("/health")
async def health():
    """Liveness probe for load balancers and deployment platforms."""
    return {"status": "ok", "service": "app-ai-base"}


@router.get("/info")
async def info():
    """Returns current model configuration for debugging."""
    return {
        "service": "app-ai-base",
        "claude_model": settings.claude_model,
        "claude_max_tokens": settings.claude_max_tokens,
        "embedding_model": "BAAI/bge-m3",
    }


class TestLLMRequest(BaseModel):
    provider: str               # "anthropic" | "openai" | "google" | "openai-compatible"
    api_key: str
    model: str
    base_url: str | None = None  # None → use the provider's default endpoint


@router.post("/test-llm")
async def test_llm(body: TestLLMRequest):
    """
    Test whether a given provider + model + API key combination is reachable.
    Called by the web settings UI when the user clicks "test connection".

    Sends a minimal one-token ping and returns ok/error so the UI can show
    a success or failure state without committing to saving the settings.
    """
    from langchain_core.messages import HumanMessage

    ping = [HumanMessage(content="Reply with one word: OK")]

    try:
        provider = body.provider.lower()

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            client = ChatAnthropic(
                model=body.model,
                max_tokens=16,
                anthropic_api_key=body.api_key,
                anthropic_api_url=body.base_url or None,
                streaming=False,
            )

        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            client = ChatOpenAI(
                model=body.model,
                max_tokens=16,
                api_key=body.api_key,
                base_url=body.base_url or None,
            )

        elif provider in ("openai-compatible", "openai_compatible"):
            from langchain_openai import ChatOpenAI
            if not body.base_url:
                return {"status": "error", "provider": provider, "model": body.model,
                        "error": "base_url is required for OpenAI-compatible providers."}
            client = ChatOpenAI(
                model=body.model,
                max_tokens=16,
                api_key=body.api_key,
                base_url=body.base_url,
            )

        elif provider in ("google", "gemini", "google_gemini"):
            from langchain_google_genai import ChatGoogleGenerativeAI
            client = ChatGoogleGenerativeAI(
                model=body.model,
                max_output_tokens=16,
                google_api_key=body.api_key,
            )

        else:
            return {
                "status": "error",
                "provider": provider,
                "model": body.model,
                "error": f"Unsupported provider '{body.provider}'. "
                         f"Supported: anthropic, openai, openai-compatible, google.",
            }

        response = await client.ainvoke(ping)
        content  = response.content
        if isinstance(content, list):
            content = "".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ).strip()

        return {"status": "ok", "provider": provider, "model": body.model, "response": content}

    except Exception as exc:
        return {"status": "error", "provider": body.provider, "model": body.model, "error": str(exc)}
