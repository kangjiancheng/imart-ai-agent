import json as _json

from fastapi import APIRouter, HTTPException, Form, UploadFile, File, Header
from fastapi.responses import StreamingResponse, JSONResponse

from src.schemas.request import AgentRequest, HistoryMessage, UserContext
from src.guardrails.checker import GuardrailChecker
from src.agent.agent_loop import run
from src.utils.file_parser import extract_text
from src.config.settings import settings

router     = APIRouter(prefix="/v1")
guardrails = GuardrailChecker()


class _UTF8JSONResponse(JSONResponse):
    """JSONResponse that outputs Unicode characters as-is instead of \\uXXXX escapes."""
    def render(self, content) -> bytes:
        return _json.dumps(content, ensure_ascii=False).encode("utf-8")


def _resolve_llm_headers(
    x_ai_api_key: str | None,
    x_ai_model: str | None,
    x_ai_base_url: str | None,
) -> tuple[str | None, str | None, str | None]:
    """
    Determine effective LLM credentials for this request.

    Rules:
    - If enable_internal_llm=True and no client headers → use server singleton (return None, None, None).
    - If enable_internal_llm=True and client headers present → override with header values.
    - If enable_internal_llm=False → client must supply X-Ai-Api-Key; raise 401 if missing.
    """
    if not settings.enable_internal_llm:
        if not x_ai_api_key:
            raise HTTPException(
                status_code=401,
                detail="Server LLM is disabled. Provide X-Ai-Api-Key in request headers.",
            )
        return x_ai_api_key, x_ai_model or None, x_ai_base_url or None

    # Internal LLM enabled — use header values only when supplied.
    if x_ai_api_key:
        return x_ai_api_key, x_ai_model or None, x_ai_base_url or None

    return None, None, None


@router.post("/agent/chat")
async def agent_chat(
    request: AgentRequest,
    x_ai_api_key:  str | None = Header(default=None),
    x_ai_model:    str | None = Header(default=None),
    x_ai_base_url: str | None = Header(default=None),
):
    """
    Main streaming endpoint. Returns tokens as Server-Sent Events (SSE).

    Flow:
      1. Run guardrails (content policy → PII redaction → injection detection).
      2. Resolve LLM credentials from X-Ai-* headers or server config.
      3. If stream=True  → return StreamingResponse with typed SSE events.
      4. If stream=False → collect all tokens, return single JSON body.

    SSE event types:
      {"type": "token",    "content": "..."}  — one streamed token
      {"type": "done"}                         — stream complete
      {"type": "error",   "message": "..."}   — agent error
      {"type": "complete","content": "..."}   — non-streaming full response
    """
    guard_result = guardrails.check(request.message)
    if not guard_result.passed:
        raise HTTPException(status_code=422, detail=guard_result.reason)

    request.message = guard_result.sanitized_message

    api_key, model, base_url = _resolve_llm_headers(x_ai_api_key, x_ai_model, x_ai_base_url)

    if request.stream:
        async def token_stream():
            try:
                async for token in run(request, llm_api_key=api_key, llm_model=model, llm_base_url=base_url):
                    if token:
                        yield f"data: {_json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
            except Exception as exc:
                yield f"data: {_json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
                return
            yield f"data: {_json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            token_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    chunks: list[str] = []
    async for token in run(request, llm_api_key=api_key, llm_model=model, llm_base_url=base_url):
        if token:
            chunks.append(token)
    return _UTF8JSONResponse(content={"type": "complete", "content": "".join(chunks)})


@router.post("/agent/chat-with-file")
async def agent_chat_with_file(
    file:              UploadFile = File(...),
    message:           str        = Form(...),
    user_id:           str        = Form(...),
    session_id:        str        = Form(...),
    subscription_tier: str        = Form(...),
    locale:            str        = Form("en-US"),
    timezone:          str        = Form("UTC"),
    stream:            bool       = Form(True),
    history_json:      str        = Form("[]"),
    x_ai_api_key:      str | None = Header(default=None),
    x_ai_model:        str | None = Header(default=None),
    x_ai_base_url:     str | None = Header(default=None),
):
    """
    Upload a document and ask a question about it in one multipart request.

    Extracts plain text from the file (PDF/DOCX/TXT via file_parser.py),
    injects it into Claude's context window as "## Uploaded Document",
    then runs the same agent loop and streaming logic as /v1/agent/chat.
    Suitable for files up to ~80k characters; use RAG ingestion for larger docs.

    Fields:
      file              — document file (PDF, DOCX, TXT, …)
      message           — user question or instruction
      user_id           — user identifier for memory recall
      session_id        — session identifier for SSE correlation
      subscription_tier — e.g. "free", "pro", "enterprise"
      locale            — optional, default "en-US"
      timezone          — optional, default "UTC"
      stream            — optional, default true
      history_json      — JSON-encoded array of past messages, optional
    """
    import json as _json_mod

    file_bytes = await file.read()
    try:
        document_text = extract_text(file.filename or "upload", file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        raw_history: list[dict] = _json_mod.loads(history_json)
        history = [HistoryMessage(**h) for h in raw_history]
    except Exception:
        history = []

    agent_request = AgentRequest(
        user_id=user_id,
        message=message,
        history=history,
        user_context=UserContext(
            subscription_tier=subscription_tier,
            locale=locale,
            timezone=timezone,
        ),
        session_id=session_id,
        stream=stream,
        document_context=document_text,
    )

    guard_result = guardrails.check(agent_request.message)
    if not guard_result.passed:
        raise HTTPException(status_code=422, detail=guard_result.reason)
    agent_request.message = guard_result.sanitized_message

    api_key, model, base_url = _resolve_llm_headers(x_ai_api_key, x_ai_model, x_ai_base_url)

    if agent_request.stream:
        async def token_stream():
            try:
                async for token in run(agent_request, llm_api_key=api_key, llm_model=model, llm_base_url=base_url):
                    if token:
                        yield f"data: {_json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
            except Exception as exc:
                yield f"data: {_json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
                return
            yield f"data: {_json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            token_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    chunks: list[str] = []
    async for token in run(agent_request, llm_api_key=api_key, llm_model=model, llm_base_url=base_url):
        if token:
            chunks.append(token)
    return _UTF8JSONResponse(content={"type": "complete", "content": "".join(chunks)})
