from fastapi import APIRouter, HTTPException, Form, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
import json

from src.schemas.request import AgentRequest, HistoryMessage, UserContext
from src.guardrails.checker import GuardrailChecker
from src.agent.agent_loop import run
from src.utils.file_parser import extract_text

router = APIRouter(prefix="/v1")
guardrails = GuardrailChecker()


@router.post("/agent/chat")
async def agent_chat(request: AgentRequest):
    """Main streaming endpoint. Returns tokens as Server-Sent Events (SSE)."""
    guard_result = guardrails.check(request.message)
    if not guard_result.passed:
        raise HTTPException(status_code=422, detail=guard_result.reason)

    request.message = guard_result.sanitized_message

    if request.stream:
        async def token_stream():
            try:
                async for token in run(request):
                    if token:
                        yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
                return
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            token_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    chunks: list[str] = []
    async for token in run(request):
        if token:
            chunks.append(token)
    return _UTF8JSONResponse(content={"type": "complete", "content": "".join(chunks)})


import json as _json

class _UTF8JSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return _json.dumps(content, ensure_ascii=False).encode("utf-8")


@router.post("/agent/chat-with-file")
async def agent_chat_with_file(
    file: UploadFile = File(...),
    message: str = Form(...),
    user_id: str = Form(...),
    session_id: str = Form(...),
    subscription_tier: str = Form(...),
    locale: str = Form("en-US"),
    timezone: str = Form("UTC"),
    stream: bool = Form(True),
    history_json: str = Form("[]"),
):
    """Upload a document and ask a question about it in one request."""
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

    if agent_request.stream:
        async def token_stream():
            try:
                async for token in run(agent_request):
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
    async for token in run(agent_request):
        if token:
            chunks.append(token)
    return _UTF8JSONResponse(content={"type": "complete", "content": "".join(chunks)})
