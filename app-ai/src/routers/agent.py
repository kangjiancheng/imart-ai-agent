# src/routers/agent.py
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS A ROUTER?
#
# A FastAPI "router" is a mini-application that groups related HTTP endpoints.
# Routers help organize code — instead of defining all routes in main.py,
# each feature gets its own router file.
#
# This file defines the main AI agent endpoint: POST /v1/agent/chat
#
# REQUEST FLOW — what happens when NestJS calls this endpoint:
#   NestJS → POST /v1/agent/chat (JSON body) →
#   → agent_chat() runs guardrails first →
#   → if blocked: return 422 HTTP error immediately →
#   → if passed: start streaming tokens via SSE →
#   → NestJS reads "data: token\n\n" events →
#   → NestJS reads "data: [DONE]\n\n" → stops reading
#
# SSE (Server-Sent Events):
#   A text streaming protocol where the server pushes lines to the client:
#     data: Hello\n\n
#     data:  world\n\n
#     data: [DONE]\n\n
#   The client reads each "data:" line as it arrives — no buffering.
#   Used here so the user sees tokens appear word-by-word in real time.
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import APIRouter, HTTPException
# APIRouter = creates a group of routes (like Express Router in Node.js)
# HTTPException = raises an HTTP error response with a status code and message

from fastapi import Form, UploadFile, File, Header
# Form      = marks a multipart form field (for /v1/agent/chat-with-file)
# UploadFile = FastAPI's type for files uploaded via multipart/form-data
# File(...)  = marks a form field as a required file upload
# Header     = FastAPI dependency for reading HTTP request headers

from fastapi.responses import StreamingResponse, JSONResponse
# StreamingResponse = a FastAPI response that streams data in chunks.
# Instead of returning one big JSON body, it sends pieces one at a time.
# The client receives each piece immediately as it's yielded.
#
# JSONResponse = a standard one-shot JSON response (used when stream=False).
# FastAPI returns the full body at once — no chunked transfer encoding.

import json
# json = Python's built-in module for encoding/decoding JSON.
# json.dumps({"type": "token", "content": "Hello"}) → '{"type": "token", "content": "Hello"}'
# Used here to format each SSE event as a typed JSON object.

from src.schemas.request import AgentRequest
# AgentRequest = Pydantic model for the incoming JSON body.
# FastAPI uses it to validate the request automatically.

from src.guardrails.checker import GuardrailChecker
# GuardrailChecker = orchestrates all 3 guardrail checks (content, PII, injection).
# See guardrails/checker.py for full explanation.

from src.agent.agent_loop import run
# run = the main async generator function that runs the ReAct agent loop.
# It yields string tokens one-by-one as Claude generates the response.
# See agent/agent_loop.py for full explanation.

from src.schemas.request import AgentRequest, HistoryMessage, UserContext
# AgentRequest  = Pydantic model for the incoming JSON body (text-only chat).
# HistoryMessage, UserContext = sub-schemas needed to build AgentRequest manually
#   for the multipart form endpoint (chat-with-file).

from src.utils.file_parser import extract_text
# extract_text(filename, bytes) → plain text from PDF / DOCX / TXT.
# Used by /v1/agent/chat-with-file to turn the uploaded file into a string
# that is injected into Claude's context window.

from src.config.settings import settings
# settings = validated config (API keys, model name, enable_internal_llm toggle, etc.)


# ── Module-level singletons ───────────────────────────────────────────────────
# PYTHON CONCEPT — module-level instantiation (singleton pattern):
#   `router = APIRouter(prefix="/v1")` runs ONCE when this module is imported.
#   All routes defined on `router` will be accessible under "/v1/..." prefix.
#   In TypeScript: const router = express.Router() or new Hono()
#
#   `guardrails = GuardrailChecker()` also runs ONCE at module import.
#   This avoids re-creating the guardrail checker on every request.
#   The checker compiles all regex patterns once and reuses them for every request.

router = APIRouter(prefix="/v1")
# prefix="/v1" means all routes in this file are under /v1/...
# @router.post("/agent/chat") → accessible at POST /v1/agent/chat

guardrails = GuardrailChecker()
# A single shared instance of the guardrail checker.
# Thread-safe because the checker doesn't store request-specific state.


def _resolve_llm_headers(
    x_ai_api_key: str | None,
    x_ai_model: str | None,
    x_ai_base_url: str | None,
) -> tuple[str | None, str | None, str | None]:
    """
    Determine the effective LLM credentials for this request.

    Rules:
    - enable_internal_llm=True + no headers  → use server singleton (return None, None, None).
    - enable_internal_llm=True + headers     → override with header values.
    - enable_internal_llm=False + no headers → raise 401; client must supply X-Ai-Api-Key.
    - enable_internal_llm=False + headers    → use header values.
    """
    if not settings.enable_internal_llm:
        if not x_ai_api_key:
            raise HTTPException(
                status_code=401,
                detail="Server LLM is disabled. Provide X-Ai-Api-Key in request headers.",
            )
        return x_ai_api_key, x_ai_model or None, x_ai_base_url or None

    # Internal LLM enabled — use header values only when the client supplies them.
    if x_ai_api_key:
        return x_ai_api_key, x_ai_model or None, x_ai_base_url or None

    return None, None, None


# ── POST /v1/agent/chat ────────────────────────────────────────────────────────
# The main agent streaming endpoint.

@router.post("/agent/chat")
async def agent_chat(
    request: AgentRequest,
    x_ai_api_key:  str | None = Header(default=None),
    x_ai_model:    str | None = Header(default=None),
    x_ai_base_url: str | None = Header(default=None),
):
    """
    Main streaming endpoint. Returns tokens as Server-Sent Events (SSE).

    The browser (via EventSource) or NestJS (via HTTP streaming) reads:
        data: Hello
        data:  world
        data: [DONE]

    PYTHON CONCEPT — async def:
      `async def agent_chat(...)` is an async function (coroutine).
      FastAPI runs it in an async event loop — it can await I/O without
      blocking other requests. Required because run() is also async.
      In TypeScript: async agentChat(request: AgentRequest): Promise<...>

    HOW FASTAPI READS THE REQUEST BODY:
      FastAPI sees `request: AgentRequest` type hint and:
        1. Reads the HTTP request body as JSON
        2. Validates it against AgentRequest (Pydantic model)
        3. Passes a real AgentRequest object to this function
        4. If validation fails → returns 422 automatically (no code needed)
    """
    # ── Step 1: Run all 3 guardrail checks ──────────────────────────────────
    guard_result = guardrails.check(request.message)
    # guardrails.check() runs content policy → PII redaction → injection detection.
    # Returns a GuardrailResult dataclass with:
    #   .passed: bool               → True if all checks passed
    #   .reason: str                → why it was blocked (empty if passed)
    #   .sanitized_message: str     → message with PII replaced (e.g. [EMAIL])
    # This is synchronous (no await) — all guardrail checks are regex-based, < 1ms.

    if not guard_result.passed:
        # PYTHON CONCEPT — raise:
        #   `raise` in Python = `throw` in JavaScript/TypeScript.
        #   `raise HTTPException(...)` immediately stops this function and
        #   tells FastAPI to return an HTTP error response.
        #
        # HTTPException(status_code=422, detail=...)
        #   status_code=422 = "Unprocessable Entity" — request was syntactically
        #   valid JSON but semantically blocked by our guardrails.
        #   detail=guard_result.reason = the human-readable reason for blocking.
        #   FastAPI converts this to: {"detail": "Request blocked by content policy."}
        raise HTTPException(status_code=422, detail=guard_result.reason)

    # ── Step 2: Replace raw message with PII-sanitized version ──────────────
    request.message = guard_result.sanitized_message
    # guard_result.sanitized_message has PII replaced (emails → [EMAIL], etc.).
    # We update request.message so run() never sees raw PII.
    # GDPR/privacy compliance: Claude's context contains sanitized text only.

    # ── Step 3: Resolve LLM credentials from headers or server config ────────
    api_key, model, base_url = _resolve_llm_headers(x_ai_api_key, x_ai_model, x_ai_base_url)
    # If the client sent X-Ai-Api-Key / X-Ai-Model / X-Ai-Base-Url headers,
    # those values override the server's configured Claude singleton for this request.
    # If enable_internal_llm=False and no headers were sent, raises HTTP 401.

    # ── Step 4: Branch on stream flag ────────────────────────────────────────
    if request.stream:
        # ── Step 3a: Define the SSE token streaming generator ───────────────
        async def token_stream():
            # PYTHON CONCEPT — nested async function (inner function / closure):
            #   `async def token_stream()` is defined INSIDE agent_chat().
            #   It's a "closure" — it captures `request` from the outer function's scope.
            #   This is similar to defining an arrow function inside another function in JS:
            #     const tokenStream = async () => { ... /* uses request from outer scope */ }
            #
            # PYTHON CONCEPT — async generator function:
            #   Because it contains `yield`, token_stream is an "async generator".
            #   Each `yield` sends one SSE event string to the client immediately.
            #   The function pauses at each yield and resumes when the client is ready.
            #   In TypeScript: async function* tokenStream() { yield ...; }

            try:
                async for token in run(request, llm_api_key=api_key, llm_model=model, llm_base_url=base_url):
                    # run(request) is the agent loop — an async generator from agent_loop.py.
                    # `async for token in run(request)` iterates over yielded tokens.
                    # Each `token` is a string (one word or piece of a word from Claude).
                    # `async for` is required because run() is async — it awaits I/O internally.
                    # In TypeScript: for await (const token of run(request))

                    if token:
                        # Skip empty strings (Claude's streaming API sometimes yields "")
                        # ensure_ascii=False → Chinese/Unicode characters are sent as-is
                        # instead of being escaped to \uXXXX sequences.
                        yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
                        # SSE event format: "data: <JSON>\n\n"
                        # Typed JSON format so NestJS can distinguish tokens from control events:
                        #   {"type": "token", "content": "Hello"}  ← text to display
                        #   {"type": "done"}                       ← stream finished
                        #   {"type": "error", "message": "..."}    ← something went wrong
                        # NestJS reads the "type" field to decide what to do with each event.

            except Exception as exc:
                # If the agent loop raises an error (e.g. Anthropic 503 no capacity),
                # catch it here and send a typed "error" SSE event to the client.
                # The client (NestJS) reads type == "error", shows the message to the user,
                # and ends the current conversation so a new chat can be started.
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
                return
                # `return` inside an async generator stops the stream immediately.
                # Without it, the "done" event below would also be sent after the error,
                # which would confuse the client into thinking the stream ended normally.

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            # After all tokens are sent, yield the "done" sentinel event.
            # NestJS (the client) watches for type == "done" to know the stream has ended.
            # Without this, the client wouldn't know when to stop reading.

        # ── Step 3b: Return a StreamingResponse wrapping the generator ───────
        return StreamingResponse(
            token_stream(),
            # token_stream() = calling the async generator function returns a generator object.
            # StreamingResponse iterates over it, sending each yielded string to the client.

            media_type="text/event-stream",
            # Tells the HTTP client this is an SSE stream, not JSON or HTML.
            # The browser's EventSource API and most HTTP clients handle this automatically.

            headers={
                "Cache-Control": "no-cache",
                # Prevents any proxy or CDN from buffering the response.
                # Without this, the client might not receive tokens until the entire
                # response is buffered somewhere in the network.

                "X-Accel-Buffering": "no",
                # Nginx-specific header: tells Nginx to disable proxy_buffering for this route.
                # Without this, Nginx would wait for the entire stream before forwarding it.
                # Essential for real-time streaming through an Nginx reverse proxy.
            },
        )

    # ── Step 3c (stream=False): Collect all tokens, return single JSON ────────
    # Instead of yielding tokens one-by-one, we accumulate them into a string,
    # then return the full answer in one JSON body.
    #
    # WHY COLLECT THEN RETURN?
    #   The client (e.g. a NestJS background job or a curl command) may not have
    #   an SSE parser. They just want one clean JSON response they can read at once.
    #   We still run the exact same agent loop — only the delivery format changes.

    chunks: list[str] = []
    # chunks = every token yielded by run() will be appended here.
    # list[str] type hint = TypeScript equivalent: string[]

    async for token in run(request, llm_api_key=api_key, llm_model=model, llm_base_url=base_url):
        # Same loop as in the streaming case — run() is unchanged.
        # `async for` iterates the async generator and awaits each token.
        if token:
            chunks.append(token)
            # Collect the token instead of yielding it immediately.

    full_content = "".join(chunks)
    # "".join(chunks) = concatenate all tokens into one string.
    # e.g. ["Hello", " world", "!"] → "Hello world!"
    # This is the complete agent answer.

    return _UTF8JSONResponse(
        content={"type": "complete", "content": full_content},
        # {"type": "complete"} mirrors the SSE typed-event convention.
        # NestJS can check `type === "complete"` and read `content` directly.
        # Consistent typing makes it easy to switch between stream/non-stream modes.
    )


# ── Configure FastAPI's JSON encoder to output Chinese characters as-is ───────
# JSONResponse internally uses Python's json module with ensure_ascii=True by default,
# which turns 中文 → \u4e2d\u6587. Override at the router level so all JSON responses
# (including error responses) use ensure_ascii=False.
import json as _json

class _UTF8JSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return _json.dumps(content, ensure_ascii=False).encode("utf-8")


# ── POST /v1/agent/chat-with-file ─────────────────────────────────────────────
# Chat endpoint that accepts a file upload alongside the user's message.
# The file text is extracted and injected directly into Claude's context window
# so Claude can read, summarize, or answer questions about the document.
#
# MULTIPART vs JSON:
#   The regular /v1/agent/chat endpoint accepts a JSON body.
#   JSON cannot carry binary data (files) — you'd have to base64-encode the file,
#   which inflates size by ~33% and requires the client to encode it first.
#   Multipart/form-data is the standard HTTP format for mixing text fields and
#   binary file uploads in a single request. Browsers and NestJS both support it.
#
# HOW IT WORKS END-TO-END:
#   1. NestJS sends multipart/form-data with: file + message + session_id + ...
#   2. FastAPI receives each field separately (UploadFile + Form fields).
#   3. We extract plain text from the file (PDF/DOCX/TXT via file_parser.py).
#   4. We build a normal AgentRequest, setting document_context = extracted text.
#   5. The agent loop reads document_context and injects it into Claude's system prompt.
#   6. Claude can now read the full document and answer the user's question.
#   7. We stream the response back as SSE (same format as /v1/agent/chat).
#
# PYTHON CONCEPT — Form() vs Body():
#   In a JSON endpoint, FastAPI reads ALL parameters from the request body as one JSON object.
#   In a multipart endpoint, each form field is a SEPARATE part of the request.
#   You declare each field with `= Form(...)` so FastAPI knows to read it from the form,
#   not from a JSON body. File fields use `= File(...)` instead.
#   The `...` (Ellipsis) in both means the field is required — no default value.

@router.post("/agent/chat-with-file")
async def agent_chat_with_file(
    # ── Required form fields ──────────────────────────────────────────────────
    file: UploadFile = File(...),
    # The uploaded document (PDF, DOCX, TXT, etc.).
    # FastAPI reads this from the multipart "file" field.
    # UploadFile gives us: .filename (str), .content_type (str), .read() (async)

    message: str = Form(...),
    # The user's question or instruction about the uploaded file.
    # e.g. "Summarize this document" or "What are the key risks in this contract?"

    user_id: str = Form(...),
    # Same as AgentRequest.user_id — identifies the user for memory recall.

    session_id: str = Form(...),
    # Same as AgentRequest.session_id — used for logging / SSE correlation.

    subscription_tier: str = Form(...),
    # User's subscription level — passed to build_system_prompt via UserContext.

    # ── Optional form fields (with defaults) ─────────────────────────────────
    locale: str = Form("en-US"),
    # User's locale for language preference. Defaults to "en-US" if not sent.

    timezone: str = Form("UTC"),
    # User's timezone for date/time context.

    stream: bool = Form(True),
    # True → stream SSE tokens (default).
    # False → return a single JSON response with the full answer.

    history_json: str = Form("[]"),
    # Conversation history sent as a JSON *string* (not a nested object),
    # because multipart forms don't support nested JSON objects natively.
    #
    # NestJS sends: history_json = '[{"role":"user","content":"Hi"},...]'
    # We parse it with json.loads() below.
    #
    # PYTHON CONCEPT — Form fields are always strings:
    #   Unlike JSON bodies where Pydantic can parse nested types directly,
    #   multipart form fields arrive as strings. Complex types like lists must
    #   be JSON-encoded by the sender and decoded manually by the receiver.
    #   This is the standard pattern for sending structured data in form submissions.

    x_ai_api_key:  str | None = Header(default=None),
    x_ai_model:    str | None = Header(default=None),
    x_ai_base_url: str | None = Header(default=None),
):
    """
    Upload a document and ask a question about it in one request.

    Accepts multipart/form-data with:
      - file:              the document file (PDF, DOCX, TXT, …)
      - message:           the user's question or instruction
      - user_id:           user identifier (for memory recall)
      - session_id:        session identifier (for SSE correlation)
      - subscription_tier: e.g. "free", "pro", "enterprise"
      - locale:            e.g. "en-US" (optional, default "en-US")
      - timezone:          e.g. "America/New_York" (optional, default "UTC")
      - stream:            true/false (optional, default true)
      - history_json:      JSON-encoded array of past messages (optional, default [])

    Returns the same SSE stream format as POST /v1/agent/chat.
    """
    import json as _json_mod
    # Import locally to keep the module-level namespace clean.

    # ── Step 1: Extract text from the uploaded file ───────────────────────────
    file_bytes = await file.read()
    # await file.read() = async read — pause until all bytes are loaded.
    # Returns raw bytes (binary data, not a string).

    try:
        document_text = extract_text(file.filename or "upload", file_bytes)
        # extract_text() dispatches to the right parser based on file extension:
        #   .pdf  → PyMuPDF (with OCR fallback for scanned pages)
        #   .docx → python-docx (paragraphs + tables)
        #   else  → UTF-8 decode
        # Returns a plain string (may be capped at 80,000 characters).
    except ValueError as exc:
        # extract_text() raises ValueError for empty files or unsupported content.
        raise HTTPException(status_code=422, detail=str(exc))

    # ── Step 2: Parse the conversation history ────────────────────────────────
    try:
        raw_history: list[dict] = _json_mod.loads(history_json)
        # json.loads() = parse JSON string → Python list of dicts.
        # e.g. '[{"role":"user","content":"Hi"}]' → [{"role": "user", "content": "Hi"}]
        history = [HistoryMessage(**h) for h in raw_history]
        # HistoryMessage(**h) = unpack each dict into a Pydantic model.
        # This validates that each item has role ("user"/"assistant") and content.
        # PYTHON CONCEPT — ** (dict unpacking):
        #   HistoryMessage(**{"role": "user", "content": "Hi"})
        #   is the same as: HistoryMessage(role="user", content="Hi")
    except Exception:
        # If history_json is malformed (bad JSON or wrong shape), default to empty.
        history = []

    # ── Step 3: Build a standard AgentRequest with document_context ───────────
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
        # document_context carries the extracted file text into the agent loop.
        # build_system_prompt() will inject it under "## Uploaded Document"
        # so Claude can read and reason about the full document content.
    )

    # ── Step 4: Run guardrails on the user's message ──────────────────────────
    # We run guardrails on the text message only (not the document content).
    # The document is the user's own file — we trust it hasn't been injected.
    guard_result = guardrails.check(agent_request.message)
    if not guard_result.passed:
        raise HTTPException(status_code=422, detail=guard_result.reason)
    agent_request.message = guard_result.sanitized_message

    # ── Step 5: Resolve LLM credentials from headers or server config ─────────
    api_key, model, base_url = _resolve_llm_headers(x_ai_api_key, x_ai_model, x_ai_base_url)

    # ── Step 6: Stream or return the agent response ───────────────────────────
    # This mirrors the logic in agent_chat() above exactly.
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

    # stream=False: collect all tokens, return single JSON response.
    chunks: list[str] = []
    async for token in run(agent_request, llm_api_key=api_key, llm_model=model, llm_base_url=base_url):
        if token:
            chunks.append(token)
    return _UTF8JSONResponse(
        content={"type": "complete", "content": "".join(chunks)}
    )
