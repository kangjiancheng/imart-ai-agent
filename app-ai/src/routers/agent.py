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

from fastapi.responses import StreamingResponse
# StreamingResponse = a FastAPI response that streams data in chunks.
# Instead of returning one big JSON body, it sends pieces one at a time.
# The client receives each piece immediately as it's yielded.

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


# ── POST /v1/agent/chat ────────────────────────────────────────────────────────
# The main agent streaming endpoint.

@router.post("/agent/chat")
async def agent_chat(request: AgentRequest):
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

    # ── Step 3: Define the SSE token streaming generator ────────────────────
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

        async for token in run(request):
            # run(request) is the agent loop — an async generator from agent_loop.py.
            # `async for token in run(request)` iterates over yielded tokens.
            # Each `token` is a string (one word or piece of a word from Claude).
            # `async for` is required because run() is async — it awaits I/O internally.
            # In TypeScript: for await (const token of run(request))

            if token:
                # Skip empty strings (Claude's streaming API sometimes yields "")
                yield f"data: {token}\n\n"
                # PYTHON CONCEPT — f-string + \n:
                #   f"data: {token}\n\n"
                #   Formats the token as an SSE event line.
                #   \n = newline, \n\n = two newlines (required by SSE protocol).
                #   SSE format: "data: <content>\n\n"
                #   The client reads everything after "data: " as the event data.

        yield "data: [DONE]\n\n"
        # After all tokens are sent, yield the sentinel value "[DONE]".
        # NestJS (the client) watches for this to know the stream has ended.
        # Without this, the client wouldn't know when to stop reading.

    # ── Step 4: Return a StreamingResponse wrapping the generator ────────────
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
