# src/main.py
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS main.py?
#
# This is the ENTRY POINT of the entire FastAPI application.
# When you run `uvicorn src.main:app`, Python:
#   1. Imports this file (src/main.py)
#   2. Finds the `app` object (a FastAPI instance)
#   3. Starts an async HTTP server that routes requests to app
#
# This file's job is minimal: create the app, configure middleware, mount routers.
# The actual business logic lives in routers/agent.py and routers/rag.py.
#
# PYTHON CONCEPT — how modules are loaded:
#   Every Python file is a "module". When this file is imported, every statement
#   at the top level runs immediately (not inside a function).
#   So `app = FastAPI(...)`, `app.add_middleware(...)`, `app.include_router(...)`
#   all execute at import time — when uvicorn starts the server.
#   In Node.js: similar to the top-level code in your server.ts file.
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI
# FastAPI = the main application class.
# Creates an HTTP server with automatic routing, request validation,
# response serialization, and OpenAPI documentation generation.
# In Node.js: similar to `const app = express()` or `new Hono()`.

from fastapi.middleware.cors import CORSMiddleware
# CORSMiddleware = handles Cross-Origin Resource Sharing headers.
# CORS is a browser security mechanism that blocks requests from one origin
# (e.g. http://localhost:3000) to another origin (e.g. http://localhost:8000)
# unless the server explicitly allows it.
# This service is called from NestJS (server-to-server), not a browser,
# but localhost origins are listed for local development convenience.

from src.routers.agent import router as agent_router
# Import the agent router from routers/agent.py.
# `as agent_router` = alias to avoid naming conflict with the rag router.
# agent_router contains POST /v1/agent/chat.

from src.routers.rag import router as rag_router
# Import the RAG router from routers/rag.py.
# rag_router contains POST /v1/rag/ingest and GET /v1/rag/ingest/{job_id}.

from src.config.settings import settings
# settings = the validated config object (reads .env). Used here to expose model info in /info.


# ── FastAPI app creation ────────────────────────────────────────────────────
# PYTHON CONCEPT — named arguments (keyword arguments):
#   FastAPI(...) accepts many optional configuration parameters.
#   `title=`, `version=`, etc. are "keyword arguments" — passed by name.
#   In TypeScript: new FastAPI({ title: "...", version: "..." })

app = FastAPI(
    title="app-ai — AI Agent Service",
    # Shown at the top of the Swagger UI at http://localhost:8000/docs

    version="1.0.0",
    # API version shown in Swagger UI and used for versioning headers.

    description="LangChain ReAct agent powered by Claude. Internal service — not public.",
    # Longer description shown in Swagger UI.

    docs_url="/docs",
    # URL for the interactive Swagger UI. Visit http://localhost:8000/docs in your browser
    # to see all endpoints, try them out, and see request/response schemas.

    redoc_url="/redoc",
    # URL for the ReDoc documentation UI — an alternative to Swagger.
    # Visit http://localhost:8000/redoc for a cleaner read-only API reference.
)


# ── CORS middleware ─────────────────────────────────────────────────────────
# PYTHON CONCEPT — app.add_middleware():
#   Middleware runs on EVERY request before it reaches your route handler,
#   and on EVERY response before it's sent back to the client.
#   Think of it as a pipeline: Request → [CORS middleware] → [route handler] → [CORS middleware] → Response.
#   app.add_middleware(MiddlewareClass, **kwargs) registers middleware globally.

app.add_middleware(
    CORSMiddleware,

    allow_origins=["http://localhost:3000", "http://localhost:4000"],
    # Which origins are allowed to make cross-origin requests.
    # http://localhost:3000 = typical NestJS dev server
    # http://localhost:4000 = typical Next.js or alternative frontend port
    # In PRODUCTION: replace with your actual domain names.
    # Using ["*"] (wildcard) would allow ANY origin — dangerous for production.

    allow_credentials=True,
    # Allow the client to send cookies and HTTP authentication headers.
    # Required if NestJS sends Authorization headers with its requests.

    allow_methods=["POST", "GET"],
    # Only POST and GET are allowed — these are the only methods this service uses.
    # Blocks PUT, PATCH, DELETE, etc. — defense in depth.
    # The agent chat endpoint is POST; the RAG status endpoint is GET.

    allow_headers=["*"],
    # Allow any HTTP headers in the request.
    # If you want stricter security, list only: ["Content-Type", "Authorization"].
)


# ── Mount routers ────────────────────────────────────────────────────────────
# PYTHON CONCEPT — app.include_router():
#   "Mounting" a router means: add all routes from that router to this app.
#   After these two calls, app knows about ALL routes from both routers:
#     POST /v1/agent/chat       (from agent_router)
#     POST /v1/rag/ingest       (from rag_router)
#     GET  /v1/rag/ingest/{id}  (from rag_router)
#
#   Why not define all routes in main.py?
#   Keeping routes in separate files makes the code easier to navigate and test.
#   main.py stays clean — it just assembles the pieces.

app.include_router(agent_router)
# Mount all routes from routers/agent.py (prefixed with /v1 already).

app.include_router(rag_router)
# Mount all routes from routers/rag.py (prefixed with /v1 already).


# ── Health check endpoint ────────────────────────────────────────────────────
# Defined directly in main.py (not a router) — it's a simple one-liner.

@app.get("/health")
async def health():
    """
    Health check endpoint used by load balancers and deployment platforms.
    Returns 200 OK with a simple JSON body if the service is running.

    PYTHON CONCEPT — @app.get() decorator:
      @app.get("/health") registers this function as the handler for GET /health.
      FastAPI calls `health()` whenever it receives a GET request to /health.
      Returning a dict → FastAPI automatically serializes it to JSON.
      In Express: app.get('/health', (req, res) => res.json({ status: 'ok' }))

    WHY A HEALTH ENDPOINT?
      - Kubernetes / Docker Compose uses it for liveness probes
      - Load balancers check it to decide if this instance should receive traffic
      - NestJS can call it before the first agent request to verify the service is up
    """
    return {"status": "ok", "service": "app-ai"}
    # FastAPI converts this dict to HTTP 200 JSON response:
    # {"status": "ok", "service": "app-ai"}


# ── Info endpoint ─────────────────────────────────────────────────────────────
# Returns the current model configuration — useful for debugging which model
# is active without having to check .env or ask the agent directly.

@app.get("/info")
async def info():
    """
    Returns current service configuration, including which Claude model is active.
    Call this to verify which model the agent is using before sending a chat request.

    Example response:
      {
        "service": "app-ai",
        "claude_model": "claude-sonnet-4-6",
        "claude_max_tokens": 4096,
        "embedding_model": "text-embedding-3-small"
      }
    """
    return {
        "service": "app-ai",
        "claude_model": settings.claude_model,
        "claude_max_tokens": settings.claude_max_tokens,
        "embedding_model": settings.embedding_model,
    }

