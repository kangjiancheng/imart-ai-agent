# src/schemas/response.py
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS THIS FILE FOR?
#
# This file defines the RESPONSE schemas — the shape of data the agent sends back.
#
# There are two ways the agent endpoint returns data:
#
#   1. STREAMING (SSE) — the /v1/agent/chat endpoint
#      Sends raw text tokens one-by-one using Server-Sent Events format:
#        data: Hello\n\n
#        data:  world\n\n
#        data: [DONE]\n\n
#      These schemas are NOT used for streaming — StreamingResponse handles that.
#
#   2. NON-STREAMING — utility endpoints and OpenAPI documentation
#      Uses AgentResponse below to describe the final structured response.
#      FastAPI auto-generates Swagger UI docs from these schemas.
#      Visit http://localhost:8000/docs to see them in the browser.
#
# WHY DEFINE RESPONSE SCHEMAS IF THE MAIN ENDPOINT STREAMS?
#   - OpenAPI/Swagger documentation shows expected response shapes
#   - Future non-streaming endpoints (batch processing, summaries) can reuse these
#   - Type-safe return values reduce bugs in non-streaming utility routes
# ─────────────────────────────────────────────────────────────────────────────

from pydantic import BaseModel
# BaseModel = the Pydantic base class all schemas must inherit from.
# See request.py for a full explanation of Pydantic and BaseModel.


# ── Document ──────────────────────────────────────────────────────────────────
# Represents a single document retrieved from the RAG knowledge base.
# Included in the response so the client knows WHICH sources were used.

class Document(BaseModel):
    # content: the text excerpt from the knowledge base document
    content: str

    # source: filename or URL of the original document
    # e.g. "product-manual.pdf", "https://docs.example.com/api"
    source: str

    # score: how semantically similar this document was to the query (0.0 → 1.0)
    # Higher = more relevant. Only documents above MIN_SCORE (0.72) are returned.
    # float = a decimal number in Python (same as JS number for most purposes)
    score: float

    # PYTHON CONCEPT — float vs int:
    #   In Python, `int` = whole numbers (1, 2, 42)
    #   `float` = decimal numbers (0.72, 1.0, 3.14)
    #   JavaScript has only `number` — Python separates these two types.
    #   Pydantic validates that score is a number with a decimal point.


# ── AgentResponse ─────────────────────────────────────────────────────────────
# The structured response shape for non-streaming agent results.
#
# WHEN IS THIS USED?
#   The /v1/agent/chat endpoint uses StreamingResponse (SSE) instead.
#   AgentResponse is used for:
#     - OpenAPI documentation (what the API "would" return if non-streaming)
#     - Potential future batch endpoints
#     - Type hints in utility functions that format agent output

class AgentResponse(BaseModel):
    # Streaming endpoint does not use this directly (it yields SSE).
    # Used by non-streaming utility endpoints and for OpenAPI docs.

    # answer: the agent's complete final response text
    # In the streaming version, this is built up token-by-token on the client.
    answer: str

    # session_id: echoes back the session_id from the request
    # Allows the client to correlate this response with the request that made it.
    session_id: str

    # sources: list of source identifiers (filenames, URLs) cited in the answer
    #
    # PYTHON CONCEPT — list[str] with default []:
    #   list[str] = a list where every element must be a string.
    #   In TypeScript: string[]
    #   = [] means: this field is OPTIONAL — if the agent found no sources,
    #   the field defaults to an empty list. The client doesn't need to send it.
    #
    # DIFFERENCE FROM Document above:
    #   sources is just a list of strings (file names or URLs).
    #   Document is the full object with content + score — used internally.
    #   The public API response only exposes the source names, not the full text,
    #   to keep the response payload small.
    sources: list[str] = []
