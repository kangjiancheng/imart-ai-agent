# src/schemas/request.py
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT ARE SCHEMAS?
#
# A "schema" defines the SHAPE of data — what fields are required, what types
# they must be, and what constraints apply (min length, max length, etc.).
#
# This file defines the shape of the JSON body NestJS sends to the /v1/agent/chat
# endpoint. FastAPI automatically validates incoming JSON against these schemas.
# If the data doesn't match, FastAPI returns a 422 error before your code runs.
#
# PYTHON CONCEPT — Pydantic BaseModel:
#   Pydantic is a Python library for data validation using class definitions.
#   You declare a class that inherits from BaseModel, then list fields as
#   class-level annotations (field_name: type = default).
#   Pydantic reads those annotations at runtime and enforces them.
#
#   In TypeScript: similar to defining an interface + running zod.parse().
#   Pydantic combines both in one class.
#
# HOW FASTAPI USES SCHEMAS:
#   When you write `async def agent_chat(request: AgentRequest):`
#   FastAPI sees the type hint `AgentRequest` and knows:
#     1. Parse the HTTP request body as JSON
#     2. Validate it against AgentRequest
#     3. If valid → call the function with a real AgentRequest object
#     4. If invalid → return 422 automatically (your code never runs)
# ─────────────────────────────────────────────────────────────────────────────

from pydantic import BaseModel, Field
# BaseModel  = the base class all Pydantic models must inherit from
# Field      = a helper to attach extra constraints (min_length, max_length, description)
#              to a field beyond just its type annotation

from typing import Literal
# Literal[...] = a special type that only allows specific exact values.
# In Python 3.9+ you can import Literal from `typing`.
# In TypeScript: `role: "user" | "assistant"`


# ── HistoryMessage ────────────────────────────────────────────────────────────
# Represents one turn of conversation history (user or assistant message).
# NestJS sends the recent conversation history so the agent has context.

class HistoryMessage(BaseModel):
    # PYTHON CONCEPT — Literal type:
    #   Literal["user", "assistant"] means this field MUST be exactly
    #   the string "user" OR the string "assistant" — nothing else.
    #   If NestJS sends role: "system", Pydantic rejects the request with 422.
    #   In TypeScript: role: "user" | "assistant"
    role: Literal["user", "assistant"]

    # content = the text of the message
    # No default → required field. NestJS must always provide this.
    content: str

    # EFFECT ON THE AGENT:
    #   These history messages are converted to LangChain HumanMessage /
    #   AIMessage objects inside build_messages() in claude_client.py.
    #   They give Claude context about what was said earlier in the session.


# ── UserContext ────────────────────────────────────────────────────────────────
# Metadata about the user making the request.
# Used by the system prompt to personalize Claude's behavior.

class UserContext(BaseModel):
    # subscription_tier: e.g. "free", "pro", "enterprise"
    # The system prompt can use this to limit or expand Claude's capabilities.
    subscription_tier: str

    # locale: language/region code for response language
    # = "en-US" means: if NestJS doesn't send this field, use "en-US" as default
    # PYTHON CONCEPT — default values:
    #   In Python class annotations, you set defaults with = after the type.
    #   `locale: str = "en-US"` means the field is OPTIONAL in the JSON body.
    #   If missing, Pydantic fills it in automatically. Same as TypeScript optional fields.
    locale: str = "en-US"

    # timezone: IANA timezone string, e.g. "America/New_York"
    timezone: str = "UTC"

    # EFFECT ON THE AGENT:
    #   Passed to build_system_prompt() in claude_client.py, where it's
    #   injected into the system prompt so Claude knows the user's subscription
    #   level, language preference, and timezone for date/time formatting.


# ── AgentRequest ──────────────────────────────────────────────────────────────
# The full JSON body that NestJS sends to POST /v1/agent/chat.
# This is the main entry point schema for the agent endpoint.

class AgentRequest(BaseModel):
    # user_id: unique identifier for the user (e.g. UUID from NestJS)
    # Used by VectorMemory to scope long-term memories to this user.
    user_id: str

    # message: the user's current question or request
    #
    # PYTHON CONCEPT — Field():
    #   Field(..., min_length=1, max_length=10_000)
    #   The first argument `...` (three dots = Ellipsis) means REQUIRED — no default.
    #   min_length=1  → rejects empty strings ""
    #   max_length=10_000 → rejects messages over 10,000 characters
    #   10_000 is valid Python — underscores in numbers are just visual separators
    #   (like 10,000 in English). In JS: 10_000 === 10000.
    #   If these constraints fail, FastAPI returns 422 before your function runs.
    message: str = Field(..., min_length=1, max_length=10_000)

    # history: list of prior conversation turns
    #
    # PYTHON CONCEPT — list[HistoryMessage]:
    #   This says: a list where every element must be a HistoryMessage object.
    #   In TypeScript: HistoryMessage[]
    #
    # = [] means: if NestJS doesn't send history at all, default to an empty list.
    # This handles the first message of a new session gracefully.
    history: list[HistoryMessage] = []

    # user_context: nested object with user metadata
    #
    # PYTHON CONCEPT — nested Pydantic models:
    #   A Pydantic model can contain another Pydantic model as a field.
    #   Pydantic validates the nested object automatically.
    #   The JSON from NestJS looks like:
    #     { "user_context": { "subscription_tier": "pro", "locale": "en-US" } }
    #   Pydantic parses that inner object into a real UserContext instance.
    #   No default → user_context is REQUIRED.
    user_context: UserContext

    # session_id: unique identifier for this conversation session
    # Used for logging, tracing, and correlating SSE events.
    session_id: str

    # stream: whether to stream the response as SSE events or return a single JSON body.
    #
    # True  (default) → stream tokens one-by-one via Server-Sent Events
    # False           → collect ALL tokens first, then return one JSON object:
    #                   {"type": "complete", "content": "full answer here"}
    #
    # When to use stream=False:
    #   - NestJS background jobs that don't need real-time display
    #   - Testing / curl requests where SSE parsing is inconvenient
    #   - Clients that can't consume EventSource (some serverless platforms)
    stream: bool = True

    # document_context: plain text extracted from an uploaded file (optional).
    #
    # When set, the agent injects this text into Claude's system prompt under
    # a "## Uploaded Document" section so Claude can read and reason about it.
    #
    # PYTHON CONCEPT — Optional field with None default:
    #   `str | None = None` means the field can be a string OR None (null).
    #   This is Python 3.10+ union syntax — equivalent to `Optional[str] = None`
    #   from the `typing` module in older Python versions.
    #   If not provided, Pydantic leaves it as None and the agent runs normally.
    #
    # This field is NOT sent by the JSON /v1/agent/chat endpoint.
    # It is populated internally by the /v1/agent/chat-with-file endpoint
    # after extracting text from the uploaded file, then passed to run().
    document_context: str | None = None
