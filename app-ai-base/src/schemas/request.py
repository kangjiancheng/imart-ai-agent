from pydantic import BaseModel, Field
from typing import Literal


class HistoryMessage(BaseModel):
    """One turn of conversation history passed from the client."""
    role: Literal["user", "assistant"]
    content: str


class UserContext(BaseModel):
    """User metadata used for system prompt personalization."""
    subscription_tier: str
    locale: str = "en-US"
    timezone: str = "UTC"


class AgentRequest(BaseModel):
    """Full JSON body for POST /v1/agent/chat."""
    user_id: str
    message: str = Field(..., min_length=1, max_length=10_000)
    history: list[HistoryMessage] = []
    user_context: UserContext
    session_id: str
    stream: bool = True
    # Populated internally by /v1/agent/chat-with-file after file text extraction.
    document_context: str | None = None
