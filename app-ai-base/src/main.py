from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routers.agent import router as agent_router
from src.routers.rag import router as rag_router
from src.config.settings import settings

app = FastAPI(
    title="app-ai-base — AI Agent Service",
    version="1.0.0",
    description="LangChain ReAct agent powered by Claude. Internal service — not public.",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:4000"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.include_router(agent_router)
app.include_router(rag_router)


@app.get("/health")
async def health():
    """Liveness probe for load balancers and deployment platforms."""
    return {"status": "ok", "service": "app-ai-base"}


@app.get("/info")
async def info():
    """Returns current model configuration for debugging."""
    return {
        "service": "app-ai-base",
        "claude_model": settings.claude_model,
        "claude_max_tokens": settings.claude_max_tokens,
        "embedding_model": "BAAI/bge-m3",
    }
