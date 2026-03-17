from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routers.agent import router as agent_router
from src.routers.rag import router as rag_router
from src.routers.common import router as common_router

app = FastAPI(
    title="app-ai-base — AI Agent Service",
    version="1.0.0",
    description="LangChain ReAct agent powered by Claude. Internal service — not public.",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", 'http://localhost:3333', "http://localhost:4000"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.include_router(common_router)
app.include_router(agent_router)
app.include_router(rag_router)
