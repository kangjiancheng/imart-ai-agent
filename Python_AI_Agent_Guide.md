# Python AI Agent — Complete Developer Guide

> **Stack:** LangChain · FastAPI · Milvus · Anthropic Claude Sonnet 4.6
> **Audience:** Web developers transitioning to Python AI development
> **Goal:** Build a production-ready RAG-powered AI Agent from scratch

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [IDE & Tooling](#2-ide--tooling)
3. [Project Structure](#3-project-structure)
4. [Core Concepts](#4-core-concepts)
5. [LangChain — Building AI Agents](#5-langchain--building-ai-agents)
6. [FastAPI — The API Layer](#6-fastapi--the-api-layer)
7. [Milvus — Vector Database](#7-milvus--vector-database)
8. [Integrating Claude Sonnet 4.6](#8-integrating-claude-sonnet-46)
9. [Putting It All Together](#9-putting-it-all-together)
10. [Testing](#10-testing)
11. [Deployment](#11-deployment)
12. [Learning Roadmap](#12-learning-roadmap)

---

## 1. Environment Setup

### 1.1 Install Python

Use **Python 3.11+** for all AI/ML projects. It has the best library compatibility and performance.

```bash
# Verify your installation
python --version        # Should show Python 3.11.x or higher
pip --version           # Should show pip 23+
```

> **Web Dev Analogy:** Python is your runtime, like Node.js. `pip` is your package manager, like `npm`. `requirements.txt` is your `package.json`.

**Download:** https://www.python.org/downloads/

- ✅ On Windows: check **"Add Python to PATH"** during installation
- ✅ On macOS: use [Homebrew](https://brew.sh) — `brew install python@3.11`
- ✅ On Linux: `sudo apt install python3.11 python3.11-venv python3-pip`

---

### 1.2 Virtual Environments

Always isolate dependencies per project using a virtual environment.

```bash
# Create a virtual environment (run inside your project folder)
python -m venv venv

# Activate it
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# Deactivate when done
deactivate
```

> **Web Dev Analogy:** A virtual environment is like your project's `node_modules` — it keeps dependencies isolated so projects don't conflict.

---

### 1.3 Install Core Dependencies

```bash
# LangChain ecosystem
pip install langchain langchain-community langchain-anthropic

# Web framework
pip install fastapi "uvicorn[standard]"

# Vector database client
pip install pymilvus

# Utilities
pip install python-dotenv pydantic httpx
```

Save dependencies:

```bash
pip freeze > requirements.txt
```

Restore on a new machine:

```bash
pip install -r requirements.txt
```

---

### 1.4 Environment Variables

Create a `.env` file at your project root to store secrets:

```env
# .env  — NEVER commit this file to Git
ANTHROPIC_API_KEY=sk-ant-your-key-here
MILVUS_HOST=localhost
MILVUS_PORT=19530
APP_ENV=development
APP_PORT=8000
```

Load it in Python:

```python
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
```

> Add `.env` to your `.gitignore` immediately.

---

## 2. IDE & Tooling

### 2.1 VS Code (Recommended)

VS Code is the best starting point for web developers moving to Python — familiar interface, great extensions.

**Download:** https://code.visualstudio.com

#### Essential Extensions

| Extension              | Purpose                                           |
| ---------------------- | ------------------------------------------------- |
| **Python** (Microsoft) | Core Python support: syntax, linting, debugger    |
| **Pylance**            | Advanced type checking and autocomplete           |
| **Ruff**               | Fast linter + formatter (replaces flake8 + black) |
| **Docker**             | Manage containers from the sidebar                |
| **REST Client**        | Test API endpoints via `.http` files              |
| **GitLens**            | Enhanced Git history and blame                    |

#### Recommended `.vscode/settings.json`

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff"
  },
  "editor.rulers": [88],
  "files.trimTrailingWhitespace": true
}
```

---

### 2.2 Alternative IDEs

| IDE                     | Best For                                                |
| ----------------------- | ------------------------------------------------------- |
| **PyCharm** (JetBrains) | Large projects, advanced refactoring, built-in debugger |
| **Cursor**              | AI-assisted coding (built-in Claude/GPT integration)    |
| **Jupyter Lab**         | Experiments, data exploration, prototyping              |

---

### 2.3 Other Essential Tools

| Tool                | Purpose             | Install                 |
| ------------------- | ------------------- | ----------------------- |
| **Git**             | Version control     | https://git-scm.com     |
| **Docker Desktop**  | Run Milvus locally  | https://docker.com      |
| **Postman / Bruno** | API testing         | https://www.postman.com |
| **TablePlus**       | Database GUI viewer | https://tableplus.com   |

---

## 3. Project Structure

A well-organized layout for a production LangChain + FastAPI + Milvus project:

```
my-ai-agent/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app entry point
│   ├── agents/                  # LangChain agent logic
│   │   ├── __init__.py
│   │   ├── rag_agent.py         # RAG agent implementation
│   │   └── tools.py             # Custom agent tools
│   ├── api/                     # HTTP route handlers
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── core/                    # App configuration
│   │   ├── __init__.py
│   │   └── config.py
│   ├── services/                # Business logic layer
│   │   ├── __init__.py
│   │   ├── vector_service.py    # Milvus operations
│   │   └── document_service.py  # Document ingestion
│   └── models/                  # Pydantic schemas
│       ├── __init__.py
│       └── schemas.py
├── tests/
│   ├── test_agent.py
│   ├── test_api.py
│   └── test_vector.py
├── scripts/
│   └── ingest_docs.py           # Load documents into Milvus
├── docker/
│   └── docker-compose.yml       # Local Milvus stack
├── .env                         # Secrets (never commit)
├── .env.example                 # Committed template (no real values)
├── .gitignore
├── requirements.txt
├── Dockerfile
└── README.md
```

### `app/core/config.py`

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    app_env: str = "development"
    app_port: int = 8000

    class Config:
        env_file = ".env"

settings = Settings()
```

---

## 4. Core Concepts

Before writing code, understand how these three components connect:

```
User Question
     │
     ▼
[FastAPI Route]          ← HTTP layer (like Express.js)
     │
     ▼
[LangChain Agent]        ← Orchestration layer
     │         │
     ▼         ▼
[Milvus]    [Claude]     ← Storage + Intelligence
(retrieve   (generate
 context)    answer)
     │         │
     └────┬────┘
          ▼
     Final Answer
```

| Component             | Role                                  | Analogy                         |
| --------------------- | ------------------------------------- | ------------------------------- |
| **FastAPI**           | Expose your agent as an HTTP API      | Express.js / Next.js API routes |
| **LangChain**         | Orchestrate LLM + tools + memory      | Business logic controller       |
| **Milvus**            | Store and search document embeddings  | MongoDB for vectors             |
| **Claude Sonnet 4.6** | Understand language, reason, generate | The AI brain                    |

---

## 5. LangChain — Building AI Agents

### 5.1 Key Concepts

| Concept              | Description                                      |
| -------------------- | ------------------------------------------------ |
| **LLM / Chat Model** | The AI model connection (Claude, GPT, etc.)      |
| **Prompt Template**  | Reusable prompt with variables                   |
| **Chain**            | A pipeline: prompt → LLM → output parser         |
| **Agent**            | LLM that dynamically decides which tools to call |
| **Memory**           | Stores conversation history across turns         |
| **Embeddings**       | Converts text into numerical vectors             |
| **Retriever**        | Fetches relevant docs from a vector store        |
| **Vector Store**     | Database of embeddings (you'll use Milvus)       |

---

### 5.2 Simple Chain Example

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize Claude Sonnet 4.6
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1024,
)

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{question}"),
])

# Build chain using pipe operator (like middleware chaining)
chain = prompt | llm | StrOutputParser()

# Invoke
response = chain.invoke({"question": "What is a vector database?"})
print(response)
```

---

### 5.3 RAG Agent with Milvus

RAG (Retrieval-Augmented Generation) = search your docs → give Claude context → get a grounded answer.

```python
# app/agents/rag_agent.py
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import os

# Embedding model (runs locally, no API needed)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to Milvus vector store
vector_store = Milvus(
    embedding_function=embeddings,
    collection_name="documents",
    connection_args={
        "host": os.getenv("MILVUS_HOST", "localhost"),
        "port": os.getenv("MILVUS_PORT", "19530"),
    },
)

# LLM
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
)

# Custom RAG prompt
rag_prompt = PromptTemplate.from_template("""
Use the following context to answer the question.
If you don't know the answer based on the context, say so.

Context:
{context}

Question: {question}

Answer:
""")

# Build RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
    chain_type_kwargs={"prompt": rag_prompt},
    return_source_documents=True,
)
```

---

### 5.4 Agent with Tools

An agent lets Claude decide which tools to use dynamically:

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define a custom tool
@tool
def search_documents(query: str) -> str:
    """Search internal documents for relevant information."""
    results = vector_store.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in results])

@tool
def get_current_date() -> str:
    """Returns today's date."""
    from datetime import date
    return str(date.today())

tools = [search_documents, get_current_date]

# Prompt with agent scratchpad
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to tools."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# Create and run agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input": "What documents do we have about AI?"})
```

---

### 5.5 Conversation Memory

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# In-memory store (use Redis/DB for production)
session_store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

# Wrap chain with history
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Use it
chain_with_memory.invoke(
    {"input": "My name is Alex"},
    config={"configurable": {"session_id": "user-123"}},
)
```

---

## 6. FastAPI — The API Layer

### 6.1 Comparison with Express.js

| Express.js (Node)           | FastAPI (Python)                 |
| --------------------------- | -------------------------------- |
| `app.get('/path', handler)` | `@app.get('/path')`              |
| `req.body`                  | Pydantic request model           |
| `res.json({...})`           | `return {...}` (auto-serialized) |
| `app.use(middleware)`       | `app.add_middleware(...)`        |
| `express.Router()`          | `APIRouter()`                    |
| Manual Swagger setup        | Auto docs at `/docs`             |
| `process.env.PORT`          | `os.getenv("PORT")`              |

---

### 6.2 Main Application

```python
# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.core.config import settings

app = FastAPI(
    title="AI Agent API",
    version="1.0.0",
    description="LangChain + Milvus + Claude AI Agent",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}
```

---

### 6.3 API Routes

```python
# app/api/routes.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from app.agents.rag_agent import rag_chain
from app.agents.agent import agent_executor

router = APIRouter()

# --- Request / Response Models ---
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []

# --- Endpoints ---
@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        result = await rag_chain.ainvoke({"query": req.message})
        sources = [
            doc.metadata.get("source", "unknown")
            for doc in result.get("source_documents", [])
        ]
        return ChatResponse(answer=result["result"], sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agent")
async def run_agent(req: ChatRequest):
    result = await agent_executor.ainvoke({"input": req.message})
    return {"output": result["output"]}
```

---

### 6.4 Running the Server

```bash
# Development (auto-reload on file changes)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Open in your browser:

- **Swagger UI (interactive docs):** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health check:** http://localhost:8000/health

---

### 6.5 Dependency Injection

FastAPI's `Depends` is like middleware but per-route — useful for auth, DB connections, etc.

```python
from fastapi import Depends, Header, HTTPException

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != os.getenv("INTERNAL_API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

@router.post("/secure-chat")
async def secure_chat(req: ChatRequest, _=Depends(verify_api_key)):
    ...
```

---

## 7. Milvus — Vector Database

### 7.1 What is a Vector Database?

When you embed text, you get a list of numbers (a **vector**) that captures semantic meaning. Milvus stores these vectors and lets you search for the most _semantically similar_ ones — not keyword matches, but meaning matches.

```
"What is machine learning?"
         │
         ▼
  Embedding Model
         │
         ▼
[0.12, -0.45, 0.89, ...]  ← 768-dimensional vector
         │
         ▼
   Stored in Milvus
         │
     (later...)
         │
"Explain ML to me"  ──► similar vector ──► finds same docs
```

---

### 7.2 Start Milvus with Docker

```yaml
# docker/docker-compose.yml
version: "3.8"

services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    volumes:
      - etcd_data:/etcd

  minio:
    image: minio/minio:RELEASE.2023-03-13T19-46-17Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - minio_data:/minio_data
    command: minio server /minio_data

  milvus:
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - etcd
      - minio
    volumes:
      - milvus_data:/var/lib/milvus

volumes:
  etcd_data:
  minio_data:
  milvus_data:
```

```bash
# Start the stack
docker compose -f docker/docker-compose.yml up -d

# Verify Milvus is running
curl http://localhost:9091/healthz
```

---

### 7.3 Vector Service

```python
# app/services/vector_service.py
from pymilvus import (
    connections, Collection, FieldSchema,
    CollectionSchema, DataType, utility
)
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

COLLECTION_NAME = "documents"
EMBEDDING_DIM = 384  # Matches all-MiniLM-L6-v2

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_vector_store() -> Milvus:
    return Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={
            "host": os.getenv("MILVUS_HOST", "localhost"),
            "port": os.getenv("MILVUS_PORT", "19530"),
        },
    )

def ingest_documents(texts: list[str], metadatas: list[dict] = None):
    """Load text documents into Milvus."""
    from langchain_core.documents import Document

    docs = [
        Document(page_content=t, metadata=m or {})
        for t, m in zip(texts, metadatas or [{}] * len(texts))
    ]

    Milvus.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={
            "host": os.getenv("MILVUS_HOST", "localhost"),
            "port": os.getenv("MILVUS_PORT", "19530"),
        },
    )
    print(f"Ingested {len(docs)} documents into Milvus.")
```

---

### 7.4 Document Ingestion Script

```python
# scripts/ingest_docs.py
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.services.vector_service import ingest_documents
from dotenv import load_dotenv

load_dotenv()

# Load documents from a folder
loader = DirectoryLoader("./docs", glob="**/*.txt", loader_cls=TextLoader)
raw_docs = loader.load()

# Split into chunks (like paginating content)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(raw_docs)

# Ingest into Milvus
texts = [c.page_content for c in chunks]
metas = [c.metadata for c in chunks]
ingest_documents(texts, metas)
```

---

## 8. Integrating Claude Sonnet 4.6

### 8.1 Setup

```bash
pip install langchain-anthropic anthropic
```

```python
import os
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=2048,
    temperature=0,        # 0 = deterministic, 1 = creative
)
```

> Get your API key at: https://console.anthropic.com

---

### 8.2 Streaming Responses

Stream tokens back to the client in real time (great for chat UIs):

```python
# Streaming with LangChain
async for chunk in llm.astream("Explain RAG in simple terms"):
    print(chunk.content, end="", flush=True)
```

**FastAPI Streaming Endpoint:**

```python
from fastapi.responses import StreamingResponse

@router.post("/chat/stream")
async def stream_chat(req: ChatRequest):
    async def event_generator():
        async for chunk in llm.astream(req.message):
            if chunk.content:
                yield f"data: {chunk.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

---

### 8.3 Claude-Specific Features

```python
# System prompt
from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="You are a senior software engineer. Be concise."),
    HumanMessage(content="What is dependency injection?"),
]

response = llm.invoke(messages)
print(response.content)
```

```python
# Multi-turn conversation
from langchain_core.messages import AIMessage

history = [
    HumanMessage(content="My name is Alex."),
    AIMessage(content="Hello Alex! How can I help you today?"),
    HumanMessage(content="What is my name?"),
]

response = llm.invoke(history)
# Claude will correctly answer: "Your name is Alex."
```

---

### 8.4 Choosing the Right Model

| Model               | Use Case                                             |
| ------------------- | ---------------------------------------------------- |
| `claude-haiku-4-5`  | Fast, cheap — simple Q&A, classification             |
| `claude-sonnet-4-6` | ✅ **Recommended** — best balance of speed + quality |
| `claude-opus-4-6`   | Most powerful — complex reasoning, long documents    |

---

## 9. Putting It All Together

### Complete RAG Agent Implementation

```python
# app/agents/rag_agent.py
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

# Components
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Milvus(
    embedding_function=embeddings,
    collection_name="documents",
    connection_args={
        "host": os.getenv("MILVUS_HOST", "localhost"),
        "port": os.getenv("MILVUS_PORT", "19530"),
    },
)

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=2048,
    temperature=0,
)

# Build conversational RAG chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True,
    verbose=True,
)

# Session memory store
session_store: dict = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]
```

---

### Complete FastAPI Route

```python
# app/api/routes.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.agents.rag_agent import qa_chain, get_session_history

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []
    session_id: str

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    history = get_session_history(req.session_id)
    chat_history = history.messages

    try:
        result = await qa_chain.ainvoke({
            "question": req.message,
            "chat_history": chat_history,
        })

        # Save to history
        from langchain_core.messages import HumanMessage, AIMessage
        history.add_message(HumanMessage(content=req.message))
        history.add_message(AIMessage(content=result["answer"]))

        sources = list({
            doc.metadata.get("source", "unknown")
            for doc in result.get("source_documents", [])
        })

        return ChatResponse(
            answer=result["answer"],
            sources=sources,
            session_id=req.session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 10. Testing

### 10.1 Setup

```bash
pip install pytest pytest-asyncio httpx
```

### 10.2 Unit Test — Agent

```python
# tests/test_agent.py
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_rag_chain_returns_answer():
    with patch("app.agents.rag_agent.qa_chain") as mock_chain:
        mock_chain.ainvoke = AsyncMock(return_value={
            "answer": "Milvus is a vector database.",
            "source_documents": [],
        })

        result = await mock_chain.ainvoke({"question": "What is Milvus?"})
        assert "Milvus" in result["answer"]
```

### 10.3 Integration Test — API

```python
# tests/test_api.py
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

@pytest.mark.asyncio
async def test_chat_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/api/v1/chat", json={
            "message": "Hello",
            "session_id": "test-session",
        })
    assert response.status_code == 200
    assert "answer" in response.json()
```

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## 11. Deployment

### 11.1 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 11.2 Docker Compose (Full Stack)

```yaml
# docker-compose.yml
version: "3.8"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      - milvus
    restart: unless-stopped

  milvus:
    image: milvusdb/milvus:v2.4.0
    ports:
      - "19530:19530"
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    depends_on:
      - etcd
      - minio

  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      ETCD_AUTO_COMPACTION_MODE: revision

  minio:
    image: minio/minio:RELEASE.2023-03-13T19-46-17Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    command: minio server /data
```

```bash
# Build and start everything
docker compose up -d --build

# View logs
docker compose logs -f api

# Stop
docker compose down
```

---

### 11.3 Cloud Deployment Options

| Platform                      | Type                  | Best For                        |
| ----------------------------- | --------------------- | ------------------------------- |
| **AWS ECS / EC2**             | Container / VM        | Full control, scalable          |
| **Google Cloud Run**          | Serverless containers | Easy scaling, pay-per-use       |
| **Azure Container Apps**      | Serverless containers | Microsoft ecosystem             |
| **Railway**                   | PaaS                  | Simplest deployment, fast setup |
| **Render**                    | PaaS                  | Free tier available             |
| **DigitalOcean App Platform** | PaaS                  | Developer-friendly, affordable  |

**General deployment steps (any cloud VM):**

```bash
# 1. SSH into your server
ssh user@your-server-ip

# 2. Install Docker
curl -fsSL https://get.docker.com | sh

# 3. Clone your repo
git clone https://github.com/yourname/my-ai-agent.git
cd my-ai-agent

# 4. Set environment variables
cp .env.example .env
nano .env   # Fill in your secrets

# 5. Deploy
docker compose up -d --build
```

---

### 11.4 Production Checklist

- [ ] Secrets stored in environment variables or a secrets manager (never in code)
- [ ] CORS origins restricted to your actual frontend domain
- [ ] API authentication enabled (API key, JWT, or OAuth)
- [ ] Rate limiting configured (use `slowapi` for FastAPI)
- [ ] HTTPS enabled (use Nginx + Let's Encrypt, or cloud load balancer)
- [ ] Health check endpoint monitored
- [ ] Logging configured (structured JSON logs)
- [ ] Milvus data volume persisted
- [ ] Backups scheduled

---

## 12. Learning Roadmap

### Suggested Timeline

| Week | Focus                                                              |
| ---- | ------------------------------------------------------------------ |
| 1–2  | Python fundamentals: syntax, classes, decorators, `async/await`    |
| 3    | FastAPI: routes, Pydantic models, middleware, dependency injection |
| 4    | Docker: Dockerfile, Compose, volumes, networking                   |
| 5    | LangChain basics: LLM calls, prompt templates, simple chains       |
| 6    | Milvus: setup, embeddings, ingestion, similarity search            |
| 7–8  | Build a full RAG agent: LangChain + FastAPI + Milvus end-to-end    |
| 9    | Testing: pytest, mocking, integration tests                        |
| 10   | Deployment: Docker Compose, cloud provider of choice               |
| 11+  | Advanced: streaming, multi-agent, LangGraph, observability         |

---

### Common Pitfalls for Web Developers

| Pitfall                       | Explanation                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------ |
| **Forgetting `async/await`**  | FastAPI is async — use `async def` and `await` for all I/O operations          |
| **Mutable default arguments** | `def f(items=[])` is a bug in Python — use `def f(items=None)` instead         |
| **Not using type hints**      | Type hints power FastAPI validation and Pylance autocomplete — always use them |
| **Committing `.env`**         | Add `.env` to `.gitignore` immediately — secrets must never be in Git          |
| **Ignoring virtual envs**     | Always activate your `venv` before running or installing anything              |
| **Blocking async loops**      | Don't use `time.sleep()` in async functions — use `await asyncio.sleep()`      |

---

### Useful Resources

| Resource                          | URL                                                           |
| --------------------------------- | ------------------------------------------------------------- |
| LangChain Docs                    | https://python.langchain.com/docs                             |
| FastAPI Docs                      | https://fastapi.tiangolo.com                                  |
| Milvus Docs                       | https://milvus.io/docs                                        |
| Anthropic API Docs                | https://docs.anthropic.com                                    |
| Claude Model Reference            | https://docs.anthropic.com/en/docs/models-overview            |
| LangChain + Anthropic             | https://python.langchain.com/docs/integrations/chat/anthropic |
| Pydantic Docs                     | https://docs.pydantic.dev                                     |
| HuggingFace Sentence Transformers | https://www.sbert.net                                         |

---

_Guide version 2.0 — Stack: Python 3.11 · LangChain 0.2 · FastAPI 0.111 · Milvus 2.4 · Claude Sonnet 4.6_
