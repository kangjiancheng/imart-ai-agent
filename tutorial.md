# Python AI Agent Project Tutorial

A comprehensive guide for building Python AI agent projects using Anthropic Claude, LangChain, FastAPI, and Milvus.

**For Web Developers Transitioning to Python AI Development**

This guide covers: Environment Setup · IDE Configuration · Project Structure · LangChain · FastAPI · Milvus · Deployment

## Prerequisites & Environment Setup

### 1. Install Python

As a web developer new to Python, start with the right Python version. Always use **Python 3.10+** for AI/ML projects (recommended: 3.11 or 3.12).

**Download & Install:**

- Official site: https://www.python.org/downloads/
- macOS: Use Homebrew (`brew install python@3.11`)
- Windows: Check "Add Python to PATH" during installation
- Linux: Use your package manager (`apt install python3.11`)

**Verify installation:**

```bash
python --version      # Should show Python 3.11.x or higher
pip --version         # Should show pip version
```

**Optional: Speed up package installation**

If you experience slow pip downloads, configure a mirror:

```bash
# Using Tsinghua mirror (good for Asia-Pacific region)
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# Or use per-command flag
pip install package -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. Virtual Environments (Essential!)

Always use a virtual environment for each project. This isolates dependencies — just like `node_modules` in Node.js.

**📌 Web Dev Analogy:**

- Virtual env = `node_modules`
- `requirements.txt` = `package.json`
- `pip` = `npm`
- `venv/bin/activate` = setting your project context

**Create and activate:**

```bash
# Create project directory
mkdir my-ai-agent
cd my-ai-agent

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate         # macOS/Linux
venv\Scripts\activate            # Windows

# Deactivate when done
deactivate
```

### 3. Core Tools & Software

| Tool               | Purpose                                                   |
| ------------------ | --------------------------------------------------------- |
| **Git**            | Version control — https://git-scm.com                     |
| **Docker Desktop** | For running Milvus vector DB locally — https://docker.com |
| **Postman/Bruno**  | API testing tool for FastAPI endpoints                    |
| **VS Code**        | Recommended IDE (see IDE section below)                   |

## IDE Recommendations

### VS Code (Recommended for Web Developers)

Since you come from web development, **VS Code** is the best choice. It's free, familiar, and has excellent Python support.

**Download:** https://code.visualstudio.com

**Essential VS Code Extensions:**

| Extension              | Purpose                                                         |
| ---------------------- | --------------------------------------------------------------- |
| **Python** (Microsoft) | Core Python extension — syntax, linting, IntelliSense, debugger |
| **Pylance**            | Advanced Python type checking and autocomplete                  |
| **Python Debugger**    | Visual debugger — set breakpoints, inspect variables            |
| **Ruff**               | Fast Python linter and formatter (replaces flake8 + black)      |
| **Docker**             | Manage Docker containers from VS Code sidebar                   |
| **REST Client**        | Test FastAPI endpoints directly in VS Code (.http files)        |
| **GitLens**            | Enhanced Git blame and history visualization                    |

**VS Code Settings for Python:**

Create `.vscode/settings.json` in your project:

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff"
  }
}
```

### Alternative IDEs

| IDE                     | When to Use                                                                                                                |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **PyCharm** (JetBrains) | Most powerful Python IDE. Professional edition has AI features. Free Community edition available. Good for large projects. |
| **Cursor**              | AI-powered code editor (based on VS Code). Excellent for AI agent development — built-in Claude/GPT integration.           |
| **Jupyter Notebook**    | Use for experiments and data exploration. Install: `pip install jupyterlab`                                                |

**✅ Recommendation:** Start with VS Code + Python/Pylance extensions. It mirrors your web dev experience.

### 4. Core Dependencies

Install these packages in your virtual environment:

```bash
# Anthropic Claude SDK
pip install anthropic

# LangChain with Anthropic integration
pip install langchain langchain-anthropic langchain-community

# FastAPI and web server
pip install fastapi uvicorn[standard]

# Vector database client
pip install pymilvus

# Utilities
pip install python-dotenv pydantic

# Development tools
pip install ruff pytest httpx
```

**Package Overview:**

| Package               | Purpose                                       |
| --------------------- | --------------------------------------------- |
| `anthropic`           | Official Anthropic Claude API client          |
| `langchain`           | Framework for building LLM applications       |
| `langchain-anthropic` | LangChain integration for Claude              |
| `fastapi`             | Modern Python web framework (like Express.js) |
| `uvicorn`             | ASGI server to run FastAPI                    |
| `pymilvus`            | Client for Milvus vector database             |
| `python-dotenv`       | Load environment variables from .env files    |
| `pydantic`            | Data validation (used by FastAPI)             |
| `ruff`                | Fast Python linter and formatter              |

## Project Structure

### Recommended Folder Structure

Here is a well-organized structure for a LangChain + FastAPI + Milvus AI agent project:

```
my-ai-agent/
├── .env                      # API keys (never commit!)
├── .gitignore
├── requirements.txt          # Python dependencies
├── README.md
├── docker/
│   └── docker-compose.yml    # Start Milvus locally
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point
│   ├── agents/              # LangChain agent logic
│   │   ├── __init__.py
│   │   └── claude_agent.py  # Claude agent implementation
│   ├── api/                 # API route handlers
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── core/                # Config, settings
│   │   ├── __init__.py
│   │   └── config.py
│   ├── services/            # Business logic
│   │   ├── __init__.py
│   │   ├── vector_service.py  # Milvus operations
│   │   └── llm_service.py     # Claude API calls
│   └── models/              # Pydantic data models
│       ├── __init__.py
│       └── schemas.py
└── tests/
    ├── __init__.py
    └── test_agent.py
```

### Key Configuration Files

**1. `.env` file (API Keys & Config):**

```bash
# Anthropic Claude API Key
ANTHROPIC_API_KEY=your-api-key-here

# Milvus Configuration
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Application Settings
APP_ENV=development
LOG_LEVEL=INFO
```

**⚠️ Important:** Never commit `.env` to git! Add it to `.gitignore`.

**2. `requirements.txt`:**

```txt
anthropic==0.39.0
langchain==0.3.0
langchain-anthropic==0.3.0
langchain-community==0.3.0
fastapi==0.115.0
uvicorn[standard]==0.32.0
pymilvus==2.4.0
python-dotenv==1.0.0
pydantic==2.9.0
pydantic-settings==2.5.0
```

**📌 Tip:** Pin your package versions (use `==`) to avoid breaking changes. Generate with: `pip freeze > requirements.txt`

**3. `.gitignore`:**

```
# Virtual environment
venv/
env/
.venv/

# Environment variables
.env
.env.local

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

## Quick Start Code

### 1. Get Your Anthropic API Key

Visit https://console.anthropic.com and create an API key. Add it to your `.env` file:

```bash
ANTHROPIC_API_KEY=sk-ant-api03-...your-key-here
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

### 2. Basic FastAPI App with Claude

Create `app/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="AI Agent API",
    version="1.0.0",
    description="Claude-powered AI Agent"
)

# CORS middleware (like in Express.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Claude
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0.7,
    max_tokens=1024
)

# Request model
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

@app.get("/")
async def root():
    return {"message": "AI Agent API is running", "model": "claude-sonnet-4-6"}

@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat endpoint using Claude"""
    response = llm.invoke(request.message)
    return {
        "response": response.content,
        "session_id": request.session_id
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### 3. Run the Development Server

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**View auto-generated API docs:**

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 4. Test Your API

Using curl:

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is AI?"}'
```

Or create a test file `test.http` (use REST Client extension in VS Code):

```http
### Test root endpoint
GET http://localhost:8000/

### Test chat endpoint
POST http://localhost:8000/chat
Content-Type: application/json

{
  "message": "Explain Python virtual environments in one sentence",
  "session_id": "test-123"
}
```

## LangChain — Building AI Agents with Claude

### What is LangChain?

LangChain is a framework for building applications powered by Large Language Models (LLMs). Think of it like **Express.js but for AI** — it provides components for chains, memory, tools, agents, and RAG (Retrieval-Augmented Generation).

**📌 Web Dev Analogy:**

| LangChain Component | Web Dev Equivalent                               |
| ------------------- | ------------------------------------------------ |
| LLM / Chat Model    | External API (like Stripe, Twilio)               |
| Prompt Template     | Template engine (Handlebars, EJS)                |
| Chain               | Middleware pipeline                              |
| Agent               | Smart router that decides which endpoint to call |
| Memory              | Session storage                                  |
| Vector Store        | Database with semantic search                    |

### Key LangChain Components

| Component            | Description                                        |
| -------------------- | -------------------------------------------------- |
| **LLM / Chat Model** | The AI brain — connects to Claude, GPT, etc.       |
| **Prompt Template**  | Reusable prompt structures with variables          |
| **Chain**            | Pipeline connecting prompts + LLM + tools          |
| **Agent**            | LLM that can decide which tools to use dynamically |
| **Memory**           | Store conversation history (like sessions)         |
| **Retriever**        | Fetch relevant docs from a vector store (Milvus)   |
| **Embeddings**       | Convert text to vectors for semantic search        |
| **Vector Store**     | Database for vectors — you'll use Milvus           |

### Simple LangChain Chain with Claude

Create `app/agents/claude_agent.py`:

```python
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Claude
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
    temperature=0.7,
    max_tokens=2048
)

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant specialized in {domain}."),
    ("human", "{question}")
])

# Create a chain: prompt -> LLM -> output parser
chain = prompt | llm | StrOutputParser()

# Use the chain
def ask_question(question: str, domain: str = "general knowledge"):
    result = chain.invoke({
        'question': question,
        'domain': domain
    })
    return result

# Example usage
if __name__ == "__main__":
    answer = ask_question(
        "What are the benefits of using virtual environments?",
        "Python development"
    )
    print(answer)
```

### Conversation Memory with Claude

Add conversation history to maintain context:

```python
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')
)

# Add memory to remember conversation
memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Multi-turn conversation
response1 = conversation.predict(input="My name is Alice")
print(response1)

response2 = conversation.predict(input="What's my name?")
print(response2)  # Claude will remember: "Your name is Alice"
```

### LangChain Agent with Tools

Create an agent that can use tools:

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_anthropic import ChatAnthropic
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate
import datetime

# Define custom tools
def get_current_time(query: str) -> str:
    """Returns the current time"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate(expression: str) -> str:
    """Evaluates a mathematical expression"""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

# Create tool list
tools = [
    Tool(
        name="GetTime",
        func=get_current_time,
        description="Get the current date and time"
    ),
    Tool(
        name="Calculator",
        func=calculate,
        description="Calculate mathematical expressions. Input should be a valid Python expression."
    )
]

# Initialize Claude
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')
)

# Create prompt for agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to tools."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Create agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run agent
result = agent_executor.invoke({
    "input": "What time is it? Also calculate 25 * 4"
})
print(result['output'])
```

## Milvus — Vector Database Setup

### What is Milvus?

Milvus is an open-source vector database used for storing and searching embeddings (numerical representations of text/images). It's the **memory store for your RAG pipeline**.

**📌 Analogy:**

- Text → Embedding Model → Vector (float array) → Stored in Milvus
- When user asks a question, the question is vectorized and Milvus finds the most similar stored vectors (documents)

### Option 1: Docker (Recommended)

Create `docker/docker-compose.yml`:

```yaml
version: "3.8"

services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - etcd_data:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - minio_data:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  milvus:
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - milvus_data:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - etcd
      - minio

volumes:
  etcd_data:
  minio_data:
  milvus_data:
```

**Start Milvus:**

```bash
cd docker
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f milvus
```

### Option 2: Milvus Lite (Development Only)

For quick prototyping without Docker:

```bash
pip install milvus

# Use in code - no separate server needed
from milvus import default_server
default_server.start()
```

### Basic Milvus Operations

Create `app/services/vector_service.py`:

```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import os

class VectorService:
    def __init__(self):
        self.connect()

    def connect(self):
        """Connect to Milvus"""
        connections.connect(
            alias="default",
            host=os.getenv("MILVUS_HOST", "localhost"),
            port=os.getenv("MILVUS_PORT", "19530")
        )
        print("Connected to Milvus")

    def create_collection(self, collection_name: str, dim: int = 1536):
        """Create a collection for storing vectors"""
        if utility.has_collection(collection_name):
            print(f"Collection {collection_name} already exists")
            return Collection(collection_name)

        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]

        schema = CollectionSchema(fields, description="Document store")
        collection = Collection(name=collection_name, schema=schema)

        # Create index for fast search
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)

        print(f"Created collection: {collection_name}")
        return collection

    def insert_documents(self, collection_name: str, texts: list, embeddings: list):
        """Insert documents with embeddings"""
        collection = Collection(collection_name)

        data = [
            texts,
            embeddings
        ]

        collection.insert(data)
        collection.flush()
        print(f"Inserted {len(texts)} documents")

    def search(self, collection_name: str, query_embedding: list, top_k: int = 3):
        """Search for similar vectors"""
        collection = Collection(collection_name)
        collection.load()

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text"]
        )

        return results

# Example usage
if __name__ == "__main__":
    service = VectorService()
    collection = service.create_collection("documents", dim=1536)
```

## RAG (Retrieval-Augmented Generation) with Claude + Milvus

Combine LangChain, Claude, and Milvus for a complete RAG system:

```python
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import os

# Initialize embeddings (using free HuggingFace model)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize Claude
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
    temperature=0.3
)

# Connect to Milvus
vector_store = Milvus(
    embedding_function=embeddings,
    collection_name="knowledge_base",
    connection_args={
        "host": os.getenv("MILVUS_HOST", "localhost"),
        "port": os.getenv("MILVUS_PORT", "19530")
    }
)

# Add documents to vector store
def add_documents(texts: list[str]):
    """Split and add documents to Milvus"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = [Document(page_content=text) for text in texts]
    split_docs = text_splitter.split_documents(docs)

    vector_store.add_documents(split_docs)
    print(f"Added {len(split_docs)} document chunks")

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# Query the RAG system
def ask_question(question: str):
    result = qa_chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": [doc.page_content for doc in result["source_documents"]]
    }

# Example usage
if __name__ == "__main__":
    # Add some knowledge
    documents = [
        "Python virtual environments isolate project dependencies.",
        "FastAPI is a modern Python web framework for building APIs.",
        "LangChain helps build applications with large language models."
    ]
    add_documents(documents)

    # Ask a question
    response = ask_question("What is FastAPI?")
    print("Answer:", response["answer"])
    print("Sources:", response["sources"])
```

## FastAPI — Building Production-Ready APIs

### FastAPI vs Express.js Comparison

| Concept          | Express.js (Node)     | FastAPI (Python)                 |
| ---------------- | --------------------- | -------------------------------- |
| Route definition | `app.get('/path')`    | `@app.get('/path')`              |
| Request body     | `req.body`            | Pydantic model parameter         |
| Response         | `res.json({...})`     | `return {...}` (auto-serialized) |
| Middleware       | `app.use(middleware)` | Dependencies / Middleware        |
| Async support    | `async/await`         | `async def`                      |
| API docs         | Manual (Swagger)      | Auto-generated at `/docs`        |
| Validation       | Manual (joi, yup)     | Built-in (Pydantic)              |

### Complete FastAPI Application

Create `app/api/routes.py`:

```python
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional
from app.agents.claude_agent import ask_question
from app.services.vector_service import VectorService

router = APIRouter()

# Request/Response models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default")
    domain: Optional[str] = Field(default="general knowledge")

class ChatResponse(BaseModel):
    response: str
    session_id: str
    model: str = "claude-sonnet-4-6"

class DocumentRequest(BaseModel):
    texts: list[str] = Field(..., min_items=1)

# Dependency injection
def get_vector_service():
    return VectorService()

# Routes
@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with Claude AI"""
    try:
        response = ask_question(request.message, request.domain)
        return ChatResponse(
            response=response,
            session_id=request.session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/documents/add")
async def add_documents(
    request: DocumentRequest,
    vector_service: VectorService = Depends(get_vector_service)
):
    """Add documents to vector store"""
    try:
        # This would integrate with your RAG system
        return {
            "message": f"Added {len(request.texts)} documents",
            "count": len(request.texts)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ai-agent"}
```

Update `app/main.py` to include routes:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Agent API",
    version="1.0.0",
    description="Claude-powered AI Agent with RAG capabilities",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1", tags=["AI Agent"])

@app.on_event("startup")
async def startup_event():
    logger.info("Starting AI Agent API...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down AI Agent API...")

@app.get("/")
async def root():
    return {
        "message": "AI Agent API",
        "version": "1.0.0",
        "docs": "/docs"
    }
```

## Deployment Options

### Docker Deployment (Recommended)

**1. Create `Dockerfile`:**

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**2. Create `docker-compose.yml` (for local development):**

```yaml
version: "3.8"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - MILVUS_HOST=milvus
      - MILVUS_PORT=19530
    depends_on:
      - milvus
    volumes:
      - ./app:/app/app
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  milvus:
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "standalone"]
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
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000

  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    command: minio server /minio_data
```

**3. Build and run:**

```bash
# Build image
docker build -t ai-agent:latest .

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop
docker-compose down
```

### Cloud Deployment Options

#### 1. AWS Deployment

**Using AWS ECS (Elastic Container Service):**

```bash
# Build and tag for ECR
docker build -t ai-agent:latest .
docker tag ai-agent:latest <account-id>.dkr.ecr.<region>.amazonaws.com/ai-agent:latest

# Push to ECR
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/ai-agent:latest

# Deploy to ECS (use AWS Console or CLI)
```

**Using AWS Lambda + API Gateway (Serverless):**

```python
# Use Mangum adapter for FastAPI on Lambda
from mangum import Mangum
from app.main import app

handler = Mangum(app)
```

#### 2. Google Cloud Platform

**Using Cloud Run:**

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/<project-id>/ai-agent
gcloud run deploy ai-agent \
  --image gcr.io/<project-id>/ai-agent \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### 3. Azure Deployment

**Using Azure Container Instances:**

```bash
az container create \
  --resource-group myResourceGroup \
  --name ai-agent \
  --image <registry>/ai-agent:latest \
  --dns-name-label ai-agent \
  --ports 8000
```

#### 4. Traditional VPS (DigitalOcean, Linode, etc.)

**Setup with systemd:**

Create `/etc/systemd/system/ai-agent.service`:

```ini
[Unit]
Description=AI Agent API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/ai-agent
Environment="PATH=/opt/ai-agent/venv/bin"
EnvironmentFile=/opt/ai-agent/.env
ExecStart=/opt/ai-agent/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

**Enable and start:**

```bash
sudo systemctl enable ai-agent
sudo systemctl start ai-agent
sudo systemctl status ai-agent
```

**Nginx reverse proxy configuration:**

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Environment Variables in Production

**Never commit secrets!** Use these approaches:

1. **AWS Secrets Manager:**

```python
import boto3
import json

def get_secret(secret_name):
    client = boto3.client('secretsmanager', region_name='us-east-1')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])
```

2. **Docker Secrets:**

```yaml
services:
  api:
    secrets:
      - anthropic_api_key

secrets:
  anthropic_api_key:
    external: true
```

3. **Environment Variables (Cloud Platforms):**

- Set via cloud provider console or CLI
- Use `.env` files only for local development

## Learning Path

### Recommended Timeline

| Timeline     | Topic                  | Key Activities                                                  |
| ------------ | ---------------------- | --------------------------------------------------------------- |
| **Week 1-2** | Python Fundamentals    | Syntax, functions, classes, async/await, decorators, type hints |
| **Week 3**   | FastAPI Basics         | Routes, Pydantic models, middleware, async endpoints, auto docs |
| **Week 4**   | Docker & Containers    | Dockerfile, docker-compose, volumes, networking                 |
| **Week 5**   | LangChain Fundamentals | LLM calls, Prompt Templates, simple chains, output parsers      |
| **Week 6**   | Vector Databases       | Milvus setup, embeddings, similarity search, indexing           |
| **Week 7-8** | RAG Implementation     | Build complete RAG: LangChain + Claude + Milvus                 |
| **Week 9**   | Production Deployment  | Docker deployment, cloud platforms, monitoring                  |
| **Week 10+** | Advanced Topics        | Streaming responses, conversation memory, multi-agent systems   |

### Week-by-Week Breakdown

**Weeks 1-2: Python Fundamentals**

- Learn Python syntax and data structures
- Understand async/await (different from JavaScript!)
- Practice with decorators and context managers
- Get comfortable with type hints

**Week 3: FastAPI**

- Build REST APIs with FastAPI
- Learn Pydantic for data validation
- Implement middleware and dependencies
- Explore auto-generated API documentation

**Week 4: Docker**

- Containerize your FastAPI application
- Use docker-compose for multi-service apps
- Understand volumes and networking
- Practice with Milvus in Docker

**Week 5: LangChain**

- Connect to Claude API
- Create prompt templates
- Build simple chains
- Understand LangChain components

**Week 6: Vector Databases**

- Set up Milvus
- Generate embeddings
- Perform similarity searches
- Optimize vector indexes

**Weeks 7-8: Complete RAG System**

- Integrate all components
- Build document ingestion pipeline
- Implement semantic search
- Create conversational interface

**Week 9: Deployment**

- Deploy to cloud platform
- Set up CI/CD pipeline
- Configure monitoring and logging
- Implement security best practices

## Essential Resources

### Official Documentation

- **Anthropic Claude API:** https://docs.anthropic.com
  - API reference, best practices, prompt engineering
- **LangChain:** https://python.langchain.com/docs
  - Comprehensive guides, examples, integrations
- **FastAPI:** https://fastapi.tiangolo.com
  - Tutorial, user guide, advanced features
- **Milvus:** https://milvus.io/docs
  - Installation, operations, performance tuning
- **Pydantic:** https://docs.pydantic.dev
  - Data validation, settings management

### Learning Resources

**Python for Web Developers:**

- Real Python (realpython.com) - Excellent tutorials
- Python Official Tutorial (docs.python.org/3/tutorial)
- "Fluent Python" by Luciano Ramalho (book)

**AI/ML Fundamentals:**

- Anthropic Prompt Engineering Guide
- LangChain Cookbook (github.com/langchain-ai/langchain)
- "Designing Data-Intensive Applications" by Martin Kleppmann

**Video Courses:**

- FastAPI - Full Course (freeCodeCamp on YouTube)
- LangChain Crash Course (various on YouTube)
- Docker for Beginners (Docker official)

### Community & Support

- **LangChain Discord:** Active community for questions
- **FastAPI Discord:** Great for API-related questions
- **Anthropic Discord:** Claude API support and discussions
- **Stack Overflow:** Tag questions with `langchain`, `fastapi`, `anthropic`

## Common Pitfalls for Web Developers

### 1. Async/Await Differences

**JavaScript:**

```javascript
async function fetchData() {
  const response = await fetch(url);
  return response.json();
}
```

**Python:**

```python
async def fetch_data():
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

# Must run with asyncio
import asyncio
asyncio.run(fetch_data())
```

**Key Difference:** Python's async requires an event loop. In FastAPI, this is handled automatically.

### 2. Indentation Matters

Python uses indentation instead of braces:

```python
# Correct
def my_function():
    if condition:
        do_something()
    else:
        do_other_thing()

# Wrong - IndentationError
def my_function():
if condition:
    do_something()
```

**Tip:** Use 4 spaces (not tabs). Configure your IDE to insert spaces.

### 3. Mutable Default Arguments

```python
# WRONG - Bug!
def add_item(item, items=[]):
    items.append(item)
    return items

# Correct
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

### 4. Global Interpreter Lock (GIL)

Python threads don't parallelize CPU work due to the GIL.

- **For I/O-bound tasks** (API calls, database): Use `async/await`
- **For CPU-bound tasks** (data processing): Use `multiprocessing`

### 5. Import System

```python
# Absolute imports (preferred)
from app.services.vector_service import VectorService

# Relative imports (use sparingly)
from ..services.vector_service import VectorService

# Avoid circular imports by restructuring code
```

### 6. Type Hints Are Optional (But Use Them!)

```python
# Without type hints (works but not recommended)
def process_data(data):
    return data.upper()

# With type hints (better)
def process_data(data: str) -> str:
    return data.upper()
```

Type hints power FastAPI's validation and IDE autocomplete.

### 7. String Formatting

```python
# Old way (avoid)
message = "Hello, %s" % name

# Better
message = "Hello, {}".format(name)

# Best (f-strings)
message = f"Hello, {name}"
```

## Troubleshooting

### Common Issues

**1. Import Errors**

```bash
# Problem: ModuleNotFoundError
# Solution: Ensure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Check Python path
which python  # Should point to venv/bin/python
```

**2. API Key Errors**

```bash
# Problem: anthropic.AuthenticationError
# Solution: Check .env file
cat .env  # Verify ANTHROPIC_API_KEY is set

# Load environment variables
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('ANTHROPIC_API_KEY'))"
```

**3. Milvus Connection Errors**

```bash
# Problem: Cannot connect to Milvus
# Solution: Check if Milvus is running
docker ps | grep milvus

# Check Milvus logs
docker logs <milvus-container-id>

# Restart Milvus
docker-compose restart milvus
```

**4. Port Already in Use**

```bash
# Problem: Address already in use
# Solution: Find and kill process
lsof -i :8000  # Find process using port 8000
kill -9 <PID>  # Kill the process

# Or use different port
uvicorn app.main:app --port 8001
```

**5. Slow Package Installation**

```bash
# Problem: pip install is very slow
# Solution: Use a mirror (especially in Asia-Pacific)
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package_name

# Or configure globally
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

**6. Docker Build Fails**

```bash
# Problem: Docker build fails with dependency errors
# Solution: Clear Docker cache
docker builder prune

# Rebuild without cache
docker build --no-cache -t ai-agent:latest .
```

## Production Checklist

Before deploying to production:

### Security

- [ ] API keys stored in secrets manager (not in code)
- [ ] CORS configured with specific origins (not `*`)
- [ ] Rate limiting implemented
- [ ] Input validation on all endpoints
- [ ] HTTPS/TLS enabled
- [ ] Authentication/authorization implemented

### Performance

- [ ] Database connection pooling configured
- [ ] Caching strategy implemented
- [ ] Async operations used for I/O
- [ ] Vector index optimized for your data size
- [ ] Load testing completed

### Monitoring

- [ ] Logging configured (structured logs)
- [ ] Health check endpoints added
- [ ] Metrics collection (Prometheus/CloudWatch)
- [ ] Error tracking (Sentry/Rollbar)
- [ ] Uptime monitoring

### Reliability

- [ ] Error handling for all external calls
- [ ] Retry logic with exponential backoff
- [ ] Circuit breakers for failing services
- [ ] Graceful shutdown handling
- [ ] Database backups configured

### Documentation

- [ ] API documentation complete (auto-generated by FastAPI)
- [ ] README with setup instructions
- [ ] Environment variables documented
- [ ] Deployment guide written
- [ ] Architecture diagram created

## Next Steps

1. **Set up your development environment**
   - Install Python 3.11+
   - Set up VS Code with Python extensions
   - Create a virtual environment

2. **Get your Anthropic API key**
   - Visit https://console.anthropic.com
   - Create an account and generate an API key
   - Add to `.env` file

3. **Build a simple FastAPI + Claude integration**
   - Follow the Quick Start section
   - Test with curl or Postman
   - Explore the auto-generated docs at `/docs`

4. **Add Milvus for RAG**
   - Start Milvus with Docker
   - Implement document ingestion
   - Build semantic search

5. **Deploy to production**
   - Containerize with Docker
   - Choose a cloud platform
   - Set up CI/CD pipeline

## Additional Resources

### Example Projects

- **LangChain Templates:** github.com/langchain-ai/langchain/tree/master/templates
- **FastAPI Examples:** github.com/tiangolo/fastapi/tree/master/docs_src
- **Milvus Bootcamp:** github.com/milvus-io/bootcamp

### Tools & Libraries

- **Ruff:** Fast Python linter and formatter
- **pytest:** Testing framework
- **httpx:** Modern HTTP client (async support)
- **loguru:** Better logging
- **tenacity:** Retry library with exponential backoff

---

**Happy coding!** 🚀

You now have a comprehensive guide to building Python AI agents with Claude, LangChain, FastAPI, and Milvus. Start with the basics, build incrementally, and don't hesitate to consult the official documentation when you get stuck.
