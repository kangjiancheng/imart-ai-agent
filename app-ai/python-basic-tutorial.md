# Python Quick Starts

A reference guide for JavaScript/TypeScript developers learning Python through this project.
Each section explains one Python concept with comparisons to JS/TS equivalents.

---

## Table of Contents

- [Python Quick Starts](#python-quick-starts)
  - [Table of Contents](#table-of-contents)
  - [0. Project setup — prerequisites and launch](#0-project-setup--prerequisites-and-launch)
    - [0.1 pyenv — Python version manager](#01-pyenv--python-version-manager)
    - [0.2 venv — virtual environment](#02-venv--virtual-environment)
    - [0.3 pip — package manager](#03-pip--package-manager)
    - [0.4 .env — environment variables](#04-env--environment-variables)
    - [0.5 uvicorn — ASGI server (how you launch the app)](#05-uvicorn--asgi-server-how-you-launch-the-app)
    - [0.6 Full setup sequence (run top to bottom, once)](#06-full-setup-sequence-run-top-to-bottom-once)
    - [0.7 Tool summary](#07-tool-summary)
  - [1. Project structure — `__init__.py`](#1-project-structure--__init__py)
  - [2. Variables and types](#2-variables-and-types)
  - [3. f-strings — template literals](#3-f-strings--template-literals)
  - [4. Functions — def and return](#4-functions--def-and-return)
  - [5. async / await](#5-async--await)
  - [6. Generators and yield](#6-generators-and-yield)
  - [7. Classes and self](#7-classes-and-self)
  - [8. Decorators — @tool, @dataclass, @property](#8-decorators--tool-dataclass-property)
  - [9. Type hints](#9-type-hints)
  - [10. Truthiness and None](#10-truthiness-and-none)
  - [11. Lists — comprehensions, enumerate, join](#11-lists--comprehensions-enumerate-join)
  - [12. Dicts — get, spread, unpacking](#12-dicts--get-spread-unpacking)
  - [13. try / except](#13-try--except)
  - [14. Imports — module system](#14-imports--module-system)
  - [15. Pydantic — validation and settings](#15-pydantic--validation-and-settings)
  - [16. Docstrings](#16-docstrings)
  - [17. `...` Ellipsis — required field marker](#17--ellipsis--required-field-marker)
  - [18. `"""` Triple quotes — multi-line strings](#18--triple-quotes--multi-line-strings)
  - [19. Quick comparison table](#19-quick-comparison-table)

---

## 0. Project setup — prerequisites and launch

Before running any Python project, you need four things: the right Python version, an isolated environment, installed packages, and environment variables. Here is each tool, what it does, and why you need it — compared to the Node.js toolchain you already know.

---

### 0.1 pyenv — Python version manager

**What it is:** Installs and switches between Python versions per project.

**JS/TS equivalent:** `nvm` (Node Version Manager)

```bash
# Install pyenv (macOS)
brew install pyenv

# Install a specific Python version
pyenv install 3.11.9

# Set the version for this project folder only
pyenv local 3.11.9     # writes a .python-version file — git-commit this

# Verify
python --version        # → Python 3.11.9
```

**Why you need it:** Your system Python is often outdated or used by the OS itself. `pyenv` lets you pin a version per project without touching the system Python — exactly like `.nvmrc` + `nvm use`.

| pyenv             | nvm                  |
| ----------------- | -------------------- |
| `pyenv install`   | `nvm install`        |
| `pyenv local X`   | `.nvmrc` + `nvm use` |
| `.python-version` | `.nvmrc`             |

---

### 0.2 venv — virtual environment

**What it is:** Creates an isolated folder (`venv/`) where packages are installed for this project only.

**JS/TS equivalent:** `node_modules/` — but in Node every project already gets its own `node_modules`. In Python, packages install globally by default unless you create a venv first.

```bash
# Create the virtual environment (run once per project)
python -m venv venv
#               ^^^^
#               folder name — conventionally "venv" or ".venv"

# Activate it — your shell prompt will show (venv) prefix
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# Deactivate when done
deactivate
```

**Why you need it:** Without activation, `pip install` puts packages into a global Python folder shared by every project on your machine — version conflicts become guaranteed. With venv, each project has its own isolated site-packages.

```
# Project layout after venv creation:
app-ai/
├── venv/              ← isolated Python + packages (git-ignore this)
│   ├── bin/python
│   └── lib/site-packages/
├── src/
└── requirements.txt
```

> **Always add `venv/` to `.gitignore`.** It is like `node_modules/` — never commit it.

---

### 0.3 pip — package manager

**What it is:** Installs Python packages from PyPI (Python Package Index).

**JS/TS equivalent:** `npm` or `yarn` or `bun`

```bash
# Install a single package
pip install fastapi

# Install everything listed in requirements.txt
pip install -r requirements.txt

# See what is installed (like npm list)
pip list

# Save current environment to requirements.txt
pip freeze > requirements.txt
```

**`requirements.txt` vs `package.json`:**

| Python                  | Node.js                             |
| ----------------------- | ----------------------------------- |
| `requirements.txt`      | `package.json` (dependencies only)  |
| `pip install -r req...` | `npm install`                       |
| `pip freeze > req...`   | auto-updated by `npm install <pkg>` |
| no lock file by default | `package-lock.json` / `yarn.lock`   |

> **Tip:** `pip freeze` pins exact versions (e.g. `fastapi==0.111.0`). Commit this file so teammates install the exact same versions.

**This project's `requirements.txt`:**

```text
fastapi
uvicorn[standard]
langchain-anthropic
langchain-core
pydantic-settings
pymilvus
openai
python-dotenv
```

---

### 0.4 .env — environment variables

**What it is:** A file containing secret API keys and config values that should not be committed to git.

**JS/TS equivalent:** `.env` loaded by `dotenv` — identical concept.

```bash
# Copy the template, then fill in your real keys
cp .env.example .env
```

```dotenv
# .env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

**How Python reads it:** This project uses `pydantic-settings` (see [section 15](#15-pydantic--validation-and-settings)). It reads `.env` automatically when `Settings()` is instantiated — no `require('dotenv').config()` call needed.

> **Always add `.env` to `.gitignore`.** Commit `.env.example` with placeholder values instead.

---

### 0.5 uvicorn — ASGI server (how you launch the app)

**What it is:** The production-grade server that runs FastAPI apps. Handles HTTP, async I/O, and hot-reload for development.

**JS/TS equivalent:** `node server.js` or `ts-node` — the runtime that actually serves requests.

```bash
# Development — hot-reload on file save
uvicorn src.main:app --reload --port 8000
#        ^^^^^^^^^  ^^^
#        module     FastAPI app object inside that module
#        path       (src/main.py → `app = FastAPI()`)

# Production — multiple workers, no reload
uvicorn src.main:app --workers 4 --port 8000
```

**Why `src.main:app` not `src/main.py`?**

Python uses dot-separated module paths, not file paths. `src.main` means "the `main` module inside the `src` package." The `:app` after the colon tells uvicorn which variable inside that module is the FastAPI instance.

```python
# src/main.py
from fastapi import FastAPI
app = FastAPI()   # ← uvicorn looks for this name
```

---

### 0.6 Full setup sequence (run top to bottom, once)

```bash
# 1. Clone the repo and enter the project
cd app-ai

# 2. Install the right Python version
pyenv install 3.11.9    # if not already installed
pyenv local 3.11.9

# 3. Create and activate the virtual environment
python -m venv venv
source venv/bin/activate     # prompt changes to (venv)

# 4. Install dependencies
pip install -r requirements.txt

# 5. Set up environment variables
cp .env.example .env
# → open .env and fill in your API keys

# 6. Launch the server
uvicorn src.main:app --reload --port 8000
# → http://localhost:8000
# → http://localhost:8000/docs  (auto-generated API docs)
```

**After the first setup**, your daily workflow is just:

```bash
cd app-ai
source venv/bin/activate
uvicorn src.main:app --reload --port 8000
```

---

### 0.7 Tool summary

| Tool               | Role                         | JS/TS equivalent     |
| ------------------ | ---------------------------- | -------------------- |
| `pyenv`            | Manage Python versions       | `nvm`                |
| `venv`             | Isolate packages per project | `node_modules` scope |
| `pip`              | Install packages             | `npm install`        |
| `requirements.txt` | List dependencies            | `package.json`       |
| `.env`             | Secret config values         | `.env` + `dotenv`    |
| `uvicorn`          | Serve the FastAPI app        | `node` / `ts-node`   |

---

## 1. Project structure — `__init__.py`

Every folder in this project contains an empty `__init__.py` file:

```
src/
├── __init__.py          ← empty
├── guardrails/
│   ├── __init__.py      ← empty
│   ├── checker.py
│   └── content_policy.py
```

**Why?** Python requires a file named `__init__.py` inside a folder to treat it as a "package" — meaning you can import from it. Without it, Python refuses to import anything from that folder.

```python
# Without __init__.py in src/guardrails/:
from src.guardrails.checker import GuardrailChecker
# → ModuleNotFoundError: No module named 'src.guardrails'

# With __init__.py present (even empty):
from src.guardrails.checker import GuardrailChecker
# → works
```

**JS/TS comparison:**

| Python                        | JavaScript/TypeScript                    |
| ----------------------------- | ---------------------------------------- |
| `__init__.py` (empty)         | no equivalent — any folder is importable |
| `__init__.py` with re-exports | `index.ts` with re-exports               |

**Can `__init__.py` have code?** Yes — you can use it to create import shortcuts:

```python
# src/guardrails/__init__.py
from .checker import GuardrailChecker  # re-export

# Then callers can write the shorter form:
from src.guardrails import GuardrailChecker
# instead of:
from src.guardrails.checker import GuardrailChecker
```

This project keeps them empty because full import paths are clear enough.

---

## 2. Variables and types

Python is dynamically typed. Type hints are optional but recommended (like TypeScript's types).

```python
# No `const`, `let`, or `var` — just assign
name = "Claude"
count = 42
score = 0.72
is_active = True
nothing = None        # equivalent to null / undefined

# Type hints (optional, for clarity and IDE support)
name: str = "Claude"
count: int = 42
score: float = 0.72
is_active: bool = True
nothing: str | None = None   # string OR null (Python 3.10+)
```

**Key type differences:**

| Python           | TypeScript                  |
| ---------------- | --------------------------- |
| `str`            | `string`                    |
| `int`            | `number` (whole)            |
| `float`          | `number` (decimal)          |
| `bool`           | `boolean`                   |
| `None`           | `null` / `undefined`        |
| `str \| None`    | `string \| null`            |
| `list[str]`      | `string[]`                  |
| `dict[str, int]` | `{ [key: string]: number }` |

---

## 3. f-strings — template literals

```python
# Python f-string
name = "Claude"
model = "claude-sonnet-4-6"
message = f"Hello from {name}, running {model}"

# Multi-line f-string
prompt = f"""
You are {name}.
Model: {model}
"""

# Expressions inside {}
price = 47.80
tip = f"Tip: {price * 0.15:.2f}"   # :.2f = format to 2 decimal places
```

```typescript
// TypeScript equivalent
const message = `Hello from ${name}, running ${model}`;
const prompt = `
You are ${name}.
Model: ${model}
`;
const tip = `Tip: ${(price * 0.15).toFixed(2)}`;
```

---

## 4. Functions — def and return

```python
# Basic function
def greet(name: str) -> str:
    return f"Hello, {name}!"

# Default parameter values
def search(query: str, top_k: int = 5) -> list[str]:
    return []

# Calling with keyword arguments (named args)
search("hello")              # top_k = 5 (default)
search("hello", top_k=3)    # top_k = 3

# Returning multiple values (a tuple)
def check(message: str) -> tuple[bool, str]:
    return True, ""          # returns (True, "")

# Unpacking the tuple
passed, reason = check("hello")
```

```typescript
// TypeScript equivalent
function greet(name: string): string {
  return `Hello, ${name}!`;
}
function search(query: string, topK = 5): string[] {
  return [];
}
function check(message: string): [boolean, string] {
  return [true, ""];
}
const [passed, reason] = check("hello");
```

---

## 5. async / await

Python's async model is the same concept as JavaScript's — non-blocking I/O.

```python
import asyncio

# Declare an async function with `async def`
async def fetch_data(url: str) -> str:
    # `await` pauses this coroutine until the result is ready
    # Other coroutines can run during this pause
    result = await some_io_call(url)
    return result

# Calling an async function — must use await
async def main():
    data = await fetch_data("http://example.com")
    print(data)

# Run the async entry point
asyncio.run(main())
```

```typescript
// TypeScript equivalent
async function fetchData(url: string): Promise<string> {
  const result = await someIoCall(url);
  return result;
}
async function main() {
  const data = await fetchData("http://example.com");
}
```

**Key difference:** In Python, calling `fetch_data(url)` without `await` returns a coroutine object (not the result). You MUST `await` it to run it. Same as calling an async function without `await` in JS returns a `Promise` instead of the value.

---

## 6. Generators and yield

A generator function produces values one at a time using `yield` instead of returning all at once.

```python
# Regular function — computes all, returns everything at once
def get_numbers() -> list[int]:
    return [1, 2, 3]

# Generator function — yields one value at a time
def count_up():
    yield 1
    yield 2
    yield 3

for n in count_up():
    print(n)    # prints 1, then 2, then 3

# Async generator — used in agent_loop.py to stream tokens
async def token_stream():
    async for token in llm.astream(messages):
        yield token        # sends one token to the caller immediately
    yield "[DONE]"

# The caller reads tokens as they arrive
async for token in token_stream():
    print(token)           # prints each token the moment it's yielded
```

```typescript
// TypeScript equivalent
async function* tokenStream() {
  for await (const token of llm.stream(messages)) {
    yield token;
  }
  yield "[DONE]";
}
for await (const token of tokenStream()) {
  console.log(token);
}
```

**Why it matters for this project:** `agent_loop.py`'s `run()` function is an async generator. It yields tokens word-by-word to `routers/agent.py`, which wraps them in SSE format and streams to NestJS in real time — no buffering, no waiting for the full response.

---

## 7. Classes and self

```python
class EmbeddingClient:
    # Class-level constant (shared by all instances)
    MODEL = "text-embedding-3-small"    # like `static readonly` in TS

    # Constructor
    def __init__(self):                 # `self` = `this` in JS/TS
        self.client = AsyncOpenAI()     # instance variable

    # Method — always takes `self` as first parameter
    async def embed(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            model=self.MODEL,           # access class constant via self
            input=text,
        )
        return response.data[0].embedding

# Instantiation (no `new` keyword in Python)
embedder = EmbeddingClient()           # calls __init__
result = await embedder.embed("hello")
```

```typescript
class EmbeddingClient {
  static readonly MODEL = "text-embedding-3-small";
  private client: AsyncOpenAI;

  constructor() {
    this.client = new AsyncOpenAI();
  }

  async embed(text: string): Promise<number[]> {
    const response = await this.client.embeddings.create({
      model: EmbeddingClient.MODEL,
      input: text,
    });
    return response.data[0].embedding;
  }
}
const embedder = new EmbeddingClient();
```

**Key difference:** Python requires `self` explicitly as the first parameter of every method. JavaScript/TypeScript has implicit `this`.

---

## 8. Decorators — @tool, @dataclass, @property

A decorator is a function that wraps another function or class to add behavior. Written with `@` above the target.

```python
# @tool — transforms a plain function into a LangChain Tool object
# LangChain reads the docstring and type hints to build the JSON schema
@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    ...

# @dataclass — auto-generates __init__, __repr__, __eq__
# No need to write a constructor manually
from dataclasses import dataclass

@dataclass
class Document:
    content: str
    source: str
    score: float

# Equivalent WITHOUT @dataclass (what Python generates for you):
class Document:
    def __init__(self, content: str, source: str, score: float):
        self.content = content
        self.source = source
        self.score = score

# @property — makes a method callable as an attribute (no parentheses)
class TokenBudget:
    USABLE_LIMIT = 193_904

    @property
    def remaining(self) -> int:
        return self.USABLE_LIMIT - self.used

budget = TokenBudget()
budget.remaining    # called like an attribute, not budget.remaining()
```

```typescript
// TypeScript equivalent of @property
class TokenBudget {
  get remaining(): number {
    return TokenBudget.USABLE_LIMIT - this.used;
  }
}
```

---

## 9. Type hints

Type hints are Python's equivalent of TypeScript types. They are optional but strongly recommended.

```python
# Basic types
name: str = "hello"
count: int = 42
score: float = 0.72
flag: bool = True

# Optional (can be None)
key: str | None = None          # Python 3.10+
key: Optional[str] = None       # older style (from typing import Optional)

# Lists and dicts
items: list[str] = []
lookup: dict[str, int] = {}

# Tuples (fixed-length, fixed-type)
result: tuple[bool, str] = (True, "")

# Function signatures
def check(message: str) -> tuple[bool, str]:
    ...

# Literal — only specific exact values allowed
from typing import Literal
role: Literal["user", "assistant"] = "user"
```

```typescript
// TypeScript equivalents
let name: string = "hello";
let key: string | null = null;
let items: string[] = [];
let lookup: { [key: string]: number } = {};
let result: [boolean, string] = [true, ""];
type Role = "user" | "assistant";
```

---

## 10. Truthiness and None

Python has the same "falsy" concept as JavaScript, but the values differ.

```python
# Falsy values in Python
None        # like null/undefined
False
0
0.0
""          # empty string
[]          # empty list
{}          # empty dict

# None checks — use `is None`, not `== None`
if value is None:       # correct
    ...
if value is not None:   # correct
    ...
if value == None:       # works but not idiomatic

# Truthiness shortcuts
if not settings.tavily_api_key:     # True when None or ""
    return "not configured"

result = snippets or "No results"   # like JS: snippets || "No results"
value = data.get("key") or "default"
```

```typescript
// TypeScript equivalents
if (value === null || value === undefined) { ... }
if (!settings.tavilyApiKey) { ... }
const result = snippets || "No results"
```

---

## 11. Lists — comprehensions, enumerate, join

```python
# List comprehension — build a list by transforming another
# [expression for item in iterable]
numbers = [1, 2, 3, 4, 5]
doubled = [n * 2 for n in numbers]          # [2, 4, 6, 8, 10]
evens = [n for n in numbers if n % 2 == 0]  # [2, 4]

# With dicts
snippets = [r.get("content", "") for r in results]

# enumerate — iterate with index
for i, doc in enumerate(docs, 1):   # start=1 makes index begin at 1
    print(f"[Document {i}] {doc.source}")

# join — concatenate list of strings
parts = ["Hello", "world", "!"]
sentence = " ".join(parts)          # "Hello world !"
sections = "\n\n---\n\n".join(parts)

# append — add to end of list
items = []
items.append("new item")
```

```typescript
// TypeScript equivalents
const doubled = numbers.map((n) => n * 2);
const evens = numbers.filter((n) => n % 2 === 0);
const snippets = results.map((r) => r.content ?? "");
docs.forEach((doc, idx) => {
  const i = idx + 1;
  console.log(`[Document ${i}]`);
});
const sentence = parts.join(" ");
items.push("new item");
```

---

## 12. Dicts — get, spread, unpacking

```python
# Create a dict
person = {"name": "Alice", "age": 30}

# Read a key
name = person["name"]           # raises KeyError if missing
name = person.get("name")       # returns None if missing
name = person.get("name", "?")  # returns "?" if missing

# Write a key
person["city"] = "NYC"

# Check if key exists
if "name" in person:
    ...

# Spread / merge dicts (** unpacking)
base = {"status": "queued"}
result = {"job_id": "123", **base}   # {"job_id": "123", "status": "queued"}

# Iterate
for key, value in person.items():
    print(f"{key}: {value}")
```

```typescript
// TypeScript equivalents
const name = person["name"]
const name2 = person.name ?? null
person.city = "NYC"
if ("name" in person) { ... }
const result = { jobId: "123", ...base }
Object.entries(person).forEach(([key, value]) => console.log(`${key}: ${value}`))
```

---

## 13. try / except

```python
# Basic try/except (same as try/catch in JS)
try:
    result = risky_operation()
except Exception as e:
    print(f"Error: {e}")

# Catch specific exception types
try:
    value = int("not a number")
except ValueError as e:
    print("Not a valid integer")
except TypeError:
    print("Wrong type")

# Silently ignore failures (use sparingly)
try:
    optional_thing()
except Exception:
    pass    # intentionally empty — failure is non-fatal

# finally — always runs (like JS finally)
try:
    f = open("file.txt")
    data = f.read()
except FileNotFoundError:
    data = ""
finally:
    f.close()   # runs whether or not an exception occurred

# raise — same as `throw` in JS
raise ValueError("Invalid input")
raise HTTPException(status_code=422, detail="Blocked")
```

```typescript
try {
  const result = await riskyOperation();
} catch (e) {
  console.error(`Error: ${e}`);
} finally {
  cleanup();
}
throw new Error("Invalid input");
```

---

## 14. Imports — module system

```python
# Import a whole module
import re
import ast
import time

re.search(pattern, text)    # use with module prefix

# Import specific names from a module
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage, AIMessage

# Import with alias
from src.routers.agent import router as agent_router

# Relative imports (within the same package)
from .checker import GuardrailChecker      # same directory
from ..config.settings import settings    # parent directory

# Deferred / local import (inside a function)
# Used when a dependency might be missing or to avoid circular imports
def embed(text):
    from pymilvus import MilvusClient      # only imported when function is called
    client = MilvusClient(...)
```

```typescript
// TypeScript equivalents
import * as re from "re";
import { FastAPI, HTTPException } from "fastapi";
import { router as agentRouter } from "./routers/agent";
import { GuardrailChecker } from "./checker";
```

**Key difference:** Python has no `export` keyword. Any name defined at the top level of a module is automatically importable. The leading underscore `_name` convention signals "private, don't import this."

---

## 15. Pydantic — validation and settings

Pydantic is Python's equivalent of `zod` + `class-validator`. It validates data at runtime using class definitions.

```python
from pydantic import BaseModel, Field

# Define a schema
class AgentRequest(BaseModel):
    user_id: str                                    # required
    message: str = Field(..., min_length=1, max_length=10_000)
    history: list[dict] = []                        # optional, default []
    session_id: str

# Instantiate — Pydantic validates automatically
req = AgentRequest(user_id="u1", message="hello", session_id="s1")

# FastAPI uses Pydantic automatically:
# async def my_endpoint(request: AgentRequest):
#   FastAPI parses JSON → validates → gives you a real AgentRequest object
#   If invalid → returns 422 before your function runs

# pydantic-settings — reads from .env
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str          # required — app won't start without it
    claude_model: str = "claude-sonnet-4-6"   # optional with default
    milvus_port: int = 19530        # auto-converts "19530" string → int

    class Config:
        env_file = ".env"           # also read from .env file

settings = Settings()   # reads .env, validates, fails fast if required key missing
```

```typescript
// TypeScript equivalent using zod
import { z } from "zod";
const AgentRequest = z.object({
  userId: z.string(),
  message: z.string().min(1).max(10_000),
  history: z.array(z.object({})).default([]),
  sessionId: z.string(),
});
type AgentRequest = z.infer<typeof AgentRequest>;
```

---

## 16. Docstrings

A docstring is a string written as the **first statement inside a function, class, or module**. It uses triple quotes `"""..."""` and serves as built-in documentation that Python stores on the object itself.

```python
def web_search(query: str) -> str:
    """Search the web for current information, news, or recent events.
    Use when the question needs data that may not be in the knowledge base."""
    # ↑ this is the docstring — first statement inside the function
    if not settings.tavily_api_key:
        ...
```

**Docstring vs regular comment:**

```python
# This is a comment — Python ignores it completely at runtime

def my_function():
    """This is a docstring — Python stores it on the function object."""
    pass

# You can read a docstring at runtime:
print(my_function.__doc__)
# → "This is a docstring — Python stores it on the function object."
```

A comment disappears at runtime. A docstring is **readable code** — Python stores it and other tools can access it.

**Why it matters in this project — docstrings are functional, not just decorative:**

The `@tool` decorator reads the docstring at runtime and sends it to Claude as the tool description:

```python
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use for arithmetic, percentages,
    and unit conversions. Example inputs: '15 * 1.08', '(100 - 20) / 4'."""
    ...
```

Claude receives exactly that string and uses it to decide **when to call this tool**. A better docstring = Claude picks the right tool more often. The docstring here is not just for humans — it shapes Claude's behavior at runtime.

**JS/TS comparison:**

```typescript
// TypeScript — JSDoc (convention only, not part of the language)
/**
 * Evaluate a mathematical expression.
 */
function calculator(expression: string): string { ... }
```

| Python                                      | TypeScript                                      |
| ------------------------------------------- | ----------------------------------------------- |
| `"""..."""` — part of the language spec     | `/** ... */` JSDoc — just a comment convention  |
| Stored on `fn.__doc__`, readable at runtime | Parsed by external tools (TypeDoc, ESLint) only |
| Read by `@tool`, `help()`, IDEs             | Read by TypeDoc, IDEs                           |

---

## 17. `...` Ellipsis — required field marker

`...` (three dots) is Python's built-in **Ellipsis** object. In Pydantic's `Field()`, it means **"this field is required — no default value."**

```python
from pydantic import BaseModel, Field

class AgentRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10_000)
    #                    ^^^
    #                    Ellipsis = "required, caller must always provide this"
    #                    If missing from JSON → Pydantic rejects with 422
```

**Required vs optional:**

```python
class AgentRequest(BaseModel):
    message: str = Field(..., min_length=1)  # required — no default, must be provided
    history: list = Field([])                # optional — default is empty list
    top_k: int    = Field(5)                 # optional — default is 5
```

You only need `Field(...)` when you want to add **extra constraints** (`min_length`, `max_length`, `gt`, etc.) alongside the required marker. For a plain required field with no constraints, just omit the default entirely:

```python
class AgentRequest(BaseModel):
    message: str        # required (no default = required, same as Field(...))
    history: list = []  # optional with default
```

**`...` appears in other contexts too:**

```python
# 1. As a function/class body placeholder (like `pass`)
def not_implemented_yet():
    ...     # same as `pass` — means "nothing here yet"

# 2. As a type hint for variable-length tuples
tuple[int, ...]   # a tuple of any number of ints

# 3. In numpy/array slicing (not used in this project)
array[..., 0]     # "all dimensions, then index 0"
```

**JS/TS comparison:**

```typescript
// Zod equivalent of Field(..., min_length=1, max_length=10_000)
const AgentRequest = z.object({
  message: z.string().min(1).max(10_000), // required by default in Zod
  history: z.array(z.object({})).default([]),
});
```

In Zod, fields are required by default — you call `.optional()` to make them optional. In Pydantic, fields with no default are required, and `Field(...)` is the explicit way to say "required AND add constraints."

---

---

## 18. `"""` Triple quotes — multi-line strings

`"""` (three double quotes) is Python's syntax for writing strings that span multiple lines. You can also use `'''` (three single quotes) — both behave identically.

```python
# Single quote — one line only
name = "Hello world"

# Triple quote — can span multiple lines freely
message = """Hello
world
this is line 3"""

# Both styles work the same way
a = """three double quotes"""
b = '''three single quotes'''   # identical behavior
```

**Two uses in Python:**

**1. Multi-line strings (anywhere in code)**

```python
# Used in claude_client.py to build the system prompt
prompt = f"""You are a helpful AI assistant.

## Rules
- Be concise.
- Use tools when needed.
{memory_section}"""
```

**2. Docstrings (first statement inside a function/class)**

```python
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.
    Use for arithmetic and percentages."""
    # ^ same triple-quote syntax, but placed here = docstring
    ...
```

The `"""` syntax is identical in both cases. What makes it a **docstring** is purely its **position** — the first statement inside a function or class body. See [section 16](#16-docstrings) for more on docstrings.

**JS/TS comparison:**

```typescript
// TypeScript uses backticks for multi-line strings
const message = `Hello
world
this is line 3`

// Python uses """ instead of backticks — no backtick strings in Python
message = """Hello
world
this is line 3"""
```

| Python              | TypeScript                    |
| ------------------- | ----------------------------- |
| `"""multi-line"""`  | `` `multi-line` `` (backtick) |
| `"single line"`     | `"single line"` (same)        |
| No backtick strings | No triple-quote strings       |

---

## 19. Quick comparison table

| Python                           | JavaScript / TypeScript       | Notes                     |
| -------------------------------- | ----------------------------- | ------------------------- |
| `def fn():`                      | `function fn() {}`            |                           |
| `async def fn():`                | `async function fn() {}`      |                           |
| `await fn()`                     | `await fn()`                  | same                      |
| `async for x in gen`             | `for await (const x of gen)`  |                           |
| `yield` in `async def`           | `yield` in `async function*`  | async generator           |
| `f"text {var}"`                  | `` `text ${var}` ``           | f-strings                 |
| `None`                           | `null` / `undefined`          |                           |
| `x is None`                      | `x === null`                  | use `is`, not `==`        |
| `x is not None`                  | `x !== null`                  |                           |
| `not x`                          | `!x`                          |                           |
| `x or "default"`                 | `x \|\| "default"`            |                           |
| `str \| None`                    | `string \| null`              |                           |
| `Literal["a","b"]`               | `"a" \| "b"`                  |                           |
| `list[str]`                      | `string[]`                    |                           |
| `dict[str, int]`                 | `{ [k: string]: number }`     |                           |
| `[x for x in list]`              | `list.map(x => x)`            | list comprehension        |
| `[x for x in list if cond]`      | `list.filter(cond).map(...)`  | filtered comprehension    |
| `for i, x in enumerate(list, 1)` | `list.forEach((x, i) => i+1)` |                           |
| `"\n".join(list)`                | `list.join("\n")`             |                           |
| `list.append(x)`                 | `list.push(x)`                |                           |
| `dict.get("key", default)`       | `dict["key"] ?? default`      |                           |
| `{**dict1, **dict2}`             | `{ ...obj1, ...obj2 }`        | spread                    |
| `raise Exception`                | `throw new Error()`           |                           |
| `try / except`                   | `try / catch`                 |                           |
| `except Exception: pass`         | `catch (e) {}`                | intentionally empty       |
| `isinstance(x, Type)`            | `x instanceof Type`           |                           |
| `type(x).__name__`               | `x.constructor.name`          |                           |
| `self`                           | `this`                        | always explicit in Python |
| `__init__(self)`                 | `constructor()`               |                           |
| `@decorator`                     | decorator pattern / HOF       |                           |
| `import x from y`                | `import { x } from 'y'`       |                           |
| `__init__.py`                    | `index.ts` (re-exports)       | marks folder as package   |
| `10_000`                         | `10_000` or `10000`           | underscore separator      |
