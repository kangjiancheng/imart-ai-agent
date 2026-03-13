# Python Quick-Start Guide for JS/TS & NestJS Developers

> Every concept here is drawn directly from the `app-ai/src/` codebase.
> Each section shows the real file and line where the pattern appears,
> explains what it does in Python, and maps it to the JS/TS equivalent you already know.

---

## Table of Contents

1. [Project Structure & Module System](#1-project-structure--module-system)
2. [Variables, Types & Type Hints](#2-variables-types--type-hints)
3. [Functions](#3-functions)
4. [Classes](#4-classes)
5. [Dataclasses — Lightweight Structs](#5-dataclasses--lightweight-structs)
6. [Pydantic — Validation + Types (like Zod + Interface)](#6-pydantic--validation--types-like-zod--interface)
7. [Pydantic Settings — Config/Env (like NestJS ConfigModule)](#7-pydantic-settings--configenv-like-nestjs-configmodule)
8. [Async / Await](#8-async--await)
9. [Async Generators — Streaming (like AsyncIterator / ReadableStream)](#9-async-generators--streaming-like-asynciterator--readablestream)
10. [Decorators](#10-decorators)
11. [List & Dict Comprehensions](#11-list--dict-comprehensions)
12. [Tuple Unpacking](#12-tuple-unpacking)
13. [Error Handling](#13-error-handling)
14. [F-Strings (Template Literals)](#14-f-strings-template-literals)
15. [Imports & Module Singletons](#15-imports--module-singletons)
16. [Regex](#16-regex)
17. [FastAPI — Web Framework (like NestJS)](#17-fastapi--web-framework-like-nestjs)
18. [Key Python Idioms Cheatsheet](#18-key-python-idioms-cheatsheet)

---

## 1. Project Structure & Module System

### `__init__.py` — The Module Marker

Every directory that Python should treat as an importable package needs an `__init__.py` file (can be empty).

```
src/
├── __init__.py          ← marks `src` as a package
├── main.py
├── config/
│   ├── __init__.py      ← marks `config` as a sub-package
│   └── settings.py
├── agent/
│   ├── __init__.py
│   └── agent_loop.py
```

| Python | JavaScript / TypeScript |
|--------|------------------------|
| `__init__.py` in each folder | `index.ts` barrel file (optional) |
| `from src.config.settings import settings` | `import { settings } from './config/settings'` |
| `import anthropic` | `import Anthropic from 'anthropic'` |

**Seen in:** every `src/*/``__init__.py` file.

### Import Style

```python
# Absolute import (preferred in this project)
from src.llm.claude_client import llm, build_messages, build_system_prompt

# Import a whole module
import anthropic

# Standard library
import re
import os
from dataclasses import dataclass
from typing import Literal
```

**TypeScript equivalent:**
```ts
import { llm, buildMessages, buildSystemPrompt } from './llm/claudeClient';
import Anthropic from 'anthropic';
```

---

## 2. Variables, Types & Type Hints

Python is **dynamically typed** but supports optional **type hints** (like TypeScript annotations, but not enforced at runtime unless you use Pydantic or mypy).

```python
# Plain variable — no type needed, but hints help readability
MAX_ITERATIONS = 10                  # const MAX_ITERATIONS = 10
tokens_used: int = 0                 # let tokensUsed: number = 0
message: str = "Hello"               # let message: string = "Hello"
iterations: list[dict] = []          # let iterations: Record<string,any>[] = []
recalled: list[str] = []             # let recalled: string[] = []
```

**Seen in:** [src/agent/agent_loop.py](src/agent/agent_loop.py) — `MAX_ITERATIONS`, `tokens_used`, `iterations`.

### Optional / Nullable Types

```python
# Python 3.10+ union syntax (used throughout this project)
anthropic_base_url: str | None = None    # string | null = null  (TS)
tavily_api_key: str | None = None

# Older Python style (still valid, avoid in new code)
from typing import Optional
anthropic_base_url: Optional[str] = None
```

**Seen in:** [src/config/settings.py](src/config/settings.py) and [src/schemas/request.py](src/schemas/request.py).

### Numeric Literal Separators

```python
# Underscore in numbers = visual separator only (just like TS)
message: str = Field(..., max_length=10_000)   # 10_000 === 10000
```

---

## 3. Functions

```python
# Basic function with type hints
def greet(name: str) -> str:
    return f"Hello, {name}"

# Function with default parameter
def connect(host: str = "localhost", port: int = 19530) -> None:
    ...   # `...` (Ellipsis) = placeholder body, like `{}` in TS

# Function returning a tuple
def check_policy(message: str) -> tuple[bool, str]:
    return True, ""    # returns (True, "")
```

**TypeScript equivalent:**
```ts
function greet(name: string): string { return `Hello, ${name}`; }
function connect(host = 'localhost', port = 19530): void { }
function checkPolicy(msg: string): [boolean, string] { return [true, '']; }
```

**Seen in:** [src/guardrails/checker.py](src/guardrails/checker.py) — `check()` method returns `GuardrailResult`.

---

## 4. Classes

```python
class GuardrailChecker:
    """Docstring = JSDoc comment — triple quotes."""

    def check(self, message: str) -> GuardrailResult:
        #       ^^^^ `self` = `this` in JS — always the FIRST parameter of any method
        result = GuardrailResult(passed=True, sanitized_message=message)
        return result
```

**Key difference from JS/TS:**
- `self` must be explicitly listed as the first parameter of every instance method.
- `self.field_name` instead of `this.fieldName`.

```python
# Constructor = __init__ (dunder = double underscore)
class MyService:
    def __init__(self, api_key: str):
        self.api_key = api_key          # this.apiKey = apiKey

    def call(self) -> str:
        return self.api_key             # return this.apiKey
```

**TypeScript equivalent:**
```ts
class MyService {
  constructor(private apiKey: string) {}
  call(): string { return this.apiKey; }
}
```

**Seen in:** [src/guardrails/checker.py](src/guardrails/checker.py), [src/rag/retriever.py](src/rag/retriever.py), [src/memory/vector_memory.py](src/memory/vector_memory.py).

### Static Methods & Properties

```python
class TokenBudget:
    @staticmethod
    def _estimate(messages: list) -> int:   # no `self` — like TS static method
        return sum(len(str(m)) for m in messages) // 4

    @property
    def is_exhausted(self) -> bool:          # accessed as budget.is_exhausted (no call!)
        return self.used >= self.limit
```

**TypeScript equivalent:**
```ts
class TokenBudget {
  static estimate(messages: any[]): number { /* ... */ }
  get isExhausted(): boolean { return this.used >= this.limit; }
}
```

**Seen in:** [src/agent/token_budget.py](src/agent/token_budget.py).

---

## 5. Dataclasses — Lightweight Structs

`@dataclass` auto-generates `__init__`, `__repr__`, and `__eq__` — no boilerplate.

```python
from dataclasses import dataclass

@dataclass
class GuardrailResult:
    passed: bool
    sanitized_message: str
    reason: str = ""    # field with default value

# Usage — positional or keyword args:
result = GuardrailResult(passed=True, sanitized_message="Hello")
result.passed          # True
result.reason          # ""
```

**TypeScript equivalent:**
```ts
interface GuardrailResult {
  passed: boolean;
  sanitizedMessage: string;
  reason?: string;
}
// No auto-constructor — you'd use a plain object literal or a class
```

**Seen in:** [src/guardrails/checker.py](src/guardrails/checker.py) — `GuardrailResult`.

---

## 6. Pydantic — Validation + Types (like Zod + Interface)

Pydantic combines TypeScript interfaces (shape definition) + Zod (runtime validation) into one class.

```python
from pydantic import BaseModel, Field
from typing import Literal

class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]   # only these two strings allowed
    content: str                          # required, no default

class AgentRequest(BaseModel):
    user_id: str
    message: str = Field(..., min_length=1, max_length=10_000)
    #               ^^^  `...` = required, no default
    history: list[HistoryMessage] = []    # optional, default empty list
    user_context: UserContext             # nested model — validated automatically
    stream: bool = True
    document_context: str | None = None  # optional nullable field
```

**TypeScript + Zod equivalent:**
```ts
import { z } from 'zod';
const AgentRequest = z.object({
  userId: z.string(),
  message: z.string().min(1).max(10000),
  history: z.array(HistoryMessage).default([]),
  userContext: UserContext,
  stream: z.boolean().default(true),
  documentContext: z.string().nullable().optional(),
});
```

### Serializing a Pydantic object → plain dict

```python
m.model_dump()    # equivalent to JSON.stringify then parse — gives plain dict
# e.g. HistoryMessage(role="user", content="Hi") → {"role": "user", "content": "Hi"}
```

**Seen in:** [src/schemas/request.py](src/schemas/request.py), [src/schemas/response.py](src/schemas/response.py), and used in [src/agent/agent_loop.py](src/agent/agent_loop.py) — `m.model_dump()`.

---

## 7. Pydantic Settings — Config/Env (like NestJS ConfigModule)

`pydantic-settings` reads `.env` automatically and validates all values at startup.

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str          # REQUIRED — app crashes at startup if missing
    claude_model: str = "claude-sonnet-4-6"   # optional with default
    claude_max_tokens: int = 4096   # auto-converts "4096" string → int
    milvus_uri: str                 # REQUIRED
    tavily_api_key: str | None = None  # optional nullable

    class Config:
        env_file = ".env"    # read .env on startup
        extra = "ignore"     # ignore unknown env vars (like NODE_ENV, PATH, etc.)

# Singleton: instantiate once at module level
settings = Settings()
```

**NestJS equivalent:**
```ts
// NestJS ConfigModule approach
@Module({ imports: [ConfigModule.forRoot({ ignoreEnvFile: false })] })
// Access: configService.get<string>('ANTHROPIC_API_KEY')
```

**Key advantage over `os.getenv()`:** Fails **fast** at startup with a clear error, not silently at runtime when you first use the missing key.

**Seen in:** [src/config/settings.py](src/config/settings.py).

---

## 8. Async / Await

Python's `async`/`await` works very similarly to JavaScript's. The main differences:

| Concept | JavaScript | Python |
|---------|-----------|--------|
| Async function | `async function foo()` | `async def foo():` |
| Await a promise/coroutine | `await foo()` | `await foo()` |
| Run async in a sync context | `foo().then(...)` | `asyncio.run(foo())` |
| Async HTTP framework | Express (with async) | FastAPI (async-native) |
| Async DB call | `await db.find(...)` | `await collection.query(...)` |

```python
async def run(request: AgentRequest):
    # `await` pauses HERE and lets other requests run on the server
    recalled = await memory.recall(request.user_id, request.message, top_k=5)
    response = await planner.ainvoke(messages)
    #                          ^^^^ LangChain's async "invoke" convention: prefix with `a`
```

### Running blocking code in a thread pool

BGE-M3 embeddings run synchronously (CPU-bound). We offload them to a thread pool so they don't block the async event loop:

```python
import asyncio

async def embed(self, texts: list[str]) -> list[list[float]]:
    loop = asyncio.get_event_loop()
    # run_in_executor = run a blocking (sync) function in a thread pool
    # None = use the default ThreadPoolExecutor
    # lambda = wrap the call so we can pass arguments
    result = await loop.run_in_executor(None, lambda: self._model.encode(texts))
    return result.tolist()
```

**JavaScript equivalent:**
```ts
// Similar to wrapping a sync function in a Worker thread
const result = await new Promise(resolve => workerPool.run(() => model.encode(texts), resolve));
```

**Seen in:** [src/rag/embeddings.py](src/rag/embeddings.py) — `embed()` method.

---

## 9. Async Generators — Streaming (like AsyncIterator / ReadableStream)

An **async generator** is a function that can `yield` many values over time — perfect for SSE streaming.

```python
# `async def` + `yield` = async generator
async def run(request: AgentRequest):
    # ... setup ...
    for i in range(MAX_ITERATIONS):
        # stream the final answer token by token
        async for chunk in planner.astream(messages):
            if chunk.content:
                yield chunk.content      # ← sends one token to the browser NOW
        break

# Caller (in the router) iterates the generator:
async for token in run(request):
    yield f"data: {token}\n\n"   # format as SSE frame
```

**TypeScript / Node.js equivalent:**
```ts
// Like an async generator function in TS:
async function* run(request: AgentRequest): AsyncGenerator<string> {
  for await (const chunk of planner.stream(messages)) {
    yield chunk.content;
  }
}
// Or like a ReadableStream / EventSource push model
```

**Seen in:** [src/agent/agent_loop.py](src/agent/agent_loop.py) — the entire `run()` function.

---

## 10. Decorators

Decorators in Python work exactly like decorators in TypeScript (NestJS uses them heavily).
They are functions that wrap other functions or classes.

```python
# @dataclass — auto-generates __init__, __repr__, __eq__
@dataclass
class GuardrailResult:
    passed: bool

# @tool — LangChain decorator that registers a function as a callable tool for the LLM
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluates a safe math expression."""    # docstring becomes the tool's description for Claude
    return str(_safe_eval(expression))

# @property — makes a method callable as an attribute (no parentheses needed)
@property
def is_exhausted(self) -> bool:
    return self.used >= self.limit

# @staticmethod — method that doesn't receive `self` (class-level utility)
@staticmethod
def _estimate(messages: list) -> int:
    return sum(len(str(m)) for m in messages) // 4
```

**NestJS / TypeScript equivalents:**
```ts
@Injectable()       // → Python: @dataclass or custom decorator
@Get('/path')       // → Python: @app.get('/path') in FastAPI
@IsString()         // → Python: Pydantic type hint + Field()
```

**Seen in:** [src/tools/calculator.py](src/tools/calculator.py), [src/guardrails/checker.py](src/guardrails/checker.py), [src/agent/token_budget.py](src/agent/token_budget.py).

---

## 11. List & Dict Comprehensions

Comprehensions are concise one-liners for building lists or dicts. Much more common in Python than in JS.

### List Comprehensions

```python
# Build a new list by transforming each item
tool_names = [it["tool"] for it in iterations]
# JS: const toolNames = iterations.map(it => it.tool)

# Filter + transform
human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
# JS: const humanMsgs = messages.filter(m => m instanceof HumanMessage)

# Serialize a list of Pydantic objects
history_dicts = [m.model_dump() for m in request.history]
# JS: const historyDicts = request.history.map(m => m.toJSON())
```

### Dict Comprehensions

```python
# Build a lookup dict: name → tool object
tool_map = {t.name: t for t in TOOLS if t.name != "rag_retrieve"}
# JS: const toolMap = Object.fromEntries(TOOLS.filter(t => t.name !== 'rag_retrieve').map(t => [t.name, t]))
```

### Generator Expressions (lazy, no brackets)

```python
# Like a lazy .map() — no intermediate list created
total = sum(len(str(m)) for m in messages)
# JS: const total = messages.reduce((acc, m) => acc + String(m).length, 0)
```

**Seen in:** [src/agent/agent_loop.py](src/agent/agent_loop.py), [src/tools/registry.py](src/tools/registry.py), [src/agent/token_budget.py](src/agent/token_budget.py).

---

## 12. Tuple Unpacking

Python functions can return multiple values as a **tuple**, and you can unpack them in one line.

```python
# Function returns a tuple (two values)
def check_content_policy(message: str) -> tuple[bool, str]:
    return True, ""          # return (True, "") — parentheses optional

# Unpacking — like JS array destructuring
passed, reason = check_content_policy(message)
#  ^^^    ^^^^   two variables assigned in one line

# If you only care about one value, use _ for the rest
passed, _ = check_content_policy(message)
```

**JavaScript equivalent:**
```ts
const [passed, reason] = checkContentPolicy(message);
```

**Seen in:** [src/guardrails/checker.py](src/guardrails/checker.py) — `passed, reason = check_content_policy(...)`.

---

## 13. Error Handling

```python
try:
    response = await planner.ainvoke(messages)
except anthropic.InternalServerError as exc:
    # Catch a specific exception type — like `catch (e instanceof SpecificError)`
    raise RuntimeError("Service temporarily unavailable.") from exc
    #                                                       ^^^^^^^^ chains the original error
except anthropic.BadRequestError as exc:
    raise RuntimeError("Malformed request.") from exc
except Exception as e:
    # Catch-all (like catch(e) in JS)
    result = f"Error: tool failed — {str(e)}"
```

**JavaScript equivalent:**
```ts
try {
  const response = await planner.invoke(messages);
} catch (e) {
  if (e instanceof InternalServerError) throw new Error('Service unavailable');
  throw e;
}
```

### Graceful Degradation (common Python pattern in this project)

```python
try:
    docs = await retriever.retrieve(query)
except Exception:
    docs = []    # Milvus is down? Return empty list, don't crash.
```

This pattern appears throughout the project: if an optional service (Milvus, Tavily) is unavailable, the code degrades gracefully rather than crashing.

**Seen in:** [src/agent/agent_loop.py](src/agent/agent_loop.py), [src/rag/retriever.py](src/rag/retriever.py), [src/memory/vector_memory.py](src/memory/vector_memory.py).

---

## 14. F-Strings (Template Literals)

F-strings are Python's template literals. Prefix a string with `f` and use `{}` for expressions.

```python
# Basic interpolation
tools_used = ", ".join(set(tool_names))
summary = f"Agent used {count} tool call(s): {tools_used}."
# JS: `Agent used ${count} tool call(s): ${toolsUsed}.`

# Expression inside braces
snippet = f"Result: {str(result)[:200]}"
# JS: `Result: ${String(result).slice(0, 200)}`

# Multi-line f-string (using parentheses to span lines)
prompt = (
    f"You are a helpful assistant.\n"
    f"User subscription: {user_context.subscription_tier}\n"
    f"Locale: {user_context.locale}\n"
)
```

**Seen in:** [src/agent/agent_loop.py](src/agent/agent_loop.py), [src/llm/claude_client.py](src/llm/claude_client.py).

---

## 15. Imports & Module Singletons

Python creates a module-level singleton by instantiating at the bottom of a file. Every other file that imports it gets the same object.

```python
# src/config/settings.py
class Settings(BaseSettings):
    ...

settings = Settings()   # ← created ONCE when the module first loads

# src/llm/claude_client.py
llm = ChatAnthropic(...)   # ← shared LLM connection — one per process

# src/tools/registry.py
TOOLS = [web_search, calculator, rag_retrieve_tool]   # ← constant list
tool_map = {t.name: t for t in TOOLS if t.name != "rag_retrieve"}  # ← lookup dict
```

**Any file that imports these gets the same instance:**
```python
from src.config.settings import settings    # always the same Settings object
from src.llm.claude_client import llm       # always the same ChatAnthropic instance
```

**NestJS equivalent:**
```ts
// NestJS singleton providers
@Injectable()
export class LlmService { /* ... */ }
// Registered as a provider → same instance injected everywhere
```

**Seen in:** [src/config/settings.py](src/config/settings.py), [src/llm/claude_client.py](src/llm/claude_client.py), [src/tools/registry.py](src/tools/registry.py).

---

## 16. Regex

```python
import re

# Define a pattern — use raw strings (r"...") to avoid double-escaping backslashes
PATTERNS = [
    r"\b(bomb|weapon|hack)\b",     # \b = word boundary
    r"(?i)ignore.{0,20}instructions",  # (?i) = case-insensitive flag inside pattern
]

# Search anywhere in the string (like JS .test())
if re.search(pattern, message):
    return False, "Harmful content detected."

# Substitution — replace all matches (like JS .replace() with global flag)
email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
cleaned = re.sub(email_pattern, "[EMAIL]", message)

# Case-insensitive flag as a separate argument
if re.search(pattern, message, re.IGNORECASE):
    ...
```

**JavaScript equivalent:**
```ts
const pattern = /\b(bomb|weapon|hack)\b/;
if (pattern.test(message)) { /* blocked */ }
const cleaned = message.replace(/email-regex/g, '[EMAIL]');
```

**Seen in:** [src/guardrails/content_policy.py](src/guardrails/content_policy.py), [src/guardrails/pii_filter.py](src/guardrails/pii_filter.py), [src/guardrails/injection_detector.py](src/guardrails/injection_detector.py).

---

## 17. FastAPI — Web Framework (like NestJS)

FastAPI is Python's closest equivalent to NestJS — dependency injection, decorators, auto-validation, auto-docs (Swagger at `/docs`).

```python
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import StreamingResponse

app = FastAPI(title="AI Agent Service")
router = APIRouter(prefix="/v1/agent", tags=["agent"])

# Decorator = route registration (like NestJS @Post())
@router.post("/chat")
async def agent_chat(request: AgentRequest):
    #                         ^^^^^^^^^^^^ FastAPI reads the type hint and:
    #                         1. Parses the HTTP body as JSON
    #                         2. Validates it against AgentRequest (Pydantic)
    #                         3. Returns 422 automatically if invalid
    ...

# Throw HTTP errors like NestJS throw new HttpException(...)
raise HTTPException(status_code=422, detail="Guardrail blocked the message.")

# Streaming response for SSE (Server-Sent Events)
async def event_stream():
    async for token in run(request):
        yield f"data: {token}\n\n"   # SSE format

return StreamingResponse(event_stream(), media_type="text/event-stream")

# Register the router with the app (like NestJS module imports)
app.include_router(router)
```

| NestJS | FastAPI |
|--------|---------|
| `@Controller('/v1/agent')` | `APIRouter(prefix='/v1/agent')` |
| `@Post('/chat')` | `@router.post('/chat')` |
| `@Body() dto: AgentRequestDto` | `request: AgentRequest` (auto-parsed) |
| `@Injectable()` service | module-level singleton |
| `new HttpException('msg', 422)` | `HTTPException(status_code=422, detail='msg')` |
| `app.useGlobalGuards(...)` | middleware or dependency injection |
| Swagger auto-docs | Built-in at `/docs` (Swagger UI) |

**Seen in:** [src/main.py](src/main.py), [src/routers/agent.py](src/routers/agent.py), [src/routers/rag.py](src/routers/rag.py).

---

## 18. Key Python Idioms Cheatsheet

Quick reference for patterns you'll see constantly in this project.

```python
# ── Truthiness ───────────────────────────────────────────────────────────────
if not iterations:     # if iterations is empty list [] or None or False
    return ""
if recalled:           # if recalled is a non-empty list
    inject_memory(recalled)

# ── String joining ────────────────────────────────────────────────────────────
tools_used = ", ".join(["calculator", "web_search"])
# JS: ["calculator", "web_search"].join(", ")

# ── set() for deduplication ───────────────────────────────────────────────────
unique_tools = set(tool_names)    # removes duplicates, like new Set(toolNames)

# ── String slicing ────────────────────────────────────────────────────────────
snippet = long_text[:200]         # first 200 chars — like .slice(0, 200) in JS
last_100 = text[-100:]            # last 100 chars — like .slice(-100)

# ── isinstance() — type checking ──────────────────────────────────────────────
if isinstance(content, str):      # like typeof content === 'string'
    yield content
if isinstance(content, list):     # like Array.isArray(content)
    for block in content: ...

# ── dict.get() — safe key access ──────────────────────────────────────────────
result = tool_call.get("result", "")   # like toolCall?.result ?? ""

# ── Ellipsis ... — placeholder body ──────────────────────────────────────────
def not_yet_implemented():
    ...    # like `{}` or `throw new Error('not implemented')`

# ── _ — throwaway variable ────────────────────────────────────────────────────
for _ in range(3):    # loop 3 times, don't care about index
    retry()

# ── Ternary expression ────────────────────────────────────────────────────────
label = "yes" if passed else "no"    # JS: passed ? "yes" : "no"

# ── for...enumerate — index + value ──────────────────────────────────────────
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk}")
# JS: chunks.forEach((chunk, i) => console.log(`Chunk ${i}: ${chunk}`))

# ── Uppercase = constant convention ──────────────────────────────────────────
MAX_ITERATIONS = 10     # convention: never reassign this
MODEL_CONTEXT_LIMIT = 200_000

# ── Private by convention: underscore prefix ──────────────────────────────────
def _safe_eval(node):   # "private" — don't call from outside this module
    ...                 # Python has no true private — _ is just a convention
```

---

## Quick Comparison: Python vs TypeScript

| Feature | TypeScript | Python |
|---------|-----------|--------|
| Type annotations | `name: string` | `name: str` |
| Optional type | `string \| null` | `str \| None` |
| Union type | `string \| number` | `str \| int` |
| List type | `string[]` or `Array<string>` | `list[str]` |
| Dict / Map type | `Record<string, any>` | `dict[str, Any]` |
| Nullable field | `field?: string` | `field: str \| None = None` |
| Required field | no default | no `=` default, or `Field(...)` |
| Async function | `async function foo()` | `async def foo():` |
| Template literal | `` `Hello ${name}` `` | `f"Hello {name}"` |
| Destructuring | `const [a, b] = fn()` | `a, b = fn()` |
| Array spread | `[...a, ...b]` | `[*a, *b]` |
| Object spread | `{...a, ...b}` | `{**a, **b}` |
| `null` | `null` | `None` |
| `undefined` | `undefined` | (doesn't exist — use `None`) |
| Class constructor | `constructor()` | `def __init__(self):` |
| `this` | `this` | `self` |
| Interface | `interface Foo {}` | Pydantic `BaseModel` or `@dataclass` |
| Enum-like | `as const` object | `Literal["a", "b"]` or `Enum` |
| Decorator | `@Injectable()` | `@dataclass`, `@property`, `@tool` |
| try/catch | `try {} catch(e) {}` | `try: ... except Exception as e:` |
| typeof check | `typeof x === 'string'` | `isinstance(x, str)` |
| Array filter | `.filter(x => x.ok)` | `[x for x in items if x.ok]` |
| Array map | `.map(x => x.id)` | `[x.id for x in items]` |

---

## Running the Project

```bash
cd app-ai
python -m venv venv          # create virtual environment (like node_modules isolation)
source venv/bin/activate     # activate venv (Windows: venv\Scripts\activate)
pip install -r requirements.txt   # like npm install
cp .env.example .env         # copy env template
# edit .env — fill in ANTHROPIC_API_KEY, MILVUS_URI etc.
uvicorn src.main:app --reload --port 8000   # like: npm run start:dev
```

| npm/Node.js | Python |
|------------|--------|
| `npm install` | `pip install -r requirements.txt` |
| `node_modules/` | `venv/` (virtual environment) |
| `package.json` | `requirements.txt` |
| `npm run start:dev` | `uvicorn src.main:app --reload` |
| `.env` | `.env` (same!) |
| `ts-node` | `python` |
| `tsc` | `mypy` (optional type checker) |

---

*This guide is generated from the actual `app-ai/src/` source code. For deeper context, read the inline comments in each file — they are written specifically for developers learning Python from a JavaScript/TypeScript background.*
