# Full-Stack Enterprise AI Application Architecture

A comprehensive reference for building production AI applications with Next.js, NestJS, FastAPI, LangChain, and Milvus.

---

## Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USERS / CLIENTS                                    │
│                    Browser  ·  Mobile App  ·  Third-party API                  │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │ HTTPS / WSS
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            EDGE / API GATEWAY LAYER                             │
│                                                                                 │
│   Nginx  (Stage 1–2)          Cloudflare / AWS API GW  (Stage 3)               │
│   ├── SSL/TLS termination     ├── Global DDoS protection                        │
│   ├── IP-level rate limiting  ├── Edge caching / CDN                            │
│   ├── Load balancing          ├── Bot mitigation                                │
│   └── proxy_buffering off ←─ critical for AI streaming                         │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │ HTTP (internal network)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         CLIENT TIER  —  Next.js (App Router)                   │
│                                                                                 │
│   app/                                                                          │
│   ├── (pages)/chat/page.tsx        UI — React components, streaming chat       │
│   ├── api/auth/[...nextauth]/      Auth.js — session cookies, OAuth            │
│   └── api/chat/route.ts            Thin proxy OR direct BFF call               │
│                                                                                 │
│   State: Zustand · React Query · Server Components                              │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │ HTTPS + WebSocket (to browser)
                                     │ REST / tRPC (to middle server)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MIDDLE TIER  —  NestJS  (Backend for Frontend)               │
│                                                                                 │
│   src/                                                                          │
│   ├── auth/              JWT validation · OAuth · session management            │
│   ├── users/             User CRUD · roles · Prisma ORM → PostgreSQL           │
│   ├── chat/                                                                     │
│   │   ├── chat.controller.ts   POST /chat  →  validate  →  call FastAPI        │
│   │   ├── chat.service.ts      Save history to DB · load context               │
│   │   └── chat.gateway.ts      WebSocket — stream tokens to browser            │
│   ├── billing/           Subscriptions · usage tracking · Stripe               │
│   ├── files/             Upload handling · S3 · pass to AI service             │
│   └── ai-proxy/          HTTP client → FastAPI (internal only)                 │
│                                                                                 │
│   Infrastructure: Rate limiting (Redis) · Caching · Logging · Metrics          │
└───────────────────────────┬─────────────────────────────────────────────────────┘
                            │ Internal HTTP  (never exposed to internet)
                            │ Message Queue for long-running jobs (Redis/RabbitMQ)
                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      AI SERVICE TIER  —  FastAPI  (Python)                      │
│                                                                                 │
│   app/                                                                          │
│   ├── routers/                                                                  │
│   │   ├── agent.py        POST /agent/chat  →  streaming SSE response          │
│   │   └── rag.py          POST /rag/query   →  document retrieval              │
│   ├── agent/                                                                    │
│   │   ├── agent_loop.py   ReAct loop: bind_tools → ainvoke → ToolMessage      │
│   │   └── token_budget.py Tracks tokens used · trims history when near limit  │
│   ├── guardrails/          Three-check pipeline before every LLM call         │
│   │   ├── content_policy.py  Block harmful requests                            │
│   │   ├── pii_redactor.py    Redact PII from user input                        │
│   │   └── injection_check.py Detect prompt-injection attempts                  │
│   ├── tools/               @tool-decorated functions bound to ChatAnthropic    │
│   │   ├── registry.py      TOOLS list — imported by agent_loop.py              │
│   │   ├── web_search.py    Tavily / SerpAPI tool                               │
│   │   └── code_exec.py     Sandboxed code execution                            │
│   ├── memory/                                                                   │
│   │   └── vector_memory.py Long-term: user preferences · past tasks (Milvus)  │
│   ├── rag/                                                                      │
│   │   ├── embeddings.py    Claude / OpenAI embedding models                    │
│   │   └── retriever.py     Milvus vector search → top-K docs                  │
│   ├── schemas/                                                                  │
│   │   └── agent_schemas.py Pydantic request / response models                  │
│   ├── llm/                                                                      │
│   │   └── claude_client.py  ChatAnthropic(model="claude-sonnet-4-6")          │
│   └── config/                                                                   │
│                                                                                 │
│   Stateless — receives full context each request (message + history + userId)  │
└──────────┬────────────────────────┬────────────────────────────────────────────┘
           │                        │
           ▼                        ▼
┌──────────────────┐    ┌───────────────────────────────────┐
│   Claude API     │    │         DATA STORES               │
│                  │    │                                   │
│  claude-         │    │  PostgreSQL  — users, sessions,   │
│  sonnet-4-6      │    │               chat history        │
│  claude-         │    │               (via NestJS/Prisma) │
│  opus-4-6        │    │                                   │
│                  │    │  Milvus      — vector embeddings   │
│  Streaming SSE   │    │               RAG document store  │
│  Tool calling    │    │               (FastAPI only)      │
└──────────────────┘    │                                   │
                        │  Redis       — rate limiting,     │
                        │               session cache,      │
                        │               job queues          │
                        │               (shared layer)     │
                        └───────────────────────────────────┘
```

---

## Data Flow: A Streaming Chat Message

```
① User types message in browser
         │
         ▼
② Next.js client component
     fetch('/api/chat', { method: 'POST', body: { message } })
     Opens EventSource for streaming tokens
         │
         ▼
③ NestJS  ChatController  POST /chat
     a. Validate JWT  →  extract userId
     b. Check rate limit (Redis)  →  reject if over limit
     c. Load chat history from PostgreSQL
     d. Forward to FastAPI:
        { message, history, userId, userContext }
         │
         ▼
④ FastAPI  /agent/chat
     a. Build LangChain prompt with full context
     b. Run guardrails (content policy → PII redact → injection check)
     c. Run agent_loop.run(request):
        ├── llm.bind_tools(TOOLS).ainvoke(messages)
        ├── If tool_calls → execute tools → append ToolMessage → loop
        └── Queries Milvus → retrieves relevant RAG documents
     d. Stream final response: llm.astream() → yield tokens via SSE
         │  SSE stream
         ▼
⑤ NestJS  (receives SSE stream from FastAPI)
     a. Pipes stream directly to WebSocket → browser
     b. On stream end: saves full response to PostgreSQL
         │  WebSocket
         ▼
⑥ Next.js renders tokens in real-time as they arrive
```

**Streaming rule:** Every layer must be configured to pass-through, never buffer.

| Layer   | Required Configuration                                           |
| ------- | ---------------------------------------------------------------- |
| FastAPI | `StreamingResponse(generator(), media_type="text/event-stream")` |
| NestJS  | `stream.pipe(res)` — never `await` the full response             |
| Nginx   | `proxy_buffering off; proxy_cache off; proxy_read_timeout 300s;` |
| Browser | `new EventSource('/api/chat/stream')`                            |

---

## Why Not Next.js → FastAPI Directly?

For an MVP, direct calls are fine. For a production product, the middle server is necessary:

| Concern              | Direct Next.js → FastAPI      | With NestJS Middle Server    |
| -------------------- | ----------------------------- | ---------------------------- |
| Auth & sessions      | Must duplicate in Python      | Single source of truth       |
| Rate limiting        | Hard to enforce consistently  | Centralized, Redis-backed    |
| User data (DB)       | Python handles non-AI SQL     | TypeScript ORM (Prisma) fits |
| Caching              | Redis in Python               | Redis in Node, shared        |
| File uploads         | Two parsers needed            | One handler, forwarded       |
| Secrets management   | API keys spread across stacks | All secrets in one place     |
| Multiple AI services | N connections from frontend   | One internal network hub     |
| WebSocket streaming  | Complex in Python             | Native in Node.js            |

**Use Next.js → FastAPI directly when:**

- Building a prototype or MVP
- Single-user or internal tool
- No user accounts, billing, or complex CRUD
- Team is 1–3 devs and speed is everything

**Use NestJS middle server when:**

- Multiple user accounts with roles and permissions
- Billing, subscriptions, or usage metering
- Multiple AI services (Claude + image gen + TTS + search)
- Audit logs, compliance, HIPAA/SOC2
- Team is growing and needs clear service boundaries

---

## Recommended Stack by Layer

### Frontend — app-web (Next.js, App Router)

```
app-web/
├── app/
│   ├── (auth)/login/page.tsx         Login page
│   ├── chat/
│   │   ├── page.tsx                  Chat UI — streams tokens via WebSocket
│   │   └── components/
│   │       ├── ChatInput.tsx
│   │       └── MessageStream.tsx     Renders SSE / WS tokens live
│   └── api/
│       ├── auth/[...nextauth]/       Auth.js — JWT + OAuth (Google, GitHub)
│       └── chat/route.ts             Thin proxy to app-service
├── src/
│   └── types/
│       └── api.ts                    Generated from app-ai OpenAPI spec
├── package.json
├── tsconfig.json
└── .eslintrc.js
```

### Middle Server — app-service (NestJS)

```
app-service/
├── src/
│   ├── auth/
│   │   ├── auth.module.ts
│   │   ├── jwt.strategy.ts           Validates Bearer tokens
│   │   └── guards/jwt-auth.guard.ts  Applied globally
│   ├── users/
│   │   ├── users.service.ts          CRUD via Prisma
│   │   └── users.controller.ts
│   ├── chat/
│   │   ├── chat.controller.ts        POST /chat → validate → forward to app-ai
│   │   ├── chat.service.ts           Load history, save response, build context
│   │   └── chat.gateway.ts           @WebSocketGateway — stream tokens to browser
│   ├── billing/
│   │   └── billing.service.ts        Stripe integration, usage limits
│   └── ai-proxy/
│       └── ai-proxy.service.ts       HTTP client → app-ai (internal only)
├── src/types/
│   └── api.ts                        Generated from app-ai OpenAPI spec
├── package.json
├── tsconfig.json
└── .eslintrc.js
```

### AI Service — app-ai (FastAPI, Python)

> For the internal agent runtime design (components, agent loop, guardrails, token budget), see [app-ai-architecture.md](app-ai-architecture.md).

```
app-ai/
├── app/
│   ├── main.py                       FastAPI app entry point
│   ├── routers/
│   │   ├── agent.py                  POST /v1/agent/chat  (streaming SSE)
│   │   └── rag.py                    POST /v1/rag/query
│   ├── agent/
│   │   ├── agent_loop.py             ReAct loop: bind_tools → ainvoke → ToolMessage → repeat
│   │   └── token_budget.py           Tracks tokens used · trims history when near limit
│   ├── guardrails/
│   │   ├── content_policy.py         Block harmful requests before LLM call
│   │   ├── pii_redactor.py           Redact PII from user input
│   │   └── injection_check.py        Detect prompt-injection attempts
│   ├── tools/
│   │   ├── registry.py               TOOLS list — @tool-decorated, imported by agent_loop.py
│   │   ├── web_search.py             Tavily / SerpAPI tool
│   │   └── code_exec.py              Sandboxed code execution
│   ├── memory/
│   │   └── vector_memory.py          Long-term: user preferences, past tasks (Milvus)
│   ├── rag/
│   │   ├── embeddings.py             Claude / OpenAI embeddings
│   │   └── retriever.py              Milvus similarity search
│   ├── schemas/
│   │   └── agent_schemas.py          Pydantic request / response models
│   ├── llm/
│   │   └── claude_client.py          ChatAnthropic wrapper, streaming
│   └── config/
│       └── settings.py               Pydantic BaseSettings — env vars
└── pyproject.toml                    uv / Poetry dependency management
```

---

## Communication Protocols

| Use Case                 | Protocol                  | Why                             |
| ------------------------ | ------------------------- | ------------------------------- |
| Browser → Next.js        | HTTPS REST                | Standard, simple                |
| Streaming tokens to user | WebSocket or SSE          | Real-time, low latency          |
| Next.js → NestJS         | REST or tRPC              | Type-safe across TS monorepo    |
| NestJS → FastAPI         | REST (internal network)   | Simple, fast on LAN             |
| High-throughput AI calls | gRPC                      | Binary, lower overhead at scale |
| Long-running AI jobs     | Redis Queue / RabbitMQ    | Async — avoids HTTP timeout     |
| FastAPI → Claude         | Anthropic SDK (streaming) | Official, SSE-native            |

---

## Key Architectural Principles

1. **Python service is stateless.** FastAPI receives full context (message + history + userId) on every request. State lives in PostgreSQL, managed by NestJS. Python never owns state.

2. **Auth never touches Python.** NestJS validates JWT, then passes only `userId` and sanitized data to FastAPI. The AI service trusts the middle server on the internal network. No auth logic in Python.

3. **Stream through every layer.** FastAPI yields SSE → NestJS pipes without buffering → WebSocket to browser. If any layer buffers the response, the streaming UX breaks entirely.

4. **Vector DB is Python-only.** Milvus is accessed exclusively from FastAPI. NestJS never queries it directly. This keeps the AI service self-contained and the vector data isolated.

5. **One database per concern.** PostgreSQL for users, sessions, and chat history (NestJS/Prisma). Milvus for vector embeddings (FastAPI). Redis for rate limiting and caching (shared). No cross-service DB access.

6. **Internal services are never internet-facing.** FastAPI and PostgreSQL have no public ports — only NestJS calls them, on the internal Docker network. Nginx only exposes ports 80 and 443.

---

## Repository Structure

### Decision: Three Separate Projects (Polyrepo)

`imart-ai-agent` is split into three independent projects, each with its own repository, dependency management, and configuration.

```
imart-ai-agent/          ← optional parent folder (not a monorepo)
├── app-web/             Next.js frontend      — own package.json, tsconfig, ESLint
├── app-service/         NestJS backend (BFF)  — own package.json, tsconfig, ESLint
└── app-ai/              FastAPI agent service  — own pyproject.toml, ruff, mypy
```

Each project is developed, versioned, and deployed independently. There are no shared packages or workspace tooling between them.

### API Contract: Source of Truth

`app-ai` owns and publishes the OpenAPI spec. `app-web` and `app-service` generate their types from it — no manual sync.

```bash
# Run in app-web and app-service after app-ai changes
npx openapi-typescript http://localhost:8000/openapi.json -o src/types/api.ts
```

- `app-ai` versions its API with a path prefix (`/v1/`, `/v2/`) so breaking changes are explicit
- Generated type files are committed per project — each project owns its copy

### Local Development

A `docker-compose.yml` at the parent folder level starts all three services together without requiring a monorepo:

```yaml
services:
  app-web:
    build: ./app-web
    ports: ["3000:3000"]
  app-service:
    build: ./app-service
    ports: ["4000:4000"]
  app-ai:
    build: ./app-ai
    ports: ["8000:8000"]
```

### Tradeoffs Accepted

| Concern                   | Impact                                              |
| ------------------------- | --------------------------------------------------- |
| Shared types              | Generated from OpenAPI spec per project             |
| Cross-service refactoring | Separate PRs per repo, coordinate manually          |
| CI/CD                     | Simpler per-repo pipelines, no cross-repo detection |
| Config (ESLint, tsconfig) | Maintained individually, may drift over time        |
| Team autonomy             | Full independence per project                       |
| Python + Node tooling     | Each project uses its native tools naturally        |
| Release cadence           | Independent release per service                     |

---

## Middle-Layer Backend: NestJS vs Java vs Go

### What the Industry Uses

| Company Type                              | Common Choice    | Reason                               |
| ----------------------------------------- | ---------------- | ------------------------------------ |
| AI startups (OpenAI, Anthropic ecosystem) | Node.js / Python | Speed, LLM ecosystem alignment       |
| Mid-size SaaS (Vercel, Linear, Notion)    | NestJS / Node.js | TypeScript across the full stack     |
| Large enterprise (JPMorgan, SAP, Oracle)  | Java Spring Boot | Existing investment, compliance      |
| Big Tech (Google, Netflix, Uber)          | Go / Java / gRPC | Scale, performance, polyglot systems |
| Korean/East Asian enterprise              | Java Spring Boot | Strong regional Java tradition       |

### NestJS vs Java Comparison

| Dimension                  | NestJS (Node.js)                               | Java (Spring Boot)                          |
| -------------------------- | ---------------------------------------------- | ------------------------------------------- |
| **Learning curve**         | Low — TypeScript/Angular patterns              | High — annotations, IoC, Spring ecosystem   |
| **Development speed**      | Fast — hot reload, JS ecosystem                | Slower — compile cycle, verbose boilerplate |
| **Performance**            | Good — async I/O, excellent for AI streaming   | Excellent — JVM throughput, CPU tasks       |
| **Streaming (SSE/WS)**     | Native, first-class (Node.js async by default) | Requires Spring WebFlux + Project Reactor   |
| **Type sharing (Next.js)** | Direct — same language, monorepo types         | Requires OpenAPI codegen or manual sync     |
| **AI/LLM ecosystem**       | Limited (LangChain.js lags Python)             | Very limited                                |
| **Docker image size**      | Small, fast startup                            | JVM overhead, slower cold start             |
| **Serverless**             | Excellent (Lambda, Vercel, Cloudflare)         | Poor — JVM startup latency                  |
| **Compliance tooling**     | Growing                                        | Best-in-class (Spring Security, Actuator)   |
| **Maturity**               | ~7 years, production-proven                    | 20+ years, battle-hardened                  |

### The Streaming Factor

Node.js is async by default — streaming is natural. Java requires WebFlux (reactive programming) for the same result:

```
NestJS (natural):
  FastAPI SSE → NestJS stream.pipe(res) → WebSocket to browser
  (non-blocking, no thread pool pressure)

Java Spring MVC (problematic for streaming):
  FastAPI SSE → blocks one thread per open connection → thread pool exhaustion

Java Spring WebFlux (solves it, but adds complexity):
  FastAPI SSE → Project Reactor Mono/Flux → browser
  (correct, but steep learning curve)
```

### The Emerging Option: Go

Growing adoption at AI infrastructure companies (Weaviate, Qdrant are written in Go):

```
Next.js → Go (Gin / Echo / Fiber) → FastAPI
```

| Advantage   | Detail                                                  |
| ----------- | ------------------------------------------------------- |
| Performance | Near-Java throughput with near-Node simplicity          |
| Concurrency | Goroutines — better streaming than both NestJS and Java |
| Deployment  | Tiny static binaries, fast cold starts, ideal for K8s   |
| Tradeoff    | Smaller talent pool; no TypeScript type sharing         |

### Recommendation

```
✅ NestJS        — TypeScript-native team, AI-first product, monorepo, streaming
✅ Spring Boot   — existing enterprise Java platform, regulated industry
✅ Go            — high-performance AI infrastructure, polyglot engineering team
```

For most AI products starting today: **NestJS is the pragmatic winner** — same language as the frontend, native streaming, monorepo-friendly. Java is the right answer when organizational constraints make it the obvious fit, not because it is technically superior for AI workloads.

---

## API Gateway Layer

### Gateway vs Middle Server — Different Responsibilities

```
                          ┌─────────────────────────────────────────┐
  INFRASTRUCTURE LAYER →  │            API GATEWAY                  │
                          │  Nginx / Cloudflare / AWS API Gateway   │
                          │                                         │
                          │  ✓ SSL/TLS termination                  │
                          │  ✓ IP-level rate limiting               │
                          │  ✓ DDoS protection                      │
                          │  ✓ Load balancing (multiple instances)  │
                          │  ✓ proxy_buffering off  (AI streaming)  │
                          └──────────────────┬──────────────────────┘
                                             │
                          ┌──────────────────▼──────────────────────┐
  APPLICATION LAYER →     │              NestJS (BFF)               │
                          │                                         │
                          │  ✓ JWT authentication                   │
                          │  ✓ User-level rate limiting             │
                          │  ✓ Business logic                       │
                          │  ✓ Database access (Prisma)             │
                          │  ✓ Chat history management              │
                          │  ✓ Proxy AI requests to FastAPI         │
                          └──────────────────┬──────────────────────┘
                                             │  internal only
                          ┌──────────────────▼──────────────────────┐
  AI LAYER →              │         FastAPI (Python)                │
                          │                                         │
                          │  ✓ LangChain agent orchestration        │
                          │  ✓ Claude API (streaming)               │
                          │  ✓ Milvus RAG                           │
                          │  ✓ Tool calling                         │
                          └─────────────────────────────────────────┘
```

They are complementary, not competing. The gateway is infrastructure; NestJS is application logic.

### Nginx Configuration (the critical streaming settings)

```nginx
upstream nestjs  { server nestjs:3000; }

server {
    listen 443 ssl http2;
    ssl_certificate     /etc/nginx/certs/fullchain.pem;
    ssl_certificate_key /etc/nginx/certs/privkey.pem;

    # Static assets → Next.js
    location / {
        proxy_pass http://nextjs:3001;
    }

    # API traffic → NestJS
    location /api/ {
        proxy_pass            http://nestjs;
        proxy_set_header      X-Real-IP      $remote_addr;
        proxy_set_header      X-Forwarded-For $proxy_add_x_forwarded_for;

        # ── Critical for AI token streaming ──────────────────────
        proxy_buffering       off;
        proxy_cache           off;
        proxy_read_timeout    300s;     # long timeout for AI responses
        proxy_http_version    1.1;
        proxy_set_header      Connection "";
        # ─────────────────────────────────────────────────────────
    }

    # WebSocket endpoint
    location /ws/ {
        proxy_pass            http://nestjs;
        proxy_http_version    1.1;
        proxy_set_header      Upgrade    $http_upgrade;
        proxy_set_header      Connection "upgrade";
    }
}
```

### NestJS Built-in Gateway (Option for MVP)

When you do not yet need Nginx load balancing, implement gateway concerns inside NestJS:

```typescript
// main.ts — application bootstrap
async function bootstrap() {
  const app = await NestFactory.create(AppModule);

  // Global JWT guard — all routes protected by default
  app.useGlobalGuards(app.get(JwtAuthGuard));

  // Global rate limiting — @nestjs/throttler
  // (configured per-module for fine-grained control)

  // Global logging interceptor
  app.useGlobalInterceptors(new LoggingInterceptor());

  await app.listen(3000);
}

// ai-proxy.service.ts — forward to FastAPI with streaming
@Get('agent/stream')
async streamAgent(@Req() req: Request, @Res() res: Response) {
  const upstream = await this.httpService.axiosRef({
    method: 'POST',
    url: `${this.fastApiUrl}/agent/chat`,
    data: req.body,
    responseType: 'stream',
  });

  // Node.js native pipe — zero buffering, tokens flow through immediately
  upstream.data.pipe(res);
}
```

### Gateway Options by Stage

| Stage        | Gateway                     | Reason                                               |
| ------------ | --------------------------- | ---------------------------------------------------- |
| MVP          | NestJS only                 | No extra infrastructure; TypeScript you already know |
| Production   | Nginx + NestJS              | Load balance multiple NestJS instances               |
| Enterprise   | Cloudflare + Nginx + NestJS | Global DDoS, edge SSL, internal load balancing       |
| Cloud-native | AWS API GW + ALB            | Managed infrastructure, pay-per-request              |

---

## Deployment Architecture

### Stage 1 — Docker Compose (Single VM)

The full stack on one server. Production-capable for early products.

```yaml
# docker-compose.yml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on: [web, nestjs]

  web:
    build:
      context: ./apps/web
      dockerfile: Dockerfile
    environment:
      NEXTAUTH_URL: https://yourdomain.com
      NEXTAUTH_SECRET: ${NEXTAUTH_SECRET}
      NESTJS_URL: http://nestjs:3000

  nestjs:
    build:
      context: ./apps/api
      dockerfile: Dockerfile
    environment:
      DATABASE_URL: postgresql://postgres:${DB_PASSWORD}@postgres:5432/app
      REDIS_URL: redis://redis:6379
      FASTAPI_URL: http://fastapi:8000
      JWT_SECRET: ${JWT_SECRET}
    depends_on: [postgres, redis]

  fastapi:
    build:
      context: ./apps/ai
      dockerfile: Dockerfile
    environment:
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      MILVUS_HOST: milvus
      MILVUS_PORT: 19530
    depends_on: [milvus]
    # FastAPI is INTERNAL ONLY — no public port exposed

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: app
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  milvus:
    image: milvusdb/milvus:v2.4.0
    environment:
      ETCD_ENDPOINTS: etcd:2379
    depends_on: [etcd, minio]

  etcd:
    image: quay.io/coreos/etcd:v3.5.5

  minio:
    image: minio/minio:latest
    command: minio server /minio_data

volumes:
  postgres_data:
  redis_data:
```

### Stage 2 — Multiple Instances (Scale-out)

When a single VM is insufficient, scale NestJS horizontally behind Nginx:

```nginx
upstream nestjs {
    server nestjs-1:3000;
    server nestjs-2:3000;
    server nestjs-3:3000;
    # Nginx handles load balancing — NestJS must be stateless (use Redis for sessions)
}
```

FastAPI can also be scaled the same way — each instance is stateless.

### Stage 3 — Kubernetes

When you need autoscaling, rolling deployments, and cloud-managed infrastructure:

```
Cloudflare / AWS API Gateway    ← edge protection, global SSL
         ↓
  Kubernetes Cluster
  ├── Ingress (Nginx Ingress Controller)   ← replaces standalone Nginx
  ├── Deployment: web (Next.js)            ← 2–N replicas
  ├── Deployment: api (NestJS)             ← 2–N replicas, HPA autoscaling
  ├── Deployment: ai  (FastAPI)            ← 2–N replicas, GPU nodes optional
  └── Managed Services
      ├── PostgreSQL  (RDS / Cloud SQL)
      ├── Redis       (ElastiCache / Memorystore)
      └── Milvus      (Zilliz Cloud / self-hosted)
```

---

## Your Practical Learning Path

> **Current skills:** TypeScript · JavaScript · NestJS · Nginx · Docker
> **New to:** Python stack · Kubernetes · Cloudflare · AWS API Gateway

### What to Learn Now (for Stage 1)

Everything at Stage 1 uses skills you already have, except FastAPI:

```
✅ Already know:   NestJS · Nginx · Docker Compose
🆕 Learn now:      FastAPI basics
                   ├── Python async/await (similar to JS async)
                   ├── Pydantic models (similar to Zod/TypeScript interfaces)
                   ├── StreamingResponse for SSE
                   └── LangChain @tool · bind_tools() · agent_loop.py pattern
```

FastAPI will feel familiar — it uses decorators like NestJS (`@app.get`, `@app.post`), dependency injection, and Pydantic schemas that mirror TypeScript interfaces. The async model is nearly identical to Node.js.

### What to Learn Later (for Stage 2–3)

Add these when scaling demands it, not before:

| Technology    | Learn when...                                     | Estimated effort |
| ------------- | ------------------------------------------------- | ---------------- |
| Kubernetes    | Docker Compose VM can no longer handle load       | 2–4 weeks        |
| Cloudflare    | Need DDoS protection or global CDN                | 1 afternoon      |
| AWS / GCP     | Need managed DB, autoscaling, or compliance certs | 2–3 weeks        |
| gRPC          | NestJS → FastAPI calls become a bottleneck        | 1 week           |
| Message Queue | AI jobs exceed HTTP timeout limits (> 30s)        | 1 week           |

### The Key Insight

**Your Nginx + Docker skills already cover Stage 1 production.** Kubernetes and cloud gateways solve scale problems, not correctness problems. Ship with what you know, then add infrastructure when the traffic demands it.

```
Stage 1  →  Docker Compose + Nginx + NestJS + FastAPI   (ship this)
Stage 2  →  Multi-instance Nginx upstream               (add when traffic grows)
Stage 3  →  Kubernetes + Cloudflare                     (add when Stage 2 is not enough)
```
