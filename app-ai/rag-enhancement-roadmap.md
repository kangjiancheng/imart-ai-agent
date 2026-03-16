# RAG Enhancement Roadmap

Analysis of current app-ai RAG implementation vs. industry enhancement techniques,
with prioritized implementation suggestions.

Reference: [Milvus — How to Enhance Your RAG](https://milvus.io/docs/how_to_enhance_your_rag.md)

---

## Current State Summary

| Component           | File                          | What it does                                            |
| ------------------- | ----------------------------- | ------------------------------------------------------- |
| Dense embedding     | `src/rag/embeddings.py`       | BGE-M3 via ModelScope, returns `dense_vecs` only        |
| Knowledge retrieval | `src/rag/retriever.py`        | Single dense vector search, MIN_SCORE=0.72, TOP_K=5     |
| Reranker            | `src/rag/reranker.py`         | Cohere cross-encoder stub — **never called**            |
| Memory              | `src/memory/vector_memory.py` | Per-user Milvus collection, stores tool-usage summaries |
| Agent loop          | `src/agent/agent_loop.py`     | ReAct loop, Claude decides when to call `rag_retrieve`  |
| Schema              | `src/rag/milvus_utils.py`     | Single `FLOAT_VECTOR` field (1024-dim), COSINE index    |
| Guardrails          | `src/guardrails/checker.py`   | Content policy → PII redaction → injection detection    |

---

## Gap Analysis: Current vs. Recommended Enhancements

### 1. Query Enhancement

| Technique              | Status             | Notes                          |
| ---------------------- | ------------------ | ------------------------------ |
| Hypothetical Questions | ❌ Not implemented | Query goes raw to Milvus       |
| HyDE                   | ❌ Not implemented | No fake-answer embedding step  |
| Sub-Queries            | ❌ Not implemented | Complex queries not decomposed |
| Stepback Prompts       | ❌ Not implemented | No query abstraction           |

**Current flow:** `user_query → embed → Milvus search`
**Impact if unchanged:** Multi-part or vague queries return poor results.

---

### 2. Indexing Enhancement

| Technique                       | Status             | Notes                                                                           |
| ------------------------------- | ------------------ | ------------------------------------------------------------------------------- |
| Chunk Merging (parent/child)    | ❌ Not implemented | Single-level chunks only, no `parent_id` field                                  |
| Hierarchical Index              | ❌ Not implemented | One collection, no summary layer                                                |
| Hybrid Retrieval (BM25 + dense) | ❌ Not implemented | BGE-M3 computes `lexical_weights` but code discards them at `embeddings.py:144` |

**Key finding:** BGE-M3 already outputs `lexical_weights` (sparse/BM25-style) on every
encode call. The data is free — it's just being thrown away. Adding hybrid search
requires a Milvus schema migration (add `SPARSE_FLOAT_VECTOR` field) but no new model.

---

### 3. Retriever Enhancement

| Technique                 | Status             | Notes                                                       |
| ------------------------- | ------------------ | ----------------------------------------------------------- |
| Sentence Window Retrieval | ❌ Not implemented | Retrieved chunk = exactly what Claude gets                  |
| Metadata Filtering        | ⚠️ Partial         | Memory uses `user_id` filter; knowledge base has no filters |

**Current:** `output_fields=["content", "source"]` — no category, date, or department.
**Memory pattern exists:** `vector_memory.py:141` already uses `filter=f'user_id == "{user_id}"'`
— the same pattern just needs to be applied to the knowledge base.

---

### 4. Generator Enhancement

| Technique          | Status             | Notes                                       |
| ------------------ | ------------------ | ------------------------------------------- |
| Prompt Compression | ❌ Not implemented | All retrieved chunks passed whole to Claude |
| Adjust Chunk Order | ❌ Not implemented | Chunks ordered by score descending only     |

**"Lost in the middle" risk:** Claude tends to ignore content in the middle of long
context windows. High-confidence chunks should be placed first and last, not just first.

---

### 5. RAG Pipeline Enhancement

| Technique                        | Status                     | Notes                                                    |
| -------------------------------- | -------------------------- | -------------------------------------------------------- |
| Self-Reflection / Corrective RAG | ❌ Not implemented         | No second-pass verification                              |
| Query Routing                    | ✅ Implemented (via tools) | Claude chooses to call `rag_retrieve` only when relevant |

**Strongest current feature:** Tool-based routing is the right pattern — Claude naturally
skips RAG for simple questions. This matches the "query routing agent" technique.

---

## Prioritized Implementation Roadmap

### Tier 1 — Fix broken/unused existing code (low effort, immediate return)

#### T1-A: Wire up the reranker

- **File:** `src/rag/retriever.py`, `src/rag/reranker.py`
- **What:** Call `rerank(query, [doc.content for doc in docs])` after `retrieve()` returns,
  before `format_for_prompt()`.
- **Why now:** Already coded. Zero schema change. Highest quality-per-effort ratio.
- **Fallback:** `reranker.py` already silently falls back to original order if Cohere is unavailable.
- **Effort:** ~10 lines in `retriever.py`

#### T1-B: Store useful memory content

- **File:** `src/agent/agent_loop.py`, `src/memory/vector_memory.py`
- **What:** Instead of storing `"Agent used 2 tool call(s): calculator"`, extract and store
  user-relevant facts: preferences, domain context, decisions made.
- **Why now:** Memory is your most differentiating feature. Right now it stores almost nothing useful.
- **Approach:** After the loop, run a short LLM call: `"Summarize any user facts, preferences,
or decisions from this conversation in 1-3 bullet points."`
- **Effort:** ~20 lines in `agent_loop.py`

#### T1-C: Add structured logging / observability

- **File:** `src/agent/agent_loop.py`
- **What:** Replace `print()` statements with structured log entries containing:
  `user_id`, `session_id`, `latency_ms`, `tokens_used`, `tools_called`, `rag_score`
- **Why now:** Cannot debug or monitor production without this.
- **Effort:** Add `logging` module, replace 1 print statement, add timing around `ainvoke`

---

### Tier 2 — RAG quality improvements (medium effort, high return)

#### T2-A: BM25 hybrid retrieval

- **Files:** `src/rag/milvus_utils.py`, `src/rag/embeddings.py`, `src/rag/retriever.py`
- **What:**
  1. Add `embed_sparse()` method to `EmbeddingClient` — return `lexical_weights` from BGE-M3
  2. Add `SPARSE_FLOAT_VECTOR` field + `SPARSE_INVERTED_INDEX` to `milvus_utils.py` schema
  3. Run two parallel searches in `retriever.py` (dense + sparse)
  4. Merge with RRF (Reciprocal Rank Fusion): `score = 1/(k + rank_dense) + 1/(k + rank_sparse)`
- **Warning:** Requires dropping and re-creating the Milvus collection (breaking schema change).
  Back up ingested documents first.
- **When it helps most:** Exact product codes, SKUs, policy article numbers, proper nouns.
- **Effort:** ~60 lines across 3 files + data re-ingestion

#### T2-B: Metadata filtering on knowledge base

- **File:** `src/rag/milvus_utils.py`, `src/routers/rag.py`, `src/rag/retriever.py`
- **What:** Add `department`, `doc_type`, `created_at` fields to the knowledge base schema.
  Accept optional filter params in `retrieve(query, filters={})`.
- **Why:** The pattern already exists in `vector_memory.py` — just apply it to the knowledge base.
- **Effort:** ~30 lines

#### T2-C: Sentence window retrieval

- **File:** `src/rag/milvus_utils.py`, `src/routers/rag.py`, `src/rag/retriever.py`
- **What:** At ingest time, store both a small embedding chunk AND a reference to its
  surrounding larger context window. At retrieval time, return the larger window to Claude.
- **Schema change:** Add `parent_content` or `context_window` field at ingest time.
- **Effort:** ~40 lines

---

### Tier 3 — Query intelligence (higher effort, enterprise value)

#### T3-A: Query routing (explicit classifier)

- **File:** `src/agent/agent_loop.py` (new pre-loop step)
- **What:** Before the ReAct loop, classify the query:
  `small_talk | factual_rag | tool_use | document_analysis`
  Route directly to the right handler, skipping unnecessary tool-binding overhead.
- **Current state:** Claude does this implicitly via tool descriptions. An explicit classifier
  is faster and more predictable at scale.
- **Effort:** ~30 lines + a small prompt

#### T3-B: Sub-query decomposition

- **File:** `src/agent/agent_loop.py` or new `src/agent/query_planner.py`
- **What:** Detect compound queries before retrieval. Split into sub-queries, retrieve for each,
  merge results before passing to Claude.
- **Example:** `"Compare Milvus and Zilliz Cloud features"` → two separate RAG calls
- **Effort:** ~50 lines

#### T3-C: Self-reflection / Corrective RAG

- **File:** `src/rag/retriever.py` or `src/agent/agent_loop.py`
- **What:** After retrieval, run a quick LLM check: `"Do these documents answer the query? yes/no"`
  If no — trigger web search fallback. Inspired by Self-RAG and Corrective RAG papers.
- **Effort:** ~40 lines + extra LLM call per RAG invocation (adds latency)

---

### Tier 4 — Enterprise-specific (beyond RAG quality)

These are not covered in the Milvus article but separate a demo from production software.

#### T4-A: Multi-tenancy / access control

- **What:** Users should only retrieve documents they have permission to see.
  Add `access_level` or `allowed_roles` to the knowledge base schema.
  Filter at retrieval time based on the authenticated user's role.
- **Required for:** Any multi-department or customer-facing deployment.

#### T4-B: Audit trail

- **What:** Persist every query, retrieved document set, and final response to a
  relational DB (PostgreSQL). Required for compliance (SOC2, GDPR, HIPAA).
- **Not in Milvus:** Milvus is not an audit log — it's a search index.

#### T4-C: Evaluation pipeline

- **What:** A test suite that measures RAG quality metrics:
  - **Retrieval precision** — are retrieved chunks relevant?
  - **Answer faithfulness** — does Claude's answer match the retrieved docs?
  - **Answer relevance** — does the answer address the query?
- **Tools:** Ragas, TruLens, or a custom eval harness.
- **Why critical:** Without metrics you cannot tell if a change improved or worsened quality.

#### T4-D: HyDE (Hypothetical Document Embeddings)

- **File:** `src/rag/retriever.py`
- **What:** Before the real search, ask Claude to generate a "fake ideal answer".
  Embed that fake answer and use it as the search vector instead of the raw query.
  Better for vague or poorly-worded queries.
- **Tradeoff:** Adds one extra LLM call per retrieval. Worth it for question-answering
  over long technical documents, not worth it for keyword lookups.

---

## Implementation Order (recommended sequence)

```
Phase 1 — Make existing code work (1-2 days)
  T1-A  Wire up reranker
  T1-B  Store useful memory content
  T1-C  Structured logging

Phase 2 — RAG quality (3-5 days)
  T2-A  BM25 hybrid retrieval        ← schema migration required
  T2-B  Metadata filtering
  T2-C  Sentence window retrieval

Phase 3 — Query intelligence (1 week)
  T3-A  Explicit query routing
  T3-B  Sub-query decomposition
  T3-C  Self-reflection / Corrective RAG

Phase 4 — Enterprise hardening (ongoing)
  T4-A  Multi-tenancy / access control
  T4-B  Audit trail (PostgreSQL)
  T4-C  Evaluation pipeline
  T4-D  HyDE
```

---

## Quick Reference: Enhancement vs. Effort vs. Impact

| Enhancement                  | Effort | Quality Impact | Schema Change | Priority |
| ---------------------------- | ------ | -------------- | ------------- | -------- |
| T1-A Wire up reranker        | Low    | High           | No            | **Now**  |
| T1-B Better memory content   | Low    | High           | No            | **Now**  |
| T1-C Structured logging      | Low    | Operational    | No            | **Now**  |
| T2-A BM25 hybrid retrieval   | Medium | High           | **Yes**       | Phase 2  |
| T2-B Metadata filtering      | Low    | Medium         | **Yes**       | Phase 2  |
| T2-C Sentence window         | Medium | Medium         | **Yes**       | Phase 2  |
| T3-A Query routing           | Medium | Medium         | No            | Phase 3  |
| T3-B Sub-query decomposition | Medium | High           | No            | Phase 3  |
| T3-C Self-reflection         | Medium | High           | No            | Phase 3  |
| T4-A Multi-tenancy           | High   | Compliance     | **Yes**       | Phase 4  |
| T4-B Audit trail             | High   | Compliance     | No            | Phase 4  |
| T4-C Eval pipeline           | High   | Operational    | No            | Phase 4  |
| T4-D HyDE                    | Low    | Medium         | No            | Phase 4  |
