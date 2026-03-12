# RAG PDF Ingestion — Comprehensive Guide

This document covers everything about PDF ingestion in the RAG pipeline:
the bug we hit, why it happened, how we fixed it step by step, and
what the industry recommends for production systems.

---

## Table of Contents

1. [How RAG Ingestion Works](#1-how-rag-ingestion-works)
2. [The Root Bug: PDF Decoded as UTF-8](#2-the-root-bug-pdf-decoded-as-utf-8)
3. [Fix 1: pypdf (first attempt)](#3-fix-1-pypdf-first-attempt)
4. [Why pypdf Still Failed on Table-Heavy PDFs](#4-why-pypdf-still-failed-on-table-heavy-pdfs)
5. [PDF Parsing Libraries — Full Comparison](#5-pdf-parsing-libraries--full-comparison)
6. [LangChain Document Loaders](#6-langchain-document-loaders)
7. [Tiered Parsing: PyMuPDF + Unstructured](#7-tiered-parsing-pymupdf--unstructured)
8. [What the Industry Uses](#8-what-the-industry-uses)
9. [Fix 2: LangChain + Unstructured (attempted, abandoned)](#9-fix-2-langchain--unstructured-attempted-abandoned)
10. [Fix 3: Unstructured Dependency Conflict Chain](#10-fix-3-unstructured-dependency-conflict-chain)
11. [Fix 4: pdfplumber (attempted, abandoned)](#11-fix-4-pdfplumber-attempted-abandoned)
12. [Final Fix: PyMuPDF + OCR Fallback (current implementation)](#12-final-fix-pymupdf--ocr-fallback-current-implementation)
13. [Tuning: RAG_MIN_SCORE for OCR-noisy text](#13-tuning-rag_min_score-for-ocr-noisy-text)
14. [New Endpoint: GET /v1/rag/content](#14-new-endpoint-get-v1ragcontent)
15. [Debug Endpoints: GET and DELETE /v1/rag/chunks](#15-debug-endpoints-get-and-delete-v1ragchunks)
16. [Re-ingesting Documents After a Parser Change](#16-re-ingesting-documents-after-a-parser-change)

---

## 1. How RAG Ingestion Works

RAG (Retrieval-Augmented Generation) lets the AI answer questions about
**your private documents** instead of only relying on its training data.

The ingestion pipeline has two phases:

```
INGEST TIME (one-time setup)
  Document file
    │
    ▼
  Parse: extract readable text from the binary file format
    │
    ▼
  Chunk: split into smaller pieces (1000 chars, 200 overlap)
    │
    ▼
  Embed: convert each chunk → 1024-float vector (BGE-M3 model)
    │
    ▼
  Store: save {text, vector, source} into Milvus

QUERY TIME (every chat message)
  User question
    │
    ▼
  Embed the question → query vector
    │
    ▼
  Search Milvus for nearby vectors (cosine similarity)
    │
    ▼
  Return top-K chunks with score ≥ MIN_SCORE (0.72)
    │
    ▼
  Claude reads the chunks and answers accurately
```

The **parse step** is where everything went wrong for PDFs.

---

## 2. The Root Bug: PDF Decoded as UTF-8

### What the original code did

```python
content = await file.read()          # raw bytes from the uploaded file
text = content.decode("utf-8", errors="ignore")  # ← WRONG for PDFs
```

### Why this is wrong

A PDF is a **binary file format**. It contains:
- Compressed content streams (zlib/deflate)
- Font programs and glyph mappings
- Binary cross-reference tables
- Embedded images

When you call `.decode("utf-8", errors="ignore")`, Python reads the binary
bytes and silently drops every byte that isn't valid UTF-8. For a PDF, that
is almost everything — leaving a nearly empty string like:

```
"PDF-1.4  obj endobj  xref  %%EOF"
```

This empty string gets chunked into nothing useful, embedded as a near-zero
vector, and stored in Milvus. At query time, cosine similarity against the
garbage chunks is far below `MIN_SCORE = 0.72`, so **RAG returns nothing**.

The agent then says "no related documents found" — even though the file was
successfully uploaded.

### Key insight

`errors="ignore"` is the silent killer here. It doesn't raise an error —
it just produces empty output. The ingest endpoint returns `{"status": "done"}`
but the knowledge base has no usable content.

---

## 3. Fix 1: pypdf (first attempt)

### What we did

Detected PDFs by MIME type or file extension and used `pypdf` to extract text:

```python
if content_type == "application/pdf" or filename_lower.endswith(".pdf"):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(content))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages_text)
else:
    text = content.decode("utf-8", errors="ignore")
```

### Why this is correct for the binary problem

`pypdf` understands the PDF binary format and extracts actual Unicode text —
Chinese characters and all. The binary decode bug was fixed.

### Install

```bash
pip install pypdf
```

### Limitation discovered next

This fixed text-heavy PDFs (articles, reports) but **still failed** on our
IoT requirements document, which is structured almost entirely as a table.

---

## 4. Why pypdf Still Failed on Table-Heavy PDFs

### The second problem: table layout

`pypdf`'s `extract_text()` reads text in the order it appears in the PDF's
content stream — which for tables is often **column-by-column or in a
non-reading order**. A table like:

```
| 功能类     | 子功能   | 功能描述         |
| 核心看板   | 设备概览 | 展示累计激活设备数 |
```

Gets extracted as something like:

```
功能类子功能核心看板设备概览展示累计激活设备数功能描述
```

This produces **incoherent chunks** with no sentence structure. The BGE-M3
embedding model cannot find semantic meaning in garbled text, so the cosine
similarity against a natural-language query ("IoT需求有哪些") is low —
below `MIN_SCORE = 0.72` — and the chunks are filtered out.

### Two-layer problem summary

| Layer | Problem | Symptom |
|-------|---------|---------|
| Layer 1 | PDF binary decoded as UTF-8 | Completely empty chunks |
| Layer 2 | pypdf extracts table text in wrong order | Incoherent chunks, low similarity scores |

Both layers must be fixed for table-heavy PDFs to work.

---

## 5. PDF Parsing Libraries — Full Comparison

### pypdf

```bash
pip install pypdf
```

- Pure Python, no system dependencies
- Good for text-heavy PDFs (academic papers, reports)
- Poor table support — reads cells out of reading order
- Best choice only when you need the smallest install

### pdfplumber

```bash
pip install pdfplumber
```

- Built on `pdfminer.six`
- Extracts tables as structured rows/columns
- Can give you `page.extract_table()` → `list[list[str]]`
- Better than pypdf for table-heavy documents
- Example:

```python
import pdfplumber
with pdfplumber.open(io.BytesIO(content)) as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()
        for table in tables:
            for row in table:
                print(" | ".join(cell or "" for cell in row))
```

### pymupdf (fitz)

```bash
pip install pymupdf
```

- Fastest parser (~10x faster than pdfminer)
- Best CJK (Chinese/Japanese/Korean) character support
- Handles complex layouts, embedded fonts well
- Simple API:

```python
import fitz
doc = fitz.open(stream=content, filetype="pdf")
text = "\n\n".join(page.get_text() for page in doc)
```

- Does not understand document structure (no concept of "this is a table cell")

### pdfminer.six

```bash
pip install pdfminer.six
```

- The lowest-level PDF text extraction library
- Most control over text position data (X/Y coordinates per character)
- Complex API — rarely used directly
- What `pdfplumber` is built on top of
- Use `pdfplumber` instead unless you need raw layout coordinates

### unstructured

```bash
pip install unstructured pdfminer.six pillow
```

- **Structure-aware**: knows the difference between a title, a table cell,
  a list item, and a paragraph — returns typed `Element` objects
- Purpose-built for RAG ingestion pipelines
- `mode="elements"` returns each structural element as a separate `Document`
- `strategy="fast"` uses pdfminer.six (no system dependencies)
- `strategy="hi_res"` uses ML layout detection (needs `detectron2` + `poppler`)
- Has a cloud API (`unstructured.io`) for high-volume production use
- Example:

```python
from unstructured.partition.pdf import partition_pdf
elements = partition_pdf("document.pdf", strategy="fast")
# elements = [Title("IoT管理后台需求"), Table("功能类 子功能..."), NarrativeText("...")]
```

### Summary table

| Library | Table support | CJK | RAG-native | Install size | Speed |
|---------|--------------|-----|------------|--------------|-------|
| `pypdf` | Poor | OK | No | Small | Fast |
| `pdfplumber` | Good | OK | No | Medium | Medium |
| `pymupdf` | Good | Excellent | No | Medium | Fastest |
| `pdfminer.six` | Medium | OK | No | Medium | Slow |
| `unstructured` | Excellent | Good | Built-in | Large | Slow |

---

## 6. LangChain Document Loaders

LangChain wraps all the parsers above behind **one consistent API**.
Every loader returns `list[Document]` — the same type — so you can swap
parsers without changing anything downstream.

```python
from langchain_community.document_loaders import PyMuPDFLoader       # uses pymupdf
from langchain_community.document_loaders import PDFPlumberLoader     # uses pdfplumber
from langchain_community.document_loaders import UnstructuredPDFLoader # uses unstructured
from langchain_community.document_loaders import PyPDFLoader          # uses pypdf
```

### Why this fits our project

The project already uses:
- `langchain-core`
- `langchain-anthropic`
- `langchain-text-splitters` (`RecursiveCharacterTextSplitter` in `chunker.py`)

Adding `langchain-community` is a natural extension — not a new philosophy.

### What a LangChain Document looks like

```python
doc.page_content  # → "展示累计激活设备数、今日新增..." (the text)
doc.metadata      # → {"source": "IoT管理后台需求.pdf", "page": 1, "category": "Table"}
```

### LangChain + text splitter integration

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# With raw text (non-PDF path):
chunks = splitter.split_text(text)           # → list[str]

# With Document objects (PDF path):
chunks = splitter.split_documents(docs)      # → list[Document]
texts  = [c.page_content for c in chunks]   # → list[str]
```

`split_documents()` preserves metadata through the chunking — each chunk
knows which page and category it came from.

---

## 7. Tiered Parsing: PyMuPDF + Unstructured

For high-volume production systems, a tiered approach avoids paying the
Unstructured penalty on every PDF:

```
PDF uploaded
    │
    ▼
PyMuPDF — fast extraction (~50ms)
    │
    ├─ avg chars/page ≥ 200? ──YES──► text-dominant PDF, use PyMuPDF → done
    │
    └─ NO (sparse text = table-heavy or scanned)
         │
         ▼
    Unstructured — structure-aware extraction (~2–5s)
         │
         ▼
    chunk_by_title() → semantic chunks → done
```

```python
import fitz
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

def extract_pdf_chunks(content: bytes) -> list[str]:
    # Fast path
    doc = fitz.open(stream=content, filetype="pdf")
    pages_text = [page.get_text() for page in doc]
    total_text = "\n\n".join(pages_text)
    avg_chars = len(total_text) / max(len(pages_text), 1)

    if avg_chars >= 200:
        return [total_text]  # chunker.py splits this further

    # Slow path — structure-aware
    import io, tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        elements = partition_pdf(tmp_path, strategy="fast")
        chunks = chunk_by_title(elements, max_characters=1000)
        return [str(c) for c in chunks]
    finally:
        os.unlink(tmp_path)
```

### When to use the tiered approach

| Scenario | Recommendation |
|----------|---------------|
| Learning project / dev | Unstructured-only (simpler) |
| Production, mixed PDF types | Tiered (PyMuPDF fast path + Unstructured fallback) |
| High volume (1000s PDFs/day) | Unstructured Cloud API |

---

## 8. What the Industry Uses

### Early-stage startups (most common pattern)

```
Unstructured → LangChain/LlamaIndex chunking → any vector DB
```

Unstructured has become the **de facto standard** for RAG ingestion because:
- Handles all file types with one API (PDF, DOCX, HTML, Excel, images)
- RAG-native design — typed elements map naturally to chunks
- Huge community and active development

### Mid-stage / scaling

```
Unstructured Cloud API → vector DB
```

Offload parsing to Unstructured's hosted service. Pay-per-page pricing.
Your server doesn't carry the compute cost.

### Enterprise / big tech

```
Docling (IBM) or Azure Document Intelligence → vector DB
```

Better accuracy for complex documents (formulas, multi-column layouts).
Enterprise SLAs and support.

### Popularity ranking

| Rank | Tool | Why popular |
|------|------|-------------|
| 1 | **Unstructured** | All file types, RAG-native, huge community |
| 2 | **LangChain document loaders** | Wraps PyMuPDF/pdfplumber, easy to swap parsers |
| 3 | **LlamaIndex readers** | Strong ecosystem, similar to LangChain |
| 4 | **Azure Document Intelligence** | Best accuracy, enterprise-grade, paid |
| 5 | **Docling (IBM)** | Rising fast, great table and formula support |

---

## 9. Fix 2: LangChain + Unstructured (attempted, abandoned)

### Why this combination looked good

- **LangChain**: already in the project's stack — consistent `list[Document]` API
- **Unstructured**: structure-aware parsing — solves the table layout problem
- **`strategy="fast"`**: uses `pdfminer.six`, no system dependencies (poppler/detectron2)
- **`mode="elements"`**: each structural element (Title, Table, NarrativeText) becomes
  its own Document → smarter chunk boundaries

> **This approach was abandoned** — see [section 10](#10-fix-3-unstructured-dependency-conflict-chain)
> for why it could not coexist with `FlagEmbedding`.

### Files changed

| File | Change |
|------|--------|
| `requirements.txt` | Removed `pypdf`, added `langchain-community`, `unstructured`, `pdfminer.six`, `pillow` |
| `src/rag/chunker.py` | Added `chunk_documents(list[Document]) -> list[str]` |
| `src/routers/rag.py` | Replaced pypdf block with `UnstructuredPDFLoader` |

### Install (attempted)

```bash
pip install langchain-community unstructured pdfminer.six pillow
```

### How it worked

```python
import os, tempfile
from langchain_community.document_loaders import UnstructuredPDFLoader
from src.rag.chunker import chunk_documents

# Write bytes to temp file (UnstructuredPDFLoader needs a file path)
with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
    tmp.write(content)
    tmp_path = tmp.name

try:
    loader = UnstructuredPDFLoader(tmp_path, mode="elements", strategy="fast")
    docs = loader.load()
    # docs = list[Document], each element typed (Title, Table, NarrativeText…)
    chunks = chunk_documents(docs)
    # chunk_documents() re-splits large elements, filters empty ones
    # returns list[str] — same type as chunk_text() for the non-PDF path
finally:
    os.unlink(tmp_path)  # always delete temp file
```

### New function in `chunker.py`

```python
def chunk_documents(docs: list, chunk_size=1000, chunk_overlap=200) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = splitter.split_documents(docs)
    return [doc.page_content for doc in split_docs if doc.page_content.strip()]
    # Returns list[str] — identical type to chunk_text() output
    # The Milvus insert loop in rag.py stays completely unchanged
```

---

## 10. Fix 3: Unstructured Dependency Conflict Chain

This is why `unstructured` was ultimately removed from the project entirely.

### The conflict chain

```
unstructured
  └─► unstructured_inference   (eager import at module load — cannot avoid it)
        └─► transformers v5.x  (unstructured_inference forces an upgrade)
              └─► BREAKS FlagEmbedding==1.3.5
                    (FlagEmbedding requires transformers==4.44.2, which removed
                     the symbol `is_torch_fx_available` in v5.x)
```

At server startup, Python imports `unstructured`, which immediately imports
`unstructured_inference` — even when `strategy="fast"` is specified. This is
unconditional: there is no way to import `unstructured` without also triggering
`unstructured_inference`.

`unstructured_inference` in turn forces `pip` to upgrade `transformers` to v5.x.
`FlagEmbedding==1.3.5` (the BGE-M3 embedding library) calls
`transformers.is_torch_fx_available`, a symbol that was **removed** in v5.x.

The resulting crash at startup:

```
ImportError: cannot import name 'is_torch_fx_available' from 'transformers'
```

### Why this is unresolvable

| Constraint | Required version |
|-----------|-----------------|
| `FlagEmbedding==1.3.5` requires | `transformers==4.44.2` |
| `unstructured-inference` requires | `transformers>=5.x` |

These two requirements directly contradict each other. There is no version of
either library that satisfies both constraints simultaneously.

### Errors encountered in sequence

```
# After pip install unstructured:
ModuleNotFoundError: No module named 'pi_heif'
→ Fix: pip install pi-heif

# After pip install pi-heif:
ModuleNotFoundError: No module named 'unstructured_inference'
→ Fix: pip install unstructured-inference

# After pip install unstructured-inference:
ImportError: cannot import name 'is_torch_fx_available' from 'transformers'
→ Unresolvable — transformers v5 was forced, breaking FlagEmbedding
```

### Resolution

Remove the entire `unstructured` ecosystem from the project:

```bash
pip uninstall unstructured unstructured-inference pi-heif langchain-community -y
```

Remove from `requirements.txt`:
- `langchain-community`
- `unstructured`
- `pdfminer.six`
- `pillow`
- `pi-heif`
- `unstructured-inference`

Switch to `pdfplumber` — see [section 11](#11-fix-4-pdfplumber-attempted-abandoned).

---

## 11. Fix 4: pdfplumber (attempted, abandoned)

### Why pdfplumber looked promising

`pdfplumber` has zero dependency conflicts and extracts tables as structured rows:

| Requirement | pdfplumber |
|-------------|-----------|
| Understands PDF binary format | Yes — built on `pdfminer.six` |
| Extracts tables as structured rows | Yes — `page.extract_tables()` |
| Works in-memory (no temp file) | Yes — accepts `io.BytesIO` |
| Conflicts with `transformers==4.44.2` | No — zero transitive conflict |

### Why it failed — Chinese CID fonts

`pdfplumber` is built on `pdfminer.six`, which cannot decode **CID (Character
ID) fonts** — a non-standard font encoding common in PDFs produced by:

- WPS Office (Chinese office suite)
- Government/enterprise Chinese tools
- Older InDesign exports with embedded CJK fonts

For CID-encoded characters, `pdfminer.six` substitutes `\u0001` control
characters instead of the actual text. The result looks like:

```
"\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001"
```

These garbled chunks get embedded as near-zero vectors and are always below
`MIN_SCORE` — so RAG returns nothing, even though chunks were stored.

### The deeper problem — image-based PDF

After switching to PyMuPDF (which does decode CID fonts), we discovered the
`IoT管理后台需求.pdf` file was **entirely image-based** (screenshot-style):
the PDF contained no text layer at all — every page was a raster image.
`page.get_text()` returned only `"IoT管理后台需求"` (10 chars — just the
document title outside the image). The actual requirements table was invisible
to any text-extraction library.

**Diagnosis command:**

```python
import fitz
doc = fitz.open("/path/to/file.pdf")
for i, page in enumerate(doc):
    print(f"Page {i+1}: {len(page.get_text())} chars text, {len(page.get_images())} images")
# Page 1: 10 chars text, 1 images
# Page 2: 0 chars text, 1 images
```

→ Switched to PyMuPDF + OCR fallback — see [section 12](#12-final-fix-pymupdf--ocr-fallback-current-implementation).

---

## 12. Final Fix: PyMuPDF + OCR Fallback (current implementation)

### Why PyMuPDF (fitz)

PyMuPDF uses the MuPDF C library which has broad CID font support — it
correctly decodes Chinese text that `pdfminer.six` renders as `\u0001`.
It also has zero dependency conflicts with `FlagEmbedding` / `transformers`.

| Requirement | PyMuPDF |
|-------------|---------|
| Decodes Chinese CID fonts | Yes — MuPDF C library |
| Table extraction (`find_tables`) | Yes — PyMuPDF ≥ 1.23 |
| Works in-memory | Yes — `fitz.open(stream=BytesIO(...))` |
| Conflicts with `transformers==4.44.2` | No |

### Why OCR fallback (tesseract)

Some PDFs — especially screenshots exported to PDF — contain **no text layer
at all**. Every page is a raster image; `page.get_text()` returns empty string
or just a title label. Tesseract OCR reads the image pixels directly.

**OCR trigger condition (two cases):**

```python
images_on_page = page.get_images()
needs_ocr = (
    not plain.strip()                           # case 1: completely empty page
    or (images_on_page and len(plain.strip()) < 50)  # case 2: image page with short text
)
```

Case 2 handles pages where `get_text()` returns only a short title (e.g.
`"IoT管理后台需求"` — 10 chars) while the actual content is embedded as an image.
The `< 50` threshold catches this without accidentally OCR-ing legitimate
short text-only pages (those won't have embedded images).

### Install

```bash
# Python packages
pip install pymupdf==1.27.2 pytesseract pillow

# System tesseract (macOS)
brew install tesseract tesseract-lang
```

### How it works in `rag.py`

```python
import io
import fitz  # pymupdf

doc = fitz.open(stream=io.BytesIO(content), filetype="pdf")
pages_text = []

for page in doc:
    page_parts = []

    # Table extraction (PyMuPDF >= 1.23)
    try:
        for tab in page.find_tables():
            for row in tab.extract():
                row_text = " | ".join(str(cell).strip() for cell in row if cell)
                if row_text:
                    page_parts.append(row_text)
    except AttributeError:
        pass  # older PyMuPDF — skip

    plain = page.get_text() or ""

    # OCR fallback for image-based pages
    images_on_page = page.get_images()
    needs_ocr = not plain.strip() or (images_on_page and len(plain.strip()) < 50)

    if needs_ocr:
        try:
            import pytesseract
            from PIL import Image
            mat = fitz.Matrix(2, 2)          # 2× zoom → better DPI → better accuracy
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(img, lang="chi_sim+eng")
            if len(ocr_text.strip()) > len(plain.strip()):
                plain = ocr_text             # use OCR only if it extracted more text
        except Exception:
            pass  # tesseract not installed — keep plain as-is

    if plain.strip():
        page_parts.append(plain.strip())

    pages_text.append("\n".join(page_parts))

doc.close()
text = "\n\n".join(pages_text)
chunks = chunk_text(text)
```

### Full ingestion pipeline (current)

```
PDF uploaded via POST /v1/rag/ingest
    │
    ▼
content = await file.read()      # raw bytes
    │
    ▼
fitz.open(stream=BytesIO(content))
    → for each page:
        page.find_tables()       → table rows as "col1 | col2 | col3"
        page.get_text()          → full page text (CID fonts decoded)
        if image-based → pytesseract OCR (chi_sim+eng, 2× zoom)
    → all page text joined into one string
    │
    ▼
chunk_text(text)
    → RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
    → returns list[str]
    │
    ▼
For each chunk:
    EmbeddingClient.embed(chunk) → 1024-float vector (BGE-M3)
    MilvusClient.insert({content, source, vector})
    │
    ▼
_jobs[job_id]["status"] = "done"
```

### Files changed (from pdfplumber → PyMuPDF + OCR)

| File | Change |
|------|--------|
| `requirements.txt` | Removed `pdfplumber`. Added `pymupdf==1.27.2`, `pytesseract`, `pillow`. |
| `src/routers/rag.py` | Replaced `pdfplumber` block with `fitz` + OCR fallback. |

---

## 13. Tuning: RAG_MIN_SCORE for OCR-noisy text

### The problem

OCR output contains noise: garbled characters (`neu:`, `a5`, `SERIE ASR`),
misaligned table columns, and partial words. This lowers cosine similarity
scores when the user asks a clean natural-language question.

With the default `rag_min_score = 0.72`, OCR chunks were consistently filtered
out — the agent returned "no documents found" despite chunks being present.

### Fix

Lowered the default in `src/config/settings.py`:

```python
# Before
rag_min_score: float = 0.72

# After
rag_min_score: float = 0.50
```

`0.50` still filters out completely unrelated content while allowing imperfect
OCR matches through. The value lives in `settings.py` — not `.env` — so it
is version-controlled as a project default (override via `RAG_MIN_SCORE=` in
`.env` if needed for specific deployments).

### Score range guide

| Score range | Meaning |
|-------------|---------|
| `0.85–1.0` | Near-identical text — exact keyword matches |
| `0.70–0.85` | Strong semantic match — clean, well-formatted text |
| `0.50–0.70` | Moderate match — acceptable for OCR-noisy or table text |
| `< 0.50` | Weak match — likely unrelated, filter out |

---

## 14. New Endpoint: GET /v1/rag/content

### Purpose

Returns the full extracted text for a source file as one joined string.
Use this to verify that OCR or text extraction produced correct, complete output
before testing retrieval — without needing to check individual chunks.

### Endpoint

```
GET /v1/rag/content?source=filename.pdf
```

### Example response

```json
{
  "source": "IoT管理后台需求.pdf",
  "total_chunks": 7,
  "total_chars": 2884,
  "content": "1oT管理后台需:\n\n功能类     子功能         功能描述..."
}
```

### How it works

```python
results = client.query(
    collection_name=settings.milvus_collection_knowledge,
    filter=f'source == "{source}"',
    output_fields=["id", "content"],
    limit=16384,
)
results.sort(key=lambda row: row["id"])   # id order ≈ insertion order ≈ page order
full_text = "\n\n".join(row["content"] for row in results)
```

### Example curl

```bash
curl 'http://127.0.0.1:8000/v1/rag/content?source=IoT%E7%AE%A1%E7%90%86%E5%90%8E%E5%8F%B0%E9%9C%80%E6%B1%82.pdf'
```

---

## 15. Debug Endpoints: GET and DELETE /v1/rag/chunks

After ingesting a PDF you need a way to **verify** that extraction worked and
**clean up** bad chunks without restarting the server or dropping the database.
Two debug endpoints were added to `rag.py` for this.

### Milvus query() vs search()

| Method | Purpose | Requires vector? |
|--------|---------|-----------------|
| `client.search()` | ANN vector search — find nearest neighbours | Yes (query embedding) |
| `client.query()` | Scalar filter fetch — like SQL `WHERE` | No |

The debug endpoints use `client.query()` and `client.delete()` with a scalar
filter on the `source` field — no embedding is needed.

---

### GET /v1/rag/chunks — inspect stored chunks

Use this after ingestion to verify the PDF was parsed and chunked correctly.

```
GET /v1/rag/chunks?source=filename.pdf
GET /v1/rag/chunks?source=filename.pdf&limit=5
```

**Query parameters:**

| Parameter | Required | Default | Description |
|-----------|---------|---------|-------------|
| `source` | Yes | — | Exact filename used when uploading (e.g. `IoT管理后台需求.pdf`) |
| `limit` | No | 20 | Max number of chunks to return |

**Example response:**

```json
{
  "source": "IoT管理后台需求.pdf",
  "total_chunks": 3,
  "chunks": [
    {
      "index": 1,
      "id": 449784762234646959,
      "content": "功能类 | 子功能 | 功能描述\n核心看板 | 设备概览 | 展示累计激活设备数",
      "source": "IoT管理后台需求.pdf",
      "char_count": 48
    }
  ]
}
```

**What to look for:**

- `content` should contain readable text — not garbled characters or empty strings
- Table content should appear as `col1 | col2 | col3` rows
- `char_count` should be substantial (typical chunks are 200–1000 chars)

**Example curl:**

```bash
curl 'http://127.0.0.1:8000/v1/rag/chunks?source=IoT%E7%AE%A1%E7%90%86%E5%90%8E%E5%8F%B0%E9%9C%80%E6%B1%82.pdf&limit=5'
```

---

### DELETE /v1/rag/chunks — delete bad chunks

Use this when chunks are garbled (uploaded with old/broken code). Deletes all
chunks for the given source file, then you can re-ingest with the corrected code.

```
DELETE /v1/rag/chunks?source=filename.pdf
```

**Query parameters:**

| Parameter | Required | Description |
|-----------|---------|-------------|
| `source` | Yes | Exact filename to delete all chunks for |

**Example response:**

```json
{
  "source": "IoT管理后台需求.pdf",
  "deleted_chunks": 20,
  "message": "Deleted 20 chunks. You can now re-ingest the file."
}
```

**How it works internally:**

```python
result = client.delete(
    collection_name=settings.milvus_collection_knowledge,
    filter=f'source == "{source}"',
    # Milvus filter syntax — equivalent to SQL:
    # DELETE FROM knowledge_base WHERE source = 'filename.pdf'
)
deleted = result.get("delete_count", 0)
```

**Example curl:**

```bash
curl -X DELETE 'http://127.0.0.1:8000/v1/rag/chunks?source=IoT%E7%AE%A1%E7%90%86%E5%90%8E%E5%8F%B0%E9%9C%80%E6%B1%82.pdf'
```

---

## 16. Re-ingesting Documents After a Parser Change

Every time the parser changes, **previously ingested documents must be
re-uploaded**. The old chunks in Milvus were created with the broken parser
and will never match queries above `MIN_SCORE`.

### Recommended workflow (with debug endpoints)

1. **Delete old chunks** using the DELETE endpoint (no server restart needed):
   ```bash
   curl -X DELETE 'http://127.0.0.1:8000/v1/rag/chunks?source=IoT管理后台需求.pdf'
   # Response: {"deleted_chunks": 20, "message": "Deleted 20 chunks. You can now re-ingest the file."}
   ```

2. **Re-ingest** the file with the corrected code:
   ```bash
   curl -X POST 'http://127.0.0.1:8000/v1/rag/ingest' \
     -H 'Content-Type: multipart/form-data' \
     -F 'file=@IoT管理后台需求.pdf;type=application/pdf'
   # Response: {"job_id": "550e8400-...", "status": "queued"}
   ```

3. **Poll until done:**
   ```bash
   curl 'http://127.0.0.1:8000/v1/rag/ingest/{job_id}'
   # Wait for: {"status": "done"}
   ```

4. **Verify chunks** look correct:
   ```bash
   curl 'http://127.0.0.1:8000/v1/rag/chunks?source=IoT管理后台需求.pdf&limit=3'
   # Check that content is readable text, not garbled characters
   ```

5. **Test retrieval:**
   ```bash
   curl -X POST 'http://127.0.0.1:8000/v1/agent/chat' \
     -H 'Content-Type: application/json' \
     -d '{"user_id": "123", "message": "IoT需求有哪些", "history": [], "stream": false}'
   ```

### Why old chunks are not automatically replaced

Milvus stores vectors by auto-generated ID. When you ingest the same file
again, it **adds new chunks** alongside the old ones — it does not replace.
Use `DELETE /v1/rag/chunks?source=filename` before re-ingesting to remove
the old data first.

In production, you would automate this in the ingest endpoint itself:

```python
# Production pattern: delete old chunks for this source before inserting new ones
client.delete(
    collection_name=settings.milvus_collection_knowledge,
    filter=f'source == "{filename}"',
)
```
