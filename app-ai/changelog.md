# Fix: pdfplumber → PyMuPDF + OCR; RAG_MIN_SCORE lowered to 0.50; new /rag/content endpoint

**Files:** `src/routers/rag.py`, `src/config/settings.py`, `requirements.txt`

## Problem chain

### Layer 1 — Chinese CID fonts
`pdfplumber` (built on `pdfminer.six`) cannot decode CID-encoded fonts common in PDFs made by WPS or Chinese enterprise tools. Unmapped characters are substituted with `\u0001` control chars — almost no readable text is extracted.

### Layer 2 — Image-based PDF
After switching to PyMuPDF, `page.get_text()` still returned only `"IoT管理后台需求"` (10 chars). The PDF is entirely screenshot-style: content is a raster image with no text layer. Any text-extraction library returns empty or near-empty output.

### Layer 3 — Low similarity scores
OCR output contains noise (`neu:`, `a5`, `SERIE ASR`, misaligned columns). Cosine similarity against clean natural-language queries was consistently below `rag_min_score = 0.72` — agent returned "no documents found" despite chunks existing.

## Fixes applied

### 1. pdfplumber → PyMuPDF + tesseract OCR

```python
# requirements.txt
# Removed: pdfplumber
# Added:
pymupdf==1.27.2
pytesseract
pillow
# System: brew install tesseract tesseract-lang
```

OCR trigger condition (two cases):
```python
needs_ocr = (
    not plain.strip()                                    # empty page
    or (images_on_page and len(plain.strip()) < 50)     # image page with short title
)
# 2× zoom matrix for better accuracy: fitz.Matrix(2, 2)
# Language pack: lang="chi_sim+eng"
```

### 2. RAG_MIN_SCORE hardcoded in settings.py

```python
# src/config/settings.py
rag_min_score: float = 0.50   # was 0.72
```

Value lives in code (not `.env`) so it is version-controlled as a project default.

### 3. New GET /v1/rag/content endpoint

Returns all chunks for a source joined into one string — useful for verifying full OCR extraction without inspecting individual chunks.

```bash
curl 'http://127.0.0.1:8000/v1/rag/content?source=IoT管理后台需求.pdf'
# → { "source": "...", "total_chunks": 7, "total_chars": 2884, "content": "..." }
```

> **Note:** Re-ingest all PDFs after this change — old chunks were extracted with pdfplumber and contain `\u0001` characters.

---

# Fix: PDF parser switched from Unstructured to pdfplumber (dependency conflict)

**Files:** `src/routers/rag.py`, `requirements.txt`

**Problem:** `unstructured` eagerly imports `unstructured_inference` at module load time regardless of `strategy="fast"`. `unstructured-inference` forces `transformers` → v5.x, which removes `is_torch_fx_available` — breaking `FlagEmbedding==1.3.5` which requires `transformers==4.44.2`.

```
unstructured → unstructured_inference → transformers v5.x → breaks FlagEmbedding
```

This is a fundamental, unresolvable version conflict in this stack.

**Fix:** Removed `unstructured`, `langchain-community`, `pi-heif`. Switched to `pdfplumber` which:

- Has zero conflicting dependencies
- Extracts tables as structured rows (`col1 | col2 | col3`)
- Works entirely in-memory with `io.BytesIO` — no temp file needed

**New PDF extraction in `rag.py`:**

```python
import io, pdfplumber

pages_text = []
with pdfplumber.open(io.BytesIO(content)) as pdf:
    for page in pdf.pages:
        page_parts = []
        for table in page.extract_tables():
            for row in table:
                row_text = " | ".join(cell.strip() for cell in row if cell)
                if row_text:
                    page_parts.append(row_text)
        plain = page.extract_text() or ""
        if plain.strip():
            page_parts.append(plain.strip())
        pages_text.append("\n".join(page_parts))

text = "\n\n".join(pages_text)
chunks = chunk_text(text)
```

**Dependencies:**

```bash
# Removed
unstructured, langchain-community, pdfminer.six, pillow, pi-heif, unstructured-inference

# Added
pip install pdfplumber
```

> **Note:** Re-ingest all PDFs after this change.

---

# Upgrade: PDF ingestion switched to LangChain + Unstructured

**Files:** `src/routers/rag.py`, `src/rag/chunker.py`, `requirements.txt`

**Problem:** `pypdf` extracted poor-quality text from table-heavy PDFs (like the IoT requirements doc). It reads table cells out of order and produces incoherent chunks, resulting in low embedding similarity scores that get filtered out by `MIN_SCORE = 0.72` — so RAG retrieval still returned nothing.

**Fix:** Replaced `pypdf` with `UnstructuredPDFLoader` (LangChain + Unstructured), which understands document structure — it distinguishes titles, table rows, paragraphs, and list items before splitting.

```
Before: PDF bytes → pypdf → flat unstructured text → poor chunks → low scores → no results
After:  PDF bytes → UnstructuredPDFLoader (elements mode) → typed elements → smart chunks → good scores → retrieval works
```

**New flow in `rag.py`:**

```python
loader = UnstructuredPDFLoader(tmp_path, mode="elements", strategy="fast")
docs = loader.load()       # list[Document] with typed structural elements
chunks = chunk_documents(docs)  # re-splits large elements, filters empty ones
```

**New function in `chunker.py`:** `chunk_documents(docs: list[Document]) -> list[str]`
uses `split_documents()` instead of `split_text()` — same `RecursiveCharacterTextSplitter` but operates on LangChain Document objects.

**Dependencies changed** in `requirements.txt`:

```bash
# Removed
pypdf==5.4.0

# Added
pip install langchain-community unstructured pdfminer.six pillow
```

> **Note:** Re-ingest any PDFs uploaded before this change — they need to be re-chunked with the new parser.

---

# Fix: PDF ingestion produces empty RAG chunks

**File:** `src/routers/rag.py`

**Problem:** The ingest endpoint read uploaded PDF files as raw bytes and decoded them with `content.decode("utf-8", errors="ignore")`. PDFs are a binary format — this silently discards all non-UTF-8 bytes, leaving almost no readable text. Milvus stored empty/garbled chunks, so RAG retrieval returned nothing.

**Fix:** Detect PDF files by MIME type (`application/pdf`) or `.pdf` extension and extract text with `pypdf` before chunking.

```python
# Before (broken for PDFs)
text = content.decode("utf-8", errors="ignore")

# After
if content_type == "application/pdf" or filename_lower.endswith(".pdf"):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(content))
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
else:
    text = content.decode("utf-8", errors="ignore")
```

**Dependency added:** `pypdf==5.4.0` in `requirements.txt`.

```bash
pip install pypdf==5.4.0
```

> **Note:** Re-ingest any PDFs that were uploaded before this fix — previously stored chunks contain no usable text.

---

# Fix: milvus-lite + setuptools ≥ 81 incompatibility

`setuptools ≥ 81` removed `pkg_resources`, which `milvus-lite` still imports. Downgrade to fix.

```bash
pip install "setuptools<81"
pip install -U "pymilvus[milvus_lite]"
python -c "from pymilvus import MilvusClient; c = MilvusClient('./test.db'); print('OK')"
```

> **Note:** Use Python 3.11 + setuptools < 81 for stable AI agent environments.
