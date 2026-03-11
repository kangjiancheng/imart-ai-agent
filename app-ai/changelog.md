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
