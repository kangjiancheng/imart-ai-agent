# src/routers/rag.py
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS THIS ROUTER?
#
# This file handles document ingestion into the RAG knowledge base.
# When you want Claude to answer questions about YOUR documents (product manuals,
# internal docs, policies), you first POST them here to be processed.
#
# TWO ENDPOINTS:
#   POST /v1/rag/ingest           → start processing a document (returns job_id)
#   GET  /v1/rag/ingest/{job_id}  → check if processing is done (poll for status)
#
# WHY TWO ENDPOINTS? — Background processing pattern:
#   Document ingestion is SLOW:
#     Read file → split into chunks → embed each chunk (OpenAI API call!) → store in Milvus
#   If we did this synchronously (blocking), NestJS would wait 30+ seconds for a response.
#   Instead:
#     1. Accept the file → return job_id immediately (< 100ms)
#     2. Process in background (async, non-blocking)
#     3. NestJS polls GET /v1/rag/ingest/{job_id} every few seconds until "done"
#
# PYTHON CONCEPT — BackgroundTasks (FastAPI):
#   FastAPI's BackgroundTasks lets you run functions AFTER the HTTP response is sent.
#   The client receives {"job_id": "...", "status": "queued"} immediately,
#   then the processing function runs in the background.
#   In Node.js: similar to setImmediate() or a job queue like Bull.
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import APIRouter, BackgroundTasks, UploadFile, File
# APIRouter     = groups related routes under a prefix (see agent.py for explanation)
# BackgroundTasks = FastAPI's built-in mechanism to run tasks after HTTP response
# UploadFile    = FastAPI's type for a file uploaded via multipart/form-data
# File(...)     = dependency injection that marks a form field as required

import uuid
# uuid = Python's built-in module for generating universally unique identifiers.
# uuid.uuid4() generates a random UUID like "550e8400-e29b-41d4-a716-446655440000"
# Used here to create a unique job_id for each ingestion request.
# In Node.js: crypto.randomUUID() or the uuid npm package.

from src.rag.milvus_utils import ensure_collection
# ensure_collection(client, name) creates the Milvus collection with the
# standard schema if it doesn't exist yet. Shared across retriever, ingest,
# and vector memory to avoid duplicating the schema/index setup code.

router = APIRouter(prefix="/v1")
# All routes in this file are accessible under /v1/...

# ── In-memory job status store ────────────────────────────────────────────────
# PYTHON CONCEPT — dict type hint:
#   `_jobs: dict = {}` declares _jobs as a dict (dictionary), initialized empty.
#   The `dict` type hint is loose — keys and values can be anything.
#   In production, replace this with Redis so job status survives server restarts.
#
# _jobs structure example:
#   {
#     "550e8400-e29b-41d4-a716-446655440000": {
#       "status": "processing",
#       "file": "manual.pdf"
#     }
#   }
_jobs: dict = {}
# In-memory job status store (replace with Redis in production)
# WARNING: Restarting the server clears all job statuses.
# In production, use Redis or a database so job status persists across restarts.


# ── POST /v1/rag/ingest ────────────────────────────────────────────────────────
# Start ingesting a document into the knowledge base.

@router.post("/rag/ingest")
async def ingest_document(
    background_tasks: BackgroundTasks,
    # FastAPI automatically injects BackgroundTasks — you don't pass this manually.
    # It's a FastAPI "dependency" — the framework creates it and passes it to your function.
    # Call background_tasks.add_task(fn, arg1, arg2) to schedule fn to run after response.

    file: UploadFile = File(...),
    # UploadFile = the uploaded file object.
    # File(...) = marks this as a required form field (... = required, no default).
    # NestJS sends a multipart/form-data request with the document file.
    # UploadFile gives you: .filename, .content_type, .read() (async)
):
    """
    Accepts a document file, chunks it, embeds it, and stores it in Milvus.
    Returns a job_id immediately; processing happens in the background.
    """
    # Step 1: Create a unique job ID for this ingestion request.
    job_id = str(uuid.uuid4())
    # uuid.uuid4() returns a UUID object; str() converts it to a string.
    # Example: "550e8400-e29b-41d4-a716-446655440000"

    # Step 2: Register the job in our in-memory store with "queued" status.
    _jobs[job_id] = {"status": "queued", "file": file.filename}
    # _jobs[job_id] = {...} creates a new key-value entry in the dict.
    # The client can poll GET /v1/rag/ingest/{job_id} to check this status.

    # ── Inner processing function (closure) ──────────────────────────────────
    # PYTHON CONCEPT — nested async function (closure):
    #   `async def _process(...)` is defined inside `ingest_document`.
    #   It captures `_jobs` from the outer module scope.
    #   It's passed to background_tasks.add_task() to run after this function returns.
    #   The leading underscore `_process` signals "private, not for external use".

    async def _process(job_id: str, file: UploadFile):
        try:
            # Update status so the polling client sees we started.
            _jobs[job_id]["status"] = "processing"
            # _jobs[job_id] = the dict for this job.
            # ["status"] = update the "status" key in that dict.

            # Step 3: Read the uploaded file content into memory.
            content = await file.read()
            # await = pause until the file bytes are fully read (non-blocking).
            # content is a `bytes` object (raw binary data).

            # Step 3b: Extract and chunk based on file type.
            #
            # WHY split by file type?
            #   PDFs are binary — they need a parser that understands the format.
            #   pdfplumber extracts table rows as structured text (col1 | col2 | col3)
            #   so table-heavy PDFs produce coherent, embeddable chunks.
            #   Plain text files can be split directly with RecursiveCharacterTextSplitter.
            #
            # WHY pdfplumber instead of unstructured?
            #   unstructured eagerly imports unstructured_inference at module load time
            #   regardless of strategy. unstructured_inference forces transformers → v5.x
            #   which breaks FlagEmbedding (needs transformers==4.44.2).
            #   pdfplumber has no such dependency conflicts — lightweight and table-aware.

            content_type = file.content_type or ""
            filename_lower = (file.filename or "").lower()

            from src.rag.chunker import chunk_text

            if content_type == "application/pdf" or filename_lower.endswith(".pdf"):
                # ── PDF path: PyMuPDF (fitz) ──────────────────────────────────
                #
                # WHY PyMuPDF instead of pdfplumber?
                #   pdfplumber uses pdfminer.six for text decoding. pdfminer.six
                #   cannot decode Chinese CID fonts (non-standard font encoding
                #   common in PDFs made by WPS, government tools, older InDesign).
                #   It substitutes unmapped characters with \u0001 control chars,
                #   so almost no text is extracted.
                #
                #   PyMuPDF uses the MuPDF C library, which has broad CID font
                #   support and correctly decodes Chinese text in these PDFs.
                #   Zero dependency conflicts with FlagEmbedding / transformers.
                #
                # API used:
                #   fitz.open(stream=BytesIO(content), filetype="pdf")
                #     → opens in-memory (no temp file needed)
                #   page.get_text()
                #     → returns full decoded text for the page (plain string)
                #   page.find_tables() (PyMuPDF >= 1.23)
                #     → detects table regions; tab.extract() → list of rows
                #     → each row is a list of cell strings
                import io
                import fitz  # pymupdf

                pages_text = []
                doc = fitz.open(stream=io.BytesIO(content), filetype="pdf")

                for page in doc:
                    page_parts = []

                    # Extract tables as "col1 | col2 | col3" rows first.
                    # find_tables() was added in PyMuPDF 1.23 — guard with try/except
                    # in case an older version is installed.
                    try:
                        for tab in page.find_tables():
                            for row in tab.extract():
                                # row = list of cell values (may be None for empty cells)
                                row_text = " | ".join(
                                    str(cell).strip() for cell in row if cell
                                )
                                if row_text:
                                    page_parts.append(row_text)
                    except AttributeError:
                        pass  # PyMuPDF < 1.23: find_tables() not available, skip

                    # Extract full page text (includes headings, paragraphs, table text).
                    # get_text() correctly decodes CID/Unicode fonts unlike pdfminer.
                    plain = page.get_text() or ""

                    # Detect pages that need OCR.
                    #
                    # Two cases where get_text() is insufficient:
                    #   1. plain is completely empty → clearly an image-only page.
                    #   2. plain is very short AND the page has embedded images →
                    #      get_text() only captured a title/metadata label, while
                    #      the actual content is inside an image (screenshot-style PDF).
                    #      Example: page 1 returns "IoT管理后台需求" (10 chars) but the
                    #      requirements table is an image — the 10 chars are just the
                    #      document title at the top of the scanned screenshot.
                    #
                    # Threshold: < 50 chars is too short to be the full content of a page.
                    # We only apply the threshold when the page contains embedded images,
                    # so we don't accidentally OCR a legitimate short text-only page.
                    images_on_page = page.get_images()
                    needs_ocr = (
                        not plain.strip()
                        or (images_on_page and len(plain.strip()) < 50)
                    )

                    if needs_ocr:
                        # Fall back to Claude Vision: render the page as a PNG image
                        # and send it to Claude to extract the text.
                        #
                        # WHY Claude Vision instead of tesseract?
                        #   tesseract struggles with Chinese text inside table cells —
                        #   it garbles characters and misreads column alignment.
                        #   Claude Vision understands Chinese + table structure natively,
                        #   producing clean, accurate text even from complex layouts.
                        #
                        # HOW IT WORKS:
                        #   1. fitz.Matrix(2, 2) = 2× zoom → higher DPI → sharper image
                        #   2. pix.tobytes("png") = render page pixels as PNG bytes
                        #   3. base64.standard_b64encode() = encode PNG as base64 string
                        #      (Anthropic API requires base64-encoded images, not raw bytes)
                        #   4. HumanMessage with image + text content → Claude reads both
                        #   5. llm.ainvoke() = async call (await pauses until Claude responds)
                        #
                        # PYTHON CONCEPT — base64 encoding:
                        #   Binary data (PNG bytes) cannot be sent directly in JSON.
                        #   base64 converts binary → ASCII string safe for JSON transport.
                        #   base64.standard_b64encode(bytes) → bytes → .decode("utf-8") → str
                        try:
                            import base64
                            from langchain_core.messages import HumanMessage as _HumanMessage
                            from src.llm.claude_client import llm

                            mat = fitz.Matrix(2, 2)
                            pix = page.get_pixmap(matrix=mat)
                            png_bytes = pix.tobytes("png")
                            image_b64 = base64.standard_b64encode(png_bytes).decode("utf-8")

                            vision_msg = _HumanMessage(content=[
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": image_b64,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": (
                                        "Extract all text from this image exactly as it appears. "
                                        "For tables, use ' | ' to separate columns and preserve each row on its own line. "
                                        "Include all Chinese and English text. "
                                        "Return only the extracted text, no commentary."
                                    ),
                                },
                            ])
                            vision_response = await llm.ainvoke([vision_msg])
                            ocr_text = vision_response.content
                            # Use Claude's result if it's longer than what get_text() returned.
                            if len(ocr_text.strip()) > len(plain.strip()):
                                plain = ocr_text
                        except Exception:
                            pass  # Claude Vision failed — keep plain as-is (may be empty)

                    if plain.strip():
                        page_parts.append(plain.strip())

                    pages_text.append("\n".join(page_parts))

                doc.close()
                text = "\n\n".join(pages_text)
                chunks = chunk_text(text)

            else:
                # ── Non-PDF path: plain text decode + RecursiveCharacterTextSplitter
                chunks = chunk_text(content.decode("utf-8", errors="ignore"))

            # Step 4: Import remaining processing dependencies.
            from src.rag.embeddings import EmbeddingClient
            from src.config.settings import settings

            # Step 5: chunks is now list[str] for both paths — same insert loop below.

            # Step 6: Create an embedding client for vectorizing chunks.
            embedder = EmbeddingClient()

            # Step 7: Connect to Milvus.
            from pymilvus import MilvusClient
            client = MilvusClient(
                uri=settings.milvus_uri,
                token=settings.milvus_token or None,
            )

            # Create the collection if it doesn't exist yet (Milvus Lite on first ingest).
            ensure_collection(client, settings.milvus_collection_knowledge)

            # Step 8: For each chunk, embed it and store it in Milvus.
            for chunk in chunks:
                # PYTHON CONCEPT — for loop over a list:
                #   Iterates over each string in the chunks list.
                #   In TypeScript: for (const chunk of chunks)

                vector = await embedder.embed(chunk)
                # Convert this text chunk to a 1024-float embedding vector (BGE-M3).

                client.insert(
                    collection_name=settings.milvus_collection_knowledge,
                    data=[{
                        "content": chunk,
                        # The raw text of this chunk.

                        "source": file.filename or "unknown",
                        # The original file name — used for citation in RAG results.
                        # `or "unknown"` handles None (UploadFile.filename can be None).
                        # In TypeScript: file.filename ?? "unknown"

                        "vector": vector,
                        # The embedding vector — what Milvus searches against.
                    }],
                )

            # Step 9: Mark job as done.
            _jobs[job_id]["status"] = "done"

        except Exception as e:
            # If any step fails, mark the job as "error" and store the message.
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(e)
            # str(e) converts the exception to a readable string.
            # The polling client sees {"status": "error", "error": "..."}.

    # Step 10: Schedule the processing function to run AFTER this response is sent.
    background_tasks.add_task(_process, job_id, file)
    # add_task(function, *args) schedules `function(job_id, file)` to run
    # after the HTTP response is returned. The function runs concurrently —
    # the client receives the job_id immediately without waiting.

    # Step 11: Return the job ID immediately.
    return {"job_id": job_id, "status": "queued"}
    # PYTHON CONCEPT — returning a dict from FastAPI:
    #   FastAPI automatically converts dicts to JSON responses.
    #   No need to call json.dumps() or Response() — FastAPI handles it.
    #   NestJS receives: {"job_id": "550e8400-...", "status": "queued"}


# ── GET /v1/rag/ingest/{job_id} ────────────────────────────────────────────────
# Poll the status of a background ingestion job.

@router.get("/rag/ingest/{job_id}")
async def get_ingest_status(job_id: str):
    """Poll the status of a background ingestion job.

    PYTHON CONCEPT — path parameter:
      `{job_id}` in the route path and `job_id: str` in the function parameter
      are automatically linked by FastAPI. The value from the URL is injected.
      Example: GET /v1/rag/ingest/550e8400-... → job_id = "550e8400-..."
      In TypeScript/Express: router.get('/:jobId', (req) => req.params.jobId)
    """
    job = _jobs.get(job_id)
    # PYTHON CONCEPT — dict.get(key):
    #   _jobs.get(job_id) = look up job_id in _jobs dict.
    #   Returns the value (the status dict) if found, or None if not found.
    #   Safer than _jobs[job_id] which raises KeyError if the key is missing.
    #   In TypeScript: _jobs.get(jobId) or _jobs[jobId]

    if job is None:
        # If job_id is not found in _jobs, the client sent an invalid ID.
        # Import HTTPException here (not at top) to keep imports organized.
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Job not found")
        # 404 = "Not Found" — standard HTTP status for missing resources.

    return {"job_id": job_id, **job}
    # PYTHON CONCEPT — dict unpacking with **:
    #   `**job` expands the job dict's key-value pairs INTO the outer dict.
    #   Example: if job = {"status": "done", "file": "manual.pdf"}
    #   Then {"job_id": job_id, **job}
    #     = {"job_id": "550e8400-...", "status": "done", "file": "manual.pdf"}
    #   In TypeScript: { jobId, ...job }
    #   This is Python's equivalent of the JavaScript spread operator for objects.


# ── GET /v1/rag/chunks ─────────────────────────────────────────────────────────
# Inspect stored chunks for a specific source file (debug / verification endpoint).
#
# WHY THIS ENDPOINT?
#   After uploading a PDF you want to verify:
#     - Was the text extracted correctly? (no garbled characters)
#     - How was the document split? (chunk boundaries make sense)
#     - How many chunks were stored?
#   This endpoint fetches every stored chunk for a given filename directly
#   from Milvus so you can see exactly what Claude will retrieve at query time.
#
# USAGE:
#   GET /v1/rag/chunks?source=your-file.pdf
#   GET /v1/rag/chunks?source=your-file.pdf&limit=5   (first 5 chunks only)
#
# HOW MILVUS QUERY WORKS (vs search):
#   client.search()  → ANN vector search (needs a query embedding)
#   client.query()   → scalar filter fetch (no vector needed, like SQL WHERE)
#   Here we use query() with filter `source == "filename"` to list all chunks
#   that belong to a specific file — no embedding required.

@router.get("/rag/chunks")
async def list_chunks(
    source: str,
    # `source` is a QUERY PARAMETER — it comes from the URL after `?`.
    # Example: GET /v1/rag/chunks?source=manual.pdf → source = "manual.pdf"
    # FastAPI reads query parameters from the function signature automatically.
    # In Express: req.query.source

    limit: int = 20,
    # Optional query parameter with a default value of 20.
    # GET /v1/rag/chunks?source=x&limit=5 → returns first 5 chunks only.
    # Useful if the PDF is large and you just want a quick sample.
):
    """
    Return stored chunks for a given source file from Milvus.
    Use this after ingestion to verify the PDF was parsed and split correctly.

    Example:
      GET /v1/rag/chunks?source=product-manual.pdf
      GET /v1/rag/chunks?source=product-manual.pdf&limit=5
    """
    from fastapi import HTTPException
    from pymilvus import MilvusClient
    from src.config.settings import settings

    try:
        client = MilvusClient(
            uri=settings.milvus_uri,
            token=settings.milvus_token or None,
        )

        # Ensure the collection exists before querying.
        # (If no documents have been ingested yet, ensure_collection creates it
        #  and query() returns an empty list — no crash.)
        ensure_collection(client, settings.milvus_collection_knowledge)

        # MILVUS CONCEPT — client.query() with a scalar filter:
        #   Unlike client.search() (which finds vectors nearest to a query vector),
        #   client.query() fetches rows that match a boolean filter expression —
        #   similar to SQL: SELECT content, source FROM knowledge_base WHERE source = '...'
        #
        # filter syntax uses Milvus expression language:
        #   `source == "filename.pdf"` — exact string match on the "source" dynamic field.
        #
        # output_fields: which fields to include in each returned row.
        #   We skip "vector" (1024 floats) — it's not human-readable.
        #   "id" is the auto-generated Milvus primary key.
        results = client.query(
            collection_name=settings.milvus_collection_knowledge,
            filter=f'source == "{source}"',
            # PYTHON CONCEPT — f-string:
            #   f'source == "{source}"' inserts the value of `source` into the string.
            #   If source = "manual.pdf", filter = 'source == "manual.pdf"'
            #   The inner double-quotes are part of the Milvus filter syntax.

            output_fields=["id", "content", "source"],
            # Return id, content, and source for each chunk.
            # "vector" is intentionally omitted — 1024 floats are not useful to read.

            limit=limit,
            # Cap the number of rows returned (default 20, configurable via ?limit=N).
        )

        if not results:
            # No chunks found for this source — either the file was never ingested,
            # the filename doesn't match exactly, or ingestion failed.
            raise HTTPException(
                status_code=404,
                detail=f"No chunks found for source='{source}'. "
                       f"Check the exact filename used during upload.",
            )

        # Build a clean response: total count + list of chunks with index labels.
        chunks = [
            {
                "index": i + 1,
                # Human-friendly 1-based index so you can say "chunk 3 looks wrong".

                "id": str(row["id"]),
                # Milvus auto-generated primary key — useful for debugging duplicates.
                # str() is required: Milvus IDs are 64-bit integers (~4.6×10¹⁷) which
                # exceed JavaScript's safe integer limit (2^53 ≈ 9×10¹⁵). Without str(),
                # the client's JSON.parse() rounds all IDs to the same value.

                "content": row["content"],
                # The actual extracted text for this chunk — the main thing to verify.

                "source": row["source"],
                # Should match the filename you queried for.

                "char_count": len(row["content"]),
                # Character count — quick way to spot empty or suspiciously short chunks.
                # In TypeScript: row.content.length
            }
            for i, row in enumerate(results)
            # PYTHON CONCEPT — list comprehension:
            #   [expression for i, row in enumerate(results)]
            #   is a compact way to build a list by iterating — equivalent to:
            #     chunks = []
            #     for i, row in enumerate(results):
            #         chunks.append({...})
            #   enumerate(results) yields (0, row0), (1, row1), ...
            #   In TypeScript: results.map((row, i) => ({ index: i + 1, ... }))
        ]

        return {
            "source": source,
            "total_chunks": len(chunks),
            # len(chunks) = number of items in the list.
            # In TypeScript: chunks.length

            "chunks": chunks,
            # The full list of chunk dicts — one per stored text segment.
        }

    except HTTPException:
        # Re-raise our own 404 without wrapping it in a 500.
        # PYTHON CONCEPT — bare `raise`:
        #   `raise` inside an except block re-raises the current exception unchanged.
        raise

    except Exception as e:
        # Any other error (Milvus connection failed, filter syntax error, etc.)
        raise HTTPException(status_code=500, detail=f"Milvus query failed: {e}")



# ── GET /v1/rag/content ────────────────────────────────────────────────────────
# Return the full extracted text for a source file as a single joined string.
#
# WHY THIS ENDPOINT?
#   GET /v1/rag/chunks returns chunks split into segments — useful for seeing
#   how the document was split. But sometimes you want to read the complete
#   extracted text as one continuous document to verify overall extraction quality.
#
#   This endpoint fetches all stored chunks for a source, sorts them by their
#   Milvus insert order (id), and joins them with double newlines — giving you
#   the full reconstructed text.
#
# USAGE:
#   GET /v1/rag/content?source=your-file.pdf
#
# NOTE ON ORDERING:
#   Milvus does not guarantee query() result order. We sort by "id" (the
#   auto-increment primary key) as a best proxy for insertion order, which
#   corresponds to page/chunk order in the original document.

@router.get("/rag/content")
async def get_full_content(
    source: str,
    # Query parameter: the exact filename used when the file was uploaded.
    # Example: GET /v1/rag/content?source=IoT管理后台需求.pdf
):
    """
    Return the full extracted text for a source file as one joined string.
    Fetches all chunks from Milvus, sorts by id, and joins with double newlines.

    Example:
      GET /v1/rag/content?source=IoT管理后台需求.pdf
    """
    from fastapi import HTTPException
    from pymilvus import MilvusClient
    from src.config.settings import settings

    try:
        client = MilvusClient(
            uri=settings.milvus_uri,
            token=settings.milvus_token or None,
        )

        ensure_collection(client, settings.milvus_collection_knowledge)

        # Fetch ALL chunks for this source.
        # limit=16384 is Milvus Lite's practical max — well above any real document.
        # We fetch id + content so we can sort by id (insertion order proxy).
        results = client.query(
            collection_name=settings.milvus_collection_knowledge,
            filter=f'source == "{source}"',
            output_fields=["id", "content"],
            limit=16384,
        )

        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"No chunks found for source='{source}'. "
                       f"Check the exact filename used during upload.",
            )

        # Sort by Milvus primary key (id) as a proxy for insertion order.
        # Chunks were inserted page-by-page, so ascending id ≈ document order.
        results.sort(key=lambda row: row["id"])

        # Join all chunk content into one continuous text.
        # Chunks overlap by 200 chars (RecursiveCharacterTextSplitter overlap),
        # so the joined text will have some repeated sentences at chunk boundaries.
        # This is expected — it does NOT mean content is duplicated in Milvus.
        full_text = "\n\n".join(row["content"] for row in results)

        return {
            "source": source,
            "total_chunks": len(results),
            # How many chunks were found and joined.

            "total_chars": len(full_text),
            # Total character count of the joined text (includes overlap regions).

            "content": full_text,
            # The full reconstructed document text — read this to verify extraction.
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Milvus query failed: {e}")


# ── DELETE /v1/rag/chunks ──────────────────────────────────────────────────────
# Delete all stored chunks for a specific source file.
#
# WHY THIS ENDPOINT?
#   If a PDF was ingested with bad/garbled content (e.g. raw binary stored instead
#   of extracted text because the server was running old code), you can clean up
#   the bad data here, then re-ingest the file with the corrected code.
#
# USAGE:
#   DELETE /v1/rag/chunks?source=your-file.pdf
#
# MILVUS DELETE:
#   client.delete() removes rows matching a scalar filter expression —
#   similar to SQL: DELETE FROM knowledge_base WHERE source = '...'
#   It returns a dict with "delete_count" showing how many rows were removed.

@router.delete("/rag/chunks")
async def delete_chunks(source: str):
    """
    Delete all stored chunks for a given source file from Milvus.
    Use this to clear bad/garbled chunks before re-ingesting a corrected file.

    Example:
      DELETE /v1/rag/chunks?source=product-manual.pdf
    """
    from fastapi import HTTPException
    from pymilvus import MilvusClient
    from src.config.settings import settings

    try:
        client = MilvusClient(
            uri=settings.milvus_uri,
            token=settings.milvus_token or None,
        )

        ensure_collection(client, settings.milvus_collection_knowledge)

        # Delete all rows where source matches the given filename.
        # Milvus filter syntax: `source == "filename"` (same as used in query above).
        result = client.delete(
            collection_name=settings.milvus_collection_knowledge,
            filter=f'source == "{source}"',
        )

        deleted = result.get("delete_count", 0)
        # result is a dict like {"delete_count": 20}.
        # .get("delete_count", 0) safely reads the count; defaults to 0 if missing.

        if deleted == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No chunks found for source='{source}'. Nothing was deleted.",
            )

        return {
            "source": source,
            "deleted_chunks": deleted,
            "message": f"Deleted {deleted} chunks. You can now re-ingest the file.",
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Milvus delete failed: {e}")
