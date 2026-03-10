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

            text = content.decode("utf-8", errors="ignore")
            # PYTHON CONCEPT — bytes.decode():
            #   `content.decode("utf-8")` converts raw bytes → Python string.
            #   errors="ignore" means: skip any characters that can't be decoded as UTF-8.
            #   Without errors="ignore", binary files (PDFs with embedded images)
            #   would raise a UnicodeDecodeError.
            #   In TypeScript: Buffer.from(content).toString("utf-8")

            # Step 4: Import processing dependencies inside the function.
            # PYTHON CONCEPT — deferred import (see web_search.py for explanation):
            #   These are imported here (not at the top) so a missing dependency
            #   only fails during actual ingestion, not at app startup.
            from src.rag.chunker import chunk_text
            from src.rag.embeddings import EmbeddingClient
            from src.config.settings import settings

            # Step 5: Split the text into chunks (paragraphs / fixed-size pieces).
            chunks = chunk_text(text)
            # chunk_text returns a list[str] — each string is one chunk.
            # Chunking is needed because embedding models have token limits (typically 8192).

            # Step 6: Create an embedding client for vectorizing chunks.
            embedder = EmbeddingClient()

            # Step 7: Connect to Milvus.
            from pymilvus import MilvusClient
            client = MilvusClient(
                uri=settings.milvus_uri,
                token=settings.milvus_token or None,
            )

            # Create the collection if it doesn't exist yet (Milvus Lite on first ingest).
            if not client.has_collection(settings.milvus_collection_knowledge):
                from src.rag.embeddings import EmbeddingClient as _EC
                from pymilvus import DataType

                # Use explicit schema so auto_id=True is guaranteed.
                # The simple create_collection(dimension=...) shorthand ignores auto_id
                # in pymilvus 2.6.x, causing: "Insert missed an field `id`".
                schema = MilvusClient.create_schema(
                    auto_id=True,
                    enable_dynamic_field=True,
                )
                schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
                schema.add_field("vector", DataType.FLOAT_VECTOR, dim=_EC.DIMENSIONS)

                index_params = client.prepare_index_params()
                index_params.add_index(field_name="vector", metric_type="COSINE")

                client.create_collection(
                    collection_name=settings.milvus_collection_knowledge,
                    schema=schema,
                    index_params=index_params,
                )

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
