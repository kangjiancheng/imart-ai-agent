import uuid

from fastapi import APIRouter, BackgroundTasks, UploadFile, File

from src.rag.milvus_utils import ensure_collection

router = APIRouter(prefix="/v1")

# In-memory job status store.
# Replace with Redis in production so status persists across server restarts.
_jobs: dict = {}


@router.post("/rag/ingest")
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Accept a document file and start background ingestion into the knowledge base.

    Returns a job_id immediately (< 100 ms). Processing runs in the background:
      Read file → extract text → chunk → embed (BGE-M3) → store in Milvus.
    Poll GET /v1/rag/ingest/{job_id} to check when processing is complete.
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "file": file.filename}

    async def _process(job_id: str, file: UploadFile):
        try:
            _jobs[job_id]["status"] = "processing"
            content = await file.read()

            content_type   = file.content_type or ""
            filename_lower = (file.filename or "").lower()

            from src.rag.chunker import chunk_text

            if content_type == "application/pdf" or filename_lower.endswith(".pdf"):
                import io
                import fitz  # pymupdf

                pages_text = []
                doc = fitz.open(stream=io.BytesIO(content), filetype="pdf")

                for page in doc:
                    page_parts = []

                    try:
                        for tab in page.find_tables():
                            for row in tab.extract():
                                row_text = " | ".join(
                                    str(cell).strip() for cell in row if cell
                                )
                                if row_text:
                                    page_parts.append(row_text)
                    except AttributeError:
                        pass

                    plain          = page.get_text() or ""
                    images_on_page = page.get_images()
                    needs_ocr      = (not plain.strip()) or (images_on_page and len(plain.strip()) < 50)

                    if needs_ocr:
                        # Claude Vision OCR fallback for scanned / image-heavy pages.
                        # Renders page as PNG → sends to Claude → extracts text.
                        # More accurate than tesseract for Chinese text and table layouts.
                        try:
                            import base64
                            from langchain_core.messages import HumanMessage as _HumanMessage
                            from src.llm.claude_client import llm

                            mat       = fitz.Matrix(2, 2)
                            pix       = page.get_pixmap(matrix=mat)
                            png_bytes = pix.tobytes("png")
                            image_b64 = base64.standard_b64encode(png_bytes).decode("utf-8")

                            vision_msg = _HumanMessage(content=[
                                {
                                    "type": "image",
                                    "source": {
                                        "type":       "base64",
                                        "media_type": "image/png",
                                        "data":       image_b64,
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
                            if len(ocr_text.strip()) > len(plain.strip()):
                                plain = ocr_text
                        except Exception:
                            pass

                    if plain.strip():
                        page_parts.append(plain.strip())
                    pages_text.append("\n".join(page_parts))

                doc.close()
                text   = "\n\n".join(pages_text)
                chunks = chunk_text(text)

            else:
                chunks = chunk_text(content.decode("utf-8", errors="ignore"))

            from src.rag.embeddings import EmbeddingClient
            from src.config.settings import settings
            from pymilvus import MilvusClient

            embedder = EmbeddingClient()
            client   = MilvusClient(
                uri=settings.milvus_uri,
                token=settings.milvus_token or None,
            )
            ensure_collection(client, settings.milvus_collection_knowledge)

            for chunk in chunks:
                vector = await embedder.embed(chunk)
                client.insert(
                    collection_name=settings.milvus_collection_knowledge,
                    data=[{
                        "content": chunk,
                        "source":  file.filename or "unknown",
                        "vector":  vector,
                    }],
                )

            _jobs[job_id]["status"] = "done"

        except Exception as e:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"]  = str(e)

    background_tasks.add_task(_process, job_id, file)
    return {"job_id": job_id, "status": "queued"}


@router.get("/rag/ingest/{job_id}")
async def get_ingest_status(job_id: str):
    """Poll the status of a background ingestion job."""
    job = _jobs.get(job_id)
    if job is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **job}


@router.get("/rag/chunks")
async def list_chunks(source: str, limit: int = 20):
    """
    Return stored chunks for a given source file from Milvus.
    Use after ingestion to verify that the PDF was parsed and split correctly.

    Example:
      GET /v1/rag/chunks?source=product-manual.pdf
      GET /v1/rag/chunks?source=product-manual.pdf&limit=5
    """
    from fastapi import HTTPException
    from pymilvus import MilvusClient
    from src.config.settings import settings

    try:
        client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token or None)
        ensure_collection(client, settings.milvus_collection_knowledge)

        results = client.query(
            collection_name=settings.milvus_collection_knowledge,
            filter=f'source == "{source}"',
            output_fields=["id", "content", "source"],
            limit=limit,
        )

        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"No chunks found for source='{source}'. Check the exact filename used during upload.",
            )

        chunks = [
            {
                "index":      i + 1,
                "id":         str(row["id"]),
                "content":    row["content"],
                "source":     row["source"],
                "char_count": len(row["content"]),
            }
            for i, row in enumerate(results)
        ]
        return {"source": source, "total_chunks": len(chunks), "chunks": chunks}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Milvus query failed: {e}")


@router.get("/rag/content")
async def get_full_content(source: str):
    """
    Return the full extracted text for a source file as one joined string.
    Fetches all chunks from Milvus, sorts by id (insertion order proxy),
    and joins with double newlines for easy review of extraction quality.

    Example:
      GET /v1/rag/content?source=product-manual.pdf
    """
    from fastapi import HTTPException
    from pymilvus import MilvusClient
    from src.config.settings import settings

    try:
        client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token or None)
        ensure_collection(client, settings.milvus_collection_knowledge)

        results = client.query(
            collection_name=settings.milvus_collection_knowledge,
            filter=f'source == "{source}"',
            output_fields=["id", "content"],
            limit=16384,
        )

        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"No chunks found for source='{source}'. Check the exact filename used during upload.",
            )

        results.sort(key=lambda row: row["id"])
        full_text = "\n\n".join(row["content"] for row in results)

        return {
            "source":       source,
            "total_chunks": len(results),
            "total_chars":  len(full_text),
            "content":      full_text,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Milvus query failed: {e}")


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
        client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token or None)
        ensure_collection(client, settings.milvus_collection_knowledge)

        result  = client.delete(
            collection_name=settings.milvus_collection_knowledge,
            filter=f'source == "{source}"',
        )
        deleted = result.get("delete_count", 0)

        if deleted == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No chunks found for source='{source}'. Nothing was deleted.",
            )

        return {
            "source":          source,
            "deleted_chunks":  deleted,
            "message":         f"Deleted {deleted} chunks. You can now re-ingest the file.",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Milvus delete failed: {e}")
