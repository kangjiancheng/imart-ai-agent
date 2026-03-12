from fastapi import APIRouter, BackgroundTasks, UploadFile, File
import uuid

from src.rag.milvus_utils import ensure_collection

router = APIRouter(prefix="/v1")

# In-memory job status store (replace with Redis in production)
_jobs: dict = {}


@router.post("/rag/ingest")
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Accept a document file and process it into the RAG knowledge base in the background."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "file": file.filename}

    async def _process(job_id: str, file: UploadFile):
        try:
            _jobs[job_id]["status"] = "processing"
            content = await file.read()

            content_type = file.content_type or ""
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
                        pass  # PyMuPDF < 1.23: find_tables() not available

                    plain = page.get_text() or ""

                    images_on_page = page.get_images()
                    needs_ocr = (
                        not plain.strip()
                        or (images_on_page and len(plain.strip()) < 50)
                    )

                    if needs_ocr:
                        try:
                            import pytesseract
                            from PIL import Image

                            mat = fitz.Matrix(2, 2)
                            pix = page.get_pixmap(matrix=mat)
                            img = Image.open(io.BytesIO(pix.tobytes("png")))
                            ocr_text = pytesseract.image_to_string(img, lang="chi_sim+eng")
                            if len(ocr_text.strip()) > len(plain.strip()):
                                plain = ocr_text
                        except Exception:
                            pass  # tesseract not installed — keep plain as-is

                    if plain.strip():
                        page_parts.append(plain.strip())

                    pages_text.append("\n".join(page_parts))

                doc.close()
                text = "\n\n".join(pages_text)
                chunks = chunk_text(text)

            else:
                chunks = chunk_text(content.decode("utf-8", errors="ignore"))

            from src.rag.embeddings import EmbeddingClient
            from src.config.settings import settings
            from pymilvus import MilvusClient

            embedder = EmbeddingClient()
            client = MilvusClient(
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
                        "source": file.filename or "unknown",
                        "vector": vector,
                    }],
                )

            _jobs[job_id]["status"] = "done"

        except Exception as e:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(e)

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
    """Return stored chunks for a given source file from Milvus."""
    from fastapi import HTTPException
    from pymilvus import MilvusClient
    from src.config.settings import settings

    try:
        client = MilvusClient(
            uri=settings.milvus_uri,
            token=settings.milvus_token or None,
        )
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
                detail=f"No chunks found for source='{source}'. "
                       f"Check the exact filename used during upload.",
            )

        chunks = [
            {
                "index": i + 1,
                "id": row["id"],
                "content": row["content"],
                "source": row["source"],
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
    """Return the full extracted text for a source file as one joined string."""
    from fastapi import HTTPException
    from pymilvus import MilvusClient
    from src.config.settings import settings

    try:
        client = MilvusClient(
            uri=settings.milvus_uri,
            token=settings.milvus_token or None,
        )
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
                detail=f"No chunks found for source='{source}'. "
                       f"Check the exact filename used during upload.",
            )

        results.sort(key=lambda row: row["id"])
        full_text = "\n\n".join(row["content"] for row in results)

        return {
            "source": source,
            "total_chunks": len(results),
            "total_chars": len(full_text),
            "content": full_text,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Milvus query failed: {e}")


@router.delete("/rag/chunks")
async def delete_chunks(source: str):
    """Delete all stored chunks for a given source file from Milvus."""
    from fastapi import HTTPException
    from pymilvus import MilvusClient
    from src.config.settings import settings

    try:
        client = MilvusClient(
            uri=settings.milvus_uri,
            token=settings.milvus_token or None,
        )
        ensure_collection(client, settings.milvus_collection_knowledge)

        result = client.delete(
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
            "source": source,
            "deleted_chunks": deleted,
            "message": f"Deleted {deleted} chunks. You can now re-ingest the file.",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Milvus delete failed: {e}")
