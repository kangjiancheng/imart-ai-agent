# src/rag/chunker.py
#
# Two chunking functions:
#   chunk_text(str)          → for plain text / non-PDF files
#   chunk_documents(docs)    → for LangChain Document objects (PDF via Unstructured)
#
# Both return list[str] so the Milvus insert loop in rag.py stays identical.

from langchain_text_splitters import RecursiveCharacterTextSplitter
# RecursiveCharacterTextSplitter splits on paragraph breaks → sentences → words,
# trying to keep semantically related text together in each chunk.


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Split a plain text string into overlapping chunks.

    Used for non-PDF uploads (Markdown, plain .txt files, etc.)

    chunk_size    = max characters per chunk (not tokens — characters)
    chunk_overlap = how many characters the next chunk re-uses from the previous one.
                    Overlap prevents cutting a sentence mid-thought at a boundary.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)
    # split_text(str) → list[str]
    # Each string in the list is one chunk, ready to be embedded.


def chunk_documents(docs: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Split LangChain Document objects into overlapping text chunks.

    Used for PDF uploads processed by UnstructuredPDFLoader.
    UnstructuredPDFLoader returns list[Document] — each Document represents
    one structural element (Title, Table row, NarrativeText paragraph).

    PYTHON CONCEPT — list[Document] vs list[str]:
      LangChain's Document dataclass has two fields:
        .page_content  → the text of this element (str)
        .metadata      → dict with source, page_number, category, etc.
      split_documents() respects these objects and carries metadata forward.
      We then extract only .page_content for Milvus (which stores plain strings).

    WHY split_documents instead of split_text?
      Unstructured elements can still be large (a big table, a long paragraph).
      split_documents() re-chunks any element that exceeds chunk_size,
      while keeping short elements (like a title) as-is — no unnecessary splitting.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = splitter.split_documents(docs)
    # split_documents(list[Document]) → list[Document]
    # Each returned Document still has .page_content and .metadata.

    return [doc.page_content for doc in split_docs if doc.page_content.strip()]
    # [doc.page_content for doc in split_docs]
    #   = list comprehension: extract just the text string from each Document.
    #   Equivalent to: results = []; for doc in split_docs: results.append(doc.page_content)
    #
    # if doc.page_content.strip()
    #   = skip empty or whitespace-only chunks (Unstructured can emit blank elements
    #     from page headers, footers, or image-only pages).
    #   .strip() removes leading/trailing whitespace; empty string is falsy in Python.
