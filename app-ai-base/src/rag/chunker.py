from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """
    Split a plain text string into overlapping chunks.

    Uses RecursiveCharacterTextSplitter which splits on paragraph breaks →
    sentences → words, keeping semantically related text together per chunk.
    chunk_overlap prevents cutting sentences at chunk boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def chunk_documents(docs: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """
    Split LangChain Document objects into overlapping text chunks.

    Used for documents returned by LangChain loaders (e.g. UnstructuredPDFLoader).
    split_documents() re-chunks large elements while preserving short ones as-is.
    Returns plain strings (page_content only) ready for Milvus insertion.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = splitter.split_documents(docs)
    return [doc.page_content for doc in split_docs if doc.page_content.strip()]
