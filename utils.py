import os
import pathlib
import asyncio

from typing import List
import functools

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

PERSIST_PATH = pathlib.Path("vector_db")
INDEX_NAME = "rag_faiss"
EMBED_MODEL = "text-embedding-3-small"    

def load_file(path: str):
    """Read a file from disk and convert it to a list of LangChain `Document`s.

    The helper chooses the appropriate **LangChain loader** based on file
    extension.

    Args:
        path: Path to the file (PDF, TXT, MD, …).

    Returns:
        A list with one or more `langchain.schema.Document` objects.  Each
        document has its text in the `page_content` attribute and any useful
        metadata (e.g., page number) in `metadata`.
    """
    path_obj = pathlib.Path(path)
    suffix = path_obj.suffix.lower()

    if suffix == ".pdf":
        return PyPDFLoader(str(path_obj)).load()
    elif suffix in {".txt", ".md", ".py"}:
        return TextLoader(str(path_obj), encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported extension: {suffix}")
    
def chunk_docs(
    docs: List, *, chunk_size: int = 1_000, chunk_overlap: int = 200
):
    """Split long documents into smaller, overlapping chunks.

    Args:
        docs: List of `Document`s to split.
        chunk_size: Target number of characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        A new list of `Document`s, each representing one chunk.  Metadata from
        the original documents is preserved and augmented with a `chunk_id`.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators = ["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


async def ingest_file(path: pathlib.Path) -> None:
    docs = chunk_docs(load_file(path))
    embedder = OpenAIEmbeddings(model=EMBED_MODEL)

    target = PERSIST_PATH / INDEX_NAME
    if target.exists():
        vectordb = FAISS.load_local(
            str(target),
            embedder,
            allow_dangerous_deserialization=True
        )
        vectordb.add_documents(docs)        # append
    else:
        # ✨ first time: build directly from docs (docstore is auto-created)
        vectordb = FAISS.from_documents(docs, embedder)

    vectordb.save_local(str(target))
