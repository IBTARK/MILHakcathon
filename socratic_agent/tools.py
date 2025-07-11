from __future__ import annotations
import pathlib
from typing import List, Dict

from typing import Any, Type
from pydantic import BaseModel, PrivateAttr

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import BaseTool
from langchain.schema import Document

from utils import INDEX_NAME, EMBED_MODEL, PERSIST_PATH

# How many chunks to retrieve
TOP_K = 2 



class RAGRetrieveChunks(BaseTool):
    """
    Tool: returns the *raw* chunks most similar to the query.
    """
    name: str = "rag_retrieve_chunks"

    description: str = (
        "Perform semantic search in the persisted FAISS index and return the "
        "matching chunks with their metadata. Does NOT call the LLM."
    )

    _vectordb: Any = PrivateAttr()
    _embeddings: Any = PrivateAttr()

    def __init__(self):
        super().__init__()

        self._embeddings = OpenAIEmbeddings(model = EMBED_MODEL)
        db_path = PERSIST_PATH / INDEX_NAME
        if not db_path.exists():
            raise RuntimeError(
                f"No vector store found at {db_path}. Upload a document first."
            )

        self._vectordb = FAISS.load_local(
            str(db_path), self._embeddings,
            allow_dangerous_deserialization=True
        )

    def _run(self, query: str) -> List[Dict]:
        """
        Returns:
            A list of dicts, each with `content` (str) and `metadata` (dict).
            This is automatically JSON-serialised when the tool call is passed
            back to the LLM.
        """
        docs: List[Document] = self._vectordb.similarity_search(query, k = TOP_K)

        result = [
            {"content": d.page_content, "metadata": d.metadata} for d in docs
        ]
        return result