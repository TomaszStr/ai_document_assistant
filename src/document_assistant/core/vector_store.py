from typing import List

import torch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class VectorStoreManager:
    """
    Manages the lifecycle of the embedding model and the ChromaDB connection.
    Follows Lazy Initialization to save memory.
    """

    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = persist_directory
        self._embeddings = None
        self._db = None

    @property
    def embeddings(self):
        """Lazy load the heavy HuggingFace model only when first requested."""
        if self._embeddings is None:
            print("Loading HuggingFace Embeddings into memory...")
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            self._embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': device}
            )
        return self._embeddings

    @property
    def db(self):
        """Lazy load the ChromaDB collection."""
        if self._db is None:
            self._db = Chroma(
                collection_name="rag_global",
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        return self._db

    def add_documents(self, documents: List[Document]):
        """Saves chunks to the database."""
        self.db.add_documents(documents)

    def delete_by_source(self, source_id: str):
        """Removes a specific document from the database."""
        if self._db:
            self._db._collection.delete(where={"source_id": source_id})
