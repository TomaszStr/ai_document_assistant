import os
import hashlib
from typing import Dict, Any
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_assistant.core.vector_store import VectorStoreManager

class DocumentIngestor:
    """
    Pure ETL Pipeline: Extracts text, transforms to chunks, and loads to Vector DB.
    Strictly stateless. Returns metadata 'receipts' to the caller.
    """
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def _get_loader(self, file_path: str):
        """Factory method for document loaders."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return PyPDFLoader(file_path)
        elif ext == '.docx':
            return Docx2txtLoader(file_path)
        elif ext == '.md':
            return UnstructuredMarkdownLoader(file_path)
        else:
            return TextLoader(file_path)

    def process(self, file_path: str, session_id: str) -> Dict[str, Any]:
        """
        Executes the ingestion pipeline and returns a metadata receipt.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_name = os.path.basename(file_path)
        source_id = hashlib.md5(f"{session_id}_{file_name}".encode()).hexdigest()

        # Extract
        print(f"[{session_id}] Extracting data from {file_name}...")
        loader = self._get_loader(file_path)
        docs = loader.load()

        file_type = os.path.splitext(file_name)[1][1:] or "txt"

        # Transform (Metadata Injection & Chunking)
        for doc in docs:
            doc.metadata.update({
                "source_id": source_id,
                "session_id": session_id,
                "file_name": file_name,
                "file_type": file_type
            })
            if "page" not in doc.metadata:
                doc.metadata["page"] = 0

        chunks = self.text_splitter.split_documents(docs)

        # Persist in Vector Storage
        print(f"[{session_id}] Generating embeddings for {len(chunks)} chunks...")
        self.vector_store.add_documents(chunks)

        # Return Receipt (For the SessionManager to save in JSON state)
        return {
            "source_id": source_id,
            "file_name": file_name,
            "file_type": file_type,
            "total_chunks": len(chunks),
            "status": "ingested"
        }