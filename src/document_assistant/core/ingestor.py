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
from pypdf import PdfReader
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

        self.truncate_summary_half_size = 2000

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

    def _extract_pdf_metadata(self, file_path: str) -> dict:
        """Helper method to extract Author, Page Count, and Table of Contents."""
        extracted_data = {
            "author": "Unknown",
            "total_pages": 0,
            "table_of_contents": "No structural Table of Contents embedded."
        }

        try:
            reader = PdfReader(file_path)

            # Extract Author and Pages
            meta = reader.metadata or {}
            extracted_data["author"] = meta.get('/Author', 'Unknown')
            extracted_data["total_pages"] = len(reader.pages)

            # Recursively parse the Table of Contents (Outline)
            toc_lines = []
            if reader.outline:
                def _parse_outline(outline_items, level=0):
                    for item in outline_items:
                        if isinstance(item, list):
                            _parse_outline(item, level + 1)
                        else:
                            title = item.title
                            page_num = reader.get_page_number(item.page) + 1
                            indent = "  " * level
                            toc_lines.append(f"{indent}- {title} (Page {page_num})")

                _parse_outline(reader.outline)

            if toc_lines:
                extracted_data["table_of_contents"] = "\n".join(toc_lines)

        except Exception as e:
            print(f"⚠️ Warning: Failed to extract deep metadata from {file_path}: {e}")

        return extracted_data

    def process(self, file_path: str, session_id: str) -> Dict[str, Any]:
        """
        Executes the ingestion pipeline and returns a metadata receipt.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_name = os.path.basename(file_path)
        file_type = os.path.splitext(file_name)[1][1:].lower() or "txt"
        source_id = hashlib.md5(f"{session_id}_{file_name}".encode()).hexdigest()

        # Initialize rich metadata
        rich_metadata = {"author": "Unknown", "total_pages": 0, "table_of_contents": "N/A"}

        # If it's a PDF, extract the deep metadata before loading chunks
        if file_type == 'pdf':
            print(f"[{session_id}] Extracting deep metadata for {file_name}...")
            rich_metadata = self._extract_pdf_metadata(file_path)

        # Extract (Standard LangChain loading)
        print(f"[{session_id}] Extracting text chunks from {file_name}...")
        loader = self._get_loader(file_path)
        docs = loader.load()

        # Combine all text to easily slice the intro and outro
        full_text = "\n".join([doc.page_content for doc in docs])

        # Safely slice (if document is smaller than truncate size chars, take the whole thing)
        if len(full_text) > self.truncate_summary_half_size * 2:
            intro_outro = (full_text[:self.truncate_summary_half_size] +
                           "\n\n...[CONTENT TRUNCATED]...\n\n" +
                           full_text[-self.truncate_summary_half_size:])
        else:
            intro_outro = full_text

        # In a real system, you would pass `intro_outro` to your LLM here to generate a clean summary.
        # Store the raw truncated text to let the Orchestrator agent summarize it at runtime, possible future improvements.
        global_summary_text = intro_outro

        # Fallback for page count if it wasn't a PDF
        if rich_metadata["total_pages"] == 0:
            rich_metadata["total_pages"] = len(docs)

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

        # Return Receipt with the metadata
        return {
            "source_id": source_id,
            "file_name": file_name,
            "file_type": file_type,
            "total_chunks": len(chunks),
            "author": rich_metadata["author"],
            "total_pages": rich_metadata["total_pages"],
            "table_of_contents": rich_metadata["table_of_contents"],
            "global_summary": global_summary_text,
            "status": "ingested"
        }
