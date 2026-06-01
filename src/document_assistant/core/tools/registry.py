from typing import List, Type
from langchain_core.tools import BaseTool, tool
from document_assistant.core.vector_store import VectorStoreManager
from document_assistant.orchestration.manager import SessionManager

class ToolRegistry:
    """
    Acts as a factory to build and return LangChain tools.
    Injects necessary managers so tools can access the data layer and session state.
    """
    def __init__(self, vector_store: VectorStoreManager, session_manager: SessionManager):
        self.vector_store = vector_store
        self.session_manager = session_manager

    def get_tools(self) -> List[BaseTool]:
        """Returns the suite of specialized tools for the General Agent."""

        # Semantic Search Tool (RAG)
        @tool
        def semantic_search(query: str) -> str:
            """
            Useful for searching the contents of the uploaded documents.
            Converts queries into vector embeddings to find the most relevant chunks.
            """
            state = self.session_manager.get_context()
            if not state or not state.registry:
                return "Error: No active documents in the current session."

            # Always lock search to the current session
            chroma_filter = {"session_id": state.session_id}

            # Apply UI Filters if they exist
            if hasattr(state, 'active_filters') and state.active_filters:
                # ChromaDB requires the $and operator to combine conditions
                chroma_filter = {
                    "$and": [
                        {"session_id": state.session_id},
                        {"source_id": {"$in": state.active_filters}}
                    ]
                }

            results = self.vector_store.db.similarity_search(
                query,
                k=4,
                filter=chroma_filter
            )

            if not results:
                return "No relevant information found in the active documents."

            formatted_results = "\n\n".join(
                [f"Source: {res.metadata.get('file_name')} (Page {res.metadata.get('page')}):\n{res.page_content}"
                 for res in results]
            )
            return formatted_results

        # Metadata/Structure Tool
        @tool
        def get_document_metadata() -> str:
            """
            Extracts the document's structure and metadata. Use this to find out
            what files are available, their names, and how large they are.
            """
            state = self.session_manager.get_context()
            if not state or not state.registry:
                return "No documents have been ingested yet."

            inventory = []
            for doc_id, meta in state.registry.items():
                inventory.append(
                    f"- File: {meta.get('file_name')} | Type: {meta.get('file_type')} | Chunks: {meta.get('total_chunks')}"
                )

            return "Active Documents:\n" + "\n".join(inventory)

        # Summarization Tool
        @tool
        def summarize_document(file_name: str) -> str:
            """
            Condenses specific files when a high-level overview is requested.
            Provide the exact file_name from the metadata tool.
            """
            state = self.session_manager.get_context()
            # Find the source_id for the given filename
            target_id = None
            for doc_id, meta in state.registry.items():
                if meta.get('file_name').lower() == file_name.lower():
                    target_id = doc_id
                    break

            if not target_id:
                return f"Error: File '{file_name}' not found in current session."

            # Fetch chunks
            results = self.vector_store.db.similarity_search(
                "introduction summary overview",
                k=5,
                filter={"source_id": target_id}
            )

            content = "\n".join([res.page_content for res in results])
            return f"Raw content extracted for summarization:\n{content}\n\n(Agent: Please synthesize this into a summary)."

        return [semantic_search, get_document_metadata, summarize_document]