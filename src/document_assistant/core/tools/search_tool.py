from langchain_core.tools import tool
from langchain_core.callbacks import CallbackManagerForToolRun
from document_assistant.core.vector_store import VectorStoreManager
from document_assistant.orchestration.manager import SessionManager


def build_semantic_search_tool(vector_store: VectorStoreManager, session_manager: SessionManager):
    # Semantic Search Tool (RAG)
    @tool
    def semantic_search(query: str, run_manager: CallbackManagerForToolRun = None) -> str:
        """
        Useful for searching the contents of the uploaded documents.
        Converts queries into vector embeddings to find the most relevant chunks.
        """
        state = session_manager.get_context()
        if not state or not state.registry:
            return "Error: No active documents in the current session."

        # Always lock search to the current session
        chroma_filter = {"session_id": state.session_id}

        # Apply UI Filters if they exist
        if hasattr(state, 'active_filters') and state.active_filters:
            chroma_filter = {
                "$and": [
                    {"session_id": state.session_id},
                    {"source_id": {"$in": state.active_filters}}
                ]
            }

        results = vector_store.db.similarity_search(
            query,
            k=4,
            filter=chroma_filter
        )

        # Manually trigger retriever callbacks to capture chunks in observability handlers
        if run_manager:
            run_manager.on_retriever_end(results)

        if not results:
            return "No relevant information found in the active documents."

        formatted_results = "\n\n".join(
            [f"Source: {res.metadata.get('file_name')} (Page {res.metadata.get('page')}):\n{res.page_content}"
             for res in results]
        )
        return formatted_results

    return semantic_search
