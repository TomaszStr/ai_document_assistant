from langchain_core.tools import tool
from document_assistant.orchestration.manager import SessionManager


def build_document_metadata_tool(session_manager: SessionManager):
    # Metadata/Structure Tool
    @tool
    def get_document_metadata() -> str:
        """
        Extracts the document's structure and metadata. Use this to find out
        what files are available, their names, and how large they are.
        """
        state = session_manager.get_context()
        if not state or not state.registry:
            return "No documents have been ingested yet."

        inventory = []
        for doc_id, meta in state.registry.items():
            inventory.append(
                f"- File: {meta.get('file_name')} | Type: {meta.get('file_type')} | Chunks: {meta.get('total_chunks')}"
            )

        return "Active Documents:\n" + "\n".join(inventory)

    return get_document_metadata


def build_toc_tool(session_manager):
    @tool
    def get_table_of_contents(file_name: str) -> str:
        """
        Retrieves the Table of Contents or structural outline for a specific document.
        Always use `get_document_metadata` first to find the exact file_name.

        Args:
            file_name (str): The exact name of the file to inspect.
        """
        state = session_manager.get_context()
        if not state or not state.registry:
            return "No documents available."

        # Find the specific document in the registry
        for doc_id, meta in state.registry.items():
            if meta.get('file_name') == file_name:
                toc = meta.get('table_of_contents')
                if not toc:
                    return f"No Table of Contents could be extracted for {file_name}."
                return f"Table of Contents for {file_name}:\n{toc}"

        return f"Error: Document '{file_name}' not found."

    return get_table_of_contents
