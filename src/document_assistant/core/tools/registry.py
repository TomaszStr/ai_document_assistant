from typing import List
from langchain_core.tools import BaseTool

from document_assistant.core.tools.summary_tool import build_fast_summary_tool, build_dynamic_section_summary_tool
from document_assistant.core.vector_store import VectorStoreManager
from document_assistant.orchestration.manager import SessionManager

from document_assistant.core.tools.metadata_tool import build_document_metadata_tool, build_toc_tool
from document_assistant.core.tools.search_tool import build_semantic_search_tool


class ToolRegistry:
    """
    Acts as a factory to build and return LangChain tools.
    Injects necessary managers so tools can access the data layer and session state.
    """

    def __init__(self, vector_store: VectorStoreManager, session_manager: SessionManager):
        self.vector_store = vector_store
        self.session_manager = session_manager

    def get_tools(self) -> List[BaseTool]:
        return [
            build_semantic_search_tool(self.vector_store, self.session_manager),
            build_fast_summary_tool(self.session_manager),
            build_dynamic_section_summary_tool(self.vector_store, self.session_manager),
            build_document_metadata_tool(self.session_manager),
            build_toc_tool(self.session_manager),
        ]
