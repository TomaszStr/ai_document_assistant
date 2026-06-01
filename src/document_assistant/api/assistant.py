from typing import Dict, Any
from document_assistant.orchestration.manager import SessionManager
from document_assistant.orchestration.agent import AgentOrchestrator

class Assistant:
    """
    The Facade Layer.
    Exposes a clean, simplified API for any UI (Streamlit, FastAPI, CLI) to consume.
    The UI should NEVER import LangChain or ChromaDB directly.
    """
    def __init__(self, session_manager: SessionManager, orchestrator: AgentOrchestrator):
        self._session_manager = session_manager
        self._orchestrator = orchestrator

    def set_session(self, session_id: str):
        """Initializes or switches the active conversational context."""
        self._session_manager.load_session(session_id)
        return f"Session '{session_id}' is now active."

    def upload_document(self, file_path: str) -> Dict[str, Any]:
        """
        Processes a document and attaches it to the active session.
        Returns a dictionary with the success status and metadata.
        """
        try:
            source_id = self._session_manager.add_document(file_path)
            return {"status": "success", "source_id": source_id}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def chat(self, message: str, allowed_sources: list[str] = None) -> str:
        """
        Sends a user message to the ReAct agent.
        Optionally filters the RAG search to specific source IDs.
        """
        if not self._session_manager.get_context():
            return "System Error: Please set a session before chatting."

        try:
            # Apply UI Checkbox Filters to the State
            self._session_manager.set_active_filters(allowed_sources)

            # Run the Agent
            response = self._orchestrator.invoke(message)
            return response
        except Exception as e:
            return f"Agent Error: {str(e)}"

    def get_status(self) -> Dict[str, Any]:
        """Returns a snapshot of the current session for the UI to render."""
        state = self._session_manager.get_context()
        if not state:
            return {"status": "No active session."}

        return {
            "session_id": state.session_id,
            "documents": list(state.registry.values()),
            "message_count": len(state.chat_history)
        }


