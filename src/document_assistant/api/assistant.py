from typing import Dict, Any, List
from document_assistant.orchestration.manager import SessionManager
from document_assistant.orchestration.single_agent import SingleAgentOrchestrator
from document_assistant.core.utils import get_ollama_models


class Assistant:
    """
    The Facade Layer.
    Exposes a clean, simplified API for any UI (Streamlit, FastAPI, CLI) to consume.
    The UI should NEVER import LangChain or ChromaDB directly.
    """

    def __init__(self, session_manager: SessionManager, orchestrator: SingleAgentOrchestrator):
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

    def chat(self, message: str, allowed_sources: list[str] = None, callbacks: list = None) -> str:
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
            response = self._orchestrator.invoke(message, additional_callbacks=callbacks)
            return response
        except Exception as e:
            return f"Agent Error: {str(e)}"

    def get_available_models(self) -> List[str]:
        """Returns a list of available local models."""
        return get_ollama_models()

    def change_model(self, model_name: str, use_local: bool = True) -> str:
        """Dynamically switches the model used by the orchestrator."""
        try:
            self._orchestrator.update_model(model=model_name, use_local=use_local)
            return f"Successfully switched to model: {model_name}"
        except Exception as e:
            return f"Failed to switch model: {str(e)}"

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
