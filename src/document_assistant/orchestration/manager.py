from typing import List

from document_assistant.core.ingestor import DocumentIngestor
from document_assistant.core.repository import SessionRepository
from document_assistant.orchestration.state import SessionState


class SessionManager:
    """
    The central orchestrator for state. It routes document tasks to the Ingestor
    and maintains the conversational context.
    """

    def __init__(self, ingestor: DocumentIngestor, repository: SessionRepository):
        self.ingestor = ingestor
        self.repository = repository
        self.active_state: SessionState = None

    def load_session(self, session_id: str):
        """Loads a session into active memory."""
        print(f"Loading context for session: {session_id}")
        self.active_state = self.repository.load(session_id)

    def save_session(self):
        """Persists the current active state to the repository."""
        if self.active_state:
            self.repository.save(self.active_state)

    def add_document(self, file_path: str) -> str:
        """
        Delegates processing to the stateless Ingestor, then updates
        the session's state with the resulting receipt.
        """
        if not self.active_state:
            raise ValueError("No active session loaded. Call load_session() first.")

        session_id = self.active_state.session_id

        # Ingest
        receipt = self.ingestor.process(file_path, session_id)

        # Update State
        source_id = receipt["source_id"]
        self.active_state.registry[source_id] = receipt

        # Save progress
        self.save_session()
        print(f"Document registered to session {session_id}. Receipt: {source_id}")
        return source_id

    def add_message(self, role: str, content: str, metadata: dict = None):
        """Appends a message to the multi-turn chat history."""
        if not self.active_state:
            raise ValueError("No active session loaded.")

        msg = {
            "role": role,
            "content": content
        }

        if metadata:
            msg["metadata"] = metadata

        self.active_state.chat_history.append(msg)
        self.save_session()

    def get_context(self) -> SessionState:
        """Returns the active state for the AgentOrchestrator to read."""
        return self.active_state

    def set_active_filters(self, source_ids: List[str] = None):
        """Temporarily scopes the session to specific documents for the next agent run."""
        if not self.active_state:
            raise ValueError("No active session loaded.")

        # If None or empty clear the filters
        self.active_state.active_filters = source_ids or []
