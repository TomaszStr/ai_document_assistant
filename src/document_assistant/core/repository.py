import os
import json
from typing import List

from document_assistant.orchestration.state import SessionState


class SessionRepository:
    """Handles the persistence of the SessionState to the filesystem."""

    def __init__(self, storage_dir: str = "./data/sessions"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def _get_filepath(self, session_id: str) -> str:
        return os.path.join(self.storage_dir, f"{session_id}.json")

    def save(self, state: SessionState):
        """Writes the state to a JSON file."""
        filepath = self._get_filepath(state.session_id)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=4)

    def load(self, session_id: str) -> SessionState:
        """Loads a state from disk, or creates a new one if it doesn't exist."""
        filepath = self._get_filepath(session_id)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return SessionState(
                    session_id=data["session_id"],
                    registry=data.get("registry", {}),
                    chat_history=data.get("chat_history", []),
                    system_memory=data.get("system_memory", []),
                    tool_logs=data.get("tool_logs", [])
                )
        # Return a fresh state if no file exists
        return SessionState(session_id=session_id)

    def list_sessions(self) -> List[str]:
        """Returns a list of session IDs found in the storage directory."""
        sessions = []
        for filename in os.listdir(self.storage_dir):
            if filename.endswith(".json"):
                sessions.append(filename[:-5])
        return sessions

    def set_active_filters(self, source_ids: List[str] = None):
        """Temporarily scopes the session to specific documents for the next agent run."""
        if not self.active_state:
            raise ValueError("No active session loaded.")

        # If None or empty, we clear the filters (search all)
        self.active_state.active_filters = source_ids or []
