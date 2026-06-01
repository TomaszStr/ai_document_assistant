from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class SessionState:
    session_id: str
    # Registry maps source_id -> Document Metadata Receipt
    registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Chat history is a list of standardized message dictionaries
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    # Tool logs for debugging the ReAct loop
    tool_logs: List[Dict[str, Any]] = field(default_factory=list)
    # Store runtime filters
    active_filters: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "registry": self.registry,
            "chat_history": self.chat_history,
            "tool_logs": self.tool_logs
        }
