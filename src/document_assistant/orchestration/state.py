from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class SessionState:
    session_id: str
    # Registry maps source_id -> Document Metadata Receipt
    registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Chat history is a list of standardized message dictionaries
    # Format: {"role": "assistant", "content": "Hello", "metadata": {"tokens": 150, "tools": [...]}}
    chat_history: List[Dict[str, Any]] = field(default_factory=list)
    # System Context & Memory (For Context Management Phase)
    # Format: "User prefers concise answers. Previously discussed Chapter 3."
    system_memory: str = ""
    # Tool logs for debugging the ReAct loop
    tool_logs: List[Dict[str, Any]] = field(default_factory=list)
    # Store runtime filters
    active_filters: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "registry": self.registry,
            "chat_history": self.chat_history,
            "system_memory": self.system_memory,
            "active_filters": self.active_filters
        }
