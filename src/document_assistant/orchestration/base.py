from abc import ABC, abstractmethod


class BaseOrchestrator(ABC):
    """The contract for any agent brain (Single or Multi)."""

    @abstractmethod
    def invoke(self, user_input: str) -> str:
        """Processes input and returns a synthesized response."""
        pass
