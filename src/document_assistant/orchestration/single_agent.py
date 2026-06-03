from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from document_assistant.core.observability import AgentObservabilityLogsHandler, TurnMetricsHandler
from document_assistant.orchestration.base import BaseOrchestrator
from document_assistant.orchestration.manager import SessionManager
from document_assistant.core.tools.registry import ToolRegistry


class SingleAgentOrchestrator(BaseOrchestrator):
    """
    The General Orchestrator Agent (Phase 1).
    Evaluates user intent, executes actions, and synthesizes responses.
    """

    def __init__(self, session_manager: SessionManager,
                 tool_registry: ToolRegistry, use_local: bool = True,
                 temperature: float = 0.0, model: str = None):
        self.session_manager = session_manager
        self.tools = tool_registry.get_tools()
        self.temperature = temperature
        self.use_local = use_local

        # Initialize LLM
        self.update_model(model="qwen3.5:4b" if model is None else model, use_local=use_local)

    def update_model(self, model: str, use_local: bool = None):
        """Dynamically updates the LLM and rebuilds the agent executor."""
        if use_local is not None:
            self.use_local = use_local

        if self.use_local:
            print(f"Initializing Local LLM via Ollama ({model})...")
            self.llm = ChatOllama(
                model=model,
                num_ctx=1024, # Limit context
                temperature=self.temperature
            )
        else:
            print(f"Initializing Cloud Fallback (OpenAI) ({model})...")
            self.llm = ChatOpenAI(
                model=model,
                temperature=self.temperature
            )

        self.agent_executor = self._build_agent()

    def _build_agent(self):
        """Constructs the LangGraph v1 Agent Loop."""

        system_prompt = """You are a specialized General Orchestrator Agent.
        Your primary goal is to interact with uploaded document data.

        CRITICAL INSTRUCTIONS:
        1. If you need to use a tool, use the provided tool-calling mechanism. DO NOT output the raw JSON of the tool call to the user.
        2. Once you receive the 'Observation' from a tool, you MUST synthesize a natural language response.
        3. NEVER end your turn with a JSON object. ALWAYS provide a conversational, synthesized final answer based on the tool's output.
        4. Synthesize your responses strictly grounded in the source material. Do not hallucinate.
        """

        agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt
        )

        return agent

    def invoke(self, user_input: str) -> str:
        """
        Takes the user input, injects the active session history,
        runs the agent graph loop, and saves the result.
        """
        state = self.session_manager.get_context()
        if not state:
            return "System Error: No active session loaded."

        # LangGraph state dictionary containing a list of standard message objects.
        messages = []
        for msg in state.chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        # Append the user input
        messages.append(HumanMessage(content=user_input))

        # Agent ReAct loop
        print(f"\n--- AGENT THINKING ---")

        logs_handler = AgentObservabilityLogsHandler()
        metrics_handler = TurnMetricsHandler()

        response = self.agent_executor.invoke(
            input={"messages": messages},
            config={"callbacks": [logs_handler, metrics_handler]}
        )

        final_message = response["messages"][-1]
        output = final_message.content if final_message.content else "I could not generate a response."
        turn_stats = metrics_handler.metrics

        # Update Session State
        self.session_manager.add_message("user", user_input)
        self.session_manager.add_message("assistant", output, metadata=turn_stats)

        return output
