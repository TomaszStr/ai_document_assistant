from typing import Annotated, Literal, Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from pydantic import BaseModel
from typing_extensions import TypedDict

from document_assistant.core.observability import AgentObservabilityLogsHandler, TurnMetricsHandler
from document_assistant.core.tools.registry import ToolRegistry
from document_assistant.orchestration.base import BaseOrchestrator
from document_assistant.orchestration.manager import SessionManager


class AgentState(TypedDict):
    """Custom State for Multi-Agent Orchestrator."""
    messages: Annotated[list[Any], add_messages]


class MultiAgentOrchestrator(BaseOrchestrator):
    """
    LangGraph-based Multi-Agent Orchestrator.
    Uses the Supervisor pattern with Tool-Based Handoffs and State Filtering.
    """

    def __init__(self, session_manager: SessionManager,
                 tool_registry: ToolRegistry, use_local: bool = True,
                 temperature: float = 0.0, model: str = None):
        self.session_manager = session_manager

        # Categorize tools by agent specialization
        all_tools = tool_registry.get_tools()
        self.search_tools = [t for t in all_tools if t.name in ["semantic_search"]]
        self.summary_tools = [t for t in all_tools if t.name in ["get_document_overview", "summarize_specific_section"]]
        self.metadata_tools = [t for t in all_tools if t.name in ["get_document_metadata", "get_table_of_contents"]]

        self.temperature = temperature
        self.use_local = use_local
        self.model_name = model

        # Initialize LLM
        self.update_model(model="llama3.2:3b" if model is None else model, use_local=use_local)

    def update_model(self, model: str, use_local: bool = None):
        """Dynamically updates the LLM and rebuilds the multi-agent graph."""
        if use_local is not None:
            self.use_local = use_local
        self.model_name = model

        if self.use_local:
            print(f"Initializing Local LLM (Multi-Agent) via Ollama ({model})...")
            self.llm = ChatOllama(
                model=model,
                num_ctx=4096,
                temperature=self.temperature
            )
        else:
            print(f"Initializing Cloud Fallback (Multi-Agent) (OpenAI) ({model})...")
            self.llm = ChatOpenAI(
                model=model,
                temperature=self.temperature
            )
        print("Initialized")

        self.graph = self._build_graph()

    def _build_graph(self):
        """Constructs the Supervisor/Worker LangGraph."""

        search_agent = create_agent(
            self.llm,
            tools=self.search_tools,
            system_prompt=(
                "You are a precise Search Specialist. Your ONLY job is to find specific facts, answer direct questions, "
                "and retrieve quotes from the uploaded documents using the `semantic_search` tool.\n"
                "ALWAYS use `semantic_search` to find the answer. NEVER rely on your general knowledge. "
                "If the search results do not contain the answer, state clearly that the information is not in the documents."
            )
        )

        summary_agent = create_agent(
            self.llm,
            tools=self.summary_tools,
            system_prompt=(
                "You are a Synthesis Specialist. Your ONLY job is to condense large amounts of text, provide high-level "
                "document overviews, or summarize specific chapters using your summary tools.\n"
                "DO NOT answer specific, highly targeted questions. If asked for a specific fact, you are the wrong agent.\n"
                "Use `get_document_overview` for full file summaries. Use `summarize_specific_section` for chapters or topics."
            )
        )

        metadata_agent = create_agent(
            self.llm,
            tools=self.metadata_tools,
            system_prompt=(
                "You are a Structural Librarian. Your ONLY job is to retrieve structural information about the documents, "
                "such as the author, page count, file name, or the Table of Contents."
            )
        )

        # Define Orchestrator Handoff Tools (Pydantic schemas)
        class TransferToSearch(BaseModel):
            """Use this tool to route the user's request to the SearchAgent. The SearchAgent is the ONLY agent capable of answering specific questions, finding facts, or checking if a specific technology (like RAG) is mentioned in the text."""
            pass

        class TransferToSummary(BaseModel):
            """Use this tool to route the user's request to the SummaryAgent. ONLY use this if the user explicitly asks for a broad 'summary', 'overview', or 'condensation' of a document or chapter."""
            pass

        class TransferToMetadata(BaseModel):
            """Use this tool to route the user's request to the MetadataAgent. ONLY use this if the user asks for the table of contents, author, or document structure."""
            pass

        orchestrator_tools = [TransferToSearch, TransferToSummary, TransferToMetadata]
        orchestrator_llm = self.llm.bind_tools(orchestrator_tools)

        # Node Functions
        def orchestrator_node(state: AgentState) -> Command[
            Literal["SearchWorker", "SummaryWorker", "MetadataWorker", "__end__"]]:
            sys_msg = SystemMessage(content=(
                "You are the Multi-Agent Orchestrator. You manage the workflow by delegating tasks to specialized workers.\n"
                "Current Workers:\n"
                "- SearchAgent: Best for finding specific facts or answering questions with document quotes.\n"
                "- SummaryAgent: Best for high-level overviews or condensing large sections.\n"
                "- MetadataAgent: Best for structural info (TOC, page counts, author).\n\n"
                "INSTRUCTIONS:\n"
                "1. If a worker is needed to fulfill the user's request, call the appropriate transfer tool.\n"
                "2. When a worker returns information, synthesize a final, natural language answer for the user.\n"
                "3. If you have enough info to answer fully, provide the final answer and do NOT call any tools.\n"
                "4. Always be conversational and grounded in the provided document data."
            ))

            messages = [sys_msg] + state["messages"]
            response = orchestrator_llm.invoke(messages)

            if response.tool_calls:
                t_name = response.tool_calls[0]["name"]

                if t_name == "TransferToSearch":
                    return Command(goto="SearchWorker", update={"messages": [response]})
                if t_name == "TransferToSummary":
                    return Command(goto="SummaryWorker", update={"messages": [response]})
                if t_name == "TransferToMetadata":
                    return Command(goto="MetadataWorker", update={"messages": [response]})

            # Final response
            return Command(goto=END, update={"messages": [response]})

        def _worker_step(agent, state: AgentState, name: str, config: RunnableConfig) -> Command[Literal["Orchestrator"]]:
            """Helper to execute a worker with State Filtering."""
            # Get the orchestrator's tool call to satisfy it upon return
            last_msg = state["messages"][-1]
            tool_call_id = last_msg.tool_calls[0]["id"]
            tool_name = last_msg.tool_calls[0]["name"]

            # STATE FILTERING:
            # We filter out the Orchestrator's internal reasoning and tool calls.
            # The worker only sees the chat history leading up to the human message + human message.
            filtered_messages = []
            for m in state["messages"]:
                if isinstance(m, HumanMessage):
                    filtered_messages.append(m)
                elif isinstance(m, AIMessage):
                    # Exclude messages with tool_calls to prevent INVALID_CHAT_HISTORY in worker
                    if not getattr(m, 'tool_calls', None):
                        filtered_messages.append(m)

            # Pass config down with run_name injected so on_chain_start can log the specific agent name
            child_config = config.copy() if config else {}
            child_config["run_name"] = name

            result = agent.invoke({"messages": filtered_messages}, config=child_config)
            worker_output = result["messages"][-1].content

            # Return the worker's output as the ToolMessage to fulfill the Orchestrator's tool call
            tool_msg = ToolMessage(content=worker_output, tool_call_id=tool_call_id, name=tool_name)
            return Command(goto="Orchestrator", update={"messages": [tool_msg]})

        def search_node(state: AgentState, config: RunnableConfig):
            return _worker_step(search_agent, state, "SearchAgent", config)

        def summary_node(state: AgentState, config: RunnableConfig):
            return _worker_step(summary_agent, state, "SummaryAgent", config)

        def metadata_node(state: AgentState, config: RunnableConfig):
            return _worker_step(metadata_agent, state, "MetadataAgent", config)

        # 4. Define Graph
        builder = StateGraph(AgentState)
        builder.add_node("Orchestrator", orchestrator_node)
        builder.add_node("SearchWorker", search_node)
        builder.add_node("SummaryWorker", summary_node)
        builder.add_node("MetadataWorker", metadata_node)

        builder.add_edge(START, "Orchestrator")

        return builder.compile()

    def invoke(self, user_input: str, additional_callbacks: list = None) -> str:
        """Standard invoke signature, matching SingleAgentOrchestrator."""
        state = self.session_manager.get_context()
        if not state:
            return "System Error: No active session loaded."

        # Reconstruct message objects from session history
        messages = []
        for msg in state.chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=user_input))

        print(f"\n--- MULTI-AGENT ORCHESTRATION ---")

        # Reuse existing observability handlers
        logs_handler = AgentObservabilityLogsHandler(orchestrator_type="Multi-Agent")
        metrics_handler = TurnMetricsHandler()

        callbacks = [logs_handler, metrics_handler]
        if additional_callbacks:
            callbacks.extend(additional_callbacks)

        # Run the LangGraph
        result = self.graph.invoke(
            {"messages": messages},
            config={"callbacks": callbacks}
        )

        final_msg = result["messages"][-1]
        output = final_msg.content if final_msg.content else "Agent could not generate a response."

        turn_stats = metrics_handler.metrics
        turn_stats["orchestrator"] = "Multi-Agent (Supervisor)"

        # Persist results to session
        self.session_manager.add_message("user", user_input)
        self.session_manager.add_message("assistant", output, metadata=turn_stats)

        return output
