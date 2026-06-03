import operator
from typing import Annotated, List, TypedDict, Optional
from langchain_core.tools import tool
from langchain_core.callbacks import CallbackManagerForToolRun, BaseCallbackHandler
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from document_assistant.core.vector_store import VectorStoreManager
from document_assistant.orchestration.manager import SessionManager


def build_fast_summary_tool(session_manager: SessionManager):
    @tool
    def get_document_overview(file_name: str) -> str:
        """
        Provides a fast, high-level executive overview of an entire document.
        Use this when the user asks "What is this document about?".
        """
        state = session_manager.get_context()
        for doc_id, meta in state.registry.items():
            if meta.get('file_name').lower() == file_name.lower():
                raw_intro_outro = meta.get('global_summary', '')
                if not raw_intro_outro:
                    return f"No overview available for {file_name}."

                return (
                    f"Raw Introduction and Conclusion for {file_name}:\n{raw_intro_outro}\n\n"
                    f"(Agent: Please synthesize this raw text into a cohesive, high-level summary for the user)."
                )

        return f"Error: File '{file_name}' not found."

    return get_document_overview


def build_dynamic_section_summary_tool(vector_store: VectorStoreManager, session_manager: SessionManager):
    @tool
    def summarize_specific_section(file_name: str, section_topic: str,
                                   run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Condenses a specific chapter, section, or topic within a document.
        Provide the file_name and the exact section_topic (e.g., 'Chapter 3' or 'Financial Risks').
        """
        state = session_manager.get_context()
        target_id = None
        for doc_id, meta in state.registry.items():
            if meta.get('file_name').lower() == file_name.lower():
                target_id = doc_id
                break

        if not target_id:
            return f"Error: File '{file_name}' not found."

        # Fetch a high number of chunks to ensure we capture the whole section
        results = vector_store.db.similarity_search(
            section_topic,
            k=15,  # Increased K to capture dense sections
            filter={"source_id": target_id}
        )

        # Manually trigger retriever callbacks to capture chunks in observability handlers
        if run_manager:
            run_manager.on_retriever_end(results)

        content = "\n".join([res.page_content for res in results])
        return (
            f"Raw content extracted for the topic '{section_topic}':\n{content}\n\n"
            f"(Agent: Synthesize this into a cohesive summary focusing exclusively on {section_topic})."
        )

    return summarize_specific_section


# ---------------------------------------------------------
# Map Reduce summary implementation
# ---------------------------------------------------------
class OverallState(TypedDict):
    """The main state holding the raw chunks and the final output."""
    contents: List[str]
    # operator.add ensures that as parallel nodes finish, they append to this list
    summaries: Annotated[list, operator.add]
    final_summary: str
    callbacks: List[BaseCallbackHandler]


class MapState(TypedDict):
    """The isolated state for a single parallel 'Map' node."""
    content: str
    callbacks: List[BaseCallbackHandler]


# Build the LangGraph Nodes
def build_map_reduce_summary_tool(vector_store: VectorStoreManager, session_manager: SessionManager, llm):
    # --- NODE: The Map Step ---
    def map_summarize(state: MapState):
        """Summarizes a single chunk of text."""
        prompt = f"Write a concise summary of the following text chunk:\n\n{state['content']}"
        # Propagate callbacks to capture nested token usage
        response = llm.invoke(prompt, config={"callbacks": state.get("callbacks")})
        return {"summaries": [response.content]}

    # --- NODE: The Reduce Step ---
    def reduce_summaries(state: OverallState):
        """Synthesizes all the chunk summaries into a master summary."""
        combined_summaries = "\n\n".join(state["summaries"])
        prompt = f"Synthesize these summaries into a cohesive, comprehensive master summary of the entire document:\n\n{combined_summaries}"
        # Propagate callbacks to capture nested token usage
        response = llm.invoke(prompt, config={"callbacks": state.get("callbacks")})
        return {"final_summary": response.content}

    # --- ROUTER: The 'Send' Fan-out Logic ---
    def map_routing(state: OverallState):
        """Spawns a parallel map_summarize node for every chunk in the document."""
        return [
            Send("map_summarize", {"content": chunk, "callbacks": state.get("callbacks")})
            for chunk in state["contents"]
        ]

    # --- COMPILE THE GRAPH ---
    graph = StateGraph(OverallState)
    graph.add_node("map_summarize", map_summarize)
    graph.add_node("reduce_summaries", reduce_summaries)

    graph.add_conditional_edges(START, map_routing, ["map_summarize"])
    graph.add_edge("map_summarize", "reduce_summaries")
    graph.add_edge("reduce_summaries", END)

    summary_app = graph.compile()

    # The LangChain Tool Wrapper
    @tool
    def generate_comprehensive_summary(file_name: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        WARNING: This is a slow, comprehensive tool.
        It reads EVERY SINGLE PAGE of a document to generate a master summary.
        Only use this if the user explicitly asks for a "deep, comprehensive summary of the whole file".
        """
        state = session_manager.get_context()
        target_id = None
        for doc_id, meta in state.registry.items():
            if meta.get('file_name').lower() == file_name.lower():
                target_id = doc_id
                break

        if not target_id:
            return f"Error: File '{file_name}' not found."

        # Fetch all chunks directly from ChromaDB
        collection_data = vector_store.db.get(where={"source_id": target_id})
        if not collection_data or not collection_data.get('documents'):
            return "Error: Could not retrieve document chunks from database."

        raw_chunks = collection_data['documents']

        try:
            # Extract callbacks from the current tool run manager
            callbacks = run_manager.get_child() if run_manager else None

            # Execute the LangGraph Map-Reduce Workflow with callback propagation
            result = summary_app.invoke(
                {"contents": raw_chunks, "callbacks": callbacks},
                config={"callbacks": callbacks}
            )
            return f"Comprehensive Summary of {file_name}:\n{result['final_summary']}"

        except Exception as e:
            return f"Failed to execute modern Map-Reduce summarization: {str(e)}"

    return generate_comprehensive_summary
