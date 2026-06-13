import logging

from langchain_core.callbacks import BaseCallbackHandler

# Setup persistent file logging
logging.basicConfig(
    filename='agent_audit.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def extract_token_usage(response) -> dict:
    """
    Robustly extracts token usage from an LLMResult across different providers (Ollama, OpenAI).
    Checks standard llm_output, usage_metadata, and response_metadata.
    """
    # Try standard llm_output['token_usage']
    if response.llm_output and "token_usage" in response.llm_output:
        return response.llm_output["token_usage"]

    # Check Generations (standard for ChatModels)
    if response.generations:
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if not msg:
                    continue

                # LangChain 0.2+ usage_metadata standard
                usage = getattr(msg, "usage_metadata", None)
                if usage:
                    return {
                        "prompt_tokens": usage.get("input_tokens", 0),
                        "completion_tokens": usage.get("output_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0)
                    }

                # response_metadata fallback (often used by Ollama/OpenAI)
                res_meta = getattr(msg, "response_metadata", {})
                if "token_usage" in res_meta:
                    return res_meta["token_usage"]

    return {}


class AgentObservabilityLogsHandler(BaseCallbackHandler):
    """Custom callback to listen to Single and Multi-Agent execution steps."""

    def __init__(self, orchestrator_type: str = "Single-Agent"):
        self.orchestrator_type = orchestrator_type

    def _get_agent_name(self, kwargs: dict) -> str:
        """Helper to extract which agent/node is currently executing."""
        tags = kwargs.get("tags", [])
        base_name = f"[{self.orchestrator_type}]"
        if tags:
            # LangGraph often injects the node name into the tags
            return f"{base_name} [{tags[0]}]"
        return f"{base_name} [System]"

    def on_llm_end(self, response, **kwargs):
        agent_name = self._get_agent_name(kwargs)
        usage = extract_token_usage(response)
        logging.info(f"{agent_name} [LLM END] Tokens Used: {usage}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        agent_name = self._get_agent_name(kwargs)
        tool_name = serialized.get("name", "unknown")
        logging.info(f"{agent_name} [TOOL START] Invoked '{tool_name}' with input: {input_str}")

    def on_tool_error(self, error: BaseException, **kwargs):
        agent_name = self._get_agent_name(kwargs)
        logging.error(f"{agent_name} [TOOL ERROR] Error: {str(error)}")

    def on_retriever_end(self, documents, **kwargs):
        agent_name = self._get_agent_name(kwargs)
        logging.info(f"{agent_name} [RETRIEVER END] Fetched {len(documents)} chunks.")
        if documents:
            meta = documents[0].metadata
            logging.info(f"   -> Top Match: {meta.get('file_name')} (Page {meta.get('page', 'Unknown')})")


class TurnMetricsHandler(BaseCallbackHandler):
    """Accumulates token, tool, and chunk metrics for a single execution turn."""

    def __init__(self):
        self.metrics = {
            "tokens": {"total": 0, "prompt": 0, "completion": 0},
            "tools_called": [],
            "source_chunks": []
        }

    def on_llm_end(self, response, **kwargs):
        """Captures token usage using the robust extraction helper."""
        usage = extract_token_usage(response)
        if usage:
            # Normalize keys across different provider naming conventions
            p_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            c_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
            t_tokens = usage.get("total_tokens") or (p_tokens + c_tokens)

            self.metrics["tokens"]["prompt"] += p_tokens
            self.metrics["tokens"]["completion"] += c_tokens
            self.metrics["tokens"]["total"] += t_tokens

    def on_tool_start(self, serialized, input_str, **kwargs):
        """Captures which tools were fired and what was sent to them."""
        tags = kwargs.get("tags", [])
        caller = tags[0] if tags else "Agent"

        self.metrics["tools_called"].append({
            "name": f"[{caller}] {serialized.get('name', 'unknown')}",
            "input": input_str
        })

    def on_chain_start(self, serialized, inputs, **kwargs):
        """Captures specialized agent invocations to display in the UI as 'actions'."""
        name = serialized.get("name", "unknown")
        if name in ["SearchAgent", "SummaryAgent", "MetadataAgent"]:
            self.metrics["tools_called"].append({
                "name": f"🔄 RouteTo({name})",
                "input": "Delegated sub-task to specialized worker."
            })

    def log_agent_routing(self, agent_name: str, reason: str):
        """Manually log an agent handoff as an action so the UI can display it."""
        self.metrics["tools_called"].append({
            "name": f"RouteTo({agent_name})",
            "input": reason
        })

    def on_retriever_end(self, documents, **kwargs):
        """Captures the exact text chunks used to ground the answer."""
        for doc in documents:
            self.metrics["source_chunks"].append({
                "file_name": doc.metadata.get("file_name", "Unknown"),
                "page": doc.metadata.get("page", 0),
                "content_preview": doc.page_content[:200] + "..."
            })
