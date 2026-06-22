import glob
import os
import time

import streamlit as st

from document_assistant.api.assistant import Assistant
from document_assistant.core.ingestor import DocumentIngestor
from document_assistant.core.repository import SessionRepository
from document_assistant.core.tools.registry import ToolRegistry
from document_assistant.core.vector_store import VectorStoreManager
from document_assistant.orchestration.manager import SessionManager
from document_assistant.orchestration.single_agent import SingleAgentOrchestrator

# Page Configuration
st.set_page_config(page_title="Local RAG Agent", page_icon="🤖", layout="wide")


def get_stored_sessions() -> list[str]:
    """Scans /data/sessions for existing JSON session files."""
    target_dir = "./data/sessions"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        return ["default"]
    files = glob.glob(os.path.join(target_dir, "*.json"))
    sessions = [os.path.basename(f).replace(".json", "") for f in files]
    return sessions if sessions else ["default"]


# Backend Initialization
@st.cache_resource
def get_assistant():
    vector_store = VectorStoreManager(persist_directory="./data/chroma_db")
    repository = SessionRepository(storage_dir="./data/sessions")
    ingestor = DocumentIngestor(vector_store=vector_store)
    session_manager = SessionManager(ingestor=ingestor, repository=repository)
    tool_registry = ToolRegistry(vector_store=vector_store, session_manager=session_manager)
    orchestrator = SingleAgentOrchestrator(session_manager=session_manager, tool_registry=tool_registry,
                                           model="llama3.2:3b", use_local=True)
    return Assistant(session_manager=session_manager, orchestrator=orchestrator, tool_registry=tool_registry)


assistant = get_assistant()


def sync_ui_with_backend(session_name: str):
    """Initializes/Switches backend session and rebuilds frontend message history."""
    assistant.set_session(session_name)
    state = assistant._session_manager.get_context()
    if state:
        st.session_state.messages = []
        for msg in state.chat_history:
            st.session_state.messages.append({
                "role": msg["role"],
                "content": msg["content"],
                "metadata": msg.get("metadata", {})
            })
    else:
        st.session_state.messages = []


def get_current_session_tokens() -> int:
    """Dynamically calculates total tokens for the current session."""
    total = 0
    for msg in st.session_state.messages:
        meta = msg.get("metadata", {})
        if "tokens" in meta:
            total += meta["tokens"].get("total", 0)
    return total


# Frontend Session State Setup
if "session_name" not in st.session_state:
    sessions = get_stored_sessions()
    initial_session = sessions[0] if sessions else "default"
    st.session_state.session_name = initial_session
    sync_ui_with_backend(initial_session)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "allowed_sources" not in st.session_state:
    st.session_state.allowed_sources = []

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ==========================================
# SIDEBAR: The Master Control Panel
# ==========================================
with st.sidebar:
    st.title("🛠️ Workspace Controls")

    # --- Configuration ---
    st.subheader("⚙️ Configuration")

    # Model Selection
    available_models = assistant.get_available_models()
    current_model = getattr(assistant._orchestrator, 'model_name', available_models[0] if available_models else "")

    selected_model = st.selectbox(
        "LLM Model",
        options=available_models,
        index=available_models.index(current_model) if current_model in available_models else 0
    )
    if selected_model != current_model:
        with st.spinner(f"Loading {selected_model}..."):
            assistant.change_model(selected_model)
            st.rerun()

    # Agent Mode
    current_mode = assistant.mode
    new_mode = st.radio(
        "Select Orchestrator",
        options=["single", "multi"],
        index=0 if current_mode == "single" else 1,
        format_func=lambda x: "Single-Agent (Legacy)" if x == "single" else "Multi-Agent (Supervisor)"
    )
    if new_mode != current_mode:
        with st.spinner(f"Switching to {new_mode}-agent mode..."):
            assistant.switch_mode(new_mode)
            st.rerun()

    st.divider()

    # --- 1. Session Management ---
    st.subheader("📁 Sessions")
    available_sessions = get_stored_sessions()

    if st.session_state.session_name not in available_sessions:
        available_sessions.append(st.session_state.session_name)

    selected_session = st.selectbox(
        "Active Session",
        options=available_sessions,
        index=available_sessions.index(st.session_state.session_name)
    )
    if selected_session != st.session_state.session_name:
        st.session_state.session_name = selected_session
        sync_ui_with_backend(selected_session)
        st.rerun()

    new_session_name = st.text_input("Create New Session", placeholder="Type name...")
    if st.button("➕ Create") and new_session_name:
        clean_name = new_session_name.strip().replace(" ", "_")
        st.session_state.session_name = clean_name
        sync_ui_with_backend(clean_name)
        st.rerun()

    st.divider()

    # Document Ingestion
    st.subheader("➕ Add Document")
    uploaded_file = st.file_uploader(
        "Upload File",
        type=["pdf", "txt", "docx"],
        label_visibility="collapsed",
        key=f"file_uploader_{st.session_state.uploader_key}"
    )

    if uploaded_file and st.button("Ingest to Workspace"):
        with st.spinner(f"Vectorizing {uploaded_file.name}..."):
            os.makedirs("./data/temp", exist_ok=True)
            temp_path = f"./data/temp/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            result = assistant.upload_document(temp_path)

            if result.get("status") == "success":
                st.success(f"✅ Ingested: {uploaded_file.name}")
                st.session_state.uploader_key += 1
            else:
                st.error(f"❌ Failed: {result.get('message')}")

            if os.path.exists(temp_path):
                os.remove(temp_path)

            time.sleep(1)
            st.rerun()

    st.divider()

    # Document Checkboxes (Source Filters)
    st.subheader("📄 Filter Documents")
    status = assistant.get_status()
    docs = status.get("documents", [])

    current_allowed = []
    if not docs:
        st.write("_No documents in workspace._")
    else:
        for doc in docs:
            name = doc.get("file_name") or doc.get("metadata", {}).get(
                "file_name") or f"Doc_{doc.get('source_id', '')[:6]}"
            source_id = doc.get("source_id")

            if st.checkbox(f"{name}", value=True, key=f"chk_{source_id}"):
                current_allowed.append(source_id)

    st.session_state.allowed_sources = current_allowed

    st.divider()

    # Dynamically calculate tokens
    session_tokens = get_current_session_tokens()
    st.metric(label="Total Session Tokens", value=session_tokens)

# ==========================================
# MAIN UI: The Chat Interface
# ==========================================
st.title(f"🤖 Agent Session: `{st.session_state.session_name}`")

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]
            if meta.get("tools_called") or meta.get("source_chunks") or meta.get("tokens") or meta.get("orchestrator"):
                with st.expander("🛠️ Tool Calls & Diagnostics"):
                    if "orchestrator" in meta:
                        st.markdown(f"**Orchestrator Engine:** `{meta['orchestrator']}`")

                    if "tokens" in meta:
                        tk = meta["tokens"]
                        st.caption(
                            f"**Tokens (This Turn):** {tk.get('total', 0)} (Prompt: {tk.get('prompt', 0)} | Generation: {tk.get('completion', 0)})")

                    if meta.get("tools_called"):
                        st.markdown("**Actions Taken:**")
                        for tool in meta["tools_called"]:
                            st.code(f"{tool['name']}(input='{tool['input']}')")

                    if meta.get("source_chunks"):
                        st.markdown("**Grounded Sources:**")
                        for chunk in meta["source_chunks"]:
                            st.info(
                                f"📄 **{chunk.get('file_name', 'Unknown')}** (Page {chunk.get('page', '?')}):  \n_{chunk.get('content_preview', '')}_")

# Chat Input Box
if prompt := st.chat_input("Ask your agent a question..."):

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Agent is reasoning..."):

            response_text = assistant.chat(prompt, allowed_sources=st.session_state.allowed_sources)

            st.markdown(response_text)

            turn_metadata = {}
            state = assistant._session_manager.get_context()
            if state and state.chat_history:
                last_turn = state.chat_history[-1]
                if last_turn.get("role") == "assistant" and "metadata" in last_turn:
                    turn_metadata = last_turn["metadata"]

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "metadata": turn_metadata
            })

            st.rerun()
