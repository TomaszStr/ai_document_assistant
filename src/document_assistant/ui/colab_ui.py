import ipywidgets as widgets
from IPython.display import display, clear_output
from document_assistant.core.vector_store import VectorStoreManager
from document_assistant.core.repository import SessionRepository
from document_assistant.core.ingestor import DocumentIngestor
from document_assistant.core.tools.registry import ToolRegistry
from document_assistant.orchestration.manager import SessionManager
from document_assistant.orchestration.single_agent import SingleAgentOrchestrator
from document_assistant.api.assistant import Assistant


def build_assistant(use_local_llm: bool = True) -> Assistant:
    """
    The Composition Root.
    Wires up the dependencies from Layer 1 (Data) up to Layer 4 (Facade).
    """
    print("Bootstrapping RAG Architecture...")

    vector_store = VectorStoreManager(persist_directory="./data/chroma_db")
    repository = SessionRepository(storage_dir="./data/sessions")

    ingestor = DocumentIngestor(vector_store=vector_store)

    session_manager = SessionManager(ingestor=ingestor, repository=repository)

    tool_registry = ToolRegistry(vector_store=vector_store, session_manager=session_manager)

    orchestrator = SingleAgentOrchestrator(
        session_manager=session_manager,
        tool_registry=tool_registry,
        use_local=use_local_llm,
        temperature=0.0
    )

    assistant = Assistant(session_manager=session_manager, orchestrator=orchestrator)

    print("System ready.")
    return assistant


# ==========================================
# Example Usage (Simulating a UI interaction)
# ==========================================

assistant = build_assistant(use_local_llm=True)

assistant.set_session("agent_systems_poc_demo")

print(assistant.upload_document("/content/Agent Systems - Agent Definition and Interaction Modelling.pdf"))

print("\n--- UI Dashboard ---")
print(assistant.get_status())

print("\n--- Chat Interface ---")
user_msg = "What documents are available currently?"
print(f"User: {user_msg}")

agent_response = assistant.chat(user_msg)
print(f"Agent: {agent_response}")


def create_colab_ui(assistant):
    """
    Builds an ipywidgets UI for the RAG Assistant in Google Colab.
    """
    # Initialize a default testing session
    assistant.set_session("agent_systems_poc_demo_ui")

    # ==========================================
    # DOCUMENT MANAGEMENT PANEL (Left Side)
    # ==========================================
    doc_header = widgets.HTML("<h3>Document Management</h3>")
    file_input = widgets.Text(
        description="File Path:",
        placeholder="/content/sample.pdf",
        layout=widgets.Layout(width='300px')
    )
    ingest_btn = widgets.Button(description="Ingest Document", button_style="info")
    ingest_output = widgets.Output()

    # Dynamic Checkbox area for sources
    sources_header = widgets.HTML("<b>Active Sources (Include in RAG):</b>")
    sources_box = widgets.VBox([])
    checkboxes_dict = {}  # Maps source_id to the Checkbox widget

    def update_sources_ui():
        """Pulls the latest registry from the Assistant and updates checkboxes."""
        status = assistant.get_status()
        if "documents" in status and status["documents"]:
            cb_list = []
            for doc in status["documents"]:
                sid = doc.get("source_id", doc['file_name'])
                # Create a checkbox if it doesn't exist
                if sid not in checkboxes_dict:
                    checkboxes_dict[sid] = widgets.Checkbox(
                        value=True,
                        description=f"{doc['file_name']} ({doc.get('total_chunks', 0)} chunks)",
                        indent=False
                    )
                cb_list.append(checkboxes_dict[sid])
            sources_box.children = tuple(cb_list)
        else:
            sources_box.children = (widgets.HTML("<i>No documents ingested yet.</i>"),)

    def on_ingest_clicked(b):
        with ingest_output:
            clear_output()
            print(f"Uploading '{file_input.value}'...")
            result = assistant.upload_document(file_input.value)

            if result.get("status") == "success":
                print(f"✅ Success! (ID: {result.get('source_id')[:8]}...)")
                update_sources_ui()
                file_input.value = ""  # Clear input
            else:
                print(f"❌ Error: {result.get('message')}")

    ingest_btn.on_click(on_ingest_clicked)

    # Assemble Left Panel
    left_panel = widgets.VBox([
        doc_header,
        widgets.HBox([file_input, ingest_btn]),
        ingest_output,
        widgets.HTML("<hr>"),
        sources_header,
        sources_box
    ], layout=widgets.Layout(width='40%', padding='10px', border='1px solid #ddd'))

    # ==========================================
    # CHAT INTERFACE PANEL (Right Side)
    # ==========================================
    chat_header = widgets.HTML("<h3>💬 Agent Chat</h3>")
    chat_output = widgets.Output(layout=widgets.Layout(
        height='350px',
        overflow='auto',
        border='1px solid #ccc',
        padding='10px',
        background_color='#f9f9f9'
    ))

    msg_input = widgets.Text(
        placeholder="Ask a question about your documents...",
        layout=widgets.Layout(width='80%')
    )
    send_btn = widgets.Button(description="Send", button_style="success")

    def on_send_clicked(b):
        user_text = msg_input.value.strip()
        if not user_text: return
        msg_input.value = ""

        # Determine which sources the user checked
        active_sources = [sid for sid, cb in checkboxes_dict.items() if cb.value]

        with chat_output:
            print(f"You: {user_text}")
            print(f"[System: Filtering to {len(active_sources)} selected sources...]")

            try:
                response = assistant.chat(user_text, allowed_sources=active_sources)

                print(f"\nAgent: {response}\n")
                print("-" * 50)
            except Exception as e:
                print(f"\n⚠️ UI Caught Error: {str(e)}\n")

    send_btn.on_click(on_send_clicked)
    msg_input.on_submit(lambda x: on_send_clicked(None))  # Allow 'Enter' to send

    # Assemble Right Panel
    right_panel = widgets.VBox([
        chat_header,
        chat_output,
        widgets.HBox([msg_input, send_btn], layout=widgets.Layout(margin='10px 0 0 0'))
    ], layout=widgets.Layout(width='55%', padding='10px'))

    # ==========================================
    # RENDER MAIN LAYOUT
    # ==========================================
    update_sources_ui()  # Initial population
    main_layout = widgets.HBox([left_panel, right_panel],
                               layout=widgets.Layout(width='100%', justify_content='space-between'))
    display(main_layout)


# Execute the UI
# Use previously initialized assistant

# Enter "/content/Agent Systems - Agent Definition and Interaction Modelling.pdf" as File Path for ingest
create_colab_ui(assistant)
