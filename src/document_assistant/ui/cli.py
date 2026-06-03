import logging
import os
import sys
import warnings

from document_assistant.core.vector_store import VectorStoreManager
from document_assistant.core.repository import SessionRepository
from document_assistant.core.ingestor import DocumentIngestor
from document_assistant.core.tools.registry import ToolRegistry
from document_assistant.orchestration.manager import SessionManager
from document_assistant.orchestration.single_agent import SingleAgentOrchestrator
from document_assistant.api.assistant import Assistant


def build_assistant() -> Assistant:
    """Composition Root: Wires up all injected dependencies."""
    vector_store = VectorStoreManager(persist_directory="./data/chroma_db")
    repository = SessionRepository(storage_dir="./data/sessions")
    ingestor = DocumentIngestor(vector_store=vector_store)
    session_manager = SessionManager(ingestor=ingestor, repository=repository)
    tool_registry = ToolRegistry(vector_store=vector_store, session_manager=session_manager)
    orchestrator = SingleAgentOrchestrator(
        session_manager=session_manager,
        tool_registry=tool_registry,
        model="llama3.2:3b",
        use_local=True  # Set to False if using OpenAI keys
    )
    return Assistant(session_manager=session_manager, orchestrator=orchestrator)


def main():
    # Mute specific Python warnings (like LangChain deprecations)
    warnings.filterwarnings("ignore", category=UserWarning)
    # If the import complains, you can use the generic DeprecationWarning category
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Hugging Face and Tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Suppress underlying logging noise from external libraries
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("chromadb").setLevel(logging.ERROR)

    print("🤖 Booting up Local RAG Assistant...")
    try:
        assistant = build_assistant()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        sys.exit(1)

    session_id = input("Enter a session name (or press Enter for 'default'): ").strip() or "default"
    assistant.set_session(session_id)
    print(f"✅ Active Session: {session_id}\n")

    print("Commands:")
    print("  /upload <filepath>  - Ingest a document")
    print("  /status             - View active documents")
    print("  /models             - List available local models")
    print("  /switch <model>     - Switch the active model")
    print("  /quit               - Exit the CLI")
    print("-" * 50)

    while True:
        try:
            user_input = input("\n> You: ").strip()

            if not user_input:
                continue
            if user_input.lower() in ['/quit', '/exit']:
                print("Goodbye!")
                break

            # Handle slash commands
            if user_input.startswith("/upload "):
                filepath = user_input.split(" ", 1)[1]
                print(f"⚙️ Uploading {filepath}...")
                result = assistant.upload_document(filepath)
                if result.get("status") == "success":
                    print(f"✅ Success! Source ID: {result.get('source_id')}")
                else:
                    print(f"❌ Error: {result.get('message')}")
                continue

            if user_input == "/status":
                status = assistant.get_status()
                print("\n📊 Session Status:")
                for doc in status.get('documents', []):
                    print(f"  - {doc['file_name']} ({doc.get('total_chunks')} chunks)")
                continue

            if user_input == "/models":
                models = assistant.get_available_models()
                print("\n🤖 Available Local Models:")
                for m in models:
                    print(f"  - {m}")
                continue

            if user_input.startswith("/switch "):
                model_name = user_input.split(" ", 1)[1]
                print(f"⚙️ Switching to {model_name}...")
                result = assistant.change_model(model_name)
                print(f"✅ {result}")
                continue

            # Standard chat routing
            print("🤖 Agent thinking...")
            response = assistant.chat(user_input)
            print(f"💬 Agent: {response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n⚠️ Unexpected Error: {str(e)}")


if __name__ == "__main__":
    main()
