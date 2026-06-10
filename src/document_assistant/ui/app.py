import sys
import os
import logging


def run():
    """
    Unified entry point for the Document Assistant.
    Usage:
        poetry run python src/document_assistant/ui/app.py [ui|cli]
    """
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "ui"

    # Ensure we are in the project root so paths match
    # Get the directory of this script and move up to 'src' then project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../.."))
    os.chdir(project_root)

    if mode == "cli":
        print("🚀 Launching CLI Mode...")
        from document_assistant.ui.cli import main
        main()

    elif mode == "ui":
        print("🎨 Launching Streamlit UI...")
        target_file = os.path.join("src", "document_assistant", "ui", "streamlit_ui.py")

        from streamlit.web import cli as stcli

        logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

        # Overwrite sys.argv so Streamlit thinks it was called from the command line
        sys.argv = ["streamlit", "run", target_file]
        sys.exit(stcli.main())

    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: ui, cli")


if __name__ == "__main__":
    run()
