import json
import urllib.request
from typing import List


def get_ollama_models() -> List[str]:
    """
    Fetches the list of available models from the local Ollama instance.
    Returns a list of model names.
    """
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags") as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                return [model["name"] for model in data.get("models", [])]
            return ["llama3.2:3b"]
    except Exception:
        # Fallback to a sensible default if Ollama is not reachable
        return ["llama3.2:3b"]
