from aider_aid.model_discovery import (
    normalize_ollama_model,
    parse_aider_list_models_output,
    parse_ollama_list_output,
)


def test_normalize_ollama_model():
    assert normalize_ollama_model("llama3") == "ollama_chat/llama3"
    assert normalize_ollama_model("ollama/llama3") == "ollama_chat/llama3"
    assert normalize_ollama_model("ollama_chat/llama3") == "ollama_chat/llama3"
    assert normalize_ollama_model("openai/gpt-4o") == "openai/gpt-4o"


def test_parse_ollama_list_output():
    output = """NAME            ID      SIZE   MODIFIED
llama3:8b       abcd    4 GB   now
qwen2.5-coder   efgh    8 GB   now
"""
    assert parse_ollama_list_output(output) == ["llama3:8b", "qwen2.5-coder"]


def test_parse_aider_list_models_output():
    output = """Models which match "ollama":
- ollama/llama3
- ollama/llama3.1
"""
    assert parse_aider_list_models_output(output) == [
        "ollama_chat/llama3",
        "ollama_chat/llama3.1",
    ]
