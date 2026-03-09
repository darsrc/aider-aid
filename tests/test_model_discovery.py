from aider_aid.model_discovery import (
    discover_ollama_models,
    normalize_ollama_model,
    parse_ollama_list_output,
)
from aider_aid.shell import CommandResult


def test_normalize_ollama_model():
    assert normalize_ollama_model("llama3") == "ollama_chat/llama3"
    assert normalize_ollama_model("ollama/llama3") == "ollama_chat/llama3"
    assert normalize_ollama_model("ollama_chat/llama3") == "ollama_chat/llama3"
    assert normalize_ollama_model("openai/gpt-4o") == "openai/gpt-4o"
    assert normalize_ollama_model("hf.co/owner/model:Q4_K_M") == "ollama_chat/hf.co/owner/model:Q4_K_M"
    assert normalize_ollama_model("huihui_ai/deepseek-r1-abliterated:14b-qwen-distill") == (
        "ollama_chat/huihui_ai/deepseek-r1-abliterated:14b-qwen-distill"
    )


def test_parse_ollama_list_output():
    output = """NAME            ID      SIZE   MODIFIED
llama3:8b       abcd    4 GB   now
qwen2.5-coder   efgh    8 GB   now
"""
    assert parse_ollama_list_output(output) == ["llama3:8b", "qwen2.5-coder"]


def test_discover_ollama_models_uses_ollama_only():
    def runner(cmd, **kwargs):  # noqa: ANN001
        if cmd == ["ollama", "list"]:
            return CommandResult(
                cmd=list(cmd),
                returncode=0,
                stdout="NAME ID SIZE MODIFIED\nllama3 abc 4GB now\n",
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {cmd}")

    discovered = discover_ollama_models(
        run=runner,
        command_exists_fn=lambda name: name == "ollama",
    )
    assert discovered.installed_models == ["ollama_chat/llama3"]
    assert discovered.combined_models == ["ollama_chat/llama3"]
    assert discovered.warnings == []
