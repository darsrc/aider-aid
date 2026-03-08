from pathlib import Path

import pytest

from aider_aid.ollama_server_store import OllamaServerError, OllamaServerStore


def test_add_list_remove_servers(tmp_path: Path):
    store = OllamaServerStore(config_root=tmp_path)
    local = store.add_server(name="local", url="http://127.0.0.1:11434")
    gpu = store.add_server(name="gpu", url="http://gpu-host:11434")
    listed = store.list_servers()

    assert [item.name for item in listed] == ["gpu", "local"]
    assert local.url == "http://127.0.0.1:11434"
    assert gpu.url == "http://gpu-host:11434"

    removed = store.remove_server("local")
    assert removed.name == "local"
    assert [item.name for item in store.list_servers()] == ["gpu"]


def test_add_server_rejects_invalid_url(tmp_path: Path):
    store = OllamaServerStore(config_root=tmp_path)
    with pytest.raises(OllamaServerError):
        store.add_server(name="bad", url="not-a-url")
