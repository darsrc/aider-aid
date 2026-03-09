from pathlib import Path
import json

import pytest
import yaml

from aider_aid import services


def test_create_profile_with_context_and_endpoint(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AIDER_AID_CONFIG_HOME", str(tmp_path))
    result = services.create_profile(
        name="dev",
        model="llama3",
        context_size=12288,
        ollama_api_base="http://127.0.0.1:11434",
    )
    profile = result.profile
    assert profile.name == "dev"
    assert profile.config["model"] == "ollama_chat/llama3"
    assert profile.config["weak-model"] == "ollama_chat/llama3"
    assert profile.config["editor-model"] == "ollama_chat/llama3"
    assert "set-env" in profile.config
    assert "OLLAMA_API_BASE=http://127.0.0.1:11434" in profile.config["set-env"]
    model_settings = Path(profile.config["model-settings-file"])
    payload = yaml.safe_load(model_settings.read_text(encoding="utf-8"))
    assert payload[0]["extra_params"]["num_ctx"] == 12288


def _read_model_metadata(model_metadata_path: Path) -> dict[str, dict[str, object]]:
    return json.loads(model_metadata_path.read_text(encoding="utf-8"))


def test_create_profile_generates_model_metadata_for_unknown_models(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AIDER_AID_CONFIG_HOME", str(tmp_path))
    result = services.create_profile(
        name="dev",
        model="hf.co/Melvin56/Phi-4-mini-instruct-abliterated-GGUF:Q6_K",
        weak_model="MFDoom/deepseek-r1-tool-calling:14b",
        context_size=12288,
    )
    profile = result.profile
    metadata_path = Path(profile.config["model-metadata-file"])
    assert metadata_path.exists()
    payload = _read_model_metadata(metadata_path)
    assert "hf.co/Melvin56/Phi-4-mini-instruct-abliterated-GGUF:Q6_K" in payload
    assert "MFDoom/deepseek-r1-tool-calling:14b" in payload
    assert payload["hf.co/Melvin56/Phi-4-mini-instruct-abliterated-GGUF:Q6_K"]["max_input_tokens"] == 12288
    assert payload["MFDoom/deepseek-r1-tool-calling:14b"]["input_cost_per_token"] == 0.0


def test_edit_profile_updates_model_and_context(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AIDER_AID_CONFIG_HOME", str(tmp_path))
    created = services.create_profile(name="dev", model="llama3", context_size=8192)
    edited = services.edit_profile(
        name=created.profile.name,
        model="llama3.1",
        context_size=16384,
        clear_ollama_api_base=True,
    )
    profile = edited.profile
    assert profile.config["model"] == "ollama_chat/llama3.1"
    assert profile.config["weak-model"] == "ollama_chat/llama3.1"
    assert profile.config["editor-model"] == "ollama_chat/llama3.1"
    assert services.extract_env_var(profile.config, "OLLAMA_API_BASE") is None
    assert services.read_config_context_size(profile.config) == 16384


def test_create_profile_accepts_explicit_role_models(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AIDER_AID_CONFIG_HOME", str(tmp_path))
    result = services.create_profile(
        name="dev",
        model="llama3",
        weak_model="qwen2.5-coder:7b",
        editor_model="openai/gpt-4o-mini",
    )
    profile = result.profile
    assert profile.config["model"] == "ollama_chat/llama3"
    assert profile.config["weak-model"] == "ollama_chat/qwen2.5-coder:7b"
    assert profile.config["editor-model"] == "openai/gpt-4o-mini"


def test_edit_profile_preserves_explicit_role_overrides(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AIDER_AID_CONFIG_HOME", str(tmp_path))
    created = services.create_profile(
        name="dev",
        model="llama3",
        weak_model="mistral",
        editor_model="openai/gpt-4o-mini",
    )
    edited = services.edit_profile(name=created.profile.name, model="llama3.1")
    profile = edited.profile
    assert profile.config["model"] == "ollama_chat/llama3.1"
    assert profile.config["weak-model"] == "ollama_chat/mistral"
    assert profile.config["editor-model"] == "openai/gpt-4o-mini"


def test_edit_profile_can_clear_role_models(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AIDER_AID_CONFIG_HOME", str(tmp_path))
    created = services.create_profile(name="dev", model="llama3")
    edited = services.edit_profile(
        name=created.profile.name,
        clear_weak_model=True,
        clear_editor_model=True,
    )
    profile = edited.profile
    assert "weak-model" not in profile.config
    assert "editor-model" not in profile.config


def test_edit_profile_updates_generated_model_metadata(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AIDER_AID_CONFIG_HOME", str(tmp_path))
    created = services.create_profile(name="dev", model="llama3", context_size=8192)
    edited = services.edit_profile(
        name=created.profile.name,
        weak_model="MFDoom/deepseek-r1-tool-calling:14b",
        context_size=16384,
    )
    profile = edited.profile
    metadata_path = Path(profile.config["model-metadata-file"])
    payload = _read_model_metadata(metadata_path)
    assert payload["ollama_chat/llama3"]["max_input_tokens"] == 16384
    assert payload["MFDoom/deepseek-r1-tool-calling:14b"]["max_output_tokens"] == 4096


def test_remove_profile_deletes_managed_settings(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AIDER_AID_CONFIG_HOME", str(tmp_path))
    created = services.create_profile(name="dev", model="llama3")
    settings_path = Path(created.profile.config["model-settings-file"])
    metadata_path = Path(created.profile.config["model-metadata-file"])
    assert settings_path.exists()
    assert metadata_path.exists()
    removed_path = services.remove_profile("dev")
    assert removed_path.exists() is False
    assert settings_path.exists() is False
    assert metadata_path.exists() is False


def test_launch_for_identifiers_builds_command(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AIDER_AID_CONFIG_HOME", str(tmp_path))
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    services.add_project(path=project_dir, name="repo")
    services.create_profile(name="dev", model="llama3")

    captured: dict[str, object] = {}

    def fake_launch_aider(*, project_path: Path, profile_path: Path, extra_args, dry_run):  # noqa: ANN001
        captured["project_path"] = project_path
        captured["profile_path"] = profile_path
        captured["extra_args"] = list(extra_args or [])
        captured["dry_run"] = dry_run
        return ["aider", "--config", str(profile_path)], 0

    monkeypatch.setattr("aider_aid.services.launch_aider", fake_launch_aider)
    result = services.launch_for_identifiers(
        project_identifier="repo",
        profile_name="dev",
        extra_args=["--yes-always"],
        dry_run=True,
    )
    assert result.returncode == 0
    assert captured["project_path"] == project_dir
    assert captured["extra_args"] == ["--yes-always"]
    assert captured["dry_run"] is True
    assert "--config" in result.command_display


def test_launch_for_identifiers_injects_runtime_metadata_for_legacy_profile(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AIDER_AID_CONFIG_HOME", str(tmp_path))
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    services.add_project(path=project_dir, name="repo")
    store = services.profile_store()
    store.ensure_dirs()
    profile_path = store.profiles_dir / "dev.aider.conf.yml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "model": "hf.co/Melvin56/Phi-4-mini-instruct-abliterated-GGUF:Q6_K",
                "weak-model": "MFDoom/deepseek-r1-tool-calling:14b",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_launch_aider(*, project_path: Path, profile_path: Path, extra_args, dry_run):  # noqa: ANN001
        config = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        metadata_path = Path(config["model-metadata-file"])
        captured["project_path"] = project_path
        captured["metadata_path"] = metadata_path
        captured["metadata_exists_during_launch"] = metadata_path.exists()
        captured["metadata_payload"] = json.loads(metadata_path.read_text(encoding="utf-8"))
        return ["aider", "--config", str(profile_path)], 0

    monkeypatch.setattr("aider_aid.services.launch_aider", fake_launch_aider)

    result = services.launch_for_identifiers(
        project_identifier="repo",
        profile_name="dev",
        dry_run=True,
    )
    assert result.returncode == 0
    assert captured["project_path"] == project_dir
    assert captured["metadata_exists_during_launch"] is True
    assert "hf.co/Melvin56/Phi-4-mini-instruct-abliterated-GGUF:Q6_K" in captured["metadata_payload"]
    assert Path(captured["metadata_path"]).exists() is False


def test_fetch_models_from_endpoint_requires_models(monkeypatch):
    monkeypatch.setattr(
        "aider_aid.services.probe_ollama_endpoint",
        lambda endpoint, api_key=None: (True, [], ""),
    )
    with pytest.raises(ValueError):
        services.fetch_models_from_endpoint("http://127.0.0.1:11434")
