from pathlib import Path

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


def test_remove_profile_deletes_managed_settings(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AIDER_AID_CONFIG_HOME", str(tmp_path))
    created = services.create_profile(name="dev", model="llama3")
    settings_path = Path(created.profile.config["model-settings-file"])
    assert settings_path.exists()
    removed_path = services.remove_profile("dev")
    assert removed_path.exists() is False
    assert settings_path.exists() is False


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


def test_fetch_models_from_endpoint_requires_models(monkeypatch):
    monkeypatch.setattr(
        "aider_aid.services.probe_ollama_endpoint",
        lambda endpoint, api_key=None: (True, [], ""),
    )
    with pytest.raises(ValueError):
        services.fetch_models_from_endpoint("http://127.0.0.1:11434")
