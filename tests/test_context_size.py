import json
from pathlib import Path

import pytest
import yaml

from aider_aid.cli import app
from aider_aid.profile_store import ProfileStore
from aider_aid.shell import CommandResult

CliRunner = pytest.importorskip("typer.testing").CliRunner


def _store_without_aider_validation(tmp_path: Path) -> ProfileStore:
    return ProfileStore(config_root=tmp_path, command_exists_fn=lambda _: False)


def _store_with_failing_validation(tmp_path: Path) -> ProfileStore:
    return ProfileStore(
        config_root=tmp_path,
        run=lambda cmd, **kwargs: CommandResult(cmd=list(cmd), returncode=1, stdout="", stderr="invalid"),
        command_exists_fn=lambda _: True,
    )


def _read_num_ctx(model_settings_path: Path) -> int:
    payload = yaml.safe_load(model_settings_path.read_text(encoding="utf-8"))
    return int(payload[0]["extra_params"]["num_ctx"])


def test_config_create_sets_default_context_size(monkeypatch, tmp_path: Path):
    runner = CliRunner()
    store = _store_without_aider_validation(tmp_path)
    monkeypatch.setattr("aider_aid.cli._profile_store", lambda: store)

    result = runner.invoke(app, ["config", "create", "dev", "--model", "llama3"])

    assert result.exit_code == 0
    profile = store.get_profile("dev")
    assert profile.config["model"] == "ollama_chat/llama3"
    assert profile.config["weak-model"] == "ollama_chat/llama3"
    assert profile.config["editor-model"] == "ollama_chat/llama3"
    model_settings_path = Path(profile.config["model-settings-file"])
    assert model_settings_path.exists()
    assert _read_num_ctx(model_settings_path) == 8192


def test_config_create_accepts_custom_context_size(monkeypatch, tmp_path: Path):
    runner = CliRunner()
    store = _store_without_aider_validation(tmp_path)
    monkeypatch.setattr("aider_aid.cli._profile_store", lambda: store)

    result = runner.invoke(
        app,
        ["config", "create", "dev", "--model", "llama3", "--context-size", "16384"],
    )

    assert result.exit_code == 0
    profile = store.get_profile("dev")
    model_settings_path = Path(profile.config["model-settings-file"])
    assert _read_num_ctx(model_settings_path) == 16384


def test_config_edit_updates_context_size(monkeypatch, tmp_path: Path):
    runner = CliRunner()
    store = _store_without_aider_validation(tmp_path)
    monkeypatch.setattr("aider_aid.cli._profile_store", lambda: store)

    created = runner.invoke(app, ["config", "create", "dev", "--model", "llama3"])
    assert created.exit_code == 0

    updated = runner.invoke(app, ["config", "edit", "dev", "--context-size", "24576"])
    assert updated.exit_code == 0

    profile = store.get_profile("dev")
    model_settings_path = Path(profile.config["model-settings-file"])
    assert _read_num_ctx(model_settings_path) == 24576


def test_launch_injects_default_context_size_for_legacy_profile(monkeypatch, tmp_path: Path):
    runner = CliRunner()
    monkeypatch.setenv("AIDER_AID_CONFIG_HOME", str(tmp_path))

    profiles_dir = tmp_path / "configs"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profiles_dir / "dev.aider.conf.yml"
    profile_path.write_text(
        yaml.safe_dump({"model": "ollama_chat/llama3"}, sort_keys=False),
        encoding="utf-8",
    )

    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    projects_file = tmp_path / "projects.config"
    projects_file.write_text(
        json.dumps(
            {
                "version": 1,
                "projects": [{"name": "repo", "path": str(project_dir)}],
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_launch_aider(*, project_path: Path, profile_path: Path, extra_args, dry_run):  # noqa: ANN001
        config = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        model_settings_path = Path(config["model-settings-file"])
        captured["config"] = config
        captured["settings_path"] = model_settings_path
        captured["settings_exists_during_launch"] = model_settings_path.exists()
        captured["num_ctx"] = _read_num_ctx(model_settings_path)
        return ["aider", "--config", str(profile_path)], 0

    monkeypatch.setattr("aider_aid.cli.launch_aider", fake_launch_aider)

    result = runner.invoke(app, ["launch", "--project", "repo", "--profile", "dev", "--dry-run"])

    assert result.exit_code == 0
    assert captured["settings_exists_during_launch"] is True
    assert captured["num_ctx"] == 8192
    assert Path(captured["settings_path"]).exists() is False
    assert "model-settings-file" not in yaml.safe_load(profile_path.read_text(encoding="utf-8"))


def test_config_create_rolls_back_model_settings_when_validation_fails(monkeypatch, tmp_path: Path):
    runner = CliRunner()
    store = _store_with_failing_validation(tmp_path)
    monkeypatch.setattr("aider_aid.cli._profile_store", lambda: store)

    result = runner.invoke(app, ["config", "create", "dev", "--model", "llama3"])

    assert result.exit_code == 1
    assert not (tmp_path / "configs" / "dev.aider.model.settings.yml").exists()


def test_config_edit_rolls_back_model_settings_when_validation_fails(monkeypatch, tmp_path: Path):
    runner = CliRunner()
    good_store = _store_without_aider_validation(tmp_path)
    monkeypatch.setattr("aider_aid.cli._profile_store", lambda: good_store)
    created = runner.invoke(app, ["config", "create", "dev", "--model", "llama3", "--context-size", "8192"])
    assert created.exit_code == 0
    model_settings_path = tmp_path / "configs" / "dev.aider.model.settings.yml"
    assert _read_num_ctx(model_settings_path) == 8192

    failing_store = _store_with_failing_validation(tmp_path)
    monkeypatch.setattr("aider_aid.cli._profile_store", lambda: failing_store)
    result = runner.invoke(app, ["config", "edit", "dev", "--context-size", "16384"])

    assert result.exit_code == 1
    assert _read_num_ctx(model_settings_path) == 8192


def test_config_create_applies_qol_preset_and_overrides(monkeypatch, tmp_path: Path):
    runner = CliRunner()
    store = _store_without_aider_validation(tmp_path)
    monkeypatch.setattr("aider_aid.cli._profile_store", lambda: store)

    result = runner.invoke(
        app,
        [
            "config",
            "create",
            "dev",
            "--model",
            "llama3",
            "--qol-preset",
            "strict-ci",
            "--auto-commits",
            "--test-cmd",
            "pytest -q",
        ],
    )

    assert result.exit_code == 0
    profile = store.get_profile("dev")
    assert profile.config["auto-commits"] is True
    assert profile.config["auto-test"] is True
    assert profile.config["test-cmd"] == "pytest -q"


def test_config_edit_auto_syncs_inherited_role_models(monkeypatch, tmp_path: Path):
    runner = CliRunner()
    store = _store_without_aider_validation(tmp_path)
    monkeypatch.setattr("aider_aid.cli._profile_store", lambda: store)

    created = runner.invoke(app, ["config", "create", "dev", "--model", "llama3"])
    assert created.exit_code == 0
    updated = runner.invoke(app, ["config", "edit", "dev", "--model", "llama3.1"])
    assert updated.exit_code == 0

    profile = store.get_profile("dev")
    assert profile.config["model"] == "ollama_chat/llama3.1"
    assert profile.config["weak-model"] == "ollama_chat/llama3.1"
    assert profile.config["editor-model"] == "ollama_chat/llama3.1"


def test_config_edit_preserves_explicit_role_models_and_allows_clear(monkeypatch, tmp_path: Path):
    runner = CliRunner()
    store = _store_without_aider_validation(tmp_path)
    monkeypatch.setattr("aider_aid.cli._profile_store", lambda: store)

    created = runner.invoke(
        app,
        [
            "config",
            "create",
            "dev",
            "--model",
            "llama3",
            "--weak-model",
            "mistral",
            "--editor-model",
            "openai/gpt-4o-mini",
        ],
    )
    assert created.exit_code == 0

    updated = runner.invoke(
        app,
        [
            "config",
            "edit",
            "dev",
            "--model",
            "llama3.1",
            "--clear-editor-model",
        ],
    )
    assert updated.exit_code == 0

    profile = store.get_profile("dev")
    assert profile.config["model"] == "ollama_chat/llama3.1"
    assert profile.config["weak-model"] == "ollama_chat/mistral"
    assert "editor-model" not in profile.config


def test_config_create_generates_model_metadata_for_unknown_models(monkeypatch, tmp_path: Path):
    runner = CliRunner()
    store = _store_without_aider_validation(tmp_path)
    monkeypatch.setattr("aider_aid.cli._profile_store", lambda: store)

    result = runner.invoke(
        app,
        [
            "config",
            "create",
            "dev",
            "--model",
            "hf.co/Melvin56/Phi-4-mini-instruct-abliterated-GGUF:Q6_K",
            "--weak-model",
            "MFDoom/deepseek-r1-tool-calling:14b",
            "--context-size",
            "12288",
        ],
    )

    assert result.exit_code == 0
    profile = store.get_profile("dev")
    metadata_path = Path(profile.config["model-metadata-file"])
    assert metadata_path.exists()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["hf.co/Melvin56/Phi-4-mini-instruct-abliterated-GGUF:Q6_K"]["max_input_tokens"] == 12288
    assert payload["MFDoom/deepseek-r1-tool-calling:14b"]["input_cost_per_token"] == 0.0


def test_config_edit_updates_model_metadata(monkeypatch, tmp_path: Path):
    runner = CliRunner()
    store = _store_without_aider_validation(tmp_path)
    monkeypatch.setattr("aider_aid.cli._profile_store", lambda: store)
    created = runner.invoke(app, ["config", "create", "dev", "--model", "llama3"])
    assert created.exit_code == 0

    updated = runner.invoke(
        app,
        ["config", "edit", "dev", "--weak-model", "MFDoom/deepseek-r1-tool-calling:14b", "--context-size", "16384"],
    )
    assert updated.exit_code == 0
    profile = store.get_profile("dev")
    metadata_path = Path(profile.config["model-metadata-file"])
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["ollama_chat/llama3"]["max_input_tokens"] == 16384
    assert payload["MFDoom/deepseek-r1-tool-calling:14b"]["max_output_tokens"] == 4096


def test_config_remove_deletes_managed_model_settings(monkeypatch, tmp_path: Path):
    runner = CliRunner()
    store = _store_without_aider_validation(tmp_path)
    monkeypatch.setattr("aider_aid.cli._profile_store", lambda: store)
    created = runner.invoke(app, ["config", "create", "dev", "--model", "llama3"])
    assert created.exit_code == 0

    model_settings_path = tmp_path / "configs" / "dev.aider.model.settings.yml"
    model_metadata_path = tmp_path / "configs" / "dev.aider.model.metadata.json"
    assert model_settings_path.exists()
    assert model_metadata_path.exists()

    removed = runner.invoke(app, ["config", "remove", "dev", "--yes"])
    assert removed.exit_code == 0
    assert not model_settings_path.exists()
    assert not model_metadata_path.exists()
