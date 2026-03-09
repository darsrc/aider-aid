import pytest
import json
import yaml
from pathlib import Path

CliRunner = pytest.importorskip("typer.testing").CliRunner

import aider_aid.cli as cli
from aider_aid.cli import app
from aider_aid.profile_store import ProfileStore


def _write_project_and_profile(
    tmp_path: Path,
    *,
    project_name: str = "repo",
    profile_name: str = "gpu-all",
    model: str = "ollama_chat/llama3",
) -> tuple[Path, Path]:
    profiles_dir = tmp_path / "configs"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profiles_dir / f"{profile_name}.aider.conf.yml"
    profile_path.write_text(
        yaml.safe_dump({"model": model}, sort_keys=False),
        encoding="utf-8",
    )

    project_dir = tmp_path / project_name
    project_dir.mkdir()
    projects_file = tmp_path / "projects.config"
    projects_file.write_text(
        json.dumps(
            {
                "version": 1,
                "projects": [{"name": project_name, "path": str(project_dir)}],
            }
        ),
        encoding="utf-8",
    )
    return project_dir, profile_path


def test_no_args_non_tty_shows_help(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("aider_aid.cli._stdin_is_tty", lambda: False)
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_no_args_tty_invokes_main_menu(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setenv("AIDER_AID_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr("aider_aid.cli._stdin_is_tty", lambda: True)
    called = {"menu": False}

    def fake_main_menu():
        called["menu"] = True

    monkeypatch.setattr("aider_aid.cli._run_textual_ui", lambda: 1)
    monkeypatch.setattr("aider_aid.cli._interactive_main_menu", fake_main_menu)
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert called["menu"] is True


def test_no_args_tty_auto_uses_textual_when_available(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setenv("AIDER_AID_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr("aider_aid.cli._stdin_is_tty", lambda: True)
    monkeypatch.setattr("aider_aid.cli._run_textual_ui", lambda: 0)

    def fail_main_menu():
        raise AssertionError("classic menu should not run when textual auto mode succeeds")

    monkeypatch.setattr("aider_aid.cli._interactive_main_menu", fail_main_menu)
    result = runner.invoke(app, [])
    assert result.exit_code == 0


def test_no_args_tty_textual_mode_propagates_failure(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setenv("AIDER_AID_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr("aider_aid.cli._stdin_is_tty", lambda: True)
    monkeypatch.setattr("aider_aid.cli._run_textual_ui", lambda: 2)
    result = runner.invoke(app, ["--ui-mode", "textual"])
    assert result.exit_code == 2


def test_subcommand_still_works_with_tty_enabled(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setenv("AIDER_AID_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr("aider_aid.cli._stdin_is_tty", lambda: True)
    result = runner.invoke(app, ["project", "list"])
    assert result.exit_code == 0
    assert "No projects found." in result.output


def test_menu_command_rejects_non_tty(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("aider_aid.cli._stdin_is_tty", lambda: False)
    result = runner.invoke(app, ["menu", "config"])
    assert result.exit_code == 1
    assert "Interactive menus require a TTY." in result.output


def test_menu_command_opens_selected_section(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("aider_aid.cli._stdin_is_tty", lambda: True)
    called = {"config": False}
    monkeypatch.setattr("aider_aid.cli._show_banner", lambda: None)
    monkeypatch.setattr("aider_aid.cli._interactive_config_menu", lambda: called.update({"config": True}))
    result = runner.invoke(app, ["menu", "config"])
    assert result.exit_code == 0
    assert called["config"] is True


def test_interactive_config_create_opens_profile_flow(monkeypatch):
    sequence = iter([1, 5])  # Create profile, then Back
    monkeypatch.setattr("aider_aid.cli._prompt_menu_choice", lambda title, options: next(sequence))
    called = {"flow": False}
    monkeypatch.setattr(
        "aider_aid.cli._interactive_create_profile_flow",
        lambda: called.update({"flow": True}),
    )
    cli._interactive_config_menu()
    assert called["flow"] is True


def test_interactive_config_edit_lists_profiles(monkeypatch, tmp_path):
    store = ProfileStore(config_root=tmp_path, command_exists_fn=lambda _: False)
    store.save_profile("gpu-all", {"model": "ollama_chat/llama3"})
    sequence = iter([3, 5])  # Edit profile, then Back
    captured: dict[str, object] = {}

    def fake_menu_choice(title, options):
        if title == "Config Profiles:":
            return next(sequence)
        if title == "Choose profile to edit:":
            assert any("gpu-all" in option for option in options)
            return 0
        raise AssertionError(f"unexpected menu title: {title}")

    def fake_edit_flow(name: str):
        captured["name"] = name

    monkeypatch.setattr("aider_aid.cli._profile_store", lambda: store)
    monkeypatch.setattr("aider_aid.cli._prompt_menu_choice", fake_menu_choice)
    monkeypatch.setattr("aider_aid.cli._interactive_edit_profile_flow", fake_edit_flow)

    cli._interactive_config_menu()
    assert captured["name"] == "gpu-all"


def test_run_config_edit_command_uses_plain_defaults(monkeypatch):
    captured: dict[str, object] = {}

    def fake_config_edit(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    monkeypatch.setattr("aider_aid.cli.config_edit", fake_config_edit)
    cli._run_config_edit_command(
        name="gpu-all",
        new_name=None,
        model=None,
        weak_model=None,
        clear_weak_model=False,
        editor_model=None,
        clear_editor_model=False,
        context_size=None,
        qol_preset=None,
        server=None,
        clear_server=False,
        ollama_api_base=None,
        clear_ollama_api_base=False,
        ollama_api_key=None,
        clear_ollama_api_key=False,
        set_env=[],
        option=[],
    )
    assert captured["name"] == "gpu-all"
    assert captured["context_size"] is None
    assert captured["interactive"] is False


def test_interactive_config_show_lists_profiles(monkeypatch, tmp_path):
    store = ProfileStore(config_root=tmp_path, command_exists_fn=lambda _: False)
    store.save_profile("gpu-all", {"model": "ollama_chat/llama3"})
    sequence = iter([2, 5])  # Show profile, then Back
    captured: dict[str, object] = {}

    def fail_prompt(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("Profile name text prompt should not be used for show")

    def fake_menu_choice(title, options):
        if title == "Config Profiles:":
            return next(sequence)
        if title == "Choose profile to show:":
            assert any("gpu-all" in option for option in options)
            return 0
        raise AssertionError(f"unexpected menu title: {title}")

    monkeypatch.setattr("aider_aid.cli._profile_store", lambda: store)
    monkeypatch.setattr("aider_aid.cli._prompt_menu_choice", fake_menu_choice)
    monkeypatch.setattr("aider_aid.cli._prompt_text", fail_prompt)
    monkeypatch.setattr("aider_aid.cli.config_show", lambda name: captured.update({"name": name}))

    cli._interactive_config_menu()
    assert captured["name"] == "gpu-all"


def test_interactive_project_remove_lists_projects(monkeypatch, tmp_path):
    monkeypatch.setenv("AIDER_AID_CONFIG_HOME", str(tmp_path))
    store = cli._project_store()
    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    store.add_project(name="repo", path=project_dir)
    sequence = iter([3, 4])  # Remove project, then Back
    captured: dict[str, object] = {}

    def fake_menu_choice(title, options):
        if title == "Projects:":
            return next(sequence)
        if title == "Choose project to remove:":
            assert any("repo" in option for option in options)
            return 0
        raise AssertionError(f"unexpected menu title: {title}")

    monkeypatch.setattr("aider_aid.cli._project_store", lambda: store)
    monkeypatch.setattr("aider_aid.cli._prompt_menu_choice", fake_menu_choice)
    monkeypatch.setattr("aider_aid.cli.project_remove", lambda identifier, yes=False: captured.update({"identifier": identifier}))
    cli._interactive_project_menu()
    assert captured["identifier"] == "repo"


def test_interactive_server_remove_lists_servers(monkeypatch, tmp_path):
    monkeypatch.setenv("AIDER_AID_CONFIG_HOME", str(tmp_path))
    store = cli._server_store()
    store.add_server(name="gpu-a", url="http://10.0.0.64:11436")
    sequence = iter([2, 3])  # Remove server, then Back
    captured: dict[str, object] = {}

    def fake_menu_choice(title, options):
        if title == "Ollama Servers:":
            return next(sequence)
        if title == "Choose server to remove:":
            assert any("gpu-a" in option for option in options)
            return 0
        raise AssertionError(f"unexpected menu title: {title}")

    monkeypatch.setattr("aider_aid.cli._server_store", lambda: store)
    monkeypatch.setattr("aider_aid.cli._prompt_menu_choice", fake_menu_choice)
    monkeypatch.setattr("aider_aid.cli.server_remove", lambda name, yes=False: captured.update({"name": name}))
    cli._interactive_server_menu()
    assert captured["name"] == "gpu-a"


def test_launch_sanitizes_legacy_profile_metadata(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setenv("AIDER_AID_CONFIG_HOME", str(tmp_path))

    profiles_dir = tmp_path / "configs"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profiles_dir / "gpu-all.aider.conf.yml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "name": "gpu-all",
                "aider-aid-ollama-server": "gpu-a",
                "model": "ollama_chat/llama3",
                "set-env": ["OLLAMA_API_BASE=http://10.0.0.64:11436"],
            },
            sort_keys=False,
        ),
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
        captured["project_path"] = project_path
        captured["config"] = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        return ["aider", "--config", str(profile_path)], 0

    monkeypatch.setattr("aider_aid.cli.launch_aider", fake_launch_aider)
    result = runner.invoke(
        app,
        ["launch", "--project", "repo", "--profile", "gpu-all", "--dry-run"],
    )
    assert result.exit_code == 0
    assert "name" not in captured["config"]
    assert "aider-aid-ollama-server" not in captured["config"]


def test_launch_maps_legacy_server_binding_to_ollama_api_base(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setenv("AIDER_AID_CONFIG_HOME", str(tmp_path))

    profiles_dir = tmp_path / "configs"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profiles_dir / "gpu-all.aider.conf.yml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "name": "gpu-all",
                "aider-aid-ollama-server": "gpu-a",
                "model": "ollama_chat/llama3",
            },
            sort_keys=False,
        ),
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

    servers_file = tmp_path / "ollama_servers.config"
    servers_file.write_text(
        json.dumps(
            {
                "version": 1,
                "servers": [{"name": "gpu-a", "url": "http://10.0.0.64:11436"}],
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_launch_aider(*, project_path: Path, profile_path: Path, extra_args, dry_run):  # noqa: ANN001
        captured["config"] = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        return ["aider", "--config", str(profile_path)], 0

    monkeypatch.setattr("aider_aid.cli.launch_aider", fake_launch_aider)
    result = runner.invoke(
        app,
        ["launch", "--project", "repo", "--profile", "gpu-all", "--dry-run"],
    )
    assert result.exit_code == 0
    assert captured["config"]["set-env"] == ["OLLAMA_API_BASE=http://10.0.0.64:11436"]


def test_launch_interactive_gate_no_skips_extra_arg_prompt(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setenv("AIDER_AID_CONFIG_HOME", str(tmp_path))
    _write_project_and_profile(tmp_path)
    monkeypatch.setattr("aider_aid.cli._stdin_is_tty", lambda: True)
    monkeypatch.setattr("aider_aid.cli.ask_confirm", lambda label, default=False: False)

    def fail_prompt(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("extra args text prompt should not be called when gate is No")

    monkeypatch.setattr("aider_aid.cli._prompt_text", fail_prompt)

    captured: dict[str, object] = {}

    def fake_launch_aider(*, project_path: Path, profile_path: Path, extra_args, dry_run):  # noqa: ANN001
        captured["extra_args"] = list(extra_args)
        return ["aider", "--config", str(profile_path)], 0

    monkeypatch.setattr("aider_aid.cli.launch_aider", fake_launch_aider)
    result = runner.invoke(
        app,
        ["launch", "--project", "repo", "--profile", "gpu-all", "--dry-run"],
    )

    assert result.exit_code == 0
    assert captured["extra_args"] == []


def test_launch_interactive_gate_yes_parses_extra_args(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setenv("AIDER_AID_CONFIG_HOME", str(tmp_path))
    _write_project_and_profile(tmp_path)
    monkeypatch.setattr("aider_aid.cli._stdin_is_tty", lambda: True)
    monkeypatch.setattr("aider_aid.cli.ask_confirm", lambda label, default=False: True)
    monkeypatch.setattr(
        "aider_aid.cli._prompt_text",
        lambda label, default=None, allow_empty=False: "--no-show-model-warnings --yes-always",
    )

    captured: dict[str, object] = {}

    def fake_launch_aider(*, project_path: Path, profile_path: Path, extra_args, dry_run):  # noqa: ANN001
        captured["extra_args"] = list(extra_args)
        return ["aider", "--config", str(profile_path)], 0

    monkeypatch.setattr("aider_aid.cli.launch_aider", fake_launch_aider)
    result = runner.invoke(
        app,
        ["launch", "--project", "repo", "--profile", "gpu-all", "--dry-run"],
    )

    assert result.exit_code == 0
    assert captured["extra_args"] == ["--no-show-model-warnings", "--yes-always"]


def test_launch_interactive_cancel_at_extra_args_prompt_aborts(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setenv("AIDER_AID_CONFIG_HOME", str(tmp_path))
    _write_project_and_profile(tmp_path)
    monkeypatch.setattr("aider_aid.cli._stdin_is_tty", lambda: True)
    monkeypatch.setattr("aider_aid.cli.ask_confirm", lambda label, default=False: True)
    monkeypatch.setattr("aider_aid.cli._prompt_text", lambda label, default=None, allow_empty=False: None)

    called = {"launch": False}

    def fake_launch_aider(*, project_path: Path, profile_path: Path, extra_args, dry_run):  # noqa: ANN001
        called["launch"] = True
        return ["aider", "--config", str(profile_path)], 0

    monkeypatch.setattr("aider_aid.cli.launch_aider", fake_launch_aider)
    result = runner.invoke(
        app,
        ["launch", "--project", "repo", "--profile", "gpu-all", "--dry-run"],
    )

    assert result.exit_code == 0
    assert called["launch"] is False


def test_launch_interactive_invalid_extra_args_does_not_launch(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setenv("AIDER_AID_CONFIG_HOME", str(tmp_path))
    _write_project_and_profile(tmp_path)
    monkeypatch.setattr("aider_aid.cli._stdin_is_tty", lambda: True)
    monkeypatch.setattr("aider_aid.cli.ask_confirm", lambda label, default=False: True)
    monkeypatch.setattr(
        "aider_aid.cli._prompt_text",
        lambda label, default=None, allow_empty=False: '--message "unterminated',
    )

    called = {"launch": False}

    def fake_launch_aider(*, project_path: Path, profile_path: Path, extra_args, dry_run):  # noqa: ANN001
        called["launch"] = True
        return ["aider", "--config", str(profile_path)], 0

    monkeypatch.setattr("aider_aid.cli.launch_aider", fake_launch_aider)
    result = runner.invoke(
        app,
        ["launch", "--project", "repo", "--profile", "gpu-all", "--dry-run"],
    )

    assert result.exit_code == 1
    assert called["launch"] is False
    assert "Invalid one-off extra aider args:" in result.output


def test_launch_non_tty_requires_explicit_project_and_profile(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setenv("AIDER_AID_CONFIG_HOME", str(tmp_path))
    _write_project_and_profile(tmp_path)
    monkeypatch.setattr("aider_aid.cli._stdin_is_tty", lambda: False)
    result = runner.invoke(app, ["launch"])
    assert result.exit_code == 1
    assert "Non-interactive launch requires --project" in result.output
