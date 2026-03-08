from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Any

import typer
import yaml

from aider_aid.doctor import DoctorResult, run_doctor
from aider_aid.launcher import format_shell_command, launch_aider
from aider_aid.model_discovery import discover_ollama_models, normalize_ollama_model
from aider_aid.ollama_server_store import (
    AIDER_AID_OLLAMA_SERVER_KEY,
    OllamaServerError,
    OllamaServerNotFoundError,
    OllamaServerStore,
)
from aider_aid.profile_store import (
    ProfileError,
    ProfileNotFoundError,
    ProfileStore,
    ProfileValidationError,
    delete_env_var,
    get_set_env_entries,
    set_set_env_entries,
    upsert_env_var,
)
from aider_aid.project_store import CorruptProjectsFileError, ProjectError, ProjectStore

app = typer.Typer(help="aider configurator, launcher, and doctor")
config_app = typer.Typer(help="Manage aider config profiles")
project_app = typer.Typer(help="Manage project directory shortcuts")
server_app = typer.Typer(help="Manage named Ollama servers")

app.add_typer(config_app, name="config")
app.add_typer(project_app, name="project")
app.add_typer(server_app, name="server")


def _profile_store() -> ProfileStore:
    return ProfileStore()


def _project_store() -> ProjectStore:
    return ProjectStore()


def _server_store() -> OllamaServerStore:
    return OllamaServerStore()


def _echo_profile_validation_note(skipped: bool, message: str) -> None:
    if skipped and message:
        typer.echo(f"Validation note: {message}")


def _parse_option_assignments(option_values: list[str]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for entry in option_values:
        if "=" not in entry:
            raise typer.BadParameter(f'Invalid --option "{entry}". Expected key=value.')
        key, value = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise typer.BadParameter(f'Invalid --option "{entry}". Empty key.')
        data[key] = yaml.safe_load(value)
    return data


def _parse_set_env(set_env: list[str]) -> list[str]:
    parsed: list[str] = []
    for entry in set_env:
        if "=" not in entry:
            raise typer.BadParameter(f'Invalid --set-env "{entry}". Expected ENV=value.')
        key, value = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise typer.BadParameter(f'Invalid --set-env "{entry}". Empty key.')
        parsed.append(f"{key}={value.strip()}")
    return parsed


def _prompt_select_index(title: str, options: list[str]) -> int:
    if not options:
        raise ValueError("No options available.")
    typer.echo(title)
    for idx, option in enumerate(options, start=1):
        typer.echo(f"{idx}. {option}")
    while True:
        raw = typer.prompt("Select number")
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return idx - 1
        typer.echo(f"Invalid selection: {raw}")


def _create_profile_interactive(store: ProfileStore, server_store: OllamaServerStore) -> str:
    name = typer.prompt("Profile name").strip()
    discovery = discover_ollama_models()
    if discovery.combined_models:
        options = discovery.combined_models + ["Enter custom model..."]
        idx = _prompt_select_index("Choose Ollama model:", options)
        if idx == len(options) - 1:
            model_input = typer.prompt("Model (for example llama3.1)")
        else:
            model_input = options[idx]
    else:
        model_input = typer.prompt("Model (for example llama3.1)")

    model = normalize_ollama_model(model_input)
    server_name: str | None = None
    servers = server_store.list_servers()
    if servers:
        options = [f"{server.name} ({server.url})" for server in servers] + ["No server binding"]
        idx = _prompt_select_index("Bind profile to named Ollama server (optional):", options)
        if idx != len(options) - 1:
            server_name = servers[idx].name

    ollama_api_base = ""
    if not server_name:
        ollama_api_base = typer.prompt(
            "OLLAMA_API_BASE (optional)",
            default="",
            show_default=False,
        ).strip()
    ollama_api_key = typer.prompt(
        "OLLAMA_API_KEY (optional)",
        default="",
        hide_input=True,
        show_default=False,
    ).strip()

    config: dict[str, Any] = {"model": model}
    set_env: list[str] = []
    if ollama_api_base:
        set_env.append(f"OLLAMA_API_BASE={ollama_api_base}")
    if ollama_api_key:
        set_env.append(f"OLLAMA_API_KEY={ollama_api_key}")
    if set_env:
        config["set-env"] = set_env
    if server_name:
        config[AIDER_AID_OLLAMA_SERVER_KEY] = server_name

    profile, validation = store.save_profile(name, config)
    _echo_profile_validation_note(validation.skipped, validation.message)
    typer.echo(f'Created profile "{profile.name}" at {profile.path}')
    return profile.name


@config_app.command("list")
def config_list() -> None:
    store = _profile_store()
    profiles = store.list_profiles()
    if not profiles:
        typer.echo("No profiles found.")
        return
    for idx, profile in enumerate(profiles, start=1):
        model = profile.config.get("model", "(unset)")
        typer.echo(f"{idx}. {profile.name} | model={model} | file={profile.path}")


@config_app.command("create")
def config_create(
    name: str | None = typer.Argument(None, help="Profile name"),
    model: str | None = typer.Option(None, "--model", help="Model value (for example ollama_chat/llama3.1)"),
    server: str | None = typer.Option(None, "--server", help="Bind to a named Ollama server"),
    ollama_api_base: str | None = typer.Option(None, "--ollama-api-base", help="Set OLLAMA_API_BASE in set-env"),
    ollama_api_key: str | None = typer.Option(
        None,
        "--ollama-api-key",
        help="Set OLLAMA_API_KEY in set-env",
        hide_input=True,
    ),
    set_env: list[str] = typer.Option([], "--set-env", help="Additional set-env value (ENV=value). Repeatable."),
    option: list[str] = typer.Option([], "--option", help="Additional aider config key=value. Repeatable."),
) -> None:
    store = _profile_store()
    server_store = _server_store()
    profile_name = name.strip() if name else typer.prompt("Profile name").strip()
    if not profile_name:
        raise typer.BadParameter("Profile name cannot be empty.")

    if model is None:
        discovery = discover_ollama_models()
        if discovery.combined_models:
            options = discovery.combined_models + ["Enter custom model..."]
            idx = _prompt_select_index("Choose Ollama model:", options)
            if idx == len(options) - 1:
                model = typer.prompt("Model (for example llama3.1)")
            else:
                model = options[idx]
        else:
            model = typer.prompt("Model (for example llama3.1)")

    model_value = normalize_ollama_model(model)
    if server and ollama_api_base:
        raise typer.BadParameter("Use either --server or --ollama-api-base, not both.")

    if server:
        try:
            server_store.get_server(server)
        except OllamaServerError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1) from exc

    env_entries = _parse_set_env(set_env)
    if ollama_api_base:
        env_entries = upsert_env_var(env_entries, "OLLAMA_API_BASE", ollama_api_base)
    if ollama_api_key:
        env_entries = upsert_env_var(env_entries, "OLLAMA_API_KEY", ollama_api_key)

    config_data = _parse_option_assignments(option)
    config_data["model"] = model_value
    if env_entries:
        config_data["set-env"] = env_entries
    if server:
        config_data[AIDER_AID_OLLAMA_SERVER_KEY] = server

    try:
        profile, validation = store.save_profile(profile_name, config_data)
    except ProfileValidationError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    except ProfileError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    _echo_profile_validation_note(validation.skipped, validation.message)
    typer.echo(f'Created profile "{profile.name}" at {profile.path}')


@config_app.command("show")
def config_show(name: str = typer.Argument(..., help="Profile name")) -> None:
    store = _profile_store()
    try:
        profile = store.get_profile(name)
    except ProfileNotFoundError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(profile.path)
    typer.echo(yaml.safe_dump(profile.config, sort_keys=False, allow_unicode=False))


@config_app.command("edit")
def config_edit(
    name: str = typer.Argument(..., help="Profile name"),
    new_name: str | None = typer.Option(None, "--name", help="New profile name"),
    model: str | None = typer.Option(None, "--model", help="New model value"),
    server: str | None = typer.Option(None, "--server", help="Bind profile to named Ollama server"),
    clear_server: bool = typer.Option(False, "--clear-server", help="Clear profile server binding"),
    ollama_api_base: str | None = typer.Option(None, "--ollama-api-base", help="Set OLLAMA_API_BASE"),
    clear_ollama_api_base: bool = typer.Option(False, "--clear-ollama-api-base", help="Clear OLLAMA_API_BASE"),
    ollama_api_key: str | None = typer.Option(None, "--ollama-api-key", hide_input=True, help="Set OLLAMA_API_KEY"),
    clear_ollama_api_key: bool = typer.Option(False, "--clear-ollama-api-key", help="Clear OLLAMA_API_KEY"),
    set_env: list[str] = typer.Option([], "--set-env", help="Upsert additional set-env entry (ENV=value)."),
    option: list[str] = typer.Option([], "--option", help="Upsert additional YAML key=value."),
    interactive: bool = typer.Option(False, "--interactive", help="Prompt for common fields."),
) -> None:
    store = _profile_store()
    server_store = _server_store()
    try:
        profile = store.get_profile(name)
    except ProfileNotFoundError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    data = dict(profile.config)
    target_name = new_name or profile.name
    model_value = model

    if interactive:
        target_name = typer.prompt("Profile name", default=target_name).strip()
        model_value = typer.prompt(
            "Model",
            default=str(data.get("model") or "ollama_chat/llama3.1"),
        ).strip()
        prompted_api_base = typer.prompt(
            "OLLAMA_API_BASE (leave blank to keep current)",
            default="",
            show_default=False,
        ).strip()
        if prompted_api_base:
            ollama_api_base = prompted_api_base

    if model_value:
        data["model"] = normalize_ollama_model(model_value)

    if server and clear_server:
        raise typer.BadParameter("Use either --server or --clear-server, not both.")
    if server and ollama_api_base:
        raise typer.BadParameter("Use either --server or --ollama-api-base, not both.")
    if server:
        try:
            selected = server_store.get_server(server)
        except OllamaServerError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1) from exc
        data[AIDER_AID_OLLAMA_SERVER_KEY] = selected.name
    if clear_server:
        data.pop(AIDER_AID_OLLAMA_SERVER_KEY, None)

    env_entries = get_set_env_entries(data)
    for entry in _parse_set_env(set_env):
        key, _, value = entry.partition("=")
        env_entries = upsert_env_var(env_entries, key, value)

    if ollama_api_base:
        env_entries = upsert_env_var(env_entries, "OLLAMA_API_BASE", ollama_api_base)
    if clear_ollama_api_base:
        env_entries = delete_env_var(env_entries, "OLLAMA_API_BASE")

    if ollama_api_key:
        env_entries = upsert_env_var(env_entries, "OLLAMA_API_KEY", ollama_api_key)
    if clear_ollama_api_key:
        env_entries = delete_env_var(env_entries, "OLLAMA_API_KEY")

    set_set_env_entries(data, env_entries)
    for key, value in _parse_option_assignments(option).items():
        data[key] = value

    try:
        updated, validation = store.save_profile(target_name, data, previous_path=profile.path)
    except ProfileError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    _echo_profile_validation_note(validation.skipped, validation.message)
    typer.echo(f'Updated profile "{updated.name}" at {updated.path}')


@config_app.command("remove")
def config_remove(
    name: str = typer.Argument(..., help="Profile name"),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation prompt"),
) -> None:
    store = _profile_store()
    if not yes and not typer.confirm(f'Remove profile "{name}"?'):
        raise typer.Exit(code=0)
    try:
        removed = store.remove_profile(name)
    except ProfileError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(f"Removed profile file: {removed}")


@project_app.command("list")
def project_list() -> None:
    store = _project_store()
    try:
        projects = store.list_projects()
    except CorruptProjectsFileError as exc:
        typer.echo(str(exc))
        typer.echo(f"Backup created: {exc.backup_path}")
        projects = store.list_projects()
    if not projects:
        typer.echo("No projects found.")
        return
    for idx, project in enumerate(projects, start=1):
        typer.echo(f"{idx}. {project.name} | {project.path}")


def _prompt_project_path() -> Path:
    raw = typer.prompt("Project directory (absolute or relative)").strip()
    path = Path(raw).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise typer.BadParameter(f"Directory does not exist: {path}")
    return path


@project_app.command("add")
def project_add(
    path: Path | None = typer.Argument(None, help="Project directory"),
    name: str | None = typer.Option(None, "--name", help="Project display name"),
) -> None:
    store = _project_store()
    project_path = path.expanduser().resolve() if path else _prompt_project_path()
    if not project_path.exists() or not project_path.is_dir():
        raise typer.BadParameter(f"Directory does not exist: {project_path}")
    project_name = name or typer.prompt("Project name", default=project_path.name).strip()
    try:
        added = store.add_project(name=project_name, path=project_path)
    except ProjectError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(f'Added project "{added.name}" => {added.path}')


@project_app.command("rename")
def project_rename(
    identifier: str = typer.Argument(..., help="Project name or 1-based index"),
    new_name: str = typer.Argument(..., help="New name"),
) -> None:
    store = _project_store()
    try:
        project = store.rename_project(identifier, new_name=new_name)
    except ProjectError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(f'Renamed project to "{project.name}"')


@project_app.command("remove")
def project_remove(
    identifier: str = typer.Argument(..., help="Project name or 1-based index"),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation prompt"),
) -> None:
    store = _project_store()
    if not yes and not typer.confirm(f'Remove project entry "{identifier}"?'):
        raise typer.Exit(code=0)
    try:
        removed = store.remove_project(identifier)
    except ProjectError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(f'Removed project entry "{removed.name}" ({removed.path})')
    typer.echo("Note: project folder was not deleted.")


@server_app.command("list")
def server_list() -> None:
    store = _server_store()
    servers = store.list_servers()
    if not servers:
        typer.echo("No Ollama servers configured.")
        return
    for idx, server in enumerate(servers, start=1):
        typer.echo(f"{idx}. {server.name} | {server.url}")


@server_app.command("add")
def server_add(
    name: str = typer.Argument(..., help="Server name"),
    url: str = typer.Argument(..., help="Ollama base URL, e.g. http://gpu-host:11434"),
    replace: bool = typer.Option(False, "--replace", help="Replace existing server with same name"),
) -> None:
    store = _server_store()
    try:
        added = store.add_server(name=name, url=url, replace=replace)
    except OllamaServerError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(f'Saved Ollama server "{added.name}" => {added.url}')


@server_app.command("remove")
def server_remove(
    name: str = typer.Argument(..., help="Server name"),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation prompt"),
) -> None:
    if not yes and not typer.confirm(f'Remove Ollama server "{name}"?'):
        raise typer.Exit(code=0)
    store = _server_store()
    try:
        removed = store.remove_server(name)
    except OllamaServerError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(f'Removed Ollama server "{removed.name}"')


def _select_or_create_project(project_store: ProjectStore) -> str:
    projects = project_store.list_projects()
    if not projects:
        typer.echo("No projects saved. Creating one now.")
        project_path = _prompt_project_path()
        project_name = typer.prompt("Project name", default=project_path.name).strip()
        project_store.add_project(name=project_name, path=project_path)
        return project_name

    options = [f"{project.name} ({project.path})" for project in projects] + ["Create New..."]
    idx = _prompt_select_index("Choose project:", options)
    if idx == len(options) - 1:
        project_path = _prompt_project_path()
        project_name = typer.prompt("Project name", default=project_path.name).strip()
        project_store.add_project(name=project_name, path=project_path)
        return project_name
    return projects[idx].name


def _select_or_create_profile(profile_store: ProfileStore, server_store: OllamaServerStore) -> str:
    profiles = profile_store.list_profiles()
    if not profiles:
        typer.echo("No profiles saved. Creating one now.")
        return _create_profile_interactive(profile_store, server_store)

    options = [f"{profile.name} ({profile.config.get('model', '(unset model)')})" for profile in profiles]
    options.append("Create New...")
    idx = _prompt_select_index("Choose profile:", options)
    if idx == len(options) - 1:
        return _create_profile_interactive(profile_store, server_store)
    return profiles[idx].name


@app.command()
def launch(
    project: str | None = typer.Option(None, "--project", help="Project name or 1-based index"),
    profile: str | None = typer.Option(None, "--profile", help="Profile name"),
    arg: list[str] = typer.Option([], "--arg", help="Extra one-off aider arg. Repeatable."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print command and exit."),
) -> None:
    profile_store = _profile_store()
    project_store = _project_store()
    server_store = _server_store()

    try:
        project_identifier = project or _select_or_create_project(project_store)
        selected_project = project_store.get_project(project_identifier)

        profile_name = profile or _select_or_create_profile(profile_store, server_store)
        selected_profile = profile_store.get_profile(profile_name)
    except (ProjectError, ProfileError, IndexError, ValueError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    server_args: list[str] = []
    bound_server = selected_profile.config.get(AIDER_AID_OLLAMA_SERVER_KEY)
    if isinstance(bound_server, str) and bound_server.strip():
        try:
            server = server_store.get_server(bound_server.strip())
        except OllamaServerNotFoundError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1) from exc
        server_args.extend(["--set-env", f"OLLAMA_API_BASE={server.url}"])

    extra_args = list(arg)
    if not extra_args and sys.stdin.isatty():
        raw = typer.prompt(
            "One-off extra aider args (optional)",
            default="",
            show_default=False,
        ).strip()
        if raw:
            extra_args = shlex.split(raw)

    final_args = server_args + extra_args
    cmd, rc = launch_aider(
        project_path=selected_project.path,
        profile_path=selected_profile.path,
        extra_args=final_args,
        dry_run=dry_run,
    )
    typer.echo(f"Project: {selected_project.path}")
    typer.echo(f"Profile: {selected_profile.path}")
    if server_args:
        typer.echo(f"Ollama server: {server.url}")
    typer.echo(f"Command: {format_shell_command(cmd)}")
    if dry_run:
        raise typer.Exit(code=0)
    if rc != 0:
        raise typer.Exit(code=rc)


def _print_doctor_result(result: DoctorResult) -> None:
    status_map = {
        "pass": "[PASS]",
        "warn": "[WARN]",
        "fail": "[FAIL]",
    }
    prefix = status_map.get(result.status, "[INFO]")
    typer.echo(f"{prefix} {result.id}: {result.message}")
    if result.details:
        typer.echo(f"  details: {result.details}")
    if result.remediation:
        typer.echo(f"  remediation: {result.remediation}")


@app.command()
def doctor() -> None:
    profile_store = _profile_store()
    server_store = _server_store()
    results = run_doctor(profile_store, server_store)
    for result in results:
        _print_doctor_result(result)

    has_fail = any(item.status == "fail" for item in results)
    raise typer.Exit(code=1 if has_fail else 0)


if __name__ == "__main__":
    app()
