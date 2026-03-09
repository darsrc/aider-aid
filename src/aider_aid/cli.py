from __future__ import annotations

from collections.abc import Callable
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

import typer
import yaml

from aider_aid.doctor import DEFAULT_OLLAMA_API_BASE, DoctorResult, probe_ollama_endpoint, run_doctor
from aider_aid.launcher import format_shell_command, launch_aider
from aider_aid.model_discovery import normalize_ollama_model
from aider_aid.ollama_server_store import (
    AIDER_AID_OLLAMA_SERVER_KEY,
    OllamaServerError,
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
    slugify_profile_name,
    strip_aider_aid_metadata,
    upsert_env_var,
)
from aider_aid.project_store import CorruptProjectsFileError, ProjectError, ProjectStore
from aider_aid.tui import (
    ask_confirm,
    ask_text,
    print_error,
    print_section,
    print_warning,
    select_index,
    show_banner,
)

app = typer.Typer(help="aider configurator, launcher, and doctor")
config_app = typer.Typer(help="Manage aider config profiles")
project_app = typer.Typer(help="Manage project directory shortcuts")
server_app = typer.Typer(help="Manage named Ollama servers")

app.add_typer(config_app, name="config")
app.add_typer(project_app, name="project")
app.add_typer(server_app, name="server")

DEFAULT_MODEL_CONTEXT_SIZE = 8192
MODEL_SETTINGS_SUFFIX = ".aider.model.settings.yml"
QOL_PRESETS = ("local-safe", "fast-iter", "strict-ci", "large-repo")
MODEL_ROLE_WEAK = "weak-model"
MODEL_ROLE_EDITOR = "editor-model"


def _profile_store() -> ProfileStore:
    return ProfileStore()


def _project_store() -> ProjectStore:
    return ProjectStore()


def _server_store() -> OllamaServerStore:
    return OllamaServerStore()


def _echo_profile_validation_note(skipped: bool, message: str) -> None:
    if skipped and message:
        typer.echo(f"Validation note: {message}")


def _stdin_is_tty() -> bool:
    return sys.stdin.isatty()


def _show_banner() -> None:
    show_banner()


def _prompt_text(
    label: str,
    *,
    default: str | None = None,
    allow_empty: bool = False,
    password: bool = False,
) -> str | None:
    if _stdin_is_tty():
        return ask_text(label, default=default, allow_empty=allow_empty, password=password)
    try:
        if default is None:
            value = typer.prompt(label, hide_input=password).strip()
        else:
            value = typer.prompt(label, default=default, hide_input=password).strip()
    except (KeyboardInterrupt, EOFError):
        typer.echo("")
        return None
    if not value and not allow_empty:
        typer.echo("Value cannot be empty.")
        return None
    return value


def _run_menu_action(action: Callable[[], None]) -> None:
    try:
        action()
    except typer.Exit as exc:
        if exc.exit_code not in (None, 0):
            print_warning(f"Action exited with code {exc.exit_code}.")
    except (KeyboardInterrupt, EOFError):
        typer.echo("")
    except Exception as exc:
        print_error(f"Error: {exc}")


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


def _parse_context_size(context_size: int | None) -> int | None:
    if context_size is None:
        return None
    if context_size <= 0:
        raise typer.BadParameter("--context-size must be greater than 0.")
    return context_size


def _normalize_optional_model(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return normalize_ollama_model(value)
    except ValueError:
        return None


def _role_model_follows_main(config: dict[str, Any], role_key: str, main_model: str | None) -> bool:
    if main_model is None:
        return False
    return _normalize_optional_model(config.get(role_key)) == main_model


def _build_context_model_settings(context_size: int) -> list[dict[str, Any]]:
    return [
        {
            "name": "aider/extra_params",
            "extra_params": {"num_ctx": context_size},
        }
    ]


def _write_context_model_settings(path: Path, context_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_context_model_settings(context_size)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _managed_model_settings_path(store: ProfileStore, profile_name: str) -> Path:
    slug = slugify_profile_name(profile_name)
    return store.profiles_dir / f"{slug}{MODEL_SETTINGS_SUFFIX}"


def _is_managed_model_settings_path(store: ProfileStore, path_value: str | None) -> bool:
    if not isinstance(path_value, str) or not path_value.strip():
        return False
    candidate = Path(path_value).expanduser()
    return candidate.parent == store.profiles_dir and candidate.name.endswith(MODEL_SETTINGS_SUFFIX)


def _stage_model_settings_write(path: Path, context_size: int) -> tuple[bool, str | None]:
    existed = path.exists()
    previous = path.read_text(encoding="utf-8") if existed else None
    _write_context_model_settings(path, context_size)
    return existed, previous


def _rollback_staged_model_settings(path: Path, existed: bool, previous: str | None) -> None:
    if existed and previous is not None:
        path.write_text(previous, encoding="utf-8")
    else:
        path.unlink(missing_ok=True)


def _has_model_settings_file(config: dict[str, Any]) -> bool:
    value = config.get("model-settings-file")
    return isinstance(value, str) and bool(value.strip())


def _create_runtime_model_settings(context_size: int) -> Path:
    with tempfile.NamedTemporaryFile(
        "w",
        suffix=MODEL_SETTINGS_SUFFIX,
        prefix="aider-aid-runtime-",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        yaml.safe_dump(_build_context_model_settings(context_size), tmp, sort_keys=False, allow_unicode=False)
        return Path(tmp.name)


def _prompt_select_index(title: str, options: list[str]) -> int:
    if not options:
        raise ValueError("No options available.")
    idx = select_index(title, options)
    if idx is None:
        raise KeyboardInterrupt
    return idx


def _resolve_model_source(
    *,
    server_store: OllamaServerStore,
    server_name: str | None,
    ollama_api_base: str | None,
) -> tuple[str, str | None]:
    if server_name and ollama_api_base:
        raise typer.BadParameter("Use either --server or --ollama-api-base, not both.")

    if server_name:
        server = server_store.get_server(server_name)
        return server.url, server.url

    if ollama_api_base:
        endpoint = ollama_api_base.strip()
        if not endpoint:
            raise typer.BadParameter("--ollama-api-base cannot be empty.")
        return endpoint, endpoint

    if _stdin_is_tty():
        servers = server_store.list_servers()
        if servers:
            options = [f"{server.name} ({server.url})" for server in servers]
            options += [
                f"Default endpoint ({DEFAULT_OLLAMA_API_BASE})",
                "Custom endpoint",
            ]
            idx = _prompt_select_index("Choose Ollama server/source:", options)
            if idx < len(servers):
                selected = servers[idx]
                return selected.url, selected.url
            if idx == len(servers):
                return DEFAULT_OLLAMA_API_BASE, None
            custom = _prompt_text("Custom OLLAMA_API_BASE")
            if not custom:
                raise typer.BadParameter("Custom OLLAMA_API_BASE cannot be empty.")
            return custom, custom

    return DEFAULT_OLLAMA_API_BASE, None


def _fetch_models_from_endpoint(endpoint: str, api_key: str | None) -> list[str]:
    ok, models, error = probe_ollama_endpoint(endpoint, api_key=api_key)
    if not ok:
        raise typer.BadParameter(
            "Unable to list models from Ollama endpoint "
            f"{endpoint}: {error}. Ensure server is reachable and has models."
        )
    cleaned = [name.strip() for name in models if isinstance(name, str) and name.strip()]
    deduped: list[str] = []
    seen: set[str] = set()
    for model_name in cleaned:
        if model_name in seen:
            continue
        seen.add(model_name)
        deduped.append(model_name)
    if not deduped:
        raise typer.BadParameter(
            f"Ollama endpoint {endpoint} returned no models. Pull a model first (for example: ollama pull llama3)."
        )
    return deduped


def _choose_model_for_profile(endpoint: str, api_key: str | None) -> str:
    models = _fetch_models_from_endpoint(endpoint, api_key)
    if _stdin_is_tty():
        idx = _prompt_select_index(f"Choose model from {endpoint}:", models)
        return normalize_ollama_model(models[idx])
    if len(models) == 1:
        return normalize_ollama_model(models[0])
    raise typer.BadParameter(
        "Multiple models are available; run in a TTY to choose one interactively, or pass --model explicitly."
    )


def _create_profile_interactive() -> str:
    if not _stdin_is_tty():
        raise typer.BadParameter(
            "Interactive profile creation requires a TTY. Use `aider-aid config create <name> --model <model>`."
        )
    name = _prompt_text("Profile name")
    if not name:
        raise typer.Exit(code=1)
    server_store = _server_store()
    model_endpoint, api_base_to_store = _resolve_model_source(
        server_store=server_store,
        server_name=None,
        ollama_api_base=None,
    )
    api_key_value = _prompt_text(
        "OLLAMA_API_KEY (optional)",
        default="",
        allow_empty=True,
        password=True,
    )
    if api_key_value is None:
        raise typer.Exit(code=0)
    model_value = _choose_model_for_profile(model_endpoint, api_key_value or None)
    context_size = _prompt_optional_positive_int(
        "Context size tokens",
        current_value=DEFAULT_MODEL_CONTEXT_SIZE,
    )
    final_context_size = context_size if context_size is not None else DEFAULT_MODEL_CONTEXT_SIZE
    _run_config_create_command(
        name=name,
        model=model_value,
        weak_model=model_value,
        editor_model=model_value,
        context_size=final_context_size,
        qol_preset=None,
        server=None,
        ollama_api_base=api_base_to_store,
        ollama_api_key=api_key_value or None,
        set_env=[],
        option=[],
    )
    return name


def _prompt_menu_choice(title: str, options: list[str]) -> int | None:
    try:
        typer.echo("")
        return _prompt_select_index(title, options)
    except (KeyboardInterrupt, EOFError):
        typer.echo("")
        return None


def _select_profile_name(profile_store: ProfileStore, *, title: str) -> str | None:
    profiles = profile_store.list_profiles()
    if not profiles:
        typer.echo("No profiles found.")
        return None
    options = [f"{profile.name} ({profile.config.get('model', '(unset model)')})" for profile in profiles]
    idx = _prompt_menu_choice(title, options + ["Back"])
    if idx is None or idx == len(options):
        return None
    return profiles[idx].name


def _select_project_identifier(project_store: ProjectStore, *, title: str) -> str | None:
    projects = project_store.list_projects()
    if not projects:
        typer.echo("No projects found.")
        return None
    options = [f"{project.name} ({project.path})" for project in projects]
    idx = _prompt_menu_choice(title, options + ["Back"])
    if idx is None or idx == len(options):
        return None
    return projects[idx].name


def _select_server_name(server_store: OllamaServerStore, *, title: str) -> str | None:
    servers = server_store.list_servers()
    if not servers:
        typer.echo("No Ollama servers configured.")
        return None
    options = [f"{server.name} ({server.url})" for server in servers]
    idx = _prompt_menu_choice(title, options + ["Back"])
    if idx is None or idx == len(options):
        return None
    return servers[idx].name


def _select_profile_for_edit(profile_store: ProfileStore) -> str | None:
    return _select_profile_name(profile_store, title="Choose profile to edit:")


def _build_qol_preset(name: str) -> dict[str, Any]:
    if name == "local-safe":
        return {
            "auto-commits": True,
            "dirty-commits": False,
            "auto-lint": True,
            "auto-test": False,
            "suggest-shell-commands": True,
        }
    if name == "fast-iter":
        return {
            "cache-prompts": True,
            "map-refresh": "files",
            "auto-commits": True,
            "auto-lint": False,
            "auto-test": False,
            "stream": True,
        }
    if name == "strict-ci":
        return {
            "auto-commits": False,
            "dirty-commits": False,
            "auto-lint": True,
            "auto-test": True,
            "show-diffs": True,
            "git-commit-verify": True,
        }
    if name == "large-repo":
        return {
            "cache-prompts": True,
            "map-refresh": "manual",
            "map-tokens": 4096,
            "max-chat-history-tokens": 16000,
        }
    raise typer.BadParameter(f'Unknown --qol-preset "{name}". Expected one of: {", ".join(QOL_PRESETS)}.')


def _apply_qol_settings(
    config: dict[str, Any],
    *,
    weak_model: str | None,
    editor_model: str | None,
    edit_format: str | None,
    editor_edit_format: str | None,
    reasoning_effort: str | None,
    thinking_tokens: str | None,
    max_chat_history_tokens: int | None,
    cache_prompts: bool | None,
    map_tokens: int | None,
    map_refresh: str | None,
    auto_commits: bool | None,
    dirty_commits: bool | None,
    auto_lint: bool | None,
    auto_test: bool | None,
    test_cmd: str | None,
    notifications: bool | None,
    notifications_command: str | None,
    suggest_shell_commands: bool | None,
    fancy_input: bool | None,
    multiline: bool | None,
) -> None:
    updates: dict[str, Any] = {
        "weak-model": weak_model,
        "editor-model": editor_model,
        "edit-format": edit_format,
        "editor-edit-format": editor_edit_format,
        "reasoning-effort": reasoning_effort,
        "thinking-tokens": thinking_tokens,
        "max-chat-history-tokens": max_chat_history_tokens,
        "cache-prompts": cache_prompts,
        "map-tokens": map_tokens,
        "map-refresh": map_refresh,
        "auto-commits": auto_commits,
        "dirty-commits": dirty_commits,
        "auto-lint": auto_lint,
        "auto-test": auto_test,
        "test-cmd": test_cmd,
        "notifications": notifications,
        "notifications-command": notifications_command,
        "suggest-shell-commands": suggest_shell_commands,
        "fancy-input": fancy_input,
        "multiline": multiline,
    }
    for key, value in updates.items():
        if value is None:
            continue
        config[key] = value


def _extract_env_var(config: dict[str, Any], key: str) -> str | None:
    lookup = key.strip()
    if not lookup:
        return None
    for entry in get_set_env_entries(config):
        if "=" not in entry:
            continue
        env_key, env_value = entry.split("=", 1)
        if env_key.strip() == lookup:
            return env_value.strip()
    return None


def _read_config_context_size(config: dict[str, Any]) -> int | None:
    value = config.get("model-settings-file")
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.exists():
        return None
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(payload, list):
        return None
    for item in payload:
        if not isinstance(item, dict):
            continue
        extra_params = item.get("extra_params")
        if not isinstance(extra_params, dict):
            continue
        num_ctx = extra_params.get("num_ctx")
        if isinstance(num_ctx, int) and num_ctx > 0:
            return num_ctx
        if isinstance(num_ctx, str) and num_ctx.isdigit() and int(num_ctx) > 0:
            return int(num_ctx)
    return None


def _prompt_optional_positive_int(label: str, *, current_value: int | None) -> int | None:
    display_current = str(current_value) if current_value is not None else "(unset)"
    while True:
        raw = _prompt_text(
            f"{label} (blank keeps current: {display_current})",
            default="",
            allow_empty=True,
        )
        if raw is None:
            raise typer.Exit(code=0)
        value = raw.strip()
        if not value:
            return None
        if value.isdigit() and int(value) > 0:
            return int(value)
        print_error("Value must be a positive integer or blank.")


def _run_config_create_command(
    *,
    name: str | None,
    model: str | None,
    weak_model: str | None,
    editor_model: str | None,
    context_size: int,
    qol_preset: str | None,
    server: str | None,
    ollama_api_base: str | None,
    ollama_api_key: str | None,
    set_env: list[str],
    option: list[str],
) -> None:
    config_create(
        name=name,
        model=model,
        context_size=context_size,
        qol_preset=qol_preset,
        weak_model=weak_model,
        editor_model=editor_model,
        edit_format=None,
        editor_edit_format=None,
        reasoning_effort=None,
        thinking_tokens=None,
        max_chat_history_tokens=None,
        cache_prompts=None,
        map_tokens=None,
        map_refresh=None,
        auto_commits=None,
        dirty_commits=None,
        auto_lint=None,
        auto_test=None,
        test_cmd=None,
        notifications=None,
        notifications_command=None,
        suggest_shell_commands=None,
        fancy_input=None,
        multiline=None,
        server=server,
        ollama_api_base=ollama_api_base,
        ollama_api_key=ollama_api_key,
        set_env=set_env,
        option=option,
    )


def _run_config_edit_command(
    *,
    name: str,
    new_name: str | None,
    model: str | None,
    weak_model: str | None,
    clear_weak_model: bool,
    editor_model: str | None,
    clear_editor_model: bool,
    context_size: int | None,
    qol_preset: str | None,
    server: str | None,
    clear_server: bool,
    ollama_api_base: str | None,
    clear_ollama_api_base: bool,
    ollama_api_key: str | None,
    clear_ollama_api_key: bool,
    set_env: list[str],
    option: list[str],
) -> None:
    config_edit(
        name=name,
        new_name=new_name,
        model=model,
        context_size=context_size,
        qol_preset=qol_preset,
        weak_model=weak_model,
        clear_weak_model=clear_weak_model,
        editor_model=editor_model,
        clear_editor_model=clear_editor_model,
        edit_format=None,
        editor_edit_format=None,
        reasoning_effort=None,
        thinking_tokens=None,
        max_chat_history_tokens=None,
        cache_prompts=None,
        map_tokens=None,
        map_refresh=None,
        auto_commits=None,
        dirty_commits=None,
        auto_lint=None,
        auto_test=None,
        test_cmd=None,
        notifications=None,
        notifications_command=None,
        suggest_shell_commands=None,
        fancy_input=None,
        multiline=None,
        server=server,
        clear_server=clear_server,
        ollama_api_base=ollama_api_base,
        clear_ollama_api_base=clear_ollama_api_base,
        ollama_api_key=ollama_api_key,
        clear_ollama_api_key=clear_ollama_api_key,
        set_env=set_env,
        option=option,
        interactive=False,
    )


def _interactive_create_profile_flow() -> None:
    print_section("Create Profile", "Fill in the sections below. Ctrl-C cancels.")

    profile_name = _prompt_text("Profile name")
    if profile_name is None:
        raise typer.Exit(code=0)

    server_store = _server_store()
    model_endpoint, api_base_to_store = _resolve_model_source(
        server_store=server_store,
        server_name=None,
        ollama_api_base=None,
    )
    api_key_value = _prompt_text(
        "OLLAMA_API_KEY (optional)",
        default="",
        allow_empty=True,
        password=True,
    )
    if api_key_value is None:
        raise typer.Exit(code=0)

    model_value = _choose_model_for_profile(model_endpoint, api_key_value or None)

    weak_model_choice = _prompt_select_index(
        "Weak model:",
        [
            f"Use main model ({model_value})",
            f"Choose from endpoint ({model_endpoint})",
            "Enter model manually",
        ],
    )
    if weak_model_choice == 0:
        weak_model_value = model_value
    elif weak_model_choice == 1:
        weak_model_value = _choose_model_for_profile(model_endpoint, api_key_value or None)
    else:
        prompted_weak = _prompt_text("Weak model", default=model_value)
        if prompted_weak is None:
            raise typer.Exit(code=0)
        weak_model_value = prompted_weak

    editor_model_choice = _prompt_select_index(
        "Editor model:",
        [
            f"Use main model ({model_value})",
            f"Choose from endpoint ({model_endpoint})",
            "Enter model manually",
        ],
    )
    if editor_model_choice == 0:
        editor_model_value = model_value
    elif editor_model_choice == 1:
        editor_model_value = _choose_model_for_profile(model_endpoint, api_key_value or None)
    else:
        prompted_editor = _prompt_text("Editor model", default=model_value)
        if prompted_editor is None:
            raise typer.Exit(code=0)
        editor_model_value = prompted_editor

    context_size_value = _prompt_optional_positive_int(
        "Context size tokens",
        current_value=DEFAULT_MODEL_CONTEXT_SIZE,
    )
    final_context_size = context_size_value if context_size_value is not None else DEFAULT_MODEL_CONTEXT_SIZE

    preset_idx = _prompt_select_index(
        "QoL preset:",
        [
            "No preset (keep explicit options only)",
            "local-safe",
            "fast-iter",
            "strict-ci",
            "large-repo",
        ],
    )
    qol_preset = None if preset_idx == 0 else QOL_PRESETS[preset_idx - 1]

    summary_lines = [
        f"Name: {profile_name}",
        f"Model: {model_value}",
        f"Weak model: {weak_model_value}",
        f"Editor model: {editor_model_value}",
        f"Context size: {final_context_size}",
        f"Ollama endpoint: {api_base_to_store or DEFAULT_OLLAMA_API_BASE} {'(stored)' if api_base_to_store else '(default)'}",
        f"QoL preset: {qol_preset or '(none)'}",
    ]
    typer.echo("\nReview:")
    for line in summary_lines:
        typer.echo(f"  - {line}")
    if not ask_confirm("Create this profile?", default=True):
        raise typer.Exit(code=0)

    _run_config_create_command(
        name=profile_name,
        model=model_value,
        weak_model=weak_model_value,
        editor_model=editor_model_value,
        context_size=final_context_size,
        qol_preset=qol_preset,
        server=None,
        ollama_api_base=api_base_to_store,
        ollama_api_key=api_key_value or None,
        set_env=[],
        option=[],
    )


def _interactive_edit_profile_flow(profile_name: str) -> None:
    store = _profile_store()
    server_store = _server_store()
    profile = store.get_profile(profile_name)
    current_model = str(profile.config.get("model") or "ollama_chat/llama3.1")
    current_weak_model = str(profile.config.get(MODEL_ROLE_WEAK) or "(unset)")
    current_editor_model = str(profile.config.get(MODEL_ROLE_EDITOR) or "(unset)")
    current_endpoint = _extract_env_var(profile.config, "OLLAMA_API_BASE") or DEFAULT_OLLAMA_API_BASE
    current_api_key = _extract_env_var(profile.config, "OLLAMA_API_KEY")
    current_context_size = _read_config_context_size(profile.config)

    print_section("Edit Profile", f"{profile.name} | model={current_model}")

    typer.echo("\nIdentity")
    target_name = _prompt_text("Profile name", default=profile.name)
    if target_name is None:
        raise typer.Exit(code=0)
    if not target_name.strip():
        raise typer.BadParameter("Profile name cannot be empty.")
    new_name = target_name if target_name != profile.name else None

    typer.echo("\nModel & Context")
    model_value: str | None = None
    model_lookup_endpoint = current_endpoint
    model_choice = _prompt_select_index(
        "Model action:",
        [
            f"Keep current model ({current_model})",
            f"Choose from current endpoint ({current_endpoint})",
            "Choose from a different endpoint",
            "Enter model manually",
        ],
    )
    if model_choice == 1:
        model_value = _choose_model_for_profile(current_endpoint, current_api_key)
    elif model_choice == 2:
        chosen_endpoint, _ = _resolve_model_source(
            server_store=server_store,
            server_name=None,
            ollama_api_base=None,
        )
        model_lookup_endpoint = chosen_endpoint
        model_value = _choose_model_for_profile(chosen_endpoint, current_api_key)
    elif model_choice == 3:
        typed_model = _prompt_text("Model", default=current_model)
        if typed_model is None:
            raise typer.Exit(code=0)
        model_value = typed_model
    next_main_model = model_value or current_model

    weak_model_value: str | None = None
    clear_weak_model = False
    weak_choice = _prompt_select_index(
        "Weak model action:",
        [
            f"Keep current weak model ({current_weak_model})",
            f"Use main model ({next_main_model})",
            f"Choose from endpoint ({model_lookup_endpoint})",
            "Enter weak model manually",
            "Clear weak model override",
        ],
    )
    if weak_choice == 1:
        weak_model_value = next_main_model
    elif weak_choice == 2:
        weak_model_value = _choose_model_for_profile(model_lookup_endpoint, current_api_key)
    elif weak_choice == 3:
        typed_weak = _prompt_text("Weak model", default=next_main_model)
        if typed_weak is None:
            raise typer.Exit(code=0)
        weak_model_value = typed_weak
    elif weak_choice == 4:
        clear_weak_model = True

    editor_model_value: str | None = None
    clear_editor_model = False
    editor_choice = _prompt_select_index(
        "Editor model action:",
        [
            f"Keep current editor model ({current_editor_model})",
            f"Use main model ({next_main_model})",
            f"Choose from endpoint ({model_lookup_endpoint})",
            "Enter editor model manually",
            "Clear editor model override",
        ],
    )
    if editor_choice == 1:
        editor_model_value = next_main_model
    elif editor_choice == 2:
        editor_model_value = _choose_model_for_profile(model_lookup_endpoint, current_api_key)
    elif editor_choice == 3:
        typed_editor = _prompt_text("Editor model", default=next_main_model)
        if typed_editor is None:
            raise typer.Exit(code=0)
        editor_model_value = typed_editor
    elif editor_choice == 4:
        clear_editor_model = True

    context_size = _prompt_optional_positive_int(
        "Context size tokens",
        current_value=current_context_size,
    )

    typer.echo("\nEndpoint & Auth")
    endpoint_choice = _prompt_select_index(
        "OLLAMA_API_BASE:",
        [
            f"Keep current ({current_endpoint})",
            "Choose named/default/custom endpoint",
            "Clear OLLAMA_API_BASE (use aider/host default)",
        ],
    )
    ollama_api_base: str | None = None
    clear_ollama_api_base = False
    if endpoint_choice == 1:
        _, api_base_to_store = _resolve_model_source(
            server_store=server_store,
            server_name=None,
            ollama_api_base=None,
        )
        ollama_api_base = api_base_to_store
        clear_ollama_api_base = api_base_to_store is None
    elif endpoint_choice == 2:
        clear_ollama_api_base = True
        ollama_api_base = None

    key_choice = _prompt_select_index(
        "OLLAMA_API_KEY:",
        [
            "Keep current value",
            "Set/replace API key",
            "Clear API key",
        ],
    )
    ollama_api_key: str | None = None
    clear_ollama_api_key = False
    if key_choice == 1:
        prompted_key = _prompt_text(
            "OLLAMA_API_KEY",
            default="",
            allow_empty=True,
            password=True,
        )
        if prompted_key is None:
            raise typer.Exit(code=0)
        ollama_api_key = prompted_key or None
    elif key_choice == 2:
        clear_ollama_api_key = True

    typer.echo("\nQoL")
    preset_choice = _prompt_select_index(
        "QoL preset:",
        [
            "Keep current profile QoL settings",
            "local-safe",
            "fast-iter",
            "strict-ci",
            "large-repo",
        ],
    )
    qol_preset = None if preset_choice == 0 else QOL_PRESETS[preset_choice - 1]

    typer.echo("\nReview:")
    typer.echo(f"  - Profile: {profile.name} -> {target_name}")
    typer.echo(f"  - Model: {model_value or '(keep current)'}")
    typer.echo(
        "  - Weak model: "
        + ("(clear)" if clear_weak_model else (weak_model_value if weak_model_value else "(keep current)"))
    )
    typer.echo(
        "  - Editor model: "
        + (
            "(clear)"
            if clear_editor_model
            else (editor_model_value if editor_model_value else "(keep current)")
        )
    )
    typer.echo(f"  - Context size: {context_size if context_size is not None else '(keep current)'}")
    typer.echo(
        "  - OLLAMA_API_BASE: "
        + (
            "(clear)"
            if clear_ollama_api_base
            else (ollama_api_base if ollama_api_base else "(keep current)")
        )
    )
    typer.echo(
        "  - OLLAMA_API_KEY: "
        + ("(clear)" if clear_ollama_api_key else ("(set)" if ollama_api_key else "(keep current)"))
    )
    typer.echo(f"  - QoL preset: {qol_preset or '(keep current)'}")

    if not ask_confirm("Save these profile changes?", default=True):
        raise typer.Exit(code=0)

    _run_config_edit_command(
        name=profile.name,
        new_name=new_name,
        model=model_value,
        weak_model=weak_model_value,
        clear_weak_model=clear_weak_model,
        editor_model=editor_model_value,
        clear_editor_model=clear_editor_model,
        context_size=context_size,
        qol_preset=qol_preset,
        server=None,
        clear_server=False,
        ollama_api_base=ollama_api_base,
        clear_ollama_api_base=clear_ollama_api_base,
        ollama_api_key=ollama_api_key,
        clear_ollama_api_key=clear_ollama_api_key,
        set_env=[],
        option=[],
    )


def _interactive_config_menu() -> None:
    store = _profile_store()
    while True:
        choice = _prompt_menu_choice(
            "Config Profiles:",
            [
                "List profiles",
                "Create profile",
                "Show profile",
                "Edit profile",
                "Remove profile",
                "Back",
            ],
        )
        if choice is None or choice == 5:
            return
        if choice == 0:
            _run_menu_action(config_list)
        elif choice == 1:
            _run_menu_action(_interactive_create_profile_flow)
        elif choice == 2:
            name = _select_profile_name(store, title="Choose profile to show:")
            if name:
                _run_menu_action(lambda: config_show(name))
        elif choice == 3:
            name = _select_profile_for_edit(store)
            if name:
                _run_menu_action(lambda: _interactive_edit_profile_flow(name))
        elif choice == 4:
            name = _select_profile_name(store, title="Choose profile to remove:")
            if name:
                _run_menu_action(lambda: config_remove(name=name, yes=False))


def _interactive_project_menu() -> None:
    store = _project_store()
    while True:
        choice = _prompt_menu_choice(
            "Projects:",
            [
                "List projects",
                "Add project",
                "Rename project",
                "Remove project",
                "Back",
            ],
        )
        if choice is None or choice == 4:
            return
        if choice == 0:
            _run_menu_action(project_list)
        elif choice == 1:
            _run_menu_action(lambda: project_add(path=None, name=None))
        elif choice == 2:
            identifier = _select_project_identifier(store, title="Choose project to rename:")
            new_name = _prompt_text("New project name")
            if identifier and new_name:
                _run_menu_action(lambda: project_rename(identifier=identifier, new_name=new_name))
        elif choice == 3:
            identifier = _select_project_identifier(store, title="Choose project to remove:")
            if identifier:
                _run_menu_action(lambda: project_remove(identifier=identifier, yes=False))


def _interactive_server_menu() -> None:
    store = _server_store()
    while True:
        choice = _prompt_menu_choice(
            "Ollama Servers:",
            [
                "List servers",
                "Add server",
                "Remove server",
                "Back",
            ],
        )
        if choice is None or choice == 3:
            return
        if choice == 0:
            _run_menu_action(server_list)
        elif choice == 1:
            name = _prompt_text("Server name")
            url = _prompt_text("Server URL (http(s)://host:port)")
            if not name or not url:
                continue
            replace = ask_confirm("Replace existing server if present?", default=False)
            if replace is None:
                continue
            _run_menu_action(lambda: server_add(name=name, url=url, replace=replace))
        elif choice == 2:
            name = _select_server_name(store, title="Choose server to remove:")
            if name:
                _run_menu_action(lambda: server_remove(name=name, yes=False))


def _interactive_doctor_menu() -> None:
    while True:
        choice = _prompt_menu_choice("Doctor:", ["Run doctor", "Back"])
        if choice is None or choice == 1:
            return
        _run_menu_action(doctor)


def _interactive_main_menu() -> None:
    _show_banner()
    while True:
        choice = _prompt_menu_choice(
            "Main Menu:",
            [
                "Launch Aider",
                "Config Profiles",
                "Projects",
                "Ollama Servers",
                "Doctor",
                "Exit",
            ],
        )
        if choice is None or choice == 5:
            return
        if choice == 0:
            _run_menu_action(lambda: launch(project=None, profile=None, arg=[], dry_run=False))
        elif choice == 1:
            _interactive_config_menu()
        elif choice == 2:
            _interactive_project_menu()
        elif choice == 3:
            _interactive_server_menu()
        elif choice == 4:
            _interactive_doctor_menu()


def _run_textual_ui() -> int:
    try:
        from aider_aid.textual_app import run_textual_app
    except Exception as exc:  # pragma: no cover - defensive import fallback
        print_warning(
            "Textual UI unavailable "
            f"({exc}). Install/upgrade with `python -m pip install --upgrade textual aider-aid`, "
            "then rerun with `--ui-mode textual`."
        )
        return 1
    return run_textual_app()


@app.callback(invoke_without_command=True)
def app_callback(
    ctx: typer.Context,
    ui_mode: Literal["auto", "classic", "textual"] = typer.Option(
        "auto",
        "--ui-mode",
        help="Interactive UI mode when running without a subcommand.",
    ),
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    if not _stdin_is_tty():
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)
    if ui_mode in {"auto", "textual"}:
        rc = _run_textual_ui()
        if ui_mode == "textual" and rc != 0:
            raise typer.Exit(code=rc)
        if rc == 0:
            raise typer.Exit(code=0)
    _interactive_main_menu()
    raise typer.Exit(code=0)


@app.command("menu", hidden=True)
def menu(
    section: Literal["main", "config", "projects", "servers", "doctor"] = typer.Argument(
        "main",
        help="Interactive menu section to open.",
    ),
) -> None:
    if not _stdin_is_tty():
        typer.echo("Interactive menus require a TTY.")
        raise typer.Exit(code=1)
    if section == "main":
        _interactive_main_menu()
    elif section == "config":
        _show_banner()
        _interactive_config_menu()
    elif section == "projects":
        _show_banner()
        _interactive_project_menu()
    elif section == "servers":
        _show_banner()
        _interactive_server_menu()
    elif section == "doctor":
        _show_banner()
        _interactive_doctor_menu()
    raise typer.Exit(code=0)


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
    context_size: int = typer.Option(
        DEFAULT_MODEL_CONTEXT_SIZE,
        "--context-size",
        help="Model context window size in tokens (writes num_ctx via model settings).",
    ),
    qol_preset: str | None = typer.Option(
        None,
        "--qol-preset",
        help=f"Apply QoL defaults. Choices: {', '.join(QOL_PRESETS)}.",
    ),
    weak_model: str | None = typer.Option(None, "--weak-model", help="Set aider weak model."),
    editor_model: str | None = typer.Option(None, "--editor-model", help="Set aider editor model."),
    edit_format: str | None = typer.Option(None, "--edit-format", help="Set aider edit format."),
    editor_edit_format: str | None = typer.Option(None, "--editor-edit-format", help="Set editor edit format."),
    reasoning_effort: str | None = typer.Option(None, "--reasoning-effort", help="Set reasoning effort."),
    thinking_tokens: str | None = typer.Option(None, "--thinking-tokens", help="Set thinking token budget."),
    max_chat_history_tokens: int | None = typer.Option(
        None,
        "--max-chat-history-tokens",
        help="Set max chat history tokens.",
    ),
    cache_prompts: bool | None = typer.Option(None, "--cache-prompts/--no-cache-prompts"),
    map_tokens: int | None = typer.Option(None, "--map-tokens", help="Set repo map token budget."),
    map_refresh: str | None = typer.Option(None, "--map-refresh", help="Set repo map refresh mode."),
    auto_commits: bool | None = typer.Option(None, "--auto-commits/--no-auto-commits"),
    dirty_commits: bool | None = typer.Option(None, "--dirty-commits/--no-dirty-commits"),
    auto_lint: bool | None = typer.Option(None, "--auto-lint/--no-auto-lint"),
    auto_test: bool | None = typer.Option(None, "--auto-test/--no-auto-test"),
    test_cmd: str | None = typer.Option(None, "--test-cmd", help="Set test command."),
    notifications: bool | None = typer.Option(None, "--notifications/--no-notifications"),
    notifications_command: str | None = typer.Option(None, "--notifications-command", help="Set notify command."),
    suggest_shell_commands: bool | None = typer.Option(
        None,
        "--suggest-shell-commands/--no-suggest-shell-commands",
    ),
    fancy_input: bool | None = typer.Option(None, "--fancy-input/--no-fancy-input"),
    multiline: bool | None = typer.Option(None, "--multiline/--no-multiline"),
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
    if name:
        profile_name = name.strip()
    else:
        if not _stdin_is_tty():
            typer.echo("Non-interactive mode requires a profile name argument.")
            raise typer.Exit(code=1)
        prompted = _prompt_text("Profile name")
        if prompted is None:
            raise typer.Exit(code=0)
        profile_name = prompted
    if not profile_name:
        raise typer.BadParameter("Profile name cannot be empty.")

    try:
        parsed_context_size = _parse_context_size(context_size)
    except typer.BadParameter as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    try:
        model_endpoint, api_base_to_store = _resolve_model_source(
            server_store=server_store,
            server_name=server,
            ollama_api_base=ollama_api_base,
        )
    except (typer.BadParameter, OllamaServerError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    if model is None:
        try:
            model_value = _choose_model_for_profile(model_endpoint, ollama_api_key)
        except typer.BadParameter as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1) from exc
    else:
        try:
            model_value = normalize_ollama_model(model)
        except ValueError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1) from exc
    try:
        weak_model_value = model_value if weak_model is None else normalize_ollama_model(weak_model)
        editor_model_value = model_value if editor_model is None else normalize_ollama_model(editor_model)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    env_entries = _parse_set_env(set_env)
    if api_base_to_store:
        env_entries = upsert_env_var(env_entries, "OLLAMA_API_BASE", api_base_to_store)
    if ollama_api_key:
        env_entries = upsert_env_var(env_entries, "OLLAMA_API_KEY", ollama_api_key)

    config_data: dict[str, Any] = {}
    if qol_preset:
        try:
            config_data.update(_build_qol_preset(qol_preset))
        except typer.BadParameter as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1) from exc
    _apply_qol_settings(
        config_data,
        weak_model=weak_model,
        editor_model=editor_model,
        edit_format=edit_format,
        editor_edit_format=editor_edit_format,
        reasoning_effort=reasoning_effort,
        thinking_tokens=thinking_tokens,
        max_chat_history_tokens=max_chat_history_tokens,
        cache_prompts=cache_prompts,
        map_tokens=map_tokens,
        map_refresh=map_refresh,
        auto_commits=auto_commits,
        dirty_commits=dirty_commits,
        auto_lint=auto_lint,
        auto_test=auto_test,
        test_cmd=test_cmd,
        notifications=notifications,
        notifications_command=notifications_command,
        suggest_shell_commands=suggest_shell_commands,
        fancy_input=fancy_input,
        multiline=multiline,
    )
    for key, value in _parse_option_assignments(option).items():
        config_data[key] = value
    config_data["model"] = model_value
    config_data[MODEL_ROLE_WEAK] = weak_model_value
    config_data[MODEL_ROLE_EDITOR] = editor_model_value
    staged_model_settings: tuple[Path, bool, str | None] | None = None
    if parsed_context_size is not None and not _has_model_settings_file(config_data):
        model_settings_path = _managed_model_settings_path(store, profile_name)
        try:
            existed, previous = _stage_model_settings_write(model_settings_path, parsed_context_size)
        except OSError as exc:
            typer.echo(f"Unable to write model settings file: {exc}")
            raise typer.Exit(code=1) from exc
        staged_model_settings = (model_settings_path, existed, previous)
        config_data["model-settings-file"] = str(model_settings_path)
    if env_entries:
        config_data["set-env"] = env_entries

    try:
        profile, validation = store.save_profile(profile_name, config_data)
    except ProfileValidationError as exc:
        if staged_model_settings:
            path, existed, previous = staged_model_settings
            _rollback_staged_model_settings(path, existed, previous)
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    except ProfileError as exc:
        if staged_model_settings:
            path, existed, previous = staged_model_settings
            _rollback_staged_model_settings(path, existed, previous)
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
    context_size: int | None = typer.Option(
        None,
        "--context-size",
        help="Model context window size in tokens (writes num_ctx via model settings).",
    ),
    qol_preset: str | None = typer.Option(
        None,
        "--qol-preset",
        help=f"Apply QoL defaults. Choices: {', '.join(QOL_PRESETS)}.",
    ),
    weak_model: str | None = typer.Option(None, "--weak-model", help="Set aider weak model."),
    clear_weak_model: bool = typer.Option(False, "--clear-weak-model", help="Clear weak model override."),
    editor_model: str | None = typer.Option(None, "--editor-model", help="Set aider editor model."),
    clear_editor_model: bool = typer.Option(False, "--clear-editor-model", help="Clear editor model override."),
    edit_format: str | None = typer.Option(None, "--edit-format", help="Set aider edit format."),
    editor_edit_format: str | None = typer.Option(None, "--editor-edit-format", help="Set editor edit format."),
    reasoning_effort: str | None = typer.Option(None, "--reasoning-effort", help="Set reasoning effort."),
    thinking_tokens: str | None = typer.Option(None, "--thinking-tokens", help="Set thinking token budget."),
    max_chat_history_tokens: int | None = typer.Option(
        None,
        "--max-chat-history-tokens",
        help="Set max chat history tokens.",
    ),
    cache_prompts: bool | None = typer.Option(None, "--cache-prompts/--no-cache-prompts"),
    map_tokens: int | None = typer.Option(None, "--map-tokens", help="Set repo map token budget."),
    map_refresh: str | None = typer.Option(None, "--map-refresh", help="Set repo map refresh mode."),
    auto_commits: bool | None = typer.Option(None, "--auto-commits/--no-auto-commits"),
    dirty_commits: bool | None = typer.Option(None, "--dirty-commits/--no-dirty-commits"),
    auto_lint: bool | None = typer.Option(None, "--auto-lint/--no-auto-lint"),
    auto_test: bool | None = typer.Option(None, "--auto-test/--no-auto-test"),
    test_cmd: str | None = typer.Option(None, "--test-cmd", help="Set test command."),
    notifications: bool | None = typer.Option(None, "--notifications/--no-notifications"),
    notifications_command: str | None = typer.Option(None, "--notifications-command", help="Set notify command."),
    suggest_shell_commands: bool | None = typer.Option(
        None,
        "--suggest-shell-commands/--no-suggest-shell-commands",
    ),
    fancy_input: bool | None = typer.Option(None, "--fancy-input/--no-fancy-input"),
    multiline: bool | None = typer.Option(None, "--multiline/--no-multiline"),
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
    parsed_context_size = context_size
    old_main_model = _normalize_optional_model(profile.config.get("model"))
    next_main_model = old_main_model
    main_changed = False

    if interactive:
        _interactive_edit_profile_flow(profile.name)
        return

    if weak_model and clear_weak_model:
        raise typer.BadParameter("Use either --weak-model or --clear-weak-model, not both.")
    if editor_model and clear_editor_model:
        raise typer.BadParameter("Use either --editor-model or --clear-editor-model, not both.")

    try:
        parsed_context_size = _parse_context_size(parsed_context_size)
    except typer.BadParameter as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    if model_value:
        try:
            next_main_model = normalize_ollama_model(model_value)
        except ValueError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1) from exc
        data["model"] = next_main_model
        main_changed = next_main_model != old_main_model

    if qol_preset:
        try:
            data.update(_build_qol_preset(qol_preset))
        except typer.BadParameter as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1) from exc

    _apply_qol_settings(
        data,
        weak_model=weak_model,
        editor_model=editor_model,
        edit_format=edit_format,
        editor_edit_format=editor_edit_format,
        reasoning_effort=reasoning_effort,
        thinking_tokens=thinking_tokens,
        max_chat_history_tokens=max_chat_history_tokens,
        cache_prompts=cache_prompts,
        map_tokens=map_tokens,
        map_refresh=map_refresh,
        auto_commits=auto_commits,
        dirty_commits=dirty_commits,
        auto_lint=auto_lint,
        auto_test=auto_test,
        test_cmd=test_cmd,
        notifications=notifications,
        notifications_command=notifications_command,
        suggest_shell_commands=suggest_shell_commands,
        fancy_input=fancy_input,
        multiline=multiline,
    )
    try:
        if weak_model is not None:
            data[MODEL_ROLE_WEAK] = normalize_ollama_model(weak_model)
        elif clear_weak_model:
            data.pop(MODEL_ROLE_WEAK, None)
        elif main_changed and _role_model_follows_main(profile.config, MODEL_ROLE_WEAK, old_main_model):
            data[MODEL_ROLE_WEAK] = next_main_model

        if editor_model is not None:
            data[MODEL_ROLE_EDITOR] = normalize_ollama_model(editor_model)
        elif clear_editor_model:
            data.pop(MODEL_ROLE_EDITOR, None)
        elif main_changed and _role_model_follows_main(profile.config, MODEL_ROLE_EDITOR, old_main_model):
            data[MODEL_ROLE_EDITOR] = next_main_model
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

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
        ollama_api_base = selected.url
    if clear_server:
        clear_ollama_api_base = True

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

    staged_model_settings: list[tuple[Path, bool, str | None]] = []
    cleanup_managed_model_settings: list[Path] = []
    old_model_settings_value = profile.config.get("model-settings-file")
    old_managed_path = Path(old_model_settings_value).expanduser() if _is_managed_model_settings_path(store, old_model_settings_value) else None
    if parsed_context_size is not None:
        model_settings_path = _managed_model_settings_path(store, target_name)
        try:
            existed, previous = _stage_model_settings_write(model_settings_path, parsed_context_size)
        except OSError as exc:
            typer.echo(f"Unable to write model settings file: {exc}")
            raise typer.Exit(code=1) from exc
        staged_model_settings.append((model_settings_path, existed, previous))
        data["model-settings-file"] = str(model_settings_path)
        if old_managed_path and old_managed_path != model_settings_path:
            cleanup_managed_model_settings.append(old_managed_path)
    elif target_name != profile.name and old_managed_path:
        migrated_model_settings_path = _managed_model_settings_path(store, target_name)
        if migrated_model_settings_path != old_managed_path and old_managed_path.exists():
            try:
                existed, previous = (
                    migrated_model_settings_path.exists(),
                    migrated_model_settings_path.read_text(encoding="utf-8")
                    if migrated_model_settings_path.exists()
                    else None,
                )
                migrated_model_settings_path.parent.mkdir(parents=True, exist_ok=True)
                migrated_model_settings_path.write_text(old_managed_path.read_text(encoding="utf-8"), encoding="utf-8")
            except OSError as exc:
                typer.echo(f"Unable to migrate managed model settings file: {exc}")
                raise typer.Exit(code=1) from exc
            staged_model_settings.append((migrated_model_settings_path, existed, previous))
            data["model-settings-file"] = str(migrated_model_settings_path)
            cleanup_managed_model_settings.append(old_managed_path)

    try:
        updated, validation = store.save_profile(target_name, data, previous_path=profile.path)
    except ProfileError as exc:
        for path, existed, previous in reversed(staged_model_settings):
            _rollback_staged_model_settings(path, existed, previous)
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    for stale_path in cleanup_managed_model_settings:
        stale_path.unlink(missing_ok=True)

    _echo_profile_validation_note(validation.skipped, validation.message)
    typer.echo(f'Updated profile "{updated.name}" at {updated.path}')


@config_app.command("remove")
def config_remove(
    name: str = typer.Argument(..., help="Profile name"),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation prompt"),
) -> None:
    store = _profile_store()
    try:
        profile = store.get_profile(name)
    except ProfileError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    if not yes and not typer.confirm(f'Remove profile "{name}"?'):
        raise typer.Exit(code=0)
    try:
        removed = store.remove_profile(name)
    except ProfileError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    settings_value = profile.config.get("model-settings-file")
    if _is_managed_model_settings_path(store, settings_value):
        Path(settings_value).expanduser().unlink(missing_ok=True)
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
    if not _stdin_is_tty():
        raise typer.BadParameter(
            "Non-interactive mode requires a project path argument. Use `aider-aid project add <path> --name <name>`."
        )
    raw = _prompt_text("Project directory (absolute or relative)")
    if raw is None:
        raise typer.Exit(code=0)
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
    if name:
        project_name = name
    else:
        prompted_name = _prompt_text("Project name", default=project_path.name)
        if prompted_name is None:
            raise typer.Exit(code=0)
        project_name = prompted_name
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
        if not _stdin_is_tty():
            raise ProjectError(
                "No projects saved. Run `aider-aid project add <path> --name <name>` or pass --project."
            )
        typer.echo("No projects saved. Creating one now.")
        project_path = _prompt_project_path()
        project_name = _prompt_text("Project name", default=project_path.name)
        if project_name is None:
            raise typer.Exit(code=0)
        project_store.add_project(name=project_name, path=project_path)
        return project_name

    if not _stdin_is_tty():
        raise ProjectError("Non-interactive launch requires --project (name or 1-based index).")

    options = [f"{project.name} ({project.path})" for project in projects] + ["Create New..."]
    idx = _prompt_select_index("Choose project:", options)
    if idx == len(options) - 1:
        project_path = _prompt_project_path()
        project_name = _prompt_text("Project name", default=project_path.name)
        if project_name is None:
            raise typer.Exit(code=0)
        project_store.add_project(name=project_name, path=project_path)
        return project_name
    return projects[idx].name


def _select_or_create_profile(profile_store: ProfileStore) -> str:
    profiles = profile_store.list_profiles()
    if not profiles:
        if not _stdin_is_tty():
            raise ProfileError(
                "No profiles saved. Run `aider-aid config create <name> --model <model>` or pass --profile."
            )
        typer.echo("No profiles saved. Creating one now.")
        return _create_profile_interactive()

    if not _stdin_is_tty():
        raise ProfileError("Non-interactive launch requires --profile (name).")

    options = [f"{profile.name} ({profile.config.get('model', '(unset model)')})" for profile in profiles]
    options.append("Create New...")
    idx = _prompt_select_index("Choose profile:", options)
    if idx == len(options) - 1:
        return _create_profile_interactive()
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

        profile_name = profile or _select_or_create_profile(profile_store)
        selected_profile = profile_store.get_profile(profile_name)
    except (ProjectError, ProfileError, IndexError, ValueError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    extra_args = list(arg)
    if not extra_args and _stdin_is_tty():
        include_extra_args = ask_confirm("Add one-off extra aider args?", default=False)
        if include_extra_args is None:
            raise typer.Exit(code=0)
        if include_extra_args:
            raw = _prompt_text("One-off extra aider args", allow_empty=True)
            if raw is None:
                raise typer.Exit(code=0)
            if raw:
                try:
                    extra_args = shlex.split(raw)
                except ValueError as exc:
                    print_error(f"Invalid one-off extra aider args: {exc}")
                    raise typer.Exit(code=1) from exc

    final_args = list(extra_args)
    runtime_profile_path = selected_profile.path
    cleanup_runtime_profile = False
    runtime_model_settings_path: Path | None = None
    cleanup_runtime_model_settings = False
    sanitized_config = strip_aider_aid_metadata(selected_profile.config)
    legacy_server_name = selected_profile.config.get(AIDER_AID_OLLAMA_SERVER_KEY)
    if isinstance(legacy_server_name, str) and legacy_server_name.strip():
        try:
            server = server_store.get_server(legacy_server_name.strip())
            env_entries = get_set_env_entries(sanitized_config)
            has_api_base = any(entry.startswith("OLLAMA_API_BASE=") for entry in env_entries)
            if not has_api_base:
                env_entries = upsert_env_var(env_entries, "OLLAMA_API_BASE", server.url)
                set_set_env_entries(sanitized_config, env_entries)
                typer.echo(f'Using legacy server binding "{server.name}" => {server.url}')
        except OllamaServerError:
            typer.echo(
                f'Warning: legacy server binding "{legacy_server_name}" was not found in configured servers.'
            )
    if not _has_model_settings_file(sanitized_config):
        try:
            runtime_model_settings_path = _create_runtime_model_settings(DEFAULT_MODEL_CONTEXT_SIZE)
        except OSError as exc:
            typer.echo(f"Unable to create runtime model settings file: {exc}")
            raise typer.Exit(code=1) from exc
        sanitized_config["model-settings-file"] = str(runtime_model_settings_path)
        cleanup_runtime_model_settings = True
    if sanitized_config != selected_profile.config:
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".aider.conf.yml",
            prefix="aider-aid-runtime-",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            yaml.safe_dump(sanitized_config, tmp, sort_keys=False, allow_unicode=False)
            runtime_profile_path = Path(tmp.name)
            cleanup_runtime_profile = True

    try:
        cmd, rc = launch_aider(
            project_path=selected_project.path,
            profile_path=runtime_profile_path,
            extra_args=final_args,
            dry_run=dry_run,
        )
    finally:
        if cleanup_runtime_profile:
            runtime_profile_path.unlink(missing_ok=True)
        if cleanup_runtime_model_settings and runtime_model_settings_path:
            runtime_model_settings_path.unlink(missing_ok=True)

    typer.echo(f"Project: {selected_project.path}")
    typer.echo(f"Profile: {selected_profile.path}")
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
