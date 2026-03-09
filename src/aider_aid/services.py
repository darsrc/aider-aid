from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from aider_aid.doctor import DEFAULT_OLLAMA_API_BASE, DoctorResult, probe_ollama_endpoint, run_doctor
from aider_aid.launcher import format_shell_command, launch_aider
from aider_aid.model_discovery import normalize_ollama_model
from aider_aid.ollama_server_store import AIDER_AID_OLLAMA_SERVER_KEY, OllamaServerStore
from aider_aid.profile_store import (
    Profile,
    ProfileError,
    ProfileStore,
    ProfileValidationError,
    ValidationResult,
    delete_env_var,
    get_set_env_entries,
    set_set_env_entries,
    slugify_profile_name,
    strip_aider_aid_metadata,
    upsert_env_var,
)
from aider_aid.project_store import Project, ProjectStore

DEFAULT_MODEL_CONTEXT_SIZE = 8192
MODEL_SETTINGS_SUFFIX = ".aider.model.settings.yml"
QOL_PRESETS = ("local-safe", "fast-iter", "strict-ci", "large-repo")
MODEL_ROLE_WEAK = "weak-model"
MODEL_ROLE_EDITOR = "editor-model"


@dataclass(frozen=True)
class ProfileMutationResult:
    profile: Profile
    validation: ValidationResult


@dataclass(frozen=True)
class LaunchResult:
    command: list[str]
    command_display: str
    returncode: int
    project_path: Path
    profile_path: Path


def profile_store() -> ProfileStore:
    return ProfileStore()


def project_store() -> ProjectStore:
    return ProjectStore()


def server_store() -> OllamaServerStore:
    return OllamaServerStore()


def list_profiles() -> list[Profile]:
    return profile_store().list_profiles()


def get_profile(name: str) -> Profile:
    return profile_store().get_profile(name)


def list_projects() -> list[Project]:
    return project_store().list_projects()


def list_servers() -> list[tuple[str, str]]:
    servers = server_store().list_servers()
    return [(server.name, server.url) for server in servers]


def _parse_context_size(context_size: int | None) -> int | None:
    if context_size is None:
        return None
    if context_size <= 0:
        raise ValueError("Context size must be greater than 0.")
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
    raise ValueError(f'Unknown QoL preset "{name}". Expected one of: {", ".join(QOL_PRESETS)}.')


def _parse_set_env(set_env: list[str]) -> list[str]:
    parsed: list[str] = []
    for entry in set_env:
        if "=" not in entry:
            raise ValueError(f'Invalid set-env "{entry}". Expected ENV=value.')
        key, value = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f'Invalid set-env "{entry}". Empty key.')
        parsed.append(f"{key}={value.strip()}")
    return parsed


def _parse_option_assignments(option_values: list[str]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for entry in option_values:
        if "=" not in entry:
            raise ValueError(f'Invalid option "{entry}". Expected key=value.')
        key, value = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f'Invalid option "{entry}". Empty key.')
        data[key] = yaml.safe_load(value)
    return data


def extract_env_var(config: dict[str, Any], key: str) -> str | None:
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


def read_config_context_size(config: dict[str, Any]) -> int | None:
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


def fetch_models_from_endpoint(endpoint: str, api_key: str | None = None) -> list[str]:
    ok, models, error = probe_ollama_endpoint(endpoint, api_key=api_key)
    if not ok:
        raise ValueError(
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
        raise ValueError(
            f"Ollama endpoint {endpoint} returned no models. Pull a model first (for example: ollama pull llama3)."
        )
    return deduped


def probe_endpoint(endpoint: str, api_key: str | None = None) -> tuple[bool, list[str], str]:
    return probe_ollama_endpoint(endpoint, api_key=api_key)


def create_profile(
    *,
    name: str,
    model: str,
    weak_model: str | None = None,
    editor_model: str | None = None,
    context_size: int = DEFAULT_MODEL_CONTEXT_SIZE,
    qol_preset: str | None = None,
    ollama_api_base: str | None = None,
    ollama_api_key: str | None = None,
    set_env: list[str] | None = None,
    option: list[str] | None = None,
) -> ProfileMutationResult:
    store = profile_store()
    profile_name = name.strip()
    if not profile_name:
        raise ValueError("Profile name cannot be empty.")

    parsed_context_size = _parse_context_size(context_size)
    model_value = normalize_ollama_model(model)
    weak_model_value = model_value if weak_model is None else normalize_ollama_model(weak_model)
    editor_model_value = model_value if editor_model is None else normalize_ollama_model(editor_model)

    env_entries = _parse_set_env(set_env or [])
    if ollama_api_base:
        env_entries = upsert_env_var(env_entries, "OLLAMA_API_BASE", ollama_api_base)
    if ollama_api_key:
        env_entries = upsert_env_var(env_entries, "OLLAMA_API_KEY", ollama_api_key)

    config_data: dict[str, Any] = {}
    if qol_preset:
        config_data.update(_build_qol_preset(qol_preset))
    for key, value in _parse_option_assignments(option or []).items():
        config_data[key] = value
    config_data["model"] = model_value
    config_data[MODEL_ROLE_WEAK] = weak_model_value
    config_data[MODEL_ROLE_EDITOR] = editor_model_value

    staged_model_settings: tuple[Path, bool, str | None] | None = None
    if parsed_context_size is not None and not _has_model_settings_file(config_data):
        model_settings_path = _managed_model_settings_path(store, profile_name)
        existed, previous = _stage_model_settings_write(model_settings_path, parsed_context_size)
        staged_model_settings = (model_settings_path, existed, previous)
        config_data["model-settings-file"] = str(model_settings_path)
    if env_entries:
        config_data["set-env"] = env_entries

    try:
        profile, validation = store.save_profile(profile_name, config_data)
    except (ProfileValidationError, ProfileError):
        if staged_model_settings:
            path, existed, previous = staged_model_settings
            _rollback_staged_model_settings(path, existed, previous)
        raise
    return ProfileMutationResult(profile=profile, validation=validation)


def edit_profile(
    *,
    name: str,
    new_name: str | None = None,
    model: str | None = None,
    weak_model: str | None = None,
    clear_weak_model: bool = False,
    editor_model: str | None = None,
    clear_editor_model: bool = False,
    context_size: int | None = None,
    qol_preset: str | None = None,
    ollama_api_base: str | None = None,
    clear_ollama_api_base: bool = False,
    ollama_api_key: str | None = None,
    clear_ollama_api_key: bool = False,
    set_env: list[str] | None = None,
    option: list[str] | None = None,
) -> ProfileMutationResult:
    store = profile_store()
    profile = store.get_profile(name)
    if weak_model and clear_weak_model:
        raise ValueError("Use either weak_model or clear_weak_model, not both.")
    if editor_model and clear_editor_model:
        raise ValueError("Use either editor_model or clear_editor_model, not both.")

    data = dict(profile.config)
    target_name = (new_name or profile.name).strip()
    if not target_name:
        raise ValueError("Profile name cannot be empty.")

    parsed_context_size = _parse_context_size(context_size)
    old_main_model = _normalize_optional_model(profile.config.get("model"))
    next_main_model = old_main_model
    main_changed = False
    if model:
        next_main_model = normalize_ollama_model(model)
        data["model"] = next_main_model
        main_changed = next_main_model != old_main_model
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
    if qol_preset:
        data.update(_build_qol_preset(qol_preset))

    env_entries = get_set_env_entries(data)
    for entry in _parse_set_env(set_env or []):
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
    for key, value in _parse_option_assignments(option or []).items():
        data[key] = value

    staged_model_settings: list[tuple[Path, bool, str | None]] = []
    cleanup_managed_model_settings: list[Path] = []
    old_model_settings_value = profile.config.get("model-settings-file")
    old_managed_path = (
        Path(old_model_settings_value).expanduser()
        if _is_managed_model_settings_path(store, old_model_settings_value)
        else None
    )

    if parsed_context_size is not None:
        model_settings_path = _managed_model_settings_path(store, target_name)
        existed, previous = _stage_model_settings_write(model_settings_path, parsed_context_size)
        staged_model_settings.append((model_settings_path, existed, previous))
        data["model-settings-file"] = str(model_settings_path)
        if old_managed_path and old_managed_path != model_settings_path:
            cleanup_managed_model_settings.append(old_managed_path)
    elif target_name != profile.name and old_managed_path:
        migrated = _managed_model_settings_path(store, target_name)
        if migrated != old_managed_path and old_managed_path.exists():
            existed = migrated.exists()
            previous = migrated.read_text(encoding="utf-8") if existed else None
            migrated.parent.mkdir(parents=True, exist_ok=True)
            migrated.write_text(old_managed_path.read_text(encoding="utf-8"), encoding="utf-8")
            staged_model_settings.append((migrated, existed, previous))
            data["model-settings-file"] = str(migrated)
            cleanup_managed_model_settings.append(old_managed_path)

    try:
        updated, validation = store.save_profile(target_name, data, previous_path=profile.path)
    except ProfileError:
        for path, existed, previous in reversed(staged_model_settings):
            _rollback_staged_model_settings(path, existed, previous)
        raise

    for stale_path in cleanup_managed_model_settings:
        stale_path.unlink(missing_ok=True)

    return ProfileMutationResult(profile=updated, validation=validation)


def remove_profile(name: str) -> Path:
    store = profile_store()
    profile = store.get_profile(name)
    removed = store.remove_profile(name)
    settings_value = profile.config.get("model-settings-file")
    if _is_managed_model_settings_path(store, settings_value):
        Path(settings_value).expanduser().unlink(missing_ok=True)
    return removed


def add_project(path: Path, name: str) -> Project:
    store = project_store()
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"Directory does not exist: {resolved}")
    return store.add_project(name=name, path=resolved)


def rename_project(identifier: str, new_name: str) -> Project:
    return project_store().rename_project(identifier, new_name=new_name)


def remove_project(identifier: str) -> Project:
    return project_store().remove_project(identifier)


def add_server(name: str, url: str, *, replace: bool = False) -> tuple[str, str]:
    server = server_store().add_server(name=name, url=url, replace=replace)
    return server.name, server.url


def remove_server(name: str) -> tuple[str, str]:
    server = server_store().remove_server(name)
    return server.name, server.url


def run_doctor_checks() -> list[DoctorResult]:
    return run_doctor(profile_store(), server_store())


def launch_for_identifiers(
    *,
    project_identifier: str,
    profile_name: str,
    extra_args: list[str] | None = None,
    dry_run: bool = False,
) -> LaunchResult:
    p_store = profile_store()
    j_store = project_store()
    s_store = server_store()

    selected_project = j_store.get_project(project_identifier)
    selected_profile = p_store.get_profile(profile_name)

    final_args = list(extra_args or [])
    runtime_profile_path = selected_profile.path
    cleanup_runtime_profile = False
    runtime_model_settings_path: Path | None = None
    cleanup_runtime_model_settings = False

    sanitized_config = strip_aider_aid_metadata(selected_profile.config)
    legacy_server_name = selected_profile.config.get(AIDER_AID_OLLAMA_SERVER_KEY)
    if isinstance(legacy_server_name, str) and legacy_server_name.strip():
        server = s_store.get_server(legacy_server_name.strip())
        env_entries = get_set_env_entries(sanitized_config)
        has_api_base = any(entry.startswith("OLLAMA_API_BASE=") for entry in env_entries)
        if not has_api_base:
            env_entries = upsert_env_var(env_entries, "OLLAMA_API_BASE", server.url)
            set_set_env_entries(sanitized_config, env_entries)

    if not _has_model_settings_file(sanitized_config):
        runtime_model_settings_path = _create_runtime_model_settings(DEFAULT_MODEL_CONTEXT_SIZE)
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

    return LaunchResult(
        command=cmd,
        command_display=format_shell_command(cmd),
        returncode=rc,
        project_path=selected_project.path,
        profile_path=selected_profile.path,
    )
