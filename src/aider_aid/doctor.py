from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import yaml

from aider_aid.model_discovery import is_known_provider_model, normalize_ollama_model, strip_ollama_prefix
from aider_aid.ollama_server_store import OllamaServerStore
from aider_aid.paths import PROFILE_SUFFIX
from aider_aid.profile_store import MODEL_ROLE_KEYS, Profile, ProfileStore, get_set_env_entries, parse_set_env
from aider_aid.shell import CommandResult, command_exists, run_command

DEFAULT_OLLAMA_API_BASE = "http://127.0.0.1:11434"
INSTALL_URL_AIDER = "https://aider.chat/docs/install.html"
INSTALL_URL_OLLAMA = "https://ollama.com/download"


@dataclass(frozen=True)
class DoctorResult:
    id: str
    status: str
    message: str
    details: str = ""
    remediation: str = ""


@dataclass(frozen=True)
class DoctorFixResult:
    updated_profiles: list[Path]
    unchanged_profiles: list[Path]
    skipped_profiles: list[Path]


def _model_needs_prefix_fix(model: str) -> bool:
    value = model.strip()
    if not value:
        return False
    if value.startswith(("ollama_chat/", "ollama/")):
        return value.startswith("ollama/")
    if is_known_provider_model(value):
        return False
    return True


def fix_profile_model_prefixes(profile_store: ProfileStore) -> DoctorFixResult:
    updated_profiles: list[Path] = []
    unchanged_profiles: list[Path] = []
    skipped_profiles: list[Path] = []

    for path in sorted(profile_store.profiles_dir.glob(f"*{PROFILE_SUFFIX}")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (yaml.YAMLError, OSError):
            skipped_profiles.append(path)
            continue

        if data is None:
            data = {}
        if not isinstance(data, dict):
            skipped_profiles.append(path)
            continue

        changed = False
        for role_key in MODEL_ROLE_KEYS:
            model = data.get(role_key)
            if not isinstance(model, str) or not model.strip():
                continue
            if not _model_needs_prefix_fix(model):
                continue
            normalized = normalize_ollama_model(model)
            if normalized == model:
                continue
            data[role_key] = normalized
            changed = True

        if not changed:
            unchanged_profiles.append(path)
            continue

        path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        updated_profiles.append(path)

    return DoctorFixResult(
        updated_profiles=updated_profiles,
        unchanged_profiles=unchanged_profiles,
        skipped_profiles=skipped_profiles,
    )


def probe_ollama_endpoint(
    endpoint: str,
    timeout: float = 2.0,
    api_key: str | None = None,
) -> tuple[bool, list[str], str]:
    url = endpoint.rstrip("/") + "/api/tags"
    request = urllib.request.Request(url, method="GET")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        return False, [], str(exc)
    except OSError as exc:
        return False, [], str(exc)

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        return False, [], f"Invalid JSON response: {exc}"

    models: list[str] = []
    for item in payload.get("models", []):
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name.strip():
            models.append(name.strip())
    return True, models, ""


def _collect_ollama_profile_data(
    profiles: list[Profile],
    env: Mapping[str, str],
) -> tuple[set[str], set[str], list[DoctorResult]]:
    endpoints: set[str] = set()
    configured_models: set[str] = set()
    warnings: list[DoctorResult] = []

    env_default_endpoint = env.get("OLLAMA_API_BASE", DEFAULT_OLLAMA_API_BASE)
    for profile in profiles:
        has_ollama_role_model = False
        for role_key in MODEL_ROLE_KEYS:
            model = profile.config.get(role_key)
            if not isinstance(model, str) or not model.strip():
                continue

            if _model_needs_prefix_fix(model):
                warnings.append(
                    DoctorResult(
                        id="profile.model.prefix",
                        status="warn",
                        message=(
                            f'Profile "{profile.name}" {role_key} uses "{model}". '
                            "Use ollama_chat/ for better aider compatibility."
                        ),
                        remediation=(
                            "Run `aider-aid doctor --fix` or `aider-aid config edit` and migrate the model to "
                            f'"{normalize_ollama_model(model)}".'
                        ),
                    )
                )

            try:
                normalized = normalize_ollama_model(model)
            except ValueError:
                continue
            if not normalized.startswith("ollama_chat/"):
                continue
            has_ollama_role_model = True
            configured_models.add(strip_ollama_prefix(normalized))

        if has_ollama_role_model:
            profile_env = parse_set_env(get_set_env_entries(profile.config))
            endpoint = profile_env.get("OLLAMA_API_BASE", env_default_endpoint)
            endpoints.add(endpoint)

    if configured_models and not endpoints:
        endpoints.add(env_default_endpoint)

    return endpoints, configured_models, warnings


def _check_profile_files_valid(profile_store: ProfileStore) -> DoctorResult:
    paths = sorted(profile_store.profiles_dir.glob(f"*{PROFILE_SUFFIX}"))
    if not paths:
        return DoctorResult(
            id="config.files.valid",
            status="pass",
            message="No profile files found.",
        )

    invalid: list[str] = []
    for path in paths:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (yaml.YAMLError, OSError) as exc:
            invalid.append(f"{path}: {exc}")
            continue
        if data is not None and not isinstance(data, dict):
            invalid.append(f"{path}: profile file must be a YAML mapping.")

    if invalid:
        return DoctorResult(
            id="config.files.valid",
            status="fail",
            message=f"{len(invalid)} invalid profile file(s) found.",
            details="\n".join(invalid),
            remediation="Fix or remove invalid profile files under ~/.config/aider-aid/configs/.",
        )
    return DoctorResult(
        id="config.files.valid",
        status="pass",
        message=f"Validated {len(paths)} profile file(s).",
    )


def _check_model_settings_files_exist(profiles: list[Profile]) -> DoctorResult:
    missing: list[str] = []
    for profile in profiles:
        model_settings_file = profile.config.get("model-settings-file")
        if not isinstance(model_settings_file, str) or not model_settings_file.strip():
            continue
        candidate = Path(model_settings_file).expanduser()
        if not candidate.exists():
            missing.append(f'{profile.name}: {candidate}')

    if missing:
        return DoctorResult(
            id="config.model_settings_file.exists",
            status="fail",
            message=f"{len(missing)} profile(s) reference missing model-settings files.",
            details="\n".join(missing),
            remediation=(
                "Run `aider-aid config edit <name> --context-size 8192` to regenerate managed settings files "
                "or update/remove model-settings-file in the profile."
            ),
        )
    return DoctorResult(
        id="config.model_settings_file.exists",
        status="pass",
        message="All referenced model-settings files exist.",
    )


def _check_textual_available() -> DoctorResult:
    try:
        import textual
    except Exception as exc:
        return DoctorResult(
            id="ui.textual.available",
            status="warn",
            message="Textual is not available; polished dashboard UI will fall back to classic mode.",
            details=str(exc),
            remediation="Install/upgrade with: python -m pip install --upgrade textual aider-aid",
        )
    version = getattr(textual, "__version__", "unknown")
    return DoctorResult(
        id="ui.textual.available",
        status="pass",
        message=f"Textual available (version {version}).",
    )


def run_doctor(
    profile_store: ProfileStore,
    server_store: OllamaServerStore | None = None,
    *,
    run: Callable[..., CommandResult] = run_command,
    command_exists_fn: Callable[[str], bool] = command_exists,
    env: Mapping[str, str] | None = None,
    probe_endpoint: Callable[[str], tuple[bool, list[str], str]] = probe_ollama_endpoint,
) -> list[DoctorResult]:
    _ = server_store
    env_map: Mapping[str, str] = env if env is not None else os.environ
    results: list[DoctorResult] = []

    if command_exists_fn("aider"):
        version = run(["aider", "--version"])
        if version.returncode == 0:
            results.append(
                DoctorResult(
                    id="aider.binary.available",
                    status="pass",
                    message=version.stdout.strip() or "aider is available.",
                )
            )
        else:
            err = (version.stderr or version.stdout).strip()
            results.append(
                DoctorResult(
                    id="aider.binary.available",
                    status="fail",
                    message="aider binary is present but failed to execute.",
                    details=err,
                    remediation=f"Reinstall aider: {INSTALL_URL_AIDER}",
                )
            )
    else:
        results.append(
            DoctorResult(
                id="aider.binary.available",
                status="fail",
                message="aider was not found in PATH.",
                remediation=f"Install aider: {INSTALL_URL_AIDER}",
            )
        )

    ollama_installed_models: set[str] = set()
    if command_exists_fn("ollama"):
        ollama_result = run(["ollama", "list"])
        if ollama_result.returncode == 0:
            for line in ollama_result.stdout.splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                name = parts[0]
                if name.upper() == "NAME":
                    continue
                ollama_installed_models.add(name)
            results.append(
                DoctorResult(
                    id="ollama.binary.available",
                    status="pass",
                    message=f"ollama CLI available ({len(ollama_installed_models)} local models listed).",
                )
            )
        else:
            err = (ollama_result.stderr or ollama_result.stdout).strip()
            results.append(
                DoctorResult(
                    id="ollama.binary.available",
                    status="warn",
                    message="ollama CLI found but `ollama list` failed.",
                    details=err,
                    remediation="Check Ollama daemon status and local permissions.",
                )
            )
    else:
        results.append(
            DoctorResult(
                id="ollama.binary.available",
                status="warn",
                message="ollama was not found in PATH.",
                remediation=f"Install Ollama for local model workflows: {INSTALL_URL_OLLAMA}",
            )
        )

    results.append(_check_profile_files_valid(profile_store))
    results.append(_check_textual_available())

    profiles = profile_store.list_profiles()
    results.append(_check_model_settings_files_exist(profiles))
    endpoints, configured_models, prefix_warnings = _collect_ollama_profile_data(
        profiles,
        env_map,
    )
    results.extend(prefix_warnings)

    endpoint_models: set[str] = set()
    if endpoints:
        for endpoint in sorted(endpoints):
            ok, models, err = probe_endpoint(endpoint)
            if ok:
                endpoint_models.update(models)
                results.append(
                    DoctorResult(
                        id="ollama.endpoint.reachable",
                        status="pass",
                        message=f"Ollama endpoint reachable: {endpoint}",
                        details=f"Discovered {len(models)} models from /api/tags.",
                    )
                )
            else:
                severity = "fail" if configured_models else "warn"
                results.append(
                    DoctorResult(
                        id="ollama.endpoint.reachable",
                        status=severity,
                        message=f"Ollama endpoint not reachable: {endpoint}",
                        details=err,
                        remediation=(
                            "Start Ollama and verify OLLAMA_API_BASE "
                            "(for example: http://127.0.0.1:11434)."
                        ),
                    )
                )

    if configured_models:
        installed = set(ollama_installed_models) | set(endpoint_models)
        missing = sorted(model for model in configured_models if model not in installed)
        if missing:
            pull_cmds = "\n".join(f"ollama pull {model}" for model in missing)
            results.append(
                DoctorResult(
                    id="ollama.model.available",
                    status="warn",
                    message=f"{len(missing)} configured model(s) are not installed locally.",
                    details=", ".join(missing),
                    remediation=f"Pull missing models:\n{pull_cmds}",
                )
            )
        else:
            results.append(
                DoctorResult(
                    id="ollama.model.available",
                    status="pass",
                    message="All configured Ollama profile models are installed.",
                )
            )

    return results
