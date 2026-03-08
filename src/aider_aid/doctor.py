from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Mapping

from aider_aid.model_discovery import normalize_ollama_model, strip_ollama_prefix
from aider_aid.ollama_server_store import (
    AIDER_AID_OLLAMA_SERVER_KEY,
    OllamaServerStore,
)
from aider_aid.profile_store import Profile, ProfileStore, get_set_env_entries, parse_set_env
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


def probe_ollama_endpoint(endpoint: str, timeout: float = 2.0) -> tuple[bool, list[str], str]:
    url = endpoint.rstrip("/") + "/api/tags"
    request = urllib.request.Request(url, method="GET")
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
    server_urls: Mapping[str, str],
) -> tuple[set[str], set[str], list[DoctorResult]]:
    endpoints: set[str] = set()
    configured_models: set[str] = set()
    warnings: list[DoctorResult] = []

    env_default_endpoint = env.get("OLLAMA_API_BASE", DEFAULT_OLLAMA_API_BASE)
    for profile in profiles:
        model = profile.config.get("model")
        if not isinstance(model, str) or not model.strip():
            continue

        if model.startswith("ollama/"):
            warnings.append(
                DoctorResult(
                    id="profile.model.prefix",
                    status="warn",
                    message=(
                        f'Profile "{profile.name}" uses "{model}". '
                        "Use ollama_chat/ for better aider compatibility."
                    ),
                    remediation=(
                        "Run `aider-aid config edit` and migrate the model to "
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
        configured_models.add(strip_ollama_prefix(normalized))

        profile_env = parse_set_env(get_set_env_entries(profile.config))
        server_name = profile.config.get(AIDER_AID_OLLAMA_SERVER_KEY)
        endpoint = env_default_endpoint
        if isinstance(server_name, str) and server_name.strip():
            endpoint = server_urls.get(server_name.strip(), "")
            if not endpoint:
                warnings.append(
                    DoctorResult(
                        id="ollama.server.binding",
                        status="fail",
                        message=(
                            f'Profile "{profile.name}" references unknown Ollama server '
                            f'"{server_name}".'
                        ),
                        remediation="Add the server via `aider-aid server add` or edit the profile.",
                    )
                )
                continue
        elif "OLLAMA_API_BASE" in profile_env:
            endpoint = profile_env["OLLAMA_API_BASE"]
        endpoints.add(endpoint)

    if configured_models and not endpoints:
        endpoints.add(env_default_endpoint)

    return endpoints, configured_models, warnings


def run_doctor(
    profile_store: ProfileStore,
    server_store: OllamaServerStore | None = None,
    *,
    run: Callable[..., CommandResult] = run_command,
    command_exists_fn: Callable[[str], bool] = command_exists,
    env: Mapping[str, str] | None = None,
    probe_endpoint: Callable[[str], tuple[bool, list[str], str]] = probe_ollama_endpoint,
) -> list[DoctorResult]:
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

    profiles = profile_store.list_profiles()
    server_urls: dict[str, str] = {}
    if server_store is not None:
        for server in server_store.list_servers():
            server_urls[server.name] = server.url

    endpoints, configured_models, prefix_warnings = _collect_ollama_profile_data(
        profiles,
        env_map,
        server_urls,
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
