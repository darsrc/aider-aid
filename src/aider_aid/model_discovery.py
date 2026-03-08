from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from aider_aid.shell import CommandResult, command_exists, run_command


def normalize_ollama_model(model: str) -> str:
    value = model.strip()
    if not value:
        raise ValueError("Model cannot be empty.")

    if value.startswith("ollama_chat/"):
        return value
    if value.startswith("ollama/"):
        return "ollama_chat/" + value.split("/", 1)[1]
    if "/" not in value:
        return "ollama_chat/" + value
    return value


def strip_ollama_prefix(model: str) -> str:
    if model.startswith("ollama_chat/"):
        return model.split("/", 1)[1]
    if model.startswith("ollama/"):
        return model.split("/", 1)[1]
    return model


def parse_ollama_list_output(output: str) -> list[str]:
    models: list[str] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Error:") or line.startswith("Warning:"):
            continue
        if line.upper().startswith("NAME "):
            continue
        first_col = line.split()[0]
        if first_col.upper() == "NAME":
            continue
        models.append(first_col)
    return models


def parse_aider_list_models_output(output: str) -> list[str]:
    models: list[str] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        value = line[2:].strip()
        if not value:
            continue
        try:
            models.append(normalize_ollama_model(value))
        except ValueError:
            continue
    return models


@dataclass(frozen=True)
class ModelDiscovery:
    installed_models: list[str] = field(default_factory=list)
    aider_known_models: list[str] = field(default_factory=list)
    combined_models: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _dedupe_keep_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def discover_ollama_models(
    *,
    run: Callable[..., CommandResult] = run_command,
    command_exists_fn: Callable[[str], bool] = command_exists,
) -> ModelDiscovery:
    warnings: list[str] = []
    installed: list[str] = []
    aider_known: list[str] = []

    if command_exists_fn("ollama"):
        result = run(["ollama", "list"])
        if result.returncode == 0:
            installed = [normalize_ollama_model(m) for m in parse_ollama_list_output(result.stdout)]
        else:
            err = (result.stderr or result.stdout).strip()
            warnings.append(f"Unable to list local Ollama models: {err}")
    else:
        warnings.append("Ollama CLI not found in PATH.")

    if command_exists_fn("aider"):
        result = run(["aider", "--list-models", "ollama"])
        if result.returncode == 0:
            aider_known = parse_aider_list_models_output(result.stdout)
        else:
            err = (result.stderr or result.stdout).strip()
            warnings.append(f"Unable to list aider-known Ollama models: {err}")
    else:
        warnings.append("aider binary not found in PATH.")

    combined = _dedupe_keep_order(installed + aider_known)
    return ModelDiscovery(
        installed_models=installed,
        aider_known_models=aider_known,
        combined_models=combined,
        warnings=warnings,
    )
