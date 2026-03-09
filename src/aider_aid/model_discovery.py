from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from aider_aid.shell import CommandResult, command_exists, run_command

KNOWN_PROVIDER_PREFIXES = (
    "openai/",
    "anthropic/",
    "deepseek/",
    "google/",
    "gemini/",
    "groq/",
    "xai/",
    "openrouter/",
    "cohere/",
    "mistral/",
    "vertex_ai/",
    "bedrock/",
    "azure/",
)


def is_known_provider_model(model: str) -> bool:
    value = model.strip()
    if not value:
        return False
    return value.startswith(KNOWN_PROVIDER_PREFIXES)


def normalize_ollama_model(model: str) -> str:
    value = model.strip()
    if not value:
        raise ValueError("Model cannot be empty.")

    if value.startswith("ollama_chat/"):
        return value
    if value.startswith("ollama/"):
        return "ollama_chat/" + value.split("/", 1)[1]
    if value.startswith("hf.co/"):
        return "ollama_chat/" + value
    if "/" not in value:
        return "ollama_chat/" + value
    if is_known_provider_model(value):
        return value
    return "ollama_chat/" + value


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


@dataclass(frozen=True)
class ModelDiscovery:
    installed_models: list[str] = field(default_factory=list)
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

    if command_exists_fn("ollama"):
        result = run(["ollama", "list"])
        if result.returncode == 0:
            installed = [normalize_ollama_model(m) for m in parse_ollama_list_output(result.stdout)]
        else:
            err = (result.stderr or result.stdout).strip()
            warnings.append(f"Unable to list local Ollama models: {err}")
    else:
        warnings.append("Ollama CLI not found in PATH.")

    combined = _dedupe_keep_order(installed)
    return ModelDiscovery(
        installed_models=installed,
        combined_models=combined,
        warnings=warnings,
    )
