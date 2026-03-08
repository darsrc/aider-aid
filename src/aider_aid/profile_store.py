from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from aider_aid.model_discovery import normalize_ollama_model
from aider_aid.paths import PROFILE_SUFFIX, ensure_config_dirs, get_profiles_dir
from aider_aid.shell import CommandResult, command_exists, run_command


class ProfileError(Exception):
    """Base profile error."""


class ProfileNotFoundError(ProfileError):
    """Raised when profile cannot be found."""


class ProfileValidationError(ProfileError):
    """Raised when aider rejects a generated config."""


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    skipped: bool
    message: str = ""


@dataclass(frozen=True)
class Profile:
    name: str
    path: Path
    config: dict[str, Any]


def slugify_profile_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    if not slug:
        raise ValueError("Profile name is empty after normalization.")
    return slug


def profile_path_to_slug(path: Path) -> str:
    name = path.name
    if name.endswith(PROFILE_SUFFIX):
        return name[: -len(PROFILE_SUFFIX)]
    return path.stem


def get_set_env_entries(config: dict[str, Any]) -> list[str]:
    raw = config.get("set-env")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        values: list[str] = []
        for item in raw:
            if isinstance(item, str):
                values.append(item)
        return values
    return []


def set_set_env_entries(config: dict[str, Any], entries: list[str]) -> None:
    if entries:
        config["set-env"] = entries
    elif "set-env" in config:
        del config["set-env"]


def parse_set_env(entries: list[str]) -> dict[str, str]:
    env_map: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        env_map[key.strip()] = value.strip()
    return env_map


def serialize_set_env(env_map: dict[str, str]) -> list[str]:
    return [f"{key}={value}" for key, value in env_map.items()]


def upsert_env_var(entries: list[str], key: str, value: str) -> list[str]:
    env_map = parse_set_env(entries)
    env_map[key] = value
    return serialize_set_env(env_map)


def delete_env_var(entries: list[str], key: str) -> list[str]:
    env_map = parse_set_env(entries)
    env_map.pop(key, None)
    return serialize_set_env(env_map)


def canonicalize_profile_model(config: dict[str, Any]) -> None:
    model = config.get("model")
    if not isinstance(model, str) or not model.strip():
        return
    config["model"] = normalize_ollama_model(model)


class ProfileStore:
    def __init__(
        self,
        *,
        config_root: Path | None = None,
        run: Callable[..., CommandResult] = run_command,
        command_exists_fn: Callable[[str], bool] = command_exists,
    ) -> None:
        self._profiles_dir = get_profiles_dir(config_root)
        self._run = run
        self._command_exists = command_exists_fn

    @property
    def profiles_dir(self) -> Path:
        return self._profiles_dir

    def ensure_dirs(self) -> None:
        ensure_config_dirs(self._profiles_dir.parent)

    def _get_profile_path(self, name: str) -> Path:
        slug = slugify_profile_name(name)
        return self._profiles_dir / f"{slug}{PROFILE_SUFFIX}"

    def _load_profile_from_path(self, path: Path) -> Profile:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ProfileError(f"Invalid YAML in {path}: {exc}") from exc
        except OSError as exc:
            raise ProfileError(f"Unable to read profile {path}: {exc}") from exc

        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ProfileError(f"Profile file must contain a YAML mapping: {path}")

        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            slug = profile_path_to_slug(path)
            name = slug.replace("-", " ").title()
        return Profile(name=name, path=path, config=data)

    def list_profiles(self) -> list[Profile]:
        if not self._profiles_dir.exists():
            return []
        profiles: list[Profile] = []
        for path in sorted(self._profiles_dir.glob(f"*{PROFILE_SUFFIX}")):
            try:
                profiles.append(self._load_profile_from_path(path))
            except ProfileError:
                continue
        return sorted(profiles, key=lambda item: item.name.lower())

    def get_profile(self, name_or_slug: str) -> Profile:
        value = name_or_slug.strip()
        if not value:
            raise ProfileNotFoundError("Profile name cannot be empty.")

        slug_lookup: str | None = None
        try:
            slug_lookup = slugify_profile_name(value)
            path_guess = self._profiles_dir / f"{slug_lookup}{PROFILE_SUFFIX}"
            if path_guess.exists():
                return self._load_profile_from_path(path_guess)
        except ValueError:
            slug_lookup = None

        for profile in self.list_profiles():
            if profile.name == value:
                return profile
            if slug_lookup and slugify_profile_name(profile.name) == slug_lookup:
                return profile
        raise ProfileNotFoundError(f'Profile "{value}" not found.')

    def validate_profile_file(self, profile_path: Path) -> ValidationResult:
        if not self._command_exists("aider"):
            return ValidationResult(
                ok=True,
                skipped=True,
                message="Skipped validation because aider is not installed.",
            )

        result = self._run(["aider", "--config", str(profile_path), "--version"])
        if result.returncode == 0:
            return ValidationResult(ok=True, skipped=False, message="")

        err = (result.stderr or result.stdout).strip()
        return ValidationResult(ok=False, skipped=False, message=err)

    def save_profile(
        self,
        name: str,
        config: dict[str, Any],
        *,
        previous_path: Path | None = None,
    ) -> tuple[Profile, ValidationResult]:
        self.ensure_dirs()
        data = dict(config)
        data["name"] = name
        canonicalize_profile_model(data)

        target_path = self._get_profile_path(name)
        temp_path = target_path.with_suffix(target_path.suffix + ".tmp")

        try:
            temp_path.write_text(
                yaml.safe_dump(data, sort_keys=False, allow_unicode=False),
                encoding="utf-8",
            )
        except OSError as exc:
            raise ProfileError(f"Unable to write profile temp file: {exc}") from exc

        validation = self.validate_profile_file(temp_path)
        if not validation.ok:
            temp_path.unlink(missing_ok=True)
            raise ProfileValidationError(
                "aider rejected this profile configuration:\n"
                f"{validation.message or 'No error details available.'}"
            )

        temp_path.replace(target_path)
        if previous_path and previous_path != target_path and previous_path.exists():
            previous_path.unlink()

        return self._load_profile_from_path(target_path), validation

    def remove_profile(self, name_or_slug: str) -> Path:
        profile = self.get_profile(name_or_slug)
        profile.path.unlink(missing_ok=True)
        return profile.path
