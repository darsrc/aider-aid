from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "aider-aid"
CONFIG_DIR_NAME = "configs"
PROJECTS_FILE_NAME = "projects.config"
OLLAMA_SERVERS_FILE_NAME = "ollama_servers.config"
PROFILE_SUFFIX = ".aider.conf.yml"


def get_config_root() -> Path:
    override = os.environ.get("AIDER_AID_CONFIG_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / APP_NAME


def get_profiles_dir(config_root: Path | None = None) -> Path:
    root = config_root or get_config_root()
    return root / CONFIG_DIR_NAME


def get_projects_file(config_root: Path | None = None) -> Path:
    root = config_root or get_config_root()
    return root / PROJECTS_FILE_NAME


def get_ollama_servers_file(config_root: Path | None = None) -> Path:
    root = config_root or get_config_root()
    return root / OLLAMA_SERVERS_FILE_NAME


def ensure_config_dirs(config_root: Path | None = None) -> None:
    profiles_dir = get_profiles_dir(config_root)
    profiles_dir.mkdir(parents=True, exist_ok=True)
