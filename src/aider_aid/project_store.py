from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from aider_aid.paths import get_projects_file


class ProjectError(Exception):
    """Base project store error."""


class ProjectNotFoundError(ProjectError):
    """Raised when a project entry is missing."""


class CorruptProjectsFileError(ProjectError):
    """Raised when projects config is invalid JSON."""

    def __init__(self, message: str, backup_path: Path) -> None:
        super().__init__(message)
        self.backup_path = backup_path


@dataclass(frozen=True)
class Project:
    name: str
    path: Path


class ProjectStore:
    def __init__(self, *, config_root: Path | None = None) -> None:
        self._projects_file = get_projects_file(config_root)

    @property
    def projects_file(self) -> Path:
        return self._projects_file

    def _default_data(self) -> dict[str, object]:
        return {"version": 1, "projects": []}

    def _load_data(self) -> dict[str, object]:
        if not self._projects_file.exists():
            return self._default_data()

        try:
            data = json.loads(self._projects_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
            backup = self._projects_file.with_suffix(f".corrupt-{timestamp}.bak")
            shutil.copy2(self._projects_file, backup)
            self._write_data(self._default_data())
            raise CorruptProjectsFileError(
                f"Projects config was invalid JSON and has been reset: {exc}",
                backup_path=backup,
            ) from exc
        except OSError as exc:
            raise ProjectError(f"Unable to read projects config: {exc}") from exc

        if not isinstance(data, dict):
            raise ProjectError("Projects config must be a JSON object.")

        version = data.get("version", 1)
        projects = data.get("projects", [])
        if not isinstance(version, int):
            raise ProjectError("Projects config 'version' must be an integer.")
        if not isinstance(projects, list):
            raise ProjectError("Projects config 'projects' must be a list.")

        normalized: list[dict[str, str]] = []
        for item in projects:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            path = item.get("path")
            if not isinstance(name, str) or not isinstance(path, str):
                continue
            normalized.append({"name": name, "path": path})
        return {"version": version, "projects": normalized}

    def _write_data(self, data: dict[str, object]) -> None:
        self._projects_file.parent.mkdir(parents=True, exist_ok=True)
        self._projects_file.write_text(
            json.dumps(data, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )

    def list_projects(self) -> list[Project]:
        data = self._load_data()
        projects_data = data.get("projects", [])
        projects: list[Project] = []
        if not isinstance(projects_data, list):
            return projects
        for item in projects_data:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            path = item.get("path")
            if isinstance(name, str) and isinstance(path, str):
                projects.append(Project(name=name, path=Path(path)))
        return projects

    def _persist(self, projects: list[Project]) -> None:
        serialized = [{"name": item.name, "path": str(item.path)} for item in projects]
        self._write_data({"version": 1, "projects": serialized})

    def _assert_valid_new_project(self, projects: list[Project], name: str, path: Path) -> None:
        if not name.strip():
            raise ProjectError("Project name cannot be empty.")
        if not path.is_absolute():
            raise ProjectError("Project path must be absolute.")
        for existing in projects:
            if existing.name == name:
                raise ProjectError(f'Project name "{name}" already exists.')
            if existing.path == path:
                raise ProjectError(f'Project path "{path}" is already registered.')

    def add_project(self, *, name: str, path: Path) -> Project:
        projects = self.list_projects()
        resolved_path = path.expanduser().resolve()
        self._assert_valid_new_project(projects, name=name, path=resolved_path)
        project = Project(name=name, path=resolved_path)
        projects.append(project)
        self._persist(projects)
        return project

    def _resolve_identifier(self, identifier: str, projects: list[Project]) -> int:
        value = identifier.strip()
        if value.isdigit():
            idx = int(value) - 1
            if 0 <= idx < len(projects):
                return idx
            raise ProjectNotFoundError(f"Project index out of range: {value}")

        for idx, project in enumerate(projects):
            if project.name == value:
                return idx
        raise ProjectNotFoundError(f'Project "{value}" not found.')

    def rename_project(self, identifier: str, *, new_name: str) -> Project:
        projects = self.list_projects()
        idx = self._resolve_identifier(identifier, projects)
        if not new_name.strip():
            raise ProjectError("New project name cannot be empty.")

        for i, existing in enumerate(projects):
            if i == idx:
                continue
            if existing.name == new_name:
                raise ProjectError(f'Project name "{new_name}" already exists.')

        updated = Project(name=new_name, path=projects[idx].path)
        projects[idx] = updated
        self._persist(projects)
        return updated

    def get_project(self, identifier: str) -> Project:
        projects = self.list_projects()
        idx = self._resolve_identifier(identifier, projects)
        return projects[idx]

    def remove_project(self, identifier: str) -> Project:
        projects = self.list_projects()
        idx = self._resolve_identifier(identifier, projects)
        removed = projects.pop(idx)
        self._persist(projects)
        return removed
