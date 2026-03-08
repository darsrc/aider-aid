import json
from pathlib import Path

import pytest

from aider_aid.project_store import ProjectError, ProjectStore


def test_project_add_remove_never_deletes_folder(tmp_path: Path):
    store = ProjectStore(config_root=tmp_path)
    project_dir = tmp_path / "repo"
    project_dir.mkdir()

    added = store.add_project(name="repo", path=project_dir)
    assert added.path == project_dir
    assert project_dir.exists()

    removed = store.remove_project("repo")
    assert removed.name == "repo"
    assert project_dir.exists()


def test_project_rejects_duplicate_name_and_path(tmp_path: Path):
    store = ProjectStore(config_root=tmp_path)
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()

    store.add_project(name="one", path=one)
    with pytest.raises(ProjectError):
        store.add_project(name="one", path=two)
    with pytest.raises(ProjectError):
        store.add_project(name="two", path=one)


def test_corrupt_projects_file_is_reset(tmp_path: Path):
    store = ProjectStore(config_root=tmp_path)
    store.projects_file.parent.mkdir(parents=True, exist_ok=True)
    store.projects_file.write_text("{not-json", encoding="utf-8")

    with pytest.raises(Exception):
        store.list_projects()

    data = json.loads(store.projects_file.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["projects"] == []
