from pathlib import Path

import pytest

from aider_aid.profile_store import ProfileStore, ProfileValidationError
from aider_aid.shell import CommandResult


def _runner_ok(cmd, **kwargs):  # noqa: ANN001
    return CommandResult(cmd=list(cmd), returncode=0, stdout="aider 0.0", stderr="")


def _runner_fail(cmd, **kwargs):  # noqa: ANN001
    return CommandResult(cmd=list(cmd), returncode=1, stdout="", stderr="invalid config")


def test_save_profile_and_preserve_unknown_keys(tmp_path: Path):
    store = ProfileStore(
        config_root=tmp_path,
        run=_runner_ok,
        command_exists_fn=lambda name: name == "aider",
    )

    profile, validation = store.save_profile(
        "My Profile",
        {"model": "ollama/llama3", "foo": "bar", "set-env": ["A=1"]},
    )
    assert profile.config["foo"] == "bar"
    assert profile.config["model"] == "ollama_chat/llama3"
    assert validation.ok

    data = dict(profile.config)
    data["model"] = "llama3.1"
    updated, _ = store.save_profile("My Profile", data, previous_path=profile.path)
    assert updated.config["foo"] == "bar"
    assert updated.config["model"] == "ollama_chat/llama3.1"


def test_validation_failure_rejected(tmp_path: Path):
    store = ProfileStore(
        config_root=tmp_path,
        run=_runner_fail,
        command_exists_fn=lambda name: name == "aider",
    )

    with pytest.raises(ProfileValidationError):
        store.save_profile("Broken", {"model": "ollama/llama3"})
