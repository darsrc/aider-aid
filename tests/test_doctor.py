from pathlib import Path

import yaml

from aider_aid.doctor import run_doctor
from aider_aid.profile_store import ProfileStore
from aider_aid.shell import CommandResult


def _runner(cmd, **kwargs):  # noqa: ANN001
    if cmd[:2] == ["aider", "--version"]:
        return CommandResult(cmd=list(cmd), returncode=0, stdout="aider 0.86.2\n", stderr="")
    if cmd[:2] == ["ollama", "list"]:
        out = "NAME ID SIZE MODIFIED\nllama3 abc 4GB now\n"
        return CommandResult(cmd=list(cmd), returncode=0, stdout=out, stderr="")
    return CommandResult(cmd=list(cmd), returncode=0, stdout="", stderr="")


def test_doctor_warns_on_noncanonical_ollama_model(tmp_path: Path):
    store = ProfileStore(
        config_root=tmp_path,
        run=lambda *a, **k: CommandResult(cmd=[], returncode=0, stdout="ok", stderr=""),
        command_exists_fn=lambda name: name == "aider",
    )
    store.ensure_dirs()
    profile_file = store.profiles_dir / "dev.aider.conf.yml"
    profile_file.write_text(
        yaml.safe_dump({"name": "dev", "model": "ollama/llama3"}, sort_keys=False),
        encoding="utf-8",
    )

    results = run_doctor(
        store,
        run=_runner,
        command_exists_fn=lambda name: name in {"aider", "ollama"},
        env={},
        probe_endpoint=lambda endpoint: (True, ["llama3"], ""),
    )
    assert any(item.id == "profile.model.prefix" and item.status == "warn" for item in results)


def test_doctor_fails_when_aider_missing(tmp_path: Path):
    store = ProfileStore(config_root=tmp_path)
    results = run_doctor(
        store,
        run=_runner,
        command_exists_fn=lambda name: False,
        env={},
        probe_endpoint=lambda endpoint: (False, [], "nope"),
    )
    assert any(item.id == "aider.binary.available" and item.status == "fail" for item in results)


def test_doctor_flags_missing_model_settings_file(tmp_path: Path):
    store = ProfileStore(
        config_root=tmp_path,
        run=lambda *a, **k: CommandResult(cmd=[], returncode=0, stdout="ok", stderr=""),
        command_exists_fn=lambda name: False,
    )
    store.ensure_dirs()
    profile_file = store.profiles_dir / "dev.aider.conf.yml"
    profile_file.write_text(
        yaml.safe_dump(
            {
                "name": "dev",
                "model": "ollama_chat/llama3",
                "model-settings-file": str(tmp_path / "missing.model.settings.yml"),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    results = run_doctor(
        store,
        run=_runner,
        command_exists_fn=lambda name: False,
        env={},
        probe_endpoint=lambda endpoint: (False, [], "nope"),
    )
    assert any(item.id == "config.model_settings_file.exists" and item.status == "fail" for item in results)


def test_doctor_flags_invalid_profile_file(tmp_path: Path):
    store = ProfileStore(config_root=tmp_path)
    store.ensure_dirs()
    bad_profile = store.profiles_dir / "bad.aider.conf.yml"
    bad_profile.write_text("{not-valid", encoding="utf-8")

    results = run_doctor(
        store,
        run=_runner,
        command_exists_fn=lambda name: False,
        env={},
        probe_endpoint=lambda endpoint: (False, [], "nope"),
    )
    assert any(item.id == "config.files.valid" and item.status == "fail" for item in results)


def test_doctor_includes_role_models_for_prefix_and_endpoint_checks(tmp_path: Path):
    store = ProfileStore(
        config_root=tmp_path,
        run=lambda *a, **k: CommandResult(cmd=[], returncode=0, stdout="ok", stderr=""),
        command_exists_fn=lambda name: False,
    )
    store.ensure_dirs()
    profile_file = store.profiles_dir / "dev.aider.conf.yml"
    profile_file.write_text(
        yaml.safe_dump(
            {
                "name": "dev",
                "model": "openai/gpt-4o",
                "weak-model": "ollama/llama3",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    results = run_doctor(
        store,
        run=_runner,
        command_exists_fn=lambda name: False,
        env={},
        probe_endpoint=lambda endpoint: (False, [], "unreachable"),
    )
    assert any(
        item.id == "profile.model.prefix" and item.status == "warn" and "weak-model" in item.message
        for item in results
    )
    assert any(item.id == "ollama.endpoint.reachable" and item.status == "fail" for item in results)
