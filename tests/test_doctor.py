from pathlib import Path

import yaml

from aider_aid.doctor import run_doctor
from aider_aid.ollama_server_store import AIDER_AID_OLLAMA_SERVER_KEY, OllamaServerStore
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


def test_doctor_fails_on_missing_bound_server(tmp_path: Path):
    store = ProfileStore(
        config_root=tmp_path,
        run=lambda *a, **k: CommandResult(cmd=[], returncode=0, stdout="ok", stderr=""),
        command_exists_fn=lambda name: name == "aider",
    )
    store.ensure_dirs()
    profile_file = store.profiles_dir / "remote.aider.conf.yml"
    profile_file.write_text(
        yaml.safe_dump(
            {
                "name": "remote",
                "model": "ollama_chat/llama3",
                AIDER_AID_OLLAMA_SERVER_KEY: "gpu-cluster",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    empty_servers = OllamaServerStore(config_root=tmp_path)
    results = run_doctor(
        store,
        empty_servers,
        run=_runner,
        command_exists_fn=lambda name: name in {"aider", "ollama"},
        env={},
        probe_endpoint=lambda endpoint: (False, [], "down"),
    )
    assert any(item.id == "ollama.server.binding" and item.status == "fail" for item in results)
