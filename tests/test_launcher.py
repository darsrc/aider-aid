from pathlib import Path

from aider_aid.launcher import build_aider_command, format_shell_command


def test_build_aider_command_order():
    profile = Path("/tmp/dev.aider.conf.yml")
    cmd = build_aider_command(profile, extra_args=["--model", "ollama_chat/llama3", "--yes-always"])
    assert cmd == [
        "aider",
        "--config",
        "/tmp/dev.aider.conf.yml",
        "--model",
        "ollama_chat/llama3",
        "--yes-always",
    ]


def test_format_shell_command_quotes_spaces():
    cmd = ["aider", "--config", "/tmp/my profile.aider.conf.yml"]
    rendered = format_shell_command(cmd)
    assert rendered == "aider --config '/tmp/my profile.aider.conf.yml'"
