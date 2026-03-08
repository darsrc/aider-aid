from __future__ import annotations

import shlex
from pathlib import Path
from typing import Sequence

from aider_aid.shell import run_interactive_command


def build_aider_command(profile_path: Path, extra_args: Sequence[str] | None = None) -> list[str]:
    cmd = ["aider", "--config", str(profile_path)]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def format_shell_command(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def launch_aider(
    *,
    project_path: Path,
    profile_path: Path,
    extra_args: Sequence[str] | None = None,
    dry_run: bool = False,
) -> tuple[list[str], int]:
    cmd = build_aider_command(profile_path, extra_args=extra_args)
    if dry_run:
        return cmd, 0
    rc = run_interactive_command(cmd, cwd=project_path)
    return cmd, rc
