from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CommandResult:
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str


def command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    check: bool = False,
    capture_output: bool = True,
) -> CommandResult:
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)

    completed = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        text=True,
        capture_output=capture_output,
        timeout=timeout,
        check=check,
    )
    return CommandResult(
        cmd=list(cmd),
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def run_interactive_command(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)

    completed = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        text=True,
    )
    return completed.returncode
