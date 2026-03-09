from __future__ import annotations

from rich.console import Console
from rich.panel import Panel

import questionary
from questionary import Choice

CONSOLE = Console()


def show_banner() -> None:
    CONSOLE.print(
        Panel.fit(
            "[bold]aider-aid[/bold]\nInteractive mode",
            border_style="cyan",
        )
    )


def print_section(title: str, description: str | None = None) -> None:
    body = title if not description else f"{title}\n{description}"
    CONSOLE.print(Panel.fit(body, border_style="blue"))


def select_index(title: str, options: list[str]) -> int | None:
    if not options:
        raise ValueError("No options available.")
    choices = [Choice(title=option, value=idx) for idx, option in enumerate(options)]
    try:
        result = questionary.select(
            title,
            choices=choices,
            qmark="",
            pointer="›",
            use_shortcuts=True,
            instruction="(Use arrow keys + Enter)",
        ).ask()
    except (KeyboardInterrupt, EOFError):
        return None
    if result is None:
        return None
    return int(result)


def ask_text(
    label: str,
    *,
    default: str | None = None,
    allow_empty: bool = False,
    password: bool = False,
) -> str | None:
    kwargs: dict[str, object] = {}
    if default is not None and not password:
        kwargs["default"] = default

    if not allow_empty:
        kwargs["validate"] = lambda value: bool(value.strip()) or "Value cannot be empty."

    prompt = questionary.password if password else questionary.text
    try:
        result = prompt(
            label,
            qmark="",
            instruction="(Enter to submit)" if not password else None,
            **kwargs,
        ).ask()
    except (KeyboardInterrupt, EOFError):
        return None
    if result is None:
        return None
    value = result.strip()
    if not value and not allow_empty:
        return None
    return value


def ask_confirm(label: str, *, default: bool = False) -> bool | None:
    try:
        result = questionary.confirm(
            label,
            default=default,
            qmark="",
            instruction="(y/n)",
        ).ask()
    except (KeyboardInterrupt, EOFError):
        return None
    if result is None:
        return None
    return bool(result)


def print_info(message: str) -> None:
    CONSOLE.print(message)


def print_warning(message: str) -> None:
    CONSOLE.print(f"[yellow]{message}[/yellow]")


def print_error(message: str) -> None:
    CONSOLE.print(f"[red]{message}[/red]")
