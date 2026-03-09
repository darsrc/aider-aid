import asyncio
import contextlib
import inspect
from pathlib import Path

import pytest

textual_app = pytest.importorskip("aider_aid.textual_app")


def test_textual_actions_are_native_only():
    for screen in textual_app.SCREENS.values():
        for action in screen.actions:
            assert action.kind in {"route", "op", "quit"}


def test_textual_nav_tree_mentions_full_flow():
    tree = textual_app.NAV_TREE
    assert "Launch" in tree
    assert "Profiles" in tree
    assert "Projects" in tree
    assert "Servers" in tree
    assert "Doctor" in tree
    assert "Confirm & Run" in tree


def test_textual_app_does_not_shell_out_for_menu_actions():
    source = inspect.getsource(textual_app)
    assert "subprocess.run" not in source


def test_action_for_hotkey_maps_visible_actions():
    action = textual_app.action_for_hotkey("profiles", 1)
    assert action is not None
    assert action.label == "List Profiles"
    assert textual_app.action_for_hotkey("profiles", 999) is None
    assert textual_app.action_for_hotkey("profiles", 0) is None


def test_build_help_text_includes_global_and_screen_actions():
    help_text = textual_app.build_help_text("servers")
    assert "Global shortcuts:" in help_text
    assert "h        Open/close this help" in help_text
    assert "Current screen: Servers" in help_text
    assert "1: List Servers" in help_text
    assert "4: Probe Server" in help_text


def test_bindings_include_keyboard_navigation_and_help():
    keymap = {entry[0] for entry in textual_app.AiderAidTextualApp.BINDINGS}
    assert "h" in keymap
    assert "left" in keymap
    assert "right" in keymap
    assert "up" in keymap
    assert "down" in keymap
    assert "1" in keymap
    assert "8" in keymap


def test_modal_blocks_global_navigation_actions(monkeypatch):
    app = textual_app.AiderAidTextualApp()
    monkeypatch.setattr(app, "_modal_open", lambda: True)
    assert app.check_action("move_down", ()) is False
    assert app.check_action("run_action_1", ()) is False
    assert app.check_action("focus_next", ()) is True
    assert app.check_action("focus_prev", ()) is True


def test_check_action_allows_global_navigation_when_not_modal(monkeypatch):
    app = textual_app.AiderAidTextualApp()
    monkeypatch.setattr(app, "_modal_open", lambda: False)
    assert app.check_action("move_down", ()) is True
    assert app.check_action("run_action_1", ()) is True


def test_launch_run_uses_selected_mode_without_extra_dry_run_prompt(monkeypatch):
    app = textual_app.AiderAidTextualApp()

    async def fake_choose_project_name(*, title):  # noqa: ANN202
        return "repo"

    async def fake_choose_profile_name(*, title):  # noqa: ANN202
        return "dev"

    async def fake_ask_text(**kwargs):  # noqa: ANN003, ANN202
        return ""

    confirm_titles: list[str] = []

    async def fake_ask_confirm(*, title, message):  # noqa: ANN202
        _ = message
        confirm_titles.append(title)
        return True

    captured: dict[str, object] = {}

    def fake_launch_for_identifiers(*, project_identifier, profile_name, extra_args, dry_run):  # noqa: ANN001
        captured["project"] = project_identifier
        captured["profile"] = profile_name
        captured["extra_args"] = list(extra_args)
        captured["dry_run"] = dry_run
        return textual_app.services.LaunchResult(
            command=["aider", "--config", "/tmp/dev.aider.conf.yml"],
            command_display="aider --config /tmp/dev.aider.conf.yml",
            returncode=0,
            project_path=Path("/tmp/repo"),
            profile_path=Path("/tmp/dev.aider.conf.yml"),
        )

    monkeypatch.setattr(app, "_choose_project_name", fake_choose_project_name)
    monkeypatch.setattr(app, "_choose_profile_name", fake_choose_profile_name)
    monkeypatch.setattr(app, "_ask_text", fake_ask_text)
    monkeypatch.setattr(app, "_ask_confirm", fake_ask_confirm)
    monkeypatch.setattr(app, "_set_status", lambda message: None)
    monkeypatch.setattr(app, "suspend", lambda: contextlib.nullcontext())
    monkeypatch.setattr(textual_app.services, "launch_for_identifiers", fake_launch_for_identifiers)

    asyncio.run(app._launch_run(dry_run=False))

    assert confirm_titles == ["Confirm Launch"]
    assert captured["project"] == "repo"
    assert captured["profile"] == "dev"
    assert captured["extra_args"] == []
    assert captured["dry_run"] is False
