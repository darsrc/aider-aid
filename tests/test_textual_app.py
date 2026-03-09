import inspect

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
