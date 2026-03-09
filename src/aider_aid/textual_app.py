from __future__ import annotations

import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Select, Static

from aider_aid.doctor import DEFAULT_OLLAMA_API_BASE
from aider_aid import services

ScreenName = Literal["dashboard", "launch", "profiles", "projects", "servers", "doctor"]


@dataclass(frozen=True)
class ScreenAction:
    label: str
    kind: Literal["route", "op", "quit"]
    payload: str


@dataclass(frozen=True)
class ScreenModel:
    title: str
    description: str
    actions: list[ScreenAction]


NAV_TREE = (
    "Navigation Tree\n"
    "Dashboard\n"
    "├─ Launch\n"
    "│  ├─ Select Project\n"
    "│  ├─ Select Profile\n"
    "│  ├─ Extra Args (optional)\n"
    "│  └─ Confirm & Run\n"
    "├─ Profiles\n"
    "│  ├─ List / View\n"
    "│  ├─ Create\n"
    "│  ├─ Edit\n"
    "│  └─ Remove\n"
    "├─ Projects\n"
    "│  ├─ List\n"
    "│  ├─ Add\n"
    "│  ├─ Rename\n"
    "│  └─ Remove\n"
    "├─ Servers\n"
    "│  ├─ List\n"
    "│  ├─ Add\n"
    "│  ├─ Remove\n"
    "│  └─ Probe\n"
    "└─ Doctor\n"
    "   ├─ Run Checks\n"
    "   └─ Remediation View"
)


SCREENS: dict[ScreenName, ScreenModel] = {
    "dashboard": ScreenModel(
        title="Dashboard",
        description="Open a flow and complete it in-app. Textual mode keeps the full tree native.",
        actions=[
            ScreenAction("Launch Workflow", "route", "launch"),
            ScreenAction("Profiles", "route", "profiles"),
            ScreenAction("Projects", "route", "projects"),
            ScreenAction("Servers", "route", "servers"),
            ScreenAction("Doctor", "route", "doctor"),
            ScreenAction("Refresh Summary", "op", "dashboard_summary"),
            ScreenAction("Quit", "quit", ""),
        ],
    ),
    "launch": ScreenModel(
        title="Launch",
        description="Select project/profile, optional args, then run aider from inside this flow.",
        actions=[
            ScreenAction("Run Launch Workflow", "op", "launch_run"),
            ScreenAction("Dry-Run Launch Workflow", "op", "launch_dry_run"),
            ScreenAction("Back to Dashboard", "route", "dashboard"),
        ],
    ),
    "profiles": ScreenModel(
        title="Profiles",
        description="Manage model profiles, context settings, and endpoint/auth bindings.",
        actions=[
            ScreenAction("List Profiles", "op", "profiles_list"),
            ScreenAction("View Profile", "op", "profiles_view"),
            ScreenAction("Create Profile", "op", "profiles_create"),
            ScreenAction("Edit Profile", "op", "profiles_edit"),
            ScreenAction("Remove Profile", "op", "profiles_remove"),
            ScreenAction("Back to Dashboard", "route", "dashboard"),
        ],
    ),
    "projects": ScreenModel(
        title="Projects",
        description="Manage saved project shortcuts used by launch workflows.",
        actions=[
            ScreenAction("List Projects", "op", "projects_list"),
            ScreenAction("Add Project", "op", "projects_add"),
            ScreenAction("Rename Project", "op", "projects_rename"),
            ScreenAction("Remove Project", "op", "projects_remove"),
            ScreenAction("Back to Dashboard", "route", "dashboard"),
        ],
    ),
    "servers": ScreenModel(
        title="Servers",
        description="Manage named Ollama endpoints and probe availability/models.",
        actions=[
            ScreenAction("List Servers", "op", "servers_list"),
            ScreenAction("Add Server", "op", "servers_add"),
            ScreenAction("Remove Server", "op", "servers_remove"),
            ScreenAction("Probe Server", "op", "servers_probe"),
            ScreenAction("Back to Dashboard", "route", "dashboard"),
        ],
    ),
    "doctor": ScreenModel(
        title="Doctor",
        description="Run diagnostics and inspect remediation guidance.",
        actions=[
            ScreenAction("Run Doctor Checks", "op", "doctor_run"),
            ScreenAction("Back to Dashboard", "route", "dashboard"),
        ],
    ),
}

NAV_BUTTON_IDS = [
    "nav-dashboard",
    "nav-launch",
    "nav-profiles",
    "nav-projects",
    "nav-servers",
    "nav-doctor",
    "nav-quit",
]


def action_for_hotkey(screen_name: ScreenName, index: int) -> ScreenAction | None:
    if index <= 0:
        return None
    screen = SCREENS[screen_name]
    if index > len(screen.actions):
        return None
    return screen.actions[index - 1]


def build_help_text(screen_name: ScreenName) -> str:
    screen = SCREENS[screen_name]
    lines = [
        "Keyboard Help",
        "",
        "Global shortcuts:",
        "  h        Open/close this help",
        "  q        Quit",
        "  d/l/p/j/s/x  Jump to Dashboard/Launch/Profiles/Projects/Servers/Doctor",
        "  Left/Right    Focus Sidebar/Actions",
        "  Tab/Shift+Tab Focus next/previous item",
        "  Up/Down       Move within focused region",
        "  Home/End      Jump first/last in region",
        "  1-8           Run visible action by index",
        "  Esc           Back to Dashboard (or close help/modal)",
        "",
        f"Current screen: {screen.title}",
        "Screen actions:",
    ]
    for idx, action in enumerate(screen.actions, start=1):
        lines.append(f"  {idx}: {action.label}")
    return "\n".join(lines)


class TextPromptDialog(ModalScreen[str | None]):
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]
    CSS = """
    Screen {
      align: center middle;
    }
    #dialog {
      width: 72;
      max-width: 92%;
      padding: 1 2;
      border: round #6dc7ff;
      background: #102538;
    }
    #dialog-title {
      text-style: bold;
      margin-bottom: 1;
      color: #e9f8ff;
    }
    #dialog-label {
      margin-bottom: 1;
      color: #d5e8f5;
    }
    #dialog-error {
      margin-top: 1;
      color: #ffb5b5;
    }
    #dialog-buttons {
      margin-top: 1;
      height: auto;
    }
    """

    def __init__(
        self,
        *,
        title: str,
        label: str,
        default: str = "",
        placeholder: str = "",
        password: bool = False,
        allow_empty: bool = False,
    ) -> None:
        super().__init__()
        self._title = title
        self._label = label
        self._default = default
        self._placeholder = placeholder
        self._password = password
        self._allow_empty = allow_empty

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(self._title, id="dialog-title")
            yield Static(self._label, id="dialog-label")
            yield Input(
                value=self._default,
                placeholder=self._placeholder,
                password=self._password,
                id="dialog-input",
            )
            with Horizontal(id="dialog-buttons"):
                yield Button("Save", id="dialog-save", variant="success")
                yield Button("Cancel", id="dialog-cancel", variant="default")
            yield Static("", id="dialog-error")

    def on_mount(self) -> None:
        self.query_one("#dialog-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "dialog-input":
            self._submit()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "dialog-save":
            self._submit()
            return
        if event.button.id == "dialog-cancel":
            self.dismiss(None)

    def _submit(self) -> None:
        value = self.query_one("#dialog-input", Input).value.strip()
        if not self._allow_empty and not value:
            self.query_one("#dialog-error", Static).update("Value cannot be empty.")
            return
        self.dismiss(value)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ChoiceDialog(ModalScreen[str | None]):
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("enter", "save", "Select"),
    ]
    CSS = """
    Screen {
      align: center middle;
    }
    #dialog {
      width: 72;
      max-width: 92%;
      padding: 1 2;
      border: round #6dc7ff;
      background: #102538;
    }
    #dialog-title {
      text-style: bold;
      margin-bottom: 1;
      color: #e9f8ff;
    }
    #dialog-label {
      margin-bottom: 1;
      color: #d5e8f5;
    }
    #dialog-buttons {
      margin-top: 1;
      height: auto;
    }
    """

    def __init__(
        self,
        *,
        title: str,
        label: str,
        options: list[tuple[str, str]],
    ) -> None:
        super().__init__()
        self._title = title
        self._label = label
        self._options = options

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(self._title, id="dialog-title")
            yield Static(self._label, id="dialog-label")
            yield Select(self._options, prompt="Choose option", allow_blank=False, id="dialog-select")
            with Horizontal(id="dialog-buttons"):
                yield Button("Select", id="dialog-save", variant="success")
                yield Button("Cancel", id="dialog-cancel", variant="default")

    def on_mount(self) -> None:
        self.query_one("#dialog-select", Select).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "dialog-cancel":
            self.dismiss(None)
            return
        if event.button.id != "dialog-save":
            return
        self.action_save()

    def action_save(self) -> None:
        select = self.query_one("#dialog-select", Select)
        value = select.value
        if value in (Select.NULL, Select.BLANK):
            self.dismiss(None)
            return
        self.dismiss(str(value))

    def action_cancel(self) -> None:
        self.dismiss(None)


class ConfirmDialog(ModalScreen[bool]):
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("enter", "confirm", "Confirm"),
    ]
    CSS = """
    Screen {
      align: center middle;
    }
    #dialog {
      width: 72;
      max-width: 92%;
      padding: 1 2;
      border: round #6dc7ff;
      background: #102538;
    }
    #dialog-title {
      text-style: bold;
      margin-bottom: 1;
      color: #e9f8ff;
    }
    #dialog-message {
      color: #d5e8f5;
    }
    #dialog-buttons {
      margin-top: 1;
      height: auto;
    }
    """

    def __init__(self, *, title: str, message: str) -> None:
        super().__init__()
        self._title = title
        self._message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(self._title, id="dialog-title")
            yield Static(self._message, id="dialog-message")
            with Horizontal(id="dialog-buttons"):
                yield Button("Confirm", id="dialog-confirm", variant="success")
                yield Button("Cancel", id="dialog-cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "dialog-confirm":
            self.action_confirm()
            return
        if event.button.id == "dialog-cancel":
            self.action_cancel()

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class HelpDialog(ModalScreen[None]):
    BINDINGS = [
        ("h", "close_help", "Close"),
        ("escape", "close_help", "Close"),
        ("enter", "close_help", "Close"),
        ("q", "close_help", "Close"),
    ]

    CSS = """
    Screen {
      align: center middle;
    }
    #help-dialog {
      width: 90;
      max-width: 96%;
      height: 85%;
      border: round #6dc7ff;
      background: #102538;
      padding: 1 2;
    }
    #help-title {
      text-style: bold;
      color: #e9f8ff;
      margin-bottom: 1;
    }
    #help-body {
      height: 1fr;
      overflow-y: auto;
      color: #d5e8f5;
    }
    #help-close {
      margin-top: 1;
      width: 14;
    }
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def compose(self) -> ComposeResult:
        with Vertical(id="help-dialog"):
            yield Static("Keyboard Shortcuts", id="help-title")
            yield Static(self._text, id="help-body")
            yield Button("Close", id="help-close", variant="default")

    def on_mount(self) -> None:
        self.query_one("#help-close", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "help-close":
            self.action_close_help()

    def action_close_help(self) -> None:
        self.dismiss(None)


class AiderAidTextualApp(App[None]):
    CSS = """
    Screen {
      background: #091520;
      color: #eff6ff;
    }
    #shell {
      height: 1fr;
      padding: 1 2;
    }
    #sidebar {
      width: 34%;
      border: round #33c5ff;
      padding: 1;
      margin-right: 2;
      background: #0f2436;
    }
    #content {
      width: 66%;
      border: round #5be39a;
      padding: 1;
      background: #102a1e;
    }
    #brand {
      color: #7ee0ff;
      margin-bottom: 1;
    }
    .nav-btn {
      width: 1fr;
      margin-bottom: 1;
    }
    #tree {
      color: #9cdfff;
      margin-top: 1;
    }
    #screen-title {
      text-style: bold;
      color: #e7ffe5;
      margin-bottom: 1;
    }
    #screen-desc {
      color: #d8efe1;
      margin-bottom: 1;
    }
    #actions {
      width: 1fr;
      margin-bottom: 1;
    }
    .action-btn {
      width: 1fr;
      margin-bottom: 1;
    }
    .nav-btn:focus, .action-btn:focus {
      background: #1d4f73;
      color: #ffffff;
      text-style: bold;
    }
    .nav-btn.-active {
      border: round #7de59a;
    }
    #status {
      border: round #8fdab6;
      padding: 1;
      color: #d6f5e2;
      height: 1fr;
      overflow-y: auto;
    }
    """

    BINDINGS = [
        ("h", "toggle_help", "Help"),
        ("q", "quit", "Quit"),
        ("escape", "escape_context", "Back"),
        ("d", "switch_dashboard", "Dashboard"),
        ("l", "switch_launch", "Launch"),
        ("p", "switch_profiles", "Profiles"),
        ("j", "switch_projects", "Projects"),
        ("s", "switch_servers", "Servers"),
        ("x", "switch_doctor", "Doctor"),
        ("left", "focus_sidebar", "Sidebar"),
        ("right", "focus_actions", "Actions"),
        ("tab", "focus_next", "Next"),
        ("shift+tab", "focus_prev", "Prev"),
        ("up", "move_up", "Up"),
        ("down", "move_down", "Down"),
        ("home", "focus_first", "First"),
        ("end", "focus_last", "Last"),
        ("1", "run_action_1", "Action 1"),
        ("2", "run_action_2", "Action 2"),
        ("3", "run_action_3", "Action 3"),
        ("4", "run_action_4", "Action 4"),
        ("5", "run_action_5", "Action 5"),
        ("6", "run_action_6", "Action 6"),
        ("7", "run_action_7", "Action 7"),
        ("8", "run_action_8", "Action 8"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._screen_name: ScreenName = "dashboard"
        self._focus_region: Literal["sidebar", "actions"] = "actions"
        self._sidebar_focus_index = 0
        self._action_focus_index = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="shell"):
            with Vertical(id="sidebar"):
                yield Static("aider-aid\nNative Textual UI", id="brand")
                yield Button("Dashboard", id="nav-dashboard", classes="nav-btn")
                yield Button("Launch", id="nav-launch", classes="nav-btn")
                yield Button("Profiles", id="nav-profiles", classes="nav-btn")
                yield Button("Projects", id="nav-projects", classes="nav-btn")
                yield Button("Servers", id="nav-servers", classes="nav-btn")
                yield Button("Doctor", id="nav-doctor", classes="nav-btn")
                yield Button("Quit", id="nav-quit", classes="nav-btn")
                yield Static(NAV_TREE, id="tree")
            with Vertical(id="content"):
                yield Static("", id="screen-title")
                yield Static("", id="screen-desc")
                with Vertical(id="actions"):
                    for idx in range(8):
                        yield Button("", id=f"action-{idx}", classes="action-btn")
                yield Static("Ready.", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self._render_screen("dashboard")
        self._dashboard_summary()
        self._focus_actions()

    def action_switch_dashboard(self) -> None:
        if self._modal_open():
            return
        self._render_screen("dashboard")
        self._dashboard_summary()
        self._focus_actions()

    def action_switch_launch(self) -> None:
        if self._modal_open():
            return
        self._render_screen("launch")
        self._focus_actions()

    def action_switch_profiles(self) -> None:
        if self._modal_open():
            return
        self._render_screen("profiles")
        self._focus_actions()

    def action_switch_projects(self) -> None:
        if self._modal_open():
            return
        self._render_screen("projects")
        self._focus_actions()

    def action_switch_servers(self) -> None:
        if self._modal_open():
            return
        self._render_screen("servers")
        self._focus_actions()

    def action_switch_doctor(self) -> None:
        if self._modal_open():
            return
        self._render_screen("doctor")
        self._focus_actions()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "nav-dashboard":
            self._focus_region = "sidebar"
            self._sidebar_focus_index = 0
            self.action_switch_dashboard()
            return
        if button_id == "nav-launch":
            self._focus_region = "sidebar"
            self._sidebar_focus_index = 1
            self.action_switch_launch()
            return
        if button_id == "nav-profiles":
            self._focus_region = "sidebar"
            self._sidebar_focus_index = 2
            self.action_switch_profiles()
            return
        if button_id == "nav-projects":
            self._focus_region = "sidebar"
            self._sidebar_focus_index = 3
            self.action_switch_projects()
            return
        if button_id == "nav-servers":
            self._focus_region = "sidebar"
            self._sidebar_focus_index = 4
            self.action_switch_servers()
            return
        if button_id == "nav-doctor":
            self._focus_region = "sidebar"
            self._sidebar_focus_index = 5
            self.action_switch_doctor()
            return
        if button_id == "nav-quit":
            self.exit()
            return
        if not button_id.startswith("action-"):
            return

        raw_idx = button_id.replace("action-", "", 1)
        if not raw_idx.isdigit():
            return
        idx = int(raw_idx)
        screen = SCREENS[self._screen_name]
        if idx >= len(screen.actions):
            return
        self._focus_region = "actions"
        self._action_focus_index = idx
        self._execute_action(screen.actions[idx])

    def _render_screen(self, screen_name: ScreenName) -> None:
        self._screen_name = screen_name
        screen = SCREENS[screen_name]
        self.query_one("#screen-title", Static).update(screen.title)
        self.query_one("#screen-desc", Static).update(screen.description)

        buttons = [self.query_one(f"#action-{idx}", Button) for idx in range(8)]
        for idx, button in enumerate(buttons):
            if idx < len(screen.actions):
                button.label = screen.actions[idx].label
                button.display = True
            else:
                button.label = ""
                button.display = False
        self._sync_nav_active_state()
        self._clamp_focus_indices()
        self._focus_current_region()
        self._set_status(f"Ready: {screen.title}")

    def _set_status(self, message: str) -> None:
        self.query_one("#status", Static).update(message.strip() if message.strip() else "Ready.")

    def _modal_open(self) -> bool:
        return isinstance(self.screen, ModalScreen)

    def _nav_buttons(self) -> list[Button]:
        return [self.query_one(f"#{item_id}", Button) for item_id in NAV_BUTTON_IDS]

    def _visible_action_buttons(self) -> list[Button]:
        visible: list[Button] = []
        for idx in range(8):
            button = self.query_one(f"#action-{idx}", Button)
            if button.display:
                visible.append(button)
        return visible

    def _clamp_focus_indices(self) -> None:
        nav_count = len(self._nav_buttons())
        action_count = len(self._visible_action_buttons())
        if nav_count:
            self._sidebar_focus_index = max(0, min(self._sidebar_focus_index, nav_count - 1))
        else:
            self._sidebar_focus_index = 0
        if action_count:
            self._action_focus_index = max(0, min(self._action_focus_index, action_count - 1))
        else:
            self._action_focus_index = 0
            self._focus_region = "sidebar"

    def _focus_current_region(self) -> None:
        if self._modal_open():
            return
        if self._focus_region == "sidebar":
            buttons = self._nav_buttons()
            if not buttons:
                return
            buttons[self._sidebar_focus_index].focus()
            return
        buttons = self._visible_action_buttons()
        if not buttons:
            self._focus_region = "sidebar"
            self._focus_current_region()
            return
        buttons[self._action_focus_index].focus()

    def _focus_sidebar(self) -> None:
        if self._modal_open():
            return
        self._focus_region = "sidebar"
        self._clamp_focus_indices()
        self._focus_current_region()

    def _focus_actions(self) -> None:
        if self._modal_open():
            return
        self._focus_region = "actions"
        self._clamp_focus_indices()
        self._focus_current_region()

    def _sync_nav_active_state(self) -> None:
        nav_map = {
            "dashboard": "nav-dashboard",
            "launch": "nav-launch",
            "profiles": "nav-profiles",
            "projects": "nav-projects",
            "servers": "nav-servers",
            "doctor": "nav-doctor",
        }
        active_id = nav_map[self._screen_name]
        for button_id in NAV_BUTTON_IDS:
            button = self.query_one(f"#{button_id}", Button)
            if button_id == active_id:
                button.add_class("-active")
            else:
                button.remove_class("-active")

    def action_toggle_help(self) -> None:
        if self._modal_open():
            if isinstance(self.screen, HelpDialog):
                self.screen.dismiss(None)
            return
        self.push_screen(HelpDialog(build_help_text(self._screen_name)))

    def action_escape_context(self) -> None:
        if self._modal_open():
            if isinstance(self.screen, HelpDialog):
                self.screen.dismiss(None)
            return
        if self._screen_name != "dashboard":
            self.action_switch_dashboard()

    def action_focus_sidebar(self) -> None:
        self._focus_sidebar()

    def action_focus_actions(self) -> None:
        self._focus_actions()

    def action_focus_next(self) -> None:
        if self._modal_open():
            return
        if self._focus_region == "sidebar":
            buttons = self._nav_buttons()
            if buttons:
                self._sidebar_focus_index = (self._sidebar_focus_index + 1) % len(buttons)
        else:
            buttons = self._visible_action_buttons()
            if buttons:
                self._action_focus_index = (self._action_focus_index + 1) % len(buttons)
        self._focus_current_region()

    def action_focus_prev(self) -> None:
        if self._modal_open():
            return
        if self._focus_region == "sidebar":
            buttons = self._nav_buttons()
            if buttons:
                self._sidebar_focus_index = (self._sidebar_focus_index - 1) % len(buttons)
        else:
            buttons = self._visible_action_buttons()
            if buttons:
                self._action_focus_index = (self._action_focus_index - 1) % len(buttons)
        self._focus_current_region()

    def action_move_up(self) -> None:
        self.action_focus_prev()

    def action_move_down(self) -> None:
        self.action_focus_next()

    def action_focus_first(self) -> None:
        if self._modal_open():
            return
        if self._focus_region == "sidebar":
            self._sidebar_focus_index = 0
        else:
            self._action_focus_index = 0
        self._focus_current_region()

    def action_focus_last(self) -> None:
        if self._modal_open():
            return
        if self._focus_region == "sidebar":
            buttons = self._nav_buttons()
            if buttons:
                self._sidebar_focus_index = len(buttons) - 1
        else:
            buttons = self._visible_action_buttons()
            if buttons:
                self._action_focus_index = len(buttons) - 1
        self._focus_current_region()

    def _run_action_hotkey(self, index: int) -> None:
        if self._modal_open():
            return
        action = action_for_hotkey(self._screen_name, index)
        if action is None:
            self._set_status(f"No action bound to {index} on {SCREENS[self._screen_name].title}.")
            return
        self._focus_region = "actions"
        self._action_focus_index = index - 1
        self._focus_current_region()
        self._execute_action(action)

    def action_run_action_1(self) -> None:
        self._run_action_hotkey(1)

    def action_run_action_2(self) -> None:
        self._run_action_hotkey(2)

    def action_run_action_3(self) -> None:
        self._run_action_hotkey(3)

    def action_run_action_4(self) -> None:
        self._run_action_hotkey(4)

    def action_run_action_5(self) -> None:
        self._run_action_hotkey(5)

    def action_run_action_6(self) -> None:
        self._run_action_hotkey(6)

    def action_run_action_7(self) -> None:
        self._run_action_hotkey(7)

    def action_run_action_8(self) -> None:
        self._run_action_hotkey(8)

    def _execute_action(self, action: ScreenAction) -> None:
        if action.kind == "quit":
            self.exit()
            return
        if action.kind == "route":
            self._render_screen(action.payload)  # type: ignore[arg-type]
            if action.payload == "dashboard":
                self._dashboard_summary()
            return
        if action.kind != "op":
            return

        self.run_worker(
            self._run_operation(action.payload),
            name=f"op:{action.payload}",
            group="ops",
            exclusive=True,
            exit_on_error=False,
        )

    async def _run_operation(self, op: str) -> None:
        handlers: dict[str, Callable[[], object]] = {
            "dashboard_summary": self._dashboard_summary,
            "profiles_list": self._profiles_list,
            "profiles_view": self._profiles_view,
            "profiles_create": self._profiles_create,
            "profiles_edit": self._profiles_edit,
            "profiles_remove": self._profiles_remove,
            "projects_list": self._projects_list,
            "projects_add": self._projects_add,
            "projects_rename": self._projects_rename,
            "projects_remove": self._projects_remove,
            "servers_list": self._servers_list,
            "servers_add": self._servers_add,
            "servers_remove": self._servers_remove,
            "servers_probe": self._servers_probe,
            "doctor_run": self._doctor_run,
            "launch_run": lambda: self._launch_run(dry_run=False),
            "launch_dry_run": lambda: self._launch_run(dry_run=True),
        }
        handler = handlers.get(op)
        if handler is None:
            self._set_status(f"Unsupported action: {op}")
            return
        try:
            outcome = handler()
            if hasattr(outcome, "__await__"):
                await outcome  # type: ignore[misc]
        except Exception as exc:
            self._set_status(f"Error: {exc}")

    async def _ask_text(
        self,
        *,
        title: str,
        label: str,
        default: str = "",
        placeholder: str = "",
        password: bool = False,
        allow_empty: bool = False,
    ) -> str | None:
        result = await self.push_screen_wait(
            TextPromptDialog(
                title=title,
                label=label,
                default=default,
                placeholder=placeholder,
                password=password,
                allow_empty=allow_empty,
            )
        )
        if result is None:
            return None
        return str(result)

    async def _ask_choice(self, *, title: str, label: str, options: list[tuple[str, str]]) -> str | None:
        if not options:
            return None
        result = await self.push_screen_wait(ChoiceDialog(title=title, label=label, options=options))
        if result is None:
            return None
        return str(result)

    async def _ask_confirm(self, *, title: str, message: str) -> bool:
        result = await self.push_screen_wait(ConfirmDialog(title=title, message=message))
        return bool(result)

    async def _choose_endpoint_source(self) -> tuple[str, str | None] | None:
        servers = services.list_servers()
        options: list[tuple[str, str]] = [(f"Default endpoint ({DEFAULT_OLLAMA_API_BASE})", "__default__")]
        for name, url in servers:
            options.append((f"{name} ({url})", f"server:{name}"))
        options.append(("Custom endpoint", "__custom__"))
        choice = await self._ask_choice(title="Endpoint", label="Choose model source endpoint:", options=options)
        if choice is None:
            return None
        if choice == "__default__":
            return DEFAULT_OLLAMA_API_BASE, None
        if choice.startswith("server:"):
            server_name = choice.split(":", 1)[1]
            for name, url in servers:
                if name == server_name:
                    return url, url
            self._set_status(f'Unknown server "{server_name}".')
            return None
        custom = await self._ask_text(
            title="Custom Endpoint",
            label="Enter OLLAMA_API_BASE URL:",
            placeholder="http://host:11434",
        )
        if custom is None:
            return None
        custom_url = custom.strip()
        if not custom_url:
            self._set_status("Endpoint cannot be empty.")
            return None
        return custom_url, custom_url

    async def _choose_profile_name(self, *, title: str) -> str | None:
        profiles = services.list_profiles()
        if not profiles:
            self._set_status("No profiles found.")
            return None
        options = [(f"{profile.name} ({profile.config.get('model', '(unset model)')})", profile.name) for profile in profiles]
        return await self._ask_choice(title=title, label="Choose profile:", options=options)

    async def _choose_project_name(self, *, title: str) -> str | None:
        projects = services.list_projects()
        if not projects:
            self._set_status("No projects found.")
            return None
        options = [(f"{project.name} ({project.path})", project.name) for project in projects]
        return await self._ask_choice(title=title, label="Choose project:", options=options)

    async def _choose_server_name(self, *, title: str) -> str | None:
        servers = services.list_servers()
        if not servers:
            self._set_status("No servers found.")
            return None
        options = [(f"{name} ({url})", name) for name, url in servers]
        return await self._ask_choice(title=title, label="Choose server:", options=options)

    def _dashboard_summary(self) -> None:
        profiles = services.list_profiles()
        projects = services.list_projects()
        servers = services.list_servers()
        lines = [
            "Environment Summary",
            f"- Profiles: {len(profiles)}",
            f"- Projects: {len(projects)}",
            f"- Servers: {len(servers)}",
        ]
        if profiles:
            lines.append(f"- Active profile sample: {profiles[0].name}")
        if projects:
            lines.append(f"- Active project sample: {projects[0].name}")
        self._set_status("\n".join(lines))

    def _profiles_list(self) -> None:
        profiles = services.list_profiles()
        if not profiles:
            self._set_status("No profiles found.")
            return
        lines = ["Profiles"]
        for idx, profile in enumerate(profiles, start=1):
            lines.append(f"{idx}. {profile.name} | model={profile.config.get('model', '(unset)')} | {profile.path}")
        self._set_status("\n".join(lines))

    async def _profiles_view(self) -> None:
        name = await self._choose_profile_name(title="View Profile")
        if not name:
            return
        profile = services.get_profile(name)
        payload = yaml.safe_dump(profile.config, sort_keys=False, allow_unicode=False)
        self._set_status(f"Profile: {profile.name}\nFile: {profile.path}\n\n{payload}")

    async def _profiles_create(self) -> None:
        profile_name = await self._ask_text(title="Create Profile", label="Profile name:")
        if profile_name is None:
            return

        endpoint_selection = await self._choose_endpoint_source()
        if endpoint_selection is None:
            return
        endpoint_for_model, api_base_to_store = endpoint_selection

        api_key_raw = await self._ask_text(
            title="Create Profile",
            label="OLLAMA_API_KEY (optional):",
            allow_empty=True,
            password=True,
        )
        if api_key_raw is None:
            return
        api_key = api_key_raw.strip() or None

        try:
            models = services.fetch_models_from_endpoint(endpoint_for_model, api_key=api_key)
        except Exception as exc:
            self._set_status(f"Unable to fetch models: {exc}")
            return
        model_options = [(model, model) for model in models]
        model_options.append(("Enter model manually", "__manual__"))
        model_choice = await self._ask_choice(title="Create Profile", label="Choose model:", options=model_options)
        if model_choice is None:
            return
        if model_choice == "__manual__":
            manual_model = await self._ask_text(
                title="Create Profile",
                label="Model name:",
                placeholder="ollama_chat/llama3",
            )
            if manual_model is None:
                return
            selected_model = manual_model
        else:
            selected_model = model_choice

        weak_action = await self._ask_choice(
            title="Create Profile",
            label="Weak model:",
            options=[
                (f"Use main model ({selected_model})", "__main__"),
                (f"Choose from endpoint ({endpoint_for_model})", "__select__"),
                ("Enter model manually", "__manual__"),
            ],
        )
        if weak_action is None:
            return
        if weak_action == "__main__":
            weak_model = selected_model
        elif weak_action == "__select__":
            weak_options = [(model, model) for model in models]
            weak_choice = await self._ask_choice(title="Create Profile", label="Choose weak model:", options=weak_options)
            if weak_choice is None:
                return
            weak_model = weak_choice
        else:
            manual_weak = await self._ask_text(
                title="Create Profile",
                label="Weak model:",
                default=selected_model,
            )
            if manual_weak is None:
                return
            weak_model = manual_weak

        editor_action = await self._ask_choice(
            title="Create Profile",
            label="Editor model:",
            options=[
                (f"Use main model ({selected_model})", "__main__"),
                (f"Choose from endpoint ({endpoint_for_model})", "__select__"),
                ("Enter model manually", "__manual__"),
            ],
        )
        if editor_action is None:
            return
        if editor_action == "__main__":
            editor_model = selected_model
        elif editor_action == "__select__":
            editor_options = [(model, model) for model in models]
            editor_choice = await self._ask_choice(
                title="Create Profile",
                label="Choose editor model:",
                options=editor_options,
            )
            if editor_choice is None:
                return
            editor_model = editor_choice
        else:
            manual_editor = await self._ask_text(
                title="Create Profile",
                label="Editor model:",
                default=selected_model,
            )
            if manual_editor is None:
                return
            editor_model = manual_editor

        context_raw = await self._ask_text(
            title="Create Profile",
            label="Context size tokens:",
            default=str(services.DEFAULT_MODEL_CONTEXT_SIZE),
        )
        if context_raw is None:
            return
        if not context_raw.isdigit() or int(context_raw) <= 0:
            self._set_status("Context size must be a positive integer.")
            return
        context_size = int(context_raw)

        preset_options = [("No preset", "__none__")] + [(preset, preset) for preset in services.QOL_PRESETS]
        preset_choice = await self._ask_choice(title="Create Profile", label="QoL preset:", options=preset_options)
        if preset_choice is None:
            return
        qol_preset = None if preset_choice == "__none__" else preset_choice

        summary = [
            f"Name: {profile_name}",
            f"Model: {selected_model}",
            f"Weak model: {weak_model}",
            f"Editor model: {editor_model}",
            f"Context size: {context_size}",
            f"Endpoint: {api_base_to_store or DEFAULT_OLLAMA_API_BASE} {'(stored)' if api_base_to_store else '(default)'}",
            f"QoL preset: {qol_preset or '(none)'}",
        ]
        if not await self._ask_confirm(title="Confirm Create", message="\n".join(summary)):
            return

        try:
            result = services.create_profile(
                name=profile_name,
                model=selected_model,
                weak_model=weak_model,
                editor_model=editor_model,
                context_size=context_size,
                qol_preset=qol_preset,
                ollama_api_base=api_base_to_store,
                ollama_api_key=api_key,
            )
        except Exception as exc:
            self._set_status(f"Create failed: {exc}")
            return
        note = (
            f"\nValidation note: {result.validation.message}"
            if result.validation.skipped and result.validation.message
            else ""
        )
        self._set_status(f'Created profile "{result.profile.name}" at {result.profile.path}{note}')

    async def _profiles_edit(self) -> None:
        selected = await self._choose_profile_name(title="Edit Profile")
        if not selected:
            return
        profile = services.get_profile(selected)
        current_model = str(profile.config.get("model") or "ollama_chat/llama3.1")
        current_weak_model = str(profile.config.get("weak-model") or "(unset)")
        current_editor_model = str(profile.config.get("editor-model") or "(unset)")
        current_endpoint = services.extract_env_var(profile.config, "OLLAMA_API_BASE") or DEFAULT_OLLAMA_API_BASE
        current_api_key = services.extract_env_var(profile.config, "OLLAMA_API_KEY")
        current_context = services.read_config_context_size(profile.config)

        target_name = await self._ask_text(title="Edit Profile", label="Profile name:", default=profile.name)
        if target_name is None:
            return
        new_name = target_name if target_name != profile.name else None

        endpoint_action = await self._ask_choice(
            title="Edit Profile",
            label=f"Endpoint action (current: {current_endpoint}):",
            options=[
                ("Keep current endpoint", "__keep__"),
                ("Choose a different endpoint", "__change__"),
                ("Clear OLLAMA_API_BASE", "__clear__"),
            ],
        )
        if endpoint_action is None:
            return

        ollama_api_base: str | None = None
        clear_ollama_api_base = False
        endpoint_for_model = current_endpoint
        if endpoint_action == "__change__":
            endpoint_selection = await self._choose_endpoint_source()
            if endpoint_selection is None:
                return
            endpoint_for_model, api_base_to_store = endpoint_selection
            ollama_api_base = api_base_to_store
            clear_ollama_api_base = api_base_to_store is None
        elif endpoint_action == "__clear__":
            clear_ollama_api_base = True
            endpoint_for_model = DEFAULT_OLLAMA_API_BASE

        key_action = await self._ask_choice(
            title="Edit Profile",
            label="OLLAMA_API_KEY action:",
            options=[
                ("Keep current key", "__keep__"),
                ("Set/replace key", "__set__"),
                ("Clear key", "__clear__"),
            ],
        )
        if key_action is None:
            return
        ollama_api_key: str | None = None
        clear_ollama_api_key = False
        if key_action == "__set__":
            prompted_key = await self._ask_text(
                title="Edit Profile",
                label="OLLAMA_API_KEY:",
                password=True,
            )
            if prompted_key is None:
                return
            ollama_api_key = prompted_key
        elif key_action == "__clear__":
            clear_ollama_api_key = True

        key_for_model_lookup = current_api_key
        if key_action == "__set__":
            key_for_model_lookup = ollama_api_key
        elif key_action == "__clear__":
            key_for_model_lookup = None

        model_action = await self._ask_choice(
            title="Edit Profile",
            label=f"Model action (current: {current_model}):",
            options=[
                ("Keep current model", "__keep__"),
                (f"Choose from endpoint ({endpoint_for_model})", "__select__"),
                ("Enter model manually", "__manual__"),
            ],
        )
        if model_action is None:
            return
        model_value: str | None = None
        if model_action == "__select__":
            try:
                models = services.fetch_models_from_endpoint(endpoint_for_model, api_key=key_for_model_lookup)
            except Exception as exc:
                self._set_status(f"Unable to fetch models: {exc}")
                return
            options = [(model, model) for model in models]
            chosen = await self._ask_choice(title="Edit Profile", label="Choose model:", options=options)
            if chosen is None:
                return
            model_value = chosen
        elif model_action == "__manual__":
            manual = await self._ask_text(title="Edit Profile", label="Model:", default=current_model)
            if manual is None:
                return
            model_value = manual
        next_main_model = model_value or current_model

        weak_model_value: str | None = None
        clear_weak_model = False
        weak_action = await self._ask_choice(
            title="Edit Profile",
            label=f"Weak model action (current: {current_weak_model}):",
            options=[
                ("Keep current weak model", "__keep__"),
                (f"Use main model ({next_main_model})", "__main__"),
                (f"Choose from endpoint ({endpoint_for_model})", "__select__"),
                ("Enter model manually", "__manual__"),
                ("Clear weak model override", "__clear__"),
            ],
        )
        if weak_action is None:
            return
        if weak_action == "__main__":
            weak_model_value = next_main_model
        elif weak_action == "__select__":
            try:
                weak_models = services.fetch_models_from_endpoint(endpoint_for_model, api_key=key_for_model_lookup)
            except Exception as exc:
                self._set_status(f"Unable to fetch models: {exc}")
                return
            weak_options = [(item, item) for item in weak_models]
            weak_choice = await self._ask_choice(title="Edit Profile", label="Choose weak model:", options=weak_options)
            if weak_choice is None:
                return
            weak_model_value = weak_choice
        elif weak_action == "__manual__":
            manual_weak = await self._ask_text(title="Edit Profile", label="Weak model:", default=next_main_model)
            if manual_weak is None:
                return
            weak_model_value = manual_weak
        elif weak_action == "__clear__":
            clear_weak_model = True

        editor_model_value: str | None = None
        clear_editor_model = False
        editor_action = await self._ask_choice(
            title="Edit Profile",
            label=f"Editor model action (current: {current_editor_model}):",
            options=[
                ("Keep current editor model", "__keep__"),
                (f"Use main model ({next_main_model})", "__main__"),
                (f"Choose from endpoint ({endpoint_for_model})", "__select__"),
                ("Enter model manually", "__manual__"),
                ("Clear editor model override", "__clear__"),
            ],
        )
        if editor_action is None:
            return
        if editor_action == "__main__":
            editor_model_value = next_main_model
        elif editor_action == "__select__":
            try:
                editor_models = services.fetch_models_from_endpoint(endpoint_for_model, api_key=key_for_model_lookup)
            except Exception as exc:
                self._set_status(f"Unable to fetch models: {exc}")
                return
            editor_options = [(item, item) for item in editor_models]
            editor_choice = await self._ask_choice(
                title="Edit Profile",
                label="Choose editor model:",
                options=editor_options,
            )
            if editor_choice is None:
                return
            editor_model_value = editor_choice
        elif editor_action == "__manual__":
            manual_editor = await self._ask_text(title="Edit Profile", label="Editor model:", default=next_main_model)
            if manual_editor is None:
                return
            editor_model_value = manual_editor
        elif editor_action == "__clear__":
            clear_editor_model = True

        context_default = str(current_context) if current_context is not None else ""
        context_raw = await self._ask_text(
            title="Edit Profile",
            label="Context size tokens (blank keeps current):",
            default=context_default,
            allow_empty=True,
        )
        if context_raw is None:
            return
        context_size: int | None = None
        if context_raw.strip():
            if not context_raw.isdigit() or int(context_raw) <= 0:
                self._set_status("Context size must be a positive integer or blank.")
                return
            context_size = int(context_raw)

        preset_options = [("Keep current QoL settings", "__keep__")] + [(preset, preset) for preset in services.QOL_PRESETS]
        preset_choice = await self._ask_choice(title="Edit Profile", label="QoL preset:", options=preset_options)
        if preset_choice is None:
            return
        qol_preset = None if preset_choice == "__keep__" else preset_choice

        summary = [
            f"Profile: {profile.name} -> {target_name}",
            f"Model: {model_value or '(keep current)'}",
            f"Weak model: {'(clear)' if clear_weak_model else (weak_model_value or '(keep current)')}",
            f"Editor model: {'(clear)' if clear_editor_model else (editor_model_value or '(keep current)')}",
            f"Context size: {context_size if context_size is not None else '(keep current)'}",
            f"OLLAMA_API_BASE: {'(clear)' if clear_ollama_api_base else (ollama_api_base or '(keep current)')}",
            f"OLLAMA_API_KEY: {'(clear)' if clear_ollama_api_key else ('(set)' if ollama_api_key else '(keep current)')}",
            f"QoL preset: {qol_preset or '(keep current)'}",
        ]
        if not await self._ask_confirm(title="Confirm Edit", message="\n".join(summary)):
            return

        try:
            result = services.edit_profile(
                name=profile.name,
                new_name=new_name,
                model=model_value,
                weak_model=weak_model_value,
                clear_weak_model=clear_weak_model,
                editor_model=editor_model_value,
                clear_editor_model=clear_editor_model,
                context_size=context_size,
                qol_preset=qol_preset,
                ollama_api_base=ollama_api_base,
                clear_ollama_api_base=clear_ollama_api_base,
                ollama_api_key=ollama_api_key,
                clear_ollama_api_key=clear_ollama_api_key,
            )
        except Exception as exc:
            self._set_status(f"Edit failed: {exc}")
            return
        note = (
            f"\nValidation note: {result.validation.message}"
            if result.validation.skipped and result.validation.message
            else ""
        )
        self._set_status(f'Updated profile "{result.profile.name}" at {result.profile.path}{note}')

    async def _profiles_remove(self) -> None:
        selected = await self._choose_profile_name(title="Remove Profile")
        if not selected:
            return
        if not await self._ask_confirm(title="Confirm Remove", message=f'Remove profile "{selected}"?'):
            return
        try:
            removed_path = services.remove_profile(selected)
        except Exception as exc:
            self._set_status(f"Remove failed: {exc}")
            return
        self._set_status(f"Removed profile file: {removed_path}")

    def _projects_list(self) -> None:
        projects = services.list_projects()
        if not projects:
            self._set_status("No projects found.")
            return
        lines = ["Projects"]
        for idx, project in enumerate(projects, start=1):
            lines.append(f"{idx}. {project.name} | {project.path}")
        self._set_status("\n".join(lines))

    async def _projects_add(self) -> None:
        path_raw = await self._ask_text(
            title="Add Project",
            label="Project directory path:",
            placeholder="/absolute/path/to/repo",
        )
        if path_raw is None:
            return
        project_path = Path(path_raw).expanduser().resolve()
        name_default = project_path.name if project_path.name else "project"
        project_name = await self._ask_text(title="Add Project", label="Project name:", default=name_default)
        if project_name is None:
            return
        try:
            added = services.add_project(path=project_path, name=project_name)
        except Exception as exc:
            self._set_status(f"Add failed: {exc}")
            return
        self._set_status(f'Added project "{added.name}" => {added.path}')

    async def _projects_rename(self) -> None:
        selected = await self._choose_project_name(title="Rename Project")
        if not selected:
            return
        new_name = await self._ask_text(title="Rename Project", label="New project name:")
        if new_name is None:
            return
        try:
            renamed = services.rename_project(selected, new_name)
        except Exception as exc:
            self._set_status(f"Rename failed: {exc}")
            return
        self._set_status(f'Renamed project to "{renamed.name}"')

    async def _projects_remove(self) -> None:
        selected = await self._choose_project_name(title="Remove Project")
        if not selected:
            return
        if not await self._ask_confirm(title="Confirm Remove", message=f'Remove project "{selected}"?'):
            return
        try:
            removed = services.remove_project(selected)
        except Exception as exc:
            self._set_status(f"Remove failed: {exc}")
            return
        self._set_status(
            f'Removed project entry "{removed.name}" ({removed.path})\nNote: project folder was not deleted.'
        )

    def _servers_list(self) -> None:
        server_entries = services.list_servers()
        if not server_entries:
            self._set_status("No Ollama servers configured.")
            return
        lines = ["Ollama Servers"]
        for idx, (name, url) in enumerate(server_entries, start=1):
            lines.append(f"{idx}. {name} | {url}")
        self._set_status("\n".join(lines))

    async def _servers_add(self) -> None:
        name = await self._ask_text(title="Add Server", label="Server name:")
        if name is None:
            return
        url = await self._ask_text(
            title="Add Server",
            label="Server URL:",
            placeholder="http://host:11434",
        )
        if url is None:
            return
        replace = await self._ask_confirm(
            title="Add Server",
            message="Replace existing server if this name already exists?",
        )
        try:
            saved_name, saved_url = services.add_server(name=name, url=url, replace=replace)
        except Exception as exc:
            self._set_status(f"Add failed: {exc}")
            return
        self._set_status(f'Saved Ollama server "{saved_name}" => {saved_url}')

    async def _servers_remove(self) -> None:
        selected = await self._choose_server_name(title="Remove Server")
        if not selected:
            return
        if not await self._ask_confirm(title="Confirm Remove", message=f'Remove server "{selected}"?'):
            return
        try:
            removed_name, _ = services.remove_server(selected)
        except Exception as exc:
            self._set_status(f"Remove failed: {exc}")
            return
        self._set_status(f'Removed Ollama server "{removed_name}"')

    async def _servers_probe(self) -> None:
        server_entries = services.list_servers()
        options = [(f"{name} ({url})", f"server:{name}") for name, url in server_entries]
        options.append((f"Default endpoint ({DEFAULT_OLLAMA_API_BASE})", "__default__"))
        options.append(("Custom endpoint", "__custom__"))
        selection = await self._ask_choice(title="Probe Server", label="Choose endpoint to probe:", options=options)
        if selection is None:
            return

        endpoint = DEFAULT_OLLAMA_API_BASE
        if selection == "__custom__":
            custom = await self._ask_text(title="Probe Server", label="Custom endpoint URL:")
            if custom is None:
                return
            endpoint = custom
        elif selection.startswith("server:"):
            selected_name = selection.split(":", 1)[1]
            for name, url in server_entries:
                if name == selected_name:
                    endpoint = url
                    break

        api_key = await self._ask_text(
            title="Probe Server",
            label="OLLAMA_API_KEY (optional):",
            allow_empty=True,
            password=True,
        )
        if api_key is None:
            return
        cleaned_api_key = api_key.strip() or None

        ok, models, error = services.probe_endpoint(endpoint, api_key=cleaned_api_key)
        if not ok:
            self._set_status(f"Probe failed for {endpoint}\nError: {error}")
            return
        lines = [f"Probe success: {endpoint}", f"Models: {len(models)}"]
        for model in models[:25]:
            lines.append(f"- {model}")
        if len(models) > 25:
            lines.append(f"... and {len(models) - 25} more")
        self._set_status("\n".join(lines))

    async def _doctor_run(self) -> None:
        results = services.run_doctor_checks()
        status_map = {"pass": "[PASS]", "warn": "[WARN]", "fail": "[FAIL]"}
        lines: list[str] = []
        for result in results:
            prefix = status_map.get(result.status, "[INFO]")
            lines.append(f"{prefix} {result.id}: {result.message}")
            if result.details:
                lines.append(f"  details: {result.details}")
            if result.remediation:
                lines.append(f"  remediation: {result.remediation}")
        self._set_status("\n".join(lines))

    async def _launch_run(self, *, dry_run: bool) -> None:
        project_name = await self._choose_project_name(title="Launch: Project")
        if not project_name:
            return
        profile_name = await self._choose_profile_name(title="Launch: Profile")
        if not profile_name:
            return
        raw_args = await self._ask_text(
            title="Launch",
            label="One-off extra aider args (optional):",
            allow_empty=True,
            placeholder="--yes-always --no-pretty",
        )
        if raw_args is None:
            return
        extra_args: list[str] = []
        if raw_args.strip():
            try:
                extra_args = shlex.split(raw_args)
            except ValueError as exc:
                self._set_status(f"Invalid one-off extra args: {exc}")
                return

        if not dry_run:
            dry_run = await self._ask_confirm(
                title="Launch",
                message="Run as dry-run only?\nChoose Confirm for dry-run, Cancel for real launch.",
            )

        summary = [
            f"Project: {project_name}",
            f"Profile: {profile_name}",
            f"Extra args: {' '.join(extra_args) if extra_args else '(none)'}",
            f"Mode: {'dry-run' if dry_run else 'run'}",
        ]
        if not await self._ask_confirm(title="Confirm Launch", message="\n".join(summary)):
            return

        try:
            with self.suspend():
                result = services.launch_for_identifiers(
                    project_identifier=project_name,
                    profile_name=profile_name,
                    extra_args=extra_args,
                    dry_run=dry_run,
                )
        except Exception as exc:
            self._set_status(f"Launch failed: {exc}")
            return
        output = [
            f"Project: {result.project_path}",
            f"Profile: {result.profile_path}",
            f"Command: {result.command_display}",
            f"Exit code: {result.returncode}",
        ]
        self._set_status("\n".join(output))


def run_textual_app() -> int:
    app = AiderAidTextualApp()
    app.run()
    return 0
