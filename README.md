# aider-aid

`aider-aid` is a local-first helper around `aider` with three commands families:

- `config`: manage named aider config profiles.
- `project`: manage project directory shortcuts.
- `launch`: choose a project + profile and run `aider`.
- `doctor`: run setup checks for aider + Ollama workflows.

## Install from release (Alpha 1.0)

```bash
# 1) Download wheel from:
# https://github.com/darsrc/aider-aid/releases/tag/v1.0.0-alpha1
curl -L -o aider_aid-0.1.0-py3-none-any.whl \
  https://github.com/darsrc/aider-aid/releases/download/v1.0.0-alpha1/aider_aid-0.1.0-py3-none-any.whl
python -m venv .venv
source .venv/bin/activate
python -m pip install ./aider_aid-0.1.0-py3-none-any.whl
aider-aid --help
```

## Install from source

```bash
git clone https://github.com/darsrc/aider-aid.git
cd aider-aid
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
aider-aid --help
```

## Quick start

```bash
aider-aid
# (no args in a terminal opens interactive mode)

aider-aid config create
aider-aid config create my-profile --context-size 8192
aider-aid config create my-profile --model llama3 --qol-preset fast-iter
aider-aid project add
aider-aid server add gpu-a http://gpu-a.internal:11434
aider-aid launch
aider-aid doctor
aider-aid menu config
```

## Notes

- Profiles are saved under `~/.config/aider-aid/configs/`.
- Projects are saved in `~/.config/aider-aid/projects.config`.
- Named Ollama servers are saved in `~/.config/aider-aid/ollama_servers.config`.
- Launch uses native aider configs via `aider --config <profile-file>`.
- `config create` defaults model context size to 8192 tokens; use `--context-size` on create/edit to override.
- `config create/edit` supports `--qol-preset` and targeted QoL options (for example `--map-tokens`, `--auto-test`).
- Run no-arg interactive mode with `--ui-mode auto|classic|textual` (default `auto`).
- If Textual is unavailable in `--ui-mode auto`, aider-aid prints an install hint and falls back to classic interactive mode.
- In `--ui-mode textual`, Launch/Profiles/Projects/Servers/Doctor flows stay inside native Textual screens (final aider run uses terminal handoff).
- Textual mode supports keyboard-first operation: section jump keys (`d/l/p/j/s/x`), action hotkeys (`1-8`), focus controls (`left/right/up/down/home/end`), and `h` for the in-app shortcut help overlay.
- `menu` opens a focused interactive section directly (`main|config|projects|servers|doctor`).
