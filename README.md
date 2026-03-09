# aider-aid

`aider-aid` is a local-first companion for `aider` that manages profiles, projects, Ollama endpoints, and launch workflows through both classic interactive menus and a full Textual TUI.

## Highlights

- Full navigation tree in Textual TUI (`Dashboard`, `Launch`, `Profiles`, `Projects`, `Servers`, `Doctor`).
- Keyboard-first navigation (`Tab`, arrows, hotkeys `1-8`, section jumps, `h` for help).
- Profile management with model role support (`model`, `weak-model`, `editor-model`).
- Managed context settings (`model-settings-file`) with default 8k context for parity.
- Managed model metadata (`model-metadata-file`) for local/unknown models so aider receives explicit context/cost metadata.
- Optional `--ollama-only` profile guard to block non-Ollama model overrides at launch.
- Launch flow that works with both modern and legacy profiles (runtime sanitization and compatibility handling).

## Install from GitHub release

Download assets from the latest release page:

- https://github.com/darsrc/aider-aid/releases/latest

Example (wheel install):

```bash
VERSION=v0.2.0
curl -L -o aider_aid-0.2.0-py3-none-any.whl \
  "https://github.com/darsrc/aider-aid/releases/download/${VERSION}/aider_aid-0.2.0-py3-none-any.whl"
python -m pip install ./aider_aid-0.2.0-py3-none-any.whl
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
# no args in a TTY opens interactive mode

aider-aid config create my-profile --model llama3 --context-size 8192
aider-aid project add
aider-aid server add gpu-a http://gpu-a.internal:11434
aider-aid launch
aider-aid doctor
```

## UI modes

- `--ui-mode auto` (default): tries Textual first, falls back to classic menus.
- `--ui-mode textual`: Textual only.
- `--ui-mode classic`: classic interactive menus.

## Data locations

- Profiles: `~/.config/aider-aid/configs/`
- Projects: `~/.config/aider-aid/projects.config`
- Servers: `~/.config/aider-aid/ollama_servers.config`

## Development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]
pytest -q
```
