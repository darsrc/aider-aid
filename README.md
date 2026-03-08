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
aider-aid config create
aider-aid project add
aider-aid server add gpu-a http://gpu-a.internal:11434
aider-aid launch
aider-aid doctor
```

## Notes

- Profiles are saved under `~/.config/aider-aid/configs/`.
- Projects are saved in `~/.config/aider-aid/projects.config`.
- Named Ollama servers are saved in `~/.config/aider-aid/ollama_servers.config`.
- Launch uses native aider configs via `aider --config <profile-file>`.
