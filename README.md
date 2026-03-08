# aider-aid

`aider-aid` is a local-first helper around `aider` with three commands families:

- `config`: manage named aider config profiles.
- `project`: manage project directory shortcuts.
- `launch`: choose a project + profile and run `aider`.
- `doctor`: run setup checks for aider + Ollama workflows.

## Install from release (Alpha 1.0)

```bash
# Download the v1.0.0-alpha1 source archive from this repo's Releases page
tar -xzf aider-aid-1.0.0-alpha1.tar.gz
cd aider-aid-1.0.0-alpha1
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

## Install from source

```bash
git clone <repo-url>
cd aider-aid
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
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
