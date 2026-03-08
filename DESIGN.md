# aider-aid Design Document

- Status: Draft v0.1
- Date: 2026-03-08
- Project: `aider-aid`
- Authors: Project team

## 1. Summary

`aider-aid` is a local-first helper for `aider` with three core capabilities:

1. Aider Configurator (primary): create and manage named aider model/config profiles.
2. Aider Launcher: launch aider with a selected project and profile.
3. Aider Doctor: detect common setup issues and guide fixes.

The first milestone prioritizes Ollama support and local management workflows.

## 2. Problem Statement

Running `aider` repeatedly across multiple repositories and model setups is operationally noisy:

- Users must remember model names, provider-specific options, and CLI flags.
- Users often work across many local project folders.
- Environment problems (missing `aider`, misconfigured local model runtime) are discovered late.

`aider-aid` reduces this friction with persistent local profiles, a project registry, and pre-flight checks.

## 3. Goals and Non-Goals

### 3.1 Goals

- Make model/profile setup fast and repeatable.
- Make project selection and launch consistent.
- Keep all state local to user config directories.
- Provide actionable diagnostics when setup is broken.
- Provide a strong first experience for Ollama-backed usage.

### 3.2 Non-Goals (v1)

- Cloud sync of profiles/projects.
- Remote project/workspace orchestration.
- Automatic dependency installation.
- Managing secrets in external vaults.

## 4. Primary Users

- Solo developers running aider locally.
- Developers using Ollama models on their own machines.
- Users with many repositories and repeated launch patterns.

## 5. Scope

### 5.1 In Scope (v1)

- Profile management under `~/.config/aider-aid/configs/`.
- Project registry management in `~/.config/aider-aid/projects.config`.
- Interactive launch flow combining selected profile + selected project.
- Doctor checks for `aider` availability, plus architecture hooks for more checks.

### 5.2 Planned Next

- Config verifier for malformed profile files.
- Ollama connectivity/model availability test.

## 6. Functional Requirements

### FR-1: Configuration Profiles (Aider Configurator)

- Users can create a named profile.
- Users can list existing profiles.
- Users can view and edit profile values.
- Users can delete a profile.
- Profiles are dynamically discovered from `~/.config/aider-aid/configs/`.
- Profile names must be unique within that directory.

Minimum v1 profile fields:

- `name` (string)
- `provider` (enum, initially `ollama`)
- `model` (string)
- Optional launch flags (array/map as needed)

### FR-2: Project Registry (Aider Projects)

- Users can add a project with:
  - Display name
  - Absolute directory path
- Projects are persisted in `~/.config/aider-aid/projects.config`.
- Users can list, rename, and remove project entries.
- Removing a project entry never deletes the underlying filesystem directory.
- During launch, users can select an existing project or choose `Create New...`.
- `Create New...` allows selecting a directory and optionally naming the project.

### FR-3: Aider Launcher

- Users can run `aider-aid launch` to start an interactive selection flow.
- Launcher flow:
  1. Select project (or `Create New...`).
  2. Select profile (or create one inline).
  3. Confirm derived aider command.
  4. Execute aider in selected project directory.
- Launcher must fail fast with clear guidance if required data is missing.

### FR-4: Aider Doctor

- Users can run `aider-aid doctor`.
- Doctor v1 checks:
  - Is `aider` installed and executable in `PATH`?
- If missing, doctor prints a direct install guidance message and link.
- Doctor framework supports additional checks without breaking output contracts.

## 7. Non-Functional Requirements

- Local-first: no network required except for optional runtime checks (future).
- Safe operations: never delete user project directories from registry operations.
- Predictable storage: deterministic file paths and formats.
- Human-readable configs and diagnostic output.
- Fast startup (<300ms target excluding external subprocess calls).

## 8. Data and Storage Design

## 8.1 Directory Layout

```text
~/.config/aider-aid/
  configs/
    <profile-name>.yaml
  projects.config
```

## 8.2 Profile File Format

Per-profile YAML file:

```yaml
name: local-llama
provider: ollama
model: llama3.1
args:
  - --yes
```

Notes:

- Filename slug should derive from profile name.
- File content remains source of truth (dynamic loading).
- Unknown fields should be preserved when possible to support forward compatibility.

## 8.3 Project Registry Format

`projects.config` should be JSON for strict parsing and simple migration:

```json
{
  "version": 1,
  "projects": [
    {
      "name": "aider-aid",
      "path": "/home/user/dev/aider-aid"
    }
  ]
}
```

Validation rules:

- `path` must be absolute.
- Duplicate paths are not allowed.
- Duplicate names are allowed only if explicitly chosen by user (discouraged with warning).

## 9. Command and UX Design

Proposed command surface:

- `aider-aid config list`
- `aider-aid config create`
- `aider-aid config edit <name>`
- `aider-aid config remove <name>`
- `aider-aid project list`
- `aider-aid project add`
- `aider-aid project remove <name-or-id>`
- `aider-aid launch`
- `aider-aid doctor`

UX principles:

- Default to interactive prompts where values are required.
- Offer explicit confirmation before destructive config actions.
- Print exact filesystem paths in all warnings/errors.

## 10. Ollama-First Design

v1 provider handling is Ollama-focused:

- Profile creation provides Ollama as default provider.
- Optional future integration can query local Ollama models and offer selection.
- Model value is stored as explicit string chosen by user.

Future provider abstraction:

- Provider adapters convert profile settings into aider CLI arguments.
- Adapter interface keeps launcher logic provider-agnostic.

## 11. Doctor Checks Architecture

Doctor executes independent checks with standardized result objects:

- `id`: stable check identifier
- `status`: `pass | warn | fail`
- `message`: user-facing summary
- `details`: optional remediation instructions

Initial checks:

1. `aider.binary.available`

Planned checks:

1. `config.files.valid`
2. `ollama.endpoint.reachable`
3. `ollama.model.available`

## 12. Error Handling and Recovery

- Missing config root: create lazily with correct permissions.
- Corrupt `projects.config`: back up file with timestamp suffix, then offer reset.
- Invalid profile YAML: skip invalid file in launcher list and report in doctor.
- Launch failure from aider subprocess: bubble up exit code and command context.

## 13. Security and Privacy

- No telemetry in v1.
- No remote calls during normal launch unless user enables optional checks.
- Avoid storing secrets in project registry.
- Log minimal local diagnostics with redaction for paths only when necessary.

## 14. Testing Strategy

- Unit tests:
  - Profile parse/serialize and validation.
  - Project registry CRUD and corruption handling.
  - Doctor check evaluation logic.
- Integration tests:
  - End-to-end launch flow with fixture projects and profiles.
  - Error path behavior when `aider` is absent.
- Regression tests:
  - Ensure project removal never deletes directories.

## 15. Milestones

1. M1: Storage + CRUD
   - Config/profile management
   - Project registry management
2. M2: Launcher
   - Interactive selection and subprocess execution
3. M3: Doctor v1
   - `aider` binary check + remediation output
4. M4: Doctor expansion
   - Config verifier + Ollama checks

## 16. Risks and Mitigations

- Risk: Config format drift as features expand.
  - Mitigation: include schema versioning and migration paths.
- Risk: Ambiguity between profile-level and launch-time flags.
  - Mitigation: explicit precedence rules (launch-time overrides profile).
- Risk: Broken local environments create poor first experience.
  - Mitigation: run doctor suggestions proactively on first launch failure.

## 17. Open Questions

- Should `projects.config` allow tags/groups for large project sets?
- Should profile files be YAML or TOML long-term?
- Should launcher cache last-used project/profile for one-command repeat runs?
- Should doctor auto-open install documentation when `aider` is missing?

## 18. Acceptance Criteria (v1)

- User can create, list, edit, and delete profiles in `~/.config/aider-aid/configs/`.
- User can add, list, and remove project entries in `~/.config/aider-aid/projects.config`.
- `aider-aid launch` can select a project and profile and invoke aider successfully.
- Removing a project entry leaves project folders untouched.
- `aider-aid doctor` correctly identifies missing `aider` and prints install guidance.
