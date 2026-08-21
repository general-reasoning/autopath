---
name: use_autopath_conda_env
description: Always use the autopath conda environment when running Python commands, tests, or CLI tools in this workspace.
---

# Use Conda Environment `autopath`

When executing shell commands, running Python scripts, running tests, or invoking CLI tools (such as `dbx`, `dbx.pprint`, `pytest`, etc.) in this workspace, always run them within the `autopath` conda environment.

## Paths and Usage

- Conda Env Path: `/home/t-9dkarp/miniconda3/envs/autopath`
- Python Binary: `/home/t-9dkarp/miniconda3/envs/autopath/bin/python`
- Executable Binaries: `/home/t-9dkarp/miniconda3/envs/autopath/bin/<command>` (e.g., `/home/t-9dkarp/miniconda3/envs/autopath/bin/dbx.pprint`)

Alternatively, prefix commands with `conda run -n autopath <command>`.

## Dataclass Parameters Rule

- **NEVER add defaults to required dataclass parameters**: Do not add default values or optional `None` fallbacks to required fields in dataclass or `VAR` definitions to fix caller argument mismatches. Always fix the callsite / caller arguments instead.

