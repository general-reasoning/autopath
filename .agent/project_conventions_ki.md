# Project Autopath Conventions

This Knowledge Item documents the persistent conventions and operational rules for the Autopath project.

## Access & Permissions
- **Full Access**: The agent is granted explicit permission to access and modify any file within the project repository. This includes all source code, documentation, and the `.agent/` directory.

## Environment & Operations
- **Conda Environment**: The `autopath` environment should be used for all tasks.
  - Python Path: `/home/t-9dkarp/miniconda3/envs/autopath/bin/python`
  - Tectonic Path: `/home/t-9dkarp/miniconda3/envs/autopath/bin/tectonic`
- **Execution**: Always set `PYTHONPATH=.` when running scripts from the project root.

## Technical Standards
- **Commit Messages**: All commits must be prefixed with `autopath: `.
- **File Organization**: 
  - Agent-specific files (verification scripts, logs, etc.) must be stored in subfolders of `.agent/`.
  - Use subfolders for categorization, e.g., `.agent/verify/` for temporary scripts.
  - **Git Hygiene**: High-level agent files (rules, KIs, workflows) should be committed. Temporary files in `.agent/verify/` should be ignored.


## Documentation
- The primary technical documentation for the Hydro model is located at `autopath/docs/models/hydro.tex`.
- Any architectural changes to the Hydro model should be reflected in this file and verified via Tectonic compilation.
