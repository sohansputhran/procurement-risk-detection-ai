# Contributing

Thanks for your interest! For this project we use:
- Issues and Milestones for Agile planning (Sprints)
- Conventional commits for messages (e.g., feat:, fix:, chore:)
- Black + Ruff + pre-commit for code quality

## Setup
1. Create a virtual env, install deps from `requirements*.txt`.
2. `pre-commit install` to enable local hooks.
3. `pytest -q` before pushing.

## Branching
- `main`: stable
- `dev`: integration branch
- feature branches: `feat/<short-name>`
