# AGENTS.md

BoltzGen: a protein design pipeline (diffusion design -> inverse folding -> folding -> affinity -> analysis -> filtering), installed as the `boltzgen` CLI (`src/boltzgen/cli/boltzgen.py`, entrypoint `boltzgen = boltzgen.cli.boltzgen:main`).

## Install / environment
- Requires Python >=3.11, editable install: `pip install -e .`
- Optional extras: `pip install -e .[dev]` (adds ruff, pytest, wandb, etc.)
- Running the pipeline needs a GPU and downloads ~6GB of model weights to `~/.cache` on first run (override with `--cache` or `$HF_HOME`).

## Testing
- Run: `pytest tests/`
- Most heavy dependencies (torch, hydra, biotite, gemmi, rdkit, etc.) are real by default; tests import them normally.
- `tests/conftest.py` provides a `--mock-heavy-deps` pytest flag that stubs out heavy deps with `MagicMock` for parser-only tests (e.g. `pytest --mock-heavy-deps tests/test_residue_constraints.py`). Only use this for tests that exercise pure parsing/schema logic — it will break anything that actually touches torch/gemmi/etc.
- There is no CI test workflow (`.github/workflows/python-publish.yml` only builds/publishes to PyPI on release). Local `pytest` is the only signal.

## Lint/type-check
- Lint: `ruff` (config in `pyproject.toml`), ruleset is `ALL` with an explicit ignore list — check that list before "fixing" a violation, it may be intentionally ignored.
- `[tool.mypy]` requires typed defs (`disallow_untyped_defs = true`) but excludes `configs/`, `build/`, `dist/`, `docs/`, `tests/`.
- No pre-commit config or lint/format CI step exists; nothing enforces these automatically, but new code should still conform to the ruff config.

## Design-spec YAML conventions (easy to get wrong)
- Residue indices in `.yaml` design specs are **1-indexed** and refer to mmcif `label_asym_id`/`label_seq_id`, **not** the author (`auth_asym_id`) numbering shown in some viewers.
- File paths referenced inside a design-spec `.yaml` (e.g. `.cif` paths) are resolved **relative to the yaml file's own directory**, not the CWD.
- Always sanity-check a new/edited design spec with `boltzgen check <spec.yaml>` before running the full pipeline.

## Repo layout notes
- `src/boltzgen/` — the actual package (`cli/`, `data/`, `model/`, `task/`, `utils/`, `resources/`). Pipeline steps live under `task/` (e.g. `task/analyze/analyze.py`).
- `example/` — canonical example design-spec YAMLs referenced throughout the README; `example/README.md` has the full yaml-spec reference.
- Root of the repo also accumulates ad-hoc experiment scratch files (`design_*_spec.yaml`, `material_spec*.yaml`, `material_out_*/`, `test_out*/`, one-off scripts like `material_builder.py`, `get_diff_history.py`). These are untracked/gitignored working files, not part of the package — don't assume they reflect the supported API surface, and don't worry about cleaning them up unless asked.
- `filter.ipynb` is the recommended interactive way to re-run just the filtering step cheaply (~15s) after a full run, instead of rerunning `boltzgen run --steps filtering`.
