# Development

## Tools

The development tools are installed by `uv sync`:

```bash
uv run ruff check        # lint
uv run ruff check --fix  # lint and apply the safe fixes
uv run ruff format       # format
uv run ty check          # type check
```

All three pass on the whole repository and are configured in `pyproject.toml`:

- **ruff** checks pycodestyle errors, pyflakes, import order and bugbear (`E`, `F`, `I`, `B`) with a line length of 160. `src/models/minimamba.py` is allowed single-letter names, which follow the notation of the reference implementation.
- **ruff format** formats the code with the same line length and keeps the quote style of each string. The vendored `src/sync_batchnorm/` package is not formatted.
- **ty** checks `src/`. The vendored `src/sync_batchnorm/` package is excluded. The optional `mamba_ssm` imports are ignored inline.

The configuration groups (`cfgs.DATA`, `cfgs.MODEL`, ...) are filled dynamically from the YAML files. For the type checker they accept any attribute (see `make_empty_object` in `src/utils/misc.py`), so a misspelled option name is not a type error: it fails when the config file is loaded.

### Pre-commit hook

Install the git hook once per clone:

```bash
uv run pre-commit install
```

Every `git commit` then runs `ruff check --fix` and `ruff format` on the staged Python files and `ty check` on the whole project (`.pre-commit-config.yaml`). The hooks call `uv run`, so they use the tool versions pinned in `uv.lock`. When ruff fixes or reformats a file the commit stops: review the change, `git add` it and commit again. `uv run pre-commit run --all-files` runs the hooks on every file (it needs Git 2.31 or newer).

If the environment is not in `.venv` (see [installation.md](installation.md#install)), `UV_PROJECT_ENVIRONMENT` has to be set in the process that runs `git commit`.

## Dependencies

`pyproject.toml` is the only list of dependencies and `uv.lock` pins them.

```bash
uv add <package>                    # core dependency
uv add --optional extract <package> # extra
uv add --dev <package>              # development tool
uv lock --upgrade                   # update the pinned versions
```

`polars` is held below 1.0: the data loader maps signs to classes with `Expr.replace`, whose behaviour changed in 1.0.

## Checking a change for regressions

There are no unit tests. Because seeded runs are deterministic (see [reproducibility.md](reproducibility.md)), a refactor can be checked by comparing the metrics of short runs before and after it:

```bash
./scripts/dev/regression_check.sh /tmp/before     # on the base commit
# make the change
./scripts/dev/regression_check.sh /tmp/after
diff /tmp/before/metrics.txt /tmp/after/metrics.txt && echo identical
```

The script trains each flow for two epochs with seed 42 and records the validation and test metrics: classification with the Transformer on four data pipelines, ST-GCN and conv1d, both generators, dataset generation (it checksums the generated clips) and synthetic pretraining. It needs the `INCLUDE` and `INCLUDE_official` datasets in `HANDCRAFT_DATA` and takes about 12 minutes on one GPU.

An empty diff means the change did not alter the behaviour of any of those flows. This is how the lint and type fixes were verified.
