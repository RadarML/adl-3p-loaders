# Development Environment

!!! info "`adl-3p-loaders` uses `uv`"

    While `pyproject.toml` reflects known limitations on dependency versions, we use `uv` to manage a lock file used for development.

    On a fresh linux install, you will need to [install `uv`](https://docs.astral.sh/uv/getting-started/installation/):
    ```sh
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc
    ```

## Setup

```sh
git clone git@github.com:RadarML/adl-3p-loaders.git
cd adl-3p-loaders
uv sync --all-extras --frozen
uv run pre-commit install
```

## Tests

Run tests:
```sh
uv run --extra dev pytest -ra --cov --cov-report=html --cov-report=term -- tests
```

!!! tip

    HTML-format code coverage is saved to `./htmlcov`; view these with `cd htmlcov; python -m http.server 8001`.

By default, tests only use small synthetic datasets. To also test against the real datasets, set the corresponding environment variable to the dataset root:

| Dataset | Environment Variable |
|---------|----------------------|
| ColoRadar | `COLORADAR_PATH` |

```sh
export COLORADAR_PATH=/path/to/coloradar
uv run --extra dev pytest -ra -- tests
```

!!! info

    The tests are also run by our [pre-commit](https://pre-commit.com/) hooks, which you can manually trigger with `uv run pre-commit run`; these hooks (`ruff + pyright + pytest`) mirror the CI.

## Documentation

Build and serve the documentation locally:
```sh
uv run --extra docs mkdocs serve
```
