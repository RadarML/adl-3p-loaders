# Agent Instructions for `adl-3p-loaders`

- Use the instructions in this file to accomplish common tasks requested by the user. If a task is not covered here, please ask the user for more details.
- This project uses `uv`. All python-related scripts and commands should use `uv run` (including linters and checkers like `ruff` and `pyright`).

## Code Style

### Formatting

- **Line length**: 80 columns soft limit on code files. Don't break lines where it would hurt readability (e.g. long URLs, Markdown tables, multi-part jaxtyping annotations, or long string literals).
    - This rule does not apply to markdown files, which should be viewed with word wrapping enabled and should not have unnecessary hard line breaks.
- **Imports**: standard library → third-party → local, each group separated by a blank line (isort order).

### Type Annotations

Type-annotate everything. Use [`jaxtyping`](https://github.com/patrick-kidger/jaxtyping) for tensor/array types:

```python
from jaxtyping import Bool, Float, Shaped
from torch import Tensor

def forward(
    self, x: Float[Tensor, "batch *spatial channels"]
) -> Float[Tensor, "batch *spatial channels"]: ...
```

Named dimensions should be descriptive (`batch`, `spatial`, `channels`, `elevation`, `azimuth`, `range`, etc.). Use `*` for variadic axes and `#` to allow broadcasting.

`# type: ignore` comments are acceptable, especially for third-party libraries with incomplete type stubs (e.g. PyTorch, Lightning).

## Documentation

### Mkdocs

The documentation is built with mkdocs and mkdocs-material.

If you modify a documentation file in docs/, the documentation preview is reloaded automatically. If you modify a docstring, use
```
touch docs/index.md
```
to trigger a reload.

### Docstrings

Use **Google-style** docstrings.

- Document `__init__` parameters in the *class* docstring, not `__init__` itself.
- `forward` / `__call__` methods that are self-evident can have a one-line docstring; add `Args` / `Returns` sections when the behaviour is non-trivial.
- Use a single backtick for inline code.
- Avoid using non-ASCII characters outside of mathematical notation. For example, do not use em-dashes if a hyphen suffices.

```python
class MyModule(nn.Module):
    """Short one-line summary.

    Longer description of the module's behaviour, with any relevant math,
    paper references, or important caveats.

    Args:
        channels: number of input/output channels.
        scale: scale factor applied before the transform.
    """

    def __init__(self, channels: int, scale: float = 1.0) -> None: ...

    def forward(
        self, x: Float[Tensor, "batch channels"]
    ) -> Float[Tensor, "batch channels"]:
        """Apply the transform."""
        ...
```

### mkdocs-material Directives

The docs are built with mkdocs-material, so use its admonition / details directives directly in docstrings:

| Directive | Purpose |
|-----------|---------|
| `!!! note` | General informational note |
| `!!! info` | Supplementary detail |
| `!!! warning` / `!!! warning "Title"` | Non-obvious gotcha or side-effect |
| `??? quote "Sample Config"` | Collapsible block (e.g. config snippets) |

Code blocks inside docstrings support language tags and titles:

````python
"""
??? quote "Sample Config"

    ```yaml title="config/foo.yaml"
    foo:
      bar: 1
    ```
"""
````

### Cross-References (mkdocstrings-python-xref)

`relative_crossrefs = true` is enabled, so you can use relative paths instead of fully-qualified names. The path is resolved relative to the current symbol's location.

| Syntax | Expands to | When to use |
|--------|-----------|-------------|
| `[ClassName][^.]` | `{current_module}.ClassName` | Sibling class / function in the same module |
| `[ClassName][.]` | `{current_module}.ClassName` | Class from a module docstring |
| `[method][^.]` | `{current_class}.method` | Another method in the same class (from inside a method docstring) |
| `[method][.]` | `{current_class}.method` | Another method in the same class (from inside a class docstring) |
| `[pkg.mod.Cls][pkg.mod.Cls]` | absolute | External or cross-package reference |

The `.` at the end of a path appends the link title automatically, so `[Foo][^.]` inside `example.library.Bar`'s docstring resolves to `example.library.Foo`.

Prefix a reference with `?` to suppress cross-reference validation for that link: `[MissingType][?adl_3p_loaders.foo.MissingType]`.

## Adding a Dataset

Each dataset lives in its own subpackage, `src/adl_3p_loaders/{dataset}/`, using [`coloradar`](src/adl_3p_loaders/coloradar/) as the reference implementation:

- `types.py`: sample and calibration types. Use `abstract_dataloader.ext.types.dataclass` (not `dataclasses.dataclass`) and `Generic[TArray]` so that types are registered with optree. Each array has leading `batch` and `t` axes; mark shape-like fields as static with `optree.dataclasses.field(pytree_node=False)`.
- `sensors.py`: one `abstract_dataloader.abstract.Sensor` subclass per sensor, with a `{Sensor}Metadata` dataclass containing `timestamps` (float64 unix epoch seconds) and any calibration.
- `trace.py`: `Trace` and `Dataset` classes with `from_config` constructors. Default to `generic.Empty()` synchronization.
- `__init__.py`: module docstring describing the dataset, the expected on-disk layout, and a sensor table.

Also:

- Add the dataset to the tables in `README.md`, `docs/index.md`, and `src/adl_3p_loaders/__init__.py`, and add API pages under `docs/{dataset}/` (plus `mkdocs.yml` nav and `llmstxt` sections).
- Add tests in `tests/test_{dataset}.py`, using a small synthetic dataset written to `tmp_path`. Tests which use the real dataset should be skipped unless a `{DATASET}_PATH` environment variable is set; document the variable in `docs/develop.md`.
- Verify with `uv run --extra dev ruff check ./src ./tests`, `uv run --extra dev pyright ./src`, `uv run --extra dev pytest -ra -- tests`, and `uv run --extra docs mkdocs build --strict`.
