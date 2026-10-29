# `adl-3p-loaders`: Abstract Dataloader-Compliant Loaders for Third-Party Datasets

[![GitHub](https://img.shields.io/github/v/release/RadarML/adl-3p-loaders)](https://github.com/RadarML/adl-3p-loaders)
![MIT License](https://img.shields.io/badge/license-MIT-green)
![Supports Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
![Typed](https://img.shields.io/badge/types-typed-limegreen)
[![CI](https://github.com/RadarML/adl-3p-loaders/actions/workflows/ci.yml/badge.svg)](https://github.com/RadarML/adl-3p-loaders/actions/workflows/ci.yml)
![GitHub issues](https://img.shields.io/github/issues/RadarML/adl-3p-loaders)

> [!IMPORTANT]
> See our [documentation site](https://radarml.github.io/adl-3p-loaders) for more details!

`adl-3p-loaders` provides data loaders for third-party datasets that implement the [abstract dataloader](https://radarml.github.io/abstract-dataloader/) specification. Each dataset is exposed as ADL sensors, traces, and datasets, which return typed, batched numpy dataclasses.

## Supported Datasets

| Module | Dataset | Sensors |
|--------|---------|---------|
| [`adl_3p_loaders.coloradar`](https://radarml.github.io/adl-3p-loaders/coloradar/) | [ColoRadar](https://arpg.github.io/coloradar/) | Single-chip (AWR1843) radar ADC samples, Ouster OS1-64 lidar point clouds |
| [`adl_3p_loaders.grade`](https://radarml.github.io/adl-3p-loaders/grade/) | [GRADE](https://phi-lab-rice.github.io/GRADE/) | Single-chip (IWR1843) radar ADC samples, ZED 2i stereo depth, MAX30105 smoke sensor; requires the `grade` extra |

## Setup

`adl-3p-loaders` is intended to be used as a library; we recommend installing it as a submodule:

```sh
git submodule add git@github.com:RadarML/adl-3p-loaders.git
uv pip install -e ./adl-3p-loaders
```

Or, via `uv` + `pyproject.toml`:

```toml
[project]
dependencies = ["adl-3p-loaders"]

[tool.uv.sources]
adl-3p-loaders = { path = "./adl-3p-loaders", editable = true }
```

## Usage

```python
from abstract_dataloader import generic
from adl_3p_loaders import coloradar

dataset = coloradar.Dataset.from_config(
    "/path/to/coloradar", sync=generic.Nearest("radar", tol=0.1))
sample = dataset[0]
sample["radar"].iq        # int16 (1, 1, chirp, tx, rx, sample, iq)
sample["lidar"].points    # float32 (1, 1, 65536, xyz + intensity)
```

> [!WARNING]
> Traces use `generic.Empty()` synchronization by default, which yields no samples; pass a synchronization protocol to pair sensors.
