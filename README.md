# `adl-3p-loaders`: Abstract Dataloader-Compliant Loaders for Third-Party Datasets

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/adl-3p-loaders)
![License - MIT](https://img.shields.io/badge/license-MIT-green)
![PyPI - Types](https://img.shields.io/pypi/types/adl-3p-loaders)
[![CI](https://github.com/RadarML/adl-3p-loaders/actions/workflows/ci.yml/badge.svg)](https://github.com/RadarML/adl-3p-loaders/actions/workflows/ci.yml)
![GitHub issues](https://img.shields.io/github/issues/RadarML/adl-3p-loaders)

`adl-3p-loaders` provides data loaders for third-party datasets that implement the [abstract dataloader](https://radarml.github.io/abstract-dataloader/) specification. Each dataset is exposed as ADL sensors, traces, and datasets, which return typed, batched numpy dataclasses.

See our [documentation site](https://radarml.github.io/adl-3p-loaders/) for more details.

## Supported Datasets

| Module | Dataset | Sensors |
|--------|---------|---------|
| [`adl_3p_loaders.coloradar`](https://radarml.github.io/adl-3p-loaders/coloradar/) | [ColoRadar](https://arpg.github.io/coloradar/) | Single-chip (AWR1843) radar ADC samples, Ouster OS1-64 lidar point clouds |

## Install

`adl-3p-loaders` can be installed from github:

```sh
pip install git+ssh://git@github.com/RadarML/adl-3p-loaders.git
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
