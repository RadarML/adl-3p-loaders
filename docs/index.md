# ADL Third-Party Loaders

`adl-3p-loaders` provides data loaders for third-party datasets that implement the [abstract dataloader](https://radarml.github.io/abstract-dataloader/) (ADL) specification.

Each dataset module follows the same structure:

- **Sensors** ([`abstract.Sensor`][abstract_dataloader.abstract.Sensor]) load individual frames from a single sensor, and expose their timestamps and calibration through `.metadata`.
- **Traces** ([`abstract.Trace`][abstract_dataloader.abstract.Trace]) combine the sensors of a single recording using an ADL [synchronization protocol][abstract_dataloader.spec.Synchronization].
- **Datasets** ([`abstract.Dataset`][abstract_dataloader.abstract.Dataset]) concatenate multiple traces.

Each sample is a typed dataclass of numpy arrays with leading `batch` and `t` (time) axes, which are both `1` when loading single frames. Sample types are registered with [optree](https://github.com/metaopt/optree) using [`abstract_dataloader.ext.types.dataclass`][abstract_dataloader.ext.types.dataclass], so they can be collated and converted with any optree-aware tooling.

!!! warning

    This library is primarily written and maintained by LLM agents tasked with implementing data loaders following the [abstract dataloader](https://radarml.github.io/abstract-dataloader/) specification. While spec-compliance is largely guaranteed, the software provided by each dataset's original authors should be considered authoritative.

## Supported Datasets

| Module | Dataset | Sensors |
|--------|---------|---------|
| [`coloradar`][adl_3p_loaders.coloradar] | [ColoRadar](https://arpg.github.io/coloradar/) | Single-chip (AWR1843) radar ADC samples, Ouster OS1-64 lidar point clouds |
| [`grade`][adl_3p_loaders.grade] | [GRADE](https://phi-lab-rice.github.io/GRADE/) | Single-chip (IWR1843) radar ADC samples, ZED 2i stereo depth, MAX30105 smoke sensor; requires the `grade` extra |

## Setup

`adl-3p-loaders` is intended to be used as a library; we recommend installing it as a submodule:

=== "As Submodule"

    ```sh
    git submodule add git@github.com:RadarML/adl-3p-loaders.git
    uv pip install -e ./adl-3p-loaders
    ```

=== "Via `uv` + `pyproject.toml`"

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
