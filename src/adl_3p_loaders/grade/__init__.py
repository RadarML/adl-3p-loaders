"""GRADE dataset loader.

Loads IWR1843BOOST radar ADC samples, ZED 2i stereo depth, and MAX30105 smoke
sensor readings from the raw release of the
[GRADE dataset](https://huggingface.co/datasets/phi-lab-rice/GRADE_Dataset)
(Zhao et al., MobiCom 2026), which contains indoor sequences with and without
smoke.

!!! info "Requires the `grade` extra"

    The GRADE loader requires `h5py`; install with
    `uv pip install -e ./adl-3p-loaders[grade]`, or add
    `adl-3p-loaders[grade]` to your dependencies.

```python
from abstract_dataloader import generic
from adl_3p_loaders import grade

dataset = grade.Dataset.from_config(
    "/path/to/grade", split="eval",
    sync=generic.Nearest("radar", tol=0.05))
sample = dataset[0]  # {"radar": RadarADC, "depth": DepthImage}
```

??? info "Expected data layout"

    ```
    grade/
    ├── GRADE_Train_Raw/{sequence}/
    │   ├── radar_{time}.h5
    │   ├── camera_timestamps_{time}.h5
    │   └── zed_depth.h5
    └── GRADE_Eval_Raw/{sequence}/
        ├── radar_{time}.h5
        ├── camera_timestamps_{time}.h5
        ├── zed_depth.h5
        └── max30105_{time}.h5
    ```

    The released `*.mkv` videos and `dji_timestamps_*.h5` are not used.

| Sensor | File | Data Type | Metadata |
|--------|------|-----------|----------|
| [`Radar`][.] | `radar_*.h5` | [`RadarADC`][.types.] | [`RadarMetadata`][.] |
| [`Depth`][.] | `zed_depth.h5` | [`DepthImage`][.types.] | [`DepthMetadata`][.] |
| [`Smoke`][.] | `max30105_*.h5` | [`SmokeReading`][.types.] | [`SmokeMetadata`][.] |

The splits follow the dataset card (see [`Dataset.find`][.]): `train` (40
sequences) and `val` (10 sequences) are drawn from `GRADE_Train_Raw`, and
`eval` (20 sequences) is `GRADE_Eval_Raw`. In the GRADE evaluation,
`GRADE_Eval_Raw` sequences ending in `-0` are clear, and the others are
smoke, which is further stratified by the [MAX30105 IR
reading][.types.SmokeReading].

!!! warning

    By default, traces use [`generic.Empty`][abstract_dataloader.generic.Empty]
    synchronization, which yields no samples. Pass a synchronization protocol
    (e.g. [`generic.Nearest`][abstract_dataloader.generic.Nearest]) to pair
    radar and depth frames; GRADE uses a tolerance of 50 ms.

!!! warning

    The radar waveform is not included in the dataset, and is inferred from
    the GRADE processing code; see [`default_waveform`][.].
"""

from . import types
from .sensors import (
    Depth,
    DepthMetadata,
    Radar,
    RadarMetadata,
    Smoke,
    SmokeMetadata,
    default_waveform,
    find_file,
    iq_from_iiqq,
)
from .trace import (
    DEFAULT_SENSORS,
    SENSOR_TYPES,
    SPLIT_PATHS,
    VAL_KEYWORDS,
    Dataset,
    Trace,
)

__all__ = [
    "types",
    "Depth", "DepthMetadata", "Radar", "RadarMetadata",
    "Smoke", "SmokeMetadata", "default_waveform", "find_file", "iq_from_iiqq",
    "DEFAULT_SENSORS", "SENSOR_TYPES", "SPLIT_PATHS", "VAL_KEYWORDS",
    "Dataset", "Trace",
]
