"""ColoRadar dataset loader.

Loads single-chip (AWR1843) radar ADC samples and Ouster OS1-64 lidar point
clouds from the [ColoRadar dataset](https://arpg.github.io/coloradar/)
(Kramer et al., IJRR 2021), using the extracted (non-rosbag) data format.

```python
from abstract_dataloader import generic
from adl_3p_loaders import coloradar

dataset = coloradar.Dataset.from_config(
    "/path/to/coloradar", sync=generic.Nearest("radar", tol=0.05))
sample = dataset[0]  # {"radar": RadarADC, "lidar": LidarPoints}
```

??? info "Expected data layout"

    ```
    coloradar/
    ├── utils/calib/
    │   ├── single_chip/waveform_cfg.txt
    │   └── transforms/{base_to_single_chip,base_to_lidar}.txt
    └── {run}/
        ├── single_chip/adc_samples/
        │   ├── timestamps.txt
        │   └── data/frame_{i}.bin
        └── lidar/
            ├── timestamps.txt
            └── pointclouds/lidar_pointcloud_{i}.bin
    ```

| Sensor | Directory | Data Type | Metadata |
|--------|-----------|-----------|----------|
| [`Radar`][.] | `single_chip` | [`RadarADC`][.types.] | [`RadarMetadata`][.] |
| [`Lidar`][.] | `lidar` | [`LidarPoints`][.types.] | [`LidarMetadata`][.] |

!!! warning

    By default, traces use [`generic.Empty`][abstract_dataloader.generic.Empty]
    synchronization, which yields no samples. Pass a synchronization protocol
    (e.g. [`generic.Nearest`][abstract_dataloader.generic.Nearest]) to pair
    radar and lidar frames.
"""

from . import types
from .calib import get_transform, read_transform, read_waveform
from .sensors import Lidar, LidarMetadata, Radar, RadarMetadata
from .trace import SENSOR_PATHS, SENSOR_TYPES, Dataset, Trace, split

__all__ = [
    "types",
    "get_transform", "read_transform", "read_waveform",
    "Lidar", "LidarMetadata", "Radar", "RadarMetadata",
    "SENSOR_PATHS", "SENSOR_TYPES", "Dataset", "Trace", "split",
]
