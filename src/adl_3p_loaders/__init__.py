"""Abstract Dataloader-compliant loaders for third-party datasets.

| Module | Dataset | Sensors |
|--------|---------|---------|
| [`coloradar`][.] | [ColoRadar](https://arpg.github.io/coloradar/) | radar ADC, lidar |
"""

from . import coloradar

__all__ = ["coloradar"]
