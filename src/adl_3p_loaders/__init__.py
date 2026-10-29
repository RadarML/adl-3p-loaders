"""Abstract Dataloader-compliant loaders for third-party datasets.

| Module | Dataset | Sensors |
|--------|---------|---------|
| [`coloradar`][.] | [ColoRadar](https://arpg.github.io/coloradar/) | radar ADC, lidar |
| [`grade`][.] | [GRADE](https://phi-lab-rice.github.io/GRADE/) | radar ADC, depth, smoke |

!!! warning

    Datasets are not imported by default, and must be imported explicitly, e.g.
    `from adl_3p_loaders import grade`.
"""
