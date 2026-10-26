"""ColoRadar data types."""

from typing import Generic

from abstract_dataloader.ext.types import TArray, dataclass
from jaxtyping import Float32, Float64, Int16
from optree.dataclasses import field


@dataclass
class Waveform(Generic[TArray]):
    """Radar chirp waveform parameters.

    Units follow the ColoRadar `waveform_cfg.txt` calibration files (SI).

    Attributes:
        start_frequency: chirp start frequency in Hz.
        frequency_slope: chirp frequency slope in Hz/s.
        sample_rate: ADC sample rate in Hz.
        idle_time: idle time between chirps in s.
        adc_start_time: ADC start time relative to chirp start in s.
        ramp_end_time: ramp end time in s.
        shape: **`(static)`** ADC frame shape as `(chirp, tx, rx, sample)`.
    """

    start_frequency: Float32[TArray, "#batch"]
    frequency_slope: Float32[TArray, "#batch"]
    sample_rate: Float32[TArray, "#batch"]
    idle_time: Float32[TArray, "#batch"]
    adc_start_time: Float32[TArray, "#batch"]
    ramp_end_time: Float32[TArray, "#batch"]
    shape: tuple[int, int, int, int] = field(pytree_node=False)


@dataclass
class Transform(Generic[TArray]):
    """Rigid transform from a sensor frame to the ColoRadar base frame.

    Attributes:
        translation: `(x, y, z)` translation in meters.
        rotation: rotation as an `(x, y, z, w)` quaternion.
    """

    translation: Float64[TArray, "#batch 3"]
    rotation: Float64[TArray, "#batch 4"]


@dataclass
class RadarADC(Generic[TArray]):
    """Single-chip radar ADC samples.

    Axes follow the TI radar convention (`chirp, tx, rx, sample`), with the
    in-phase and quadrature components stacked along a trailing `iq` axis.

    Attributes:
        iq: raw ADC samples; `iq[..., 0]` is I and `iq[..., 1]` is Q.
        timestamps: unix epoch seconds for each frame.
        waveform: chirp waveform parameters.
        extrinsics: transform from the radar frame to the base frame.
    """

    iq: Int16[TArray, "batch t chirp tx rx sample 2"]
    timestamps: Float64[TArray, "batch t"]
    waveform: Waveform[TArray]
    extrinsics: Transform[TArray]


@dataclass
class LidarPoints(Generic[TArray]):
    """Ouster OS1-64 lidar point clouds.

    !!! warning

        Point clouds are returned as stored, with a fixed number of points per
        frame; points with no return are all-zero and are not removed.

    Attributes:
        points: `(x, y, z, intensity)` for each point, in the lidar frame.
        timestamps: unix epoch seconds for each frame.
        extrinsics: transform from the lidar frame to the base frame.
    """

    points: Float32[TArray, "batch t n 4"]
    timestamps: Float64[TArray, "batch t"]
    extrinsics: Transform[TArray]
