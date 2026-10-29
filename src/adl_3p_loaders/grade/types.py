"""GRADE data types."""

from typing import Generic

from abstract_dataloader.ext.types import TArray, dataclass
from jaxtyping import Float32, Float64, Int16, UInt16, UInt32
from optree.dataclasses import field


@dataclass
class Waveform(Generic[TArray]):
    """Radar chirp waveform parameters (SI units).

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
class RadarADC(Generic[TArray]):
    """IWR1843BOOST radar ADC samples.

    Axes follow the TI radar convention (`chirp, tx, rx, sample`), with the
    in-phase and quadrature components stacked along a trailing `iq` axis.

    Attributes:
        iq: raw ADC samples; `iq[..., 0]` is I and `iq[..., 1]` is Q.
        timestamps: unix epoch seconds for each frame.
        waveform: chirp waveform parameters.
    """

    iq: Int16[TArray, "batch t chirp tx rx sample 2"]
    timestamps: Float64[TArray, "batch t"]
    waveform: Waveform[TArray]


@dataclass
class DepthImage(Generic[TArray]):
    """ZED 2i stereo depth, aligned with the rectified left camera.

    Attributes:
        depth: depth in millimeters; `0` marks invalid pixels.
        timestamps: unix epoch seconds for each frame.
        intrinsics: pinhole intrinsics `(fx, fy, cx, cy)` in pixels.
    """

    depth: UInt16[TArray, "batch t height width"]
    timestamps: Float64[TArray, "batch t"]
    intrinsics: Float32[TArray, "#batch 4"]


@dataclass
class SmokeReading(Generic[TArray]):
    """MAX30105 particle sensor readings.

    GRADE uses the infrared channel as a smoke density measure: light smoke
    is `IR < 2000`, medium is `2000 <= IR < 4000`, and heavy is `IR >= 4000`.

    Attributes:
        samples: raw `(red, infrared, green)` LED readings.
        timestamps: unix epoch seconds for each reading.
    """

    samples: UInt32[TArray, "batch t 3"]
    timestamps: Float64[TArray, "batch t"]
