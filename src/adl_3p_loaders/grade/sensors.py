"""GRADE sensors."""

import glob
import os
from dataclasses import dataclass

import h5py
import numpy as np
from abstract_dataloader import abstract
from jaxtyping import Float32, Float64, Int16

from . import types


def find_file(path: str, pattern: str) -> str:
    """Find a (timestamped) file in a GRADE sequence directory.

    Recordings are named with their start time, e.g.
    `radar_20260228_001319.h5`; if more than one file matches, the latest is
    used, following the official GRADE processing code.

    Args:
        path: path to the sequence directory.
        pattern: glob pattern, e.g. `radar_*.h5`.

    Returns:
        Path to the matching file.

    Raises:
        FileNotFoundError: if no file matches.
    """
    matches = sorted(glob.glob(os.path.join(path, pattern)))
    if len(matches) == 0:
        raise FileNotFoundError(f"No `{pattern}` in {path}.")
    return matches[-1]


def default_waveform() -> types.Waveform[np.ndarray]:
    """Get the (assumed) GRADE radar waveform.

    GRADE does not publish its chirp configuration. The range resolution used
    by the GRADE processing code (`0.04375` m) matches the red-rover indoor
    (`grt-i`) configuration used by I/Q-1M, which GRADE also pretrains on, so
    we assume the same waveform.

    !!! warning

        The range resolution is verified against the GRADE code, but the chirp
        timing (`idle_time`, `ramp_end_time`, and therefore the Doppler
        resolution) is not.
    """
    def _f32(x: float) -> np.ndarray:
        return np.array([x], dtype=np.float32)

    return types.Waveform(
        start_frequency=_f32(77e9),
        frequency_slope=_f32(67.012e12),
        sample_rate=_f32(5e6),
        idle_time=_f32(200e-6),
        adc_start_time=_f32(5.7e-6),
        ramp_end_time=_f32(59e-6),
        shape=(64, 3, 4, 256))


def iq_from_iiqq(
    iiqq: Int16[np.ndarray, "... n"]
) -> Int16[np.ndarray, "... n/2 2"]:
    """Un-interleave DCA1000 `IIQQ` samples into `(..., sample, iq)`.

    Each group of 4 values is `(Q0, Q1, I0, I1)`, following the convention
    used by both GRADE and [`xwr`](https://radarml.github.io/xwr/).
    """
    iq = np.empty((*iiqq.shape[:-1], iiqq.shape[-1] // 2, 2), dtype=np.int16)
    iq[..., 0::2, 1] = iiqq[..., 0::4]
    iq[..., 1::2, 1] = iiqq[..., 1::4]
    iq[..., 0::2, 0] = iiqq[..., 2::4]
    iq[..., 1::2, 0] = iiqq[..., 3::4]
    return iq


def _dataset(f: h5py.File, key: str) -> h5py.Dataset:
    ds = f[key]
    if not isinstance(ds, h5py.Dataset):
        raise ValueError(f"`{key}` in {f.filename} is not a dataset.")
    return ds


def _read_timestamps(path: str) -> Float64[np.ndarray, "N"]:
    with h5py.File(path, "r") as f:
        return np.asarray(
            _dataset(f, "timestamps_ms")[:], dtype=np.float64) / 1000.0


@dataclass
class RadarMetadata:
    """Radar metadata.

    Attributes:
        timestamps: unix epoch seconds for each frame.
        waveform: chirp waveform parameters.
    """

    timestamps: Float64[np.ndarray, "N"]
    waveform: types.Waveform[np.ndarray]


@dataclass
class DepthMetadata:
    """ZED depth metadata.

    Attributes:
        timestamps: unix epoch seconds for each frame.
        intrinsics: pinhole intrinsics `(fx, fy, cx, cy)` in pixels.
    """

    timestamps: Float64[np.ndarray, "N"]
    intrinsics: Float32[np.ndarray, "1 4"]


@dataclass
class SmokeMetadata:
    """MAX30105 metadata.

    Attributes:
        timestamps: unix epoch seconds for each reading.
    """

    timestamps: Float64[np.ndarray, "N"]


class Radar(abstract.Sensor[types.RadarADC[np.ndarray], RadarMetadata]):
    """IWR1843BOOST radar ADC samples (`radar_*.h5`).

    Frames are stored as `int16` DCA1000 `IIQQ` samples in
    `(chirp, tx, rx, sample)` order, and are returned with a trailing `iq`
    axis.

    !!! warning

        If `past > 0`, the caller is responsible for ensuring that indices
        `< past` are not accessed, e.g. by applying a synchronization margin.

    Args:
        path: path to a sequence directory.
        waveform: chirp waveform parameters; if `None`, uses
            [`default_waveform`][^.].
        past: number of past frames to include.
    """

    def __init__(
        self, path: str, waveform: types.Waveform[np.ndarray] | None = None,
        past: int = 0
    ) -> None:
        self.past = past
        self._path = find_file(path, "radar_*.h5")
        if waveform is None:
            waveform = default_waveform()

        with h5py.File(self._path, "r") as f:
            shape = _dataset(f, "radar_data").shape[1:]
        n_chirp, n_tx, n_rx, n_sample = waveform.shape
        if shape != (n_chirp, n_tx, n_rx, 2 * n_sample):
            raise ValueError(
                f"Radar data in {self._path} has frame shape {shape}; "
                f"expected IIQQ data for waveform shape {waveform.shape}.")

        super().__init__(metadata=RadarMetadata(
            timestamps=_read_timestamps(self._path), waveform=waveform),
            name=os.path.basename(os.path.abspath(path)))

    def __getitem__(self, index: int | np.integer) -> types.RadarADC[np.ndarray]:
        """Read a radar frame, along with `past` previous frames."""
        index = int(index)
        if index < self.past:
            raise IndexError(
                f"Radar frame {index} does not have {self.past} past frames.")
        with h5py.File(self._path, "r") as f:
            iiqq = np.asarray(
                _dataset(f, "radar_data")[index - self.past:index + 1])
        return types.RadarADC(
            iq=iq_from_iiqq(iiqq)[None],
            timestamps=self.metadata.timestamps[
                index - self.past:index + 1][None],
            waveform=self.metadata.waveform)


class Depth(abstract.Sensor[types.DepthImage[np.ndarray], DepthMetadata]):
    """ZED 2i stereo depth (`zed_depth.h5`).

    Depth frame `i` corresponds to entry `i` in `camera_timestamps_*.h5`;
    surplus trailing timestamps are discarded, following the GRADE processing
    code.

    !!! warning

        The GRADE processing code discards ZED frames after index `10000` in
        the `Razor-1` sequence; this loader does not.

    Args:
        path: path to a sequence directory.
    """

    def __init__(self, path: str) -> None:
        self._path = os.path.join(path, "zed_depth.h5")
        with h5py.File(self._path, "r") as f:
            n_frames = _dataset(f, "depth_mm").shape[0]
            intrinsics = np.asarray(
                f.attrs["intrinsics_fx_fy_cx_cy"], dtype=np.float32)

        timestamps = _read_timestamps(
            find_file(path, "camera_timestamps_*.h5"))
        if len(timestamps) < n_frames:
            raise ValueError(
                f"{path} has {n_frames} depth frames, but only "
                f"{len(timestamps)} camera timestamps.")

        super().__init__(metadata=DepthMetadata(
            timestamps=timestamps[:n_frames], intrinsics=intrinsics[None]),
            name=os.path.basename(os.path.abspath(path)))

    def __getitem__(
        self, index: int | np.integer
    ) -> types.DepthImage[np.ndarray]:
        """Read a depth image."""
        index = int(index)
        with h5py.File(self._path, "r") as f:
            depth = np.asarray(_dataset(f, "depth_mm")[index])
        return types.DepthImage(
            depth=depth[None, None],
            timestamps=self.metadata.timestamps[index:index + 1][None],
            intrinsics=self.metadata.intrinsics)


class Smoke(abstract.Sensor[types.SmokeReading[np.ndarray], SmokeMetadata]):
    """MAX30105 particle sensor (`max30105_*.h5`).

    !!! warning

        Only evaluation sequences (`GRADE_Eval_Raw`) have a smoke sensor.

    Args:
        path: path to a sequence directory.
    """

    def __init__(self, path: str) -> None:
        self._path = find_file(path, "max30105_*.h5")
        super().__init__(metadata=SmokeMetadata(
            timestamps=_read_timestamps(self._path)),
            name=os.path.basename(os.path.abspath(path)))

    def __getitem__(
        self, index: int | np.integer
    ) -> types.SmokeReading[np.ndarray]:
        """Read a smoke sensor reading."""
        index = int(index)
        with h5py.File(self._path, "r") as f:
            samples = np.asarray(_dataset(f, "samples_uint32")[index])
        return types.SmokeReading(
            samples=samples[None, None],
            timestamps=self.metadata.timestamps[index:index + 1][None])
