"""ColoRadar sensors."""

import os
from dataclasses import dataclass

import numpy as np
from abstract_dataloader import abstract
from jaxtyping import Float64

from . import types
from .calib import read_transform, read_waveform


@dataclass
class RadarMetadata:
    """Single-chip radar metadata.

    Attributes:
        timestamps: unix epoch seconds for each frame.
        waveform: chirp waveform parameters.
        extrinsics: transform from the radar frame to the base frame.
    """

    timestamps: Float64[np.ndarray, "N"]
    waveform: types.Waveform[np.ndarray]
    extrinsics: types.Transform[np.ndarray]


@dataclass
class LidarMetadata:
    """Lidar metadata.

    Attributes:
        timestamps: unix epoch seconds for each frame.
        extrinsics: transform from the lidar frame to the base frame.
    """

    timestamps: Float64[np.ndarray, "N"]
    extrinsics: types.Transform[np.ndarray]


class Radar(abstract.Sensor[types.RadarADC[np.ndarray], RadarMetadata]):
    """Single-chip (AWR1843) radar ADC samples.

    Each frame is stored as interleaved `int16` I/Q samples in
    `(tx, rx, chirp, sample)` order, and is returned in TI
    `(chirp, tx, rx, sample)` order.

    Args:
        path: path to the `single_chip` directory of a run.
        calib: path to the ColoRadar calibration directory (`utils/calib`).
    """

    def __init__(self, path: str, calib: str) -> None:
        self._path = os.path.join(path, "adc_samples")
        waveform = read_waveform(
            os.path.join(calib, "single_chip", "waveform_cfg.txt"))
        super().__init__(metadata=RadarMetadata(
            timestamps=np.loadtxt(
                os.path.join(self._path, "timestamps.txt"),
                dtype=np.float64, ndmin=1),
            waveform=waveform,
            extrinsics=read_transform(
                os.path.join(calib, "transforms", "base_to_single_chip.txt")),
        ), name=os.path.basename(os.path.dirname(os.path.abspath(path))))

    def __getitem__(self, index: int | np.integer) -> types.RadarADC[np.ndarray]:
        """Read a radar frame."""
        n_chirp, n_tx, n_rx, n_sample = self.metadata.waveform.shape
        raw = np.fromfile(
            os.path.join(self._path, "data", f"frame_{index}.bin"),
            dtype="<i2")
        expected = n_tx * n_rx * n_chirp * n_sample * 2
        if raw.size != expected:
            raise ValueError(
                f"Radar frame {index} has {raw.size} values; expected "
                f"{expected} for shape {self.metadata.waveform.shape}.")
        iq = np.ascontiguousarray(
            raw.reshape(n_tx, n_rx, n_chirp, n_sample, 2)
            .transpose(2, 0, 1, 3, 4))
        return types.RadarADC(
            iq=iq[None, None],
            timestamps=self.metadata.timestamps[index:index + 1][None],
            waveform=self.metadata.waveform,
            extrinsics=self.metadata.extrinsics)


class Lidar(abstract.Sensor[types.LidarPoints[np.ndarray], LidarMetadata]):
    """Ouster OS1-64 lidar point clouds.

    Args:
        path: path to the `lidar` directory of a run.
        calib: path to the ColoRadar calibration directory (`utils/calib`).
    """

    def __init__(self, path: str, calib: str) -> None:
        self._path = path
        super().__init__(metadata=LidarMetadata(
            timestamps=np.loadtxt(
                os.path.join(path, "timestamps.txt"),
                dtype=np.float64, ndmin=1),
            extrinsics=read_transform(
                os.path.join(calib, "transforms", "base_to_lidar.txt")),
        ), name=os.path.basename(os.path.dirname(os.path.abspath(path))))

    def __getitem__(
        self, index: int | np.integer
    ) -> types.LidarPoints[np.ndarray]:
        """Read a lidar point cloud."""
        points = np.fromfile(
            os.path.join(
                self._path, "pointclouds", f"lidar_pointcloud_{index}.bin"),
            dtype="<f4").reshape(-1, 4)
        return types.LidarPoints(
            points=points[None, None],
            timestamps=self.metadata.timestamps[index:index + 1][None],
            extrinsics=self.metadata.extrinsics)
