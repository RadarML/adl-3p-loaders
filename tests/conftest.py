"""Shared test fixtures."""

import os
from dataclasses import dataclass

import h5py
import numpy as np
import pytest

WAVEFORM_CFG = """num_rx 4
num_tx 3
num_adc_samples_per_chirp 8
num_chirps_per_frame 6
adc_sample_frequency 10666000
start_frequency 76999999488.0
idle_time 0.000110000000859
adc_start_time 7.00000009601e-06
ramp_end_time 1.99999994948e-05
frequency_slope 1.00000000377e+14
"""


@dataclass
class FakeColoRadar:
    """Synthetic ColoRadar root with a single run."""

    root: str
    run: str
    radar: np.ndarray  # complex, (frame, tx, rx, chirp, sample)
    radar_ts: np.ndarray
    lidar: np.ndarray  # float32, (frame, n, 4)
    lidar_ts: np.ndarray


@pytest.fixture
def fake(tmp_path) -> FakeColoRadar:
    """Write a synthetic ColoRadar dataset to a temporary directory."""
    root = str(tmp_path)
    run = "run0"
    rng = np.random.default_rng(42)

    calib = os.path.join(root, "utils", "calib")
    os.makedirs(os.path.join(calib, "single_chip"))
    os.makedirs(os.path.join(calib, "transforms"))
    with open(os.path.join(calib, "single_chip", "waveform_cfg.txt"), "w") as f:
        f.write(WAVEFORM_CFG)
    with open(os.path.join(calib, "transforms", "base_to_single_chip.txt"), "w") as f:
        f.write("-0.145 0.09 -0.025\n0.0 0.0 0.706825181105 0.707388269167\n")
    with open(os.path.join(calib, "transforms", "base_to_lidar.txt"), "w") as f:
        f.write("-0.075 -0.02 0.03618\n0.0 0.0 0.721382357437 -0.692536998563\n")

    adc = os.path.join(root, run, "single_chip", "adc_samples")
    os.makedirs(os.path.join(adc, "data"))
    radar = (
        rng.integers(-1000, 1000, size=(3, 3, 4, 6, 8))
        + 1j * rng.integers(-1000, 1000, size=(3, 3, 4, 6, 8)))
    radar_ts = 1608589697.0 + 0.1 * np.arange(3)
    np.savetxt(os.path.join(adc, "timestamps.txt"), radar_ts, fmt="%.6f")
    for i, frame in enumerate(radar):
        interleaved = np.stack([frame.real, frame.imag], axis=-1)
        interleaved.astype("<i2").tofile(
            os.path.join(adc, "data", f"frame_{i}.bin"))

    lidar_dir = os.path.join(root, run, "lidar")
    os.makedirs(os.path.join(lidar_dir, "pointclouds"))
    # Lidar frames bracket the radar frames so that every radar frame has a
    # nearest lidar frame.
    lidar = rng.normal(size=(4, 16, 4)).astype(np.float32)
    lidar_ts = radar_ts[0] - 0.03 + 0.1 * np.arange(4)
    np.savetxt(os.path.join(lidar_dir, "timestamps.txt"), lidar_ts, fmt="%.6f")
    for i, cloud in enumerate(lidar):
        cloud.astype("<f4").tofile(
            os.path.join(lidar_dir, "pointclouds", f"lidar_pointcloud_{i}.bin"))

    return FakeColoRadar(
        root=root, run=run, radar=radar, radar_ts=radar_ts,
        lidar=lidar, lidar_ts=lidar_ts)


@dataclass
class FakeGRADE:
    """Synthetic GRADE root with train, validation, and eval sequences."""

    root: str
    seq: str  # eval sequence with all sensors
    radar: np.ndarray  # complex, (frame, chirp, tx, rx, sample)
    radar_ts: np.ndarray  # seconds
    depth: np.ndarray  # uint16, (frame, height, width)
    depth_ts: np.ndarray  # seconds
    smoke: np.ndarray  # uint32, (n, 3)
    smoke_ts: np.ndarray  # seconds


def _write_grade_sequence(
    path: str, rng: np.random.Generator, smoke: bool
) -> tuple[np.ndarray, ...]:
    os.makedirs(path)
    shape = (3, 64, 3, 4, 256)
    radar = (
        rng.integers(-1000, 1000, size=shape)
        + 1j * rng.integers(-1000, 1000, size=shape))
    # DCA1000 IIQQ: (Q0, Q1, I0, I1) for each pair of samples.
    iiqq = np.zeros((*shape[:-1], 512), dtype=np.int16)
    iiqq[..., 0::4] = radar.imag[..., 0::2]
    iiqq[..., 1::4] = radar.imag[..., 1::2]
    iiqq[..., 2::4] = radar.real[..., 0::2]
    iiqq[..., 3::4] = radar.real[..., 1::2]
    radar_ts = 1772259202.0 + 0.1 * np.arange(3)
    with h5py.File(os.path.join(path, "radar_20260228_001319.h5"), "w") as f:
        f.create_dataset("radar_data", data=iiqq)
        f.create_dataset("timestamps_ms", data=radar_ts * 1000)

    # Depth brackets radar; one surplus trailing camera timestamp.
    depth = rng.integers(0, 10000, size=(4, 6, 8)).astype(np.uint16)
    camera_ts = radar_ts[0] - 0.03 + 0.1 * np.arange(5)
    with h5py.File(os.path.join(path, "zed_depth.h5"), "w") as f:
        f.create_dataset("depth_mm", data=depth)
        f.attrs["intrinsics_fx_fy_cx_cy"] = np.array(
            [521.4, 521.4, 636.3, 373.1], dtype=np.float32)
    with h5py.File(
        os.path.join(path, "camera_timestamps_20260228_001321.h5"), "w"
    ) as f:
        f.create_dataset("timestamps_ms", data=camera_ts * 1000)

    samples = rng.integers(0, 5000, size=(8, 3)).astype(np.uint32)
    smoke_ts = radar_ts[0] - 0.05 + 0.04 * np.arange(8)
    if smoke:
        with h5py.File(
            os.path.join(path, "max30105_20260228_001320.h5"), "w"
        ) as f:
            f.create_dataset("samples_uint32", data=samples)
            f.create_dataset("timestamps_ms", data=smoke_ts * 1000)

    return radar, radar_ts, depth, camera_ts[:4], samples, smoke_ts


@pytest.fixture
def fake_grade(tmp_path) -> FakeGRADE:
    """Write a synthetic GRADE dataset to a temporary directory."""
    root = str(tmp_path)
    rng = np.random.default_rng(42)

    for seq in ["AlicePratt-1", "Smoke-Dell-1"]:
        _write_grade_sequence(
            os.path.join(root, "GRADE_Train_Raw", seq), rng, smoke=False)
    os.makedirs(os.path.join(root, "GRADE_Train_Raw", "not-a-sequence"))

    seq = "GRADE_Eval_Raw/Smoke-keck-1-2"
    radar, radar_ts, depth, depth_ts, smoke, smoke_ts = _write_grade_sequence(
        os.path.join(root, seq), rng, smoke=True)
    return FakeGRADE(
        root=root, seq=seq, radar=radar, radar_ts=radar_ts,
        depth=depth, depth_ts=depth_ts, smoke=smoke, smoke_ts=smoke_ts)
