"""ColoRadar calibration file parsers."""

import os

import numpy as np
from jaxtyping import Float64
from scipy.spatial.transform import Rotation

from . import types


def read_waveform(path: str) -> types.Waveform[np.ndarray]:
    """Read a ColoRadar `waveform_cfg.txt` file.

    Args:
        path: path to the waveform configuration file.

    Returns:
        Chirp waveform parameters.
    """
    with open(path) as f:
        cfg = {
            k: v for k, v in (
                line.split() for line in f if line.strip())}

    def _f32(key: str) -> np.ndarray:
        return np.array([float(cfg[key])], dtype=np.float32)

    return types.Waveform(
        start_frequency=_f32("start_frequency"),
        frequency_slope=_f32("frequency_slope"),
        sample_rate=_f32("adc_sample_frequency"),
        idle_time=_f32("idle_time"),
        adc_start_time=_f32("adc_start_time"),
        ramp_end_time=_f32("ramp_end_time"),
        shape=(
            int(cfg["num_chirps_per_frame"]), int(cfg["num_tx"]),
            int(cfg["num_rx"]), int(cfg["num_adc_samples_per_chirp"])))


def read_transform(path: str) -> types.Transform[np.ndarray]:
    """Read a ColoRadar `base_to_*.txt` transform file.

    The first line contains the translation `tx ty tz`, and the second line
    contains the rotation quaternion `qx qy qz qw`.

    Args:
        path: path to the transform file.

    Returns:
        Transform from the sensor frame to the base frame.
    """
    with open(path) as f:
        lines = [line.split() for line in f if line.strip()]
    return types.Transform(
        translation=np.array([lines[0]], dtype=np.float64),
        rotation=np.array([lines[1]], dtype=np.float64))


def get_transform(
    calib: str, from_sensor: str, to_sensor: str
) -> Float64[np.ndarray, "4 4"]:
    """Get the rigid transform between two ColoRadar sensor frames.

    Sensors are named by their calibration file suffix, i.e.
    `transforms/base_to_{sensor}.txt` (e.g. `lidar`, `single_chip`); `base`
    refers to the base frame itself.

    Args:
        calib: path to the ColoRadar calibration directory (`utils/calib`).
        from_sensor: source frame.
        to_sensor: target frame.

    Returns:
        4x4 homogeneous transform which maps points in the `from_sensor` frame
            to the `to_sensor` frame.
    """
    def _to_base(sensor: str) -> Float64[np.ndarray, "4 4"]:
        res = np.eye(4)
        if sensor != "base":
            tf = read_transform(
                os.path.join(calib, "transforms", f"base_to_{sensor}.txt"))
            res[:3, :3] = Rotation.from_quat(tf.rotation[0]).as_matrix()
            res[:3, 3] = tf.translation[0]
        return res

    return np.linalg.inv(_to_base(to_sensor)) @ _to_base(from_sensor)
