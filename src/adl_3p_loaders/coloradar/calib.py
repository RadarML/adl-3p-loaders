"""ColoRadar calibration file parsers."""

import numpy as np

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
