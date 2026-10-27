"""ColoRadar loader tests.

Set `COLORADAR_PATH` to the ColoRadar dataset root to also run tests against
the real dataset.
"""

import os

import numpy as np
import pytest
from abstract_dataloader import generic, spec

from adl_3p_loaders import coloradar


def test_calib(fake):
    """Calibration files are parsed into batched arrays."""
    calib = os.path.join(fake.root, "utils", "calib")
    waveform = coloradar.read_waveform(
        os.path.join(calib, "single_chip", "waveform_cfg.txt"))
    assert waveform.shape == (6, 3, 4, 8)
    assert waveform.sample_rate.shape == (1,)
    assert np.isclose(waveform.sample_rate[0], 10666000)

    tf = coloradar.read_transform(
        os.path.join(calib, "transforms", "base_to_lidar.txt"))
    assert tf.translation.shape == (1, 3)
    assert tf.rotation.shape == (1, 4)
    assert np.allclose(tf.translation, [[-0.075, -0.02, 0.03618]])


def test_radar(fake):
    """Radar frames are loaded in (chirp, tx, rx, sample, iq) order."""
    path = os.path.join(fake.root, fake.run, "single_chip")
    calib = os.path.join(fake.root, "utils", "calib")
    radar = coloradar.Radar(path, calib)

    assert isinstance(radar, spec.Sensor)
    assert len(radar) == 3
    assert np.allclose(radar.metadata.timestamps, fake.radar_ts)

    for i in range(len(radar)):
        sample = radar[i]
        assert sample.iq.shape == (1, 1, 6, 3, 4, 8, 2)
        assert sample.iq.dtype == np.int16
        assert np.allclose(sample.timestamps, fake.radar_ts[i])
        iq = sample.iq[0, 0, ..., 0] + 1j * sample.iq[0, 0, ..., 1]
        assert np.array_equal(iq, fake.radar[i].transpose(2, 0, 1, 3))


def test_lidar(fake):
    """Lidar point clouds are loaded as stored."""
    path = os.path.join(fake.root, fake.run, "lidar")
    calib = os.path.join(fake.root, "utils", "calib")
    lidar = coloradar.Lidar(path, calib)

    assert isinstance(lidar, spec.Sensor)
    assert len(lidar) == len(fake.lidar)
    for i in range(len(lidar)):
        sample = lidar[i]
        assert sample.points.shape == (1, 1, 16, 4)
        assert sample.points.dtype == np.float32
        assert np.array_equal(sample.points[0, 0], fake.lidar[i])


def test_trace(fake):
    """Traces synchronize radar and lidar; default sync yields no samples."""
    path = os.path.join(fake.root, fake.run)

    empty = coloradar.Trace.from_config(path)
    assert len(empty) == 0
    assert isinstance(empty["radar"], coloradar.Radar)

    trace = coloradar.Trace.from_config(
        path, sync=generic.Nearest("radar", tol=0.05))
    assert len(trace) == 3
    sample = trace[1]
    assert set(sample.keys()) == {"radar", "lidar"}
    assert np.allclose(sample["lidar"].timestamps, fake.lidar_ts[1])

    radar_only = coloradar.Trace.from_config(
        path, sensors={"radar": "Radar"}, sync=None)
    assert len(radar_only) == 3


def test_trace_errors(fake):
    """Unknown sensor names and types are rejected."""
    path = os.path.join(fake.root, fake.run)
    with pytest.raises(ValueError):
        coloradar.Trace.from_config(path, sensors={"cascade": None})
    with pytest.raises(ValueError):
        coloradar.Trace.from_config(path, sensors={"radar": "Cascade"})


def test_dataset(fake):
    """Datasets discover runs and skip the calibration directory."""
    assert coloradar.Dataset.find(fake.root) == [fake.run]
    for workers in (0, -1):
        dataset = coloradar.Dataset.from_config(
            fake.root, sync=generic.Nearest("radar", tol=0.05),
            workers=workers)
        assert len(dataset) == 3
        assert dataset[2]["radar"].iq.shape == (1, 1, 6, 3, 4, 8, 2)


def test_radar_size_mismatch(fake):
    """Truncated radar frames raise an error."""
    path = os.path.join(fake.root, fake.run, "single_chip")
    calib = os.path.join(fake.root, "utils", "calib")
    np.zeros(10, dtype="<i2").tofile(
        os.path.join(path, "adc_samples", "data", "frame_0.bin"))
    with pytest.raises(ValueError):
        coloradar.Radar(path, calib)[0]


@pytest.mark.skipif(
    "COLORADAR_PATH" not in os.environ, reason="COLORADAR_PATH is not set.")
def test_real_data():
    """Load the real ColoRadar dataset."""
    root = os.environ["COLORADAR_PATH"]
    dataset = coloradar.Dataset.from_config(
        root, sync=generic.Nearest("radar", tol=0.1), workers=-1)
    assert len(dataset.traces) > 0
    assert len(dataset) > 0

    trace = dataset.traces[0]
    assert isinstance(trace, coloradar.Trace)
    radar = trace["radar"]
    assert isinstance(radar, coloradar.Radar)
    sample = radar[0]

    # Reference: official ColoRadar loader (dataset_loaders.get_adc_frame)
    n_chirp, n_tx, n_rx, n_sample = radar.metadata.waveform.shape
    vals = np.fromfile(
        os.path.join(
            root, trace.name, "single_chip", "adc_samples", "data",
            "frame_0.bin"), dtype="<i2").astype(float)
    ref = (vals[:-1:2] + 1j * vals[1::2]).reshape(
        n_tx, n_rx, n_chirp, n_sample)
    iq = sample.iq[0, 0, ..., 0] + 1j * sample.iq[0, 0, ..., 1]
    assert np.array_equal(iq, ref.transpose(2, 0, 1, 3))

    points = trace["lidar"][0].points
    assert points.shape == (1, 1, 65536, 4)


def test_get_transform(fake):
    """Transforms compose through the base frame and invert consistently."""
    calib = os.path.join(fake.root, "utils", "calib")
    l2r = coloradar.get_transform(calib, "lidar", "single_chip")
    r2l = coloradar.get_transform(calib, "single_chip", "lidar")
    assert l2r.shape == (4, 4)
    assert np.allclose(l2r @ r2l, np.eye(4))
    assert np.allclose(np.linalg.det(l2r[:3, :3]), 1.0)

    # The lidar origin is at the lidar translation in the base frame.
    l2b = coloradar.get_transform(calib, "lidar", "base")
    assert np.allclose(l2b[:3, 3], [-0.075, -0.02, 0.03618])
    assert np.allclose(
        coloradar.get_transform(calib, "base", "base"), np.eye(4))

    # Both sensors are yawed ~90 / ~-92 deg, so lidar -> radar is ~180 deg.
    assert l2r[0, 0] < -0.99 and l2r[1, 1] < -0.99


def test_split(fake):
    """Splits take contiguous proportions of each trace."""
    calib = os.path.join(fake.root, "utils", "calib")
    trace = coloradar.Trace.from_config(
        os.path.join(fake.root, fake.run),
        sync=generic.Nearest("radar", tol=0.1), calib=calib)
    dataset = coloradar.Dataset(traces=[trace, trace])

    head = coloradar.split(trace, start=0.0, end=0.5)
    tail = coloradar.split(trace, start=0.5, end=1.0)
    assert isinstance(head, coloradar.Trace)
    assert len(head) + len(tail) == len(trace)
    assert np.allclose(
        tail[0]["radar"].timestamps, trace[len(head)]["radar"].timestamps)

    sub = coloradar.split(dataset, start=0.0, end=0.5)
    assert isinstance(sub, coloradar.Dataset)
    assert len(sub) == 2 * len(head)

    unsynced = coloradar.Trace.from_config(
        os.path.join(fake.root, fake.run), sensors={"radar": None},
        sync=None, calib=calib)
    assert len(coloradar.split(unsynced, end=0.5)) == 1

    with pytest.raises(ValueError):
        coloradar.split(trace, start=0.5, end=0.5)
