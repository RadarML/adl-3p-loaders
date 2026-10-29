"""GRADE loader tests.

Set `GRADE_PATH` to the GRADE dataset root to also run tests against the real
dataset.
"""

import dataclasses
import os

import h5py
import numpy as np
import pytest
from abstract_dataloader import generic, spec

from adl_3p_loaders import grade


def test_radar(fake_grade):
    """Radar frames are un-interleaved into (chirp, tx, rx, sample, iq)."""
    radar = grade.Radar(os.path.join(fake_grade.root, fake_grade.seq))

    assert isinstance(radar, spec.Sensor)
    assert len(radar) == 3
    assert np.allclose(radar.metadata.timestamps, fake_grade.radar_ts)
    assert radar.metadata.waveform.shape == (64, 3, 4, 256)

    for i in range(len(radar)):
        sample = radar[i]
        assert sample.iq.shape == (1, 1, 64, 3, 4, 256, 2)
        assert sample.iq.dtype == np.int16
        assert np.allclose(sample.timestamps, fake_grade.radar_ts[i])
        iq = sample.iq[0, 0, ..., 0] + 1j * sample.iq[0, 0, ..., 1]
        assert np.array_equal(iq, fake_grade.radar[i])


def test_radar_past(fake_grade):
    """Past frames are stacked along the time axis."""
    radar = grade.Radar(os.path.join(fake_grade.root, fake_grade.seq), past=1)
    sample = radar[2]
    assert sample.iq.shape == (1, 2, 64, 3, 4, 256, 2)
    assert np.allclose(sample.timestamps[0], fake_grade.radar_ts[1:3])
    with pytest.raises(IndexError):
        radar[0]


def test_radar_shape_mismatch(fake_grade):
    """Waveforms which do not match the stored data are rejected."""
    waveform = dataclasses.replace(
        grade.default_waveform(), shape=(32, 3, 4, 256))
    with pytest.raises(ValueError):
        grade.Radar(
            os.path.join(fake_grade.root, fake_grade.seq), waveform=waveform)


def test_range_resolution():
    """The default waveform matches the GRADE range resolution."""
    wf = grade.default_waveform()
    res = 299792458 * wf.sample_rate / (
        2 * wf.frequency_slope * wf.shape[-1])
    assert np.isclose(res[0], 0.04375, atol=1e-4)


def test_depth(fake_grade):
    """Depth frames use the (truncated) camera timestamps."""
    depth = grade.Depth(os.path.join(fake_grade.root, fake_grade.seq))

    assert isinstance(depth, spec.Sensor)
    assert len(depth) == 4
    assert np.allclose(depth.metadata.timestamps, fake_grade.depth_ts)
    sample = depth[1]
    assert sample.depth.shape == (1, 1, 6, 8)
    assert sample.depth.dtype == np.uint16
    assert np.array_equal(sample.depth[0, 0], fake_grade.depth[1])
    assert sample.intrinsics.shape == (1, 4)


def test_depth_missing_timestamps(fake_grade):
    """Depth files with more frames than timestamps are rejected."""
    path = os.path.join(fake_grade.root, fake_grade.seq)
    with h5py.File(os.path.join(path, "zed_depth.h5"), "a") as f:
        del f["depth_mm"]
        f.create_dataset("depth_mm", data=np.zeros((9, 6, 8), np.uint16))
    with pytest.raises(ValueError):
        grade.Depth(path)


def test_smoke(fake_grade):
    """Smoke readings are loaded as stored."""
    smoke = grade.Smoke(os.path.join(fake_grade.root, fake_grade.seq))

    assert isinstance(smoke, spec.Sensor)
    assert len(smoke) == len(fake_grade.smoke)
    sample = smoke[3]
    assert sample.samples.shape == (1, 1, 3)
    assert np.array_equal(sample.samples[0, 0], fake_grade.smoke[3])

    with pytest.raises(FileNotFoundError):
        grade.Smoke(
            os.path.join(fake_grade.root, "GRADE_Train_Raw", "AlicePratt-1"))


def test_trace(fake_grade):
    """Traces synchronize radar and depth; default sync yields no samples."""
    path = os.path.join(fake_grade.root, fake_grade.seq)

    empty = grade.Trace.from_config(path)
    assert len(empty) == 0
    assert empty.name == "Smoke-keck-1-2"
    assert isinstance(empty["radar"], grade.Radar)

    trace = grade.Trace.from_config(
        path, sensors={"radar": None, "depth": "Depth", "smoke": None},
        sync=generic.Nearest("radar", tol=0.05))
    assert len(trace) == 3
    sample = trace[1]
    assert set(sample.keys()) == {"radar", "depth", "smoke"}
    assert np.allclose(sample["depth"].timestamps, fake_grade.depth_ts[1])

    with pytest.raises(ValueError):
        grade.Trace.from_config(path, sensors={"lidar": None})
    with pytest.raises(ValueError):
        grade.Trace.from_config(path, sensors={"radar": "Lidar"})


def test_dataset(fake_grade):
    """Datasets discover sequences and apply the dataset card splits."""
    root = fake_grade.root
    assert grade.Dataset.find(root, split="train") == [
        "GRADE_Train_Raw/AlicePratt-1"]
    assert grade.Dataset.find(root, split="val") == [
        "GRADE_Train_Raw/Smoke-Dell-1"]
    assert grade.Dataset.find(root, split="eval") == [fake_grade.seq]
    assert len(grade.Dataset.find(root)) == 3
    with pytest.raises(ValueError):
        grade.Dataset.find(root, split="test")  # type: ignore

    for workers in (0, -1):
        dataset = grade.Dataset.from_config(
            root, sync=generic.Nearest("radar", tol=0.05), workers=workers)
        assert len(dataset) == 9
        assert dataset[8]["depth"].depth.shape == (1, 1, 6, 8)


@pytest.mark.skipif(
    "GRADE_PATH" not in os.environ, reason="GRADE_PATH is not set.")
def test_real_data():
    """Load the real GRADE dataset."""
    root = os.environ["GRADE_PATH"]
    dataset = grade.Dataset.from_config(
        root, split="eval", sensors={"radar": None, "depth": None,
                                     "smoke": None},
        sync=generic.Nearest("radar", tol=0.05), workers=-1)
    assert len(dataset.traces) > 0
    assert len(dataset) > 0

    trace = dataset.traces[0]
    assert isinstance(trace, grade.Trace)
    sample = trace[0]
    assert sample["radar"].iq.shape == (1, 1, 64, 3, 4, 256, 2)
    assert sample["depth"].depth.shape == (1, 1, 720, 1280)
    assert sample["smoke"].samples.shape == (1, 1, 3)

    # Reference: official GRADE processing code (`iiqq_to_iq`)
    radar = trace["radar"]
    assert isinstance(radar, grade.Radar)
    with h5py.File(grade.find_file(
        os.path.join(root, dataset.find(root, split="eval")[0]),
        "radar_*.h5"
    ), "r") as f:
        iiqq = f["radar_data"][0]
    ref = np.zeros((*iiqq.shape[:-1], iiqq.shape[-1] // 2), np.complex64)
    ref[..., 0::2] = 1j * iiqq[..., 0::4] + iiqq[..., 2::4]
    ref[..., 1::2] = 1j * iiqq[..., 1::4] + iiqq[..., 3::4]
    iq = radar[0].iq[0, 0]
    assert np.array_equal(iq[..., 0] + 1j * iq[..., 1], ref)
