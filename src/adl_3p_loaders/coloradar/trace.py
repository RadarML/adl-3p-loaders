"""ColoRadar traces and datasets."""

import os
from collections.abc import Callable, Mapping, Sequence
from multiprocessing.pool import ThreadPool
from typing import Self, overload

import numpy as np
from abstract_dataloader import abstract, generic, spec
from jaxtyping import Integer

from . import sensors as cr_sensors

#: Default sensor type for each sensor name.
SENSOR_TYPES: dict[str, Callable[[str, str], spec.Sensor]] = {
    "radar": cr_sensors.Radar,
    "lidar": cr_sensors.Lidar,
}

#: Subdirectory of a run containing each sensor's data.
SENSOR_PATHS: dict[str, str] = {
    "radar": "single_chip",
    "lidar": "lidar",
}


class Trace(abstract.Trace):
    """A single ColoRadar run.

    Args:
        sensors: sensors which make up this trace.
        sync: synchronization protocol used to create global samples from
            asynchronous time series. If `Mapping`; the provided indices are
            used directly; if `None`, sensors are expected to already be
            synchronous (equivalent to passing `{k: np.arange(N), ...}`).
        name: friendly name; should only be used for debugging and inspection.
    """

    def __init__(
        self, sensors: Mapping[str, spec.Sensor],
        sync: (
            spec.Synchronization | Mapping[str, Integer[np.ndarray, "N"]]
            | None) = None,
        name: str = "trace"
    ) -> None:
        # Passthrough so that we get types in the docstring.
        super().__init__(sensors=sensors, sync=sync, name=name)

    @classmethod
    def from_config(
        cls, path: str,
        sensors: Mapping[
            str, Callable[[str, str], spec.Sensor] | str | None] | None = None,
        sync: spec.Synchronization | None = generic.Empty(),
        calib: str | None = None
    ) -> Self:
        """Create a trace from sensor configurations.

        `sensors` maps each sensor name (a key in `SENSOR_PATHS`, i.e.
        `radar` or `lidar`) to one of:

        - `Callable[[str, str], Sensor]`: a sensor constructor, which is
            called with the sensor directory and calibration directory.
        - `str`: the name of a sensor class in
            [`coloradar`][adl_3p_loaders.coloradar], e.g. `Radar`.
        - `None`: use the default type for that sensor name.

        If `sensors` is `None`, all supported sensors are loaded.

        Args:
            path: path to the run directory.
            sensors: sensor configuration.
            sync: synchronization protocol to apply.
            calib: path to the calibration directory; if `None`, uses
                `utils/calib` in the parent directory of `path`.
        """
        if sensors is None:
            sensors = {k: None for k in SENSOR_TYPES}
        if calib is None:
            calib = os.path.join(
                os.path.dirname(os.path.abspath(path)), "utils", "calib")

        def _infer_type(
            k: str, v: Callable[[str, str], spec.Sensor] | str | None
        ) -> Callable[[str, str], spec.Sensor]:
            if isinstance(v, str):
                try:
                    return getattr(cr_sensors, v)
                except AttributeError as e:
                    raise ValueError(f"Unknown sensor type: {v}") from e
            elif v is None:
                try:
                    return SENSOR_TYPES[k]
                except KeyError as e:
                    raise ValueError(f"Unknown sensor name: {k}") from e
            return v

        def _sensor_path(k: str) -> str:
            try:
                return os.path.join(path, SENSOR_PATHS[k])
            except KeyError as e:
                raise ValueError(f"Unknown sensor name: {k}") from e

        init = {
            k: _infer_type(k, v)(_sensor_path(k), calib)
            for k, v in sensors.items()}
        return cls(sensors=init, sync=sync, name=os.path.basename(path))


class Dataset(abstract.Dataset):
    """A ColoRadar dataset, consisting of multiple runs."""

    @staticmethod
    def find(path: str) -> list[str]:
        """List all runs in a ColoRadar dataset directory.

        A run is any subdirectory containing a `single_chip` or `lidar`
        directory.

        Args:
            path: path to the dataset root.

        Returns:
            Sorted list of run names.
        """
        return sorted(
            d for d in os.listdir(path)
            if any(
                os.path.isdir(os.path.join(path, d, s))
                for s in SENSOR_PATHS.values()))

    @classmethod
    def from_config(
        cls, path: str, traces: Sequence[str] | None = None,
        sensors: Mapping[
            str, Callable[[str, str], spec.Sensor] | str | None] | None = None,
        sync: spec.Synchronization | None = generic.Empty(),
        calib: str | None = None, workers: int = 0
    ) -> Self:
        """Create a dataset from sensor configurations.

        Args:
            path: path to the dataset root.
            traces: run names to include; if `None`, all runs are included
                (see [`find`][^.]).
            sensors: sensor configuration; see [`Trace.from_config`][^^.].
            sync: synchronization protocol to apply; see
                [`Trace.from_config`][^^.].
            calib: path to the calibration directory; if `None`, uses
                `{path}/utils/calib`.
            workers: number of threads to use when loading traces; if `0`,
                load traces sequentially, and if negative, use one thread per
                trace.
        """
        if traces is None:
            traces = cls.find(path)
        if calib is None:
            calib = os.path.join(path, "utils", "calib")

        paths = [os.path.join(path, t) for t in traces]

        def _load(p: str) -> Trace:
            return Trace.from_config(p, sensors=sensors, sync=sync, calib=calib)

        if workers < 0:
            workers = len(paths)
        if workers == 0:
            loaded = [_load(p) for p in paths]
        else:
            with ThreadPool(workers) as pool:
                loaded = pool.map(_load, paths)
        return cls(traces=loaded)


@overload
def split[TSample](
    dataset: abstract.Trace[TSample], start: float = 0.0, end: float = 1.0
) -> Trace: ...

@overload
def split[TSample](
    dataset: abstract.Dataset[TSample], start: float = 0.0, end: float = 1.0
) -> Dataset: ...

def split[TSample](
    dataset: abstract.Dataset[TSample] | abstract.Trace[TSample],
    start: float = 0.0, end: float = 1.0
) -> Dataset | Trace:
    """Get a contiguous sub-trace or sub-dataset.

    Each trace is split separately, so splitting a dataset takes the same
    `start` to `end` proportion of every trace.

    Args:
        dataset: trace or dataset to split.
        start: start of the split, as a proportion of the trace length (`0-1`).
        end: end of the split, as a proportion of the trace length (`0-1`).

    Returns:
        Trace or dataset with a contiguous subset of samples according to the
            start and end indices.

    Raises:
        ValueError: if the split range is invalid, or a trace has no sensors.
        TypeError: if a dataset contains a trace which is not an
            `abstract.Trace` (and therefore does not expose its sensors and
            indices).
    """
    if not (0 <= start < end <= 1):
        raise ValueError(
            f"Invalid split range: {start} - {end} (must be in [0, 1])")

    if isinstance(dataset, abstract.Trace):
        if dataset.indices is None:
            indices = {
                k: np.arange(len(v), dtype=np.int32)
                for k, v in dataset.sensors.items()}
        else:
            indices = dataset.indices

        for v in indices.values():
            istart = int(len(v) * start)
            iend = int(len(v) * end)
            break
        else:
            raise ValueError("There must be at least one sensor.")

        return Trace(
            sensors=dataset.sensors,
            sync={k: np.copy(v[istart:iend]) for k, v in indices.items()},
            name=dataset.name)

    traces: list[Trace] = []
    for t in dataset.traces:
        if not isinstance(t, abstract.Trace):
            raise TypeError(
                f"Cannot split trace of type {type(t).__name__}; must be an "
                f"`abstract.Trace`.")
        traces.append(split(t, start=start, end=end))
    return Dataset(traces=traces)
