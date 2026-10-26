"""ColoRadar traces and datasets."""

import os
from collections.abc import Callable, Mapping, Sequence
from multiprocessing.pool import ThreadPool

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
    ) -> "Trace":
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
    ) -> "Dataset":
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
