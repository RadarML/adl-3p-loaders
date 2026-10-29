"""GRADE traces and datasets."""

import os
from collections.abc import Callable, Mapping, Sequence
from multiprocessing.pool import ThreadPool
from typing import Literal, Self

import numpy as np
from abstract_dataloader import abstract, generic, spec
from jaxtyping import Integer

from . import sensors as gr_sensors

#: Default sensor type for each sensor name.
SENSOR_TYPES: dict[str, Callable[[str], spec.Sensor]] = {
    "radar": gr_sensors.Radar,
    "depth": gr_sensors.Depth,
    "smoke": gr_sensors.Smoke,
}

#: Sensors loaded by default; `smoke` is only present in evaluation sequences.
DEFAULT_SENSORS: tuple[str, ...] = ("radar", "depth")

#: Subdirectory of the dataset root containing each split's sequences.
SPLIT_PATHS: dict[str, str] = {
    "train": "GRADE_Train_Raw",
    "val": "GRADE_Train_Raw",
    "eval": "GRADE_Eval_Raw",
}

#: Sequences in `GRADE_Train_Raw` whose (lowercase) names contain any of these
#: are used for validation instead of training.
VAL_KEYWORDS: tuple[str, ...] = ("dell", "keck")


class Trace(abstract.Trace):
    """A single GRADE sequence.

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
            str, Callable[[str], spec.Sensor] | str | None] | None = None,
        sync: (
            spec.Synchronization | Mapping[str, Integer[np.ndarray, "N"]]
            | None) = generic.Empty()
    ) -> Self:
        """Create a trace from sensor configurations.

        `sensors` maps each sensor name (a key in `SENSOR_TYPES`, i.e.
        `radar`, `depth`, or `smoke`) to one of:

        - `Callable[[str], Sensor]`: a sensor constructor, which is called
            with the sequence directory.
        - `str`: the name of a sensor class in
            [`grade`][adl_3p_loaders.grade], e.g. `Radar`.
        - `None`: use the default type for that sensor name.

        If `sensors` is `None`, the sensors in `DEFAULT_SENSORS` (`radar` and
        `depth`) are loaded.

        Args:
            path: path to the sequence directory.
            sensors: sensor configuration.
            sync: synchronization protocol to apply, or explicit indices for
                each sensor; see [`Trace`][^^.].
        """
        if sensors is None:
            sensors = {k: None for k in DEFAULT_SENSORS}

        def _infer_type(
            k: str, v: Callable[[str], spec.Sensor] | str | None
        ) -> Callable[[str], spec.Sensor]:
            if isinstance(v, str):
                try:
                    return getattr(gr_sensors, v)
                except AttributeError as e:
                    raise ValueError(f"Unknown sensor type: {v}") from e
            elif v is None:
                try:
                    return SENSOR_TYPES[k]
                except KeyError as e:
                    raise ValueError(f"Unknown sensor name: {k}") from e
            return v

        init = {k: _infer_type(k, v)(path) for k, v in sensors.items()}
        return cls(
            sensors=init, sync=sync,
            name=os.path.basename(os.path.normpath(path)))


class Dataset(abstract.Dataset):
    """A GRADE dataset, consisting of multiple sequences."""

    @staticmethod
    def find(
        path: str, split: Literal["train", "val", "eval"] | None = None
    ) -> list[str]:
        """List sequences in the GRADE dataset.

        Sequences are subdirectories of `GRADE_Train_Raw` or `GRADE_Eval_Raw`
        which contain a `radar_*.h5` file. Following the GRADE dataset card,
        `GRADE_Train_Raw` sequences with `Dell` or `Keck` (case-insensitive)
        in their names are used for validation, and all `GRADE_Eval_Raw`
        sequences are used for evaluation.

        Args:
            path: path to the dataset root.
            split: split to list; if `None`, lists all sequences.

        Returns:
            Sorted list of sequence paths, relative to `path`, e.g.
                `GRADE_Eval_Raw/Smoke-dell-1-0`.
        """
        if split is None:
            splits = sorted(set(SPLIT_PATHS.values()))
        elif split in SPLIT_PATHS:
            splits = [SPLIT_PATHS[split]]
        else:
            raise ValueError(f"Unknown split: {split}")

        res = []
        for s in splits:
            root = os.path.join(path, s)
            if not os.path.isdir(root):
                continue
            for d in os.listdir(root):
                if not any(
                    f.startswith("radar_") and f.endswith(".h5")
                    for f in os.listdir(os.path.join(root, d))
                ):
                    continue
                is_val = any(k in d.lower() for k in VAL_KEYWORDS)
                if split == "train" and is_val:
                    continue
                if split == "val" and not is_val:
                    continue
                res.append(os.path.join(s, d))
        return sorted(res)

    @classmethod
    def from_config(
        cls, path: str, traces: Sequence[str] | None = None,
        split: Literal["train", "val", "eval"] | None = None,
        sensors: Mapping[
            str, Callable[[str], spec.Sensor] | str | None] | None = None,
        sync: spec.Synchronization | None = generic.Empty(),
        workers: int = 0
    ) -> Self:
        """Create a dataset from sensor configurations.

        Args:
            path: path to the dataset root.
            traces: sequence paths to include, relative to `path`; if `None`,
                all sequences in `split` are included (see [`find`][^.]).
            split: split to load if `traces` is not specified.
            sensors: sensor configuration; see [`Trace.from_config`][^^.].
            sync: synchronization protocol to apply; see
                [`Trace.from_config`][^^.].
            workers: number of threads to use when loading traces; if `0`,
                load traces sequentially, and if negative, use one thread per
                trace.
        """
        if traces is None:
            traces = cls.find(path, split=split)

        paths = [os.path.join(path, t) for t in traces]

        def _load(p: str) -> Trace:
            return Trace.from_config(p, sensors=sensors, sync=sync)

        if workers < 0:
            workers = len(paths)
        if workers == 0:
            loaded = [_load(p) for p in paths]
        else:
            with ThreadPool(workers) as pool:
                loaded = pool.map(_load, paths)
        return cls(traces=loaded)
