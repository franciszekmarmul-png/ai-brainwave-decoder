from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

N_SAMPLES = 256
_RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

if "MNE_DATA" not in os.environ:
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["MNE_DATA"] = str(_RAW_DIR.resolve())


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    moabb_name: str
    classes: tuple[str, ...]
    n_channels: int
    subjects: tuple[int, ...]

    @property
    def chance(self) -> float:
        return 1 / len(self.classes)


DATASETS = {
    spec.key: spec
    for spec in (
        DatasetSpec(
            "bnci2014_001", "BCI IV 2a", "BNCI2014_001",
            ("left_hand", "right_hand", "feet", "tongue"), 22, tuple(range(1, 10)),
        ),
        DatasetSpec(
            "bnci2014_004", "BCI IV 2b", "BNCI2014_004",
            ("left_hand", "right_hand"), 3, tuple(range(1, 10)),
        ),
        DatasetSpec(
            "physionet_mi", "PhysioNet", "PhysionetMI",
            ("left_hand", "right_hand"), 64, tuple(range(1, 11)),
        ),
    )
}
DEFAULT_DATASET = DATASETS["bnci2014_001"]
CLASSES = DEFAULT_DATASET.classes
N_CHANNELS = DEFAULT_DATASET.n_channels


@dataclass
class EEGData:
    X: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    session: np.ndarray

    def subset(self, idx: np.ndarray) -> "EEGData":
        return EEGData(self.X[idx], self.y[idx], self.subject[idx], self.session[idx])


def validate(data: EEGData, spec: DatasetSpec = DEFAULT_DATASET) -> None:
    if data.X.ndim != 3 or data.X.shape[1:] != (spec.n_channels, N_SAMPLES):
        raise ValueError(f"expected trials shaped (n, {spec.n_channels}, {N_SAMPLES}), got {data.X.shape}")
    if not len(data.X) == len(data.y) == len(data.subject) == len(data.session):
        raise ValueError("X, y, subject and session have different lengths")
    if not np.isfinite(data.X).all():
        raise ValueError("trials contain non-finite values")
    known = np.arange(len(spec.classes))
    if not np.isin(data.y, known).all():
        raise ValueError(f"unknown labels: {sorted(set(data.y.tolist()) - set(known.tolist()))}")


def load_dataset(
    subjects: tuple[int, ...] | None = None,
    cache_path: Path | None = None,
    force: bool = False,
    dataset: str = DEFAULT_DATASET.key,
) -> EEGData:
    spec = DATASETS[dataset]
    subjects = spec.subjects if subjects is None else tuple(subjects)
    cache_path = Path(cache_path or PROCESSED_DIR / f"{spec.key}.npz")
    cached = None
    if cache_path.exists() and not force:
        cached = _read_cache(cache_path, spec)
        if set(subjects) <= set(cached.subject.tolist()):
            return cached.subset(np.flatnonzero(np.isin(cached.subject, subjects)))
    missing = subjects if cached is None else tuple(s for s in subjects if s not in set(cached.subject.tolist()))
    try:
        fetched = _assemble(*_fetch_moabb(missing, spec), spec)
    except Exception as err:
        raise RuntimeError(
            f"Could not build {cache_path} from the downloaded EEG data: {err}. "
            f"Check your connection and retry with force=True. If it keeps failing, "
            f"delete {_RAW_DIR} so the recordings download again."
        ) from err
    validate(fetched, spec)
    data = fetched if cached is None else _merge(cached, fetched)
    _write_cache(data, cache_path)
    return data.subset(np.flatnonzero(np.isin(data.subject, subjects)))


def _fetch_moabb(subjects: tuple[int, ...], spec: DatasetSpec = DEFAULT_DATASET):
    from moabb import datasets
    from moabb.paradigms import LeftRightImagery, MotorImagery

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    window = dict(fmin=4, fmax=40, tmin=0.5, tmax=2.5, resample=128)
    if len(spec.classes) == 2:
        paradigm = LeftRightImagery(**window)
    else:
        paradigm = MotorImagery(n_classes=len(spec.classes), **window)
    source = getattr(datasets, spec.moabb_name)()
    return paradigm.get_data(dataset=source, subjects=list(subjects))


def _assemble(X: np.ndarray, labels: np.ndarray, metadata: pd.DataFrame, spec: DatasetSpec = DEFAULT_DATASET) -> EEGData:
    session_index = {name: i for i, name in enumerate(sorted(metadata["session"].unique()))}
    table = pd.DataFrame(
        {
            "subject": metadata["subject"].to_numpy(),
            "session": [session_index[s] for s in metadata["session"]],
            "run": pd.to_numeric(metadata["run"], errors="coerce").fillna(0).astype(int).to_numpy(),
        }
    )
    order = table.sort_values(["subject", "session", "run"], kind="stable").index.to_numpy()
    label_codes = {name: code for code, name in enumerate(spec.classes)}
    return EEGData(
        X=np.asarray(X, dtype=np.float32)[order][:, :, :N_SAMPLES],
        y=np.array([label_codes[name] for name in labels], dtype=np.int64)[order],
        subject=table["subject"].to_numpy(dtype=np.int64)[order],
        session=table["session"].to_numpy(dtype=np.int64)[order],
    )


def _merge(a: EEGData, b: EEGData) -> EEGData:
    joined = EEGData(
        X=np.concatenate([a.X, b.X]),
        y=np.concatenate([a.y, b.y]),
        subject=np.concatenate([a.subject, b.subject]),
        session=np.concatenate([a.session, b.session]),
    )
    order = np.argsort(joined.subject, kind="stable")
    return joined.subset(order)


def _read_cache(path: Path, spec: DatasetSpec = DEFAULT_DATASET) -> EEGData:
    try:
        with np.load(path, allow_pickle=False) as f:
            data = EEGData(X=f["X"], y=f["y"], subject=f["subject"], session=f["session"])
        validate(data, spec)
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as err:
        raise RuntimeError(
            f"Cache file {path} can't be read ({err}). Rebuild it with load_dataset(force=True)."
        ) from err
    return data


def _write_cache(data: EEGData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as f:
        np.savez(f, X=data.X, y=data.y, subject=data.subject, session=data.session)
    temporary.replace(path)
