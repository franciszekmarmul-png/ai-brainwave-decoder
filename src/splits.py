from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data import CLASSES, EEGData

_N_FOLDS = 4
_N_CROSS_SUBJECT_TRAIN = 5
_N_CROSS_SUBJECT_VAL = 3
_CALIBRATION_VAL_FRACTION = 0.1
_FLAT_CHANNEL_STD = 1e-6


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def within_subject_splits(data: EEGData, subject: int) -> list[Split]:
    _require_subject(data, subject)
    blocks = np.array_split(np.flatnonzero(data.subject == subject), _N_FOLDS)
    splits = []
    for k in range(_N_FOLDS):
        val_block = (k + 1) % _N_FOLDS
        train_blocks = [blocks[i] for i in range(_N_FOLDS) if i not in (k, val_block)]
        splits.append(Split(train=np.concatenate(train_blocks), val=blocks[val_block], test=blocks[k]))
    return splits


def cross_subject_splits(data: EEGData, test_subject: int, n_repeats: int = 10, seed: int = 0) -> list[Split]:
    others = _other_subjects(data, test_subject)
    if len(others) != _N_CROSS_SUBJECT_TRAIN + _N_CROSS_SUBJECT_VAL:
        raise ValueError(f"cross-subject splits need 8 other people, got {len(others)}")
    rng = np.random.default_rng([seed, test_subject])
    test = _session_trials(data, test_subject, session=1)
    splits = []
    for _ in range(n_repeats):
        shuffled = rng.permutation(others)
        train_people = shuffled[:_N_CROSS_SUBJECT_TRAIN]
        val_people = shuffled[_N_CROSS_SUBJECT_TRAIN:]
        in_session = data.session == 0
        splits.append(
            Split(
                train=np.flatnonzero(np.isin(data.subject, train_people) & in_session),
                val=np.flatnonzero(np.isin(data.subject, val_people) & in_session),
                test=test,
            )
        )
    return splits


def calibration_base_split(data: EEGData, test_subject: int, seed: int = 0) -> Split:
    others = _other_subjects(data, test_subject)
    rng = np.random.default_rng([seed, test_subject])
    train, val = [], []
    for person in others:
        for label in range(len(CLASSES)):
            trials = np.flatnonzero((data.subject == person) & (data.session == 0) & (data.y == label))
            shuffled = rng.permutation(trials)
            n_val = round(len(trials) * _CALIBRATION_VAL_FRACTION)
            val.append(shuffled[:n_val])
            train.append(shuffled[n_val:])
    return Split(
        train=np.sort(np.concatenate(train)),
        val=np.sort(np.concatenate(val)),
        test=_session_trials(data, test_subject, session=1),
    )


def calibration_selection(data: EEGData, subject: int, n_trials: int, seed: int) -> np.ndarray:
    _require_subject(data, subject)
    if n_trials < 0 or n_trials % len(CLASSES):
        raise ValueError(f"n_trials must be a non-negative multiple of {len(CLASSES)}, got {n_trials}")
    per_class = n_trials // len(CLASSES)
    rng = np.random.default_rng([seed, subject])
    picks = []
    for label in range(len(CLASSES)):
        trials = np.flatnonzero((data.subject == subject) & (data.session == 0) & (data.y == label))
        if per_class > len(trials):
            raise ValueError(f"asked for {per_class} trials per class, but class {label} has {len(trials)}")
        picks.append(rng.permutation(trials)[:per_class])
    return np.sort(np.concatenate(picks))


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wide = X.astype(np.float64)
    mean = wide.mean(axis=(0, 2), keepdims=True)
    std = wide.std(axis=(0, 2), keepdims=True)
    return mean, np.where(std < _FLAT_CHANNEL_STD, 1.0, std)


def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def _require_subject(data: EEGData, subject: int) -> None:
    if subject not in data.subject:
        raise ValueError(f"no trials for subject {subject}")


def _other_subjects(data: EEGData, subject: int) -> np.ndarray:
    _require_subject(data, subject)
    return np.setdiff1d(np.unique(data.subject), [subject])


def _session_trials(data: EEGData, subject: int, session: int) -> np.ndarray:
    return np.flatnonzero((data.subject == subject) & (data.session == session))
