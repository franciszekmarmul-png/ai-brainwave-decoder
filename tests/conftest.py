import numpy as np
import torch

from src.data import EEGData

torch.set_num_threads(1)


def make_synthetic(n_subjects: int = 4, n_per_class: int = 72, seed: int = 0, rhythm_hz: float = 10.0) -> EEGData:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(4), n_per_class)
    y = np.concatenate([rng.permutation(labels) for _ in range(2 * n_subjects)])
    n_session = len(labels)
    subject = np.repeat(np.arange(1, n_subjects + 1), 2 * n_session)
    session = np.tile(np.repeat([0, 1], n_session), n_subjects)

    t = np.arange(256) / 128
    phase = rng.uniform(0, 2 * np.pi, (len(y), 1))
    rhythm = np.sin(2 * np.pi * rhythm_hz * t + phase)
    active = np.arange(22) % 4 == y[:, None]
    X = rng.normal(size=(len(y), 22, 256)) + 3.0 * active[:, :, None] * rhythm[:, None, :]
    return EEGData(
        X=X.astype(np.float32),
        y=y.astype(np.int64),
        subject=subject.astype(np.int64),
        session=session.astype(np.int64),
    )
