import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import make_synthetic

from src import data as data_module
from src.data import EEGData, load_dataset, validate


def fake_moabb(subjects, spec=None, n_samples=257, value=0.0):
    rows = [(s, sess, run) for s in subjects for sess in ("0train", "1test") for run in ("0", "1")]
    metadata = pd.DataFrame(rows, columns=["subject", "session", "run"])
    X = np.full((len(rows), 22, n_samples), value, dtype=np.float64)
    return X, np.array(["left_hand"] * len(rows)), metadata


def test_validate_accepts_good_data():
    validate(make_synthetic(n_subjects=1, n_per_class=2))


def test_validate_rejects_wrong_sample_count():
    d = make_synthetic(n_subjects=1, n_per_class=2)
    d.X = d.X[:, :, :255]
    with pytest.raises(ValueError, match="256"):
        validate(d)


def test_validate_rejects_wrong_channel_count():
    d = make_synthetic(n_subjects=1, n_per_class=2)
    d.X = d.X[:, :21, :]
    with pytest.raises(ValueError, match="22"):
        validate(d)


def test_validate_rejects_nan():
    d = make_synthetic(n_subjects=1, n_per_class=2)
    d.X[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate(d)


def test_validate_rejects_unknown_label():
    d = make_synthetic(n_subjects=1, n_per_class=2)
    d.y[0] = 4
    with pytest.raises(ValueError, match="label"):
        validate(d)


def test_validate_rejects_misaligned_fields():
    d = make_synthetic(n_subjects=1, n_per_class=2)
    d.subject = d.subject[:-1]
    with pytest.raises(ValueError, match="length"):
        validate(d)


def test_subset_keeps_every_field_aligned():
    d = EEGData(
        X=np.arange(3 * 22 * 256, dtype=np.float32).reshape(3, 22, 256),
        y=np.array([0, 1, 2]),
        subject=np.array([1, 2, 3]),
        session=np.array([0, 1, 0]),
    )
    s = d.subset(np.array([2, 0]))
    assert s.y.tolist() == [2, 0]
    assert s.subject.tolist() == [3, 1]
    assert s.session.tolist() == [0, 0]
    assert s.X[0, 0, 0] == d.X[2, 0, 0]


def test_download_is_sorted_labelled_trimmed_and_typed(tmp_path, monkeypatch):
    rows = [
        (2, "0train", "0", "tongue"),
        (1, "1test", "0", "feet"),
        (1, "0train", "10", "right_hand"),
        (1, "0train", "2", "left_hand"),
        (1, "0train", "2", "tongue"),
    ]
    metadata = pd.DataFrame([r[:3] for r in rows], columns=["subject", "session", "run"])
    labels = np.array([r[3] for r in rows])
    X = np.zeros((5, 22, 257), dtype=np.float64)
    X[:, 0, 0] = [0, 1, 2, 3, 4]
    monkeypatch.setattr(data_module, "_fetch_moabb", lambda subjects, spec=None: (X, labels, metadata))

    d = load_dataset(subjects=(1, 2), cache_path=tmp_path / "c.npz")

    assert d.X[:, 0, 0].tolist() == [3, 4, 2, 1, 0]
    assert d.y.tolist() == [0, 3, 1, 2, 3]
    assert d.subject.tolist() == [1, 1, 1, 1, 2]
    assert d.session.tolist() == [0, 0, 0, 1, 0]
    assert d.X.shape == (5, 22, 256)
    assert d.X.dtype == np.float32
    assert d.y.dtype == np.int64


def test_short_trials_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "_fetch_moabb", lambda subjects, spec=None: fake_moabb(subjects, n_samples=255))
    with pytest.raises(ValueError, match="256"):
        load_dataset(subjects=(1,), cache_path=tmp_path / "c.npz")


def test_invalid_download_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "_fetch_moabb", lambda subjects, spec=None: fake_moabb(subjects, value=np.nan))
    cache = tmp_path / "c.npz"
    with pytest.raises(ValueError, match="finite"):
        load_dataset(subjects=(1,), cache_path=cache)
    assert not cache.exists()


def test_second_load_comes_from_cache(tmp_path, monkeypatch):
    cache = tmp_path / "sub" / "c.npz"
    monkeypatch.setattr(data_module, "_fetch_moabb", lambda subjects, spec=None: fake_moabb(subjects, value=7.0))
    first = load_dataset(subjects=(1,), cache_path=cache)

    def no_download(subjects, spec=None):
        raise AssertionError("should have used the cache")

    monkeypatch.setattr(data_module, "_fetch_moabb", no_download)
    second = load_dataset(subjects=(1,), cache_path=cache)
    assert np.array_equal(first.X, second.X)
    assert second.X[0, 0, 0] == 7.0
    assert second.session.tolist() == first.session.tolist()


def test_subset_of_cached_subjects_returns_only_those(tmp_path, monkeypatch):
    cache = tmp_path / "c.npz"
    monkeypatch.setattr(data_module, "_fetch_moabb", fake_moabb)
    load_dataset(subjects=(1, 2), cache_path=cache)

    def no_download(subjects, spec=None):
        raise AssertionError("should have used the cache")

    monkeypatch.setattr(data_module, "_fetch_moabb", no_download)
    assert set(load_dataset(subjects=(2,), cache_path=cache).subject.tolist()) == {2}


def test_subject_missing_from_cache_is_downloaded(tmp_path, monkeypatch):
    cache = tmp_path / "c.npz"
    monkeypatch.setattr(data_module, "_fetch_moabb", fake_moabb)
    load_dataset(subjects=(1,), cache_path=cache)
    assert set(load_dataset(subjects=(1, 2), cache_path=cache).subject.tolist()) == {1, 2}


@pytest.mark.parametrize("corruption", ["junk", "truncated"])
def test_corrupt_cache_error_names_file_and_force(tmp_path, monkeypatch, corruption):
    cache = tmp_path / "c.npz"
    if corruption == "junk":
        cache.write_bytes(b"junk")
    else:
        monkeypatch.setattr(data_module, "_fetch_moabb", fake_moabb)
        load_dataset(subjects=(1,), cache_path=cache)
        cache.write_bytes(cache.read_bytes()[:200])
    with pytest.raises(RuntimeError, match=r"c\.npz.*force=True"):
        load_dataset(subjects=(1,), cache_path=cache)


def test_force_rebuilds_corrupt_cache(tmp_path, monkeypatch):
    cache = tmp_path / "c.npz"
    cache.write_bytes(b"junk")
    monkeypatch.setattr(data_module, "_fetch_moabb", fake_moabb)
    d = load_dataset(subjects=(1,), cache_path=cache, force=True)
    assert len(d.X) == 4
    assert len(load_dataset(subjects=(1,), cache_path=cache).X) == 4


def test_failed_download_error_names_file_and_force(tmp_path, monkeypatch):
    def offline(subjects, spec=None):
        raise ConnectionError("no route to host")

    monkeypatch.setattr(data_module, "_fetch_moabb", offline)
    with pytest.raises(RuntimeError, match=r"c\.npz.*no route to host.*force=True"):
        load_dataset(subjects=(1,), cache_path=tmp_path / "c.npz")


def test_moabb_downloads_stay_in_the_project_and_leave_the_home_folder_alone(tmp_path):
    project = Path(__file__).resolve().parents[1]
    script = (
        "import src.data\n"
        "from moabb.datasets.download import get_dataset_path\n"
        "print(get_dataset_path('BNCI2014-001', None))\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("MNE_")}
    env["HOME"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, cwd=project, capture_output=True, text=True, check=True
    )
    assert Path(result.stdout.strip().splitlines()[-1]) == project / "data" / "raw"
    assert not (tmp_path / ".mne" / "mne-python.json").exists()
    assert not (tmp_path / "mne_data").exists()


@pytest.mark.data
def test_real_dataset_shape_and_counts():
    d = load_dataset()
    assert d.X.shape == (9 * 576, 22, 256)
    for s in range(1, 10):
        assert (d.subject == s).sum() == 576
        for session in (0, 1):
            mask = (d.subject == s) & (d.session == session)
            assert np.bincount(d.y[mask], minlength=4).tolist() == [72] * 4


@pytest.mark.data
def test_real_first_trials_match_their_raw_recording():
    import mne
    from moabb.datasets import BNCI2014_001

    d = load_dataset()
    raw = BNCI2014_001().get_data([1])[1]["0train"]["0"]
    events = mne.find_events(raw, shortest_event=0, verbose=False)[:3]
    assert d.y[:3].tolist() == (events[:, 2] - 1).tolist()

    filtered = raw.copy().pick("eeg").filter(4, 40, method="iir", verbose=False)

    def rebuilt(seconds_after_cue):
        start = 2 + seconds_after_cue
        epochs = mne.Epochs(filtered, events, tmin=start, tmax=start + 2, baseline=None, preload=True, verbose=False)
        return epochs.resample(128, verbose=False).get_data(copy=True)[:, :, :256] * 1e6

    assert np.allclose(rebuilt(0.5), d.X[:3], atol=1e-2)
    assert np.corrcoef(rebuilt(0.0).ravel(), d.X[:3].ravel())[0, 1] < 0.5
