import numpy as np
import pytest
from conftest import make_synthetic

from src.splits import (
    apply_standardizer,
    calibration_base_split,
    calibration_selection,
    cross_subject_splits,
    fit_standardizer,
    within_subject_splits,
)


@pytest.fixture(scope="module")
def d():
    return make_synthetic(n_subjects=9)


def test_within_subject_blocks_never_overlap_and_tests_cover_everyone(d):
    splits = within_subject_splits(d, 1)
    assert len(splits) == 4
    for s in splits:
        assert not set(s.train) & set(s.val) | set(s.train) & set(s.test) | set(s.val) & set(s.test)
        assert len(s.train) == 288 and len(s.val) == 144 and len(s.test) == 144
    assert sorted(np.concatenate([s.test for s in splits])) == sorted(np.where(d.subject == 1)[0])


def test_within_subject_blocks_are_consecutive_and_rotate(d):
    s = within_subject_splits(d, 2)[1]
    assert s.test.tolist() == list(range(720, 864))
    assert s.val.tolist() == list(range(864, 1008))
    assert sorted(s.train.tolist()) == list(range(576, 720)) + list(range(1008, 1152))
    wrapped = within_subject_splits(d, 2)[3]
    assert wrapped.test.tolist() == list(range(1008, 1152))
    assert wrapped.val.tolist() == list(range(576, 720))


def test_cross_subject_keeps_test_person_out_and_uses_session_2(d):
    splits = cross_subject_splits(d, 3)
    assert len(splits) == 10
    for s in splits:
        assert 3 not in set(d.subject[np.r_[s.train, s.val]])
        assert len(set(d.subject[s.train])) == 5 and len(set(d.subject[s.val])) == 3
        assert set(d.subject[s.test]) == {3} and set(d.session[s.test]) == {1}
        assert set(d.session[np.r_[s.train, s.val]]) == {0}
        assert not set(d.subject[s.train]) & set(d.subject[s.val])
        assert len(s.test) == 288


def test_cross_subject_repeats_pick_different_people(d):
    picks = {tuple(sorted(set(d.subject[s.train].tolist()))) for s in cross_subject_splits(d, 3)}
    assert len(picks) > 1


def test_cross_subject_same_seed_repeats_and_different_seed_changes(d):
    first = [s.train.tolist() for s in cross_subject_splits(d, 3, seed=1)]
    assert first == [s.train.tolist() for s in cross_subject_splits(d, 3, seed=1)]
    assert first != [s.train.tolist() for s in cross_subject_splits(d, 3, seed=2)]


def test_calibration_base_split_uses_8_people_and_holds_out_10_percent(d):
    s = calibration_base_split(d, 3)
    assert len(set(d.subject[s.train])) == 8 and 3 not in set(d.subject[s.train])
    assert set(d.subject[s.test]) == {3} and set(d.session[s.test]) == {1}
    assert abs(len(s.val) / (len(s.train) + len(s.val)) - 0.1) < 0.02


def test_calibration_base_split_stratifies_validation_by_person_and_class(d):
    s = calibration_base_split(d, 3)
    assert not set(s.train) & set(s.val)
    assert set(d.session[np.r_[s.train, s.val]]) == {0}
    assert len(s.train) + len(s.val) == 8 * 288
    for person in (1, 2, 4, 5, 6, 7, 8, 9):
        people = d.y[s.val][d.subject[s.val] == person]
        assert np.bincount(people, minlength=4).tolist() == [7, 7, 7, 7]


@pytest.mark.parametrize("n", [12, 24, 48, 96])
def test_calibration_selection_is_balanced_and_from_session_1(d, n):
    idx = calibration_selection(d, 2, n, seed=0)
    assert np.bincount(d.y[idx], minlength=4).tolist() == [n // 4] * 4
    assert set(d.subject[idx]) == {2} and set(d.session[idx]) == {0}
    assert len(set(idx.tolist())) == n


def test_calibration_selection_zero_is_empty_and_uneven_is_rejected(d):
    assert len(calibration_selection(d, 2, 0, seed=0)) == 0
    with pytest.raises(ValueError):
        calibration_selection(d, 2, 10, seed=0)


@pytest.mark.parametrize("n", [-4, 292])
def test_calibration_selection_rejects_impossible_sizes(d, n):
    with pytest.raises(ValueError):
        calibration_selection(d, 2, n, seed=0)


def test_same_seed_repeats_and_different_seed_changes(d):
    assert np.array_equal(calibration_selection(d, 2, 24, 1), calibration_selection(d, 2, 24, 1))
    assert not np.array_equal(calibration_selection(d, 2, 24, 1), calibration_selection(d, 2, 24, 2))


@pytest.mark.parametrize(
    "make_split",
    [
        lambda d: within_subject_splits(d, 99),
        lambda d: cross_subject_splits(d, 99),
        lambda d: calibration_base_split(d, 99),
        lambda d: calibration_selection(d, 99, 12, 0),
    ],
)
def test_unknown_subject_is_rejected(d, make_split):
    with pytest.raises(ValueError, match="99"):
        make_split(d)


def test_cross_subject_needs_8_other_people():
    small = make_synthetic(n_subjects=4, n_per_class=2)
    with pytest.raises(ValueError, match="8"):
        cross_subject_splits(small, 1)


def test_standardizer_is_channel_wise_and_handles_flat_channel():
    X = np.random.default_rng(0).normal(5, 2, (50, 22, 256)).astype("float32")
    X[:, 0] = 1.0
    m, s = fit_standardizer(X)
    Z = apply_standardizer(X, m, s)
    assert np.isfinite(Z).all() and abs(Z[:, 1].mean()) < 1e-3
    assert abs(Z[:, 1].std() - 1) < 1e-3
    assert Z.dtype == np.float32


def test_standardizer_treats_rounding_noise_on_a_flat_channel_as_flat():
    X = np.full((50, 22, 256), 0.1, dtype=np.float32)
    Z = apply_standardizer(X, *fit_standardizer(X))
    assert np.abs(Z).max() < 1e-3


def test_standardizer_applies_training_statistics_to_new_trials():
    train = np.zeros((1, 22, 256), dtype=np.float32)
    train[0, 0] = np.tile([2.0, 4.0], 128)
    train[0, 1] = np.tile([0.0, 20.0], 128)
    new = np.zeros((1, 22, 256), dtype=np.float32)
    new[0, 0] = 5.0
    new[0, 1] = 30.0
    Z = apply_standardizer(new, *fit_standardizer(train))
    assert Z[0, 0, 0] == pytest.approx(2.0)
    assert Z[0, 1, 0] == pytest.approx(2.0)
