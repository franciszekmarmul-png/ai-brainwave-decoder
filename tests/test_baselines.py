import numpy as np
import pytest
from conftest import make_synthetic

from src.baselines import CSPLDA, FBCSP
from src.data import load_dataset
from src.splits import apply_standardizer, fit_standardizer


def split(d):
    train, held_out = d.session == 0, d.session == 1
    return d.X[train], d.y[train], d.X[held_out][:40], d.y[held_out][:40], d.X[held_out][40:], d.y[held_out][40:]


@pytest.fixture(scope="module")
def d():
    return make_synthetic(n_subjects=1, n_per_class=36)


def test_fbcsp_learns_synthetic_signal(d):
    X_train, y_train, X_val, y_val, X_test, y_test = split(d)
    model = FBCSP().fit(X_train, y_train, X_val, y_val)
    assert (model.predict(X_test) == y_test).mean() > 0.5


@pytest.mark.data
@pytest.mark.parametrize("cls", [CSPLDA, FBCSP])
def test_learns_a_real_persons_imagined_movements(cls):
    real = load_dataset(subjects=(1,))
    first, second = real.session == 0, real.session == 1
    mean, std = fit_standardizer(real.X[first])
    X_train = apply_standardizer(real.X[first], mean, std)
    X_test = apply_standardizer(real.X[second], mean, std)
    y_train, y_test = real.y[first], real.y[second]
    model = cls().fit(X_train[:-40], y_train[:-40], X_train[-40:], y_train[-40:])
    assert (model.predict(X_test) == y_test).mean() > 0.5


@pytest.mark.parametrize("rhythm_hz", [6.0, 38.0])
def test_fbcsp_finds_signal_at_both_ends_of_its_range(rhythm_hz):
    d = make_synthetic(n_subjects=1, n_per_class=36, rhythm_hz=rhythm_hz)
    X_train, y_train, X_val, y_val, X_test, y_test = split(d)
    model = FBCSP().fit(X_train, y_train, X_val, y_val)
    assert (model.predict(X_test) == y_test).mean() > 0.5


def test_fbcsp_settings_match_paper():
    m = FBCSP()
    assert m.bands == [(4 + 4 * i, 8 + 4 * i) for i in range(9)]
    assert m.n_csp_components == 4 and m.l1_ratio == 0.95


def test_fbcsp_gives_each_one_vs_rest_classifier_36_features(d):
    X_train, y_train, X_val, y_val, _, _ = split(d)
    model = FBCSP().fit(X_train, y_train, X_val, y_val)
    assert [c.coef_.shape for c in model.classifiers_] == [(1, 36)] * 4


@pytest.mark.parametrize("grid", [[1e-6, 10.0], [10.0, 1e-6]])
def test_fbcsp_picks_the_c_with_best_validation_accuracy(d, grid):
    X_train, y_train, X_val, y_val, _, _ = split(d)
    model = FBCSP()
    model.c_grid = grid
    model.fit(X_train, y_train, X_val, y_val)
    assert model.best_c_ == 10.0


def test_fbcsp_uses_validation_data_only_to_choose_c(d):
    X_train, y_train, X_val, y_val, X_test, _ = split(d)
    first, second = FBCSP(), FBCSP()
    first.c_grid = second.c_grid = [1.0]
    first.fit(X_train, y_train, X_val, y_val)
    second.fit(X_train, y_train, X_val[::-1] * 3.0, (y_val + 1) % 4)
    assert np.array_equal(first.predict(X_test), second.predict(X_test))
    assert all(np.array_equal(a.coef_, b.coef_) for a, b in zip(first.classifiers_, second.classifiers_))


def test_fbcsp_same_seed_repeats(d):
    X_train, y_train, X_val, y_val, _, _ = split(d)
    first = FBCSP(seed=1).fit(X_train, y_train, X_val, y_val)
    second = FBCSP(seed=1).fit(X_train, y_train, X_val, y_val)
    assert all(np.array_equal(a.coef_, b.coef_) for a, b in zip(first.classifiers_, second.classifiers_))


@pytest.mark.parametrize("cls", [CSPLDA, FBCSP])
def test_fit_and_predict_leave_inputs_unchanged(cls, d):
    X_train, y_train, X_val, y_val, X_test, _ = split(d)
    X_train, X_test = X_train.copy(), X_test.copy()
    before_train, before_test = X_train.copy(), X_test.copy()
    cls().fit(X_train, y_train, X_val, y_val).predict(X_test)
    assert np.array_equal(X_train, before_train)
    assert np.array_equal(X_test, before_test)


@pytest.mark.parametrize("cls", [CSPLDA, FBCSP])
def test_predictions_use_the_original_label_values(cls, d):
    X_train, y_train, X_val, y_val, X_test, _ = split(d)
    predictions = cls().fit(X_train, y_train, X_val, y_val).predict(X_test)
    assert set(predictions.tolist()) <= {0, 1, 2, 3}
    assert predictions.shape == (len(X_test),)


@pytest.mark.parametrize("cls", [CSPLDA, FBCSP])
def test_fitting_does_not_log_details_for_every_fit(cls, d, caplog):
    X_train, y_train, X_val, y_val, _, _ = split(d)
    cls().fit(X_train, y_train, X_val, y_val)
    assert [r for r in caplog.records if r.name == "mne"] == []
