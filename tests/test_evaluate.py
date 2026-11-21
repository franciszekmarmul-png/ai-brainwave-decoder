import json

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import make_synthetic

from src import evaluate
from src.evaluate import CALIBRATION_SIZES, RESULT_COLUMNS, main, run_experiment, train_eegnet_fold
from src.splits import calibration_base_split, cross_subject_splits, within_subject_splits
from src.train import TrainConfig

QUICK = TrainConfig(max_epochs=1, device="cpu")


@pytest.fixture(scope="module")
def d():
    return make_synthetic(n_subjects=9, n_per_class=8)


def test_within_writes_one_row_per_fold(d, tmp_path):
    df = run_experiment("within", "eegnet", d, tmp_path / "r.csv", subjects=[1], config=QUICK)
    assert len(df) == 4 and list(df.columns) == RESULT_COLUMNS
    assert sorted(df.fold) == [0, 1, 2, 3]
    assert set(df.subject) == {1} and set(df.experiment) == {"within"} and set(df.model) == {"eegnet"}
    assert df.accuracy.between(0, 1).all()
    assert df.best_epoch.notna().all() and set(df.device) == {"cpu"}


def test_cross_writes_ten_rows_per_person(d, tmp_path):
    df = run_experiment("cross", "eegnet", d, tmp_path / "r.csv", subjects=[2, 3], config=QUICK)
    assert len(df) == 20
    assert sorted(df[df.subject == 3].fold) == list(range(10))


@pytest.mark.parametrize("model", ["csp_lda", "fbcsp"])
def test_baselines_run_and_leave_best_epoch_blank(d, tmp_path, model):
    df = run_experiment("within", model, d, tmp_path / "r.csv", subjects=[1], config=QUICK)
    assert len(df) == 4 and df.best_epoch.isna().all()
    assert df.accuracy.between(0, 1).all()


def test_resume_skips_finished_runs(d, tmp_path, monkeypatch):
    out = tmp_path / "r.csv"
    run_experiment("within", "eegnet", d, out, subjects=[1], config=QUICK)

    def fail(*args, **kwargs):
        raise AssertionError("a finished run was trained again")

    monkeypatch.setattr(evaluate, "train_model", fail)
    assert len(run_experiment("within", "eegnet", d, out, subjects=[1], config=QUICK)) == 4


def test_resume_only_runs_what_is_missing(d, tmp_path, monkeypatch):
    out = tmp_path / "r.csv"
    run_experiment("within", "eegnet", d, out, subjects=[1], config=QUICK)
    lines = out.read_text().splitlines(keepends=True)
    out.write_text("".join(lines[:3]))

    calls = []
    real = evaluate.train_model

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(evaluate, "train_model", counted)
    df = run_experiment("within", "eegnet", d, out, subjects=[1], config=QUICK)
    assert len(calls) == 2
    assert sorted(df.fold) == [0, 1, 2, 3]


def test_resume_ignores_partial_last_row(d, tmp_path):
    out = tmp_path / "r.csv"
    run_experiment("within", "eegnet", d, out, subjects=[1], config=QUICK)
    out.write_text(out.read_text()[:-15])

    df = run_experiment("within", "eegnet", d, out, subjects=[1], config=QUICK)

    assert sorted(df.fold) == [0, 1, 2, 3]
    lines = out.read_text().splitlines()
    assert len(lines) == 5
    assert all(line.count(",") == len(RESULT_COLUMNS) - 1 for line in lines)
    assert pd.read_csv(out).device.notna().all()


def test_resume_survives_file_cut_inside_the_header(d, tmp_path):
    out = tmp_path / "r.csv"
    out.write_text("experiment,mod")
    assert len(run_experiment("within", "csp_lda", d, out, subjects=[1], config=QUICK)) == 4


def test_resume_with_different_settings_is_rejected(d, tmp_path):
    out = tmp_path / "r.csv"
    run_experiment("within", "eegnet", d, out, subjects=[1], config=QUICK)
    with pytest.raises(ValueError, match="max_epochs"):
        run_experiment("within", "eegnet", d, out, subjects=[1], config=TrainConfig(max_epochs=2, device="cpu"))


def test_resume_can_change_device_and_add_people(d, tmp_path):
    out = tmp_path / "r.csv"
    run_experiment("within", "csp_lda", d, out, subjects=[1], config=QUICK)
    df = run_experiment("within", "csp_lda", d, out, subjects=[1, 2], config=TrainConfig(max_epochs=1, device="mps"))
    assert sorted(set(df.subject)) == [1, 2] and len(df) == 8


def test_failed_run_keeps_finished_rows_and_names_the_run(d, tmp_path, monkeypatch):
    out = tmp_path / "r.csv"
    real = evaluate.train_model
    calls = []

    def fails_on_third(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("non-finite training loss (nan)")
        return real(*args, **kwargs)

    monkeypatch.setattr(evaluate, "train_model", fails_on_third)
    with pytest.raises(RuntimeError, match=r"subject 1.*fold 2.*non-finite"):
        run_experiment("within", "eegnet", d, out, subjects=[1], config=QUICK)
    assert sorted(pd.read_csv(out).fold) == [0, 1]


def test_model_trains_and_is_scored_on_the_right_trials(d, tmp_path, monkeypatch):
    split = within_subject_splits(d, 2)[1]
    train = d.X[split.train].astype(np.float64)
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = train.std(axis=(0, 2), keepdims=True)
    fit_calls, score_calls = {}, {}
    real_train = evaluate.train_model

    def spy_train(model, X_train, y_train, X_val, y_val, config):
        fit_calls[config.seed] = (X_train, y_train, X_val, y_val)
        return real_train(model, X_train, y_train, X_val, y_val, config)

    def always_class_zero(model, X, device=None):
        score_calls[len(score_calls)] = X
        p = np.zeros((len(X), 4))
        p[:, 0] = 1
        return p

    monkeypatch.setattr(evaluate, "train_model", spy_train)
    monkeypatch.setattr(evaluate, "predict_proba", always_class_zero)
    df = run_experiment("within", "eegnet", d, tmp_path / "r.csv", subjects=[2], config=QUICK)

    row = df[df.fold == 1].iloc[0]
    X_train, y_train, X_val, y_val = fit_calls[row.seed]
    assert np.array_equal(y_train, d.y[split.train]) and np.array_equal(y_val, d.y[split.val])
    assert np.allclose(X_train, (d.X[split.train] - mean) / std, atol=1e-4)
    assert np.allclose(X_val, (d.X[split.val] - mean) / std, atol=1e-4)
    assert np.allclose(score_calls[1], (d.X[split.test] - mean) / std, atol=1e-4)
    assert row.accuracy == pytest.approx((d.y[split.test] == 0).mean())


def test_standardizer_is_fit_on_training_trials_only(d, tmp_path, monkeypatch):
    real = evaluate.fit_standardizer
    sizes = []

    def spy(X):
        sizes.append(len(X))
        return real(X)

    monkeypatch.setattr(evaluate, "fit_standardizer", spy)
    run_experiment("within", "eegnet", d, tmp_path / "w.csv", subjects=[1], config=QUICK)
    run_experiment("cross", "csp_lda", d, tmp_path / "c.csv", subjects=[1], config=QUICK)
    train_size = len(within_subject_splits(d, 1)[0].train)
    assert sizes[:4] == [train_size] * 4
    assert sizes[4:] == [len(s.train) for s in cross_subject_splits(d, 1)]


@pytest.mark.parametrize("experiment, dropout", [("within", 0.5), ("cross", 0.25)])
def test_dropout_follows_experiment(d, tmp_path, monkeypatch, experiment, dropout):
    real = evaluate.EEGNet
    seen = []

    def spy(**kwargs):
        seen.append(kwargs.get("dropout"))
        return real(**kwargs)

    monkeypatch.setattr(evaluate, "EEGNet", spy)
    run_experiment(experiment, "eegnet", d, tmp_path / "r.csv", subjects=[1], config=QUICK)
    assert set(seen) == {dropout}


def test_each_run_records_its_own_seed(d, tmp_path):
    df = run_experiment("within", "eegnet", d, tmp_path / "r.csv", subjects=[1, 2], config=QUICK)
    assert df.seed.nunique() == 8


def test_same_settings_repeat_exactly(d, tmp_path):
    first = run_experiment("within", "eegnet", d, tmp_path / "a.csv", subjects=[1], config=QUICK)
    second = run_experiment("within", "eegnet", d, tmp_path / "b.csv", subjects=[1], config=QUICK)
    assert first.accuracy.tolist() == second.accuracy.tolist()


def test_config_seed_changes_the_runs(d, tmp_path):
    first = run_experiment("within", "eegnet", d, tmp_path / "a.csv", subjects=[1], config=QUICK)
    other = TrainConfig(max_epochs=1, device="cpu", seed=5)
    second = run_experiment("within", "eegnet", d, tmp_path / "b.csv", subjects=[1], config=other)
    assert not set(first.seed) & set(second.seed)


def test_config_file_is_written(d, tmp_path):
    run_experiment("within", "eegnet", d, tmp_path / "r.csv", subjects=[1], config=QUICK)
    saved = json.loads((tmp_path / "r.csv.config.json").read_text())
    assert saved["max_epochs"] == 1 and saved["seed"] == 0
    assert saved["experiment"] == "within" and saved["model"] == "eegnet" and saved["dropout"] == 0.5


def test_workers_give_the_same_results_as_one_process(d, tmp_path):
    one = run_experiment("within", "eegnet", d, tmp_path / "a.csv", subjects=[1], config=QUICK)
    two = run_experiment("within", "eegnet", d, tmp_path / "b.csv", subjects=[1], config=QUICK, workers=2)
    assert one.sort_values("fold").accuracy.tolist() == two.sort_values("fold").accuracy.tolist()


@pytest.mark.parametrize("experiment, model", [("nope", "eegnet"), ("within", "nope")])
def test_unknown_names_are_rejected(d, tmp_path, experiment, model):
    with pytest.raises(ValueError, match="nope"):
        run_experiment(experiment, model, d, tmp_path / "r.csv", subjects=[1], config=QUICK)


def test_command_line_runs_and_reports_progress(d, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(evaluate, "load_dataset", lambda: d)
    out = tmp_path / "sub" / "r.csv"
    main(
        ["--experiment", "within", "--model", "eegnet", "--subjects", "1", "--max-epochs", "1", "--device", "cpu",
         "--out", str(out)]
    )
    assert len(pd.read_csv(out)) == 4
    printed = capsys.readouterr().out
    assert "4/4" in printed and "subject 1" in printed


def test_other_csv_file_is_not_overwritten(d, tmp_path):
    out = tmp_path / "notes.csv"
    out.write_text("name,score\nalex,1\n")
    with pytest.raises(ValueError, match="isn't a results file"):
        run_experiment("within", "csp_lda", d, out, subjects=[1], config=QUICK)
    assert out.read_text() == "name,score\nalex,1\n"


@pytest.fixture(scope="module")
def big():
    return make_synthetic(n_subjects=9, n_per_class=24)


class FineTuneSpy:
    def __init__(self, monkeypatch, run_real=False):
        self.calls = []
        self.real = evaluate.fine_tune
        self.run_real = run_real
        monkeypatch.setattr(evaluate, "fine_tune", self)

    def __call__(self, model, X, y, **kwargs):
        self.calls.append(
            {"weights": model.temporal.weight.detach().clone(), "X": X, "y": y, "kwargs": kwargs, "model": model}
        )
        if self.run_real:
            self.real(model, X, y, **kwargs)


def test_calibration_row_count(big, tmp_path, monkeypatch):
    FineTuneSpy(monkeypatch)
    df = run_experiment("calibration", "eegnet", big, tmp_path / "c.csv", subjects=[1], config=QUICK)
    assert len(df) == 1 + 4 * 5
    assert set(df.n_calibration) == set(CALIBRATION_SIZES)
    assert sorted(df[df.n_calibration == 12].selection) == [0, 1, 2, 3, 4]
    assert df[df.n_calibration == 0].selection.tolist() == [0]
    assert df.seed.nunique() == 21
    assert df.accuracy.between(0, 1).all() and df.best_epoch.notna().all()


def test_fine_tuning_starts_from_a_fresh_copy_of_the_base_model(big, tmp_path, monkeypatch):
    spy = FineTuneSpy(monkeypatch, run_real=True)
    run_experiment("calibration", "eegnet", big, tmp_path / "c.csv", subjects=[1], config=QUICK)
    assert len(spy.calls) == 20
    assert all(torch.equal(call["weights"], spy.calls[0]["weights"]) for call in spy.calls)
    assert not torch.equal(spy.calls[-1]["model"].temporal.weight.detach().cpu(), spy.calls[0]["weights"].cpu())


def test_base_model_is_trained_once_per_subject(big, tmp_path, monkeypatch):
    FineTuneSpy(monkeypatch)
    real = evaluate.train_model
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(evaluate, "train_model", counted)
    run_experiment("calibration", "eegnet", big, tmp_path / "c.csv", subjects=[1, 2], config=QUICK)
    assert len(calls) == 2


def test_resumed_calibration_reuses_saved_base_models(big, tmp_path, monkeypatch):
    FineTuneSpy(monkeypatch)
    out = tmp_path / "c.csv"
    first = run_experiment("calibration", "eegnet", big, out, subjects=[1], config=QUICK)
    lines = out.read_text().splitlines(keepends=True)
    out.write_text("".join(lines[:11]))

    def fail(*args, **kwargs):
        raise AssertionError("the saved base model should have been reused")

    monkeypatch.setattr(evaluate, "train_model", fail)
    second = run_experiment("calibration", "eegnet", big, out, subjects=[1], config=QUICK)
    assert len(second) == 21
    columns = ["n_calibration", "selection"]
    assert second.sort_values(columns).accuracy.tolist() == first.sort_values(columns).accuracy.tolist()


def test_calibration_uses_only_the_new_persons_first_session_and_scores_on_their_second(big, tmp_path, monkeypatch):
    spy = FineTuneSpy(monkeypatch)
    scored = []
    real_predict = evaluate.predict_proba

    def spy_predict(model, X, device=None):
        scored.append(X)
        return real_predict(model, X, device)

    monkeypatch.setattr(evaluate, "predict_proba", spy_predict)
    run_experiment("calibration", "eegnet", big, tmp_path / "c.csv", subjects=[3], config=QUICK)

    base = calibration_base_split(big, 3, seed=0)
    train = big.X[base.train].astype(np.float64)
    mean, std = train.mean(axis=(0, 2), keepdims=True), train.std(axis=(0, 2), keepdims=True)
    first_session = (big.subject == 3) & (big.session == 0)
    candidates = (big.X[first_session] - mean) / std
    for call in spy.calls:
        assert len(call["X"]) in (12, 24, 48, 96)
        for trial in call["X"]:
            assert np.isclose(candidates, trial, atol=1e-3).all(axis=(1, 2)).any()
        assert np.bincount(call["y"], minlength=4).tolist() == [len(call["y"]) // 4] * 4

    test_trials = (big.subject == 3) & (big.session == 1)
    expected = (big.X[test_trials] - mean) / std
    assert all(np.allclose(X, expected, atol=1e-3) for X in scored)
    assert len(scored) == 21


def test_calibration_selections_change_with_size_and_selection(big, tmp_path, monkeypatch):
    spy = FineTuneSpy(monkeypatch)
    run_experiment("calibration", "eegnet", big, tmp_path / "c.csv", subjects=[1], config=QUICK)
    twelves = [call["X"] for call in spy.calls if len(call["X"]) == 12]
    assert len(twelves) == 5
    assert not np.array_equal(twelves[0], twelves[1])


def test_calibration_uses_lower_dropout(big, tmp_path, monkeypatch):
    FineTuneSpy(monkeypatch)
    real = evaluate.EEGNet
    seen = []

    def spy(**kwargs):
        seen.append(kwargs.get("dropout"))
        return real(**kwargs)

    monkeypatch.setattr(evaluate, "EEGNet", spy)
    run_experiment("calibration", "eegnet", big, tmp_path / "c.csv", subjects=[1], config=QUICK)
    assert set(seen) == {0.25}


@pytest.mark.parametrize("model", ["fbcsp", "csp_lda"])
def test_calibration_only_supports_eegnet(big, tmp_path, model):
    with pytest.raises(ValueError, match="eegnet"):
        run_experiment("calibration", model, big, tmp_path / "c.csv", subjects=[1], config=QUICK)


def test_calibration_workers_give_the_same_results_as_one_process(big, tmp_path):
    columns = ["n_calibration", "selection"]
    one = run_experiment("calibration", "eegnet", big, tmp_path / "a.csv", subjects=[1], config=QUICK)
    two = run_experiment("calibration", "eegnet", big, tmp_path / "b.csv", subjects=[1], config=QUICK, workers=2)
    assert one.sort_values(columns).accuracy.tolist() == two.sort_values(columns).accuracy.tolist()


def test_train_eegnet_fold_reproduces_the_saved_run(d, tmp_path):
    df = run_experiment("within", "eegnet", d, tmp_path / "r.csv", subjects=[2], config=QUICK)
    fold = train_eegnet_fold("within", d, 2, 1, QUICK)
    split = within_subject_splits(d, 2)[1]
    assert np.array_equal(fold.test_indices, split.test)
    accuracy = (evaluate.predict_proba(fold.model, fold.X_test).argmax(axis=1) == fold.y_test).mean()
    assert accuracy == df[df.fold == 1].accuracy.iloc[0]
