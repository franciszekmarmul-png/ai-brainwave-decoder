import numpy as np
import pytest
import torch
from conftest import make_synthetic

from src.eegnet import EEGNet
from src.train import TrainConfig, fine_tune, pick_device, predict_proba, set_seed, train_model


@pytest.fixture(scope="module")
def d():
    return make_synthetic(n_subjects=1, n_per_class=8)


def accuracy(model, X, y):
    return (predict_proba(model, X, "cpu").argmax(1) == y).mean()


def count_optimizer_steps(monkeypatch):
    calls = []
    real = EEGNet.apply_max_norm

    def counted(self):
        calls.append(1)
        real(self)

    monkeypatch.setattr(EEGNet, "apply_max_norm", counted)
    return calls


def test_overfits_small_batch(d):
    m = EEGNet(dropout=0.0)
    train_model(m, d.X, d.y, d.X, d.y, TrainConfig(max_epochs=200, batch_size=8, device="cpu"))
    assert accuracy(m, d.X, d.y) > 0.95


def test_restores_lowest_validation_loss_weights(d):
    wrong = (d.y + 1) % 4
    m = EEGNet(dropout=0.0)
    result = train_model(m, d.X, d.y, d.X, wrong, TrainConfig(max_epochs=40, batch_size=8, device="cpu"))
    assert result.best_epoch < 39
    assert result.history[-1]["val_loss"] > result.best_val_loss
    p = predict_proba(m, d.X, "cpu")
    loss = -np.log(p[np.arange(len(wrong)), wrong]).mean()
    assert loss == pytest.approx(result.best_val_loss, abs=1e-4)


def test_history_records_every_epoch_and_finds_the_minimum(d):
    result = train_model(EEGNet(), d.X, d.y, d.X, d.y, TrainConfig(max_epochs=5, device="cpu"))
    assert [h["epoch"] for h in result.history] == [0, 1, 2, 3, 4]
    assert result.best_val_loss == min(h["val_loss"] for h in result.history)
    assert result.history[result.best_epoch]["val_loss"] == result.best_val_loss


def test_same_seed_gives_same_result(d):
    def run(seed):
        set_seed(seed)
        model = EEGNet()
        return train_model(model, d.X, d.y, d.X, d.y, TrainConfig(max_epochs=3, seed=seed, device="cpu")).best_val_loss

    assert run(1) == run(1)
    assert run(1) != run(2)


@pytest.mark.parametrize("n_trials, steps", [(33, 1), (34, 2), (64, 2)])
def test_final_batch_of_one_trial_is_dropped(d, monkeypatch, n_trials, steps):
    calls = count_optimizer_steps(monkeypatch)
    train_model(EEGNet(), d.X[:n_trials], d.y[:n_trials], d.X, d.y, TrainConfig(max_epochs=1, device="cpu"))
    assert len(calls) == steps


def test_fine_tune_also_drops_final_batch_of_one_trial(d, monkeypatch):
    calls = count_optimizer_steps(monkeypatch)
    fine_tune(EEGNet(), d.X[:33], d.y[:33], epochs=1, device="cpu")
    assert len(calls) == 1


def test_non_finite_training_loss_raises(d):
    X = d.X.copy()
    X[0] = np.inf
    with pytest.raises(RuntimeError, match="non-finite"):
        train_model(EEGNet(), X, d.y, d.X, d.y, TrainConfig(max_epochs=1, device="cpu"))


def test_non_finite_validation_loss_raises(d):
    X = d.X.copy()
    X[0] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        train_model(EEGNet(), d.X, d.y, X, d.y, TrainConfig(max_epochs=1, device="cpu"))


def test_fine_tune_non_finite_loss_raises(d):
    X = d.X.copy()
    X[0] = np.inf
    with pytest.raises(RuntimeError, match="non-finite"):
        fine_tune(EEGNet(), X, d.y, epochs=1, device="cpu")


def test_training_keeps_max_norm_limits(d):
    m = EEGNet(dropout=0.0)
    train_model(m, d.X, d.y, d.X, d.y, TrainConfig(max_epochs=3, lr=0.1, device="cpu"))
    assert m.spatial.weight.flatten(1).norm(dim=1).max() <= 1 + 1e-5
    assert m.classifier.weight.norm(dim=1).max() <= 0.25 + 1e-5


def test_fine_tune_keeps_max_norm_limits(d):
    m = EEGNet(dropout=0.0)
    fine_tune(m, d.X, d.y, epochs=3, lr=0.1, device="cpu")
    assert m.spatial.weight.flatten(1).norm(dim=1).max() <= 1 + 1e-5
    assert m.classifier.weight.norm(dim=1).max() <= 0.25 + 1e-5


def test_fine_tune_changes_weights_and_leaves_model_in_eval_mode(d):
    m = EEGNet()
    before = m.temporal.weight.detach().clone()
    fine_tune(m, d.X, d.y, epochs=1, device="cpu")
    assert not torch.equal(m.temporal.weight, before)
    assert m.training is False


def test_fine_tune_learns_a_new_labelling(d):
    swapped = (d.y + 1) % 4
    m = EEGNet(dropout=0.0)
    train_model(m, d.X, d.y, d.X, d.y, TrainConfig(max_epochs=100, batch_size=8, device="cpu"))
    assert accuracy(m, d.X, swapped) < 0.3
    fine_tune(m, d.X, swapped, epochs=100, lr=1e-3, batch_size=8, device="cpu")
    assert accuracy(m, d.X, swapped) > 0.8


def test_fine_tune_same_seed_gives_same_weights(d):
    def run(seed):
        set_seed(0)
        m = EEGNet()
        fine_tune(m, d.X, d.y, epochs=2, seed=seed, device="cpu")
        return m.temporal.weight.detach().clone()

    assert torch.equal(run(1), run(1))
    assert not torch.equal(run(1), run(2))


def test_predict_proba_rows_are_probabilities_in_eval_mode(d):
    m = EEGNet()
    first = predict_proba(m, d.X, "cpu")
    assert first.shape == (len(d.X), 4)
    assert np.allclose(first.sum(axis=1), 1, atol=1e-5)
    assert np.array_equal(first, predict_proba(m, d.X, "cpu"))


def test_predict_proba_uses_the_models_device_by_default(d):
    m = EEGNet().to("cpu")
    assert np.array_equal(predict_proba(m, d.X[:4]), predict_proba(m, d.X[:4], "cpu"))


def test_pick_device_falls_back_to_cpu(monkeypatch, capsys):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert pick_device().type == "cpu"
    assert "CPU" in capsys.readouterr().out


def test_pick_device_falls_back_when_mps_is_requested_but_missing(monkeypatch, capsys):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert pick_device("mps").type == "cpu"
    assert "CPU" in capsys.readouterr().out


def test_pick_device_prefers_mps(monkeypatch, capsys):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert pick_device().type == "mps"
    assert capsys.readouterr().out == ""


def test_pick_device_honours_cpu_request_silently(monkeypatch, capsys):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert pick_device("cpu").type == "cpu"
    assert capsys.readouterr().out == ""


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs the Mac GPU")
def test_trains_and_predicts_on_the_mac_gpu(d):
    m = EEGNet()
    train_model(m, d.X, d.y, d.X, d.y, TrainConfig(max_epochs=1, device="mps"))
    fine_tune(m, d.X, d.y, epochs=1, device="mps")
    assert next(m.parameters()).device.type == "mps"
    assert predict_proba(m, d.X).shape == (len(d.X), 4)
