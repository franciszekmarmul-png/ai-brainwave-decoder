import numpy as np
import pytest
from conftest import make_synthetic

from src import live
from src.eegnet import EEGNet
from src.live import RunRequest, summarize
from src.train import TrainConfig, train_model


@pytest.fixture
def one_person(monkeypatch):
    data = make_synthetic(n_subjects=1, n_per_class=24)
    monkeypatch.setattr(live, "load_dataset", lambda subjects, dataset: data)
    return data


def test_summarize_picks_the_first_time_the_best_accuracy_is_reached_within_the_limit():
    points = [(1.0, 0.4), (2.0, 0.6), (3.0, 0.6), (9.0, 0.9)]
    assert summarize(points, time_limit=5) == {"best_accuracy": 0.6, "best_time": 2.0}


def test_summarize_falls_back_to_the_first_point_when_nothing_fits_the_limit():
    assert summarize([(7.0, 0.5)], time_limit=5) == {"best_accuracy": 0.5, "best_time": 7.0}


@pytest.mark.parametrize(
    "request_args, message",
    [
        (("nope", 1, "eegnet", 60), "dataset"),
        (("bnci2014_001", 99, "eegnet", 60), "person"),
        (("bnci2014_001", 1, "svm", 60), "model"),
        (("bnci2014_001", 1, "eegnet", 9999), "time limit"),
    ],
)
def test_bad_requests_are_rejected(request_args, message):
    with pytest.raises(ValueError, match=message):
        RunRequest(*request_args).check()


def test_train_model_stops_when_the_callback_asks():
    data = make_synthetic(n_subjects=1, n_per_class=8)
    seen = []
    train_model(EEGNet(), data.X, data.y, data.X, data.y, TrainConfig(max_epochs=50, device="cpu"),
                on_epoch=lambda info: seen.append(info) or len(seen) == 3)
    assert len(seen) == 3
    assert {"epoch", "train_loss", "val_loss", "improved", "elapsed"} <= set(seen[0])


def test_train_model_respects_the_time_limit():
    data = make_synthetic(n_subjects=1, n_per_class=8)
    seen = []
    train_model(EEGNet(), data.X, data.y, data.X, data.y, TrainConfig(max_epochs=500, device="cpu"),
                on_epoch=seen.append, time_limit=0)
    assert len(seen) == 1


@pytest.mark.parametrize("model", ["csp_lda", "fbcsp"])
def test_baseline_runs_report_steps_and_one_result(one_person, model):
    events = []
    summary = live.run(RunRequest("bnci2014_001", 1, model, 60), events.append)
    kinds = [e["type"] for e in events]
    assert kinds[:3] == ["start", "loading", "loaded"]
    assert kinds.count("result") == 1 and kinds[-1] == "done"
    assert "step" in kinds
    assert summary["best_accuracy"] == next(e["test_accuracy"] for e in events if e["type"] == "result")


def test_eegnet_run_streams_epochs_and_can_be_stopped(one_person):
    events = []
    live.run(RunRequest("bnci2014_001", 1, "eegnet", 60), events.append,
             should_stop=lambda: sum(e["type"] == "epoch" for e in events) >= 2)
    epochs = [e for e in events if e["type"] == "epoch"]
    assert [e["epoch"] for e in epochs] == [1, 2]
    assert all(0 <= e["test_accuracy"] <= 1 for e in epochs)
    assert events[-2]["type"] == "kept" and events[-1]["stopped"] is True
    assert np.diff([e["t"] for e in epochs]).min() >= 0
