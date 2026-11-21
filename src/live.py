from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from src.baselines import CSPLDA, FBCSP
from src.data import DATASETS, load_dataset
from src.eegnet import EEGNet
from src.splits import apply_standardizer, fit_standardizer, within_subject_splits
from src.train import TrainConfig, predict_proba, set_seed, train_model

MODELS = {
    "eegnet": "EEGNet",
    "fbcsp": "FBCSP",
    "csp_lda": "CSP + LDA",
}
MAX_TIME_LIMIT = 600


@dataclass
class RunRequest:
    dataset: str
    person: int
    model: str
    time_limit: float = 120

    def check(self) -> None:
        if self.dataset not in DATASETS:
            raise ValueError(f"unknown dataset {self.dataset!r}")
        if self.person not in DATASETS[self.dataset].subjects:
            raise ValueError(f"{DATASETS[self.dataset].title} has no person {self.person}")
        if self.model not in MODELS:
            raise ValueError(f"unknown model {self.model!r}")
        if not 5 <= self.time_limit <= MAX_TIME_LIMIT:
            raise ValueError(f"time limit must be between 5 and {MAX_TIME_LIMIT} seconds")

    @property
    def command(self) -> str:
        return (
            f"python -m src.live --dataset {self.dataset} --person {self.person} "
            f"--model {self.model} --time-limit {int(self.time_limit)}"
        )


Emit = Callable[[dict], None]


def run(request: RunRequest, emit: Emit, should_stop: Callable[[], bool] = lambda: False) -> dict:
    request.check()
    spec = DATASETS[request.dataset]
    emit({"type": "start", "command": request.command, "dataset": spec.key, "title": spec.title,
          "person": request.person, "model": request.model, "time_limit": request.time_limit,
          "chance": spec.chance, "classes": list(spec.classes)})

    emit({"type": "loading", "dataset": spec.title})
    data = load_dataset(subjects=(request.person,), dataset=spec.key)
    split = within_subject_splits(data, request.person)[0]
    mean, std = fit_standardizer(data.X[split.train])
    X = {part: apply_standardizer(data.X[getattr(split, part)], mean, std) for part in ("train", "val", "test")}
    y = {part: data.y[getattr(split, part)] for part in ("train", "val", "test")}
    emit({"type": "loaded", "trials": int(len(data.X)), "channels": int(data.X.shape[1]),
          "samples": int(data.X.shape[2]), "train": len(y["train"]), "val": len(y["val"]), "test": len(y["test"])})

    started = time.monotonic()
    clock = lambda: round(time.monotonic() - started, 3)
    points = []

    if request.model == "eegnet":
        torch.set_num_threads(1)
        seed = request.person * 1000
        set_seed(seed)
        model = EEGNet(n_channels=X["train"].shape[1], n_classes=len(spec.classes), dropout=0.5)
        config = TrainConfig(seed=seed, device="cpu")

        def on_epoch(info: dict) -> bool:
            accuracy = float((predict_proba(model, X["test"]).argmax(1) == y["test"]).mean())
            t = clock()
            points.append((t, accuracy))
            emit({"type": "epoch", "t": t, "epoch": info["epoch"] + 1, "train_loss": round(info["train_loss"], 4),
                  "val_loss": round(info["val_loss"], 4), "test_accuracy": accuracy, "improved": info["improved"]})
            return should_stop()

        result = train_model(model, X["train"], y["train"], X["val"], y["val"], config,
                             on_epoch=on_epoch, time_limit=request.time_limit)
        kept = float((predict_proba(model, X["test"]).argmax(1) == y["test"]).mean())
        emit({"type": "kept", "t": clock(), "epoch": result.best_epoch + 1, "test_accuracy": kept})
    else:
        model = FBCSP(seed=0) if request.model == "fbcsp" else CSPLDA()

        def on_step(step: dict) -> None:
            emit({"type": "step", "t": clock(), **step})

        model.fit(X["train"], y["train"], X["val"], y["val"], on_step=on_step)
        accuracy = float((model.predict(X["test"]) == y["test"]).mean())
        points.append((clock(), accuracy))
        emit({"type": "result", "t": points[-1][0], "test_accuracy": accuracy})

    summary = summarize(points, request.time_limit)
    summary.update({"type": "done", "t": clock(), "stopped": should_stop()})
    emit(summary)
    return summary


def summarize(points: list[tuple[float, float]], time_limit: float) -> dict:
    within = [(t, a) for t, a in points if t <= time_limit] or points[:1]
    if not within:
        return {"best_accuracy": None, "best_time": None}
    best = max(a for _, a in within)
    first = next(t for t, a in within if a == best)
    return {"best_accuracy": best, "best_time": first}


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Train one model on one person and print each step as JSON.")
    parser.add_argument("--dataset", default="bnci2014_001", choices=sorted(DATASETS))
    parser.add_argument("--person", type=int, default=1)
    parser.add_argument("--model", default="eegnet", choices=sorted(MODELS))
    parser.add_argument("--time-limit", type=float, default=120)
    args = parser.parse_args(argv)
    request = RunRequest(args.dataset, args.person, args.model, args.time_limit)
    run(request, lambda event: print(json.dumps(event), flush=True))


if __name__ == "__main__":
    main()
