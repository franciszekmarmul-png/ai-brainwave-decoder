from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.baselines import CSPLDA, FBCSP
from src.data import EEGData, load_dataset
from src.eegnet import EEGNet
from src.splits import (
    Split,
    apply_standardizer,
    calibration_base_split,
    calibration_selection,
    cross_subject_splits,
    fit_standardizer,
    within_subject_splits,
)
from src.train import TrainConfig, fine_tune, pick_device, predict_proba, set_seed, train_model

RESULT_COLUMNS = [
    "experiment",
    "model",
    "subject",
    "fold",
    "n_calibration",
    "selection",
    "seed",
    "accuracy",
    "best_epoch",
    "device",
]

CALIBRATION_SIZES = (0, 12, 24, 48, 96)

_N_FOLDS = {"within": 4, "cross": 10}
_DROPOUT = {"within": 0.5, "cross": 0.25, "calibration": 0.25}
_N_SELECTIONS = 5
_BASE_MODEL_OFFSET = 999
_BASELINES = {"csp_lda": lambda seed: CSPLDA(), "fbcsp": lambda seed: FBCSP(seed=seed)}
_MODELS = ("eegnet", *_BASELINES)
_SETTINGS_THAT_MUST_MATCH = ("experiment", "model", "max_epochs", "batch_size", "lr", "seed", "dropout")


@dataclass(frozen=True)
class _Run:
    experiment: str
    model: str
    subject: int
    fold: int
    n_calibration: int | None = None
    selection: int | None = None

    def key(self) -> tuple:
        return (self.subject, self.fold, self.n_calibration, self.selection)

    def describe(self) -> str:
        if self.experiment == "calibration":
            return f"calibration subject {self.subject} with {self.n_calibration} trials, selection {self.selection}"
        return f"{self.experiment} {self.model} subject {self.subject} fold {self.fold}"


@dataclass
class TrainedFold:
    model: EEGNet
    best_epoch: int
    mean: np.ndarray
    std: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_indices: np.ndarray


def train_eegnet_fold(
    experiment: str,
    data: EEGData,
    subject: int,
    fold: int,
    config: TrainConfig,
    saved_model: Path | None = None,
) -> TrainedFold:
    split = _split_for(_Run(experiment, "eegnet", subject, fold), data, config)
    mean, std = fit_standardizer(data.X[split.train])
    seed = _seed(config, subject, fold)
    set_seed(seed)
    model = EEGNet(dropout=_DROPOUT[experiment])
    if saved_model is not None and Path(saved_model).exists():
        saved = torch.load(saved_model, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["state"])
        best_epoch = saved["best_epoch"]
    else:
        result = train_model(
            model,
            apply_standardizer(data.X[split.train], mean, std),
            data.y[split.train],
            apply_standardizer(data.X[split.val], mean, std),
            data.y[split.val],
            replace(config, seed=seed),
        )
        best_epoch = result.best_epoch
        if saved_model is not None:
            _save_model(model, best_epoch, Path(saved_model))
    return TrainedFold(
        model=model,
        best_epoch=best_epoch,
        mean=mean,
        std=std,
        X_test=apply_standardizer(data.X[split.test], mean, std),
        y_test=data.y[split.test],
        test_indices=split.test,
    )


def run_experiment(
    experiment: str,
    model_name: str,
    data: EEGData,
    out_csv: Path,
    subjects: Iterable[int] = range(1, 10),
    config: TrainConfig | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    if experiment not in _DROPOUT:
        raise ValueError(f"unknown experiment {experiment!r}, expected one of {sorted(_DROPOUT)}")
    if model_name not in _MODELS:
        raise ValueError(f"unknown model {model_name!r}, expected one of {sorted(_MODELS)}")
    if experiment == "calibration" and model_name != "eegnet":
        raise ValueError(f"the calibration experiment only supports eegnet, not {model_name}")
    config = config or TrainConfig()
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    subjects = [int(s) for s in subjects]

    _save_settings(out_csv, experiment, model_name, subjects, config, workers)
    finished = _finished_keys(out_csv)
    planned = _plan_runs(experiment, model_name, subjects)
    pending = [run for run in planned if run.key() not in finished]

    done = len(planned) - len(pending)

    def save(run: _Run, row: dict) -> None:
        nonlocal done
        with open(out_csv, "a", newline="") as f:
            csv.writer(f, lineterminator="\n").writerow([row[c] for c in RESULT_COLUMNS])
        done += 1
        print(f"[{done}/{len(planned)}] {run.describe()}: accuracy {row['accuracy']:.3f}", flush=True)

    torch.set_num_threads(1)
    if experiment == "calibration":
        _train_missing_base_models(pending, data, config, out_csv, workers)
    if workers > 1:
        jobs = [(run, config, out_csv) for run in pending]
        _map_in_workers(_run_in_worker, jobs, data, workers, lambda result: save(*result))
    else:
        for run in pending:
            save(run, _run_labelled(run, data, config, out_csv))

    columns = ["subject", "fold", "n_calibration", "selection"]
    return pd.read_csv(out_csv).sort_values(columns).reset_index(drop=True)


def _plan_runs(experiment: str, model_name: str, subjects: list[int]) -> list[_Run]:
    if experiment == "calibration":
        return [
            _Run(experiment, model_name, subject, 0, size, selection)
            for subject in subjects
            for size in CALIBRATION_SIZES
            for selection in range(_N_SELECTIONS if size else 1)
        ]
    return [
        _Run(experiment, model_name, subject, fold) for subject in subjects for fold in range(_N_FOLDS[experiment])
    ]


def _run_labelled(run: _Run, data: EEGData, config: TrainConfig, out_csv: Path) -> dict:
    try:
        return _run_once(run, data, config, out_csv)
    except Exception as err:
        raise RuntimeError(f"{run.describe()} failed: {err}") from err


def _run_once(run: _Run, data: EEGData, config: TrainConfig, out_csv: Path) -> dict:
    if run.experiment == "calibration":
        return _run_calibration(run, data, config, out_csv)
    seed = _seed(config, run.subject, run.fold)
    if run.model == "eegnet":
        fold = train_eegnet_fold(run.experiment, data, run.subject, run.fold, config)
        predictions = predict_proba(fold.model, fold.X_test).argmax(axis=1)
        device = next(fold.model.parameters()).device.type
        return _row(run, seed, float((predictions == fold.y_test).mean()), fold.best_epoch, device)

    split = _split_for(run, data, config)
    mean, std = fit_standardizer(data.X[split.train])
    model = _BASELINES[run.model](seed).fit(
        apply_standardizer(data.X[split.train], mean, std),
        data.y[split.train],
        apply_standardizer(data.X[split.val], mean, std),
        data.y[split.val],
    )
    predictions = model.predict(apply_standardizer(data.X[split.test], mean, std))
    return _row(run, seed, float((predictions == data.y[split.test]).mean()), None, "cpu")


def _run_calibration(run: _Run, data: EEGData, config: TrainConfig, out_csv: Path) -> dict:
    fold = _base_model(run.subject, data, config, out_csv)
    seed = _seed(config, run.subject, run.n_calibration * 10 + run.selection)
    model = fold.model.to(pick_device(config.device))
    if run.n_calibration:
        chosen = calibration_selection(data, run.subject, run.n_calibration, seed=config.seed * 1000 + run.selection)
        X = apply_standardizer(data.X[chosen], fold.mean, fold.std)
        fine_tune(model, X, data.y[chosen], seed=seed, device=config.device)
    predictions = predict_proba(model, fold.X_test).argmax(axis=1)
    device = next(model.parameters()).device.type
    return _row(run, seed, float((predictions == fold.y_test).mean()), fold.best_epoch, device)


def _base_model(subject: int, data: EEGData, config: TrainConfig, out_csv: Path) -> TrainedFold:
    path = out_csv.parent / "models" / f"{out_csv.stem}_subject_{subject}.pt"
    return train_eegnet_fold("calibration", data, subject, _BASE_MODEL_OFFSET, config, saved_model=path)


def _train_missing_base_models(
    pending: list[_Run], data: EEGData, config: TrainConfig, out_csv: Path, workers: int
) -> None:
    subjects = sorted({run.subject for run in pending})
    missing = [s for s in subjects if not (out_csv.parent / "models" / f"{out_csv.stem}_subject_{s}.pt").exists()]

    def report(subject: int) -> None:
        print(f"Trained the base model for subject {subject}", flush=True)

    if workers > 1:
        jobs = [(subject, config, out_csv) for subject in missing]
        _map_in_workers(_train_base_in_worker, jobs, data, workers, report)
    else:
        for subject in missing:
            _base_model(subject, data, config, out_csv)
            report(subject)


def _save_model(model: EEGNet, best_epoch: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    state = {name: value.cpu() for name, value in model.state_dict().items()}
    torch.save({"state": state, "best_epoch": best_epoch}, temporary)
    temporary.replace(path)


def _seed(config: TrainConfig, subject: int, offset: int) -> int:
    return config.seed * 1_000_000 + subject * 1000 + offset


def _row(run: _Run, seed: int, accuracy: float, best_epoch: int | None, device: str) -> dict:
    return {
        "experiment": run.experiment,
        "model": run.model,
        "subject": run.subject,
        "fold": run.fold,
        "n_calibration": run.n_calibration,
        "selection": run.selection,
        "seed": seed,
        "accuracy": accuracy,
        "best_epoch": best_epoch,
        "device": device,
    }


def _split_for(run: _Run, data: EEGData, config: TrainConfig) -> Split:
    if run.experiment == "within":
        return within_subject_splits(data, run.subject)[run.fold]
    if run.experiment == "calibration":
        return calibration_base_split(data, run.subject, seed=config.seed)
    return cross_subject_splits(data, run.subject, n_repeats=_N_FOLDS["cross"], seed=config.seed)[run.fold]


_worker_data: EEGData | None = None


def _start_worker(data: EEGData) -> None:
    global _worker_data
    _worker_data = data
    torch.set_num_threads(1)


def _run_in_worker(run: _Run, config: TrainConfig, out_csv: Path) -> tuple[_Run, dict]:
    return run, _run_labelled(run, _worker_data, config, out_csv)


def _train_base_in_worker(subject: int, config: TrainConfig, out_csv: Path) -> int:
    _base_model(subject, _worker_data, config, out_csv)
    return subject


def _map_in_workers(
    function: Callable, jobs: list[tuple], data: EEGData, workers: int, on_result: Callable[..., None]
) -> None:
    if not jobs:
        return
    os.environ.update(OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=context, initializer=_start_worker, initargs=(data,)) as pool:
        futures = [pool.submit(function, *job) for job in jobs]
        try:
            for future in as_completed(futures):
                on_result(future.result())
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise


def _finished_keys(out_csv: Path) -> set[tuple]:
    header = ",".join(RESULT_COLUMNS) + "\n"
    text = out_csv.read_text() if out_csv.exists() else ""
    complete = text[: text.rfind("\n") + 1]
    if complete and not complete.startswith(header):
        raise ValueError(f"{out_csv} exists but isn't a results file, so it won't be overwritten")
    complete = complete or header
    if complete != text:
        out_csv.write_text(complete)
    return {
        (int(row.subject), int(row.fold), _whole_or_none(row.n_calibration), _whole_or_none(row.selection))
        for row in pd.read_csv(out_csv).itertuples()
    }


def _whole_or_none(value: float) -> int | None:
    return None if np.isnan(value) else int(value)


def _save_settings(
    out_csv: Path, experiment: str, model_name: str, subjects: list[int], config: TrainConfig, workers: int
) -> None:
    settings = {
        "experiment": experiment,
        "model": model_name,
        "dropout": _DROPOUT.get(experiment),
        "subjects": subjects,
        "workers": workers,
        **asdict(config),
    }
    path = out_csv.with_name(out_csv.name + ".config.json")
    if path.exists():
        saved = json.loads(path.read_text())
        changed = [k for k in _SETTINGS_THAT_MUST_MATCH if saved.get(k) != settings[k]]
        if changed:
            raise ValueError(
                f"{out_csv} was started with different settings ({', '.join(changed)}). "
                "Use a new --out file so results from different settings don't get mixed."
            )
    path.write_text(json.dumps(settings, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--max-epochs", type=int, default=TrainConfig.max_epochs)
    parser.add_argument("--device", default=None, help="cpu or mps. By default, mps if it's available.")
    parser.add_argument("--workers", type=int, default=1, help="Runs to do at once, each on one CPU core.")
    args = parser.parse_args(argv)
    config = TrainConfig(max_epochs=args.max_epochs, device=args.device)
    run_experiment(args.experiment, args.model, load_dataset(), args.out, args.subjects, config, args.workers)


if __name__ == "__main__":
    main()
