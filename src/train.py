from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.nn import functional as F

from src.eegnet import EEGNet

_EVAL_BATCH_SIZE = 512


@dataclass
class TrainConfig:
    max_epochs: int = 500
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 0
    device: str | None = None


@dataclass
class TrainResult:
    best_epoch: int
    best_val_loss: float
    history: list[dict] = field(default_factory=list)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def pick_device(preferred: str | None = None) -> torch.device:
    if preferred not in (None, "mps"):
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    print("The Mac GPU (MPS) isn't available, so this will run on the CPU.")
    return torch.device("cpu")


def train_model(
    model: EEGNet,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig,
    on_epoch: Callable[[dict], bool | None] | None = None,
    time_limit: float | None = None,
) -> TrainResult:
    started = time.monotonic()
    device = pick_device(config.device)
    model.to(device)
    set_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    X_train_t, y_train_t = _to_tensors(X_train, y_train, device)
    X_val_t, y_val_t = _to_tensors(X_val, y_val, device)

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    history = []
    for epoch in range(config.max_epochs):
        train_loss = _train_epoch(model, X_train_t, y_train_t, optimizer, config.batch_size, generator)
        val_loss = _mean_loss(model, X_val_t, y_val_t)
        _require_finite(val_loss, "validation")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        elapsed = time.monotonic() - started
        stop = on_epoch({**history[-1], "improved": improved, "elapsed": elapsed}) if on_epoch else False
        if stop or (time_limit is not None and elapsed >= time_limit):
            break
    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(best_epoch=best_epoch, best_val_loss=best_val_loss, history=history)


def fine_tune(
    model: EEGNet,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-4,
    batch_size: int = 32,
    seed: int = 0,
    device: str | None = None,
) -> None:
    target = _model_device(model) if device is None else pick_device(device)
    model.to(target)
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X_t, y_t = _to_tensors(X, y, target)
    for _ in range(epochs):
        _train_epoch(model, X_t, y_t, optimizer, batch_size, generator)
    model.eval()


@torch.no_grad()
def predict_proba(model: EEGNet, X: np.ndarray, device: str | None = None) -> np.ndarray:
    target = _model_device(model) if device is None else pick_device(device)
    model.to(target)
    model.eval()
    trials = torch.as_tensor(X, dtype=torch.float32, device=target)
    batches = [
        torch.softmax(model(trials[start : start + _EVAL_BATCH_SIZE]), dim=1).cpu()
        for start in range(0, len(trials), _EVAL_BATCH_SIZE)
    ]
    return torch.cat(batches).numpy()


def _train_epoch(
    model: EEGNet,
    X: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    generator: torch.Generator,
) -> float:
    model.train()
    order = torch.randperm(len(X), generator=generator).to(X.device)
    total, count = 0.0, 0
    for start in range(0, len(X), batch_size):
        batch = order[start : start + batch_size]
        if len(batch) == 1:
            break
        optimizer.zero_grad()
        loss = F.cross_entropy(model(X[batch]), y[batch])
        _require_finite(loss.item(), "training")
        loss.backward()
        optimizer.step()
        model.apply_max_norm()
        total += loss.item() * len(batch)
        count += len(batch)
    return total / count


@torch.no_grad()
def _mean_loss(model: EEGNet, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    total = 0.0
    for start in range(0, len(X), _EVAL_BATCH_SIZE):
        end = start + _EVAL_BATCH_SIZE
        total += F.cross_entropy(model(X[start:end]), y[start:end], reduction="sum").item()
    return total / len(X)


def _require_finite(loss: float, stage: str) -> None:
    if not np.isfinite(loss):
        raise RuntimeError(f"non-finite {stage} loss ({loss}), so this run's results would be meaningless")


def _to_tensors(X: np.ndarray, y: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(X, dtype=torch.float32, device=device),
        torch.as_tensor(y, dtype=torch.long, device=device),
    )


def _model_device(model: EEGNet) -> torch.device:
    return next(model.parameters()).device
