from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import PercentFormatter

from src.data import CLASSES, EEGData, load_dataset
from src.evaluate import train_eegnet_fold
from src.train import TrainConfig, predict_proba

CHANCE = 0.25
PAPER_DPI = 200

PAPER_VALUES = {
    ("within", "fbcsp"): 0.68,
    ("within", "eegnet"): 0.68,
    ("cross", "fbcsp"): 0.32,
    ("cross", "eegnet"): 0.39,
}

_SURFACE = "#fcfcfb"
_TEXT = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_BASELINE = "#c3c2b7"
_MODEL_COLORS = {"fbcsp": "#2a78d6", "eegnet": "#eb6834"}
_MODEL_NAMES = {"fbcsp": "FBCSP", "eegnet": "EEGNet"}
_GOOD = "#0ca30c"
_CRITICAL = "#d03b3b"
_CLASS_NAMES = {"left_hand": "Left hand", "right_hand": "Right hand", "feet": "Feet", "tongue": "Tongue"}
_DEMO_CHANNELS = {"C3": 7, "Cz": 9, "C4": 11}
_DEMO_TRIALS = 12
_DEMO_FOLD = 0
_DEMO_COLUMNS = 4
_DEMO_HEADER_INCHES = 1.4
_DEMO_ROW_INCHES = 2.75
_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "text.color": _TEXT,
    "axes.labelcolor": _TEXT_SECONDARY,
    "axes.edgecolor": _BASELINE,
    "axes.facecolor": _SURFACE,
    "figure.facecolor": _SURFACE,
    "xtick.color": _TEXT_SECONDARY,
    "ytick.color": _MUTED,
}


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["experiment", "model"], sort=False)["accuracy"]
    return grouped.agg(mean="mean", sem="sem", n="count").reset_index()


def person_summary(df: pd.DataFrame) -> pd.DataFrame:
    per_person = df.groupby(["experiment", "model", "subject"], sort=False)["accuracy"].mean().reset_index()
    return summarize(per_person)


def paired_difference(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    differences = a.groupby("subject")["accuracy"].mean() - b.groupby("subject")["accuracy"].mean()
    return {"mean": differences.mean(), "sem": differences.sem(), "n": int(differences.count())}


def calibration_curve(df: pd.DataFrame) -> pd.DataFrame:
    per_person = df.groupby(["n_calibration", "subject"])["accuracy"].mean().reset_index()
    curve = per_person.groupby("n_calibration")["accuracy"].agg(mean="mean", sem="sem", n="count")
    return curve.reset_index()


def replication_figure(results_dir: Path) -> Figure:
    results_dir = Path(results_dir)
    frames = [
        _read_results(results_dir / f"{experiment}_{model}.csv")
        for experiment in ("within", "cross")
        for model in ("fbcsp", "eegnet")
    ]
    summary = summarize(pd.concat(frames)).set_index(["experiment", "model"])

    with matplotlib.rc_context(_STYLE):
        fig = Figure(figsize=(7.2, 4.8))
        ax = fig.add_axes((0.09, 0.15, 0.88, 0.67))
        _style_percent_axis(ax)
        ax.set_xlim(-0.6, 1.6)
        column_width = 0.22
        for position, experiment in enumerate(("within", "cross")):
            for side, model in zip((-0.5, 0.5), ("fbcsp", "eegnet")):
                row = summary.loc[(experiment, model)]
                x = position + side * (column_width + 0.03)
                _rounded_column(ax, x - column_width / 2, column_width, row["mean"], _MODEL_COLORS[model])
                ax.errorbar(
                    x, row["mean"], yerr=2 * row["sem"], fmt="none", ecolor=_TEXT, elinewidth=0.8, capsize=3, zorder=4
                )
                paper = PAPER_VALUES[(experiment, model)]
                ax.plot([x], [paper], marker="D", linestyle="none", markersize=6, color=_TEXT,
                        markeredgecolor=_SURFACE, markeredgewidth=1.2, zorder=5)
                label_y = max(row["mean"] + 2 * row["sem"], paper) + 0.025
                ax.text(x, label_y, f"{row['mean']:.0%}", ha="center", va="bottom", fontsize=11)

        ax.axhline(CHANCE, color=_MUTED, linewidth=0.8, zorder=3)
        ax.text(1.6, CHANCE + 0.012, "Guessing: 25%", ha="right", va="bottom", color=_TEXT_SECONDARY, fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [
                f"Trained on the same person\nwithin-subject, {_runs_label(summary, 'within')}",
                f"Tested on a new person\ncross-subject, {_runs_label(summary, 'cross')}",
            ]
        )
        ax.tick_params(axis="x", length=0, pad=8)
        ax.legend(
            handles=[
                Patch(color=_MODEL_COLORS["fbcsp"], label=_MODEL_NAMES["fbcsp"]),
                Patch(color=_MODEL_COLORS["eegnet"], label=_MODEL_NAMES["eegnet"]),
                Line2D([], [], marker="D", linestyle="none", color=_TEXT, markersize=6,
                       label="Paper's value (read from its charts)"),
            ],
            loc="upper right",
            frameon=False,
            fontsize=9,
        )
        fig.text(0.09, 0.93, "How well each model decodes imagined movements", fontsize=14, fontweight="bold")
        fig.text(
            0.09, 0.875, "Mean accuracy over all runs, with error bars of ±2 standard errors over runs",
            fontsize=9.5, color=_TEXT_SECONDARY,
        )
    return fig


def calibration_figure(csv: Path) -> Figure:
    csv = Path(csv)
    df = _read_results(csv)
    curve = calibration_curve(df)
    sizes = curve["n_calibration"].to_numpy()
    mean, sem = curve["mean"].to_numpy(), curve["sem"].to_numpy()
    color = _MODEL_COLORS["eegnet"]

    with matplotlib.rc_context(_STYLE):
        fig = Figure(figsize=(7.2, 4.8))
        ax = fig.add_axes((0.09, 0.13, 0.88, 0.69))
        _style_percent_axis(ax)
        right = sizes.max() * 1.22 if sizes.max() else 1
        ax.set_xlim(-sizes.max() * 0.03, right)

        per_person = df.groupby(["subject", "n_calibration"])["accuracy"].mean().unstack("subject")
        for subject in per_person.columns:
            ax.plot(per_person.index, per_person[subject], color=_BASELINE, linewidth=0.9, zorder=2)
        ax.fill_between(sizes, mean - sem, mean + sem, color=color, alpha=0.12, linewidth=0, zorder=3)
        ax.plot(sizes, mean, color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=_SURFACE,
                markeredgewidth=1.3, zorder=4, solid_capstyle="round")

        ax.axhline(CHANCE, color=_MUTED, linewidth=0.8, zorder=1)
        ax.text(right, CHANCE - 0.012, "Guessing: 25%", ha="right", va="top", color=_TEXT_SECONDARY, fontsize=9)
        same_person = csv.parent / "within_eegnet.csv"
        if same_person.exists():
            level = pd.read_csv(same_person)["accuracy"].mean()
            ax.axhline(level, color=_TEXT_SECONDARY, linewidth=0.8, zorder=1)
            ax.text(right, level + 0.012, f"Average when trained on the same person (288 trials, different test trials): {level:.0%}",
                    ha="right", va="bottom",
                    color=_TEXT_SECONDARY, fontsize=9)

        ax.text(sizes[-1] + sizes.max() * 0.02, mean[-1], f"{mean[-1]:.0%} average\nof {int(curve['n'].iloc[-1])} people",
                ha="left", va="center", fontsize=9.5)
        ax.text(sizes[0], mean[0] + 0.045, f"{mean[0]:.0%}", ha="center", va="bottom", fontsize=9.5)
        ax.set_xticks(sizes)
        ax.tick_params(axis="x", length=0, pad=6)
        ax.set_xlabel("Trials from the new person used for fine-tuning")
        fig.text(0.09, 0.93, "How much practice data a new person needs", fontsize=14, fontweight="bold")
        fig.text(
            0.09, 0.905,
            "EEGNet trained on 8 other people, then fine-tuned on the new person.\n"
            "Grey lines: each person. Band: ±1 standard error across people.",
            fontsize=9.5, color=_TEXT_SECONDARY, va="top", linespacing=1.4,
        )
    return fig


def demo_chart(data: EEGData, subject: int, config: TrainConfig, saved_model: Path | None = None) -> Figure:
    torch.set_num_threads(1)
    fold = train_eegnet_fold("within", data, subject, _DEMO_FOLD, config, saved_model=saved_model)
    probabilities = predict_proba(fold.model, fold.X_test)
    predictions = probabilities.argmax(axis=1)
    accuracy = (predictions == fold.y_test).mean()
    n_shown = min(_DEMO_TRIALS, len(fold.X_test))
    n_right = int((predictions[:n_shown] == fold.y_test[:n_shown]).sum())
    n_rows = int(np.ceil(n_shown / _DEMO_COLUMNS))

    height = _DEMO_HEADER_INCHES + _DEMO_ROW_INCHES * n_rows
    with matplotlib.rc_context(_STYLE):
        fig = Figure(figsize=(12.5, height))
        grid = fig.add_gridspec(
            n_rows, _DEMO_COLUMNS, left=0.075, right=0.985, top=1 - _DEMO_HEADER_INCHES / height, bottom=0.025,
            hspace=0.3, wspace=0.42,
        )
        for trial in range(n_shown):
            cell = grid[trial // _DEMO_COLUMNS, trial % _DEMO_COLUMNS].subgridspec(
                2, 1, height_ratios=(1.25, 1), hspace=0.08
            )
            _demo_traces(fig.add_subplot(cell[0]), fold.X_test[trial], fold.y_test[trial], predictions[trial])
            _demo_probabilities(fig.add_subplot(cell[1]), probabilities[trial], fold.y_test[trial])
        fig.text(
            0.04, 1 - 0.42 / height, f"Person {subject}: the model got {n_right} of {n_shown} unseen trials right",
            fontsize=15, fontweight="bold",
        )
        fig.text(
            0.04, 1 - 0.74 / height,
            f"The first {n_shown} trials of a test block, not hand-picked. Across all {len(fold.X_test)} trials in "
            f"this block, accuracy was {accuracy:.0%}. Traces show 2 s of EEG; bold marks what the person imagined.",
            fontsize=9.5, color=_TEXT_SECONDARY,
        )
    return fig


def demo_figure(data: EEGData, subject: int, out: Path, config: TrainConfig) -> None:
    out = Path(out)
    _save(demo_chart(data, subject, config, saved_model=out.parent / "models" / f"demo_subject_{subject}.pt"), out)


def plot_replication(results_dir: Path, out: Path) -> None:
    _save(replication_figure(results_dir), out)


def plot_calibration(csv: Path, out: Path) -> None:
    _save(calibration_figure(csv), out)


def _demo_traces(ax, trial: np.ndarray, label: int, predicted: int) -> None:
    right = label == predicted
    for row, (name, channel) in enumerate(_DEMO_CHANNELS.items()):
        offset = -6.0 * row
        ax.plot(trial[channel] + offset, color=_TEXT_SECONDARY, linewidth=0.55)
        ax.text(-6, offset, name, ha="right", va="center", fontsize=8, color=_MUTED)
    ax.set_xlim(0, trial.shape[1])
    ax.axis("off")
    ax.text(0, 1.09, "✓" if right else "✗", transform=ax.transAxes, fontsize=12, fontweight="bold",
            color=_GOOD if right else _CRITICAL)
    if right:
        heading = f"Correct: {_CLASS_NAMES[CLASSES[label]].lower()}"
    else:
        heading = f"Wrong: said {_CLASS_NAMES[CLASSES[predicted]].lower()}"
    ax.text(0.07, 1.09, heading, transform=ax.transAxes, fontsize=10.5, fontweight="bold")


def _demo_probabilities(ax, probabilities: np.ndarray, label: int) -> None:
    predicted = int(probabilities.argmax())
    positions = np.arange(len(CLASSES))
    colors = [_MODEL_COLORS["eegnet"] if i == predicted else _BASELINE for i in positions]
    ax.barh(positions, probabilities, height=0.62, color=colors)
    ax.set_yticks(positions, [_CLASS_NAMES[name] for name in CLASSES], fontsize=8.5)
    ax.get_yticklabels()[label].set_fontweight("bold")
    ax.get_yticklabels()[label].set_color(_TEXT)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.18)
    ax.text(probabilities[predicted] + 0.02, predicted, f"{probabilities[predicted]:.0%}", va="center", fontsize=8.5)
    ax.tick_params(axis="both", length=0)
    ax.set_xticks([])
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)


def _read_results(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Results file {path} doesn't exist. Run its experiment first.")
    return pd.read_csv(path)


def _save(fig: Figure, out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=PAPER_DPI, facecolor=_SURFACE)


def _runs_label(summary: pd.DataFrame, experiment: str) -> str:
    counts = [int(summary.loc[(experiment, model), "n"]) for model in ("fbcsp", "eegnet")]
    if counts[0] == counts[1]:
        return f"{counts[0]} runs"
    return f"FBCSP {counts[0]} runs, EEGNet {counts[1]} runs"


def _style_percent_axis(ax) -> None:
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="y", color=_GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def _rounded_column(ax, left: float, width: float, height: float, color: str, radius_px: float = 4) -> None:
    (x0, y0), (x1, y1) = ax.transData.transform([(0, 0), (1, 1)])
    scale = ax.figure.dpi / 100
    rx = min(radius_px * scale / (x1 - x0), width / 2)
    ry = min(radius_px * scale / (y1 - y0), height)
    right = left + width
    vertices = [
        (left, 0),
        (left, height - ry),
        (left, height),
        (left + rx, height),
        (right - rx, height),
        (right, height),
        (right, height - ry),
        (right, 0),
        (left, 0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=color, edgecolor="none", zorder=3))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Draw the charts from saved results.")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--demo-subject", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=TrainConfig.max_epochs)
    args = parser.parse_args(argv)
    plot_replication(args.results, args.results / "replication.png")
    plot_calibration(args.results / "calibration_eegnet.csv", args.results / "calibration.png")
    config = TrainConfig(max_epochs=args.max_epochs, device="cpu")
    demo_figure(load_dataset(), args.demo_subject, args.results / "demo.png", config)


if __name__ == "__main__":
    main()
