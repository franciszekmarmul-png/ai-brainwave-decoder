import numpy as np
import pandas as pd
import pytest
from conftest import make_synthetic
from matplotlib.backends.backend_agg import FigureCanvasAgg

from src import evaluate, plots
from src.evaluate import RESULT_COLUMNS, run_experiment, train_eegnet_fold
from src.plots import (
    calibration_curve,
    calibration_figure,
    demo_chart,
    demo_figure,
    main,
    paired_difference,
    person_summary,
    plot_calibration,
    plot_replication,
    replication_figure,
    summarize,
)
from src.train import TrainConfig, predict_proba

QUICK = TrainConfig(max_epochs=2, device="cpu")


def results_frame(experiment, model, accuracies, subjects=None):
    n = len(accuracies)
    return pd.DataFrame(
        {
            "experiment": experiment,
            "model": model,
            "subject": subjects if subjects is not None else [1] * n,
            "fold": range(n),
            "n_calibration": np.nan,
            "selection": np.nan,
            "seed": range(n),
            "accuracy": accuracies,
            "best_epoch": 10,
            "device": "cpu",
        }
    )[RESULT_COLUMNS]


def write_replication_results(folder):
    results_frame("within", "fbcsp", [0.6, 0.8]).to_csv(folder / "within_fbcsp.csv", index=False)
    results_frame("within", "eegnet", [0.5, 0.5]).to_csv(folder / "within_eegnet.csv", index=False)
    results_frame("cross", "fbcsp", [0.3, 0.3, 0.3]).to_csv(folder / "cross_fbcsp.csv", index=False)
    results_frame("cross", "eegnet", [0.4, 0.4, 0.4]).to_csv(folder / "cross_eegnet.csv", index=False)


def calibration_results():
    rows = [(1, 0, 0, 0.3), (2, 0, 0, 0.5), (1, 12, 0, 0.4), (1, 12, 1, 0.6), (2, 12, 0, 0.7), (2, 12, 1, 0.7)]
    frame = pd.DataFrame(rows, columns=["subject", "n_calibration", "selection", "accuracy"])
    for column, value in {"experiment": "calibration", "model": "eegnet", "fold": 0, "seed": 1, "best_epoch": 10,
                          "device": "cpu"}.items():
        frame[column] = value
    return frame[RESULT_COLUMNS]


def line_with_constant(ax, y):
    return [line for line in ax.lines if len(line.get_ydata()) and np.allclose(line.get_ydata(), y)]


def test_summarize_mean_and_standard_error():
    df = pd.DataFrame({"experiment": ["within"] * 2, "model": ["eegnet"] * 2, "accuracy": [0.5, 0.7]})
    row = summarize(df).iloc[0]
    assert row["mean"] == pytest.approx(0.6) and row["sem"] == pytest.approx(0.1) and row["n"] == 2


def test_summarize_keeps_models_and_experiments_apart():
    df = pd.concat([results_frame("within", "eegnet", [0.5, 0.7]), results_frame("cross", "eegnet", [0.2, 0.2])])
    summary = summarize(df).set_index("experiment")
    assert summary.loc["within", "mean"] == pytest.approx(0.6)
    assert summary.loc["cross", "mean"] == pytest.approx(0.2)


def test_calibration_curve_averages_within_each_person_first():
    curve = calibration_curve(calibration_results()).set_index("n_calibration")
    assert curve.loc[12, "mean"] == pytest.approx(0.6)
    assert curve.loc[12, "sem"] == pytest.approx(0.1)
    assert curve.loc[12, "n"] == 2
    assert curve.loc[0, "mean"] == pytest.approx(0.4)


def test_plots_write_files(tmp_path):
    write_replication_results(tmp_path)
    calibration_results().to_csv(tmp_path / "calibration_eegnet.csv", index=False)
    plot_replication(tmp_path, tmp_path / "replication.png")
    plot_calibration(tmp_path / "calibration_eegnet.csv", tmp_path / "calibration.png")
    for name in ("replication.png", "calibration.png"):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG")


def test_replication_chart_labels_each_model_mean_and_run_count(tmp_path):
    write_replication_results(tmp_path)
    ax = replication_figure(tmp_path).axes[0]
    labels = [text.get_text() for text in ax.texts]
    assert sorted(label for label in labels if label.endswith("%") and label[0].isdigit()) == [
        "30%", "40%", "50%", "70%",
    ]
    ticks = " ".join(tick.get_text() for tick in ax.get_xticklabels())
    assert "2 runs" in ticks and "3 runs" in ticks


def test_replication_chart_shows_guessing_level_and_paper_values(tmp_path):
    write_replication_results(tmp_path)
    ax = replication_figure(tmp_path).axes[0]
    assert line_with_constant(ax, 0.25)
    legend = [text.get_text() for text in ax.get_legend().get_texts()]
    assert legend[:2] == ["FBCSP", "EEGNet"] and any("paper" in label.lower() for label in legend)
    papers = [line for line in ax.lines if line.get_marker() == "D"]
    assert sorted(float(y) for line in papers for y in line.get_ydata()) == [0.32, 0.39, 0.68, 0.68]


def test_plot_calibration_includes_chance_and_zero_trial_point(tmp_path):
    calibration_results().to_csv(tmp_path / "c.csv", index=False)
    ax = calibration_figure(tmp_path / "c.csv").axes[0]
    assert line_with_constant(ax, 0.25)
    assert any(0 in line.get_xdata() and 12 in line.get_xdata() for line in ax.lines)


def test_calibration_chart_shows_each_person_and_the_mean(tmp_path):
    calibration_results().to_csv(tmp_path / "c.csv", index=False)
    ax = calibration_figure(tmp_path / "c.csv").axes[0]
    curves = sorted(tuple(np.round(line.get_ydata(), 3)) for line in ax.lines if len(line.get_ydata()) == 2)
    assert (0.3, 0.5) in curves
    assert (0.5, 0.7) in curves
    assert (0.4, 0.6) in curves


def test_calibration_chart_adds_same_person_level_when_available(tmp_path):
    calibration_results().to_csv(tmp_path / "c.csv", index=False)
    results_frame("within", "eegnet", [0.9, 0.7]).to_csv(tmp_path / "within_eegnet.csv", index=False)
    ax = calibration_figure(tmp_path / "c.csv").axes[0]
    assert line_with_constant(ax, 0.8)


def test_missing_results_file_gives_clear_error(tmp_path):
    write_replication_results(tmp_path)
    (tmp_path / "cross_eegnet.csv").unlink()
    with pytest.raises(FileNotFoundError, match="cross_eegnet.csv"):
        plot_replication(tmp_path, tmp_path / "out.png")
    with pytest.raises(FileNotFoundError, match="nothing.csv"):
        plot_calibration(tmp_path / "nothing.csv", tmp_path / "out.png")


def test_plots_create_their_output_folder(tmp_path):
    write_replication_results(tmp_path)
    plot_replication(tmp_path, tmp_path / "new" / "replication.png")
    assert (tmp_path / "new" / "replication.png").exists()


def test_replication_chart_says_when_models_have_different_run_counts(tmp_path):
    write_replication_results(tmp_path)
    results_frame("cross", "eegnet", [0.4]).to_csv(tmp_path / "cross_eegnet.csv", index=False)
    ticks = " ".join(tick.get_text() for tick in replication_figure(tmp_path).axes[0].get_xticklabels())
    assert "FBCSP 3 runs, EEGNet 1 runs" in ticks


def test_error_bars_are_drawn_over_the_columns(tmp_path):
    write_replication_results(tmp_path)
    ax = replication_figure(tmp_path).axes[0]
    columns = max(patch.get_zorder() for patch in ax.patches)
    assert ax.collections and all(bars.get_zorder() > columns for bars in ax.collections)


@pytest.fixture(scope="module")
def d():
    return make_synthetic(n_subjects=3, n_per_class=8)


def figure_text(fig):
    return " ".join(text.get_text() for text in fig.texts)


def test_demo_uses_the_folds_first_12_unseen_trials(d):
    fold = train_eegnet_fold("within", d, 2, 0, QUICK)
    probabilities = predict_proba(fold.model, fold.X_test[:12])
    right = int((probabilities.argmax(axis=1) == fold.y_test[:12]).sum())

    fig = demo_chart(d, 2, QUICK)

    assert f"{right} of 12" in figure_text(fig)
    icons = [text.get_text() for ax in fig.axes for text in ax.texts if text.get_text() in ("✓", "✗")]
    assert icons.count("✓") == right and icons.count("✗") == 12 - right


def test_demo_probability_bars_match_the_model(d):
    fold = train_eegnet_fold("within", d, 2, 0, QUICK)
    probabilities = predict_proba(fold.model, fold.X_test[:12])
    fig = demo_chart(d, 2, QUICK)
    bars = [ax for ax in fig.axes if ax.patches]
    assert len(bars) == 12
    for ax, expected in zip(bars, probabilities):
        assert [round(p.get_width(), 4) for p in ax.patches] == [round(float(v), 4) for v in expected]


def test_demo_traces_are_the_trials_motor_channels(d):
    fold = train_eegnet_fold("within", d, 2, 0, QUICK)
    fig = demo_chart(d, 2, QUICK)
    traces = [ax for ax in fig.axes if ax.lines]
    assert len(traces) == 12
    for channel, line in zip((7, 9, 11), traces[3].lines):
        y = line.get_ydata()
        shape = y - y.mean()
        expected = fold.X_test[3, channel] - fold.X_test[3, channel].mean()
        assert np.allclose(shape, expected, atol=1e-3)


def test_demo_states_the_saved_runs_accuracy(d, tmp_path):
    df = run_experiment("within", "eegnet", d, tmp_path / "r.csv", subjects=[2], config=QUICK)
    accuracy = df[df.fold == 0].accuracy.iloc[0]
    assert f"{accuracy:.0%}" in figure_text(demo_chart(d, 2, QUICK))


def test_demo_figure_saves_and_reuses_its_model(d, tmp_path, monkeypatch):
    out = tmp_path / "demo.png"
    demo_figure(d, 2, out, QUICK)
    assert out.read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "models" / "demo_subject_2.pt").exists()
    first = figure_text(demo_chart(d, 2, QUICK, saved_model=tmp_path / "models" / "demo_subject_2.pt"))

    def fail(*args, **kwargs):
        raise AssertionError("the saved model should have been reused")

    monkeypatch.setattr(evaluate, "train_model", fail)
    demo_figure(d, 2, out, QUICK)
    assert figure_text(demo_chart(d, 2, QUICK, saved_model=tmp_path / "models" / "demo_subject_2.pt")) == first


def test_demo_rejects_an_unknown_person(d):
    with pytest.raises(ValueError, match="99"):
        demo_chart(d, 99, QUICK)


def test_command_line_writes_all_three_charts(d, tmp_path, monkeypatch):
    write_replication_results(tmp_path)
    calibration_results().to_csv(tmp_path / "calibration_eegnet.csv", index=False)
    monkeypatch.setattr(plots, "load_dataset", lambda: d)
    main(["--results", str(tmp_path), "--demo-subject", "1", "--max-epochs", "1"])
    for name in ("replication.png", "calibration.png", "demo.png"):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG")


def test_demo_text_fits_the_figure_without_colliding(d):
    fig = demo_chart(d, 2, QUICK)
    renderer = FigureCanvasAgg(fig).get_renderer()
    headline_and_subtitle = [text.get_window_extent(renderer) for text in fig.texts]
    panel_text = [text.get_window_extent(renderer) for ax in fig.axes for text in ax.texts]
    tick_labels = [label.get_window_extent(renderer) for ax in fig.axes for label in ax.get_yticklabels()]
    for box in headline_and_subtitle + panel_text + tick_labels:
        assert box.x0 >= 0 and box.x1 <= fig.bbox.x1 and box.y0 >= 0 and box.y1 <= fig.bbox.y1
    for top in headline_and_subtitle:
        assert not any(top.overlaps(box) for box in panel_text)


def test_person_summary_uses_one_value_per_person():
    df = results_frame("within", "eegnet", [0.4, 0.6, 0.7], subjects=[1, 1, 2])
    row = person_summary(df).iloc[0]
    assert row["mean"] == pytest.approx(0.6) and row["sem"] == pytest.approx(0.1) and row["n"] == 2


def test_paired_difference_compares_models_within_each_person():
    a = results_frame("within", "eegnet", [0.5, 0.9], subjects=[1, 2])
    b = results_frame("within", "fbcsp", [0.4, 0.6], subjects=[1, 2])
    result = paired_difference(a, b)
    assert result["mean"] == pytest.approx(0.2) and result["sem"] == pytest.approx(0.1) and result["n"] == 2


def test_replication_chart_says_error_bars_are_over_runs(tmp_path):
    write_replication_results(tmp_path)
    assert "over runs" in " ".join(text.get_text() for text in replication_figure(tmp_path).texts)


def test_calibration_chart_explains_its_band_and_reference_line(tmp_path):
    calibration_results().to_csv(tmp_path / "c.csv", index=False)
    results_frame("within", "eegnet", [0.9, 0.7]).to_csv(tmp_path / "within_eegnet.csv", index=False)
    fig = calibration_figure(tmp_path / "c.csv")
    assert "±1 standard error across people" in " ".join(text.get_text() for text in fig.texts)
    labels = " ".join(text.get_text() for text in fig.axes[0].texts)
    assert "Average when trained on the same person" in labels and "288 trials" in labels
