from __future__ import annotations

from collections.abc import Callable

import mne
import numpy as np
from mne.decoding import CSP
from mne.filter import filter_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

mne.set_log_level("WARNING")

StepCallback = Callable[[dict], None] | None


class CSPLDA:
    def __init__(self):
        self._model = make_pipeline(
            CSP(n_components=4, log=True, reg="ledoit_wolf"),
            LinearDiscriminantAnalysis(),
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, on_step: StepCallback = None) -> "CSPLDA":
        csp, lda = (step for _, step in self._model.steps)
        wide = X_train.astype(np.float64)
        csp.fit(wide, y_train)
        _report(on_step, stage="csp", n_filters=4)
        lda.fit(csp.transform(wide), y_train)
        _report(on_step, stage="lda")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X.astype(np.float64))


class FBCSP:
    def __init__(self, sfreq: float = 128.0, seed: int = 0):
        self.sfreq = sfreq
        self.seed = seed
        self.bands = [(4 + 4 * i, 8 + 4 * i) for i in range(9)]
        self.n_csp_components = 4
        self.l1_ratio = 0.95
        self.c_grid = [float(c) for c in np.logspace(-3, 2, 6)]

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, on_step: StepCallback = None) -> "FBCSP":
        self.classes_ = np.unique(y_train)
        train_bands = self._filter_bands(X_train, on_step)
        val_bands = self._filter_bands(X_val)
        self.csps_, self.scalers_ = [], []
        train_features, val_features = [], []
        for label in self.classes_:
            csps = [self._new_csp().fit(band, y_train == label) for band in train_bands]
            scaler = StandardScaler().fit(_features(csps, train_bands))
            self.csps_.append(csps)
            self.scalers_.append(scaler)
            train_features.append(scaler.transform(_features(csps, train_bands)))
            val_features.append(scaler.transform(_features(csps, val_bands)))
            _report(on_step, stage="csp", label=int(label))

        best_accuracy = -1.0
        for c in self.c_grid:
            classifiers = [
                self._new_classifier(c).fit(features, y_train == label)
                for label, features in zip(self.classes_, train_features)
            ]
            predictions = self._choose_class(classifiers, val_features)
            accuracy = (predictions == y_val).mean()
            _report(on_step, stage="tune", c=c, val_accuracy=float(accuracy))
            if accuracy > best_accuracy:
                best_accuracy, self.best_c_, self.classifiers_ = accuracy, c, classifiers
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        bands = self._filter_bands(X)
        features = [scaler.transform(_features(csps, bands)) for csps, scaler in zip(self.csps_, self.scalers_)]
        return self._choose_class(self.classifiers_, features)

    def _filter_bands(self, X: np.ndarray, on_step: StepCallback = None) -> list[np.ndarray]:
        wide = X.astype(np.float64)
        bands = []
        for low, high in self.bands:
            bands.append(filter_data(wide, self.sfreq, low, high, method="iir", verbose=False))
            _report(on_step, stage="filter", band=[low, high])
        return bands

    def _new_csp(self) -> CSP:
        return CSP(n_components=self.n_csp_components, log=True, reg="ledoit_wolf", component_order="alternate")

    def _new_classifier(self, c: float) -> LogisticRegression:
        return LogisticRegression(C=c, l1_ratio=self.l1_ratio, solver="saga", max_iter=5000, random_state=self.seed)

    def _choose_class(self, classifiers: list[LogisticRegression], features: list[np.ndarray]) -> np.ndarray:
        scores = np.column_stack([c.decision_function(f) for c, f in zip(classifiers, features)])
        return self.classes_[scores.argmax(axis=1)]


def _features(csps: list[CSP], bands: list[np.ndarray]) -> np.ndarray:
    return np.hstack([csp.transform(band) for csp, band in zip(csps, bands)])


def _report(on_step: StepCallback, **event) -> None:
    if on_step is not None:
        on_step(event)
