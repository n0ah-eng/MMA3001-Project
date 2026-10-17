"""Machine-learning gap filling: learn a correction to linear interpolation.

Idea
----
Linear interpolation gets the *trend* across a gap right but cannot know the
shape inside it (e.g. the afternoon warm-up in a 6-hour gap). A regression
model is trained to predict the **residual**

    r = y_true - y_linear

from features describing *where* the missing sample sits (time of day,
weekday/weekend, position in the gap) and *what happened yesterday* over
the same window. The reconstruction is ``y_linear + r_hat``.

Why predict the residual instead of the value itself?

* Tree models (random forest) cannot extrapolate: if the test period is
  warmer than anything seen in training, a value-predicting forest is
  capped at the training maximum. Residuals are centred on zero in every
  season, so this problem disappears.
* If the model learns nothing useful it predicts ~0 and falls back to
  linear interpolation, so it should rarely be much *worse* than the
  baseline.

Features (computed only from data that would exist during a real outage)
-----------------------------------------------------------------------
==================  ==========================================================
hour_sin, hour_cos  Time of day on a circle, so 23:50 is next to 00:00.
is_weekend          1 on Saturday/Sunday (different building use).
gap_hours           Length of the gap containing the sample.
pos_frac            Position within the gap: 0 at the start, 1 at the end.
slope_per_h         Linear trend across the gap (end - start) / gap_hours.
analog_minus_lin    "Yesterday's shape": y[gap start] + (y_yesterday[t] -
                    y_yesterday[gap start]) minus the linear estimate.
analog_missing      1 if yesterday's values were unavailable (feature set 0).
==================  ==========================================================

Other variables of the same sensor (e.g. humidity) are deliberately **not**
used: when a sensor drops out, all of its channels are lost together.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sensorgap.validation import insert_gaps, sample_gap_starts

#: Samples per day on the 10-minute grid.
DAY = 144

FEATURES = ["hour_sin", "hour_cos", "is_weekend", "gap_hours", "pos_frac",
            "slope_per_h", "analog_minus_lin", "analog_missing"]


def gap_features(series: pd.Series) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build the feature table for every interior missing sample.

    Args:
        series: Regular 10-minute series with a DatetimeIndex; NaN = missing.

    Returns:
        ``(X, y_lin, pos)``: feature DataFrame (columns :data:`FEATURES`), the
        linear-interpolation estimate at each row, and the integer positions
        of those samples in ``series``. Leading/trailing NaN are excluded
        (no bracketing samples, so no gap is defined).
    """
    y = series.to_numpy(dtype=float)
    n = len(y)
    known = ~np.isnan(y)
    idx = np.arange(n)

    # Index of the previous / next known sample for every position.
    prev = np.where(known, idx, -1)
    prev = np.maximum.accumulate(prev)
    nxt = np.where(known, idx, n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]

    pos = np.flatnonzero(~known & (prev >= 0) & (nxt < n))
    a, b = prev[pos], nxt[pos]
    span = (b - a).astype(float)
    frac = (pos - a) / span
    y_lin = y[a] + (y[b] - y[a]) * frac
    gap_hours = span * 10 / 60

    # Yesterday's shape anchored at the gap start ("analog" forecast).
    ja, j = a - DAY, pos - DAY
    valid = ja >= 0
    analog = np.full(len(pos), np.nan)
    analog[valid] = y[a[valid]] + (y[j[valid]] - y[ja[valid]])
    analog_missing = np.isnan(analog)
    analog_minus_lin = np.where(analog_missing, 0.0, analog - y_lin)

    t = series.index[pos]
    hour = t.hour + t.minute / 60
    X = pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "is_weekend": (t.dayofweek >= 5).astype(float),
        "gap_hours": gap_hours,
        "pos_frac": frac,
        "slope_per_h": (y[b] - y[a]) / gap_hours,
        "analog_minus_lin": analog_minus_lin,
        "analog_missing": analog_missing.astype(float),
    })
    return X, y_lin, pos


def make_training_set(
    train: pd.Series,
    gap_lengths: list[int],
    n_gaps: int = 30,
    seeds: tuple[int, ...] = (100, 101, 102),
    n_blocks: int = 5,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Create supervised examples by hiding data in the training period.

    Args:
        train: Training-period series only (the test period must be excluded
            by the caller, so no test value can leak into training).
        gap_lengths: Gap lengths (samples) to simulate.
        n_gaps: Gaps per (seed, gap length).
        seeds: Seeds for gap placement. Deliberately different from the
            evaluation seeds.
        n_blocks: Number of contiguous time blocks used as CV groups.

    Returns:
        ``(X, r, groups, gap_len)``: features, residual targets
        ``y_true - y_linear``, the time-block id of each example (for blocked
        cross-validation) and the simulated gap length that produced it.
    """
    observed = train.notna().to_numpy()
    block_of = np.minimum(np.arange(len(train)) * n_blocks // len(train), n_blocks - 1)
    Xs, rs, gs, ls = [], [], [], []
    for seed in seeds:
        for gap_len in gap_lengths:
            rng = np.random.default_rng([seed, gap_len])
            starts = sample_gap_starts(observed, gap_len, n_gaps, rng)
            if len(starts) == 0:
                continue
            masked, hidden = insert_gaps(train, starts, gap_len)
            X, y_lin, pos = gap_features(masked)
            keep = hidden[pos]  # only rows with a true value to learn from
            Xs.append(X[keep])
            rs.append(train.to_numpy()[pos[keep]] - y_lin[keep])
            gs.append(block_of[pos[keep]])
            ls.append(np.full(keep.sum(), gap_len))
    return (pd.concat(Xs, ignore_index=True), np.concatenate(rs),
            np.concatenate(gs), np.concatenate(ls))


#: Candidate models and the hyperparameter grids searched by cross-validation.
MODEL_GRIDS: dict[str, tuple[RegressorMixin, dict]] = {
    "ridge": (Pipeline([("scale", StandardScaler()), ("model", Ridge())]),
              {"model__alpha": [0.1, 1.0, 10.0, 100.0]}),
    "knn": (Pipeline([("scale", StandardScaler()), ("model", KNeighborsRegressor())]),
            {"model__n_neighbors": [5, 20, 50]}),
    "random_forest": (RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1),
                      {"min_samples_leaf": [5, 20, 50], "max_depth": [8, None]}),
}


class MLGapFiller:
    """Gap filler = linear interpolation + a learned residual correction.

    Usage::

        filler = MLGapFiller("ridge").fit(train_series, gap_lengths=[1, 6, 18, 36, 144])
        filled = filler(masked_series)      # same signature as interpolation methods

    Attributes:
        name: Key into :data:`MODEL_GRIDS`.
        model_: Fitted scikit-learn estimator (best by cross-validation).
        best_params_: Hyperparameters chosen by cross-validation.
        cv_rmse_: Cross-validated RMSE of the chosen model.
        cv_by_gap_: Table of cross-validated RMSE per simulated gap length,
            for the model and for plain linear interpolation.
        min_gap_len_: Shortest gap (samples) on which the ML correction is
            applied; shorter gaps fall back to linear interpolation.
        fit_time_s_: Wall-clock time for tuning + fitting.
        n_train_: Number of training examples.
    """

    def __init__(self, name: str, gated: bool = True):
        """
        Args:
            name: Model key in :data:`MODEL_GRIDS`.
            gated: If True (default), apply the ML correction only on gap
                lengths where cross-validation showed it beats linear
                interpolation. If False, always apply it.

        Raises:
            ValueError: If ``name`` is unknown.
        """
        if name not in MODEL_GRIDS:
            raise ValueError(f"Unknown model {name!r}; choose from {list(MODEL_GRIDS)}")
        self.name = name
        self.gated = gated
        self.model_ = None

    def fit(self, train: pd.Series, gap_lengths: list[int], n_cv_folds: int = 5) -> "MLGapFiller":
        """Tune hyperparameters with blocked cross-validation, then fit.

        Blocked (grouped-by-time) folds are used instead of random folds:
        neighbouring 10-minute samples are almost identical, so random folds
        would let the model validate on near-copies of its training data and
        report an optimistic error.

        Args:
            train: Training-period series (no test-period data).
            gap_lengths: Gap lengths to simulate for training examples.
            n_cv_folds: Number of time blocks / CV folds.

        Returns:
            ``self`` (fitted).
        """
        t0 = time.perf_counter()
        X, r, groups, gap_len = make_training_set(train, gap_lengths, n_blocks=n_cv_folds)
        estimator, grid = MODEL_GRIDS[self.name]
        cv = GroupKFold(n_splits=n_cv_folds)
        search = GridSearchCV(clone(estimator), grid, cv=cv,
                              scoring="neg_root_mean_squared_error")
        search.fit(X[FEATURES], r, groups=groups)
        self.model_ = search.best_estimator_
        self.best_params_ = search.best_params_
        self.cv_rmse_ = -search.best_score_

        # Out-of-fold predictions -> does the correction help, per gap length?
        # Linear interpolation corresponds to predicting a residual of 0.
        r_oof = cross_val_predict(clone(search.best_estimator_), X[FEATURES], r,
                                  groups=groups, cv=cv)
        df = pd.DataFrame({"gap_len": gap_len, "lin_sq": r**2, "ml_sq": (r - r_oof) ** 2})
        by_gap = df.groupby("gap_len")[["lin_sq", "ml_sq"]].mean() ** 0.5
        self.cv_by_gap_ = by_gap.rename(columns={"lin_sq": "rmse_linear", "ml_sq": "rmse_ml"})
        # Use ML from the shortest gap length at and beyond which it always wins.
        wins = (self.cv_by_gap_["rmse_ml"] < self.cv_by_gap_["rmse_linear"]).to_numpy()
        lengths = self.cv_by_gap_.index.to_numpy()
        self.min_gap_len_ = np.inf
        for L, all_win in zip(lengths[::-1], np.logical_and.accumulate(wins[::-1])):
            if all_win:
                self.min_gap_len_ = L
        if not self.gated:
            self.min_gap_len_ = 0
        self.fit_time_s_ = time.perf_counter() - t0
        self.n_train_ = len(X)
        return self

    def __call__(self, series: pd.Series) -> pd.Series:
        """Fill interior gaps in ``series`` (same contract as the interpolation methods).

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.model_ is None:
            raise RuntimeError("MLGapFiller must be fitted before use")
        X, y_lin, pos = gap_features(series)
        out = series.copy()
        if len(pos):
            correction = self.model_.predict(X[FEATURES])
            # Number of missing samples in each sample's gap = span - 1.
            n_missing = np.round(X["gap_hours"].to_numpy() * 6).astype(int) - 1
            correction[n_missing < self.min_gap_len_] = 0.0  # short gap -> linear
            out.iloc[pos] = y_lin + correction
        return out
