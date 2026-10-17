"""Validation by artificial gaps ("mask-and-reconstruct").

The true value of a genuinely missing sample is unknown, so accuracy cannot
be measured on real gaps. Instead, contiguous blocks of *observed* data are
hidden, each method reconstructs them, and the reconstruction is compared
with the hidden truth. Repeating this over many randomly placed gaps and
several random seeds gives an error distribution, not a single lucky number.

Design choices:

* **Gap placement** - random but reproducible (seeded). Gaps never overlap,
  are separated by at least one gap length of untouched data so each gap has
  genuine neighbours, and stay away from the record ends (no extrapolation).
* **Natural gaps are kept** - the series already has ~10 % missing samples.
  They stay missing inside and around artificial gaps, exactly as in real
  use. Error is evaluated only on hidden samples that had a true value.
* **Metrics** - RMSE (penalises large errors), MAE, maximum absolute error
  (worst case) and bias (systematic over/under-estimation), plus the
  *overshoot fraction*: the share of filled values lying outside the range
  of the two known samples bracketing the gap. Overshoot is a physical
  plausibility check - e.g. CO2 cannot dip below what either end of the
  gap shows unless something happened that the data never recorded.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from sensorgap.interpolation import FillFunction


def sample_gap_starts(
    observed: np.ndarray,
    gap_len: int,
    n_gaps: int,
    rng: np.random.Generator,
    min_observed_frac: float = 0.8,
    allowed: np.ndarray | None = None,
) -> np.ndarray:
    """Choose start indices for non-overlapping artificial gaps.

    Args:
        observed: Boolean array, True where the original series has a value.
        gap_len: Number of consecutive samples in each gap (>= 1).
        n_gaps: Number of gaps wanted.
        rng: NumPy random generator (seeded by the caller for reproducibility).
        min_observed_frac: A candidate window is accepted only if at least this
            fraction of it was observed, so there is enough truth to score.
        allowed: Optional boolean array; gaps (and their buffers) are placed
            only where it is True. Used to restrict testing to the held-out
            test period so ML models are never scored on data they trained on.

    Returns:
        Sorted array of start indices. May contain fewer than ``n_gaps``
        entries if the series is too short or too sparse.

    Raises:
        ValueError: If ``gap_len < 1``.
    """
    if gap_len < 1:
        raise ValueError("gap_len must be >= 1")
    n = len(observed)
    buffer = gap_len  # untouched data required either side of each gap
    lo, hi = buffer, n - gap_len - buffer
    if hi <= lo:
        return np.array([], dtype=int)

    # Fraction observed in every window, via a cumulative sum (O(n)).
    csum = np.concatenate([[0], np.cumsum(observed)])
    window_frac = (csum[gap_len:] - csum[:-gap_len]) / gap_len
    candidates = np.arange(lo, hi)
    candidates = candidates[window_frac[candidates] >= min_observed_frac]
    # Both edge samples must be observed so each gap has true neighbours.
    candidates = candidates[observed[candidates - 1] & observed[candidates + gap_len]]
    if allowed is not None:
        # The whole window [s - buffer, s + gap_len + buffer) must be allowed.
        span = gap_len + 2 * buffer
        acs = np.concatenate([[0], np.cumsum(allowed)])
        ok = (acs[candidates - buffer + span] - acs[candidates - buffer]) == span
        candidates = candidates[ok]
    rng.shuffle(candidates)

    taken = np.zeros(n, dtype=bool)  # samples reserved by accepted gaps + buffers
    starts = []
    for s in candidates:
        a, b = s - buffer, s + gap_len + buffer
        if not taken[a:b].any():
            taken[a:b] = True
            starts.append(s)
            if len(starts) == n_gaps:
                break
    return np.sort(np.array(starts, dtype=int))


def insert_gaps(series: pd.Series, starts: np.ndarray, gap_len: int) -> tuple[pd.Series, np.ndarray]:
    """Hide ``gap_len`` samples at each start index.

    Args:
        series: Original series.
        starts: Start indices from :func:`sample_gap_starts`.
        gap_len: Samples per gap.

    Returns:
        ``(masked, hidden)``: a copy of ``series`` with the gaps set to NaN, and
        a boolean array that is True where a *true* value was hidden (i.e. the
        positions on which the reconstruction is scored).
    """
    in_gap = np.zeros(len(series), dtype=bool)
    for s in starts:
        in_gap[s:s + gap_len] = True
    hidden = in_gap & series.notna().to_numpy()
    masked = series.copy()
    masked.iloc[np.flatnonzero(in_gap)] = np.nan
    return masked, hidden


def score(truth: pd.Series, filled: pd.Series, masked: pd.Series, hidden: np.ndarray) -> dict[str, float]:
    """Error metrics on the hidden samples.

    Args:
        truth: Original series (ground truth).
        filled: Reconstruction returned by a fill method.
        masked: The series the method was given (used for overshoot bounds).
        hidden: Boolean array of positions to score.

    Returns:
        Dict with ``rmse``, ``mae``, ``max_abs``, ``bias`` (mean of
        filled - true), ``overshoot_frac`` and ``n_scored``.

    Raises:
        ValueError: If the method left any hidden sample unfilled.
    """
    y = truth.to_numpy(dtype=float)[hidden]
    yhat = filled.to_numpy(dtype=float)[hidden]
    if np.isnan(yhat).any():
        raise ValueError("Fill method left hidden samples as NaN")
    err = yhat - y

    # Bracketing known values: last known before / next known after each sample.
    before = masked.ffill().to_numpy(dtype=float)[hidden]
    after = masked.bfill().to_numpy(dtype=float)[hidden]
    lo, hi = np.minimum(before, after), np.maximum(before, after)
    tol = 1e-9 * np.maximum(1.0, np.abs(hi))  # ignore floating-point round-off
    overshoot = (yhat < lo - tol) | (yhat > hi + tol)

    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
        "bias": float(np.mean(err)),
        "overshoot_frac": float(np.mean(overshoot)),
        "n_scored": int(hidden.sum()),
    }


def run_gap_experiment(
    series: pd.Series,
    methods: dict[str, FillFunction],
    gap_lengths: list[int],
    n_gaps: int = 30,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    allowed: np.ndarray | None = None,
) -> pd.DataFrame:
    """Score every method on every gap length over several random seeds.

    For each ``(seed, gap_len)`` one masked series is built and *all* methods
    are given the identical masked series, so differences between methods
    are not caused by different gap placements (a paired comparison).

    Args:
        series: Regularly spaced series with a DatetimeIndex (NaN = missing).
        methods: ``{name: fill_function}``.
        gap_lengths: Gap lengths in samples, e.g. ``[1, 6, 18, 36, 144]``.
        n_gaps: Gaps inserted per (seed, gap length).
        seeds: Random seeds; one repeat per seed.
        allowed: Optional boolean array restricting where gaps may be placed
            (see :func:`sample_gap_starts`).

    Returns:
        Long-format DataFrame, one row per (seed, gap_len, method), with the
        metrics from :func:`score`, ``n_gaps`` actually placed and the
        wall-clock ``runtime_s`` of the fill call.
    """
    observed = series.notna().to_numpy()
    rows = []
    for seed in seeds:
        for gap_len in gap_lengths:
            rng = np.random.default_rng([seed, gap_len])
            starts = sample_gap_starts(observed, gap_len, n_gaps, rng, allowed=allowed)
            if len(starts) == 0:
                continue
            masked, hidden = insert_gaps(series, starts, gap_len)
            for name, fill in methods.items():
                t0 = time.perf_counter()
                filled = fill(masked)
                runtime = time.perf_counter() - t0
                rows.append({"seed": seed, "gap_len": gap_len, "method": name,
                             "n_gaps": len(starts), "runtime_s": runtime,
                             **score(series, filled, masked, hidden)})
    return pd.DataFrame(rows)


def add_skill(results: pd.DataFrame, naive: str = "mean",
              pair: tuple[str, ...] = ("sensor_id", "variable", "seed", "gap_len")) -> pd.DataFrame:
    """Add a paired skill score relative to a naive reference method.

    ``skill = 1 - RMSE_method / RMSE_naive``, where both RMSEs come from the
    *same* gaps (same keys in ``pair``). 0 = no better than the naive method,
    1 = perfect, negative = worse than the naive method.

    Args:
        results: Long table from :func:`run_gap_experiment`, with the pairing
            columns added by the caller (e.g. ``sensor_id``, ``variable``).
        naive: Name of the reference method in ``results["method"]``.
        pair: Columns that identify one set of identical gaps.

    Returns:
        Copy of ``results`` with ``rmse_naive`` and ``skill`` columns.
    """
    pair = list(pair)
    ref = results[results["method"] == naive].set_index(pair)["rmse"].rename("rmse_naive")
    out = results.join(ref, on=pair)
    out["skill"] = 1 - out["rmse"] / out["rmse_naive"]
    return out


def summarise(results: pd.DataFrame, thresholds: dict[str, float], min_skill: float,
              keys: tuple[str, ...] = ("sensor_id", "location", "variable", "gap_len", "method"),
              ) -> pd.DataFrame:
    """Average the per-seed results and apply the acceptance criteria.

    A method *passes* a (sensor, variable, gap length) case if its mean RMSE is
    within ``thresholds[variable]`` **and** its mean skill is at least
    ``min_skill``.

    Args:
        results: Output of :func:`add_skill`.
        thresholds: Maximum acceptable RMSE per variable (variable units).
        min_skill: Minimum acceptable skill score.
        keys: Grouping columns.

    Returns:
        One row per group with ``rmse_mean``, ``rmse_std``, ``skill``,
        ``max_abs_mean``, ``bias_mean``, ``overshoot_frac``, ``runtime_ms``,
        ``n_gaps``, ``gap_hours``, ``threshold`` and ``passes``.
    """
    summary = results.groupby(list(keys)).agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"), skill=("skill", "mean"),
        max_abs_mean=("max_abs", "mean"), bias_mean=("bias", "mean"),
        overshoot_frac=("overshoot_frac", "mean"),
        runtime_ms=("runtime_s", lambda s: 1000 * s.mean()),
        n_gaps=("n_gaps", "mean"),
    ).reset_index()
    summary["gap_hours"] = summary["gap_len"] * 10 / 60
    summary["threshold"] = summary["variable"].map(thresholds)
    summary["passes"] = (summary["rmse_mean"] <= summary["threshold"]) & (summary["skill"] >= min_skill)
    return summary
