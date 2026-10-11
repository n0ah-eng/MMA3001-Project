"""Univariate interpolation methods for filling gaps in a regular time series.

Every method has the same signature::

    fill(series: pd.Series) -> pd.Series

It receives a regularly spaced series with ``NaN`` marking missing samples
and returns a copy in which every *interior* ``NaN`` has been replaced.
Leading and trailing ``NaN`` are left untouched: filling them would be
extrapolation, which none of these methods support reliably.

Methods (all exact interpolants, i.e. they pass through every known point):

========  =================================================================
linear    Straight line between the two neighbouring known samples.
          C0 continuous, never overshoots, cheapest. The baseline.
cubic     Cubic spline (not-a-knot end conditions). C2 smooth. Built by
          solving a tridiagonal linear system for the knot curvatures, so
          cost is O(n). Can overshoot between knots.
pchip     Piecewise Cubic Hermite Interpolating Polynomial. C1 smooth and
          shape-preserving: on each interval the curve stays between the
          two end values, so it never overshoots.
akima     Akima spline. C1, slopes from a weighted local average, so it is
          less prone to wiggles than ``cubic`` but can still overshoot.
========  =================================================================
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator

#: Type of every gap-filling function in this package.
FillFunction = Callable[[pd.Series], pd.Series]


def _split_known(series: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x_known, y_known, x_query, query_positions)`` for interior gaps.

    ``x`` is elapsed time in hours from the first sample. Using physical time
    (rather than sample index) keeps derivatives in meaningful units
    (e.g. °C per hour) and stays correct if the grid spacing changes.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must have a DatetimeIndex")
    y = series.to_numpy(dtype=float)
    x = (series.index - series.index[0]).total_seconds().to_numpy() / 3600.0
    known = ~np.isnan(y)
    if known.sum() < 2:
        raise ValueError("At least two known samples are required to interpolate")

    first, last = np.flatnonzero(known)[[0, -1]]
    interior_missing = ~known
    interior_missing[:first] = False  # leading NaN: would be extrapolation
    interior_missing[last + 1:] = False  # trailing NaN: would be extrapolation
    pos = np.flatnonzero(interior_missing)
    return x[known], y[known], x[pos], pos


def _apply(series: pd.Series, values: np.ndarray, pos: np.ndarray) -> pd.Series:
    """Return a copy of ``series`` with ``values`` written at ``pos``."""
    out = series.copy()
    out.iloc[pos] = values
    return out


def fill_linear(series: pd.Series) -> pd.Series:
    """Fill interior gaps by piecewise-linear interpolation (baseline method).

    Args:
        series: Regularly spaced series with a DatetimeIndex; NaN = missing.

    Returns:
        Copy of ``series`` with interior NaN replaced.

    Raises:
        TypeError: If the index is not a DatetimeIndex.
        ValueError: If fewer than two samples are known.
    """
    xk, yk, xq, pos = _split_known(series)
    return _apply(series, np.interp(xq, xk, yk), pos)


def fill_cubic(series: pd.Series) -> pd.Series:
    """Fill interior gaps with a not-a-knot cubic spline through all known samples.

    "Not-a-knot" is used because no physical end-slope information exists;
    its influence is limited to the ends of the record, which are excluded
    from validation.

    Args / Returns / Raises: as :func:`fill_linear`.
    """
    xk, yk, xq, pos = _split_known(series)
    return _apply(series, CubicSpline(xk, yk, bc_type="not-a-knot")(xq), pos)


def fill_pchip(series: pd.Series) -> pd.Series:
    """Fill interior gaps with a shape-preserving PCHIP interpolant.

    Args / Returns / Raises: as :func:`fill_linear`.
    """
    xk, yk, xq, pos = _split_known(series)
    return _apply(series, PchipInterpolator(xk, yk)(xq), pos)


def fill_akima(series: pd.Series) -> pd.Series:
    """Fill interior gaps with an Akima spline.

    Args / Returns / Raises: as :func:`fill_linear`.
    """
    xk, yk, xq, pos = _split_known(series)
    return _apply(series, Akima1DInterpolator(xk, yk)(xq), pos)


def fill_mean(series: pd.Series) -> pd.Series:
    """Naive reference: fill every interior gap with the mean of all known samples.

    Not a serious method. It is the "no-skill" yardstick that every other
    method must clearly beat; its RMSE is roughly the series' standard
    deviation. If a method cannot beat it, the method adds no value.

    Args / Returns / Raises: as :func:`fill_linear`.
    """
    _, yk, _, pos = _split_known(series)
    return _apply(series, np.full(len(pos), yk.mean()), pos)


#: The naive reference method used to compute skill scores.
NAIVE_BASELINE: dict[str, FillFunction] = {"mean": fill_mean}

#: Registry used by the validation scripts: name -> fill function.
INTERPOLATION_METHODS: dict[str, FillFunction] = {
    "linear": fill_linear,
    "cubic": fill_cubic,
    "pchip": fill_pchip,
    "akima": fill_akima,
}
