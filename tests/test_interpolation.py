"""Verification of the interpolation methods against known (manufactured) solutions."""

import numpy as np
import pandas as pd
import pytest

from sensorgap.interpolation import (
    INTERPOLATION_METHODS,
    fill_cubic,
    fill_linear,
    fill_mean,
    fill_pchip,
)
from sensorgap.validation import score

ALL = list(INTERPOLATION_METHODS.items())


def series_from(f, n=200, step_min=10):
    """Regular 10-min series sampling f(t), t in hours."""
    idx = pd.date_range("2024-03-01", periods=n, freq=f"{step_min}min", tz="Australia/Melbourne")
    t = np.arange(n) * step_min / 60
    return pd.Series(f(t), index=idx), t


@pytest.mark.parametrize("name,fill", ALL)
def test_known_points_unchanged(name, fill):
    s, _ = series_from(np.sin)
    masked = s.copy()
    masked.iloc[50:60] = np.nan
    out = fill(masked)
    known = masked.notna()
    assert np.allclose(out[known], s[known], atol=1e-12)  # exact interpolant
    assert out.notna().all()


@pytest.mark.parametrize("name,fill", ALL)
def test_reproduces_straight_line_exactly(name, fill):
    # Every method should reproduce a linear function (a manufactured solution).
    s, _ = series_from(lambda t: 3.0 * t + 20.0)
    masked = s.copy()
    masked.iloc[[10, 11, 12, 80, 150]] = np.nan
    assert np.allclose(fill(masked), s, atol=1e-9)


def test_cubic_reproduces_cubic_polynomial():
    # A not-a-knot cubic spline is exact for any cubic polynomial.
    s, _ = series_from(lambda t: 0.01 * t**3 - 0.2 * t**2 + t + 5)
    masked = s.copy()
    masked.iloc[40:70] = np.nan
    assert np.allclose(fill_cubic(masked), s, atol=1e-8)


@pytest.mark.parametrize("fill,expected_order", [(fill_linear, 2), (fill_cubic, 4)])
def test_convergence_order(fill, expected_order):
    """Halving the sample spacing h should cut the error by 2**p (p = order)."""
    errors = []
    for step in (20, 10):  # minutes; t in hours, f = sin(t)
        s, _ = series_from(np.sin, n=int(48 * 60 / step), step_min=step)
        masked = s.copy()
        masked.iloc[1:-1:2] = np.nan  # hide every other sample
        mid = slice(len(s) // 4, 3 * len(s) // 4)  # avoid end effects
        errors.append(np.max(np.abs(fill(masked) - s).iloc[mid]))
    observed_order = np.log2(errors[0] / errors[1])
    assert observed_order == pytest.approx(expected_order, abs=0.3)


def test_pchip_and_linear_never_overshoot_but_cubic_does():
    # A step change (e.g. CO2 when a room suddenly fills) at sample 90, then a
    # dropout just after it. The steep slope at the step makes the C2 cubic
    # spline "ring" above 800 ppm inside the gap; PCHIP and linear cannot.
    s, _ = series_from(lambda t: np.where(t < 15, 450.0, 800.0))
    masked = s.copy()
    masked.iloc[91:93] = np.nan
    hidden = masked.isna().to_numpy()
    assert score(s, fill_linear(masked), masked, hidden)["overshoot_frac"] == 0
    assert score(s, fill_pchip(masked), masked, hidden)["overshoot_frac"] == 0
    assert score(s, fill_cubic(masked), masked, hidden)["overshoot_frac"] > 0
    assert fill_cubic(masked).iloc[91:93].max() > 800


@pytest.mark.parametrize("name,fill", ALL)
def test_leading_and_trailing_nan_not_extrapolated(name, fill):
    s, _ = series_from(np.sin, n=50)
    masked = s.copy()
    masked.iloc[:3] = np.nan
    masked.iloc[-2:] = np.nan
    out = fill(masked)
    assert out.iloc[:3].isna().all() and out.iloc[-2:].isna().all()


@pytest.mark.parametrize("name,fill", ALL)
def test_input_validation(name, fill):
    with pytest.raises(TypeError):
        fill(pd.Series([1.0, np.nan, 3.0]))  # no DatetimeIndex
    s, _ = series_from(np.sin, n=5)
    with pytest.raises(ValueError):
        fill(pd.Series([1.0, np.nan, np.nan, np.nan, np.nan], index=s.index))


def test_fill_mean_uses_known_mean():
    s, _ = series_from(lambda t: t, n=10)  # values 0, 1/6, ..., 1.5 hours
    masked = s.copy()
    masked.iloc[4] = np.nan
    out = fill_mean(masked)
    assert out.iloc[4] == pytest.approx(masked.mean())


@pytest.mark.parametrize("name,fill", ALL)
def test_input_not_modified(name, fill):
    s, _ = series_from(np.sin, n=50)
    s.iloc[20] = np.nan
    before = s.copy()
    fill(s)
    pd.testing.assert_series_equal(s, before)
