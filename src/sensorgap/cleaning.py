"""Auditing and cleaning the environmental sensor table.

Three data-quality problems were found in the raw export (see README):

1. **Stale re-uploads** - after mid-June 2024 the gateways repeatedly re-sent
   old readings (one reading appears 29,342 times). These are duplicates in
   ``(sensor_id, time)`` and must be removed before any analysis.
2. **Irregular coverage** - only two sensors have substantial data, with long
   outages between deployments.
3. **Physically implausible spikes** - e.g. VOC readings of 8,051 ppb.

The functions here fix (1) and (3) and expose (2) so that the gap
reconstruction methods operate on a regular, honest time grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Physically plausible ranges for indoor measurements (inclusive).
#: Values outside are treated as sensor faults and set to NaN.
#: Sources: sensor operating ranges and indoor-environment norms
#: (CO2 cannot fall far below the outdoor ~400 ppm background).
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "temperature_C": (0.0, 45.0),
    "humidity_RH": (0.0, 100.0),
    "co2_ppm": (350.0, 5000.0),
    "pressure_mb": (950.0, 1050.0),
}


def drop_stale_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove repeated uploads of the same reading.

    A reading is identified by ``(sensor_id, time)`` where ``time`` is the
    sensor's own measurement timestamp. The earliest upload is kept.

    Args:
        df: Wide table from :func:`sensorgap.io.load_env_csv`.

    Returns:
        A new DataFrame without duplicates, sorted by sensor then time. The
        number of rows removed is stored in ``out.attrs["n_duplicates"]``.
    """
    sort_cols = ["sensor_id", "time"] + (["upload_time"] if "upload_time" in df else [])
    out = df.sort_values(sort_cols).drop_duplicates(["sensor_id", "time"], keep="first")
    out = out.reset_index(drop=True)
    out.attrs["n_duplicates"] = len(df) - len(out)
    return out


def apply_physical_bounds(
    df: pd.DataFrame, bounds: dict[str, tuple[float, float]] | None = None
) -> pd.DataFrame:
    """Replace physically implausible values with NaN.

    Args:
        df: Table containing some or all of the columns in ``bounds``.
        bounds: ``{column: (low, high)}``; defaults to :data:`PHYSICAL_BOUNDS`.

    Returns:
        A copy of ``df`` with out-of-range values set to NaN. Counts of values
        removed per column are stored in ``out.attrs["n_out_of_bounds"]``.
    """
    bounds = PHYSICAL_BOUNDS if bounds is None else bounds
    out = df.copy()
    removed = {}
    for col, (lo, hi) in bounds.items():
        if col not in out:
            continue
        bad = (out[col] < lo) | (out[col] > hi)
        removed[col] = int(bad.sum())
        out.loc[bad, col] = np.nan
    out.attrs["n_out_of_bounds"] = removed
    return out


def audit_sensors(df: pd.DataFrame, locations: pd.DataFrame | None = None) -> pd.DataFrame:
    """Summarise coverage and sampling for each sensor.

    Args:
        df: De-duplicated wide table.
        locations: Optional lookup from :func:`sensorgap.io.load_locations`.

    Returns:
        One row per sensor with ``n_readings``, ``start``, ``end``,
        ``median_interval_min`` and ``longest_gap_days`` (plus location
        columns if ``locations`` is given), sorted by ``n_readings``.
    """
    g = df.sort_values("time").groupby("sensor_id")["time"]
    summary = pd.DataFrame(
        {
            "n_readings": g.size(),
            "start": g.min(),
            "end": g.max(),
            "median_interval_min": g.apply(lambda t: t.diff().dt.total_seconds().median() / 60),
            "longest_gap_days": g.apply(lambda t: t.diff().dt.total_seconds().max() / 86400),
        }
    )
    if locations is not None:
        summary = summary.join(locations, how="left")
    return summary.sort_values("n_readings", ascending=False)


def to_regular_grid(
    df: pd.DataFrame,
    sensor_id: str,
    columns: list[str],
    freq: str = "10min",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Resample one sensor's readings onto a regular time grid.

    Readings are averaged within each bin; bins with no reading become NaN.
    A regular grid is required by the interpolation and ML methods, and makes
    missing data explicit instead of silently absent.

    Args:
        df: De-duplicated wide table.
        sensor_id: Sensor to extract.
        columns: Variables to keep, e.g. ``["temperature_C", "co2_ppm"]``.
        freq: Grid spacing (pandas offset alias). The sensors report every
            10 minutes, so ``"10min"`` preserves the native resolution.
        start, end: Optional window limits (inclusive). Defaults to the
            sensor's first and last reading.

    Returns:
        DataFrame indexed by a tz-aware ``DatetimeIndex`` at spacing ``freq``.

    Raises:
        KeyError: If ``sensor_id`` is not present in ``df``.
    """
    sub = df[df["sensor_id"] == sensor_id]
    if sub.empty:
        raise KeyError(f"Sensor {sensor_id!r} not found")
    series = sub.set_index("time")[columns].sort_index()
    tz = series.index.tz
    if start is not None:
        series = series[series.index >= _localise(start, tz)]
    if end is not None:
        series = series[series.index <= _localise(end, tz)]
    # Bin in UTC: local time repeats an hour when daylight saving ends
    # (e.g. 2024-04-07 02:00-03:00 in Melbourne), which makes local bins ambiguous.
    grid = series.tz_convert("UTC").resample(freq).mean()
    return grid.tz_convert(tz)


def _localise(ts: str | pd.Timestamp, tz) -> pd.Timestamp:
    """Return ``ts`` as a Timestamp in ``tz`` (naive inputs are assumed local)."""
    ts = pd.Timestamp(ts)
    return ts.tz_localize(tz) if ts.tz is None else ts.tz_convert(tz)


def find_gaps(series: pd.Series) -> pd.DataFrame:
    """Locate contiguous runs of missing values in a regularly spaced series.

    Args:
        series: Regularly spaced series (e.g. one column of
            :func:`to_regular_grid`) where NaN marks a missing sample.

    Returns:
        DataFrame with one row per gap: ``start`` and ``end`` (timestamps of the
        first and last missing sample) and ``n_missing`` (number of samples).
        Empty if there are no gaps.
    """
    missing = series.isna().to_numpy()
    if not missing.any():
        return pd.DataFrame(columns=["start", "end", "n_missing"])
    # Pad with False so every run has a rising and falling edge.
    edges = np.diff(np.concatenate([[False], missing, [False]]).astype(int))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return pd.DataFrame(
        {"start": series.index[starts], "end": series.index[ends], "n_missing": ends - starts + 1}
    )
