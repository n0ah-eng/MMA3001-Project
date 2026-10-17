"""Performance analysis: runtime scaling, memory, FLOP estimates and an RF trade-off.

Usage (from the repository root, after scripts/01_build_clean_dataset.py):
    python scripts/04_profile_performance.py

What it measures
----------------
1. **Runtime scaling** - median wall-clock time of each fill method on synthetic
   series of n = 1e3 ... 1e5 samples (10 % missing). The log-log slope gives the
   empirical complexity, compared with theory (linear and spline ~ O(n)).
2. **Peak memory** of each method at n = 1e5 (tracemalloc).
3. **FLOP / arithmetic-intensity estimates** for linear and cubic-spline filling,
   to explain *why* they run at the speed they do (memory-bound, not compute-bound).
4. **Profiler breakdown** (cProfile) of fitting the random-forest gap filler -
   where the time actually goes.
5. **Optimisation trade-off**: cross-validated RMSE and fit time of the random
   forest versus number of trees, to justify ``n_estimators``.
6. **Optimisation applied**: random-forest prediction with parallel workers
   (original) versus serial (final), across series lengths.

Outputs:
    results/perf_scaling.csv, results/perf_memory.csv, results/perf_flops.csv,
    results/perf_rf_trees.csv, results/perf_rf_parallel.csv, results/perf_profile_rf.txt
"""

import cProfile
import io
import pstats
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_score

from sensorgap.interpolation import INTERPOLATION_METHODS
from sensorgap.ml import FEATURES, MLGapFiller, make_training_set

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"
GAP_LENGTHS = [1, 6, 18, 36, 144]
SIZES = [1_000, 3_000, 10_000, 30_000, 100_000]
REPEATS = 5


def synthetic(n: int, seed: int = 0) -> pd.Series:
    """Temperature-like series with ~10 % single-sample dropouts (like the real data)."""
    idx = pd.date_range("2024-03-01", periods=n, freq="10min", tz="Australia/Melbourne")
    t = np.arange(n) / 6
    rng = np.random.default_rng(seed)
    s = pd.Series(22 + 2 * np.sin(2 * np.pi * t / 24) + 0.1 * rng.standard_normal(n), index=idx)
    s.iloc[rng.choice(np.arange(1, n - 1), size=n // 10, replace=False)] = np.nan
    return s


def median_time(fn, arg, repeats=REPEATS) -> float:
    """Median wall-clock seconds of ``fn(arg)`` over ``repeats`` runs (after one warm-up)."""
    fn(arg)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(arg)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main() -> None:
    real = pd.read_parquet(PROC / "grid_6012002000869.parquet")["temperature_C"]
    split = int(0.7 * len(real))
    print("Fitting random forest filler on G.38 temperature (training period)...")
    rf = MLGapFiller("random_forest").fit(real.iloc[:split], GAP_LENGTHS)
    rf.min_gap_len_ = 0  # time the model on every gap, not just long ones
    methods = {**INTERPOLATION_METHODS, "random_forest": rf}

    # 1. Runtime scaling --------------------------------------------------------
    rows = []
    for n in SIZES:
        s = synthetic(n)
        for name, fill in methods.items():
            rows.append({"method": name, "n": n, "time_s": median_time(fill, s)})
    scaling = pd.DataFrame(rows)
    # Fit the slope on n >= 1e4 only: below that, fixed Python/pandas overheads
    # dominate and hide the algorithm's true growth rate.
    big = scaling[scaling.n >= 10_000]
    slopes = {m: np.polyfit(np.log10(g.n), np.log10(g.time_s), 1)[0]
              for m, g in big.groupby("method")}
    scaling["loglog_slope"] = scaling["method"].map(slopes)
    scaling.to_csv(RES / "perf_scaling.csv", index=False)
    print("\nEmpirical complexity (log-log slope of time vs n for n >= 1e4; 1.0 = O(n)):")
    for m, k in slopes.items():
        t = scaling[(scaling.method == m) & (scaling.n == SIZES[-1])].time_s.iloc[0]
        print(f"  {m:14s} slope {k:4.2f}   time at n=1e5: {1000 * t:8.2f} ms")

    # 2. Peak memory --------------------------------------------------------------
    s = synthetic(SIZES[-1])
    mem = []
    for name, fill in methods.items():
        tracemalloc.start()
        fill(s)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem.append({"method": name, "n": SIZES[-1], "peak_MB": peak / 1e6})
    mem = pd.DataFrame(mem)
    mem.to_csv(RES / "perf_memory.csv", index=False)
    print("\nPeak memory at n=1e5 (series itself = 0.8 MB of float64):")
    print(mem.round(2).to_string(index=False))

    # 3. FLOP and arithmetic-intensity estimates ----------------------------------
    # m known points, q query points; float64 = 8 bytes.
    n = SIZES[-1]
    q = n // 10
    m = n - q
    flops = pd.DataFrame([
        # Linear: per query, binary search (log2 m comparisons, not FLOPs) then
        # y0 + (y1 - y0) * (x - x0) / (x1 - x0): 2 sub + 1 div + 1 mul + 1 add = 5 FLOPs.
        {"method": "linear", "flops": 5 * q,
         "bytes": 8 * (4 * q + q),  # read x0,x1,y0,y1 per query; write result
         "note": "5 FLOPs per query + binary search"},
        # Cubic spline: build tridiagonal system (~6 FLOPs/knot), Thomas-style
        # banded solve (~8 FLOPs/knot), coefficients (~6 FLOPs/knot), then Horner
        # evaluation of a cubic (3 mul + 3 add) per query.
        {"method": "cubic", "flops": 20 * m + 6 * q,
         "bytes": 8 * (8 * m + 5 * q),  # ~8 arrays of length m touched in the solve
         "note": "O(n) tridiagonal solve + 6 FLOPs/query"},
    ])
    flops["arithmetic_intensity"] = flops["flops"] / flops["bytes"]
    flops.to_csv(RES / "perf_flops.csv", index=False)
    print("\nFLOP estimates at n=1e5:")
    print(flops.round(3).to_string(index=False))
    print("  Arithmetic intensity << 1 FLOP/byte -> memory-bound: speed is limited by "
          "moving data, not by arithmetic.")

    # 4. Profile the random-forest fit ----------------------------------------------
    prof = cProfile.Profile()
    prof.enable()
    MLGapFiller("random_forest").fit(real.iloc[:split], GAP_LENGTHS)
    prof.disable()
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(15)
    (RES / "perf_profile_rf.txt").write_text(buf.getvalue())
    print("\ncProfile of random-forest fit saved to results/perf_profile_rf.txt")

    # 5. Trade-off: number of trees vs accuracy and fit time -------------------------
    X, r, groups, _ = make_training_set(real.iloc[:split], GAP_LENGTHS)
    trees = []
    for n_est in [10, 25, 50, 100, 200]:
        model = RandomForestRegressor(n_estimators=n_est, min_samples_leaf=50,
                                      random_state=0, n_jobs=-1)
        t0 = time.perf_counter()
        scores = cross_val_score(model, X[FEATURES], r, groups=groups, cv=GroupKFold(5),
                                 scoring="neg_root_mean_squared_error")
        trees.append({"n_estimators": n_est, "cv_rmse": -scores.mean(),
                      "cv_time_s": time.perf_counter() - t0})
    trees = pd.DataFrame(trees)
    trees.to_csv(RES / "perf_rf_trees.csv", index=False)
    print("\nRandom forest: trees vs cross-validated RMSE and cost:")
    print(trees.round(4).to_string(index=False))

    # 6. Parallel (original) vs serial (final) prediction ----------------------------
    par = []
    for n in [1_000, 10_000, 100_000]:
        s = synthetic(n)
        rf.model_.set_params(n_jobs=-1)
        t_par = median_time(rf, s)
        rf.model_.set_params(n_jobs=1)
        t_ser = median_time(rf, s)
        par.append({"n": n, "parallel_ms": 1000 * t_par, "serial_ms": 1000 * t_ser,
                    "speedup": t_par / t_ser})
    par = pd.DataFrame(par)
    par.to_csv(RES / "perf_rf_parallel.csv", index=False)
    print("\nRandom-forest prediction, parallel workers vs serial:")
    print(par.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
