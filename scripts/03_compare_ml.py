"""Compare ML gap fillers with the interpolation baselines on held-out data.

Usage (from the repository root, after scripts/01_build_clean_dataset.py):
    python scripts/03_compare_ml.py

Protocol
--------
1. Each sensor record is split chronologically: first 70 % = training,
   last 30 % = test. (A random split would leak: the hours either side of a
   test gap would be in the training set.)
2. ML models are trained only on artificial gaps in the training period, with
   hyperparameters chosen by blocked 5-fold cross-validation.
3. Cross-validation also decides, per gap length, whether the ML correction
   beats plain linear interpolation; on shorter gaps the model falls back to
   linear ("gating"). An ungated random forest is kept for comparison.
4. All methods - naive mean, interpolation and ML - are then scored on the
   *identical* artificial gaps placed only in the test period.

Outputs:
    results/ml_comparison.csv          one row per (sensor, variable, seed, gap, method)
    results/ml_comparison_summary.csv  mean over seeds, skill, pass/fail
    results/ml_model_selection.csv     chosen hyperparameters, CV RMSE, fit time
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from sensorgap.interpolation import INTERPOLATION_METHODS, NAIVE_BASELINE
from sensorgap.ml import MODEL_GRIDS, MLGapFiller
from sensorgap.validation import add_skill, run_gap_experiment, summarise

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"

SENSORS = {"6012002000869": "G.38 (conditioned)", "6012002000326": "unmapped (unconditioned)"}
VARIABLES = ["temperature_C", "co2_ppm"]
GAP_LENGTHS = [1, 6, 18, 36, 144]          # 10 min, 1 h, 3 h, 6 h, 24 h
ACCEPTANCE_RMSE = {"temperature_C": 0.5, "co2_ppm": 50.0}
MIN_SKILL = 0.5
TRAIN_FRACTION = 0.7
N_GAPS = 30
SEEDS = (0, 1, 2, 3, 4)


def main() -> None:
    frames, selection = [], []
    for sid, label in SENSORS.items():
        grid = pd.read_parquet(PROC / f"grid_{sid}.parquet")
        split = int(TRAIN_FRACTION * len(grid))
        test_region = np.arange(len(grid)) >= split
        print(f"\n{sid}: train {grid.index[0]:%Y-%m-%d} to {grid.index[split - 1]:%Y-%m-%d}, "
              f"test {grid.index[split]:%Y-%m-%d} to {grid.index[-1]:%Y-%m-%d}")

        for var in VARIABLES:
            series = grid[var]
            methods = {**NAIVE_BASELINE, **INTERPOLATION_METHODS}
            for name in MODEL_GRIDS:
                methods[name] = MLGapFiller(name).fit(series.iloc[:split], GAP_LENGTHS)
            # Same fitted forest with gating switched off (no refit needed).
            ungated = copy.copy(methods["random_forest"])
            ungated.gated, ungated.min_gap_len_ = False, 0
            methods["random_forest_ungated"] = ungated
            for label_m in [*MODEL_GRIDS, "random_forest_ungated"]:
                filler = methods[label_m]
                selection.append({"sensor_id": sid, "variable": var, "model": label_m,
                                  "best_params": filler.best_params_, "cv_rmse": filler.cv_rmse_,
                                  "ml_from_gap_len": filler.min_gap_len_,
                                  "fit_time_s": filler.fit_time_s_, "n_train": filler.n_train_})
                print(f"  {var:14s} {label_m:22s} CV RMSE {filler.cv_rmse_:.3f}  "
                      f"ML used for gaps >= {filler.min_gap_len_} samples  "
                      f"fit {filler.fit_time_s_:.1f}s  {filler.best_params_}")

            res = run_gap_experiment(series, methods, GAP_LENGTHS, N_GAPS, SEEDS, allowed=test_region)
            frames.append(res.assign(sensor_id=sid, location=label, variable=var))

    results = add_skill(pd.concat(frames, ignore_index=True), naive="mean")
    results.to_csv(RES / "ml_comparison.csv", index=False)
    pd.DataFrame(selection).to_csv(RES / "ml_model_selection.csv", index=False)

    summary = summarise(results, ACCEPTANCE_RMSE, MIN_SKILL)
    summary.to_csv(RES / "ml_comparison_summary.csv", index=False)

    pd.set_option("display.width", 200)
    cols = ["mean", "linear", "pchip", "ridge", "knn", "random_forest", "random_forest_ungated"]
    for (sid, var), g in summary.groupby(["sensor_id", "variable"]):
        rmse = g.pivot(index="gap_hours", columns="method", values="rmse_mean")[cols].round(3)
        skill = g.pivot(index="gap_hours", columns="method", values="skill")[cols[1:]].round(2)
        print(f"\n{sid} ({SENSORS[sid]})  {var}   test period only")
        print("RMSE:\n", rmse, "\nskill:\n", skill)


if __name__ == "__main__":
    main()
