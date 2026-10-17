"""Validate the interpolation baselines with artificial gaps on the real sensors.

Usage (from the repository root, after scripts/01_build_clean_dataset.py):
    python scripts/02_validate_interpolation.py

Outputs:
    results/interpolation_validation.csv  one row per (sensor, variable, seed, gap, method)
    results/interpolation_summary.csv     mean +/- std over seeds, skill and pass/fail

Skill score (relative to the naive mean-fill baseline, paired by seed and gap):

    skill = 1 - RMSE_method / RMSE_mean_fill

0 = no better than filling with the average; 1 = perfect.
"""

from pathlib import Path

import pandas as pd

from sensorgap.interpolation import INTERPOLATION_METHODS, NAIVE_BASELINE
from sensorgap.validation import add_skill, run_gap_experiment, summarise

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"

SENSORS = {"6012002000869": "G.38 (conditioned)", "6012002000326": "unmapped (unconditioned)"}
VARIABLES = ["temperature_C", "co2_ppm"]

#: Gap lengths in 10-minute samples: 10 min, 1 h, 3 h, 6 h, 24 h.
GAP_LENGTHS = [1, 6, 18, 36, 144]

#: Acceptance criteria on RMSE, set at roughly typical indoor-sensor accuracy:
#: a reconstruction is "acceptable" if its error is no worse than the
#: uncertainty of a real reading. (Exact sensor model unidentified - see README.)
ACCEPTANCE_RMSE = {"temperature_C": 0.5, "co2_ppm": 50.0}

#: ...AND it must remove at least half the error of the naive mean fill.
#: Needed because these rooms vary so little in CO2 that mean-filling alone
#: already scores ~25-40 ppm, i.e. "passes" the absolute threshold.
MIN_SKILL = 0.5

N_GAPS = 30
SEEDS = (0, 1, 2, 3, 4)


def main() -> None:
    frames = []
    for sid, label in SENSORS.items():
        grid = pd.read_parquet(PROC / f"grid_{sid}.parquet")
        for var in VARIABLES:
            print(f"{sid} {var} ...")
            methods = {**NAIVE_BASELINE, **INTERPOLATION_METHODS}
            res = run_gap_experiment(grid[var], methods, GAP_LENGTHS, N_GAPS, SEEDS)
            frames.append(res.assign(sensor_id=sid, location=label, variable=var))
    results = pd.concat(frames, ignore_index=True)

    # Paired skill: compare each method with mean-fill on the *same* gaps.
    results = add_skill(results, naive="mean")
    results.to_csv(RES / "interpolation_validation.csv", index=False)

    summary = summarise(results, ACCEPTANCE_RMSE, MIN_SKILL)
    summary.to_csv(RES / "interpolation_summary.csv", index=False)

    pd.set_option("display.width", 200)
    for (sid, var), g in summary.groupby(["sensor_id", "variable"]):
        table = g.pivot(index="gap_hours", columns="method", values="rmse_mean").round(3)
        skill = g.pivot(index="gap_hours", columns="method", values="skill").round(2)
        print(f"\n{sid} ({SENSORS[sid]})  {var}  [RMSE <= {ACCEPTANCE_RMSE[var]}, skill >= {MIN_SKILL}]")
        print("RMSE:\n", table, "\nskill:\n", skill.drop(columns="mean"))


if __name__ == "__main__":
    main()
