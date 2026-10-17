"""Make the report figures from the saved results.

Usage (from the repository root, after scripts 01-04):
    python scripts/05_make_figures.py

Outputs (results/figures/):
    fig1_example_gap.png       a 24 h temperature gap: truth vs linear vs random forest
    fig2_rmse_vs_gap.png       RMSE vs gap length, both sensors and variables (test period)
    fig3_runtime_scaling.png   fill time vs series length (log-log)

Colour: a fixed categorical order validated for colour-vision deficiency
(all pairs) - linear = blue, cubic spline = orange, random forest = aqua; the
naive mean reference is grey and dashed. Every series also has a distinct
marker and a legend, so identity never relies on colour alone.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sensorgap.interpolation import fill_linear
from sensorgap.ml import MLGapFiller
from sensorgap.validation import insert_gaps, sample_gap_starts

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"
FIG = RES / "figures"

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STYLE = {  # method -> (label, colour, marker, linestyle)
    "mean": ("Mean fill (naive)", "#9a9994", "", "--"),
    "linear": ("Linear", "#2a78d6", "o", "-"),
    "cubic": ("Cubic spline", "#eb6834", "s", "-"),
    "random_forest": ("Random forest (gated)", "#1baf7a", "^", "-"),
}
UNITS = {"temperature_C": "°C", "co2_ppm": "ppm"}
NAMES = {"temperature_C": "Temperature", "co2_ppm": "CO₂"}
SENSOR_NAMES = {"6012002000869": "G.38 (conditioned)", "6012002000326": "Unmapped (unconditioned)"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def fig_example_gap() -> None:
    """24 h gap in the G.38 test period: where linear interpolation misses the daily cycle."""
    s = pd.read_parquet(PROC / "grid_6012002000869.parquet")["temperature_C"]
    split = int(0.7 * len(s))
    allowed = np.arange(len(s)) >= split
    start = sample_gap_starts(s.notna().to_numpy(), 144, 1, np.random.default_rng(7), allowed=allowed)[0]
    masked, _ = insert_gaps(s, np.array([start]), 144)
    rf = MLGapFiller("random_forest").fit(s.iloc[:split], [1, 6, 18, 36, 144])

    win = slice(start - 144, start + 288)
    t = s.index[win]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axvspan(s.index[start], s.index[start + 143], color=GRID, alpha=0.6, lw=0)
    top = s.iloc[win].max() + 0.6
    ax.set_ylim(s.iloc[win].min() - 0.3, top)
    ax.text(s.index[start + 72], top - 0.1, "hidden 24 h gap", ha="center", va="top", color=INK_2)
    ax.plot(t, s.iloc[win], color=INK, lw=1.0, label="Measured (truth)")
    for key, filled in [("linear", fill_linear(masked)), ("random_forest", rf(masked))]:
        label, colour, _, ls = STYLE[key]
        g = slice(start - 1, start + 145)
        ax.plot(s.index[g], filled.iloc[g], color=colour, lw=2, ls=ls, label=label)
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Example reconstruction: G.38 temperature, test period", loc="left", color=INK)
    fig.autofmt_xdate()
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.32), ncols=3)
    fig.savefig(FIG / "fig1_example_gap.png")
    plt.close(fig)


def fig_rmse_vs_gap() -> None:
    """2x2 small multiples: RMSE vs gap length for each sensor and variable."""
    summ = pd.read_csv(RES / "ml_comparison_summary.csv", dtype={"sensor_id": str})
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5), sharex=True)
    for row, var in enumerate(["temperature_C", "co2_ppm"]):
        for col, sid in enumerate(SENSOR_NAMES):
            ax = axes[row, col]
            g = summ[(summ.sensor_id == sid) & (summ.variable == var)]
            for key, (label, colour, marker, ls) in STYLE.items():
                d = g[g.method == key].sort_values("gap_hours")
                ax.plot(d.gap_hours, d.rmse_mean, color=colour, marker=marker, ms=5, ls=ls,
                        lw=2 if key != "mean" else 1.2, label=label,
                        markeredgecolor="white", markeredgewidth=1)
            thr = g.threshold.iloc[0]
            ax.axhline(thr, color=INK_2, lw=0.8)
            ax.text(0.17, thr * 1.08, f"acceptance {thr:g} {UNITS[var]}", color=INK_2, fontsize=8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks([1 / 6, 1, 3, 6, 24], ["10 min", "1 h", "3 h", "6 h", "24 h"])
            ax.minorticks_off()
            ax.set_title(f"{NAMES[var]} · {SENSOR_NAMES[sid]}", loc="left", fontsize=9, color=INK)
            if col == 0:
                ax.set_ylabel(f"RMSE ({UNITS[var]})")
            if row == 1:
                ax.set_xlabel("Gap length")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    fig.savefig(FIG / "fig2_rmse_vs_gap.png")
    plt.close(fig)


def fig_runtime_scaling() -> None:
    """Log-log runtime vs series length, with an O(n) reference slope."""
    sc = pd.read_csv(RES / "perf_scaling.csv")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for key in ["linear", "cubic", "random_forest"]:
        label, colour, marker, _ = STYLE[key]
        d = sc[sc.method == key]
        ax.plot(d.n, 1000 * d.time_s, color=colour, marker=marker, ms=5, lw=2, label=label,
                markeredgecolor="white", markeredgewidth=1)
    n = np.array([1e4, 1e5])
    ref = sc[(sc.method == "cubic") & (sc.n == 10_000)].time_s.iloc[0] * 1000
    ax.plot(n, ref * n / 1e4, color=INK_2, lw=0.8, ls=":", label="O(n) reference")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Series length n (samples)")
    ax.set_ylabel("Fill time (ms, median of 5)")
    ax.set_title("Runtime scaling of gap-filling methods", loc="left", color=INK)
    ax.legend()
    fig.savefig(FIG / "fig3_runtime_scaling.png")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig_example_gap()
    fig_rmse_vs_gap()
    fig_runtime_scaling()
    print(f"Figures written to {FIG}")


if __name__ == "__main__":
    main()
