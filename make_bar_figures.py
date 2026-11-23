"""
make_bar_figures.py - Grouped bar charts for the paper, drawn from results/summary.csv.

Run after run_experiments.py (no new simulations are needed):
    python make_bar_figures.py

Creates in ./results:
    fig11_bars_success_vs_uavs.png      success rate per method for 3, 4, 5 and 6 UAVs   (experiment E2)
    fig12_bars_accviol_vs_accuracy.png  accuracy violation rate per method for each
                                         lowest required accuracy                         (experiment E4)
Error bars are the 95% confidence intervals over the seeds, as in the line figures.
Colours and method order are the same as in run_experiments.py, so a method keeps its colour
in every figure.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
plt.style.use(["science", "ieee", "no-latex"])

from run_experiments import COLORS, MAIN

OUT = "results"


def grouped_bars(summary, exp, metric, xlabel, ylabel, fname, xfmt="{:g}"):
    data = summary[summary.experiment == exp]
    xs = sorted(data.x.unique())
    n = len(MAIN)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(4.2, 2.4), dpi=600)        # one IEEE column
    for i, m in enumerate(MAIN):
        g = data[data.method == m].set_index("x").reindex(xs)
        pos = np.arange(len(xs)) - 0.4 + width * (i + 0.5)
        ax.bar(pos, g[metric], width * 0.9, label=m,
               color=COLORS[i % len(COLORS)], edgecolor="black", linewidth=0.4)
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels([xfmt.format(x) for x in xs], fontsize=7)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    top = (data[metric] + data[metric + "_ci95"]).max()
    ax.set_ylim(0, top * 1.4)                                   # bars start at zero; headroom for the legend
    ticks = matplotlib.ticker.MaxNLocator(6, steps=[1, 2, 5, 10]).tick_values(0, top)
    ax.set_yticks([t for t in ticks if t <= top * 1.05])        # no tick labels in the legend area
    ax.tick_params(labelsize=7)
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=6, frameon=True, edgecolor="#bbbbbb", fancybox=False, ncol=3, loc="upper left",
              handlelength=1.2, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)


def main():
    summary = pd.read_csv(os.path.join(OUT, "summary.csv"))
    grouped_bars(summary, "uavs", "success_rate", "Number of UAVs", "Task success rate",
                 "fig11_bars_success_vs_uavs.png", xfmt="{:.0f}")
    grouped_bars(summary, "accuracy", "accuracy_violation_rate", "Lowest required accuracy",
                 "Accuracy violation rate", "fig12_bars_accviol_vs_accuracy.png", xfmt="{:.2f}")
    print("Saved fig11_bars_success_vs_uavs.png and fig12_bars_accviol_vs_accuracy.png in", OUT)


if __name__ == "__main__":
    main()