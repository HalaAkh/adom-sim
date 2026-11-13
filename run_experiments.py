"""
run_experiments.py - Runs every experiment of the paper and saves tables and figures.

Five experiments (all with the same tasks for every method, 30 seeds, 95% CIs):
  E1 load      : vary the arrival rate                      (main result)
  E2 uavs      : vary the number of UAVs (3-6)              at medium-high load
  E3 deadline  : vary the deadlines (x0.75, x1, x1.5)        at medium-high load
  E4 accuracy  : vary the required accuracy range            at medium-high load
  E5 margin    : AD-OM only, vary the margin fraction kappa (sensitivity; justifies kappa = 0.15)

Usage:
    python run_experiments.py              # all experiments, 30 seeds
    python run_experiments.py --seeds 5    # quick run
Outputs (in ./results): raw_results.csv, summary.csv, fig_*.png
"""
import argparse
import os
from dataclasses import replace
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

from config import SimConfig
from simulator import Simulator, acc
from policies import all_methods, ADOM

ARRIVAL_RATES = [0.04, 0.06, 0.08, 0.10, 0.12, 0.14]   # tasks/s per user (x30 users)
FIXED_RATE = 0.10                                       # load used by E2-E5

# Each experiment: name -> list of (x value shown on the plot, config changes)
EXPERIMENTS = {
    "load":     [(r * 30, {"arrival_rate": r}) for r in ARRIVAL_RATES],
    "uavs":     [(n, {"arrival_rate": FIXED_RATE, "n_uavs": n}) for n in (3, 4, 5, 6)],
    "deadline": [(f, {"arrival_rate": FIXED_RATE, "deadlines_s": tuple(round(d * f, 3) for d in (2.0, 2.5, 3.0))})
                 for f in (0.75, 1.0, 1.5)],
    "accuracy": [(lo, {"arrival_rate": FIXED_RATE, "acc_req_range": (lo, 0.67)}) for lo in (0.60, 0.62, 0.64)],
    "margin":   [(k, {"arrival_rate": r, "kappa_init": k}) for k in (0.05, 0.15, 0.25, 0.40) for r in (0.06, 0.12)],
}


def metrics(tasks):
    """Per-run metrics (the ones listed in the Plan Forward)."""
    n = max(1, len(tasks))
    done = [k for k in tasks if k.result_time is not None]
    late = [k for k in tasks if k.result_time is None or k.result_time - k.t_arr > k.t_max]
    acc_fail = [k for k in tasks if k.config is not None and acc(k.config) < k.a_req]
    succ = [k for k in tasks if k.success]
    return {
        "tasks": len(tasks),
        "success_rate": len(succ) / n,
        "accuracy_violation_rate": len(acc_fail) / n,
        "deadline_violation_rate": len(late) / n,
        "drop_rate": sum(k.dropped for k in tasks) / n,
        "mean_latency_s": float(np.mean([k.result_time - k.t_arr for k in done])) if done else np.nan,
        "mean_delivered_accuracy": float(np.mean([acc(k.config) for k in succ])) if succ else np.nan,
        "mean_wearable_energy_j": float(np.mean([k.user_energy_j for k in tasks])) if tasks else np.nan,
        "migrations_per_task": float(np.mean([k.n_migrations for k in tasks])) if tasks else 0.0,
        "downgrades_per_task": float(np.mean([k.n_downgrades for k in tasks])) if tasks else 0.0,
    }


def one_run(job):
    exp, x, changes, method_idx, seed = job
    policy = ADOM() if exp == "margin" else all_methods()[method_idx]
    if exp == "margin":                      # report the load inside the method name
        policy.name = f"AD-OM, load {changes['arrival_rate'] * 30:.1f} tasks/s"
    cfg = replace(SimConfig(), **changes)
    tasks = Simulator(cfg, policy, seed).run()
    return {"experiment": exp, "x": x, "method": policy.name, "seed": seed, **metrics(tasks)}


def summarize(df):
    rows = []
    metric_cols = [c for c in df.columns if c not in ("experiment", "x", "method", "seed")]
    for (e, m, x), g in df.groupby(["experiment", "method", "x"], sort=False):
        row = {"experiment": e, "method": m, "x": x}
        for c in metric_cols:
            v = g[c].dropna().values
            row[c] = v.mean() if len(v) else np.nan
            row[c + "_ci95"] = (stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
                                if len(v) > 1 else 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


# Fixed categorical colours (validated palette) + a marker per series,
# so identity never relies on colour alone. A method keeps its colour in every figure.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
MAIN = ["AD-OM", "QTM", "BLTM", "DTM", "AAO", "Nearest UAV"]
# QTM [Zhang 2026], BLTM [Meng 2026], DTM [Wang 2025], AAO [Yan 2025]
ABLATION = ["AD-OM", "AD-OM w/o accuracy check", "AD-OM w/o early trigger", "AD-OM w/o migration"]
XLABEL = {
    "load": "Total arrival rate (tasks/s)",
    "uavs": "Number of UAVs",
    "deadline": "Deadline scale (x of 2-3 s)",
    "accuracy": "Lowest required accuracy",
    "margin": "Margin fraction $\\kappa$",
}


def plot(summary, exp, methods, metric, ylabel, fname, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = summary[summary.experiment == exp]
    if methods is None:
        methods = list(dict.fromkeys(data.method))
    fig, ax = plt.subplots(figsize=(4.2, 3.4), dpi=200)   # one IEEE column
    for i, m in enumerate(methods):
        g = data[data.method == m].sort_values("x")
        if g.empty:
            continue
        ax.errorbar(g.x, g[metric], yerr=g[metric + "_ci95"], label=m, color=COLORS[i % 6],
                    marker=MARKERS[i % 6], markersize=5, linewidth=1.6, capsize=2)
    ax.set_xlabel(XLABEL[exp], fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, color="#dddddd", linewidth=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, fname))
    plt.close(fig)


def draw_all(summary, out):
    # E1: main result versus load
    plot(summary, "load", MAIN, "success_rate", "Task success rate", "fig1_success_vs_load.png", out)
    plot(summary, "load", MAIN, "accuracy_violation_rate", "Accuracy violation rate", "fig2_accviol_vs_load.png", out)
    plot(summary, "load", MAIN, "deadline_violation_rate", "Deadline violation rate", "fig3_deadline_vs_load.png", out)
    plot(summary, "load", MAIN, "migrations_per_task", "Migrations per task", "fig4_migrations_vs_load.png", out)
    plot(summary, "load", ABLATION, "success_rate", "Task success rate", "fig5_ablation_vs_load.png", out)
    # E2-E4: sensitivity
    plot(summary, "uavs", MAIN, "success_rate", "Task success rate", "fig6_success_vs_uavs.png", out)
    plot(summary, "deadline", MAIN, "success_rate", "Task success rate", "fig7_success_vs_deadline.png", out)
    plot(summary, "accuracy", MAIN, "success_rate", "Task success rate", "fig8_success_vs_accuracy.png", out)
    plot(summary, "accuracy", MAIN, "accuracy_violation_rate", "Accuracy violation rate",
         "fig9_accviol_vs_accuracy.png", out)
    # E5: margin sensitivity
    plot(summary, "margin", None, "success_rate", "Task success rate", "fig10_margin_sensitivity.png", out)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--plot-only", action="store_true", help="redraw figures from results/summary.csv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.plot_only:
        draw_all(pd.read_csv(os.path.join(a.out, "summary.csv")), a.out)
        return

    n_methods = len(all_methods())
    jobs = []
    for exp, points in EXPERIMENTS.items():
        for x, changes in points:
            method_ids = [0] if exp == "margin" else range(n_methods)
            for mi in method_ids:
                for s in range(a.seeds):
                    jobs.append((exp, x, changes, mi, s))
    print(f"Running {len(jobs)} simulations on {a.workers} workers ...")
    with Pool(a.workers) as pool:
        rows = pool.map(one_run, jobs, chunksize=4)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "raw_results.csv"), index=False)
    summary = summarize(df)
    summary.to_csv(os.path.join(a.out, "summary.csv"), index=False)

    draw_all(summary, a.out)

    cols = ["experiment", "method", "x", "success_rate", "accuracy_violation_rate",
            "deadline_violation_rate", "migrations_per_task", "downgrades_per_task"]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    print(summary[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
