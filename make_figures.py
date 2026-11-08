"""
make_figures.py - Illustration figures for the paper (in addition to the result figures
produced by run_experiments.py).

    python make_figures.py

Creates in ./results:
    fig_scenario.png    top view of the simulated area: users, UAVs, coverage, hosted configurations
    fig_tradeoff.png    accuracy vs. response time of the four LLaVA-HR configurations
    fig_casestudy.png   one task followed slot by slot under AD-OM and under Wang (deadline-triggered, accuracy-blind)
"""
import math
import os
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SimConfig, PROFILES, CONFIG_ORDER
from simulator import Simulator, acc, link_rate_bps
from policies import ADOM, WangDeadline

OUT = "results"
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#888780"
UAV_COLORS = [ORANGE, BLUE, AQUA]          # UAV 1 small, UAV 2 large, UAV 3 medium
UAV_MARKERS = ["^", "s", "D"]


def style(ax):
    ax.tick_params(labelsize=7)
    ax.grid(True, color="#dddddd", linewidth=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------------------
# Figure A: simulated scenario (like the hazard map of the reference paper)
# ---------------------------------------------------------------------------
def fig_scenario(seed=0):
    cfg = SimConfig()
    sim = Simulator(cfg, ADOM(), seed)          # only used to build the same network as the experiments
    fig, ax = plt.subplots(figsize=(4.2, 4.0), dpi=200)
    r_h = math.sqrt(cfg.g2a_range_m ** 2 - cfg.uav_altitude_m ** 2)   # horizontal coverage radius
    for u in sim.uavs:
        col = UAV_COLORS[u.id % 3]
        ax.add_patch(plt.Circle(u.pos[:2], r_h, color=col, alpha=0.08, linewidth=0))
        ax.add_patch(plt.Circle(u.pos[:2], r_h, fill=False, color=col, linewidth=1.0, linestyle="--"))
    ux = [p.pos[0] for p in sim.users]
    uy = [p.pos[1] for p in sim.users]
    ax.scatter(ux, uy, s=14, color=GRAY, marker="o", label="Wearable user", zorder=3)
    for u in sim.uavs:
        cfgs = ", ".join(u.configs)
        ax.scatter(*u.pos[:2], s=90, color=UAV_COLORS[u.id % 3], marker=UAV_MARKERS[u.id % 3],
                   edgecolor="white", linewidth=1.2, zorder=4,
                   label=f"UAV {u.id + 1}: {cfgs}")
    ax.set_xlim(0, cfg.area_m)
    ax.set_ylim(0, cfg.area_m)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)
    style(ax)
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_scenario.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure B: accuracy vs. response time of each configuration
# ---------------------------------------------------------------------------
def fig_tradeoff(distance_m=200.0):
    cfg = SimConfig()
    rate = link_rate_bps(cfg, cfg.user_tx_w, distance_m)
    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=200)
    ax.axhspan(60, 67, color=YELLOW, alpha=0.15, linewidth=0)
    ax.text(1.45, 60.4, "Required accuracy range of tasks (60-67%)", fontsize=6, color="#633806")
    xs, ys = [], []
    for c in CONFIG_ORDER:
        p = PROFILES[c]
        upload = p["input_mb"] * 8e6 / rate
        infer = cfg.answer_tokens_mean / p["speed"]
        xs.append(upload + infer)
        ys.append(p["accuracy"] * 100)
    ax.plot(xs, ys, color=BLUE, linewidth=1.6, marker="o", markersize=6, zorder=3)
    for c, x, y in zip(CONFIG_ORDER, xs, ys):
        off = {"384p": (6, 4), "768p": (6, -14), "1024p": (-46, 0), "1536p": (6, -14)}[c]
        ax.annotate(f"{c}\n{PROFILES[c]['input_mb']:.2f} MB", (x, y), textcoords="offset points",
                    xytext=off, fontsize=6)
    ax.set_xlabel(f"Upload + inference time of a {cfg.answer_tokens_mean:.0f}-token answer (s)", fontsize=8)
    ax.set_ylabel("TextVQA accuracy (%)", fontsize=8)
    ax.set_xlim(0.7, max(xs) + 0.35)
    ax.set_ylim(58, 70)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_tradeoff.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure C: case study - one task under AD-OM and under Wang
# ---------------------------------------------------------------------------
def find_case(cfg, seeds=range(10)):
    """First task that AD-OM rescues by migrating it to a UAV that is still accurate enough, while
    Wang moves it to a UAV whose model is not accurate enough - and both start from the same state
    (same UAV, configuration and spare time on arrival), so the two curves are directly comparable."""
    for seed in seeds:
        a = {k.id: k for k in Simulator(cfg, ADOM(), seed).run()}
        w = {k.id: k for k in Simulator(cfg, WangDeadline(), seed).run()}
        for tid in sorted(a):
            ka, kw = a[tid], w[tid]
            if not (ka.success and ka.migrated and kw.migrated
                    and acc(kw.config) < kw.a_req and ka.uav != kw.uav):
                continue
            _, pa = trace(cfg, ADOM(), seed, tid)
            _, pw = trace(cfg, WangDeadline(), seed, tid)
            if pa and pw and pa[0][2:] == pw[0][2:] and abs(pa[0][1] - pw[0][1]) < 0.02:
                return seed, tid
    return None


def trace(cfg, policy, seed, tid):
    sim = Simulator(cfg, policy, seed)
    sim.trace_ids = {tid}
    tasks = sim.run()
    task = next(k for k in tasks if k.id == tid)
    pts = [(max(0.0, t), s, u, c) for t, s, u, c in sim.traces[tid]]
    return task, pts


def fig_casestudy(rate=0.10):
    cfg = replace(SimConfig(), arrival_rate=rate)
    found = find_case(cfg)
    if found is None:
        print("No case-study task found; skipping fig_casestudy.png")
        return
    seed, tid = found
    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=200)
    task = None
    for policy, col, mk, name in ((ADOM(), BLUE, "o", "AD-OM"), (WangDeadline(), ORANGE, "s", "Wang (deadline)")):
        task, pts = trace(cfg, policy, seed, tid)
        t = [p[0] for p in pts] + [task.result_time - task.t_arr]
        s = [p[1] for p in pts] + [pts[-1][1]]
        ax.step(t, s, where="post", color=col, linewidth=1.6, label=name)
        # mark the moment the UAV or the configuration changed
        for (t0, s0, u0, c0), (t1, s1, u1, c1) in zip(pts, pts[1:]):
            if (u0, c0) != (u1, c1):
                what = (f"downgrade {c0}→{c1}\n(same UAV {u1 + 1})" if u0 == u1
                        else f"migrate to UAV {u1 + 1}, {c1}\n({acc(c1) * 100:.1f}% < {task.a_req * 100:.1f}%)"
                        if acc(c1) < task.a_req else f"migrate to UAV {u1 + 1}, {c1}")
                ax.scatter([t1], [s1], color=col, marker=mk, s=30, zorder=4)
                ax.annotate(what, (t1, s1), textcoords="offset points", xytext=(6, 16) if name == "AD-OM" or s1 > 0.15 else (8, 10),
                            fontsize=6, color="#2c2c2a")
        ok = "success" if task.success else ("late" if task.result_time - task.t_arr > task.t_max
                                             else "on time, accuracy too low")
        ax.scatter([t[-1]], [s[-1]], color=col, marker=mk, s=30, zorder=4, facecolor="white")
        ax.annotate(f"answer: {ok}", (t[-1], s[-1]), textcoords="offset points", xytext=(4, 4),
                    fontsize=6, color="#2c2c2a")
    margin = max(cfg.kappa_init * task.t_max, cfg.slot_s)
    ax.axhline(margin, color=GRAY, linestyle="--", linewidth=0.9)
    ax.text(0.0, margin + 0.03, f"margin $\\delta_k$ = {margin:.2f} s", fontsize=6, color="#444441")
    ax.axhline(0, color="#444441", linewidth=0.8)
    ax.set_xlim(-0.05, ax.get_xlim()[1] + 0.45)
    ax.set_xlabel("Time since the task arrived (s)", fontsize=8)
    ax.set_ylabel("Spare time $s_k$ (s)", fontsize=8)
    ax.set_title(f"Task needing accuracy {task.a_req * 100:.1f}% within {task.t_max:g} s", fontsize=8)
    style(ax)
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_casestudy.png"))
    plt.close(fig)
    print(f"Case study: seed {seed}, task {tid}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_scenario()
    fig_tradeoff()
    fig_casestudy()
    print("Saved fig_scenario.png, fig_tradeoff.png, fig_casestudy.png in", OUT)
