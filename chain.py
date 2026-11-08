"""
chain.py - Step-by-step build-up from an existing method (Yan 2025) to AD-OM.

Each step adds ONE feature to the previous one:
  S0  Yan 2025                     fastest option meeting accuracy + deadline, no migration
  S1  + no-harm check              same, but never make an already-queued task late (F4, AD-OM Stage 1)
  S2  + late-trigger migration     migrate when the task is going to be late, accuracy-blind (Wang-style trigger)
  S3  + accuracy check             migrate only to UAVs/configurations that still meet a_k (F1, F5)
  S4  + early trigger  = AD-OM     migrate as soon as spare time < margin, before it is late

Run:  python chain.py --seeds 30
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

from config import SimConfig
from simulator import Simulator, acc
from policies import ADOM, YanNoMigration


STEPS = [
    ("S0 Yan (existing)",        lambda: YanNoMigration()),
    ("S1 + no-harm check",      lambda: ADOM(use_migration=False, name="S1 + no-harm check")),
    ("S2 + late-trigger migration",      lambda: ADOM(acc_aware_migration=False, early_trigger=False, name="S2 + late-trigger migration")),
    ("S3 + accuracy check",      lambda: ADOM(early_trigger=False, name="S3 + accuracy check")),
    ("S4 + early trigger = AD-OM", lambda: ADOM(name="S4 + early trigger = AD-OM")),
]


def one(job):
    step, rate, seed = job
    cfg = SimConfig(arrival_rate=rate)
    pol = dict(STEPS)[step]()
    tasks = Simulator(cfg, pol, seed).run()
    n = len(tasks)
    succ = sum(t.success for t in tasks) / n
    accv = sum(t.config is not None and acc(t.config) < t.a_req for t in tasks) / n   # same as run_experiments
    mig = sum(t.n_migrations for t in tasks) / n
    dg = sum(t.n_downgrades for t in tasks) / n
    return dict(step=step, load=round(rate * cfg.n_users, 1), seed=seed,
                success=succ, acc_violation=accv, migrations=mig, downgrades=dg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--rates", type=float, nargs="+", default=[0.06, 0.10, 0.14])
    a = ap.parse_args()
    jobs = [(s, r, k) for s, _ in STEPS for r in a.rates for k in range(a.seeds)]
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(one, jobs, chunksize=4))
    df = pd.DataFrame(rows)
    df.to_csv("results/chain_raw.csv", index=False)
    g = df.groupby(["step", "load"], sort=False)
    out = g.mean(numeric_only=True).drop(columns="seed")
    ci = 1.96 * g.std(numeric_only=True)["success"] / np.sqrt(a.seeds)
    out["success_ci95"] = ci
    out = out.reset_index()
    out.to_csv("results/chain_summary.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
