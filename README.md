# AD-OM simulator

Time-slotted Python simulator for the paper *Accuracy- and Deadline-Aware Multimodal AI Task
Offloading and Migration in Cooperative Multi-UAV Edge Computing*.

## Run it

```bash
pip install numpy pandas scipy matplotlib
python run_experiments.py --seeds 5     # quick check
python run_experiments.py               # paper run: 30 seeds, all 5 experiments
python make_figures.py                  # scenario map, accuracy-time trade-off, case study
```
Results go to `results/`: `summary.csv` (mean + 95% CI), `raw_results.csv`, and `fig_*.png`.

## Files and how they map to the paper

| File | What it does | Paper section |
|---|---|---|
| `config.py` | Every parameter, with its source paper. LLaVA-HR profiles. | Simulation setup table |
| `entities.py` | `Task`, `User`, `UAV` data | III-A |
| `simulator.py` | Channel (Eq. 1), time estimates (Eqs. 2-3), queues, batteries, success (Eq. 4) | III |
| `policies.py` | AD-OM (Stage 1, Stage 2, margin, F1-F6) and all benchmarks | IV + Plan Forward |
| `run_experiments.py` | Sweeps load and seeds, computes metrics and 95% CIs, draws figures | Results |
| `make_figures.py` | Scenario map, accuracy-time trade-off, one-task case study (AD-OM vs Wang) | Results (illustrations) |

## One slot of the simulation

1. New tasks arrive (Poisson). The policy picks local or a (UAV, configuration) - `assign`.
2. The policy re-checks waiting tasks - `on_slot` (AD-OM: stay / migrate / best-effort).
3. Each UAV processes its queue for 0.1 s (EDF, non-preemptive; best-effort queue last).
4. Batteries drain (hover + computing). A UAV at 15% hands its waiting tasks over and leaves.
5. Finished tasks are scored: success = accurate enough AND on time AND batteries OK.

## Methods compared

| Name | Arrival rule | Migration trigger | Accuracy-aware migration |
|---|---|---|---|
| AD-OM | fastest feasible, with margin + no-harm | own spare time < margin (kappa = 0.15) | yes |
| AD-OM w/o accuracy check | ablation | same | no |
| AD-OM w/o early trigger | ablation | only when it will be late (s < 0) | yes |
| AD-OM w/o migration | ablation | none | - |
| Zhang (queue) | same as AD-OM | queue > Gamma/2 | no |
| Meng (battery/load) | same as AD-OM | load >= 80% | no |
| Wang (deadline) | same as AD-OM | estimated completion > deadline | no |
| Yan (no migration) | lowest-latency feasible | none | - |
| Nearest UAV | nearest UAV | none | - |

## Simplifying assumptions (state them in the paper)

- Users and UAVs are static; channels are line-of-sight with no interference.
- The AI model is not executed: accuracy, speed and input size come from published LLaVA-HR profiles.
- A UAV that reaches its reserve leaves and does not return within the simulated 5 minutes.
- Benchmarks keep only their trigger and destination rules; when a destination lacks the task's
  configuration they use the most accurate lighter one (inputs cannot be upscaled).
- All methods migrate a task at most once (forced moves when a UAV leaves are exempt).

## Experiments (run_experiments.py)

| Name | What varies | Fixed load |
|---|---|---|
| E1 load | arrival rate 0.04-0.14 tasks/s per user (1.2-4.2 tasks/s total) | - |
| E2 uavs | 3, 4, 5, 6 UAVs | 0.10 |
| E3 deadline | deadlines x0.75, x1, x1.5 | 0.10 |
| E4 accuracy | required accuracy from [0.60, 0.62, 0.64] up to 0.67 | 0.10 |
| E5 margin | AD-OM margin fraction kappa 0.05-0.40 (sensitivity) | 0.06 and 0.12 |

## Step-by-step build-up from Yan (chain.py)

`python chain.py --seeds 30` starts from Yan 2025 and adds one AD-OM feature per step:
S0 Yan -> S1 + no-harm check -> S2 + late-trigger migration (accuracy-blind)
-> S3 + accuracy check -> S4 + early trigger = AD-OM. Output: results/chain_summary.csv.

Note: AD-OM no longer downgrades. Placing each task at the fastest feasible configuration means
it already runs the lightest configuration that meets a_k on its UAV, so no lighter configuration
can pass the accuracy check (0 downgrades measured). The code path is kept (`use_downgrade`).

Local-only is not a method: the wearable's 384p model (59.6%) is below every required
accuracy (0.60-0.67), so no task can succeed locally. AD-OM still checks the local option.
