"""
policies.py - The decision rules: our AD-OM (Section IV) and every benchmark / baseline.

Each policy answers four questions for the simulator:
  assign(sim, task, now)            -> where does a NEW task go?          (offloading)
  on_slot(sim, now, load_slots)     -> do we change any WAITING task?     (downgrade / migration)
  relocate(sim, tasks, from_uav, now) -> a UAV is leaving to recharge: where do its tasks go?
  on_complete(sim, task)            -> learn from a finished task (AD-OM's adaptive margin)
"""
import math
from config import PROFILES, CONFIG_ORDER
from simulator import acc, lighter_configs


# ===========================================================================
# Shared helper: test one option (UAV n, configuration c) for a task.
# Implements the feasibility checks F2 (spare time), F3 (battery) and F4 (no harm).
# F1 (accuracy) and F6 (range) are checked by the caller before calling this.
# ===========================================================================
def evaluate_option(sim, task, uav, config, now, ready_time, theta, check_no_harm=True):
    """Returns (feasible, spare_time, estimated_result_time)."""
    remove = {task.id}
    fin = sim.estimate_finish_times(uav, now, extra={task: (config, ready_time)}, remove=remove)
    dl = sim.download_time(task, uav.id)
    result_time = fin[task.id] + dl
    spare = task.deadline - result_time
    if spare < theta:                                   # F2
        return False, spare, result_time
    if not sim.uav_energy_ok(uav, fin, now, {task: config}, remove=remove):   # F3
        return False, spare, result_time
    if check_no_harm:                                   # F4
        before = sim.estimate_finish_times(uav, now, remove=remove)
        for j in uav.queue:
            if j.id == task.id or j.deadline <= task.deadline:
                continue
            dlj = sim.download_time(j, uav.id)
            if j.deadline - (before[j.id] + dlj) >= 0 and j.deadline - (fin[j.id] + dlj) < 0:
                return False, spare, result_time
    return True, spare, result_time


def offload_energy(sim, task, upload, result_time, now):
    """Wearable energy of an offloaded task, Eq. (2)."""
    c = sim.cfg
    return c.user_tx_w * upload + c.user_idle_w * max(0.0, (result_time - now) - upload)


def local_option(sim, task, now):
    """Running the lightest configuration (384p) on the wearable itself."""
    c = sim.cfg
    user = sim.users[task.user]
    start = max(now, user.local_busy_until)
    result_time = start + task.o_hat / c.local_speed
    energy = c.user_comp_w * (result_time - start)
    return result_time, energy


def place_best_effort(sim, task, now):
    """No option meets the requirements: answer it anyway, late (best-effort queue)."""
    in_range = [u for u in sim.available_uavs() if sim.user_in_range(task.user, u.id)]
    if not in_range:
        sim.place(task, None, "384p", now)             # nobody in range: run locally
        return
    candidates = []
    for u in in_range:
        for cf in u.configs:
            up = sim.upload_time(task, u.id, cf)
            fin = sim.estimate_finish_times(u, now, extra={task: (cf, now + up)})
            candidates.append((acc(cf) >= task.a_req, -fin[task.id], acc(cf), u.id, cf))
    # Prefer an option that meets the accuracy and finishes first; otherwise the most accurate one.
    accurate = [x for x in candidates if x[0]]
    if accurate:
        best = max(accurate, key=lambda x: (x[1],))
    else:
        best = max(candidates, key=lambda x: (x[2], x[1]))
    sim.place(task, best[3], best[4], now, best_effort=True)


def dest_config_unaware(task, dst_uav):
    """Benchmarks ignore accuracy: keep the same configuration if hosted,
    otherwise the most accurate lighter one (inputs cannot be upscaled)."""
    if dst_uav.hosts(task.config):
        return task.config
    for cf in lighter_configs(task.config):
        if dst_uav.hosts(cf):
            return cf
    return None


# ===========================================================================
# Base class
# ===========================================================================
class Policy:
    name = "base"

    def reset(self, sim):
        self.kappa = {}

    def margin(self, sim, task):
        return max(sim.cfg.kappa_init * task.t_max, sim.cfg.slot_s)

    def assign(self, sim, task, now):
        raise NotImplementedError

    def on_slot(self, sim, now, load_slots):
        pass

    def on_complete(self, sim, task):
        pass

    def relocate(self, sim, tasks, from_uav, now):
        """Default (benchmarks): move each task to the least-loaded reachable UAV, accuracy-unaware."""
        for k in sorted(tasks, key=lambda x: x.deadline):
            dsts = [u for u in sim.available_uavs()
                    if u.id != from_uav and sim.uavs_in_range(from_uav, u.id)
                    and dest_config_unaware(k, u) is not None]
            if not dsts:
                sim.drop(k)
                continue
            dst = min(dsts, key=lambda u: len(u.queue) + len(u.be_queue))
            sim.migrate(k, dst.id, dest_config_unaware(k, dst), now)


# ===========================================================================
# AD-OM (ours)
# ===========================================================================
class ADOM(Policy):
    """Accuracy- and Deadline-aware Offloading and Migration.
    Flags let us build the ablations of the paper."""

    def __init__(self, use_downgrade=True, use_migration=True, adaptive_margin=False, name="AD-OM"):
        self.use_downgrade = use_downgrade
        self.use_migration = use_migration
        self.adaptive_margin = adaptive_margin   # default: fixed margin kappa = 0.25 (see sensitivity sweep)
        self.name = name

    # ---------------- adaptive safety margin, Eq. (6) ----------------
    def margin(self, sim, task):
        kappa = self.kappa.get(task.t_max, sim.cfg.kappa_init)
        return max(kappa * task.t_max, sim.cfg.slot_s)

    def on_complete(self, sim, task):
        if not self.adaptive_margin or task.uav is None and task.origin_uav is None:
            return
        c = sim.cfg
        k = self.kappa.get(task.t_max, c.kappa_init)
        late = task.result_time is None or task.result_time - task.t_arr > task.t_max
        k = k + c.kappa_up if late else k - c.kappa_down   # grow after a miss, relax after a success
        self.kappa[task.t_max] = min(c.kappa_max, max(c.kappa_min, k))

    # ---------------- Stage 1: offloading at arrival ----------------
    def assign(self, sim, task, now):
        for theta in (self.margin(sim, task), 0.0):
            options = []          # (accuracy, spare, -energy, uav_id or None, config)
            if acc("384p") >= task.a_req:                         # local option (F1)
                rt, en = local_option(sim, task, now)
                spare = task.deadline - rt
                if spare >= theta and sim.users[task.user].battery_j > en:
                    options.append((acc("384p"), spare, -en, None, "384p"))
            for u in sim.available_uavs():
                if not sim.user_in_range(task.user, u.id):       # F6
                    continue
                for cf in u.configs:
                    if acc(cf) < task.a_req:                      # F1
                        continue
                    up = sim.upload_time(task, u.id, cf)
                    ok, spare, rt = evaluate_option(sim, task, u, cf, now, now + up, theta)  # F2-F4
                    if ok:
                        options.append((acc(cf), spare, -offload_energy(sim, task, up, rt, now), u.id, cf))
            if options:
                # most accurate, then largest spare time, then lowest wearable energy
                best = max(options, key=lambda x: (x[0], x[1], x[2]))
                sim.place(task, best[3], best[4], now)
                return
        place_best_effort(sim, task, now)

    # ---------------- Stage 2: downgrade and migration ----------------
    def on_slot(self, sim, now, load_slots):
        for u in sim.available_uavs():
            waiting = [k for k in u.queue if k.ready_time <= now]
            if not waiting:
                continue
            fin = sim.estimate_finish_times(u, now)
            order = sorted(waiting, key=lambda k: k.deadline - (fin[k.id] + sim.download_time(k, u.id)))
            for k in order:
                if k not in u.queue:
                    continue
                fin = sim.estimate_finish_times(u, now)          # queue may have changed
                s = k.deadline - (fin[k.id] + sim.download_time(k, u.id))
                delta = self.margin(sim, k)
                if s >= delta:
                    continue                                     # safe: stay
                acted = False
                for theta in (delta, 0.0):
                    if self.use_downgrade and self._try_downgrade(sim, k, u, now, theta):
                        acted = True
                        break
                    if self.use_migration and not k.migrated and self._try_migrate(sim, k, u, now, theta):
                        acted = True
                        break
                if not acted and s < 0:
                    sim.make_best_effort(k)                      # will miss its deadline everywhere

    def _try_downgrade(self, sim, k, u, now, theta):
        for cf in lighter_configs(k.config):                     # most accurate lighter first
            if not u.hosts(cf) or acc(cf) < k.a_req:             # F1
                continue
            ok, _, _ = evaluate_option(sim, k, u, cf, now, k.ready_time, theta)
            if ok:
                sim.downgrade(k, cf)
                return True
        return False

    def _migration_options(self, sim, k, src, now, theta, require_accuracy=True):
        opts = []
        for dst in sim.available_uavs():
            if dst.id == src.id or not sim.uavs_in_range(src.id, dst.id):   # F6
                continue
            for cf in dst.configs:
                if CONFIG_ORDER.index(cf) > CONFIG_ORDER.index(k.config):    # F5: no upscaling
                    continue
                if require_accuracy and acc(cf) < k.a_req:                   # F1
                    continue
                t_mig = sim.migration_time(cf, src.id, dst.id)
                ok, spare, _ = evaluate_option(sim, k, dst, cf, now, max(now, k.ready_time) + t_mig, theta)
                if ok:
                    c = sim.cfg
                    cost = t_mig / k.t_max + c.uav_tx_w * t_mig / c.mig_energy_ref_j   # J_mig
                    opts.append((cost, -acc(cf), dst.id, cf))
        return opts

    def _try_migrate(self, sim, k, u, now, theta):
        opts = self._migration_options(sim, k, u, now, theta)
        if not opts:
            return False
        cost, _, dst, cf = min(opts)
        sim.migrate(k, dst, cf, now)
        return True

    # ---------------- UAV leaving to recharge ----------------
    def relocate(self, sim, tasks, from_uav, now):
        src = sim.uavs[from_uav]
        for k in sorted(tasks, key=lambda x: x.deadline):
            opts = self._migration_options(sim, k, src, now, 0.0)   # forced move: allowed even if migrated
            if opts:
                _, _, dst, cf = min(opts)
                sim.migrate(k, dst, cf, now)
                continue
            # no on-time option: keep accuracy, go best-effort to the reachable UAV that finishes first
            best = None
            for dst in sim.available_uavs():
                if dst.id == from_uav or not sim.uavs_in_range(from_uav, dst.id):
                    continue
                for cf in dst.configs:
                    if CONFIG_ORDER.index(cf) <= CONFIG_ORDER.index(k.config) and acc(cf) >= k.a_req:
                        best = (dst.id, cf)
                        break
                if best:
                    break
            if best:
                k.best_effort = True
                sim.migrate(k, best[0], best[1], now)
            else:
                sim.drop(k)


# ===========================================================================
# Benchmarks that share AD-OM's arrival rule (so only the migration rule differs)
# ===========================================================================
class _SameArrival(Policy):
    """Uses AD-OM's Stage 1 (same fixed margin), then its own migration trigger."""

    def reset(self, sim):
        super().reset(sim)
        self._arrival = ADOM(adaptive_margin=False)
        self._arrival.reset(sim)

    def assign(self, sim, task, now):
        self._arrival.assign(sim, task, now)


class ZhangQueue(_SameArrival):
    """[Zhang 2026] Migrate when a UAV's queue exceeds Gamma/2, to the neighbour with the best
    distance + queue score. Accuracy-unaware."""
    name = "Zhang (queue)"

    def on_slot(self, sim, now, load_slots):
        cap = sim.cfg.zhang_queue_cap
        max_d = sim.cfg.a2a_range_m
        for u in sim.available_uavs():
            while len(u.queue) > cap / 2:
                tail = [k for k in u.queue if k.ready_time <= now and not k.migrated]
                if not tail:
                    break
                k = max(tail, key=lambda x: x.deadline)            # back of the EDF queue
                best, best_score = None, -1e9
                for d in sim.available_uavs():
                    if d.id == u.id or not sim.uavs_in_range(u.id, d.id) or len(d.queue) >= cap / 2:
                        continue
                    if dest_config_unaware(k, d) is None:
                        continue
                    score = (1 - sim.dist(u.pos, d.pos) / max_d) + (1 - len(d.queue) / cap)
                    if score > best_score:
                        best, best_score = d, score
                if best is None:
                    break
                sim.migrate(k, best.id, dest_config_unaware(k, best), now)


class MengLoad(_SameArrival):
    """[Meng 2026] Migrate when a UAV's load reaches 80% (battery <= 15% is handled by relocate),
    to the least-loaded UAV. Accuracy-unaware."""
    name = "Meng (battery/load)"

    def on_slot(self, sim, now, load_slots):
        th = sim.cfg.meng_load_threshold
        for u in sim.available_uavs():
            if u.load(load_slots) < th:
                continue
            tail = [k for k in u.queue if k.ready_time <= now and not k.migrated]
            if not tail:
                continue
            k = max(tail, key=lambda x: x.deadline)
            dsts = [d for d in sim.available_uavs()
                    if d.id != u.id and sim.uavs_in_range(u.id, d.id)
                    and d.load(load_slots) < th and dest_config_unaware(k, d) is not None]
            if not dsts:
                continue
            d = min(dsts, key=lambda x: x.load(load_slots))
            sim.migrate(k, d.id, dest_config_unaware(k, d), now)


class WangDeadline(_SameArrival):
    """[Wang 2025] Migrate a task whose estimated completion exceeds its deadline,
    to the UAV with the lowest estimated completion time. No accuracy check."""
    name = "Wang (deadline)"

    def on_slot(self, sim, now, load_slots):
        for u in sim.available_uavs():
            for k in sorted([k for k in u.queue if k.ready_time <= now and not k.migrated],
                            key=lambda x: x.deadline):
                if k not in u.queue:
                    continue
                fin = sim.estimate_finish_times(u, now)
                cur = fin[k.id] + sim.download_time(k, u.id)
                if cur <= k.deadline:
                    continue
                best = None
                for d in sim.available_uavs():
                    if d.id == u.id or not sim.uavs_in_range(u.id, d.id):
                        continue
                    cf = dest_config_unaware(k, d)
                    if cf is None:
                        continue
                    ready = now + sim.migration_time(cf, u.id, d.id)
                    f = sim.estimate_finish_times(d, now, extra={k: (cf, ready)})
                    t = f[k.id] + sim.download_time(k, d.id)
                    if best is None or t < best[0]:
                        best = (t, d.id, cf)
                if best and best[0] < cur:
                    sim.migrate(k, best[1], best[2], now)


# ===========================================================================
# Benchmarks / baselines with their own arrival rule and no migration
# ===========================================================================
class YanNoMigration(Policy):
    """[Yan 2025] Accuracy- and deadline-feasible assignment with the lowest latency. No migration."""
    name = "Yan (no migration)"

    def assign(self, sim, task, now):
        options = []
        for u in sim.available_uavs():
            if not sim.user_in_range(task.user, u.id):
                continue
            for cf in u.configs:
                if acc(cf) < task.a_req:
                    continue
                up = sim.upload_time(task, u.id, cf)
                ok, _, rt = evaluate_option(sim, task, u, cf, now, now + up, 0.0, check_no_harm=False)
                if ok:
                    options.append((rt, u.id, cf))
        if options:
            _, uid, cf = min(options)
            sim.place(task, uid, cf, now)
        else:
            place_best_effort(sim, task, now)


class NearestUAV(Policy):
    """Baseline: nearest UAV, lightest configuration that meets the accuracy (else the best it has)."""
    name = "Nearest UAV"

    def assign(self, sim, task, now):
        in_range = [u for u in sim.available_uavs() if sim.user_in_range(task.user, u.id)]
        if not in_range:
            sim.place(task, None, "384p", now)
            return
        u = min(in_range, key=lambda x: sim.dist(sim.users[task.user].pos, x.pos))
        ok = [cf for cf in u.configs if acc(cf) >= task.a_req]
        sim.place(task, u.id, ok[0] if ok else u.configs[-1], now)


class LocalOnly(Policy):
    """Baseline: everything runs on the wearable (384p only)."""
    name = "Local only"

    def assign(self, sim, task, now):
        sim.place(task, None, "384p", now)


def all_methods():
    """Every method compared in the paper (AD-OM, its ablations, benchmarks and baselines).
    Local-only is not included: the wearable's 384p model (59.6%) is below every accuracy
    requirement (0.60-0.67), so it can never succeed."""
    return [
        ADOM(),
        ADOM(use_downgrade=False, name="AD-OM w/o downgrade"),
        ADOM(use_migration=False, name="AD-OM w/o migration"),
        ADOM(adaptive_margin=True, name="AD-OM adaptive margin"),
        ZhangQueue(),
        MengLoad(),
        WangDeadline(),
        YanNoMigration(),
        NearestUAV(),
    ]
