"""
simulator.py - The time-slotted simulation engine (Section III of the paper).

The engine is the same for every method. It:
  1. creates users, UAVs and tasks (same random seed => same tasks for every method),
  2. asks the policy where each new task goes            (policy.assign)
  3. asks the policy to re-check queued tasks every slot (policy.on_slot)
  4. runs the UAV processors, sends results back, updates batteries,
  5. records whether each task succeeded (Eq. 4).

It also offers "estimation" helpers that policies use to predict times
(Eqs. 1-3): upload time, waiting time, inference time, spare time.
"""
import math
import numpy as np

from config import SimConfig, PROFILES, CONFIG_ORDER
from entities import Task, User, UAV


# ---------------------------------------------------------------------------
# Channel model, Eq. (1):  R = B log2(1 + p * beta0 / (sigma^2 * d^2))   [Naouri 2026]
# ---------------------------------------------------------------------------
def link_rate_bps(cfg: SimConfig, tx_power_w: float, distance_m: float, bandwidth_hz: float = None) -> float:
    d2 = max(distance_m, 1.0) ** 2
    snr = tx_power_w * cfg.beta0 / (cfg.noise_w * d2)
    bw = cfg.bandwidth_hz if bandwidth_hz is None else bandwidth_hz
    return bw * math.log2(1.0 + snr)


def acc(config_name: str) -> float:
    return PROFILES[config_name]["accuracy"]


def lighter_configs(config_name: str) -> list:
    """Configurations lighter than config_name, most accurate first."""
    idx = CONFIG_ORDER.index(config_name)
    return list(reversed(CONFIG_ORDER[:idx]))


class Simulator:
    def __init__(self, cfg: SimConfig, policy, seed: int):
        self.cfg = cfg
        self.policy = policy
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.now = 0.0
        self.tasks: list[Task] = []
        self._build_network()
        self._pregenerate_tasks()

    # ======================================================================
    # Setup
    # ======================================================================
    def _build_network(self):
        c = self.cfg
        rng = self.rng
        self.users = [
            User(i, np.array([*rng.uniform(0, c.area_m, 2), 0.0]), c.user_battery_j)
            for i in range(c.n_users)
        ]
        # UAVs evenly spread on a circle around the centre, at fixed altitude.
        self.uavs = []
        for n in range(c.n_uavs):
            angle = 2 * math.pi * n / c.n_uavs
            r = c.area_m * 0.25
            x = c.area_m / 2 + r * math.cos(angle)
            y = c.area_m / 2 + r * math.sin(angle)
            soc = rng.uniform(*c.uav_initial_soc)
            self.uavs.append(UAV(
                id=n,
                pos=np.array([x, y, c.uav_altitude_m]),
                configs=list(c.uav_config_sets[n % len(c.uav_config_sets)]),
                battery_j=soc * c.uav_battery_j,
                battery_max_j=c.uav_battery_j,
            ))

    def _pregenerate_tasks(self):
        """Poisson arrivals per user. Generated in advance so every method sees the same tasks."""
        c = self.cfg
        rng = np.random.default_rng(self.seed + 10_000)   # separate stream: identical across methods
        tid = 0
        self._arrivals = []
        n_slots = int(c.gen_time_s / c.slot_s)
        for s in range(n_slots):
            t = s * c.slot_s
            for u in range(c.n_users):
                if rng.random() < c.arrival_rate * c.slot_s:
                    o_act = max(1.0, rng.normal(c.answer_tokens_mean, c.answer_tokens_std))
                    self._arrivals.append(Task(
                        id=tid, user=u,
                        t_arr=t + rng.uniform(0, c.slot_s),
                        a_req=rng.uniform(*c.acc_req_range),
                        t_max=float(rng.choice(c.deadlines_s)),
                        o_hat=c.answer_tokens_mean,
                        o_act=o_act,
                    ))
                    tid += 1
        self._arrivals.sort(key=lambda k: k.t_arr)

    # ======================================================================
    # Geometry and link helpers
    # ======================================================================
    def dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def user_in_range(self, user_id: int, uav_id: int) -> bool:
        return self.dist(self.users[user_id].pos, self.uavs[uav_id].pos) <= self.cfg.g2a_range_m

    def uavs_in_range(self, u1: int, u2: int) -> bool:
        return self.dist(self.uavs[u1].pos, self.uavs[u2].pos) <= self.cfg.a2a_range_m

    def rate_user_uav(self, user_id: int, uav_id: int) -> float:
        d = self.dist(self.users[user_id].pos, self.uavs[uav_id].pos)
        return link_rate_bps(self.cfg, self.cfg.user_tx_w, d)

    def rate_uav_user(self, uav_id: int, user_id: int) -> float:
        d = self.dist(self.users[user_id].pos, self.uavs[uav_id].pos)
        return link_rate_bps(self.cfg, self.cfg.uav_tx_w, d)

    def rate_uav_uav(self, u1: int, u2: int) -> float:
        return link_rate_bps(self.cfg, self.cfg.uav_tx_w, self.dist(self.uavs[u1].pos, self.uavs[u2].pos),
                             self.cfg.bandwidth_a2a_hz)

    def available_uavs(self):
        """UAVs that can accept new work (active and not leaving)."""
        return [u for u in self.uavs if u.active and not u.leaving]

    # ======================================================================
    # Time estimates used by the policies (Eqs. 1-3)
    # ======================================================================
    def upload_time(self, task: Task, uav_id: int, config: str) -> float:
        bits = PROFILES[config]["input_mb"] * 8e6
        return bits / self.rate_user_uav(task.user, uav_id)

    def migration_time(self, config: str, src: int, dst: int) -> float:
        bits = PROFILES[config]["input_mb"] * 8e6
        return bits / self.rate_uav_uav(src, dst)

    def inference_time(self, uav_id: int, config: str, tokens: float) -> float:
        return tokens / PROFILES[config]["speed"]

    def download_time(self, task: Task, uav_id: int) -> float:
        """Result return. Relayed through the origin UAV if the user is out of range."""
        bits = self.cfg.result_kb * 8e3
        if self.user_in_range(task.user, uav_id):
            return bits / self.rate_uav_user(uav_id, task.user)
        relay = task.origin_uav if task.origin_uav is not None else uav_id
        if relay != uav_id and self.uavs_in_range(uav_id, relay) and self.user_in_range(task.user, relay):
            return bits / self.rate_uav_uav(uav_id, relay) + bits / self.rate_uav_user(relay, task.user)
        return math.inf  # cannot reach the user

    def estimate_finish_times(self, uav: UAV, now: float, extra: dict = None, remove: set = None,
                              cfg_override: dict = None) -> dict:
        """
        Predict when each task in the on-time queue of `uav` will finish processing (EDF order).
          extra       : {task: (config, ready_time)} tasks to add hypothetically
          remove      : task ids to leave out
          cfg_override: {task_id: config} hypothetical new configuration
        Returns {task_id: estimated processing end time}.
        """
        remove = remove or set()
        cfg_override = cfg_override or {}
        items = []
        for k in uav.queue:
            if k.id in remove:
                continue
            cfgk = cfg_override.get(k.id, k.config)
            items.append((k.deadline, k.id, k.ready_time, cfgk, k.o_hat))
        if extra:
            for k, (cfgk, ready) in extra.items():
                items.append((k.deadline, k.id, ready, cfgk, k.o_hat))
        items.sort()
        # The task being processed occupies the processor until (estimated) end.
        if uav.in_service is not None:
            k = uav.in_service
            cursor = max(now, uav.service_start + self.inference_time(uav.id, k.config, k.o_hat))
        else:
            cursor = now
        finish = {}
        for _, tid, ready, cfgk, o_hat in items:
            start = max(cursor, ready)
            cursor = start + self.inference_time(uav.id, cfgk, o_hat)
            finish[tid] = cursor
        return finish

    def spare_times(self, uav: UAV, now: float, finish: dict, tasks: list, cfg_override: dict = None,
                    host: int = None) -> dict:
        """Spare time s_k (Eq. 3) = deadline - (estimated finish + result return)."""
        out = {}
        for k in tasks:
            dl = self.download_time(k, host if host is not None else uav.id)
            out[k.id] = k.deadline - (finish[k.id] + dl)
        return out

    def uav_energy_ok(self, uav: UAV, finish: dict, now: float, extra_cfg: dict, remove: set = None) -> bool:
        """(F3) The UAV can finish its queue (plus new work) without going below its reserve."""
        c = self.cfg
        remove = remove or set()
        t_empty = max(finish.values()) - now if finish else 0.0
        comp = sum(self.inference_time(uav.id, k.config, k.o_hat) for k in uav.queue if k.id not in remove)
        comp += sum(self.inference_time(uav.id, cf, k.o_hat) for k, cf in extra_cfg.items())
        need = c.uav_hover_w * t_empty + c.uav_comp_w * comp
        return uav.battery_j - c.uav_reserve_frac * uav.battery_max_j >= need

    # ======================================================================
    # Actions the policies call
    # ======================================================================
    def place(self, task: Task, uav_id, config: str, now: float, best_effort=False):
        """Send a new task (just arrived) to a UAV, or run it locally (uav_id=None)."""
        task.config = config
        if uav_id is None:                          # local execution on the wearable
            task.uav = None
            user = self.users[task.user]
            start = max(now, user.local_busy_until)
            run = task.o_act / self.cfg.local_speed
            user.local_busy_until = start + run
            task.result_time = start + run
            task.user_energy_j = self.cfg.user_comp_w * run
            user.battery_j -= task.user_energy_j
            self._finish(task, uav_battery_ok=True)
            return
        task.uav = uav_id
        task.origin_uav = uav_id
        task.upload_time = self.upload_time(task, uav_id, config)
        task.ready_time = now + task.upload_time
        task.best_effort = best_effort
        uav = self.uavs[uav_id]
        (uav.be_queue if best_effort else uav.queue).append(task)

    def downgrade(self, task: Task, new_config: str):
        task.config = new_config
        task.n_downgrades += 1

    def migrate(self, task: Task, dst: int, new_config: str, now: float):
        """Move a waiting task to another UAV (input downscaled to new_config before transfer)."""
        src = task.uav
        src_uav, dst_uav = self.uavs[src], self.uavs[dst]
        for q in (src_uav.queue, src_uav.be_queue):
            if task in q:
                q.remove(task)
        t_mig = self.migration_time(new_config, src, dst)
        task.migration_energy_j += self.cfg.uav_tx_w * t_mig
        task.uav = dst
        task.config = new_config
        task.ready_time = max(now, task.ready_time) + t_mig
        task.migrated = True
        task.n_migrations += 1
        (dst_uav.be_queue if task.best_effort else dst_uav.queue).append(task)

    def make_best_effort(self, task: Task):
        uav = self.uavs[task.uav]
        if task in uav.queue:
            uav.queue.remove(task)
        task.best_effort = True
        uav.be_queue.append(task)

    def drop(self, task: Task):
        if task.uav is not None:
            uav = self.uavs[task.uav]
            for q in (uav.queue, uav.be_queue):
                if task in q:
                    q.remove(task)
        task.dropped = True
        task.result_time = None
        self._finish(task, uav_battery_ok=False)

    # ======================================================================
    # Main loop
    # ======================================================================
    def run(self) -> list:
        c = self.cfg
        n_slots = int((c.gen_time_s + c.drain_time_s) / c.slot_s)
        load_slots = max(1, int(c.meng_load_window_s / c.slot_s))
        next_arrival = 0
        self.policy.reset(self)
        trace_ids = getattr(self, "trace_ids", set())   # tasks to follow slot by slot (case study)
        self.traces = {tid: [] for tid in trace_ids}
        for s in range(n_slots):
            t0 = s * c.slot_s
            t1 = t0 + c.slot_s
            self.now = t0
            # 1) New tasks -> policy decides where they go (Stage 1)
            while next_arrival < len(self._arrivals) and self._arrivals[next_arrival].t_arr < t1:
                task = self._arrivals[next_arrival]
                next_arrival += 1
                self.tasks.append(task)
                self.now = max(t0, task.t_arr)
                self.policy.assign(self, task, self.now)
            self.now = t0
            # 2) Re-check queued tasks (Stage 2 / benchmark migration triggers)
            self.policy.on_slot(self, t0, load_slots)
            if trace_ids:
                self._record_traces(t0, trace_ids)
            # 3) Run processors for this slot
            for uav in self.uavs:
                if uav.active:
                    busy = self._run_processor(uav, t0, t1)
                    uav.busy_history.append(busy / c.slot_s)
                    uav.battery_j -= c.uav_hover_w * c.slot_s + c.uav_comp_w * busy
            # 4) Battery reserve: UAV hands over its waiting tasks and leaves
            for uav in self.uavs:
                if uav.active and not uav.leaving and uav.battery_j <= c.uav_reserve_frac * uav.battery_max_j:
                    uav.leaving = True
                    waiting = list(uav.queue) + list(uav.be_queue)
                    uav.queue.clear()
                    uav.be_queue.clear()
                    for k in waiting:
                        k.uav = uav.id   # still formally on this UAV until relocated
                    self.policy.relocate(self, waiting, uav.id, t1)
                if uav.leaving and uav.in_service is None and not uav.queue and not uav.be_queue:
                    uav.active = False
        # Anything never finished counts as dropped
        for k in self.tasks:
            if k.result_time is None and not k.dropped:
                k.dropped = True
                self._finish(k, uav_battery_ok=False)
        return self.tasks

    def _record_traces(self, now: float, trace_ids: set):
        """Case study: record the spare time (Eq. 3), UAV and configuration of followed tasks."""
        for k in self.tasks:
            if k.id not in trace_ids or k.result_time is not None or k.dropped or k.uav is None:
                continue
            uav = self.uavs[k.uav]
            if uav.in_service is k:
                end = uav.service_end
            elif k in uav.queue or k in uav.be_queue:
                fin = self.estimate_finish_times(uav, now, extra={k: (k.config, k.ready_time)} if k in uav.be_queue else None)
                end = fin[k.id]
            else:
                continue
            spare = k.deadline - (end + self.download_time(k, uav.id))
            self.traces[k.id].append((now - k.t_arr, spare, k.uav, k.config))

    def _run_processor(self, uav: UAV, t0: float, t1: float) -> float:
        """Serve tasks on one UAV during [t0, t1). Non-preemptive; on-time queue (EDF) first."""
        time = t0
        busy = 0.0
        while True:
            if uav.in_service is not None:
                end = uav.service_end
                if end > t1:
                    busy += t1 - max(time, uav.service_start)
                    break
                busy += end - max(time, uav.service_start)
                time = end
                self._complete(uav, uav.in_service, end)
                uav.in_service = None
            if uav.leaving:
                break
            nxt = self._pick_next(uav, time)
            if nxt is None:
                pending = [k.ready_time for k in uav.queue + uav.be_queue if k.ready_time > time]
                if pending and min(pending) < t1:
                    time = min(pending)
                    continue
                break
            (uav.queue if nxt in uav.queue else uav.be_queue).remove(nxt)
            uav.in_service = nxt
            uav.service_start = time
            uav.service_end = time + self.inference_time(uav.id, nxt.config, nxt.o_act)
        return busy

    def _pick_next(self, uav: UAV, time: float):
        ready = [k for k in uav.queue if k.ready_time <= time]
        if ready:
            return min(ready, key=lambda k: (k.deadline, k.id))
        ready_be = [k for k in uav.be_queue if k.ready_time <= time]
        if ready_be:
            return ready_be[0]
        return None

    def _complete(self, uav: UAV, task: Task, end: float):
        dl = self.download_time(task, uav.id)
        task.result_time = end + dl if math.isfinite(dl) else None
        if task.result_time is None:
            task.dropped = True
        user = self.users[task.user]
        # Wearable energy, Eq. (2): transmit, then idle until the result arrives [Khater 2025]
        latency = (task.result_time if task.result_time else end) - task.t_arr
        task.user_energy_j = (self.cfg.user_tx_w * task.upload_time
                              + self.cfg.user_idle_w * max(0.0, latency - task.upload_time))
        user.battery_j -= task.user_energy_j
        self._finish(task, uav_battery_ok=uav.battery_j > 0)

    def _finish(self, task: Task, uav_battery_ok: bool):
        """Success condition, Eq. (4): accurate enough AND on time AND batteries > 0."""
        ok_acc = task.config is not None and acc(task.config) >= task.a_req
        ok_time = task.result_time is not None and (task.result_time - task.t_arr) <= task.t_max + 1e-9
        ok_batt = self.users[task.user].battery_j > 0 and uav_battery_ok
        task.success = bool(ok_acc and ok_time and ok_batt and not task.dropped)
        self.policy.on_complete(self, task)
