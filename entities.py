"""
entities.py - The three "things" in the system model: tasks, users (wearables) and UAVs.
They only hold data; all decisions are made in policies.py and all timing in simulator.py.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass(eq=False)   # eq=False: each task is unique (hashable by identity)
class Task:
    """One visual question: an image + a text question from a wearable."""
    id: int
    user: int                  # index of the user who generated it
    t_arr: float               # arrival time t_k^arr (s)
    a_req: float               # required accuracy a_k
    t_max: float               # relative deadline T_k^max (s)
    o_hat: float               # expected answer length (tokens), known when deciding
    o_act: float               # actual answer length (tokens), known only after inference

    # --- decisions (filled by the policy) ---
    uav: Optional[int] = None  # serving UAV (None = local execution)
    config: Optional[str] = None
    origin_uav: Optional[int] = None   # first UAV, used to relay the result if needed
    best_effort: bool = False  # True if it can no longer meet its requirements
    migrated: bool = False
    n_migrations: int = 0
    n_downgrades: int = 0
    ready_time: float = 0.0    # time its input is fully at the UAV (after upload/migration)
    upload_time: float = 0.0
    migration_energy_j: float = 0.0

    # --- outcome (filled by the simulator) ---
    result_time: Optional[float] = None
    dropped: bool = False
    success: bool = False
    user_energy_j: float = 0.0

    @property
    def deadline(self) -> float:
        """Absolute deadline d_k = t_arr + T_max."""
        return self.t_arr + self.t_max


@dataclass
class User:
    id: int
    pos: np.ndarray            # (x, y, 0)
    battery_j: float
    local_busy_until: float = 0.0


@dataclass
class UAV:
    id: int
    pos: np.ndarray            # (x, y, altitude)
    configs: list              # C_n, hosted configurations (nested)
    battery_j: float
    battery_max_j: float
    active: bool = True        # False once it reached its reserve and left
    leaving: bool = False      # reached reserve: finishes current task, then leaves

    queue: list = field(default_factory=list)      # on-time queue Q_n (EDF)
    be_queue: list = field(default_factory=list)   # best-effort queue Q_n^BE (FIFO)

    in_service: Optional[Task] = None
    service_start: float = 0.0
    service_end: float = 0.0

    busy_history: list = field(default_factory=list)  # busy fraction of recent slots

    def hosts(self, cfg: str) -> bool:
        return cfg in self.configs

    def load(self, n_slots: int) -> float:
        """Fraction of the last n_slots during which the processor was busy (Meng's load)."""
        recent = self.busy_history[-n_slots:]
        return float(np.mean(recent)) if recent else 0.0
