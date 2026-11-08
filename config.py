"""
config.py - All simulation parameters in one place.

Every number used by the simulator lives here, next to the paper it comes from,
so the "Simulation Setup" table of the paper can be written directly from this file.
Values marked "ASSUMPTION" are our own choices: state them in the paper or replace them.
"""
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 1. AI model profiles (LLaVA-HR at four input resolutions)
#    accuracy  = share of TextVQA questions answered correctly      [Luo 2024, Li VLM 2025]
#    speed     = tokens generated per second                         [Luo 2024, Li VLM 2025]
#    input_mb  = image size sent over the network (MB)               [Li VLM 2025]
#    768p and 1024p sizes are scaled by pixel count from the 384p/1536p endpoints.
# ---------------------------------------------------------------------------
PROFILES = {
    "384p":  {"accuracy": 0.5963, "speed": 23.8, "input_mb": 0.42},
    "768p":  {"accuracy": 0.6436, "speed": 19.9, "input_mb": 1.68},
    "1024p": {"accuracy": 0.6711, "speed": 19.7, "input_mb": 2.99},
    "1536p": {"accuracy": 0.6796, "speed": 12.6, "input_mb": 6.74},
}
# Ordered from lightest to most accurate. "Lighter" always means "earlier in this list".
CONFIG_ORDER = ["384p", "768p", "1024p", "1536p"]


@dataclass
class SimConfig:
    # ---------------- Area and nodes ----------------
    area_m: float = 500.0            # square area side (m)                    [Meng 2026, Yan 2025: 500 x 500 m]
    n_users: int = 30                # number of wearable users                ASSUMPTION (Naouri: 4-25, Yan: 10)
    uav_altitude_m: float = 100.0    # fixed hovering altitude (m)             [Naouri 2026: 100 m; You 2026: 50-120 m]
    # Hardware-limited, nested configuration sets C_n (Fig. 1 of the paper).
    # UAV 1 small, UAV 2 large, UAV 3 medium. Repeats if more UAVs are used.
    uav_config_sets: list = field(default_factory=lambda: [
        ["384p", "768p"],
        ["384p", "768p", "1024p", "1536p"],
        ["384p", "768p", "1024p"],
    ])
    n_uavs: int = 3                  # number of UAVs                          [You 2026, Yan 2025: 3 UAVs]
    g2a_range_m: float = 300.0       # user-UAV coverage radius, 3-D (m)       [Tambe 2026: 150-350 m]
    a2a_range_m: float = 600.0       # UAV-UAV link range (m)                  ASSUMPTION

    # ---------------- Time ----------------
    slot_s: float = 0.1              # slot length Delta (s), << 2-3 s deadlines  [paper Sec. III]
    gen_time_s: float = 300.0        # tasks are generated during this period (s)
    drain_time_s: float = 15.0       # extra time to finish queued tasks (s)

    # ---------------- Tasks ----------------
    arrival_rate: float = 0.06       # tasks per second per user (Poisson)     ASSUMPTION (swept)
    deadlines_s: tuple = (2.0, 2.5, 3.0)   # T_max values                      [Yan 2025: 2-3 s]
    acc_req_range: tuple = (0.60, 0.67)    # required accuracy a_k             [Li VLM 2025]
    answer_tokens_mean: float = 20.0 # expected answer length o_hat (tokens)  ASSUMPTION (verify)
    answer_tokens_std: float = 5.0   # actual length o_k varies around o_hat   ASSUMPTION
    result_kb: float = 1.0           # size of the text answer sent back (KB)  ASSUMPTION

    # ---------------- Channel (free-space LoS, Eq. 1) ----------------
    bandwidth_hz: float = 10e6       # B, user-UAV bandwidth (10 MHz)         [Meng 2026: B1]
    bandwidth_a2a_hz: float = 15e6   # UAV-UAV bandwidth (15 MHz)              [Meng 2026: B2]
    beta0: float = 1e-3              # channel gain at 1 m (-30 dB)            [Meng 2026]
    noise_w: float = 3.162e-12       # sigma^2 = -85 dBm                       [Meng 2026]
    user_tx_w: float = 0.2           # wearable transmit power p_i            [Meng 2026, Khater 2025]
    uav_tx_w: float = 0.5            # UAV transmit power                      [Meng 2026]

    # ---------------- Energy ----------------
    user_idle_w: float = 0.1         # wearable idle power                     [Khater 2025]
    user_comp_w: float = 1.0         # wearable computing power (local run)    ASSUMPTION
    user_battery_j: float = 15000.0  # ~ 300 mAh x 3.85 V smartwatch           ASSUMPTION
    local_speed: float = 2.0         # tokens/s of 384p on the wearable        ASSUMPTION
    uav_hover_w: float = 138.1       # P1 + P2 at v = 0                        [Meng 2026, Eq. 3]
    uav_comp_w: float = 25.0         # Jetson Orin NX power                    [Yan 2025]
    uav_battery_j: float = 432000.0  # 5400 mAh 6S pack ~ 120 Wh               [Promise 2026, Fig. 2b]
    uav_initial_soc: tuple = (0.18, 1.0)   # initial state of charge range     ASSUMPTION (lets reserve trigger)
    uav_reserve_frac: float = 0.15   # B_min = 0.15 B_max                      [Meng 2026]

    # ---------------- AD-OM ----------------
    kappa_init: float = 0.15         # margin fraction kappa (fixed in AD-OM)  chosen by sensitivity sweep (fig10)
                                     # (adaptive variant starts here and moves within [kappa_min, kappa_max])
    kappa_up: float = 0.02           # increase after a missed deadline
    kappa_down: float = 0.001        # decrease after a success
    kappa_min: float = 0.02
    kappa_max: float = 0.40
    mig_energy_ref_j: float = 1.0    # E_ref used to normalize migration energy ASSUMPTION

    # ---------------- Benchmarks ----------------
    zhang_queue_cap: int = 10        # Gamma (tasks); migrate when queue > Gamma/2  [rule: Zhang 2026, Eq. 65; value: ASSUMPTION]
    meng_load_threshold: float = 0.80   # migrate when load >= 80%           [Meng 2026]
    meng_load_window_s: float = 10.0 # window used to measure load             ASSUMPTION
