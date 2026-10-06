"""Attitude control simulation for the 3U CubeSat (Project B).

Two runners share the truth model and time stepping below:

* :func:`run` (Group 6, M7): **perfect knowledge** - the controller sees the
  true attitude, rate, wheel momentum and magnetic field. Spec:
  ``docs/specs/adcs_control.md`` section 3.
* :func:`run_closed_loop` (Group 8, M9): sensors (with optional truth-side
  errors, :class:`SensorErrors`) -> MEKF -> mode machine -> controller on the
  **estimated** attitude. Spec: ``docs/specs/adcs_closed_loop.md`` section 3.

Time stepping (zero-order hold, as flight software would do it): every control
period ``T`` (``MagnetorquerConfig.control_period``) starting at ``t_k = k T``

1. sample the field ``b_b = R(q)^T B_eci(t_k)`` (the coils were off),
2. compute the dipole ``m`` and the wheel torque ``u_w`` for the period,
3. integrate the rigid-body dynamics (``dynamics.rk4_step``) with the coils on
   for ``on_time = duty T`` and off for the rest.

Held over the period: ``m``, ``u_w``, the disturbance torque (evaluated at
``t_k`` with the state at ``t_k``) and ``B_eci(t_k)``. The magnetorquer torque
``m x R(q)^T B_eci`` is re-evaluated with the current ``q`` at every RK4 stage
(the body can turn ~10 deg per period while tumbling). Holding the disturbance
torque (~1e-7 N m) and the inertial field (rotates ~0.1 deg/s) over 1 s is an
approximation recorded in ``assumptions.md`` [B-17].
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field

import numpy as np

from common import quaternion as quat
from adcs.actuators import (MagnetorquerConfig, WheelConfig, limit_wheel_torque,
                            wheel_speed)
from adcs.config import DEFAULT, SimConfig
from adcs.control import (PointingGains, bdot_dipole, bdot_gain, dump_active,
                          dump_dipole, nadir_reference, pointing_torque)
from adcs.disturbances import total_disturbance_torque
from adcs.dynamics import _cross, rk4_step
from adcs.environment import FieldTable, sunlit
from adcs.estimation import MEKF, MekfConfig, css_vector_complete, nees, triad_init, vector_angle
from adcs.sensors import (Gyro, GyroConfig, MagnetometerConfig, SunSensorConfig, albedo_currents,
                          gauss_markov, magnetometer, sun_sensor_currents, sun_vector_from_currents)
from adcs.orbit import Ephemeris, initial_state, orbital_period

__all__ = [
    "ControlConfig",
    "SimResult",
    "run",
    "tumble_state",
    "lvlh_state",
    "detumble_time",
    "MODE_DETUMBLE",
    "MODE_POINTING",
    "SensorErrors",
    "IDEAL_SENSORS",
    "REALISTIC_SENSORS",
    "MEKF_G7",
    "MEKF_DESIGN",
    "ModeConfig",
    "ClosedLoopResult",
    "run_closed_loop",
    "summarize_closed_loop",
]

MODE_DETUMBLE = 0
MODE_POINTING = 1


@dataclass(frozen=True)
class ControlConfig:
    """Controller settings (perfect knowledge). See ``assumptions.md`` [B-16]."""

    bdot_k: float | None = None          # N m s; None -> bdot_gain(inertia, a, inc)
    gains: PointingGains = field(default_factory=PointingGains)
    dumping: bool = True
    dump_k: float = 1e-3                 # 1/s, cross-product dumping gain
    dump_on: float = 0.3                 # fraction of max wheel speed: dumping on
    dump_off: float = 0.05               # fraction of max wheel speed: dumping off
    detumble_threshold: float = float(np.deg2rad(0.5))   # rad/s
    detumble_hold: float = 60.0          # s below threshold before auto -> pointing
    disturbances: bool = True


@dataclass
class SimResult:
    """Logged simulation output; arrays have time on axis 0 (SI units, body axes)."""

    t: np.ndarray            # (N,) s since epoch
    q: np.ndarray            # (N, 4) body-to-ECI
    w: np.ndarray            # (N, 3) rad/s
    h_w: np.ndarray          # (N, 3) N m s
    wheel_speed: np.ndarray  # (N, 3) rad/s relative to the body
    m: np.ndarray            # (N, 3) A m^2, dipole applied during the period starting at t
    u_w: np.ndarray          # (N, 3) N m, wheel torque delivered during that period
    tau_dist: np.ndarray     # (N, 3) N m, disturbance torque at t
    tau_mag: np.ndarray      # (N, 3) N m, magnetorquer torque at t (coils on)
    nadir_err: np.ndarray    # (N,) rad, angle to the LVLH target
    rate: np.ndarray         # (N,) rad/s, |w|
    mode: np.ndarray         # (N,) int, MODE_DETUMBLE / MODE_POINTING
    dump: np.ndarray         # (N,) bool, dumping active
    meta: dict


def _rot_inv(q, v) -> np.ndarray:
    """``R(q)^T v`` for one quaternion and one 3-vector (fast path of ``quat.rotate_inverse``)."""
    u = q[1:4]
    uv = _cross(u, v)
    return v - 2.0 * q[0] * uv + 2.0 * _cross(u, uv)


def tumble_state(rate: float, seed: int | None = None) -> np.ndarray:
    """Initial state ``[q, w, h_w]`` (10,) with a uniformly random attitude and rate axis.

    ``rate`` (rad/s) is ``|w|``; ``h_w = 0``.
    """
    rng = np.random.default_rng(seed)
    q = quat.canonicalize(quat.normalize(rng.normal(size=4)))
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return np.concatenate([q, float(rate) * axis, np.zeros(3)])


def lvlh_state(r_eci, v_eci, offset=(0.0, 0.0, 0.0), w_rel=(0.0, 0.0, 0.0)) -> np.ndarray:
    """State near the nadir target: ``q = q_ref (x) exp(offset)``, ``w = R(exp(offset))^T w_ref + w_rel``.

    ``offset`` rotation vector (rad, body axes); ``w_rel`` rad/s body axes; ``h_w = 0``.
    """
    q_ref, w_ref = nadir_reference(r_eci, v_eci)
    dq = quat.from_rotvec(np.asarray(offset, float))
    q = quat.canonicalize(quat.multiply(q_ref, dq))
    w = quat.rotate_inverse(dq, w_ref) + np.asarray(w_rel, float)
    return np.concatenate([q, w, np.zeros(3)])


def detumble_time(t, rate, threshold: float) -> float:
    """Earliest ``t_i`` with ``rate[j] < threshold`` for every ``j >= i``; ``nan`` if none."""
    t = np.asarray(t, float)
    above = np.asarray(rate, float) >= threshold
    if above[-1]:
        return float("nan")
    idx = np.nonzero(above)[0]
    return float(t[0] if idx.size == 0 else t[idx[-1] + 1])


def run(y0, t_end: float, mode: str = "detumble", cfg: SimConfig = DEFAULT,
        ctrl: ControlConfig = ControlConfig(), mtq: MagnetorquerConfig = MagnetorquerConfig(),
        wheel: WheelConfig = WheelConfig(), log_every: int = 1) -> SimResult:
    """Simulate the attitude loop with perfect knowledge from ``y0`` for ``t_end`` s.

    ``y0`` = ``[q, w, h_w]`` (10,) at the epoch (t = 0); ``mode`` one of
    ``"detumble"`` (B-dot only, wheels idle), ``"pointing"`` (nadir PD on the
    wheels + momentum dumping), ``"auto"`` (detumble, then pointing once
    ``|w| < detumble_threshold`` has held for ``detumble_hold`` s).
    """
    if mode not in ("detumble", "pointing", "auto"):
        raise ValueError(f"unknown mode {mode!r}")
    T = float(mtq.control_period)
    n_steps = int(round(float(t_end) / T))
    if n_steps < 1:
        raise ValueError("t_end shorter than one control period")
    t_grid = np.arange(n_steps + 1) * T
    inertia = cfg.spacecraft.inertia
    k_bdot = (bdot_gain(inertia, cfg.orbit.a, cfg.orbit.inc)
              if ctrl.bdot_k is None else float(ctrl.bdot_k))

    r0, v0 = initial_state(cfg)
    eph = Ephemeris(r0, v0, t_grid[-1] + 10.0)
    table = FieldTable(eph, cfg, t_grid[-1] + 10.0)
    r_all, v_all = eph.at(t_grid)
    b_all = table.b_eci(t_grid)
    qref_all, wref_all = nadir_reference(r_all, v_all)

    on_t = float(mtq.on_time)
    off_t = T - on_t
    y = np.asarray(y0, dtype=float).copy()
    y[0:4] = quat.normalize(y[0:4])

    cur_mode = MODE_POINTING if mode == "pointing" else MODE_DETUMBLE
    below_since = None
    switch_time = 0.0 if mode == "pointing" else float("nan")
    dumping = False
    b_prev = None

    n_log = n_steps // log_every + 1
    log = {name: np.zeros((n_log, d)) for name, d in
           (("q", 4), ("w", 3), ("h_w", 3), ("wheel_speed", 3), ("m", 3), ("u_w", 3),
            ("tau_dist", 3), ("tau_mag", 3))}
    log_t = np.zeros(n_log)
    log_err = np.zeros(n_log)
    log_mode = np.zeros(n_log, dtype=int)
    log_dump = np.zeros(n_log, dtype=bool)
    j = 0

    for k in range(n_steps + 1):
        t = t_grid[k]
        q, w, h = y[0:4], y[4:7], y[7:10]
        r, v, b_eci = r_all[k], v_all[k], b_all[k]
        b_b = quat.rotate_inverse(q, b_eci)

        # auto mode: switch once to pointing after the rate stayed low
        rate = float(np.linalg.norm(w))
        if mode == "auto" and cur_mode == MODE_DETUMBLE:
            if rate < ctrl.detumble_threshold:
                below_since = t if below_since is None else below_since
                if t - below_since >= ctrl.detumble_hold:
                    cur_mode = MODE_POINTING
                    switch_time = t
            else:
                below_since = None

        m = np.zeros(3)
        u_w = np.zeros(3)
        if cur_mode == MODE_DETUMBLE:
            if b_prev is not None:
                m = bdot_dipole(b_b, b_prev, T, k_bdot, mtq.m_max)
        else:
            omega = wheel_speed(h, w, wheel)
            frac = float(np.max(np.abs(omega))) / wheel.max_speed
            dumping = ctrl.dumping and dump_active(dumping, frac, ctrl.dump_on, ctrl.dump_off)
            if dumping:
                m = dump_dipole(h, b_b, ctrl.dump_k, mtq.m_max)
            tau, _ = pointing_torque(q, w, h, qref_all[k], wref_all[k], inertia, ctrl.gains)
            tau_mag_ff = mtq.duty * _cross(m, b_b)       # period-average coil torque
            u_w = limit_wheel_torque(-(tau - tau_mag_ff), h, w, wheel, dt=T)
        b_prev = b_b

        tau_d = (total_disturbance_torque(r, v, q, b_eci, cfg.spacecraft)
                 if ctrl.disturbances else np.zeros(3))

        if k % log_every == 0:
            log["q"][j], log["w"][j], log["h_w"][j] = q, w, h
            log["wheel_speed"][j] = wheel_speed(h, w, wheel)
            log["m"][j], log["u_w"][j], log["tau_dist"][j] = m, u_w, tau_d
            log["tau_mag"][j] = _cross(m, b_b)
            log_t[j] = t
            dq = quat.multiply(quat.conjugate(qref_all[k]), q)
            log_err[j] = np.linalg.norm(quat.to_rotvec(dq))
            log_mode[j] = cur_mode
            log_dump[j] = dumping
            j += 1
        if k == n_steps:
            break

        def ext_on(_t, qq, _w, _h, m=m, b_eci=b_eci, tau_d=tau_d):
            return _cross(m, _rot_inv(qq, b_eci)) + tau_d

        y = rk4_step(t, y, on_t, inertia, u_w, ext_on)
        if off_t > 0.0:
            y = rk4_step(t + on_t, y, off_t, inertia, u_w, tau_d)

    meta = {
        "mode": mode,
        "t_end": float(t_grid[-1]),
        "control_period": T,
        "bdot_k": k_bdot,
        "switch_time": switch_time,
        "orbital_period": float(orbital_period(cfg.orbit.a)),
        "ctrl": asdict(ctrl),
        "mtq": asdict(mtq),
        "wheel": asdict(wheel) | {"max_speed": wheel.max_speed, "spin_inertia": wheel.spin_inertia},
    }
    return SimResult(t=log_t[:j], q=log["q"][:j], w=log["w"][:j], h_w=log["h_w"][:j],
                     wheel_speed=log["wheel_speed"][:j], m=log["m"][:j], u_w=log["u_w"][:j],
                     tau_dist=log["tau_dist"][:j], tau_mag=log["tau_mag"][:j],
                     nadir_err=log_err[:j], rate=np.linalg.norm(log["w"][:j], axis=1),
                     mode=log_mode[:j], dump=log_dump[:j], meta=meta)


# ---------------------------------------------------------------------------
# Closed loop on the estimated attitude (Group 8, M9)
# ---------------------------------------------------------------------------
#
# Spec: docs/specs/adcs_closed_loop.md section 3. Per control period starting
# at t_k (sampling model of docs/specs/adcs_estimation.md section 0):
#
# 1. sensors from the truth: gyro sample k = mean rate over [t_{k-1}, t_k]
#    (from q_{k-1}, q_k) + bias + ARW; magnetometer with the coils off from the
#    *truth* field (full IGRF + unmodelled residual) + bias + noise; CSS
#    currents (FOV cut, albedo, noise);
# 2. estimator: TRIAD initialisation once the init conditions hold, then one
#    MEKF cycle per period against the *onboard* field model;
# 3. mode machine on the estimate (detumble -> pointing once the MEKF has
#    converged and the bias-corrected rate is low; fallback to detumble if the
#    estimated rate stays high);
# 4. control from measurements/estimates only (B-dot on the measured field;
#    PD on q_hat and w_hat = gyro - b_hat; dumping on the bias-corrected
#    measured field); wheel momentum from perfect tachometers, orbit (nadir
#    reference) from perfect onboard navigation [B-23];
# 5. truth integration exactly as in run() with the *true* inertia.

@dataclass(frozen=True)
class SensorErrors:
    """Truth-side sensor and environment errors for the closed loop [B-21].

    Defaults (``IDEAL_SENSORS``) reproduce the Group 7 sensor models.
    """

    mag_bias_sigma: float = 0.0          # T per axis, per-run constant body-frame bias draw
    mag_bias: tuple | None = None        # T, explicit bias (overrides the draw)
    field_residual_sigma: float = 0.0    # T per ECI axis, Gauss-Markov unmodelled field
    field_residual_tau: float = 600.0    # s
    onboard_field_degree: int = 13       # IGRF degree of the filter's reference field
    css_fov: float | None = None         # rad, CSS half-angle field of view (None = hemisphere)
    albedo: float = 0.0                  # mean Earth albedo coefficient on the CSS (0 = off)
    albedo_sigma: float = 0.0            # Gauss-Markov std of the albedo coefficient
    albedo_tau: float = 300.0            # s
    albedo_clip: tuple = (0.05, 0.9)     # bounds of the albedo coefficient


IDEAL_SENSORS = SensorErrors()
REALISTIC_SENSORS = SensorErrors(mag_bias_sigma=200e-9, field_residual_sigma=50e-9,
                                 onboard_field_degree=8, css_fov=float(np.deg2rad(60.0)),
                                 albedo=0.3, albedo_sigma=0.1)

#: Group 7 filter (6 states, CSS vector) with the TRIAD angle guard.
MEKF_G7 = MekfConfig(triad_min_angle=float(np.deg2rad(20.0)))
#: Group 8 design filter [B-22]: magnetometer-bias states, R inflated for the
#: onboard field-model error, per-diode CSS updates with FOV gating and
#: onboard albedo compensation.
MEKF_DESIGN = MekfConfig(mag_bias=True, sigma_mag=100e-9, mag_bias_rw=1e-11,
                         css_mode="diodes", css_fov=float(np.deg2rad(60.0)),
                         albedo_model=0.3, sigma_albedo=0.5,
                         triad_min_angle=float(np.deg2rad(20.0)),
                         init_att_sigma_floor=float(np.deg2rad(5.0)))


@dataclass(frozen=True)
class ModeConfig:
    """Mode-machine thresholds on the estimate [B-23]."""

    switch_att_sigma: float = float(np.deg2rad(2.0))  # rad, MEKF sqrt(trace P_att) below this
    switch_min_filter_time: float = 300.0             # s since MEKF initialisation
    safe_rate: float = float(np.deg2rad(2.0))         # rad/s, pointing -> detumble above this
    safe_hold: float = 10.0                           # s the estimated rate must stay above


@dataclass
class ClosedLoopResult(SimResult):
    """:class:`SimResult` plus estimator output (NaN before MEKF initialisation)."""

    q_est: np.ndarray        # (N, 4) estimated body-to-ECI
    b_est: np.ndarray        # (N, 3) rad/s, estimated gyro bias
    bm_est: np.ndarray       # (N, 3) T, estimated magnetometer bias (0 for the 6-state filter)
    att_sigma: np.ndarray    # (N,) rad, sqrt(trace P_att)
    P_att: np.ndarray        # (N, 3, 3) rad^2, attitude-error covariance
    know_err: np.ndarray     # (N, 3) rad, q_true = q_est (x) exp(know_err)
    est_nadir_err: np.ndarray  # (N,) rad, angle of q_est to the LVLH target (what the controller sees)
    w_est: np.ndarray        # (N, 3) rad/s, gyro - b_est used by the controller (raw gyro before init)
    gyro_bias: np.ndarray    # (N, 3) rad/s, true gyro bias
    sun_used: np.ndarray     # (N,) int, sun measurements used at t (diodes, or 1 for the vector)
    lit: np.ndarray          # (N,) bool, truth shadow flag


def run_closed_loop(y0, t_end: float, seed: int | None = 0, cfg: SimConfig = DEFAULT,
                    ctrl: ControlConfig = ControlConfig(), mtq: MagnetorquerConfig = MagnetorquerConfig(),
                    wheel: WheelConfig = WheelConfig(), sensors: SensorErrors = IDEAL_SENSORS,
                    mekf_cfg: MekfConfig = MEKF_G7, modes: ModeConfig = ModeConfig(),
                    mag_cfg: MagnetometerConfig = MagnetometerConfig(),
                    css_cfg: SunSensorConfig = SunSensorConfig(), gyro_cfg: GyroConfig = GyroConfig(),
                    inertia_true=None, log_every: int = 1) -> ClosedLoopResult:
    """Closed-loop ADCS from ``y0`` = ``[q, w, h_w]`` (10,) for ``t_end`` s on the estimated attitude.

    Starts in detumble (B-dot on the measured field); the MEKF is initialised
    by TRIAD when a sun measurement is available (diodes mode: all three CSS
    axes lit above ``diode_min_current``) and the field-sun angle is at least
    ``mekf_cfg.triad_min_angle``; the switch to nadir pointing needs the MEKF
    running for ``modes.switch_min_filter_time``, ``sqrt(trace P_att) <
    modes.switch_att_sigma`` and the bias-corrected gyro rate below
    ``ctrl.detumble_threshold`` for ``ctrl.detumble_hold`` s. ``inertia_true``
    (3, 3) is the truth inertia (default: the nominal ``cfg`` inertia, which the
    controller always uses). Random streams ``default_rng(seed).spawn(4)`` =
    (gyro, magnetometer, sun sensor, truth errors), the first three as in
    ``adcs.sensors.measure_series``.
    """
    T = float(mtq.control_period)
    n_steps = int(round(float(t_end) / T))
    if n_steps < 1:
        raise ValueError("t_end shorter than one control period")
    t_grid = np.arange(n_steps + 1) * T
    inertia_nom = cfg.spacecraft.inertia
    inertia = inertia_nom if inertia_true is None else np.asarray(inertia_true, float)
    k_bdot = (bdot_gain(inertia_nom, cfg.orbit.a, cfg.orbit.inc)
              if ctrl.bdot_k is None else float(ctrl.bdot_k))

    rng_g, rng_m, rng_s, rng_truth = np.random.default_rng(seed).spawn(4)
    rng_bias, rng_field, rng_alb = rng_truth.spawn(3)

    # --- truth environment and onboard models -----------------------------
    r0, v0 = initial_state(cfg)
    eph = Ephemeris(r0, v0, t_grid[-1] + 10.0)
    table = FieldTable(eph, cfg, t_grid[-1] + 10.0)
    if sensors.onboard_field_degree == table.max_degree:
        table_ob = table
    else:
        table_ob = FieldTable(eph, cfg, t_grid[-1] + 10.0, max_degree=sensors.onboard_field_degree)
    r_all, v_all = eph.at(t_grid)
    b_ob_all = table_ob.b_eci(t_grid)                    # filter reference field
    b_all = table.b_eci(t_grid) + gauss_markov(n_steps + 1, T, sensors.field_residual_sigma,
                                                sensors.field_residual_tau, rng_field)
    s_all = table.sun_dir_eci(t_grid)
    lit_all = sunlit(r_all, s_all)
    qref_all, wref_all = nadir_reference(r_all, v_all)
    if sensors.albedo > 0.0:
        a_all = np.clip(sensors.albedo + gauss_markov(n_steps + 1, T, sensors.albedo_sigma,
                                                      sensors.albedo_tau, rng_alb, dim=1)[:, 0],
                        *sensors.albedo_clip)
    else:
        a_all = None
    bias_draw = rng_bias.standard_normal(3)
    mag_bias = (np.asarray(sensors.mag_bias, float) if sensors.mag_bias is not None
                else sensors.mag_bias_sigma * bias_draw)
    css_true = SunSensorConfig(sigma_diode=css_cfg.sigma_diode, threshold=css_cfg.threshold,
                               fov=sensors.css_fov)
    gyro = Gyro(gyro_cfg, rng_g)

    on_t = float(mtq.on_time)
    off_t = T - on_t
    y = np.asarray(y0, dtype=float).copy()
    y[0:4] = quat.normalize(y[0:4])

    f: MEKF | None = None
    k_init = -1
    cur_mode = MODE_DETUMBLE
    below_since = None
    above_since = None
    switch_time = float("nan")
    switch_times: list = []
    fallback_times: list = []
    dumping = False
    mag_prev = None
    q_prev = None

    n_log = n_steps // log_every + 1
    vec = {name: np.full((n_log, d), np.nan) for name, d in
           (("q", 4), ("w", 3), ("h_w", 3), ("wheel_speed", 3), ("m", 3), ("u_w", 3),
            ("tau_dist", 3), ("tau_mag", 3), ("q_est", 4), ("b_est", 3), ("bm_est", 3),
            ("know_err", 3), ("w_est", 3), ("gyro_bias", 3))}
    sca = {name: np.full(n_log, np.nan) for name in ("t", "nadir_err", "est_nadir_err", "att_sigma")}
    log_mode = np.zeros(n_log, dtype=int)
    log_dump = np.zeros(n_log, dtype=bool)
    log_used = np.zeros(n_log, dtype=int)
    log_lit = np.zeros(n_log, dtype=bool)
    log_P = np.full((n_log, 3, 3), np.nan)
    j = 0

    for k in range(n_steps + 1):
        t = t_grid[k]
        q, w, h = y[0:4], y[4:7], y[7:10]
        r, v, b_eci, s_eci = r_all[k], v_all[k], b_all[k], s_all[k]

        # ---- 1. sensors ---------------------------------------------------
        gyro_k = None
        if q_prev is not None:
            w_mean = quat.to_rotvec(quat.multiply(quat.conjugate(q_prev), q)) / T
            gyro_k = gyro.sample(w_mean, T)
        mag_k = magnetometer(_rot_inv(q, b_eci), rng_m, mag_cfg, bias=mag_bias)
        alb = None if a_all is None else albedo_currents(q, r, s_eci, a_all[k])
        currents = sun_sensor_currents(_rot_inv(q, s_eci), bool(lit_all[k]), rng_s, css_true,
                                       albedo=alb)
        sun_k, sun_valid = sun_vector_from_currents(currents, css_true)
        sun_valid = bool(sun_valid)

        # ---- 2. estimator -------------------------------------------------
        used = 0
        if f is None:
            ok = sun_valid
            if ok and mekf_cfg.css_mode == "diodes":
                ok = css_vector_complete(currents, mekf_cfg.diode_min_current)
            if ok and mekf_cfg.triad_min_angle > 0.0:
                ok = vector_angle(mag_k, sun_k) >= mekf_cfg.triad_min_angle
            if ok:
                q0, P0 = triad_init(mag_k, sun_k, b_ob_all[k], s_eci, mekf_cfg)
                f = MEKF(q0, np.zeros(3), P0, mekf_cfg)
                k_init = k
        elif gyro_k is not None:
            n_sun = len(f.ekf.history.get("sun", ()))
            n_css = len(f.ekf.history.get("css", ()))
            f.step(gyro_k, T, mag_k, b_ob_all[k], sun_k, s_eci, sun_valid, currents, r, t)
            if len(f.ekf.history.get("sun", ())) > n_sun:
                used = 1
            hist = f.ekf.history.get("css", ())
            if len(hist) > n_css:
                used = int(hist[-1].residual.size)

        # ---- 3. mode machine on the estimate ------------------------------
        if f is not None:
            q_hat, b_hat, bm_hat = f._q, f._b, f._bm
            w_hat = None if gyro_k is None else gyro_k - b_hat
        else:
            q_hat = b_hat = None
            bm_hat = np.zeros(3)
            w_hat = gyro_k
        rate_hat = np.inf if w_hat is None else float(np.linalg.norm(w_hat))
        if cur_mode == MODE_DETUMBLE:
            ready = (f is not None and (k - k_init) * T >= modes.switch_min_filter_time
                     and f.att_sigma < modes.switch_att_sigma)
            if ready and rate_hat < ctrl.detumble_threshold:
                below_since = t if below_since is None else below_since
                if t - below_since >= ctrl.detumble_hold:
                    cur_mode = MODE_POINTING
                    switch_times.append(float(t))
                    if np.isnan(switch_time):
                        switch_time = float(t)
                    above_since = None
            else:
                below_since = None
        else:
            if rate_hat > modes.safe_rate:
                above_since = t if above_since is None else above_since
                if t - above_since >= modes.safe_hold:
                    cur_mode = MODE_DETUMBLE
                    fallback_times.append(float(t))
                    below_since = None
                    dumping = False
            else:
                above_since = None

        # ---- 4. control from measurements / estimates ---------------------
        m = np.zeros(3)
        u_w = np.zeros(3)
        if cur_mode == MODE_DETUMBLE:
            if mag_prev is not None:
                m = bdot_dipole(mag_k, mag_prev, T, k_bdot, mtq.m_max)
        else:
            b_corr = mag_k - bm_hat
            omega = wheel_speed(h, w, wheel)               # tachometers (truth)
            frac = float(np.max(np.abs(omega))) / wheel.max_speed
            dumping = ctrl.dumping and dump_active(dumping, frac, ctrl.dump_on, ctrl.dump_off)
            if dumping:
                m = dump_dipole(h, b_corr, ctrl.dump_k, mtq.m_max)
            tau, _ = pointing_torque(q_hat, w_hat, h, qref_all[k], wref_all[k], inertia_nom, ctrl.gains)
            tau_mag_ff = mtq.duty * _cross(m, b_corr)
            u_w = limit_wheel_torque(-(tau - tau_mag_ff), h, w, wheel, dt=T)
        mag_prev = mag_k
        q_prev = q.copy()

        tau_d = (total_disturbance_torque(r, v, q, b_eci, cfg.spacecraft)
                 if ctrl.disturbances else np.zeros(3))

        if k % log_every == 0:
            vec["q"][j], vec["w"][j], vec["h_w"][j] = q, w, h
            vec["wheel_speed"][j] = wheel_speed(h, w, wheel)
            vec["m"][j], vec["u_w"][j], vec["tau_dist"][j] = m, u_w, tau_d
            vec["tau_mag"][j] = _cross(m, _rot_inv(q, b_eci))
            vec["gyro_bias"][j] = gyro.bias
            if w_hat is not None:
                vec["w_est"][j] = w_hat
            sca["t"][j] = t
            dq = quat.multiply(quat.conjugate(qref_all[k]), q)
            sca["nadir_err"][j] = np.linalg.norm(quat.to_rotvec(dq))
            if f is not None:
                vec["q_est"][j], vec["b_est"][j], vec["bm_est"][j] = q_hat, b_hat, bm_hat
                vec["know_err"][j] = quat.error_rotvec(q_hat, q)
                sca["att_sigma"][j] = f.att_sigma
                log_P[j] = f.ekf._P[:3, :3]
                dqe = quat.multiply(quat.conjugate(qref_all[k]), q_hat)
                sca["est_nadir_err"][j] = np.linalg.norm(quat.to_rotvec(dqe))
            log_mode[j] = cur_mode
            log_dump[j] = dumping
            log_used[j] = used
            log_lit[j] = lit_all[k]
            j += 1
        if k == n_steps:
            break

        def ext_on(_t, qq, _w, _h, m=m, b_eci=b_eci, tau_d=tau_d):
            return _cross(m, _rot_inv(qq, b_eci)) + tau_d

        y = rk4_step(t, y, on_t, inertia, u_w, ext_on)
        if off_t > 0.0:
            y = rk4_step(t + on_t, y, off_t, inertia, u_w, tau_d)

    nis = {}
    if f is not None:
        for name in ("mag", "sun", "css"):
            lg = f.ekf.innovations(name)
            nis[name] = {"t": lg.t, "nis": lg.nis, "dof": lg.dof, "accepted": lg.accepted}
    meta = {
        "mode": "closed_loop",
        "seed": seed,
        "t_end": float(t_grid[-1]),
        "control_period": T,
        "bdot_k": k_bdot,
        "switch_time": switch_time,
        "switch_times": switch_times,
        "fallback_times": fallback_times,
        "t_init": float(t_grid[k_init]) if k_init >= 0 else float("nan"),
        "orbital_period": float(orbital_period(cfg.orbit.a)),
        "mag_bias_true": mag_bias,
        "inertia_true": inertia.copy(),
        "nis": nis,
        "ctrl": asdict(ctrl),
        "sensors": asdict(sensors),
        "mekf": asdict(mekf_cfg),
        "modes": asdict(modes),
        "mtq": asdict(mtq),
        "wheel": asdict(wheel) | {"max_speed": wheel.max_speed, "spin_inertia": wheel.spin_inertia},
    }
    sl = slice(0, j)
    return ClosedLoopResult(
        t=sca["t"][sl], q=vec["q"][sl], w=vec["w"][sl], h_w=vec["h_w"][sl],
        wheel_speed=vec["wheel_speed"][sl], m=vec["m"][sl], u_w=vec["u_w"][sl],
        tau_dist=vec["tau_dist"][sl], tau_mag=vec["tau_mag"][sl], nadir_err=sca["nadir_err"][sl],
        rate=np.linalg.norm(vec["w"][sl], axis=1), mode=log_mode[sl], dump=log_dump[sl], meta=meta,
        q_est=vec["q_est"][sl], b_est=vec["b_est"][sl], bm_est=vec["bm_est"][sl],
        att_sigma=sca["att_sigma"][sl], P_att=log_P[sl], know_err=vec["know_err"][sl],
        est_nadir_err=sca["est_nadir_err"][sl], w_est=vec["w_est"][sl],
        gyro_bias=vec["gyro_bias"][sl], sun_used=log_used[sl], lit=log_lit[sl])


def summarize_closed_loop(res: ClosedLoopResult, settle: float | None = None) -> dict:
    """Scalar metrics of a closed-loop run (degrees, seconds, orbits; JSON-ready).

    Steady-state window: mode == pointing and ``t >= first switch + settle``
    (default one orbit). Knowledge error = ``|know_err|``; NEES uses ``P_att``.
    Keys with no data are ``nan``. Definitions in ``docs/specs/adcs_closed_loop.md`` section 3.
    """
    m = res.meta
    P = m["orbital_period"]
    settle = P if settle is None else float(settle)
    deg = np.rad2deg
    nan = float("nan")
    sw = m["switch_time"]
    td = detumble_time(res.t, res.rate, m["ctrl"]["detumble_threshold"])
    out = {
        "t_end_s": m["t_end"],
        "t_init_s": m["t_init"],
        "switch_time_s": sw,
        "switch_time_orbits": sw / P,
        "detumble_time_s": td,
        "detumble_time_orbits": td / P,
        "n_switches": len(m["switch_times"]),
        "n_fallbacks": len(m["fallback_times"]),
    }
    win = (res.mode == MODE_POINTING) & (res.t >= sw + settle) if not np.isnan(sw) else np.zeros(res.t.size, bool)
    out["n_steady"] = int(win.sum())
    keys = ("nadir_rms_deg", "nadir_p95_deg", "nadir_p99_deg", "nadir_max_deg", "know_rms_deg",
            "know_rms_sunlit_deg", "know_rms_eclipse_deg", "know_max_deg", "ctrl_rms_deg", "ctrl_max_deg",
            "nees_mean", "wheel_max_frac", "dump_frac", "dipole_max")
    out.update({k: nan for k in keys})
    if win.any():
        e = deg(res.nadir_err[win])
        ke_vec = res.know_err[win]
        ke = deg(np.linalg.norm(ke_vec, axis=1))
        lit = res.lit[win]
        ce = deg(res.est_nadir_err[win])
        rms = lambda x: float(np.sqrt(np.mean(x ** 2))) if x.size else nan
        out.update(
            nadir_rms_deg=rms(e), nadir_p95_deg=float(np.percentile(e, 95)),
            nadir_p99_deg=float(np.percentile(e, 99)), nadir_max_deg=float(e.max()),
            know_rms_deg=rms(ke), know_rms_sunlit_deg=rms(ke[lit]), know_rms_eclipse_deg=rms(ke[~lit]),
            know_max_deg=float(ke.max()), ctrl_rms_deg=rms(ce), ctrl_max_deg=float(ce.max()),
            nees_mean=float(np.mean(nees(ke_vec, res.P_att[win]))),
            wheel_max_frac=float(np.max(np.abs(res.wheel_speed[win])) / m["wheel"]["max_speed"]),
            dump_frac=float(np.mean(res.dump[win])),
            dipole_max=float(np.max(np.abs(res.m[win]))))
    pt = res.mode == MODE_POINTING
    out["wheel_max_frac_pointing"] = (float(np.max(np.abs(res.wheel_speed[pt])) / m["wheel"]["max_speed"])
                                      if pt.any() else nan)
    ok = np.isfinite(res.b_est[:, 0])
    out["gyro_bias_err_end_degps"] = (float(deg(np.linalg.norm(res.b_est[ok][-1] - res.gyro_bias[ok][-1])))
                                      if ok.any() else nan)
    out["mag_bias_true_nT"] = float(np.linalg.norm(m["mag_bias_true"]) * 1e9)
    out["mag_bias_err_end_nT"] = (float(np.linalg.norm(res.bm_est[ok][-1] - m["mag_bias_true"]) * 1e9)
                                  if ok.any() else nan)
    t_ok = (m["t_init"] + 300.0) if np.isfinite(m["t_init"]) else np.inf
    for name, lg in m["nis"].items():
        sel = lg["accepted"] & (lg["t"] >= t_ok)
        out[f"nis_per_dof_{name}"] = float(np.mean(lg["nis"][sel] / lg["dof"][sel])) if sel.any() else nan
    out["success"] = bool(out["n_switches"] >= 1 and out["n_fallbacks"] == 0 and out["n_steady"] > 0)
    return out


def _summary(res: SimResult) -> str:
    P = res.meta["orbital_period"]
    lines = [f"mode={res.meta['mode']}  t_end={res.meta['t_end']:.0f} s ({res.meta['t_end'] / P:.2f} orbits)"]
    td = detumble_time(res.t, res.rate, ControlConfig().detumble_threshold)
    lines.append(f"rate: start {np.rad2deg(res.rate[0]):.3f} deg/s, end {np.rad2deg(res.rate[-1]):.4f} deg/s, "
                 f"below 0.5 deg/s from t = {td:.0f} s ({td / P:.2f} orbits)")
    pt = res.mode == MODE_POINTING
    if pt.any():
        t_sw = res.meta["switch_time"]
        late = pt & (res.t >= t_sw + P)
        err = np.rad2deg(res.nadir_err[late]) if late.any() else np.array([np.nan])
        frac = np.max(np.abs(res.wheel_speed[pt]), axis=1) / res.meta["wheel"]["max_speed"]
        lines.append(f"pointing from t = {t_sw:.0f} s; nadir error after 1 orbit settling: "
                     f"max {np.max(err):.4f} deg, rms {np.sqrt(np.mean(err**2)):.4f} deg")
        lines.append(f"wheel speed fraction max {np.max(frac):.3f}; dumping active {np.mean(res.dump[pt]):.1%}")
    return "\n".join(lines)


def _summary_closed_loop(res: ClosedLoopResult) -> str:
    sm = summarize_closed_loop(res)
    lines = [f"closed loop  t_end={sm['t_end_s']:.0f} s ({sm['t_end_s'] / res.meta['orbital_period']:.2f} orbits)  "
             f"sensors={res.meta['sensors_preset']}  filter={res.meta['mekf_preset']}",
             f"MEKF init t = {sm['t_init_s']:.0f} s; rate below 0.5 deg/s from {sm['detumble_time_s']:.0f} s "
             f"({sm['detumble_time_orbits']:.2f} orbits); pointing from {sm['switch_time_s']:.0f} s "
             f"({sm['switch_time_orbits']:.2f} orbits); fallbacks {sm['n_fallbacks']}"]
    if sm["n_steady"]:
        lines.append(f"steady state (switch + 1 orbit): nadir error rms {sm['nadir_rms_deg']:.3f} deg, "
                     f"p95 {sm['nadir_p95_deg']:.3f}, max {sm['nadir_max_deg']:.3f}; knowledge rms "
                     f"{sm['know_rms_deg']:.3f} deg (sunlit {sm['know_rms_sunlit_deg']:.3f}, eclipse "
                     f"{sm['know_rms_eclipse_deg']:.3f}); NEES {sm['nees_mean']:.2f}")
        lines.append(f"wheel speed fraction max {sm['wheel_max_frac']:.3f}; dumping active {sm['dump_frac']:.1%}; "
                     f"mag-bias error {sm['mag_bias_err_end_nT']:.1f} nT of {sm['mag_bias_true_nT']:.1f} nT")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description="ADCS simulation: closed loop on the estimate (default, Group 8) "
                                            "or perfect knowledge (Group 6).")
    p.add_argument("--knowledge", choices=("estimated", "perfect"), default="estimated")
    p.add_argument("--sensors", choices=("realistic", "ideal"), default="realistic",
                   help="closed loop: truth-side sensor errors (REALISTIC_SENSORS / IDEAL_SENSORS)")
    p.add_argument("--filter", choices=("design", "g7"), default="design",
                   help="closed loop: MEKF_DESIGN or MEKF_G7")
    p.add_argument("--scenario", choices=("detumble", "pointing", "auto"), default="auto",
                   help="perfect knowledge only")
    p.add_argument("--orbits", type=float, default=None)
    p.add_argument("--rate-deg", type=float, default=10.0, help="initial tumble rate (deg/s)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-dumping", action="store_true")
    a = p.parse_args(argv)
    P = float(orbital_period(DEFAULT.orbit.a))
    ctrl = ControlConfig(dumping=not a.no_dumping)
    if a.knowledge == "estimated":
        orbits = 6.0 if a.orbits is None else a.orbits
        sensors = REALISTIC_SENSORS if a.sensors == "realistic" else IDEAL_SENSORS
        mekf_cfg = MEKF_DESIGN if a.filter == "design" else MEKF_G7
        y0 = tumble_state(np.deg2rad(a.rate_deg), a.seed)
        res = run_closed_loop(y0, orbits * P, seed=a.seed, ctrl=ctrl, sensors=sensors, mekf_cfg=mekf_cfg)
        res.meta["sensors_preset"], res.meta["mekf_preset"] = a.sensors, a.filter
        print(_summary_closed_loop(res))
        return
    orbits = a.orbits if a.orbits is not None else {"detumble": 3, "pointing": 15, "auto": 6}[a.scenario]
    if a.scenario == "pointing":
        r0, v0 = initial_state(DEFAULT)
        y0 = lvlh_state(r0, v0, offset=np.deg2rad([20.0, -15.0, 30.0]), w_rel=np.deg2rad([0.1, -0.1, 0.2]))
    else:
        y0 = tumble_state(np.deg2rad(a.rate_deg), a.seed)
    res = run(y0, orbits * P, a.scenario, ctrl=ctrl, log_every=10)
    print(_summary(res))


if __name__ == "__main__":
    main()
