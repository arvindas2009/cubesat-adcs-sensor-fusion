"""Attitude control simulation for the 3U CubeSat (Project B).

Group 6 (M7): **perfect knowledge** - the controller sees the true attitude,
rate, wheel momentum and magnetic field. Group 8 adds sensors and the MEKF.
Spec: ``docs/specs/adcs_control.md`` section 3.

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
from adcs.environment import FieldTable
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


def main(argv=None):
    p = argparse.ArgumentParser(description="Perfect-knowledge ADCS simulation (Group 6).")
    p.add_argument("--scenario", choices=("detumble", "pointing", "auto"), default="auto")
    p.add_argument("--orbits", type=float, default=None)
    p.add_argument("--rate-deg", type=float, default=10.0, help="initial tumble rate (deg/s)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-dumping", action="store_true")
    a = p.parse_args(argv)
    P = float(orbital_period(DEFAULT.orbit.a))
    orbits = a.orbits if a.orbits is not None else {"detumble": 3, "pointing": 15, "auto": 6}[a.scenario]
    if a.scenario == "pointing":
        r0, v0 = initial_state(DEFAULT)
        y0 = lvlh_state(r0, v0, offset=np.deg2rad([20.0, -15.0, 30.0]), w_rel=np.deg2rad([0.1, -0.1, 0.2]))
    else:
        y0 = tumble_state(np.deg2rad(a.rate_deg), a.seed)
    res = run(y0, orbits * P, a.scenario, ctrl=ControlConfig(dumping=not a.no_dumping), log_every=10)
    print(_summary(res))


if __name__ == "__main__":
    main()
