"""Group 5 physics figures and metrics (spec docs/specs/adcs_physics.md section 6).

Run:  .venv/Scripts/python -m adcs.report.physics_figures

Outputs
-------
``adcs/report/figures/{orbit_vs_sgp4,eclipse_and_field,disturbance_magnitudes,conservation_10orbits}.png``
and ``adcs/report/physics_metrics.json``. Every plotted/reported number is computed here from the
``adcs`` modules; nothing is tuned. Anomalies are reported, never corrected.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

from adcs import dynamics as dyn  # noqa: E402
from adcs.config import DEFAULT  # noqa: E402
from adcs.disturbances import disturbance_breakdown  # noqa: E402
from adcs.environment import (  # noqa: E402
    beta_angle,
    eclipse_fraction,
    magnetic_field_eci,
    sun_direction_eci,
    sunlit,
)
from adcs.orbit import (  # noqa: E402
    Ephemeris,
    initial_state,
    orbital_period,
    propagate,
    sgp4_ephemeris,
    specific_energy,
)
from common import quaternion as quat  # noqa: E402
from common.plotting import label_axes, save_fig  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parent
FIG_DIR = REPORT_DIR / "figures"
METRICS_PATH = REPORT_DIR / "physics_metrics.json"

# python-sgp4 documentation example (ISS-like), same TLE as adcs/tests/test_orbit_env.py
TLE1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
TLE2 = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472 20569"

DEG = np.pi / 180.0
SOURCES = ("gravity_gradient", "residual_dipole", "aero")
SOURCE_LABELS = {"gravity_gradient": "gravity gradient", "residual_dipole": "residual dipole",
                 "aero": "aerodynamic"}


def _summary(name: str, text: str) -> None:
    print(f"[{name}] {text}")


# ---------------------------------------------------------------------------------------------
# 1. orbit vs SGP4
# ---------------------------------------------------------------------------------------------
def orbit_vs_sgp4() -> dict:
    from sgp4.api import Satrec

    sat = Satrec.twoline2rv(TLE1, TLE2)
    jd0 = sat.jdsatepoch + sat.jdsatepochF
    period = 2.0 * np.pi / (sat.no_kozai / 60.0)           # s, from TLE mean motion
    t = np.arange(0.0, 86400.0 + 1e-9, 30.0)
    r_s, v_s = sgp4_ephemeris(TLE1, TLE2, jd0, t)
    r_o, _ = propagate(r_s[0], v_s[0], t, j2=True)

    d = r_o - r_s
    rh = r_s / np.linalg.norm(r_s, axis=1, keepdims=True)
    nh = np.cross(r_s, v_s)
    nh /= np.linalg.norm(nh, axis=1, keepdims=True)
    th = np.cross(nh, rh)
    rad, alo, crs = (np.sum(d * u, axis=1) for u in (rh, th, nh))
    dn = np.linalg.norm(d, axis=1)
    dradius = np.linalg.norm(r_o, axis=1) - np.linalg.norm(r_s, axis=1)

    one = t <= period
    fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6.5))
    th_h = t / 3600.0
    a1.plot(th_h, dn / 1e3, "k", lw=1.6, label="|Δr|")
    a1.plot(th_h, np.abs(rad) / 1e3, lw=0.9, label="|radial|")
    a1.plot(th_h, np.abs(alo) / 1e3, lw=0.9, label="|along-track|")
    a1.plot(th_h, np.abs(crs) / 1e3, lw=0.9, label="|cross-track|")
    a1.axvline(period / 3600.0, color="gray", ls=":", label="end of orbit 1")
    a1.legend(ncol=5, fontsize=8, loc="upper left")
    label_axes(a1, "", "position difference magnitude [km]",
               "ISS TLE (epoch 2019-12-09): own J2 propagation vs SGP4, 1 day")
    a2.plot(th_h, rad / 1e3, label="radial")
    a2.plot(th_h, alo / 1e3, label="along-track")
    a2.plot(th_h, crs / 1e3, label="cross-track")
    a2.plot(th_h, dradius / 1e3, "k--", lw=0.9, label="|r_ours| - |r_sgp4|")
    a2.legend(ncol=4, fontsize=8, loc="upper left")
    label_axes(a2, "time since TLE epoch [h]", "signed difference, RTN of SGP4 orbit [km]")
    save_fig(fig, FIG_DIR / "orbit_vs_sgp4.png")

    out = {
        "tle_line1": TLE1,
        "tle_line2": TLE2,
        "period_s": float(period),
        "sgp4_radius_m": float(np.linalg.norm(r_s[0])),
        "sgp4_speed_m_s": float(np.linalg.norm(v_s[0])),
        "max_dr_first_orbit_m": float(dn[one].max()),
        "max_dr_one_day_m": float(dn.max()),
        "final_dr_one_day_m": float(dn[-1]),
        "max_abs_radial_one_day_m": float(np.abs(rad).max()),
        "max_abs_along_track_one_day_m": float(np.abs(alo).max()),
        "max_abs_cross_track_one_day_m": float(np.abs(crs).max()),
        "max_abs_radius_difference_one_day_m": float(np.abs(dradius).max()),
        "note": "difference expected to be dominated by drag (B* = 3.88e-5), J3/J4 and osculating vs "
                "mean-element mismatch; own model is two-body + J2 only",
    }
    _summary("orbit_vs_sgp4",
             f"max |dr| = {out['max_dr_first_orbit_m'] / 1e3:.3f} km (1 orbit), "
             f"{out['max_dr_one_day_m'] / 1e3:.3f} km (1 day); max |d radius| = "
             f"{out['max_abs_radius_difference_one_day_m'] / 1e3:.3f} km")
    return out


# ---------------------------------------------------------------------------------------------
# shared default-config trajectory over 3 orbits
# ---------------------------------------------------------------------------------------------
def _default_orbit(n_orbits: int, dt: float = 10.0):
    cfg = DEFAULT
    r0, v0 = initial_state(cfg)
    period = float(orbital_period(cfg.orbit.a))
    t_end = n_orbits * period
    eph = Ephemeris(r0, v0, t_end)
    t = np.arange(0.0, t_end + 1e-9, dt)
    t = t[t <= t_end]
    r, v = eph.at(t)
    return cfg, period, t, r, v


def eclipse_and_field() -> dict:
    cfg, period, t, r, v = _default_orbit(3)
    jd = cfg.jd_epoch + t / 86400.0
    s = sun_direction_eci(jd)
    lit = sunlit(r, s)
    b = magnetic_field_eci(r, jd, cfg.epoch)
    bmag = np.linalg.norm(b, axis=1) * 1e6                 # uT
    beta = beta_angle(r, v, s)

    # 1-day eclipse fraction (the plan's gate) for reference
    t_day = np.arange(0.0, 86400.0 + 1e-9, 10.0)
    eph_day = Ephemeris(*initial_state(cfg), 86400.0)
    r_d, v_d = eph_day.at(t_day)
    s_d = sun_direction_eci(cfg.jd_epoch + t_day / 86400.0)
    beta_d = beta_angle(r_d, v_d, s_d)

    fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6),
                                 gridspec_kw={"height_ratios": [1, 2]})
    tm = t / 60.0
    a1.fill_between(tm, 0, 1, where=~lit, step="post", color="0.25", alpha=0.6, label="eclipse")
    a1.step(tm, lit.astype(float), where="post", color="C1", lw=1.2, label="sunlit flag")
    a1.set_yticks([0, 1])
    a1.set_yticklabels(["shadow", "sunlit"])
    a1.set_ylim(-0.1, 1.15)
    a1.legend(loc="center right", fontsize=8)
    label_axes(a1, "", "illumination [-]",
               f"Default 3U orbit (400 km, 51.6 deg, epoch {cfg.epoch:%Y-%m-%d %H:%M} UTC), 3 orbits")
    a2.plot(tm, bmag, "C0")
    for k in range(1, 3):
        a2.axvline(k * period / 60.0, color="gray", ls=":", lw=0.8)
    label_axes(a2, "time since epoch [min]", "|B| (IGRF) [µT]")
    save_fig(fig, FIG_DIR / "eclipse_and_field.png")

    out = {
        "n_orbits": 3,
        "period_s": period,
        "eclipse_fraction_3_orbits": eclipse_fraction(r, s),
        "eclipse_fraction_1_day": eclipse_fraction(r_d, s_d),
        "B_min_uT": float(bmag.min()),
        "B_mean_uT": float(bmag.mean()),
        "B_max_uT": float(bmag.max()),
        "beta_deg_at_epoch": float(beta[0] / DEG),
        "beta_deg_min_3_orbits": float(beta.min() / DEG),
        "beta_deg_max_3_orbits": float(beta.max() / DEG),
        "beta_deg_min_1_day": float(beta_d.min() / DEG),
        "beta_deg_max_1_day": float(beta_d.max() / DEG),
        "analytic_eclipse_fraction_beta0": float(np.arcsin(6378137.0 / cfg.orbit.a) / np.pi),
    }
    _summary("eclipse_and_field",
             f"eclipse fraction {out['eclipse_fraction_3_orbits']:.4f} (3 orbits), "
             f"{out['eclipse_fraction_1_day']:.4f} (1 day); |B| min/mean/max = "
             f"{out['B_min_uT']:.2f}/{out['B_mean_uT']:.2f}/{out['B_max_uT']:.2f} uT; "
             f"beta(epoch) = {out['beta_deg_at_epoch']:.3f} deg")
    return out


# ---------------------------------------------------------------------------------------------
# disturbances
# ---------------------------------------------------------------------------------------------
def disturbance_magnitudes() -> dict:
    cfg, period, t, r, v = _default_orbit(3)
    jd = cfg.jd_epoch + t / 86400.0
    b = magnetic_field_eci(r, jd, cfg.epoch)
    q_lvlh = quat.from_dcm(np.swapaxes(_dcm_lvlh(r, v), -1, -2))

    axes = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
    cases = {"nadir-aligned": None}
    for ang in (10, 20):
        for ax in axes:
            cases[f"+{ang} deg about {ax}"] = (ax, ang)

    results: dict = {}
    series: dict = {}
    for name, spec in cases.items():
        if spec is None:
            q = q_lvlh
        else:
            ax, ang = spec
            dq = quat.from_axis_angle(axes[ax], ang * DEG)
            q = quat.multiply(q_lvlh, np.broadcast_to(dq, q_lvlh.shape))   # q_lvlh (x) exp(err), body-frame offset
        br = disturbance_breakdown(r, v, q, b, cfg.spacecraft)
        entry = {}
        ser = {}
        for src in SOURCES:
            mag = np.linalg.norm(br[src], axis=-1)
            ser[src] = mag
            entry[src] = {"rms_Nm": float(np.sqrt(np.mean(mag**2))), "max_Nm": float(mag.max())}
        rms = {s_: entry[s_]["rms_Nm"] for s_ in SOURCES}
        top = max(rms, key=rms.get)
        entry["largest_source_by_rms"] = top
        results[name] = entry
        series[name] = ser

    # statements generated from the numbers
    per_att = {n: e["largest_source_by_rms"] for n, e in results.items()}
    counts = {s_: sum(1 for x in per_att.values() if x == s_) for s_ in SOURCES}
    lines = []
    for n, e in results.items():
        rms = {s_: e[s_]["rms_Nm"] for s_ in SOURCES}
        order = sorted(SOURCES, key=lambda k: -rms[k])
        lines.append(f"{n}: largest orbit-RMS = {SOURCE_LABELS[order[0]]} ({rms[order[0]]:.3e} N m), "
                     f"then {SOURCE_LABELS[order[1]]} ({rms[order[1]]:.3e}), "
                     f"{SOURCE_LABELS[order[2]]} ({rms[order[2]]:.3e})")
    statement = (" | ".join(f"{SOURCE_LABELS[s_]} is the largest orbit-RMS source in {counts[s_]} of "
                            f"{len(per_att)} attitudes" for s_ in SOURCES))

    # figure: nadir-aligned and +10 deg about y (and the other +10 deg offsets as thin lines)
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(9, 7))
    tm = t / 60.0
    floor = 1e-14
    for ax_, name in zip(axs, ("nadir-aligned", "+10 deg about y")):
        for k, src in enumerate(SOURCES):
            ax_.semilogy(tm, np.maximum(series[name][src], floor), color=f"C{k}", lw=1.3,
                         label=SOURCE_LABELS[src])
        for k in range(1, 3):
            ax_.axvline(k * period / 60.0, color="gray", ls=":", lw=0.8)
        ax_.set_ylim(floor, 1e-5)
        label_axes(ax_, "", "torque magnitude [N·m]",
                   f"Disturbance torques, default 3U, 3 orbits, attitude: {name} (LVLH target)")
        ax_.legend(fontsize=8, ncol=3, loc="lower right")
    axs[-1].set_xlabel("time since epoch [min]")
    fig.text(0.01, 0.005, f"values below {floor:.0e} N·m are clipped to the axis floor", fontsize=7)
    save_fig(fig, FIG_DIR / "disturbance_magnitudes.png")

    out = {
        "n_orbits": 3,
        "attitudes": results,
        "largest_source_by_rms_per_attitude": per_att,
        "largest_source_statement": statement,
        "per_attitude_ranking": lines,
        "gate_range_Nm": [1e-10, 1e-6],
        "values_outside_gate": [
            f"{n}/{s_}/{k}" for n, e in results.items() for s_ in SOURCES
            for k in ("rms_Nm", "max_Nm") if not (1e-10 <= e[s_][k] <= 1e-6)],
    }
    _summary("disturbance_magnitudes", statement)
    for n in ("nadir-aligned", "+10 deg about y"):
        e = results[n]
        _summary("disturbance_magnitudes",
                 f"{n}: RMS gg/dip/aero = " + "/".join(f"{e[s_]['rms_Nm']:.2e}" for s_ in SOURCES) + " N m")
    return out


def _dcm_lvlh(r, v):
    from common.frames import dcm_eci_to_lvlh
    return dcm_eci_to_lvlh(r, v)


# ---------------------------------------------------------------------------------------------
# conservation over 10 orbits
# ---------------------------------------------------------------------------------------------
def conservation_10orbits() -> dict:
    cfg = DEFAULT
    inertia = cfg.spacecraft.inertia
    period = float(orbital_period(cfg.orbit.a))
    t_end = 10.0 * period
    dt = 10.0
    t = np.arange(0.0, t_end + 1e-9, dt)
    t = t[t <= t_end]

    # --- free tumble (test 11) ---
    y0 = np.concatenate([[1.0, 0.0, 0.0, 0.0], [0.05, 0.12, 0.03], np.zeros(3)])
    t0 = time.perf_counter()
    y = dyn.integrate(y0, t, inertia, rtol=1e-11)
    t_int = time.perf_counter() - t0
    # raw (un-renormalised) quaternion norm: same call as dyn.integrate, before its output renormalisation
    t0 = time.perf_counter()
    sol = solve_ivp(lambda tt, yy: dyn.derivative(tt, yy, inertia, None, None), (t[0], t[-1]), y0,
                    method="DOP853", t_eval=t, rtol=1e-11, atol=1e-12)
    t_raw = time.perf_counter() - t0
    qn_raw = np.linalg.norm(sol.y.T[:, :4], axis=1)
    raw_matches_integrate = bool(np.allclose(quat.normalize(sol.y.T[:, :4]), y[:, :4], atol=1e-15, rtol=0))

    h = dyn.angular_momentum_inertial(y, inertia)
    hn = np.linalg.norm(h, axis=1)
    h_norm_drift = hn / hn[0] - 1.0
    h_vec_drift = np.linalg.norm(h - h[0], axis=1) / hn[0]
    tk = dyn.kinetic_energy(y, inertia)
    t_drift = tk / tk[0] - 1.0
    q_drift = qn_raw - 1.0

    # --- J2 orbit (test 3): energy and h_z ---
    r0, v0 = initial_state(cfg)
    rr, vv = propagate(r0, v0, t, j2=True)
    e = specific_energy(rr, vv, j2=True)
    hz = np.cross(rr, vv)[:, 2]
    e_drift = e / e[0] - 1.0
    hz_drift = hz / hz[0] - 1.0

    fl = 1e-18
    fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
    th = t / 3600.0
    a1.semilogy(th, np.maximum(np.abs(h_norm_drift), fl), label="||H_i|| / ||H_i(0)|| - 1")
    a1.semilogy(th, np.maximum(h_vec_drift, fl), label="||H_i - H_i(0)|| / ||H_i(0)||")
    a1.semilogy(th, np.maximum(np.abs(t_drift), fl), label="T / T(0) - 1")
    a1.semilogy(th, np.maximum(np.abs(q_drift), fl), label="||q|| - 1 (raw DOP853 output)")
    a1.legend(fontsize=8, ncol=2, loc="lower right")
    label_axes(a1, "", "|relative drift| [-]",
               "Torque-free asymmetric tumble, 3U inertia, 10 orbits, DOP853 rtol = 1e-11")
    a2.semilogy(th, np.maximum(np.abs(e_drift), fl), label="specific energy (J2 potential)")
    a2.semilogy(th, np.maximum(np.abs(hz_drift), fl), label="angular momentum z-component h_z")
    a2.legend(fontsize=8, loc="lower right")
    label_axes(a2, "time since epoch [h]", "|relative drift| [-]",
               "J2 orbit propagation, 10 orbits, DOP853 rtol = 1e-12 (values below 1e-18 clipped)")
    save_fig(fig, FIG_DIR / "conservation_10orbits.png")

    out = {
        "n_orbits": 10,
        "t_end_s": float(t_end),
        "output_step_s": dt,
        "tumble_omega0_rad_s": [0.05, 0.12, 0.03],
        "tumble_rtol": 1e-11,
        "H_i_norm_final_drift": float(h_norm_drift[-1]),
        "H_i_norm_max_abs_drift": float(np.abs(h_norm_drift).max()),
        "H_i_vector_final_drift": float(h_vec_drift[-1]),
        "H_i_vector_max_drift": float(h_vec_drift.max()),
        "T_final_drift": float(t_drift[-1]),
        "T_max_abs_drift": float(np.abs(t_drift).max()),
        "q_norm_raw_final_drift": float(q_drift[-1]),
        "q_norm_raw_max_abs_drift": float(np.abs(q_drift).max()),
        "q_norm_after_integrate_max_abs_drift": float(np.abs(np.linalg.norm(y[:, :4], axis=1) - 1).max()),
        "raw_solve_matches_integrate_output": raw_matches_integrate,
        "j2_energy_final_drift": float(e_drift[-1]),
        "j2_energy_max_abs_drift": float(np.abs(e_drift).max()),
        "j2_hz_final_drift": float(hz_drift[-1]),
        "j2_hz_max_abs_drift": float(np.abs(hz_drift).max()),
        "runtime_integrate_s": t_int,
        "runtime_raw_solve_s": t_raw,
    }
    _summary("conservation_10orbits",
             f"tumble |H| drift {out['H_i_norm_max_abs_drift']:.2e}, H vector {out['H_i_vector_max_drift']:.2e}, "
             f"T {out['T_max_abs_drift']:.2e}, |q|-1 raw {out['q_norm_raw_max_abs_drift']:.2e}; "
             f"J2 energy {out['j2_energy_max_abs_drift']:.2e}, h_z {out['j2_hz_max_abs_drift']:.2e}")
    return out


def main() -> None:
    t_start = time.perf_counter()
    metrics: dict = {"config": {
        "epoch_utc": DEFAULT.epoch.isoformat(),
        "altitude_m": DEFAULT.orbit.altitude,
        "inclination_deg": DEFAULT.orbit.inc / DEG,
        "mass_kg": DEFAULT.spacecraft.mass,
        "inertia_diag_kg_m2": np.diag(DEFAULT.spacecraft.inertia).tolist(),
        "residual_dipole_A_m2": list(DEFAULT.spacecraft.residual_dipole_b),
        "cd": DEFAULT.spacecraft.cd,
        "com_offset_b_m": list(DEFAULT.spacecraft.com_offset_b),
    }}
    timings = {}
    for name, fn in (("orbit_vs_sgp4", orbit_vs_sgp4), ("eclipse_and_field", eclipse_and_field),
                     ("disturbance_magnitudes", disturbance_magnitudes),
                     ("conservation_10orbits", conservation_10orbits)):
        t0 = time.perf_counter()
        metrics[name] = fn()
        timings[name] = time.perf_counter() - t0
    metrics["runtime_s"] = {**timings, "total": time.perf_counter() - t_start}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"wrote {METRICS_PATH} (total {metrics['runtime_s']['total']:.1f} s)")


if __name__ == "__main__":
    main()
