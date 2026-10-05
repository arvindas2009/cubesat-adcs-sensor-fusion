"""Project A drift study: pure raw strapdown dead reckoning vs PX4 EKF2 (spec section 5).

Run:  .venv/Scripts/python -m sensor_fusion.report.dr_study

For each log in ``PINNED_LOGS`` this script

1. loads the log (``load_log``, cropped to the armed window +/- 2 s);
2. **initialises** the strapdown from EKF2's own state at the first IMU sample:
   attitude ``q0`` = slerp of ``vehicle_attitude.q`` at ``imu.t[0]``, velocity ``v0`` and
   position ``p0`` = linear interpolation of ``vehicle_local_position`` (NED, relative to
   EKF2's reference point) at ``imu.t[0]`` (``interp_to`` holds the end value if EKF2's
   first sample lies slightly after the first IMU sample);
3. integrates the **raw** IMU over the whole flight window -- no bias removal, no
   aiding, no leveling, flat non-rotating Earth, per-log local WGS-84 normal gravity
   ``g = normal_gravity(ekf2.ref_lat, ref_alt + median(gps.alt - gps.alt_msl))`` at the EKF2
   origin, ``g_ned = (0, 0, g)`` (the G0 = 9.80665 baseline is also run, for ``g_effect``);
   a SECOND run is re-seeded from EKF2 at (first EKF2 quaternion reset + 1 s) and reported
   under ``post_reset`` so the yaw-reset step is not attributed to drift;
4. compares to EKF2 **on the EKF2 grids**: the dead-reckoned position/velocity are
   linearly interpolated onto ``ekf2.t_pos``, the attitude is slerped onto ``ekf2.t_att``.
   Errors are position/velocity vector norms and the principal attitude angle
   (``common.quaternion.angle_between``).

EKF2 is the reference, NOT ground truth. Anomalies are reported, never corrected.
Outputs: ``sensor_fusion/report/dr_metrics_<id>.json`` and
``sensor_fusion/report/figures/dr_<id>_{poserr,atterr,traj}.png``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import quaternion as quat  # noqa: E402
from common.frames import G0, normal_gravity  # noqa: E402
from common.plotting import deg, label_axes, save_fig  # noqa: E402
from sensor_fusion.dead_reckoning import strapdown  # noqa: E402
from sensor_fusion.download_logs import PINNED_LOGS  # noqa: E402
from sensor_fusion.interp import interp_to  # noqa: E402
from sensor_fusion.loader import load_log  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parent
FIG_DIR = REPORT_DIR / "figures"
CHECKPOINTS_S = (1.0, 5.0, 10.0, 30.0, 60.0)
FIT_WINDOW_S = (2.0, 30.0)
CAVEAT = "EKF2 is the reference, not ground truth"


def _at(t_query: float, t: np.ndarray, y: np.ndarray):
    """Value of error curve y(t) at t_query (linear interp); None if outside the data span."""
    if t_query < t[0] or t_query > t[-1]:
        return None
    return float(np.interp(t_query, t, y))


def _checkpoints(t_rel, err, scale=1.0) -> dict:
    out = {f"{int(c)}s": (None if (v := _at(c, t_rel, err)) is None else v * scale) for c in CHECKPOINTS_S}
    out["end"] = float(err[-1]) * scale
    out["end_time_s"] = float(t_rel[-1])
    out["max"] = float(np.max(err)) * scale
    out["t_of_max_s"] = float(t_rel[int(np.argmax(err))])
    return out


def _growth_exponent(t_rel, err) -> dict:
    lo, hi = FIT_WINDOW_S
    m = (t_rel >= lo) & (t_rel <= hi) & (err > 0)
    if m.sum() < 3:
        return {"exponent": None, "intercept_log10": None, "n_points": int(m.sum()), "window_s": list(FIT_WINDOW_S)}
    x, y = np.log10(t_rel[m]), np.log10(err[m])
    slope, icpt = np.polyfit(x, y, 1)
    resid = y - (slope * x + icpt)
    return {"exponent": float(slope), "intercept_log10": float(icpt), "n_points": int(m.sum()),
            "rms_resid_log10": float(np.sqrt(np.mean(resid ** 2))), "window_s": list(FIT_WINDOW_S)}


def _window_run(log, t_start: float, g: float, tag: str) -> dict:
    """Strapdown from the first IMU sample at/after ``t_start``, seeded from EKF2 there.

    Returns metrics (``m``), arrays for the figures (``arr``) and the gravity-effect table.
    """
    imu, ek = log.imu, log.ekf2
    i0 = int(np.searchsorted(imu.t, t_start, side="left"))
    t_s = float(imu.t[i0])  # actual seed time (first IMU sample at/after t_start)
    t_end = float(imu.t[-1])
    q0 = interp_to([t_s], ek.t_att, ek.q, quaternion=True)[0]
    v0 = interp_to([t_s], ek.t_pos, ek.vel_ned)[0]
    p0 = interp_to([t_s], ek.t_pos, ek.pos_ned)[0]

    sl = slice(i0, None)
    dr = strapdown(imu.t[sl], imu.gyro[sl], imu.accel[sl], q0, v0=v0, p0=p0, g_ned=(0.0, 0.0, g))
    finite = {k: bool(np.all(np.isfinite(getattr(dr, k)))) for k in ("q", "v", "p")}
    dr_g0 = strapdown(imu.t[sl], imu.gyro[sl], imu.accel[sl], q0, v0=v0, p0=p0, g_ned=(0.0, 0.0, G0))

    mp = (ek.t_pos >= t_s) & (ek.t_pos <= t_end)
    ma = (ek.t_att >= t_s) & (ek.t_att <= t_end)
    tp, ta = ek.t_pos[mp], ek.t_att[ma]
    p_dr = interp_to(tp, dr.t, dr.p)
    v_dr = interp_to(tp, dr.t, dr.v)
    q_dr = interp_to(ta, dr.t, dr.q, quaternion=True)
    pos_err = np.linalg.norm(p_dr - ek.pos_ned[mp], axis=1)
    vel_err = np.linalg.norm(v_dr - ek.vel_ned[mp], axis=1)
    att_err = quat.angle_between(ek.q[ma], q_dr)
    tp_rel, ta_rel = tp - t_s, ta - t_s

    p_g0 = interp_to(tp, dr_g0.t, dr_g0.p)
    pos_err_g0 = np.linalg.norm(p_g0 - ek.pos_ned[mp], axis=1)
    dg = g - G0
    g_effect = {"g_local_m_s2": g, "g0_m_s2": G0, "delta_g_m_s2": dg, "checkpoints_s": list(CHECKPOINTS_S[2:]),
                "note": "pos_err = |p_DR - p_EKF2|; dr_position_difference = |p_DR(G0) - p_DR(g_local)| (exactly the "
                        "gravity effect on the dead-reckoned position); predicted = 0.5*|dg|*t^2",
                "entries": {}}
    for c in CHECKPOINTS_S[2:]:
        e_loc, e_g0 = _at(c, tp_rel, pos_err), _at(c, tp_rel, pos_err_g0)
        d = _at(c, tp_rel, np.linalg.norm(p_g0 - p_dr, axis=1))
        g_effect["entries"][f"{int(c)}s"] = {
            "pos_err_g_local_m": e_loc, "pos_err_G0_m": e_g0,
            "pos_err_change_m": None if e_loc is None else e_g0 - e_loc,
            "dr_position_difference_m": d, "predicted_half_dg_t2_m": 0.5 * abs(dg) * c * c}

    m = {
        "tag": tag,
        "init": {"source": "EKF2 attitude/velocity/position interpolated at the seed time (raw IMU, no biases removed)",
                 "t_start_s": t_s, "ekf2_att_first_t_s": float(ek.t_att[0]), "ekf2_pos_first_t_s": float(ek.t_pos[0]),
                 "q0": q0.tolist(), "v0_ned_m_s": v0.tolist(), "p0_ned_m": p0.tolist()},
        "g_ned_m_s2": [0.0, 0.0, g],
        "duration_s": float(t_end - t_s),
        "n_ekf2_pos_samples_compared": int(mp.sum()),
        "n_ekf2_att_samples_compared": int(ma.sum()),
        "all_finite": finite,
        "position_error_m": _checkpoints(tp_rel, pos_err),
        "velocity_error_m_s": _checkpoints(tp_rel, vel_err),
        "attitude_error_deg": _checkpoints(ta_rel, att_err, scale=180.0 / np.pi),
        "position_growth_fit": _growth_exponent(tp_rel, pos_err),
        "final_position_dr_ned_m": dr.p[-1].tolist(),
        "final_position_ekf2_ned_m": ek.pos_ned[mp][-1].tolist(),
        "dr_max_speed_m_s": float(np.max(np.linalg.norm(dr.v, axis=1))),
        "ekf2_max_speed_m_s": float(np.max(np.linalg.norm(ek.vel_ned[mp], axis=1))),
    }
    arr = dict(t_s=t_s, tp=tp, ta=ta, tp_rel=tp_rel, ta_rel=ta_rel, pos_err=pos_err, att_err=att_err,
               p_dr=p_dr, mp=mp)
    return {"m": m, "arr": arr, "g_effect": g_effect}


def run_log(entry: dict) -> dict:
    log_id, role = entry["id"], entry["role"]
    log = load_log(log_id)
    imu, ek = log.imu, log.ekf2

    # ---- per-log local gravity at the EKF2 origin (ellipsoidal height = MSL ref_alt + geoid separation)
    geoid_sep = float(np.median(log.gps.alt - log.gps.alt_msl))
    h0_ell = float(ek.ref_alt) + geoid_sep
    g = float(normal_gravity(float(ek.ref_lat), h0_ell))

    # ---- run 1: from EKF2 state at the first IMU sample (documented above)
    t_start = float(imu.t[0])
    r1 = _window_run(log, t_start, g, "full")
    dr_m, A = r1["m"], r1["arr"]
    tp, ta, tp_rel, ta_rel = A["tp"], A["ta"], A["tp_rel"], A["ta_rel"]
    pos_err, att_err, p_dr, mp = A["pos_err"], A["att_err"], A["p_dr"], A["mp"]
    t_end = float(imu.t[-1])

    # ---- EKF2 quaternion resets falling inside the run, and the error step across each
    resets = [float(x) for x in log.info.get("ekf2_quat_reset_times_s", []) if t_start <= x <= t_end]
    reset_effects = []
    for tr in resets:
        r = tr - t_start
        b, a = _at(r - 0.5, ta_rel, att_err), _at(r + 0.5, ta_rel, att_err)
        reset_effects.append({"t_s": tr, "att_err_deg_before_0p5s": None if b is None else float(np.degrees(b)),
                              "att_err_deg_after_0p5s": None if a is None else float(np.degrees(a))})

    # ---- run 2: re-seeded from EKF2 at (first reset + 1 s) so the yaw-reset step is not counted as drift
    post, A2, r2 = None, None, None
    if resets:
        r2 = _window_run(log, resets[0] + 1.0, g, "post_reset")
        post, A2 = r2["m"], r2["arr"]
        post["reset_time_s"] = resets[0]
        post["seed_offset_after_reset_s"] = A2["t_s"] - resets[0]
        post["further_resets_in_window_s"] = [x for x in resets if x > A2["t_s"]]

    # ---- context for anomalies: when is the vehicle most aggressive?
    wn = np.linalg.norm(imu.gyro, axis=1)
    i_w = int(np.argmax(wn))
    t_rel_imu = imu.t - t_start
    t_pk = float(t_rel_imu[i_w])
    peak_event = {"t_peak_gyro_s": t_pk, "peak_gyro_deg_s": float(np.degrees(wn[i_w])), "samples": []}
    for dtau in (-5.0, -1.0, 1.0, 5.0):
        pe, ae = _at(t_pk + dtau, tp_rel, pos_err), _at(t_pk + dtau, ta_rel, att_err)
        peak_event["samples"].append({"offset_s": dtau, "pos_err_m": pe,
                                      "att_err_deg": None if ae is None else float(np.degrees(ae))})

    metrics = {
        "log_id": log_id,
        "role": role,
        "reference": CAVEAT,
        "gravity": {"ref_lat_rad": float(ek.ref_lat), "ref_lon_rad": float(ek.ref_lon),
                    "ref_alt_msl_m": float(ek.ref_alt), "median_geoid_separation_m": geoid_sep,
                    "ellipsoidal_height_m": h0_ell, "g_local_m_s2": g, "G0_baseline_m_s2": G0,
                    "delta_g_m_s2": g - G0,
                    "source": "common.frames.normal_gravity(ref_lat, ref_alt + median(gps.alt - gps.alt_msl))"},
        "imu_rate_hz": float(imu.rate_hz),
        "imu_source": log.info.get("imu_source"),
        "n_imu_samples": int(imu.t.size),
        "imu_max_abs_gyro_deg_s": float(np.degrees(wn[i_w])),
        "imu_t_of_max_gyro_s": float(t_rel_imu[i_w]),
        "imu_clipped_samples": log.info.get("imu_clipped_samples"),
        "ekf2_quat_reset_times_s": resets,
        "ekf2_quat_reset_effects": reset_effects,
        "ekf2_local_pos_resets": log.info.get("ekf2_local_pos_resets"),
        "ekf2_ref_changes": log.info.get("ekf2_ref_changes"),
        "peak_gyro_event": peak_event,
        "g_effect": r1["g_effect"],
        "g_effect_post_reset": None if r2 is None else r2["g_effect"],
        "post_reset": post,
    }
    # run-1 results stay at the top level (same keys as before)
    for k in ("init", "duration_s", "n_ekf2_pos_samples_compared", "n_ekf2_att_samples_compared", "all_finite",
              "position_error_m", "velocity_error_m_s", "attitude_error_deg", "position_growth_fit",
              "final_position_dr_ned_m", "final_position_ekf2_ned_m", "dr_max_speed_m_s", "ekf2_max_speed_m_s"):
        metrics[k] = dr_m[k]
    (REPORT_DIR / f"dr_metrics_{log_id}.json").write_text(json.dumps(metrics, indent=2))

    # ------------------------------------------------------------------ figures
    ttl = (f"Raw strapdown dead reckoning vs EKF2, {role} flight {log_id[:8]} "
           f"({imu.rate_hz:.0f} Hz IMU, {metrics['duration_s']:.0f} s)")
    cap = f"{CAVEAT}. Initialised from EKF2 at t = 0, no bias removal, local g = {g:.4f} m/s$^2$."

    def mark_resets(ax):
        for k, tr in enumerate(resets):
            ax.axvline(tr - t_start, color="tab:red", ls="--", lw=1.2, label="EKF2 quaternion reset" if k == 0 else None)

    def overlay_post(ax, key_t, key_e, scale=1.0):
        """Post-reset run, plotted on the same time axis (time since the start of run 1)."""
        if A2 is None:
            return
        t = A2[key_t] + (A2["t_s"] - t_start)
        e = A2[key_e] * scale
        m2 = (A2[key_t] > 0) & (e > 0)
        ax.semilogy(t[m2], e[m2], lw=1.0, color="tab:green", label="re-seeded at reset + 1 s")

    # position error: linear time / log-y and log-log with fit
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    m = (tp_rel > 0) & (pos_err > 0)
    a1.semilogy(tp_rel[m], pos_err[m], lw=1.0, label="|p_DR - p_EKF2| (seeded at t = 0)")
    overlay_post(a1, "tp_rel", "pos_err")
    mark_resets(a1)
    label_axes(a1, "time since start of dead reckoning [s]", "3-D position error [m]")
    a1.legend(loc="lower right")
    a2.loglog(tp_rel[m], pos_err[m], lw=1.0, label="|p_DR - p_EKF2|")
    fit = metrics["position_growth_fit"]
    if fit["exponent"] is not None:
        xs = np.array(FIT_WINDOW_S)
        a2.loglog(xs, 10 ** fit["intercept_log10"] * xs ** fit["exponent"], "k-", lw=2.5, alpha=0.7,
                  label=f"fit 2-30 s: t^{fit['exponent']:.2f}")
    a2.axvspan(*FIT_WINDOW_S, color="0.85", zorder=0)
    label_axes(a2, "time since start of dead reckoning [s]", "3-D position error [m]")
    a2.legend(loc="lower right")
    fig.suptitle(f"{ttl}\nPosition error. {cap}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_fig(fig, FIG_DIR / f"dr_{log_id}_poserr.png")

    # attitude error
    fig, ax = plt.subplots(figsize=(9, 4.6))
    m = (ta_rel > 0) & (att_err > 0)
    ax.semilogy(ta_rel[m], deg(att_err[m]), lw=1.0, label="angle(q_EKF2, q_DR), seeded at t = 0")
    overlay_post(ax, "ta_rel", "att_err", scale=180.0 / np.pi)
    mark_resets(ax)
    label_axes(ax, "time since start of dead reckoning [s]", "attitude error angle [deg]")
    ax.legend(loc="lower right")
    fig.suptitle(f"{ttl}\nAttitude error. {cap}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_fig(fig, FIG_DIR / f"dr_{log_id}_atterr.png")

    # trajectory overlay: north-east plan view and height (= -down) vs time
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.2]})
    a1.plot(ek.pos_ned[mp][:, 1], ek.pos_ned[mp][:, 0], "k-", lw=1.5, label="EKF2")
    a1.plot(p_dr[:, 1], p_dr[:, 0], "tab:orange", lw=1.0, label="dead reckoning")
    a1.plot(*ek.pos_ned[mp][0, [1, 0]], "go", label="start")
    a1.set_aspect("equal", adjustable="datalim")
    label_axes(a1, "east [m]", "north [m]", "plan view (NED, EKF2 reference point)")
    a1.legend(loc="upper left")
    ins = a1.inset_axes([0.58, 0.05, 0.4, 0.4])  # EKF2 track alone (invisible at the DR error scale)
    ins.plot(ek.pos_ned[mp][:, 1], ek.pos_ned[mp][:, 0], "k-", lw=1.0)
    ins.set_aspect("equal", adjustable="datalim")
    ins.set_title("EKF2 only", fontsize=7)
    ins.tick_params(labelsize=6)
    ins.set_xlabel("east [m]", fontsize=6)
    ins.set_ylabel("north [m]", fontsize=6)
    a2.plot(tp_rel, -ek.pos_ned[mp][:, 2], "k-", lw=1.5, label="EKF2")
    a2.plot(tp_rel, -p_dr[:, 2], "tab:orange", lw=1.0, label="dead reckoning")
    label_axes(a2, "time since start of dead reckoning [s]", "height above EKF2 origin (= -down) [m]",
               "vertical channel")
    a2.legend()
    fig.suptitle(f"{ttl}\nTrajectory overlay. {cap}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_fig(fig, FIG_DIR / f"dr_{log_id}_traj.png")
    return metrics


def main() -> None:
    rows = [run_log(e) for e in PINNED_LOGS]
    keys = ("1s", "5s", "10s", "30s", "60s", "end")
    f = lambda x: "       n/a" if x is None else f"{x:10.3g}"  # noqa: E731
    for label, sel in (("RUN 1: seeded from EKF2 at t = 0", lambda m: m),
                       ("RUN 2 (post_reset): seeded at first EKF2 quaternion reset + 1 s", lambda m: m["post_reset"])):
        for qty, name, fmt in (("position_error_m", "pos [m]", f), ("velocity_error_m_s", "vel [m/s]", f),
                               ("attitude_error_deg", "att [deg]", f)):
            print(f"\n{label} -- {name}")
            print(f"{'log':9s} {'role':10s} {'g':>8s} {'T[s]':>7s} | " + " ".join(f"{k:>10s}" for k in keys) + " | exp")
            for m in rows:
                r = sel(m)
                if r is None:
                    continue
                ex = r["position_growth_fit"]["exponent"]
                print(f"{m['log_id'][:8]:9s} {m['role']:10s} {m['gravity']['g_local_m_s2']:8.4f} {r['duration_s']:7.1f} | "
                      + " ".join(fmt(r[qty][k]) for k in keys) + (f" | {ex:.2f}" if qty == "position_error_m" else ""))
    print("\ng_effect (run 1): pos error with G0 vs local g; dr difference; 0.5*|dg|*t^2")
    for m in rows:
        for k, e in m["g_effect"]["entries"].items():
            print(f"{m['log_id'][:8]} dg={m['g_effect']['delta_g_m_s2']:+.4f} t={k:>4s} G0 {f(e['pos_err_G0_m'])} "
                  f"local {f(e['pos_err_g_local_m'])} diff {f(e['dr_position_difference_m'])} "
                  f"pred {f(e['predicted_half_dg_t2_m'])}")


if __name__ == "__main__":
    main()
