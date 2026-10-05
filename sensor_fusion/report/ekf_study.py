"""Project A flight study: 15-state ES-EKF (increments 1-3) vs PX4 EKF2 on the pinned logs (spec section 8).

Run:  .venv/Scripts/python -m sensor_fusion.report.ekf_study [--logs hover aggressive <id> ...] [--increments 1 2 3]

For every selected log (``PINNED_LOGS``; loaded once) and increment:

* ``S = streams_from_log(log)``, ``ref = ekf2_reference(log, S.origin_lla)``,
  ``res = run_filter(S, EkfConfig(increment=i))``;
* the filter output is interpolated onto the **EKF2 grids** (linear for p, v; slerp for q) and compared:
  horizontal position difference norm, vertical (down) difference, velocity difference norm, tilt difference
  (angle between the body down axes ``R(q) [0, 0, 1]`` of the two attitudes) and wrapped ZYX yaw difference;
* statistics (RMS / max / final) use only samples with ``t >= 5 s`` (the first 5 s are initialisation) and, for the
  attitude quantities, exclude +/- 1 s around every EKF2 quaternion reset (``ref["reset_times"]``);
* per-sensor NIS (accepted updates), update counts, final bias estimates +/- 1 sigma, runtime;
* EKF2's own bias estimates from ``estimator_sensor_bias`` (pyulog; the instance whose accel/gyro device ids equal
  ``estimator_selector_status`` -- the primary EKF2 instance).

EKF2 is a reference, NOT ground truth. Anomalies are reported, never corrected.
Outputs: ``sensor_fusion/report/ekf_metrics_<id>.json`` and
``sensor_fusion/report/figures/ekf_<id>_{pos,att,bias,nis}.png``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import quaternion as quat  # noqa: E402
from common.ekf import nis_bounds, nis_fraction_inside  # noqa: E402
from common.plotting import deg, label_axes, save_fig  # noqa: E402
from sensor_fusion.download_logs import PINNED_LOGS  # noqa: E402
from sensor_fusion.ekf_drone import (IDX_BA, IDX_BG, EkfConfig, ekf2_reference, run_filter,  # noqa: E402
                                     streams_from_log, wrap_angle)
from sensor_fusion.interp import interp_to  # noqa: E402
from sensor_fusion.loader import load_log  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parent
FIG_DIR = REPORT_DIR / "figures"
CAVEAT = "EKF2 reference, not truth"
SKIP_INIT_S = 5.0          # initialisation period excluded from the statistics
RESET_EXCL_S = 1.0         # +/- exclusion around each EKF2 quaternion reset (attitude statistics)
NIS_MA = 20                # moving-average length for the NIS figure [samples]
PLOT_MAX_POINTS = 4000     # decimation limit for the long filter-rate time series
SENSOR_ORDER = ("gps_pos", "gps_vel", "baro", "mag_heading")
INC_COLORS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
INC_LABEL = {1: "inc 1: GPS+baro", 2: "inc 2: +mag", 3: "inc 3: +biases"}
Z_DOWN = np.array([0.0, 0.0, 1.0])


# ------------------------------------------------------------------ helpers
def _clean(o):
    """JSON-safe copy: numpy -> python, NaN/inf -> None."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _stats(t, x, scale=1.0) -> dict:
    """RMS / max-abs / final of x(t) over the already-masked samples (``scale`` converts units)."""
    if x.size == 0:
        return {"rms": None, "max_abs": None, "final": None, "n": 0}
    return {"rms": float(np.sqrt(np.mean(x ** 2)) * scale), "max_abs": float(np.max(np.abs(x)) * scale),
            "final": float(x[-1] * scale), "n": int(x.size)}


def _decim(n: int) -> slice:
    return slice(None, None, max(1, n // PLOT_MAX_POINTS))


def _tilt_angle(q1, q2) -> np.ndarray:
    """Angle [rad] between the body down axes R(q1) z and R(q2) z (yaw-independent)."""
    a, b = quat.rotate(q1, Z_DOWN), quat.rotate(q2, Z_DOWN)
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.sum(a * b, axis=1))


def _yaw(q) -> np.ndarray:
    return quat.to_euler_zyx(q)[2]


def _reset_mask(t, resets, half=RESET_EXCL_S) -> np.ndarray:
    ok = np.ones(t.shape, bool)
    for tr in resets:
        ok &= np.abs(t - tr) > half
    return ok


# ------------------------------------------------------------------ EKF2 bias reference (pyulog)
def ekf2_bias_reference(log) -> dict:
    """EKF2's bias estimates (``estimator_sensor_bias``) of the primary instance, in repository time.

    The instance is matched by ``accel_device_id`` / ``gyro_device_id`` to ``estimator_selector_status``.
    Restricted to the loader window ``[0, imu.t[-1]]`` (t = (timestamp - log.t0_us) * 1e-6).
    """
    from pyulog import ULog
    ul = ULog(str(log.path), message_name_filter_list=["estimator_sensor_bias", "estimator_selector_status"])
    sel = [d for d in ul.data_list if d.name == "estimator_selector_status"]
    if not sel:
        return {"available": False, "reason": "no estimator_selector_status topic"}
    sel = sel[0].data
    acc_id = int(np.median(sel["accel_device_id"]))
    gyr_id = int(np.median(sel["gyro_device_id"]))
    out = {"available": False, "selector_accel_device_id": acc_id, "selector_gyro_device_id": gyr_id,
           "selector_device_ids_constant": bool(np.all(sel["accel_device_id"] == acc_id)
                                                and np.all(sel["gyro_device_id"] == gyr_id)),
           "selector_primary_instance_values": sorted({int(v) for v in sel["primary_instance"]}),
           "units": {"accel_bias": "m/s^2", "gyro_bias": "rad/s"}}
    for d in ul.data_list:
        if d.name != "estimator_sensor_bias":
            continue
        da = d.data["accel_device_id"]
        dg = d.data["gyro_device_id"]
        if not (np.all(da == acc_id) and np.all(dg == gyr_id)):
            continue
        t = (d.data["timestamp"].astype(np.int64) - np.int64(log.t0_us)) * 1e-6
        m = (t >= 0.0) & (t <= float(log.imu.t[-1]))
        if not m.any():
            out["reason"] = "matched instance has no samples inside the loader window"
            return out
        ab = np.column_stack([d.data[f"accel_bias[{i}]"] for i in range(3)])[m]
        gb = np.column_stack([d.data[f"gyro_bias[{i}]"] for i in range(3)])[m]
        tm = t[m]
        out.update({
            "available": True, "multi_id": int(d.multi_id), "n_samples": int(m.sum()),
            "t": tm, "accel_bias": ab, "gyro_bias": gb,
            "accel_bias_median": np.median(ab, axis=0).tolist(), "accel_bias_start": ab[0].tolist(),
            "accel_bias_end": ab[-1].tolist(), "gyro_bias_median": np.median(gb, axis=0).tolist(),
            "gyro_bias_start": gb[0].tolist(), "gyro_bias_end": gb[-1].tolist(),
            "t_start_s": float(tm[0]), "t_end_s": float(tm[-1])})
        return out
    out["reason"] = "no estimator_sensor_bias instance matches the selector device ids"
    return out


# ------------------------------------------------------------------ one filter run vs EKF2
def compare_run(res, ref, S) -> dict:
    """Differences ours - EKF2 on the EKF2 grids plus all statistics for one filter run."""
    t0, t1 = float(res.t[0]), float(res.t[-1])
    t_eval0 = t0 + SKIP_INIT_S
    resets = [r for r in ref["reset_times"]]

    tp = np.asarray(ref["t_pos"], float)
    mp = (tp >= t0) & (tp <= t1)
    tp = tp[mp]
    dp = interp_to(tp, res.t, res.p) - np.asarray(ref["pos_ned"])[mp]
    dv = interp_to(tp, res.t, res.v) - np.asarray(ref["vel_ned"])[mp]
    pos_h, pos_d = np.linalg.norm(dp[:, :2], axis=1), dp[:, 2]
    vel_n = np.linalg.norm(dv, axis=1)

    ta = np.asarray(ref["t_att"], float)
    ma = (ta >= t0) & (ta <= t1)
    ta = ta[ma]
    q_ref = np.asarray(ref["q"])[ma]
    q_ours = interp_to(ta, res.t, res.q, quaternion=True)
    tilt = _tilt_angle(q_ours, q_ref)
    yaw = wrap_angle(_yaw(q_ours) - _yaw(q_ref))

    ev_p = tp >= t_eval0
    ev_a = (ta >= t_eval0) & _reset_mask(ta, resets)
    cmp = {
        "pos_horizontal_m": _stats(tp[ev_p], pos_h[ev_p]),
        "pos_vertical_m": _stats(tp[ev_p], pos_d[ev_p]),
        "vel_m_s": _stats(tp[ev_p], vel_n[ev_p]),
        "tilt_deg": _stats(ta[ev_a], tilt[ev_a], 180.0 / np.pi),
        "yaw_deg": _stats(ta[ev_a], yaw[ev_a], 180.0 / np.pi),
        "max_incl_init": {"pos_horizontal_m": float(np.max(pos_h)), "pos_vertical_m": float(np.max(np.abs(pos_d))),
                          "vel_m_s": float(np.max(vel_n))},
        "yaw_signed_mean_deg": float(np.degrees(np.mean(yaw[ev_a]))) if ev_a.any() else None,
        "vertical_signed_mean_m": float(np.mean(pos_d[ev_p])) if ev_p.any() else None,
        "n_att_samples_excluded_resets": int((ta >= t_eval0).sum() - ev_a.sum()),
    }
    arr = {"tp": tp, "pos_h": pos_h, "pos_d": pos_d, "vel": vel_n, "ta": ta, "tilt": tilt, "yaw": yaw}
    return {"cmp": cmp, "arr": arr}


def nis_summary(res, t_eval0: float) -> dict:
    out = {}
    for name in SENSOR_ORDER:
        il = res.innovations.get(name)
        if il is None:
            continue
        acc = il.accepted.astype(bool)
        nis = il.nis[acc]
        dofs = np.unique(il.dof[acc]) if acc.any() else np.array([], int)
        dof = int(dofs[0]) if dofs.size else None
        n = int(nis.size)
        entry = {"n_total": int(il.nis.size), "n_accepted": n, "n_rejected": int(il.nis.size - n), "dof": dof,
                 "dof_values": dofs.tolist()}
        if n and dof is not None:
            lo, hi = nis_bounds(dof, n)
            entry.update({"mean_nis": float(np.mean(nis)), "median_nis": float(np.median(nis)),
                          "fraction_inside_95": nis_fraction_inside(nis, dof),
                          "mean_bounds_95": [lo, hi], "mean_inside_bounds": bool(lo <= np.mean(nis) <= hi)})
            late = acc & (il.t >= t_eval0)
            if late.any():
                nl = il.nis[late]
                lo2, hi2 = nis_bounds(dof, int(nl.size))
                entry["after_5s"] = {"n": int(nl.size), "mean_nis": float(np.mean(nl)),
                                     "fraction_inside_95": nis_fraction_inside(nl, dof),
                                     "mean_bounds_95": [lo2, hi2]}
        out[name] = entry
    return out


def bias_summary(res) -> dict:
    sd = res.std[-1]
    return {"b_g_rad_s": res.b_g[-1].tolist(), "b_g_sigma_rad_s": sd[IDX_BG].tolist(),
            "b_a_m_s2": res.b_a[-1].tolist(), "b_a_sigma_m_s2": sd[IDX_BA].tolist(),
            "b_g_init_rad_s": list(res.meta["init"]["b_g"]), "b_a_init_m_s2": list(res.meta["init"]["b_a"])}


# ------------------------------------------------------------------ figures
def _title(role, log_id, dur):
    return f"{role} flight {log_id[:8]} ({dur:.0f} s)"


def fig_pos(runs, role, log_id, dur):
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8.5))
    for i, r in sorted(runs.items()):
        a = r["arr"]
        s = _decim(a["tp"].size)
        c, lab = INC_COLORS[i], INC_LABEL[i]
        axs[0].plot(a["tp"][s], a["pos_h"][s], c=c, lw=0.9, label=lab)
        axs[1].plot(a["tp"][s], a["pos_d"][s], c=c, lw=0.9, label=lab)
        axs[2].plot(a["tp"][s], a["vel"][s], c=c, lw=0.9, label=lab)
    for ax in axs:
        ax.axvspan(runs[min(runs)]["arr"]["tp"][0], runs[min(runs)]["arr"]["tp"][0] + SKIP_INIT_S,
                   color="0.9", zorder=0)
    label_axes(axs[0], "", "horizontal position difference [m]")
    label_axes(axs[1], "", "vertical (down) difference, ours - EKF2 [m]")
    label_axes(axs[2], "time since window start [s]", "velocity difference norm [m/s]")
    axs[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"ES-EKF vs EKF2, {_title(role, log_id, dur)}: position and velocity difference\n"
                 f"{CAVEAT}; first {SKIP_INIT_S:.0f} s (grey) excluded from the statistics", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, FIG_DIR / f"ekf_{log_id}_pos.png")


def fig_att(runs, resets, role, log_id, dur):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 6.5))
    for i, r in sorted(runs.items()):
        a = r["arr"]
        s = _decim(a["ta"].size)
        axs[0].plot(a["ta"][s], deg(a["tilt"][s]), c=INC_COLORS[i], lw=0.9, label=INC_LABEL[i])
        axs[1].plot(a["ta"][s], deg(a["yaw"][s]), c=INC_COLORS[i], lw=0.9, label=INC_LABEL[i])
    for ax in axs:
        for k, tr in enumerate(resets):
            ax.axvspan(tr - RESET_EXCL_S, tr + RESET_EXCL_S, color="tab:red", alpha=0.25, lw=0,
                       label=f"EKF2 quaternion reset (+/-{RESET_EXCL_S:.0f} s excluded)" if k == 0 else None)
            ax.axvline(tr, color="tab:red", ls="--", lw=0.8)
    label_axes(axs[0], "", "tilt difference [deg]")
    label_axes(axs[1], "time since window start [s]", "yaw difference, ours - EKF2 (wrapped) [deg]")
    axs[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"ES-EKF vs EKF2, {_title(role, log_id, dur)}: attitude difference\n{CAVEAT}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, FIG_DIR / f"ekf_{log_id}_att.png")


def fig_bias(res, bias_ref, role, log_id, dur):
    t = res.t
    s = _decim(t.size)
    fig, axs = plt.subplots(2, 3, sharex=True, figsize=(14, 6.5))
    sd = res.std
    ax_names = "xyz"
    for j in range(3):
        for row, (est, sig, unit, scale, key, ylab) in enumerate((
                (res.b_g, sd[:, IDX_BG], "rad/s", 180.0 / np.pi, "gyro_bias", "gyro bias [deg/s]"),
                (res.b_a, sd[:, IDX_BA], "m/s$^2$", 1.0, "accel_bias", "accel bias [m/s$^2$]"))):
            ax = axs[row, j]
            e, sg = est[:, j] * scale, sig[:, j] * scale
            ax.plot(t[s], e[s], c="tab:green", lw=1.0, label="ES-EKF inc 3 estimate")
            ax.fill_between(t[s], (e - sg)[s], (e + sg)[s], color="tab:green", alpha=0.25, lw=0, label="$\\pm 1\\sigma$")
            if bias_ref.get("available"):
                ax.plot(bias_ref["t"], bias_ref[key][:, j] * scale, "k--", lw=1.2, label="EKF2 (primary instance)")
            label_axes(ax, "time since window start [s]" if row == 1 else "",
                       ylab.replace("bias", f"bias {ax_names[j]}"))
    axs[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(f"Bias estimates, increment 3, {_title(role, log_id, dur)}\n"
                 f"{CAVEAT} (EKF2 biases: estimator_sensor_bias, dashed)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_fig(fig, FIG_DIR / f"ekf_{log_id}_bias.png")


def fig_nis(res, role, log_id, dur):
    names = [n for n in SENSOR_ORDER if n in res.innovations and res.innovations[n].nis.size >= NIS_MA]
    fig, axs = plt.subplots(len(names), 1, sharex=True, figsize=(10, 2.4 * len(names) + 1.0), squeeze=False)
    for ax, name in zip(axs[:, 0], names):
        il = res.innovations[name]
        acc = il.accepted.astype(bool)
        t, nis = il.t[acc], il.nis[acc]
        dof = int(np.unique(il.dof[acc])[0])
        ma = np.convolve(nis, np.ones(NIS_MA) / NIS_MA, mode="valid")
        tm = np.convolve(t, np.ones(NIS_MA) / NIS_MA, mode="valid")
        lo, hi = nis_bounds(dof, NIS_MA)
        ax.axhspan(lo, hi, color="tab:blue", alpha=0.2, lw=0, label=f"95 % band of a {NIS_MA}-sample mean, $\\chi^2({dof})/n$")
        ax.axhline(dof, color="tab:blue", lw=0.8, ls="--", label=f"expected mean = dof = {dof}")
        s = _decim(ma.size)
        ax.plot(tm[s], ma[s], c="tab:green", lw=0.9, label=f"NIS, {NIS_MA}-sample moving average")
        ax.set_yscale("log")
        label_axes(ax, "time since window start [s]" if ax is axs[-1, 0] else "", f"{name} NIS [-]")
        ax.legend(loc="upper right", fontsize=7)
    fig.suptitle(f"Normalised innovation squared per sensor, increment 3, {_title(role, log_id, dur)}\n{CAVEAT}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, FIG_DIR / f"ekf_{log_id}_nis.png")


# ------------------------------------------------------------------ per log
def run_log(entry: dict, increments) -> dict:
    log_id, role = entry["id"], entry["role"]
    log = load_log(log_id)                       # loaded once, reused for every increment
    S = streams_from_log(log)
    ref = ekf2_reference(log, S.origin_lla)
    bias_ref = ekf2_bias_reference(log)
    dur = float(log.imu.t[-1] - log.imu.t[0])
    lat0, lon0, alt0 = S.origin_lla

    runs, results = {}, {}
    for inc in increments:
        res = run_filter(S, EkfConfig(increment=inc))
        c = compare_run(res, ref, S)
        finite = {k: bool(np.all(np.isfinite(getattr(res, k)))) for k in ("p", "v", "q", "b_g", "b_a", "std")}
        results[inc] = {
            "all_finite": finite,
            "comparison_vs_ekf2": c["cmp"],
            "nis": nis_summary(res, float(res.t[0]) + SKIP_INIT_S),
            "n_updates": res.meta["n_updates"],
            "mag_skipped_tilt": res.meta["mag_skipped_tilt"],
            "final_biases": bias_summary(res),
            "runtime_s": res.meta["runtime_s"],
            "config": res.meta["config"],
        }
        runs[inc] = {"arr": c["arr"], "res": res}
        print(f"[{log_id[:8]} {role}] increment {inc}: {res.meta['runtime_s']:.1f} s, "
              f"posH RMS {c['cmp']['pos_horizontal_m']['rms']:.3f} m", flush=True)

    bias_json = {k: v for k, v in bias_ref.items() if k not in ("t", "accel_bias", "gyro_bias")}
    metrics = {
        "log_id": log_id, "role": role, "reference": CAVEAT,
        "note": ("EKF2 is a reference, not truth. Statistics skip the first "
                 f"{SKIP_INIT_S:.0f} s (initialisation) and, for tilt/yaw, +/-{RESET_EXCL_S:.0f} s around each EKF2 "
                 "quaternion reset. max_incl_init is over all samples."),
        "skip_initial_s": SKIP_INIT_S, "reset_exclusion_s": RESET_EXCL_S,
        "units": {"position": "m", "velocity": "m/s", "tilt_yaw": "deg", "bias_gyro": "rad/s", "bias_accel": "m/s^2"},
        "g_m_s2": float(S.g), "declination_deg": float(np.degrees(S.declination)),
        "origin": {"lat_deg": float(np.degrees(lat0)), "lon_deg": float(np.degrees(lon0)),
                   "alt_ellipsoid_m": float(alt0)},
        "gps_utc_start": S.meta.get("gps_utc_start"),
        "ekf2_reset_times_s": [float(x) for x in ref["reset_times"]],
        "ekf2_origin_offset_ne_m": ref["origin_offset_ne"], "ekf2_vertical_offset_m": ref["vertical_offset"],
        "n_imu": int(log.imu.t.size), "duration_s": dur,
        "ekf2_bias_reference": bias_json,
        "increments": results,
    }
    (REPORT_DIR / f"ekf_metrics_{log_id}.json").write_text(json.dumps(_clean(metrics), indent=2))

    # ---- figures
    fig_pos(runs, role, log_id, dur)
    fig_att(runs, ref["reset_times"], role, log_id, dur)
    if 3 in runs:
        fig_bias(runs[3]["res"], bias_ref, role, log_id, dur)
        fig_nis(runs[3]["res"], role, log_id, dur)
    else:
        print(f"[{log_id[:8]}] increment 3 not run: bias/NIS figures skipped")
    return metrics


def _select(tokens):
    if not tokens:
        return list(PINNED_LOGS)
    sel = []
    for tok in tokens:
        hit = [e for e in PINNED_LOGS if tok == e["role"] or e["id"].startswith(tok)]
        if not hit:
            raise SystemExit(f"unknown log '{tok}' (roles: {[e['role'] for e in PINNED_LOGS]})")
        sel += [h for h in hit if h not in sel]
    return sel


def print_table(all_metrics) -> None:
    f = lambda x, p=3: "    n/a" if x is None else f"{x:7.{p}f}"  # noqa: E731
    print(f"\nSummary vs EKF2 ({CAVEAT}); statistics skip first {SKIP_INIT_S:.0f} s; attitude excludes "
          f"+/-{RESET_EXCL_S:.0f} s around EKF2 resets")
    print(f"{'log':9s} {'role':10s} {'inc':>3s} | {'posH':>7s} {'posD':>7s} {'vel':>7s} {'tilt':>7s} {'yaw':>7s} | "
          + " ".join(f"{'NIS ' + n:>13s}" for n in SENSOR_ORDER) + " | runtime")
    print(f"{'':9s} {'':10s} {'':>3s} | {'RMS[m]':>7s} {'RMS[m]':>7s} {'[m/s]':>7s} {'[deg]':>7s} {'[deg]':>7s} |")
    for m in all_metrics:
        for i, r in m["increments"].items():
            c = r["comparison_vs_ekf2"]
            nis = []
            for n in SENSOR_ORDER:
                e = r["nis"].get(n)
                nis.append(f"{e['mean_nis']:6.2f}/{e['dof']}" if e and e.get("mean_nis") is not None else "-")
            print(f"{m['log_id'][:8]:9s} {m['role']:10s} {i:>3} | {f(c['pos_horizontal_m']['rms'])} "
                  f"{f(c['pos_vertical_m']['rms'])} {f(c['vel_m_s']['rms'])} {f(c['tilt_deg']['rms'], 2)} "
                  f"{f(c['yaw_deg']['rms'], 2)} | " + " ".join(f"{x:>13s}" for x in nis)
                  + f" | {r['runtime_s']:5.1f} s")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--logs", nargs="*", default=None, help="role or log id (prefix) of PINNED_LOGS; default all")
    ap.add_argument("--increments", nargs="*", type=int, default=[1, 2, 3])
    args = ap.parse_args(argv)
    incs = sorted(set(args.increments))
    if not set(incs) <= {1, 2, 3}:
        raise SystemExit("increments must be in {1, 2, 3}")
    all_metrics = [run_log(e, incs) for e in _select(args.logs)]
    print_table(all_metrics)


if __name__ == "__main__":
    main()
