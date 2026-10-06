"""Validation metrics for the Project A ES-EKF vs the PX4 EKF2 reference (spec ``docs/specs/tuning_writeup.md`` section 1).

Pure functions (no plotting; file I/O only in ``ekf2_bias_reference``, which reads a ``.ulg`` through pyulog).
Extracted behaviour-identically from ``sensor_fusion/report/ekf_study.py``.

Units: positions m, velocities m/s, angles rad internally (``error_stats`` can scale, e.g. to degrees), biases
rad/s and m/s^2, time seconds since the loader window start. Frames: navigation NED, body FRD, quaternions
scalar-first Hamilton body-to-NED (``CONVENTIONS.md``). EKF2 is a reference, NOT ground truth.
"""
from __future__ import annotations

import math

import numpy as np

from common import quaternion as quat
from common.ekf import nis_bounds, nis_fraction_inside
from sensor_fusion.ekf_drone import IDX_BA, IDX_BG, wrap_angle
from sensor_fusion.interp import interp_to

SKIP_INIT_S = 5.0          # initialisation period excluded from the statistics [s]
RESET_EXCL_S = 1.0         # +/- exclusion around each EKF2 quaternion reset (attitude statistics) [s]
SENSOR_ORDER = ("gps_pos", "gps_vel", "baro", "mag_heading")
Z_DOWN = np.array([0.0, 0.0, 1.0])


def clean_json(o):
    """JSON-safe copy of a nested structure: numpy -> python types, NaN/inf -> None."""
    if isinstance(o, dict):
        return {str(k): clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean_json(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def error_stats(x, scale: float = 1.0) -> dict:
    """RMS / max-abs / final / count of the already-masked series ``x`` (``scale`` converts units, e.g. rad -> deg)."""
    x = np.asarray(x)
    if x.size == 0:
        return {"rms": None, "max_abs": None, "final": None, "n": 0}
    return {"rms": float(np.sqrt(np.mean(x ** 2)) * scale), "max_abs": float(np.max(np.abs(x)) * scale),
            "final": float(x[-1] * scale), "n": int(x.size)}


def tilt_angle(q1, q2) -> np.ndarray:
    """Angle [rad], shape (N,), between the body down axes R(q1) z and R(q2) z of two attitude series (yaw-independent)."""
    a, b = quat.rotate(q1, Z_DOWN), quat.rotate(q2, Z_DOWN)
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.sum(a * b, axis=1))


def yaw_of(q) -> np.ndarray:
    """Aerospace ZYX yaw [rad] of body-to-NED quaternions ``q`` (N, 4), in (-pi, pi]."""
    return quat.to_euler_zyx(q)[2]


def reset_mask(t, resets, half: float = RESET_EXCL_S) -> np.ndarray:
    """Boolean (N,) mask, True where ``|t - t_r| > half`` [s] for every reset time ``t_r`` [s]."""
    t = np.asarray(t)
    ok = np.ones(t.shape, bool)
    for tr in resets:
        ok &= np.abs(t - tr) > half
    return ok


def ekf2_bias_reference(log) -> dict:
    """EKF2's bias estimates (``estimator_sensor_bias``) of the primary instance, in repository time.

    The instance is matched by ``accel_device_id`` / ``gyro_device_id`` to ``estimator_selector_status``.
    Restricted to the loader window ``[0, imu.t[-1]]`` (t = (timestamp - log.t0_us) * 1e-6, s).
    Biases in rad/s (gyro) and m/s^2 (accel), FRD body axes.
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


def compare_to_ekf2(res, ref, skip_init_s: float = SKIP_INIT_S, reset_excl_s: float = RESET_EXCL_S) -> dict:
    """Differences ours - EKF2 on the EKF2 grids plus all statistics for one filter run.

    ``res``: ``EkfResult`` (t, p, v, q); ``ref``: ``ekf2_reference`` dict (NED m, m/s, quaternion body-to-NED).
    Statistics skip the first ``skip_init_s`` s of the filter window; attitude statistics also exclude
    +/- ``reset_excl_s`` s around each EKF2 quaternion reset. Returns ``{"cmp": stats dict, "arr": difference arrays}``.
    """
    t0, t1 = float(res.t[0]), float(res.t[-1])
    t_eval0 = t0 + skip_init_s
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
    tilt = tilt_angle(q_ours, q_ref)
    yaw = wrap_angle(yaw_of(q_ours) - yaw_of(q_ref))

    ev_p = tp >= t_eval0
    ev_a = (ta >= t_eval0) & reset_mask(ta, resets, reset_excl_s)
    cmp = {
        "pos_horizontal_m": error_stats(pos_h[ev_p]),
        "pos_vertical_m": error_stats(pos_d[ev_p]),
        "vel_m_s": error_stats(vel_n[ev_p]),
        "tilt_deg": error_stats(tilt[ev_a], 180.0 / np.pi),
        "yaw_deg": error_stats(yaw[ev_a], 180.0 / np.pi),
        "max_incl_init": {"pos_horizontal_m": float(np.max(pos_h)), "pos_vertical_m": float(np.max(np.abs(pos_d))),
                          "vel_m_s": float(np.max(vel_n))},
        "yaw_signed_mean_deg": float(np.degrees(np.mean(yaw[ev_a]))) if ev_a.any() else None,
        "vertical_signed_mean_m": float(np.mean(pos_d[ev_p])) if ev_p.any() else None,
        "n_att_samples_excluded_resets": int((ta >= t_eval0).sum() - ev_a.sum()),
    }
    arr = {"tp": tp, "pos_h": pos_h, "pos_d": pos_d, "vel": vel_n, "ta": ta, "tilt": tilt, "yaw": yaw}
    return {"cmp": cmp, "arr": arr}


def nis_summary(res, t_eval0: float) -> dict:
    """Per-sensor NIS statistics of accepted updates (dimensionless), plus the subset with ``t >= t_eval0`` s."""
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
    """Final bias estimates and 1-sigma (gyro rad/s, accel m/s^2, FRD body axes) plus the initial values."""
    sd = res.std[-1]
    return {"b_g_rad_s": res.b_g[-1].tolist(), "b_g_sigma_rad_s": sd[IDX_BG].tolist(),
            "b_a_m_s2": res.b_a[-1].tolist(), "b_a_sigma_m_s2": sd[IDX_BA].tolist(),
            "b_g_init_rad_s": list(res.meta["init"]["b_g"]), "b_a_init_m_s2": list(res.meta["init"]["b_a"])}


def consistency_score(nis: dict, min_updates: int = 5) -> dict:
    """Scalar NIS-consistency score from a ``nis_summary`` dict (dimensionless).

    For every sensor with ``after_5s`` present and ``after_5s["n"] >= min_updates``:
    ``ratio = after_5s["mean_nis"] / dof``, ``log_ratio = ln(ratio)``. ``score = sqrt(mean(log_ratio^2))`` over those
    sensors (0 for a perfectly consistent filter; ratios 2 and 0.5 score the same); ``None`` if no sensor qualifies.
    """
    per, logs = {}, []
    for name, e in nis.items():
        if not e or e.get("dof") is None:
            continue
        late = e.get("after_5s")
        if not late or late["n"] < min_updates:
            continue
        ratio = late["mean_nis"] / e["dof"]
        lr = math.log(ratio) if ratio > 0 else float("-inf")
        per[name] = {"ratio": float(ratio), "log_ratio": float(lr),
                     "fraction_inside_95": late["fraction_inside_95"], "n": int(late["n"])}
        logs.append(lr)
    score = float(np.sqrt(np.mean(np.square(logs)))) if logs else None
    return {"per_sensor": per, "score": score, "n_sensors": len(logs)}
