"""Vertical (down) channel diagnostics on the three pinned PX4 logs (Group 4, M4).

Run::

    .venv/Scripts/python -m sensor_fusion.report.height_diagnostics

Writes ``sensor_fusion/report/height_diagnostics.json`` and ``figures/height_diagnostics_<id>.png`` (150 dpi).
Purpose: traceable evidence for the aggressive-log baro NIS and the vertical-datum caveat (CONVENTIONS section 6,
assumptions A-08, A-10). Down position in the repository NED frame (origin = first GPS fix, ellipsoidal height), m,
down positive. Sources compared at the end of the loader window and at t = 40 s, 100 s (if the log is that long):

* ours: final tuned ``EkfConfig`` defaults, increment 3;
* EKF2: ``ekf2_reference`` (translated into our frame); a reference, NOT truth;
* GPS position D, nearest receiver sample, GPS delay ignored;
* baro: ``streams.baro_d`` (baro height change tied to p_D at the start), nearest sample;
* integrated GPS vertical velocity: ``p_D(0)`` (first GPS D) + trapezoid integral of ``gps_vel_ned[:, 2]`` over the GPS
  receiver stamps, read at the nearest GPS sample.

The landing pad height equal to the take-off height is an ASSUMPTION, not a result; no source here establishes it.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.plotting import label_axes, save_fig  # noqa: E402
from sensor_fusion.download_logs import PINNED_LOGS  # noqa: E402
from sensor_fusion.ekf_drone import EkfConfig, run_filter  # noqa: E402
from sensor_fusion.loader import _resolve_path  # noqa: E402
from sensor_fusion.metrics import SENSOR_ORDER, SKIP_INIT_S, clean_json, nis_summary  # noqa: E402
from sensor_fusion.run_flight import load_context, run_one  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parent
FIG_DIR = REPORT_DIR / "figures"
CAVEAT = "EKF2 reference, not truth"
QUERY_TIMES_S = (40.0, 100.0)
ASSUMPTION = ("ASSUMPTION (not a result): landing pad height = take-off height. Nothing in the logs verifies it; "
              "all 'height drift' statements that rely on it inherit this assumption.")


def _nearest(t: np.ndarray, tq: float) -> int:
    return int(np.argmin(np.abs(np.asarray(t, float) - tq)))


def cumulative_gps_height(S) -> np.ndarray:
    """p_D(0) + cumulative trapezoid integral of GPS vertical velocity at each GPS receiver stamp [m]."""
    t = np.asarray(S.gps_t, float)
    vd = np.asarray(S.gps_vel_ned, float)[:, 2]
    inc = 0.5 * (vd[1:] + vd[:-1]) * np.diff(t)
    return float(S.gps_pos_ned[0, 2]) + np.concatenate([[0.0], np.cumsum(inc)])


def ekf2_reported_accuracy(log_id: str, t0_us: int, t_end: float) -> dict:
    """Median EKF2 eph / epv / evh [m] from ``vehicle_local_position`` inside the loader window (pyulog)."""
    from pyulog import ULog
    ul = ULog(str(_resolve_path(log_id)), message_name_filter_list=["vehicle_local_position"])
    ds = [d for d in ul.data_list if d.name == "vehicle_local_position"]
    if not ds:
        return {"available": False, "reason": "no vehicle_local_position topic in the log"}
    d = sorted(ds, key=lambda x: x.multi_id)[0].data
    t = (d["timestamp"].astype(np.int64) - int(t0_us)) * 1e-6
    m = (t >= 0.0) & (t <= t_end)
    out = {"available": True, "n_samples": int(m.sum()), "unit": "m"}
    for k in ("eph", "epv", "evh"):
        if k in d:
            out[f"median_{k}"] = float(np.median(d[k][m].astype(float)))
        else:
            out[f"median_{k}"] = None
            out.setdefault("missing_fields", []).append(k)
    return out


def sample_at(label_t: float, t_end: float, S, ref, res, cum) -> dict:
    """All down-position sources at time ``label_t`` [s] (nearest samples), plus the two requested differences."""
    ours = float(res.p[_nearest(res.t, label_t), 2])
    e2 = float(ref["pos_ned"][_nearest(ref["t_pos"], label_t), 2])
    ig = _nearest(S.gps_t, label_t)
    gps = float(S.gps_pos_ned[ig, 2])
    ib = _nearest(S.baro_t, label_t)
    baro = float(S.baro_d[ib])
    igv = float(cum[ig])
    return {"t_requested_s": float(label_t),
            "t_used_s": {"ours": float(res.t[_nearest(res.t, label_t)]),
                         "ekf2": float(ref["t_pos"][_nearest(ref["t_pos"], label_t)]),
                         "gps": float(S.gps_t[ig]), "baro": float(S.baro_t[ib])},
            "down_position_m": {"ours": ours, "ekf2": e2, "gps": gps, "baro": baro, "gps_vel_integrated": igv},
            "baro_minus_gps_m": baro - gps,
            "gps_pos_minus_gps_vel_integrated_m": gps - igv,
            "ours_minus_ekf2_m": ours - e2}


def figure(role, log_id, dur, S, ref, res, cum, out_dir=None) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(res.t, res.p[:, 2], c="tab:green", lw=1.2, label="ours (ES-EKF, final config, increment 3)")
    ax.plot(ref["t_pos"], ref["pos_ned"][:, 2], c="tab:red", lw=1.0, ls="--", label="EKF2 (translated to our frame)")
    ax.plot(S.gps_t, S.gps_pos_ned[:, 2], c="tab:blue", lw=1.0, label="GPS position D")
    ax.plot(S.baro_t, S.baro_d, c="tab:orange", lw=0.8, alpha=0.9, label="baro-derived D (height change from start)")
    ax.plot(S.gps_t, cum, c="k", lw=1.0, ls=":", label="GPS vertical velocity integrated from first fix")
    ax.invert_yaxis()   # down positive: plot with up at the top
    label_axes(ax, "time since window start [s]", "down position D, + below take-off [m] (axis inverted)",
               f"Vertical channel, {role} flight {log_id[:8]} ({dur:.0f} s)")
    ax.legend(loc="best", fontsize=8)
    fig.text(0.01, 0.005, f"{CAVEAT}. Heights are ellipsoidal changes from the first GPS fix; "
             "landing height = take-off height is an assumption.", fontsize=7, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return save_fig(fig, Path(FIG_DIR if out_dir is None else out_dir) / f"height_diagnostics_{log_id}.png")


def analyse(entry: dict, make_figure: bool = True) -> dict:
    log_id, role = entry["id"], entry["role"]
    ctx = load_context(entry)
    S, ref, log = ctx["S"], ctx["ref"], ctx["log"]
    cfg = EkfConfig(increment=3)
    res = run_filter(S, cfg)
    cum = cumulative_gps_height(S)
    t_end = float(res.t[-1])

    samples = {"end_of_window": sample_at(t_end, t_end, S, ref, res, cum)}
    for tq in QUERY_TIMES_S:
        if tq <= t_end:
            samples[f"t_{tq:.0f}s"] = sample_at(tq, t_end, S, ref, res, cum)
        else:
            samples[f"t_{tq:.0f}s"] = None

    dh = S.gps_pos_ned[-1, :2] - S.gps_pos_ned[0, :2]
    # NIS from run_one (final config, increment 3), plus an internal consistency check against our own run
    m = run_one(entry, 3, None, figures=False, write=False, ctx=ctx)
    nis_ratio = {}
    for n in SENSOR_ORDER:
        e = m["nis"].get(n)
        if e and e.get("after_5s") and e.get("dof"):
            nis_ratio[n] = {"mean_nis_after_5s": e["after_5s"]["mean_nis"], "dof": e["dof"],
                            "ratio": e["after_5s"]["mean_nis"] / e["dof"], "n": e["after_5s"]["n"]}
    own = nis_summary(res, float(res.t[0]) + SKIP_INIT_S)
    consistent = all(abs(own[n]["after_5s"]["mean_nis"] - m["nis"][n]["after_5s"]["mean_nis"]) < 1e-9
                     for n in nis_ratio)

    out = {
        "log_id": log_id, "role": role, "reference": CAVEAT, "window_duration_s": t_end,
        "assumption_landing_height": ASSUMPTION,
        "sign_convention": "down positive; positive end value = below the take-off height",
        "units": {"position": "m", "time": "s", "eph_epv_evh": "m", "s_acc": "m/s"},
        "config": "EkfConfig defaults (final tuned values), increment 3",
        "samples": samples,
        "gps_first_to_last_horizontal_distance_m": float(np.hypot(*dh)),
        "gps_first_to_last_ned_delta_m": [float(x) for x in (S.gps_pos_ned[-1] - S.gps_pos_ned[0])],
        "receiver_median": {"eph_m": float(np.median(S.gps_pos_sigma[:, 0])),
                            "epv_m": float(np.median(S.gps_pos_sigma[:, 2])),
                            "s_acc_m_s": float(np.median(S.gps_vel_sigma)), "n_gps": int(S.gps_t.size)},
        "ekf2_reported_median": ekf2_reported_accuracy(log_id, log.t0_us, t_end),
        "ekf2_vertical_offset_m": ref["vertical_offset"],
        "nis_ratio_after_5s": nis_ratio,
        "nis_own_run_matches_run_one": bool(consistent),
    }
    if make_figure:
        out["figure"] = str(figure(role, log_id, t_end, S, ref, res, cum).relative_to(REPORT_DIR.parent.parent)
                            ).replace("\\", "/")
    return out


def main() -> int:
    import time
    t0 = time.perf_counter()
    logs = {}
    for e in PINNED_LOGS:
        t1 = time.perf_counter()
        r = analyse(e)
        r["analysis_runtime_s"] = time.perf_counter() - t1
        logs[e["id"]] = r
        d = r["samples"]["end_of_window"]["down_position_m"]
        print(f"[{e['id'][:8]} {e['role']}] end {r['window_duration_s']:.1f} s: " +
              " ".join(f"{k} {v:+.2f}" for k, v in d.items()) + f"  ({r['analysis_runtime_s']:.1f} s)", flush=True)
    result = clean_json({"reference": CAVEAT, "assumption_landing_height": ASSUMPTION,
                         "total_runtime_s": time.perf_counter() - t0, "logs": logs})
    (REPORT_DIR / "height_diagnostics.json").write_text(json.dumps(result, indent=2))
    print(f"total {result['total_runtime_s']:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
