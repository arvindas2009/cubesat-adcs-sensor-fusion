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
from pathlib import Path

import numpy as np

from sensor_fusion.download_logs import PINNED_LOGS
from sensor_fusion.ekf_drone import EkfConfig, ekf2_reference, run_filter, streams_from_log
from sensor_fusion.loader import load_log
from sensor_fusion.metrics import (RESET_EXCL_S, SENSOR_ORDER, SKIP_INIT_S, bias_summary, clean_json,
                                   compare_to_ekf2, ekf2_bias_reference, nis_summary)
from sensor_fusion.report.ekf_figures import fig_att, fig_bias, fig_nis, fig_pos

REPORT_DIR = Path(__file__).resolve().parent
CAVEAT = "EKF2 reference, not truth"

# EkfConfig defaults as of Group 3 (pre-tuning), written out explicitly so that this study keeps reproducing
# ``ekf_metrics_<id>.json`` even after the ``EkfConfig`` defaults are tuned (Group 4).
G3_DEFAULTS = {
    "accel_noise": 0.035,        # m/s^2/sqrt(Hz)
    "gyro_noise": 1.5e-3,        # rad/s/sqrt(Hz)
    "gyro_bias_rw": 1.0e-4,      # rad/s^2/sqrt(Hz)
    "accel_bias_rw": 3.0e-4,     # m/s^3/sqrt(Hz)
    "sigma_p0": 1.0,             # m
    "sigma_v0": 0.5,             # m/s
    "sigma_att0": 0.1,           # rad
    "sigma_bg0": 0.1,            # rad/s
    "sigma_ba0": 0.2,            # m/s^2
    "gps_delay_s": 0.11,         # s
    "gps_pos_floor_h": 0.5,      # m
    "gps_pos_floor_v": 0.5,      # m
    "gps_vel_floor": 0.3,        # m/s
    "baro_sigma": 1.5,           # m
    "mag_heading_sigma": 0.07,   # rad
}


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
        res = run_filter(S, EkfConfig(increment=inc, **G3_DEFAULTS))
        c = compare_to_ekf2(res, ref)
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
    (REPORT_DIR / f"ekf_metrics_{log_id}.json").write_text(json.dumps(clean_json(metrics), indent=2))

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
