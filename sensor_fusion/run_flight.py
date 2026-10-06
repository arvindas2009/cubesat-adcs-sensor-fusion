"""Run the Project A ES-EKF on a pinned PX4 log and compare with EKF2 (spec ``docs/specs/tuning_writeup.md`` section 3).

Run::

    .venv/Scripts/python -m sensor_fusion.run_flight <log> [--increment 3] [--set key=value ...]
                                                     [--no-figures] [--out-dir DIR] [--prefix final]

``<log>``: role (``hover``, ``aggressive``, ``other``), full id or id prefix of ``PINNED_LOGS``; ``all`` runs the three.
Uses the *current* ``EkfConfig`` defaults (the tuned values) plus ``--set`` overrides (any ``EkfConfig`` field, value
parsed with ``ast.literal_eval``). Writes ``<out-dir>/<prefix>_metrics_<id>.json`` (default out-dir
``sensor_fusion/report``, prefix ``final``) and, unless ``--no-figures``, ``<out-dir>/figures/<prefix>_<id>_{pos,att,bias,nis}.png``
(bias / NIS figures only for increment 3). Units: m, m/s, deg in tables (rad internally), NIS dimensionless.
EKF2 is a reference, NOT ground truth.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import json
from pathlib import Path

import numpy as np

from sensor_fusion.download_logs import PINNED_LOGS
from sensor_fusion.ekf_drone import EkfConfig, ekf2_reference, run_filter, streams_from_log
from sensor_fusion.loader import load_log
from sensor_fusion.metrics import (RESET_EXCL_S, SENSOR_ORDER, SKIP_INIT_S, bias_summary, clean_json,
                                   compare_to_ekf2, consistency_score, ekf2_bias_reference, nis_summary)

REPORT_DIR = Path(__file__).resolve().parent / "report"
CAVEAT = "EKF2 reference, not truth"


# ------------------------------------------------------------------ CLI helpers (unit-testable)
def config_fields() -> list[str]:
    """Names of all ``EkfConfig`` fields (valid ``--set`` keys)."""
    return [f.name for f in dataclasses.fields(EkfConfig)]


def parse_overrides(items) -> dict:
    """Parse ``["key=value", ...]`` into ``{key: python value}`` (value via ``ast.literal_eval``).

    Accepts float / int / bool / None / str literals (e.g. ``gyro_noise=0.002``, ``gate_prob=None``,
    ``use_mag=True``, ``discretization='van_loan'``). Unknown keys, a missing ``=`` or a non-literal value raise
    ``SystemExit`` listing the valid field names.
    """
    valid = config_fields()
    out = {}
    for item in items or []:
        key, sep, val = str(item).partition("=")
        key = key.strip()
        if not sep:
            raise SystemExit(f"--set expects key=value, got '{item}'")
        if key not in valid:
            raise SystemExit(f"unknown EkfConfig field '{key}'; valid names: {', '.join(valid)}")
        try:
            out[key] = ast.literal_eval(val.strip())
        except (ValueError, SyntaxError):
            raise SystemExit(f"value of '{key}' is not a Python literal: '{val}'") from None
    return out


def build_config(increment: int = 3, overrides: dict | None = None) -> EkfConfig:
    """``EkfConfig(increment=increment, **overrides)``; an ``increment`` entry in ``overrides`` takes precedence."""
    kw = {"increment": increment}
    kw.update(overrides or {})
    unknown = sorted(set(kw) - set(config_fields()))
    if unknown:
        raise SystemExit(f"unknown EkfConfig field(s) {unknown}; valid names: {', '.join(config_fields())}")
    try:
        return EkfConfig(**kw)
    except (ValueError, TypeError) as e:
        raise SystemExit(f"invalid EkfConfig: {e}") from None


def select_logs(token: str) -> list[dict]:
    """``PINNED_LOGS`` entries matching ``token``: ``all``, a role, a full id or an id prefix."""
    if token == "all":
        return list(PINNED_LOGS)
    hit = [e for e in PINNED_LOGS if token == e["role"] or e["id"].startswith(token)]
    if not hit:
        raise SystemExit(f"unknown log '{token}' (use 'all', a role {[e['role'] for e in PINNED_LOGS]} or an id prefix)")
    return hit


# ------------------------------------------------------------------ one run
def load_context(log_entry: dict) -> dict:
    """Load a pinned log once: ``{log, S, ref, bias_ref, dur}`` (reusable across many ``run_one`` calls)."""
    log = load_log(log_entry["id"])
    S = streams_from_log(log)
    return {"log": log, "S": S, "ref": ekf2_reference(log, S.origin_lla), "bias_ref": ekf2_bias_reference(log),
            "dur": float(log.imu.t[-1] - log.imu.t[0])}


def run_one(log_entry: dict, increment: int = 3, overrides: dict | None = None, out_dir=None,
            prefix: str = "final", figures: bool = True, write: bool = True, ctx: dict | None = None) -> dict:
    """Run the filter on one pinned log and compare with EKF2; returns the metrics dict (JSON-safe, no arrays).

    ``overrides``: ``EkfConfig`` field overrides (an ``increment`` key overrides the ``increment`` argument).
    ``out_dir``: output directory (default ``sensor_fusion/report``); the JSON goes to
    ``<out_dir>/<prefix>_metrics_<id>.json`` if ``write`` and the figures to ``<out_dir>/figures/`` if ``figures``.
    ``ctx``: optional result of ``load_context`` to avoid reloading the log. Statistics skip the first 5 s and, for
    tilt / yaw, +/- 1 s around each EKF2 quaternion reset.
    """
    cfg = build_config(increment, overrides)
    ctx = load_context(log_entry) if ctx is None else ctx
    log_id, role = log_entry["id"], log_entry["role"]
    S, ref, bias_ref, dur = ctx["S"], ctx["ref"], ctx["bias_ref"], ctx["dur"]
    lat0, lon0, alt0 = S.origin_lla

    res = run_filter(S, cfg)
    c = compare_to_ekf2(res, ref)
    nis = nis_summary(res, float(res.t[0]) + SKIP_INIT_S)
    metrics = {
        "log_id": log_id, "role": role, "reference": CAVEAT,
        "note": ("EKF2 is a reference, not truth. Statistics skip the first "
                 f"{SKIP_INIT_S:.0f} s (initialisation) and, for tilt/yaw, +/-{RESET_EXCL_S:.0f} s around each EKF2 "
                 "quaternion reset. max_incl_init is over all samples."),
        "increment": cfg.increment, "overrides": dict(overrides or {}), "prefix": prefix,
        "skip_initial_s": SKIP_INIT_S, "reset_exclusion_s": RESET_EXCL_S,
        "units": {"position": "m", "velocity": "m/s", "tilt_yaw": "deg", "bias_gyro": "rad/s", "bias_accel": "m/s^2"},
        "g_m_s2": float(S.g), "declination_deg": float(np.degrees(S.declination)),
        "origin": {"lat_deg": float(np.degrees(lat0)), "lon_deg": float(np.degrees(lon0)),
                   "alt_ellipsoid_m": float(alt0)},
        "gps_utc_start": S.meta.get("gps_utc_start"),
        "ekf2_reset_times_s": [float(x) for x in ref["reset_times"]],
        "ekf2_origin_offset_ne_m": ref["origin_offset_ne"], "ekf2_vertical_offset_m": ref["vertical_offset"],
        "n_imu": int(ctx["log"].imu.t.size), "duration_s": dur,
        "ekf2_bias_reference": {k: v for k, v in bias_ref.items() if k not in ("t", "accel_bias", "gyro_bias")},
        "all_finite": {k: bool(np.all(np.isfinite(getattr(res, k)))) for k in ("p", "v", "q", "b_g", "b_a", "std")},
        "comparison_vs_ekf2": c["cmp"],
        "nis": nis,
        "consistency": consistency_score(nis),
        "n_updates": res.meta["n_updates"],
        "mag_skipped_tilt": res.meta["mag_skipped_tilt"],
        "final_biases": bias_summary(res),
        "runtime_s": res.meta["runtime_s"],
        "config": res.meta["config"],
    }
    metrics = clean_json(metrics)

    base = Path(REPORT_DIR if out_dir is None else out_dir)
    if write:
        base.mkdir(parents=True, exist_ok=True)
        (base / f"{prefix}_metrics_{log_id}.json").write_text(json.dumps(metrics, indent=2))
    if figures:
        from sensor_fusion.report import ekf_figures as F
        fd = base / "figures"
        runs = {cfg.increment: {"arr": c["arr"]}}
        F.fig_pos(runs, role, log_id, dur, prefix=prefix, out_dir=fd)
        F.fig_att(runs, ref["reset_times"], role, log_id, dur, prefix=prefix, out_dir=fd)
        if cfg.increment == 3:
            F.fig_bias(res, bias_ref, role, log_id, dur, prefix=prefix, out_dir=fd)
            F.fig_nis(res, role, log_id, dur, prefix=prefix, out_dir=fd)
        else:
            print(f"[{log_id[:8]}] increment {cfg.increment} != 3: bias/NIS figures skipped")
    return metrics


# ------------------------------------------------------------------ summary + CLI
def print_summary(all_metrics) -> None:
    """Table of RMS differences vs EKF2 (m, m/s, deg), NIS mean/dof per sensor, consistency score, runtime."""
    f = lambda x, p=3: "    n/a" if x is None else f"{x:7.{p}f}"  # noqa: E731
    print(f"\nSummary vs EKF2 ({CAVEAT}); statistics skip first {SKIP_INIT_S:.0f} s; attitude excludes "
          f"+/-{RESET_EXCL_S:.0f} s around EKF2 resets")
    print(f"{'log':9s} {'role':10s} {'inc':>3s} | {'posH':>7s} {'posD':>7s} {'vel':>7s} {'tilt':>7s} {'yaw':>7s} | "
          + " ".join(f"{'NIS ' + n:>13s}" for n in SENSOR_ORDER) + " | score | runtime")
    print(f"{'':9s} {'':10s} {'':>3s} | {'RMS[m]':>7s} {'RMS[m]':>7s} {'[m/s]':>7s} {'[deg]':>7s} {'[deg]':>7s} |")
    for m in all_metrics:
        c = m["comparison_vs_ekf2"]
        nis = []
        for n in SENSOR_ORDER:
            e = m["nis"].get(n)
            nis.append(f"{e['mean_nis']:6.2f}/{e['dof']}" if e and e.get("mean_nis") is not None else "-")
        sc = m["consistency"]["score"]
        print(f"{m['log_id'][:8]:9s} {m['role']:10s} {m['increment']:>3} | {f(c['pos_horizontal_m']['rms'])} "
              f"{f(c['pos_vertical_m']['rms'])} {f(c['vel_m_s']['rms'])} {f(c['tilt_deg']['rms'], 2)} "
              f"{f(c['yaw_deg']['rms'], 2)} | " + " ".join(f"{x:>13s}" for x in nis)
              + f" | {'  n/a' if sc is None else f'{sc:5.3f}'} | {m['runtime_s']:5.1f} s")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the ES-EKF on a pinned PX4 log and compare with EKF2 (a reference, not truth).")
    ap.add_argument("log", help="role (hover/aggressive/other), id or id prefix of PINNED_LOGS, or 'all'")
    ap.add_argument("--increment", type=int, default=3, choices=(1, 2, 3))
    ap.add_argument("--set", dest="overrides", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                    help="EkfConfig override(s), value parsed with ast.literal_eval")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--out-dir", default=None, help="output directory (default sensor_fusion/report)")
    ap.add_argument("--prefix", default="final")
    args = ap.parse_args(argv)

    overrides = parse_overrides(args.overrides)
    build_config(args.increment, overrides)          # validate before loading any log
    all_metrics = []
    for entry in select_logs(args.log):
        m = run_one(entry, args.increment, overrides, args.out_dir, args.prefix, figures=not args.no_figures)
        all_metrics.append(m)
        print(f"[{entry['id'][:8]} {entry['role']}] increment {m['increment']}: {m['runtime_s']:.1f} s, "
              f"posH RMS {m['comparison_vs_ekf2']['pos_horizontal_m']['rms']:.3f} m", flush=True)
    print_summary(all_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
