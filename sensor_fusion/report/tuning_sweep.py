"""Covariance sweep of the Project A ES-EKF (spec ``docs/specs/tuning_writeup.md`` section 5).

Run::

    .venv/Scripts/python -m sensor_fusion.report.tuning_sweep [--budget-min 30] [--workers 20] [--out-dir DIR]

Increment 3 (the final filter), the three pinned PX4 logs, ``multiprocessing`` (each worker loads each log once).
Baseline = the pre-tuning (Group 3) values ``G3_DEFAULTS`` of ``ekf_study`` (fixed, so the sweep stays reproducible after the ``EkfConfig`` defaults change). The sweep never edits ``EkfConfig``; every config passes all knob values as overrides.

* Stage 1 (one at a time): baseline + each of the 9 knobs x factor in {1/4, 1/2, 2, 4}.
* Stage 2 (joint): per knob the factor with the best stage-1 ``total_score`` among *admissible* configs (the baseline,
  factor 1, competes; ties -> factor closest to 1) -> ``joint``; every selected factor moved one ladder step toward 1
  -> ``joint_halfstep``; ``joint`` with the mag sigma left at baseline -> ``joint_no_mag``. Skipped if stage 1 took
  longer than ``--budget-min``.
* Score (fixed, not tuned): per log ``score_nis`` = ``metrics.consistency_score`` score (RMS of ln(NIS mean / dof)
  over the sensors, after the first 5 s); ``total_score`` = mean over the three logs. A config is *admissible* only if
  on every log posH, vel and tilt RMS vs EKF2 are each <= 1.15 x the baseline value on that log. The lowest
  ``total_score`` among admissible configs wins. EKF2 is a reference, NOT ground truth.

Outputs (default dir ``sensor_fusion/report``): ``tuning_sweep.json``, ``tuning_sweep.md``,
``figures/tuning_sweep_<knob>.png``, ``figures/tuning_sweep_nis_ratio.png``. Units: m, m/s, deg, NIS dimensionless.
Deterministic: results are keyed by (config, log) and independent of worker scheduling.
"""
from __future__ import annotations

import os

# One BLAS thread per worker process (set before numpy is imported; spawned workers inherit the environment):
# 20 workers x all-core OpenBLAS pools exhaust memory on Windows.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.plotting import label_axes, save_fig  # noqa: E402
from sensor_fusion.download_logs import PINNED_LOGS  # noqa: E402
from sensor_fusion.report.ekf_study import G3_DEFAULTS  # noqa: E402
from sensor_fusion.metrics import SENSOR_ORDER, clean_json  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parent
KNOBS = ("accel_noise", "gyro_noise", "gyro_bias_rw", "accel_bias_rw", "gps_pos_floor_h", "gps_pos_floor_v",
         "gps_vel_floor", "baro_sigma", "mag_heading_sigma")
FACTORS = (0.25, 0.5, 2.0, 4.0)
LADDER = (0.25, 0.5, 1.0, 2.0, 4.0)
ADMISSIBLE_TOL = 1.15                      # RMS vs EKF2 may grow by at most 15 % on any log
GUARDED = ("pos_horizontal_m", "vel_m_s", "tilt_deg")
NIS_ANOMALY = 100.0
INCREMENT = 3
CAVEAT = "EKF2 reference, not truth"
LOG_COLORS = {"hover": "tab:blue", "aggressive": "tab:red", "other": "tab:green"}


# ------------------------------------------------------------------ config bookkeeping
def fac_str(f: float) -> str:
    return f"{f:g}"


def knob_name(knob: str, f: float) -> str:
    return f"{knob}_x{fac_str(f)}"


def baseline_values() -> dict:
    """Pre-tuning (Group 3) values, fixed: the sweep must stay reproducible after the EkfConfig defaults change."""
    return {k: float(G3_DEFAULTS[k]) for k in KNOBS}


def overrides_from_factors(factors: dict, base: dict) -> dict:
    """``{knob: baseline * factor}`` for EVERY knob (factor 1 unless given), so a config never depends on the
    current EkfConfig defaults."""
    return {k: float(base[k] * factors.get(k, 1.0)) for k in KNOBS}


def step_toward_one(f: float) -> float:
    i = LADDER.index(f)
    return LADDER[i + (1 if i < LADDER.index(1.0) else -1 if i > LADDER.index(1.0) else 0)]


def stage1_configs(base: dict) -> list[dict]:
    cfgs = [{"name": "baseline", "stage": 1, "knob": None, "factor": 1.0, "factors": {}, "overrides": overrides_from_factors({}, base)}]
    for k in KNOBS:
        for f in FACTORS:
            cfgs.append({"name": knob_name(k, f), "stage": 1, "knob": k, "factor": f, "factors": {k: f},
                         "overrides": overrides_from_factors({k: f}, base)})
    return cfgs


# ------------------------------------------------------------------ one (config, log) run (worker)
_CTX: dict = {}


def _get_ctx(entry: dict) -> dict:
    from sensor_fusion.run_flight import load_context
    if entry["id"] not in _CTX:
        _CTX[entry["id"]] = load_context(entry)
    return _CTX[entry["id"]]


def make_record(m: dict) -> dict:
    """Reduce a ``run_one`` metrics dict to the per-(config, log) sweep record."""
    c = m["comparison_vs_ekf2"]
    cons = m["consistency"]
    rms = {k: c[k]["rms"] for k in ("pos_horizontal_m", "pos_vertical_m", "vel_m_s", "tilt_deg", "yaw_deg")}
    per_sensor = {}
    for s in SENSOR_ORDER:
        e = m["nis"].get(s, {})
        late = e.get("after_5s") or {}
        ps = cons["per_sensor"].get(s, {})
        per_sensor[s] = {"dof": e.get("dof"), "n_rejected": e.get("n_rejected"), "n_after_5s": late.get("n"),
                         "mean_nis_after_5s": late.get("mean_nis"), "ratio": ps.get("ratio"),
                         "fraction_inside_95": ps.get("fraction_inside_95")}
    fb = m["final_biases"]
    finite = bool(all(m["all_finite"].values()) and all(v is not None and math.isfinite(v) for v in rms.values())
                  and cons["score"] is not None and math.isfinite(cons["score"]))
    return {"score_nis": cons["score"], "n_sensors": cons["n_sensors"], "per_sensor": per_sensor, "rms": rms,
            "final_biases": {"b_g_rad_s": fb["b_g_rad_s"], "b_g_sigma_rad_s": fb["b_g_sigma_rad_s"],
                             "b_a_m_s2": fb["b_a_m_s2"], "b_a_sigma_m_s2": fb["b_a_sigma_m_s2"]},
            "runtime_s": m["runtime_s"], "finite": finite, "error": None}


def _worker(task):
    name, overrides, entry = task
    from sensor_fusion.run_flight import run_one
    t0 = time.perf_counter()
    try:
        m = run_one(entry, INCREMENT, overrides, figures=False, write=False, ctx=_get_ctx(entry))
        rec = make_record(m)
    except Exception as e:  # recorded, not fixed
        rec = {"score_nis": None, "n_sensors": 0, "per_sensor": {}, "rms": {}, "final_biases": None,
               "runtime_s": time.perf_counter() - t0, "finite": False, "error": f"{type(e).__name__}: {e}"}
    return name, entry["id"], clean_json(rec)


def run_batch(pool, cfgs: list[dict], results: dict, logs=PINNED_LOGS) -> None:
    tasks = [(c["name"], c["overrides"], e) for c in cfgs for e in logs]
    n = len(tasks)
    for i, (name, lid, rec) in enumerate(pool.imap_unordered(_worker, tasks, chunksize=1), 1):
        results.setdefault(name, {})[lid] = rec
        if i % 10 == 0 or i == n:
            print(f"  {i}/{n} runs done", flush=True)


# ------------------------------------------------------------------ scoring / selection
def total_score(recs: dict, log_ids) -> float | None:
    sc = [recs[l]["score_nis"] if l in recs and recs[l]["finite"] else None for l in log_ids]
    if any(s is None for s in sc):
        return None
    return float(np.mean(sc))


def admissible(recs: dict, base_recs: dict, log_ids, tol: float = ADMISSIBLE_TOL) -> bool:
    for l in log_ids:
        r, b = recs.get(l), base_recs[l]
        if r is None or not r["finite"]:
            return False
        for k in GUARDED:
            if not r["rms"][k] <= tol * b["rms"][k]:
                return False
    return True


def evaluate(results: dict, log_ids) -> dict:
    """``{config: {total_score, admissible, ...}}`` for every config in ``results``."""
    base = results["baseline"]
    out = {}
    for name, recs in results.items():
        out[name] = {"total_score": total_score(recs, log_ids), "admissible": admissible(recs, base, log_ids),
                     "all_finite": bool(all(recs[l]["finite"] for l in log_ids if l in recs))}
    return out


def rank_key(summary_entry: dict, factor: float):
    return (summary_entry["total_score"], abs(math.log(factor)))


def select_factors(cfgs1: list[dict], summ: dict) -> tuple[dict, dict]:
    """Per knob: best admissible (baseline included as factor 1) by (total_score, |ln factor|); returns
    ``(selected_factors, per_knob_ranking)``; the ranking lists admissible candidates best first."""
    selected, ranking = {}, {}
    for k in KNOBS:
        cand = [(1.0, "baseline")] + [(c["factor"], c["name"]) for c in cfgs1 if c["knob"] == k]
        ok = [(f, n) for f, n in cand if summ[n]["admissible"] and summ[n]["total_score"] is not None]
        ok.sort(key=lambda fn: rank_key(summ[fn[1]], fn[0]))
        ranking[k] = [{"factor": f, "config": n, "total_score": summ[n]["total_score"]} for f, n in ok]
        selected[k] = ok[0][0] if ok else 1.0
    return selected, ranking


def shape_of(vals: list[float | None]) -> str:
    """Classify total_score along LADDER: monotonic / valley (single interior minimum) / other (incl. missing)."""
    if any(v is None for v in vals):
        return "missing"
    d = np.sign(np.diff(vals))
    d = d[d != 0]
    changes = int(np.sum(d[1:] != d[:-1])) if d.size > 1 else 0
    if changes == 0:
        return "monotonic"
    if changes == 1 and d[0] < 0:
        return "valley"
    return "other"


# ------------------------------------------------------------------ outputs
def knob_figure(knob: str, results: dict, summ: dict, logs, base: dict, out_dir: Path) -> Path:
    names = {f: ("baseline" if f == 1.0 else knob_name(knob, f)) for f in LADDER}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for e in logs:
        lid, role = e["id"], e["role"]
        col = LOG_COLORS.get(role)
        sc = [results[names[f]][lid]["score_nis"] if names[f] in results and results[names[f]][lid]["finite"] else np.nan
              for f in LADDER]
        ph = [results[names[f]][lid]["rms"]["pos_horizontal_m"] if names[f] in results and results[names[f]][lid]["finite"]
              else np.nan for f in LADDER]
        axes[0].plot(LADDER, sc, "o-", color=col, label=f"{role} {lid[:8]}")
        axes[1].plot(LADDER, ph, "o-", color=col, label=f"{role} {lid[:8]}")
        b = results["baseline"][lid]
        axes[1].axhline(ADMISSIBLE_TOL * b["rms"]["pos_horizontal_m"], color=col, ls=":", lw=0.8)
    tot = [summ[names[f]]["total_score"] if names[f] in summ and summ[names[f]]["total_score"] is not None else np.nan
           for f in LADDER]
    axes[0].plot(LADDER, tot, "k--", marker="s", label="mean over logs (total_score)")
    for ax in axes:
        ax.set_xscale("log")
        ax.axvline(1.0, color="grey", lw=1.0)
        ax.set_xticks(LADDER)
        ax.set_xticklabels([fac_str(f) for f in LADDER])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax = axes[0]
    label_axes(ax, f"{knob} factor (x baseline {base[knob]:g}) [-]", "NIS consistency score [-]  (0 = consistent)",
               "NIS consistency score")
    ax.legend(fontsize=7)
    label_axes(axes[1], f"{knob} factor (x baseline {base[knob]:g}) [-]", "horizontal position RMS vs EKF2 [m]",
               "posH RMS vs EKF2 (dotted: 1.15 x baseline limit)")
    fig.suptitle(f"Covariance sweep, one-at-a-time: {knob} (increment 3, 3 pinned logs; baseline = grey line; "
                 f"{CAVEAT})", fontsize=9)
    fig.tight_layout()
    return save_fig(fig, out_dir / "figures" / f"tuning_sweep_{knob}.png")


def nis_ratio_figure(results: dict, logs, out_dir: Path, other: str = "joint") -> Path | None:
    if other not in results:
        return None
    fig, axes = plt.subplots(1, len(logs), figsize=(4.2 * len(logs), 4.2), sharey=True)
    x = np.arange(len(SENSOR_ORDER))
    for ax, e in zip(np.atleast_1d(axes), logs):
        lid = e["id"]
        for j, (nm, col) in enumerate((("baseline", "tab:gray"), (other, "tab:green"))):
            r = [results[nm][lid]["per_sensor"].get(s, {}).get("ratio") for s in SENSOR_ORDER]
            r = [np.nan if v is None else v for v in r]
            ax.bar(x + (j - 0.5) * 0.38, r, 0.38, color=col, label=nm)
        ax.axhline(1.0, color="k", lw=1.0)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(SENSOR_ORDER, rotation=20, fontsize=8)
        label_axes(ax, "sensor [-]", "mean NIS / dof after 5 s [-]", f"{e['role']} {lid[:8]}")
        ax.legend(fontsize=8)
    fig.suptitle(f"NIS ratio per sensor, baseline vs {other} (increment 3; ratio 1 = consistent)", fontsize=10)
    fig.tight_layout()
    return save_fig(fig, out_dir / "figures" / "tuning_sweep_nis_ratio.png")


def _fmt(x, p=3):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{p}f}"


def write_markdown(data: dict, path: Path) -> None:
    logs = data["logs"]
    L = [f"# Covariance sweep (increment 3, {len(logs)} pinned logs)", "",
         f"{CAVEAT}. Score = RMS of ln(NIS mean/dof) over sensors (after 5 s), mean over logs = `total_score`. "
         f"Admissible = posH, vel and tilt RMS vs EKF2 each <= {ADMISSIBLE_TOL} x baseline on every log. "
         f"Wall time {data['wall_time_s']:.0f} s, {data['n_runs']} runs, stage 2 "
         f"{'skipped' if data['stage2_skipped'] else 'run'}.", ""]
    w = data["winner"]
    L += [f"Winner: `{w['config']}` total_score {_fmt(w['total_score'], 4)} (baseline "
          f"{_fmt(data['summary']['baseline']['total_score'], 4)}).", ""]
    head = "| config | admissible | total |" + "".join(
        f" {l['role']} score | posH [m] | vel [m/s] | tilt [deg] | yaw [deg] |" for l in logs)
    L += [head, "|" + "---|" * (3 + 5 * len(logs))]
    for name in data["config_order"]:
        s = data["summary"][name]
        row = f"| {name} | {'yes' if s['admissible'] else 'NO'} | {_fmt(s['total_score'])} |"
        for l in logs:
            r = data["results"][name].get(l["id"])
            if not r or not r["rms"]:
                row += " err | | | | |"
                continue
            rm = r["rms"]
            row += (f" {_fmt(r['score_nis'])} | {_fmt(rm['pos_horizontal_m'])} | {_fmt(rm['vel_m_s'])} | "
                    f"{_fmt(rm['tilt_deg'], 2)} | {_fmt(rm['yaw_deg'], 2)} |")
        L.append(row)
    L += ["", "## Per-sensor NIS ratio (mean NIS / dof, after 5 s)", "",
          "| config | log | " + " | ".join(SENSOR_ORDER) + " |", "|" + "---|" * (2 + len(SENSOR_ORDER))]
    for name in ("baseline", "joint", "joint_halfstep", "joint_no_mag"):
        if name not in data["results"]:
            continue
        for l in logs:
            ps = data["results"][name][l["id"]]["per_sensor"]
            L.append(f"| {name} | {l['role']} | " + " | ".join(_fmt((ps.get(s) or {}).get("ratio")) for s in SENSOR_ORDER) + " |")
    L += ["", "## Per knob (stage 1; admissible candidates, baseline = factor 1)", "",
          "| knob | selected factor | best | runner-up | shape of total_score vs factor |", "|---|---|---|---|---|"]
    for k in data["knobs"]:
        rk = data["per_knob_ranking"][k]
        f1 = (f"x{fac_str(rk[0]['factor'])} ({_fmt(rk[0]['total_score'], 4)})" if rk else "n/a")
        f2 = (f"x{fac_str(rk[1]['factor'])} ({_fmt(rk[1]['total_score'], 4)})" if len(rk) > 1 else "n/a")
        L.append(f"| {k} | x{fac_str(data['selected_factors'][k]) if data['selected_factors'] else 'n/a'} | {f1} | {f2} | "
                 f"{data['knob_shape'][k]} |")
    L += ["", "## Anomalies", ""] + ([f"- {a}" for a in data["anomalies"]] or ["- none"])
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ driver
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Covariance sweep of the ES-EKF (increment 3, three pinned logs).")
    ap.add_argument("--budget-min", type=float, default=30.0, help="wall-clock budget [min]; stage 2 skipped if exceeded")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out-dir", default=str(REPORT_DIR))
    ap.add_argument("--replot", action="store_true", help="regenerate markdown and figures from an existing JSON (no runs)")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.replot:
        data = json.loads((out_dir / "tuning_sweep.json").read_text())
        write_markdown(data, out_dir / "tuning_sweep.md")
        for k in KNOBS:
            knob_figure(k, data["results"], data["summary"], data["logs"], data["baseline_values"], out_dir)
        nis_ratio_figure(data["results"], data["logs"], out_dir)
        print("replotted from", out_dir / "tuning_sweep.json")
        return 0

    logs = list(PINNED_LOGS)
    log_ids = [e["id"] for e in logs]
    base = baseline_values()
    t_start = time.perf_counter()
    results: dict = {}
    cfgs1 = stage1_configs(base)
    cfg_by_name = {c["name"]: c for c in cfgs1}
    n_workers = max(1, min(args.workers, len(cfgs1) * len(logs)))
    print(f"Stage 1: {len(cfgs1)} configs x {len(logs)} logs on {n_workers} workers", flush=True)

    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        run_batch(pool, cfgs1, results, logs)
        t_stage1 = time.perf_counter() - t_start
        summ1 = evaluate({n: results[n] for n in (c["name"] for c in cfgs1)}, log_ids)
        selected, ranking = select_factors(cfgs1, summ1)
        stage2_skipped = t_stage1 > args.budget_min * 60.0
        cfgs2: list[dict] = []
        if stage2_skipped:
            print(f"Stage 1 took {t_stage1:.0f} s > budget {args.budget_min} min: stage 2 skipped", flush=True)
        else:
            half = {k: step_toward_one(f) for k, f in selected.items()}
            no_mag = dict(selected, mag_heading_sigma=1.0)
            for name, fac in (("joint", selected), ("joint_halfstep", half), ("joint_no_mag", no_mag)):
                cfgs2.append({"name": name, "stage": 2, "knob": None, "factor": None, "factors": dict(fac),
                              "overrides": overrides_from_factors(fac, base)})
            print(f"Selected factors: {selected}\nStage 2: {len(cfgs2)} configs x {len(logs)} logs", flush=True)
            run_batch(pool, cfgs2, results, logs)
    wall = time.perf_counter() - t_start

    all_cfgs = cfgs1 + cfgs2
    summary = evaluate(results, log_ids)
    cand = [(n, s) for n, s in summary.items() if s["admissible"] and s["total_score"] is not None]
    cand.sort(key=lambda ns: (ns[1]["total_score"], ns[0]))
    winner = {"config": cand[0][0], "total_score": cand[0][1]["total_score"],
              "overrides": next(c for c in all_cfgs if c["name"] == cand[0][0])["overrides"],
              "factors": next(c for c in all_cfgs if c["name"] == cand[0][0])["factors"]}

    # shape of total_score along the ladder, per knob
    knob_shape = {}
    for k in KNOBS:
        nm = {f: ("baseline" if f == 1.0 else knob_name(k, f)) for f in LADDER}
        knob_shape[k] = shape_of([summary[nm[f]]["total_score"] for f in LADDER])

    # anomalies (reported, not fixed)
    anomalies = []
    for name, recs in results.items():
        for e in logs:
            r = recs.get(e["id"])
            tag = f"{name} / {e['role']}"
            if r is None:
                anomalies.append(f"{tag}: no result")
                continue
            if r["error"]:
                anomalies.append(f"{tag}: run failed ({r['error']})")
            elif not r["finite"]:
                anomalies.append(f"{tag}: non-finite result (NaN/inf in state, RMS or score)")
            for s, ps in r["per_sensor"].items():
                v = ps.get("mean_nis_after_5s")
                if v is not None and v > NIS_ANOMALY:
                    anomalies.append(f"{tag}: {s} mean NIS after 5 s = {v:.1f} > {NIS_ANOMALY:g}")
    for k in KNOBS:
        rk = ranking[k]
        if rk and rk[0]["factor"] in (FACTORS[0], FACTORS[-1]):
            anomalies.append(f"{k}: best stage-1 factor x{fac_str(rk[0]['factor'])} is at the edge of the range")
        if knob_shape[k] not in ("monotonic", "valley"):
            anomalies.append(f"{k}: total_score vs factor is not monotonic / single-valley ({knob_shape[k]})")
    if stage2_skipped:
        anomalies.append("stage 2 skipped (stage 1 exceeded the time budget)")

    data = {
        "meta": {"spec": "docs/specs/tuning_writeup.md section 5", "increment": INCREMENT, "caveat": CAVEAT,
                 "admissibility_tol": ADMISSIBLE_TOL, "guarded_rms": list(GUARDED), "budget_min": args.budget_min,
                 "workers": n_workers, "python": sys.version.split()[0], "numpy": np.__version__,
                 "platform": platform.platform(), "mp_start_method": "spawn",
                 "scoring": "score_nis = consistency_score (after 5 s); total_score = mean over logs; lowest "
                            "admissible wins; baseline counts as factor 1 in per-knob selection"},
        "knobs": list(KNOBS), "factors": list(FACTORS), "ladder": list(LADDER), "baseline_values": base,
        "logs": [{"id": e["id"], "role": e["role"]} for e in logs],
        "wall_time_s": wall, "stage1_wall_time_s": t_stage1, "stage2_skipped": stage2_skipped,
        "n_runs": sum(len(v) for v in results.values()),
        "config_order": [c["name"] for c in all_cfgs],
        "configs": {c["name"]: {"stage": c["stage"], "knob": c["knob"], "factor": c["factor"],
                                "factors": c["factors"], "overrides": c["overrides"]} for c in all_cfgs},
        "results": {c["name"]: results[c["name"]] for c in all_cfgs},
        "summary": {c["name"]: summary[c["name"]] for c in all_cfgs},
        "selected_factors": selected, "per_knob_ranking": ranking, "knob_shape": knob_shape,
        "winner": winner, "anomalies": anomalies,
    }
    data = clean_json(data)
    (out_dir / "tuning_sweep.json").write_text(json.dumps(data, indent=2))
    write_markdown(data, out_dir / "tuning_sweep.md")
    for k in KNOBS:
        knob_figure(k, results, summary, logs, base, out_dir)
    nis_ratio_figure(results, logs, out_dir)

    print(f"\nWall time {wall:.0f} s, {data['n_runs']} runs. Winner: {winner['config']} "
          f"(total {winner['total_score']:.4f}; baseline {summary['baseline']['total_score']:.4f})")
    print(f"Wrote {out_dir / 'tuning_sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
