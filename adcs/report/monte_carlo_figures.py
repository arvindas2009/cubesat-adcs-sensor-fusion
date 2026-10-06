"""Group 8 Monte Carlo figures and summary table (spec docs/specs/adcs_closed_loop.md section 5.2).

Run:  .venv/Scripts/python -m adcs.report.monte_carlo_figures [monte_carlo_design.json ...]

Inputs
------
``adcs/report/monte_carlo_{design,ideal}.json`` as written by ``python -m adcs.monte_carlo``
(whichever exist, or the paths given). JSON ``null`` is read as NaN.

Outputs
-------
``adcs/report/figures/mc_{detumble_time,acquire_time,pointing_rms,pointing_max,knowledge_rms,
wheel_speed,sensitivity}.png`` (PNG, 150 dpi; both campaigns overlaid when both exist;
requirement lines dashed, taken from ``config.requirements`` of the JSON; failing run indices
annotated) and ``adcs/report/monte_carlo_summary.md`` (tables generated only from the JSON).
Nothing is recomputed from the simulation and nothing is tuned.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.plotting import label_axes, save_fig  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parent
FIG_DIR = REPORT_DIR / "figures"
SUMMARY_PATH = REPORT_DIR / "monte_carlo_summary.md"

CAMPAIGN_ORDER = ("design", "ideal")
CAMPAIGN_STYLE = {
    "design": {"color": "tab:blue", "label": "design (realistic sensors + design MEKF)"},
    "ideal": {"color": "tab:orange", "label": "ideal (ideal sensors + Group 7 MEKF)"},
}
MAX_LABELS = 12   # max annotated run indices per axes (the rest is counted in the text)

# figure file, row key, x label, requirement key (or None), requirement name in `requirements`,
# whether the requirement is a goal, log-x
HIST_SPECS = [
    ("mc_detumble_time", "detumble_time_orbits", "time until true rate < 0.5 deg/s [orbits]",
     "detumble_orbits_max", "detumble_orbits_max", False, False),
    ("mc_acquire_time", "switch_time_orbits", "time to first switch to nadir pointing [orbits]",
     "acquire_orbits_max", "acquire_orbits_max", False, False),
    ("mc_pointing_rms", "nadir_rms_deg", "steady-state nadir pointing error RMS [deg]",
     "pointing_rms_goal_deg", "pointing_rms_goal_deg", True, True),
    ("mc_pointing_max", "nadir_max_deg", "steady-state nadir pointing error maximum [deg]",
     "pointing_max_deg", "pointing_max_deg", False, True),
    ("mc_knowledge_rms", "know_rms_deg", "steady-state attitude-knowledge error RMS [deg]",
     "knowledge_rms_deg", "knowledge_rms_deg", False, True),
    ("mc_wheel_speed", "wheel_max_frac_pointing", "maximum wheel speed in pointing [fraction of limit]",
     "wheel_frac_max", "wheel_frac_max", False, False),
]
# requirement key (config.requirements) -> name in the `requirements` (check_requirements) dict
# (same names), metric used by that check is the row key above.


# ----------------------------------------------------------------------------- loading
def _num(v) -> float:
    """JSON value -> float (``None`` -> NaN, non-numeric -> NaN)."""
    if v is None or isinstance(v, bool):
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _col(runs, key) -> np.ndarray:
    return np.array([_num(r.get(key)) for r in runs], float)


def _idx(runs) -> np.ndarray:
    return np.array([int(r["index"]) for r in runs], int)


def load_campaigns(paths=None) -> dict:
    """Read the Monte Carlo JSON files; returns ``{campaign: data}`` in design, ideal order."""
    if paths is None:
        paths = [REPORT_DIR / f"monte_carlo_{c}.json" for c in CAMPAIGN_ORDER]
    out = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        out[data.get("campaign", p.stem.replace("monte_carlo_", ""))] = data
    return dict(sorted(out.items(), key=lambda kv: CAMPAIGN_ORDER.index(kv[0])
                       if kv[0] in CAMPAIGN_ORDER else 99))


def _requirements_config(camps) -> dict:
    for d in camps.values():
        req = d.get("config", {}).get("requirements")
        if req:
            return req
    return {}


def _style(name):
    return CAMPAIGN_STYLE.get(name, {"color": "tab:green", "label": name})


def _title_suffix(camps) -> str:
    parts = []
    for name, d in camps.items():
        c = d["config"]
        parts.append(f"{name}: n={c['n']}")
    c0 = next(iter(camps.values()))["config"]
    return (f"Monte Carlo, {c0['orbits']:g} orbits per run, 2-10 deg/s initial tumble, "
            f"inertia +/-{100 * c0['inertia_tol']:g} %  ({'; '.join(parts)})")


# ----------------------------------------------------------------------------- figures
def _annotate_failures(ax, items, xlog=False, y0=0.0, dy=0.0, color="k"):
    """Mark failing runs as ticks on the x axis and label up to ``MAX_LABELS`` indices.

    ``items`` = list of ``(x, index)`` with finite x. Labels are staggered vertically.
    """
    items = sorted(items)
    if not items:
        return
    xs = np.array([x for x, _ in items])
    ax.plot(xs, np.full(xs.size, y0), "v", color=color, ms=6, mec="k", mew=0.5, clip_on=False, zorder=5)
    for j, (x, i) in enumerate(items[:MAX_LABELS]):
        ax.annotate(str(i), (x, y0), xytext=(0, 8 + 9 * (j % 3)), textcoords="offset points",
                    ha="center", fontsize=7, color=color,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=color))


def _hist_figure(camps, fname, key, xlabel, req_key, req_name, is_goal, xlog, req):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    vals_all = [v for d in camps.values() for v in _col(d["runs"], key) if np.isfinite(v)]
    thr = req.get(req_key)
    if vals_all:
        lo, hi = min(vals_all), max(vals_all)
        if thr is not None and np.isfinite(thr):
            lo, hi = min(lo, thr), max(hi, thr)
        if xlog and lo > 0:
            bins = np.geomspace(lo * 0.95, hi * 1.05, 31)
            ax.set_xscale("log")
        else:
            span = (hi - lo) or max(abs(hi), 1.0) * 0.1
            bins = np.linspace(lo - 0.02 * span, hi + 0.02 * span, 31)
    else:
        bins = 10
    notes = []
    for ci, (name, d) in enumerate(camps.items()):
        st = _style(name)
        runs = d["runs"]
        x = _col(runs, key)
        idx = _idx(runs)
        fin = np.isfinite(x)
        n_tot = len(runs)
        ax.hist(x[fin], bins=bins, color=st["color"], alpha=0.55, edgecolor="k", linewidth=0.4,
                label=f"{st['label']}: {int(fin.sum())}/{n_tot} finite, "
                      f"median {np.median(x[fin]):.3g}" if fin.any() else f"{st['label']}: no finite values")
        failing = set(d["requirements"].get(req_name, {}).get("failing", []))
        pos = [(float(x[k]), int(idx[k])) for k in range(len(runs)) if fin[k] and int(idx[k]) in failing]
        nan_fail = sorted(int(idx[k]) for k in range(len(runs)) if not fin[k])
        _annotate_failures(ax, pos, color=st["color"], y0=0.0)
        if len(pos) > MAX_LABELS:
            notes.append(f"{name}: {len(pos)} runs beyond the line (first {MAX_LABELS} labelled)")
        elif pos:
            notes.append(f"{name}: failing {sorted(i for _, i in pos)}")
        if nan_fail:
            shown = nan_fail if len(nan_fail) <= MAX_LABELS else nan_fail[:MAX_LABELS] + ["..."]
            notes.append(f"{name}: NaN (not drawn) runs {shown}")
    if thr is not None:
        ax.axvline(thr, color="r", ls="--", lw=1.5,
                   label=f"{'goal' if is_goal else 'requirement'} {thr:g}")
    label_axes(ax, xlabel, "number of runs", f"{_title_suffix(camps)}\n{key}")
    ax.legend(fontsize=8, loc="upper right")
    if notes:
        ax.text(0.01, 0.99, "\n".join(notes), transform=ax.transAxes, va="top", ha="left", fontsize=7,
                bbox=dict(boxstyle="round", fc="w", ec="0.6", alpha=0.85))
    fig.tight_layout()
    save_fig(fig, FIG_DIR / f"{fname}.png")


def _sensitivity_figure(camps, req):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    notes = []
    for name, d in camps.items():
        st = _style(name)
        runs = d["runs"]
        rate = _col(runs, "rate_degps")
        spread = np.array([(max(r["inertia_scale"]) - min(r["inertia_scale"])) if r.get("inertia_scale") else np.nan
                           for r in runs], float)
        y = _col(runs, "nadir_rms_deg")
        idx = _idx(runs)
        fin = np.isfinite(y)
        allfail = set(d["requirements"].get("all_requirements", {}).get("failing", []))
        for ax, x in zip(axes, (rate, spread)):
            ax.scatter(x[fin], y[fin], s=22, color=st["color"], alpha=0.7, edgecolor="k", linewidth=0.3,
                       label=f"{st['label']} ({int(fin.sum())}/{len(runs)} finite)")
            pos = [(float(x[k]), float(y[k]), int(idx[k])) for k in range(len(runs))
                   if fin[k] and np.isfinite(x[k]) and int(idx[k]) in allfail]
            for j, (xx, yy, i) in enumerate(pos[:MAX_LABELS]):
                ax.scatter([xx], [yy], s=70, facecolors="none", edgecolors="r", linewidths=1.2, zorder=5)
                ax.annotate(str(i), (xx, yy), xytext=(5, 5 + 7 * (j % 2)), textcoords="offset points",
                            fontsize=7, color="r")
        if allfail:
            nan_runs = sorted(int(idx[k]) for k in range(len(runs)) if not fin[k])
            msg = f"{name}: runs failing any requirement {sorted(allfail) if len(allfail) <= MAX_LABELS else str(len(allfail)) + ' runs'}"
            if nan_runs:
                msg += f"; NaN nadir RMS (not drawn) {nan_runs[:MAX_LABELS]}"
            notes.append(msg)
    goal = req.get("pointing_rms_goal_deg")
    if goal is not None:
        for ax in axes:
            ax.axhline(goal, color="r", ls="--", lw=1.3, label=f"goal {goal:g} deg")
    c0 = next(iter(camps.values()))["config"]
    axes[0].set_xlim(0.9 * c0["rate_range_degps"][0], 1.05 * c0["rate_range_degps"][1])
    label_axes(axes[0], "initial body rate [deg/s]", "steady-state nadir RMS error [deg]")
    label_axes(axes[1], "inertia scale spread, max - min of the three factors [-]", "")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8, loc="best")
    if notes:
        axes[1].text(0.99, 0.01, "\n".join(notes), transform=axes[1].transAxes, va="bottom", ha="right",
                     fontsize=7, bbox=dict(boxstyle="round", fc="w", ec="0.6", alpha=0.85))
    fig.suptitle(f"{_title_suffix(camps)}\nnadir RMS sensitivity to initial rate and truth inertia error "
                 "(red circles: runs failing a requirement)", fontsize=10)
    fig.tight_layout()
    save_fig(fig, FIG_DIR / "mc_sensitivity.png")


# ----------------------------------------------------------------------------- summary
def _fmt(x, spec=".3g") -> str:
    x = _num(x)
    return "nan" if not np.isfinite(x) else format(x, spec)


def _stats(runs, key):
    v = _col(runs, key)
    v = v[np.isfinite(v)]
    if not v.size:
        return float("nan"), float("nan"), float("nan"), 0
    return float(np.median(v)), float(np.percentile(v, 95)), float(v.max()), int(v.size)


def _fail_str(lst, maxn=15) -> str:
    if not lst:
        return "none"
    if len(lst) <= maxn:
        return ", ".join(str(i) for i in lst)
    return ", ".join(str(i) for i in lst[:maxn]) + f", ... ({len(lst)} total)"


REQ_DESCR = {
    "detumble_orbits_max": "detumble time <= {v:g} orbits",
    "acquire_orbits_max": "first switch to pointing <= {v:g} orbits",
    "pointing_max_deg": "steady-state nadir max <= {v:g} deg",
    "knowledge_rms_deg": "knowledge RMS <= {v:g} deg",
    "pointing_rms_goal_deg": "nadir RMS <= {v:g} deg (GOAL)",
    "wheel_frac_max": "wheel fraction in pointing < {v:g}",
    "success": "success (switch, no fallback, non-empty window)",
    "all_requirements": "all requirements (goal excluded)",
}

STAT_KEYS = [
    ("detumble_time_orbits", "detumble time [orbits]"),
    ("switch_time_orbits", "first switch time [orbits]"),
    ("nadir_rms_deg", "nadir RMS [deg]"),
    ("nadir_max_deg", "nadir max [deg]"),
    ("know_rms_deg", "knowledge RMS [deg]"),
    ("nees_mean", "attitude NEES (mean)"),
    ("wheel_max_frac_pointing", "wheel max fraction (pointing)"),
    ("wheel_max_frac", "wheel max fraction (window)"),
]


def _anomalies(name, d):
    """List of anomaly strings for one campaign (from the JSON only)."""
    runs = d["runs"]
    out = []
    err = [r for r in runs if "error" in r]
    for r in err:
        out.append(f"run {r['index']} crashed: {r['error']} (rate {_fmt(r.get('rate_degps'))} deg/s, "
                   f"inertia scale {[round(v, 3) for v in r.get('inertia_scale', [])]})")
    nosw = [int(r["index"]) for r in runs if not np.isfinite(_num(r.get("switch_time_s")))]
    if nosw:
        out.append(f"runs that never switch to pointing: {_fail_str(nosw)}")
    fb = [(int(r["index"]), int(_num(r.get("n_fallbacks")))) for r in runs
          if np.isfinite(_num(r.get("n_fallbacks"))) and _num(r.get("n_fallbacks")) > 0]
    if fb:
        out.append("runs with fallbacks to detumble (index: n_fallbacks): "
                   + ", ".join(f"{i}: {n}" for i, n in fb[:MAX_LABELS * 2]))
    keys = sorted({k for r in runs for k in r if k not in ("error",)})
    nan_keys = {}
    for k in keys:
        bad = [int(r["index"]) for r in runs if k in r and (r[k] is None)]
        if bad and k not in ("index",):
            nan_keys[k] = bad
    for k, bad in nan_keys.items():
        if k.startswith("nis_per_dof_") and len(bad) == len(runs):
            out.append(f"{k} is NaN in all {len(runs)} runs (structural: the filter mode of this campaign "
                       "performs no such update)")
        else:
            out.append(f"NaN metric {k}: {len(bad)} runs ({_fail_str(bad, 10)})")
    for rname, rd in d["requirements"].items():
        if np.isfinite(_num(rd["fraction"])) and rd["fraction"] < 0.95:
            out.append(f"requirement {rname}: fraction {rd['fraction']:.3f} < 0.95, failing {_fail_str(rd['failing'])}")
    return out


def _worst_runs(d, n=3, key="nadir_max_deg"):
    runs = d["runs"]
    fin = [r for r in runs if np.isfinite(_num(r.get(key)))]
    fin.sort(key=lambda r: -_num(r.get(key)))
    return fin[:n]


def write_summary(camps, path=SUMMARY_PATH) -> Path:
    req_cfg = _requirements_config(camps)
    L = ["# Monte Carlo summary", "",
         "Generated by `adcs/report/monte_carlo_figures.py` from `adcs/report/monte_carlo_{design,ideal}.json` "
         "only. Requirement thresholds are the assumed values recorded in the JSON (`config.requirements`). "
         "Angles in degrees, times in orbits unless stated; stats over runs with a finite value.", ""]
    for name, d in camps.items():
        c, runs = d["config"], d["runs"]
        L += [f"## Campaign `{name}`", "",
              f"n = {len(runs)} runs, {c['orbits']:g} orbits per run, base seed {c['base_seed']}, "
              f"initial rate {c['rate_range_degps'][0]:g}-{c['rate_range_degps'][1]:g} deg/s, "
              f"inertia tolerance +/-{100 * c['inertia_tol']:g} %, wall time {_fmt(d.get('wall_time_s'), '.0f')} s "
              f"({_fmt(_num(d.get('wall_time_s')) / 60.0, '.1f')} min).", "",
              f"Success fraction: {_fmt(d['aggregate'].get('success_fraction'), '.3f')}", "",
              "| Requirement | thresholds | fraction met | n met / n | failing run indices |",
              "|---|---|---|---|---|"]
        for rname, rd in d["requirements"].items():
            key = rname
            descr = REQ_DESCR.get(rname, rname)
            descr = descr.format(v=req_cfg.get(key, float("nan"))) if "{v" in descr else descr
            L.append(f"| `{rname}` | {descr} | {_fmt(rd['fraction'], '.3f')} | {rd['n_met']} / {rd['n']} "
                     f"| {_fail_str(rd['failing'])} |")
        L += ["", "| Metric | median | p95 | max | n finite |", "|---|---|---|---|---|"]
        for key, lab in STAT_KEYS:
            med, p95, mx, nf = _stats(runs, key)
            L.append(f"| {lab} (`{key}`) | {_fmt(med)} | {_fmt(p95)} | {_fmt(mx)} | {nf} |")
        L.append("")
        an = _anomalies(name, d)
        L += ["Anomalies:", ""] + ([f"- {a}" for a in an] if an else ["- none"]) + [""]
        worst = _worst_runs(d)
        L += ["Worst 3 runs by `nadir_max_deg`:", "",
              "| index | rate [deg/s] | inertia scale | switch time [s] | nadir RMS [deg] | nadir max [deg] "
              "| know RMS sunlit [deg] | know RMS eclipse [deg] | NEES | n_fallbacks "
              "| mag bias true norm [nT] | mag bias err end [nT] |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in worst:
            sc = ", ".join(f"{v:.3f}" for v in r.get("inertia_scale", []))
            L.append(f"| {r['index']} | {_fmt(r.get('rate_degps'), '.2f')} | [{sc}] | {_fmt(r.get('switch_time_s'), '.0f')} "
                     f"| {_fmt(r.get('nadir_rms_deg'))} | {_fmt(r.get('nadir_max_deg'))} "
                     f"| {_fmt(r.get('know_rms_sunlit_deg'))} | {_fmt(r.get('know_rms_eclipse_deg'))} "
                     f"| {_fmt(r.get('nees_mean'))} | {_fmt(r.get('n_fallbacks'), '.0f')} "
                     f"| {_fmt(r.get('mag_bias_true_nT'), '.0f')} | {_fmt(r.get('mag_bias_err_end_nT'), '.1f')} |")
        L.append("")
    path = Path(path)
    path.write_text("\n".join(L), encoding="utf-8")
    return path


# ----------------------------------------------------------------------------- entry point
def main(paths=None) -> int:
    """Make the ``mc_*.png`` figures and ``monte_carlo_summary.md`` from the campaign JSON files."""
    camps = load_campaigns(paths)
    if not camps:
        print("no Monte Carlo JSON found; nothing to do")
        return 1
    req = _requirements_config(camps)
    for fname, key, xlabel, req_key, req_name, is_goal, xlog in HIST_SPECS:
        _hist_figure(camps, fname, key, xlabel, req_key, req_name, is_goal, xlog, req)
    _sensitivity_figure(camps, req)
    write_summary(camps)
    print(f"wrote {len(HIST_SPECS) + 1} figures to {FIG_DIR} and {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
