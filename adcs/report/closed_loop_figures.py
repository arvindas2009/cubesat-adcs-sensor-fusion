"""Group 8 closed-loop figures and metrics (spec docs/specs/adcs_closed_loop.md section 5.1).

Run:  .venv/Scripts/python -m adcs.report.closed_loop_figures [--workers 20]

Outputs
-------
``adcs/report/figures/closed_loop_{vs_pk,estimation,cost,breakdown,bdot_gain}.png`` and
``adcs/report/closed_loop_metrics.json``. Every run starts from ``tumble_state(10 deg/s, s)`` with
the sensor seed ``s``, default config, ``log_every = 1`` (1 s). Workers return scalar summaries
(``summarize_closed_loop``, or the equivalent subset for the perfect-knowledge ``run(..., "auto")``);
full arrays (decimated to every 10 s) are returned only for the plotted runs: seed 0 of ``pk``,
``ideal_g7`` and ``real_design``. Every number reported comes from the JSON written here. Nothing
is tuned; anomalies are flagged in ``checks`` and reported, never corrected.

Steady-state window (all cases): mode == pointing and ``t >= first switch + 1 orbit``.
The Group 6 B-dot gain values are read from ``control_metrics.json``
(``detumble_sensitivity.gain_grid``).
"""
from __future__ import annotations

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from dataclasses import asdict, replace  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from adcs.config import DEFAULT  # noqa: E402
from adcs.control import bdot_gain  # noqa: E402
from adcs.orbit import orbital_period  # noqa: E402
from adcs.simulate import (IDEAL_SENSORS, MEKF_DESIGN, MEKF_G7, MODE_POINTING, REALISTIC_SENSORS,  # noqa: E402
                           ControlConfig, ModeConfig, detumble_time, run, run_closed_loop,
                           summarize_closed_loop, tumble_state)
from common.plotting import label_axes, save_fig  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parent
FIG_DIR = REPORT_DIR / "figures"
METRICS_PATH = REPORT_DIR / "closed_loop_metrics.json"
G6_METRICS_PATH = REPORT_DIR / "control_metrics.json"

DEG = np.pi / 180.0
PERIOD = float(orbital_period(DEFAULT.orbit.a))
RATE_DEG = 10.0
DECIM = 10                       # plotted runs are decimated to every 10 s
SEEDS5 = tuple(range(5))
SEEDS3 = tuple(range(3))
ORBITS = 6.0
BDOT_ORBITS = 3.0
BDOT_SCALES = (0.5, 1.0, 2.0, 4.0)
PLOT_KEYS = (("pk", 0), ("ideal_g7", 0), ("real_design", 0))
DETUMBLE_ONLY = ModeConfig(switch_att_sigma=0.0)
BDOT_K_NOM = float(bdot_gain(DEFAULT.spacecraft.inertia, DEFAULT.orbit.a, DEFAULT.orbit.inc))

R, D = REALISTIC_SENSORS, MEKF_DESIGN
# case key -> dict(kind, sensors, mekf, seeds, orbits, [bdot scale]); spec section 5.1 table
CASES: dict[str, dict] = {
    "pk": dict(kind="pk", seeds=SEEDS5, orbits=ORBITS),
    "ideal_g7": dict(kind="cl", sensors=IDEAL_SENSORS, mekf=MEKF_G7, seeds=SEEDS5, orbits=ORBITS),
    "real_g7": dict(kind="cl", sensors=R, mekf=MEKF_G7, seeds=SEEDS5, orbits=ORBITS),
    "real_design": dict(kind="cl", sensors=R, mekf=D, seeds=SEEDS5, orbits=ORBITS),
    "no_albedo": dict(kind="cl", sensors=replace(R, albedo=0.0), mekf=replace(D, albedo_model=0.0),
                      seeds=SEEDS3, orbits=ORBITS),
    "const_albedo": dict(kind="cl", sensors=replace(R, albedo_sigma=0.0), mekf=D, seeds=SEEDS3, orbits=ORBITS),
    "no_field_error": dict(kind="cl", sensors=replace(R, field_residual_sigma=0.0, onboard_field_degree=13),
                           mekf=D, seeds=SEEDS3, orbits=ORBITS),
    "no_fov": dict(kind="cl", sensors=replace(R, css_fov=None), mekf=replace(D, css_fov=None),
                   seeds=SEEDS3, orbits=ORBITS),
    "no_mag_bias_states": dict(kind="cl", sensors=R, mekf=replace(D, mag_bias=False),
                               seeds=SEEDS3, orbits=ORBITS),
    "design_no_albedo_truth": dict(kind="cl", sensors=IDEAL_SENSORS, mekf=D, seeds=SEEDS3, orbits=ORBITS),
}
for _s in BDOT_SCALES:
    CASES[f"bdot_x{_s:g}"] = dict(kind="bdot", sensors=R, mekf=D, seeds=SEEDS5, orbits=BDOT_ORBITS, scale=_s)

BREAKDOWN = ("no_albedo", "const_albedo", "no_field_error", "no_fov", "no_mag_bias_states",
             "design_no_albedo_truth")
BREAKDOWN_LABELS = {"real_design": "real_design\n(reference)", "no_albedo": "no_albedo", "const_albedo": "const_albedo",
                    "no_field_error": "no_field\n_error", "no_fov": "no_fov",
                    "no_mag_bias_states": "no_mag_bias\n_states",
                    "design_no_albedo_truth": "design_no_\nalbedo_truth"}
BREAKDOWN_DESC = {
    "real_design": "REALISTIC_SENSORS + MEKF_DESIGN",
    "no_albedo": "truth albedo = 0; filter albedo_model = 0",
    "const_albedo": "truth albedo_sigma = 0 (constant 0.30)",
    "no_field_error": "truth field residual = 0; onboard degree 13",
    "no_fov": "truth css_fov = None; filter css_fov = None",
    "no_mag_bias_states": "filter mag_bias = False",
    "design_no_albedo_truth": "IDEAL_SENSORS + MEKF_DESIGN (albedo model, none in truth)",
}


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def _clean(o):
    """Recursively convert to JSON-safe python types (non-finite floats -> None)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    return o


def intervals(flag) -> list[tuple[int, int]]:
    """Inclusive index runs ``(i0, i1)`` where ``flag`` is True."""
    flag = np.asarray(flag, bool)
    idx = np.flatnonzero(flag)
    if idx.size == 0:
        return []
    brk = np.flatnonzero(np.diff(idx) > 1)
    return list(zip(np.r_[idx[0], idx[brk + 1]].tolist(), np.r_[idx[brk], idx[-1]].tolist()))


def shade_eclipse(ax, t, lit, label=True):
    dt = float(t[1] - t[0])
    for j, (i0, i1) in enumerate(intervals(~lit)):
        ax.axvspan(t[i0] / PERIOD, (t[i1] + dt) / PERIOD, color="0.75", alpha=0.5, lw=0,
                   label="eclipse" if (label and j == 0) else None)


def _pk_summary(res) -> dict:
    """Perfect-knowledge subset with the closed-loop window definition (first switch + 1 orbit, pointing)."""
    sw = res.meta["switch_time"]
    out = {"switch_time_s": sw, "switch_time_orbits": sw / PERIOD,
           "detumble_time_s": detumble_time(res.t, res.rate, res.meta["ctrl"]["detumble_threshold"])}
    out["detumble_time_orbits"] = out["detumble_time_s"] / PERIOD
    nan = float("nan")
    win = (res.mode == MODE_POINTING) & (res.t >= sw + PERIOD) if np.isfinite(sw) else np.zeros(res.t.size, bool)
    out["n_steady"] = int(win.sum())
    out.update(nadir_rms_deg=nan, nadir_p95_deg=nan, nadir_p99_deg=nan, nadir_max_deg=nan, wheel_max_frac=nan)
    ms = res.meta["wheel"]["max_speed"]
    if win.any():
        e = res.nadir_err[win] / DEG
        out.update(nadir_rms_deg=float(np.sqrt(np.mean(e ** 2))), nadir_p95_deg=float(np.percentile(e, 95)),
                   nadir_p99_deg=float(np.percentile(e, 99)), nadir_max_deg=float(e.max()),
                   wheel_max_frac=float(np.max(np.abs(res.wheel_speed[win])) / ms))
    pt = res.mode == MODE_POINTING
    out["wheel_max_frac_pointing"] = float(np.max(np.abs(res.wheel_speed[pt])) / ms) if pt.any() else nan
    out["success"] = bool(np.isfinite(sw) and out["n_steady"] > 0)
    return out


def _block_mean(x, step=DECIM):
    idx = np.arange(0, x.size, step)
    return np.add.reduceat(np.asarray(x, float), idx) / np.diff(np.r_[idx, x.size])


def _plot_payload(res, closed_loop: bool) -> dict:
    """Decimated arrays (every 10 s) of a plotted run."""
    s = slice(None, None, DECIM)
    ms = res.meta["wheel"]["max_speed"]
    d = {"t": res.t[s], "rate": res.rate[s], "nadir_err": res.nadir_err[s], "mode": res.mode[s],
         "wheel_frac": (np.max(np.abs(res.wheel_speed), axis=1) / ms)[s], "t_end": res.meta["t_end"],
         "switch_time": res.meta["switch_time"]}
    if closed_loop:
        d.update(lit=res.lit[s], know_err=res.know_err[s],
                 sigma=np.sqrt(np.diagonal(res.P_att, axis1=1, axis2=2))[s], bm_est=res.bm_est[s],
                 mag_bias_true=np.asarray(res.meta["mag_bias_true"], float),
                 gyro_bias_err=np.linalg.norm(res.b_est - res.gyro_bias, axis=1)[s],
                 sun_used=_block_mean(res.sun_used), t_init=res.meta["t_init"])
    return d


# ---------------------------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------------------------
def _worker(job: dict):
    """One simulation; returns ``(case, seed, payload)``. Top-level so the pool can pickle it."""
    t0 = time.perf_counter()
    name, seed = job["case"], job["seed"]
    c = CASES[name]
    y0 = tumble_state(np.deg2rad(RATE_DEG), seed)
    t_end = c["orbits"] * PERIOD
    plot = (name, seed) in PLOT_KEYS
    if c["kind"] == "pk":
        res = run(y0, t_end, "auto", log_every=1)
        payload = {"summary": _pk_summary(res)}
        if plot:
            payload["plot"] = _plot_payload(res, False)
    else:
        ctrl, modes = ControlConfig(), ModeConfig()
        if c["kind"] == "bdot":
            ctrl = ControlConfig(bdot_k=float(c["scale"] * BDOT_K_NOM))
            modes = DETUMBLE_ONLY
        res = run_closed_loop(y0, t_end, seed=seed, sensors=c["sensors"], mekf_cfg=c["mekf"],
                              ctrl=ctrl, modes=modes, log_every=1)
        payload = {"summary": summarize_closed_loop(res)}
        if c["kind"] == "bdot":
            last = res.t >= res.t[-1] - PERIOD
            payload["summary"].update(
                bdot_k=float(res.meta["bdot_k"]), bdot_scale=float(c["scale"]),
                mean_rate_last_orbit_degps=float(np.mean(res.rate[last]) / DEG),
                final_rate_degps=float(res.rate[-1] / DEG),
                n_samples=int(res.t.size))
        if plot:
            payload["plot"] = _plot_payload(res, True)
    payload["wall_s"] = time.perf_counter() - t0
    return name, seed, payload


# ---------------------------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------------------------
def _stats(vals) -> dict:
    a = np.array([np.nan if v is None else float(v) for v in vals], float)
    f = a[np.isfinite(a)]
    return {"mean": float(f.mean()) if f.size else None, "min": float(f.min()) if f.size else None,
            "max": float(f.max()) if f.size else None, "n_finite": int(f.size), "n": int(a.size)}


def aggregate_case(per_seed: dict) -> dict:
    seeds = sorted(per_seed)
    keys = [k for k, v in per_seed[seeds[0]].items() if isinstance(v, (int, float, np.floating, np.integer))
            and not isinstance(v, (bool, np.bool_))]
    agg = {k: _stats([per_seed[s].get(k) for s in seeds]) for k in keys}
    succ = [bool(per_seed[s]["success"]) for s in seeds if "success" in per_seed[s]]
    if succ:
        agg["success_fraction"] = float(np.mean(succ))
        agg["n_success"] = int(np.sum(succ))
    return agg


def _val(case_agg, key, stat="mean"):
    return case_agg[key][stat]


# ---------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------
CASE_STYLE = {"pk": ("C2", "perfect knowledge (pk)"), "ideal_g7": ("C0", "ideal sensors + Group 7 filter"),
              "real_design": ("C3", "realistic sensors + design filter"),
              "real_g7": ("C1", "realistic sensors + Group 7 filter")}


def fig_vs_pk(plots: dict):
    lit = plots[("ideal_g7", 0)]["lit"]
    t_lit = plots[("ideal_g7", 0)]["t"]
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(11, 13),
                            gridspec_kw={"height_ratios": [2, 2, 1.5, 1.0]})
    for a in axs:
        shade_eclipse(a, t_lit, lit, label=(a is axs[0]))
    floor = 1e-3
    for key in ("pk", "ideal_g7", "real_design"):
        p = plots[(key, 0)]
        col, lab = CASE_STYLE[key]
        x = p["t"] / PERIOD
        axs[0].semilogy(x, np.maximum(p["rate"] / DEG, 1e-6), color=col, lw=1.0, label=lab)
        axs[1].semilogy(x, np.maximum(p["nadir_err"] / DEG, floor), color=col, lw=0.9, label=lab)
        axs[2].plot(x, p["wheel_frac"], color=col, lw=0.9, label=lab)
        if np.isfinite(p["switch_time"]):
            for a in axs[:3]:
                a.axvline(p["switch_time"] / PERIOD, color=col, ls=":", lw=0.9)
    axs[0].axhline(0.5, color="k", ls="--", lw=0.8, label="0.5 deg/s threshold")
    axs[0].set_ylim(1e-2, 20)
    axs[0].legend(fontsize=8, loc="upper right")
    label_axes(axs[0], "", "true angular rate |ω| [deg/s]",
               "Closed loop vs perfect knowledge, tumble from 10 deg/s, seed 0, 6 orbits "
               "(10 s samples; dotted = switch to pointing)")
    axs[1].axhline(5.0, color="k", ls="--", lw=0.8, label="5 deg (requirement)")
    axs[1].legend(fontsize=8, loc="upper right")
    label_axes(axs[1], "", f"true nadir pointing error [deg]", f"True angle to the LVLH target (values below {floor:g} deg clipped)")
    axs[2].axhline(1.0, color="k", ls="--", lw=0.8, label="rated speed")
    axs[2].legend(fontsize=8, loc="upper right")
    label_axes(axs[2], "", "max wheel speed / rated [-]", "Reaction-wheel speed fraction (max over axes)")
    for i, key in enumerate(("pk", "ideal_g7", "real_design")):
        p = plots[(key, 0)]
        col, lab = CASE_STYLE[key]
        x = p["t"] / PERIOD
        axs[3].fill_between(x, i - 0.35, i + 0.35, where=p["mode"] == MODE_POINTING, step="post",
                            color=col, alpha=0.85, lw=0)
        axs[3].fill_between(x, i - 0.35, i + 0.35, where=p["mode"] != MODE_POINTING, step="post",
                            color=col, alpha=0.2, lw=0)
    axs[3].set_yticks(range(3))
    axs[3].set_yticklabels(["pk", "ideal_g7", "real_design"])
    axs[3].set_ylim(-0.6, 2.6)
    axs[3].plot([], [], color="0.4", alpha=0.85, lw=6, label="pointing (dark)")
    axs[3].plot([], [], color="0.4", alpha=0.2, lw=6, label="detumble (light)")
    axs[3].legend(fontsize=8, loc="upper right", ncol=2)
    label_axes(axs[3], "time [orbits]", "mode", "Mode timeline")
    save_fig(fig, FIG_DIR / "closed_loop_vs_pk.png")


def fig_estimation(p: dict):
    t = p["t"]
    x = t / PERIOD
    lit = p["lit"]
    ke = p["know_err"] / DEG
    sg = p["sigma"] / DEG
    keep = t >= (p["t_init"] + 600.0)
    ax_names = ("x", "y", "z")
    fig, axs = plt.subplots(6, 1, sharex=True, figsize=(11, 17))
    for a in axs:
        shade_eclipse(a, t, lit, label=(a is axs[0]))
    for i in range(3):
        a = axs[i]
        a.fill_between(x, -3 * sg[:, i], 3 * sg[:, i], color=f"C{i}", alpha=0.2, lw=0, label="±3σ (P_att)")
        a.plot(x, ke[:, i], color=f"C{i}", lw=0.7, label=f"δθ_{ax_names[i]} (true - estimate)")
        lim = 1.2 * np.nanmax(np.r_[np.abs(ke[keep, i]), 3 * sg[keep, i]])
        a.set_ylim(-lim, lim)
        a.legend(fontsize=8, loc="upper right", ncol=3)
        label_axes(a, "", f"knowledge error δθ_{ax_names[i]} [deg]",
                   "Realistic sensors + design filter, seed 0, tumble from 10 deg/s: attitude knowledge error, body axes "
                   "(y-limits exclude the first 600 s after initialisation)" if i == 0 else None)
    a = axs[3]
    bt = p["mag_bias_true"] * 1e9
    for i in range(3):
        a.plot(x, p["bm_est"][:, i] * 1e9, color=f"C{i}", lw=0.9, label=f"estimate {ax_names[i]}")
        a.axhline(bt[i], color=f"C{i}", ls="--", lw=1.0, label=f"truth {ax_names[i]}")
    a.legend(fontsize=8, loc="upper right", ncol=3)
    label_axes(a, "", "magnetometer bias [nT]", "Magnetometer bias, estimate vs truth (body axes)")
    a = axs[4]
    a.semilogy(x, np.maximum(p["gyro_bias_err"] / DEG, 1e-6), color="C4", lw=0.9)
    if np.isfinite(p["switch_time"]):
        for aa in axs:
            aa.axvline(p["switch_time"] / PERIOD, color="r", ls="-.", lw=0.9,
                       label="switch to pointing" if aa is axs[0] else None)
    label_axes(a, "", "‖gyro bias error‖ [deg/s]", "Gyro-bias estimation error norm")
    a = axs[5]
    a.plot(x, p["sun_used"], color="C1", lw=0.8)
    label_axes(a, "time [orbits]", "CSS diodes used per update [-]",
               "Number of CSS diodes used in the update (mean over each 10 s block; 0 in eclipse)")
    axs[0].legend(fontsize=8, loc="upper right", ncol=4)
    save_fig(fig, FIG_DIR / "closed_loop_estimation.png")


def _points_and_mean(ax, xpos, vals, color, width=0.28):
    v = np.array([np.nan if y is None else y for y in vals], float)
    n = v.size
    jit = np.linspace(-width / 2, width / 2, n) if n > 1 else np.zeros(1)
    ok = np.isfinite(v)
    ax.plot(xpos + jit[ok], v[ok], "o", color=color, ms=5, alpha=0.75)
    if ok.any():
        m = float(v[ok].mean())
        ax.hlines(m, xpos - 0.38, xpos + 0.38, color="k", lw=2.0)
    return v


def fig_cost(per_case: dict):
    cases = ("pk", "ideal_g7", "real_design", "real_g7")
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, key, title in ((axs[0], "nadir_rms_deg", "steady-state nadir error RMS"),
                           (axs[1], "nadir_max_deg", "steady-state nadir error max")):
        for i, c in enumerate(cases):
            _points_and_mean(ax, i, [per_case[c][s][key] for s in SEEDS5], CASE_STYLE[c][0])
        ax.set_yscale("log")
        ax.set_xticks(range(len(cases)))
        ax.set_xticklabels(["pk\n(perfect knowledge)", "ideal_g7\n(ideal sensors,\nG7 filter)",
                            "real_design\n(realistic sensors,\ndesign filter)", "real_g7\n(realistic sensors,\nG7 filter)"],
                           fontsize=8)
        ax.plot([], [], "o", color="gray", label="seeds 0-4")
        ax.plot([], [], color="k", lw=2, label="mean")
        if key == "nadir_max_deg":
            ax.axhline(5.0, color="k", ls="--", lw=0.8, label="5 deg requirement")
        ax.legend(fontsize=8, loc="upper left")
        label_axes(ax, "", f"{title} [deg]", title[0].upper() + title[1:])
    fig.suptitle("Cost of estimation: true nadir pointing error, tumble from 10 deg/s, 6 orbits, window = first switch + 1 orbit",
                 fontsize=11)
    save_fig(fig, FIG_DIR / "closed_loop_cost.png")


def fig_breakdown(per_case: dict):
    cases = ("real_design",) + BREAKDOWN
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
    for ax, key, title in ((axs[0], "know_rms_sunlit_deg", "sunlit"), (axs[1], "know_rms_eclipse_deg", "eclipse")):
        for i, c in enumerate(cases):
            _points_and_mean(ax, i, [per_case[c][s][key] for s in SEEDS3], "C3" if c == "real_design" else "C0")
        ax.set_yscale("log")
        ax.set_xticks(range(len(cases)))
        ax.set_xticklabels([BREAKDOWN_LABELS[c] for c in cases], fontsize=7, rotation=25)
        ax.plot([], [], "o", color="gray", label="seeds 0-2")
        ax.plot([], [], color="k", lw=2, label="mean")
        ax.legend(fontsize=8, loc="lower right")
        label_axes(ax, "", f"knowledge error RMS, {title} [deg]", f"Attitude knowledge error, {title}")
    fig.suptitle("Error-source breakdown (design filter, realistic sensors with one source removed), tumble from 10 deg/s, "
                 "6 orbits, steady-state window", fontsize=11)
    save_fig(fig, FIG_DIR / "closed_loop_breakdown.png")


def fig_bdot(bdot: dict):
    top = BDOT_ORBITS * 1.12
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for i, s in enumerate(BDOT_SCALES):
        c = bdot["cases"][f"bdot_x{s:g}"]
        for j, v in enumerate(c["detumble_time_orbits_per_seed"]):
            xj = s * (1 + 0.03 * (j - 2))
            if v is None:
                ax.plot(xj, top, "x", color="C3", ms=7, mew=1.8)
            else:
                ax.plot(xj, v, ".", color="C3", ms=7, alpha=0.7)
    xs = [s for s in BDOT_SCALES if bdot["cases"][f"bdot_x{s:g}"]["mean_orbits"] is not None]
    ax.plot(xs, [bdot["cases"][f"bdot_x{s:g}"]["mean_orbits"] for s in xs], "o-", color="C3", lw=1.3,
            label="with sensors (realistic + design filter), mean of seeds 0-4")
    xp = [s for s in BDOT_SCALES if bdot["cases"][f"bdot_x{s:g}"]["mean_orbits"] is None
          and bdot["cases"][f"bdot_x{s:g}"]["mean_orbits_detumbled_only"] is not None]
    ax.plot(xp, [bdot["cases"][f"bdot_x{s:g}"]["mean_orbits_detumbled_only"] for s in xp], "o", mfc="none",
            color="C3", ms=8, label="with sensors, mean of the detumbled seeds only (others marked x)")
    xg = [s for s in BDOT_SCALES if bdot["group6"][f"scale={s:g}"]["mean_orbits"] is not None]
    ax.plot(xg, [bdot["group6"][f"scale={s:g}"]["mean_orbits"] for s in xg], "s--", color="C0", lw=1.3,
            label="Group 6 perfect knowledge, mean of seeds 0-4 (only if all detumbled)")
    for s in BDOT_SCALES:
        g = bdot["group6"][f"scale={s:g}"]["detumble_time_orbits_per_seed"]
        for j, v in enumerate(g):
            if v is not None and v <= top:
                ax.plot(s * (1 + 0.03 * (j - 2)), v, ".", color="C0", ms=5, alpha=0.5)
    ax.plot([], [], "x", color="C3", mew=1.8, label=f"x: sensor run not detumbled within {BDOT_ORBITS:g} orbits")
    ax.axvline(1.0, color="k", ls=":", lw=0.8, label="nominal Avanzini-Giulietti gain")
    ax.axhline(BDOT_ORBITS, color="gray", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(BDOT_SCALES)
    ax.set_xticklabels([f"{s:g}" for s in BDOT_SCALES])
    ax.minorticks_off()
    ax.set_ylim(0, top * 1.05)
    ax.legend(fontsize=7.5, loc="upper right")
    label_axes(ax, "B-dot gain scale [× nominal k]", "time to |ω| < 0.5 deg/s [orbits]",
               "B-dot detumble from 10 deg/s with sensor errors vs perfect knowledge\n"
               f"(sensor runs {BDOT_ORBITS:g} orbits, 1 s resolution; Group 6 runs 6 orbits, 10 s resolution)")
    save_fig(fig, FIG_DIR / "closed_loop_bdot_gain.png")


# ---------------------------------------------------------------------------------------------
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=20, help="process pool size (default 20)")
    ap.add_argument("--cache", type=str, default=None,
                    help="pickle path: reuse the raw run results if it exists, else write it after the runs")
    args = ap.parse_args(argv)
    t_start = time.perf_counter()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cache = Path(args.cache) if args.cache else None
    jobs = [dict(case=name, seed=s) for name, c in CASES.items() for s in c["seeds"]]
    jobs.sort(key=lambda j: -CASES[j["case"]]["orbits"])         # long runs first
    n_proc = max(1, min(args.workers, len(jobs)))
    per_case: dict[str, dict] = {c: {} for c in CASES}
    plots: dict = {}
    walls: dict = {}
    if cache is not None and cache.exists():
        with open(cache, "rb") as fh:
            per_case, plots, walls = pickle.load(fh)
        print(f"loaded raw results from {cache}", flush=True)
    else:
        print(f"{len(jobs)} runs on {n_proc} processes ...", flush=True)
        with ProcessPoolExecutor(max_workers=n_proc) as pool:
            futs = [pool.submit(_worker, j) for j in jobs]
            for i, fu in enumerate(as_completed(futs), 1):
                name, seed, payload = fu.result()
                per_case[name][seed] = payload["summary"]
                walls[f"{name}|{seed}"] = payload["wall_s"]
                if "plot" in payload:
                    plots[(name, seed)] = payload["plot"]
                print(f"  {i}/{len(jobs)} done: {name} seed {seed} ({payload['wall_s']:.0f} s, "
                      f"total {time.perf_counter() - t_start:.0f} s)", flush=True)
        if cache is not None:
            with open(cache, "wb") as fh:
                pickle.dump((per_case, plots, walls), fh)

    # ---- figures -------------------------------------------------------------------------
    fig_vs_pk(plots)
    fig_estimation(plots[("real_design", 0)])
    fig_cost(per_case)
    fig_breakdown(per_case)

    # ---- B-dot gain with sensors vs Group 6 ----------------------------------------------
    g6 = json.loads(G6_METRICS_PATH.read_text())["detumble_sensitivity"]["gain_grid"]
    g6_cells = g6["cells"]
    bdot = {"scales": list(BDOT_SCALES), "seeds": list(SEEDS5), "orbits": BDOT_ORBITS,
            "bdot_k_nominal_N_m_s": BDOT_K_NOM, "cases": {}, "group6": {},
            "group6_source": "adcs/report/control_metrics.json detumble_sensitivity.gain_grid "
                             "(perfect knowledge, 6-orbit runs, 10 s log resolution)",
            "note": "Group 6 detumble times above 3 orbits cannot be reproduced by the 3-orbit sensor runs "
                    "(those appear as not detumbled)"}
    for s in BDOT_SCALES:
        ps = per_case[f"bdot_x{s:g}"]
        td = [ps[k]["detumble_time_orbits"] for k in SEEDS5]
        tdf = [x for x in td if np.isfinite(x)]
        bdot["cases"][f"bdot_x{s:g}"] = {
            "bdot_k_N_m_s": float(s * BDOT_K_NOM),
            "detumble_time_orbits_per_seed": [x if np.isfinite(x) else None for x in td],
            "mean_rate_last_orbit_degps_per_seed": [ps[k]["mean_rate_last_orbit_degps"] for k in SEEDS5],
            "mean_rate_last_orbit_degps_mean": float(np.mean([ps[k]["mean_rate_last_orbit_degps"] for k in SEEDS5])),
            "final_rate_degps_per_seed": [ps[k]["final_rate_degps"] for k in SEEDS5],
            "n_not_detumbled": int(len(td) - len(tdf)),
            "mean_orbits": float(np.mean(tdf)) if tdf and len(tdf) == len(td) else None,
            "mean_orbits_detumbled_only": float(np.mean(tdf)) if tdf else None,
            "min_orbits": float(np.min(tdf)) if tdf else None,
            "max_orbits": float(np.max(tdf)) if tdf else None}
        gc = g6_cells[f"scale={s:g}"]
        bdot["group6"][f"scale={s:g}"] = {k: gc[k] for k in (
            "detumble_time_orbits_per_seed", "n_not_detumbled", "mean_orbits", "mean_orbits_detumbled_only",
            "min_orbits", "max_orbits")}
        m_s, m_g = bdot["cases"][f"bdot_x{s:g}"]["mean_orbits"], gc["mean_orbits"]
        bdot["cases"][f"bdot_x{s:g}"]["mean_ratio_sensors_over_group6"] = (
            m_s / m_g if m_s is not None and m_g is not None else None)
    fig_bdot(bdot)

    # ---- metrics -------------------------------------------------------------------------
    agg = {c: aggregate_case(per_case[c]) for c in CASES}
    ideal, real_g7, real_d = agg["ideal_g7"], agg["real_g7"], agg["real_design"]
    g7_know = [per_case["real_g7"][s]["know_rms_deg"] for s in SEEDS5]
    checks = {
        "real_design_all_seeds_success": bool(all(per_case["real_design"][s]["success"] for s in SEEDS5)),
        "real_design_nadir_max_lt_5deg_all_seeds":
            bool(all(np.isfinite(per_case["real_design"][s]["nadir_max_deg"])
                     and per_case["real_design"][s]["nadir_max_deg"] < 5.0 for s in SEEDS5)),
        "ideal_g7_nees_mean_in_2p5_3p5": bool(ideal["nees_mean"]["mean"] is not None
                                              and 2.5 <= ideal["nees_mean"]["mean"] <= 3.5),
        "real_g7_know_rms_gt_5deg_all_seeds": bool(all(np.isfinite(v) and v > 5.0 for v in g7_know)),
    }
    cfg = {
        "orbital_period_s": PERIOD, "initial_rate_deg_s": RATE_DEG, "control_period_s": 1.0, "log_every": 1,
        "orbits": ORBITS, "bdot_orbits": BDOT_ORBITS, "seeds": {"main": list(SEEDS5), "breakdown": list(SEEDS3)},
        "initial_state": "tumble_state(10 deg/s, s) with s = sensor seed",
        "steady_state_window": "mode == pointing and t >= first switch + 1 orbit (same definition for pk)",
        "plot_decimation_s": DECIM,
        "presets": {"IDEAL_SENSORS": asdict(IDEAL_SENSORS), "REALISTIC_SENSORS": asdict(REALISTIC_SENSORS),
                    "MEKF_G7": asdict(MEKF_G7), "MEKF_DESIGN": asdict(MEKF_DESIGN)},
        "cases": {name: ({"kind": c["kind"], "seeds": list(c["seeds"]), "orbits": c["orbits"]}
                         | ({"sensors": asdict(c["sensors"]), "mekf": asdict(c["mekf"])} if "sensors" in c else {})
                         | ({"bdot_scale": c["scale"], "modes": asdict(DETUMBLE_ONLY)} if "scale" in c else {}))
                  for name, c in CASES.items()},
        "breakdown_description": BREAKDOWN_DESC,
    }
    metrics = {
        "config": cfg,
        "cases": {c: {"per_seed": {str(s): per_case[c][s] for s in sorted(per_case[c])}, "aggregate": agg[c]}
                  for c in CASES if not c.startswith("bdot_x")},
        "bdot_gain": bdot,
        "bdot_per_seed_summaries": {c: {str(s): per_case[c][s] for s in sorted(per_case[c])}
                                    for c in CASES if c.startswith("bdot_x")},
        "checks": checks,
        "runtime_s": {"total": 0.0, "sum_of_individual_runs": float(sum(walls.values())), "n_runs": len(jobs),
                      "n_processes": n_proc, "per_run": walls},
    }
    metrics["runtime_s"]["total"] = time.perf_counter() - t_start
    METRICS_PATH.write_text(json.dumps(_clean(metrics), indent=2, allow_nan=False))
    print(f"wrote {METRICS_PATH} (total {metrics['runtime_s']['total']:.1f} s)")
    print("checks:", checks)


if __name__ == "__main__":
    main()
