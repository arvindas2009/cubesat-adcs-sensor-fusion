"""Group 6 control figures and metrics (spec docs/specs/adcs_control.md section 6).

Run:  .venv/Scripts/python -m adcs.report.control_figures

Outputs
-------
``adcs/report/figures/{detumble,detumble_sensitivity,pointing_15orbits,auto_detumble_to_nadir}.png``
and ``adcs/report/control_metrics.json``. All runs are perfect-knowledge simulations
(``adcs.simulate.run``); every number reported comes from the JSON written here. Nothing is tuned;
anomalies are flagged in ``checks`` and reported, never corrected.

Runs are distributed over a ``multiprocessing.Pool``. All logged runs use ``log_every = 10`` control
periods (10 s), so detumble times have a 10 s resolution.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
import numpy as np  # noqa: E402

from adcs.actuators import MagnetorquerConfig, WheelConfig, RPM  # noqa: E402
from adcs.config import DEFAULT  # noqa: E402
from adcs.control import PointingGains, bdot_gain  # noqa: E402
from adcs.orbit import initial_state, orbital_period  # noqa: E402
from adcs.simulate import (ControlConfig, MODE_POINTING, detumble_time, lvlh_state, run,  # noqa: E402
                           tumble_state)
from common.plotting import label_axes, save_fig  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parent
FIG_DIR = REPORT_DIR / "figures"
METRICS_PATH = REPORT_DIR / "control_metrics.json"

DEG = np.pi / 180.0
LOG_EVERY = 10
THRESH = float(np.deg2rad(0.5))
PERIOD = float(orbital_period(DEFAULT.orbit.a))
NOT_DETUMBLED = "not detumbled within 6 orbits"

DET_SEEDS = range(5)
SENS_RATES = (2.0, 5.0, 10.0, 20.0, 30.0)
SENS_MMAX = (0.05, 0.1, 0.2, 0.4)
SENS_SEEDS = range(3)
GAIN_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
GAIN_SEEDS = range(5)
SENS_ORBITS = 6.0
WHEEL_AX = ("x", "y", "z")


# ---------------------------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------------------------
def _worker(job: dict):
    """Run one simulation; returns ``(key, payload)``. Top-level so ``Pool`` can pickle it."""
    kind = job["kind"]
    t0 = time.perf_counter()
    if kind in ("detumble", "sens_mmax", "sens_gain"):
        y0 = tumble_state(np.deg2rad(job["rate_deg"]), job["seed"])
        mtq = MagnetorquerConfig(m_max=job.get("m_max", 0.2))
        ctrl = ControlConfig()
        if kind == "sens_gain":
            k = job["scale"] * bdot_gain(DEFAULT.spacecraft.inertia, DEFAULT.orbit.a, DEFAULT.orbit.inc)
            ctrl = ControlConfig(bdot_k=float(k))
        res = run(y0, job["orbits"] * PERIOD, "detumble", ctrl=ctrl, mtq=mtq, log_every=LOG_EVERY)
        td = detumble_time(res.t, res.rate, THRESH)
        payload = {"td_s": td, "final_rate_rad_s": float(res.rate[-1]), "bdot_k": res.meta["bdot_k"],
                   "res": res if kind == "detumble" else None}
    elif kind == "pointing":
        r0, v0 = initial_state(DEFAULT)
        y0 = lvlh_state(r0, v0, offset=np.deg2rad([20.0, -15.0, 30.0]), w_rel=np.deg2rad([0.1, -0.1, 0.2]))
        res = run(y0, job["orbits"] * PERIOD, "pointing", ctrl=ControlConfig(dumping=job["dumping"]),
                  log_every=LOG_EVERY)
        payload = {"res": res}
    elif kind == "auto":
        y0 = tumble_state(np.deg2rad(10.0), 0)
        res = run(y0, job["orbits"] * PERIOD, "auto", log_every=LOG_EVERY)
        payload = {"res": res}
    else:
        raise ValueError(kind)
    payload["wall_s"] = time.perf_counter() - t0
    return job["key"], payload


def _jobs() -> list[dict]:
    jobs = []
    # long runs first so the pool stays busy
    for dumping in (True, False):
        jobs.append(dict(kind="pointing", key=("pointing", dumping), orbits=15.0, dumping=dumping))
    jobs.append(dict(kind="auto", key=("auto",), orbits=4.0))
    for rate in SENS_RATES:
        for m_max in SENS_MMAX:
            for seed in SENS_SEEDS:
                jobs.append(dict(kind="sens_mmax", key=("sens_mmax", rate, m_max, seed), rate_deg=rate,
                                 m_max=m_max, seed=seed, orbits=SENS_ORBITS))
    for scale in GAIN_SCALES:
        for seed in GAIN_SEEDS:
            jobs.append(dict(kind="sens_gain", key=("sens_gain", scale, seed), rate_deg=10.0, m_max=0.2,
                             scale=scale, seed=seed, orbits=SENS_ORBITS))
    for seed in DET_SEEDS:
        jobs.append(dict(kind="detumble", key=("detumble", seed), rate_deg=10.0, seed=seed, orbits=3.0))
    return jobs


def _nan_to_none(x):
    x = float(x)
    return None if not np.isfinite(x) else x


# ---------------------------------------------------------------------------------------------
# 1. detumble
# ---------------------------------------------------------------------------------------------
def detumble_fig_and_metrics(results: dict) -> dict:
    fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(9, 7), gridspec_kw={"height_ratios": [2, 1]})
    per_seed = {}
    for seed in DET_SEEDS:
        p = results[("detumble", seed)]
        res = p["res"]
        a1.semilogy(res.t / PERIOD, np.maximum(res.rate, 1e-9) / DEG, lw=1.1, label=f"seed {seed}")
        td = p["td_s"]
        per_seed[str(seed)] = {
            "detumble_time_s": _nan_to_none(td),
            "detumble_time_orbits": _nan_to_none(td / PERIOD),
            "final_rate_deg_s": float(res.rate[-1] / DEG),
            "final_kinetic_energy_ratio": None,
        }
        if not np.isfinite(td):
            per_seed[str(seed)]["flag"] = "not detumbled within 3 orbits"
    a1.axhline(0.5, color="k", ls="--", lw=0.9, label="0.5 deg/s threshold")
    for k in (1, 2):
        a1.axvline(k, color="gray", ls=":", lw=0.8)
    a1.set_ylim(1e-2, 20)
    a1.legend(fontsize=8, ncol=3, loc="lower left")
    label_axes(a1, "", "angular rate |ω| [deg/s]",
               "B-dot detumbling from 10 deg/s, perfect knowledge, m_max = 0.2 A·m², 3 orbits (10 s resolution)")
    r0 = results[("detumble", 0)]["res"]
    a2.plot(r0.t / PERIOD, np.linalg.norm(r0.m, axis=1), "C0", lw=0.9, label="|m| (seed 0)")
    a2.axhline(0.2, color="r", ls=":", lw=0.9, label="axis limit 0.2 A·m² (per axis)")
    a2.legend(fontsize=8, loc="upper right")
    label_axes(a2, "time [orbits]", "dipole magnitude |m| [A·m²]",
               "Applied magnetorquer dipole, seed 0 (sampled every 10 s)")
    save_fig(fig, FIG_DIR / "detumble.png")

    tds = [v["detumble_time_orbits"] for v in per_seed.values() if v["detumble_time_orbits"] is not None]
    return {
        "initial_rate_deg_s": 10.0, "m_max_A_m2": 0.2, "n_orbits": 3, "threshold_deg_s": 0.5,
        "log_resolution_s": LOG_EVERY * 1.0,
        "bdot_k_N_m_s": results[("detumble", 0)]["bdot_k"],
        "per_seed": per_seed,
        "max_detumble_time_orbits": max(tds) if tds else None,
        "mean_detumble_time_orbits": float(np.mean(tds)) if tds else None,
        "all_seeds_detumbled": len(tds) == len(per_seed),
    }


# ---------------------------------------------------------------------------------------------
# 2. sensitivity
# ---------------------------------------------------------------------------------------------
def _cell(tds: list[float]) -> dict:
    ok = [x / PERIOD for x in tds if np.isfinite(x)]
    n_fail = len(tds) - len(ok)
    cell = {
        "detumble_time_orbits_per_seed": [_nan_to_none(x / PERIOD) for x in tds],
        "detumble_time_s_per_seed": [_nan_to_none(x) for x in tds],
        "n_not_detumbled": n_fail,
        "mean_orbits": float(np.mean(ok)) if ok and n_fail == 0 else None,
        "mean_orbits_detumbled_only": float(np.mean(ok)) if ok else None,
        "min_orbits": float(np.min(ok)) if ok else None,
        "max_orbits": float(np.max(ok)) if ok else None,
    }
    if n_fail:
        cell["flag"] = NOT_DETUMBLED
    return cell


def sensitivity_fig_and_metrics(results: dict) -> dict:
    grid = {}
    for m_max in SENS_MMAX:
        for rate in SENS_RATES:
            tds = [results[("sens_mmax", rate, m_max, s)]["td_s"] for s in SENS_SEEDS]
            grid[(m_max, rate)] = _cell(tds)
    gain = {}
    k_nom = None
    for scale in GAIN_SCALES:
        tds = [results[("sens_gain", scale, s)]["td_s"] for s in GAIN_SEEDS]
        gain[scale] = _cell(tds)
        k_nom = results[("sens_gain", scale, 0)]["bdot_k"] / scale
    top = SENS_ORBITS * 1.08

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
    n = len(SENS_MMAX)
    for i, m_max in enumerate(SENS_MMAX):
        color = f"C{i}"
        jit = 1.0 + 0.04 * (i - (n - 1) / 2)
        xs, ym, lo, hi = [], [], [], []
        for rate in SENS_RATES:
            c = grid[(m_max, rate)]
            x = rate * jit
            # individual successful seeds (small dots), failures at the top
            for v in c["detumble_time_orbits_per_seed"]:
                if v is None:
                    a1.plot(x, top, marker="x", color=color, ms=7, mew=1.8, ls="none")
                else:
                    a1.plot(x, v, marker=".", color=color, ms=5, alpha=0.6, ls="none")
            if c["mean_orbits"] is not None:
                xs.append(x)
                ym.append(c["mean_orbits"])
                lo.append(c["mean_orbits"] - c["min_orbits"])
                hi.append(c["max_orbits"] - c["mean_orbits"])
            else:
                xs.append(x)
                ym.append(np.nan)
                lo.append(0.0)
                hi.append(0.0)
        a1.errorbar(xs, ym, yerr=[lo, hi], color=color, marker="o", ms=5, capsize=3, lw=1.2,
                    label=f"m_max = {m_max:g} A·m²")
    a1.plot([], [], marker="x", color="k", ls="none", mew=1.8, label=f"x: {NOT_DETUMBLED}")
    a1.axhspan(1, 3, color="green", alpha=0.10, label="1-3 orbit band")
    a1.set_xscale("log")
    a1.set_xticks(SENS_RATES)
    a1.set_xticklabels([f"{r:g}" for r in SENS_RATES])
    a1.xaxis.set_minor_formatter(NullFormatter())
    a1.set_ylim(0, SENS_ORBITS * 1.15)
    a1.axhline(SENS_ORBITS, color="gray", ls=":", lw=0.8)
    a1.legend(fontsize=8, loc="upper left")
    label_axes(a1, "initial rate [deg/s]", "time to |ω| < 0.5 deg/s [orbits]",
               "Detumble time vs initial rate and m_max (seeds 0-2: mean, min-max; dots = seeds)")

    xs, ym, lo, hi = [], [], [], []
    for scale in GAIN_SCALES:
        c = gain[scale]
        for v in c["detumble_time_orbits_per_seed"]:
            if v is None:
                a2.plot(scale, top, marker="x", color="C0", ms=7, mew=1.8, ls="none")
            else:
                a2.plot(scale, v, marker=".", color="C0", ms=6, alpha=0.6, ls="none")
        xs.append(scale)
        if c["mean_orbits"] is not None:
            ym.append(c["mean_orbits"])
            lo.append(c["mean_orbits"] - c["min_orbits"])
            hi.append(c["max_orbits"] - c["mean_orbits"])
        else:
            ym.append(np.nan)
            lo.append(0.0)
            hi.append(0.0)
    a2.errorbar(xs, ym, yerr=[lo, hi], color="C0", marker="o", ms=5, capsize=3, lw=1.2,
                label="10 deg/s, m_max = 0.2 A·m² (seeds 0-4: mean, min-max)")
    a2.axhspan(1, 3, color="green", alpha=0.10, label="1-3 orbit band")
    a2.axvline(1.0, color="k", ls="--", lw=0.8, label="nominal Avanzini-Giulietti gain")
    a2.plot([], [], marker="x", color="k", ls="none", mew=1.8, label=f"x: {NOT_DETUMBLED}")
    a2.set_xscale("log")
    a2.set_xticks(GAIN_SCALES)
    a2.set_xticklabels([f"{s:g}" for s in GAIN_SCALES])
    a2.xaxis.set_minor_formatter(NullFormatter())
    a2.set_ylim(0, SENS_ORBITS * 1.15)
    a2.axhline(SENS_ORBITS, color="gray", ls=":", lw=0.8)
    a2.legend(fontsize=8, loc="upper right")
    label_axes(a2, "B-dot gain scale [× nominal k]", "time to |ω| < 0.5 deg/s [orbits]",
               "Detumble time vs B-dot gain scale")
    fig.suptitle("B-dot detumble sensitivity, perfect knowledge (runs up to 6 orbits, 10 s resolution)",
                 fontsize=11)
    save_fig(fig, FIG_DIR / "detumble_sensitivity.png")

    n_runs = len(SENS_RATES) * len(SENS_MMAX) * len(SENS_SEEDS) + len(GAIN_SCALES) * len(GAIN_SEEDS)
    all_cells = list(grid.values()) + list(gain.values())
    return {
        "max_orbits_simulated": SENS_ORBITS,
        "threshold_deg_s": 0.5,
        "log_resolution_s": LOG_EVERY * 1.0,
        "not_detumbled_representation": f"null + flag '{NOT_DETUMBLED}'",
        "mmax_grid": {
            "rates_deg_s": list(SENS_RATES), "m_max_A_m2": list(SENS_MMAX), "seeds": list(SENS_SEEDS),
            "bdot_k_N_m_s": k_nom,
            "cells": {f"m_max={m:g}|rate={r:g}": grid[(m, r)] for m in SENS_MMAX for r in SENS_RATES},
        },
        "gain_grid": {
            "rate_deg_s": 10.0, "m_max_A_m2": 0.2, "scales": list(GAIN_SCALES), "seeds": list(GAIN_SEEDS),
            "bdot_k_nominal_N_m_s": k_nom,
            "bdot_k_per_scale_N_m_s": {f"{s:g}": s * k_nom for s in GAIN_SCALES},
            "cells": {f"scale={s:g}": gain[s] for s in GAIN_SCALES},
        },
        "n_runs": n_runs,
        "n_cells": len(all_cells),
        "n_cells_with_failures": sum(1 for c in all_cells if c["n_not_detumbled"] > 0),
        "n_runs_not_detumbled": sum(c["n_not_detumbled"] for c in all_cells),
    }


# ---------------------------------------------------------------------------------------------
# 3. pointing 15 orbits
# ---------------------------------------------------------------------------------------------
def _shade_intervals(ax, t, flag, color="orange", alpha=0.25, label=None):
    """Shade ``t[flag]`` intervals (samples spaced by the log step)."""
    dt = t[1] - t[0]
    idx = np.flatnonzero(flag)
    if idx.size == 0:
        return
    starts = [idx[0]]
    ends = []
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            ends.append(a)
            starts.append(b)
    ends.append(idx[-1])
    for i, (s, e) in enumerate(zip(starts, ends)):
        ax.axvspan(t[s] / PERIOD, (t[e] + dt) / PERIOD, color=color, alpha=alpha, lw=0,
                   label=label if i == 0 else None)


def _pointing_metrics(res) -> dict:
    ms = res.meta["wheel"]["max_speed"]
    err = res.nadir_err / DEG
    after = res.t >= PERIOD
    frac = np.abs(res.wheel_speed) / ms                       # (N, 3)
    fmax = frac.max(axis=1)
    sat = np.flatnonzero(fmax >= 0.99)
    n_orb = int(round(res.t[-1] / PERIOD))
    per_orbit = []
    for i in range(n_orb):
        sel = (res.t >= i * PERIOD) & (res.t < (i + 1) * PERIOD)
        per_orbit.append(float(err[sel].max()))
    return {
        "max_nadir_error_deg_after_first_orbit": float(err[after].max()),
        "rms_nadir_error_deg_after_first_orbit": float(np.sqrt(np.mean(err[after] ** 2))),
        "max_nadir_error_deg_overall": float(err.max()),
        "final_nadir_error_deg": float(err[-1]),
        "per_orbit_max_nadir_error_deg": per_orbit,
        "max_wheel_speed_fraction_per_axis": {ax: float(frac[:, i].max()) for i, ax in enumerate(WHEEL_AX)},
        "max_wheel_speed_fraction": float(fmax.max()),
        "max_wheel_speed_rpm": float(np.abs(res.wheel_speed).max() / RPM),
        "first_saturation_time_s": float(res.t[sat[0]]) if sat.size else None,
        "first_saturation_time_orbits": float(res.t[sat[0]] / PERIOD) if sat.size else None,
        "fraction_of_time_dumping": float(np.mean(res.dump)),
        "max_abs_dipole_component_A_m2": float(np.abs(res.m).max()),
    }


def pointing_fig_and_metrics(results: dict) -> dict:
    cases = {"dumping on": results[("pointing", True)]["res"], "dumping off": results[("pointing", False)]["res"]}
    ms_rpm = WheelConfig().rated_speed_rpm
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
    floor = 1e-4
    for (name, res), c in zip(cases.items(), ("C0", "C3")):
        axs[0].semilogy(res.t / PERIOD, np.maximum(res.nadir_err / DEG, floor), color=c, lw=1.0, label=name)
    _shade_intervals(axs[0], cases["dumping on"].t, cases["dumping on"].dump, label="dumping active (on case)")
    axs[0].axhline(1.0, color="gray", ls=":", lw=0.8)
    axs[0].legend(fontsize=8, loc="upper right")
    label_axes(axs[0], "", "nadir pointing error [deg]",
               "Nadir pointing, 15 orbits, initial offset [20, -15, 30] deg, perfect knowledge "
               f"(values below {floor:g} deg clipped)")
    for ax, (name, res) in zip(axs[1:], cases.items()):
        rpm = res.wheel_speed / RPM
        for i, a_ in enumerate(WHEEL_AX):
            ax.plot(res.t / PERIOD, rpm[:, i], lw=0.9, label=f"wheel {a_}")
        ax.axhline(ms_rpm, color="k", ls="--", lw=0.9, label="±8000 rpm limit")
        ax.axhline(-ms_rpm, color="k", ls="--", lw=0.9)
        if name == "dumping on":
            _shade_intervals(ax, res.t, res.dump, label="dumping active")
        ax.legend(fontsize=8, ncol=5, loc="upper right")
        label_axes(ax, "", "wheel speed rel. to body [rpm]", f"Wheel speeds, {name}")
    axs[-1].set_xlabel("time [orbits]")
    save_fig(fig, FIG_DIR / "pointing_15orbits.png")

    return {
        "n_orbits": 15,
        "initial_offset_deg": [20.0, -15.0, 30.0],
        "initial_w_rel_deg_s": [0.1, -0.1, 0.2],
        "log_resolution_s": LOG_EVERY * 1.0,
        "gains": {"wn_rad_s": PointingGains().wn, "zeta": PointingGains().zeta,
                  "rate_limit_deg_s": PointingGains().rate_limit / DEG},
        "dump_k_per_s": ControlConfig().dump_k, "dump_on_fraction": ControlConfig().dump_on,
        "dump_off_fraction": ControlConfig().dump_off,
        "cases": {name.replace(" ", "_"): _pointing_metrics(res) for name, res in cases.items()},
    }


# ---------------------------------------------------------------------------------------------
# 4. auto
# ---------------------------------------------------------------------------------------------
def auto_fig_and_metrics(results: dict) -> dict:
    res = results[("auto",)]["res"]
    t_sw = res.meta["switch_time"]
    ms = res.meta["wheel"]["max_speed"]
    x = res.t / PERIOD
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 12))
    axs[0].semilogy(x, np.maximum(res.rate / DEG, 1e-6), "C0", lw=1.0)
    axs[0].axhline(0.5, color="k", ls="--", lw=0.8, label="0.5 deg/s threshold")
    label_axes(axs[0], "", "angular rate |ω| [deg/s]",
               "Auto mode: B-dot detumble from 10 deg/s (seed 0) then nadir pointing, 4 orbits")
    floor = 1e-4
    axs[1].semilogy(x, np.maximum(res.nadir_err / DEG, floor), "C1", lw=1.0)
    label_axes(axs[1], "", "nadir error [deg]", f"Angle to LVLH target (values below {floor:g} deg clipped)")
    rpm = res.wheel_speed / RPM
    for i, a_ in enumerate(WHEEL_AX):
        axs[2].plot(x, rpm[:, i], lw=0.9, label=f"wheel {a_}")
    axs[2].axhline(WheelConfig().rated_speed_rpm, color="k", ls="--", lw=0.8, label="±8000 rpm limit")
    axs[2].axhline(-WheelConfig().rated_speed_rpm, color="k", ls="--", lw=0.8)
    label_axes(axs[2], "", "wheel speed rel. to body [rpm]", "Wheel speeds")
    axs[3].plot(x, np.linalg.norm(res.m, axis=1), "C2", lw=0.9)
    label_axes(axs[3], "time [orbits]", "dipole magnitude |m| [A·m²]", "Applied magnetorquer dipole")
    for ax in axs:
        if np.isfinite(t_sw):
            ax.axvline(t_sw / PERIOD, color="r", ls="-.", lw=1.0,
                       label="switch to pointing" if ax is axs[0] else None)
    axs[0].legend(fontsize=8, loc="upper right")
    axs[2].legend(fontsize=8, ncol=4, loc="upper right")
    save_fig(fig, FIG_DIR / "auto_detumble_to_nadir.png")

    pt = res.mode == MODE_POINTING
    late = pt & (res.t >= t_sw + PERIOD)
    err = res.nadir_err[late] / DEG
    frac = np.abs(res.wheel_speed) / ms
    return {
        "initial_rate_deg_s": 10.0, "seed": 0, "n_orbits": 4,
        "log_resolution_s": LOG_EVERY * 1.0,
        "switch_time_s": _nan_to_none(t_sw),
        "switch_time_orbits": _nan_to_none(t_sw / PERIOD),
        "detumble_time_s": _nan_to_none(detumble_time(res.t, res.rate, THRESH)),
        "max_nadir_error_deg_from_switch_plus_1_orbit": float(err.max()) if err.size else None,
        "rms_nadir_error_deg_from_switch_plus_1_orbit": float(np.sqrt(np.mean(err ** 2))) if err.size else None,
        "final_nadir_error_deg": float(res.nadir_err[-1] / DEG),
        "max_wheel_speed_fraction_whole_run": float(frac.max()),
        "max_wheel_speed_fraction_pointing_phase": float(frac[pt].max()) if pt.any() else None,
        "max_wheel_speed_fraction_per_axis_pointing_phase":
            {ax: float(frac[pt][:, i].max()) for i, ax in enumerate(WHEEL_AX)} if pt.any() else None,
        "max_wheel_speed_rpm": float(np.abs(res.wheel_speed).max() / RPM),
        "fraction_of_pointing_time_dumping": float(np.mean(res.dump[pt])) if pt.any() else None,
        "max_abs_dipole_component_A_m2": float(np.abs(res.m).max()),
        "note_pointing_error_window": "samples with t >= switch time + 1 orbit",
    }


# ---------------------------------------------------------------------------------------------
def main() -> None:
    t_start = time.perf_counter()
    jobs = _jobs()
    n_proc = min(len(jobs), max(1, (os.cpu_count() or 2) - 2))
    print(f"{len(jobs)} runs on {n_proc} processes ...", flush=True)
    results: dict = {}
    walls: dict = {}
    with Pool(n_proc) as pool:
        for i, (key, payload) in enumerate(pool.imap_unordered(_worker, jobs, chunksize=1), 1):
            results[key] = payload
            walls[key] = payload["wall_s"]
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done ({time.perf_counter() - t_start:.0f} s)", flush=True)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"config": {
        "orbital_period_s": PERIOD,
        "altitude_m": DEFAULT.orbit.altitude,
        "inclination_deg": DEFAULT.orbit.inc / DEG,
        "inertia_diag_kg_m2": np.diag(DEFAULT.spacecraft.inertia).tolist(),
        "bdot_k_nominal_N_m_s": float(bdot_gain(DEFAULT.spacecraft.inertia, DEFAULT.orbit.a, DEFAULT.orbit.inc)),
        "magnetorquer": {"m_max_A_m2": MagnetorquerConfig().m_max,
                         "control_period_s": MagnetorquerConfig().control_period,
                         "duty": MagnetorquerConfig().duty, "on_time_s": MagnetorquerConfig().on_time},
        "wheel": {"max_torque_N_m": WheelConfig().max_torque, "rated_momentum_N_m_s": WheelConfig().rated_momentum,
                  "rated_speed_rpm": WheelConfig().rated_speed_rpm,
                  "max_speed_rad_s": WheelConfig().max_speed, "spin_inertia_kg_m2": WheelConfig().spin_inertia},
        "pointing_gains": {"wn_rad_s": PointingGains().wn, "zeta": PointingGains().zeta,
                           "rate_limit_rad_s": PointingGains().rate_limit},
        "control_config_defaults": {"dump_k_per_s": ControlConfig().dump_k, "dump_on": ControlConfig().dump_on,
                                    "dump_off": ControlConfig().dump_off,
                                    "detumble_threshold_rad_s": ControlConfig().detumble_threshold,
                                    "detumble_hold_s": ControlConfig().detumble_hold,
                                    "disturbances": ControlConfig().disturbances},
        "log_every_control_periods": LOG_EVERY,
        "time_resolution_of_detumble_times_s": LOG_EVERY * MagnetorquerConfig().control_period,
    }}
    metrics["detumble"] = detumble_fig_and_metrics(results)
    metrics["detumble_sensitivity"] = sensitivity_fig_and_metrics(results)
    metrics["pointing_15orbits"] = pointing_fig_and_metrics(results)
    metrics["auto_detumble_to_nadir"] = auto_fig_and_metrics(results)

    # kinetic-energy ratio for the detumble seeds (needs inertia)
    inertia = DEFAULT.spacecraft.inertia
    for seed in DET_SEEDS:
        res = results[("detumble", seed)]["res"]
        ke = 0.5 * np.einsum("ni,ij,nj->n", res.w, inertia, res.w)
        metrics["detumble"]["per_seed"][str(seed)]["final_kinetic_energy_ratio"] = float(ke[-1] / ke[0])

    # anomaly checks against the expectations named in the task / M7 exit criterion
    pm = metrics["pointing_15orbits"]["cases"]["dumping_on"]
    d = metrics["detumble"]
    checks = {
        "detumble_10deg_all_seeds_within_3_orbits": bool(d["all_seeds_detumbled"]),
        "wheel_speed_never_above_8000_rpm_x1.00001_pointing_cases":
            bool(all(c["max_wheel_speed_fraction"] <= 1.0 + 1e-5
                     for c in metrics["pointing_15orbits"]["cases"].values())),
        "wheel_speed_never_above_8000_rpm_auto":
            bool(metrics["auto_detumble_to_nadir"]["max_wheel_speed_fraction_whole_run"] <= 1.0 + 1e-5),
        "nadir_error_below_1deg_after_first_orbit_dumping_on": bool(pm["max_nadir_error_deg_after_first_orbit"] < 1.0),
    }
    metrics["checks"] = checks
    metrics["runtime_s"] = {"total": time.perf_counter() - t_start,
                            "sum_of_individual_runs": float(sum(walls.values())),
                            "n_runs": len(jobs), "n_processes": n_proc}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"wrote {METRICS_PATH} (total {metrics['runtime_s']['total']:.1f} s)")


if __name__ == "__main__":
    main()
