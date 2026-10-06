"""Group 7 estimation figures and metrics (spec docs/specs/adcs_estimation.md section 6).

Run:  .venv/Scripts/python -m adcs.report.estimation_figures

Outputs
-------
``adcs/report/figures/mekf_{pointing_attitude,pointing_bias,nis,eclipse,detumble,sensitivity,
consistency_mc}.png`` and ``adcs/report/estimation_metrics.json``. Truth comes from
``adcs.simulate.run`` (perfect-knowledge control, 1 s log); sensors from
``adcs.sensors.measure_series``; estimates from ``adcs.estimation.run_mekf``. Every number
reported comes from the JSON written here. Nothing is tuned; anomalies are flagged in
``checks`` and reported, never corrected.

Statistics windows (recorded in the JSON): "after 0.25 orbit" / "after 1 orbit" are
``t >= 0.25 P`` / ``t >= P``; NEES, NIS and the 3-sigma fractions use samples with
``k >= k_init + 300`` (the window of the test suite, so the TRIAD-start transient is excluded).
Sunlit / eclipse classification uses the true shadow flag at the sample.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from adcs.config import DEFAULT  # noqa: E402
from adcs.environment import FieldTable  # noqa: E402
from adcs.estimation import (MekfConfig, attitude_errors, nees, run_mekf, triad,  # noqa: E402
                             triad_covariance)
from adcs.orbit import Ephemeris, initial_state, orbital_period  # noqa: E402
from adcs.sensors import (GyroConfig, MagnetometerConfig, SunSensorConfig,  # noqa: E402
                          measure_series)
from adcs.simulate import lvlh_state, run, tumble_state  # noqa: E402
from common.ekf import nis_bounds, nis_fraction_inside  # noqa: E402
from common.plotting import label_axes, save_fig  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parent
FIG_DIR = REPORT_DIR / "figures"
METRICS_PATH = REPORT_DIR / "estimation_metrics.json"

DEG = np.pi / 180.0
PERIOD = float(orbital_period(DEFAULT.orbit.a))
SETTLE = 300                      # samples skipped after k_init for NEES / NIS / 3-sigma statistics
MA = 100                          # moving-mean length for the NIS / NEES plots
BIAS_FRACS = (0.0, 0.1, 0.25, 0.5, 1.0)
POINTING_OFFSET_DEG = (2.0, -1.0, 3.0)
DETUMBLE_RATE_DEG = 3.0

SENS_SEEDS = tuple(range(5))
SENS_MAG_NT = (15.0, 100.0, 300.0, 1000.0)
SENS_CSS_DEG = (2.5, 5.0, 10.0)
SENS_ORBITS = 3.0
MC_SEEDS = tuple(range(20))
MC_ORBITS = 3.0

_TRUTH: dict = {}


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


def _init_pool(truth):
    _TRUTH.update(truth)


def intervals(flag) -> list[tuple[int, int]]:
    """Inclusive index runs ``(i0, i1)`` where ``flag`` is True."""
    flag = np.asarray(flag, bool)
    idx = np.flatnonzero(flag)
    if idx.size == 0:
        return []
    brk = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[brk + 1]]
    ends = np.r_[idx[brk], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def shade_eclipse(ax, t, lit, label=True):
    for j, (i0, i1) in enumerate(intervals(~lit)):
        ax.axvspan(t[i0] / PERIOD, (t[i1] + 1.0) / PERIOD, color="0.75", alpha=0.5, lw=0,
                   label="eclipse" if (label and j == 0) else None)


def movmean(x, n=MA):
    c = np.cumsum(np.insert(np.asarray(x, float), 0, 0.0))
    return (c[n:] - c[:-n]) / n


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else float("nan")


# ---------------------------------------------------------------------------------------------
# truth
# ---------------------------------------------------------------------------------------------
def _truth_worker(job: dict):
    t0 = time.perf_counter()
    r0, v0 = initial_state(DEFAULT)
    t_end = job["orbits"] * PERIOD
    if job["name"] == "pointing":
        y0 = lvlh_state(r0, v0, offset=np.deg2rad(POINTING_OFFSET_DEG))
        res = run(y0, t_end, "pointing", log_every=1)
    else:
        y0 = tumble_state(np.deg2rad(DETUMBLE_RATE_DEG), 0)
        res = run(y0, t_end, "detumble", log_every=1)
    t = res.t
    eph = Ephemeris(r0, v0, t[-1] + 10.0)
    ft = FieldTable(eph, DEFAULT, t[-1] + 10.0)
    out = {"t": t, "q": res.q, "w": res.w, "b_eci": ft.b_eci(t), "s_eci": ft.sun_dir_eci(t),
           "lit": ft.sunlit(t), "wall_s": time.perf_counter() - t0}
    return job["name"], out


# ---------------------------------------------------------------------------------------------
# one estimation run and its statistics
# ---------------------------------------------------------------------------------------------
def _slice(truth: dict, n_orbits: float | None) -> dict:
    if n_orbits is None:
        return truth
    n = int(np.floor(n_orbits * PERIOD + 1e-9)) + 1
    return {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in truth.items()}


def estimate(truth: dict, seed: int, mag_sigma=None, css_diode=None):
    """Sensors + MEKF on a truth record; sensor and filter sigmas are matched."""
    mag = MagnetometerConfig() if mag_sigma is None else MagnetometerConfig(sigma=mag_sigma)
    css = SunSensorConfig() if css_diode is None else SunSensorConfig(sigma_diode=css_diode)
    cfg = MekfConfig(sigma_mag=mag.sigma, sigma_sun=css.sigma_vector)
    data = measure_series(truth["t"], truth["q"], truth["b_eci"], truth["s_eci"], truth["lit"],
                          seed=seed, mag=mag, css=css, gyro=GyroConfig())
    res = run_mekf(data, truth["b_eci"], truth["s_eci"], cfg)
    return data, res, cfg


def run_stats(truth: dict, data, res) -> dict:
    """Window statistics shared by the scenarios, the sensitivity grid and the Monte Carlo."""
    t, lit, k0 = truth["t"], data.lit, res.k_init
    n = t.size
    err = attitude_errors(truth["q"], res.q)                   # (N, 3) rad, NaN before k0
    en = np.linalg.norm(err, axis=1)
    fin = np.isfinite(en)
    out: dict = {"k_init": int(k0), "t_init_s": float(t[k0]) if k0 >= 0 else None}
    for tag, tmin in (("after_0p25_orbit", 0.25 * PERIOD), ("after_1_orbit", PERIOD)):
        base = fin & (t >= tmin)
        d = {}
        for name, m in (("all", base), ("sunlit", base & lit), ("eclipse", base & ~lit)):
            d[name] = {"n": int(m.sum()), "rms_deg": _rms(en[m]) / DEG,
                       "max_deg": float(en[m].max() / DEG) if m.any() else None,
                       "rms_per_axis_deg": [_rms(err[m, i]) / DEG for i in range(3)]}
        out[tag] = d
    # NEES / 3-sigma window
    ks = k0 + SETTLE
    w = np.zeros(n, bool)
    w[ks:] = True
    w &= fin
    e = nees(err[w], res.P_att[w])
    out["stat_window"] = {"k_min": int(ks), "t_min_s": float(t[ks]), "n": int(w.sum())}
    out["nees_mean"] = float(e.mean())
    out["nees_mean_sunlit"] = float(e[lit[w]].mean()) if lit[w].any() else None
    out["nees_mean_eclipse"] = float(e[~lit[w]].mean()) if (~lit[w]).any() else None
    sig_att = res.std[:, :3]
    out["frac_inside_3sigma_attitude_per_axis"] = [float(np.mean(np.abs(err[w, i]) < 3 * sig_att[w, i]))
                                                   for i in range(3)]
    b_err = data.gyro_bias - res.b                             # true - estimated, rad/s
    out["frac_inside_3sigma_bias_per_axis"] = [float(np.mean(np.abs(b_err[w, i]) < 3 * res.std[w, 3 + i]))
                                               for i in range(3)]
    # NIS
    nis = {}
    for name in ("mag", "sun"):
        tn, vn, acc = res.nis[name]
        m = (tn >= t[ks]) & acc
        v = vn[m]
        lo, hi = nis_bounds(3, int(v.size))
        mean = float(v.mean())
        nis[name] = {"n": int(v.size), "mean": mean, "bound_lo": lo, "bound_hi": hi,
                     "inside_95pct_bounds": bool(lo <= mean <= hi),
                     "fraction_of_samples_inside_chi2_95pct": nis_fraction_inside(v, 3),
                     "n_rejected_total": int((~acc).sum())}
    out["nis"] = nis
    # bias error
    be = np.linalg.norm(b_err, axis=1)
    out["bias_error_deg_s"] = {}
    for f in BIAS_FRACS:
        k = max(int(round(f * PERIOD)), k0)
        if k < n:
            out["bias_error_deg_s"][f"{f:g}_orbit"] = {"k": k, "norm": float(be[k] / DEG),
                                                       "sigma_norm": float(np.linalg.norm(res.std[k, 3:]) / DEG)}
    out["bias_error_deg_s"]["end"] = {"k": n - 1, "norm": float(be[-1] / DEG),
                                      "sigma_norm": float(np.linalg.norm(res.std[-1, 3:]) / DEG)}
    out["final_sigma_b_deg_s_per_axis"] = (res.std[-1, 3:] / DEG).tolist()
    out["final_bias_error_deg_s_per_axis"] = (b_err[-1] / DEG).tolist()
    out["true_bias0_deg_s"] = (data.gyro_bias[0] / DEG).tolist()
    return out


def triad_series(truth: dict, data, res):
    """TRIAD (magnetometer primary) at every sunlit sample with a valid sun vector."""
    idx = np.flatnonzero(data.lit & data.sun_valid)
    mag, sun = data.mag[idx], data.sun[idx]
    q_tri = triad(mag, sun, truth["b_eci"][idx], truth["s_eci"][idx])
    err = attitude_errors(truth["q"][idx], q_tri)
    s_m = MagnetometerConfig().sigma / np.linalg.norm(mag, axis=1)
    s_s = SunSensorConfig().sigma_vector
    tr = np.array([np.trace(triad_covariance(mag[i], sun[i], s_m[i], s_s)) for i in range(idx.size)])
    return idx, np.linalg.norm(err, axis=1), np.sqrt(tr)


def triad_stats(idx, err_n, pred_sqrt_tr) -> dict:
    return {"n": int(idx.size), "rms_deg": _rms(err_n) / DEG, "max_deg": float(err_n.max() / DEG),
            "predicted_rms_deg": float(np.sqrt(np.mean(pred_sqrt_tr ** 2)) / DEG),
            "predicted_mean_sqrt_trace_deg": float(pred_sqrt_tr.mean() / DEG),
            "ratio_rms_to_predicted_rms": float(_rms(err_n) / np.sqrt(np.mean(pred_sqrt_tr ** 2)))}


def eclipse_stats(truth, data, res) -> list[dict]:
    """Per complete eclipse: sqrt(trace P_att) at entry (last sunlit sample) and exit (last eclipse sample)."""
    out = []
    n = truth["t"].size
    err = attitude_errors(truth["q"], res.q)
    en = np.linalg.norm(err, axis=1)
    sig = np.sqrt(np.trace(res.P_att, axis1=1, axis2=2))
    for i0, i1 in intervals(~data.lit):
        if i0 - 1 < max(res.k_init, 0) or i1 + 1 >= n:
            continue
        d = {"entry_t_s": float(truth["t"][i0]), "exit_t_s": float(truth["t"][i1]),
             "duration_s": float(truth["t"][i1] - truth["t"][i0] + 1.0),
             "sigma_att_entry_deg": float(sig[i0 - 1] / DEG), "sigma_att_exit_deg": float(sig[i1] / DEG),
             "sigma_ratio_exit_over_entry": float(sig[i1] / sig[i0 - 1]),
             "max_err_inside_deg": float(en[i0:i1 + 1].max() / DEG),
             "err_at_exit_deg": float(en[i1] / DEG)}
        k300 = i1 + 300                        # 300 s (samples) after the last eclipse sample
        if k300 < n:
            d["sigma_att_300s_after_exit_deg"] = float(sig[k300] / DEG)
            d["sigma_ratio_300s_after_exit_over_exit"] = float(sig[k300] / sig[i1])
        out.append(d)
    return out


# ---------------------------------------------------------------------------------------------
# pool workers
# ---------------------------------------------------------------------------------------------
def _worker(job: dict):
    t0 = time.perf_counter()
    kind = job["kind"]
    if kind == "scenario":
        truth = _TRUTH[job["name"]]
        data, res, cfg = estimate(truth, job["seed"])
        stats = run_stats(truth, data, res)
        tri = triad_series(truth, data, res)
        stats["triad"] = triad_stats(*tri)
        stats["eclipses"] = eclipse_stats(truth, data, res)
        payload = {"data": data, "res": res, "stats": stats, "cfg": asdict(cfg),
                   "triad": tri}
    else:
        truth = _slice(_TRUTH["pointing"], job["orbits"])
        data, res, _ = estimate(truth, job["seed"], job.get("mag_sigma"), job.get("css_diode"))
        s = run_stats(truth, data, res)
        payload = {"rms_sunlit_deg": s["after_1_orbit"]["sunlit"]["rms_deg"],
                   "rms_eclipse_deg": s["after_1_orbit"]["eclipse"]["rms_deg"],
                   "max_deg": s["after_1_orbit"]["all"]["max_deg"], "nees_mean": s["nees_mean"],
                   "nis_mag_mean": s["nis"]["mag"]["mean"], "nis_sun_mean": s["nis"]["sun"]["mean"],
                   "nis_mag_n": s["nis"]["mag"]["n"], "nis_sun_n": s["nis"]["sun"]["n"],
                   "nis_mag_inside": s["nis"]["mag"]["inside_95pct_bounds"],
                   "nis_sun_inside": s["nis"]["sun"]["inside_95pct_bounds"],
                   "bias_err_0p25_deg_s": s["bias_error_deg_s"]["0.25_orbit"]["norm"]}
    payload["wall_s"] = time.perf_counter() - t0
    return job["key"], payload


def _jobs() -> list[dict]:
    jobs = [dict(kind="scenario", key=("scenario", "pointing"), name="pointing", seed=0),
            dict(kind="scenario", key=("scenario", "detumble"), name="detumble", seed=0)]
    for seed in SENS_SEEDS:
        for v in SENS_MAG_NT:
            jobs.append(dict(kind="sens", key=("sens_mag", v, seed), orbits=SENS_ORBITS, seed=seed,
                             mag_sigma=v * 1e-9))
        for a in SENS_CSS_DEG:
            jobs.append(dict(kind="sens", key=("sens_css", a, seed), orbits=SENS_ORBITS, seed=seed,
                             css_diode=float(np.deg2rad(a)) / 2.0))
    for seed in MC_SEEDS:
        jobs.append(dict(kind="mc", key=("mc", seed), orbits=MC_ORBITS, seed=seed))
    return jobs


# ---------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------
AX = ("x", "y", "z")


def fig_pointing_attitude(truth, p):
    t, lit = truth["t"], truth["lit"]
    res, data = p["res"], p["data"]
    err = attitude_errors(truth["q"], res.q) / DEG
    sig = res.std[:, :3] / DEG
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(11, 8.5))
    for i, ax in enumerate(axs):
        shade_eclipse(ax, t, lit, label=(i == 0))
        ax.fill_between(t / PERIOD, -3 * sig[:, i], 3 * sig[:, i], color=f"C{i}", alpha=0.2, lw=0,
                        label="±3σ (filter)")
        ax.plot(t / PERIOD, err[:, i], color=f"C{i}", lw=0.6, label=f"δθ_{AX[i]} (true - estimate)")
        ax.legend(fontsize=8, loc="upper right", ncol=3)
        label_axes(ax, "", f"attitude error δθ_{AX[i]} [deg]",
                   "MEKF attitude error, nadir pointing, 5 orbits, seed 0, body axes (q_true = q_est ⊗ exp(δθ))"
                   if i == 0 else None)
    axs[-1].set_xlabel("time [orbits]")
    save_fig(fig, FIG_DIR / "mekf_pointing_attitude.png")


def fig_pointing_bias(truth, p):
    t, lit = truth["t"], truth["lit"]
    res, data = p["res"], p["data"]
    x = t / PERIOD
    b_true = data.gyro_bias / DEG
    b_est = res.b / DEG
    b_err = b_true - b_est
    sig = res.std[:, 3:] / DEG
    fig, axs = plt.subplots(3, 2, sharex=True, figsize=(13, 8.5))
    for i in range(3):
        a1, a2 = axs[i]
        shade_eclipse(a1, t, lit, label=(i == 0))
        shade_eclipse(a2, t, lit, label=False)
        a1.plot(x, b_true[:, i], "k", lw=1.0, label="true")
        a1.plot(x, b_est[:, i], f"C{i}", lw=0.9, label="estimated")
        a1.legend(fontsize=8, loc="upper right", ncol=3)
        label_axes(a1, "", f"gyro bias b_{AX[i]} [deg/s]",
                   "Gyro bias, true and estimated (nadir pointing, 5 orbits, seed 0)" if i == 0 else None)
        a2.fill_between(x, -3 * sig[:, i], 3 * sig[:, i], color=f"C{i}", alpha=0.2, lw=0, label="±3σ (filter)")
        a2.plot(x, b_err[:, i], color=f"C{i}", lw=0.7, label="error (true - estimated)")
        a2.set_yscale("symlog", linthresh=1e-3)
        a2.legend(fontsize=8, loc="upper right")
        label_axes(a2, "", f"bias error δb_{AX[i]} [deg/s]",
                   "Bias error with ±3σ (symlog axis, linear below 1e-3 deg/s)" if i == 0 else None)
    axs[-1, 0].set_xlabel("time [orbits]")
    axs[-1, 1].set_xlabel("time [orbits]")
    save_fig(fig, FIG_DIR / "mekf_pointing_bias.png")


def fig_nis(truth, p):
    t, lit = truth["t"], truth["lit"]
    res = p["res"]
    lo, hi = nis_bounds(3, MA)
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(11, 10))
    for ax, name, title in ((axs[0], "mag", "Magnetometer NIS"), (axs[1], "sun", "Sun-sensor NIS")):
        tn, vn, acc = res.nis[name]
        shade_eclipse(ax, t, lit, label=(name == "mag"))
        # moving mean restarted on every contiguous run of samples (sun: each sunlit pass),
        # so no window mixes NIS from before and after an eclipse
        cuts = np.flatnonzero(np.diff(tn) > 1.5) + 1
        xs, ys = [], []
        for seg_t, seg_v in zip(np.split(tn, cuts), np.split(vn, cuts)):
            if seg_v.size >= MA:
                xs += [seg_t[MA - 1:] / PERIOD, [np.nan]]
                ys += [movmean(seg_v), [np.nan]]
        ax.plot(np.concatenate(xs), np.concatenate(ys), color="C0", lw=0.9,
                label=f"{MA}-sample moving mean (restarted after each gap)")
        ax.axhline(3.0, color="k", lw=0.9, label="expected value 3 (dof)")
        ax.axhspan(lo, hi, color="C2", alpha=0.2, label=f"95 % bounds nis_bounds(3, {MA}) = [{lo:.2f}, {hi:.2f}]")
        ax.legend(fontsize=8, loc="upper right", ncol=2)
        label_axes(ax, "", "mean NIS [-]", f"{title}, nadir pointing, 5 orbits, seed 0")
    err = attitude_errors(truth["q"], res.q)
    k0 = res.k_init
    e = nees(err[k0:], res.P_att[k0:])
    shade_eclipse(axs[2], t, lit, label=False)
    axs[2].plot(t[k0 + MA - 1:] / PERIOD, movmean(e), color="C1", lw=0.9, label=f"{MA}-sample moving mean")
    axs[2].axhline(3.0, color="k", lw=0.9, label="expected value 3 (dof)")
    axs[2].legend(fontsize=8, loc="upper right")
    label_axes(axs[2], "time [orbits]", "mean attitude NEES [-]", "Attitude NEES (consecutive samples are correlated)")
    save_fig(fig, FIG_DIR / "mekf_nis.png")


def fig_eclipse(truth, p):
    t, lit = truth["t"], truth["lit"]
    res = p["res"]
    k_lo, k_hi = int(2 * PERIOD), int(3 * PERIOD)
    sl = slice(k_lo, k_hi + 1)
    err = np.linalg.norm(attitude_errors(truth["q"], res.q), axis=1) / DEG
    sig3 = 3 * np.sqrt(np.trace(res.P_att, axis1=1, axis2=2)) / DEG
    idx, tr_err, _ = p["triad"]
    m = (idx >= k_lo) & (idx <= k_hi)
    x = t / PERIOD
    fig, ax = plt.subplots(figsize=(11, 5.5))
    shade_eclipse(ax, t[sl], lit[sl])
    ax.semilogy(t[idx[m]] / PERIOD, tr_err[m] / DEG, ".", color="C3", ms=2.5, alpha=0.6,
                label="TRIAD ‖δθ‖ (each sunlit sample, no filtering)")
    ax.semilogy(x[sl], err[sl], color="C0", lw=0.9, label="MEKF ‖δθ‖")
    ax.semilogy(x[sl], sig3[sl], color="C1", lw=1.4, label="3·sqrt(trace P_att) (filter)")
    ax.set_xlim(x[k_lo], x[k_hi])
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    label_axes(ax, "time [orbits]", "attitude error norm [deg]",
               "MEKF through eclipse, orbit 3 of nadir pointing, seed 0 (grey = eclipse; TRIAD needs the sun)")
    save_fig(fig, FIG_DIR / "mekf_eclipse.png")


def fig_detumble(truth, p):
    t, lit = truth["t"], truth["lit"]
    res, data = p["res"], p["data"]
    x = t / PERIOD
    err = attitude_errors(truth["q"], res.q) / DEG
    sig = res.std / DEG
    b_err = (data.gyro_bias - res.b) / DEG
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(11, 14))
    for a in axs:
        shade_eclipse(a, t, lit, label=(a is axs[0]))
    w = truth["w"] / DEG
    for i in range(3):
        axs[0].plot(x, w[:, i], color=f"C{i}", lw=0.8, label=f"ω_{AX[i]}")
    axs[0].plot(x, np.linalg.norm(w, axis=1), "k--", lw=0.9, label="|ω|")
    axs[0].legend(fontsize=8, ncol=5, loc="upper right")
    label_axes(axs[0], "", "body rate ω [deg/s]", "B-dot detumble from 3 deg/s, 2 orbits, seed 0: true body rate rel. to ECI")
    for i in range(3):
        a = axs[1 + i]
        a.fill_between(x, -3 * sig[:, i], 3 * sig[:, i], color=f"C{i}", alpha=0.2, lw=0, label="±3σ (filter)")
        a.plot(x, err[:, i], color=f"C{i}", lw=0.6, label=f"δθ_{AX[i]}")
        keep = x >= 0.05
        lim = 1.15 * max(np.nanmax(np.abs(err[keep, i])), np.nanmax(3 * sig[keep, i]))
        a.set_ylim(-lim, lim)        # initial TRIAD transient (first 0.05 orbit) is clipped
        a.legend(fontsize=8, loc="upper right", ncol=2)
        label_axes(a, "", f"δθ_{AX[i]} [deg]", "MEKF attitude error, body axes (y-limits exclude the first 0.05 orbit)" if i == 0 else None)
    for i in range(3):
        axs[4].plot(x, b_err[:, i], color=f"C{i}", lw=0.8, label=f"δb_{AX[i]}")
        axs[4].plot(x, 3 * sig[:, 3 + i], color=f"C{i}", lw=0.8, ls="--")
        axs[4].plot(x, -3 * sig[:, 3 + i], color=f"C{i}", lw=0.8, ls="--")
    axs[4].plot([], [], "k--", lw=0.8, label="±3σ (filter)")
    axs[4].set_yscale("symlog", linthresh=1e-3)
    axs[4].legend(fontsize=8, ncol=4, loc="upper right")
    label_axes(axs[4], "time [orbits]", "bias error δb [deg/s]", "Gyro bias error with ±3σ (symlog axis, linear below 1e-3 deg/s)")
    save_fig(fig, FIG_DIR / "mekf_detumble.png")


def sens_summary(results, keyname, values):
    cells = {}
    for v in values:
        runs = [results[(keyname, v, s)] for s in SENS_SEEDS]
        c = {}
        for q in ("rms_sunlit_deg", "rms_eclipse_deg", "max_deg", "nees_mean", "nis_mag_mean", "nis_sun_mean"):
            a = np.array([r[q] for r in runs])
            c[q] = {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max()),
                    "per_seed": a.tolist()}
        cells[v] = c
    return cells


def fig_sensitivity(sens):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, (key, vals, xl, title) in zip(axs, (
            ("mag", SENS_MAG_NT, "magnetometer noise σ_m [nT, per axis]",
             "vs magnetometer noise (CSS default 5° 1σ)"),
            ("css", SENS_CSS_DEG, "coarse sun sensor 1σ angle error [deg]",
             "vs sun-sensor error (magnetometer default 15 nT)"))):
        cells = sens[key]
        for qn, col, lab in (("rms_sunlit_deg", "C0", "sunlit"), ("rms_eclipse_deg", "C3", "eclipse")):
            m = np.array([cells[v][qn]["mean"] for v in vals])
            lo = np.array([cells[v][qn]["min"] for v in vals])
            hi = np.array([cells[v][qn]["max"] for v in vals])
            ax.errorbar(vals, m, yerr=[m - lo, hi - m], color=col, marker="o", capsize=3, lw=1.2,
                        label=f"RMS ‖δθ‖ {lab}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(vals)
        ax.set_xticklabels([f"{v:g}" for v in vals])
        ax.minorticks_off()
        yt = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        ax.set_yticks(yt)
        ax.set_yticklabels([f"{v:g}" for v in yt])
        ax.set_ylim(0.15, 1.2)
        ax.legend(fontsize=8, loc="upper left")
        label_axes(ax, xl, "RMS attitude error after 1 orbit [deg]", title)
    fig.suptitle("MEKF sensitivity, nadir pointing, 3 orbits, seeds 0-4 (mean, min-max), filter sigmas matched to sensors",
                 fontsize=11)
    save_fig(fig, FIG_DIR / "mekf_sensitivity.png")


def fig_mc(mc):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, key, title, c in ((axs[0], "nees_mean", "attitude NEES", "C1"),
                              (axs[1], "nis_mag_mean", "magnetometer NIS", "C0"),
                              (axs[2], "nis_sun_mean", "sun-sensor NIS", "C2")):
        v = np.array([r[key] for r in mc])
        ax.hist(v, bins=10, color=c, alpha=0.75, edgecolor="k")
        ax.axvline(3.0, color="k", lw=1.4, label="expected 3")
        if key != "nees_mean":
            n = int(np.median([r["nis_mag_n" if "mag" in key else "nis_sun_n"] for r in mc]))
            lo, hi = nis_bounds(3, n)
            ax.axvspan(lo, hi, color="gray", alpha=0.25, label=f"95 % bounds (n = {n}): [{lo:.2f}, {hi:.2f}]")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
        ax.legend(fontsize=8, loc="upper left")
        label_axes(ax, f"per-run mean {title} [-]", "number of runs", f"{title}")
    fig.suptitle("Monte Carlo consistency, nadir pointing, 3 orbits, 20 sensor-noise seeds", fontsize=11)
    save_fig(fig, FIG_DIR / "mekf_consistency_mc.png")


# ---------------------------------------------------------------------------------------------
def main() -> None:
    t_start = time.perf_counter()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    n_cpu = os.cpu_count() or 2

    # 1. truth (once per scenario, shared)
    tj = [dict(name="pointing", orbits=5.0), dict(name="detumble", orbits=2.0)]
    with Pool(2) as pool:
        truth = dict(pool.map(_truth_worker, tj))
    t_truth = time.perf_counter() - t_start
    print(f"truth done ({t_truth:.0f} s)", flush=True)
    for name, tr in truth.items():
        dt = np.diff(tr["t"])
        assert np.allclose(dt, dt[0], atol=1e-9), "non-uniform truth grid"

    # 2. estimation runs
    jobs = _jobs()
    n_proc = min(len(jobs), max(1, n_cpu - 2))
    print(f"{len(jobs)} estimation runs on {n_proc} processes ...", flush=True)
    results, walls = {}, {}
    with Pool(n_proc, initializer=_init_pool, initargs=(truth,)) as pool:
        for i, (key, payload) in enumerate(pool.imap_unordered(_worker, jobs, chunksize=1), 1):
            results[key] = payload
            walls[key] = payload["wall_s"]
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done ({time.perf_counter() - t_start:.0f} s)", flush=True)

    pp, pd_ = results[("scenario", "pointing")], results[("scenario", "detumble")]

    # 3. figures
    fig_pointing_attitude(truth["pointing"], pp)
    fig_pointing_bias(truth["pointing"], pp)
    fig_nis(truth["pointing"], pp)
    fig_eclipse(truth["pointing"], pp)
    fig_detumble(truth["detumble"], pd_)
    sens = {"mag": sens_summary(results, "sens_mag", SENS_MAG_NT), "css": sens_summary(results, "sens_css", SENS_CSS_DEG)}
    fig_sensitivity(sens)
    mc = [results[("mc", s)] for s in MC_SEEDS]
    fig_mc(mc)

    # 4. metrics
    def agg(key):
        v = np.array([r[key] for r in mc])
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "min": float(v.min()),
                "max": float(v.max()), "per_seed": v.tolist()}

    mc_summary = {"seeds": list(MC_SEEDS), "n_orbits": MC_ORBITS,
                  "nees_mean_per_run": agg("nees_mean"),
                  "nis_mag_mean_per_run": agg("nis_mag_mean"),
                  "nis_sun_mean_per_run": agg("nis_sun_mean"),
                  "fraction_runs_nis_mag_inside_95pct_bounds": float(np.mean([r["nis_mag_inside"] for r in mc])),
                  "fraction_runs_nis_sun_inside_95pct_bounds": float(np.mean([r["nis_sun_inside"] for r in mc])),
                  "nis_mag_bounds_per_run": [list(nis_bounds(3, r["nis_mag_n"])) for r in mc],
                  "nis_sun_bounds_per_run": [list(nis_bounds(3, r["nis_sun_n"])) for r in mc],
                  "pooled_nis_mag_mean": float(np.average([r["nis_mag_mean"] for r in mc], weights=[r["nis_mag_n"] for r in mc])),
                  "pooled_nis_sun_mean": float(np.average([r["nis_sun_mean"] for r in mc], weights=[r["nis_sun_n"] for r in mc]))}
    sens_json = {"seeds": list(SENS_SEEDS), "n_orbits": SENS_ORBITS,
                 "statistic": "RMS ||dtheta|| after 1 orbit, sunlit and eclipse separately, mean/min/max over seeds",
                 "mag_sigma_nT": {f"{v:g}": c for v, c in sens["mag"].items()},
                 "css_sigma_angle_deg": {f"{v:g}": c for v, c in sens["css"].items()},
                 "css_diode_sigma_rule": "sigma_diode = angle_rad / 2; filter sigma_sun = sqrt(2) sigma_diode",
                 "note": "the 5 deg CSS cell uses sigma_diode = 0.043633 (default 0.0436); the 15 nT cell is the default magnetometer"}

    scen = {"pointing": {"seed": 0, "n_orbits": 5, "initial_offset_deg": list(POINTING_OFFSET_DEG),
                         "mekf_cfg": pp["cfg"], **pp["stats"]},
            "detumble": {"seed": 0, "n_orbits": 2, "initial_rate_deg_s": DETUMBLE_RATE_DEG,
                         "truth_initial_state_seed": 0, "mekf_cfg": pd_["cfg"], **pd_["stats"]}}
    for name, tr in truth.items():
        scen[name]["eclipse_fraction"] = float(np.mean(~tr["lit"]))
        scen[name]["n_samples"] = int(tr["t"].size)

    pnt, det = scen["pointing"], scen["detumble"]
    checks = {
        "nees_in_2p55_3p45_pointing": bool(2.55 <= pnt["nees_mean"] <= 3.45),
        "nees_in_2p55_3p45_detumble": bool(2.55 <= det["nees_mean"] <= 3.45),
        "bias_error_lt_0p01_deg_s_at_0p25_orbit_pointing": bool(pnt["bias_error_deg_s"]["0.25_orbit"]["norm"] < 0.01),
        "bias_error_lt_0p01_deg_s_at_0p25_orbit_detumble": bool(det["bias_error_deg_s"]["0.25_orbit"]["norm"] < 0.01),
        "eclipse_rms_gt_sunlit_rms_pointing_after_1_orbit":
            bool(pnt["after_1_orbit"]["eclipse"]["rms_deg"] > pnt["after_1_orbit"]["sunlit"]["rms_deg"]),
    }

    metrics = {
        "config": {
            "orbital_period_s": PERIOD, "dt_s": 1.0,
            "sensors": {"magnetometer": asdict(MagnetometerConfig()), "css": {**asdict(SunSensorConfig()),
                        "sigma_vector": SunSensorConfig().sigma_vector}, "gyro": asdict(GyroConfig())},
            "mekf_default": asdict(MekfConfig()),
            "stat_window": f"NEES / NIS / 3-sigma fractions use samples k >= k_init + {SETTLE}",
            "orbit_windows": "after 0.25 orbit: t >= 0.25 P; after 1 orbit: t >= P",
            "sunlit_eclipse_classification": "true shadow flag (FieldTable.sunlit) at the sample",
            "bias_error_definition": "true gyro bias used in sample k minus estimated bias, k = round(frac P) (>= k_init)",
            "triad": {"primary": "magnetometer", "samples": "lit and sun_valid",
                      "sigma_mag": "sigma_m / |measured mag|", "sigma_sun": "SunSensorConfig.sigma_vector",
                      "predicted_rms": "sqrt(mean(trace triad_covariance))",
                      "predicted_mean_sqrt_trace": "mean(sqrt(trace triad_covariance))  (spec definition)"},
            "eclipse_stats_definition": "entry = last sunlit sample before the eclipse; exit = last eclipse sample; complete eclipses after k_init only",
            "nis_bounds": "common.ekf.nis_bounds(3, n), 95 % two-sided, n = samples in window; figure moving mean uses n = 100",
            "moving_mean_samples": MA,
            "seeds": {"scenarios": 0, "sensitivity": list(SENS_SEEDS), "monte_carlo": list(MC_SEEDS)},
        },
        "scenarios": scen,
        "sensitivity": sens_json,
        "monte_carlo": mc_summary,
        "checks": checks,
        "runtime_s": {"total": 0.0, "truth": t_truth,
                      "scenario_pointing_run": walls[("scenario", "pointing")],
                      "scenario_detumble_run": walls[("scenario", "detumble")],
                      "truth_pointing": truth["pointing"]["wall_s"], "truth_detumble": truth["detumble"]["wall_s"],
                      "sum_of_estimation_runs": float(sum(walls.values())),
                      "n_estimation_runs": len(jobs), "n_processes": n_proc},
    }
    metrics["runtime_s"]["total"] = time.perf_counter() - t_start
    METRICS_PATH.write_text(json.dumps(_clean(metrics), indent=2, allow_nan=False))
    print(f"wrote {METRICS_PATH} (total {metrics['runtime_s']['total']:.1f} s)")


if __name__ == "__main__":
    main()
