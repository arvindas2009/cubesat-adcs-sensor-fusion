"""Figures of the Project A flight study (ES-EKF vs PX4 EKF2; spec ``docs/specs/tuning_writeup.md`` section 2).

Moved unchanged from ``ekf_study.py``. Every figure function takes the keywords ``prefix`` (default ``"ekf"``) and
``out_dir`` (directory that receives the PNG; default ``sensor_fusion/report/figures``) and writes
``<out_dir>/<prefix>_<log_id>_<kind>.png`` (``kind`` in pos, att, bias, nis; 150 dpi) and returns that path.
Axes use degrees / metres / seconds; the filter itself works in radians and SI. EKF2 is a reference, not truth.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.ekf import nis_bounds  # noqa: E402
from common.plotting import deg, label_axes, save_fig  # noqa: E402
from sensor_fusion.ekf_drone import IDX_BA, IDX_BG  # noqa: E402
from sensor_fusion.metrics import RESET_EXCL_S, SENSOR_ORDER, SKIP_INIT_S  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parent
FIG_DIR = REPORT_DIR / "figures"
CAVEAT = "EKF2 reference, not truth"
NIS_MA = 20                # moving-average length for the NIS figure [samples]
PLOT_MAX_POINTS = 4000     # decimation limit for the long filter-rate time series
INC_COLORS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
INC_LABEL = {1: "inc 1: GPS+baro", 2: "inc 2: +mag", 3: "inc 3: +biases"}


def _title(role, log_id, dur):
    return f"{role} flight {log_id[:8]} ({dur:.0f} s)"


def _decim(n: int) -> slice:
    return slice(None, None, max(1, n // PLOT_MAX_POINTS))


def _path(prefix: str, out_dir, log_id: str, kind: str) -> Path:
    return Path(FIG_DIR if out_dir is None else out_dir) / f"{prefix}_{log_id}_{kind}.png"


def fig_pos(runs, role, log_id, dur, prefix: str = "ekf", out_dir: Path | None = None) -> Path:
    """Horizontal / vertical position and velocity difference vs EKF2 for ``runs = {increment: {"arr": ...}}``."""
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
    return save_fig(fig, _path(prefix, out_dir, log_id, "pos"))


def fig_att(runs, resets, role, log_id, dur, prefix: str = "ekf", out_dir: Path | None = None) -> Path:
    """Tilt and wrapped yaw difference vs EKF2, EKF2 quaternion-reset times (s) marked."""
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
    return save_fig(fig, _path(prefix, out_dir, log_id, "att"))


def fig_bias(res, bias_ref, role, log_id, dur, prefix: str = "ekf", out_dir: Path | None = None) -> Path:
    """Gyro / accel bias estimates +/- 1 sigma of an increment-3 ``EkfResult`` against EKF2's bias estimates."""
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
    return save_fig(fig, _path(prefix, out_dir, log_id, "bias"))


def fig_nis(res, role, log_id, dur, prefix: str = "ekf", out_dir: Path | None = None) -> Path:
    """Moving-average NIS per sensor (log axis) with the 95 % band of an ``NIS_MA``-sample mean."""
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
    return save_fig(fig, _path(prefix, out_dir, log_id, "nis"))
