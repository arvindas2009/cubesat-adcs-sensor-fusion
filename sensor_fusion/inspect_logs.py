"""Inspection plots and smoke check for the pinned PX4 logs (spec section 3).

``python -m sensor_fusion.inspect_logs``           -> 5 PNGs per log in ``sensor_fusion/report/figures/``
``python -m sensor_fusion.inspect_logs --smoke``   -> load every pinned log, print rates / removals / gaps

Figures (``data_<id>_*.png``): ``imu`` (gyro, accel, sampling-interval histogram),
``mag`` (norm and body components), ``alt`` (baro vs GPS altitude vs EKF2), ``track``
(GPS and EKF2 NED ground track), ``attitude`` (EKF2 Euler angles).
EKF2 is the reference, not ground truth.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import frames  # noqa: E402
from common import quaternion as quat  # noqa: E402
from common.plotting import deg, label_axes, save_fig, subplots_stacked, time_axes  # noqa: E402
from sensor_fusion.download_logs import PINNED_LOGS  # noqa: E402
from sensor_fusion.loader import FlightLog, load_log, sample_rates  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent / "report" / "figures"
CAPTION = "EKF2 = PX4 onboard estimate (reference, not ground truth)"


def _title(log: FlightLog, role: str, what: str) -> str:
    return f"{what} - {role} log {log.log_id[:8]} ({log.info['firmware']['ver_sw_release']})"


def plot_imu(log, role, out):
    imu = log.imu
    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.9], hspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2])
    for i, n in enumerate("xyz"):
        ax1.plot(imu.t, deg(imu.gyro[:, i]), lw=0.5, label=f"gyro {n}")
        ax2.plot(imu.t, imu.accel[:, i], lw=0.5, label=f"accel {n}")
    label_axes(ax1, "", "angular rate, FRD [deg/s]", _title(log, role, "Raw IMU"))
    time_axes(ax2, "specific force, FRD [m/s$^2$]")
    ax1.legend(ncol=3, loc="upper right", fontsize=8)
    ax2.legend(ncol=3, loc="upper right", fontsize=8)
    dt_ms = np.diff(imu.t) * 1e3
    ax3.hist(dt_ms, bins=np.linspace(0, max(2.0 * np.median(dt_ms), 1.0), 80), color="C0")
    ax3.set_yscale("log")
    n_big = int((dt_ms > 2.0 * np.median(dt_ms)).sum())
    label_axes(ax3, "sample interval [ms]", "count",
               f"sampling interval: median {np.median(dt_ms):.3f} ms ({imu.rate_hz:.1f} Hz), "
               f"max {dt_ms.max():.1f} ms, {n_big} intervals > 2x median")
    return save_fig(fig, out / f"data_{log.log_id}_imu.png")


def plot_mag(log, role, out):
    m = log.mag
    fig, (a1, a2) = subplots_stacked(2)
    norm_ut = np.linalg.norm(m.field, axis=1) * 1e6
    a1.plot(m.t, norm_ut, lw=0.8)
    a1.axhline(norm_ut.mean(), color="k", ls="--", lw=0.8, label=f"mean {norm_ut.mean():.1f} uT")
    a1.legend(fontsize=8)
    label_axes(a1, "", "|B| [uT]", _title(log, role, "Magnetometer norm"))
    for i, n in enumerate("xyz"):
        a2.plot(m.t, m.field[:, i] * 1e6, lw=0.8, label=f"B {n} (FRD)")
    a2.legend(ncol=3, fontsize=8)
    time_axes(a2, "B, body FRD [uT]")
    return save_fig(fig, out / f"data_{log.log_id}_mag.png")


def plot_alt(log, role, out):
    fig, (a1, a2) = subplots_stacked(2)
    a1.plot(log.baro.t, log.baro.alt_baro - log.baro.alt_baro[0], label="baro (up), minus first sample")
    a1.plot(log.gps.t, log.gps.alt - log.gps.alt[0], label="GPS altitude (up), minus first sample")
    a1.plot(log.ekf2.t_pos, -(log.ekf2.pos_ned[:, 2] - log.ekf2.pos_ned[0, 2]), label="EKF2 -z (up)", lw=0.8)
    a1.legend(fontsize=8)
    label_axes(a1, "", "height change [m]", _title(log, role, "Baro vs GPS altitude"))
    a2.plot(log.baro.t, log.baro.temperature, color="C3")
    time_axes(a2, "baro temperature [deg C]")
    return save_fig(fig, out / f"data_{log.log_id}_alt.png")


def plot_track(log, role, out):
    ned = frames.lla_to_ned(log.gps.lat, log.gps.lon, log.gps.alt, log.ekf2.ref_lat, log.ekf2.ref_lon, log.gps.alt[0])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(ned[:, 1], ned[:, 0], ".", ms=2, label="GPS (NED about EKF2 ref lat/lon)")
    ax.plot(log.ekf2.pos_ned[:, 1], log.ekf2.pos_ned[:, 0], lw=0.9, label="EKF2 local position")
    ax.plot(log.ekf2.pos_ned[0, 1], log.ekf2.pos_ned[0, 0], "go", label="start")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8)
    label_axes(ax, "East [m]", "North [m]", _title(log, role, "Ground track") + "\n" + CAPTION)
    return save_fig(fig, out / f"data_{log.log_id}_track.png")


def plot_attitude(log, role, out):
    roll, pitch, yaw = quat.to_euler_zyx(log.ekf2.q)
    fig, axes = subplots_stacked(3)
    for ax, v, n in zip(axes, (roll, pitch, yaw), ("roll", "pitch", "yaw")):
        ax.plot(log.ekf2.t_att, deg(v), lw=0.8)
        label_axes(ax, "", f"{n} [deg]")
    for ax in axes:
        for tr in log.info.get("ekf2_quat_reset_times_s", []):
            ax.axvline(tr, color="r", ls=":", lw=0.8)
    axes[0].set_title(_title(log, role, "EKF2 attitude (ZYX Euler, FRD to NED)") + "\n" + CAPTION
                      + "; red = EKF2 quaternion reset")
    time_axes(axes[2], "yaw [deg]")
    return save_fig(fig, out / f"data_{log.log_id}_attitude.png")


def make_plots(log: FlightLog, role: str, out: Path = FIG_DIR):
    return [f(log, role, out) for f in (plot_imu, plot_mag, plot_alt, plot_track, plot_attitude)]


def smoke():
    """Load every pinned log and print per-stream rates, removals and gaps."""
    for e in PINNED_LOGS:
        log = load_log(e["id"])
        for name, tt in (("imu", log.imu.t), ("mag", log.mag.t), ("baro", log.baro.t), ("gps", log.gps.t),
                         ("ekf2_att", log.ekf2.t_att), ("ekf2_pos", log.ekf2.t_pos)):
            assert np.all(np.diff(tt) > 0) and tt[0] >= 0, name
        for arr in (log.imu.gyro, log.imu.accel, log.mag.field, log.baro.alt_baro, log.gps.vel_ned, log.ekf2.q):
            assert np.isfinite(arr).all()
        assert np.allclose(np.linalg.norm(log.ekf2.q, axis=1), 1.0) and (log.ekf2.q[:, 0] >= 0).all()
        r = sample_rates(log)
        print(f"{e['id'][:8]} {e['role']:10s} window={log.info['window_duration_s']:.1f}s "
              + " ".join(f"{k}={v:.2f}Hz" for k, v in r.items()))
        print("   removed:", {k: {kk: vv for kk, vv in v.items() if kk in ('nonfinite', 'duplicate_t', 'backward_t', 'fix_lt3', 'outside_window')}
                              for k, v in log.info["removed"].items()})
        print("   gaps:", log.info["gaps"])
    print("smoke OK")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(FIG_DIR))
    args = ap.parse_args(argv)
    if args.smoke:
        smoke()
        return
    for e in PINNED_LOGS:
        log = load_log(e["id"])
        for p in make_plots(log, e["role"], Path(args.out)):
            print(p)


if __name__ == "__main__":
    main()
