"""3D attitude visualiser for Project B (M11): closed-loop result -> MP4 / GIF animation.

Spec: ``docs/specs/visualizer.md``. A visual aid only; it adds no numbers to the reports.

Conventions (``CONVENTIONS.md``): quaternions scalar-first Hamilton, body-to-ECI;
LVLH z = nadir, y = negative orbit normal, x = y x z. Everything is drawn in the
**display frame** ``p_display = DISPLAY_FLIP @ p_lvlh`` (LVLH rotated by 180 deg about
x, so zenith is up and nadir is the fixed vector ``(0, 0, -1)``).

Units: SI, radians (the on-screen text and the error panel use degrees, seconds and
orbits). Time is on axis 0 of every array.

The animation is time-warped (:func:`frame_schedule` samples fast phases densely); the
on-screen clock always shows the true simulation time.

Importing this module imports only numpy and the repo modules; matplotlib is imported
inside :func:`render`, and ``ffmpeg`` is only needed for ``.mp4`` output.

CLI: ``python -m adcs.visualize [--orbits 6] [--rate-deg 10] [--seed 0]
[--sensors realistic|ideal] [--filter design|g7] [--max-frames 600] [--fps 25]
[--dpi 100] [--out PATH]``.
"""
from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common import quaternion as quat
from common.frames import dcm_eci_to_lvlh
from adcs.config import DEFAULT, SimConfig
from adcs.environment import sun_direction_eci, sunlit
from adcs.orbit import Ephemeris, initial_state, orbital_period

__all__ = [
    "DISPLAY_FLIP",
    "body_dcm_lvlh",
    "sun_lvlh",
    "frame_schedule",
    "AnimationData",
    "build_animation_data",
    "render",
    "main",
]

#: LVLH -> display frame (proper rotation, det = +1): zenith (-LVLH z) points up.
DISPLAY_FLIP = np.diag([1.0, -1.0, -1.0])

_BODY_DIMS = np.array([0.353, 0.353, 1.2])   # plot units: 10 x 10 x 34 cm with 34 cm = 1.2
_BOX = 1.6                                   # 3D axis half-extent
_MODE_NAMES = {0: "DETUMBLE", 1: "POINTING"}


def _f(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def body_dcm_lvlh(q, r_eci, v_eci) -> np.ndarray:
    """Body axes in LVLH axes, ``(..., 3, 3)``; column ``k`` is body axis ``k`` in LVLH.

    ``q`` (..., 4) body-to-ECI (scalar-first, normalised on entry; NaN rows give NaN),
    ``r_eci`` (m) and ``v_eci`` (m/s) (..., 3). ``C = dcm_eci_to_lvlh(r, v) @ R(q)
    = R(q_ref)^T R(q)``; ``C = I`` for a body aligned with LVLH.
    """
    return dcm_eci_to_lvlh(_f(r_eci), _f(v_eci)) @ quat.to_dcm(q)


def sun_lvlh(s_eci, r_eci, v_eci) -> np.ndarray:
    """Unit Sun vector in LVLH axes, ``(..., 3)``.

    ``s_eci`` (..., 3) Earth-to-Sun direction in ECI (any length, normalised here),
    ``r_eci`` (m), ``v_eci`` (m/s) the satellite state in ECI.
    """
    s = _f(s_eci)
    s_l = np.einsum("...ij,...j->...i", dcm_eci_to_lvlh(_f(r_eci), _f(v_eci)), s)
    return s_l / np.linalg.norm(s_l, axis=-1, keepdims=True)


def _rel_angle(C: np.ndarray) -> np.ndarray:
    """Rotation angle (rad) of ``C_i^T C_{i+1}`` for consecutive matrices, shape ``(N-1,)``.

    Uses ``atan2(|skew part|, (tr - 1)/2)``, accurate for small and large angles.
    """
    rel = np.swapaxes(C[:-1], -1, -2) @ C[1:]
    s = 0.5 * np.stack([rel[:, 2, 1] - rel[:, 1, 2],
                        rel[:, 0, 2] - rel[:, 2, 0],
                        rel[:, 1, 0] - rel[:, 0, 1]], axis=-1)
    c = 0.5 * (np.trace(rel, axis1=-2, axis2=-1) - 1.0)
    return np.arctan2(np.linalg.norm(s, axis=-1), c)


def frame_schedule(t, C, sun, max_frames: int = 600, dt_min: float = 1.0,
                   dt_max: float = 60.0) -> np.ndarray:
    """Sorted integer frame indices into the ``(N,)`` time grid (motion-adaptive sampling).

    ``t`` (N,) s strictly increasing, ``C`` (N, 3, 3) body axes in LVLH, ``sun``
    (N, 3) LVLH unit vectors. Per-step motion = body rotation angle + Sun-vector angle
    (rad, NaN counts as 0). A frame is emitted once the accumulated motion since the
    last frame reaches the target angle ``a`` (and ``dt_min`` s have passed), or after
    ``dt_max`` s regardless. ``a`` in [1 deg, 90 deg] is the smallest value (bisection)
    giving at most ``max_frames`` frames; if even 90 deg gives more, that schedule is
    subsampled uniformly by index. Index 0 and ``N - 1`` are always included.
    """
    t = _f(t)
    n = t.size
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    max_frames = max(int(max_frames), 2)
    C = _f(C)
    sun = _f(sun)
    m = _rel_angle(C)
    ds = np.sum(sun[:-1] * sun[1:], axis=-1)
    m = m + np.arccos(np.clip(ds, -1.0, 1.0))
    m = np.where(np.isfinite(m), m, 0.0)
    cm = np.concatenate([[0.0], np.cumsum(m)])           # cm[j] = motion from index 0 to j

    def walk(a: float, limit: int | None = None) -> list:
        idx = [0]
        last = 0
        while last < n - 1:
            j_motion = int(np.searchsorted(cm, cm[last] + a, side="left"))
            j_min = int(np.searchsorted(t, t[last] + dt_min, side="left"))
            j_max = int(np.searchsorted(t, t[last] + dt_max, side="left"))
            j = min(max(j_motion, j_min), j_max)
            j = max(j, last + 1)
            if j >= n - 1:
                break
            idx.append(j)
            last = j
            if limit is not None and len(idx) >= limit:
                break
        idx.append(n - 1)
        return idx

    lo, hi = np.deg2rad(1.0), np.deg2rad(90.0)
    out = walk(lo, max_frames + 1)
    if len(out) > max_frames:
        out = walk(hi, max_frames + 1)
        if len(out) > max_frames:
            full = np.asarray(walk(hi))
            sel = np.unique(np.round(np.linspace(0, full.size - 1, max_frames)).astype(int))
            return full[sel].astype(int)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            cand = walk(mid, max_frames + 1)
            if len(cand) <= max_frames:
                hi, out = mid, cand
            else:
                lo = mid
    return np.asarray(out, dtype=int)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
@dataclass
class AnimationData:
    """Per-frame quantities of the selected frames (M frames); time on axis 0, SI/rad."""

    t: np.ndarray                  # (M,) s since epoch
    C: np.ndarray                  # (M, 3, 3) true body axes in LVLH (columns)
    sun: np.ndarray                # (M, 3) unit Sun vector in LVLH
    lit: np.ndarray                # (M,) bool, truth shadow flag (True = sunlit)
    mode: np.ndarray               # (M,) int, 0 = DETUMBLE, 1 = POINTING
    nadir_err: np.ndarray          # (M,) rad
    C_est: np.ndarray | None = None      # (M, 3, 3) estimated body axes in LVLH (NaN before init)
    know_err: np.ndarray | None = None   # (M,) rad, knowledge error angle
    period: float = 5560.0               # s, orbital period for the "orbits" label


def build_animation_data(res, cfg: SimConfig = DEFAULT, max_frames: int = 600,
                         dt_min: float = 1.0, dt_max: float = 60.0) -> AnimationData:
    """Select frames with :func:`frame_schedule` and evaluate all display quantities on them.

    ``res`` is a ``ClosedLoopResult`` or a ``SimResult`` (``lit`` is then computed from
    the ephemeris and the Sun). ``know_err`` rows (rad, rotation vectors) become angles.
    """
    t = _f(res.t)
    eph = Ephemeris(*initial_state(cfg), float(t[-1]) + 10.0)
    r, v = eph.at(t)
    s_eci = sun_direction_eci(cfg.jd_epoch + t / 86400.0)
    C_all = body_dcm_lvlh(res.q, r, v)
    sun_all = sun_lvlh(s_eci, r, v)
    idx = frame_schedule(t, C_all, sun_all, max_frames=max_frames, dt_min=dt_min, dt_max=dt_max)

    lit_all = getattr(res, "lit", None)
    lit = sunlit(r[idx], s_eci[idx]) if lit_all is None else np.asarray(lit_all, bool)[idx]
    C_est = know = None
    q_est = getattr(res, "q_est", None)
    if q_est is not None:
        C_est = body_dcm_lvlh(_f(q_est)[idx], r[idx], v[idx])
    ke = getattr(res, "know_err", None)
    if ke is not None:
        know = np.linalg.norm(_f(ke)[idx], axis=-1)
    return AnimationData(
        t=t[idx], C=C_all[idx], sun=sun_all[idx], lit=lit,
        mode=np.asarray(res.mode)[idx].astype(int), nadir_err=_f(res.nadir_err)[idx],
        C_est=C_est, know_err=know, period=float(orbital_period(cfg.orbit.a)))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
_FACES = (  # corner index = 4*(sx>0) + 2*(sy>0) + (sz>0); faces: -x, +x, -y, +y, -z, +z
    (0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))
_CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                    float) * 0.5 * _BODY_DIMS
_AXIS_COLORS = ("tab:red", "tab:green", "tab:blue")
_AXIS_NAMES = ("x", "y", "z")


def _prism_faces(C: np.ndarray) -> list:
    """Display-frame vertices of the six prism faces for body axes ``C`` (3, 3) in LVLH."""
    pts = _CORNERS @ (DISPLAY_FLIP @ C).T
    return [pts[list(f)] for f in _FACES]


def render(data: AnimationData, path, fps: int = 25, dpi: int = 100, title: str | None = None) -> Path:
    """Write the animation to ``path`` (``.mp4`` via ffmpeg/libx264, ``.gif`` via Pillow).

    Frames follow ``data`` (display frame, see module docstring); the final frame is held
    for one second. Parent directories are created; returns the output ``Path``.
    Raises ``ValueError`` for other extensions and ``RuntimeError`` if ``.mp4`` is
    requested without ``ffmpeg`` on ``PATH``.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in (".mp4", ".gif"):
        raise ValueError(f"unsupported extension {path.suffix!r}: use .mp4 or .gif")
    if ext == ".mp4" and shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH, which is needed for .mp4 output; "
                           "install ffmpeg or write a .gif instead")
    path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import animation
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fps = int(fps)
    dpi = int(dpi)
    w_px = 2 * int(round(12.0 * dpi / 2.0))
    h_px = 2 * int(round(6.0 * dpi / 2.0))
    fig = Figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2], left=0.0, right=0.97, bottom=0.2,
                          top=0.93, wspace=0.12)
    ax = fig.add_subplot(gs[0], projection="3d")
    ax2 = fig.add_subplot(gs[1])
    if title:
        fig.suptitle(title, fontsize=11, y=0.985)

    # --- 3D scene ---------------------------------------------------------
    ax.set_xlim(-_BOX, _BOX)
    ax.set_ylim(-_BOX, _BOX)
    ax.set_zlim(-_BOX, _BOX)
    ax.set_box_aspect((1, 1, 1), zoom=1.15)
    ax.view_init(elev=18, azim=-58)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("X (along-track)", labelpad=-6, fontsize=8)
    ax.set_ylabel("Y (orbit normal)", labelpad=-6, fontsize=8)
    ax.set_zlabel("Z (zenith)", labelpad=-6, fontsize=8)

    th = np.linspace(0.0, 2.0 * np.pi, 120)
    earth = Poly3DCollection([np.column_stack([np.cos(th), np.sin(th), np.full_like(th, -_BOX)])],
                             facecolors="#9ec5e8", edgecolors="#4b7bab", alpha=0.55)
    ax.add_collection3d(earth)
    ax.text(0.0, 0.0, -_BOX, "Earth", color="#2f5f8f", ha="center", va="center", fontsize=8)

    face_cols = ["#c9d3dc"] * 5 + ["#f2a33a"]            # body +z face differs
    body = Poly3DCollection(_prism_faces(np.eye(3)), facecolors=face_cols, edgecolors="#333333",
                            linewidths=0.8, alpha=0.9)
    ax.add_collection3d(body)

    true_lines = [ax.plot([0, 0], [0, 0], [0, 0], color=c, lw=2.2, label=f"body {n} (true)")[0]
                  for c, n in zip(_AXIS_COLORS, _AXIS_NAMES)]
    have_est = data.C_est is not None
    est_lines = []
    if have_est:
        est_lines = [ax.plot([0, 0], [0, 0], [0, 0], color=c, lw=1.0, ls="--",
                             label="estimated axes (dashed)" if n == "x" else None)[0]
                     for c, n in zip(_AXIS_COLORS, _AXIS_NAMES)]
    tip_txt = [ax.text(0, 0, 0, n, color=c, fontsize=9, fontweight="bold")
               for c, n in zip(_AXIS_COLORS, _AXIS_NAMES)]
    nadir_proxy = ax.plot([], [], [], color="black", lw=2, label="nadir")[0]
    sun_proxy = ax.plot([], [], [], color="goldenrod", lw=2, label="Sun")[0]
    leg = ax.legend(loc="upper left", fontsize=7, frameon=True, framealpha=0.85,
                    bbox_to_anchor=(0.0, 0.98))
    leg_texts = {tx.get_text(): tx for tx in leg.get_texts()}
    sun_leg_text = leg_texts["Sun"]
    sun_leg_handle = leg.legend_handles[[tx.get_text() for tx in leg.get_texts()].index("Sun")]

    arrows = {"nadir": None, "sun": None}

    def set_arrow(key, vec, color):
        if arrows[key] is not None:
            arrows[key].remove()
        arrows[key] = ax.quiver(0, 0, 0, *vec, color=color, linewidth=2.2, arrow_length_ratio=0.1)

    # --- text overlay -----------------------------------------------------
    txt = fig.text(0.02, 0.05, "", family="monospace", fontsize=10, va="bottom", ha="left",
                   bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    fig.text(0.02, 0.012, "animation is time-warped (fast phases are sampled densely); "
             "the clock shows simulated time", fontsize=7, color="0.4", ha="left")

    # --- nadir error panel ------------------------------------------------
    err_deg = np.degrees(_f(data.nadir_err))
    err_plot = np.where(err_deg > 0.0, err_deg, np.nan)
    ax2.plot(data.t, err_plot, color="tab:blue", lw=1.2)
    ax2.set_yscale("log")
    fin = err_plot[np.isfinite(err_plot)]
    if fin.size:
        ax2.set_ylim(10 ** np.floor(np.log10(fin.min()) - 0.1), 10 ** np.ceil(np.log10(fin.max()) + 0.1))
    ax2.set_xlim(float(data.t[0]), float(data.t[-1]) if data.t[-1] > data.t[0] else float(data.t[0]) + 1.0)
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("nadir error [deg]")
    ax2.set_title("Nadir pointing error (true attitude)", fontsize=10)
    ax2.grid(True, which="both", alpha=0.3)
    vline = ax2.axvline(float(data.t[0]), color="0.5", lw=0.8)
    marker, = ax2.plot([], [], "o", color="tab:red", ms=7)

    n_frames = len(data.t)

    def draw(i: int):
        C = _f(data.C[i])
        body.set_verts(_prism_faces(C))
        Cd = DISPLAY_FLIP @ C
        for k in range(3):
            true_lines[k].set_data_3d([0.0, Cd[0, k]], [0.0, Cd[1, k]], [0.0, Cd[2, k]])
            tip_txt[k].set_position_3d((1.06 * Cd[0, k], 1.06 * Cd[1, k], 1.06 * Cd[2, k]))
        if have_est:
            Ce = _f(data.C_est[i])
            ok = bool(np.all(np.isfinite(Ce)))
            Ced = DISPLAY_FLIP @ Ce if ok else None
            for k in range(3):
                if ok:
                    est_lines[k].set_data_3d([0.0, Ced[0, k]], [0.0, Ced[1, k]], [0.0, Ced[2, k]])
                else:
                    est_lines[k].set_data_3d([], [], [])
        set_arrow("nadir", 1.3 * np.array([0.0, 0.0, -1.0]), "black")
        lit = bool(data.lit[i])
        sv = 1.3 * (DISPLAY_FLIP @ _f(data.sun[i]))
        set_arrow("sun", sv, "goldenrod" if lit else "grey")
        sun_proxy.set_color("goldenrod" if lit else "grey")
        sun_leg_handle.set_color("goldenrod" if lit else "grey")
        sun_leg_text.set_text("Sun" if lit else "Sun (ECLIPSE)")

        t = float(data.t[i])
        lines = [f"t = {t:9.1f} s   ({t / data.period:5.2f} orbits)",
                 f"mode: {_MODE_NAMES.get(int(data.mode[i]), str(int(data.mode[i])))}",
                 f"nadir error:     {np.degrees(data.nadir_err[i]):8.3f} deg"]
        if data.know_err is not None and np.isfinite(data.know_err[i]):
            lines.append(f"knowledge error: {np.degrees(data.know_err[i]):8.3f} deg")
        txt.set_text("\n".join(lines))
        vline.set_xdata([t, t])
        e = err_plot[i]
        marker.set_data([t] if np.isfinite(e) else [], [e] if np.isfinite(e) else [])

    if ext == ".gif":
        writer = animation.PillowWriter(fps=fps)
    else:
        writer = animation.FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(path), dpi):
        for i in range(n_frames):
            draw(i)
            writer.grab_frame()
        for _ in range(max(fps, 1)):                       # hold the final state for ~1 s
            writer.grab_frame()
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    """Run the closed loop (as ``adcs.simulate.main`` does) and render the animation."""
    p = argparse.ArgumentParser(description="3D attitude animation of the closed-loop ADCS run.")
    p.add_argument("--orbits", type=float, default=6.0)
    p.add_argument("--rate-deg", type=float, default=10.0, help="initial tumble rate (deg/s)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sensors", choices=("realistic", "ideal"), default="realistic")
    p.add_argument("--filter", choices=("design", "g7"), default="design")
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--out", type=str,
                   default=str(Path(__file__).resolve().parent / "report" / "figures"
                               / "closed_loop_animation.mp4"))
    a = p.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    from adcs.simulate import (IDEAL_SENSORS, MEKF_DESIGN, MEKF_G7, REALISTIC_SENSORS, ControlConfig,
                               run_closed_loop, tumble_state)

    P = float(orbital_period(DEFAULT.orbit.a))
    sensors = REALISTIC_SENSORS if a.sensors == "realistic" else IDEAL_SENSORS
    mekf_cfg = MEKF_DESIGN if a.filter == "design" else MEKF_G7
    y0 = tumble_state(np.deg2rad(a.rate_deg), a.seed)
    t0 = time.perf_counter()
    res = run_closed_loop(y0, a.orbits * P, seed=a.seed, ctrl=ControlConfig(), sensors=sensors,
                          mekf_cfg=mekf_cfg)
    t1 = time.perf_counter()
    data = build_animation_data(res, max_frames=a.max_frames)
    out = render(data, a.out, fps=a.fps, dpi=a.dpi,
                 title=f"3U CubeSat closed-loop ADCS (seed {a.seed}, {a.sensors} sensors, {a.filter} filter)")
    t2 = time.perf_counter()
    n = len(data.t)
    span = float(data.t[-1] - data.t[0])
    print(f"wrote {out}")
    print(f"frames: {n} (of {res.t.size} simulated steps), animation length {n / a.fps:.1f} s at {a.fps} fps")
    print(f"time compression: {span / (n / a.fps):.0f} simulated s per animation s (time-warped schedule)")
    print(f"file size: {out.stat().st_size / 1e6:.2f} MB")
    print(f"timing: simulation {t1 - t0:.0f} s, render {t2 - t1:.0f} s")


if __name__ == "__main__":
    main()
