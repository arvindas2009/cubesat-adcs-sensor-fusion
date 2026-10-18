# Spec — 3D attitude visualizer (`adcs/visualize.py`, M11)

Contract between `py-implementer` (module) and `test-author` (tests, written from this file, not from the code).
Read `CONVENTIONS.md` first (quaternions scalar-first Hamilton, body-to-reference; LVLH definition in §4).

## 0. Purpose and scope

Turn a closed-loop result (`adcs.simulate.ClosedLoopResult`, also any `SimResult`) into a matplotlib 3D animation (MP4 and/or GIF) that shows:

* the 3U body (10 × 10 × 34 cm prism, drawn to scale) and its three body axes (x red, y green, z blue, from the **true** attitude);
* the **estimated** body axes (thin dashed, same colours) when the result has `q_est` (NaN before MEKF initialisation → not drawn);
* the **nadir** direction (arrow to the Earth centre) and a small Earth disc below as a visual anchor;
* the **Sun** direction (arrow; grey and labelled `ECLIPSE` while the truth shadow flag says the satellite is in shadow);
* text: time `t` [s] and [orbits], mode (`DETUMBLE` / `POINTING`), nadir error [deg], knowledge error [deg] (if available);
* a second panel: nadir error [deg] vs time (log y axis, all frames) with a moving marker at the current frame.

It is a visual aid only: it adds no numbers to the reports. It must not modify any other module. It depends only on numpy, matplotlib (Pillow comes with it) and the repo's `common`/`adcs` modules; MP4 output additionally needs `ffmpeg` on `PATH`.

## 1. View frame

All geometry is drawn in the **display frame** = LVLH rotated by 180° about its x axis, so that zenith is up:

`p_display = diag(1, −1, −1) · p_lvlh`  (a proper rotation, det = +1)

* display X = LVLH x (≈ along-track), display Y = −LVLH y (= +orbit normal), display Z = −LVLH z (zenith, **up**).
* Nadir is therefore the fixed vector `(0, 0, −1)` in the display frame.
* With perfect nadir pointing (`q = q_ref`) body axes coincide with LVLH axes, so body z (the long axis) points **down** (display `(0, 0, −1)`), body x along display X.

## 2. API (module `adcs/visualize.py`, `__all__` lists these)

All arrays: time on axis 0, SI units, radians; quaternions scalar-first body-to-ECI.

```python
DISPLAY_FLIP = np.diag([1.0, -1.0, -1.0])    # LVLH -> display

def body_dcm_lvlh(q, r_eci, v_eci) -> np.ndarray:
    """(..., 3, 3) matrices C with C[..., :, k] = body axis k expressed in LVLH axes (= R(q_lvlh)^T R(q))."""

def sun_lvlh(s_eci, r_eci, v_eci) -> np.ndarray:
    """(..., 3) unit Sun vector in LVLH axes."""

def frame_schedule(t, C, sun, max_frames=600, dt_min=1.0, dt_max=60.0) -> np.ndarray:
    """Sorted integer indices into the (N,) time grid for the animation frames."""

@dataclass
class AnimationData:
    t: (M,)  s;  C: (M,3,3) true body axes in LVLH;  sun: (M,3) LVLH;  lit: (M,) bool;
    mode: (M,) int;  nadir_err: (M,) rad;
    C_est: (M,3,3) | None;  know_err: (M,) rad | None;  period: float   # orbital period [s] for the "orbits" label

def build_animation_data(res, cfg=DEFAULT, max_frames=600, dt_min=1.0, dt_max=60.0) -> AnimationData:
    """Pick frames with ``frame_schedule`` and evaluate everything on them."""

def render(data: AnimationData, path, fps=25, dpi=100, title=None) -> Path:
    """Write the animation; format from the extension (``.mp4`` via ffmpeg, ``.gif`` via Pillow)."""

def main(argv=None) -> None   # CLI, see §5
```

### `body_dcm_lvlh`

`C = R(q_ref)^T R(q)` with `q_ref, _ = adcs.control.nadir_reference(r, v)` (body-aligned-with-LVLH quaternion). Equivalently `C = dcm_eci_to_lvlh(r, v) @ R(q)` (`common.frames.dcm_eci_to_lvlh`). If `q = q_ref ⊗ exp(φ)` then `C = R(exp(φ))`. Quaternions are normalised on entry; NaN rows give NaN matrices (no exception).

### `sun_lvlh`

`s_lvlh = dcm_eci_to_lvlh(r, v) @ s_eci`, normalised.

### `frame_schedule`

* `t` (N,) strictly increasing seconds, `C` (N, 3, 3), `sun` (N, 3) (all on the full simulation grid).
* Per-step **motion** `m_i = ∠(C_i, C_{i+1}) + ∠(sun_i, sun_{i+1})` [rad], with `∠(C_i, C_{i+1}) = |to_rotvec(C_iᵀ C_{i+1})|` (rotation angle of the relative body rotation as seen in LVLH) and `∠(sun_i, sun_{i+1}) = arccos(clip(sun_i · sun_{i+1}, −1, 1))`. NaN motion counts as 0.
* Algorithm: for a trial target angle `a`, walk from index 0; emit the next frame at the first index `j` such that either the accumulated motion since the last emitted frame ≥ `a` and `t_j − t_last ≥ dt_min`, or `t_j − t_last ≥ dt_max`; always emit index 0 and index N−1. Find (by bisection on `a` in `[1°, 90°]`, ≥ 30 iterations, monotone: larger `a` ⇒ fewer frames) the **smallest** `a` such that the number of frames ≤ `max_frames`. If even `a = 90°` gives more than `max_frames` frames, subsample the `a = 90°` schedule uniformly by index (keeping first and last) to `max_frames`.
* Guarantees (tested): returns strictly increasing integer indices, first = 0, last = N−1, length ≤ `max_frames` (≥ 2 when N ≥ 2); for N ≤ `max_frames` and a motion large enough that every step exceeds `a`, all indices may be returned; consecutive frames never exceed `dt_max` apart unless the `max_frames` subsampling was needed; for constant-rate rotation and constant `sun` the frames are uniform in time (± one sample).
* The animation is therefore **time-warped** (fast phases are sampled more densely in time); the on-screen text always shows the true simulation time.

### `render`

* Matplotlib is used headless (`matplotlib.use("Agg")` inside `render`/`main` before creating a figure; no state leaks into import time of the module).
* Figure ≈ 12 × 6 in at `dpi` (even pixel dimensions — required by h264), left 3D axes (≈ 60 % width), right panel nadir error (log y, deg). Axis limits fixed (no auto-scale jumps): 3D box ±1.6 with equal aspect.
* Geometry units: body drawn with its long axis (34 cm) = 1.2 plot units (so 10 cm = 0.353); body axes arrows length 1.0 from the body centre (origin); estimated axes the same length, dashed, linewidth 1; nadir arrow length 1.3 along `(0, 0, −1)`; Sun arrow length 1.3 along `diag(1,−1,−1) · sun`; Earth disc: filled circle radius 1.0 in the plane `Z = −1.6`.
* Body prism: 6 faces (`Poly3DCollection`), vertices `R(C) · diag(0.353, 0.353, 1.2)/2 · (±1, ±1, ±1)` mapped with `DISPLAY_FLIP` (body +z face coloured differently from the others so spin about the long axis is visible).
* Text overlay per frame as listed in §0; knowledge error shown only if `know_err` is given and finite. When `lit` is false the Sun arrow is grey and the label reads `Sun (ECLIPSE)`.
* Writers: `.gif` → `matplotlib.animation.PillowWriter(fps)`; `.mp4` → `FFMpegWriter(fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])`; if `ffmpeg` is not found for `.mp4`, raise `RuntimeError` with a message naming ffmpeg and suggesting `.gif`; any other extension → `ValueError`. Parent directories are created. Returns the `Path`.
* GIF loops forever; the last frame is held for ≥ 1 s (repeat the final frame `fps` times or equivalent) so the final state can be read.

### `build_animation_data`

Uses `res.t`, `res.q`, `res.lit` (if the result has no `lit` attribute — i.e. a `SimResult` — compute it from the ephemeris and Sun), `res.mode`, `res.nadir_err`, and when present `res.q_est`, `res.know_err` (**angle** of `know_err` rows, `‖know_err‖`). `r, v` from `Ephemeris(*initial_state(cfg), t[-1] + 10)`, Sun from `adcs.environment.sun_direction_eci(cfg.jd_epoch + t/86400)`. `period = orbital_period(cfg.orbit.a)`. The full-grid `C`/`sun` are evaluated first for the schedule; `AnimationData` then holds only the selected frames.

## 3. Numerical behaviour

* No input array is modified; no global RNG use; deterministic output for identical inputs.
* `C_est` rows with NaN `q_est` stay NaN and are skipped by `render` (axes not drawn for that frame).
* Memory: the full simulation grid (≤ ~40 000 steps) is processed vectorised; no per-step Python loops over the full grid except inside `frame_schedule` (a loop over steps is acceptable; N ≈ 4·10⁴ and ≈ 30 bisection iterations must finish in a few seconds — precompute `m_i` and its cumulative sum once, then each trial is a loop or searchsorted walk).

## 4. Tests (written by `test-author` from this spec) — `adcs/tests/test_visualize.py`

Independent oracles: scipy `Rotation`, construction of LVLH axes from the definition in `CONVENTIONS.md` §4 (not via `dcm_eci_to_lvlh`).

1. `body_dcm_lvlh`: `q = q_ref` → identity (1e-12); `q = q_ref ⊗ exp(φ)` → `Rotation.from_rotvec(φ).as_matrix()`; batched shapes; NaN in → NaN out; orthonormal, det +1 for random q.
2. `sun_lvlh`: equals `[s·x̂_lvlh, s·ŷ_lvlh, s·ẑ_lvlh]` with the axes built from `r, v` per CONVENTIONS §4; unit norm.
3. Display frame: `DISPLAY_FLIP` has det +1; with `C = I` the body z axis in the display frame is `(0, 0, −1)` (down) and the nadir arrow direction equals it.
4. `frame_schedule`: all guarantees of §2 (first/last, strictly increasing, ≤ `max_frames`, `dt_max` respected, uniform for constant-rate rotation, more frames in a fast segment than in a slow one of equal duration, monotone in `max_frames`).
5. `render` smoke on a small synthetic `AnimationData` (≈ 12 frames, with and without `C_est`/`know_err`, with a lit→eclipse change): GIF written to `tmp_path`, `PIL.Image.n_frames ≥ 12`, size > 0; MP4 written (skip if no `ffmpeg`), size > 0; `.avi` → `ValueError`; missing ffmpeg path (monkeypatch `shutil.which`) → `RuntimeError`.
6. `build_animation_data` on a short result (e.g. a 300 s perfect-knowledge `adcs.simulate.run` pointing run and a ≈ 300 s `run_closed_loop`): shapes consistent, `C ≈ I` within the nadir error for perfect pointing, `lit` present for both result types.
7. Determinism: two `render` calls on the same data give GIFs with identical frame count.

Keep the whole file under ~30 s.

## 5. CLI

`python -m adcs.visualize [--orbits 6] [--rate-deg 10] [--seed 0] [--sensors realistic|ideal] [--filter design|g7] [--max-frames 600] [--fps 25] [--dpi 100] [--out PATH]`

Runs `run_closed_loop` exactly as `adcs.simulate.main` does for `--knowledge estimated` (same defaults: `tumble_state(deg2rad(rate), seed)`, `REALISTIC_SENSORS`, `MEKF_DESIGN`, `ControlConfig()`), builds the animation data and renders to `--out` (default `adcs/report/figures/closed_loop_animation.mp4`; use the `.gif` extension for GIF). Prints the output path, number of frames, the schedule's time compression (simulated seconds per animation second) and the file size. Importing `adcs.visualize` must not run a simulation or import `ffmpeg`.
