# Spec: closed-loop step-by-step engine (`adcs/stepper.py`, M12)

This file is the contract between `py-implementer` (who writes the module) and `test-author` (who writes the tests from this file, not from the code). Read `CONVENTIONS.md` first: quaternions are scalar-first Hamilton, body-to-ECI; units are SI and radians. Also read `docs/specs/adcs_closed_loop.md` §3, the closed-loop algorithm this module re-hosts.

## 0. Purpose and scope

`run_closed_loop` (`adcs/simulate.py`, Group 8) runs the whole horizon in one call. The browser simulator (Groups 10–11) has to advance it **step by step**, read the state between steps, and inject disturbances. This module moves the body of `run_closed_loop` into a class, `ClosedLoopSim`, without changing any arithmetic.

* **One code path.** `run_closed_loop` becomes a thin wrapper around `ClosedLoopSim` (§4), so `monte_carlo.py`, the report scripts and `visualize.py` keep calling `run_closed_loop` unchanged.
* **Bit-for-bit equivalence.** For the same arguments, the wrapper's result must have the same bytes as the pre-refactor code in every array and every non-array `meta` leaf. The only allowed difference is a new `meta["kicks"]` key, which is `[]` when there were no kicks. The oracle is `adcs/tests/data/closed_loop_golden.npz`. It was captured from the unmodified code by `adcs/tests/golden_closed_loop.py`, which provides `CASES`, `run_case(name, runner)`, `flatten`, and `compare(res, golden, case) -> list[str]`.
* **No new physics.** Every sensor model, the RNG stream order (`default_rng(seed).spawn(4)`, then `rng_truth.spawn(3)`), the precomputation, the order of operations in a period, the logging and `meta` must be kept exactly. Moving lines is allowed; reordering floating-point operations is **not** (e.g. `a + b + c` must not become `a + (b + c)`).
* **The horizon is fixed at construction.** The ephemeris, field tables, Gauss-Markov truth series, albedo series, Sun and shadow flags are precomputed over `[0, t_end]`, exactly as now. A simulation cannot be extended past `t_end`; the server builds a new one instead.

## 1. Time indexing (the one thing to get right)

* `T = mtq.control_period`, `n_steps = int(round(t_end / T))`, `t_grid = arange(n_steps + 1) * T`.
* **Sample `k`** (`0 ≤ k ≤ n_steps`) is the work the current loop does at `t_k = t_grid[k]`:
  * sensors → estimator → mode machine → control → logging (if `k % log_every == 0`);
  * then, if `k < n_steps`, the RK4 on-phase plus off-phase integration of the truth from `t_k` to `t_{k+1}`.
* A new sim has processed no samples. Its **pending truth state** `y` is `y0` at `t_0 = 0`, with the quaternion normalised as now.
* After processing sample `k` (`k < n_steps`), the pending state is `y` at `t_{k+1}`. After processing the last sample `k = n_steps`, the sim is **done**: there is no integration, and the pending state stays at `t_end`.
* There are `n_steps + 1` samples in total. A full run is exactly the current `for k in range(n_steps + 1)` loop.

## 2. API

`adcs/stepper.py`, with `__all__ = ["ClosedLoopSim"]`. It imports the result dataclass, the presets and the private helpers it needs from `adcs.simulate`. To avoid an import cycle, `adcs.simulate.run_closed_loop` imports `ClosedLoopSim` lazily, inside the function. If the implementer prefers the reverse layout (helpers moved to `adcs/stepper.py` and re-exported from `adcs.simulate`), that is also acceptable, as long as every name in `adcs.simulate.__all__` stays importable from `adcs.simulate`.

```python
class ClosedLoopSim:
    def __init__(self, y0, t_end, seed=0, cfg=DEFAULT, ctrl=ControlConfig(), mtq=MagnetorquerConfig(),
                 wheel=WheelConfig(), sensors=IDEAL_SENSORS, mekf_cfg=MEKF_G7, modes=ModeConfig(),
                 mag_cfg=MagnetometerConfig(), css_cfg=SunSensorConfig(), gyro_cfg=GyroConfig(),
                 inertia_true=None, log_every=1): ...
```

The constructor takes the same parameter names, order and defaults as `run_closed_loop`. It does all the precomputation `run_closed_loop` does before its loop, and initialises all loop state. It processes **no** samples. Errors:
* `ValueError("t_end shorter than one control period")` if `n_steps < 1`, as now;
* `ValueError` if `log_every < 1`.

Read-only attributes (properties or plain attributes; callers must not modify them):

| Name | Type | Meaning |
|---|---|---|
| `T` | float | control period, s |
| `n_steps` | int | number of integration periods (samples = `n_steps + 1`) |
| `t_end` | float | `t_grid[-1]`, s |
| `period` | float | `orbital_period(cfg.orbit.a)`, s |
| `k` | int | index of the **next** sample to process: 0 for a new sim, `n_steps + 1` when done |
| `t` | float | time of the pending truth state, `t_grid[min(k, n_steps)]`, s |
| `done` | bool | `k > n_steps` |
| `remaining` | int | `n_steps + 1 - k` |
| `state` | ndarray (10,) | a **copy** of the pending truth state `[q, w, h_w]` |
| `t_grid`, `r_eci`, `v_eci`, `sun_eci`, `lit` | ndarrays (n_steps+1, …) | the precomputed truth environment on `t_grid`: position m, velocity m/s, unit Sun direction, shadow flag. These are the arrays the loop uses (`r_all`, `v_all`, `s_all`, `lit_all`). The server uses them to draw the orbit and eclipse intervals. |

### `step(n: int = 1) -> int`

Processes up to `n` samples (§1) and returns how many it processed, `min(n, remaining)`.
* `n` must be an `int` with `n ≥ 1` (`bool` is rejected), else `ValueError`.
* If the sim is already done, raises `RuntimeError("horizon reached")`.
* `step(remaining)` from a new sim is a full run.
* Splitting the samples across calls in any way, e.g. `step(1)` 7201 times or `step(5000)` then `step(2201)`, gives bit-identical results.

### `kick(dw_body) -> None`

An instantaneous external angular impulse on the **truth**:
* `y = y.copy(); y[4:7] = y[4:7] + dw`, where `dw = np.asarray(dw_body, float)` has shape (3,) and is in rad/s, body axes.
* `q` and `h_w` (total wheel momentum, body axes) are unchanged. The impulse is therefore `ΔL = I_true Δω`, and the wheels' speeds relative to the body change by `−a·Δω`. The impulse is so short that the bearings transmit no axial torque.
* It applies to the pending state at time `t`. The next sample processed is `k`, at `t_k = t`. Its gyro reading is the mean rate over `[t_{k−1}, t_k]` and does **not** see the kick. Its controller output is computed from that reading. The integration from `t_k` starts from the kicked rate, so gyro sample `k+1` is the first to see it.
* Several kicks before the same sample add up.
* Records `{"t": float(t), "k": int(k), "dw": [float, float, float]}` in a list that `result().meta["kicks"]` returns.
* Errors:
  * `ValueError` if `dw` is not shape (3,) or not finite;
  * `ValueError` if `‖w + dw‖ > MAX_KICK_RATE` (module constant `np.deg2rad(30.0)` rad/s, the same as the server's initial-rate cap). The reason: RK4 with the 0.8 s + 0.2 s sub-steps loses energy non-physically at high rates. Over 600 s torque-free, ΔE/E is −1.2e-4 at 0.5 rad/s, −7.8e-3 at 1 rad/s and −0.38 at 2 rad/s. Also, the gyro's mean rate `to_rotvec(q_prev* q)/T` wraps above π rad/s. The state is unchanged after a rejected kick. This is recorded in `assumptions.md` [B-25].
  * `RuntimeError("horizon reached")` if the sim is done, **or** if `k == n_steps`. In the second case no integration is left, so the kick could only change the last logged `w`.
* A kick before the first sample is allowed. It acts like a modified `y0` but is still recorded.

### Failure latch

* If any exception escapes the per-sample loop inside `step()`, the sim sets `failed = True` (a read-only bool property, `False` otherwise) and re-raises.
* After that, `step`, `kick` and `snapshot` raise `RuntimeError("simulation failed; build a new one")`. `result()` still works and returns whatever was logged.
* Reason: the loop advances the RNG streams, the MEKF and the logs before it commits `k`, so a retry would replay samples on streams that have already advanced.
* Argument validation errors (`ValueError` for a bad `n` or `dw`) are raised **before** any state is touched, and do not latch.

### `snapshot() -> dict`

Describes the **most recently processed sample** `k_last = k − 1`. It raises `RuntimeError` if no sample has been processed yet.

All values are copies: numpy arrays (float64) or Python scalars. Before MEKF initialisation, estimator fields are `None` (vectors) or `nan` (scalars). The values must equal what the logger writes for that sample: same formulas, same bytes. So for `log_every = 1`, `snapshot()` after each `step(1)` matches row `k_last` of `result()`.

| Key | Shape/type | Definition (as in the logger) |
|---|---|---|
| `k` | int | `k_last` |
| `t` | float | `t_grid[k_last]`, s |
| `q`, `w`, `h_w` | (4,), (3,), (3,) | truth state at `t_k` (before that sample's integration) |
| `wheel_speed` | (3,) | `wheel_speed(h, w, wheel)`, rad/s |
| `rate` | float | `‖w‖` |
| `m`, `u_w` | (3,) | dipole A·m² and wheel torque N·m commanded for the period starting at `t_k` |
| `tau_dist`, `tau_mag` | (3,) | as logged |
| `mode` | int | `MODE_DETUMBLE` (0) or `MODE_POINTING` (1) after the mode machine at `t_k` |
| `dump` | bool | dumping flag at `t_k` |
| `lit` | bool | truth shadow flag at `t_k` |
| `sun_used` | int | as logged (`used`) |
| `r_eci`, `v_eci` | (3,) | m, m/s, `r_all[k]`, `v_all[k]` |
| `sun_eci` | (3,) | `s_all[k]` (unit) |
| `b_eci` | (3,) | truth field incl. the Gauss-Markov residual, `b_all[k]`, T |
| `q_ref` | (4,) | nadir reference `qref_all[k]` |
| `gmst` | float | `common.frames.gmst(cfg.jd_epoch + t / 86400.0)`, rad (Earth rotation angle for display) |
| `nadir_err` | float | as logged |
| `filter_ready` | bool | the MEKF exists (TRIAD has initialised) |
| `q_est`, `b_est`, `bm_est` | (4,), (3,), (3,) or `None` | `q_hat`, `b_hat`, `bm_hat` |
| `w_est` | (3,) or `None` | `w_hat` (raw gyro before init; `None` at `k = 0`) |
| `know_err` | (3,) or `None` | `quat.error_rotvec(q_hat, q)` |
| `att_sigma` | float | `f.att_sigma`, or `nan` |
| `P_att` | (3, 3) or `None` | `f.ekf._P[:3, :3]` (copy) |
| `est_nadir_err` | float | as logged, or `nan` |
| `t_init` | float | time of TRIAD initialisation, or `nan` |
| `switch_times`, `fallback_times` | list[float] | copies of the event lists so far |
| `n_kicks` | int | kicks recorded so far |

**Performance rule:** `step()` must not compute any snapshot-only quantity per sample. Store references to that sample's already-computed values. Every derived value (`know_err`, `gmst`, `P_att` copy, …) is computed inside `snapshot()`. The per-sample cost of a full run must stay within 5 % of the pre-refactor wall time. The golden capture records the pre-refactor rate.

Aliasing: the stored references must not change after the sample is processed. This holds because `rk4_step` returns a new array, `kick` copies `y` first, and the MEKF does not mutate `q_hat` in place. If the implementer finds any in-place mutation, they copy at storage time and report it.

### `result() -> ClosedLoopResult`

It can be called at any time, including mid-run, and more than once. It returns the logged rows so far (`[0:j]`) and `meta`, built exactly as `run_closed_loop` builds them now, plus `"kicks"` (a copy of the kick list). `meta["t_end"]` is the horizon `t_grid[-1]` even mid-run. `meta["t_init"]` is `nan` if TRIAD has not run. The event lists are copies. Later `step()` calls must not change a result that has already been returned.

## 3. What must not change

* The constructor's precomputation and the per-sample sequence. The current code at `adcs/simulate.py:378–574` is the reference; moving it into methods is the whole task.
* `ClosedLoopResult`, `SimResult`, `summarize_closed_loop`, `run`, the presets, `tumble_state` and the CLI keep their behaviour.
* The loop's local `def ext_on(...)` closure binds `m`, `b_eci` and `tau_d` per sample. Keep that binding, so the closure sees the values of the sample being integrated.

## 4. `run_closed_loop` wrapper

```python
def run_closed_loop(y0, t_end, seed=0, ..., inertia_true=None, log_every=1) -> ClosedLoopResult:
    """(docstring unchanged, plus one line: implemented by adcs.stepper.ClosedLoopSim)"""
    from adcs.stepper import ClosedLoopSim
    sim = ClosedLoopSim(y0, t_end, seed=seed, cfg=cfg, ...)   # every argument forwarded by keyword
    sim.step(sim.remaining)
    return sim.result()
```

## 5. Tests (`adcs/tests/test_stepper.py`)

Write these from this spec. The oracles are the golden file (pre-refactor code) and exact algebra.

1. **Golden equivalence.** For every case in `golden_closed_loop.CASES`, `compare(run_case(name), golden, name) == []` with the default runner (the wrapper).
   * Also with a runner that drives `ClosedLoopSim` in **irregular chunks**, e.g. step sizes cycling 1, 7, 333, 1000, until done.
   * `meta["kicks"] == []` in both.
   * The golden file must be loaded with `allow_pickle=False`.
   * Runtime: these are the slow tests. Mark the 2-orbit cases `@pytest.mark.slow` only if the repo already defines that marker; otherwise run them all (about 10 runs).
2. **Snapshot = log row.**
   * A `log_every=1`, 600 s realistic+design run, stepped one sample at a time. After each step, every snapshot key that has a logged counterpart equals the row bit for bit, including NaN rows before init (snapshot `None` ↔ row all-NaN). The counterparts are `t q w h_w wheel_speed m u_w tau_dist tau_mag mode dump lit sun_used nadir_err q_est b_est bm_est know_err w_est att_sigma est_nadir_err`, plus `P_att` ↔ `P_att` row and `rate` ↔ `rate`.
   * `r_eci`, `v_eci`, `sun_eci` and `lit` equal `sim.r_eci[k]` etc.
   * `gmst` equals `common.frames.gmst(DEFAULT.jd_epoch + t/86400)`.
   * The quaternion fields are unit to 1e-12.
3. **Indexing.**
   * New sim: `k == 0`, `t == 0`, `remaining == n_steps + 1`, `done` false, `snapshot()` raises `RuntimeError`, and `state` equals `y0` with the quaternion normalised.
   * After `step(1)`: `k == 1`, `t == T`, `snapshot()["t"] == 0`.
   * `step(10**9)` returns the remaining count, after which `done` is true and `t == t_end`.
   * A further `step()` or `kick()` raises `RuntimeError`.
   * `step(0)`, `step(-1)`, `step(1.5)` and `step(True)` raise `ValueError`.
   * `t_end < T` raises `ValueError`.
4. **`state` is a copy:** mutating it does not change the sim.
5. **Kick = exact Δω.**
   * `s0 = sim.state`; `sim.kick(dw)`; `s1 = sim.state`. Then `s1[4:7] == s0[4:7] + dw` exactly (`array_equal`), and `s1[0:4] == s0[0:4]`, `s1[7:10] == s0[7:10]` exactly.
   * Two kicks add up.
   * `meta["kicks"]` records `t`, `k` and `dw`.
   * Bad shape or NaN raises `ValueError`.
6. **Kick physics (independent oracle).**
   * The total inertial angular momentum `H = R(q)(I w + h_w)` jumps by exactly `R(q) I_true Δω` across the kick. Check to 1e-15 relative using the `state` before and after; `I_true` is `cfg.spacecraft.inertia` or `inertia_true`.
   * The wheel speed relative to the body, `wheel_speed(h, w, wheel)`, changes by `−(a·Δω)` per wheel, using the `WheelConfig` axis definition.
   * The next snapshot's `w` equals the kicked rate (the sample after the kick logs the truth state at `t_k`).
7. **Kick response.** In a realistic+design run that has switched to POINTING, so it is 2 orbits long, starting from a pointing state:
   * Pick a pointing state cheaply. Option one: the golden `real_design_s0` case, stepped to just after `meta["switch_time"] + 600 s`. Option two: a shorter run started near LVLH with `lvlh_state`, in which case the test documents how many seconds it simulates.
   * A kick of 5 °/s about body x makes the mode machine fall back to DETUMBLE within `modes.safe_hold + 2 T` seconds of the kick.
   * `fallback_times` gains an entry ≥ the kick time.
   * The run without the kick, stepped over the same samples, has no fallback.
8. **`result()` mid-run.**
   * After `step(100)` with `log_every=10`: `len(res.t) == 10`, and `res.t` equals `arange(10)*10*T`.
   * A second `result()` after more steps is longer, and the first one is unchanged (keep a copy and compare).
   * `result()` on a new sim has zero rows, and `meta["t_init"]` is nan.
9. **Wrapper identity.** `run_closed_loop(**kw)` equals `ClosedLoopSim(**kw)` followed by `step(remaining)` and `result()`, per `compare`, for one short case.

10. **Golden portability.** The golden file is bit-exact only on its capture platform. `golden_closed_loop.environment_matches(golden) -> (bool, reason)` compares `info_json` (python, numpy, scipy versions, `platform.machine()`) with the running interpreter. The equivalence tests call `pytest.skip(reason)` on a mismatch, so the reason is explicit. Re-capture **only** from pre-refactor code: the HEAD-at-capture module, `git show 137bc1d:adcs/simulate.py`.
11. **Fallback and dumping coverage.** Golden case `lvlh_wheels_fallback` starts near LVLH with wheels pre-loaded and a low `safe_rate`. It was captured from the HEAD-at-capture `run_closed_loop`, so it covers the fallback and dumping branches the other cases never reach.
12. **Kick limits and latch.**
    * A kick that would make `‖w+dw‖` exceed `MAX_KICK_RATE` raises `ValueError` and leaves `state` unchanged.
    * A kick at `k == n_steps` raises `RuntimeError`.
    * Failure latch: monkeypatch a dependency to raise mid-chunk, e.g. on sample 5 of `step(10)`. Then `failed` is true, `step`, `kick` and `snapshot` raise `RuntimeError`, and `result()` works.

Existing tests (`test_closed_loop.py`, `test_monte_carlo.py`, `test_visualize.py`, …) must pass **unchanged**.
