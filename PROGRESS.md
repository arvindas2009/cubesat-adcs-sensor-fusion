# Progress

Read this file and `docs/IMPLEMENTATION_PLAN.md` at the start of every group. Updated by the main session at every group exit.

| Group | Milestones | Status |
|---|---|---|
| 1 Foundations | M0 | **done** |
| 2 A data + dead reckoning | M1, M2 | **done** |
| 3 A 15-state EKF | M3 | **done** |
| 4 A tuning + writeup | M4 | **done** |
| 5 B physics | M5, M6 | next |
| 6 B control | M7 | not started |
| 7 B estimation | M8 | not started |
| 8 B closed loop + MC + doc | M9 | not started |
| 9 Packaging + visualizer | M10, M11 | not started |

## Group 1 — Foundations (M0) — done

**Delivered**
- Repo scaffold: git, `.gitignore`, `.gitattributes` (LF), `pyproject.toml` (editable install), `requirements.txt`, packages `common/`, `sensor_fusion/`, `adcs/`.
- Environment: `.venv` on **Python 3.14.2** — all deps have 3.14 wheels (no 3.12 fallback needed). numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2, pytest 9.1.1, pyulog 1.2.4, ppigrf 2.1.0 (pulls pandas 3.0.6), sgp4 2.27, requests 2.34.2.
- `CONVENTIONS.md`, `docs/specs/common_api.md`, skeleton `README.md`, `assumptions.md` (G-01…G-09 filled).
- `common/quaternion.py` (+ `inject_attitude_error`, `attitude_reset_jacobian` — the single shared ES-EKF injection/reset), `common/frames.py`, `common/ekf.py` (generic ES-EKF core: Joseph form, block + sequential updates, gating, NIS log, Van Loan, chi² bounds).
- Tests: 445 passing (`pytest`, ~10 s). Quaternion verified against scipy `Rotation`; frames against closed-form WGS-84 and Vallado GMST examples; EKF against a textbook KF, DARE steady state, analytic Van Loan, Monte-Carlo NIS/NEES, nonlinear pendulum.
- Subagents defined in `.claude/agents/` (py-implementer, test-author, data-wrangler, figures-runner, doc-writer = sonnet; math-reviewer = opus).

**★ Opus math review:** no blockers. A mutation test (26 deliberate bugs) was caught 22/26; the 4 escapes are float-equivalent or immaterial. Follow-ups applied: Joseph-form test now actually discriminates vs short form; shared attitude inject/reset helpers; LVLH/NED DCM ↔ quaternion link tests (`from_dcm(C.T)` is the body-aligned quaternion); ragged `innovations()` logs with `dof`; ES-EKF contract (inject after *every* update) in `ekf.py` docstring.

**Notes for later groups**
- Custom agents in `.claude/agents/` only load at session start. In the session that created them they were run as general-purpose agents with the model set (sonnet/opus) and told to follow the agent file; from a fresh session use them by name.
- `common/plotting.py` deferred to Group 2 (owned by figures-runner).
- ppigrf 2.1.0: `ppigrf.igrf(lon_deg, lat_deg, h_km, datetime)` → `(Be, Bn, Bu)` in **nT**, local geodetic **ENU**; scalar input returns shape `(1,)` arrays (use `[0]`). Needed in Group 5.
- sgp4: `Satrec.twoline2rv` + `sat.sgp4(jd, fr)` → TEME km, km/s (convert to m).
- Group 3 decision pending (from review): PX4 GPS arrives ~100–200 ms delayed — buffer-and-replay vs ignore + inflate R. `van_loan` costs ~160 µs/step at n=15 (fine; first-order discretisation optional).
- GMST oracle (Vallado Ex. 3-5) matches to 7.6e-8°; tests use 1e-6°.

## Group 2 — Project A data + dead reckoning (M1, M2) — done

**Delivered**
- `sensor_fusion/download_logs.py` (3 pinned Flight Review logs, retries, ULog-magic check), `loader.py` (pyulog → time-aligned SI arrays; `FlightLog` with `imu/mag/baro/gps/ekf2/info`; cleaning, gap report, armed-window crop), `interp.py` (`interp_to`, slerp for quaternions), `inspect_logs.py` (+ `--smoke`), `common/plotting.py`, 15 inspection figures, `sensor_fusion/data/NOTES.md`.
- `sensor_fusion/dead_reckoning.py`: `strapdown` (trapezoid gyro, mid-interval attitude for accel rotation, trapezoid v/p; vectorised prefix-scan quaternion product), `initial_attitude_from_accel_mag`.
- `common/frames.py::normal_gravity` (Somigliana + free-air) added after the ★ review.
- `CONVENTIONS.md` §6 finalised (PX4 → repo mapping; three height datums; local gravity; EKF2 = reference not truth); `assumptions.md` A-07…A-12; `docs/specs/sensor_fusion_data.md`.
- Tests: **605 passing** (`pytest`, ~17 s) — new: 34 dead-reckoning (scipy/analytic oracles, order-2 convergence), ~94 loader/download (checked against raw pyulog reads, ppigrf inclination/declination, gyro-vs-EKF2 attitude rate, accel-vs-gravity), 11 `normal_gravity`.

**Pinned logs** (all gitignored `.ulg`, 46 MB in `sensor_fusion/data/`)

| ID | Role | HW / FW | Window | IMU rate |
|---|---|---|---|---|
| `f4e81624-7ccc-4859-bb45-d02fa0246b77` | hover (slow position-mode flight) | FMU_V6C, v1.17.0 | 280 s | 221.7 Hz |
| `7cd0a812-d6d6-4c67-9da8-1f3a9cc460b0` | aggressive (stabilized, gyro to 386 °/s, ends in impact) | FMU_V6C, v1.17.0 | 169 s | 221.6 Hz |
| `78808b7c-d9e9-4350-b311-3bb510f29a7b` | other (long, 9 m/s, RTK GPS 2 Hz, old int message format) | FMU_V5, v1.14.0 | 674 s | 199.4 Hz |

Other stream rates: mag ≈ 2 Hz, baro ≈ 5–6 Hz, GPS 2–10 Hz, EKF2 attitude 20 Hz / position 10 Hz.

**★ Opus review (PX4 mapping)** — core mapping confirmed independently (R(q_EKF2)·f_b ≈ −g down; gyro vs EKF2 attitude rate rms 0.013–0.022 rad/s, zero lag; GPS vel sign; GPS→NED vs EKF2 0.05–0.3 m rms). Findings fixed in this group: red baro test (EKF2 height is GPS-referenced on 2 of 3 logs and GPS height drifts ~9 m in the aggressive log → baro test criterion changed per spec §6), wrong EKF2-origin claim in NOTES, loader dropped GPS MSL height (`gps.alt_msl` added), per-log local gravity (G0 would add up to 0.028 m/s²), mag/baro now stamped with `timestamp_sample`, GPS latency is measurable (≈0.10–0.18 s; `EKF2_GPS_DELAY` = 110 ms).

**Measured dead-reckoning drift** (raw IMU, no bias removal, seeded from EKF2 at t=0, local g; EKF2 is the reference, not ground truth; `report/dr_metrics_<id>.json`)

| Log | Pos err @10 s | @30 s | @60 s | Att err @10 s | @60 s | Growth exponent (2–30 s) |
|---|---|---|---|---|---|---|
| hover | 10.8 m | 97 m | 695 m | 1.1° | 13.3° | 1.93 |
| aggressive | 9.1 m | 109 m | 320 m | 1.5° | 5.3° | 2.20 |
| other | 2.4 m | 49.7 m | 256 m | 1.0° | 7.7° | 2.77 |

Whole-flight end errors reach 3 km – 172 km (e.g. hover 78 km after 280 s). Re-seeding after the EKF2 yaw reset (`post_reset` in the JSON) gives similar growth (exponent 1.74–2.14). Using G0 instead of local g shifts position by exactly ½|Δg|t² (hover 51 m at 60 s) — real but small next to bias-driven error. Exponents near 2 point to accel-bias-dominated error and 2.77 on the long log to gyro-bias gravity coupling; **the split was not tested** — Group 3 estimates the biases.

**Open decisions / notes for Group 3**
- Mag: fuse **tilt-compensated heading only**, σψ ≈ 3–5° (measured heading offset vs IGRF 1.2–6.6°, likely uncorrected hard-iron on the hover log; IGRF date from `time_utc_usec`). Mag is only ~2 Hz in the default log profile.
- Baro: use as height change vs mean of the first 1 s on the ground, σ ≈ 1.5 m (to be confirmed by NIS); baro offset drifts by metres and its scale differs from GPS by ~10 %.
- GPS delay: re-stamp as `t − 0.11 s` and process in time order (offline, exact); sensitivity sweep τ ∈ {0, 0.11, 0.2} s. (This resolves the Group 1 buffer-and-replay vs inflate-R question.)
- Local gravity per log via `normal_gravity` (9.7784 / 9.7859 / 9.8119 m/s²); NED origin = first valid GPS fix (ellipsoidal height); compare with EKF2 using the constant origin offset (CONVENTIONS §6).
- Each log has one EKF2 pure-yaw quaternion reset (−4.92° @16.1 s, −0.98° @4.2 s, −3.53° @7.4 s): mark or exclude in filter-vs-EKF2 yaw statistics.
- Aggressive log: baro vs GPS 10 s-increment correlation is only r = 0.34 (non-overlapping windows) because of GPS vertical drift; and the log ends in an impact at t ≈ 167 s.
- `strapdown` gyro scheme (trapezoid) carries a ≈ dt/2 lag vs PX4's interval-end means; within 0.01° of the best scheme over 1 s windows — accepted.
- Citations in `assumptions.md` A-07…A-12 marked "unverified" (Hofmann-Wellenhof & Moritz section, PX4 doc pages, IGRF details, Snyder) need checking before the writeup. Recomputed IGRF inclinations using the per-log UTC date are TBD.
- `ppigrf` dates: 78808b7c is from 2024-12-19.

Group 3 resolved: GPS delay re-stamped `t − 0.11 s` (A-09); mag fused as tilt-compensated heading; baro as height change (A-10); local g per log.

## Group 3 — Project A 15-state ES-EKF (M3) — done

**Delivered**
- `docs/specs/ekf_drone.md` (model, API, generator, tests, study) and `sensor_fusion/ekf_drone.py`: 15-state error-state EKF on `common/ekf.py`. Error state `[δp, δv, δθ (body, right), δb_g, δb_a]`; F/G/H derivation in the module docstring; nominal propagation identical to `strapdown` (trapezoid, mid-interval attitude); first-order Φ/Qd (Van Loan option); GPS pos/vel (delay-corrected, applied at the nearest IMU sample); baro down-position; tilt-compensated mag heading with full residual Jacobian; inject + reset after every update; frozen bias states for increments 1–2 (`EkfConfig.increment`). Static init: accel leveling + mag heading + IGRF declination; origin = first GPS fix. `streams_from_log`, `ekf2_reference` (EKF2 translated into our NED; reviewer-verified: GPS − EKF2 median ≤ 0.12 m on all logs).
- `sensor_fusion/synthetic.py` (analytic figure-8 / hover / static truth, instantaneous IMU with noise + bias random walk, delayed GPS, baro, mag) — written by test-author from the spec.
- `sensor_fusion/report/ekf_study.py` → `report/ekf_metrics_<id>.json` + `figures/ekf_<id>_{pos,att,bias,nis}.png` (3 logs × 3 increments, ~3 min total; EKF2 biases from `estimator_sensor_bias` as reference).
- `assumptions.md` A-03 (IMU noise from logged EKF2 params `EKF2_ACC_NOISE` 0.35 m/s² etc., converted to densities with `EKF2_PREDICT_US` = 10 ms), A-04 (R values and floors), A-05 (IGRF declination −0.31° / 0.00° / +0.45°).
- Tests: **664 passing** (~60 s). New: 22 generator self-checks (strapdown on noise-free synthetic IMU converges at ratio 4.000 per dt halving) and 37 filter tests: numerical F/G and H Jacobians, first-order vs Van Loan, noise-free tracking, bias convergence, **NEES Monte Carlo 9.86 (99 % band 5.9–12.8, 10 seeds)**, GPS NIS 0.958 inside 95 %, increments, SPD/unit norm, GPS-delay benefit, static init, real-log smoke.

**★ Opus review** — no blockers. Numerically verified: F/G signs (FD 2e-6), discretisation (≈ 1e-5 vs Van Loan), yaw Jacobian, declination sign, reset, update ordering, timing, EKF2 frame translation. Fixed: **mag heading used only the yaw Jacobian although the measurement is tilt-compensated with the estimated tilt** (tilt error leaks × tan I: 0.95 / 0.54 / 2.32 on hover / aggressive / other) → full residual Jacobian `mag_heading_jacobian`. Independently confirmed by the synthetic NEES test (failed before, NEES ≈ 14; passes after). Nits fixed: `meta.init.static`, accepted-only update counts, `ekf2_reference` origin guard, docstring bounds and the dt/2 IMU-timing lag (0.01° effect).

**Flight results** (RMS vs EKF2 after 5 s; attitude excludes ±1 s around EKF2 yaw resets; EKF2 is a reference, not truth)

| Log | Inc | posH [m] | posD [m] | vel [m/s] | tilt [°] | yaw [°] | NIS gps_pos / vel / baro / mag (dof 3/3/1/1) |
|---|---|---|---|---|---|---|---|
| hover | 1 | 0.165 | 0.96 | 0.224 | 1.55 | 6.13 | 0.76 / 0.67 / 1.27 / – |
| hover | 2 | 0.150 | 0.96 | 0.214 | 1.68 | 4.75 | 0.74 / 0.62 / 1.27 / 0.47 |
| hover | 3 | 0.050 | 1.04 | 0.105 | 1.05 | 5.78 | 0.78 / 0.22 / 1.16 / 0.09 |
| aggressive | 1 | 0.155 | 0.81 | 0.225 | 1.18 | 2.58 | 2.73 / 1.39 / 37.7 / – |
| aggressive | 2 | 0.156 | 0.81 | 0.224 | 1.10 | 1.29 | 2.72 / 1.39 / 37.7 / 0.24 |
| aggressive | 3 | 0.055 | 0.77 | 0.088 | 0.37 | 1.38 | 2.37 / 0.87 / 38.0 / 0.05 |
| other | 1 | 0.268 | 0.80 | 0.256 | 0.64 | 2.00 | 0.80 / 0.98 / 0.93 / – |
| other | 2 | 0.304 | 0.80 | 0.297 | 0.84 | 2.88 | 0.89 / 1.24 / 0.93 / 0.23 |
| other | 3 | 0.254 | 0.80 | 0.231 | 0.56 | 1.80 | 0.69 / 0.87 / 0.94 / 0.31 |

Runtime 9–32 s per run (≈ 15k IMU steps/s). Bias estimation (inc 3) cuts horizontal position and velocity differences by ~3× on hover/aggressive and tilt up to 3× (aggressive 1.10 → 0.37°). Gyro bias agrees with EKF2's `estimator_sensor_bias` to ≈ 0.001 rad/s.

**Interpretation notes / open issues for Group 4**
- **Hover static tilt offset 1.35°** between accel leveling and EKF2 equals the tilt implied by EKF2's own accel bias [−0.141, −0.192, −0.009] m/s². It is not a convention error.
- **Hover horizontal accel bias is unobservable** (8° yaw range): σ stays ≈ 0.13–0.15 m/s², close to the prior; the estimate wanders along the tilt/bias ridge (ended at −0.25 before the mag fix, at x +0.04 / y −0.31 after). Report it only with ±σ, and do not tune `accel_bias_rw` to "fix" it. Vertical accel bias is well determined (σ ≈ 0.01).
- **Yaw vs EKF2**: hover is −5…−8° once mag is fused (uncorrected hard iron, A-11; EKF2 estimates mag bias). On "other", inc 2 yaw is worse than inc 1 (≈ 3° mag-vs-IGRF heading offset, Group 2). Options: mag bias states (beyond the 15-state scope) or larger σψ on hover — a Group 4 decision.
- **Mag heading is strongly nonlinear at high inclination** (the Jacobian changes ~15 per rad at I = 67°); keep σψ at several degrees or more on the "other" log.
- **Baro NIS ≈ 38 on aggressive** (GPS vertical drift vs baro, A-08/A-10) → baro offset state or larger σ_baro.
- **NIS low** for mag (0.05–0.47) and hover GPS velocity (0.22; 0.3 m/s floor vs s_acc 0.05) → R too conservative. GPS-position NIS is 2.7 on aggressive.
- **Vertical difference 0.8–1.0 m RMS** on all logs, partly datum: GPS D − EKF2 D is −0.27…−0.33 m in the first 5 s on all logs (reviewer), and EKF2 follows GPS height on 2 logs and baro on 1.
- Synthetic: with a frozen 0.01 rad/s gyro bias, increment 2 is overconfident (0.2 rad yaw error, σ 0.004) — the expected motivation for increment 3.
- `run_filter` applies the GPS σ floors; `DroneEKF.update_*` take σ as given.
- All increments were implemented together behind `EkfConfig.increment` and run on all 3 flights after the review, not strictly one after another.

## Group 4 — Project A tuning, validation, writeup (M4) — done · **Project A complete**

**Delivered**
- `docs/specs/tuning_writeup.md` (spec for everything below).
- `sensor_fusion/metrics.py` (extracted behaviour-identically from `ekf_study.py`: `compare_to_ekf2`, `nis_summary` with χ² bounds, `bias_summary`, `ekf2_bias_reference`; new `consistency_score` = RMS of ln(mean NIS/dof)), `sensor_fusion/report/ekf_figures.py` (figure functions with `prefix`/`out_dir`), `sensor_fusion/run_flight.py` CLI (`python -m sensor_fusion.run_flight <role|id|all> [--increment 3] [--set key=value] [--prefix final]` → `report/final_metrics_<id>.json` + `figures/final_<id>_{pos,att,bias,nis}.png`). Regression: new metrics reproduce the stored G3 hover numbers bit-identically (104 numbers). `ekf_study.py` now uses `G3_DEFAULTS` (the pre-tuning values) so the increment study stays reproducible.
- `sensor_fusion/report/tuning_sweep.py` → `tuning_sweep.json/.md`, 10 figures: 120 runs (40 configs × 3 logs), 500 s wall time on 20 workers; baseline pinned to `G3_DEFAULTS` after review (re-run reproduces the committed tables).
- `sensor_fusion/report/height_diagnostics.py` → `height_diagnostics.json` + figures (end-of-window down position ours/EKF2/GPS/baro/integrated GPS velocity, reported GPS accuracy, NIS ratios).
- `sensor_fusion/report/writeup.md` (draft from the JSON files; ★ reviewed).
- `assumptions.md` A-13 (tuning procedure and decision); A-03/A-04/A-09/A-11 corrected.
- Tests: **722 passing** (~90 s). New: 58 (`test_metrics.py`, `test_run_flight.py`; independent oracles: hand-computed values, scipy `Rotation`, `scipy.stats.chi2`).

**Tuning outcome (small, and that is the finding).** Sweep winner `joint` (total score 1.816 → 1.717) was **not** adopted: the ★ review showed the gain of lower `gyro_bias_rw` comes from the mag NIS (too-large R offset by too-small Q) and makes our gyro bias overconfident vs EKF2's `estimator_sensor_bias`. **Only `gps_pos_floor_v` 0.5 → 0.125 m was adopted** (RTK log reports epv 0.010–0.026 m; total score 1.816 → 1.770). All other `EkfConfig` defaults are unchanged. The choice deviates from the pre-set rule's winner after seeing results (recorded as post hoc in A-13).

**Final numbers** (increment 3, RMS difference to EKF2 after 5 s — a reference, not truth; `final_metrics_<id>.json`)

| Log | posH [m] | posD [m] | vel [m/s] | tilt [°] | yaw [°] | NIS ratio gps_pos / gps_vel / baro / mag | score |
|---|---|---|---|---|---|---|---|
| hover | 0.050 | 1.040 | 0.105 | 1.05 | 5.78 | 0.27 / 0.075 / 1.18 / 0.089 | 1.892 |
| aggressive | 0.055 | 0.753 | 0.090 | 0.37 | 1.38 | 0.79 / 0.30 / 39.4 / 0.053 | 2.435 |
| other | 0.255 | 0.929 | 0.216 | 0.56 | 1.80 | 0.42 / 0.27 / 1.10 / 0.31 | 0.981 |

**Findings to carry forward**
- NIS: only baro on hover/other is near 1. Mag, GPS velocity and GPS position ratios are 0.05–0.42, i.e. the noise models are conservative; aggressive-log baro ratio ≈ 39 (not sensitive to knobs except `baro_sigma` and the vertical GPS floor) is consistent with GPS vertical position drifting ≈ 7.4 m against its own integrated velocity and the baro (`height_diagnostics.json`); the filter's height follows GPS. A baro-offset state is untested future work. Mean-NIS χ² bounds are very narrow with thousands of correlated samples; ratios and the effective-sample-size reading are used instead.
- `fraction_inside_95` is insensitive to over-conservative R (stays ≈ 0.9 even for ratio 0.05).
- Aggressive-log accel bias disagrees with EKF2 (y axis opposite sign, ≈ 24σ; gyro biases agree) — reported, not explained. The log ends in an impact; the landing-height closure assumes landing height = take-off height (ASSUMPTION).
- GPS-delay sensitivity (A-09) was **not** run in Group 4.
- Citations in A-07…A-12 marked "unverified" still need checking before any external use.
- Reviewer-only numbers in A-13 (bias-normalised differences, EKF2 bias wander rates) come from scratch scripts that are not in the repo.
- `tuning_sweep.py` sets `OPENBLAS_NUM_THREADS=1` itself (20 workers otherwise hit OpenBLAS allocation failures).

**Next: Group 5** (M5, M6: Project B physics — orbit, environment, dynamics, disturbances). Recommended main-session model: **Sonnet** (`/model sonnet`).
