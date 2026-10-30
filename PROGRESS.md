# Progress

Read this file and `docs/IMPLEMENTATION_PLAN.md` at the start of every group. Updated by the main session at every group exit.

| Group | Milestones | Status |
|---|---|---|
| 1 Foundations | M0 | **done** |
| 2 A data + dead reckoning | M1, M2 | **done** |
| 3 A 15-state EKF | M3 | **done** |
| 4 A tuning + writeup | M4 | **done** |
| 5 B physics | M5, M6 | **done** |
| 6 B control | M7 | **done** |
| 7 B estimation | M8 | **done** |
| 8 B closed loop + MC + doc | M9 | **done** |
| 9 Packaging + visualizer | M10, M11 | **done** |
| 10 Browser sim: engine stepper + server | M12 | **done** |
| 11 Browser sim: ADCS frontend | M13 | next |
| 12 Browser sim: PX4 replay + docs | M14 | planned |

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
- **Open: EKF2 hover yaw-reset time** is 16.06 s in `sensor_fusion/data/NOTES.md` (line 151) but 16.08 s in `final_metrics_<id>.json` (`ekf2_reset_times_s`); A-08 was changed to 16.08. The cause (rounding vs window t0) was not checked and NOTES.md was not edited.
- **n_eff check:** the reviewer's n_eff numbers for GPS position were wrong; the writeup uses the recomputed values (scipy `chi2` with n_eff × dof degrees of freedom: aggressive gps_pos 0.79 needs n_eff ≈ 52, other gps_pos 6, hover gps_pos 3; gps_vel 2–3; mag 3/4/10; baro 1 / 266 / 822). The n_eff table is a calculation from the ratios, not stored in a JSON.
- Group 3 notes above say "GPS-position NIS is 2.7 on aggressive" — that is a mean NIS for 3 dof (ratio 0.79); use ratios.
- The suite now takes ~85–90 s (was ~60 s; `CLAUDE.md` command comment updated); the two hover smoke tests in `test_run_flight.py` take ~20 s each.
- `writeup.md` is ~3,300 words with tables (spec said ~2 pages) and has had one math-review pass plus one fix pass; a final read-through by the user before external use is advisable. `sensor_fusion/data/NOTES.md` was not touched in Group 4.
- Not verified/not done in Group 4: GPS-delay sensitivity; baro-offset state; GPS lever arm; accel-bias disagreement with EKF2 on the aggressive log (unexplained).

## Group 5 — Project B physics (M5, M6) — done

**Delivered**
- `docs/specs/adcs_physics.md` (spec), `adcs/config.py` (written by the main session, not an agent, because every other module depends on it: `SimConfig`/`SpacecraftConfig`/`OrbitConfig`, epoch 2026-03-20 12:00 UTC, 400 km / 51.6° circular, RAAN 0, 3U 4 kg box inertia diag(0.041867, 0.041867, 0.006667) kg·m², CoM offset (5, 5, 10) mm, residual dipole (1, 1, 1) mA·m², Cd 2.2).
- `adcs/orbit.py` (Kepler ↔ rv, two-body + J2 `accel_j2`, `specific_energy`, DOP853 `propagate`, dense `Ephemeris`, `sgp4_ephemeris`), `adcs/environment.py` (Vallado Alg. 29 sun, cylindrical shadow, β angle, exponential density, IGRF via ppigrf rotated to ECI, `FieldTable` cubic-spline cache so closed-loop sims do not call ppigrf per step), `adcs/dynamics.py` (state `[q, ω, h_w]`; `I ω̇ = −ω×(Iω + h_w) − u_w + τ_ext`, `ḣ_w = u_w`; `derivative`, `rk4_step`, `integrate`, `angular_momentum_inertial`, `kinetic_energy`, `lvlh_error`, `nadir_angle`), `adcs/disturbances.py` (gravity gradient, residual dipole, flat-plate aero with co-rotating atmosphere, breakdown).
- `adcs/report/physics_figures.py` → `physics_metrics.json` + 4 figures (`orbit_vs_sgp4`, `eclipse_and_field`, `disturbance_magnitudes`, `conservation_10orbits`); `python -m adcs.report.physics_figures` (~33 s).
- `assumptions.md` B-03, B-08…B-15 filled with measured numbers.
- Tests: **898 passing** (~90 s; 176 new, `adcs/tests/test_{config,orbit_env,dynamics_dist}.py`). The test-author tests were written from the spec; independent oracles: sgp4, closed-form eclipse/Kepler/J2 formulas, explicit ENU→ECI rotation, scipy `Rotation`, analytic symmetric-top and gyroscopic solutions.

**★ Opus review** — no blockers; 21/21 sign/frame mutants (GMST direction, NED DCM, gravity-gradient sign, wheel reaction sign, kinematics side, LVLH transpose, aero sign, J2, obliquity, β, shadow …) are caught. Fixed after the review: field upper bound (see below), raw quaternion-norm check (`integrate(normalize=False)`), stale doc claims, spec wording on gravity-gradient sign, density-error wording, `h_w` / wheel-speed note for Group 6, ppigrf pole NaN note.

**Measured numbers** (`adcs/report/physics_metrics.json`)

| Check | Result |
|---|---|
| J2 orbit vs SGP4 (ISS TLE, started from SGP4 state) | max \|Δr\| 103 m over 1 orbit, 2.34 km over 1 day (along-track 2.34 km, radial 386 m, cross 121 m); drag/J3/J4 not modelled |
| Two-body / J2 conservation, 10 orbits | energy 3.4e-12 / 3.5e-12, h_z 1.7e-12 (relative); nodal regression within 0.5 % of formula |
| Eclipse fraction, default config | 0.3905 (3 orbits), 0.3939 (1 day); analytic β = 0 value 0.3901; β = 0.05° at epoch |
| \|B\| at 400 km, 3 orbits | 18.98 / 34.98 / 49.31 µT (min/mean/max) |
| Free tumble, 10 orbits (rtol 1e-11) | \|H_i\| 5.9e-14, T 1.1e-13, H_i vector 1.6e-10, raw \|q\|−1 1.4e-10 |
| Disturbance torques, orbit-RMS, 7 attitudes | aero 8.4e-8…1.3e-7, residual dipole 4.8e-8…5.5e-8, gravity gradient 2.3e-8…4.4e-8 N·m (tilted); **aero is largest in 7/7** |

**Decisions and deviations (record for later groups)**
- **The plan's 25–45 µT field gate does not hold.** IGRF over the ±51.6° band at 400 km spans 18.98 µT (South Atlantic Anomaly) to 54.13 µT. Test bounds revised 20–50 → 15–55 µT, with an independent grid-scan oracle for the floor; the first 3 orbits peak at 49.3 µT but one day reaches 54.0 µT. **Size magnetorquers/magnetometers for ≈ 19–54 µT.**
- **"Gravity gradient dominates" does not hold** for this 3U: aero torque is the largest source in every tested attitude (it scales with the CoM offset and density, ×10 swing with solar activity). The sanity gate was reinterpreted as an order-of-magnitude check (1e-10…1e-6 N·m, except gravity gradient exactly at principal-axis alignment, which is 0 by symmetry). With Ix = Iy the 3U is only marginally stable in yaw under gravity gradient.
- `propagate` default `rtol` 1e-12 (spec said 1e-10: the energy-drift test failed at 1e-10). The 10-orbit tumble test uses `integrate(rtol=1e-11)` (default 1e-10 gives 1.8e-9 H_i drift from DOP853 phase error); thresholds were not loosened. The old "raw |q| = 1 to 1e-12" claim was unattainable and untested (renormalisation); the test now checks the raw output at 1e-9 (1.4e-10 measured).
- `h_w` is the **total** wheel momentum with the rotors' spin-axis inertia excluded from `I`; Group 6 computes wheel speed as `(h_w·a)/J_s − a·ω`.
- Aero uses the spherical height `|r| − R_E` for density (≤ 25 % density error at 51.6°); solar radiation pressure neglected (hand estimate 1.5e-9 N·m); inertia ignores the CoM offset.
- `magnetic_field_eci` is NaN exactly at a geographic pole (ppigrf); irrelevant for 51.6°, relevant for polar configs. `FieldTable` queries outside `[0, t_end]` raise.
- Open nits not fixed: `atol` 1e-6 applies to both metres and m/s in `propagate`; `sgp4_ephemeris` takes `jd0` as one double (≤ 0.3 m); the spec test 6 mentions both 0.05° and 0.1° for the equinox declination (0.05° used); the Sun-algorithm transcription test is not independent (the equinox/solstice tests are).
- Citations in B-01…B-15 marked "unverified" (CubeSat Design Spec rev., Vallado table/equation numbers, Wertz, IGRF) must be checked before external use. TLE: python-sgp4 documentation example (epoch 2019-12-09), used only for the SGP4 comparison.

## Group 6 — Project B control, perfect knowledge (M7) — done

**Delivered**
- `docs/specs/adcs_control.md` (spec; test thresholds set after a main-session prototype, measured values in brackets).
- `adcs/actuators.py` (py-implementer): `MagnetorquerConfig` (0.2 A·m²/axis, ISIS iMTQ; 1 s control period, duty 0.8 — magnetometer sampled with coils off), `WheelConfig` (CubeSpace CW0017: 0.23 mN·m, 1.77 mN·m·s at 8000 rpm, derived J_s = 2.1128e-6 kg·m²; 3 wheels on body axes), `saturate_dipole`, `magnetorquer_torque`, `wheel_speed`/`wheel_momentum`, `limit_wheel_torque` (torque clip + speed limit that a held command cannot cross). Datasheet values read from the vendor pages on 2026-10-06 (main session, WebFetch).
- `adcs/control.py` (main): B-dot on the field direction with the Avanzini & Giulietti (2012) gain `k = 2n(1 + sin i)J_min` = 2.69e-5 N·m·s; rate-limited quaternion-error PD (wn 0.1 rad/s, ζ 0.7, slew limit 0.5 deg/s) with gyroscopic and reference-rate feed-forward (error dynamics `I δω̇ = −Kp err − Kd δω + τ_d`, derivation in the docstring); `nadir_reference` (LVLH rotates at −n about LVLH y); cross-product dumping `m = (k/|B|²) h × B` (k 1e-3 1/s) with hysteresis 0.3/0.05 of max wheel speed; magnetorquer torque fed forward to the wheels.
- `adcs/simulate.py` (main): perfect-knowledge runner, ZOH per 1 s control period, RK4 0.8 s (coils on) + 0.2 s (off), modes `detumble`/`pointing`/`auto` (switch after |ω| < 0.5 deg/s for 60 s), `SimResult`, `tumble_state`, `lvlh_state`, `detumble_time`, CLI `python -m adcs.simulate --scenario …`. Group 8 adds sensors + MEKF here.
- `adcs/dynamics.py`: internal speed-up only (`_cross` for 3-vectors; `np.cross` cost ~40 µs/call); API and results unchanged (existing tests green).
- `adcs/report/control_figures.py` → `control_metrics.json` + `figures/{detumble,detumble_sensitivity,pointing_15orbits,auto_detumble_to_nadir}.png` (93 runs, 562 s on 20 processes).
- `assumptions.md` B-05, B-06 filled; B-16 (control laws/gains/decisions), B-17 (discretisation), B-18 (detumble benchmark) added.
- Tests: **1054 passing** (~130 s; 156 new in `adcs/tests/test_{actuators,control,simulate_pk}.py`). Oracles: hand values, closed-form B-dot dissipation for finite dt, own RK4 loop for energy monotonicity, finite-difference LVLH rate, scipy `Rotation`, H_i conservation. The simulation tests run their five scenarios in a process pool (fixture), ~80 s of the suite.

**Measured (`adcs/report/control_metrics.json`, perfect knowledge)**

| Check | Result | Target |
|---|---|---|
| Detumble 10 deg/s → < 0.5 deg/s, 0.2 A·m², seeds 0–4 | 1.14 / 1.62 / 1.68 / 0.88 / 1.51 orbits (mean 1.36) | 1–3 orbits ✅ |
| Rate floor after detumble | ≈ 0.1–0.2 deg/s (≈ 2n field-tracking) | — |
| Sensitivity (mean, seeds 0–2) | 2 → 30 deg/s: 0.66 → 2.2 orbits at 0.2 A·m²; only 0.05 A·m² at 30 deg/s exceeds 3 orbits (3.50) | — |
| Gain scale ×0.25 / 0.5 / 1 / 2 / 4 at 10 deg/s | 3 of 5 not detumbled in 6 orbits / 2.92 / 1.36 / 0.73 / 0.43 orbits | — |
| Nadir pointing 15 orbits, dumping on (after orbit 1) | max 0.078°, RMS 0.053°; wheels ≤ 0.30 of max speed | < 1° ✅ |
| Dumping off | y wheel saturates at 3.73 orbits; attitude lost (max 179.9°) | demonstrates need ✅ |
| Auto 10 deg/s → nadir | switch at 1.15 orbits; error ≤ 0.066° from switch + 1 orbit | — |
| No disturbances (test) | 0.0013° after 1000 s, H_i drift 2e-12 | — |

**Decisions and findings (record for later groups)**
- **B-dot gain kept at the cited nominal value**, although 2× / 4× detumble faster with perfect knowledge (0.73 / 0.43 orbits): a higher gain amplifies magnetometer noise in the finite-difference Ḃ. Revisit in Group 8 with sensor noise.
- `m_max` barely binds at the nominal gain (0.2 and 0.4 A·m² curves nearly coincide; the peak dipole in auto mode is 0.144 A·m²).
- **Dumping is effectively continuous**: in the 15-orbit run it is active 92 % of the time; the y wheel settles near 0.1 of max speed (~700 rpm in the figure), where the dumping torque balances the secular aero torque (mean τ_y ≈ 8.8e-8 N·m, prototype), so the 0.05 off threshold is never reached. Peak dipole only 0.01 A·m². Harmless with perfect knowledge; in Group 8 the coil activity affects the magnetometer only through the duty cycle (sampled with coils off). Raising `dump_off` to ~0.15 or `dump_k` would make the hysteresis cycle — not done.
- Steady pointing error (~0.05–0.08°) is set by disturbance / Kp (z axis has the smallest inertia); no integral term (not needed for the 1° target).
- `nadir_reference` ignores the J2 out-of-plane LVLH rotation (≈ 1e-6 rad/s, test 9).
- Held wheel torque lets the wheel speed exceed the limit by ≤ 1.3e-6 relative (ω changes within the period); tests allow 1e-5.
- The pointing test scenario preloads the y wheel to 0.9 h_max so that dumping on/off is tested within 1 orbit (secular growth ≈ 28 % of h_max per orbit).
- B-17: no sub-stepping convergence check of the 1 s ZOH / RK4 discretisation was run.
- Citations: vendor pages (read 2026-10-06) for iMTQ and CW0017; Avanzini & Giulietti 2012 and the LAPAN 3U study (B-18), Wie and Numerical Recipes (added by doc-writer) are marked unverified.
- No ★ review in this group (none in the task table); signs were checked by tests (dissipation, closed-loop error dynamics, LVLH rate vs finite differences, dumping torque identity).

## Group 7 — Project B attitude determination (M8) — done

**Delivered**
- `docs/specs/adcs_estimation.md` (spec: §0 sampling model, sensors, TRIAD + covariance derivation, MEKF API, prototype numbers, tests, figures).
- `adcs/sensors.py` (py-implementer): magnetometer **PNI RM3100** (σ 15 nT/axis), gyro **ADXRS453** (ARW 0.65 °/√h = 1.8908e-4 rad/√s; RRW 1.0409e-6 rad/s/√s from 16 °/h bias stability over one orbit — ASSUMPTION; turn-on bias σ 0.1 °/s), six cosine CSS **Tensor Tech CSS-10** (σ_d 0.0436 → 5° 1σ, validity threshold 0.6). Vendor pages read 2026-10-06 (main session). `Gyro` (stateful), `measure_series` (independent `spawn(3)` streams), `mean_rates`. **Sampling model**: everything at 1 Hz; gyro sample k = exact mean rate `to_rotvec(q_{k−1}* ⊗ q_k)/T` + bias + ARW (ideal rate-integrating gyro), so the filter's propagation model is exact for the truth.
- `adcs/estimation.py`: `triad`, `triad_covariance` (py-implementer; covariance derived by main, body frame, verified by MC to 1.5 % of λ_max); **MEKF** (main): error state `[δθ, δb]` on `common.ekf.ErrorStateEKF`, closed-form Φ/Qd (`discrete_model`, = `van_loan` to 1e-11), vector updates `H = [[R(q̂)ᵀv×], 0]` for magnetometer (field in T, 3 dof) and CSS (unnormalised difference vector, 3 dof), inject + reset after every update, TRIAD initialisation at the first valid sun sample (primary = magnetometer), optional χ² gate (off by default); `run_mekf`, `attitude_errors`, `nees`.
- `adcs/report/estimation_figures.py` → `estimation_metrics.json` + 7 figures `figures/mekf_*.png` (57 runs, ~110 s on 20 processes).
- `assumptions.md` B-07 (sensors), B-19 (MEKF design), B-20 (measured performance + ★-review optimism numbers).
- Tests: **1177 passing** (123 new: `test_sensors.py` 44, `test_triad.py` 37, `test_mekf.py` 42; oracles: scipy `Rotation`, `van_loan`, finite differences, χ² statistics, MC covariance).

**Measured** (`adcs/report/estimation_metrics.json`; matched noise models — an ideal-sensor floor, see below)

| Check | Result |
|---|---|
| Nadir pointing, 5 orbits: RMS ‖δθ‖ after 1 orbit, sunlit / eclipse | 0.346° / 0.731° (max 2.16°) |
| Detumble from 3 °/s, 2 orbits | 0.352° / 0.449° (max 0.94°) |
| Attitude NEES (dof 3), pointing / detumble | 3.00 / 3.21 |
| NIS mag / sun, pointing | 2.995 / 2.976 (inside 95 % bounds); detumble sun 2.940 is 6e-4 below its lower bound (one seed) |
| 20-seed MC (3 orbits): per-run NEES / NIS mag / NIS sun | 2.95 ± 0.15 / 3.00 ± 0.02 / 3.00 ± 0.02 |
| Bias error 0.185 °/s → at 0.25 orbit | 0.0010 °/s; final σ_b ≈ 0.001 °/s/axis |
| TRIAD RMS (sunlit) vs predicted √(mean trace P) | 4.89° vs 4.91° (MEKF sunlit is ~14× better) |
| Eclipse σ_att entry → exit (pointing) | 0.29° → 1.04–1.07° (×3.6–3.8); back to 0.28° 300 s after exit |
| Sensitivity (white, matched): σ_m 15 → 1000 nT; CSS 2.5 → 10° | sunlit 0.31 → 0.40°; 0.20 → 0.48°; eclipse 0.63–0.73° throughout |

**★ Opus review** — no blockers. Verified numerically: Φ vs nonlinear truth (second order), series branch, H/residual/reset signs, unit conversions (ARW = ND/√2 makes 0.015 °/s/√Hz and 0.65 °/√h agree — a doc-writer "discrepancy" was wrong and corrected), whiteness of magnetometer innovations (lag-1 −0.001). Applied: NEES moving-mean spikes (≈ 10) explained as the about-field error's ~150-sample correlation, largest one mid-sunlit, not at eclipse exit; B-20 now quantifies the optimism; sun-NIS moving mean restarted per sunlit pass; "300 s after exit" index; spec 3σ-fraction and TRIAD-predictor wording; Q-discretisation note for the per-sample bias step (2e-5 relative); test 9 now compares at the last eclipse sample and checks the about-field axis is weakest; test 7 now also checks bias 3σ fractions (0.998–0.9996).

**Findings / open items for Group 8**
- **The results are optimistic.** Filter reference field = truth field, no magnetometer bias, white CSS noise only. Review scratch runs (not in repo, 4 seeds): unmodelled 300 nT mag bias → NEES ≈ 815 (inflating σ_m to 300 nT still NEES ≈ 29); hard ±60° CSS FOV → 8.1° sunlit RMS; crude albedo → 6.4° sunlit. Group 8 needs: an onboard field model independent of truth (e.g. lower-degree IGRF / orbit error), magnetometer bias (states or consider covariance), CSS FOV handling (reconstruct only from lit diodes / least squares) and possibly albedo — at least as a sensitivity case. These scratch numbers should be reproduced by a repo script if they go into the design document; the reviewer's scripts are kept for reference in `docs/review_scratch/group7/` (unreviewed).
- **Gating**: don't simply enable `gate_prob=0.9973` with mismatched models (NIS ≈ 123 would reject nearly every mag update). Monitor NIS first.
- **Closed loop**: generate the gyro inside the control loop from q_{k−1}, q_k of each 1 s step (`measure_series` on a `log_every > 1` record would silently use dt = 10 s); controller gets ω̂ = mean rate of the previous period (T/2 lag + 0.011 °/s per-sample ARW in the D term); never propagate with gyro row 0 (NaN). Factor `run_mekf`'s loop body into a public stepper shared by batch and closed loop (it currently reads `f._q`, `f.ekf._P`). Keep the `spawn(3)` stream order.
- **Initialisation**: no estimate until the first valid sun sample (up to ~2170 s if starting in eclipse); `auto` mode's 0.5 °/s detumble threshold would see an uncorrected gyro bias (0.17 °/s norm, 1σ) — use B-dot (magnetometer only) for detumble and gate the switch on the MEKF. `triad_init` has no minimum-angle guard: add α ≥ ~20° (about-field σ = 0.0617/sin α; the orbit's minimum field–sun angle is 20.4°).
- 10° CSS cell: sun NIS 3.10 — the 0.6 threshold is only 2.8σ / 1.9σ from eclipse / lit at that noise (false detections). Scale the threshold with σ_d if noisier CSS is used.
- Statistical nits not fixed: test 7's ±5 % pooled-NIS band (~15 standard errors) would not catch a 2 % σ_sun error.
- Mag-only vs both at the last eclipse sample: σ about the field 0.666° vs 0.660° — almost all sun information is gone by the end of an eclipse.
- Citations in B-07/B-19 (ADXRS453 datasheet PDF, CSS-10 manual, Markley & Crassidis, Shuster & Oh, Crassidis & Junkins, IEEE 952) are marked unverified; one `TBD` (IGRF model error magnitude) remains in B-07.

Group 8 resolved: closed loop on the estimate with a public MEKF stepper and in-loop gyro sampling; onboard field model ≠ truth, magnetometer bias, CSS FOV and albedo added as truth-side errors with estimator mitigations; TRIAD 20° guard; detumble → pointing switch gated on the MEKF; gating left off.

## Group 8 — Project B closed loop, Monte Carlo, design document (M9) — done · **Project B complete**

**Delivered**
- `docs/specs/adcs_closed_loop.md` (spec §0–§5 with the main-session prototype numbers and the design-filter tuning record).
- Sensor realism (py-implementer): `environment.magnetic_field_eci/FieldTable(max_degree=…)`; `sensors.magnetometer(bias=…)`, `gauss_markov`, `SunSensorConfig.fov` (hard cut), `sun_sensor_currents(albedo=…)`, `earth_view_factor` (plate-to-sphere, checked by ray Monte Carlo), `albedo_currents`. All defaults = Group 7 behaviour.
- `adcs/estimation.py` (main): optional magnetometer-bias states (9-state MEKF), per-diode CSS block updates (gate on the *predicted* cosine > 0.55; gating on the measured current biased NEES to 3.3–4.8), onboard albedo compensation with albedo uncertainty in R, `triad_min_angle`, `init_att_sigma_floor`, public `MEKF.step` (shared by `run_mekf` and the closed loop), `vector_angle`, `css_vector_complete`, `run_mekf(r_eci=…)`, `meta["nis_dof"]`. `MekfConfig()` unchanged.
- `adcs/simulate.py` (main): `run_closed_loop` (sensors in the loop, gyro from consecutive true attitudes, MEKF against the onboard field, mode machine on the estimate, fallback on estimated rate, controller on q̂/ω̂, truth with `inertia_true`), `ClosedLoopResult`, `summarize_closed_loop`, presets `IDEAL_SENSORS`, `REALISTIC_SENSORS`, `MEKF_G7`, `MEKF_DESIGN`, `ModeConfig`, `SensorErrors`; CLI `python -m adcs.simulate` now runs the closed loop by default (`--knowledge perfect` for Group 6).
- `adcs/monte_carlo.py` (py-implementer): `draw_run`, `run_one`, `run_campaign`, `aggregate`, `check_requirements`, `REQUIREMENTS`, `save_json`, CLI `python -m adcs.monte_carlo`.
- Report scripts (figures-runner, main): `adcs/report/closed_loop_figures.py` → `closed_loop_metrics.json` + 5 figures (58 runs, 719 s); `adcs/report/monte_carlo_figures.py` → `figures/mc_*.png` (7) + `monte_carlo_summary.md`; `adcs/report/mc_worst_runs.py` → `mc_worst_runs.json` + `figures/mc_worst_run.png`; `monte_carlo_{design,ideal}.json` (1346 s / 1156 s on 20 workers).
- `adcs/report/design_document.md` (≈ 6500 words incl. tables): requirements, hardware, architecture/rationale, results by phase (doc-writer draft, hand calculations spot-checked), Monte Carlo, failure modes, conclusions, omissions, traceability.
- `assumptions.md` B-21 (truth-side errors), B-22 (design MEKF), B-23 (mode machine and onboard knowledge), B-24 (assumed requirements + MC results); B-07 cross-reference.
- Tests: **1388 passing** (~230 s idle). New: `test_sensor_realism.py` 82, `test_mekf_ext.py` 35, `test_closed_loop.py` 54, `test_monte_carlo.py` 40 (oracles: ppigrf chain, ray Monte Carlo, scipy `Rotation`, finite-difference Jacobians with a textbook EKF update, `van_loan`, χ², the Group 6 perfect-knowledge runner, reconstructed random streams).

**Measured** (simulation; `closed_loop_metrics.json`: 10 °/s, 6 orbits, seeds 0–4, means; `monte_carlo_*.json`: 100 runs each, 2–10 °/s, ±10 % truth inertia)

| Case | Nadir RMS | Nadir max | Knowledge RMS | NEES |
|---|---|---|---|---|
| Perfect knowledge | 0.054° | 0.068° | – | – |
| Ideal sensors + Group 7 filter | 0.53° | 1.92° | 0.51° | 3.05 |
| Realistic sensors + Group 7 filter | 24.4° | 57.2° | 24.1° | ≈ 6400 |
| Realistic sensors + design filter | 0.99° | 3.02° | 0.96° | 13.7 |
| Low-noise check (σ × 0.01, 2 °/s, main-session prototype; test 1 of `test_closed_loop.py`) | 0.0533° vs PK 0.0532°, same switch time | | | |

| Monte Carlo (median / p95 / max) | design campaign | ideal campaign |
|---|---|---|
| Detumble [orbits] | 1.22 / 1.88 / 2.29 | 1.21 / 1.63 / 1.89 |
| Switch to pointing [orbits] | 1.21 / 1.88 / 2.23 | 1.17 / 1.66 / 1.85 |
| Nadir RMS [°] | 0.962 / 1.25 / 1.46 | 0.503 / 0.635 / 0.733 |
| Nadir max [°] | 3.15 / 4.44 / 5.19 | 1.84 / 2.90 / 3.59 |
| All requirements met | 99/100 (run 54 fails REQ-PNT at 5.19°) | 100/100 |
| 1° RMS goal | 58/100 | 100/100 |

**Decisions and findings (record for Group 9 / later)**
- **Design filter tuning was time-boxed and post hoc** (σ_mag ∈ {65, 45, 100} nT, σ_albedo ∈ {0.1, 0.25, 0.5}, ARW ×3, RRW ×10 on seeds 0–2); chosen σ_mag 100 nT, σ_albedo 0.5 (B-22). It remains **overconfident** (NEES ≈ 14); the breakdown shows the field-model error drives this (NEES 2.3 without it); bias states matter most (without them NEES 157, nadir RMS 1.45°).
- **Worst Monte Carlo excursions are knowledge errors outside the filter's 3σ** (control error ≤ 0.36°), 4 of 5 in sunlight 660–825 s after eclipse exit, run 54 433 s after eclipse entry; largest component about body z (`mc_worst_runs.json`). Cause not isolated further.
- **Late initialisation**: 14/100 design runs initialise TRIAD after 1000 s (all three CSS axes lit + 20° guard); in 2 runs acquisition is delayed 1605 / 2955 s past detumble. TRIAD (diodes mode) cannot initialise in nadir pointing at β ≈ 0 → no re-initialisation in pointing.
- **The naive filter fails silently**: `real_g7` counts as `success` at 24° true error; the rate-based fallback cannot see a wrong estimate. One RRW ×10 scratch tuning run diverged to 180° unnoticed (not reproduced by a repo script). An innovation monitor is recommended, not implemented.
- **B-dot gain** (Group 6 open item): with sensors, detumble times at ×1/×2/×4 match Group 6 within 0.2 %; nominal gain kept. Dumping hysteresis not revisited.
- `success` in `summarize_closed_loop` is internal (switch, no fallback) — it does not look at the true error; use the requirement checks.
- Mode switch times track the perfect-knowledge detumble (within 163 s on seeds 0–4); nadir RMS shows no visible trend with initial rate or truth inertia error (visual, `mc_sensitivity.png`).
- Not done: gating in the closed loop (left off), NIS monitor/FDIR, albedo state, orbit-knowledge error, dumping-threshold revisit, sub-stepping convergence (B-17), ★ review (none in the Group 8 task table).
- Suite time grew to ~230 s (closed-loop and MEKF-extension fixtures ≈ 50 s); `CLAUDE.md` updated.

## Group 9 — Packaging, README, smoke test, 3D visualizer (M10, M11) — done

**Delivered**
- `README.md` rewritten (doc-writer, edited by main): headline tables, layout, setup, reproduction commands with measured wall times and the real worker defaults, gallery, 3D visualizer section, reproducibility check, known limitations.
- `docs/specs/visualizer.md` (contract) → `adcs/visualize.py` (py-implementer): display frame = LVLH rotated 180° about x (zenith up, nadir = (0,0,-1)); `body_dcm_lvlh`, `sun_lvlh`, `frame_schedule` (time-warped frame choice by bisection on a target angle, 1–60 s spacing), `build_animation_data`, `render` (Agg; MP4 via ffmpeg/libx264, GIF via Pillow); CLI `python -m adcs.visualize`.
- `adcs/tests/test_visualize.py` (test-author, from the spec): 48 tests.
- `adcs/report/figures/closed_loop_animation.mp4` (0.80 MB) and `.gif` (1.5 MB, from the MP4 with ffmpeg). Visual aid only; no numbers added.
- Tests: **1436 passing** (1388 + 48), no failures/errors/skips, 260 s run file by file with `OPENBLAS_NUM_THREADS=1`.

**Smoke test (fresh venv, Python 3.14.2, versions identical to Group 1)**
- Fresh log download SHA-256 identical to the committed `.ulg` files.
- Fresh-venv suite: 1388 tests, no failures (before the visualizer tests; counted from dots).
- Regenerated and compared with the committed copies: all Project A scripts, `physics_figures`, `control_figures`, `estimation_figures`, `closed_loop_figures`, default `adcs.simulate`. Every number reproduces (estimation ≤ 1.75e-12 relative, closed loop bit-identical, default `adcs.simulate` = closed-loop seed 0). Regenerated files were reverted with `git checkout`.
- "altitude" sweep: every hit is genuine height.

**Not done / open**
- `python -m adcs.monte_carlo`, `monte_carlo_figures` and `mc_worst_runs` were **not** regenerated in the fresh venv (run stopped at the user's request on a memory-starved machine; ~45+ min on 20 workers, consider `--workers 8`). The committed Monte Carlo results are the Group 8 ones.
- Wrong worker-default claim in the first README draft was corrected to the real defaults (README "Worker counts").
- Open note carried over: hover EKF2 yaw-reset time 16.06 s (`NOTES.md`) vs 16.08 s (`final_metrics_<id>.json`).
- Note: `pyproject.toml` sets `addopts = "-q"`; an extra `-q` hides the "N passed" line.

All nine original groups are complete. Anything further (MC regeneration, the unexplained worst Monte Carlo excursions, an innovation monitor) is outside the plan. Groups 10–12 (browser simulator) were added afterwards; see below.

## Group 10 — Browser sim: engine stepper + server API (M12) — done

**Delivered**
- **Specs.** `docs/specs/stepper.md` and `docs/specs/webapp_protocol.md`.
  - The stepper spec was extended after the ★ review with the kick cap, the failure latch and golden portability.
  - The protocol spec fixes the frame layout, including the display frame `C = DISPLAY_FLIP @ body_dcm_lvlh(q, r, v)`, the same frame as the Group 9 visualizer.
- **Golden equivalence gate** (figures-runner, captured *before* any edit).
  - The capture is `adcs/tests/data/closed_loop_golden.npz` (3.0 MB) with its helper, `adcs/tests/golden_closed_loop.py`. The helper provides `CASES`, `run_case`, `flatten`, `compare` (strict `tobytes` per array, `meta_json` string) and `environment_matches`.
  - There are 7 cases:
    - ideal+G7 and realistic+design, seeds 0 and 1, **2.0 orbits** (the plan said 1.5; seed 1 only switches at 1.64 orbits);
    - a 600 s `log_every=1` case;
    - ×1.1 truth inertia with dumping off;
    - `lvlh_wheels_fallback`, which adds 4 switches, 4 fallbacks and 16 dumping rows. It was captured from the HEAD module after the review showed that no other case reaches the fallback or dumping branches.
  - The capture was reproduced in a second process. Re-capture **only** from `git show 137bc1d:adcs/simulate.py`.
- **`adcs/stepper.py`** (py-implementer): `ClosedLoopSim`.
  - Methods: `step(n)`, `kick(dw)`, `snapshot()`, `result()`.
  - State: `k`, `t`, `done`, `remaining`, `state` (a copy), `failed`, and the read-only environment arrays `t_grid`/`r_eci`/`v_eci`/`sun_eci`/`lit`.
  - `MAX_KICK_RATE` = 30 °/s. A kick at the last sample is rejected. Any exception mid-step latches the sim as failed.
  - `run_closed_loop` is now a thin wrapper, so there is one code path. Eleven orphaned imports were removed from `simulate.py`.
  - The kick model is recorded as `assumptions.md` [B-25]: Δω with `q` and the total wheel momentum held, so ΔL = I_true Δω.
- **`webapp/`** (py-implementer):
  - `python -m webapp` (FastAPI + uvicorn, binds to 127.0.0.1:8000, shows a "frontend not built" page until Group 11);
  - `server/schemas.py` (`SessionParams`, commands, `build_sim_kwargs`);
  - `server/adcs_session.py` (`AdcsSession`, `SessionRegistry`, the pure `frame()` builder; streaming loop at ~30 Hz using `asyncio.to_thread`);
  - `server/app.py` (REST + WebSocket + static mount);
  - `server/tests/ws_smoke.py`.
  - `pyproject.toml` has a `[web]` extra (fastapi 0.162, uvicorn 0.54, httpx 0.28; websockets 17.4 comes with uvicorn), and `webapp` is in `testpaths`. Every webapp test module importorskips fastapi.
- **Tests: 1742 passing** (242 s, `OPENBLAS_NUM_THREADS=1`). Of these, 118 are new in `test_stepper.py` and 188 in `webapp/server/tests/` (`test_schemas` 92, `test_adcs_session` 42, `test_app` 54).
  - Oracles: the golden file, run plain and in irregular chunks; scipy DOP853 free rotation, with a kick and a full inertia matrix; conservation of H; independent frame recomputation from `snapshot()`; the CLI's `run_closed_loop` (server defaults are bit-identical to `python -m adcs.simulate`).
  - The server suite passed 5 runs in a row after the disconnect fixes below.
  - There is one third-party warning: the starlette TestClient deprecates httpx.

**★ Review (math-reviewer): OK to proceed, no blockers.**
- The step body matches HEAD line for line. The reviewer's own byte comparison against HEAD covered the branches the golden misses (fallbacks, dumping, vector CSS, `duty=1`, explicit `mag_bias`) and found 0 differences.
- The kick physics are correct.
- All should-fix findings were addressed:
  - kick cap, because RK4 at 1 s loses energy at high rates (ΔE/E over 600 s torque-free: −1.2e-4 at 0.5 rad/s, −7.8e-3 at 1 rad/s);
  - failure latch;
  - golden skip on a version or platform mismatch;
  - fallback and dumping golden case;
  - weak tests replaced: the momentum oracle now integrates with I_true against DOP853; the kick-fallback test states and asserts x-wheel saturation (ΔL 3.65e-3 > h_max 1.77e-3) and has a 2.25 °/s counter-case without a fallback; `cfg` forwarding is checked with a 550 km orbit.

**Bugs found by the server tests and fixed (main session)**
- **`_receive`.** It awaited its cancelled helper futures, which let a `CancelledError` escape when the ASGI server cancelled the scope after a disconnect. About 8–10 of the 54 `test_app` tests failed per run, a different set each time.
- **`serve()` cleanup.** It cleared the client only after an await, so a cancellation left the session marked connected. Such a session could never be evicted and caused permanent 429s.

**Measured (exit criteria)**
- **Engine** (`ClosedLoopSim`, realistic+design, 2 orbits, one core): 594–595 steps/s, i.e. **≈ 595× real time**; ideal+G7 665–715×.
  - The refactor costs no speed. HEAD and the wrapper alternated in one process ran within 3 % of each other, the wrapper slightly faster.
  - **The ≥ 500× target holds only when the laptop runs at full speed.** The same run measured 300–390 steps/s a few minutes later with the machine idle (2 % load), so it is run-to-run CPU variation (boost/power state). The first golden capture's 357–480 steps/s was also measured while another agent was using the CPU.
  - The profile shows no stepper overhead. The time goes to the MEKF (30 %), disturbance torques (29 %, aero 17 %), RK4 (20 %) and albedo. `np.cross` on 3-vectors costs about 26 % cumulative, mostly argument-handling overhead. A byte-preserving speed-up (swap in `_cross`) is possible but not done.
- **Server streaming** (`python -m webapp`, client in a separate process, default session, 10 s per speed setting):
  - at 100×: **30.1 Hz, 100×** achieved;
  - at 500× and 2000×: saturates at **≈ 400×, ≈ 10 Hz**. The engine alone measured 350–390 steps/s over the same sim span just afterwards, so streaming adds no measurable overhead. When the CPU cannot keep up, frames drop to about 10 Hz because each batch is capped at ~0.1 s of compute (spec §3.3), and `speed_actual` reports the shortfall.
  - Building a 15-orbit session takes 1.3 s.
- **Scripted WebSocket client** (`python -m webapp.server.tests.ws_smoke`, server and client in one process): OK, 26–31 frames in 2 s at 330–370×, `t` strictly increasing. It reads lower than the separate-process benchmark because the client competes with the server for the GIL.

**Decisions and notes**
- The speed setting survives `reset`. The `speed` and `play` commands send no frame (only state-changing commands do). `speed_actual` is 0 while paused.
- The orbit ring (`info.orbit_track_eci_m`) closes with a copy of its first point. With J2, a real one-period position does not close. This is display only.
- `result()` after a failure latch returns only the rows of fully committed `step()` chunks.
- A `kick` is applied to the pending state. Its own sample's gyro, estimator and controller do not see it; the next gyro sample does (spec §2).
- Not tested: the `n_max` batch cap and the dropped-budget behaviour. Both depend on wall-clock timing, and the spec gives no observable bound.
- Process note: two agents killed stray Python processes broadly (`pkill -f python`, `taskkill /IM python.exe`) during the session. All tests were re-run afterwards. Future agent prompts should forbid broad process kills.
- The 27 report files that were already modified before this group (Group 9 smoke-test regeneration leftovers, line endings / last-digit diffs) are **not** part of the Group 10 commit.
