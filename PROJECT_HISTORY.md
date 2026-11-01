# Project History — Everything Built, and Why

A complete narrative of the work in this repository: what was built, in what order, the reasoning behind each design decision, the numbers measured, the bugs found, and what is still open.

It is assembled from `PROGRESS.md`, `CONVENTIONS.md`, `assumptions.md`, `README.md`, `docs/IMPLEMENTATION_PLAN.md`, `docs/specs/*` and the git history (19 commits, 2026-10-05 to 2026-10-31). It does not re-run anything. Every number below is one the repo already records, with its source file. Where the records say something is unverified, unexplained or not done, this document says so too.

---

## 1. The big picture

The repository contains **two linked, pure-Python projects** that share one core of quaternion math and an error-state Kalman filter:

| | Project A — Sensor fusion | Project B — CubeSat ADCS |
|---|---|---|
| Goal | Estimate a drone's position, velocity and attitude from raw sensors | Simulate a 3U CubeSat that detumbles, then points at Earth, using only its own noisy sensors |
| Data | Three **real** PX4 flight logs (`.ulg`) | **Simulated** (orbit, environment, sensors, actuators) |
| Filter | 15-state error-state EKF (ES-EKF) | 6- or 9-state multiplicative EKF (MEKF) |
| Sensors | IMU, GPS, barometer, magnetometer | Magnetometer, six coarse sun sensors, gyro |
| Judged against | PX4's onboard EKF2 (a *reference*, not truth) | Simulated truth |
| Plus | | Control (B-dot, reaction-wheel PD, momentum dumping), Monte Carlo, design document |

On top of both sits a **browser simulator** (FastAPI + Vite/TypeScript/Three.js) that runs the CubeSat closed loop live and replays the PX4 logs with a 3D drone and plots.

Terminology: this is **attitude** (orientation) determination and control. The word "altitude" appears only for genuine height (barometric altitude, orbit altitude). A sweep at the end of Group 9 confirmed every "altitude" hit is genuine height.

### Why two projects on one core?
Both problems are the same estimation problem in different clothes: a rigid body with a gyro, noisy vector observations, and a bias to estimate. Writing the quaternion algebra, frame transforms and error-state filter once in `common/` and testing it hard (Group 1) means both projects inherit trusted math. Project A shows the filter on real, messy data. Project B shows it inside a feedback loop, where estimation errors feed back into control.

### The working method
The project was run as a **plan of 12 groups**, one at a time. At the end of each group the rules were:
1. The full `pytest` suite is green.
2. `PROGRESS.md` and the plan's status board are updated.
3. One git commit is made.
4. Stop and wait for the user's next prompt.

Work was delegated to specialised subagents defined in `.claude/agents/`:

| Agent | Model | Role |
|---|---|---|
| `py-implementer` | Sonnet | Implement one fully specified Python module and run its tests |
| `test-author` | Sonnet | Write pytest suites **from the written spec, not the implementation**, using independent oracles |
| `data-wrangler` | Sonnet | PX4 log download/parse |
| `figures-runner` | Sonnet | Run sims/sweeps, produce figures and metrics JSON; never edits algorithm code |
| `doc-writer` | Sonnet | README, assumptions, progress notes from measured outputs |
| `math-reviewer` | Opus, read-only | Review math-critical code at ★ checkpoints; returns findings, never edits |
| `web-implementer` | Sonnet | Browser simulator frontend/server (added in Group 11) |

Two rules shaped the quality of everything:
- **Spec first.** A spec in `docs/specs/` is the contract between the implementer and the test-author. The tests are written from the spec by a different agent, so the tests are not just a mirror of the code.
- **Every reported number must trace to** a conservation law, an independent library oracle (scipy, sgp4, ppigrf, chi²), the PX4 log's EKF2, or a cited source. Assumptions go in `assumptions.md` with an ID (A-xx for Project A, B-xx for Project B, G-xx for general), and any value in code with no entry there is considered a bug.

---

## 2. Fixed conventions (`CONVENTIONS.md`)

Decided in Group 1 and never changed silently, because a sign or frame slip here would corrupt everything downstream.

- **Units:** SI internally (m, s, kg, rad, N·m, A·m², tesla). Degrees appear only in plots and tables. Orbit state in metres (the `sgp4` library returns km and is converted at the boundary).
- **Arrays:** time on axis 0. Vector series `(N,3)`, quaternion series `(N,4)`, covariance series `(N,n,n)`.
- **Quaternions:** scalar-first `[w,x,y,z]`, **Hamilton** product, representing the **body-to-reference** rotation: `v_ref = q ⊗ [0,v_body] ⊗ q*`. This matches `scipy.spatial.transform.Rotation` and PX4's `vehicle_attitude.q`, so no conjugations are needed at the PX4 boundary. Unique representative: `w ≥ 0`. Kinematics `q̇ = ½ q ⊗ [0, ω_b]`; discrete step `q_{k+1} = q_k ⊗ exp(ω dt)`.
- **Error-state filtering:** the covariance never contains a 4-element quaternion (it would be singular). The attitude error is a 3-element rotation vector δθ in **body axes** (right/local perturbation): `q_true = q_est ⊗ exp(δθ)`. After every update the error is **injected** into the nominal state and **reset** to zero, with the reset Jacobian `G ≈ I − ½[δθ×]`. Covariance update uses the **Joseph form**, symmetrised. Consistency is judged by **NIS** (normalised innovation squared) against χ² bounds.
- **Frames:** ECI (approximated by a GMST rotation from ECEF; precession/nutation neglected, < 0.5° in 2026), ECEF (WGS-84), LLA, NED (drone navigation, origin at first GPS fix), PX4 FRD body, spacecraft body (z along the 34 cm axis), LVLH (z = nadir).
- **Project A simplification:** NED is treated as inertial (flat, non-rotating Earth). Earth rate (7.3e-5 rad/s) is below consumer MEMS gyro bias instability.
- **PX4 mapping (§6, finalised after the Group 2 review):**
  - one place (`sensor_fusion/loader.py`) converts PX4 units/fields;
  - GPS data are valid at `t − 0.11 s` because PX4 does not stamp receiver latency;
  - **three height datums are never mixed:** WGS-84 ellipsoidal (ours), MSL (EKF2's `ref_alt − z`) and ISA pressure altitude (baro);
  - baro is used only as a height *change* relative to the first second on the ground;
  - local gravity per log instead of 9.80665;
  - EKF2 is a reference only, and its in-flight yaw resets must be marked in every comparison.

---

## 3. Chronological history

### Group 1 — Foundations (M0) · commit `8d78ae7`, 2026-10-05

**Built**
- Repo scaffold: git, `.gitignore`, `.gitattributes` (LF), `pyproject.toml` (editable install), packages `common/`, `sensor_fusion/`, `adcs/`.
- Environment: `.venv` on Python 3.14.2 (numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2, pytest 9.1.1, pyulog 1.2.4, ppigrf 2.1.0, sgp4 2.27, requests).
- `common/quaternion.py`: the algebra plus the shared `inject_attitude_error` and `attitude_reset_jacobian`, which are the single implementation of ES-EKF injection and reset for both projects.
- `common/frames.py`: WGS-84 LLA/ECEF/NED, GMST, ECI/ECEF.
- `common/ekf.py`: a generic error-state EKF core with Joseph form, block and sequential updates, gating, NIS logging, Van Loan discretisation and χ² bounds.

**Why it matters.** Everything else stands on this. The quaternion code is checked against scipy; frames against closed-form WGS-84 values and the Vallado GMST example (matches to 7.6e-8°); the EKF against a textbook Kalman filter, the discrete algebraic Riccati equation steady state, analytic Van Loan, Monte Carlo NIS/NEES, and a nonlinear pendulum.

**Result:** 445 tests passing.

**★ Opus review:** no blockers. A mutation test injected 26 deliberate bugs and the tests caught 22; the 4 escapes were float-equivalent or immaterial. Follow-ups: the Joseph-form test was made to actually discriminate it from the short form, the attitude inject/reset helpers were shared, LVLH/NED DCM↔quaternion link tests were added, and the "inject after every update" contract was documented.

### Group 2 — Project A data and dead reckoning (M1, M2) · commit `ec49fa7`, 2026-10-05

**Built**
- `download_logs.py`: three pinned Flight Review logs (retries, ULog magic-byte check; SHA-256 later confirmed identical on a fresh download).
- `loader.py`: pyulog → time-aligned SI arrays in a `FlightLog` (`imu/mag/baro/gps/ekf2/info`), with cleaning, gap report and armed-window crop. `interp.py` (linear and slerp). `inspect_logs.py` plus 15 inspection figures. `sensor_fusion/data/NOTES.md`.
- `dead_reckoning.py`: `strapdown` (trapezoid gyro, mid-interval attitude for the accel rotation, vectorised quaternion prefix-scan) and a static initial attitude from accel + mag.
- `common/frames.py::normal_gravity` (Somigliana + free-air), added after the review.

**The three logs**

| Role | ID prefix | Hardware / firmware | Window | Notes |
|---|---|---|---|---|
| hover | `f4e81624` | FMU_V6C, v1.17.0 | 280 s | slow position-mode flight |
| aggressive | `7cd0a812` | FMU_V6C, v1.17.0 | 169 s | stabilized, gyro to 386 °/s, ends in an impact |
| other | `78808b7c` | FMU_V5, v1.14.0 | 674 s | long, 9 m/s, RTK GPS 2 Hz, old integer message format |

**Why dead reckoning first?** It's the baseline that motivates the filter. Integrating the raw IMU with no bias removal drifts enormously:

| Log | Position error @ 10 s | @ 60 s | Growth exponent |
|---|---|---|---|
| hover | 10.8 m | 695 m | 1.93 |
| aggressive | 9.1 m | 320 m | 2.20 |
| other | 2.4 m | 256 m | 2.77 |

An exponent near 2 points to accelerometer-bias-dominated error (bias × t²/2). 2.77 on the long log suggests gyro bias coupling gravity into the horizontal axes. This split was *not* tested at this point; the filter estimates the biases in Group 3. Using nominal G0 instead of local gravity shifts position by exactly ½|Δg|t² (51 m at 60 s on hover), real but small.

**★ Opus review of the PX4 mapping.** Confirmed core mapping independently (R(q_EKF2)·f_b ≈ −g down; gyro vs EKF2 attitude rate rms 0.013–0.022 rad/s with zero lag). Found and fixed:
- a red baro test (EKF2 height is GPS-referenced on two of three logs, and GPS height drifts ~9 m on the aggressive log, so the criterion changed);
- a wrong EKF2-origin claim in the notes;
- the loader dropping GPS MSL height (`gps.alt_msl` added);
- use of G0 rather than per-log gravity;
- mag/baro not stamped with `timestamp_sample`;
- GPS latency is measurable (≈ 0.10–0.18 s; `EKF2_GPS_DELAY` = 110 ms).

**Decisions handed to Group 3**
- Mag: fuse **tilt-compensated heading only**, σ ≈ 3–5°.
- Baro: height change vs. first second; σ ≈ 1.5 m.
- GPS delay: offline, so re-stamp `t − 0.11 s` and process in time order (exact; resolves the earlier "buffer-and-replay vs inflate R" question).
- Local gravity per log (9.7784 / 9.7859 / 9.8119 m/s²).
- Each log has one EKF2 pure-yaw reset (−4.92° @ 16.1 s, −0.98° @ 4.2 s, −3.53° @ 7.4 s) that must be handled in yaw statistics.

**Result:** 605 tests.

### Group 3 — Project A 15-state ES-EKF (M3) · commit `92e3c4b`, 2026-10-05

**Built**
- `sensor_fusion/ekf_drone.py`, a 15-state error-state EKF on the shared core.
  - **Error state** `[δp, δv, δθ (body, right), δb_g, δb_a]`.
  - Nominal propagation identical to `strapdown`.
  - First-order Φ/Qd with a Van Loan option.
  - **Updates:** GPS position and velocity (delay-corrected, applied at the nearest IMU sample), barometer down-position, tilt-compensated magnetometer heading.
  - Inject and reset after every update.
  - Static initialisation by accel levelling + mag heading + IGRF declination; NED origin at the first GPS fix.
  - `EkfConfig.increment` stages the build: 1 = bias states frozen, GPS+baro; 2 = adds mag; 3 = full bias estimation.
- `synthetic.py`: an analytic figure-8 / hover / static truth generator with instantaneous IMU, noise, bias random walk, delayed GPS, baro and mag. It lets the filter be tested against *known truth*, the one thing real logs cannot offer.
- `report/ekf_study.py`: 3 logs × 3 increments with figures.
- `assumptions.md` A-03 (IMU noise taken from PX4's own logged EKF2 parameters, converted to densities), A-04 (R values), A-05 (IGRF declination).

**Key verification:** NEES Monte Carlo = 9.86 for a 99 % band of 5.9–12.8 (10 seeds); GPS NIS 0.958 inside the 95 % bounds.

**★ Opus review. No blockers, but one real bug.** The magnetometer heading used only the yaw Jacobian, although the measurement is *tilt-compensated using the estimated tilt*, so tilt error leaks into the residual (× tan I, inclination). Leakage factors: 0.95 / 0.54 / 2.32 on hover / aggressive / other. The fix was a full residual Jacobian `mag_heading_jacobian`. The synthetic NEES test independently confirmed it: it failed before (NEES ≈ 14) and passed after.

**Flight results** (RMS difference vs EKF2 after 5 s; EKF2 is a reference): bias estimation (increment 3) cut horizontal position and velocity differences by about 3× on hover and aggressive, and tilt by up to 3× on aggressive (1.10° → 0.37°). Gyro bias agrees with EKF2's `estimator_sensor_bias` to ≈ 0.001 rad/s. Runtime 9–32 s per run (≈ 15 k IMU steps/s).

**Findings carried forward**
- Hover's 1.35° static tilt offset equals the tilt implied by EKF2's own accel bias. It is *not* a convention error.
- Hover horizontal accel bias is **unobservable** (only an 8° yaw range). It must be reported with ±σ and not tuned away.
- Hover yaw is −5…−8° off EKF2 once mag is fused, from uncorrected hard iron.
- Baro NIS ≈ 38 on aggressive (GPS vertical drift vs baro).
- With a frozen 0.01 rad/s gyro bias, increment 2 was overconfident, which is the motivation for increment 3.

**Result:** 664 tests.

### Group 4 — Project A tuning and writeup (M4) · commits `dc2d128`, `6738c0e`, 2026-10-06 · **Project A complete**

**Built**
- `metrics.py` (extracted bit-identically from the study script: `compare_to_ekf2`, `nis_summary` with χ² bounds, `bias_summary`, plus a new `consistency_score` = RMS of ln(mean NIS/dof)); `report/ekf_figures.py`; the CLI `python -m sensor_fusion.run_flight <role|id|all>`.
- `report/tuning_sweep.py`: 120 runs (40 configs × 3 logs) in 500 s on 20 workers.
- `report/height_diagnostics.py`; `report/writeup.md` (≈ 3,300 words with tables).
- `assumptions.md` A-13 (tuning procedure and decision).

**The honest tuning result.** The sweep's winner, `joint`, improved the total score from 1.816 to 1.717. The ★ review showed that this came from the mag NIS, where a too-large R was being offset by a too-small Q. It also made the gyro bias overconfident versus EKF2's estimate. So it was **not adopted**. Only one change was kept: `gps_pos_floor_v` 0.5 → 0.125 m, justified by the RTK receiver reporting epv of 0.010–0.026 m (total score 1.816 → 1.770). This deviates from the pre-set rule's winner after seeing the results, and A-13 records it as post hoc.

**Final numbers** (increment 3, RMS difference to EKF2 after 5 s)

| Log | posH [m] | posD [m] | vel [m/s] | tilt [°] | yaw [°] |
|---|---|---|---|---|---|
| hover | 0.050 | 1.040 | 0.105 | 1.05 | 5.78 |
| aggressive | 0.055 | 0.753 | 0.090 | 0.37 | 1.38 |
| other | 0.255 | 0.929 | 0.216 | 0.56 | 1.80 |

NIS ratios (mean NIS / dof; 1 = consistent): mag 0.05–0.34, GPS velocity 0.08–0.30, hover GPS position 0.27, i.e. the noise models are conservative. Aggressive-log baro ≈ 39, consistent with GPS height drifting ≈ 7.4 m against its own integrated velocity and the baro.

**Caveats the writeup keeps**
- Both filters fuse the same GPS, so the small horizontal difference is not an accuracy claim.
- The posD column is not an accuracy measure.
- `fraction_inside_95` is insensitive to over-conservative R.
- Aggressive-log accel bias disagrees with EKF2 (y axis opposite sign, ≈ 24σ); reported, not explained.

**Not done:** GPS-delay sensitivity, a baro-offset state, a GPS lever arm.

**Result:** 722 tests.

### Group 5 — Project B physics (M5, M6) · commit `53987dc`, 2026-10-06

**Built** (all verified before any control was attempted)
- `adcs/config.py`: `SimConfig`/`SpacecraftConfig`/`OrbitConfig`. Epoch 2026-03-20 12:00 UTC, 400 km / 51.6° circular orbit, 3U 4 kg box with inertia diag(0.041867, 0.041867, 0.006667) kg·m², CoM offset (5, 5, 10) mm, residual dipole (1, 1, 1) mA·m², Cd 2.2. Written by the main session because every other module depends on it.
- `orbit.py`: Kepler ↔ state, two-body + J2, DOP853 propagation with a dense `Ephemeris`, an SGP4 comparison.
- `environment.py`: Sun position (Vallado), cylindrical eclipse shadow, β angle, exponential atmosphere, IGRF magnetic field through ppigrf rotated to ECI, and a cubic-spline `FieldTable` cache so closed-loop sims don't call ppigrf each step.
- `dynamics.py`: state `[q, ω, h_w]` with `I ω̇ = −ω×(Iω + h_w) − u_w + τ_ext`, `ḣ_w = u_w`; RK4 and adaptive integrators; conserved-quantity helpers.
- `disturbances.py`: gravity gradient, residual magnetic dipole, flat-plate aerodynamic drag with a co-rotating atmosphere.

**Measured**

| Check | Result |
|---|---|
| J2 orbit vs SGP4 (ISS TLE) | max \|Δr\| 103 m after 1 orbit, 2.34 km after 1 day (drag, J3, J4 not modelled) |
| Two-body / J2 conservation over 10 orbits | energy 3.4e-12 / 3.5e-12, h_z 1.7e-12 relative |
| Eclipse fraction | 0.3905 (analytic β=0 value 0.3901) |
| \|B\| at 400 km | 18.98 / 34.98 / 49.31 µT (min/mean/max over 3 orbits) |
| Free tumble, 10 orbits | \|H_i\| drift 5.9e-14, kinetic energy 1.1e-13 |

**Two plan assumptions turned out to be wrong, and were corrected rather than papered over**
1. **The plan's 25–45 µT field gate does not hold.** Over the whole ±51.6° band at 400 km, the field spans 18.98 µT (South Atlantic Anomaly) to 54.13 µT. Tests were revised to 15–55 µT with an independent grid-scan oracle for the floor. Hardware should be sized for ≈ 19–54 µT.
2. **"Gravity gradient dominates" does not hold** for this 3U. Aerodynamic torque is the largest source in all 7 tested attitudes (it scales with CoM offset and density, a ×10 swing with solar activity). With Ix = Iy the satellite is also only marginally stable in yaw under gravity gradient. The test became an order-of-magnitude sanity check.

**★ Opus review:** no blockers; 21 of 21 sign/frame mutants caught (GMST direction, NED DCM, gravity-gradient sign, wheel reaction sign, kinematics side, LVLH transpose, aero sign, J2, obliquity, β, shadow, …).

**Notable modelling choices:** `h_w` is the *total* wheel momentum, with the rotors' spin-axis inertia excluded from `I`. Aero uses spherical height for density (≤ 25 % density error). Solar radiation pressure is neglected (hand estimate 1.5e-9 N·m).

**Result:** 898 tests.

### Group 6 — Project B control with perfect knowledge (M7) · commit `2b08dcd`, 2026-10-06

Control was designed first with *perfect* knowledge of the state, so any later degradation can be attributed to estimation.

**Built**
- `actuators.py`: magnetorquers (0.2 A·m² per axis, ISIS iMTQ-class, 1 s control period, duty 0.8 so the magnetometer samples with coils off) and three reaction wheels (CubeSpace CW0017: 0.23 mN·m, 1.77 mN·m·s at 8000 rpm). Torque limiting and speed limiting that a held command cannot cross.
- `control.py`:
  - **B-dot detumble** on the field direction, gain `k = 2n(1+sin i)J_min` = 2.69e-5 N·m·s (Avanzini & Giulietti 2012).
  - **Quaternion-error PD** for nadir pointing (ωn 0.1 rad/s, ζ 0.7, 0.5 °/s slew limit) with gyroscopic and reference-rate feed-forward.
  - **Cross-product momentum dumping** `m = (k/|B|²) h × B` with hysteresis, feeding magnetorquer torque forward to the wheels.
- `simulate.py`: the perfect-knowledge runner (ZOH per 1 s period, RK4 0.8 s coils on + 0.2 s off), modes `detumble` / `pointing` / `auto`.

**Measured (perfect knowledge)**

| Check | Result |
|---|---|
| Detumble 10 °/s → < 0.5 °/s | 0.88–1.68 orbits (mean 1.36); target 1–3 ✅ |
| Nadir pointing, 15 orbits | max 0.078°, RMS 0.053° (target < 1°) ✅ |
| Dumping off | y wheel saturates at 3.73 orbits and attitude is lost (179.9°): demonstrates dumping is needed ✅ |
| No-disturbance test | 0.0013° after 1000 s, H_i drift 2e-12 |

**Findings**
- B-dot gain kept at the cited value, though 2×/4× detumble faster (0.73 / 0.43 orbits) with perfect knowledge. A higher gain would amplify magnetometer noise in the finite-difference Ḃ, so this was revisited in Group 8.
- Dumping is effectively continuous (active 92 % of the time): it balances the secular aero torque (~8.8e-8 N·m), so the hysteresis off-threshold is never reached.
- No sub-stepping convergence check of the 1 s ZOH was run.

**Result:** 1054 tests.

### Group 7 — Project B attitude determination (M8) · commits `c7ed854`, `7eff545`, 2026-10-06

**Built**
- `sensors.py`: PNI RM3100 magnetometer (σ 15 nT/axis); ADXRS453 gyro (ARW 0.65 °/√h, RRW from 16 °/h bias stability, turn-on bias σ 0.1 °/s); six cosine coarse sun sensors (Tensor Tech CSS-10, σ 5° 1σ, validity threshold 0.6). Sampling model: everything at 1 Hz, and the gyro sample is the *exact* mean rate over the interval plus bias plus ARW, so the filter's propagation is exact for the truth.
- `estimation.py`: `triad` and its covariance (derived by hand, verified by Monte Carlo to 1.5 % of λ_max); the **MEKF** with error state `[δθ, δb]` on the shared EKF core, closed-form Φ/Qd (equals Van Loan to 1e-11), vector updates for magnetometer and sun sensors, TRIAD initialisation at the first valid sun sample.

**Measured (matched noise models, so an ideal-sensor floor)**

| Check | Result |
|---|---|
| Nadir pointing RMS ‖δθ‖, sunlit / eclipse | 0.346° / 0.731° |
| Attitude NEES (dof 3) | 3.00 (pointing), 3.21 (detumble) |
| 20-seed MC: NEES / NIS mag / NIS sun | 2.95 ± 0.15 / 3.00 ± 0.02 / 3.00 ± 0.02 |
| Bias error 0.185 °/s → | 0.0010 °/s within 0.25 orbit |
| TRIAD RMS vs predicted | 4.89° vs 4.91°; MEKF sunlit is ~14× better |
| Eclipse σ_att entry → exit | 0.29° → ≈ 1.05° (×3.6–3.8) |

**★ Opus review: no blockers.** Verified Φ against the nonlinear truth, H/residual/reset signs, unit conversions and whiteness of innovations (lag-1 autocorrelation −0.001). A doc-writer "discrepancy" between 0.015 °/s/√Hz and 0.65 °/√h turned out to be wrong (ARW = ND/√2 reconciles them) and was corrected.

**The review's most important finding: these results are optimistic.** The filter used the truth field as its reference and had no magnetometer bias, only white CSS noise. Scratch runs: an unmodelled 300 nT mag bias gave NEES ≈ 815; a hard ±60° CSS field of view gave 8.1° sunlit RMS; crude albedo gave 6.4°. That became the work list for Group 8.

**Result:** 1177 tests.

### Group 8 — Project B closed loop, Monte Carlo, design document (M9) · commit `e97d443`, 2026-10-06 · **Project B complete**

**Built**
- **Sensor realism:** magnetometer bias with Gauss-Markov drift, a degree-8 truncated onboard field model (≠ truth), hard CSS field of view, Earth albedo (plate-to-sphere view factor, ray-Monte-Carlo checked). All defaults still reproduce Group 7.
- **Estimator hardening:** optional magnetometer-bias states (9-state MEKF), per-diode CSS updates gated on the *predicted* cosine (gating on the measured current biased NEES to 3.3–4.8), onboard albedo compensation with albedo uncertainty in R, a TRIAD minimum-angle guard (about-field σ = 0.0617/sin α), and a public `MEKF.step` shared by batch and closed-loop runs.
- **`run_closed_loop`:** sensors in the loop, gyro generated from consecutive true attitudes, MEKF against the onboard field, mode machine on the *estimate*, fallback on estimated rate, controller on q̂/ω̂, truth integrated with a perturbed inertia.
- **`monte_carlo.py`:** `draw_run`, `run_one`, `run_campaign`, `aggregate`, `check_requirements`.
- A ≈ 6,500-word `design_document.md`, plus `assumptions.md` B-21…B-24.

**Measured** (10 °/s, 6 orbits, seeds 0–4, means)

| Case | Nadir RMS | Nadir max | Knowledge RMS | NEES |
|---|---|---|---|---|
| Perfect knowledge | 0.054° | 0.068° | – | – |
| Ideal sensors + Group 7 filter | 0.53° | 1.92° | 0.51° | 3.05 |
| **Realistic sensors + Group 7 filter** | **24.4°** | 57.2° | 24.1° | ≈ 6400 |
| Realistic sensors + design filter | 0.99° | 3.02° | 0.96° | 13.7 |

**Monte Carlo** (100 runs per campaign, 2–10 °/s, ±10 % truth inertia, base seed 2026)

| Median / p95 / max | Design campaign | Ideal campaign |
|---|---|---|
| Detumble [orbits] | 1.22 / 1.88 / 2.29 | 1.21 / 1.63 / 1.89 |
| Nadir RMS [°] | 0.962 / 1.25 / 1.46 | 0.503 / 0.635 / 0.733 |
| Nadir max [°] | 3.15 / 4.44 / 5.19 | 1.84 / 2.90 / 3.59 |
| All requirements met | **99/100** (run 54 fails the 5° max at 5.19°) | 100/100 |
| 1° RMS goal | 58/100 | 100/100 |

The requirement thresholds (3-orbit detumble, 4-orbit acquisition, 5° max, 2° knowledge RMS) are **assumed values fixed before the campaign** (B-24), not derived mission needs.

**Findings stated plainly**
- **The naive filter fails silently.** The Group 7 filter under realistic errors reaches 24° true error while the internal `success` flag stays true, because the rate-based fallback cannot see a wrong attitude. An innovation monitor is recommended but not implemented.
- The design filter's tuning was time-boxed and post hoc (σ_mag 100 nT, σ_albedo 0.5). It remains overconfident (NEES ≈ 14). The breakdown shows the field-model error is the cause (NEES 2.3 without it), and the bias states matter most (without them NEES 157, nadir RMS 1.45°).
- Worst MC excursions are knowledge errors outside the filter's 3σ while control error stays ≤ 0.36°; 4 of 5 happen in sunlight 660–825 s after eclipse exit. Cause not isolated further.
- Late initialisation: 14 of 100 design runs initialise TRIAD after 1000 s, and TRIAD can't re-initialise in nadir pointing at β ≈ 0.
- Not done: closed-loop gating, FDIR, an albedo state, orbit-knowledge error, dumping-threshold revisit, sub-stepping convergence.

**Result:** 1388 tests.

### Group 9 — Packaging, README, smoke test, 3D visualizer (M10, M11) · commit `137bc1d`, 2026-10-18; follow-up `6e010e0`, 2026-10-31

**Built**
- `README.md` rewritten with headline tables, layout, setup, reproduction commands with measured wall times, gallery and known limitations.
- `adcs/visualize.py` (spec `visualizer.md`): a display frame with LVLH rotated 180° about x (zenith up, nadir = (0,0,−1)), a time-warped frame schedule chosen by bisection on a target angle, and MP4/GIF rendering. Output: `closed_loop_animation.mp4` (0.80 MB) and `.gif` (1.5 MB). 48 tests. Visual aid only; it adds no numbers.

**Reproducibility smoke test** in a *fresh venv* (Python 3.14.2): log downloads have identical SHA-256, the suite passes, and every regenerated report reproduces (estimation ≤ 1.75e-12 relative; closed loop bit-identical).

**Follow-up:** the Monte Carlo was reproduced in a second fresh venv (slightly newer numpy 2.5.4, pyulog 1.3.0, pandas 3.1.0). Design (1109 s) and ideal (892 s) campaigns gave **bit-identical non-timing numbers**, with `mc_worst_runs.json` and every `mc_*.png` byte-identical. This closed the last open reproducibility item.

**Result:** 1436 tests. All nine original groups were complete at this point.

### Group 10 — Browser sim: engine stepper and server API (M12) · commit `d13035e`, 2026-10-30

Goal: make the closed-loop simulation steppable so a browser can drive it live.

**Built**
- **Golden equivalence gate, captured *before any edit*.** `adcs/tests/data/closed_loop_golden.npz` (3 MB), 7 cases covering ideal+G7, realistic+design, a 600 s full-rate case, a ×1.1 inertia case with dumping off, and a fallback/dumping case. The refactor must reproduce it **byte-for-byte**. This is how a risky refactor of the heart of the simulator was made safe.
- `adcs/stepper.py`: `ClosedLoopSim` with `step(n)`, `kick(dw)`, `snapshot()`, `result()`. `run_closed_loop` becomes a thin wrapper so there is one code path. A "kick" (instant Δω) is capped at 30 °/s and models ΔL = I_true Δω with wheel momentum held (`assumptions.md` B-25). Any mid-step exception latches the sim as failed.
- `webapp/` package: `python -m webapp` starts FastAPI + uvicorn on 127.0.0.1:8000. It has REST + WebSocket streaming at ~30 Hz (`asyncio.to_thread`), `SessionParams` schemas, a `SessionRegistry` and a pure `frame()` builder that defines the wire format, including the display frame `C = DISPLAY_FLIP @ body_dcm_lvlh(q, r, v)`, identical to the Group 9 visualizer.

**Measured**
- Engine: ≈ 595× real time (594–595 steps/s) for realistic+design on one core. The ≥ 500× target holds only when the laptop runs at full boost; the same run measured 300–390 steps/s minutes later, so this is CPU run-to-run variation. The profile shows time goes to the MEKF (30 %), disturbance torques (29 %) and RK4 (20 %), with no stepper overhead.
- Streaming: 100× achieved exactly at 30.1 Hz; at 500×/2000× it saturates at ≈ 400× / 10 Hz, and `speed_actual` reports the shortfall.

**★ Review: OK, no blockers.** The reviewer byte-compared the step body against HEAD for branches the golden misses and found zero differences. Should-fix items done: kick cap (RK4 at 1 s loses energy at high rates), failure latch, golden portability skip, fallback and dumping golden case, weak tests replaced.

**Bugs found by the server tests (main session fix)**
- `_receive` awaited its cancelled helper futures, so a `CancelledError` escaped on disconnect, making about 8–10 of 54 `test_app` tests fail per run, a different set each time.
- `serve()` cleared the client only after an await, so a cancellation left a session marked connected, never evictable, causing permanent 429s.

**Process lesson:** two agents killed stray Python processes broadly (`pkill -f python`). Future prompts forbid broad kills.

**Result:** 1742 tests.

### Group 11 — Browser sim: ADCS frontend (M13) · commit `b1b8710`, 2026-10-31

**Built**
- Spec `frontend.md`, which fixes **one** mapping into Three.js (Y-up): `(x,y,z) → (x, z, −y)`. One mapping serves the ECI scene and the LVLH display scene, and it coincides with `SphereGeometry`'s UV layout, so the Earth mesh needs `rotation.y = gmst` with no offset.
- `web-implementer` agent definition, carrying the process-hygiene rules.
- Earth texture: NASA Blue Marble (Reto Stöckli, public domain), 3840×1920, fetched through the Wikimedia Commons API with JPEG magic bytes and dimensions verified; provenance in `manifest.json` and `CREDITS.md`.
- `webapp/frontend/` (Vite 8, TypeScript strict, three.js, uPlot, vitest):
  - `OrbitView`: textured rotating Earth, a real terminator from the Sun direction, the orbit ring and a trail;
  - `SatView`: the 3U to scale, an axis triad, an estimate "ghost", nadir/Sun/B arrows, an ECLIPSE label and wheel bars;
  - `Panels`: parameter form, transport with a log-speed slider, a kick panel, readouts, a mode/eclipse/event timeline and four plot strips;
  - a history ring buffer, WebSocket client and frame math.

**Tests:** 68 vitest. The oracle is Python-generated fixtures (`common.quaternion.to_dcm`, `eci_to_ecef`, real `ClosedLoopSim` frames), so the TypeScript math is checked against the Python truth. The Earth-texture test samples a real `SphereGeometry` mesh by nearest UV at 12 ground points × 3 GMST values, and sensitivity checks prove that the wrong signs (`−gmst`, `gmst+90°`) *fail*.

**Browser check** (claude-in-chrome; GIF `webapp/docs/adcs_browser_check.gif`): realistic+design, 6 orbits, 10 °/s. TRIAD init at 269 s (matches the golden `real_design_s0`), POINTING by 1.23 orbits (golden: 6491 s). A +5 °/s kick about body x drove the x wheel to 100 % speed, and the sim fell back to DETUMBLE and re-acquired POINTING about one orbit later with nadir error back to 0.30°. No console errors.

**Fixed during the check:** the fallback marker hidden under the kick marker (verified by reading canvas pixels); a stale-bundle caching problem, where Vite's hashed assets are now served immutable and everything else `no-cache` (5 tests).
**Not a bug:** Chrome throttled the automation window to ~1 Hz rAF behind other windows, verified by a probe showing an idle main thread.

**Result:** 1747 pytest + 68 vitest.

### Group 12 — Browser sim: PX4 replay and docs (M14) · commit `02f2deb`, 2026-10-31

**Built**
- `webapp/server/px4_replay.py`: `build_replay` runs `run_filter` once and returns ours vs EKF2 (NaN outside EKF2's span), differences (using the `metrics` helpers), decimated NIS with per-sample χ² 95 % bands, EKF2 reset times and the 5 s init window. `ReplayStore` keys a disk cache (`webapp/.cache/`) by the sha256 of cache version, id, rate, `EkfConfig`, and `.ulg` size/mtime, with atomic writes and one build per log at a time. Endpoints `GET /api/px4/logs`, `POST /api/px4/replay/{log}`, plus gzip.
- Frontend: `ned.ts` (NED → Three.js, a proper rotation, fixture-tested), and `src/px4/*` for a 3D drone, path view, replay clock, plots and the replay page. Tabs with a `#adcs` / `#px4` hash.
- README "Interactive simulator" section, `CLAUDE.md` update.

**Measured**
- First builds: hover 14.1 s / 4.6 MB, aggressive 8.6 s / 2.9 MB, other 29.6 s / 11.1 MB. Cached + gzip: 0.15–0.61 s, 1.1–4.3 MB.
- **Replay vs the committed report agree** (hover tilt RMS 1.049 vs 1.049°, yaw 5.778 vs 5.778°; aggressive tilt 0.371 vs 0.366°; other horizontal 0.258 vs 0.255 m). Small differences come from the 50 Hz grid vs EKF2's own sample times. No implementation bugs found by the 52 new server tests.

**Browser check:** the extension wasn't connected, so a headless Chrome driven by `puppeteer-core` was used (installed only in the session scratchpad). All three logs load and play at 8×. Jumping to the previous EKF2 reset lands at 16.08 / 4.21 / 7.38 s, which are the report's reset times. Fixed: stale N/E labels after switching logs (three.js fires `removed` only on the removed root), and an EKF2 ghost drawn as edge outlines so it no longer hides our drone.

The replay adds **no numbers** to any report, and the page footer says so.

**Result:** 1799 passed + 1 opt-in skip, plus 107 vitest.

### After Group 12 — fixes and two visual redesigns · 2026-10-31

**`2b92871` — "Start session" looked dead.** A new ADCS session opened paused at t = 0. The page now sends `play` on the first `hello` (Reset still leaves it paused).

**A clarification, not a bug.** The vertical-position and hover-yaw differences that wander in the PX4 replay plots are the filter-vs-EKF2 differences already documented in the writeup. The replay reproduces the report statistics.

**`488e3e8` — "Colour blocks" theme** (`docs/specs/ui_theme.md`). Light and dark modes with a header toggle (default from the system setting, stored in `localStorage`), sunflower/periwinkle/mint region blocks, dark plum 3D screens, pill controls, a bundled Schibsted Grotesk font. Charts and timelines recolour live without losing the session. Data colours are tested for ≥ 3:1 contrast in both modes (several light-mode values were darkened to pass). 131 vitest.

**`e669a21` — "Flight-deck console" redesign** (`docs/specs/ui_console.md`, which supersedes the theme spec). Requested by the user: plots that rise over the 3D views instead of page scrolling, an expand button per plot, a compact top bar, consistent indicators, more accessible readouts, and a less playful, more technical look.
- **Look:** graphite/steel surfaces, amber as the one action colour, cyan/teal data, red alarms; 4 px corners with corner brackets; Barlow Semi Condensed + JetBrains Mono (bundled). Text contrast ≥ 4.5:1 and data/focus edges ≥ 3:1, tested in both modes.
- **Layout:** 40 px top bar with arrow-key tabs and a live status slot. The page never scrolls: the timeline + plots form a **sheet** raised by a scroll notch, PageUp/PageDown, the handle or a button (parallax + dim; instant under reduced motion). 3D zoom moved to Ctrl + scroll / pinch / middle-drag (the user's choice). Below 1100 × 600 px it falls back to normal scrolling.
- **Charts:** chart code extracted into `ui/chart.ts`, `adcs/charts.ts`, `px4/charts.ts`; an expand `<dialog>` (focus trap, Esc) with live data, drag-to-zoom that holds while data streams, and plain-language "how to read this" text.
- **Legends and readouts:** one DOM legend system with glyphs matching the drawn marks; grouped readouts with units and keyboard-accessible "?" help; a keyboard-operable PX4 timeline slider; text summaries for screen readers.
- **Checks:** Chrome in both modes and tabs (sheet raise, PageUp/PageDown, dialog focus, 10 open/close cycles leaving exactly 4 charts, help tips, Shift+→ seek), and headless Chrome at five viewport sizes with no overflow or console errors. Fixed during checks: a squashed timeline band, a double help button, footer text under 12 px.
- 239 vitest, build clean.

---

## 4. Test growth

| After | Tests | Notable new oracles |
|---|---|---|
| Group 1 | 445 | scipy Rotation, closed-form WGS-84, Vallado GMST, textbook KF, DARE |
| Group 2 | 605 | raw pyulog reads, ppigrf, gyro vs EKF2 rate |
| Group 3 | 664 | analytic synthetic truth, NEES Monte Carlo |
| Group 4 | 722 | hand-computed values, `scipy.stats.chi2` |
| Group 5 | 898 | sgp4, closed-form eclipse/Kepler/J2, analytic gyroscopic solutions |
| Group 6 | 1054 | closed-form B-dot dissipation, finite-difference LVLH rate |
| Group 7 | 1177 | `van_loan`, finite differences, MC covariance |
| Group 8 | 1388 | ppigrf chain, ray Monte Carlo, perfect-knowledge runner |
| Group 9 | 1436 | visualizer geometry from spec |
| Group 10 | 1742 | golden file, DOP853 free rotation, conservation of H |
| Group 11 | 1747 + 68 vitest | Python-generated frame fixtures, real `SphereGeometry` UV sampling |
| Group 12 | 1799 + 1 skip, + 107 vitest | independent `run_filter`, scipy slerp, χ² |
| UI redesign | 1800 collected, 239 vitest | contrast, sheet model, a11y, legends |

---

## 5. Recurring patterns (why the project is trustworthy)

1. **Independent oracles.** The test-author wrote tests from the spec, using libraries or closed forms the implementation never touched.
2. **Mutation and sign-flip testing.** Reviews deliberately broke the code (26 mutants in Group 1, 21 in Group 5) to prove the tests would notice.
3. **Opus ★ reviews found real bugs**, not just nits: the mag-heading Jacobian (G3), the baro test criterion and GPS MSL handling (G2), the optimistic MEKF assumptions (G7), the kick energy cap (G10).
4. **Honesty about results.** A tuning winner was rejected, a plan gate was disproved (field strength, gravity-gradient dominance), post-hoc choices are labelled, the failure of the naive filter is shown, not hidden.
5. **Golden gating before refactoring** (Group 10) and **bit-identical fresh-venv reproduction** (Group 9).
6. **Every claim about EKF2 says "reference, not truth"**, in captions, the web page footer and a permanent badge.

---

## 6. Open items and known limitations (as recorded)

**Project A**
- GPS-delay sensitivity ({0, 0.11, 0.2} s) planned but not run; baro-offset state, GPS lever arm and mag bias states untried.
- Aggressive-log accel-bias disagreement with EKF2 unexplained.
- Hover EKF2 yaw-reset time: 16.06 s in `NOTES.md` vs 16.08 s in `final_metrics_<id>.json` (cause unchecked).
- Citations marked "unverified" in A-07…A-12 still need checking before any external use. Some reviewer-only numbers in A-13 come from scratch scripts not in the repo.
- Hover horizontal accel bias is unobservable (report with ±σ).

**Project B**
- All results are simulation results; the requirement thresholds are assumed.
- Design filter still overconfident (NEES ≈ 14); the Monte Carlo worst-case knowledge excursions are not root-caused.
- No innovation monitor/FDIR, no closed-loop gating, no albedo state, no orbit-knowledge error; the dumping hysteresis never cycles; no ZOH sub-stepping convergence check.
- Citations for B-01…B-20 marked "unverified" must be checked; one `TBD` (IGRF model error magnitude) remains in B-07.

**Browser simulator**
- The ≥ 500× engine target depends on CPU boost state; streaming saturates at ≈ 400× / 10 Hz when the CPU can't keep up.
- A byte-preserving `np.cross` speed-up (~26 % of cumulative time) is possible but not done.
- `n_max` batch cap and dropped-budget behaviour are untested because they depend on wall-clock timing.
- Process hygiene: never run `python -` with a heredoc in Git Bash (it opens a REPL), kill by PID only, and use timeouts.

---

## 7. How to use the repo

```bash
.venv/Scripts/python -m pytest -q                       # all tests (~245–300 s)
.venv/Scripts/python -m pip install -e .[web]           # add FastAPI/uvicorn/httpx
.venv/Scripts/python -m webapp                          # simulator at http://127.0.0.1:8000
cd webapp/frontend && npm ci && npm run build && npm test

.venv/Scripts/python -m sensor_fusion.run_flight all    # Project A final metrics
.venv/Scripts/python -m adcs.simulate                   # Project B closed loop (--knowledge perfect for G6)
.venv/Scripts/python -m adcs.monte_carlo                # 100-run campaigns (~20 min)
.venv/Scripts/python -m adcs.visualize                  # 3D animation
```

### Layout
```
common/          quaternion.py  ekf.py  frames.py  plotting.py            (shared core)
sensor_fusion/   loader, dead_reckoning, ekf_drone, synthetic, metrics, run_flight, report/   (Project A)
adcs/            config, orbit, environment, dynamics, disturbances, actuators, control,
                 sensors, estimation, simulate, stepper, monte_carlo, visualize, report/       (Project B)
webapp/          server/ (FastAPI, px4_replay.py)   frontend/ (Vite + TS + Three.js)
docs/            IMPLEMENTATION_PLAN.md  specs/*.md
CONVENTIONS.md  assumptions.md  PROGRESS.md  README.md  CLAUDE.md
```

### Where to find a number
`PROGRESS.md` (per-group results) → the metrics JSON it names in `*/report/` → `assumptions.md` (why a value was chosen) → `docs/specs/*` (the contract the code was written to).
