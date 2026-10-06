# Spec: Project B attitude determination (Group 7, M8)

Source of truth for `py-implementer`, `test-author`, `figures-runner`, `doc-writer`. Conventions per `CONVENTIONS.md`: SI, scalar-first Hamilton **body-to-ECI** quaternions, body angular velocity relative to ECI in body axes, time on axis 0, error state `[δθ(3), δb_g(3)]` with `q_true = q_est ⊗ exp(δθ)` (right / body-frame error). Reuse `common.quaternion` (`skew`, `to_dcm`, `from_dcm`, `canonicalize`, `rotate_inverse`, `from_rotvec`, `to_rotvec`, `multiply`, `conjugate`, `inject_attitude_error`, `attitude_reset_jacobian`, `error_rotvec`, `angle_between`), `common.ekf` (`ErrorStateEKF`, `van_loan`, `chi2_gate`, `nis_bounds`), `adcs.config`, `adcs.orbit` (`initial_state`, `Ephemeris`, `orbital_period`), `adcs.environment` (`FieldTable`), `adcs.simulate` (`run`, `lvlh_state`, `tumble_state`). Do not edit those modules or `CONVENTIONS.md`.

General: array-likes in, float arrays out; no mutation of inputs; no hidden global state (randomness only through a `numpy.random.Generator` passed in or a `seed`); docstrings state units and frames.

## 0. Sampling model (shared by sensors, estimator, Group 8)

All sensors are sampled once per control period `T = 1 s` at `t_k = k T` (the magnetometer with the coils off, Group 6 duty cycle).

- **Gyro** sample `k ≥ 1` is the **mean body rate over `[t_{k−1}, t_k]`** plus bias and noise. The noise-free mean rate is defined from the true attitudes as the constant rate that maps `q_{k−1}` to `q_k` exactly:
  `ω̄_k = to_rotvec(q_{k−1}* ⊗ q_k) / T` (an ideal, coning-compensated rate-integrating gyro). Sample 0 does not exist (NaN row).
- **Magnetometer** and **sun sensor** samples `k ≥ 0` are instantaneous at `t_k`, using the true `q_k`.
- The estimator at step `k` propagates with gyro sample `k` from `t_{k−1}` to `t_k`, then applies the magnetometer update, then the sun update (if valid). With this definition the filter's propagation model `q_k = q_{k−1} ⊗ exp(ω̄_k T)` is exact for the truth, so all estimation error comes from sensor noise and bias.

## 1. `adcs/sensors.py` (py-implementer)

Parameters (recorded in `assumptions.md` B-07; vendor pages read 2026-10-06 by the main session):

| Sensor | Part | Datasheet value | Model parameter |
|---|---|---|---|
| Magnetometer | PNI RM3100 | noise 15 nT | white noise **σ_m = 15e-9 T per axis**, no bias, no scale/misalignment, no coil or spacecraft fields (sampled with coils off) |
| Gyro | Analog Devices ADXRS453 | ARW 0.65 °/√h; in-run bias stability 16 °/h; rate noise density 0.015 °/s/√Hz | **ARW σ_v = 0.65·(π/180)/60 = 1.8908e-4 rad/√s**; bias random walk **σ_u = b_inst/√P_orbit = 7.7570e-5/√5553.6 = 1.0409e-6 rad/s/√s** (ASSUMPTION: the 1σ bias drift over one orbit equals the datasheet in-run bias stability); initial (turn-on) bias per axis ~ N(0, **σ_b0 = 0.1 °/s = 1.7453e-3 rad/s**) (design choice) |
| Coarse sun sensor | Tensor Tech CSS-10, ×6 (one per face ±x, ±y, ±z) | "< 5 deg (1-sigma) of sun determination accuracy" after calibration; ±60° FOV per sensor | ideal cosine photodiodes with hemispherical response (FOV limit, albedo and Earth IR **not modelled**); per-diode white noise **σ_d = 0.0436** of full-scale current, chosen so that the reconstructed vector has per-axis noise `√2 σ_d = 0.0617 rad` and total angle error `√2 · 0.0617 rad = 5.0°` (1σ) |

### API

| Name | Behaviour |
|---|---|
| `CSS_NORMALS` | module constant `(6, 3)`: rows `+x, −x, +y, −y, +z, −z` |
| `MagnetometerConfig` | frozen dataclass: `sigma: float = 15e-9` (T, per axis) |
| `SunSensorConfig` | frozen dataclass: `sigma_diode: float = 0.0436` (fraction of full-scale current), `threshold: float = 0.6` (eclipse false detection at 5.6σ of the summed noise √6 σ_d = 0.107; lit misses, sum ≥ 1, at 3.7σ); property `sigma_vector = √2 · sigma_diode` (per-axis std of the reconstructed vector) |
| `GyroConfig` | frozen dataclass: `arw: float = 1.8908e-4` (rad/√s), `rrw: float = 1.0409e-6` (rad/s/√s, bias random walk), `bias0_sigma: float = 1.7453e-3` (rad/s, per-axis std of the turn-on bias) |
| `magnetometer(b_body, rng, cfg=MagnetometerConfig())` | `b_body (..., 3)` T → `b_body + N(0, σ_m² I)` |
| `sun_sensor_currents(s_body, lit, rng, cfg=SunSensorConfig())` | `s_body (..., 3)` unit sun vector in body, `lit (...)` bool → currents `(..., 6)` = `where(lit, max(0, CSS_NORMALS @ s_body), 0) + N(0, σ_d²)` (noise on all six, also in eclipse; currents may be slightly negative) |
| `sun_vector_from_currents(currents, cfg=SunSensorConfig())` | → `(s_meas (..., 3), valid (...) bool)`: `s_meas = currents[..., 0::2] − currents[..., 1::2]` (not normalised: noise-free and lit it equals `s_body` exactly because `max(0,x) − max(0,−x) = x`); `valid = currents.sum(-1) > cfg.threshold` |
| `class Gyro(cfg=GyroConfig(), rng=None, bias0=None)` | stateful gyro. `bias0` None → draw `N(0, bias0_sigma² I)` from `rng`. Property `bias` (copy, rad/s). `sample(w_mean, dt)`: first steps the bias `b ← b + rrw·√dt·N(0, I)`, then returns `w_mean + b + (arw/√dt)·N(0, I)` (rad/s) |
| `mean_rates(q, dt)` | `q (N, 4)` → `(N, 3)`; row 0 NaN, row k = `to_rotvec(q_{k−1}* ⊗ q_k) / dt` |
| `SensorData` | dataclass: `t (N,)`, `gyro (N,3)` (row 0 NaN), `gyro_bias (N,3)` true bias used in sample k (row 0 = bias0), `mag (N,3)` T, `css_currents (N,6)`, `sun (N,3)` (= `s_meas`), `sun_valid (N,)` bool, `lit (N,)` bool, `meta: dict` (configs as dicts, seed) |
| `measure_series(t, q, b_eci, s_eci, lit, seed=None, mag=MagnetometerConfig(), css=SunSensorConfig(), gyro=GyroConfig(), bias0=None)` | uniform time grid `t (N,)` (dt = t[1]−t[0]; raise `ValueError` if not uniform within 1e-9 s); true `q (N,4)`; `b_eci (N,3)` T; `s_eci (N,3)` unit; `lit (N,)`. Random streams: `rng_g, rng_m, rng_s = np.random.default_rng(seed).spawn(3)` (gyro, magnetometer, sun sensor), so changing one sensor's config does not change another's noise. Body vectors `R(q)ᵀ b_eci`, `R(q)ᵀ s_eci`. Gyro: `Gyro(gyro, rng_g, bias0)`, sample k = `sample(mean_rates(q, dt)[k], dt)` for k ≥ 1. |

### Tests (`adcs/tests/test_sensors.py`, test-author)

1. Magnetometer noise: 20 000 samples of a fixed vector → per-axis std within 3 % of σ, mean within 4σ/√N, axes uncorrelated (|corr| < 0.03).
2. CSS noise-free (`sigma_diode = 0`): for 1000 random unit vectors `s_meas == s_body` to 1e-15; `valid` true; currents ≥ 0 and exactly the cosine law on lit faces, 0 on the others; in eclipse all currents 0 and `valid` false.
3. CSS noisy: per-axis std of `s_meas − s_body` ≈ `√2 σ_d` (±3 %); the mean angle error in degrees ≈ the Rayleigh mean `√(π/2)·√2σ_d` in degrees (±3 %); eclipse never valid (10 000 samples), lit always valid.
4. Gyro: zero bias RW and ARW → returns `w_mean + bias0` exactly; ARW only → sample std `arw/√dt` (±3 %) for dt ∈ {0.1, 1, 10}; RRW only → after n steps the bias variance across 2000 independent gyros ≈ `rrw² · n·dt` (±7 %); default `bias0` drawn with std `bias0_sigma` (±7 % over 2000 gyros).
5. `mean_rates`: for constant body rate (truth by `quaternion.propagate`) returns that rate to 1e-12; for a random sequence of attitudes `q_{k−1} ⊗ exp(ω̄_k dt) = q_k` (scipy `Rotation` oracle).
6. `measure_series`: shapes; reproducible for equal seeds; changing `css` config leaves `gyro` and `mag` identical (independent streams); with all noise off the magnetometer equals `R(q)ᵀ b_eci` (scipy oracle) and `gyro[k] − gyro_bias[k]` equals the mean rate; non-uniform `t` raises.

## 2. TRIAD (in `adcs/estimation.py`, py-implementer)

| Name | Behaviour |
|---|---|
| `triad(v1_b, v2_b, v1_r, v2_r)` | body vectors `v1_b, v2_b`, reference (ECI) vectors `v1_r, v2_r`, each `(..., 3)`, any non-zero length (normalised internally). `v1` is the **primary** (matched exactly). Triads `t1 = v̂1`, `t2 = (v1 × v2)/‖v1 × v2‖`, `t3 = t1 × t2` in each frame; `R = [t1r t2r t3r] [t1b t2b t3b]ᵀ` (body → ref); returns `canonicalize(from_dcm(R))` `(..., 4)`. Raises `ValueError` if `‖v̂1 × v̂2‖ < 1e-6` in either frame for any element. |
| `triad_covariance(v1_b, v2_b, sigma1, sigma2)` | single (unbatched) case, `(3, 3)` body-frame covariance of the TRIAD attitude error `δθ` (`q_true = q_triad ⊗ exp(δθ)`), small-noise approximation. `sigma_i` (rad) is the per-axis std of the **unit-vector** noise perpendicular to `v̂_i` (for the magnetometer `σ_m / ‖b‖`; for the CSS `sigma_vector`). With `e1 = v̂1`, `e2 = (v̂1 × v̂2)/sin α`, `e3 = e1 × e2`, `cos α = v̂1·v̂2`: `P = σ1² (I − e1e1ᵀ) + ((σ2² + σ1² cos²α)/sin²α) e1e1ᵀ − (σ1² cos α / sin α)(e1e3ᵀ + e3e1ᵀ)`. (Derivation: the primary fixes `δθ⊥ = v̂1 × n1`; the out-of-plane component of `v2`'s noise, minus the part explained by the tilt of `v1`, sets the rotation about `v1`.) |

### Tests (`adcs/tests/test_triad.py`, test-author)

1. Noise-free: 500 random attitudes (scipy `Rotation.random`, seeded) and random vector pairs (angle between them ≥ 10°): `triad` returns the true quaternion (angle error < 1e-9 rad), batched and unbatched agree; inputs of arbitrary length give the same result.
2. Primary exactness: with noise on both body vectors, `R(q) v̂1_b == v̂1_r` to 1e-12; `v2` is generally not matched.
3. Swapping primary/secondary changes the result only when there is noise.
4. Degenerate (parallel / antiparallel) pairs raise `ValueError`.
5. Covariance Monte Carlo: for several geometries (α = 20°, 60°, 90°, 150°) and σ1 ≠ σ2 (e.g. 1e-3 and 3e-3 rad, and reversed), 20 000 noisy samples (noise drawn perpendicular to each true unit vector, then renormalised); the sample covariance of `error_rotvec(q_triad, q_true)` (note: `q_true = q_triad ⊗ exp(δθ)` ⇒ `δθ = error_rotvec(q_triad, q_true)`) matches `triad_covariance` element-wise within 5 % of its largest eigenvalue, and the mean error is < 4 standard errors.
6. Sign check of the off-diagonal term: the MC correlation between the e1 and e3 components has the sign of `−cos α` (for α = 60° and 150°).

## 3. MEKF (in `adcs/estimation.py`, main session)

Multiplicative EKF on `common.ekf.ErrorStateEKF`. Nominal state `q̂` (body→ECI) and `b̂` (gyro bias, rad/s); error state `[δθ, δb]` (6).

Error dynamics with `ω̂ = ω_m − b̂`, `W = [ω̂×]`: `δθ̇ = −W δθ − δb − n_v`, `δḃ = n_u` (`E[n_v n_vᵀ] = σ_v² I δ`, `E[n_u n_uᵀ] = σ_u² I δ`). Discretisation over `dt` with constant `ω̂` is **closed form** (`discrete_model(w_hat, dt, arw, rrw) -> (Phi, Qd)`), equal to `van_loan` of the continuous model to round-off. Nominal propagation `q̂ ← q̂ ⊗ exp(ω̂ dt)`, `b̂` constant.

Vector measurement `z = R(q)ᵀ v_eci + n`, `E[n nᵀ] = σ² I`: residual `r = z − R(q̂)ᵀ v_eci`, `H = [[R(q̂)ᵀ v_eci ×], 0₃]` (3 × 6). Magnetometer: `v_eci = B_eci(t_k)` (T), σ = σ_m. Sun: `v_eci = ŝ_eci(t_k)`, `z = s_meas`, σ = `sigma_vector`. After every accepted update: inject (`inject_attitude_error`, `b̂ += δb`) and reset `P ← G P Gᵀ` with `G = blockdiag(attitude_reset_jacobian(δθ), I)`.

Initialisation from TRIAD at the first sample with a valid sun measurement: primary = the vector with the smaller unit-vector σ (the magnetometer by default), `P0 = blockdiag(triad_covariance · init_inflate, σ_b0² I)`, `b̂0 = 0`. Before initialisation the outputs are NaN.

API (main session writes it; test-author tests it): `MekfConfig` (arw, rrw, sigma_mag, sigma_sun, sigma_b0, init_inflate, gate_prob — `None` = no gating), `MEKF(q0, b0, P0, cfg)` with `.q`, `.b`, `.P`, `.std`, `.ekf`, `.propagate(w_meas, dt)`, `.update_vector(z, v_eci, sigma, name, t=None)`, `.update_mag(z, b_eci, t=None)`, `.update_sun(z, s_eci, t=None)`; `EstimationResult` dataclass (`t, q, b, std (N,6), P_att (N,3,3), k_init, nis: dict name -> (t, nis, accepted)`); `run_mekf(data: SensorData, b_eci, s_eci, cfg=MekfConfig()) -> EstimationResult`; `attitude_errors(q_true, q_est) -> (N,3)` body-frame `δθ` with `q_true = q_est ⊗ exp(δθ)`.

Tests (`adcs/tests/test_mekf.py`, test-author, thresholds set by the main session after a prototype run) are listed in §5.

### Implemented API (main session, `adcs/estimation.py`)

- `discrete_model(w_hat, dt, arw, rrw) -> (Phi (6,6), Qd (6,6))`; derivation in the comment block above it (series for θ = |ω̂|dt < 1e-2).
- `MekfConfig(arw, rrw, sigma_mag, sigma_sun, sigma_b0, init_inflate=1.0, gate_prob=None, use_mag=True, use_sun=True)`; defaults = the sensor defaults of §1.
- `MEKF(q0, b0, P0, cfg)`: `.q`, `.b` (copies), `.P`, `.std`, `.ekf` (the `ErrorStateEKF`, state names `STATE_NAMES`); `.propagate(w_meas, dt)`; `.update_vector(z, v_eci, sigma, name, t=None) -> UpdateResult`; `.update_mag(z, b_eci, t)` (name `"mag"`), `.update_sun(z, s_eci, t)` (name `"sun"`).
- `triad_init(mag_b, sun_b, b_eci, s_eci, cfg) -> (q0, P0)`.
- `run_mekf(data, b_eci, s_eci, cfg=MekfConfig(), q0=None, P0=None, b0=None, k_init=None) -> EstimationResult(t, q, b, std, P_att, k_init, nis, meta)`; `nis[name] = (t, nis, accepted)` arrays. Without `q0` it starts from TRIAD at the first `sun_valid` sample; rows before `k_init` are NaN.
- `attitude_errors(q_true, q_est)` (`= error_rotvec(q_est, q_true)`), `nees(err (N,m), P (N,m,m))`.

### Prototype results (main session, seed 0 unless noted; truth from `adcs.simulate.run`, sensors with default parameters, TRIAD start at t = 0 in sunlight)

| Scenario | NEES (att, dof 3) | NIS mag / sun (dof 3) | RMS ‖δθ‖ after 1 orbit, sunlit / eclipse | max ‖δθ‖ after 1 orbit | bias error ‖·‖ at 0 → 0.25 orbit |
|---|---|---|---|---|---|
| nadir pointing, 3 orbits | 3.00 | 3.005 / 2.98 | 0.32° / 0.78° | 2.16° | 0.185 → 0.0010 °/s |
| detumble from 3 deg/s (B-dot), 2 orbits | 3.10 | 3.03 / 2.93 | 0.35° / 0.45° | 0.94° | 0.185 → 0.0014 °/s |
| nadir, σ_m = 300 nT, seed 1 | 2.63 | 3.04 / 3.01 | 0.33° / 0.56° | 1.66° | 0.085 → 0.0013 °/s |

Fraction of samples with |δθ_i| < 3σ_i (final 5-orbit pointing run, `estimation_metrics.json`): 0.9969 / 0.9980 / 0.9979 (Gaussian value 0.9973); bias 0.9990 / 0.9988 / 0.9975. Closed-form `discrete_model` vs `van_loan`: Φ to 1.4e-14, Qd to 8e-12 relative (worst near θ = 1e-2). Run time ≈ 2.4 s per orbit (5554 steps).

## 5. MEKF tests (`adcs/tests/test_mekf.py`, test-author)

Keep the file under ~25 s. Build **synthetic** truth for most tests (fast): `q` by `quaternion.propagate` with a slowly varying body rate (e.g. orbit-rate pitch `n = 1.13e-3 rad/s` about body y plus a small sinusoidal wobble), `dt = 1 s`; reference vectors: `b_eci(t)` = 30 µT × a unit vector rotating at `2n` about an axis tilted 30° from the rotation plane (so the field direction in body changes), `s_eci` fixed unit vector, `lit(t)` = sunlit for the first 61 % of each 5554 s period. Sensors via `adcs.sensors.measure_series`. One module-scoped fixture runs the **real** environment: 1 orbit of `adcs.simulate.run(lvlh_state(...), P, "pointing", log_every=1)` plus `FieldTable` reference vectors (≈ 5 s).

1. `discrete_model` vs `common.ekf.van_loan` (F = [[−[ω̂×], −I],[0,0]], G = diag(−I, I), Qc = diag(σv² I, σu² I)): θ = |ω̂|dt ∈ {0, 1e-6, 5e-3, 9.9e-3, 1.01e-2, 0.2, 1, 3}, dt ∈ {0.1, 1, 10}: `Phi` abs error < 1e-12; `Qd` with arw = 0, rrw = 1: abs error < 1e-9 × max |off-diagonal of the oracle|; with defaults: relative < 1e-9.
2. Measurement Jacobian by central finite differences of `R(q̂ ⊗ exp(δθ))ᵀ v` w.r.t. δθ (step 1e-6): equals `skew(R(q̂)ᵀ v)` to 1e-8 relative; the bias columns are zero.
3. Exact propagation: sensors with zero noise and zero bias (GyroConfig(arw=0, rrw=0, bias0_sigma=0)), MEKF started at the true attitude, `use_mag=False, use_sun=False`: estimate equals truth to 1e-9 rad after 2000 steps (checks the §0 sampling model and `propagate`).
4. Housekeeping on a 1-orbit synthetic run: `|q| = 1` (1e-12) at every step; every logged `P_att` symmetric positive definite; `std` finite after `k_init`; rows before `k_init` NaN when the run starts in eclipse (shift `lit` so that t = 0 is in eclipse) and `k_init` = the first `sun_valid` index.
5. TRIAD initialisation: `run_mekf(...).q[k_init]` equals `triad_init(...)` q0; `P0` attitude block equals `triad_covariance` with the primary = the magnetometer for default σs (and the sun sensor when `sigma_mag` is made huge, e.g. 1e-4 T); bias block `σ_b0² I`.
6. Bias convergence (synthetic, default noise, true turn-on bias 0.1 °/s on every axis): after 0.25 orbit ‖b − b̂‖ < 0.01 °/s; at the end of 1 orbit each axis within 3σ and σ_b < 0.005 °/s per axis. [prototype: 0.185 → 0.0010 °/s within 0.25 orbit, σ_b ≈ 0.001 °/s]
7. Consistency Monte Carlo (synthetic, 10 seeds × 1 orbit, start from TRIAD): pooled mean NIS for mag and for sun within ±5 % of 3; pooled mean attitude NEES (`nees(attitude_errors, P_att)`, after `k_init + 300`) within [2.55, 3.45]; fraction of |δθ_i| < 3σ_i ≥ 0.98 per axis.
8. Eclipse degradation (synthetic, one seed, sun available only while `lit`): `σ_att = sqrt(trace(P_att))` at the end of an eclipse ≥ 1.5 × its value at eclipse entry; 300 s after the sun returns it is ≤ 0.7 × the end-of-eclipse value; and RMS ‖δθ‖ over eclipse > RMS over sunlight (both after the first 0.25 orbit). [prototype on the real pointing case: RMS 0.32° sunlit vs 0.78° eclipse]
9. Mag-only sanity: with `use_sun=False` (magnetometer only, started from TRIAD) the filter still runs (no exception, P SPD) and its attitude σ about the instantaneous field direction is larger than with both sensors (end of the run).
10. Gating: `MekfConfig(gate_prob=0.9973)`; one magnetometer sample replaced by `z + 100 σ_m` along a direction ⟂ to the field: that update is rejected (`accepted` False in `nis["mag"]`), the estimate at that step changes only by propagation (compare to a run where that update is skipped, `atol` 1e-12).
11. Real-environment smoke (fixture): attitude NEES mean over the orbit after `k_init + 300` in [2.0, 4.0]; max ‖δθ‖ after 0.25 orbit < 3°; NIS mag and sun means in [2.7, 3.3].

Statistical thresholds were chosen from the prototype numbers above; if one fails, report the measured value to the main session instead of loosening it.

## 6. Figures and metrics (`adcs/report/estimation_figures.py`, figures-runner)

`python -m adcs.report.estimation_figures` → `adcs/report/estimation_metrics.json` + figures in `adcs/report/figures/`. Use a process pool for independent runs (set `OPENBLAS_NUM_THREADS=1` before importing numpy in workers, as `sensor_fusion/report/tuning_sweep.py` does). Truth trajectories are computed once per scenario and shared.

Scenarios (default `SimConfig`, default sensors, seed 0 unless stated):
- **pointing**: `simulate.run(lvlh_state(r0, v0, offset=deg2rad([2, −1, 3])), 5 P, "pointing", log_every=1)`.
- **detumble**: `simulate.run(tumble_state(deg2rad(3), 0), 2 P, "detumble", log_every=1)`.

Reference vectors from `Ephemeris(r0, v0, t_end + 10)` + `FieldTable(eph, DEFAULT, t_end + 10)`: `b_eci`, `sun_dir_eci`, `sunlit` at the truth times.

Figures (PNG 150 dpi, units on axes, eclipse intervals shaded grey, time in orbits):
1. `mekf_pointing_attitude.png` — δθ per axis (deg) with ±3σ envelopes, 5 orbits.
2. `mekf_pointing_bias.png` — true and estimated bias per axis (deg/s), and bias error with ±3σ.
3. `mekf_nis.png` — magnetometer and sun NIS, 100-sample moving mean, with the two-sided 95 % bounds `nis_bounds(3, 100)`; attitude NEES moving mean in a third panel.
4. `mekf_eclipse.png` — one-orbit zoom (orbit 3): ‖δθ‖ and `3 sqrt(trace P_att)` vs time, plus the TRIAD error (every sunlit sample) for comparison.
5. `mekf_detumble.png` — detumble scenario: body rate (deg/s), δθ per axis with ±3σ, bias error.
6. `mekf_sensitivity.png` — RMS ‖δθ‖ after 1 orbit (sunlit and eclipse separately) vs σ_m ∈ {15, 100, 300, 1000} nT (CSS default) and vs CSS 1σ angle ∈ {2.5, 5, 10}° (`sigma_diode = angle_rad / 2`, magnetometer default; the filter's `sigma_sun` = `√2 sigma_diode` matched), pointing truth, 3 orbits, seeds 0–4 (mean and min/max over seeds).
7. `mekf_consistency_mc.png` — Monte Carlo, pointing truth (first 3 orbits), 20 seeds: histogram of per-run mean NEES and per-run mean NIS (mag, sun) with the reference value 3.

Metrics JSON (every number in the figures; seeds and configs recorded): per scenario: init index/time, RMS and max ‖δθ‖ (after 0.25 orbit and after 1 orbit; sunlit vs eclipse), per-axis RMS, fraction inside 3σ (attitude and bias), mean NEES, mean NIS (mag, sun) with `nis_bounds(3, n)`, bias error norm at 0 / 0.1 / 0.25 / 0.5 / 1 orbit and at the end, final σ_b; TRIAD error statistics over sunlit samples (RMS, max, and predicted RMS = `sqrt(mean(trace(triad_covariance)))`, the like-for-like predictor of an RMS; `mean(sqrt(trace))` is also stored); eclipse growth: for each complete eclipse, `sqrt(trace P_att)` at entry and exit and the max ‖δθ‖ inside; sensitivity grid; MC summary (mean, std, min, max of per-run NEES/NIS; fraction of runs whose mean NIS is inside the 95 % bounds); run times. A `checks` dict: NEES within [2.55, 3.45] for each scenario; bias error < 0.01 °/s at 0.25 orbit; eclipse RMS > sunlit RMS (pointing).
