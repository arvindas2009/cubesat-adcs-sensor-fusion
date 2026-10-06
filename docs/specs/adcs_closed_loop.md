# Spec: Project B closed loop, sensor realism, Monte Carlo (Group 8, M9)

Source of truth for `py-implementer`, `test-author`, `figures-runner`, `doc-writer`. Conventions per `CONVENTIONS.md`: SI, scalar-first Hamilton **body-to-ECI** quaternions, body rates relative to ECI in body axes, time on axis 0, right/body-frame attitude error `q_true = q_est ⊗ exp(δθ)`. Builds on `docs/specs/adcs_physics.md`, `adcs_control.md`, `adcs_estimation.md` (read §0 sampling model there). Do not edit `CONVENTIONS.md`, `common/`, or modules not named in your task.

General: array-likes in, float arrays out; no mutation of inputs; randomness only through a `numpy.random.Generator` passed in (or a `seed`); docstrings state units and frames; existing defaults and existing tests must stay unchanged (every new parameter defaults to the Group 7 behaviour).

## 0. Why: the Group 7 results are an ideal-sensor floor

Group 7 (`assumptions.md` B-20) ran the MEKF with the filter's reference field equal to the truth field, no magnetometer bias, and ideal hemispherical cosine sun sensors. ★-review scratch runs showed this is optimistic by up to an order of magnitude (300 nT unmodelled mag bias → NEES ≈ 815; hard ±60° CSS field of view → 8.1° sunlit RMS; crude albedo → 6.4°). Group 8 therefore adds **truth-side sensor errors** (§1) and **estimator-side mitigations** (§2), and runs the closed loop with both (§3).

Truth-side error sources added (all off by default in the low-level functions):

| Source | Truth model | Default in the closed-loop "realistic" sensor set |
|---|---|---|
| Onboard field model error | Filter reference = IGRF truncated to degree 8 (`max_degree=8`); truth = full IGRF-14 (degree 13) **plus** an unmodelled field (first-order Gauss–Markov per ECI axis) | degree 8 (prototype: 67 nT rms / 158 nT max vs degree 13 over 3 orbits, 0.10° rms direction); residual σ = 50 nT per axis, τ = 600 s (ASSUMPTION) |
| Magnetometer bias | constant body-frame bias, per run `N(0, σ_mb² I)` | σ_mb = 200 nT per axis (ASSUMPTION: post-calibration residual hard-iron, order of the field of the 1 mA·m² residual dipole B-08 at ~0.1 m) |
| CSS field of view | hard cutoff of the direct-sun current at the vendor ±60° half-angle | fov = 60° |
| Earth albedo on the CSS | Lambertian Earth, plate-to-sphere view factor, illumination ∝ cos of the solar zenith angle at the sub-satellite point; albedo coefficient = 0.30 + Gauss–Markov (σ 0.10, τ 300 s), clipped to [0.05, 0.9] | as stated |

## 1. Sensor and environment additions (py-implementer; tests by test-author)

### 1.1 `adcs/environment.py`

| Name | Change |
|---|---|
| `magnetic_field_eci(r_eci, jd, date, max_degree=13)` | new keyword passed to `ppigrf.igrf(..., max_degree=max_degree)` (ppigrf 2.1.0 signature `igrf(lon, lat, h, date, coeff_fn=..., min_degree=1, max_degree=13)`). Default output identical to today. |
| `FieldTable(ephemeris, cfg=DEFAULT, t_end=None, dt=10.0, max_degree=13)` | new keyword forwarded to `magnetic_field_eci`; store as attribute `max_degree`. Default identical to today. |

### 1.2 `adcs/sensors.py`

| Name | Behaviour |
|---|---|
| `magnetometer(b_body, rng, cfg=MagnetometerConfig(), bias=None)` | returns `b_body + bias + N(0, σ_m² I)`; `bias` (3,) T in body axes (broadcast), `None` = 0. The noise draw (`rng.normal(0, σ, size=b.shape)`) is **identical** with or without `bias`, so for equal generator state `magnetometer(b, rng1, bias=β) − magnetometer(b, rng2) == β` exactly. |
| `gauss_markov(n, dt, sigma, tau, rng, dim=3)` | stationary first-order Gauss–Markov series `(n, dim)`: `x_0 = σ w_0`, `x_k = a x_{k−1} + σ √(1 − a²) w_k`, `a = exp(−dt/τ)`, `w = rng.standard_normal((n, dim))` drawn in one call (row k ↔ w_k). `sigma == 0` → zeros **without** drawing. `ValueError` if `n < 1`, `dt <= 0`, `tau <= 0` or `sigma < 0`. Unit = unit of `sigma`. (`scipy.signal.lfilter` may be used for speed.) |
| `SunSensorConfig.fov: float \| None = None` | new field: half-angle of each diode's field of view (rad). `None` = full hemisphere (Group 7). |
| `sun_sensor_currents(s_body, lit, rng, cfg=SunSensorConfig(), albedo=None)` | direct current `c_i = n_i · s_body`; direct term = `where(lit, c_i if c_i > cos(fov) else 0, 0)` when `fov` is set, `where(lit, max(0, c_i), 0)` when `fov is None` (unchanged); then `+ albedo` (array `(..., 6)` of extra currents, `None` = 0; added regardless of `lit` and of `fov`); then `+ N(0, σ_d²)` on all six (draw identical with/without `albedo`/`fov`). |
| `earth_view_factor(phi, r_norm, re=R_EARTH)` | view factor `F` from a differential plate to a sphere of radius `re` at distance `r_norm` (m) from its centre, `phi` = angle (rad) between the plate normal and the direction to the sphere centre; broadcasting. With `H = r_norm/re`, `ρ = asin(1/H)`, `x = √(H² − 1)`: `F = cos φ / H²` for `φ ≤ π/2 − ρ`; `F = 1/2 − asin(x/(H sin φ))/π + (cos φ · acos(−x cot φ) − x √(1 − H² cos² φ)) / (π H²)` for `π/2 − ρ < φ < π/2 + ρ`; `F = 0` for `φ ≥ π/2 + ρ`. Clip arguments of `asin/acos/√` to their domains. `ValueError` if any `r_norm <= re`. Prototype (400 km): F(0) = 0.8855, F(30°) = 0.7712, F(60°) = 0.5372, F(90°) = 0.2887, F(110°) = 0.1495 (each matched by a 2e6-ray Monte Carlo to 4e-4). Source: standard configuration factor "differential planar element to sphere" (e.g. Howell, *A Catalog of Radiation Heat Transfer Configuration Factors*; catalog entry unverified). |
| `albedo_currents(q, r_eci, s_eci, albedo, re=R_EARTH)` | extra CSS currents `(..., 6)` [fraction of full scale] from Earth albedo. `q (..., 4)` body→ECI, `r_eci (..., 3)` m, `s_eci (..., 3)` unit Earth→Sun, `albedo` scalar or `(...)`. `nadir_b = R(q)ᵀ (−r̂)`; `φ_i = acos(clip(CSS_NORMALS[i] · nadir_b, −1, 1))`; `I_i = albedo · earth_view_factor(φ_i, |r|) · max(0, ŝ · r̂)`. Assumptions (record in docstring): diode full scale = 1 solar constant at normal incidence; Lambertian Earth with uniform albedo; illumination of the whole visible cap approximated by the cosine of the solar zenith angle at the sub-satellite point (so zero whenever `ŝ·r̂ ≤ 0`, which includes eclipse); diode cosine response over the full view factor (no FOV limit on albedo). |

`measure_series` is unchanged.

### 1.3 Tests (`adcs/tests/test_sensor_realism.py`, test-author, written from this spec)

1. `max_degree`: `FieldTable(...)` with default equals `max_degree=13` exactly; degree 8 vs 13 over one orbit differs by an rms in [20, 200] nT; `magnetic_field_eci(max_degree=8)` at 5 points equals an independent chain (`ppigrf.igrf(..., max_degree=8)` → ENU → ECEF → ECI built in the test) to 1e-12 T.
2. Magnetometer bias: exact difference identity above; mean of 20 000 samples within 4σ/√N of `b + β`.
3. `gauss_markov`: shape and dtype; reproducible with equal seeds; `sigma=0` → zeros and the generator state unchanged; std over 400 independent series of length 2000 within 3 % of σ at k = 0 and at k = 1999 (stationary); lag-m autocorrelation ≈ `exp(−m dt/τ)` within 0.03 for m ∈ {1, 10, 50} (dt = 1, τ = 30); invalid arguments raise.
4. FOV: noise-free (σ_d = 0), 2000 random unit vectors: diode currents are `c_i` where `c_i > cos(fov)` and 0 elsewhere; `fov=None` reproduces the current function exactly (same seed); `fov = π/2` equals `fov=None` for noise-free input; eclipse all zero.
5. `earth_view_factor`: φ = 0 gives `(re/r)²`; both branches agree at `φ = π/2 − ρ` within 1e-12; zero for `φ ≥ π/2 + ρ`; non-increasing in φ; Monte Carlo oracle (cosine-weighted rays from the plate, hit test against the sphere; ≥ 1e6 rays, seeded) at φ ∈ {0, 20°, 45°, 70°, 90°, 105°} within 0.003, for r = R_E + 400 km and R_E + 1000 km; invalid `r_norm` raises.
6. `albedo_currents`: all zero when `ŝ·r̂ ≤ 0`; body with +z towards nadir and Sun at the zenith of the spacecraft (`ŝ = r̂`): current on +z = `albedo · (re/|r|)²`, on −z = 0, on ±x, ±y = `albedo · F(π/2)`; linear in `albedo`; `nadir_b` checked against scipy `Rotation` for random `q`; batched shapes `(N, 6)`.
7. `sun_sensor_currents(albedo=...)`: with σ_d = 0 equals direct + albedo; with noise, same seed, difference with/without `albedo` equals `albedo` exactly.

## 2. Estimator additions (`adcs/estimation.py`, main session — implemented)

All new `MekfConfig` fields default to the Group 7 filter; `MekfConfig()` is unchanged (6 states, CSS vector update, no TRIAD guard).

| Name | Behaviour |
|---|---|
| `MekfConfig.mag_bias: bool = False` | estimate a constant magnetometer bias: error state `[δθ, δb_g, δb_m]` (9), `b_m,true = b̂_m + δb_m` (T, body axes). Property `n_states` (6 or 9). Constant `STATE_NAMES_MAG_BIAS`. |
| `sigma_mb0 = 200e-9` (T), `mag_bias_rw = 0.0` (T/√s) | initial std per axis (b̂_m0 = 0) and random walk (Qd block `mag_bias_rw² dt I`). |
| propagation | `Phi = blockdiag(Phi6, I3)`, `Qd = blockdiag(Qd6, mag_bias_rw² dt I3)` with `(Phi6, Qd6) = discrete_model(...)` (equal to `van_loan` of the augmented `F = [[−W, −I, 0], [0, 0, 0], [0, 0, 0]]`). |
| magnetometer update (9 states) | `z = R(q)ᵀ b_eci + b_m + n`: residual `z − (R(q̂)ᵀ b_eci + b̂_m)`, `H = [[v_b×], 0, I]`; injection `b̂_m += δb_m`; reset `G = blockdiag(attitude_reset_jacobian(δθ), I, I)`. |
| `css_mode = "vector"` or `"diodes"` | `"diodes"`: `MEKF.update_css_diodes(currents, s_eci, r_eci=None, t=None)`, one block update (name `"css"`) of the usable diodes. Diode i: `I_i = n_iᵀ R(q)ᵀ s + a_i + n`, `a_i` = onboard albedo `albedo_currents(q̂, r_eci, s_eci, albedo_model)[i]` (attitude dependence neglected in H), `H_i = [n_iᵀ [s_b×], 0…]`, `R_ii = σ_d² + (sigma_albedo · a_i / albedo_model)²`. Usable: predicted direct cosine `n_i · R(q̂)ᵀ s > diode_min_current` (0.55) and `> cos(css_fov)` if set, and measured `I_i > diode_min_current / 2`. Gating on the prediction avoids selecting samples by their noise (gating on the measured current gave NEES 3.3–4.8; on the prediction 2.6–3.5). Returns the `UpdateResult` or `None`. |
| `sigma_diode`, `diode_min_current`, `css_fov`, `albedo_model`, `sigma_albedo` | defaults `SunSensorConfig().sigma_diode`, 0.55, None, 0, 0 |
| `triad_min_angle` (rad, default 0) | `run_mekf` (and the closed loop) initialise only at a sample whose field–sun angle (`vector_angle(mag, sun)`) is ≥ this; diodes mode additionally needs `css_vector_complete(currents, diode_min_current)` (each body axis has one diode above the threshold, so the reconstructed vector is fully measured) |
| `init_att_sigma_floor` (rad, default 0) | `triad_init`: attitude block `init_inflate · P_triad + floor² I`; 9-state: magnetometer unit-vector σ uses `hypot(sigma_mag, sigma_mb0)/|b|`, `P0[6:,6:] = sigma_mb0² I` |
| `MEKF.step(gyro, dt, mag=None, b_eci=None, sun=None, s_eci=None, sun_valid=False, currents=None, r_eci=None, t=None)` | public cycle shared by `run_mekf` and the closed loop: propagate; mag update (if `use_mag` and `mag` given); sun update (if `use_sun`): vector mode if `sun_valid`, diodes mode with `currents` |
| `MEKF.bm`, `MEKF.att_sigma` | magnetometer-bias estimate (zeros for 6 states); `sqrt(trace P_att)` |
| `vector_angle(a, b)`, `css_vector_complete(currents, min_current)` | helpers |
| `run_mekf(..., r_eci=None)` | uses `step`; `EstimationResult.bm` (N, 3) for 9 states else None; `meta["nis_dof"][name]` = dof of each NIS row; `nis` keeps the `(t, nis, accepted)` tuples (`"css"` key only when diode updates happened) |

### Prototype numbers (main session scratch runs on the PK pointing truth of `adcs_estimation.md` §6, 3 orbits, seeds 0–5)

- 9 states, matched model (constant bias [200, −150, 100] nT added to `SensorData.mag`, `sigma_mb0 = 300 nT`, `mag_bias_rw = 0`, σ_m = 15 nT): attitude NEES (from k_init + 300) 2.75–3.18; bias error at end 0.27–0.50 nT (σ_norm 0.38 nT), after 1 orbit 0.38–0.79 nT; mag NIS/3 0.989–1.011; sunlit RMS 0.27–0.37°.
- Diodes mode, filter started at truth ⊗ exp([3, −2, 2]°) with `P0 = diag((5°)² I, σ_b0² I)` (TRIAD cannot initialise in nadir pointing because the sun stays in the body x–z plane, so one axis is never lit): truth CSS FOV 60° with no albedo, filter `css_fov = 60°`: NIS per dof 0.968–1.020, NEES 2.58–3.44, sunlit RMS 0.51–0.87°, mean 1.21 diodes per update. Same with truth albedo constant 0.3 and `albedo_model = 0.3` (σ_albedo 0): NIS/dof 0.99–1.03, NEES 2.70–3.48.

### Tests (`adcs/tests/test_mekf_ext.py`, test-author, from this spec)

Oracles: finite differences, an independent textbook EKF update written in the test, `common.ekf.van_loan`, χ² statistics, scipy `Rotation`.

1. Defaults: `MekfConfig().n_states == 6`; `MEKF(..., MekfConfig(mag_bias=True))` rejects a 6 × 6 `P0`; `css_mode="x"` raises.
2. 9-state propagation: after one `propagate`, `P` equals `Phi P0 Phiᵀ + Qd` with `(Phi, Qd) = van_loan(F9, Qc9, dt, G9)` built in the test (F9 above, `Qc9 = diag(arw², rrw², mag_bias_rw²)`), to 1e-10 relative; nominal `q`, `b`, `bm` as expected.
3. Mag update with bias states: for random `q̂`, `b̂_m`, `P0` (SPD), `z`: `res.dx` equals `K r` of an independent EKF update using `H_fd` from central differences of `h(q̂ ⊗ exp(δθ), b̂_m + δb_m) = R(·)ᵀ b_eci + b_m` (step 1e-7) and the Joseph-form `P` equals the test's own, both to 1e-6 relative; injected `q`, `b`, `bm` follow `inject_attitude_error` / addition; reset applied.
4. Diode update: same oracle for `update_css_diodes` with `albedo_model = 0` (H from central differences of `n_iᵀ R(q)ᵀ s` over the selected diodes); selection: a diode with predicted cosine 0.5 < c < 0.55 is not used, one with measured current below `0.55/2` is not used although predicted lit, `css_fov` removes diodes whose predicted cosine is below `cos(fov)`; no usable diode → returns `None` and leaves `q`, `P` untouched; with `albedo_model > 0` the residual equals `I − (cos_pred + albedo_currents(q̂, r, s, albedo_model))` and `R_ii` equals the formula.
5. Bias convergence and consistency (matched model, prototype setup above, ≥ 6 seeds, 2–3 orbits of pointing truth from `adcs.simulate.run`): mean attitude NEES pooled over seeds within the χ² 99 % band for the pooled sample size treated with an effective sample size of n/150 (correlated errors; G7 review) **or** simply in [2.4, 3.6]; bias error at the end < 3 nT for every seed; `bm` 3σ covers the error at the end for ≥ 5 of 6 seeds.
6. Diodes consistency (prototype setup, ≥ 6 seeds): pooled NIS per dof in [0.95, 1.05]; pooled attitude NEES in [2.4, 3.8]; sunlit RMS < 1.5° for every seed.
7. `triad_init` 9-state: `P0` shape, blocks, floor added in quadrature (attitude block minus floor² I equals `triad_covariance` with the inflated magnetometer σ); `css_vector_complete` truth table; `vector_angle` vs `np.arccos` for random vectors.
8. `run_mekf` with `triad_min_angle`: the initialisation index satisfies the guard and is the first index that does (construct data where the first valid-sun samples have a field–sun angle below the guard); with an impossible guard (179°) → `k_init == -1`, all NaN.
9. `step` equivalence: a manual `propagate` + `update_mag` + `update_sun` sequence gives bit-identical `q`, `b`, `P` to `step(...)` (vector mode), and to `update_css_diodes` in diodes mode.

## 3. Closed-loop runner (`adcs/simulate.py`, main session — implemented)

`run_closed_loop(y0, t_end, seed=0, cfg=DEFAULT, ctrl=ControlConfig(), mtq=..., wheel=..., sensors=IDEAL_SENSORS, mekf_cfg=MEKF_G7, modes=ModeConfig(), mag_cfg=..., css_cfg=..., gyro_cfg=..., inertia_true=None, log_every=1) -> ClosedLoopResult`; module docstring block above it lists the per-period sequence. Key facts:

- Random streams `rng_g, rng_m, rng_s, rng_truth = default_rng(seed).spawn(4)` (first three = `measure_series` order); `rng_truth.spawn(3)` → (magnetometer-bias draw: `standard_normal(3)` always drawn, field residual, albedo).
- Truth field `FieldTable` (degree 13) + `gauss_markov(field_residual_sigma, tau)` in ECI, used for the magnetometer **and** the magnetorquer torque; filter reference = `FieldTable(max_degree=sensors.onboard_field_degree)`. Sun direction and shadow from the same models (onboard = truth). Albedo coefficient `clip(albedo + GM(albedo_sigma, albedo_tau), *albedo_clip)`.
- Gyro sample k from `q_{k−1}, q_k` of the truth inside the loop (`Gyro.sample(to_rotvec(q_{k−1}* q_k)/T, T)`); no sample at k = 0.
- MEKF initialised at the first k with: sun valid (diodes mode: `css_vector_complete`), angle(mag, sun) ≥ `triad_min_angle`; from then `MEKF.step` every period with the onboard field.
- Modes: start `MODE_DETUMBLE` (B-dot on the measured field, bias included). Switch to `MODE_POINTING` when the MEKF has run ≥ `switch_min_filter_time` (300 s), `att_sigma < switch_att_sigma` (2°) and `|gyro − b̂| < ctrl.detumble_threshold` (0.5 °/s) held `ctrl.detumble_hold` (60 s). Fallback to detumble when `|gyro − b̂| > safe_rate` (2 °/s) for `safe_hold` (10 s). `ModeConfig(switch_att_sigma=0.0)` never switches (detumble-only runs).
- Pointing: `pointing_torque(q̂, ω̂ = gyro − b̂, h_w, q_ref, w_ref, I_nominal)`; dumping and torque feed-forward with `mag − b̂_m`; wheel speed from perfect tachometers (truth `h_w`, `ω`); nadir reference from the truth orbit (perfect onboard navigation, ASSUMPTION). Truth dynamics with `inertia_true`.
- Presets: `IDEAL_SENSORS` (Group 7), `REALISTIC_SENSORS` (§0 table), `MEKF_G7` (= `MekfConfig(triad_min_angle=20°)`), `MEKF_DESIGN` (`mag_bias=True, sigma_mag=100 nT, mag_bias_rw=1e-11, css_mode="diodes", css_fov=60°, albedo_model=0.3, sigma_albedo=0.5, triad_min_angle=20°, init_att_sigma_floor=5°`).
- `ClosedLoopResult(SimResult)` adds `q_est, b_est, bm_est, att_sigma, P_att, know_err, est_nadir_err, w_est, gyro_bias, sun_used, lit`; `meta` adds `seed, switch_time (first), switch_times, fallback_times, t_init, mag_bias_true, inertia_true, nis{name: {t, nis, dof, accepted}}, sensors, mekf, modes`.
- `summarize_closed_loop(res, settle=None)`: steady-state window = pointing and `t ≥ first switch + settle` (default 1 orbit). Keys: `t_init_s, switch_time_s/_orbits, detumble_time_s/_orbits` (truth rate, as Group 6), `n_switches, n_fallbacks, n_steady, nadir_{rms,p95,p99,max}_deg` (true pointing error), `know_{rms,rms_sunlit,rms_eclipse,max}_deg`, `ctrl_{rms,max}_deg` (estimate vs target), `nees_mean`, `wheel_max_frac` (window), `wheel_max_frac_pointing`, `dump_frac`, `dipole_max`, `gyro_bias_err_end_degps`, `mag_bias_true_nT`, `mag_bias_err_end_nT`, `nis_per_dof_<mag|sun|css>` (accepted, t ≥ t_init + 300 s), `success` (≥ 1 switch, no fallback, non-empty window). NaN where undefined.

### Prototype numbers (main session; `tumble_state(rate, seed)`, `seed` = sensor seed)

| Case | rate, orbits, seeds | switch [orbits] | nadir RMS / p95 / max [°] | know sunlit / eclipse [°] | NEES | other |
|---|---|---|---|---|---|---|
| low noise (all sensor σ × 0.01, filter matched, `MEKF_G7` noise fields matched) | 2 °/s, 2.5, 0 | 0.359 (1994 s, = PK) | 0.0533 / – / 0.0675 | 0.0042 total | 2.76 | PK same y0: switch 1994 s, RMS 0.0532, max 0.0695 |
| ideal + `MEKF_G7` | 2 °/s, 3, 0–2 | 0.37 / 0.80 / 0.85 | 0.38–0.60 / 0.74–1.28 / 1.05–2.28 | 0.30–0.37 / 0.48–0.79 | 2.78–3.06 | NIS/dof mag, sun 0.995–1.010 |
| realistic + `MEKF_G7` | 2 °/s, 3, 0–2 | as above | 24.5–26.6 / 42–50 / 51–53 | 26.8–28.2 / 20.3–23.2 | 6300–8300 | NIS/dof mag 63–86, sun 12.8–14.3 |
| realistic + `MEKF_DESIGN` | 2 °/s, 3, 0–2 | 0.37 / 0.80 / 0.85 | 0.67–0.92 / 1.36–2.01 / 1.71–2.76 | 0.71–1.03 / 0.58–1.00 | 11.2–17.4 | mag-bias error at end 26–48 nT of 365–403 nT; NIS/dof mag 0.18–0.22, css 0.78–0.81 |
| ideal + `MEKF_G7` | 10 °/s, 6, 0–2 | 1.16 / 1.64 / 1.70 | 0.47–0.60 / 0.93–1.25 / 1.49–2.18 | 0.34–0.38 / 0.55–0.78 | 2.96–3.11 | |
| realistic + `MEKF_DESIGN` | 10 °/s, 6, 0–2 | 1.17 / 1.64 / 1.70 | 0.98–1.11 / 1.87–2.28 / 2.51–3.38 | 0.93–0.95 / 0.97–1.21 | 13.6–16.1 | wheel ≤ 0.30 of max speed |
| ideal sensors + `MEKF_DESIGN` (albedo model with no albedo in truth) | 10 °/s, 6, 0–2 | | 1.56–1.80 / 3.0–3.5 / 4.8–5.5 | 1.81–2.06 / 0.96–1.35 | 5.6–7.3 | |

Wall time ≈ 1.9 ms per control period uncontended (≈ 27 s for 2.5 orbits; ≈ 3.5 × slower with 21 parallel processes).

Design-filter tuning (time-boxed, recorded for `assumptions.md`): σ_mag ∈ {65, 45, 100} nT and σ_albedo ∈ {0.1, 0.25, 0.5} were tried on seeds 0–2 (10 °/s, 5 orbits and 2 °/s, 3.5 orbits); ARW × 3 brings NEES to 5–7 but costs accuracy (not adopted); RRW × 10 diverged on one seed (attitude lost, no fallback — the rate-based fallback cannot detect a wrong attitude estimate). Chosen: σ_mag 100 nT, σ_albedo 0.5. The design filter stays overconfident (NEES 11–17) because the remaining errors (albedo, field model) are time-correlated.

### Tests (`adcs/tests/test_closed_loop.py`, test-author, from this spec)

Run the long scenarios once in a session-scoped fixture with a process pool (pattern of `adcs/tests/test_simulate_pk.py`); added suite wall time ≤ 60 s.

1. **Low-noise equivalence**: low-noise case of the table (all sensor σ × 0.01, `MEKF_G7` with `arw, rrw, sigma_mag, sigma_sun` matched) from `tumble_state(2°/s, 0)`, 2.5 orbits: switch time equals the PK `run(y0, ·, "auto")` switch time within 30 s; steady nadir RMS within 0.01° of the PK RMS over the same window (PK switch + 1 orbit); knowledge RMS < 0.02°.
2. **Ideal sensors**: `IDEAL_SENSORS` + `MEKF_G7`, 2 °/s, seed 0, 2.5 orbits: `success`; nadir RMS in (3 × PK RMS, 1.0°); NEES in [2.2, 4.0]; NIS/dof mag and sun in [0.9, 1.1].
3. **Realistic sensors, Group 7 filter fails**: `REALISTIC_SENSORS` + `MEKF_G7`, same y0/seed: knowledge RMS > 5°, NIS/dof mag > 10.
4. **Realistic sensors, design filter**: `REALISTIC_SENSORS` + `MEKF_DESIGN`, same y0/seed: `success`; nadir max < 5°; nadir RMS < 1.5°; mag-bias error at end < 0.3 × true bias norm; gyro-bias error at end < 0.02 °/s.
5. **Controller acts on the estimate**: in case 4, `ctrl_rms_deg < 0.5 × know_rms_deg` and `|nadir_rms − know_rms| < 0.3 × know_rms` (pointing error is dominated by knowledge error); the triangle inequality `nadir_err ≤ |know_err| + est_nadir_err + 1e-9` holds sample-wise.
6. **Mode machine** (short runs ≤ 0.4 orbit): no `MODE_POINTING` sample before `t_init + 300 s`; `att_sigma < 2°` at the switch; `ModeConfig(switch_att_sigma=0.0)` → never pointing, `switch_time` NaN, `success` False; `ModeConfig(safe_rate=1e-6)` → a fallback is recorded within `safe_hold + 2 s` after the switch and the mode returns to detumble; `MekfConfig(triad_min_angle=179°)` → never initialised (`t_init` NaN, `q_est` all NaN, never pointing).
7. **Reproducibility and streams**: identical results for equal seeds, different for different seeds; logged `gyro_bias[0]` equals the turn-on bias drawn by `Gyro(GyroConfig(), default_rng(seed).spawn(4)[0])`; `meta["mag_bias_true"]` equals `sensors.mag_bias_sigma × default_rng(seed).spawn(4)[3].spawn(3)[0].standard_normal(3)`; explicit `SensorErrors(mag_bias=...)` is used verbatim.
8. **Inertia**: `inertia_true=None` equals passing the nominal inertia (bit-identical); a 10 % different inertia changes the truth trajectory.
9. `summarize_closed_loop`: on a hand-built `ClosedLoopResult` (small arrays) the window, RMS/percentiles, NEES and `success` match hand computation.

## 4. Monte Carlo (`adcs/monte_carlo.py`, py-implementer; tests by test-author)

| Name | Behaviour |
|---|---|
| `CAMPAIGNS` | `{"design": (REALISTIC_SENSORS, MEKF_DESIGN), "ideal": (IDEAL_SENSORS, MEKF_G7)}` from `adcs.simulate` |
| `REQUIREMENTS` | dict (assumed mission requirements, set before the campaign; `assumptions.md` B-24): `detumble_orbits_max = 3.0` (truth rate < 0.5 °/s), `acquire_orbits_max = 4.0` (first switch to pointing), `pointing_max_deg = 5.0` (steady-state max nadir error), `knowledge_rms_deg = 2.0`, `pointing_rms_goal_deg = 1.0` (goal, not requirement), `wheel_frac_max = 1.0` (never saturate) |
| `RunSpec` | frozen dataclass: `index: int`, `base_seed: int`, `rate: float` (rad/s), `y0: tuple` (10 floats), `inertia_scale: tuple` (3 floats); property `sim_seed = (base_seed, index, 1)` |
| `draw_run(index, base_seed=2026, rate_range=(deg2rad(2), deg2rad(10)), inertia_tol=0.10)` | `rng = default_rng([base_seed, index, 0])`; draw in this order: `q = canonicalize(normalize(rng.normal(size=4)))`, `axis = rng.normal(size=3)` normalised, `rate = rng.uniform(*rate_range)`, `inertia_scale = rng.uniform(1 − tol, 1 + tol, size=3)`; `y0 = [q, rate·axis, 0, 0, 0]` |
| `run_one(spec, campaign="design", orbits=6.0, cfg=DEFAULT)` | `inertia_true = diag(diag(cfg.spacecraft.inertia) · inertia_scale)` (controller keeps the nominal inertia); `run_closed_loop(spec.y0, orbits · P, seed=list(spec.sim_seed), sensors, mekf_cfg, inertia_true=..., log_every=1)`; returns `{"index", "rate_degps", "inertia_scale" (list), **summarize_closed_loop(res), "runtime_s"}` (JSON-ready: floats, lists, bools; NaN allowed here) |
| `aggregate(rows)` | for every numeric key of the rows: `{"mean", "std", "median", "p5", "p95", "min", "max", "n_finite"}` over finite values (NaN-safe; all-NaN → NaN stats, n_finite 0); `"success_fraction"` |
| `check_requirements(rows, req=REQUIREMENTS)` | fraction of runs meeting each: `detumble_time_orbits ≤ detumble_orbits_max`, `switch_time_orbits ≤ acquire_orbits_max`, `nadir_max_deg ≤ pointing_max_deg`, `know_rms_deg ≤ knowledge_rms_deg`, `nadir_rms_deg ≤ pointing_rms_goal_deg`, `wheel_max_frac_pointing < wheel_frac_max`, `success`; NaN counts as not met; plus `"all_requirements"` = fraction meeting all except the goal; list of failing run indices per item |
| `run_campaign(campaign, n=100, orbits=6.0, workers=None, base_seed=2026)` | `draw_run(i)` for i in range(n), `ProcessPoolExecutor(max_workers=workers or max(1, os.cpu_count() − 2))`; sets `os.environ["OPENBLAS_NUM_THREADS"] = "1"` before creating the pool; rows sorted by index; returns `{"campaign", "config" (n, orbits, base_seed, rate_range_degps, inertia_tol, sensors/mekf as dicts, requirements), "runs", "aggregate", "requirements", "wall_time_s"}` |
| `save_json(result, path)` | writes with NaN → `null`, numpy scalars/arrays → Python floats/lists, indent 1 |
| CLI `python -m adcs.monte_carlo [--campaign design\|ideal\|all] [--n 100] [--orbits 6] [--workers N] [--base-seed 2026] [--out adcs/report] [--no-figures]` | default `all`; writes `<out>/monte_carlo_<campaign>.json`; prints a short table (success fraction, requirement fractions, median/p95/max of nadir RMS and max, detumble time); unless `--no-figures`, calls `adcs.report.monte_carlo_figures.main([...json paths])` if that module exists |

Expected cost: ≈ 220 s per 6-orbit run with 20 workers → ≈ 20 min per 100-run campaign.

### Tests (`adcs/tests/test_monte_carlo.py`, test-author)

1. `draw_run`: deterministic for equal (index, base_seed); different indices differ; over 2000 draws: rates within the range and roughly uniform (KS test p > 0.01), inertia scales within ±10 %, unit quaternion with w ≥ 0, axis uniform (mean ≈ 0 within 0.05); the documented draw order reproduces `y0` from a manually seeded generator.
2. `aggregate` / `check_requirements` on hand-built rows (including NaN and failing rows): exact values, failing indices.
3. `run_one` short (orbits = 0.03, campaign "ideal"): keys present, `inertia_scale` applied (compare `run_closed_loop` called directly with the same arguments → identical summary).
4. `run_campaign(n=2, orbits=0.03, workers=2)`: structure, rows sorted, equal to serial `run_one` results (excluding `runtime_s`); `save_json` output parses with `json.load` and contains `null` for NaN.

## 5. Figures and metrics (figures-runner)

### 5.1 `adcs/report/closed_loop_figures.py` → `adcs/report/closed_loop_metrics.json` + `adcs/report/figures/closed_loop_*.png`

All runs from `tumble_state(10 °/s, s)` with sensor seed `s`, default config, `log_every=1` (summaries) — keep raw arrays only for the runs that are plotted. Use a process pool (20 workers, `OPENBLAS_NUM_THREADS=1`).

| Case key | Runner / config | Seeds | Orbits |
|---|---|---|---|
| `pk` | `run(y0, ·, "auto")` (perfect knowledge, Group 6) — metrics with the same window definition (first switch + 1 orbit, pointing): nadir RMS/p95/p99/max, wheel max | 0–4 | 6 |
| `ideal_g7` | `IDEAL_SENSORS`, `MEKF_G7` | 0–4 | 6 |
| `real_g7` | `REALISTIC_SENSORS`, `MEKF_G7` (naive filter) | 0–4 | 6 |
| `real_design` | `REALISTIC_SENSORS`, `MEKF_DESIGN` | 0–4 | 6 |
| breakdown: `no_albedo` | realistic with `albedo=0`; design with `albedo_model=0` | 0–2 | 6 |
| `const_albedo` | realistic with `albedo_sigma=0` | 0–2 | 6 |
| `no_field_error` | realistic with `field_residual_sigma=0, onboard_field_degree=13` | 0–2 | 6 |
| `no_fov` | realistic with `css_fov=None`; design with `css_fov=None` | 0–2 | 6 |
| `no_mag_bias_states` | realistic; design with `mag_bias=False` | 0–2 | 6 |
| `design_no_albedo_truth` | `IDEAL_SENSORS`, `MEKF_DESIGN` (onboard albedo model, none in truth) | 0–2 | 6 |
| B-dot gain with sensors: `bdot_x{0.5,1,2,4}` | `REALISTIC_SENSORS`, `MEKF_DESIGN`, `ModeConfig(switch_att_sigma=0.0)` (detumble only), `ControlConfig(bdot_k = scale × bdot_gain(...))` | 0–4 | 3 |

Metrics JSON: per case and seed `summarize_closed_loop` (or the PK subset), per case aggregate (mean, min, max), and for the B-dot runs `detumble_time_orbits` and the mean true rate over the last orbit (°/s), compared with the Group 6 perfect-knowledge values in `control_metrics.json` (`detumble_sensitivity.gain_grid`). Record `config` (presets as dicts, seeds, orbits) and `checks`: `real_design` all seeds `success` and nadir max < 5°; `ideal_g7` NEES mean in [2.5, 3.5]; `real_g7` knowledge RMS > 5° for all seeds.

Figures (PNG 150 dpi, axis labels with units, eclipse shading where time is on the x axis):
1. `closed_loop_vs_pk.png` — seed 0, `pk` vs `ideal_g7` vs `real_design`: true rate (log), true nadir error (log, deg), wheel speed fraction, mode timeline.
2. `closed_loop_estimation.png` — `real_design` seed 0: knowledge error per axis with ±3σ (from `P_att` diagonal), magnetometer-bias estimate vs truth per axis (nT), gyro-bias error norm (°/s, log), number of CSS diodes used.
3. `closed_loop_cost.png` — nadir RMS and max (log y) per case `pk / ideal_g7 / real_design / real_g7`, seeds 0–4 as points + mean bar; caption: cost of estimation.
4. `closed_loop_breakdown.png` — knowledge RMS sunlit and eclipse per breakdown case vs `real_design` (seeds 0–2).
5. `closed_loop_bdot_gain.png` — detumble time vs gain scale with sensors (seeds 0–4) and the Group 6 perfect-knowledge means.

### 5.2 `adcs/report/monte_carlo_figures.py` → `adcs/report/figures/mc_*.png`

`main(paths=None)`: reads `adcs/report/monte_carlo_{design,ideal}.json` (whichever exist). Histograms (both campaigns overlaid when both exist, requirement lines dashed): `mc_detumble_time.png` (orbits), `mc_acquire_time.png` (switch time, orbits), `mc_pointing_rms.png`, `mc_pointing_max.png` (5° line), `mc_knowledge_rms.png` (2° line), `mc_wheel_speed.png` (max fraction in pointing); scatter `mc_sensitivity.png`: nadir RMS vs initial rate and vs `max(inertia_scale) − min(inertia_scale)`, coloured by campaign. Failing runs (if any) annotated with their index.
