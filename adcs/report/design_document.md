# 3U CubeSat ADCS — design document (Project B)

Simulation design study of the attitude determination and control system (ADCS) of a 3U CubeSat in a 400 km, 51.6° orbit: B-dot detumbling with magnetorquers, reaction-wheel nadir pointing with magnetic momentum dumping, TRIAD + multiplicative EKF (MEKF) attitude determination from a magnetometer, six coarse sun sensors (CSS) and a MEMS gyro, a closed loop that acts only on the estimate, and a 100-run Monte Carlo campaign.

**Summary** (simulation results; sources in sections 4–6):

- The closed loop runs autonomously on the estimated attitude from a 2–10 °/s tumble to nadir pointing. In 100 Monte Carlo runs with realistic sensor errors and ±10 % inertia uncertainty, detumbling takes 1.22 orbits (median; max 2.29) and pointing is acquired after 1.21 orbits (median; max 2.23).
- Steady-state nadir pointing error is 0.96° RMS (median; p95 1.25°), with a worst-case maximum of 5.19°. 99 of 100 runs meet all requirements (REQ-ROB ≥ 95 % is met). The 1° RMS goal is met in 58 runs.
- Estimation sets the pointing error: perfect knowledge gives 0.054° RMS, ideal sensors with a matched filter 0.53°, and realistic sensor errors with the design filter 0.99° (5-seed means).
- The Group 7 filter fails with realistic sensor errors (≈ 24° RMS) without any onboard indication. Magnetometer-bias states and per-diode sun-sensor updates fix it.
- The design filter stays overconfident (NEES ≈ 14 against 3), because the field-model error is time-correlated. The few largest pointing excursions are knowledge errors outside the filter's own 3σ.

Conventions: `CONVENTIONS.md` (scalar-first Hamilton quaternions, body-to-ECI; right/body-frame attitude error; SI units, degrees only in tables and figures). Assumptions are cited by ID from `assumptions.md` (`[B-xx]`). Every number below comes from a metrics file named next to it (paths relative to `adcs/report/`) or is labelled as a hand calculation. **All results are simulation results**; section 8 lists what the simulation leaves out.

## 1. Mission context and requirements

There is no real payload behind this study. The requirements below are **assumed** for a typical 3U mission with a nadir-pointing patch antenna or wide-field camera and were fixed before the Monte Carlo campaign was run [B-24]; they are the yardstick for the design, not derived needs.

| ID | Requirement | Verification |
|---|---|---|
| REQ-DET | After separation with a body rate up to 10 °/s, reduce the rate below 0.5 °/s within 3 orbits using magnetorquers only | Monte Carlo (truth rate), section 6 |
| REQ-ACQ | Acquire nadir pointing autonomously (switch to pointing on the estimate, no ground command) within 4 orbits | Monte Carlo, section 6 |
| REQ-PNT | Steady-state nadir pointing error ≤ 5° at all times (window: from one orbit after the switch to pointing) | Monte Carlo, section 6 |
| REQ-KNW | Attitude knowledge error ≤ 2° RMS in steady-state pointing | Monte Carlo, section 6 |
| REQ-MOM | Reaction wheels never reach their speed limit in pointing | Monte Carlo, section 6 |
| REQ-ROB | The above hold for sensor noise, realistic sensor errors [B-21], random initial tumble and ±10 % inertia uncertainty in ≥ 95 % of Monte Carlo runs | Monte Carlo, section 6 |
| GOAL-PNT | Steady-state nadir pointing error ≤ 1° RMS (goal, not a requirement) | Monte Carlo, section 6 |

"Pointing error" is the true angle between the body frame and the LVLH target frame (all three axes); "knowledge error" is the angle of `q_true ⊗ q_est*`.

## 2. Spacecraft, environment and hardware

| Item | Model | Ref. |
|---|---|---|
| Spacecraft | 3U, 4 kg, uniform box inertia diag(0.0419, 0.0419, 0.0067) kg·m², CoM offset (5, 5, 10) mm | B-01…B-03, B-09 |
| Orbit | circular 400 km, 51.6°, two-body + J2; epoch 2026-03-20 (β ≈ 0, eclipse fraction 0.39 — longest eclipses) | B-04, B-13…B-15 |
| Geomagnetic field | IGRF-14 (degree 13) via `ppigrf`; 19–54 µT along the orbit | B-11 |
| Disturbances | gravity gradient, residual dipole 1 mA·m²/axis, aerodynamic (exponential atmosphere); aero is the largest (≈ 1e-7 N·m) | B-08, B-10, B-12 |
| Magnetorquers | 3 × 0.2 A·m² (ISIS iMTQ class), 1 s control period, 0.8 duty (magnetometer read with coils off) | B-05, B-17 |
| Reaction wheels | 3 × CubeSpace CW0017: 0.23 mN·m, 1.77 mN·m·s at 8000 rpm | B-06 |
| Magnetometer | PNI RM3100, 15 nT white noise | B-07 |
| Gyro | ADXRS453: ARW 0.65 °/√h, bias random walk from 16 °/h in-run stability, turn-on bias 0.1 °/s per axis | B-07 |
| Coarse sun sensors | 6 × cosine photodiodes (Tensor Tech CSS-10 class), one per face, 5° (1σ) vector accuracy | B-07 |

**Realistic sensor errors.** Group 7 verified the estimator with ideal sensors (white noise only, filter reference field = truth field). Group 8 adds truth-side errors that a flight system meets [B-21]: the onboard field model is IGRF truncated to degree 8 while the truth is the full model plus an unmodelled Gauss–Markov field (50 nT/axis, τ 600 s); a constant magnetometer bias drawn per run (200 nT/axis); a hard ±60° CSS field of view; and Earth albedo on the CSS (Lambertian Earth, plate-to-sphere view factor, coefficient 0.3 ± 0.1 varying with τ 300 s). These values are assumptions of plausible magnitude, not measured hardware data.

## 3. Architecture

### 3.1 Modes

```
           separation
               |
               v
   +-----------------------+   MEKF initialised (TRIAD) and converged (sqrt tr P_att < 2 deg,
   | DETUMBLE              |   >= 300 s of filtering) and |gyro - b_hat| < 0.5 deg/s for 60 s
   | B-dot on measured B   | ------------------------------------------------------------+
   | wheels idle           |                                                             |
   | MEKF runs in background| <--------------------------+                                v
   +-----------------------+   |gyro - b_hat| > 2 deg/s  |                 +---------------------------+
                               for 10 s (fallback)       +---------------- | NADIR POINTING            |
                                                                          | PD on (q_hat, w_hat),      |
                                                                          | wheels + magnetic dumping  |
                                                                          +---------------------------+
```

The estimator runs from the first usable sun measurement onwards, also during detumbling: TRIAD needs a valid sun vector and a field–sun angle of at least 20° (the TRIAD rotation about the primary vector has σ ∝ 1/sin α [B-19]); with the design filter all three CSS axes must be lit so that the sun vector is fully measured. The switch to pointing is gated on the filter itself (covariance and time) and on the **bias-corrected** gyro rate, because the uncorrected turn-on bias (0.17 °/s norm, 1σ [B-07]) is a third of the 0.5 °/s threshold [B-23].

### 3.2 Control laws (unchanged from the perfect-knowledge design [B-16])

- **B-dot** on the measured field direction, `m = −(k/|B|) d(b̂)/dt`, `k = 2n(1 + sin i) J_min` = 2.69e-5 N·m·s (Avanzini & Giulietti gain). It needs only the magnetometer, so detumbling does not depend on the estimator.
- **Nadir pointing**: rate-limited quaternion-error PD with gyroscopic and reference-rate feed-forward (ω_n = 0.1 rad/s, ζ = 0.7, slew limit 0.5 °/s), on the wheels.
- **Momentum dumping**: cross-product law `m = (k/|B|²) h_w × B` (k = 1e-3 1/s) with hysteresis on wheel speed; the expected magnetorquer torque is fed forward to the wheels.

With estimation the controller receives `q̂` and `ω̂ = gyro − b̂` (mean rate of the previous period, T/2 = 0.5 s lag); dumping and feed-forward use the measured field minus the estimated magnetometer bias. Wheel speeds come from the wheel tachometers and the nadir reference from onboard orbit knowledge, both taken as perfect [B-23].

### 3.3 Estimator

The MEKF (error state `[δθ, δb_g]`, closed-form discretisation, Joseph-form updates, inject-and-reset after every update [B-19]) propagates with the gyro at 1 Hz and updates with the magnetometer and the CSS. Two configurations are compared:

| | Group 7 filter (`MEKF_G7`) | Design filter (`MEKF_DESIGN`) [B-22] |
|---|---|---|
| States | 6: attitude, gyro bias | 9: + magnetometer bias |
| Magnetometer R | 15 nT (sensor noise) | 100 nT (sensor noise + onboard field-model error) |
| Sun measurement | reconstructed CSS vector `I(+axis) − I(−axis)`, 3-dof | per-diode scalar updates, only diodes whose predicted incidence is below 56.6° (predicted cosine > 0.55; FOV 60°), onboard albedo subtracted, albedo uncertainty in R |
| Assumes | ideal hemispherical CSS, exact field model, no magnetometer bias | ±60° FOV, albedo, bias, field-model error |

Design rationale:
- **Magnetometer bias states** instead of a larger R: an unmodelled constant bias is a correlated error, which a larger R cannot represent (Group 7 review: 300 nT bias gave NEES ≈ 815, and still ≈ 29 with R inflated to 300 nT [B-20]). The bias is observable because the body-frame field direction changes over the orbit.
- **Per-diode updates** instead of the reconstructed vector: with a ±60° FOV the reconstruction `I(+) − I(−)` silently zeroes every sun-vector component below cos 60° = 0.5, so the vector is wrong by tens of degrees for most directions (Group 7 review: 14.6° RMS vector error [B-20]). A scalar update per diode uses only what a diode actually measures. Diodes are selected on the *predicted* incidence, not on the measured current, because selecting on the noisy current biases the innovations (matched-model NEES 3.3–4.8 with current gating vs 2.6–3.5 with predicted gating, spec §2).
- **R inflation** for the field-model error, which is spatially correlated along the orbit and cannot be estimated with a few states.

### 3.4 Timing

One control period T = 1 s: sample all sensors at `t_k` (magnetometer with the coils off), run one MEKF cycle, run the mode machine, compute the dipole and wheel torque, hold them for the period (coils on for 0.8 s) [B-17, B-23]. Computation latency is taken as zero.

## 4. Results by phase

All results are simulation results. Unless stated, the closed loop starts from a 10 °/s tumble (`tumble_state(10 °/s, s)`, sensor seed `s` = 0–4), runs 6 orbits (orbital period 5553.6 s, `closed_loop_metrics.json` `config.orbital_period_s`), and statistics use the steady-state window "pointing and t ≥ first switch + 1 orbit" (same definition for the perfect-knowledge case). Four cases are compared:

| Case key | Sensors | Filter |
|---|---|---|
| `pk` | perfect knowledge (Group 6 controller on the truth state) | none |
| `ideal_g7` | ideal (white noise only, onboard field = truth) | `MEKF_G7` |
| `real_design` | realistic [B-21] | `MEKF_DESIGN` [B-22] |
| `real_g7` | realistic [B-21] | `MEKF_G7` (naive filter) |

Values are mean (min–max) over seeds, from `closed_loop_metrics.json` `cases.<key>.aggregate` unless another source is named. Three-seed breakdown cases (section 4.5) use seeds 0–2.

### 4.1 Detumbling

Detumbling uses B-dot on the *measured* field and does not depend on the estimator [B-16]. The true-rate detumble time (rate stays below 0.5 °/s, as B-18) in the 6-orbit runs with automatic switching is:

| Case | Detumble time [orbits] | Source |
|---|---|---|
| `pk` | 1.399 (0.946–1.680) | `closed_loop_metrics.json` `cases.pk` |
| `ideal_g7` | 1.406 (0.953–1.695) | `cases.ideal_g7` |
| `real_design` | 1.406 (0.953–1.693) | `cases.real_design` |
| `real_g7` | 1.405 (0.955–1.693) | `cases.real_g7` |

Group 6 (perfect knowledge, B-dot only, `control_metrics.json`) gives 1.365 orbits mean, 1.680 maximum.

**B-dot gain sweep with sensors.** Realistic sensors, design filter, detumble-only runs (`ModeConfig(switch_att_sigma=0.0)`), 3 orbits, seeds 0–4, compared with the Group 6 perfect-knowledge sweep of the same seeds (6-orbit runs). Source: `closed_loop_metrics.json` `bdot_gain`; figure `figures/closed_loop_bdot_gain.png`.

| Gain scale | `bdot_k` [N·m·s] | With sensors, mean (min–max) [orbits] | Group 6 mean [orbits] | Ratio | Mean true rate, last orbit [°/s] |
|---|---|---|---|---|---|
| 0.5 | 1.345e-5 | 2 of 5 seeds below 0.5 °/s within 3 orbits (2.386, 1.837); 3 not | 2.917 (6-orbit runs; seeds 1, 2, 4: 3.317, 3.306, 3.798) | n/a | 0.710 |
| 1 (nominal) | 2.691e-5 | 1.363 (0.877–1.677) | 1.365 | 0.9986 | 0.170 |
| 2 | 5.381e-5 | 0.732 (0.513–0.919) | 0.732 | 0.9998 | 0.101 |
| 4 | 1.076e-4 | 0.425 (0.300–0.492) | 0.426 | 0.9985 | 0.104 |

At the nominal gain the per-seed times with sensors (1.143, 1.618, 1.677, 0.877, 1.498) match Group 6 (1.136, 1.621, 1.680, 0.879, 1.507) to within 0.01 orbit. The 0.5× row is limited by the 3-orbit run length, not by sensors: the three seeds that did not finish are the ones Group 6 needs more than 3 orbits for, and the two that finished agree with Group 6 (2.386 vs 2.325 and 1.837 vs 1.840 orbits).

Conclusions: (1) magnetometer noise, bias, and the onboard-model error do not change the detumble time at any tested gain (ratios 0.9985–0.9998 where defined); (2) the concern behind the Group 6 decision to keep the nominal gain, noise amplification in the finite-difference Ḃ [B-16], does not appear in these metrics, since 2× and 4× the gain detumble faster with no visible penalty in the residual rate. **The nominal gain is nevertheless kept**: it meets REQ-DET with margin (section 5), and the faster gains were not examined for other effects (for example the hand-over to pointing), so they are listed as a possible improvement, not adopted.

### 4.2 Initialisation and acquisition

The MEKF is initialised by TRIAD at the first sample that meets the guards of section 3.1; the switch to pointing follows once the filter has run 300 s, `√tr P_att` < 2° and the bias-corrected rate has stayed below 0.5 °/s for 60 s [B-23].

| Case | `t_init` [s] | First switch [orbits] | Switch times, seeds 0–4 [s] |
|---|---|---|---|
| `pk` | n/a | 1.375 (0.889–1.691) | 6367, 9060, 9390, 4939, 8424 |
| `ideal_g7` | 0 | 1.387 (0.919–1.698) | 6422, 9109, 9429, 5102, 8460 |
| `real_g7` | 0 | 1.385 (0.922–1.693) | 6422, 9109, 9403, 5120, 8398 |
| `real_design` | 141 (55–269) | 1.388 (0.919–1.695) | 6491, 9109, 9415, 5102, 8423 |

Source: `closed_loop_metrics.json` `cases.<key>` (per-seed switch times and aggregates). In the three closed-loop cases no fallback to detumble occurred (`n_fallbacks` = 0) and every seed is `success`.

The switch happens within 163 s (0.029 orbit, hand calculation 163 s / 5553.6 s) of the perfect-knowledge switch in every seed and case; the largest difference is seed 3 (4939 s for `pk`, 5102 s for `ideal_g7` and `real_design`). Acquisition time is therefore set by the detumble, not by the estimator, when the filter initialises early. The design filter initialises later than the Group 7 filter (141 s against 0 s) because it needs all three CSS axes lit [B-22]. With the CSS field-of-view limit removed (`no_fov`, seeds 0–2) `t_init` is 30.7 s (16–45), against 177 s (120–269) for `real_design` on the same seeds. In the Monte Carlo campaign the tail is longer (section 5).

### 4.3 Attitude determination

**Ideal-sensor floor.** Group 7 on a prescribed pointing truth (5 orbits, seed 0, after one orbit): knowledge RMS 0.531° (sunlit 0.346°, eclipse 0.731°), NEES 2.999 (`estimation_metrics.json` `scenarios.pointing`). The closed loop with ideal sensors reproduces it (`ideal_g7`): knowledge RMS 0.513° (0.445–0.627), sunlit 0.347° (0.320–0.380), eclipse 0.697° (0.547–0.908), NEES 3.05 (2.96–3.21), NIS per dof magnetometer 1.001 and sun 1.000, gyro-bias error at the end 0.0023 °/s (0.0009–0.0040). The filter is consistent here because its noise parameters equal the simulated ones [B-19]. Eclipse knowledge is worse than sunlit because without the sun vector the filter relies on the change of the field direction relative to the spacecraft [B-19].

**Design filter with realistic sensors** (`real_design`, seeds 0–4; figure `figures/closed_loop_estimation.png` shows seed 0):

| Quantity | Value |
|---|---|
| Knowledge RMS, all / sunlit / eclipse [°] | 0.960 (0.768–1.067) / 0.917 (0.787–0.993) / 1.021 (0.735–1.214) |
| Magnetometer-bias error at end [nT] | 26.2 (12.5–40.6) against true norm 380.5 (183.1–558.8) |
| Same, as fraction of the true norm, per seed | 6.9 %, 10.3 %, 7.5 %, 2.2 %, 12.6 % (hand calculation from per-seed values) |
| Gyro-bias error at end [°/s] | 0.0033 (0.0011–0.0066) |
| NIS per dof, magnetometer / CSS | 0.181 (0.168–0.196) / 0.799 (0.787–0.806) |
| Attitude NEES (mean) | 13.7 (11.8–16.1) |

Source: `closed_loop_metrics.json` `cases.real_design`.

- **Accuracy.** The knowledge RMS is about 1° with little sunlit/eclipse contrast, in contrast to the ideal case (0.35°/0.70°). The cause was not isolated.
- **Bias estimation.** The magnetometer bias is recovered to 2–13 % of its norm and the gyro bias to a few 1e-3 °/s.
- **Consistency.** NEES 13.7 against 3 for a consistent 3-dof error: the filter is **overconfident** by a factor of about 4.6 (hand ratio 13.7/3), with magnetometer NIS per dof far below 1 (R inflated to 100 nT) and CSS NIS per dof 0.80. The inflation cannot fix this because the remaining errors are time-correlated (section 4.5). The ±3σ bands in `figures/closed_loop_estimation.png` are therefore too tight in places.

### 4.4 Pointing and the cost of estimation

Steady-state true nadir pointing error and knowledge, mean (min–max) over seeds 0–4. Source: `closed_loop_metrics.json` `cases.<key>.aggregate`; figures `figures/closed_loop_cost.png` and `figures/closed_loop_vs_pk.png` (seed 0, time histories).

| Case | Nadir RMS [°] | p95 [°] | Max [°] | Knowledge RMS [°] | NEES |
|---|---|---|---|---|---|
| `pk` | 0.0537 (0.0532–0.0540) | 0.0663 (0.0660–0.0666) | 0.0679 (0.0676–0.0690) | n/a | n/a |
| `ideal_g7` | 0.532 (0.466–0.633) | 1.130 (0.933–1.407) | 1.915 (1.492–2.341) | 0.513 (0.445–0.627) | 3.05 (2.96–3.21) |
| `real_design` | 0.987 (0.786–1.113) | 1.965 (1.506–2.283) | 3.015 (2.508–3.375) | 0.960 (0.768–1.067) | 13.7 (11.8–16.1) |
| `real_g7` | 24.4 (22.8–25.8) | 42.0 (34.5–48.0) | 57.2 (54.9–61.2) | 24.1 (22.4–25.4) | 6377 (4865–7823) |

- **Perfect knowledge to ideal-sensor estimation: ×10** in nadir RMS (0.532 / 0.0537 = 9.9, hand calculation). The pointing error becomes knowledge-limited: nadir RMS 0.532° against knowledge RMS 0.513°. The controller's own tracking error, estimate against target, stays at 0.129° RMS (0.124–0.134) (`ctrl_rms_deg`).
- **Ideal-sensor to realistic sensors: ×1.9** (0.987 / 0.532 = 1.86, hand calculation), about ×18 against perfect knowledge. Knowledge RMS (0.960°) and nadir RMS (0.987°) are again close; the controller-visible error is 0.139° (0.133–0.147).
- **Requirements.** All five `real_design` seeds succeed with nadir max below 5° (`checks`), but the 1° RMS goal is exceeded in three of five seeds (1.113°, 1.013°, 1.048°; seeds 0, 2, 4).
- **The naive filter is not usable.** `real_g7` points to about 24° RMS with maxima of 55–61°, NEES in the thousands, magnetometer NIS per dof 89 (17.9–185.8), sun NIS per dof 13.1, and a gyro-bias error that is still 0.079 °/s (0.079–0.081) at the end (for scale, the turn-on bias norm is 0.17 °/s, 1σ [B-07]). The unestimated magnetometer bias (error at the end equals the true norm) and the vector-reconstruction error from the FOV limit and albedo [B-20, B-21] explain this qualitatively; the individual contributions were not separated for the naive filter. The mode logic cannot see it: all five seeds switch once, never fall back, and are flagged `success`, because the controller sees only its own estimate (`ctrl_rms_deg` 0.738°, 0.677–0.779) while the true error is 24°. This is the same weakness as the missing attitude-consistency monitor [B-23]; see section 6 (F4) and section 8.

### 4.5 Error-source breakdown

Each row removes or changes one error source relative to `real_design` (descriptions in `closed_loop_metrics.json` `config.breakdown_description`), 10 °/s, 6 orbits, **seeds 0–2**. Source: `closed_loop_metrics.json` `cases.<key>.aggregate`; the `real_design` row is the **hand mean over seeds 0–2** of `cases.real_design.per_seed` (the 5-seed aggregate in section 4.4 is not comparable). Figure: `figures/closed_loop_breakdown.png`.

| Case | Change | Nadir RMS [°] | p95 [°] | Max [°] | Knowledge sunlit / eclipse [°] | NEES | Mag-bias error at end [nT] |
|---|---|---|---|---|---|---|---|
| `real_design` (seeds 0–2) | none | 1.034 | 2.024 | 2.889 | 0.935 / 1.091 | 14.5 | 31.9 |
| `no_albedo` | truth albedo 0, filter albedo model 0 | 0.851 | 1.721 | 2.611 | 0.734 / 0.933 | 14.0 | 31.9 |
| `const_albedo` | truth albedo constant 0.30 | 0.875 | 1.753 | 2.577 | 0.756 / 0.958 | 14.0 | 31.9 |
| `no_field_error` | truth field residual 0, onboard degree 13 | 0.990 | 1.994 | 2.716 | 0.888 / 1.048 | 2.29 | 0.25 |
| `no_fov` | no FOV limit in truth and filter | 1.034 | 2.024 | 2.889 | 0.935 / 1.091 | 14.4 | 31.7 |
| `no_mag_bias_states` | filter without bias states | 1.451 | 2.546 | 3.300 | 1.327 / 1.533 | 157 | 386.9 (= true norm) |
| `design_no_albedo_truth` | ideal sensors, filter keeps its albedo model | 1.713 | 3.250 | **5.253** | 1.953 / 1.151 | 6.49 | 0.35 (true bias 0) |

**Seed-spread caveat.** There are 3 seeds. Across seeds 0–2 the `real_design` nadir RMS ranges 0.976–1.113° (0.14°) and the maximum 2.51–3.37°. The nadir-RMS differences from the `real_design` hand mean (0.18° `no_albedo`, 0.16° `const_albedo`, 0.04° `no_field_error`, 0.00° `no_fov`; hand calculation) are of the order of this spread and **are not significant**. The robust conclusions are the large effects:

1. **The bias states matter most.** Without them NEES rises from 14.5 to 157, nadir RMS from 1.03° to 1.45°, and the bias is of course not estimated.
2. **The field-model error drives the overconfidence.** Without it NEES is 2.29 (2.23–2.38), close to the consistent value of 3, and the bias error falls to 0.25 nT. The knowledge RMS barely changes (0.954° against 0.999°), so accuracy is not limited by the field error alone.
3. **An albedo model that does not match the truth hurts.** With no albedo in the truth but the filter subtracting 0.3, nadir RMS is 1.71° and the maximum exceeds the 5° requirement (5.25°, 4.82–5.53). The sunlit knowledge error (1.95°) is larger than the eclipse error (1.15°), consistent with the albedo term acting only in sunlight [B-21], which was not tested separately. The design filter depends on its albedo model being representative of the truth.
4. **The FOV limit has no measurable effect on pointing** (nadir RMS, p95, max identical to `real_design` to the printed digits); the filter's diode gate (predicted cosine > 0.55) lies inside the cutoff (cos 60° = 0.5) [B-22]. It does delay initialisation (section 4.2).

### 4.6 Momentum management

Dumping is a cross-product law with on/off thresholds at 0.3 and 0.05 of the maximum wheel speed [B-16]. `wheel_max_frac_pointing` is the maximum wheel speed fraction over the whole pointing phase; `dump_frac` is the fraction of steady-state-window samples with dumping active. Source: `closed_loop_metrics.json` `cases.<key>.aggregate`; Group 6 values from `control_metrics.json` (`pointing_15orbits.cases`, `fraction_of_time_dumping`).

| Case | Peak wheel fraction, pointing phase | Dumping active, steady window |
|---|---|---|
| `pk` | 0.3003 (0.3000–0.3013) | not in `closed_loop_metrics.json`; Group 6, 15-orbit run: 0.922 |
| `ideal_g7` | 0.3011 (0.3004–0.3017) | 0.932 (0.812–1.000) |
| `real_design` | 0.3019 (0.3000–0.3080) | 0.944 (0.831–1.000) |
| `real_g7` | 0.3044 (0.3001–0.3133) | 0.403 (0.332–0.469) |

The peak fraction sits at the dump-on threshold of 0.3 in every case, so the wheels stay far from the limit (REQ-MOM requires < 1). Dumping is active for most of the steady state, as in Group 6 (the wheel settles where dumping balances the secular aerodynamic torque, an interpretation recorded in B-16). In `real_g7` dumping is active only 40 % of the time; the dumping law uses the measured field minus the estimated bias, which is not estimated there, and the attitude is wrong. The mechanism was not analysed. The peak steady-window dipole of `real_design` is below 0.0204 A·m² in the Monte Carlo (section 5) against the 0.2 A·m² limit [B-05].

## 5. Monte Carlo campaign

### 5.1 Setup

Source for all numbers in this section: `monte_carlo_design.json`, `monte_carlo_ideal.json` (`config`, `aggregate`, `requirements`) and `monte_carlo_summary.md`, which is generated from those files only.

| Item | Value |
|---|---|
| Runs per campaign | 100 |
| Duration per run | 6 orbits |
| Base seed | 2026 (run `i` uses `default_rng([2026, i, 0])` for the initial condition and sensor seed `(2026, i, 1)`) |
| Initial body rate | uniform 2–10 °/s (drawn range 2.006–9.934 °/s), random axis |
| Initial attitude | uniformly random quaternion |
| Inertia | each principal moment scaled by U(0.9, 1.1) on the **truth only**; the controller and filter keep the nominal inertia [B-23] |
| Sensor errors | per-run seeds for gyro, magnetometer and CSS noise; in the `design` campaign also per-run draws of the magnetometer bias (`σ` 200 nT per axis), field residual and albedo [B-21] |
| `design` campaign | `REALISTIC_SENSORS` + `MEKF_DESIGN`; wall time 1346 s (20 workers) |
| `ideal` campaign | `IDEAL_SENSORS` + `MEKF_G7` (Group 7 floor); wall time 1156 s |
| Requirements | assumed and fixed before the campaign [B-24], section 1 |

Both campaigns use the same 100 initial conditions (the draw does not depend on the campaign, spec `docs/specs/adcs_closed_loop.md` §4). In the `design` campaign the true magnetometer-bias norm has mean 316 nT, median 320 nT, 5th–95th percentile 119–549 nT and maximum 709 nT (`aggregate.mag_bias_true_nT`); the expected norm for σ = 200 nT is 319 nT [B-21].

### 5.2 Requirement compliance

| Requirement | Criterion | `design`: met (failing run indices) | `ideal`: met (failing run indices) |
|---|---|---|---|
| REQ-DET | detumble time ≤ 3 orbits | 100/100 (none) | 100/100 (none) |
| REQ-ACQ | first switch ≤ 4 orbits | 100/100 (none) | 100/100 (none) |
| REQ-PNT | steady-state nadir max ≤ 5° | 99/100 (**54**) | 100/100 (none) |
| REQ-KNW | knowledge RMS ≤ 2° | 100/100 (none) | 100/100 (none) |
| REQ-MOM | peak wheel fraction in pointing < 1 | 100/100 (none) | 100/100 (none) |
| `success` | switch, no fallback, non-empty window | 100/100 (none) | 100/100 (none) |
| All requirements (goal excluded) | | 0.99 (54) | 1.00 (none) |
| GOAL-PNT | nadir RMS ≤ 1° | 58/100 (42 runs, listed in `monte_carlo_design.json` `requirements.pointing_rms_goal_deg.failing`) | 100/100 (none) |

REQ-ROB (≥ 95 % of runs meet all requirements) is met by both campaigns (0.99 and 1.00). With 100 runs the resolution is 1 %.

### 5.3 Distributions

Median / 95th percentile / maximum over the 100 runs (`monte_carlo_summary.md`).

| Metric | `design` | `ideal` |
|---|---|---|
| Detumble time [orbits] | 1.22 / 1.88 / 2.29 | 1.21 / 1.63 / 1.89 |
| First switch time [orbits] | 1.21 / 1.88 / 2.23 | 1.17 / 1.66 / 1.85 |
| Nadir RMS [°] | 0.962 / 1.25 / 1.46 | 0.503 / 0.635 / 0.733 |
| Nadir max [°] | 3.15 / 4.44 / 5.19 | 1.84 / 2.90 / 3.59 |
| Knowledge RMS [°] | 0.946 / 1.22 / 1.43 | 0.481 / 0.617 / 0.717 |
| Attitude NEES (mean) | 13.7 / 16.4 / 19.9 | 3.00 / 3.25 / 3.59 |
| Peak wheel fraction, pointing phase | 0.302 / 0.374 / 0.390 | 0.301 / 0.365 / 0.376 |
| Peak wheel fraction, steady window | 0.300 / 0.303 / 0.307 | 0.300 / 0.303 / 0.305 |

Further `design` statistics (`monte_carlo_design.json` `aggregate`): `t_init` median 116.5 s, 95th percentile 10088 s, maximum 12024 s; magnetometer-bias error at end 22.1 / 32.9 / 42.5 nT; gyro-bias error at end 0.0020 / 0.0034 / 0.0072 °/s; peak steady-window dipole 0.0117 / 0.0169 / 0.0204 A·m²; dumping active 0.961 of the window (median; minimum 0.817). The `design` acquisition maximum (12384 s) follows the largest `t_init` by 360 s, so in the longest run the time to pointing is set by when the filter could initialise. The `design` detumble tail (95th percentile 1.88 against 1.63 orbits) differs from `ideal` on identical initial conditions; the cause was not investigated.

The 5-seed closed-loop results of section 4.4 understate the tail: their `real_design` nadir maximum is at most 3.37°, while 100 runs reach 5.19°. The `ideal` campaign (nadir RMS median 0.503°) agrees with the 5-seed `ideal_g7` mean of 0.532°.

### 5.4 Figures

All in `figures/`, both campaigns overlaid, requirement or goal lines dashed, failing runs annotated with their index.

| Figure | Content |
|---|---|
| `figures/mc_detumble_time.png` | Histogram of true-rate detumble time [orbits] |
| `figures/mc_acquire_time.png` | Histogram of the first switch time [orbits] |
| `figures/mc_pointing_rms.png` | Histogram of steady-state nadir RMS [°], 1° goal |
| `figures/mc_pointing_max.png` | Histogram of steady-state nadir maximum [°], 5° requirement |
| `figures/mc_knowledge_rms.png` | Histogram of knowledge RMS [°], 2° requirement |
| `figures/mc_wheel_speed.png` | Histogram of peak wheel speed fraction in pointing |
| `figures/mc_sensitivity.png` | Nadir RMS against initial rate and against the spread of the inertia scale factors (max − min), per campaign |

### 5.5 Margin of the pointing requirement

The `design` campaign has little margin on REQ-PNT. The 95th percentile of the steady-state nadir maximum is 4.44°, 0.56° below the 5° limit (hand calculation); the three largest values are 5.19° (run 54, the one failure), 4.96° (run 20) and 4.78° (run 35), so the next two runs are within 0.04° and 0.22° of the limit (`monte_carlo_summary.md`, worst-three table). The `ideal` campaign has 1.41° of margin at its maximum (5 − 3.59, hand calculation). The knowledge requirement has 0.57° of margin at the `design` maximum (2 − 1.43). The 99 % compliance meets REQ-ROB but is a point estimate from 100 runs; the `design` maxima lie well above the `ideal` ones on identical initial conditions (median 3.15° against 1.84°), so the margin is consumed by the realistic sensor errors (sections 4.4–4.5). The causes of run 54 are discussed in section 6.

## 6. Failure modes

### 6.1 Observed in the Monte Carlo

**F1 — Pointing excursions are knowledge errors the filter does not know about.** The one requirement failure (run 54, 5.19°) and the next four largest maxima were re-run with identical inputs (`mc_worst_runs.json`, `python -m adcs.report.mc_worst_runs`; every run reproduced its campaign maximum exactly). At the maximum the knowledge error is 4.17–4.93° while the controller's own error (estimate vs target) is 0.10–0.36°, so the attitude loop is doing what it is told; the estimate is wrong. In all five runs the knowledge error at the maximum exceeds the filter's 3σ (`√tr P_att`: 2.67–3.39°), and the knowledge error is outside 3σ for 2–11 % of the steady-state time. Four of the five maxima occur in sunlight 660–825 s after eclipse exit, one (run 54) 433 s after eclipse entry, at the end of an error build-up during the preceding sunlit pass (`figures/mc_worst_run.png`). The largest component is about body z (−4.2 … +3.4°), the axis with the smallest inertia and, in nadir pointing with the sun near the orbit plane, the axis along which a single lit side diode gives little information. This is the same overconfidence as in section 4.3 (NEES 13.7 median over the campaign): time-correlated errors (field model, albedo variation) that R inflation cannot represent. The mechanism behind the specific timing was not isolated further.

**F2 — Late filter initialisation delays acquisition.** The design filter initialises TRIAD only when all three CSS axes are lit and the field–sun angle is ≥ 20° [B-22]. In 14 of 100 design runs `t_init` exceeds 1000 s (up to 12024 s, `monte_carlo_design.json`); in two of them the switch to pointing comes 1605 s and 2955 s after the rate is already below threshold (hand calculation from `switch_time_s − detumble_time_s`), i.e. acquisition is initialisation-limited. All runs still acquire within 2.23 orbits (REQ-ACQ ≤ 4). The condition becomes rare once the body rate is low, because the sun then stays in a near-fixed body direction for long periods.

**F3 — No fallbacks and no lost runs.** Neither campaign had a fallback, a crashed run or a run that failed to switch (`monte_carlo_*.json`). Nadir RMS shows no visible trend with the initial rate or the ±10 % truth inertia error (`figures/mc_sensitivity.png`; visual reading, not tested statistically); wheel speeds stayed at the 0.3 dump-on threshold (peak 0.39 in pointing).

### 6.2 Demonstrated outside the nominal design

**F4 — An estimator built on ideal-sensor assumptions fails silently.** With realistic sensor errors the Group 7 filter points to ≈ 24° RMS (section 4.4), yet every run switches to pointing, never falls back and counts as `success`: the controller sees only its estimate (0.74° error). Nothing onboard flags the problem except the innovation statistics (magnetometer NIS per dof 89, sun 13).

**F5 — Albedo-model mismatch.** If the onboard albedo model is wrong (here: albedo assumed, none present), the pointing maximum exceeds the requirement (5.25°, section 4.5). The design depends on a representative albedo model.

**F6 — Divergence invisible to the rate-based fallback (indicative).** In one time-boxed tuning run with a mistuned filter (gyro-bias random walk × 10, main-session scratch run, not reproduced by a repo script [B-23]) the attitude estimate diverged to 180° with a low estimated rate, so the spacecraft held a wrong attitude without any mode change.

### 6.3 Mitigations (recommended, not implemented)

| Failure | Mitigation |
|---|---|
| F1 | Better sun information in pointing (more CSS heads at canted angles so ≥ 2 diodes see the sun, or a fine sun sensor); in-flight magnetometer calibration and a higher-degree onboard field model with orbit knowledge; consider-states or Schmidt–Kalman treatment of the field-model error; widen the requirement margin. |
| F2 | Initialise from a least-squares sun vector over several samples or from per-diode updates with a coarse attitude prior; command a short slew if no complete CSS vector is seen within a time limit. |
| F4, F6 | Innovation monitor: moving-mean NIS per sensor against χ² bounds, with re-initialisation or safe mode on persistent violation; cross-check with TRIAD whenever a complete sun vector is available. |
| F5 | Fit the albedo coefficient online (one extra state) or de-weight diodes with large predicted albedo. |

## 7. Design conclusions

1. **The architecture works end to end on the estimate.** B-dot detumbling, TRIAD initialisation, MEKF and the rate-limited PD with magnetic dumping run autonomously from a 2–10 °/s tumble to nadir pointing in all 200 Monte Carlo runs, with acquisition set by detumbling in most runs (sections 4.2, 5).
2. **Estimation sets the pointing error.** Perfect knowledge points to 0.054° RMS; ideal sensors with a matched filter to 0.53° (×10); realistic sensor errors with the design filter to 0.99° (×18) (section 4.4). Control-law tuning is not the lever; sensors and the estimator are.
3. **The estimator must be designed for the sensor errors.** Magnetometer-bias states and per-diode CSS updates are what make the realistic case work (24° → 1°); the bias states alone are worth NEES 157 → 14 (section 4.5).
4. **Requirements**: REQ-DET, REQ-ACQ, REQ-KNW, REQ-MOM met in 100 % of design runs, REQ-PNT in 99 % (REQ-ROB met); the 1° RMS goal in 58 %. The pointing margin is small (p95 of the maximum 4.44° vs 5°) and the filter is overconfident, so the result should be read as "a few-degree ADCS with coarse sensors", not as a guaranteed 5°.
5. **B-dot gain**: sensor noise does not change detumbling at 1–4× the nominal gain; the nominal gain is kept (section 4.1).

## 8. What a real mission needs that the simulation omits

- **Sensor calibration and characterisation**: magnetometer scale factor, misalignment and coil/harness residual fields; CSS calibration (scale, temperature, the real angular response instead of a cosine with a hard 60° cut); gyro scale factor and misalignment. Only a constant magnetometer bias and the CSS FOV/albedo are modelled [B-21].
- **Orbit knowledge**: the nadir reference, the onboard field model and the albedo model use the true position; a real system uses GNSS or a TLE (km-level along-track error), which adds field-model and reference errors.
- **Better field and albedo models or better sensors**: the knowledge error floor (≈ 1° RMS) comes from coarse sensors with correlated errors. A fine sun sensor, a star tracker or an in-flight-calibrated magnetometer would be needed for sub-degree pointing; this study shows the coarse-sensor ADCS meets a few-degree requirement, not more.
- **Fault detection, isolation and recovery (FDIR)**: the only fallback is on the estimated rate. It cannot detect a wrong attitude estimate with a low rate (section 6, F4 and F6). An innovation (NIS) monitor, a sanity check against TRIAD when available, and a safe-mode sun-pointing law are needed.
- **Re-initialisation in pointing**: the design filter's TRIAD needs all three CSS axes lit, which never happens in nadir pointing with the sun near the orbit plane (one body axis never sees the sun); a re-initialisation strategy (e.g. a short slew, or initialisation from per-diode least squares over several samples) is required.
- **Flight software effects**: computation latency, fixed-point/word-length effects, sensor timing jitter, actuator command quantisation, magnetorquer/magnetometer interference timing, wheel friction and jitter, wheel speed reversals.
- **Environment**: penumbra, Earth infrared on the CSS, solar radiation pressure, density variability with solar activity (×10 in aero torque [B-10]), flexible modes (deployables), CoM shifts.
- **Verification**: hardware-in-the-loop and air-bearing tests; the Monte Carlo varies only what is modelled here.

## 9. Traceability

| Requirement | Result (design campaign, 100 runs) | Status | Evidence |
|---|---|---|---|
| REQ-DET ≤ 3 orbits | median 1.22, max 2.29 orbits | met (100 %) | §5.2–5.3, `monte_carlo_design.json`, `figures/mc_detumble_time.png` |
| REQ-ACQ ≤ 4 orbits | median 1.21, max 2.23 orbits | met (100 %) | §5.2–5.3, `figures/mc_acquire_time.png`; F2 |
| REQ-PNT ≤ 5° max | median 3.15°, p95 4.44°, max 5.19° | 99 % (run 54 fails) | §5.2, §5.5, §6.1 F1, `mc_worst_runs.json`, `figures/mc_pointing_max.png` |
| REQ-KNW ≤ 2° RMS | median 0.946°, max 1.43° | met (100 %) | §5.3, `figures/mc_knowledge_rms.png` |
| REQ-MOM < 1 | peak wheel fraction max 0.39 | met (100 %) | §4.6, §5.3, `figures/mc_wheel_speed.png` |
| REQ-ROB ≥ 95 % | 99 % meet all requirements | met | §5.2 |
| GOAL-PNT ≤ 1° RMS | median 0.962°, max 1.46° | 58 % | §5.2, `figures/mc_pointing_rms.png` |

Reproduction: `python -m adcs.simulate` (one closed-loop run), `python -m adcs.report.closed_loop_figures` (sections 4.1–4.6, ≈ 12 min), `python -m adcs.monte_carlo` (both campaigns and `figures/mc_*.png`, ≈ 45 min on 20 processes), `python -m adcs.report.mc_worst_runs` (section 6.1, ≈ 3 min). Specification: `docs/specs/adcs_closed_loop.md`.
