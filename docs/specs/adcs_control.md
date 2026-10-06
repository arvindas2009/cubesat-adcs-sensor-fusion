# Spec: Project B control with perfect knowledge (Group 6, M7)

Source of truth for `py-implementer`, `test-author`, `figures-runner`, `doc-writer`. Conventions per `CONVENTIONS.md` (SI, scalar-first Hamilton body-to-ECI quaternions, body angular velocity relative to ECI in body axes, time on axis 0). Reuse `common.quaternion`, `common.frames` (`GM_EARTH`, `dcm_eci_to_lvlh`), `adcs.config` (`SimConfig`, `DEFAULT`), `adcs.orbit` (`initial_state`, `Ephemeris`, `orbital_period`), `adcs.environment` (`FieldTable`), `adcs.dynamics` (`rk4_step`, `lvlh_error`, `nadir_angle`), `adcs.disturbances` (`total_disturbance_torque`). Do not edit those modules or `CONVENTIONS.md`.

General: array-likes in, float arrays out; no mutation of inputs; no hidden global state; docstrings state units and frames.

## 1. `adcs/actuators.py` (py-implementer)

Hardware (recorded in `assumptions.md` B-05, B-06):

- **Magnetorquers**: three orthogonal coils along body x, y, z; maximum dipole **0.2 A·m² per axis** (ISIS iMTQ magnetorquer board, nominal actuation level 0.2 A·m² at 20 °C, 5 V).
- **Reaction wheels**: three identical wheels with spin axes along body x, y, z. **CubeSpace CubeWheel Gen 2 CW0017**: maximum torque 0.23 mN·m, rated momentum 1.77 mN·m·s at 8000 rpm (maximum speed 10000 rpm at nominal voltage is *not* used; the rated point is the operational limit). Spin inertia `J_s = 1.77e-3 / (8000·2π/60) = 2.1128e-6 kg·m²` (derived, not a datasheet number). No friction, no torque ripple, no motor dynamics.

| Name | Behaviour |
|---|---|
| `RPM` | module constant `2π/60` (rad/s per rpm) |
| `@dataclass(frozen=True) MagnetorquerConfig` | `m_max: float = 0.2` (A·m², per axis); `control_period: float = 1.0` (s); `duty: float = 0.8` (fraction of each control period the coils are driven; the magnetometer is sampled in the remaining off part so the coils do not corrupt it). Property `on_time = duty * control_period` (s). `__post_init__` raises `ValueError` unless `m_max > 0`, `control_period > 0`, `0 < duty <= 1`. |
| `@dataclass(frozen=True) WheelConfig` | `max_torque: float = 0.23e-3` (N·m); `rated_momentum: float = 1.77e-3` (N·m·s); `rated_speed_rpm: float = 8000.0`. Properties: `max_speed = rated_speed_rpm * RPM` (rad/s, the operational speed limit), `spin_inertia = rated_momentum / max_speed` (kg·m²), `h_max = rated_momentum` (N·m·s). Positive values required (`ValueError`). |
| `saturate_dipole(m_cmd, m_max, mode="scale")` | `(…,3)` A·m². `"scale"`: if `max_i |m_i| > m_max` scale the whole vector by `m_max / max_i |m_i|` (direction preserved); `"clip"`: clip each component to `[−m_max, m_max]`. Unknown mode → `ValueError`. |
| `magnetorquer_torque(m_b, b_b)` | `m_b × b_b` (N·m, body axes); `m_b` A·m², `b_b` T, broadcast. |
| `wheel_speed(h_w, w, cfg=WheelConfig())` | wheel speeds relative to the body (rad/s), `(…,3)`: `Ω = h_w / J_s − ω` (wheel i spins about body axis i; `h_w` is the total wheel momentum including the body rate, see `adcs/dynamics.py`). |
| `wheel_momentum(speed, w, cfg=WheelConfig())` | inverse: `h_w = J_s (Ω + ω)` (N·m·s). |
| `limit_wheel_torque(u_cmd, h_w, w, cfg=WheelConfig(), dt=None)` | wheel torque command `u_w` (N·m, applied **to** the wheels, `ḣ_w = u_w`, body axes), returns the torque the wheels actually deliver, `(3,)`. (1) clip each component to `±max_torque`. (2) speed limit with `h_hi = J_s (max_speed + ω)`, `h_lo = J_s (−max_speed + ω)` per axis: if `dt is None`, set `u_i = 0` where `h_i ≥ h_hi_i and u_i > 0` or `h_i ≤ h_lo_i and u_i < 0`; if `dt` is given (the command is held for `dt` seconds), clip `u_i` to `[min(0, (h_lo_i − h_i)/dt), max(0, (h_hi_i − h_i)/dt)]` so a held command cannot carry the wheel past the limit (and never pushes a wheel already beyond a limit further out; it also never forces a reversal). Order: speed limit after the torque clip; the result satisfies both. |

## 2. `adcs/control.py` (main session)

| Name | Behaviour |
|---|---|
| `bdot_gain(inertia, a, inc, mu=GM_EARTH)` | Avanzini & Giulietti (2012) gain `k = 2 n (1 + sin ξ_m) J_min` (N·m·s), `n = sqrt(μ/a³)`, `J_min` = smallest eigenvalue of `inertia`, `ξ_m` approximated by the orbit inclination `inc` (the geomagnetic inclination varies about it by the ≈ 10° dipole tilt). |
| `bdot_dipole(b_b, b_b_prev, dt, k, m_max)` | B-dot on the field **direction**: `m = −(k/|B|) (b̂ − b̂_prev)/dt`, `b̂ = b_b/|b_b|`, `|B| = |b_b|`; saturated with `saturate_dipole(…, "scale")`. With the instantaneous rate `ḃ̂ ≈ −ω × b̂` (inertial field change neglected) the torque is `m × B = −k ω_⊥` (component of ω perpendicular to B), so `ω · τ ≤ 0`. |
| `nadir_reference(r, v)` | `(q_ref, w_ref)`: `q_ref = from_dcm(dcm_eci_to_lvlh(r, v).T)` (body-to-ECI attitude of a body aligned with LVLH, w ≥ 0), `w_ref = [0, −|r × v|/|r|², 0]` = angular velocity of LVLH relative to ECI **in LVLH axes** (rad/s; for the circular orbit `−n` about LVLH y, i.e. `+n` about the orbit normal). |
| `@dataclass(frozen=True) PointingGains` | `wn = 0.1` rad/s, `zeta = 0.7`, `rate_limit = deg2rad(0.5)` rad/s. `Kp = wn² I`, `Kd = 2 zeta wn I` (matrices, I = inertia). |
| `pointing_torque(q, w, h_w, q_ref, w_ref, inertia, gains)` | Returns `(tau, err)`: `err = to_rotvec(conj(q_ref) ⊗ q)` (body axes, w ≥ 0 shortest rotation, equals `dynamics.lvlh_error` for the nadir reference); `w_ref_b = R(δq)ᵀ w_ref` with `δq = conj(q_ref) ⊗ q`; `δω = ω − w_ref_b`; rate-limited PD: `ω_cmd = −sat((wn/(2 zeta)) err, rate_limit)` (vector-norm saturation: scaled to norm `rate_limit` if larger); `tau = −Kd (δω − ω_cmd) + ω × (I ω + h_w) − I (ω × w_ref_b)`. `tau` is the desired torque **on the body** (N·m, body axes). Unsaturated, the closed-loop error dynamics are `I δω̇ = −Kp err − Kd δω + τ_dist` (small angles). |
| `dump_dipole(h_w, b_b, k, m_max, h_target=0)` | cross-product momentum dumping `m = (k/|B|²) ((h_w − h_target) × b_b)`, saturated (`"scale"`). Unsaturated torque `m × b_b = −k (h_w − h_target)_⊥` (perpendicular to B). `k` in 1/s. |
| `dump_active(prev, speed_frac, on_frac, off_frac)` | hysteresis switch: returns `True` if `speed_frac ≥ on_frac`, `False` if `speed_frac ≤ off_frac`, else `prev`. `speed_frac = max_i |Ω_i| / max_speed`. |

Wheel command (in the simulator): `u_w = −(tau − tau_mag)`, where `tau_mag` is the magnetorquer torque the controller expects over the coming control period (feed-forward, perfect knowledge), then `limit_wheel_torque(u_w, h_w, ω, wheel, dt=control_period)`.

## 3. `adcs/simulate.py` (main session) — perfect-knowledge runner (Group 8 adds estimation)

- `@dataclass(frozen=True) ControlConfig`: `bdot_k: float | None = None` (None → `bdot_gain`), `gains: PointingGains`, `dumping: bool = True`, `dump_k`, `dump_on`, `dump_off` (fractions of max wheel speed), `detumble_threshold = deg2rad(0.5)`, `detumble_hold = 60.0` s, `disturbances: bool = True`.
- `run(y0, t_end, mode="detumble"|"pointing"|"auto", cfg=DEFAULT, ctrl=ControlConfig(), mtq=MagnetorquerConfig(), wheel=WheelConfig(), log_every=1)` → `SimResult`.
- Loop per control period `T = mtq.control_period` starting at `t_k = k T`: sample the true field `b_b = R(q)ᵀ B_eci(t_k)` (perfect magnetometer; coils were off), compute commands, then integrate with `dynamics.rk4_step`: one step of `mtq.on_time` with the dipole on, then (if `duty < 1`) one step of `T − on_time` with the coils off. Held during the period: dipole `m`, wheel torque `u_w`, disturbance torque (evaluated at `t_k` with the state at `t_k`) and `B_eci(t_k)`; the magnetorquer torque `m × R(q)ᵀ B_eci` is re-evaluated with the current `q` at every RK4 stage. Orbit from `Ephemeris` (J2), field from `FieldTable(dt=10)`.
- Modes: `detumble` (B-dot; wheels held at zero torque), `pointing` (PD on wheels + dumping), `auto` (detumble, switching once to pointing after `|ω| < detumble_threshold` has held for `detumble_hold` s).
- `SimResult` fields (logged every `log_every` periods): `t, q, w, h_w, wheel_speed, m (applied dipole), u_w (delivered), tau_dist, tau_mag (at t_k), nadir_err (rad), rate (|ω|, rad/s), mode (0 detumble / 1 pointing), dump (bool)`, `meta` dict (configs, switch time, period).
- `tumble_state(rate, seed)`: uniformly random attitude and rate axis (`numpy.random.default_rng(seed)`), `|ω| = rate`, `h_w = 0`.
- `lvlh_state(r, v, offset=zeros(3), w_rel=zeros(3))`: `q = q_ref ⊗ exp(offset)`, `ω = R(exp(offset))ᵀ w_ref + w_rel`, `h_w = 0`.
- `detumble_time(t, rate, threshold)`: earliest `t_i` such that `rate[j] < threshold` for all `j ≥ i`; `nan` if the last sample is not below.
- `python -m adcs.simulate --scenario detumble|pointing|auto [--orbits N]` prints a summary.

## 4. Tests (`test-author`, from this spec; independent oracles; thresholds fixed here and never loosened — if one fails, report it)

Files: `adcs/tests/test_actuators.py`, `adcs/tests/test_control.py`, `adcs/tests/test_simulate_pk.py`. Thresholds were set after a prototype run by the main session; the measured value is given in brackets.

**Actuators**
1. `WheelConfig()`: `max_speed = 8000·2π/60 = 837.758 rad/s`, `spin_inertia = 2.1128e-6 kg·m²` (hand values, 1e-4 rel), `h_max = 1.77e-3`. Invalid configs raise. `MagnetorquerConfig().on_time = 0.8`; `duty = 0` or `> 1` raises.
2. `saturate_dipole`: within limits unchanged; `"scale"` keeps the direction (cross product with input = 0) and `max |m_i| = m_max`; `"clip"` per component; batched input `(N,3)` handled row by row; bad mode raises.
3. `magnetorquer_torque` = hand cross product; ⟂ to `m` and `B`.
4. `wheel_speed` / `wheel_momentum` round trip; `Ω = h/J_s − ω` by hand.
5. `limit_wheel_torque`: random property test (1000 cases, random `u`, `h` within ±1.2 `h_max`, small `ω`, `dt ∈ {None, 0.1, 1}`): `|u_i| ≤ max_torque`; with `dt`: if the wheel starts inside the limits, `h + u dt` stays within `[h_lo, h_hi]` (1e-15 abs); a wheel beyond a limit never gets torque further out; torque away from a limit is not reduced by the speed limit; never reverses the sign of `u_i`.

**Control**
6. `bdot_gain(DEFAULT inertia, a, 51.6°)` equals `2·sqrt(μ/a³)·(1 + sin 51.6°)·0.0066667` (1e-12 rel, with the exact Izz).
7. `bdot_dipole` dissipation (exact for finite dt): body rotating at constant `ω` in a constant inertial field: `b_now = rotate_inverse(from_rotvec(ω dt), b_prev)`; for 1000 random `ω` (|ω|dt < 3), `b`, `k`, `m_max`: `ω · (m × b_now) ≤ 1e-25`. Small-dt limit: `m → (k/|B|)(ω × b̂)` (rel 1e-3 at |ω|dt = 1e-4, unsaturated). Saturation respected.
8. B-dot energy (own loop, independent of `simulate.py`): constant inertial field 30 µT, DEFAULT inertia, `ω0` = 10 deg/s random axis, `dynamics.rk4_step` with `ext_torque = m × R(q)ᵀB` re-evaluated per stage, `m` held over `dt = 0.1` s from the last two samples: kinetic energy non-increasing at every step (tolerance 1e-12·T0) over 2000 steps; also `H_i · B̂` constant (torque ⟂ B; rel 1e-9).
9. `nadir_reference`: `q_ref` gives `dynamics.lvlh_error = 0` (1e-12); `w_ref` vs central finite difference of `q_ref` along `orbit.propagate` (`ω = 2 vec(q_ref* ⊗ q̇_ref)`, step 0.5 s) at 10 points: two-body (`j2=False`) within 1e-9 rad/s; J2 orbit within 5e-6 rad/s (the out-of-plane LVLH rotation due to J2 is not in `w_ref`, ≈ 1e-6 rad/s).
10. `pointing_torque`: (a) `q = q_ref`, `ω = w_ref_b`, `h = 0` → `tau = ω × Iω` (1e-15 abs; pure gyroscopic feed-forward since `ω × w_ref_b = 0`), `err = 0`; (b) `q_ref = identity`, `w_ref = 0`, `ω = 0`, `h = 0`, `q = exp(δθ)` with |δθ| = 1e-3 rad → `tau = −wn² I δθ` (rel 1e-6); (c) 90° error about a random axis, `ω = 0`, `w_ref = 0` → `tau = −2 zeta wn I · rate_limit · ê` (rate saturation; 1e-12 rel); (d) `err` equals `dynamics.lvlh_error` for the nadir reference at random attitudes; (e) `h_w` enters only through `ω × h_w`.
11. `dump_dipole`: unsaturated `m × B = −k (h − (h·B̂)B̂)` (1e-12 rel); `m ⟂ B`; `h ∥ B` → `m = 0`; saturation via scale. `dump_active` hysteresis truth table.

**Simulation (perfect knowledge)** — each `run` costs ~1 ms per control period plus ~4 s setup; keep runs ≤ 2 orbits.
12. Detumble, `tumble_state(deg2rad(10), seed=0)`, `run(…, 2 orbits, "detumble")`: `detumble_time(…, 0.5 deg/s) < 2 orbits` [1.14]; final kinetic energy < 0.01 of the initial [≈ 4e-4]; wheel torque and `h_w` stay 0; `|m_i| ≤ 0.2`. (Exit criterion M7: < 0.5 deg/s within 1–3 orbits.)
13. Pointing without disturbances or dumping, `lvlh_state(r0, v0, offset=deg2rad([10, 0, 0]))`, 1 orbit: nadir error < 0.01° for `t > 1000 s` [0.0013°]; `H_i` drift (max component) < 1e-9 of `|H_i|` [2e-12] (only internal wheel torques act).
14. Momentum dumping, `lvlh_state(r0, v0)` with `y0[8] = +0.9 h_max` (y-wheel preloaded in the direction the secular aero torque drives it), 1 orbit with disturbances: **dumping off** → largest wheel-speed fraction reaches ≥ 0.99 [saturates at 2573 s] and nadir error exceeds 10° afterwards [172°]; **dumping on** → speed fraction never exceeds 0.9 + 1e-3 and ends ≤ 0.3 [0.153]; nadir error < 1° for `t > 2000 s` [0.063°]. In both: `|Ω_i| ≤ max_speed·(1 + 1e-5)` at every sample [overshoot 1.3e-6, from ω changing within a held period].
15. Auto mode, `tumble_state(deg2rad(0.3), seed=5)`, 600 s: `mode` switches exactly once, from 0 to 1, with `meta["switch_time"] = 60 s` (rate below threshold from t = 0, plus `detumble_hold`).
16. Plumbing: `SimResult` array shapes with `log_every = 10`; `detumble_time` hand cases (never below → nan, always below → t[0], dips below then above → later time); `lvlh_state` gives zero nadir error for zero offset; `tumble_state` unit-norm `q`, `|ω| = rate`, reproducible per seed; bad `mode` raises.

## 5. Figures and metrics (`figures-runner`; edit only `adcs/report/` scripts)

`adcs/report/control_figures.py` (`python -m adcs.report.control_figures`; `multiprocessing.Pool`, set `OPENBLAS_NUM_THREADS=1` before importing numpy) → `adcs/report/control_metrics.json` and PNGs (150 dpi, labelled axes with units) in `adcs/report/figures/`:
- `detumble.png`: |ω| (deg/s, log y) vs time (orbits) for `tumble_state(10 deg/s, seed)` seeds 0–4, 3 orbits, 0.5 deg/s threshold line; second panel |m| (A·m²) for seed 0. JSON: detumble time per seed (s and orbits), final rate.
- `detumble_sensitivity.png`: detumble time (orbits) vs initial rate {2, 5, 10, 20, 30} deg/s for `m_max` ∈ {0.05, 0.1, 0.2, 0.4} A·m² (seeds 0–2: mean with min–max bars; runs up to 6 orbits; not-detumbled runs reported as such, never dropped); second panel: detumble time vs gain scale {0.25, 0.5, 1, 2, 4} × `bdot_gain` at 10 deg/s, `m_max` 0.2, seeds 0–4. Shade the 1–3 orbit band. JSON: full grid.
- `pointing_15orbits.png`: pointing scenario of the CLI (`lvlh_state` offset deg2rad([20, −15, 30]), w_rel deg2rad([0.1, −0.1, 0.2]), `h_w = 0`), 15 orbits, dumping on vs off: nadir error (deg, log y) vs orbits; wheel speeds (rpm) per axis with ±8000 rpm limit lines for each case; shade dumping-active intervals. JSON: per case max/RMS nadir error after the first orbit, per-orbit max error, max wheel-speed fraction per axis, time of first saturation (fraction ≥ 0.99) or null, fraction of time dumping.
- `auto_detumble_to_nadir.png`: `run(tumble_state(10 deg/s, 0), 4 orbits, "auto")`: rate, nadir error, wheel speeds and |m| vs orbits with the switch time marked. JSON: switch time, max/RMS nadir error from switch + 1 orbit to the end, max wheel-speed fraction.
- JSON also records `bdot_k`, gains, actuator configs, control period/duty, and wall time.
Use `log_every=10` for long runs. Report anomalies instead of changing algorithm code.

## 6. Assumptions (`doc-writer`, `assumptions.md`)

Fill B-05 (iMTQ, 0.2 A·m² nominal at 20 °C / 5 V, per axis; https://www.isispace.nl/?p=2384 and the iMTQ brochure; applied as a hard limit with duty cycle 0.8 per 1 s period), B-06 (CW0017: 0.23 mN·m max torque, 1.77 mN·m·s rated momentum at 8000 rpm, 10000 rpm max speed at nominal voltage not used, 60 g; https://cubesatshop.com/product/gen-2-cubewheel-cw0017 and satsearch; spin inertia 2.1128e-6 kg·m² derived; three wheels along body axes, no friction/jitter). Add B-16 (control laws and gains with design rationale: Avanzini–Giulietti B-dot gain, PD `wn` 0.1 rad/s, ζ 0.7, slew-rate limit 0.5 deg/s so that `I_x·rate_limit` = 3.7e-4 N·m·s ≈ 21 % of `h_max`, gyroscopic + reference-rate feed-forward, dumping gain 1e-3 1/s, on/off 0.3/0.05 of max speed, magnetorquer feed-forward to the wheels, detumble threshold 0.5 deg/s held 60 s), B-17 (simulation discretisation: 1 s control period, ZOH of dipole/wheel torque/disturbance torque/`B_eci`, RK4 0.8 s + 0.2 s, magnetometer sampled with coils off, perfect knowledge), B-18 (detumble benchmark: published 1–3 orbits for 3U-class B-dot; Avanzini & Giulietti, "Magnetic detumbling of a rigid spacecraft", JGCD 35(4):1326–1334, 2012; a LAPAN 3U study reporting a best transient of about two orbits from 10 deg/s, https://karya.brin.go.id/11429/1/Jurnal%20TD_Robertus_Pus%20KKPA_2017.pdf — mark details unverified). Measured numbers only from `adcs/report/control_metrics.json`.

## 7. Out of scope

Estimation and sensor noise (Group 7), the closed loop on estimated attitude and Monte Carlo (Group 8), wheel friction/jitter, coil dynamics and magnetometer–coil coupling beyond duty cycling, wheel pyramid/redundancy.
