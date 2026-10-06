# Spec: Project A 15-state error-state EKF (Group 3, M3)

Source of truth for `sensor_fusion/ekf_drone.py` (main session), `sensor_fusion/synthetic.py` + tests (`test-author`), the flight study (`figures-runner`) and the ★ review (`math-reviewer`). Conventions per `CONVENTIONS.md` (§2 quaternions, §3 error state, §4 frames/IMU model, §6 PX4 mapping). The full derivation lives in the module docstring of `sensor_fusion/ekf_drone.py`; this file fixes names, signatures, behaviour and the test list.

## 1. Model

Nominal state `x = (p, v, q, b_g, b_a)`: NED position [m] and velocity [m/s], FRD→NED quaternion, gyro bias [rad/s], accel bias [m/s²].
Error state (fixed order, 15): `δx = [δp(0:3), δv(3:6), δθ(6:9), δb_g(9:12), δb_a(12:15)]`, `q_true = q ⊗ exp(δθ)` (body-frame, right perturbation), all other components additive (`p_true = p + δp`, ...).

IMU model (CONVENTIONS §4): `ω_m = ω + b_g + n_g`, `f_m = Rᵀ(a − g_n) + b_a + n_a`, `ḃ_g = n_bg`, `ḃ_a = n_ba`, `g_n = [0, 0, g]`.

Continuous error dynamics, with `ω = ω_m − b_g`, `f = f_m − b_a`, `R = R(q)`:

```
δṗ  = δv
δv̇  = −R [f×] δθ − R δb_a − R n_a
δθ̇  = −[ω×] δθ − δb_g − n_g
δḃ_g = n_bg
δḃ_a = n_ba
```

`F` (15×15) and `G` (15×12, noise order `w = [n_a, n_g, n_bg, n_ba]`) follow directly. `Qc = diag(σ_a² I3, σ_g² I3, σ_bg² I3, σ_ba² I3)` with densities `σ_a` [m/s²/√Hz], `σ_g` [rad/s/√Hz], `σ_bg` [rad/s²/√Hz], `σ_ba` [m/s³/√Hz].

Measurements (residual `r = z − h(x)`; Jacobians w.r.t. `δx`):

| Name | `z` | `h(x)` | `H` | `R` |
|---|---|---|---|---|
| `gps_pos` | GPS position in repo NED | `p` | `[I 0 0 0 0]` | `diag(σ_h², σ_h², σ_v²)`, `σ_h = max(eph, floor_h)`, `σ_v = max(epv, floor_v)` |
| `gps_vel` | GPS NED velocity | `v` | `[0 I 0 0 0]` | `max(s_acc, floor_vel)² I` |
| `baro` | `z_D,baro` (down position from baro, §3) | `p_D` | `e_3ᵀ` on δp | `σ_baro²` |
| `mag_heading` | `ψ_m = decl + ψ_mag`, tilt-compensated with the **estimated** roll/pitch | `ψ(q)` (ZYX yaw) | `[0, 0, h_m, 0, 0]`, `h_m` below (full residual Jacobian) | `σ_ψ²` |

Mag heading: with roll φ, pitch θ of the current estimate, `m_h = Ry(θ) Rx(φ) m_b`, `ψ_mag = −atan2(m_h,y, m_h,x)`, `z = wrap(decl + ψ_mag)`, residual wrapped to (−π, π]. Yaw `ψ = atan2(R10, R00)`; with `R_true = R (I + [δθ×])` the Jacobian is

`h_ψ = [0, R10 R02 − R00 R12, R00 R11 − R10 R01] / (R00² + R10²)`.

**Changed after the ★ review:** because `z` uses the estimated tilt, a tilt error leaks into the residual (×tan I). The update therefore uses the full residual Jacobian `h_m = mag_heading_jacobian(q, m_b) = −(m_h,x e_yᵀ − m_h,y e_xᵀ) T̂ [m_b×] / (m_h,x² + m_h,y²)`, `T̂ = Ry(θ̂)Rx(φ̂)`, instead of `h_ψ` (still exported as `yaw_jacobian`). With `h_ψ` the synthetic NEES Monte Carlo failed (mean ≈ 14 vs 99 % band 5.9–12.8); with `h_m` it passes (9.86).

The heading update is skipped (counted in `meta["mag_skipped_tilt"]`) when `R00² + R10² < cos²(60°)` (|pitch| > 60°).

## 2. Discretisation and timing

- Nominal propagation over each IMU interval `[t_k, t_{k+1}]` uses the **same scheme as `dead_reckoning.strapdown`**: trapezoid (mean of the two bracketing samples) for `ω` and `f`, `q_{k+1} = q_k ⊗ exp(ω dt)`, `a = R(q_mid) f + g_n` with `q_mid = q_k ⊗ exp(ω dt/2)`, `v += a dt`, `p += ½(v_k + v_{k+1}) dt`. Biases constant.
- Covariance: `F` evaluated with `R(q_mid)`, the trapezoid `ω`, `f`. Default `discretization="first_order"`: `Φ = I + F dt + ½(F dt)²`, `Qd = G Qc Gᵀ dt`. Alternative `"van_loan"` (`common.ekf.van_loan`) for tests.
- Measurements are applied at the IMU sample **nearest** to their (delay-corrected) time stamp, after the propagation step ending at that sample; timing error ≤ dt/2 (2.3 ms at 222 Hz). Within one sample, order is `gps_pos, gps_vel, baro, mag_heading`; each update is followed by injection and reset (ES-EKF contract of `common/ekf.py`). Measurements outside `[t_imu[0], t_imu[-1]]` are dropped.
- GPS delay: GPS samples (stamped `t_gps` by the receiver driver) are valid at `t_gps − gps_delay_s` (A-09).
- Injection: `p += δp`, `v += δv`, `q ← inject_attitude_error(q, δθ)`, biases `+=`. Reset: `P ← G_r P G_rᵀ`, `G_r = I` except the attitude block `attitude_reset_jacobian(δθ)`.

## 3. Increments

`EkfConfig.increment` ∈ {1, 2, 3}:

1. GPS position + velocity + baro. All 9 navigation error states active (tilt is observable through GPS velocity); **no mag** (yaw only through manoeuvres); biases **frozen** at their initial values.
2. Increment 1 + `mag_heading`.
3. Increment 2 + gyro and accel bias estimation.

Frozen states: `P0` variance `FROZEN_VAR = 1e-18` and zero process noise for those states (so their gain is ~0); the bias values stay exactly at their initial values — implementation also zeroes `dx` for frozen states before injection.

## 4. API (`sensor_fusion/ekf_drone.py`)

```python
STATE_NAMES: tuple[str, ...]          # 15 names: "p_n","p_e","p_d","v_n",...,"th_x",...,"bg_x",...,"ba_x",...
IDX_P, IDX_V, IDX_TH, IDX_BG, IDX_BA  # slices
FROZEN_VAR = 1e-18

@dataclass
class EkfConfig:
    increment: int = 3
    # process noise densities (A-03)
    accel_noise: float = 0.035        # m/s^2/sqrt(Hz)
    gyro_noise: float = 1.5e-3        # rad/s/sqrt(Hz)
    gyro_bias_rw: float = 1.0e-4      # rad/s^2/sqrt(Hz)
    accel_bias_rw: float = 3.0e-4     # m/s^3/sqrt(Hz)
    # initial 1-sigma (A-03)
    sigma_p0: float = 1.0; sigma_v0: float = 0.5; sigma_att0: float = 0.1
    sigma_bg0: float = 0.1; sigma_ba0: float = 0.2
    # measurements (A-04)
    gps_delay_s: float = 0.11
    gps_pos_floor_h: float = 0.5; gps_pos_floor_v: float = 0.125; gps_vel_floor: float = 0.3   # floor_v tuned in Group 4 (A-13), was 0.5
    baro_sigma: float = 1.5
    mag_heading_sigma: float = 0.07   # rad (4 deg)
    gate_prob: float | None = None    # chi2 gating probability per update; None = no gating
    init_window_s: float = 1.0        # static initialisation window
    discretization: str = "first_order"   # or "van_loan"
    cov_every: int = 0                # store full P every k IMU samples (0 = never; final P always stored)
    use_mag: bool | None = None       # override of the increment's default (None = from increment)
    estimate_biases: bool | None = None

@dataclass
class SensorStreams:                  # filter input; synthetic.py produces the same structure
    imu_t: (N,); gyro: (N,3); accel: (N,3)
    gps_t: (M,)  # receiver time stamps (NOT delay-corrected)
    gps_pos_ned: (M,3); gps_vel_ned: (M,3); gps_pos_sigma: (M,3) [σ_n, σ_e, σ_d]; gps_vel_sigma: (M,)
    baro_t: (K,); baro_d: (K,)        # down position from baro [m]
    mag_t: (L,); mag_field: (L,3)     # body FRD, tesla
    g: float                          # local gravity magnitude [m/s^2]
    declination: float                # rad, east positive
    origin_lla: tuple[float, float, float]   # rad, rad, m ellipsoidal
    meta: dict

@dataclass
class InitialState:
    p: (3,); v: (3,); q: (4,); b_g: (3,) = 0; b_a: (3,) = 0

@dataclass
class EkfResult:
    t: (N,); p, v: (N,3); q: (N,4); b_g, b_a: (N,3)   # posterior at every IMU sample
    std: (N,15)                       # sqrt(diag(P)) at every IMU sample
    P_hist_t: (J,); P_hist: (J,15,15) # if cov_every > 0
    P_final: (15,15)
    innovations: dict[str, InnovationLog]   # keys gps_pos, gps_vel, baro, mag_heading (present if used)
    config: EkfConfig
    meta: dict                        # init details, counts, runtime_s, mag_skipped_tilt, ...

def error_state_jacobians(q, f_body, omega_body) -> tuple[F (15,15), G (15,12)]
def process_noise_psd(cfg, active_mask) -> Qc (12,12)
def mag_heading_measurement(q, mag_body, declination) -> float     # z (rad, wrapped)
def yaw_jacobian(q) -> (3,)                                         # h_psi
def propagate_nominal(p, v, q, b_g, b_a, gyro0, gyro1, acc0, acc1, dt, g) -> (p, v, q, q_mid, omega, f)
def static_initial_state(streams, window_s=1.0) -> InitialState    # static leveling + mag heading + decl; p from first GPS fix, v = 0

class DroneEKF:
    def __init__(self, cfg, init: InitialState, g: float)
    # .p .v .q .b_g .b_a, .ekf (common.ekf.ErrorStateEKF)
    def predict(self, gyro0, gyro1, acc0, acc1, dt)
    def update_gps_pos(self, z, sigma3, t=None) -> UpdateResult   # σ used as given; floors are applied in run_filter
    def update_gps_vel(self, z, sigma, t=None) -> UpdateResult
    def update_baro(self, z_d, sigma, t=None) -> UpdateResult
    def update_mag_heading(self, mag_body, declination, sigma, t=None) -> UpdateResult | None

def run_filter(streams, cfg=EkfConfig(), init: InitialState | None = None) -> EkfResult
def streams_from_log(log: FlightLog) -> SensorStreams
def ekf2_reference(log: FlightLog, origin_lla) -> dict   # EKF2 in the filter's NED frame
```

`run_filter` with `init=None` uses `static_initial_state(streams, cfg.init_window_s)`. `P0 = diag(σ_p0², σ_v0², σ_att0², σ_bg0², σ_ba0²)` per axis (frozen biases: `FROZEN_VAR`).

## 5. Real-log inputs (`streams_from_log`)

- NED origin = first GPS sample of the window (lat, lon, ellipsoidal alt); `gps_pos_ned = lla_to_ned(...)`.
- `g = normal_gravity(lat0, alt0)` (A-07). `declination = atan2(B_e, B_n)` from `ppigrf.igrf` at the origin on `info["gps_utc_start"]` (A-05/A-11).
- `gps_pos_sigma = [eph, eph, epv]`, `gps_vel_sigma = s_variance_m_s` (floors applied in the filter).
- Baro: `alt_ref` = mean `alt_baro` over `t ≤ baro.t[0] + 1 s`; `baro_d = d0 − (alt_baro − alt_ref)`, `d0` = GPS down position at the first fix (≈ 0) (A-10).
- `ekf2_reference(log, origin_lla)` returns `t_pos, pos_ned, vel_ned, t_att, q, reset_times` with EKF2 translated into the repo frame: horizontal `p_NE − lla_to_ned(origin; EKF2 ref)_NE`, vertical `D = z − (ref_alt − alt_msl(origin))` (CONVENTIONS §6).

## 6. Synthetic truth generator (`sensor_fusion/synthetic.py`, test-author)

```python
@dataclass
class SyntheticNoise:
    accel_noise=0.0; gyro_noise=0.0; gyro_bias_rw=0.0; accel_bias_rw=0.0   # densities, same units as EkfConfig
    gps_pos_sigma=(0.0, 0.0, 0.0); gps_vel_sigma=0.0; baro_sigma=0.0; mag_sigma=0.0  # white; mag in tesla per axis

@dataclass
class SyntheticTruth:
    t: (N,); p, v, a: (N,3); q: (N,4); omega: (N,3) body; b_g, b_a: (N,3)

def make_synthetic_flight(trajectory="figure8", duration=60.0, imu_rate=100.0, gps_rate=5.0,
                          baro_rate=10.0, mag_rate=10.0, *, noise=SyntheticNoise(),
                          gyro_bias=(0,0,0), accel_bias=(0,0,0), gps_delay=0.0,
                          declination=0.1, field_ned=None, g=G0, seed=0
                          ) -> tuple[SensorStreams, SyntheticTruth]
```

- Analytic trajectories (closed-form `p, v, a` and Euler angles + their rates; `ω_b` from the ZYX Euler-rate relation, never from finite differences):
  - `"figure8"`: horizontal figure-eight of ~20 m, period ~20 s, gentle climb/descent, yaw rotating (so all bias axes are observable), roll/pitch oscillations of 5–15°.
  - `"hover"`: fixed position, roll/pitch oscillations ≤ 3°, **constant yaw** (yaw unobservable without mag).
  - `"static"`: at rest with a fixed attitude (for initialisation tests).
- IMU samples are **instantaneous** values on a uniform grid: `gyro = ω_b + b_g(t) + noise`, `accel = R(q)ᵀ(a − g_n) + b_a(t) + noise`; white-noise per-sample σ = density·√(imu_rate); bias random walk increments density·√dt.
- GPS: true `p`, `v` at time `t` plus white noise, **stamped `t + gps_delay`** (so `SensorStreams.gps_t` is the receiver stamp); `gps_pos_sigma` / `gps_vel_sigma` arrays = the noise σ used. Baro: `p_D + noise`. Mag: `R(q)ᵀ m_n + noise`, `m_n` from `field_ned` (default: 50 µT total, inclination 65°, horizontal component rotated east by `declination`).
- `origin_lla = (0, 0, 0)`, `meta = {"synthetic": True, ...}`.

## 7. Tests (`sensor_fusion/tests/test_synthetic.py`, `test_ekf_drone.py`)

Generator self-checks (independent): `ω_b` vs finite-difference of the truth quaternion (`2 q* ⊗ q̇`); `a` vs finite-difference of `v`; noise-free IMU through `dead_reckoning.strapdown` reproduces truth `p, v, q` (second-order: error shrinks ~4× on doubling the rate).

Filter tests:
1. **F Jacobian** — numerical: for random nominal state/inputs, perturb the true state by a small `δx` (via the error definitions in §1), integrate truth and nominal over a tiny dt, compare `(δx(dt) − δx(0))/dt` to `F δx(0)`. Also `G` noise mapping for each noise channel.
2. **Yaw Jacobian** vs finite differences of `ψ(q ⊗ exp(ε e_i))`; `mag_heading_measurement` returns the true yaw for a noise-free field at random tilts ≤ 50°.
3. **Discretisation**: first-order vs `van_loan` covariance after 10 s of propagation agree to ≤ 2 % (diagonal, relative).
4. **Noise-free truth tracking**: `figure8`, noise-free sensors, `init` = truth, increment 3: position < 0.05 m, velocity < 0.05 m/s, attitude < 0.05° for the whole run.
5. **Convergence with noise + biases** (increment 3, `figure8`, constant biases e.g. `b_g = [0.01, −0.008, 0.005] rad/s`, `b_a = [0.1, −0.15, 0.2] m/s²`, init biases 0, filter noise = generator noise): final bias errors < 3σ per axis and gyro bias error < 0.002 rad/s; attitude error < 1° after 30 s.
6. **Consistency (NEES/NIS)**: Monte-Carlo over ≥ 10 seeds — mean NEES of (δp, δv, δθ) within the two-sided 99 % chi-square band for 9·runs dof (time-averaged after 20 s); GPS-position NIS fraction inside the 95 % chi²(3) interval ≥ 0.85.
7. **Increments**: `hover`, increment 1 → yaw σ grows (noise-free IMU; with a noisy IMU the noisy specific force in `F` gives slight spurious yaw observability, so only "stays near the prior" is required), bias estimates identically equal to their initial value; increment 2 → yaw σ bounded (< 3 σ_ψ after 20 s); increment 3 → biases move.
8. **SPD and unit norm**: every stored `P_hist` (cov_every > 0) and `P_final` SPD (`common.ekf.is_spd`); `|q| = 1` to 1e-12 at every sample.
9. **GPS delay**: `gps_delay=0.2` on `figure8` with GPS noise; position RMSE with `cfg.gps_delay_s=0.2` < with `0.0`.
10. **Static init**: `static` with roll 5°, pitch −3°, yaw 40°, declination 0.1 rad, noise-free → `static_initial_state` attitude within 0.01° of truth (yaw includes declination).
11. **Real-log smoke** (skip if `.ulg` missing): `streams_from_log` on the hover log — `gps_pos_ned[0] ≈ 0`, `baro_d` ≈ 0 over the first second (±0.5 m), `g == normal_gravity(...)`, declination equals the `ppigrf` value at the origin/date; `run_filter` on the first 40 s (slice the streams) of the hover log, increment 3: all finite, P SPD, horizontal position within 3 m of the EKF2 reference after 10 s.

Runtime budget: whole new test set ≤ 40 s (use 100 Hz IMU and ≤ 60 s runs).

## 8. Flight study (`sensor_fusion/report/ekf_study.py`, figures-runner)

For each pinned log × increment {1, 2, 3}: `run_filter(streams_from_log(log), EkfConfig(increment=i))`, compare with `ekf2_reference` on the EKF2 grids (excluding ±1 s around each EKF2 quaternion reset for attitude statistics). Metrics JSON `sensor_fusion/report/ekf_metrics_<id>.json`: per increment — RMS/max/final of horizontal position, vertical position, velocity, tilt and yaw differences vs EKF2 (tilt = angle of the attitude difference with yaw removed; yaw = wrapped yaw difference); per-sensor NIS mean, dof, fraction inside 95 % interval, `nis_bounds`; final bias estimates ±1σ; runtime. **Bias reference:** EKF2's own bias estimates are in the ULog topic `estimator_sensor_bias` (read with `pyulog` in the study script, not the loader): use the instance whose `accel_device_id` / `gyro_device_id` equals `estimator_selector_status.accel_device_id` / `gyro_device_id` (the primary EKF2 instance; hover log: 4259850). Report EKF2 `accel_bias`, `gyro_bias` (median over the flight and the values at start/end) next to ours, and plot them as dashed lines in the bias figure. (Main-session check, hover log: the 1.35° static tilt difference between accel leveling and EKF2 equals the tilt implied by EKF2's accel bias [−0.141, −0.192, −0.009] m/s².) Figures `figures/ekf_<id>_{pos,att,bias,nis}.png` (increments overlaid; caption "EKF2 reference, not truth").
