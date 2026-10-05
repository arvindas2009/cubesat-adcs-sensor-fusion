# Spec: Project A data layer + dead reckoning (Group 2)

Source of truth for `data-wrangler`, `py-implementer`, `test-author`. Conventions per `CONVENTIONS.md` (SI units, time on axis 0, scalar-first Hamilton quaternions, FRD body, NED local frame).

## 1. `sensor_fusion/download_logs.py`

- `PINNED_LOGS`: dict/tuple of exactly 3 entries, each `{id, role, notes}`; `role` in `{"hover", "aggressive", "other"}`. At least one hover-like and one aggressive-manoeuvre flight. IDs are Flight Review log UUIDs (review.px4.io), picked once by a one-off screening step and then **hard-coded**.
- Screening criteria (documented in the module docstring and `data/NOTES.md`): multirotor, GPS fix 3D for the bulk of flight, duration 2–15 min, contains topics `sensor_combined` (or `vehicle_imu`), `vehicle_magnetometer` (or `sensor_mag`), `vehicle_air_data`, `vehicle_gps_position`, `vehicle_local_position`, `vehicle_attitude`; file size reasonable (< ~40 MB each). Prefer logs with IMU at ≥ 200 Hz.
- `download_log(log_id, dest_dir=DATA_DIR, session=None) -> Path`: downloads `https://review.px4.io/download?log=<id>` to `dest_dir/<id>.ulg`; skips if file exists and is non-empty; timeout, ≤ 3 retries with exponential backoff; verifies the `ULog` magic bytes `b"ULog\x01\x12\x35"`.
- `python -m sensor_fusion.download_logs` downloads all pinned logs. Screening helpers (querying `review.px4.io/browse_data_retrieval` or equivalent) live in a separate function/CLI flag `--screen` and are never needed at run time.
- `DATA_DIR = sensor_fusion/data` (`*.ulg` is gitignored).

## 2. `sensor_fusion/loader.py`

```python
@dataclass
class FlightLog:
    log_id: str
    path: Path
    t0_us: int                 # boot-time µs of the first sample of the common window
    imu: ImuData
    mag: MagData
    baro: BaroData
    gps: GpsData
    ekf2: Ekf2Data             # PX4's onboard EKF2 outputs (reference only, not truth)
    info: dict                 # airframe/params/firmware strings, arming windows, removals log

def load_log(path_or_id, *, crop_to_flight=True) -> FlightLog
```

All time arrays are `t` in **seconds since boot minus `t0`** (so the common window starts at 0), `float64`, strictly increasing. Every container is a dataclass of numpy arrays with time on axis 0:

| Container | Fields (SI, repository frames) |
|---|---|
| `ImuData` | `t (N,)`, `gyro (N,3)` rad/s FRD, `accel (N,3)` m/s² FRD specific force, `dt (N,)` s (actual sample interval; `dt[0]` = median), `rate_hz` (median) |
| `MagData` | `t`, `field (N,3)` **tesla** FRD (PX4 gauss × 1e-4) |
| `BaroData` | `t`, `pressure (N,)` Pa, `alt_baro (N,)` m (height **up**, relative to the log's reference), `temperature (N,)` °C |
| `GpsData` | `t`, `lat (N,)` rad, `lon (N,)` rad, `alt (N,)` m above ellipsoid (use `alt_ellipsoid` if present, else MSL and say so in `info`), `alt_msl (N,)` m above the geoid (same message; gives the geoid separation `alt − alt_msl`), `vel_ned (N,3)` m/s, `eph (N,)`, `epv (N,)` m, `s_variance_m_s`, `fix_type (N,)`, `n_sat (N,)`, `t_sample (N,)` (time of the measurement if the log carries it, else `t`) |
| `Ekf2Data` | `t_att`, `q (N,4)` FRD→NED scalar-first canonical; `t_pos`, `pos_ned (N,3)`, `vel_ned (N,3)`, `ref_lat`, `ref_lon`, `ref_alt` of EKF2's local frame |

Rules
1. Preserve native sample rates; **no resampling** in the loader. A separate helper `interp_to(t_target, t, x)` (linear; `slerp` for quaternions via `common.quaternion.slerp`) is provided for consumers.
2. If `sensor_combined` has multiple IMU instances or `vehicle_imu` is used instead, document the choice. If `vehicle_imu` delta-angle/delta-velocity is used, convert to rate = Δθ/dt and accel = Δv/dt.
3. Clean explicitly: drop non-finite rows, duplicate/non-increasing timestamps, and GPS rows with `fix_type < 3`; record counts per stream in `info["removed"]`. Gaps: report any inter-sample gap > 5× median interval in `info["gaps"]` as `(stream, t_start, t_end)`.
4. `crop_to_flight`: crop all streams to the window in which the vehicle is armed (from `vehicle_status.arming_state` / `actuator_armed`, fall back to `vehicle_land_detected`), with 2 s margin. Record the window in `info`.
5. Magnetometer frame: `vehicle_magnetometer` is already in the vehicle body frame (FRD) after PX4's rotation handling; confirm for each log and note it. GPS lat/lon in the raw topic are integer degrees × 1e-7 in older formats — handle both; confirm which in the log.
6. Quaternions in the output are canonical (`w ≥ 0`) and normalised.

## 3. `sensor_fusion/data/NOTES.md` and inspection plots

`NOTES.md` records, per log: ID, role, firmware, airframe, duration, flight window, sample rates per stream, units/frames found, removals/gaps, anything odd (e.g. mag calibration, clipped accel). It also lists the **PX4 → repository mapping questions** for the ★ review (units, sign of baro altitude, GPS scaling, which topics, frame of each stream).

Inspection plots (via `common/plotting.py` helpers, created by `figures-runner`/`data-wrangler`; PNG 150 dpi, labelled axes with units) saved to `sensor_fusion/report/figures/data_<id>_*.png`: raw gyro/accel with sampling-interval histogram, mag norm vs time, baro vs GPS altitude, GPS ground track + EKF2 NED track, EKF2 attitude (Euler) for orientation sanity.

## 4. `sensor_fusion/dead_reckoning.py`

```python
@dataclass
class DeadReckoningResult:
    t: (N,)  q: (N,4)  v: (N,3)  p: (N,3)      # NED, p relative to p0

def strapdown(t, gyro, accel, q0, v0=None, p0=None, *,
              gyro_bias=None, accel_bias=None, g_ned=(0, 0, G0)) -> DeadReckoningResult
```

- Model per `CONVENTIONS.md` §4: `f_b = R(q)ᵀ(a_n − g_n) + b_a`, so `a_n = R(q)(f_b − b_a) + g_n`; gyro measures `ω_b + b_g`.
- Attitude propagated with `common.quaternion.propagate(q, ω dt)` using the **mean of the two bracketing gyro samples** (trapezoid). Note (★ review): PX4 `sensor_combined` values are means over the interval *ending* at `timestamp`, so the trapezoid carries a ≈ dt/2 lag (2.25 ms at 222 Hz); in 1 s windows vs EKF2 this is within 0.01° of the interval-end scheme and better than interval-start. Documented, accepted. Velocity/position: trapezoid on `a_n` (v) and on `v` (p) with `a_n` rotated by the mid-interval attitude. `t` need not be uniform; use actual `dt`.
- Flat, non-rotating Earth (Project A assumption). No Earth-rate or transport-rate terms.
- `q0` required; `v0`, `p0` default to zeros. Biases default to zero.
- Helper `initial_attitude_from_accel_mag(accel, mag)` (static leveling: roll/pitch from mean specific force over a window, yaw from mag tilt-compensated heading with declination = 0) — a coarse alternative to EKF2's initial quaternion. Default for the drift study is to **initialise from EKF2's attitude at t=0** and say so.

### Tests (independent oracles)
1. **Constant body rate, no gravity effect** (zero accel, g=0): attitude equals `from_rotvec(ω t)` to 1e-10; also equals scipy `Rotation.from_rotvec(ω t)`.
2. **Static, level**: `accel = [0, 0, −g]` in FRD (specific force of a vehicle resting level, down-axis reaction = −g) with q=identity → `v, p` stay zero to 1e-9.
3. **Constant acceleration**: known `a_n`, attitude rotated by a constant arbitrary q; resulting `p(t) = ½ a t²`, `v = a t` to 1e-9 (trapezoid is exact for constant `a`).
4. **Circular motion / coning**: body rotating about a tilted axis at constant rate, accel computed analytically from a known inertial trajectory → position error bounded and decreasing with `dt` (convergence order ≥ 1.8 on halving `dt` for the chosen scheme).
5. **Bias drift analytic**: constant accel bias `b` with level attitude → `p_err = ½ b t²`; constant gyro bias on a static level vehicle → attitude error angle `|b| t` (first-order position error `~ ½ g |b| t³/3` for small angles, check against the leading-order analytic expression with tolerance stated in the test).
6. Non-uniform `t` accepted; mismatched lengths raise `ValueError`.

## 5. Drift study (`figures-runner`)

For each pinned log: strapdown from EKF2 initial state over the flight window, compare with EKF2 NED position/velocity/attitude (`interp_to` onto the IMU grid, or onto the EKF2 grid). Metrics JSON `sensor_fusion/report/dr_metrics_<id>.json`: position error at 10 s, 30 s, 60 s, end; velocity error same; attitude error angle (`common.quaternion.angle_between`) same; the fitted growth exponent of position error. Figures: position error vs time (log-y), attitude error vs time, trajectory overlay, all `figures/dr_<id>_*.png`. Plain statements of what the numbers show (gyro-bias-driven attitude error → gravity-coupling → cubic position growth) belong in `PROGRESS.md`, not invented here.

Caveat to state in every caption: EKF2 is the reference, **not ground truth**.

## 6. Additions after the ★ review (Group 2)

- `common/frames.py::normal_gravity(lat, alt)` → m/s²: WGS-84 normal gravity (Somigliana closed form, `γ = 9.7803253359 (1 + 0.00193185265241 sin²φ)/√(1 − 0.00669437999013 sin²φ)`) plus the free-air correction `−3.086e-6 · h`. Batched; `lat` rad, `alt` m above ellipsoid. Oracle values: equator 9.7803253359, pole 9.8321849378 m/s² (sea level).
- `MagData` and `BaroData` time stamps use `timestamp_sample` where the topic has it.
- `GpsData.alt_msl` added (see §2).
- Baro test criterion: baro height *changes* are compared with EKF2 height only when the log has `EKF2_HGT_REF == 0` (baro); on GPS-referenced logs EKF2 height inherits GPS vertical drift (~9 m in the aggressive log), so the oracle there is: baro returns to take-off level within 5 m by the end of the flight and baro increments over 10 s windows correlate with GPS height increments (r > 0.3) over the whole flight.
- Drift study: per-log `g` from `normal_gravity` at the EKF2 origin; initial state = EKF2 state at t=0 (`p0 = ekf2.pos_ned[0]`); metrics also reported on a window starting 1 s after the (single) EKF2 yaw reset with the initial state re-seeded from EKF2 at that time.
