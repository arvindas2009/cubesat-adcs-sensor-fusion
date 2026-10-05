# Conventions

Fixed in Group 1 (M0). Every module, test, plot and report follows this file. Changes require an explicit decision recorded in `PROGRESS.md`.

> It is **attitude** determination and control (orientation). "Altitude" is used only for genuine height (barometric altitude, orbit altitude).

## 1. Units

- SI everywhere internally: m, s, kg, rad, rad/s, N·m, A·m², T (tesla), Pa.
- Angles are **radians** internally. Degrees appear only in plot axes, printed tables and reports.
- Orbit positions/velocities are in **m** and **m/s** (not km). Libraries returning km (e.g. `sgp4`) are converted at the boundary.
- Time series are numpy arrays with **time on axis 0**: a vector series has shape `(N, 3)`, a quaternion series `(N, 4)`, a covariance series `(N, n, n)`. Time stamps are float seconds.

## 2. Quaternions

- Storage: **scalar-first** `q = [w, x, y, z]`, numpy float arrays, shape `(4,)` or batched `(..., 4)`.
- Algebra: **Hamilton** product (`i² = j² = k² = ijk = −1`), right-handed.
- Meaning: a quaternion attached to a body represents the **body-to-reference** rotation (reference = inertial ECI for the spacecraft, local NED for the drone):

  `v_ref = R(q) v_body = q ⊗ [0, v_body] ⊗ q*`

  ```
  R(q) = [[1-2(y²+z²),  2(xy-wz),    2(xz+wy)  ],
          [2(xy+wz),    1-2(x²+z²),  2(yz-wx)  ],
          [2(xz-wy),    2(yz+wx),    1-2(x²+y²)]]
  ```
  This is the same convention as `scipy.spatial.transform.Rotation.from_quat([x, y, z, w]).as_matrix()` and as PX4's `vehicle_attitude.q` (FRD body → NED).
- Composition: `q_AC = q_AB ⊗ q_BC`.
- Unit norm is enforced after every propagation/update step (`normalize`). Sign ambiguity: where a unique representative is needed (`from_dcm`, `to_rotvec`, comparisons), use **w ≥ 0**.
- Kinematics: `q̇ = ½ q ⊗ [0, ω_b]`, where `ω_b` is the body angular velocity relative to the reference frame, **expressed in body axes**.
- Discrete propagation over `dt` with constant rate: `q_{k+1} = q_k ⊗ exp(ω_b dt)`, where `exp(φ) = [cos(|φ|/2), sin(|φ|/2) φ/|φ|]` (`from_rotvec`).
- Euler angles (only for display and PX4 comparisons): aerospace **ZYX** (yaw ψ, pitch θ, roll φ), `R = Rz(ψ) Ry(θ) Rx(φ)`.

## 3. Error-state filtering

- The covariance never contains a 4-element quaternion. Attitude error is a **3-element rotation vector δθ in body axes** (local / right perturbation):

  `q_true = q_est ⊗ exp(δθ)`,  `exp(δθ) ≈ [1, δθ/2]` for small δθ.
- Error state ordering is fixed per filter and documented in that filter's module docstring. Project A (15 states): `[δp(3), δv(3), δθ(3), δb_g(3), δb_a(3)]`. Project B MEKF (6 states): `[δθ(3), δb_g(3)]`.
- Measurement residual: `r = z − h(x̂_nominal)`. After an update the error is **injected** into the nominal state (`p += δp`, `q ← q ⊗ exp(δθ)`, ...) and reset to zero; the covariance reset Jacobian is applied where it matters (`P ← G P Gᵀ`, `G ≈ I − ½[δθ×]` for the attitude block).
- Covariance update uses the **Joseph form** and is symmetrised after every step.
- Filter consistency is judged by **NIS** `rᵀ S⁻¹ r` against chi-square bounds.

## 4. Frames

| Frame | Definition | Used by |
|---|---|---|
| **ECI** | Earth-centred inertial, z = Earth rotation axis, x = vernal equinox. Approximated as the true-of-date frame obtained from ECEF by a GMST rotation only (precession, nutation, polar motion neglected; < 0.5° in 2026, irrelevant for magnetic-field and sun-direction modelling). `sgp4` TEME output is treated as this frame. | Project B |
| **ECEF** | WGS-84 Earth-centred Earth-fixed. `r_ecef = Rz(GMST) r_eci`. | IGRF lookups, GPS |
| **LLA** | WGS-84 geodetic latitude, longitude (rad), height above ellipsoid (m). | GPS, IGRF |
| **NED** | Local tangent plane at a fixed reference LLA (first valid GPS fix of the flight). North-East-Down. | Project A navigation frame |
| **Body (drone)** | PX4 **FRD**: x forward, y right, z down. Origin at the IMU. | Project A |
| **Body (spacecraft)** | Origin at CoM; x, y along the two 10 cm faces, z along the 34 cm long axis. | Project B |
| **LVLH** | z = −r̂ (nadir), y = −(r × v)/‖r × v‖ (negative orbit normal), x = y × z (≈ velocity direction for a circular orbit). | Nadir pointing target |

Project A treats NED as inertial (flat, non-rotating Earth). Neglected Earth rate (7.3e-5 rad/s) is below consumer MEMS gyro bias instability; transport rate is negligible over a few-km flight. Gravity in NED is `g_n = [0, 0, +g]`; `g = 9.80665 m/s²` (`G0`) for synthetic tests, the local WGS-84 normal gravity for real logs (see §6).

IMU model (both projects): the accelerometer measures **specific force** in body axes, `f_b = R(q)ᵀ (a_n − g_n) + b_a + noise`; the gyro measures `ω_b + b_g + noise`.

## 5. Time

- Project A: PX4 timestamps (µs since boot) converted to float seconds since boot.
- Project B: a UTC `datetime` epoch in `adcs/config.py`; simulation time is float seconds since epoch. UT1 ≈ UTC (|ΔUT1| < 0.9 s neglected).

## 6. PX4 → repository mapping (Project A)

Finalised in Group 2 after the ★ review. Verified on the three pinned logs (PX4 v1.14.0 and v1.17.0; evidence in `sensor_fusion/data/NOTES.md`). The loader (`sensor_fusion/loader.py`) is the only place where PX4 units/fields are converted.

**Time.** PX4 `timestamp` is µs since boot (uint64). Repository time `t = (t_us − t0_us)·1e-6` s, `t0_us` = first sample of the loader's common (armed) window. A measurement is valid at its `timestamp_sample` where the topic has one (mag, baro, GPS). PX4 does not stamp GPS receiver latency (`timestamp_sample == timestamp`): GPS position/velocity are valid at `t − τ_gps`, `τ_gps = EKF2_GPS_DELAY` (0.11 s in all pinned logs; independent estimate 0.10–0.18 s). UTC (`vehicle_gps_position.time_utc_usec`) is used to date the IGRF model.

**Frames.** Body = PX4 FRD, navigation = NED, for every topic below. No FLU/ENU conversion and no quaternion conjugation anywhere. Gyro and accel values are means over the integration interval ending at `timestamp`.

| Quantity | PX4 field | PX4 unit / frame | Repository |
|---|---|---|---|
| angular rate | `sensor_combined.gyro_rad[3]` | rad/s, FRD | `imu.gyro` rad/s FRD |
| specific force | `sensor_combined.accelerometer_m_s2[3]` | m/s², FRD, specific force (≈ −g on z at rest, level) | `imu.accel` = `f_b` of §4 |
| magnetic field | `vehicle_magnetometer.magnetometer_ga[3]` | gauss, FRD, calibrated but **not** corrected by EKF2's in-flight bias | `mag.field` = gauss × 1e-4 → **T** |
| baro | `vehicle_air_data.baro_alt_meter`, `baro_pressure_pa` | m **up**, ISA altitude for `SENS_BARO_QNH` (1013.25 hPa); Pa | `baro.alt_baro` m up (datum below); `baro.pressure` Pa |
| GPS position | v1.15+: `latitude_deg`, `longitude_deg` [deg], `altitude_ellipsoid_m` [m]; ≤ v1.14: `lat`, `lon` [int, 1e-7 deg], `alt_ellipsoid` [int, mm] | WGS-84 | `gps.lat/lon` **rad**, `gps.alt` **m above WGS-84 ellipsoid** |
| GPS MSL height | `altitude_msl_m` / `alt` [mm] | m above geoid | `gps.alt_msl`; used for the geoid separation `N = alt − alt_msl` |
| GPS velocity | `vel_n_m_s`, `vel_e_m_s`, `vel_d_m_s` | m/s NED (down positive) | `gps.vel_ned` |
| EKF2 attitude | `vehicle_attitude.q[4]` | `[w,x,y,z]` Hamilton, FRD → NED | `ekf2.q`, normalised, w ≥ 0 (matches §2 with reference = NED) |
| EKF2 position/velocity | `vehicle_local_position.x,y,z,vx,vy,vz`; `ref_lat`, `ref_lon` [deg], `ref_alt` [m **MSL**] | NED about EKF2's own origin, spherical azimuthal-equidistant projection (R = 6371 km) | `ekf2.pos_ned/vel_ned` unchanged; `ref_lat/lon` rad, `ref_alt` m MSL |

**Heights — three datums, never mix.** (1) Repository LLA/NED: WGS-84 ellipsoidal height. (2) EKF2 `ref_alt − z`: MSL. (3) `baro_alt_meter`: ISA pressure altitude, offset from MSL by +8…+28 m and drifting by metres within a flight. Convert MSL ↔ ellipsoid only with the per-sample `N` from the same GPS message. Use baro only as a height change: `z_D,baro = −(alt_baro − alt_baro_ref)`, `alt_baro_ref` = mean over the first 1 s of the window (vehicle on the ground), tied to `p_D` at that time. `vehicle_air_data.ambient_temperature` is a 15 °C placeholder on v1.17 — do not use it.

**Local frame.** Repository NED origin = first valid GPS fix in the loader window (lat, lon, ellipsoidal height); lat/lon → NED via `common.frames.lla_to_ned` (ellipsoidal tangent plane; differs from PX4's spherical projection by ≤ 0.3 % of distance). EKF2's origin is set at the first fix after boot, before logging starts, so EKF2 position at t = 0 is not zero (up to 3.9 m horizontal, 11.7 m vertical in the pinned logs). Compare with EKF2 by translating it by the constant origin offset (horizontal: our origin projected in EKF2's frame; vertical: `ref_alt − alt_msl(origin)`), never by subtracting first samples.

**Gravity.** `g_n = [0, 0, g]` with `g` = WGS-84 normal gravity at the NED origin (`common.frames.normal_gravity`: Somigliana + free-air −3.086e-6 h); the pinned logs give 9.7785, 9.7859, 9.8119 m/s² (vs `G0` = 9.80665, which would add up to 0.028 m/s² vertical error). The value used is recorded per log in `assumptions.md`.

**EKF2 is a reference, not ground truth.** It uses its own bias, mag-bias and Earth-field states (not logged in the default profile). Its height follows GPS when `EKF2_HGT_REF = 1` (both v1.17 logs; GPS height drifts ~9 m in the aggressive log) and baro when `= 0` (v1.14 log). Its attitude contains in-flight yaw resets (`vehicle_attitude.quat_reset_counter`, `delta_q_reset`; 1–5° in the pinned logs); the reset times are in `info["ekf2_quat_reset_times_s"]` and must be marked and handled in every comparison. Every figure caption comparing against EKF2 says "EKF2 reference, not truth".

## 7. Code conventions

- Packages: `common/`, `sensor_fusion/` (Project A), `adcs/` (Project B). Tests in `<package>/tests/`.
- Functions take and return numpy arrays; no hidden global state. Physical constants live in `common/frames.py` (Earth) or `adcs/config.py` (spacecraft).
- Every module has pytest tests before the next module starts; `pytest` is green at every group exit.
- Figures: matplotlib, PNG 150 dpi, axis labels with units, saved under `<package>/report/figures/`.
