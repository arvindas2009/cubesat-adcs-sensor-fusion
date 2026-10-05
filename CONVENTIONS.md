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

Project A treats NED as inertial (flat, non-rotating Earth). Neglected Earth rate (7.3e-5 rad/s) is below consumer MEMS gyro bias instability; transport rate is negligible over a few-km flight. Gravity in NED is `g_n = [0, 0, +g]` with `g = 9.80665 m/s²` unless a local value is stated in `assumptions.md`.

IMU model (both projects): the accelerometer measures **specific force** in body axes, `f_b = R(q)ᵀ (a_n − g_n) + b_a + noise`; the gyro measures `ω_b + b_g + noise`.

## 5. Time

- Project A: PX4 timestamps (µs since boot) converted to float seconds since boot.
- Project B: a UTC `datetime` epoch in `adcs/config.py`; simulation time is float seconds since epoch. UT1 ≈ UTC (|ΔUT1| < 0.9 s neglected).

## 6. PX4 → repository mapping

Placeholder, finalised in Group 2 after the loader is reviewed (★). Known so far: body FRD, local NED, `vehicle_attitude.q` already scalar-first Hamilton FRD→NED (matches §2 with reference = NED). Magnetometer units, barometric altitude sign and GPS lat/lon scaling to be confirmed per log format and recorded here.

## 7. Code conventions

- Packages: `common/`, `sensor_fusion/` (Project A), `adcs/` (Project B). Tests in `<package>/tests/`.
- Functions take and return numpy arrays; no hidden global state. Physical constants live in `common/frames.py` (Earth) or `adcs/config.py` (spacecraft).
- Every module has pytest tests before the next module starts; `pytest` is green at every group exit.
- Figures: matplotlib, PNG 150 dpi, axis labels with units, saved under `<package>/report/figures/`.
