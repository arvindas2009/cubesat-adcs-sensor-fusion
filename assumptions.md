# Assumptions

Every physical assumption used anywhere in this repository is recorded here with its value, unit, justification and citation. Reports and docstrings cite entries by ID (e.g. `[A-03]`, `[B-07]`). A value in code that has no entry here is a bug. Entries marked `TBD(...)` are placeholders to be filled by the named group; citations marked "unverified" must be checked before the final report.

Table format: `| ID | Quantity | Value | Unit | Justification | Source |`

## General / Earth

| ID | Quantity | Value | Unit | Justification | Source |
|---|---|---|---|---|---|
| G-01 | WGS-84 semi-major axis a | 6378137.0 | m | Defining WGS-84 ellipsoid constant; used for LLA/ECEF/NED. | NIMA TR8350.2, WGS-84 |
| G-02 | WGS-84 flattening f | 1/298.257223563 | – | Defining WGS-84 ellipsoid constant. | NIMA TR8350.2, WGS-84 |
| G-03 | Earth gravitational parameter GM | 3.986004418e14 | m³/s² | Two-body and J2 orbit propagation. | WGS-84 (NIMA TR8350.2) |
| G-04 | Earth J2 zonal harmonic | 1.08262668e-3 | – | Dominant oblateness perturbation; only J2 is modelled. | EGM96 / WGS-84 |
| G-05 | Earth rotation rate | 7.2921150e-5 | rad/s | ECI/ECEF rotation via GMST. | WGS-84 |
| G-06 | Standard gravity g | 9.80665 | m/s² | Constant gravity for NED (Project A) unless a local value is stated. | 3rd CGPM (1901) |
| G-07 | ECI realisation | GMST rotation only | – | Precession, nutation and polar motion neglected; error < 0.5° in 2026, irrelevant for magnetic-field and sun-direction modelling. `sgp4` TEME output treated as this frame. | CONVENTIONS.md §4 |
| G-08 | UT1 − UTC | ≈ 0 (\|ΔUT1\| < 0.9 s) | s | UT1 approximated by UTC; effect on GMST is below 0.004°. | IERS convention (UTC leap-second tolerance) |
| G-09 | GMST formula | IAU-1982 | – | Standard closed-form GMST from Julian date. | Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Eq. 3-47 (equation number unverified) |

## Project A — PX4 sensor fusion

| ID | Quantity | Value | Unit | Justification | Source |
|---|---|---|---|---|---|
| A-01 | Navigation frame | Flat, non-rotating NED treated as inertial | – | Earth rate 7.3e-5 rad/s is below consumer MEMS gyro bias instability; transport rate negligible over a few-km flight. | CONVENTIONS.md §4; G-05 |
| A-02 | Gravity in NED | Constant, `[0, 0, +g]`; g = 9.80665 for synthetic tests, per-log local value for real logs (A-07) | m/s² | Flight area spans a few km; gravity variation within a flight is negligible. | G-06; A-07 |
| A-03 | IMU noise densities and bias random walks | TBD(Group 3) | TBD | TBD(Group 3) | TBD(Group 3) |
| A-04 | GPS, barometer, magnetometer noise (R) | TBD(Group 3) | TBD | TBD(Group 3) | TBD(Group 3) |
| A-05 | Magnetic declination / field model | IGRF dated from log UTC; see A-11. Final per-log declination handling TBD(Group 3) | deg | See A-11. | See A-11 |
| A-06 | PX4 log sensor units and sign conventions | Fixed in CONVENTIONS.md §6 (gauss → T, FRD body, NED, baro up, GPS ellipsoidal height) | – | Verified on the three pinned logs; evidence in `sensor_fusion/data/NOTES.md` §4. | PX4 uORB message definitions; Group 2 ★ review |
| A-07 | Local gravity per log (hover / aggressive / other) | 9.7785 / 9.7859 / 9.8119 (G0 = 9.80665 not used) | m/s² | WGS-84 normal gravity at each log's NED origin (latitude and ellipsoidal height), computed by `common.frames.normal_gravity` (Somigliana closed form plus linear free-air term −3.086e-6 m/s² per m). Differs from G0 by up to 0.028 m/s². Plausibility check: measured `R(q)f` down component 9.786 / 9.782 / 9.814 m/s² (includes accel bias and vibration, so not a measurement of g). Deflection of the vertical and local anomalies neglected. | NIMA TR8350.2 (WGS-84, normal gravity formula); Hofmann-Wellenhof & Moritz, *Physical Geodesy*, 2nd ed., 2006 (free-air gradient; section number unverified); Group 2 ★ review (values) |
| A-08 | EKF2 output is a reference, not truth | Used as comparison reference only; attitude resets excluded from or reported separately in comparisons | – | EKF2 runs its own bias, mag-bias and Earth-field states (not logged in the default profile). Each log has one pure-yaw quaternion reset (−4.92° at 16.06 s, −0.98° at 4.21 s, −3.53° at 7.38 s for hover/aggressive/other) and no position/velocity resets. EKF2 height follows GPS when `EKF2_HGT_REF` = 1 (both v1.17 logs) and baro when 0 (v1.14 log); GPS height drifts by −9.2 / −11.6 / −8.9 m relative to take-off at 40 / 100 / 169 s on the aggressive log. Drift/filter comparisons start after the yaw reset or report tilt and yaw separately; figure captions say "EKF2 reference, not truth". | PX4 EKF2 documentation (docs.px4.io, EKF2 tuning page; page version unverified); Group 2 ★ review (numbers) |
| A-09 | GPS measurement delay τ_gps | 0.11 (sweep {0, 0.11, 0.2} for sensitivity) | s | PX4 does not stamp receiver latency (`timestamp_sample == timestamp`). Value is `EKF2_GPS_DELAY` = 110 ms, set in all pinned logs. Independent estimates: best lag 0.10 / 0.14 / 0.18 s (GPS velocity vs IMU-integrated velocity) and 0.10 / 0.11 / 0.15 s (vs EKF2 velocity) for hover/aggressive/other. GPS data are valid at `t − τ_gps`. The sweep bounds the effect of the 0.10–0.18 s spread. | PX4 parameter reference `EKF2_GPS_DELAY` (default value not checked, unverified); Group 2 ★ review (lag estimates) |
| A-10 | Barometer treatment | Height change vs mean of first 1 s on the ground, `z_D,baro = −(alt_baro − alt_baro_ref)`; σ ≈ 1.5 m; ambient temperature not used | m | `baro_alt_meter` is ISA altitude for `SENS_BARO_QNH` = 1013.25 hPa, not equal to MSL or take-off height; baro − EKF2 height has std 1.0 / 3.2 / 1.3 m (hover/aggressive/other), scale versus GPS height change 0.91 (hover) and 1.11 (other), so the offset is neither constant nor exactly unit-scale. `ambient_temperature` is a 15 °C placeholder on v1.17. σ ≈ 1.5 m is a design choice bracketing the 1.0–3.2 m std; to be confirmed by NIS in Group 3. | ISA (ISO 2533:1975, definition of pressure altitude); Group 2 ★ review (numbers) |
| A-11 | Magnetometer treatment | Tilt-compensated heading only; σ_ψ ≈ 3–5; field model IGRF dated from `time_utc_usec` | deg | Measured heading disagreement with IGRF is 6.6 / 1.2 / 3.0° and median field-vector angle 6.0 / 2.7 / 1.6° (hover/aggressive/other). Hover log has likely uncorrected hard-iron (mean body-frame bias ≈ [4.2, 1.4, 1.7] µT, unobservable with an 8.4° yaw range); the other two logs have bias < 1 µT. Logged rate is about 2 Hz. Heading only keeps tilt out of the mag update; a field-vector R from the \|B\| scatter would not represent the heading error. IGRF date must come from the log's UTC (78808b7c: 2024-12-19), not the wall clock. Exact IGRF realisation and σ_ψ per log: TBD(Group 3). | IGRF-13 (Alken et al., 2021, *Earth Planets Space* 73:49; citation details unverified) via `ppigrf`; Group 2 ★ review (numbers) |
| A-12 | Local-frame projection | Ellipsoidal tangent-plane NED (`common.frames.lla_to_ned`) vs PX4's spherical azimuthal-equidistant projection (R = 6371 km) | m | Repository NED uses WGS-84 (G-01, G-02). PX4's `vehicle_local_position` uses a spherical azimuthal-equidistant projection about its own origin. The two differ by ≤ 0.32 m on the 99 m RTK track; comparisons with EKF2 position take this as a bound on the projection error and translate EKF2 by the constant origin offset (CONVENTIONS.md §6). | Snyder, *Map Projections – A Working Manual*, USGS Prof. Paper 1395, 1987 (azimuthal equidistant; page unverified); PX4 `lib/geo` map projection (source file unverified); Group 2 ★ review (0.32 m) |

## Project B — CubeSat ADCS

| ID | Quantity | Value | Unit | Justification | Source |
|---|---|---|---|---|---|
| B-01 | Form factor (3U) | 0.10 × 0.10 × 0.34 | m | Standard 3U CubeSat envelope. | CubeSat Design Specification Rev. 14, Cal Poly |
| B-02 | Mass | 4 | kg | Chosen design value; below the 3U limit of 6 kg in Rev. 14. | CubeSat Design Specification Rev. 14, Cal Poly (limit); design choice (value) |
| B-03 | Inertia tensor | Uniform solid box about CoM, TBD(Group 5) | kg·m² | Simple estimate from B-01 and B-02; ±10% randomised in Monte Carlo. | TBD(Group 5) |
| B-04 | Orbit | 400 km circular, 51.6° inclination | km, deg | ISS-like orbit, representative LEO CubeSat deployment. | Design choice (ISS-like) |
| B-05 | Magnetorquer maximum dipole | 0.2 per axis | A·m² | Typical CubeSat torquer-rod class. | TBD(Group 6: datasheet citation) |
| B-06 | Reaction wheel parameters (max torque, momentum, inertia, speed) | TBD(Group 6) | TBD | TBD(Group 6) | TBD(Group 6: datasheet) |
| B-07 | Sensor noise (magnetometer, sun sensor, gyro ARW and bias random walk) | TBD(Group 7) | TBD | TBD(Group 7) | TBD(Group 7: datasheets) |
| B-08 | Residual magnetic dipole | TBD(Group 5) | A·m² | TBD(Group 5) | TBD(Group 5) |
| B-09 | Drag coefficient | TBD(Group 5) | – | TBD(Group 5) | TBD(Group 5) |
| B-10 | Atmospheric density model (exponential) | TBD(Group 5) | kg/m³ | TBD(Group 5) | TBD(Group 5) |
