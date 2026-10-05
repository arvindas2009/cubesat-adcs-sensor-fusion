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
| A-02 | Gravity in NED | Constant, `[0, 0, +g]`, g = 9.80665 | m/s² | Flight area spans a few km; gravity variation is negligible. | G-06 |
| A-03 | IMU noise densities and bias random walks | TBD(Group 3) | TBD | TBD(Group 3) | TBD(Group 3) |
| A-04 | GPS, barometer, magnetometer noise (R) | TBD(Group 3) | TBD | TBD(Group 3) | TBD(Group 3) |
| A-05 | Magnetic declination / field model | TBD(Group 2/3) | TBD | TBD(Group 2/3) | TBD(Group 2/3) |
| A-06 | PX4 log sensor units and sign conventions | TBD(Group 2) | TBD | TBD(Group 2) | PX4 uORB message definitions |

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
