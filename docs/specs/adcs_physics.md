# Spec: Project B physics — orbit, environment, dynamics, disturbances (Group 5, M5 + M6)

Source of truth for `py-implementer`, `test-author`, `figures-runner`, `doc-writer`. Conventions per `CONVENTIONS.md` (SI, scalar-first Hamilton body-to-ECI quaternions, time on axis 0, ECI = GMST-only true-of-date frame). Reuse `common.frames` (`R_EARTH`, `GM_EARTH`, `J2_EARTH`, `OMEGA_EARTH`, `julian_date`, `gmst`, `dcm_eci_to_ecef`, `eci_to_ecef`, `ecef_to_eci`, `ecef_to_lla`, `dcm_ecef_to_ned`, `dcm_eci_to_lvlh`) and `common.quaternion`. `adcs/config.py` already exists (`SimConfig`, `DEFAULT`, `inertia_box`) — do not edit it.

General: array-likes in, float arrays out; batched over leading dims where stated; no mutation of inputs; no hidden global state; docstrings state units and frames.

## 1. `adcs/orbit.py`

| Function | Behaviour |
|---|---|
| `kepler_to_rv(a, ecc, inc, raan, argp, nu, mu=GM_EARTH)` | classical elements (m, –, rad…) → ECI `(r (3,), v (3,))` m, m/s (perifocal → ECI via 3-1-3 rotation R3(−Ω)R1(−i)R3(−ω)); handles `ecc` 0 |
| `rv_to_kepler(r, v, mu=GM_EARTH)` | inverse; returns dict `a, ecc, inc, raan, argp, nu` (used by tests as the round-trip oracle) |
| `initial_state(cfg: SimConfig = DEFAULT)` | `kepler_to_rv` from `cfg.orbit` |
| `orbital_period(a, mu=GM_EARTH)` | `2π sqrt(a³/μ)` |
| `accel_j2(r, mu=GM_EARTH, j2=J2_EARTH, re=R_EARTH)` | batched `(…,3)`; two-body + J2 in ECI (z = Earth axis): `a = −μ r/|r|³ [1 + 1.5 J2 (Re/|r|)² (1 − 5 z²/|r|²)]` for x,y, and `[… (3 − 5 z²/|r|²)]` for z (standard form, Vallado Eq. 8-30). `j2=0` gives pure two-body. |
| `specific_energy(r, v, j2=True)` | `v²/2 + Φ(r)` with `Φ = −μ/r · [1 − J2 (Re/r)² (3 z²/r² − 1)/2]` (`j2=False` drops the J2 term); conserved by the J2 flow (oracle for the integrator) |
| `propagate(r0, v0, t, j2=True, rtol=1e-12, atol=1e-6)` | `t` = increasing 1-D array of times (s since start, `t[0]` may be 0); `scipy.integrate.solve_ivp` method `DOP853`; returns `r (N,3), v (N,3)` |
| `class Ephemeris` | `Ephemeris(r0, v0, t_end, j2=True)` integrates once with `dense_output=True`; `.at(t)` → `(r, v)` for scalar or array `t` (shape `(…,3)`); attribute `t_end`; raises `ValueError` outside `[0, t_end]` |
| `sgp4_ephemeris(line1, line2, jd0, t)` | SGP4 (`sgp4.api.Satrec.twoline2rv`, WGS72) at times `jd0 + t/86400` (t in s array) → `r (N,3), v (N,3)` in **m, m/s**, TEME treated as ECI. Raises `RuntimeError` if SGP4 returns an error code. |

## 2. `adcs/environment.py`

| Function | Behaviour |
|---|---|
| `sun_vector_eci(jd)` | Vallado Algorithm 29 (low-precision Astronomical Almanac, ≈ 0.01°): with `T = (jd − 2451545.0)/36525`, `λ_M = 280.460° + 36000.771 T`, `M = 357.5291092° + 35999.05034 T`, `λ = λ_M + 1.914666471° sin M + 0.019994643° sin 2M`, `r = 1.000140612 − 0.016708617 cos M − 0.000139589 cos 2M` AU, `ε = 23.439291° − 0.0130042 T`; `r_sun = r·AU·[cos λ, cos ε sin λ, sin ε sin λ]`, `AU = 1.495978707e11 m`. Treat UTC as UT1/TDB (error ≪ 0.01°). Batched over `jd` → `(…,3)` m. |
| `sun_direction_eci(jd)` | unit vector |
| `sunlit(r_eci, s_hat, re=R_EARTH)` | cylindrical shadow: returns bool array, `False` iff `r·ŝ < 0` and `|r − (r·ŝ) ŝ| < re` |
| `eclipse_fraction(r_eci_series, s_hat_series)` | mean of `~sunlit` |
| `beta_angle(r_eci, v_eci, s_hat)` | angle (rad, signed) between ŝ and the orbit plane: `asin(ŝ · ĥ)`, `ĥ = r×v/|r×v|` |
| `atmospheric_density(h)` | exponential, `ρ = ρ0 exp(−(h − h0)/H)`, `h0 = 400e3 m`, `ρ0 = 3.725e-12 kg/m³`, `H = 58.515e3 m` (Vallado Table 8-4, 400–450 km row; **unverified citation**, valid roughly 300–500 km, medium solar activity; real density varies ×10 with solar activity — recorded in `assumptions.md` B-10). `h` = geodetic height above the ellipsoid (m) → kg/m³; for `h` < 0 raise `ValueError`. |
| `geodetic_height(r_eci, jd)` | height above WGS-84 (m) via `eci_to_ecef` + `ecef_to_lla` |
| `magnetic_field_eci(r_eci, jd, date)` | IGRF via `ppigrf` → tesla in ECI. `r_eci (N,3)` m, `jd (N,)` or scalar, `date` = `datetime` at which the IGRF model is evaluated (constant; secular variation over days ≪ 1 %, recorded B-11). Steps: `eci_to_ecef(r, jd)` → `ecef_to_lla` → `ppigrf.igrf(lon_deg, lat_deg, h_km, date)` returns `(Be, Bn, Bu)` in **nT**, local geodetic ENU, shape `(1, N)` for array input (use `[0]`) → T → NED-rows → ECEF with `dcm_ecef_to_ned(lat, lon).T` (NED = (n, e, −u)) → ECI with `ecef_to_eci`'s rotation (rotate the vector, no position-offset). Scalar and `(N,3)` inputs both supported. |
| `class FieldTable` | `FieldTable(ephemeris, cfg: SimConfig, t_end, dt=10.0)`: precomputes `B_eci` (tesla) with `magnetic_field_eci` on a grid `t = 0, dt, … ≥ t_end` along `ephemeris`, builds a `scipy.interpolate.CubicSpline` per component; `.b_eci(t)` → `(…,3)` T (scalar/array `t`, seconds since `cfg.epoch`). Purpose: closed-loop sims must not call ppigrf per step (≈ 14 ms/call). Also `.r_eci(t)`, `.v_eci(t)`, `.sun_dir_eci(t)`, `.sunlit(t)` for convenience (delegating to the ephemeris and `sun_direction_eci(cfg.jd_epoch + t/86400)`). |

## 3. `adcs/dynamics.py`

State vector `y = [q (4), ω (3), h_w (3)]`, length 10: `q` body→ECI scalar-first, `ω` = body angular velocity relative to ECI in **body axes** (rad/s), `h_w` = total reaction-wheel angular momentum in **body axes** (N·m·s). `I` = 3×3 inertia (kg m²) about the CoM.

Equations (the contract):

- `q̇ = ½ q ⊗ [0, ω]` (`common.quaternion.derivative`)
- `I ω̇ = −ω × (I ω + h_w) − ḣ_w + τ_ext`
- `ḣ_w = u_w`: the commanded torque applied **to the wheels** (N·m, body axes); the equal and opposite reaction acts on the body (that is the `−ḣ_w` term). `h_w` is an input state; wheel speed limits/saturation belong to Group 6.

Total inertial angular momentum `H_i = R(q) (I ω + h_w)`; with `u_w = 0`, `τ_ext = 0` it is constant. Kinetic energy `T = ½ ωᵀ I ω` is also constant when `h_w` is constant (derivation: `ω·(ω × h_w) = 0`).

| Function | Behaviour |
|---|---|
| `derivative(t, y, inertia, wheel_torque, ext_torque)` | `wheel_torque`/`ext_torque` are callables `f(t, q, ω, h_w) -> (3,)` **or** constant 3-vectors (convert internally; default zeros). Returns `ẏ` (10,). |
| `rk4_step(t, y, dt, inertia, wheel_torque, ext_torque)` | one classic RK4 step; renormalises `q` afterwards; returns `y_next` |
| `integrate(y0, t, inertia, wheel_torque=None, ext_torque=None, rtol=1e-10, atol=1e-12, normalize=True)` | `solve_ivp` DOP853 at the requested increasing times `t`; renormalises `q` at output; returns `y (N,10)` |
| `angular_momentum_inertial(y, inertia)` | `R(q)(Iω + h_w)` → `(…,3)`; batched over `(N,10)` |
| `kinetic_energy(y, inertia)` | `½ ωᵀ I ω` → `(…)` |
| `lvlh_error(q, r_eci, v_eci)` | attitude error rotation vector `to_rotvec(conj(q_lvlh) ⊗ q)` (body-frame error, CONVENTIONS §3) w.r.t. the LVLH nadir target `q_lvlh = from_dcm(dcm_eci_to_lvlh(r, v).T)`; and `nadir_angle(q, r_eci, v_eci)` = its norm (rad). |

## 4. `adcs/disturbances.py`

All torques are in **body axes**, N·m, about the CoM. `q` body→ECI, positions in ECI.

| Function | Behaviour |
|---|---|
| `gravity_gradient_torque(r_eci, q, inertia, mu=GM_EARTH)` | `τ = 3μ/|r|³ · r̂_b × (I r̂_b)` with `r̂_b = rotate_inverse(q, r/|r|)` (unit vector from Earth centre to spacecraft, in body axes). Spherical Earth. |
| `residual_dipole_torque(m_b, b_b)` | `m_b × b_b` (A m², T, body axes) |
| `residual_dipole_torque_eci(m_b, q, b_eci)` | convenience: rotates `b_eci` into the body with `q` first |
| `aero_torque(r_eci, v_eci, q, cfg=DEFAULT.spacecraft, rho=None)` | flat-plate model of the six faces of the box: face normals `±x, ±y, ±z` with areas `A_x = b·c, A_y = a·c, A_z = a·b` (`size = (a,b,c)`); atmosphere co-rotates with Earth: `v_rel = v − Ω_E ẑ × r` (ECI, `OMEGA_EARTH`); `v̂_b = rotate_inverse(q, v_rel/|v_rel|)`; per face `cosα = n̂·v̂_b`, only faces with `cosα > 0` (facing the flow; convex body, no shadowing, no lift) contribute: `F_i = −½ ρ |v_rel|² Cd A_i cosα · v̂_b`, `τ = Σ (r_i − r_com) × F_i` with `r_i` = face centre (`n̂ · size/2` per axis) minus `cfg.com_offset_b` (the CoM is displaced from the geometric centre; `r_i` is the centre of pressure vector relative to the CoM). `rho` default: `atmospheric_density(|r| − R_EARTH)` (spherical height; this ignores Earth oblateness — up to up to 21 km (poles; ≈ 13 km at 51.6°) height error, i.e. up to ≈ 44 % (≈ 25 %) in ρ, state it in the docstring; there is no `jd` argument so `geodetic_height` is not used here) |
| `total_disturbance_torque(r_eci, v_eci, q, b_eci, cfg=DEFAULT.spacecraft, mu=GM_EARTH)` | sum of the three; also `disturbance_breakdown(...)` returning a dict `{"gravity_gradient", "residual_dipole", "aero"}` of `(3,)` torques |

Solar radiation pressure is neglected (≈ 1.5e-9 N·m for the 3U at a 1 cm lever arm; assumptions `B-12`).

## 5. Tests (written from this spec by `test-author`, independent oracles; thresholds below are fixed in advance and must not be loosened — if one fails, report it)

`adcs/tests/test_orbit_env.py`, `adcs/tests/test_dynamics_dist.py`, `adcs/tests/test_config.py` (small).

**Config:** `inertia_box` vs hand values (Ixx = Iyy = 4/12·(0.1² + 0.34²) = 0.041867, Izz = 4/12·(0.1² + 0.1²) = 0.006667); frozen dataclasses; `jd_epoch == 2461120.0` (2026-03-20 12:00 UTC).

**Orbit**
1. `kepler_to_rv` ↔ `rv_to_kepler` round trip on random elements (incl. `ecc = 0.01…0.5`); circular orbit: `|r| = a`, `|v| = sqrt(μ/a)`, `r·v = 0`.
2. Two-body (`j2=False`) circular orbit over 10 periods: `|r|` constant to 1e-3 m, specific energy and `h` conserved to relative 1e-10; returns to the start after one period to < 1 m (2-body `a` exact period). Elliptical orbit (e = 0.3): matches the Kepler-equation solution (solve Kepler's equation independently with `scipy.optimize.brentq`) at several times to < 1 m.
3. J2: `specific_energy(…, j2=True)` and the z-component of angular momentum conserved to relative 1e-9 over 10 orbits; **nodal regression** over 10 orbits from `rv_to_kepler` (osculating Ω averaged — fit a line to Ω(t)) matches `dΩ/dt = −1.5 J2 n (Re/p)² cos i` to within 3 %. Finite-difference check of `accel_j2` against `−∇Φ` (the potential in `specific_energy`) to 1e-8 relative.
4. SGP4 comparison (ISS-like): use the TLE `1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991` / `2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472 20569` (the python-sgp4 documentation example; confirm `twoline2rv` error 0 and its checksums). Initialise `propagate(j2=True)` from the SGP4 state at epoch and propagate 1 day: |Δr| < 5 km for the first orbit and < 30 km over the day; Δ radius (|r_ours| − |r_sgp4|) bounded 5 km; the difference is expected to be dominated by drag (B* = 3.88e-5) and J3/J4 — the test documents that in a comment. `sgp4_ephemeris` units: `|r|` ≈ 6.78e6 m, `|v|` ≈ 7.66e3 m/s.
5. `Ephemeris.at` agrees with `propagate` at the same times to 1e-6 m; out-of-range raises.

**Environment**
6. Sun: ecliptic longitude consistent with the equinoxes/solstices (UTC): 2026-03-20 14:46 (equinox) declination `|δ| < 0.05°` (±0.25° tolerance on the timing effect is automatic since δ changes 0.4°/day… use the instant stated, tolerance 0.1°); 2026-06-21 08:24: δ = 23.44° ± 0.1°; 2026-09-23 00:05: |δ| < 0.1°; 2026-12-21 20:50: δ = −23.44° ± 0.1°. Distance: perihelion near 2026-01-03 → 0.9833 AU ± 0.001; aphelion near 2026-07-06 → 1.0167 AU ± 0.001. `sun_direction_eci` unit norm. Right ascension at the March equinox ≈ 0 ± 0.1°.
7. Eclipse: cylindrical-shadow fraction of a 400 km circular orbit with ŝ in the orbit plane (β = 0) equals the analytic `asin(Re/(Re + h))/π` (= 0.390) within 0.002 when sampled at ≥ 2000 points per orbit; with β ≠ 0: analytic `acos( sqrt(h² + 2 Re h) / ((Re + h) cos β) ) / π` for `|β| < β* = asin(Re/(Re+h))` within 0.002, and 0 for `|β| > β*`. `beta_angle` of a constructed orbit matches the constructed β (1e-12). For the **default config** (`DEFAULT`, 1 day of J2 propagation, `sun_direction_eci`) the eclipse fraction is between 0.35 and 0.40 (the plan's gate).
8. Magnetic field: `magnetic_field_eci` magnitude along the default 400 km / 51.6° orbit over 3 orbits is within 15–55 µT everywhere (revised after the review: IGRF at 400 km within ±51.6° has an 18.98 µT SAA minimum and a 54.13 µT maximum; the plan's 25–45 µT gate does not hold; report min/max) and mean within 25–40; compare with an independent evaluation: for 20 random (lat, lon) with `h = 400 km`, rotate the ppigrf ENU components into ECEF manually with explicit east/north/up unit vectors (not `dcm_ecef_to_ned`) and into ECI with an explicit `Rz(−GMST)`, tolerance 1e-12 relative. Dipole sanity: |B| at 400 km magnetic-equator-ish vs pole ratio ≈ 0.5 (loose, 0.35–0.65 of the maximum over the orbit vs the minimum), and the field direction mostly aligned with the axis `B·ẑ_eci` sign: field points **northward** (`B_z_ecef > 0`) near the geographic equator (Earth's field lines, IGRF: the geomagnetic north pole is a south magnetic pole, field at the equator points north). `FieldTable.b_eci(t)` equals direct `magnetic_field_eci` at off-grid times to < 100 nT (1 % of the field) over 3 orbits with `dt = 10`; shape/units tests.
9. `atmospheric_density(400e3) == 3.725e-12`; one scale height up → `ρ/e`; monotone decreasing; `h < 0` raises. `geodetic_height` of `[R_EARTH + 400e3, 0, 0]` at `jd` with GMST irrelevant = 400 km ± 1 m (equatorial point).
10. `sunlit` basic cases (point behind the Earth at 7000 km: dark; at 7000 km abeam: lit; sun side: lit).

**Dynamics**
11. Torque-free asymmetric tumble (I = box inertia; `ω0 = [0.05, 0.12, 0.03]` rad/s) over 10 orbit periods (5.57e4 s) with `integrate(..., rtol=1e-11)` (the default 1e-10 gives 1.8e-9 H_i drift from DOP853 phase error): `H_i` constant to 1e-9 relative, `T` constant to 1e-9 relative, raw (`normalize=False`) `|q| − 1` < 1e-9 (1.5e-10 measured).
12. Symmetric top analytic: `I = diag(Ix, Ix, Iz)`, `ω0 = [A, 0, ωz]` with `A = 0.02`, `ωz = 0.1`: body rates `ωx = A cos λt`, `ωy = A sin λt`, `λ = (Iz − Ix) ωz / Ix` to 1e-8 over 200 s; also attitude: `H_i` direction fixed and the angle between `H_i` and the body z-axis constant (nutation angle) to 1e-9; inertial precession rate of the symmetry axis `= |H|/Ix` to 1e-6 relative (fit the angle of the body-z projection about `H` direction).
13. Momentum bias: `Ix = Iy`, constant `h_w = [0, 0, hz]`, `ω0 = [A, 0, 0]`, `Iz` arbitrary: `ωx = A cos(λ t)`, `ωy = A sin(λ t)` with `λ = hz/Ix` (gyroscopic nutation; derive from the equations of §3 and put the derivation in the test comment), tol 1e-8 relative. With the full state (also `ω_z = 0` stays 0).
14. Momentum exchange: start at rest, spin a wheel about body z by a constant wheel torque `u_w = [0, 0, 1e-4]` for 50 s: `h_w = u_w t`, body rate `ω_z = −h_w/Iz` exactly (principal axis, no coupling), `H_i` constant. Wheel torque about a non-principal combination conserves `H_i` as well.
15. `rk4_step` converges at order 4 (halve `dt`: error ratio 14–18) against `integrate` as the reference; `derivative` vs finite differences of `integrate`. `lvlh_error`: zero for the LVLH-aligned quaternion at several orbit points; a known 5° rotation about body y gives `nadir_angle = 5°` to 1e-10; the error vector is **body-frame** (rotating the target by a rotation about body z gives a z-aligned error).
16. External torque callable receives `(t, q, ω, h_w)`; constant 3-vector torque on a principal-axis rest body gives `ω = τ t/I` exactly.

**Disturbances**
17. Gravity gradient: zero when a principal axis is along `r` (all three); for the 3U with `r̂_b` in the x–z plane at angle θ from body z: `τ_y = ±(3μ/r³)(Iz − Ix) sinθ cosθ` (sign `(Ix − Iz)` if θ is the angle of r̂_b from body z toward +x; `(Iz − Ix)` if θ is the body rotation about +y), sign checked against the analytic formula (derive in the test, independent of `np.cross`-based code, e.g. explicit components); peak magnitude `1.5 μ/r³ |Iz − Ix|` (6.76e-8 N·m at 400 km); torque is perpendicular to `r̂_b`; the total `τ·(I ω)`-independent stability check: equilibrium at nadir-aligned long axis is *stable* (restoring torque sign: a small positive rotation about body y gives a negative τ_y and vice versa, for the long axis along z).
18. Residual dipole: `m × B`, perpendicular to both, zero for parallel `m ∥ B`; unit test with hand cross product; eci variant equals body variant after rotation.
19. Aero: with `ρ` given, flow along `+x_b` (body velocity direction `v̂_b = +x̂`): only the `+x` face contributes, force `= −½ρv²Cd A_x x̂` and torque `= r_face × F` = hand value; zero lever (com_offset equal to 0 and flow along an axis) gives zero torque; torque ∝ ρ and ∝ v²; opposite faces never both contribute; the drag force always opposes `v_rel` (check via recomputing `F` from the torque sum is not needed — expose an internal `_aero_force_torque` returning `(F, τ)` and test `F·v̂ ≤ 0`). Co-rotation: `v_rel = v − Ω ẑ × r` verified by a hand case at the equator. Magnitude at 400 km, ρ = 3.725e-12, `v ≈ 7.67 km/s`, `Cd = 2.2`, flow along x with the default CoM offset → within 10 % of the hand computation (document it).
20. **Magnitude gate (reported, not asserted as dominance):** for the default config at 6 attitudes (nadir-aligned, and +10° / +20° offsets about each axis), orbit-RMS and max of each disturbance torque are in `1e-10 … 1e-6 N·m` (typical 3U at 400 km, order-of-magnitude check, source: Wertz SMAD ch. 17 / typical CubeSat values, unverified). `disturbance_breakdown` sums to `total_disturbance_torque`.

## 6. Figures and metrics (`figures-runner`, after implementation + tests pass)

`adcs/report/physics_figures.py` (edit only report scripts; never `adcs/*.py` algorithm modules). Outputs under `adcs/report/figures/` (PNG, 150 dpi, axis labels with units) and `adcs/report/physics_metrics.json`:
- `orbit_vs_sgp4.png`: |Δr|, along/cross/radial components (RTN of the SGP4 orbit) vs time over 1 day; JSON max |Δr| over 1 orbit and 1 day.
- `eclipse_and_field.png`: sunlit flag + |B| (µT) over 3 orbits of the default config; JSON eclipse fraction, B min/mean/max, β angle.
- `disturbance_magnitudes.png`: gravity-gradient / residual-dipole / aero torque magnitudes (log scale) over 3 orbits for the nadir-aligned and a +10° offset attitude; JSON orbit-RMS and max per source and attitude; text in the JSON states which source is largest — nothing else (no "GG dominates" unless the numbers show it).
- `conservation_10orbits.png`: relative drift of `|H_i|`, `T`, `|q| − 1` over 10 orbits for the free tumble of test 11 (JSON final drifts) and the J2 energy/`h_z` drift (test 3).
Every plotted number comes from the code; the script prints a one-line summary per figure.

## 7. Assumptions to record (`doc-writer`, `assumptions.md`, ID `B-nn`)

Fill B-03 (box inertia values and formula), B-08 (residual dipole `[0.001, 0.001, 0.001]` A·m², "typical well-controlled CubeSat 0.001–0.01 A·m², design choice, citation unverified"), B-09 (Cd = 2.2, Vallado §8.6 typical; CoM offset (5, 5, 10) mm within the CDS ±2 cm lateral / ±7 cm longitudinal limits for a 3U), B-10 (exponential density: values above, solar-activity caveat), and add B-11 (IGRF evaluated at the fixed epoch date; ppigrf; field at 400 km ≈ 19–54 µT (SAA minimum 18.98 µT, maximum 54.13 µT, IGRF)), B-12 (disturbance model: spherical-Earth gravity gradient, flat-plate aero with co-rotating atmosphere, SRP and gravity-gradient J2 neglected, no eddy currents / no residual-dipole variation), B-13 (sun model: Vallado Alg. 29 accuracy ≈ 0.01°, cylindrical shadow ignores penumbra, ECI = GMST-only per G-07), B-14 (epoch 2026-03-20 12:00 UTC, RAAN 0 → β ≈ 0; why: long eclipses are the worst case for the sun-sensor/MEKF study), B-15 (orbit propagation: two-body + J2 only, DOP853 rtol 1e-10; compared with SGP4 whose drag/J3/J4 are not modelled). Mark citations that were not checked as "unverified". Fill measured numbers only from `adcs/report/physics_metrics.json`.

## 8. Out of scope for Group 5

Actuators, sensors, control, estimation, simulate loop (Groups 6–9); penumbra; drag lift; SRP; gravity-gradient with J2; wheel speed limits.
