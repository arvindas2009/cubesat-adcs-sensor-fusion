# Spec: `common/` public API (Group 1)

Source of truth for names, signatures and behaviour. Implementer and test author both work from this file. All conventions per `CONVENTIONS.md`.

General rules
- Inputs are array-likes, converted with `np.asarray(..., dtype=float)`. Outputs are numpy float arrays.
- "Batched" means leading dimensions `(...)` broadcast with numpy rules. Shapes below are given for a single item.
- No function mutates its inputs.

---

## `common/quaternion.py`

All functions batched over leading dims unless stated.

| Function | Returns | Behaviour |
|---|---|---|
| `identity()` | `(4,)` | `[1, 0, 0, 0]` |
| `skew(v)` | `(3,3)` | `[v×]` such that `skew(a) @ b == cross(a, b)` |
| `multiply(p, q)` | `(4,)` | Hamilton product `p ⊗ q` |
| `conjugate(q)` | `(4,)` | `[w, −x, −y, −z]` |
| `inverse(q)` | `(4,)` | `conjugate(q) / |q|²` |
| `norm(q)` | `()` | Euclidean norm |
| `normalize(q)` | `(4,)` | `q / |q|`; raises `ValueError` if any norm < 1e-12 |
| `canonicalize(q)` | `(4,)` | returns `q` or `−q` so that `w ≥ 0` (if `w == 0`, first non-zero of x,y,z positive) |
| `rotate(q, v)` | `(3,)` | `R(q) v` (body → reference). Must equal `to_dcm(q) @ v`. |
| `rotate_inverse(q, v)` | `(3,)` | `R(q)ᵀ v` (reference → body) |
| `to_dcm(q)` | `(3,3)` | `R(q)` as in CONVENTIONS §2. Normalises `q` first. |
| `from_dcm(R)` | `(4,)` | Inverse of `to_dcm`; robust for all rotations (Shepperd / largest-diagonal method); canonical `w ≥ 0`. |
| `from_axis_angle(axis, angle)` | `(4,)` | rotation of `angle` rad about `axis` (normalised internally) |
| `from_rotvec(phi)` | `(4,)` | exponential map `[cos(|φ|/2), sin(|φ|/2) φ/|φ|]`; for `|φ| < 1e-8` use Taylor series (no division by zero); exact identity for `φ = 0` |
| `to_rotvec(q)` | `(3,)` | logarithm map, inverse of `from_rotvec`, result has `|φ| ≤ π` (canonicalise first); numerically stable near identity (use `atan2`) |
| `from_euler_zyx(roll, pitch, yaw)` | `(4,)` | `R = Rz(yaw) Ry(pitch) Rx(roll)`; args broadcast |
| `to_euler_zyx(q)` | `tuple(roll, pitch, yaw)` | inverse; roll, yaw in `(−π, π]`, pitch in `[−π/2, π/2]`; pitch argument clipped to `[−1, 1]` before `arcsin` |
| `small_angle_quat(dtheta)` | `(4,)` | `normalize([1, dθ/2])` |
| `error_rotvec(q_ref, q)` | `(3,)` | `to_rotvec(conjugate(q_ref) ⊗ q)`: the body-frame error `e` with `q = q_ref ⊗ exp(e)` |
| `angle_between(q1, q2)` | `()` | principal angle of the relative rotation, in `[0, π]`; sign-invariant (`q` vs `−q`) |
| `derivative(q, omega)` | `(4,)` | `½ q ⊗ [0, ω]` (ω body rate in body axes) |
| `omega_matrix(omega)` | `(4,4)` | `Ω(ω)` with `derivative(q, ω) == ½ Ω(ω) @ q` |
| `propagate(q, omega, dt)` | `(4,)` | `normalize(q ⊗ from_rotvec(ω dt))` — exact for constant body rate |
| `slerp(q0, q1, t)` | `(4,)` | spherical interpolation along the shortest arc; `t` in `[0, 1]` (array `t` broadcasts); `t=0 → q0`, `t=1 → ±q1` |
| `inject_attitude_error(q, dtheta)` | `(4,)` | `normalize(q ⊗ from_rotvec(dθ))` — the one shared error-injection used by every ES-EKF/MEKF |
| `attitude_reset_jacobian(dtheta)` | `(3,3)` | `I − ½[dθ×]`; attitude block of the reset Jacobian, `P_att ← G P_att Gᵀ` (Solà 2017 Eq. 287) |

Edge cases the tests must exercise: identity, 180° rotations (`from_dcm`, `to_rotvec`), tiny angles (1e-10 rad), `q` vs `−q`, batched inputs `(N,4)` and `(N,3)`.

Oracle: `scipy.spatial.transform.Rotation` with `from_quat([x, y, z, w])` (scalar-last) or `scalar_first=True`; `Rotation.as_matrix()` equals `to_dcm`; `Rotation.apply(v)` equals `rotate`; `Rotation.as_rotvec()` equals `to_rotvec`; `Rotation.from_euler('ZYX', [yaw, pitch, roll])` equals `from_euler_zyx(roll, pitch, yaw)`.

---

## `common/frames.py`

Constants (module level, SI):

| Name | Value |
|---|---|
| `WGS84_A` | 6378137.0 m |
| `WGS84_F` | 1 / 298.257223563 |
| `WGS84_B` | `WGS84_A * (1 − WGS84_F)` (= 6356752.314245 m) |
| `WGS84_E2` | `WGS84_F * (2 − WGS84_F)` |
| `GM_EARTH` | 3.986004418e14 m³/s² |
| `J2_EARTH` | 1.08262668e-3 |
| `R_EARTH` | `WGS84_A` |
| `OMEGA_EARTH` | 7.2921150e-5 rad/s |
| `G0` | 9.80665 m/s² |

Functions (batched over leading dims):

| Function | Returns | Behaviour |
|---|---|---|
| `lla_to_ecef(lat, lon, alt)` | `(3,)` | geodetic (rad, rad, m) → ECEF m |
| `ecef_to_lla(r)` | `(lat, lon, alt)` | inverse; accuracy ≤ 1e-6 m in height and ≤ 1e-11 rad in latitude for heights −1 km … 1000 km; correct at the poles and on the equator. Iterative (Bowring, converge to tolerance) or closed form (Zhu/Heikkinen). |
| `dcm_ecef_to_ned(lat, lon)` | `(3,3)` | `C` with `v_ned = C @ v_ecef`; rows are N, E, D unit vectors expressed in ECEF |
| `ecef_to_ned(r_ecef, lat0, lon0, alt0)` | `(3,)` | position of `r_ecef` relative to the reference point, in that point's NED |
| `ned_to_ecef(r_ned, lat0, lon0, alt0)` | `(3,)` | inverse |
| `lla_to_ned(lat, lon, alt, lat0, lon0, alt0)` | `(3,)` | convenience: `ecef_to_ned(lla_to_ecef(...), ...)` |
| `julian_date(dt)` | `float` | Julian date of a `datetime`; naive datetimes are UTC, aware ones converted to UTC; includes fractional day incl. microseconds. `2000-01-01 12:00 → 2451545.0`. |
| `gmst(jd_ut1)` | `()` | Greenwich mean sidereal time, rad in `[0, 2π)`, IAU-1982 formula (Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Eq. 3-47): `θ = 67310.54841 s + (876600·3600 + 8640184.812866) T + 0.093104 T² − 6.2e-6 T³` with `T = (jd − 2451545.0)/36525`, seconds of time → rad via `·2π/86400`. Array `jd` allowed. |
| `dcm_eci_to_ecef(jd)` | `(3,3)` | `Rz(gmst)` passive: `[[c, s, 0], [−s, c, 0], [0, 0, 1]]`, `c = cos θ`, `s = sin θ` |
| `eci_to_ecef(r_eci, jd)` / `ecef_to_eci(r_ecef, jd)` | `(3,)` | position rotation only (no velocity transport term) |
| `dcm_eci_to_lvlh(r_eci, v_eci)` | `(3,3)` | rows are LVLH x, y, z axes in ECI, per CONVENTIONS §4: `z = −r̂`, `y = −(r×v)/|r×v|`, `x = y × z`. Maps reference→frame; the body-to-ECI quaternion of an LVLH-aligned body is `from_dcm(C.T)` |

Oracles for tests: `(0,0,0) → (a,0,0)`; `(π/2, 0, 0) → (0,0,b)`; `(0, π/2, 1000) → (0, a+1000, 0)`; LLA↔ECEF↔NED round trips on random points (fixed seed); NED of a point 1 m north/east/down of the reference; GMST at JD 2451545.0 = 280.46061837° (Vallado); Vallado Example 3-5: 1992-08-20 12:14:00 UT1 → GMST 152.578787886°; `julian_date(2000-01-01 12:00) == 2451545.0`; LVLH rows orthonormal and right-handed (`det = +1`), and for a circular equatorial orbit `r=[R,0,0], v=[0,V,0]` → `x=[0,1,0]`, `y=[0,0,−1]`, `z=[−1,0,0]`.

---

## `common/ekf.py`

Written by the main session; its spec is the module docstring of `common/ekf.py`.
