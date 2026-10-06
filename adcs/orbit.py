"""Orbit propagation for Project B (SI units, ECI frame).

Frame: ECI = true-of-date frame obtained from ECEF by a GMST rotation only
(CONVENTIONS.md section 4); z = Earth rotation axis. Positions in m,
velocities in m/s, angles in rad, times in s since the start of propagation.

Force model: two-body + J2 (spherical-harmonic zonal term, Vallado Eq. 8-30),
integrated with SciPy's DOP853. SGP4 (WGS72, TEME treated as ECI) is provided
only as an independent comparison.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from common.frames import GM_EARTH, J2_EARTH, R_EARTH
from adcs.config import DEFAULT, SimConfig

__all__ = [
    "kepler_to_rv",
    "rv_to_kepler",
    "initial_state",
    "orbital_period",
    "accel_j2",
    "specific_energy",
    "propagate",
    "Ephemeris",
    "sgp4_ephemeris",
]

_TWO_PI = 2.0 * np.pi


def _rot1(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def _rot3(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def kepler_to_rv(a, ecc, inc, raan, argp, nu, mu: float = GM_EARTH):
    """Classical orbital elements to ECI position and velocity.

    Args:
        a: semi-major axis (m); ecc: eccentricity (-), 0 <= ecc < 1;
        inc, raan, argp, nu: inclination, right ascension of the ascending
        node, argument of periapsis, true anomaly (rad); mu: GM (m^3/s^2).

    Returns:
        (r, v): ECI position (3,) m and velocity (3,) m/s. The perifocal
        vectors are rotated to ECI with R3(-raan) R1(-inc) R3(-argp).
        Works for ecc = 0 (argp and nu then only enter through argp + nu).
    """
    a, ecc, inc, raan, argp, nu = (float(x) for x in (a, ecc, inc, raan, argp, nu))
    p = a * (1.0 - ecc**2)
    cn, sn = np.cos(nu), np.sin(nu)
    rmag = p / (1.0 + ecc * cn)
    r_pf = np.array([rmag * cn, rmag * sn, 0.0])
    k = np.sqrt(mu / p)
    v_pf = np.array([-k * sn, k * (ecc + cn), 0.0])
    # perifocal -> ECI: R3(-raan) R1(-inc) R3(-argp); _rotN(a) is the passive rotation
    rot = _rot3(-raan) @ _rot1(-inc) @ _rot3(-argp)
    return rot @ r_pf, rot @ v_pf


def rv_to_kepler(r, v, mu: float = GM_EARTH) -> dict:
    """ECI position/velocity to classical orbital elements (inverse of ``kepler_to_rv``).

    Args:
        r: ECI position (3,) m; v: ECI velocity (3,) m/s; mu: GM (m^3/s^2).

    Returns:
        dict with ``a`` (m), ``ecc`` (-), ``inc`` (rad, 0..pi), ``raan``,
        ``argp``, ``nu`` (rad, in [0, 2 pi)). Singular cases: for a circular
        orbit argp = 0 and nu is the argument of latitude (or true longitude
        if also equatorial); for an equatorial orbit raan = 0 and argp is
        the longitude of periapsis. Elliptic orbits only.
    """
    r = np.asarray(r, float).reshape(3)
    v = np.asarray(v, float).reshape(3)
    rn = np.linalg.norm(r)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    h_hat = h / hn
    evec = np.cross(v, h) / mu - r / rn
    ecc = float(np.linalg.norm(evec))
    a = 1.0 / (2.0 / rn - v @ v / mu)
    inc = float(np.arccos(np.clip(h_hat[2], -1.0, 1.0)))

    nvec = np.array([-h[1], h[0], 0.0])
    nn = np.linalg.norm(nvec)
    tol_c = 1e-11
    tol_i = 1e-11
    circular = ecc < tol_c
    equatorial = nn < tol_i * hn

    if equatorial:
        raan = 0.0
        n_hat = np.array([1.0, 0.0, 0.0])
    else:
        n_hat = nvec / nn
        raan = float(np.arctan2(n_hat[1], n_hat[0]))
    # in-plane unit vector 90 deg ahead of n_hat
    m_hat = np.cross(h_hat, n_hat)

    if circular:
        argp = 0.0
        nu = float(np.arctan2(r @ m_hat, r @ n_hat))
        ecc = 0.0 if ecc < tol_c else ecc
    else:
        e_hat = evec / ecc
        argp = float(np.arctan2(e_hat @ m_hat, e_hat @ n_hat))
        q_hat = np.cross(h_hat, e_hat)
        nu = float(np.arctan2(r @ q_hat, r @ e_hat))
    return {
        "a": float(a),
        "ecc": ecc,
        "inc": inc,
        "raan": raan % _TWO_PI,
        "argp": argp % _TWO_PI,
        "nu": nu % _TWO_PI,
    }


def initial_state(cfg: SimConfig = DEFAULT):
    """Initial ECI state (r (3,) m, v (3,) m/s) from ``cfg.orbit`` (osculating elements)."""
    o = cfg.orbit
    return kepler_to_rv(o.a, o.ecc, o.inc, o.raan, o.argp, o.nu)


def orbital_period(a, mu: float = GM_EARTH):
    """Keplerian period 2 pi sqrt(a^3/mu) in s for semi-major axis ``a`` (m)."""
    return _TWO_PI * np.sqrt(np.asarray(a, float) ** 3 / mu)


def accel_j2(r, mu: float = GM_EARTH, j2: float = J2_EARTH, re: float = R_EARTH):
    """Two-body + J2 gravitational acceleration in ECI (z = Earth axis).

    Args:
        r: ECI position(s), shape (..., 3), m.
    Returns:
        acceleration (..., 3) in m/s^2 (Vallado Eq. 8-30). ``j2 = 0`` gives pure two-body.
    """
    r = np.asarray(r, float)
    rn2 = np.sum(r * r, axis=-1, keepdims=True)
    rn = np.sqrt(rn2)
    z2r2 = r[..., 2:3] ** 2 / rn2
    k = 1.5 * j2 * (re / rn) ** 2
    fxy = 1.0 + k * (1.0 - 5.0 * z2r2)
    fz = 1.0 + k * (3.0 - 5.0 * z2r2)
    fac = np.concatenate([fxy, fxy, fz], axis=-1)
    return -mu * r / (rn2 * rn) * fac


def specific_energy(r, v, j2: bool = True, mu: float = GM_EARTH, jj2: float = J2_EARTH,
                    re: float = R_EARTH):
    """Specific mechanical energy v^2/2 + Phi(r) in J/kg (ECI, batched over leading dims).

    Phi = -mu/r [1 - J2 (Re/r)^2 (3 z^2/r^2 - 1)/2]; ``j2=False`` drops the J2
    term. Conserved by the J2 flow of ``accel_j2`` (= -grad Phi).
    """
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    rn = np.linalg.norm(r, axis=-1)
    corr = 1.0
    if j2:
        corr = 1.0 - jj2 * (re / rn) ** 2 * (3.0 * (r[..., 2] / rn) ** 2 - 1.0) / 2.0
    return 0.5 * np.sum(v * v, axis=-1) - mu / rn * corr


def _rhs(j2: bool):
    j2c = J2_EARTH if j2 else 0.0

    def f(_t, y):
        return np.concatenate([y[3:], accel_j2(y[:3], j2=j2c)])

    return f


def propagate(r0, v0, t, j2: bool = True, rtol: float = 1e-12, atol: float = 1e-6):
    """Propagate an ECI state with two-body (+ J2) dynamics, DOP853.

    Args:
        r0, v0: initial ECI position (3,) m / velocity (3,) m/s at ``t[0]``.
        t: increasing 1-D array of times (s since start; t[0] may be 0).
        j2: include the J2 term; rtol/atol: DOP853 tolerances (default rtol 1e-12, tighter
        than the 1e-10 first specified, so that 10-orbit two-body drift is < 1 mm).
    Returns:
        (r, v): arrays (N, 3) in m and m/s at the times ``t``.
    """
    t = np.atleast_1d(np.asarray(t, float))
    if t.ndim != 1 or np.any(np.diff(t) <= 0.0):
        raise ValueError("t must be a strictly increasing 1-D array")
    y0 = np.concatenate([np.asarray(r0, float).reshape(3), np.asarray(v0, float).reshape(3)])
    if t.size == 1:
        return y0[None, :3].copy(), y0[None, 3:].copy()
    sol = solve_ivp(_rhs(j2), (t[0], t[-1]), y0, method="DOP853", t_eval=t, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"orbit integration failed: {sol.message}")
    y = sol.y.T
    return y[:, :3], y[:, 3:]


class Ephemeris:
    """Dense-output ECI ephemeris, integrated once over [0, t_end].

    Args:
        r0, v0: ECI state at t = 0 (m, m/s); t_end: end time (s) > 0; j2: include J2.
    ``at(t)`` returns (r, v) in m, m/s, with shape (..., 3) for scalar or array ``t``
    (s since start); raises ``ValueError`` outside [0, t_end].
    """

    def __init__(self, r0, v0, t_end: float, j2: bool = True, rtol: float = 1e-12,
                 atol: float = 1e-6):
        t_end = float(t_end)
        if not t_end > 0.0:
            raise ValueError("t_end must be positive")
        self.t_end = t_end
        self.j2 = bool(j2)
        y0 = np.concatenate([np.asarray(r0, float).reshape(3), np.asarray(v0, float).reshape(3)])
        sol = solve_ivp(_rhs(self.j2), (0.0, t_end), y0, method="DOP853", dense_output=True,
                        rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"orbit integration failed: {sol.message}")
        self._sol = sol.sol

    def at(self, t):
        """ECI position and velocity at times ``t`` (s since start): (r, v), (..., 3) each."""
        t = np.asarray(t, float)
        if t.size and (np.nanmin(t) < 0.0 or np.nanmax(t) > self.t_end or np.any(np.isnan(t))):
            raise ValueError(f"t outside [0, {self.t_end}] s")
        y = self._sol(t.ravel()).T.reshape(t.shape + (6,))
        return y[..., :3], y[..., 3:]


def sgp4_ephemeris(line1: str, line2: str, jd0: float, t):
    """SGP4 (WGS72) position/velocity at ``jd0 + t/86400``; TEME treated as ECI.

    Args:
        line1, line2: two-line element set; jd0: Julian date (days);
        t: 1-D array of times (s since ``jd0``).
    Returns:
        (r, v): (N, 3) in m and m/s (converted from SGP4's km, km/s).
    Raises:
        RuntimeError: if SGP4 returns a non-zero error code for any time.
    """
    from sgp4.api import Satrec

    sat = Satrec.twoline2rv(line1, line2)
    t = np.atleast_1d(np.asarray(t, float))
    jd = np.full(t.shape, float(jd0))
    fr = t / 86400.0
    err, r, v = sat.sgp4_array(jd, fr)
    err = np.asarray(err)
    if np.any(err != 0):
        raise RuntimeError(f"SGP4 error code(s) {sorted(set(int(e) for e in err[err != 0]))}")
    return np.asarray(r, float) * 1e3, np.asarray(v, float) * 1e3
