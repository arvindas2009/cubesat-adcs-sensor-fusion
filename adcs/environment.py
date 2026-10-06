"""Project B space environment: Sun, eclipse, atmosphere, geomagnetic field.

Conventions (CONVENTIONS.md): SI units, radians internally, ECI = GMST-only
true-of-date frame (z = Earth axis), time on axis 0, simulation time = seconds
since ``cfg.epoch`` (UTC, UT1 ~ UTC). Positions in m, vectors batched over
leading dimensions.

Public API: ``sun_vector_eci``, ``sun_direction_eci``, ``sunlit``,
``eclipse_fraction``, ``beta_angle``, ``atmospheric_density``,
``geodetic_height``, ``magnetic_field_eci``, ``FieldTable``.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np
from scipy.interpolate import CubicSpline

from common.frames import (
    R_EARTH,
    dcm_ecef_to_ned,
    ecef_to_eci,
    ecef_to_lla,
    eci_to_ecef,
)

from .config import DEFAULT, SimConfig

#: Astronomical unit [m] (IAU 2012).
AU = 1.495978707e11

# Exponential atmosphere, Vallado Table 8-4 (400-450 km row; citation unverified) [B-10].
_RHO0 = 3.725e-12      # kg/m^3 at h0
_H0 = 400e3            # m
_SCALE_H = 58.515e3    # m

_J2000_JD = 2451545.0


def _f(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


# --------------------------------------------------------------------------
# Sun
# --------------------------------------------------------------------------
def sun_vector_eci(jd) -> np.ndarray:
    """Geocentric Sun position in ECI (m), shape ``(..., 3)``.

    ``jd`` = Julian date (days, UTC taken as UT1/TDB; scalar or array).
    Vallado Algorithm 29 (low-precision Astronomical Almanac, ~0.01 deg),
    mean-equinox-of-date frame, used here as the GMST-only ECI frame.
    """
    jd = _f(jd)
    t = (jd - _J2000_JD) / 36525.0
    lam_m = 280.460 + 36000.771 * t                      # deg, mean longitude
    m = np.deg2rad(357.5291092 + 35999.05034 * t)        # rad, mean anomaly
    lam = np.deg2rad(lam_m + 1.914666471 * np.sin(m) + 0.019994643 * np.sin(2.0 * m))
    r_au = 1.000140612 - 0.016708617 * np.cos(m) - 0.000139589 * np.cos(2.0 * m)
    eps = np.deg2rad(23.439291 - 0.0130042 * t)
    r = r_au * AU
    return np.stack([r * np.cos(lam),
                     r * np.cos(eps) * np.sin(lam),
                     r * np.sin(eps) * np.sin(lam)], axis=-1)


def sun_direction_eci(jd) -> np.ndarray:
    """Unit vector from Earth centre to the Sun in ECI, shape ``(..., 3)``.

    ``jd`` = Julian date (days).
    """
    s = sun_vector_eci(jd)
    return s / np.linalg.norm(s, axis=-1, keepdims=True)


def sunlit(r_eci, s_hat, re: float = R_EARTH) -> np.ndarray:
    """Cylindrical-shadow illumination flag (bool array, shape of the broadcast leading dims).

    ``r_eci`` spacecraft position (m, ECI), ``s_hat`` unit Earth-to-Sun vector (ECI),
    both ``(..., 3)``; ``re`` = shadow cylinder radius (m). Returns ``False`` iff
    ``r.s < 0`` and the distance from the shadow axis is ``< re``. No penumbra.
    """
    r, s = np.broadcast_arrays(_f(r_eci), _f(s_hat))
    along = np.sum(r * s, axis=-1)
    perp = np.linalg.norm(r - along[..., None] * s, axis=-1)
    return ~((along < 0.0) & (perp < re))


def eclipse_fraction(r_eci_series, s_hat_series) -> float:
    """Fraction of samples in shadow (mean of ``~sunlit``).

    ``r_eci_series`` ``(N, 3)`` m, ``s_hat_series`` ``(N, 3)`` or ``(3,)`` (unit, ECI).
    Dimensionless, in [0, 1].
    """
    return float(np.mean(~sunlit(r_eci_series, s_hat_series)))


def beta_angle(r_eci, v_eci, s_hat) -> np.ndarray:
    """Signed beta angle (rad): angle between the Sun direction and the orbit plane.

    ``asin(s_hat . h_hat)`` with ``h_hat = r x v / |r x v|``. Inputs ``(..., 3)``
    in ECI (m, m/s, unit). Positive when the Sun is on the orbit-normal side.
    """
    r, v, s = np.broadcast_arrays(_f(r_eci), _f(v_eci), _f(s_hat))
    h = np.cross(r, v)
    h_hat = h / np.linalg.norm(h, axis=-1, keepdims=True)
    return np.arcsin(np.clip(np.sum(h_hat * s, axis=-1), -1.0, 1.0))


# --------------------------------------------------------------------------
# Atmosphere and geodetic height
# --------------------------------------------------------------------------
def atmospheric_density(h) -> np.ndarray:
    """Exponential atmospheric density (kg/m^3) at geodetic height ``h`` (m).

    ``rho = rho0 exp(-(h - h0)/H)``, ``h0 = 400 km``, ``rho0 = 3.725e-12``,
    ``H = 58.515 km`` (Vallado Table 8-4, medium solar activity; valid roughly
    300-500 km; unverified citation, assumptions B-10). Raises ``ValueError``
    for ``h < 0``.
    """
    h = _f(h)
    if np.any(h < 0.0):
        raise ValueError("atmospheric_density: height must be >= 0 m")
    return _RHO0 * np.exp(-(h - _H0) / _SCALE_H)


def geodetic_height(r_eci, jd) -> np.ndarray:
    """Height above the WGS-84 ellipsoid (m) of ECI position(s) ``r_eci`` (m, ``(..., 3)``)
    at Julian date ``jd`` (days, UT1 ~ UTC)."""
    _, _, h = ecef_to_lla(eci_to_ecef(r_eci, jd))
    return h


# --------------------------------------------------------------------------
# Geomagnetic field
# --------------------------------------------------------------------------
def magnetic_field_eci(r_eci, jd, date: _dt.datetime) -> np.ndarray:
    """IGRF magnetic field in ECI (tesla) via ``ppigrf``.

    ``r_eci`` ``(3,)`` or ``(N, 3)`` position (m, ECI); ``jd`` scalar or ``(N,)``
    Julian date (days) for the Earth rotation (GMST); ``date`` = ``datetime``
    at which the IGRF coefficients are evaluated (held constant, assumptions B-11).
    Returns an array of the shape of ``r_eci``.

    Chain: ECI -> ECEF (GMST) -> geodetic lat/lon/height -> ``ppigrf.igrf``
    (nT, local geodetic east/north/up) -> T -> NED = (n, e, -u) -> ECEF with
    ``dcm_ecef_to_ned(lat, lon).T`` -> ECI (vector rotation by -GMST only).
    """
    import ppigrf  # local import: slow to load, only needed here

    r = _f(r_eci)
    single = r.ndim == 1
    r2 = np.atleast_2d(r)
    jd_arr = np.broadcast_to(_f(jd), r2.shape[:1])

    # NOTE: ppigrf divides by sin(colatitude): NaN/inf exactly at a geographic pole
    # (cannot occur for the 51.6 deg orbit; relevant for polar configs).
    lat, lon, h = ecef_to_lla(eci_to_ecef(r2, jd_arr))
    be, bn, bu = ppigrf.igrf(np.rad2deg(lon), np.rad2deg(lat), h * 1e-3, date)
    be, bn, bu = be[0], bn[0], bu[0]                       # (N,) nT
    b_ned = np.stack([bn, be, -bu], axis=-1) * 1e-9        # T
    c = dcm_ecef_to_ned(lat, lon)                          # (N, 3, 3), v_ned = C v_ecef
    b_ecef = np.einsum("nji,nj->ni", c, b_ned)             # C^T b_ned
    b_eci = ecef_to_eci(b_ecef, jd_arr)
    return b_eci[0] if single else b_eci


class FieldTable:
    """Precomputed geomagnetic field (and orbit/Sun helpers) along an ephemeris.

    ``FieldTable(ephemeris, cfg, t_end, dt=10.0)``: evaluates
    :func:`magnetic_field_eci` (tesla, ECI) once on ``t = 0, dt, ... >= t_end``
    (s since ``cfg.epoch``) and interpolates each component with a cubic
    spline, so closed-loop simulations never call ppigrf per step. The IGRF
    date is ``cfg.epoch``. If the last grid point would exceed
    ``ephemeris.t_end`` it is clipped to ``t_end`` (requires
    ``ephemeris.t_end >= t_end``). Queries outside ``[0, t_end]`` raise
    ``ValueError``.
    """

    def __init__(self, ephemeris, cfg: SimConfig = DEFAULT, t_end: float | None = None,
                 dt: float = 10.0):
        if t_end is None:
            t_end = ephemeris.t_end
        t_end, dt = float(t_end), float(dt)
        if dt <= 0.0 or t_end <= 0.0:
            raise ValueError("FieldTable: t_end and dt must be > 0")
        eph_end = float(getattr(ephemeris, "t_end", np.inf))
        if eph_end < t_end - 1e-9:
            raise ValueError("FieldTable: ephemeris.t_end is shorter than t_end")
        n = int(np.ceil(t_end / dt - 1e-12))
        grid = np.arange(n + 1, dtype=float) * dt
        if grid[-1] > eph_end:
            grid[-1] = t_end          # t_end > grid[-2] by construction
        self._eph = ephemeris
        self._cfg = cfg
        self.t_grid = grid
        self.t_end = float(grid[-1])
        r, _ = ephemeris.at(grid)
        b = magnetic_field_eci(r, cfg.jd_epoch + grid / 86400.0, cfg.epoch)
        self._spline = CubicSpline(grid, b, axis=0)

    def _check(self, t) -> np.ndarray:
        t = _f(t)
        if np.any(t < -1e-9) or np.any(t > self.t_end + 1e-9):
            raise ValueError(f"FieldTable: t outside [0, {self.t_end}] s")
        return np.clip(t, 0.0, self.t_end)

    def b_eci(self, t) -> np.ndarray:
        """Magnetic field in ECI (T), shape ``t.shape + (3,)``; ``t`` in s since ``cfg.epoch``."""
        return self._spline(self._check(t))

    def r_eci(self, t) -> np.ndarray:
        """Spacecraft position in ECI (m) from the ephemeris, shape ``t.shape + (3,)``."""
        return self._eph.at(self._check(t))[0]

    def v_eci(self, t) -> np.ndarray:
        """Spacecraft velocity in ECI (m/s) from the ephemeris, shape ``t.shape + (3,)``."""
        return self._eph.at(self._check(t))[1]

    def sun_dir_eci(self, t) -> np.ndarray:
        """Unit Earth-to-Sun vector in ECI, shape ``t.shape + (3,)``."""
        return sun_direction_eci(self._cfg.jd_epoch + self._check(t) / 86400.0)

    def sunlit(self, t) -> np.ndarray:
        """Bool array (cylindrical shadow) at ``t`` s since ``cfg.epoch``."""
        return sunlit(self.r_eci(t), self.sun_dir_eci(t))
