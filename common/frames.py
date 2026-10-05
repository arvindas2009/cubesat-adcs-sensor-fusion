"""Earth constants and reference-frame transformations (numpy only).

Frames (see ``CONVENTIONS.md`` sections 4 and 5)
------------------------------------------------
ECI
    Earth-centred inertial. Approximated as the true-of-date frame obtained
    from ECEF by a single rotation about z by the Greenwich mean sidereal time
    (GMST). Precession, nutation, polar motion and the UT1-UTC offset are
    neglected (< 0.5 deg in 2026). ``sgp4`` TEME output is treated as ECI.
ECEF
    WGS-84 Earth-centred Earth-fixed, ``r_ecef = Rz(GMST) r_eci`` (passive
    rotation).
LLA
    WGS-84 geodetic latitude (rad), longitude (rad), height above the
    ellipsoid (m).
NED
    Local tangent plane (North-East-Down) at a reference LLA.
LVLH
    Local-vertical local-horizontal: z = -r_hat (nadir),
    y = -(r x v)/|r x v| (negative orbit normal), x = y x z.

Units
-----
SI throughout: metres, seconds, radians, rad/s. Julian dates are in days.
All functions accept array-likes (converted with ``np.asarray(..., float)``),
broadcast over leading dimensions, and never mutate their inputs. Vectors
have the last axis of length 3; direction-cosine matrices (DCMs) have shape
``(..., 3, 3)`` and act as ``v_to = C @ v_from``.

This module stands alone and depends on numpy and the standard library only.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence

import numpy as np

# --------------------------------------------------------------------------
# Constants (SI)
# --------------------------------------------------------------------------
WGS84_A: float = 6378137.0                      # semi-major axis [m]
WGS84_F: float = 1.0 / 298.257223563            # flattening [-]
WGS84_B: float = WGS84_A * (1.0 - WGS84_F)      # semi-minor axis [m]
WGS84_E2: float = WGS84_F * (2.0 - WGS84_F)     # first eccentricity squared [-]
GM_EARTH: float = 3.986004418e14                # gravitational parameter [m^3/s^2]
J2_EARTH: float = 1.08262668e-3                 # J2 zonal harmonic [-]
R_EARTH: float = WGS84_A                        # equatorial radius [m]
OMEGA_EARTH: float = 7.2921150e-5               # Earth rotation rate [rad/s]
G0: float = 9.80665                             # standard gravity [m/s^2]

_WGS84_EP2: float = WGS84_E2 / (1.0 - WGS84_E2)  # second eccentricity squared
_J2000_JD: float = 2451545.0
_TWO_PI: float = 2.0 * np.pi


def _f(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


# --------------------------------------------------------------------------
# Geodetic <-> ECEF
# --------------------------------------------------------------------------
def lla_to_ecef(lat, lon, alt) -> np.ndarray:
    """Geodetic to ECEF position.

    Parameters: ``lat``, ``lon`` in rad, ``alt`` height above the WGS-84
    ellipsoid in m (broadcast together). Returns ECEF position in m,
    shape ``(..., 3)``.
    """
    lat, lon, alt = np.broadcast_arrays(_f(lat), _f(lon), _f(alt))
    slat, clat = np.sin(lat), np.cos(lat)
    slon, clon = np.sin(lon), np.cos(lon)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * slat * slat)  # prime-vertical radius
    x = (n + alt) * clat * clon
    y = (n + alt) * clat * slon
    z = (n * (1.0 - WGS84_E2) + alt) * slat
    return np.stack([x, y, z], axis=-1)


def ecef_to_lla(r):
    """ECEF position to geodetic coordinates.

    Parameters: ``r`` ECEF position in m, shape ``(..., 3)``.
    Returns ``(lat, lon, alt)``: latitude in ``[-pi/2, pi/2]`` rad, longitude
    in ``(-pi, pi]`` rad, height above the WGS-84 ellipsoid in m; each of
    shape ``(...)``.

    Bowring's iteration on the parametric latitude, run to convergence
    (``|dlat| < 1e-14`` rad, at most 10 iterations). Height is evaluated as
    ``p cos(lat) + z sin(lat) - a sqrt(1 - e2 sin^2(lat))``, which is free of
    the ``p / cos(lat)`` singularity at the poles. Exact on the z-axis and in
    the equatorial plane. The geocentre (``r = 0``) is not meaningful.
    """
    r = _f(r)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    p = np.hypot(x, y)
    lon = np.arctan2(y, x)

    # Initial parametric latitude beta (spherical-ish starting guess).
    beta = np.arctan2((1.0 - WGS84_F) * z, p)
    lat = beta
    for _ in range(10):
        sb, cb = np.sin(beta), np.cos(beta)
        lat_new = np.arctan2(z + _WGS84_EP2 * WGS84_B * sb**3,
                             p - WGS84_E2 * WGS84_A * cb**3)
        # Parametric latitude from geodetic latitude: tan(beta) = (1-f) tan(lat)
        beta = np.arctan2((1.0 - WGS84_F) * np.sin(lat_new), np.cos(lat_new))
        converged = np.all(np.abs(lat_new - lat) < 1e-14)
        lat = lat_new
        if converged:
            break

    slat, clat = np.sin(lat), np.cos(lat)
    alt = p * clat + z * slat - WGS84_A * np.sqrt(1.0 - WGS84_E2 * slat * slat)
    return lat, lon, alt


# --------------------------------------------------------------------------
# NED
# --------------------------------------------------------------------------
def dcm_ecef_to_ned(lat, lon) -> np.ndarray:
    """DCM ``C`` with ``v_ned = C @ v_ecef`` at geodetic ``lat``, ``lon`` (rad).

    Rows are the North, East and Down unit vectors expressed in ECEF.
    Shape ``(..., 3, 3)``.
    """
    lat, lon = np.broadcast_arrays(_f(lat), _f(lon))
    slat, clat = np.sin(lat), np.cos(lat)
    slon, clon = np.sin(lon), np.cos(lon)
    zero = np.zeros_like(lat)
    c = np.stack([
        np.stack([-slat * clon, -slat * slon, clat], axis=-1),
        np.stack([-slon, clon, zero], axis=-1),
        np.stack([-clat * clon, -clat * slon, -slat], axis=-1),
    ], axis=-2)
    return c


def ecef_to_ned(r_ecef, lat0, lon0, alt0) -> np.ndarray:
    """Position of ``r_ecef`` (m, ECEF) relative to the reference point
    (``lat0``, ``lon0`` rad, ``alt0`` m), expressed in that point's NED
    axes (m). Shape ``(..., 3)``.
    """
    r_ecef = _f(r_ecef)
    r0 = lla_to_ecef(lat0, lon0, alt0)
    c = dcm_ecef_to_ned(lat0, lon0)
    return np.einsum("...ij,...j->...i", c, r_ecef - r0)


def ned_to_ecef(r_ned, lat0, lon0, alt0) -> np.ndarray:
    """Inverse of :func:`ecef_to_ned`: NED offset (m) from the reference
    point (``lat0``, ``lon0`` rad, ``alt0`` m) to ECEF position (m).
    Shape ``(..., 3)``.
    """
    r_ned = _f(r_ned)
    r0 = lla_to_ecef(lat0, lon0, alt0)
    c = dcm_ecef_to_ned(lat0, lon0)
    return np.einsum("...ji,...j->...i", c, r_ned) + r0


def lla_to_ned(lat, lon, alt, lat0, lon0, alt0) -> np.ndarray:
    """Geodetic point (rad, rad, m) to NED offset (m) from the reference
    (``lat0``, ``lon0`` rad, ``alt0`` m). Shape ``(..., 3)``.
    """
    return ecef_to_ned(lla_to_ecef(lat, lon, alt), lat0, lon0, alt0)


# --------------------------------------------------------------------------
# Time and ECI <-> ECEF
# --------------------------------------------------------------------------
def _julian_date_one(dt: _dt.datetime) -> float:
    if dt.tzinfo is not None:
        dt = dt.astimezone(_dt.timezone.utc).replace(tzinfo=None)
    delta = dt - _dt.datetime(2000, 1, 1, 12, 0, 0)
    day_frac = (delta.seconds + delta.microseconds * 1e-6) / 86400.0
    return _J2000_JD + delta.days + day_frac


def julian_date(dt):
    """Julian date (days) of a ``datetime`` (UTC; UT1 ~ UTC).

    Naive datetimes are taken as UTC, aware ones are converted to UTC.
    Microseconds are included. A sequence (or array) of datetimes returns a
    float array of the same shape. ``2000-01-01 12:00 -> 2451545.0``.
    """
    if isinstance(dt, _dt.datetime):
        return _julian_date_one(dt)
    if isinstance(dt, (Sequence, np.ndarray)) and not isinstance(dt, (str, bytes)):
        arr = np.asarray(dt, dtype=object)
        out = np.empty(arr.shape, dtype=float)
        for idx in np.ndindex(arr.shape):
            out[idx] = _julian_date_one(arr[idx])
        return out
    raise TypeError(f"julian_date expects a datetime or a sequence of them, got {type(dt)!r}")


def gmst(jd_ut1):
    """Greenwich mean sidereal time in rad, in ``[0, 2*pi)``.

    ``jd_ut1`` is the UT1 Julian date in days (scalar or array). IAU-1982
    formula (Vallado, Eq. 3-47): ``theta = 67310.54841 s + (876600*3600 +
    8640184.812866) T + 0.093104 T^2 - 6.2e-6 T^3``, with
    ``T = (jd - 2451545.0)/36525`` (Julian centuries), converted from seconds
    of time to rad by ``2*pi/86400``.
    """
    jd = _f(jd_ut1)
    t = (jd - _J2000_JD) / 36525.0
    theta_s = (67310.54841
               + (876600.0 * 3600.0 + 8640184.812866) * t
               + 0.093104 * t**2
               - 6.2e-6 * t**3)
    theta = np.mod(theta_s, 86400.0) * (_TWO_PI / 86400.0)
    theta = np.mod(theta, _TWO_PI)  # guard against rounding to exactly 2*pi
    return theta


def dcm_eci_to_ecef(jd) -> np.ndarray:
    """Passive rotation ``Rz(gmst)`` taking ECI vectors to ECEF.

    ``jd`` is the UT1 Julian date (days, scalar or array). Returns shape
    ``(..., 3, 3)``: ``[[c, s, 0], [-s, c, 0], [0, 0, 1]]``,
    ``c = cos(gmst)``, ``s = sin(gmst)``.
    """
    th = gmst(jd)
    c, s = np.cos(th), np.sin(th)
    zero, one = np.zeros_like(th), np.ones_like(th)
    return np.stack([
        np.stack([c, s, zero], axis=-1),
        np.stack([-s, c, zero], axis=-1),
        np.stack([zero, zero, one], axis=-1),
    ], axis=-2)


def eci_to_ecef(r_eci, jd) -> np.ndarray:
    """Rotate an ECI position (m) to ECEF (m) at UT1 Julian date ``jd``.

    Position rotation only (no velocity transport term). ``r_eci`` has shape
    ``(..., 3)``; ``jd`` broadcasts against its leading dimensions.
    """
    r = _f(r_eci)
    th = gmst(jd)
    c, s = np.cos(th), np.sin(th)
    x, y, z = np.broadcast_arrays(r[..., 0], r[..., 1], r[..., 2], c)[:3]
    return np.stack([c * x + s * y, -s * x + c * y, z], axis=-1)


def ecef_to_eci(r_ecef, jd) -> np.ndarray:
    """Inverse of :func:`eci_to_ecef`: ECEF position (m) to ECI (m) at UT1
    Julian date ``jd``. Position rotation only.
    """
    r = _f(r_ecef)
    th = gmst(jd)
    c, s = np.cos(th), np.sin(th)
    x, y, z = np.broadcast_arrays(r[..., 0], r[..., 1], r[..., 2], c)[:3]
    return np.stack([c * x - s * y, s * x + c * y, z], axis=-1)


# --------------------------------------------------------------------------
# LVLH
# --------------------------------------------------------------------------
def dcm_eci_to_lvlh(r_eci, v_eci) -> np.ndarray:
    """DCM taking ECI vectors to LVLH, ``v_lvlh = C @ v_eci``.

    ``r_eci`` (m) and ``v_eci`` (m/s) have shape ``(..., 3)``. Rows of the
    returned ``(..., 3, 3)`` matrix are the LVLH x, y, z axes in ECI:
    ``z = -r/|r|``, ``y = -(r x v)/|r x v|``, ``x = y x z``.

    Note: like every DCM in this module this maps *reference -> frame*. The
    body-to-reference quaternion of a body aligned with LVLH (the nadir-pointing
    target) is therefore ``quaternion.from_dcm(C.T)``, not ``from_dcm(C)``.
    """
    r, v = np.broadcast_arrays(_f(r_eci), _f(v_eci))
    z = -r / np.linalg.norm(r, axis=-1, keepdims=True)
    h = np.cross(r, v)
    y = -h / np.linalg.norm(h, axis=-1, keepdims=True)
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=-2)
