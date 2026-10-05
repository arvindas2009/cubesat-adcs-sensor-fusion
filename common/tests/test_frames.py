"""Tests for ``common.frames`` written from ``docs/specs/common_api.md`` and
``CONVENTIONS.md`` sections 4-5 (not from the implementation).

Independent oracles used here:
  * closed-form geodetic -> ECEF written out in the test (``_oracle_lla_to_ecef``),
  * closed-form NED rotation matrix written out in the test (``_oracle_c_ecef_ned``),
  * ``scipy.spatial.transform.Rotation`` for the passive Rz(theta),
  * ``datetime.toordinal`` for Julian dates,
  * Vallado, Fundamentals of Astrodynamics and Applications, 4th ed.,
    Example 3-5 and Eq. 3-47 for sidereal time.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from common import frames as F
from common import quaternion as Q

# --------------------------------------------------------------------------- #
# Independent reference values / helpers
# --------------------------------------------------------------------------- #
A = 6378137.0
FLAT = 1.0 / 298.257223563
B = A * (1.0 - FLAT)
E2 = FLAT * (2.0 - FLAT)

TWO_PI = 2.0 * np.pi
SIDEREAL_DAY_S = 86164.0905  # mean sidereal day, s (Vallado Table / IERS)
JD_J2000 = 2451545.0


def _wrap(x):
    """Wrap an angle difference into [-pi, pi)."""
    return (np.asarray(x) + np.pi) % TWO_PI - np.pi


def _oracle_lla_to_ecef(lat, lon, h):
    """Closed form: N = a / sqrt(1 - e^2 sin^2 phi)."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    h = np.asarray(h, dtype=float)
    n = A / np.sqrt(1.0 - E2 * np.sin(lat) ** 2)
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - E2) + h) * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def _oracle_c_ecef_ned(lat, lon):
    """Textbook ECEF -> NED rotation (rows N, E, D in ECEF)."""
    sp, cp = np.sin(lat), np.cos(lat)
    sl, cl = np.sin(lon), np.cos(lon)
    return np.array(
        [
            [-sp * cl, -sp * sl, cp],
            [-sl, cl, 0.0],
            [-cp * cl, -cp * sl, -sp],
        ]
    )


def _oracle_gmst_formula(jd):
    """IAU-1982 GMST, Vallado Eq. 3-47 as written in the spec (rad, [0, 2pi))."""
    t = (np.asarray(jd, dtype=float) - JD_J2000) / 36525.0
    sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * t
        + 0.093104 * t**2
        - 6.2e-6 * t**3
    )
    return np.mod(sec * TWO_PI / 86400.0, TWO_PI)


def _oracle_jd(d: dt.datetime) -> float:
    """JD of a naive-UTC datetime via the proleptic-Gregorian ordinal.

    ordinal 1 (0001-01-01 00:00) == JD 1721425.5.
    """
    frac = (
        d.hour * 3600 + d.minute * 60 + d.second + d.microsecond * 1e-6
    ) / 86400.0
    return d.toordinal() + 1721424.5 + frac


def _is_orthonormal(c, tol=1e-14):
    c = np.asarray(c)
    eye = np.broadcast_to(np.eye(3), c.shape)
    return np.max(np.abs(c @ np.swapaxes(c, -1, -2) - eye)) < tol


def _random_lla(rng, n, h_lo=-1e3, h_hi=1e6):
    lat = rng.uniform(-np.pi / 2, np.pi / 2, n)
    lon = rng.uniform(-np.pi, np.pi, n)
    lon = np.where(lon == -np.pi, np.pi, lon)  # (-pi, pi]
    h = rng.uniform(h_lo, h_hi, n)
    return lat, lon, h


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
class TestConstants:
    def test_wgs84_a(self):
        assert F.WGS84_A == 6378137.0

    def test_wgs84_f(self):
        assert F.WGS84_F == pytest.approx(1.0 / 298.257223563, rel=1e-15)

    def test_wgs84_b_numeric_value(self):
        # Spec quotes 6356752.314245 m (12 significant digits) -> 1e-6 m.
        assert F.WGS84_B == pytest.approx(6356752.314245, abs=1e-6)

    def test_wgs84_b_definition(self):
        assert F.WGS84_B == pytest.approx(A * (1.0 - FLAT), rel=1e-15)

    def test_wgs84_e2(self):
        assert F.WGS84_E2 == pytest.approx(FLAT * (2.0 - FLAT), rel=1e-14)
        # Published WGS-84 first eccentricity squared (NIMA TR8350.2), 15 digits.
        assert F.WGS84_E2 == pytest.approx(0.00669437999014, abs=1e-14)
        # Consistency with semi-minor axis.
        assert F.WGS84_E2 == pytest.approx(1.0 - (B / A) ** 2, rel=1e-12)

    def test_gm_earth(self):
        assert F.GM_EARTH == 3.986004418e14

    def test_j2_earth(self):
        assert F.J2_EARTH == 1.08262668e-3

    def test_r_earth_is_wgs84_a(self):
        assert F.R_EARTH == F.WGS84_A

    def test_omega_earth(self):
        assert F.OMEGA_EARTH == 7.2921150e-5

    def test_g0(self):
        assert F.G0 == 9.80665


# --------------------------------------------------------------------------- #
# lla_to_ecef
# --------------------------------------------------------------------------- #
class TestLlaToEcef:
    # Absolute tolerance 1e-8 m: coordinates are ~6.4e6 m, double resolution is
    # ~1e-9 m, so a handful of ulps of slack.
    TOL = 1e-8

    def test_origin_on_equator(self):
        r = F.lla_to_ecef(0.0, 0.0, 0.0)
        assert r.shape == (3,)
        np.testing.assert_allclose(r, [A, 0.0, 0.0], atol=self.TOL, rtol=0)

    def test_north_pole(self):
        r = F.lla_to_ecef(np.pi / 2, 0.0, 0.0)
        # x = N cos(pi/2) ~ 6.4e6 * 6.1e-17 = 3.9e-10 m, below TOL.
        np.testing.assert_allclose(r, [0.0, 0.0, B], atol=self.TOL, rtol=0)

    def test_south_pole(self):
        r = F.lla_to_ecef(-np.pi / 2, 1.0, 0.0)
        np.testing.assert_allclose(r, [0.0, 0.0, -B], atol=self.TOL, rtol=0)

    def test_lon_90_with_height(self):
        r = F.lla_to_ecef(0.0, np.pi / 2, 1000.0)
        # x = (A+1000) cos(pi/2) ~ 4e-10 m
        np.testing.assert_allclose(r, [0.0, A + 1000.0, 0.0], atol=self.TOL, rtol=0)

    def test_lon_180_equator(self):
        r = F.lla_to_ecef(0.0, np.pi, 0.0)
        np.testing.assert_allclose(r, [-A, 0.0, 0.0], atol=self.TOL, rtol=0)

    def test_height_along_pole_axis(self):
        r = F.lla_to_ecef(np.pi / 2, 0.3, 400e3)
        np.testing.assert_allclose(r, [0.0, 0.0, B + 400e3], atol=self.TOL, rtol=0)

    def test_mid_latitude_hand_computed(self):
        # Hand-written closed form, independent of the helper above.
        lat = np.deg2rad(45.5)
        lon = np.deg2rad(-73.6)
        h = 250.0
        n = A / np.sqrt(1.0 - E2 * np.sin(lat) ** 2)
        expected = np.array(
            [
                (n + h) * np.cos(lat) * np.cos(lon),
                (n + h) * np.cos(lat) * np.sin(lon),
                (n * (1.0 - E2) + h) * np.sin(lat),
            ]
        )
        r = F.lla_to_ecef(lat, lon, h)
        np.testing.assert_allclose(r, expected, atol=self.TOL, rtol=0)
        # Sanity on the hand calculation itself: radius is between b and a + h.
        assert B < np.linalg.norm(r) < A + h

    def test_random_points_vs_oracle(self):
        rng = np.random.default_rng(1001)
        lat, lon, h = _random_lla(rng, 200)
        r = F.lla_to_ecef(lat, lon, h)
        assert r.shape == (200, 3)
        np.testing.assert_allclose(
            r, _oracle_lla_to_ecef(lat, lon, h), atol=self.TOL, rtol=0
        )

    def test_broadcasting(self):
        lat = np.array([0.1, 0.2, 0.3])
        r = F.lla_to_ecef(lat, 0.5, 100.0)
        assert r.shape == (3, 3)
        np.testing.assert_allclose(
            r, _oracle_lla_to_ecef(lat, 0.5, 100.0), atol=self.TOL, rtol=0
        )

    def test_ellipsoid_surface_equation(self):
        # Points at h = 0 satisfy (x^2+y^2)/a^2 + z^2/b^2 = 1.
        rng = np.random.default_rng(1002)
        lat, lon, _ = _random_lla(rng, 100)
        r = F.lla_to_ecef(lat, lon, np.zeros(100))
        val = (r[:, 0] ** 2 + r[:, 1] ** 2) / A**2 + r[:, 2] ** 2 / B**2
        # 1e-14: relative rounding of squares of ~6e6 m quantities.
        np.testing.assert_allclose(val, 1.0, atol=1e-14, rtol=0)

    def test_inputs_not_mutated(self):
        lat = np.array([0.1, 0.2])
        lon = np.array([0.3, 0.4])
        h = np.array([10.0, 20.0])
        copies = [lat.copy(), lon.copy(), h.copy()]
        F.lla_to_ecef(lat, lon, h)
        for orig, cp in zip((lat, lon, h), copies):
            np.testing.assert_array_equal(orig, cp)


# --------------------------------------------------------------------------- #
# ecef_to_lla
# --------------------------------------------------------------------------- #
class TestEcefToLla:
    H_TOL = 1e-6  # m (spec)
    ANG_TOL = 1e-11  # rad (spec)

    def _check(self, lat, lon, h, r=None):
        if r is None:
            r = _oracle_lla_to_ecef(lat, lon, h)
        lat2, lon2, h2 = F.ecef_to_lla(r)
        lat2, lon2, h2 = (np.asarray(v, dtype=float) for v in (lat2, lon2, h2))
        assert np.max(np.abs(h2 - h)) <= self.H_TOL
        assert np.max(np.abs(lat2 - lat)) <= self.ANG_TOL
        assert np.max(np.abs(_wrap(lon2 - lon))) <= self.ANG_TOL
        return lat2, lon2, h2

    def test_round_trip_1000_random_points(self):
        rng = np.random.default_rng(20240601)
        lat, lon, h = _random_lla(rng, 1000)
        lat2, lon2, h2 = self._check(lat, lon, h)
        assert lat2.shape == (1000,)
        assert lon2.shape == (1000,)
        assert h2.shape == (1000,)
        assert np.all(lat2 >= -np.pi / 2 - 1e-12) and np.all(lat2 <= np.pi / 2 + 1e-12)
        assert np.all(lon2 >= -np.pi - 1e-12) and np.all(lon2 <= np.pi + 1e-12)

    def test_round_trip_with_implementation_forward(self):
        # Consistency between the two public functions as well.
        rng = np.random.default_rng(77)
        lat, lon, h = _random_lla(rng, 200)
        r = F.lla_to_ecef(lat, lon, h)
        lat2, lon2, h2 = F.ecef_to_lla(r)
        assert np.max(np.abs(np.asarray(h2) - h)) <= self.H_TOL
        assert np.max(np.abs(np.asarray(lat2) - lat)) <= self.ANG_TOL
        assert np.max(np.abs(_wrap(np.asarray(lon2) - lon))) <= self.ANG_TOL

    @pytest.mark.parametrize("sign", [+1.0, -1.0])
    @pytest.mark.parametrize("h", [0.0, 400e3, -500.0])
    def test_exact_poles_on_z_axis(self, sign, h):
        z = sign * (B + h)
        lat, lon, alt = F.ecef_to_lla(np.array([0.0, 0.0, z]))
        lat, lon, alt = float(lat), float(lon), float(alt)
        assert np.isfinite(lat) and np.isfinite(lon) and np.isfinite(alt)
        assert abs(lat - sign * np.pi / 2) <= self.ANG_TOL
        assert abs(alt - h) <= self.H_TOL  # longitude is undefined at a pole

    def test_poles_batched_in_one_call(self):
        r = np.array([[0.0, 0.0, B], [0.0, 0.0, -B], [0.0, 0.0, B + 400e3]])
        lat, lon, alt = F.ecef_to_lla(r)
        np.testing.assert_allclose(lat, [np.pi / 2, -np.pi / 2, np.pi / 2], atol=self.ANG_TOL, rtol=0)
        np.testing.assert_allclose(alt, [0.0, 0.0, 400e3], atol=self.H_TOL, rtol=0)
        assert np.all(np.isfinite(lon))

    @pytest.mark.parametrize("eps", [1e-3, 1e-6, 1e-9])
    @pytest.mark.parametrize("h", [0.0, 400e3])
    @pytest.mark.parametrize("sign", [+1.0, -1.0])
    def test_near_pole(self, eps, h, sign):
        lat = sign * (np.pi / 2 - eps)
        lon = 2.0
        self._check(np.array([lat]), np.array([lon]), np.array([h]))

    @pytest.mark.parametrize("lon", [0.0, 0.5, np.pi / 2, np.pi, -2.5])
    @pytest.mark.parametrize("h", [0.0, 400e3, -1e3, 1e6])
    def test_equator(self, lon, h):
        self._check(np.array([0.0]), np.array([lon]), np.array([h]))

    def test_equator_point_exact(self):
        lat, lon, alt = F.ecef_to_lla(np.array([A, 0.0, 0.0]))
        assert abs(float(lat)) <= self.ANG_TOL
        assert abs(float(lon)) <= self.ANG_TOL
        assert abs(float(alt)) <= self.H_TOL

    def test_400km_points(self):
        rng = np.random.default_rng(400)
        lat = rng.uniform(-np.deg2rad(51.6), np.deg2rad(51.6), 200)
        lon = rng.uniform(-np.pi, np.pi, 200)
        h = np.full(200, 400e3)
        self._check(lat, lon, h)

    def test_lon_branch_near_pi(self):
        lon = np.array([np.pi - 1e-9, -np.pi + 1e-9, np.pi])
        lat = np.array([0.3, -0.3, 0.1])
        self._check(lat, lon, np.array([10.0, 10.0, 10.0]))

    def test_single_vector_output_shapes(self):
        lat, lon, alt = F.ecef_to_lla(np.array([A + 400e3, 0.0, 0.0]))
        assert np.shape(lat) == () and np.shape(lon) == () and np.shape(alt) == ()
        assert float(alt) == pytest.approx(400e3, abs=self.H_TOL)

    def test_batched_leading_dims(self):
        rng = np.random.default_rng(5)
        lat, lon, h = _random_lla(rng, 12)
        r = _oracle_lla_to_ecef(lat, lon, h).reshape(3, 4, 3)
        lat2, lon2, h2 = F.ecef_to_lla(r)
        assert np.shape(lat2) == (3, 4)
        assert np.shape(lon2) == (3, 4)
        assert np.shape(h2) == (3, 4)
        np.testing.assert_allclose(np.ravel(h2), h, atol=self.H_TOL, rtol=0)

    def test_input_not_mutated(self):
        r = _oracle_lla_to_ecef(0.5, 0.5, 100.0)
        r0 = r.copy()
        F.ecef_to_lla(r)
        np.testing.assert_array_equal(r, r0)


# --------------------------------------------------------------------------- #
# dcm_ecef_to_ned
# --------------------------------------------------------------------------- #
class TestDcmEcefToNed:
    def test_origin_axes(self):
        c = F.dcm_ecef_to_ned(0.0, 0.0)
        assert c.shape == (3, 3)
        np.testing.assert_allclose(c[0], [0, 0, 1], atol=1e-15)  # N = +z_ecef
        np.testing.assert_allclose(c[1], [0, 1, 0], atol=1e-15)  # E = +y_ecef
        np.testing.assert_allclose(c[2], [-1, 0, 0], atol=1e-15)  # D = -x_ecef

    def test_north_pole_down_is_minus_z(self):
        c = F.dcm_ecef_to_ned(np.pi / 2, 0.7)
        # cos(pi/2) = 6e-17 -> atol 1e-15
        np.testing.assert_allclose(c[2], [0, 0, -1], atol=1e-15)
        # East stays horizontal, tangent to the parallel.
        np.testing.assert_allclose(c[1], [-np.sin(0.7), np.cos(0.7), 0], atol=1e-15)

    def test_south_pole_down_is_plus_z(self):
        c = F.dcm_ecef_to_ned(-np.pi / 2, 0.0)
        np.testing.assert_allclose(c[2], [0, 0, 1], atol=1e-15)

    def test_random_orthonormal_and_proper(self):
        rng = np.random.default_rng(11)
        lat, lon, _ = _random_lla(rng, 200)
        for la, lo in zip(lat, lon):
            c = F.dcm_ecef_to_ned(la, lo)
            assert _is_orthonormal(c)
            assert np.linalg.det(c) == pytest.approx(1.0, abs=1e-14)

    def test_random_vs_textbook_matrix(self):
        rng = np.random.default_rng(12)
        lat, lon, _ = _random_lla(rng, 100)
        for la, lo in zip(lat, lon):
            np.testing.assert_allclose(
                F.dcm_ecef_to_ned(la, lo), _oracle_c_ecef_ned(la, lo), atol=1e-14
            )

    def test_right_handed_triad(self):
        c = F.dcm_ecef_to_ned(0.6, -1.2)
        np.testing.assert_allclose(np.cross(c[0], c[1]), c[2], atol=1e-14)

    def test_down_is_opposite_to_ellipsoid_normal(self):
        # Geodetic normal direction = gradient of the ellipsoid surface; compare
        # with the finite-difference direction of increasing height.
        lat, lon = 0.9, 0.4
        up = (
            _oracle_lla_to_ecef(lat, lon, 1.0) - _oracle_lla_to_ecef(lat, lon, 0.0)
        )
        np.testing.assert_allclose(F.dcm_ecef_to_ned(lat, lon)[2], -up, atol=1e-9)

    def test_batched_shape(self):
        lat = np.linspace(-1.0, 1.0, 5)
        lon = np.linspace(-3.0, 3.0, 5)
        c = F.dcm_ecef_to_ned(lat, lon)
        assert c.shape == (5, 3, 3)
        for i in range(5):
            np.testing.assert_allclose(c[i], _oracle_c_ecef_ned(lat[i], lon[i]), atol=1e-14)


    def test_ned_aligned_body_quaternion_z_is_down(self):
        # A body aligned with NED has body-to-ECEF quaternion from_dcm(C.T) (same
        # rule as LVLH). Body z is "down": minus the geodetic up vector, written
        # out here in closed form: up = [cos(lat)cos(lon), cos(lat)sin(lon), sin(lat)].
        rng = np.random.default_rng(69)
        lat = rng.uniform(-np.pi / 2 + 0.01, np.pi / 2 - 0.01, 40)
        lon = rng.uniform(-np.pi, np.pi, 40)
        up = np.stack(
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
        )
        east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], axis=-1)
        north = np.cross(up, east)
        q = Q.from_dcm(np.swapaxes(F.dcm_ecef_to_ned(lat, lon), -1, -2))   # batched
        np.testing.assert_allclose(Q.rotate(q, [0, 0, 1]), -up, atol=1e-12)
        np.testing.assert_allclose(Q.rotate(q, [0, 1, 0]), east, atol=1e-12)
        np.testing.assert_allclose(Q.rotate(q, [1, 0, 0]), north, atol=1e-12)
        # scalar call agrees
        q0 = Q.from_dcm(F.dcm_ecef_to_ned(lat[0], lon[0]).T)
        np.testing.assert_allclose(Q.rotate(q0, [0, 0, 1]), -up[0], atol=1e-12)

    def test_ned_transposed_dcm_gives_wrong_quaternion(self):
        lat, lon = 0.6, -1.2
        up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
        q_wrong = Q.from_dcm(F.dcm_ecef_to_ned(lat, lon))
        assert np.linalg.norm(Q.rotate(q_wrong, [0, 0, 1]) + up) > 0.1


# --------------------------------------------------------------------------- #
# ecef_to_ned / ned_to_ecef / lla_to_ned
# --------------------------------------------------------------------------- #
class TestNed:
    TOL = 1e-6  # m (spec/task); actual error ~1e-9 m (ulp of 6.4e6 m)

    def _ref(self):
        lat0, lon0, alt0 = np.deg2rad(37.4), np.deg2rad(-122.1), 30.0
        return lat0, lon0, alt0, _oracle_lla_to_ecef(lat0, lon0, alt0)

    def test_reference_point_is_origin(self):
        lat0, lon0, alt0, r0 = self._ref()
        n = F.ecef_to_ned(r0, lat0, lon0, alt0)
        assert n.shape == (3,)
        np.testing.assert_allclose(n, 0.0, atol=self.TOL)

    @pytest.mark.parametrize(
        "row,expected", [(0, [1, 0, 0]), (1, [0, 1, 0]), (2, [0, 0, 1])]
    )
    def test_one_metre_along_each_axis(self, row, expected):
        lat0, lon0, alt0, r0 = self._ref()
        axis_ecef = _oracle_c_ecef_ned(lat0, lon0)[row]  # N/E/D unit vector in ECEF
        n = F.ecef_to_ned(r0 + 1.0 * axis_ecef, lat0, lon0, alt0)
        np.testing.assert_allclose(n, expected, atol=self.TOL)

    def test_ned_to_ecef_unit_vectors(self):
        lat0, lon0, alt0, r0 = self._ref()
        c = _oracle_c_ecef_ned(lat0, lon0)
        for i, e in enumerate(np.eye(3)):
            r = F.ned_to_ecef(e, lat0, lon0, alt0)
            np.testing.assert_allclose(r, r0 + c[i], atol=self.TOL)

    def test_100m_directly_above(self):
        lat0, lon0, alt0, _ = self._ref()
        above = _oracle_lla_to_ecef(lat0, lon0, alt0 + 100.0)
        n = F.ecef_to_ned(above, lat0, lon0, alt0)
        np.testing.assert_allclose(n, [0.0, 0.0, -100.0], atol=self.TOL)

    def test_100m_directly_above_at_pole_and_equator(self):
        for lat0, lon0 in [(np.pi / 2, 0.0), (0.0, 1.0), (-np.pi / 2, 0.0)]:
            above = _oracle_lla_to_ecef(lat0, lon0, 100.0)
            n = F.ecef_to_ned(above, lat0, lon0, 0.0)
            np.testing.assert_allclose(n, [0.0, 0.0, -100.0], atol=self.TOL)

    def test_round_trip_ecef_ned_ecef(self):
        rng = np.random.default_rng(31)
        lat0, lon0, alt0 = _random_lla(rng, 20, h_lo=-500, h_hi=5000)
        for la, lo, al in zip(lat0, lon0, alt0):
            r = _oracle_lla_to_ecef(la, lo, al) + rng.normal(scale=5e4, size=3)
            n = F.ecef_to_ned(r, la, lo, al)
            np.testing.assert_allclose(F.ned_to_ecef(n, la, lo, al), r, atol=self.TOL)

    def test_round_trip_ned_ecef_ned(self):
        rng = np.random.default_rng(32)
        lat0, lon0, alt0 = _random_lla(rng, 20, h_lo=-500, h_hi=5000)
        for la, lo, al in zip(lat0, lon0, alt0):
            n = rng.normal(scale=1e4, size=3)
            r = F.ned_to_ecef(n, la, lo, al)
            np.testing.assert_allclose(F.ecef_to_ned(r, la, lo, al), n, atol=self.TOL)

    def test_ned_preserves_distance_from_reference(self):
        lat0, lon0, alt0, r0 = self._ref()
        r = r0 + np.array([1234.0, -4321.0, 777.0])
        n = F.ecef_to_ned(r, lat0, lon0, alt0)
        assert np.linalg.norm(n) == pytest.approx(np.linalg.norm(r - r0), abs=self.TOL)

    def test_batched_points(self):
        lat0, lon0, alt0, r0 = self._ref()
        rng = np.random.default_rng(33)
        offs = rng.normal(scale=1e3, size=(50, 3))
        n = F.ecef_to_ned(r0 + offs, lat0, lon0, alt0)
        assert n.shape == (50, 3)
        expected = offs @ _oracle_c_ecef_ned(lat0, lon0).T
        np.testing.assert_allclose(n, expected, atol=self.TOL)
        back = F.ned_to_ecef(n, lat0, lon0, alt0)
        assert back.shape == (50, 3)
        np.testing.assert_allclose(back, r0 + offs, atol=self.TOL)

    def test_lla_to_ned_consistency(self):
        lat0, lon0, alt0, _ = self._ref()
        rng = np.random.default_rng(34)
        for _ in range(20):
            lat = lat0 + rng.uniform(-0.05, 0.05)
            lon = lon0 + rng.uniform(-0.05, 0.05)
            alt = rng.uniform(0, 3000)
            expected = _oracle_c_ecef_ned(lat0, lon0) @ (
                _oracle_lla_to_ecef(lat, lon, alt) - _oracle_lla_to_ecef(lat0, lon0, alt0)
            )
            got = F.lla_to_ned(lat, lon, alt, lat0, lon0, alt0)
            np.testing.assert_allclose(got, expected, atol=self.TOL)
            np.testing.assert_allclose(
                got,
                F.ecef_to_ned(F.lla_to_ecef(lat, lon, alt), lat0, lon0, alt0),
                atol=self.TOL,
            )

    def test_lla_to_ned_same_point_and_height_only(self):
        lat0, lon0, alt0, _ = self._ref()
        np.testing.assert_allclose(
            F.lla_to_ned(lat0, lon0, alt0, lat0, lon0, alt0), 0.0, atol=self.TOL
        )
        np.testing.assert_allclose(
            F.lla_to_ned(lat0, lon0, alt0 + 50.0, lat0, lon0, alt0),
            [0.0, 0.0, -50.0],
            atol=self.TOL,
        )

    def test_lla_to_ned_small_angular_offsets_match_local_radii(self):
        # Independent physical check: for a small offset the north/east distances
        # are M*dlat and N*cos(lat)*dlon with the meridian/prime-vertical radii.
        lat0, lon0, alt0 = np.deg2rad(52.0), np.deg2rad(5.0), 0.0
        dlat = dlon = 1e-5  # ~ 64 m
        s2 = np.sin(lat0) ** 2
        n_pv = A / np.sqrt(1 - E2 * s2)
        m_mer = A * (1 - E2) / (1 - E2 * s2) ** 1.5
        got = F.lla_to_ned(lat0 + dlat, lon0 + dlon, alt0, lat0, lon0, alt0)
        # Second-order terms: ~ (dlat)^2 * R ~ 1e-10 * 6e6 = 6e-4 m -> 5e-3 m slack.
        assert got[0] == pytest.approx(m_mer * dlat, abs=5e-3)
        assert got[1] == pytest.approx(n_pv * np.cos(lat0) * dlon, abs=5e-3)
        # Down: surface curves away from the tangent plane, a few 1e-4 m.
        assert abs(got[2]) < 5e-3

    def test_lla_to_ned_batched(self):
        lat0, lon0, alt0, _ = self._ref()
        lat = lat0 + np.linspace(0, 0.01, 7)
        lon = lon0 + np.linspace(0, 0.01, 7)
        alt = np.linspace(0, 600, 7)
        got = F.lla_to_ned(lat, lon, alt, lat0, lon0, alt0)
        assert got.shape == (7, 3)
        for i in range(7):
            np.testing.assert_allclose(
                got[i], F.lla_to_ned(lat[i], lon[i], alt[i], lat0, lon0, alt0), atol=self.TOL
            )


# --------------------------------------------------------------------------- #
# julian_date
# --------------------------------------------------------------------------- #
class TestJulianDate:
    def test_j2000(self):
        jd = F.julian_date(dt.datetime(2000, 1, 1, 12, 0, 0))
        assert isinstance(jd, (float, np.floating))
        assert jd == 2451545.0

    def test_vallado_example_3_5(self):
        jd = F.julian_date(dt.datetime(1992, 8, 20, 12, 14, 0))
        assert jd == pytest.approx(2448855.009722, abs=1e-6)

    def test_unix_epoch(self):
        assert F.julian_date(dt.datetime(1970, 1, 1)) == pytest.approx(2440587.5, abs=1e-9)

    def test_leap_day(self):
        assert F.julian_date(dt.datetime(2024, 2, 29, 12)) == pytest.approx(2460370.0, abs=1e-9)
        assert F.julian_date(dt.datetime(2024, 3, 1)) == pytest.approx(2460370.5, abs=1e-9)

    def test_naive_is_utc(self):
        naive = dt.datetime(2026, 3, 4, 5, 6, 7)
        aware = naive.replace(tzinfo=dt.timezone.utc)
        assert F.julian_date(naive) == F.julian_date(aware)

    def test_microseconds_half_second(self):
        base = F.julian_date(dt.datetime(2026, 1, 1, 0, 0, 0))
        half = F.julian_date(dt.datetime(2026, 1, 1, 0, 0, 0, 500000))
        # 0.5 s = 5.787e-6 day. A JD near 2.46e6 has float spacing 4.7e-10 day
        # (40 us), so allow 1e-9 day.
        assert half - base == pytest.approx(0.5 / 86400.0, abs=1e-9)

    def test_microseconds_single_digit_fraction(self):
        base = F.julian_date(dt.datetime(2026, 1, 1, 0, 0, 0))
        jd = F.julian_date(dt.datetime(2026, 1, 1, 0, 0, 0, 123456))
        assert jd - base == pytest.approx(0.123456 / 86400.0, abs=1e-9)
        assert jd > base

    def test_microseconds_are_not_dropped_close_to_oracle(self):
        d = dt.datetime(2031, 7, 9, 13, 14, 15, 987654)
        assert F.julian_date(d) == pytest.approx(_oracle_jd(d), abs=1e-9)

    def test_timezone_aware_converted_to_utc(self):
        tz = dt.timezone(dt.timedelta(hours=5))
        aware = dt.datetime(2000, 1, 1, 17, 0, 0, tzinfo=tz)  # = 12:00 UTC
        assert F.julian_date(aware) == 2451545.0

    def test_timezone_aware_negative_offset_crosses_midnight(self):
        tz = dt.timezone(dt.timedelta(hours=-8))
        aware = dt.datetime(2000, 1, 1, 20, 30, 0, tzinfo=tz)  # = 2000-01-02 04:30 UTC
        expected = _oracle_jd(dt.datetime(2000, 1, 2, 4, 30, 0))
        assert F.julian_date(aware) == pytest.approx(expected, abs=1e-9)
        assert expected == pytest.approx(2451545.0 + 16.5 / 24.0, abs=1e-12)

    def test_random_datetimes_vs_ordinal_oracle(self):
        rng = np.random.default_rng(2000)
        for _ in range(100):
            d = dt.datetime(1950, 1, 1) + dt.timedelta(
                seconds=float(rng.uniform(0, 120 * 365.25 * 86400))
            )
            d = d.replace(microsecond=int(rng.integers(0, 1_000_000)))
            # 1e-9 day = 86 us, ~2 float spacings at JD ~ 2.4e6.
            assert F.julian_date(d) == pytest.approx(_oracle_jd(d), abs=1e-9)

    def test_monotonic_one_day_step(self):
        d0 = dt.datetime(2026, 5, 17, 3, 0, 0)
        d1 = d0 + dt.timedelta(days=1)
        assert F.julian_date(d1) - F.julian_date(d0) == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# gmst
# --------------------------------------------------------------------------- #
class TestGmst:
    def test_j2000_value(self):
        assert np.rad2deg(F.gmst(JD_J2000)) == pytest.approx(280.46061837, abs=1e-6)

    def test_vallado_example_3_5(self):
        # JD(UT1) of 1992-08-20 12:14:00 = 2448854.5 + 734/1440 (exact fraction).
        jd = 2448854.5 + 734.0 / 1440.0
        assert np.rad2deg(F.gmst(jd)) == pytest.approx(152.578787886, abs=1e-6)

    def test_vallado_example_via_julian_date(self):
        jd = F.julian_date(dt.datetime(1992, 8, 20, 12, 14, 0))
        # jd float spacing 4.7e-10 day * 360.98 deg/day = 1.7e-7 deg < 1e-6.
        assert np.rad2deg(F.gmst(jd)) == pytest.approx(152.578787886, abs=1e-6)

    def test_scalar_returns_scalar(self):
        out = F.gmst(JD_J2000 + 123.456)
        assert np.ndim(out) == 0

    def test_range_zero_to_two_pi(self):
        rng = np.random.default_rng(7)
        jd = rng.uniform(JD_J2000 - 36525.0, JD_J2000 + 36525.0, 5000)  # +-100 yr
        th = F.gmst(jd)
        assert th.shape == jd.shape
        assert np.all(th >= 0.0)
        assert np.all(th < TWO_PI)

    def test_array_input_matches_scalar_and_spec_formula(self):
        rng = np.random.default_rng(8)
        jd = rng.uniform(JD_J2000 - 20000.0, JD_J2000 + 20000.0, 50)
        th = F.gmst(jd)
        assert th.shape == (50,)
        for i in range(0, 50, 7):
            assert float(F.gmst(jd[i])) == pytest.approx(th[i], abs=1e-12)
        # Spec formula: sec reaches ~3e9 s at |T|=0.5; double rounding ~ 5e-7 s
        # -> 3e-11 rad; allow 1e-9 rad on the wrapped difference.
        assert np.max(np.abs(_wrap(th - _oracle_gmst_formula(jd)))) < 1e-9

    def test_2d_array_shape_preserved(self):
        jd = JD_J2000 + np.arange(6.0).reshape(2, 3)
        assert F.gmst(jd).shape == (2, 3)

    def test_one_sidereal_day_returns_to_same_angle(self):
        rng = np.random.default_rng(9)
        for jd0 in rng.uniform(JD_J2000 - 10000.0, JD_J2000 + 10000.0, 20):
            jd1 = jd0 + SIDEREAL_DAY_S / 86400.0
            d = _wrap(F.gmst(jd1) - F.gmst(jd0))
            # Sidereal-day constant is quoted to 5e-5 s -> 3.6e-9 rad; JD float
            # spacing 4.7e-10 day -> 3e-9 rad. 1e-6 rad leaves wide margin.
            assert abs(d) < 1e-6

    def test_mean_rate_matches_omega_earth(self):
        jd0 = JD_J2000 + 5000.0
        dt_s = 3600.0
        jd1 = jd0 + dt_s / 86400.0
        # unwrapped advance over 1 h is 0.26 rad < pi, so a wrapped diff is fine.
        rate = float(_wrap(F.gmst(jd1) - F.gmst(jd0))) / dt_s
        # True sidereal rate is 7.2921159e-5; OMEGA_EARTH (7.2921150e-5) differs by
        # 1.2e-7 relative; JD float spacing adds ~1e-8. Use 1e-6 relative.
        assert rate == pytest.approx(F.OMEGA_EARTH, rel=1e-6)

    def test_increasing_over_one_day_fraction(self):
        jd = JD_J2000 + np.linspace(0.0, 0.2, 50)
        d = np.diff(np.unwrap(F.gmst(jd)))
        assert np.all(d > 0)


# --------------------------------------------------------------------------- #
# dcm_eci_to_ecef / eci_to_ecef / ecef_to_eci
# --------------------------------------------------------------------------- #
class TestEciEcef:
    def _passive_rz(self, th):
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])

    def test_matches_spec_form(self):
        rng = np.random.default_rng(21)
        for jd in rng.uniform(JD_J2000 - 5000, JD_J2000 + 15000, 20):
            th = _oracle_gmst_formula(jd)
            c = F.dcm_eci_to_ecef(jd)
            assert c.shape == (3, 3)
            np.testing.assert_allclose(c, self._passive_rz(th), atol=1e-9)

    def test_at_j2000_explicit_angle(self):
        th = np.deg2rad(280.46061837)
        np.testing.assert_allclose(
            F.dcm_eci_to_ecef(JD_J2000), self._passive_rz(th), atol=1e-9
        )

    def test_matches_scipy_rotation(self):
        # Passive Rz(theta) == active rotation by -theta about z.
        jd = JD_J2000 + 1234.5678
        th = _oracle_gmst_formula(jd)
        expected = Rotation.from_euler("z", -th).as_matrix()
        np.testing.assert_allclose(F.dcm_eci_to_ecef(jd), expected, atol=1e-9)

    def test_orthonormal_and_proper(self):
        rng = np.random.default_rng(22)
        for jd in rng.uniform(JD_J2000 - 5000, JD_J2000 + 15000, 20):
            c = F.dcm_eci_to_ecef(jd)
            assert _is_orthonormal(c)
            assert np.linalg.det(c) == pytest.approx(1.0, abs=1e-14)

    def test_z_vector_unchanged(self):
        out = F.eci_to_ecef(np.array([0.0, 0.0, 7.0e6]), JD_J2000 + 100.3)
        np.testing.assert_allclose(out, [0.0, 0.0, 7.0e6], atol=1e-9)

    def test_eci_x_axis_direction_at_j2000(self):
        th = np.deg2rad(280.46061837)
        out = F.eci_to_ecef(np.array([1.0, 0.0, 0.0]), JD_J2000)
        np.testing.assert_allclose(out, [np.cos(th), -np.sin(th), 0.0], atol=1e-9)

    def test_greenwich_meridian_direction_in_eci(self):
        # ECEF x-axis (Greenwich meridian) is at right ascension = GMST in ECI.
        jd = JD_J2000 + 321.123
        th = _oracle_gmst_formula(jd)
        out = F.ecef_to_eci(np.array([1.0, 0.0, 0.0]), jd)
        np.testing.assert_allclose(out, [np.cos(th), np.sin(th), 0.0], atol=1e-9)

    def test_round_trip(self):
        rng = np.random.default_rng(23)
        for jd in rng.uniform(JD_J2000 - 5000, JD_J2000 + 15000, 10):
            r = rng.normal(scale=7e6, size=3)
            np.testing.assert_allclose(
                F.ecef_to_eci(F.eci_to_ecef(r, jd), jd), r, atol=1e-8
            )
            np.testing.assert_allclose(
                F.eci_to_ecef(F.ecef_to_eci(r, jd), jd), r, atol=1e-8
            )

    def test_norm_preserved(self):
        r = np.array([4.0e6, -5.0e6, 3.0e6])
        out = F.eci_to_ecef(r, JD_J2000 + 77.7)
        assert np.linalg.norm(out) == pytest.approx(np.linalg.norm(r), rel=1e-14)

    def test_eci_to_ecef_matches_dcm(self):
        jd = JD_J2000 + 4242.4242
        r = np.array([1.0e6, 2.0e6, -3.0e6])
        np.testing.assert_allclose(
            F.eci_to_ecef(r, jd), F.dcm_eci_to_ecef(jd) @ r, atol=1e-8
        )

    def test_batch_jd_dcm(self):
        jd = JD_J2000 + np.linspace(0.0, 1.0, 9)
        c = F.dcm_eci_to_ecef(jd)
        assert c.shape == (9, 3, 3)
        for i in range(9):
            np.testing.assert_allclose(c[i], F.dcm_eci_to_ecef(jd[i]), atol=1e-12)
            np.testing.assert_allclose(c[i], self._passive_rz(_oracle_gmst_formula(jd[i])), atol=1e-9)

    def test_batch_vectors_and_jd(self):
        rng = np.random.default_rng(24)
        jd = JD_J2000 + np.linspace(0.0, 1.0, 6)
        r = rng.normal(scale=7e6, size=(6, 3))
        out = F.eci_to_ecef(r, jd)
        assert out.shape == (6, 3)
        for i in range(6):
            np.testing.assert_allclose(out[i], F.eci_to_ecef(r[i], jd[i]), atol=1e-8)
        np.testing.assert_allclose(F.ecef_to_eci(out, jd), r, atol=1e-8)

    def test_earth_rotation_rate_visible_in_ecef(self):
        # A fixed inertial vector appears to rotate at -OMEGA_EARTH in ECEF.
        jd0 = JD_J2000 + 100.0
        dt_s = 600.0
        r = np.array([1.0, 0.0, 0.0])
        a0 = F.eci_to_ecef(r, jd0)
        a1 = F.eci_to_ecef(r, jd0 + dt_s / 86400.0)
        ang = np.arctan2(a1[1], a1[0]) - np.arctan2(a0[1], a0[0])
        # relative 1e-6 as in the mean-rate test above
        assert float(_wrap(ang)) / dt_s == pytest.approx(-F.OMEGA_EARTH, rel=1e-6)

    def test_inputs_not_mutated(self):
        r = np.array([1.0, 2.0, 3.0])
        r0 = r.copy()
        F.eci_to_ecef(r, JD_J2000)
        F.ecef_to_eci(r, JD_J2000)
        np.testing.assert_array_equal(r, r0)


# --------------------------------------------------------------------------- #
# dcm_eci_to_lvlh
# --------------------------------------------------------------------------- #
class TestLvlh:
    R0 = 6778137.0  # 400 km orbit radius
    V0 = np.sqrt(3.986004418e14 / 6778137.0)

    def test_equatorial_circular_example(self):
        c = F.dcm_eci_to_lvlh(np.array([self.R0, 0, 0]), np.array([0, self.V0, 0]))
        assert c.shape == (3, 3)
        np.testing.assert_allclose(c[0], [0, 1, 0], atol=1e-15)  # x ~ velocity
        np.testing.assert_allclose(c[1], [0, 0, -1], atol=1e-15)  # y = -orbit normal
        np.testing.assert_allclose(c[2], [-1, 0, 0], atol=1e-15)  # z = nadir

    def test_retrograde_flips_y_and_x(self):
        c = F.dcm_eci_to_lvlh(np.array([self.R0, 0, 0]), np.array([0, -self.V0, 0]))
        np.testing.assert_allclose(c[0], [0, -1, 0], atol=1e-15)
        np.testing.assert_allclose(c[1], [0, 0, 1], atol=1e-15)
        np.testing.assert_allclose(c[2], [-1, 0, 0], atol=1e-15)

    def _random_orbit_states(self, rng, n, circular):
        rhat = rng.normal(size=(n, 3))
        rhat /= np.linalg.norm(rhat, axis=1, keepdims=True)
        radius = rng.uniform(6.6e6, 8.0e6, n)
        r = rhat * radius[:, None]
        nrm = rng.normal(size=(n, 3))
        nrm -= np.sum(nrm * rhat, axis=1, keepdims=True) * rhat  # make n perp r
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        vhat = np.cross(nrm, rhat)  # unit, perpendicular to r
        speed = np.sqrt(F.GM_EARTH / radius)
        v = vhat * speed[:, None]
        if not circular:
            # add a radial component (flight-path angle up to ~ +-10 deg)
            gamma = rng.uniform(-0.17, 0.17, n)
            v = (
                np.cos(gamma)[:, None] * v
                + np.sin(gamma)[:, None] * speed[:, None] * rhat
            )
        return r, v

    @pytest.mark.parametrize("circular", [True, False])
    def test_orthonormal_proper_random_orbits(self, circular):
        rng = np.random.default_rng(41 + int(circular))
        r, v = self._random_orbit_states(rng, 200, circular)
        for ri, vi in zip(r, v):
            c = F.dcm_eci_to_lvlh(ri, vi)
            assert _is_orthonormal(c)
            assert np.linalg.det(c) == pytest.approx(1.0, abs=1e-14)

    @pytest.mark.parametrize("circular", [True, False])
    def test_axis_definitions(self, circular):
        rng = np.random.default_rng(51 + int(circular))
        r, v = self._random_orbit_states(rng, 100, circular)
        for ri, vi in zip(r, v):
            c = F.dcm_eci_to_lvlh(ri, vi)
            x, y, z = c
            rhat = ri / np.linalg.norm(ri)
            h = np.cross(ri, vi)
            hhat = h / np.linalg.norm(h)
            assert float(z @ rhat) == pytest.approx(-1.0, abs=1e-14)
            np.testing.assert_allclose(z, -rhat, atol=1e-14)
            assert float(y @ hhat) == pytest.approx(-1.0, abs=1e-14)
            np.testing.assert_allclose(y, -hhat, atol=1e-14)
            np.testing.assert_allclose(x, np.cross(y, z), atol=1e-14)

    def test_x_matches_velocity_for_circular_orbit(self):
        rng = np.random.default_rng(61)
        r, v = self._random_orbit_states(rng, 100, circular=True)
        for ri, vi in zip(r, v):
            x = F.dcm_eci_to_lvlh(ri, vi)[0]
            np.testing.assert_allclose(x, vi / np.linalg.norm(vi), atol=1e-14)

    def test_x_has_positive_velocity_component_for_eccentric_orbit(self):
        rng = np.random.default_rng(62)
        r, v = self._random_orbit_states(rng, 100, circular=False)
        for ri, vi in zip(r, v):
            x = F.dcm_eci_to_lvlh(ri, vi)[0]
            assert float(x @ vi) > 0.0

    def test_maps_position_to_minus_z_times_radius(self):
        # In LVLH the position vector is [0, 0, -|r|].
        rng = np.random.default_rng(63)
        r, v = self._random_orbit_states(rng, 20, circular=False)
        for ri, vi in zip(r, v):
            out = F.dcm_eci_to_lvlh(ri, vi) @ ri
            # 1e-8 m: |r| ~ 7e6 m with ~1e-16 relative rounding.
            np.testing.assert_allclose(out, [0, 0, -np.linalg.norm(ri)], atol=1e-8)

    def test_orbit_normal_maps_to_minus_y(self):
        rng = np.random.default_rng(64)
        r, v = self._random_orbit_states(rng, 20, circular=False)
        for ri, vi in zip(r, v):
            h = np.cross(ri, vi)
            out = F.dcm_eci_to_lvlh(ri, vi) @ (h / np.linalg.norm(h))
            np.testing.assert_allclose(out, [0, -1, 0], atol=1e-14)

    def test_inclined_example_51_6_deg(self):
        inc = np.deg2rad(51.6)
        # Satellite at the ascending node: r along +x, v tilted by inc about x.
        r = np.array([self.R0, 0.0, 0.0])
        v = self.V0 * np.array([0.0, np.cos(inc), np.sin(inc)])
        c = F.dcm_eci_to_lvlh(r, v)
        np.testing.assert_allclose(c[2], [-1, 0, 0], atol=1e-15)
        np.testing.assert_allclose(c[0], [0, np.cos(inc), np.sin(inc)], atol=1e-15)
        np.testing.assert_allclose(c[1], [0, np.sin(inc), -np.cos(inc)], atol=1e-15)

    def test_scale_invariance(self):
        r = np.array([7.0e6, 1.0e6, -2.0e6])
        v = np.array([-500.0, 7000.0, 2000.0])
        np.testing.assert_allclose(
            F.dcm_eci_to_lvlh(r, v), F.dcm_eci_to_lvlh(2.0 * r, 3.0 * v), atol=1e-14
        )

    def test_batched(self):
        rng = np.random.default_rng(65)
        r, v = self._random_orbit_states(rng, 10, circular=False)
        c = F.dcm_eci_to_lvlh(r, v)
        assert c.shape == (10, 3, 3)
        for i in range(10):
            np.testing.assert_allclose(c[i], F.dcm_eci_to_lvlh(r[i], v[i]), atol=1e-14)
        assert np.max(np.abs(np.linalg.det(c) - 1.0)) < 1e-14

    # ---- link to the quaternion convention (body-to-ECI = from_dcm(C.T)) ----
    def test_lvlh_aligned_body_quaternion_points_nadir(self):
        # A body whose axes coincide with LVLH has body-to-ECI quaternion
        # from_dcm(C.T) (spec). Its body z (nadir) must point to -r_hat, its body y
        # to the negative orbit normal, and (circular orbit) body x to v_hat.
        rng = np.random.default_rng(66)
        r, v = self._random_orbit_states(rng, 50, circular=True)
        rhat = r / np.linalg.norm(r, axis=1, keepdims=True)
        vhat = v / np.linalg.norm(v, axis=1, keepdims=True)
        hhat = np.cross(r, v)
        hhat /= np.linalg.norm(hhat, axis=1, keepdims=True)
        for i in range(50):
            q = Q.from_dcm(F.dcm_eci_to_lvlh(r[i], v[i]).T)
            # Tolerance: float64 algebra on O(1) unit vectors (inputs are ~1e7, ~1e4).
            np.testing.assert_allclose(Q.rotate(q, [0, 0, 1]), -rhat[i], atol=1e-12)
            np.testing.assert_allclose(Q.rotate(q, [1, 0, 0]), vhat[i], atol=1e-12)
            np.testing.assert_allclose(Q.rotate(q, [0, 1, 0]), -hhat[i], atol=1e-12)

    def test_lvlh_quaternion_batched(self):
        rng = np.random.default_rng(67)
        r, v = self._random_orbit_states(rng, 30, circular=True)
        rhat = r / np.linalg.norm(r, axis=1, keepdims=True)
        c = F.dcm_eci_to_lvlh(r, v)                        # (30, 3, 3)
        q = Q.from_dcm(np.swapaxes(c, -1, -2))
        np.testing.assert_allclose(Q.rotate(q, [0.0, 0.0, 1.0]), -rhat, atol=1e-12)

    def test_lvlh_transposed_dcm_gives_wrong_quaternion(self):
        # The classic mistake from_dcm(C) (instead of from_dcm(C.T)) is the inverse
        # rotation, so body z does NOT point nadir for a generic orbit.
        rng = np.random.default_rng(68)
        r, v = self._random_orbit_states(rng, 30, circular=True)
        for i in range(30):
            rhat = r[i] / np.linalg.norm(r[i])
            q_wrong = Q.from_dcm(F.dcm_eci_to_lvlh(r[i], v[i]))
            z_body = Q.rotate(q_wrong, [0, 0, 1])
            # angle between the wrong body-z and nadir; >= 0.05 rad for these generic orbits
            ang = np.arccos(np.clip(z_body @ -rhat, -1.0, 1.0))
            assert ang > 0.05

    def test_inputs_not_mutated(self):
        r = np.array([self.R0, 0.0, 0.0])
        v = np.array([0.0, self.V0, 0.0])
        r0, v0 = r.copy(), v.copy()
        F.dcm_eci_to_lvlh(r, v)
        np.testing.assert_array_equal(r, r0)
        np.testing.assert_array_equal(v, v0)


# --------------------------------------------------------------------------- #
# normal_gravity (WGS-84 Somigliana + free-air), docs/specs/sensor_fusion_data.md section 6
# --------------------------------------------------------------------------- #
GAMMA_E = 9.7803253359      # WGS-84 equatorial normal gravity, m/s^2 (NGA TR8350.2)
GAMMA_P = 9.8321849378      # WGS-84 polar normal gravity, m/s^2
FREE_AIR = -3.086e-6        # free-air gradient, (m/s^2)/m


def _oracle_somigliana(lat):
    """Independent (a, b, gamma_e, gamma_p) form of the Somigliana formula (Torge, Geodesy;
    Hofmann-Wellenhof & Moritz, Physical Geodesy eq. 2-146) - not the k/e^2 form."""
    s2, c2 = np.sin(lat) ** 2, np.cos(lat) ** 2
    return (A * GAMMA_E * c2 + B * GAMMA_P * s2) / np.sqrt(A * A * c2 + B * B * s2)


class TestNormalGravity:
    def test_equator_and_pole_sea_level(self):
        # Defining WGS-84 constants -> exact algebra, 1e-9 as specified.
        assert F.normal_gravity(0.0, 0.0) == pytest.approx(GAMMA_E, abs=1e-9)
        assert F.normal_gravity(np.pi / 2, 0.0) == pytest.approx(GAMMA_P, abs=1e-9)
        assert F.normal_gravity(-np.pi / 2, 0.0) == pytest.approx(GAMMA_P, abs=1e-9)

    def test_alt_defaults_to_sea_level(self):
        assert F.normal_gravity(0.3) == pytest.approx(F.normal_gravity(0.3, 0.0), abs=0.0)

    def test_mid_latitude_textbook_value(self):
        # 45 deg geodetic: 9.80619... m/s^2 (Somigliana closed form); +-1e-5 per the spec.
        assert F.normal_gravity(np.radians(45.0), 0.0) == pytest.approx(9.80619, abs=1e-5)

    def test_matches_independent_somigliana_form_over_latitude(self):
        # The two algebraic forms are identical up to the 1e-12-level rounding of the
        # published constants (k, e^2 are derived from gamma_p/gamma_e, a, b): 1e-9 m/s^2.
        lat = np.radians(np.linspace(-90.0, 90.0, 181))
        np.testing.assert_allclose(F.normal_gravity(lat, 0.0), _oracle_somigliana(lat), rtol=0, atol=1e-9)

    def test_symmetric_in_latitude_and_monotonic_to_pole(self):
        lat = np.radians(np.linspace(0.0, 90.0, 91))
        g = F.normal_gravity(lat, 0.0)
        np.testing.assert_allclose(F.normal_gravity(-lat, 0.0), g, rtol=0, atol=1e-15)
        assert np.all(np.diff(g) > 0.0)

    def test_free_air_gradient_finite_difference(self):
        h = np.linspace(0.0, 5000.0, 51)
        for lat_deg in (0.0, 27.5, 45.0, 80.0):
            g = F.normal_gravity(np.radians(lat_deg), h)
            # g is exactly affine in h here; finite difference of the whole profile
            grad = np.diff(g) / np.diff(h)
            # roundoff: eps*g/dh = 2e-16*10/100 ~ 2e-17 absolute -> 1e-9 relative is generous
            np.testing.assert_allclose(grad, FREE_AIR, rtol=1e-9, atol=0)
            # and the 0-5 km total change
            assert (g[-1] - g[0]) / 5000.0 == pytest.approx(FREE_AIR, rel=1e-9)

    def test_gravity_decreases_with_height(self):
        assert F.normal_gravity(0.5, 1000.0) < F.normal_gravity(0.5, 0.0)

    def test_height_term_independent_of_latitude(self):
        lat = np.radians([0.0, 30.0, 60.0, 90.0])
        d = F.normal_gravity(lat, 2000.0) - F.normal_gravity(lat, 0.0)
        np.testing.assert_allclose(d, FREE_AIR * 2000.0, rtol=1e-9)

    def test_scalar_and_batch_shapes(self):
        assert np.ndim(F.normal_gravity(0.1, 10.0)) == 0
        lat = np.radians(np.linspace(-80, 80, 7))
        assert F.normal_gravity(lat, 0.0).shape == (7,)
        assert F.normal_gravity(lat, 100.0).shape == (7,)
        assert F.normal_gravity(0.2, np.arange(5.0)).shape == (5,)
        assert F.normal_gravity(lat[:, None], np.arange(4.0)[None, :]).shape == (7, 4)
        assert F.normal_gravity(np.zeros((2, 3, 4)), np.zeros((2, 3, 4))).shape == (2, 3, 4)

    def test_batch_equals_elementwise(self):
        rng = np.random.default_rng(20260605)
        lat = rng.uniform(-np.pi / 2, np.pi / 2, (5, 6))
        alt = rng.uniform(-100.0, 5000.0, (5, 6))
        out = F.normal_gravity(lat, alt)
        expect = _oracle_somigliana(lat) + FREE_AIR * alt
        np.testing.assert_allclose(out, expect, rtol=0, atol=1e-9)
        for i in range(5):
            for j in range(6):
                assert out[i, j] == pytest.approx(F.normal_gravity(lat[i, j], alt[i, j]), abs=1e-15)

    def test_inputs_not_mutated(self):
        lat = np.radians(np.array([10.0, 20.0]))
        alt = np.array([5.0, 6.0])
        l0, a0 = lat.copy(), alt.copy()
        F.normal_gravity(lat, alt)
        np.testing.assert_array_equal(lat, l0)
        np.testing.assert_array_equal(alt, a0)
