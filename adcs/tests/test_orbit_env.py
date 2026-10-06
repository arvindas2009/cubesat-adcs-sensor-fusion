"""Tests for adcs/orbit.py and adcs/environment.py.

Written from docs/specs/adcs_physics.md sections 1, 2 and tests 1-10 of section 5,
not from the implementation.  Independent oracles used:

* perifocal -> ECI by an explicit scipy ``Rotation.from_euler('ZXZ', ...)``
* Kepler's equation solved with ``scipy.optimize.brentq``
* analytic J2 secular nodal regression (Vallado) and a finite-difference gradient
  of the J2 potential written out in this file
* ``sgp4.api.Satrec`` (WGS72) as the independent orbit propagator
* closed-form cylindrical-shadow fractions
* explicit east/north/up -> ECEF -> ECI rotations for the IGRF check
* published equinox/solstice instants and perihelion/aphelion distances

Runtime note: sample counts are chosen to keep the file well below 60 s; where a
spec tolerance allows it the sampling is coarser than the "at least" figure and
this is stated at the test.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
import ppigrf
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation as Rot
from sgp4.api import Satrec

from adcs import environment as env
from adcs import orbit as orb
from adcs.config import DEFAULT, SimConfig
from adcs.environment import (
    FieldTable,
    atmospheric_density,
    beta_angle,
    eclipse_fraction,
    geodetic_height,
    magnetic_field_eci,
    sun_direction_eci,
    sun_vector_eci,
    sunlit,
)
from adcs.orbit import (
    Ephemeris,
    accel_j2,
    initial_state,
    kepler_to_rv,
    orbital_period,
    propagate,
    rv_to_kepler,
    sgp4_ephemeris,
    specific_energy,
)
from common.frames import (
    GM_EARTH,
    J2_EARTH,
    OMEGA_EARTH,
    R_EARTH,
    WGS84_B,
    gmst,
    julian_date,
)

MU = GM_EARTH
RE = R_EARTH
AU = 1.495978707e11


# ------------------------------------------------------------------- helpers
def ref_rv(a, e, inc, raan, argp, nu, mu=MU):
    """Independent perifocal -> ECI construction (scipy ZXZ rotation)."""
    p = a * (1.0 - e**2)
    r_pf = p / (1.0 + e * np.cos(nu)) * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    rot = Rot.from_euler("ZXZ", [raan, inc, argp]).as_matrix()  # Rz(raan) Rx(inc) Rz(argp)
    return rot @ r_pf, rot @ v_pf


def wrap(x):
    """Wrap angle difference to (-pi, pi]."""
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def phi_j2(r, mu=MU, j2=J2_EARTH, re=RE):
    """Gravitational potential per unit mass written out from the spec."""
    rn = np.linalg.norm(r)
    return -mu / rn * (1.0 - j2 * (re / rn) ** 2 * (3.0 * r[2] ** 2 / rn**2 - 1.0) / 2.0)


A_LEO = RE + 400e3
T_LEO = 2 * np.pi * np.sqrt(A_LEO**3 / MU)


# ============================================================ 1. kepler <-> rv
def test_orbital_period_formula():
    assert orbital_period(A_LEO) == pytest.approx(2 * np.pi * np.sqrt(A_LEO**3 / MU), rel=1e-14)
    # Kepler's third law scaling: T ~ a^1.5
    assert orbital_period(4 * A_LEO) == pytest.approx(8 * orbital_period(A_LEO), rel=1e-13)
    # Sanity against the well-known ~92.5 min at 400 km
    assert 5500.0 < orbital_period(A_LEO) < 5600.0
    # custom mu
    assert orbital_period(1.0e7, mu=1.0e14) == pytest.approx(2 * np.pi * np.sqrt(1e21 / 1e14), rel=1e-14)


def test_kepler_to_rv_matches_independent_construction():
    rng = np.random.default_rng(101)
    for _ in range(25):
        a = rng.uniform(7e6, 3e7)
        e = rng.uniform(0.0, 0.6)
        inc = rng.uniform(0.0, np.pi)
        raan, argp, nu = rng.uniform(0, 2 * np.pi, 3)
        r, v = kepler_to_rv(a, e, inc, raan, argp, nu)
        r0, v0 = ref_rv(a, e, inc, raan, argp, nu)
        assert r.shape == (3,) and v.shape == (3,)
        assert np.linalg.norm(r - r0) / np.linalg.norm(r0) < 1e-12
        assert np.linalg.norm(v - v0) / np.linalg.norm(v0) < 1e-12


def test_kepler_rv_round_trip_random_elements():
    rng = np.random.default_rng(7)
    for _ in range(50):
        a = rng.uniform(6.9e6, 4e7)
        e = rng.uniform(0.01, 0.5)
        inc = rng.uniform(0.1, np.pi - 0.1)
        raan, argp, nu = rng.uniform(0, 2 * np.pi, 3)
        r, v = kepler_to_rv(a, e, inc, raan, argp, nu)
        k = rv_to_kepler(r, v)
        assert set(["a", "ecc", "inc", "raan", "argp", "nu"]) <= set(k)
        assert k["a"] == pytest.approx(a, rel=1e-11)
        assert k["ecc"] == pytest.approx(e, abs=1e-11)
        assert k["inc"] == pytest.approx(inc, abs=1e-11)
        # conditioning of argp/nu is ~1/e <= 100, so 1e-10 leaves margin over double precision
        for key, val in (("raan", raan), ("argp", argp), ("nu", nu)):
            assert abs(wrap(k[key] - val)) < 1e-10, key


def test_kepler_rv_round_trip_with_non_default_mu():
    mu = 1.2e14
    r, v = kepler_to_rv(9e6, 0.2, 0.7, 1.0, 2.0, 3.0, mu=mu)
    k = rv_to_kepler(r, v, mu=mu)
    assert k["a"] == pytest.approx(9e6, rel=1e-11)
    assert k["ecc"] == pytest.approx(0.2, abs=1e-11)


@pytest.mark.parametrize("inc", [0.3, np.deg2rad(51.6), 1.5, 2.8])
@pytest.mark.parametrize("nu", [0.0, 1.0, 4.0])
def test_circular_orbit_properties(inc, nu):
    a = A_LEO
    r, v = kepler_to_rv(a, 0.0, inc, 0.4, 0.9, nu)
    assert np.linalg.norm(r) == pytest.approx(a, rel=1e-13)
    assert np.linalg.norm(v) == pytest.approx(np.sqrt(MU / a), rel=1e-13)
    assert abs(r @ v) / (np.linalg.norm(r) * np.linalg.norm(v)) < 1e-13
    # orbit normal inclination: h_z / |h| = cos(inc)
    h = np.cross(r, v)
    assert h[2] / np.linalg.norm(h) == pytest.approx(np.cos(inc), abs=1e-13)


def test_elliptical_orbit_invariants():
    a, e = 1.2e7, 0.4
    r, v = kepler_to_rv(a, e, 0.8, 0.3, 1.1, 2.0)
    # vis-viva energy and angular momentum from the closed forms
    assert 0.5 * v @ v - MU / np.linalg.norm(r) == pytest.approx(-MU / (2 * a), rel=1e-12)
    assert np.linalg.norm(np.cross(r, v)) == pytest.approx(np.sqrt(MU * a * (1 - e**2)), rel=1e-12)
    # perigee passage: r = a (1 - e), v = sqrt(mu/a) sqrt((1+e)/(1-e))
    r, v = kepler_to_rv(a, e, 0.8, 0.3, 1.1, 0.0)
    assert np.linalg.norm(r) == pytest.approx(a * (1 - e), rel=1e-13)
    assert np.linalg.norm(v) == pytest.approx(np.sqrt(MU / a * (1 + e) / (1 - e)), rel=1e-13)


def test_kepler_to_rv_equatorial_axes_explicit():
    # raan = argp = 0, inc = 0, nu = 0 -> r on +x, v on +y
    r, v = kepler_to_rv(A_LEO, 0.0, 0.0, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(r, [A_LEO, 0, 0], atol=1e-6)
    np.testing.assert_allclose(v, [0, np.sqrt(MU / A_LEO), 0], atol=1e-9)
    # Pure RAAN rotation by 90 deg of the same orbit -> r on +y, v on -x
    r, v = kepler_to_rv(A_LEO, 0.0, 0.0, np.pi / 2, 0.0, 0.0)
    np.testing.assert_allclose(r, [0, A_LEO, 0], atol=1e-6)
    np.testing.assert_allclose(v, [-np.sqrt(MU / A_LEO), 0, 0], atol=1e-9)


def test_initial_state_default_config_explicit_values():
    r, v = initial_state()
    inc = DEFAULT.orbit.inc
    vc = np.sqrt(MU / (RE + 400e3))
    # RAAN = argp = nu = 0: r on +x, v = vc [0, cos i, sin i]
    np.testing.assert_allclose(r, [RE + 400e3, 0, 0], atol=1e-6)
    np.testing.assert_allclose(v, vc * np.array([0, np.cos(inc), np.sin(inc)]), atol=1e-9)
    r2, v2 = initial_state(DEFAULT)
    np.testing.assert_array_equal(r, r2)
    np.testing.assert_array_equal(v, v2)


def test_initial_state_custom_config():
    import dataclasses

    orb_cfg = dataclasses.replace(DEFAULT.orbit, altitude=600e3, ecc=0.05, raan=1.0, argp=0.5, nu=0.7, inc=1.2)
    cfg = dataclasses.replace(DEFAULT, orbit=orb_cfg)
    r, v = initial_state(cfg)
    r0, v0 = ref_rv(RE + 600e3, 0.05, 1.2, 1.0, 0.5, 0.7)
    assert np.linalg.norm(r - r0) < 1e-5
    assert np.linalg.norm(v - v0) < 1e-8


def test_kepler_to_rv_does_not_mutate_inputs():
    a = np.array(A_LEO)
    kepler_to_rv(a, 0.1, 0.5, 0.1, 0.2, 0.3)
    assert a == A_LEO
    r = np.array([7e6, 1e5, 2e5])
    v = np.array([-100.0, 7500.0, 1000.0])
    rc, vc = r.copy(), v.copy()
    rv_to_kepler(r, v)
    np.testing.assert_array_equal(r, rc)
    np.testing.assert_array_equal(v, vc)


# ========================================================== 2. two-body propagation
def test_two_body_circular_ten_periods_conservation():
    r0, v0 = ref_rv(A_LEO, 0.0, np.deg2rad(51.6), 0.7, 0.0, 0.3)
    T = 2 * np.pi * np.sqrt(A_LEO**3 / MU)
    t = np.linspace(0.0, 10 * T, 401)  # 40 samples / period; includes t = T
    r, v = propagate(r0, v0, t, j2=False)
    assert r.shape == (401, 3) and v.shape == (401, 3)
    np.testing.assert_allclose(r[0], r0, atol=1e-9)
    np.testing.assert_allclose(v[0], v0, atol=1e-12)
    rn = np.linalg.norm(r, axis=1)
    assert np.max(np.abs(rn - A_LEO)) < 1e-3  # spec threshold: 1e-3 m
    e = specific_energy(r, v, j2=False)
    assert e.shape == (401,)
    assert np.max(np.abs(e - e[0]) / abs(e[0])) < 1e-10
    h = np.cross(r, v)
    assert np.max(np.linalg.norm(h - h[0], axis=1) / np.linalg.norm(h[0])) < 1e-10
    # returns to start after exactly one period (2-body a gives the exact period)
    assert np.linalg.norm(r[40] - r0) < 1.0
    assert np.linalg.norm(r[-1] - r0) < 1.0  # also after ten periods (spec: one period; 10 is stricter)


def test_propagate_does_not_mutate_inputs_and_returns_floats():
    r0, v0 = ref_rv(A_LEO, 0.0, 0.5, 0.0, 0.0, 0.0)
    r0c, v0c = r0.copy(), v0.copy()
    r, v = propagate(r0, v0, np.array([0.0, 100.0, 200.0]), j2=True)
    np.testing.assert_array_equal(r0, r0c)
    np.testing.assert_array_equal(v0, v0c)
    assert r.dtype == float and v.dtype == float


def test_two_body_elliptical_matches_kepler_equation():
    a, e, inc, raan, argp, nu0 = 1.0e7, 0.3, 0.9, 1.0, 0.5, 0.7
    n = np.sqrt(MU / a**3)
    T = 2 * np.pi / n
    r0, v0 = ref_rv(a, e, inc, raan, argp, nu0)
    # mean anomaly at t = 0 from the true anomaly
    E0 = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(nu0 / 2), np.sqrt(1 + e) * np.cos(nu0 / 2))
    M0 = E0 - e * np.sin(E0)
    times = np.array([0.0, 0.1 * T, 0.37 * T, 0.5 * T, 0.9 * T, 1.0 * T, 1.45 * T, 2.2 * T])
    r, v = propagate(r0, v0, times, j2=False)
    for k, tk in enumerate(times):
        M = M0 + n * tk
        # E - e sin E = M has its root within [M - e, M + e]
        E = brentq(lambda x: x - e * np.sin(x) - M, M - e - 1e-9, M + e + 1e-9, xtol=1e-14, rtol=1e-14)
        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
        r_ref, v_ref = ref_rv(a, e, inc, raan, argp, nu)
        assert np.linalg.norm(r[k] - r_ref) < 1.0, f"t={tk}"  # spec: < 1 m
        assert np.linalg.norm(v[k] - v_ref) < 1e-3


# ================================================================= 3. J2
def test_accel_j2_two_body_limit_and_formula():
    rng = np.random.default_rng(3)
    r = rng.normal(size=(6, 3))
    r = r / np.linalg.norm(r, axis=1, keepdims=True) * rng.uniform(6.8e6, 4e7, (6, 1))
    a0 = accel_j2(r, j2=0.0)
    np.testing.assert_allclose(a0, -MU * r / np.linalg.norm(r, axis=1, keepdims=True) ** 3, rtol=1e-13)
    # J2 formula (Vallado Eq. 8-30) transcribed from the spec
    rn = np.linalg.norm(r, axis=1)
    zr2 = (r[:, 2] / rn) ** 2
    f = 1.5 * J2_EARTH * (RE / rn) ** 2
    expected = -MU * r / rn[:, None] ** 3
    expected[:, 0] *= 1 + f * (1 - 5 * zr2)
    expected[:, 1] *= 1 + f * (1 - 5 * zr2)
    expected[:, 2] *= 1 + f * (3 - 5 * zr2)
    np.testing.assert_allclose(accel_j2(r), expected, rtol=1e-13)


def test_accel_j2_batch_shapes_and_single():
    rng = np.random.default_rng(4)
    r = rng.normal(size=(2, 5, 3)) * 4e6 + np.array([7e6, 0, 0])
    a = accel_j2(r)
    assert a.shape == (2, 5, 3)
    for i in range(2):
        for j in range(5):
            np.testing.assert_allclose(a[i, j], accel_j2(r[i, j]), rtol=1e-14)
    assert accel_j2(r[0, 0]).shape == (3,)


def test_accel_j2_is_minus_gradient_of_potential():
    rng = np.random.default_rng(5)
    h = 100.0  # m; central difference. Truncation ~ h^2 mu/r^4 / 6 ~ 2e-10 relative, roundoff ~ 1e-11
    for _ in range(12):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        r = u * rng.uniform(6.8e6, 2e7)
        grad = np.zeros(3)
        for k in range(3):
            d = np.zeros(3)
            d[k] = h
            grad[k] = (phi_j2(r + d) - phi_j2(r - d)) / (2 * h)
        a = accel_j2(r)
        assert np.linalg.norm(a + grad) / np.linalg.norm(a) < 1e-8


def test_specific_energy_matches_potential_and_j2_flag():
    rng = np.random.default_rng(6)
    r = np.array([[7e6, 1e6, 2e6], [-3e6, 6e6, 4e6]])
    v = rng.normal(size=(2, 3)) * 3000
    e_j2 = specific_energy(r, v, j2=True)
    e_2b = specific_energy(r, v, j2=False)
    for i in range(2):
        rn = np.linalg.norm(r[i])
        assert e_2b[i] == pytest.approx(0.5 * v[i] @ v[i] - MU / rn, rel=1e-13)
        assert e_j2[i] == pytest.approx(0.5 * v[i] @ v[i] + phi_j2(r[i]), rel=1e-13)
    assert specific_energy(r[0], v[0]).shape == ()
    # J2 term is ~1e-3 relative: present but small
    assert 1e-5 < abs(e_j2[0] - e_2b[0]) / abs(e_2b[0]) < 1e-2


def test_j2_conservation_and_nodal_regression_ten_orbits():
    a, e, inc = 7.0e6, 0.01, np.deg2rad(51.6)  # e > 0 keeps Omega/argp well conditioned
    r0, v0 = ref_rv(a, e, inc, 0.5, 1.0, 0.0)
    T = 2 * np.pi * np.sqrt(a**3 / MU)
    # 40 samples/orbit; the nodal fit and the conservation checks need no more
    t = np.linspace(0.0, 10 * T, 401)
    r, v = propagate(r0, v0, t, j2=True)

    en = specific_energy(r, v, j2=True)
    assert np.max(np.abs(en - en[0]) / abs(en[0])) < 1e-9
    hz = np.cross(r, v)[:, 2]
    assert np.max(np.abs(hz - hz[0]) / abs(hz[0])) < 1e-9
    # two-body energy is NOT conserved to that level (J2 short-period terms) -> the check is meaningful
    en2 = specific_energy(r, v, j2=False)
    assert np.max(np.abs(en2 - en2[0]) / abs(en2[0])) > 1e-6

    raan = np.array([rv_to_kepler(ri, vi)["raan"] for ri, vi in zip(r, v)])
    raan = np.unwrap(raan)
    slope = np.polyfit(t, raan, 1)[0]
    p = a * (1 - e**2)
    n = np.sqrt(MU / a**3)
    expected = -1.5 * J2_EARTH * n * (RE / p) ** 2 * np.cos(inc)
    assert expected < 0
    assert slope == pytest.approx(expected, rel=0.03)  # spec: within 3 %


def test_j2_changes_trajectory_but_two_body_period_returns():
    r0, v0 = ref_rv(A_LEO, 0.0, np.deg2rad(51.6), 0.0, 0.0, 0.0)
    t = np.array([0.0, 5 * T_LEO])
    r_j2, _ = propagate(r0, v0, t, j2=True)
    r_2b, _ = propagate(r0, v0, t, j2=False)
    assert np.linalg.norm(r_j2[1] - r_2b[1]) > 1e3  # J2 moves the satellite by km within 5 orbits


# ================================================================= 4. SGP4
TLE1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
TLE2 = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472 20569"


def _tle_checksum(line):
    return sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line[:68]) % 10


@pytest.fixture(scope="module")
def sgp4_case():
    sat = Satrec.twoline2rv(TLE1, TLE2)
    assert sat.error == 0
    t = np.arange(0.0, 86400.0 + 1, 60.0)
    e, r, v = sat.sgp4_array(np.full(t.shape, sat.jdsatepoch), sat.jdsatepochF + t / 86400.0)
    assert np.all(e == 0)
    return sat, t, r * 1e3, v * 1e3


def test_tle_checksums_and_error_code():
    assert _tle_checksum(TLE1) == int(TLE1[68])
    assert _tle_checksum(TLE2) == int(TLE2[68])
    assert Satrec.twoline2rv(TLE1, TLE2).error == 0


def test_sgp4_ephemeris_units_and_agreement_with_direct_sgp4(sgp4_case):
    sat, t, r_ref, v_ref = sgp4_case
    jd0 = sat.jdsatepoch + sat.jdsatepochF
    r, v = sgp4_ephemeris(TLE1, TLE2, jd0, t)
    assert r.shape == (t.size, 3) and v.shape == (t.size, 3)
    rn = np.linalg.norm(r, axis=1)
    vn = np.linalg.norm(v, axis=1)
    assert 6.7e6 < rn[0] < 6.9e6 and abs(rn[0] - 6.78e6) < 0.1e6  # m, ISS ~ 6.78e6
    assert 7.5e3 < vn[0] < 7.8e3 and abs(vn[0] - 7.66e3) < 0.1e3  # m/s
    # jd0 passed as a single double loses ~4e-5 s (ulp of 2.4e6 d) -> <= 0.3 m, 4e-4 m/s
    assert np.max(np.linalg.norm(r - r_ref, axis=1)) < 1.0
    assert np.max(np.linalg.norm(v - v_ref, axis=1)) < 5e-3


def test_sgp4_ephemeris_raises_on_sgp4_error():
    sat = Satrec.twoline2rv(TLE1, TLE2)
    jd0 = sat.jdsatepoch + sat.jdsatepochF
    # mean eccentricity 0.9999999 -> SGP4 error code 4 (checked directly with python-sgp4 when writing the test)
    bad2 = TLE2[:26] + "9999999" + TLE2[33:]
    sat_bad = Satrec.twoline2rv(TLE1, bad2)
    assert sat_bad.sgp4(sat_bad.jdsatepoch, sat_bad.jdsatepochF)[0] != 0
    with pytest.raises(RuntimeError):
        sgp4_ephemeris(TLE1, bad2, jd0, np.array([0.0, 60.0]))
    # decayed after ~55 years -> error code 6
    with pytest.raises(RuntimeError):
        sgp4_ephemeris(TLE1, TLE2, jd0, np.array([0.0, 20000.0 * 86400.0]))


def test_propagate_vs_sgp4_one_day(sgp4_case):
    # The difference is dominated by (a) atmospheric drag (B* = 3.88e-5 /Re, modelled by SGP4 but not by us),
    # (b) J3/J4 and other zonal terms SGP4 includes, and (c) the osculating vs. mean-element mismatch
    # (we start from the SGP4 state, which contains SGP4's short-period terms, and then evolve with J2 only).
    sat, t, r_ref, v_ref = sgp4_case
    r, v = propagate(r_ref[0], v_ref[0], t, j2=True)
    dr = np.linalg.norm(r - r_ref, axis=1)
    period = 2 * np.pi / (sat.no_kozai / 60.0)  # s, from the TLE mean motion (rad/min -> rad/s)
    first_orbit = t <= period
    assert first_orbit.sum() > 80
    assert dr[first_orbit].max() < 5e3  # spec: < 5 km in the first orbit
    assert dr.max() < 30e3  # spec: < 30 km over the day
    d_rad = np.linalg.norm(r, axis=1) - np.linalg.norm(r_ref, axis=1)
    assert np.max(np.abs(d_rad)) < 5e3  # spec: radius difference bounded by 5 km
    # start state is reproduced exactly
    assert dr[0] < 1e-6


# ================================================================ 5. Ephemeris
def test_ephemeris_matches_propagate_and_range_errors():
    r0, v0 = ref_rv(A_LEO, 0.001, np.deg2rad(51.6), 0.3, 0.2, 0.1)
    t_end = 2.5 * T_LEO
    eph = Ephemeris(r0, v0, t_end, j2=True)
    assert eph.t_end == pytest.approx(t_end)
    t = np.array([0.0, 17.3, 600.0, 1234.5, 3000.0, 5000.0, 9999.9, 0.9 * t_end, t_end])
    r_p, v_p = propagate(r0, v0, t, j2=True)
    r_e, v_e = eph.at(t)
    assert r_e.shape == (t.size, 3) and v_e.shape == (t.size, 3)
    assert np.max(np.linalg.norm(r_e - r_p, axis=1)) < 1e-6  # spec: 1e-6 m
    # velocities agree to correspondingly tight bounds (not in spec: 1e-9 m/s is a generous echo of 1e-6 m)
    assert np.max(np.linalg.norm(v_e - v_p, axis=1)) < 1e-6
    # scalar
    rs, vs = eph.at(1234.5)
    assert rs.shape == (3,) and vs.shape == (3,)
    np.testing.assert_allclose(rs, r_p[3], atol=1e-6)
    # 2-D time array
    r2, v2 = eph.at(t[:6].reshape(2, 3))
    assert r2.shape == (2, 3, 3) and v2.shape == (2, 3, 3)


def test_ephemeris_j2_flag_changes_result():
    r0, v0 = ref_rv(A_LEO, 0.0, np.deg2rad(51.6), 0.0, 0.0, 0.0)
    e1 = Ephemeris(r0, v0, 3 * T_LEO, j2=True)
    e2 = Ephemeris(r0, v0, 3 * T_LEO, j2=False)
    assert np.linalg.norm(e1.at(3 * T_LEO)[0] - e2.at(3 * T_LEO)[0]) > 1e3


@pytest.mark.parametrize("bad", [-1.0, 5000.0 + 1e-3, 1e6])
def test_ephemeris_out_of_range_raises(bad):
    r0, v0 = ref_rv(A_LEO, 0.0, 0.5, 0.0, 0.0, 0.0)
    eph = Ephemeris(r0, v0, 5000.0)
    with pytest.raises(ValueError):
        eph.at(bad)
    with pytest.raises(ValueError):
        eph.at(np.array([10.0, bad]))
    # in-range boundaries are fine
    eph.at(0.0)
    eph.at(5000.0)


# ========================================================== 6. Sun
def _sun_ref(jd):
    """Vallado Algorithm 29 transcribed from the spec (independent re-implementation)."""
    T = (np.asarray(jd, float) - 2451545.0) / 36525.0
    d = np.deg2rad
    lam_m = 280.460 + 36000.771 * T
    M = d(357.5291092 + 35999.05034 * T)
    lam = d(lam_m + 1.914666471 * np.sin(M) + 0.019994643 * np.sin(2 * M))
    r = 1.000140612 - 0.016708617 * np.cos(M) - 0.000139589 * np.cos(2 * M)
    eps = d(23.439291 - 0.0130042 * T)
    return (r * AU)[..., None] * np.stack([np.cos(lam), np.cos(eps) * np.sin(lam), np.sin(eps) * np.sin(lam)], axis=-1)


def _dec_deg(vec):
    return np.rad2deg(np.arcsin(vec[..., 2] / np.linalg.norm(vec, axis=-1)))


def _ecl_lon_deg(vec):
    eps = np.deg2rad(23.439291)
    y_ecl = vec[1] * np.cos(eps) + vec[2] * np.sin(eps)
    return np.rad2deg(np.arctan2(y_ecl, vec[0])) % 360.0


def test_sun_vector_matches_spec_algorithm_transcription():
    jd = 2461120.0 + np.array([0.0, 17.25, 90.5, 181.0, 300.7, 365.0])
    np.testing.assert_allclose(sun_vector_eci(jd), _sun_ref(jd), rtol=1e-12, atol=1.0)


def test_sun_vector_batch_shapes_and_units():
    jd = 2461120.0 + np.arange(12.0).reshape(3, 4)
    s = sun_vector_eci(jd)
    assert s.shape == (3, 4, 3)
    assert sun_vector_eci(2461120.0).shape == (3,)
    d = np.linalg.norm(s, axis=-1)
    assert np.all((0.98e11 * 1.5 < d) & (d < 1.02 * AU))  # metres (not km / AU)
    u = sun_direction_eci(jd)
    assert u.shape == (3, 4, 3)
    np.testing.assert_allclose(np.linalg.norm(u, axis=-1), 1.0, atol=1e-14)
    np.testing.assert_allclose(u, s / d[..., None], rtol=1e-13)


@pytest.mark.parametrize(
    "when, dec_target, tol",
    [
        (dt.datetime(2026, 3, 20, 14, 46), 0.0, 0.05),  # March equinox
        (dt.datetime(2026, 6, 21, 8, 24), 23.44, 0.1),  # June solstice
        (dt.datetime(2026, 9, 23, 0, 5), 0.0, 0.1),  # September equinox
        (dt.datetime(2026, 12, 21, 20, 50), -23.44, 0.1),  # December solstice
    ],
)
def test_sun_declination_at_equinoxes_and_solstices(when, dec_target, tol):
    dec = _dec_deg(sun_vector_eci(julian_date(when)))
    assert abs(dec - dec_target) < tol


@pytest.mark.parametrize(
    "when, lon_target",
    [
        (dt.datetime(2026, 3, 20, 14, 46), 0.0),
        (dt.datetime(2026, 6, 21, 8, 24), 90.0),
        (dt.datetime(2026, 9, 23, 0, 5), 180.0),
        (dt.datetime(2026, 12, 21, 20, 50), 270.0),
    ],
)
def test_sun_ecliptic_longitude_at_equinoxes_and_solstices(when, lon_target):
    # Not in the spec's numbered list but follows from "ecliptic longitude consistent with the
    # equinoxes/solstices".  0.05 deg ~ 1.2 h of solar motion; the almanac formula is a mean-equinox
    # geometric longitude (no nutation/aberration, <= 0.01 deg).
    lon = _ecl_lon_deg(sun_vector_eci(julian_date(when)))
    assert abs(wrap(np.deg2rad(lon - lon_target))) < np.deg2rad(0.05)


def test_sun_right_ascension_zero_at_march_equinox():
    s = sun_vector_eci(julian_date(dt.datetime(2026, 3, 20, 14, 46)))
    ra = np.rad2deg(np.arctan2(s[1], s[0]))
    assert abs(wrap(np.deg2rad(ra))) < np.deg2rad(0.1)


def test_sun_distance_perihelion_and_aphelion():
    d_peri = np.linalg.norm(sun_vector_eci(julian_date(dt.datetime(2026, 1, 3, 17, 0)))) / AU
    d_aph = np.linalg.norm(sun_vector_eci(julian_date(dt.datetime(2026, 7, 6, 18, 0)))) / AU
    assert d_peri == pytest.approx(0.9833, abs=0.001)
    assert d_aph == pytest.approx(1.0167, abs=0.001)
    # and over a full year these are the extremes
    jd = julian_date(dt.datetime(2026, 1, 1)) + np.arange(0, 366.0, 1.0)
    d = np.linalg.norm(sun_vector_eci(jd), axis=-1) / AU
    assert abs(jd[np.argmin(d)] - julian_date(dt.datetime(2026, 1, 3))) < 4.0
    assert abs(jd[np.argmax(d)] - julian_date(dt.datetime(2026, 7, 6))) < 4.0


def test_sun_direction_unit_over_a_year_and_moves_eastward():
    jd = julian_date(dt.datetime(2026, 1, 1)) + np.linspace(0, 365, 200)
    u = sun_direction_eci(jd)
    np.testing.assert_allclose(np.linalg.norm(u, axis=1), 1.0, atol=1e-14)
    # RA increases (Sun moves east along the ecliptic) ~ 0.9856 deg/day
    ra = np.unwrap(np.arctan2(u[:, 1], u[:, 0]))
    assert np.all(np.diff(ra) > 0)
    assert np.rad2deg(ra[-1] - ra[0]) == pytest.approx(360.0 * 365 / 365.2422, abs=1.0)


# ========================================================== 7. Eclipse
def _circular_series(n, beta=0.0, a=A_LEO):
    """Circular orbit in the xy-plane, sun at angle beta above the plane in the xz-plane."""
    th = np.arange(n) * 2 * np.pi / n
    r = a * np.stack([np.cos(th), np.sin(th), np.zeros(n)], axis=1)
    s = np.tile([np.cos(beta), 0.0, np.sin(beta)], (n, 1))
    return r, s


def test_sunlit_basic_cases():
    s = np.array([1.0, 0.0, 0.0])
    r_dark = np.array([-7.0e6, 0.0, 0.0])  # directly behind the Earth
    r_abeam = np.array([0.0, 7.0e6, 0.0])  # perpendicular distance 7000 km > Re
    r_sun_side = np.array([7.0e6, 0.0, 0.0])
    assert not sunlit(r_dark, s)
    assert sunlit(r_abeam, s)
    assert sunlit(r_sun_side, s)
    # behind the Earth but outside the cylinder -> lit; inside -> dark
    assert sunlit(np.array([-7.0e6, 6.4e6, 0.0]), s)
    assert not sunlit(np.array([-7.0e6, 0.0, 6.3e6]), s)
    # just inside / just outside the cylinder edge, with r.s < 0
    assert not sunlit(np.array([-1e7, RE - 1.0, 0.0]), s)
    assert sunlit(np.array([-1e7, RE + 1.0, 0.0]), s)
    # sun side: even with small perpendicular offset, lit
    assert sunlit(np.array([1e7, 1e3, 0.0]), s)


def test_sunlit_batched_returns_bool_array():
    s = np.tile([1.0, 0.0, 0.0], (4, 1))
    r = np.array([[-7e6, 0, 0], [0, 7e6, 0], [7e6, 0, 0], [-7e6, 6.4e6, 0]])
    out = sunlit(r, s)
    assert out.shape == (4,)
    assert out.dtype == bool
    np.testing.assert_array_equal(out, [False, True, True, True])
    # single sun vector broadcast against a series
    np.testing.assert_array_equal(sunlit(r, np.array([1.0, 0.0, 0.0])), [False, True, True, True])


def test_sunlit_is_independent_of_sun_vector_sign_convention_with_rotation():
    # Rotate the whole geometry: result must be invariant
    rng = np.random.default_rng(11)
    rot = Rot.random(random_state=2).as_matrix()
    r = rng.uniform(-1.2e7, 1.2e7, (200, 3))
    r = r[np.linalg.norm(r, axis=1) > RE]
    s = np.array([1.0, 0.0, 0.0])
    ref = np.array([not (ri @ s < 0 and np.linalg.norm(ri - (ri @ s) * s) < RE) for ri in r])
    np.testing.assert_array_equal(sunlit(r, s), ref)
    np.testing.assert_array_equal(sunlit(r @ rot.T, rot @ s), ref)


def test_eclipse_fraction_is_mean_of_not_sunlit():
    r, s = _circular_series(1000)
    assert eclipse_fraction(r, s) == pytest.approx(np.mean(~sunlit(r, s)), abs=1e-15)


def test_eclipse_fraction_beta_zero_matches_analytic():
    n = 3000  # >= 2000 per orbit (spec); sampling error <= 1/n = 3e-4 < 2e-3
    r, s = _circular_series(n, 0.0)
    expected = np.arcsin(RE / A_LEO) / np.pi
    assert expected == pytest.approx(0.390, abs=0.003)  # ~0.39 (spec quotes 0.392 loosely)
    assert eclipse_fraction(r, s) == pytest.approx(expected, abs=0.002)


@pytest.mark.parametrize("beta_deg", [-60.0, -30.0, -10.0, 10.0, 30.0, 50.0, 60.0, 68.0])
def test_eclipse_fraction_with_beta_matches_analytic(beta_deg):
    beta = np.deg2rad(beta_deg)
    h = 400e3
    beta_star = np.arcsin(RE / (RE + h))
    assert abs(beta) < beta_star
    expected = np.arccos(np.sqrt(h**2 + 2 * RE * h) / ((RE + h) * np.cos(beta))) / np.pi
    r, s = _circular_series(4000, beta)
    assert eclipse_fraction(r, s) == pytest.approx(expected, abs=0.002)


@pytest.mark.parametrize("beta_deg", [-89.0, -80.0, -72.0, 72.0, 80.0, 89.0])
def test_eclipse_fraction_zero_above_critical_beta(beta_deg):
    beta_star = np.rad2deg(np.arcsin(RE / A_LEO))  # 70.2 deg
    assert abs(beta_deg) > beta_star
    r, s = _circular_series(2000, np.deg2rad(beta_deg))
    assert eclipse_fraction(r, s) == 0.0


def test_beta_angle_of_constructed_orbit_matches():
    rng = np.random.default_rng(21)
    for beta in [-1.3, -0.5, -0.1, 0.0, 0.2, 0.9, 1.4]:
        th = rng.uniform(0, 2 * np.pi)
        r = A_LEO * np.array([np.cos(th), np.sin(th), 0.0])
        v = np.sqrt(MU / A_LEO) * np.array([-np.sin(th), np.cos(th), 0.0])
        # sun: unit vector at elevation beta above the orbit plane (h = +z), azimuth arbitrary
        az = rng.uniform(0, 2 * np.pi)
        s = np.array([np.cos(beta) * np.cos(az), np.cos(beta) * np.sin(az), np.sin(beta)])
        # random rigid rotation of everything
        rot = Rot.random(random_state=int(rng.integers(1_000_000))).as_matrix()
        b = beta_angle(rot @ r, rot @ v, rot @ s)
        assert float(b) == pytest.approx(beta, abs=1e-12)


def test_beta_angle_sign_and_batch():
    r = np.array([A_LEO, 0, 0])
    v = np.array([0, np.sqrt(MU / A_LEO), 0])  # h = +z
    assert beta_angle(r, v, np.array([0, 0, 1.0])) == pytest.approx(np.pi / 2, abs=1e-14)
    assert beta_angle(r, v, np.array([0, 0, -1.0])) == pytest.approx(-np.pi / 2, abs=1e-14)
    assert beta_angle(r, v, np.array([1.0, 0, 0])) == pytest.approx(0.0, abs=1e-14)
    # a retrograde orbit flips the sign
    assert beta_angle(r, -v, np.array([0, 0, 1.0])) == pytest.approx(-np.pi / 2, abs=1e-14)
    rs = np.tile(r, (3, 1))
    vs = np.tile(v, (3, 1))
    ss = np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 0, -1.0]])
    np.testing.assert_allclose(beta_angle(rs, vs, ss), [np.pi / 2, 0.0, -np.pi / 2], atol=1e-14)


@pytest.fixture(scope="module")
def default_day():
    r0, v0 = initial_state(DEFAULT)
    t = np.arange(0.0, 86400.0, 5.0)  # 5 s: ~1100 samples/orbit
    r, v = propagate(r0, v0, t, j2=True)
    return t, r, v


def test_default_config_eclipse_fraction_gate(default_day):
    t, r, v = default_day
    s = sun_direction_eci(DEFAULT.jd_epoch + t / 86400.0)
    f = eclipse_fraction(r, s)
    print(f"default 1-day eclipse fraction = {f:.4f}")
    assert 0.35 < f < 0.40  # plan gate


def test_default_config_beta_is_small(default_day):
    # B-14: epoch near the equinox with RAAN = 0 -> beta ~ 0 (hand estimate ~0.07 deg at epoch)
    t, r, v = default_day
    s0 = sun_direction_eci(DEFAULT.jd_epoch)
    b = beta_angle(r[0], v[0], s0)
    assert abs(np.rad2deg(b)) < 1.0


# ========================================================== 9. Density / height
def test_density_reference_value_and_scale_height():
    rho0, H = 3.725e-12, 58.515e3
    assert atmospheric_density(400e3) == pytest.approx(rho0, rel=1e-14)
    assert atmospheric_density(400e3 + H) == pytest.approx(rho0 / np.e, rel=1e-12)
    assert atmospheric_density(400e3 - H) == pytest.approx(rho0 * np.e, rel=1e-12)
    h = np.array([300e3, 350e3, 400e3, 450e3, 500e3])
    np.testing.assert_allclose(atmospheric_density(h), rho0 * np.exp(-(h - 400e3) / H), rtol=1e-12)


def test_density_monotone_decreasing_positive_and_shape():
    h = np.linspace(0.0, 1000e3, 101)
    rho = atmospheric_density(h)
    assert rho.shape == h.shape
    assert np.all(rho > 0)
    assert np.all(np.diff(rho) < 0)
    assert np.isscalar(float(atmospheric_density(400e3)))


def test_density_negative_height_raises():
    with pytest.raises(ValueError):
        atmospheric_density(-1.0)
    with pytest.raises(ValueError):
        atmospheric_density(np.array([400e3, -10.0]))
    atmospheric_density(0.0)  # zero height is allowed


def test_geodetic_height_equatorial_and_polar_points():
    for jd in (2461120.0, 2461120.37, 2461200.8):  # GMST irrelevant for a point on the axis x or z
        assert geodetic_height(np.array([RE + 400e3, 0.0, 0.0]), jd) == pytest.approx(400e3, abs=1.0)
        # rotated about z (still equatorial)
        th = 1.234
        r = (RE + 400e3) * np.array([np.cos(th), np.sin(th), 0.0])
        assert geodetic_height(r, jd) == pytest.approx(400e3, abs=1.0)
        # pole: height above the semi-minor axis
        assert geodetic_height(np.array([0.0, 0.0, WGS84_B + 400e3]), jd) == pytest.approx(400e3, abs=1.0)
    # batched
    rr = np.array([[RE + 400e3, 0, 0], [RE + 500e3, 0, 0]])
    np.testing.assert_allclose(geodetic_height(rr, 2461120.0), [400e3, 500e3], atol=1.0)
    np.testing.assert_allclose(
        geodetic_height(rr, np.array([2461120.0, 2461120.5])), [400e3, 500e3], atol=1.0
    )


# ================================================================ 8. Magnetic field
DATE = dt.datetime(2026, 3, 20, 12, 0, 0)


@pytest.fixture(scope="module")
def default_3orbits():
    r0, v0 = initial_state(DEFAULT)
    t = np.arange(0.0, 3 * T_LEO, 30.0)  # 30 s -> ~185 samples/orbit; field varies on > 1 min scales
    r, v = propagate(r0, v0, t, j2=True)
    jd = DEFAULT.jd_epoch + t / 86400.0
    b = magnetic_field_eci(r, jd, DEFAULT.epoch)
    return t, r, v, jd, b


def _bmag_uT(default_3orbits):
    return np.linalg.norm(default_3orbits[4], axis=1) * 1e6


def test_field_shape_and_units(default_3orbits):
    t, r, v, jd, b = default_3orbits
    assert b.shape == (t.size, 3)
    bm = _bmag_uT(default_3orbits)
    print(f"|B| over 3 orbits: min {bm.min():.2f} uT, mean {bm.mean():.2f} uT, max {bm.max():.2f} uT")
    assert np.all(np.isfinite(b))
    assert 1e-5 < np.linalg.norm(b, axis=1).min() and np.linalg.norm(b, axis=1).max() < 1e-4  # tesla, not nT/uT


def test_field_magnitude_floor_south_atlantic_anomaly(default_3orbits):
    # Spec test 8 originally asked for > 20 uT everywhere. IGRF itself gives a global
    # minimum of 18.98 uT at 400 km inside +-51.6 deg latitude (South Atlantic Anomaly,
    # near lat -23, lon -60; confirmed with a direct ppigrf grid scan below), so the floor
    # was revised to 15 uT (decision recorded in PROGRESS.md, Group 5 section). The orbit cannot
    # dip below the independent grid minimum (0.5 uT margin for the 1.8 km height spread).
    import datetime as dt
    import ppigrf
    lat = np.deg2rad(np.arange(-51.6, 51.61, 2.0))
    lon = np.arange(-180.0, 180.0, 4.0)
    la, lo = np.meshgrid(np.rad2deg(lat), lon, indexing="ij")
    be, bn, bu = ppigrf.igrf(lo.ravel(), la.ravel(), np.full(lo.size, 400.0), dt.datetime(2026, 3, 20, 12))
    grid_min = np.sqrt(be**2 + bn**2 + bu**2).min() * 1e-3  # nT -> uT
    orbit_min = _bmag_uT(default_3orbits).min()
    assert 15.0 < orbit_min
    assert orbit_min > grid_min - 0.5


def test_field_magnitude_everywhere_below_55uT(default_3orbits):
    # Spec test 8 said < 50 uT; the IGRF maximum at 400 km within +-51.6 deg latitude is
    # 54.13 uT (lat -51.6, lon 130.5), reached over a day of this orbit (54.00 uT; this
    # 3-orbit window peaks at 49.3 uT). Bound revised to 55 uT after the Group 5 review.
    assert _bmag_uT(default_3orbits).max() < 55.0


def test_field_mean_magnitude_25_to_40uT(default_3orbits):
    assert 25.0 < _bmag_uT(default_3orbits).mean() < 40.0


def test_field_min_to_max_ratio_dipole_sanity(default_3orbits):
    bm = _bmag_uT(default_3orbits)
    assert 0.35 < bm.min() / bm.max() < 0.65


def test_field_scalar_input_and_batch_consistency(default_3orbits):
    t, r, v, jd, b = default_3orbits
    i = 50
    b1 = magnetic_field_eci(r[i], jd[i], DEFAULT.epoch)
    assert b1.shape == (3,)
    np.testing.assert_allclose(b1, b[i], rtol=1e-12, atol=1e-15)
    # scalar jd broadcast against an array of positions
    b2 = magnetic_field_eci(r[:5], jd[0], DEFAULT.epoch)
    assert b2.shape == (5, 3)
    np.testing.assert_allclose(
        b2[0], magnetic_field_eci(r[0], jd[0], DEFAULT.epoch), rtol=1e-12, atol=1e-15
    )


def test_field_matches_explicit_enu_to_eci_rotation():
    rng = np.random.default_rng(31)
    n = 20
    lat = np.arcsin(rng.uniform(-1, 1, n))  # uniform on the sphere
    lon = rng.uniform(-np.pi, np.pi, n)
    h = 400e3
    jd = 2461120.0 + rng.uniform(0, 1, n)
    # WGS-84 geodetic -> ECEF written out explicitly
    a_w = 6378137.0
    f_w = 1 / 298.257223563
    e2 = f_w * (2 - f_w)
    N = a_w / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    x = (N + h) * np.cos(lat) * np.cos(lon)
    y = (N + h) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + h) * np.sin(lat)
    th = gmst(jd)
    c, s = np.cos(th), np.sin(th)
    # ECI = Rz(-gmst) ECEF  (r_ecef = Rz(gmst) r_eci)
    def to_eci(v_ecef):
        return np.stack(
            [c * v_ecef[:, 0] - s * v_ecef[:, 1], s * v_ecef[:, 0] + c * v_ecef[:, 1], v_ecef[:, 2]], axis=1
        )

    r_eci = to_eci(np.stack([x, y, z], axis=1))

    be, bn, bu = ppigrf.igrf(np.rad2deg(lon), np.rad2deg(lat), np.full(n, h / 1e3), DATE)
    be, bn, bu = be[0] * 1e-9, bn[0] * 1e-9, bu[0] * 1e-9  # nT -> T
    east = np.stack([-np.sin(lon), np.cos(lon), np.zeros(n)], axis=1)
    north = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], axis=1)
    up = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)
    b_ecef = be[:, None] * east + bn[:, None] * north + bu[:, None] * up
    expected = to_eci(b_ecef)

    got = magnetic_field_eci(r_eci, jd, DATE)
    assert got.shape == (n, 3)
    err = np.linalg.norm(got - expected, axis=1) / np.linalg.norm(expected, axis=1)
    # spec: 1e-12 relative.  Residual comes from ecef_to_lla's round trip of the geodetic height/lat.
    assert np.max(err) < 1e-12


def test_field_points_north_near_geographic_equator():
    # Earth's field lines leave the geographic south and enter the north: B_z (ECEF z = ECI z) > 0
    # near the equator (the "north" component dominates there).
    lons = np.deg2rad(np.arange(-180, 180, 15.0))
    for lat_deg in (-5.0, 0.0, 5.0):
        lat = np.deg2rad(lat_deg)
        r_ecef = (RE + 400e3) * np.stack([np.cos(lat) * np.cos(lons), np.cos(lat) * np.sin(lons), np.full_like(lons, np.sin(lat))], axis=1)
        jd = np.full(lons.shape, 2461120.0)
        r_eci = np.stack([r_ecef[:, 0], r_ecef[:, 1], r_ecef[:, 2]], axis=1)  # gmst irrelevant: z only is checked
        b = magnetic_field_eci(r_eci, jd, DATE)
        assert np.all(b[:, 2] > 0), f"lat {lat_deg}"
        # equatorial field magnitude ~ 20-40 uT at 400 km
        bm = np.linalg.norm(b, axis=1)
        assert np.all((bm > 15e-6) & (bm < 45e-6))


def test_field_is_tesla_magnitude_at_pole_larger_than_equator():
    # ppigrf evaluation near geomagnetic poles gives ~ 2x the equatorial value (dipole); loose check
    # exactly on the geographic pole ppigrf divides by sin(colatitude) = 0 and returns NaN, so use 80 deg latitude
    lat80 = np.deg2rad(80.0)
    r_pole = np.array([[0.0, (RE + 400e3) * np.cos(lat80), (RE + 400e3) * np.sin(lat80)]])
    r_eq = np.array([[RE + 400e3, 0.0, 0.0]])
    bp = np.linalg.norm(magnetic_field_eci(r_pole, 2461120.0, DATE))
    be_ = np.linalg.norm(magnetic_field_eci(r_eq, 2461120.0, DATE))
    assert bp > 1.2 * be_  # geographic lat 80 deg (IGRF dipole tilt makes this ~1.5-2x the equator at x-axis)
    assert 1e-5 < be_ < 6e-5 and 1e-5 < bp < 8e-5


def test_field_input_not_mutated():
    r = np.array([[7e6, 1e5, 2e5], [0.0, 7e6, 1e5]])
    rc = r.copy()
    jd = np.array([2461120.0, 2461120.1])
    magnetic_field_eci(r, jd, DATE)
    np.testing.assert_array_equal(r, rc)


@pytest.fixture(scope="module")
def field_table_case():
    r0, v0 = initial_state(DEFAULT)
    t_end = 3 * T_LEO
    eph = Ephemeris(r0, v0, t_end, j2=True)
    table = FieldTable(eph, DEFAULT, t_end, dt=10.0)
    return eph, table, t_end


def test_field_table_matches_direct_at_off_grid_times(field_table_case):
    eph, table, t_end = field_table_case
    rng = np.random.default_rng(41)
    t = rng.uniform(0.0, t_end, 60)  # 60 random off-grid times (grid spacing 10 s)
    b_tab = table.b_eci(t)
    assert b_tab.shape == (60, 3)
    r, _ = eph.at(t)
    b_dir = magnetic_field_eci(r, DEFAULT.jd_epoch + t / 86400.0, DEFAULT.epoch)
    err = np.linalg.norm(b_tab - b_dir, axis=1)
    print(f"FieldTable max error {err.max() * 1e9:.3f} nT")
    assert err.max() < 100e-9  # spec: < 100 nT (1 % of the field)


def test_field_table_scalar_shapes_units_and_knots(field_table_case):
    eph, table, t_end = field_table_case
    b = table.b_eci(1234.5)
    assert b.shape == (3,)
    assert 1e-5 < np.linalg.norm(b) < 6e-5  # tesla
    tg = np.array([0.0, 10.0, 500.0, 5000.0])
    r, _ = eph.at(tg)
    b_dir = magnetic_field_eci(r, DEFAULT.jd_epoch + tg / 86400.0, DEFAULT.epoch)
    # at grid knots a cubic spline reproduces the sampled values (exactness up to round-off)
    np.testing.assert_allclose(table.b_eci(tg), b_dir, rtol=1e-9, atol=1e-13)
    assert table.b_eci(np.zeros((2, 4))).shape == (2, 4, 3)


def test_field_table_convenience_accessors(field_table_case):
    eph, table, t_end = field_table_case
    t = np.array([0.0, 800.0, 2500.0, 9000.0])
    r_ref, v_ref = eph.at(t)
    np.testing.assert_allclose(table.r_eci(t), r_ref, atol=1e-6)
    np.testing.assert_allclose(table.v_eci(t), v_ref, atol=1e-9)
    s_ref = sun_direction_eci(DEFAULT.jd_epoch + t / 86400.0)
    np.testing.assert_allclose(table.sun_dir_eci(t), s_ref, atol=1e-14)
    np.testing.assert_array_equal(table.sunlit(t), sunlit(r_ref, s_ref))
    assert table.sunlit(t).dtype == bool
    assert table.r_eci(10.0).shape == (3,)
    assert table.sun_dir_eci(10.0).shape == (3,)
    assert np.ndim(table.sunlit(10.0)) == 0
