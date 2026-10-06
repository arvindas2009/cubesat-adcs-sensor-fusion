"""Tests for the Group 8 sensor-realism additions (docs/specs/adcs_closed_loop.md, 1.1-1.3).

Written from the spec, not from the implementation.  Independent oracles:

* explicit geodetic -> ECEF -> ECI position and ENU -> ECEF -> ECI field chain around
  ``ppigrf.igrf(..., max_degree=8)`` (same construction as test_orbit_env.py)
* closed-form statistics of a first-order Gauss-Markov process (marginal std,
  autocorrelation exp(-m dt / tau)) and the exact recursion driven by an identically
  seeded standard-normal draw
* cosine-weighted ray Monte Carlo for the plate-to-sphere view factor (vectorised, seeded)
* the closed-form ``F(pi/2) = rho/pi - sqrt(H^2-1)/(pi H^2)`` and ``F(0) = (re/r)^2``
* scipy ``Rotation`` for the nadir direction in body axes (body-to-ECI quaternions)
* explicit ``where`` formulas for the sun-sensor FOV cutoff and albedo addition

Runtime: ppigrf is called at 5 points plus three one-orbit FieldTables at dt = 60 s; the
Monte Carlo uses 2e6 rays shared by all (phi, r) cases.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
import ppigrf
from scipy.spatial.transform import Rotation as Rot

from adcs import sensors as sens
from adcs.config import DEFAULT
from adcs.environment import FieldTable, magnetic_field_eci
from adcs.orbit import Ephemeris, initial_state
from adcs.sensors import CSS_NORMALS, MagnetometerConfig, SunSensorConfig
from common.frames import GM_EARTH, R_EARTH, gmst

RE = R_EARTH
MU = GM_EARTH
DATE = dt.datetime(2026, 3, 20, 12, 0, 0)
A_LEO = RE + 400e3
T_LEO = 2 * np.pi * np.sqrt(A_LEO**3 / MU)


# ------------------------------------------------------------------- helpers
def wxyz_from_rot(rot):
    """scipy Rotation -> scalar-first quaternion(s)."""
    return np.roll(rot.as_quat(), 1, axis=-1)


def rot_from_wxyz(q):
    return Rot.from_quat(np.roll(np.asarray(q, float), -1, axis=-1))


def random_unit(rng, n, dim=3):
    v = rng.normal(size=(n, dim))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def f_spec(phi, r, re=RE):
    """Plate-to-sphere view factor transcribed from spec 1.2 (independent of the implementation)."""
    phi = np.asarray(phi, float)
    r = np.asarray(r, float)
    H = r / re
    rho = np.arcsin(1.0 / H)
    x = np.sqrt(H**2 - 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        b1 = np.cos(phi) / H**2
        arg = np.clip(x / (H * np.sin(phi)), -1.0, 1.0)
        acarg = np.clip(-x / np.tan(phi), -1.0, 1.0)
        rad = np.clip(1.0 - H**2 * np.cos(phi) ** 2, 0.0, None)
        b2 = 0.5 - np.arcsin(arg) / np.pi + (np.cos(phi) * np.arccos(acarg) - x * np.sqrt(rad)) / (np.pi * H**2)
    return np.where(phi <= np.pi / 2 - rho, b1, np.where(phi < np.pi / 2 + rho, b2, 0.0))


def f_at_90deg(r, re=RE):
    """Closed form of F(pi/2) = rho/pi - sqrt(H^2-1)/(pi H^2), derived from the middle branch."""
    H = r / re
    return np.arcsin(1.0 / H) / np.pi - np.sqrt(H**2 - 1.0) / (np.pi * H**2)


# =========================================================== 1. max_degree
@pytest.fixture(scope="module")
def one_orbit_tables():
    r0, v0 = initial_state(DEFAULT)
    t_end = T_LEO
    eph = Ephemeris(r0, v0, t_end, j2=True)
    t_default = FieldTable(eph, DEFAULT, t_end, dt=60.0)
    t13 = FieldTable(eph, DEFAULT, t_end, dt=60.0, max_degree=13)
    t8 = FieldTable(eph, DEFAULT, t_end, dt=60.0, max_degree=8)
    return t_default, t13, t8, t_end


def test_field_table_default_equals_degree_13_exactly(one_orbit_tables):
    t_default, t13, _, t_end = one_orbit_tables
    t = np.linspace(0.0, t_end, 97)
    np.testing.assert_array_equal(t_default.b_eci(t), t13.b_eci(t))
    assert t_default.max_degree == 13
    assert t13.max_degree == 13


def test_field_table_degree_8_attribute_and_rms_difference(one_orbit_tables):
    _, t13, t8, t_end = one_orbit_tables
    assert t8.max_degree == 8
    t = np.arange(0.0, t_end, 60.0)
    diff = t8.b_eci(t) - t13.b_eci(t)
    rms_nt = np.sqrt(np.mean(np.sum(diff**2, axis=1))) * 1e9
    print(f"degree 8 vs 13 over one orbit: rms {rms_nt:.1f} nT")
    assert 20.0 <= rms_nt <= 200.0  # spec band; prototype 67 nT over 3 orbits


def test_magnetic_field_eci_max_degree_matches_independent_chain():
    rng = np.random.default_rng(1301)
    n = 5
    lat = np.arcsin(rng.uniform(-1, 1, n))
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

    def to_eci(v):  # r_ecef = Rz(gmst) r_eci  =>  r_eci = Rz(-gmst) r_ecef
        return np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2]], axis=1)

    r_eci = to_eci(np.stack([x, y, z], axis=1))

    def chain(max_degree):
        be, bn, bu = ppigrf.igrf(np.rad2deg(lon), np.rad2deg(lat), np.full(n, h / 1e3), DATE, max_degree=max_degree)
        be, bn, bu = be[0] * 1e-9, bn[0] * 1e-9, bu[0] * 1e-9  # nT -> T
        east = np.stack([-np.sin(lon), np.cos(lon), np.zeros(n)], axis=1)
        north = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], axis=1)
        up = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)
        return to_eci(be[:, None] * east + bn[:, None] * north + bu[:, None] * up)

    exp8, exp13 = chain(8), chain(13)
    got8 = magnetic_field_eci(r_eci, jd, DATE, max_degree=8)
    got_default = magnetic_field_eci(r_eci, jd, DATE)
    got13 = magnetic_field_eci(r_eci, jd, DATE, max_degree=13)
    assert got8.shape == (n, 3)
    np.testing.assert_allclose(got8, exp8, rtol=0.0, atol=1e-12)  # spec: 1e-12 T
    np.testing.assert_allclose(got13, exp13, rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(got_default, got13)
    # the keyword must actually change the answer (degree 9-13 content is ~tens of nT)
    assert np.max(np.abs(got8 - got13)) > 1e-9


# ========================================================= 2. magnetometer bias
def test_magnetometer_bias_exact_difference_identity():
    rng_b = np.random.default_rng(2)
    b = rng_b.normal(size=(50, 3)) * 3e-5
    beta = np.array([2.0e-7, -1.5e-7, 3.1e-7])
    out_bias = sens.magnetometer(b, np.random.default_rng(77), bias=beta)
    out_none = sens.magnetometer(b, np.random.default_rng(77))
    assert out_bias.shape == b.shape
    # identical noise draw => difference is beta; a few ulps of a 3e-5 T number (~7e-21 T)
    # are lost to floating-point addition order, hence 1e-19 T instead of bit equality
    np.testing.assert_allclose(out_bias - out_none, np.broadcast_to(beta, b.shape), rtol=0.0, atol=1e-19)
    # single vector, list bias
    b1 = np.array([1e-5, -2e-5, 3e-5])
    d = sens.magnetometer(b1, np.random.default_rng(5), bias=[1e-7, 2e-7, -3e-7]) - sens.magnetometer(
        b1, np.random.default_rng(5)
    )
    np.testing.assert_allclose(d, [1e-7, 2e-7, -3e-7], rtol=0.0, atol=1e-19)


def test_magnetometer_bias_none_and_zero_are_the_old_behaviour():
    b = np.random.default_rng(3).normal(size=(20, 3)) * 3e-5
    a0 = sens.magnetometer(b, np.random.default_rng(9))
    a1 = sens.magnetometer(b, np.random.default_rng(9), bias=None)
    a2 = sens.magnetometer(b, np.random.default_rng(9), bias=np.zeros(3))
    np.testing.assert_array_equal(a0, a1)
    np.testing.assert_allclose(a2, a0, rtol=0.0, atol=1e-19)
    # explicit Group 7 formula
    expected = b + np.random.default_rng(9).normal(0.0, MagnetometerConfig().sigma, size=b.shape)
    np.testing.assert_array_equal(a0, expected)


def test_magnetometer_bias_does_not_mutate_inputs():
    b = np.array([[1e-5, 2e-5, 3e-5]] * 3)
    beta = np.array([1e-7, 1e-7, 1e-7])
    bc, betac = b.copy(), beta.copy()
    sens.magnetometer(b, np.random.default_rng(0), bias=beta)
    np.testing.assert_array_equal(b, bc)
    np.testing.assert_array_equal(beta, betac)


def test_magnetometer_sample_mean_is_field_plus_bias():
    n = 20000
    b0 = np.array([1.2e-5, -2.3e-5, 3.4e-5])
    beta = np.array([2.0e-7, -3.0e-7, 1.0e-7])
    cfg = MagnetometerConfig()
    out = sens.magnetometer(np.tile(b0, (n, 1)), np.random.default_rng(123), cfg=cfg, bias=beta)
    err = out.mean(axis=0) - (b0 + beta)
    assert np.all(np.abs(err) < 4.0 * cfg.sigma / np.sqrt(n))
    # and the sample std recovers sigma (so the bias did not touch the noise)
    assert out.std(axis=0, ddof=1) == pytest.approx(np.full(3, cfg.sigma), rel=0.05)


# ============================================================= 3. gauss_markov
def test_gauss_markov_shape_dtype_and_default_dim():
    x = sens.gauss_markov(100, 1.0, 2.0, 30.0, np.random.default_rng(0))
    assert x.shape == (100, 3)
    assert x.dtype == np.float64
    x5 = sens.gauss_markov(7, 0.5, 2.0, 30.0, np.random.default_rng(0), dim=5)
    assert x5.shape == (7, 5)
    assert sens.gauss_markov(1, 1.0, 1.0, 10.0, np.random.default_rng(0)).shape == (1, 3)
    assert np.all(np.isfinite(x))


def test_gauss_markov_reproducible_and_seed_sensitive():
    a = sens.gauss_markov(500, 1.0, 1.0, 30.0, np.random.default_rng(42))
    b = sens.gauss_markov(500, 1.0, 1.0, 30.0, np.random.default_rng(42))
    c = sens.gauss_markov(500, 1.0, 1.0, 30.0, np.random.default_rng(43))
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_gauss_markov_exact_recursion_from_one_standard_normal_draw():
    n, dim, dt_, sigma, tau = 300, 3, 2.0, 5e-8, 600.0
    x = sens.gauss_markov(n, dt_, sigma, tau, np.random.default_rng(8), dim=dim)
    w = np.random.default_rng(8).standard_normal((n, dim))  # spec: one call, row k <-> w_k
    a = np.exp(-dt_ / tau)
    ref = np.empty((n, dim))
    ref[0] = sigma * w[0]
    for k in range(1, n):
        ref[k] = a * ref[k - 1] + sigma * np.sqrt(1.0 - a * a) * w[k]
    np.testing.assert_allclose(x, ref, rtol=1e-12, atol=1e-12 * sigma)


def test_gauss_markov_sigma_zero_gives_zeros_without_drawing():
    rng = np.random.default_rng(5)
    state_before = rng.bit_generator.state
    x = sens.gauss_markov(50, 1.0, 0.0, 30.0, rng)
    assert x.shape == (50, 3)
    np.testing.assert_array_equal(x, np.zeros((50, 3)))
    assert rng.bit_generator.state == state_before
    nxt = rng.standard_normal()
    assert nxt == np.random.default_rng(5).standard_normal()


@pytest.fixture(scope="module")
def gm_ensemble():
    """4000 independent series of length 2000 (columns), sigma = 2.5, dt = 1, tau = 30."""
    sigma, dt_, tau = 2.5, 1.0, 30.0
    x = sens.gauss_markov(2000, dt_, sigma, tau, np.random.default_rng(20260101), dim=4000)
    return x, sigma, dt_, tau


def test_gauss_markov_marginal_std_is_stationary(gm_ensemble):
    # The spec states 400 series within 3 %.  The sampling error of a std from M Gaussian
    # samples is 1/sqrt(2M): 3.5 % for M = 400 (a 3 % bound would be a coin flip), so M = 4000
    # (1.1 %, the 3 % bound is 2.7 sigma) is used with the spec tolerance unchanged.
    x, sigma, _, _ = gm_ensemble
    assert x.shape == (2000, 4000)
    for k in (0, 1999):
        assert np.std(x[k], ddof=1) == pytest.approx(sigma, rel=0.03), f"k={k}"
    assert np.mean(x[0]) == pytest.approx(0.0, abs=5 * sigma / np.sqrt(x.shape[1]))
    # more power: the std pooled over a block of late rows (stationarity: same sigma everywhere)
    assert np.sqrt(np.mean(x[1000:1200] ** 2)) == pytest.approx(sigma, rel=0.01)


@pytest.mark.parametrize("m", [1, 10, 50])
def test_gauss_markov_autocorrelation(gm_ensemble, m):
    x, sigma, dt_, tau = gm_ensemble
    rho_hat = np.mean(x[:-m] * x[m:]) / sigma**2
    assert rho_hat == pytest.approx(np.exp(-m * dt_ / tau), abs=0.03)


def test_gauss_markov_dim_columns_are_uncorrelated(gm_ensemble):
    x, sigma, _, _ = gm_ensemble
    c = np.mean(x[:, 0:1000] * x[:, 1000:2000]) / sigma**2
    assert abs(c) < 0.03


def test_gauss_markov_tau_limits():
    # tau >> n dt: series almost constant; tau << dt: white (lag-1 correlation ~ 0)
    long_tau = sens.gauss_markov(200, 1.0, 1.0, 1e9, np.random.default_rng(1), dim=2)
    assert np.max(np.abs(np.diff(long_tau, axis=0))) < 1e-3
    white = sens.gauss_markov(20000, 1.0, 1.0, 1e-3, np.random.default_rng(1), dim=1)[:, 0]
    assert abs(np.mean(white[:-1] * white[1:])) < 0.05


@pytest.mark.parametrize(
    "args",
    [
        (0, 1.0, 1.0, 30.0),
        (-3, 1.0, 1.0, 30.0),
        (10, 0.0, 1.0, 30.0),
        (10, -1.0, 1.0, 30.0),
        (10, 1.0, 1.0, 0.0),
        (10, 1.0, 1.0, -5.0),
        (10, 1.0, -1.0, 30.0),
    ],
)
def test_gauss_markov_invalid_arguments_raise(args):
    with pytest.raises(ValueError):
        sens.gauss_markov(*args, np.random.default_rng(0))


# ================================================================= 4. CSS FOV
def _direct_currents(s, lit, fov=None):
    c = s @ CSS_NORMALS.T
    if fov is None:
        cut = np.maximum(c, 0.0)
    else:
        cut = np.where(c > np.cos(fov), c, 0.0)
    return np.where(lit[:, None], cut, 0.0)


def test_css_fov_noise_free_matches_cutoff_formula():
    rng = np.random.default_rng(401)
    s = random_unit(rng, 2000)
    lit = np.ones(2000, dtype=bool)
    fov = np.deg2rad(60.0)
    cfg = SunSensorConfig(sigma_diode=0.0, fov=fov)
    got = sens.sun_sensor_currents(s, lit, np.random.default_rng(0), cfg)
    assert got.shape == (2000, 6)
    c = s @ CSS_NORMALS.T
    np.testing.assert_allclose(got, np.where(c > np.cos(fov), c, 0.0), rtol=0.0, atol=1e-15)
    # the test is only meaningful if the cutoff removed something that cosine response keeps
    assert np.sum((c > 0) & (c <= np.cos(fov))) > 500
    assert np.all(got[(c > 0) & (c <= np.cos(fov))] == 0.0)
    # surviving currents are all above cos(fov)
    assert np.all(got[got != 0.0] > np.cos(fov))


@pytest.mark.parametrize("fov_deg", [30.0, 45.0, 60.0, 80.0])
def test_css_fov_random_lit_mask_and_angles(fov_deg):
    rng = np.random.default_rng(402)
    s = random_unit(rng, 2000)
    lit = rng.uniform(size=2000) < 0.6
    fov = np.deg2rad(fov_deg)
    cfg = SunSensorConfig(sigma_diode=0.0, fov=fov)
    got = sens.sun_sensor_currents(s, lit, np.random.default_rng(1), cfg)
    np.testing.assert_allclose(got, _direct_currents(s, lit, fov), rtol=0.0, atol=1e-15)
    assert np.all(got[~lit] == 0.0)


def test_css_fov_none_reproduces_group7_function_exactly_with_noise():
    rng = np.random.default_rng(403)
    s = random_unit(rng, 500)
    lit = rng.uniform(size=500) < 0.7
    cfg = SunSensorConfig()  # default sigma_diode, fov None
    assert cfg.fov is None
    got = sens.sun_sensor_currents(s, lit, np.random.default_rng(11), cfg)
    expected = _direct_currents(s, lit) + np.random.default_rng(11).normal(0.0, cfg.sigma_diode, size=(500, 6))
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-15)
    # default argument (no cfg) behaves identically
    np.testing.assert_array_equal(got, sens.sun_sensor_currents(s, lit, np.random.default_rng(11)))


def test_css_fov_half_pi_equals_no_fov_noise_free():
    rng = np.random.default_rng(404)
    s = random_unit(rng, 2000)
    lit = rng.uniform(size=2000) < 0.8
    a = sens.sun_sensor_currents(s, lit, np.random.default_rng(0), SunSensorConfig(sigma_diode=0.0, fov=np.pi / 2))
    b = sens.sun_sensor_currents(s, lit, np.random.default_rng(0), SunSensorConfig(sigma_diode=0.0))
    np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-15)


def test_css_fov_eclipse_all_zero_noise_free():
    s = random_unit(np.random.default_rng(405), 300)
    lit = np.zeros(300, dtype=bool)
    for fov in (None, np.deg2rad(60.0), np.pi / 2):
        got = sens.sun_sensor_currents(s, lit, np.random.default_rng(0), SunSensorConfig(sigma_diode=0.0, fov=fov))
        np.testing.assert_array_equal(got, np.zeros((300, 6)))


def test_css_fov_noise_draw_unchanged_by_fov():
    rng = np.random.default_rng(406)
    s = random_unit(rng, 400)
    lit = rng.uniform(size=400) < 0.75
    fov = np.deg2rad(60.0)
    with_fov = sens.sun_sensor_currents(s, lit, np.random.default_rng(3), SunSensorConfig(fov=fov))
    without = sens.sun_sensor_currents(s, lit, np.random.default_rng(3), SunSensorConfig())
    c = s @ CSS_NORMALS.T
    removed = np.where(lit[:, None] & (c > 0) & (c <= np.cos(fov)), c, 0.0)
    np.testing.assert_allclose(without - with_fov, removed, rtol=0.0, atol=1e-14)


def test_css_fov_scalar_single_sample_shape():
    out = sens.sun_sensor_currents(
        np.array([0.0, 0.0, 1.0]), np.array(True), np.random.default_rng(0), SunSensorConfig(sigma_diode=0.0, fov=1.0)
    )
    assert out.shape == (6,)
    np.testing.assert_allclose(out, [0, 0, 0, 0, 1.0, 0], atol=1e-15)
    # 70 deg incidence on +x, fov 60 deg: cut; fov None: 0.342 kept
    s = np.array([np.cos(np.deg2rad(70.0)), np.sin(np.deg2rad(70.0)), 0.0])
    cut = sens.sun_sensor_currents(s, np.array(True), np.random.default_rng(0), SunSensorConfig(sigma_diode=0.0, fov=np.deg2rad(60)))
    keep = sens.sun_sensor_currents(s, np.array(True), np.random.default_rng(0), SunSensorConfig(sigma_diode=0.0))
    assert cut[0] == 0.0 and keep[0] == pytest.approx(np.cos(np.deg2rad(70.0)), abs=1e-15)
    assert cut[2] == pytest.approx(np.sin(np.deg2rad(70.0)), abs=1e-15)  # +y at 20 deg incidence is inside


# ========================================================= 5. earth_view_factor
R400 = RE + 400e3
R1000 = RE + 1000e3


@pytest.mark.parametrize("r", [R400, R1000, RE + 36e3, 2.0 * RE])
def test_view_factor_zero_angle_is_inverse_square(r):
    F = sens.earth_view_factor(0.0, r)
    assert float(F) == pytest.approx((RE / r) ** 2, rel=1e-14)


def test_view_factor_custom_re_and_default_re():
    assert float(sens.earth_view_factor(0.0, 9.0e6, re=6.0e6)) == pytest.approx((6.0e6 / 9.0e6) ** 2, rel=1e-14)
    np.testing.assert_array_equal(sens.earth_view_factor(0.7, R400), sens.earth_view_factor(0.7, R400, re=RE))


@pytest.mark.parametrize("r", [R400, R1000, 1.5 * RE])
def test_view_factor_branches_agree_at_first_boundary(r):
    H = r / RE
    rho = np.arcsin(1.0 / H)
    phi_b = np.pi / 2 - rho
    first = np.cos(phi_b) / H**2
    # NB: the middle-branch formula cannot be evaluated *at* phi_b in double precision to 1e-12
    # (sqrt(1 - H^2 cos^2 phi) is the square root of round-off, ~1e-8 error), so the test-side
    # transcription is compared just inside the branch, where the implementation must match it
    # to 1e-12 and where it must be within slope*delta of the branch-1 value.
    delta = 1e-4
    inside = float(sens.earth_view_factor(phi_b + delta, r))
    assert inside == pytest.approx(float(f_spec(phi_b + delta, r)), abs=1e-12)
    assert abs(inside - first) < 1.0 * delta  # |dF/dphi| <= 1 (F <= 1, F(0) - F(pi/2) < 1 over pi/2)
    assert float(sens.earth_view_factor(phi_b, r)) == pytest.approx(first, abs=1e-12)
    # continuity of the implementation across the boundary (derivative |dF/dphi| < 1 => 1e-9 for 2e-10 spread)
    eps = 1e-10
    lo = float(sens.earth_view_factor(phi_b - eps, r))
    hi = float(sens.earth_view_factor(phi_b + eps, r))
    assert abs(hi - lo) < 1e-9


@pytest.mark.parametrize("r", [R400, R1000])
def test_view_factor_middle_branch_matches_spec_transcription(r):
    H = r / RE
    rho = np.arcsin(1.0 / H)
    frac = np.linspace(0.1, 0.9, 17)
    phi = np.pi / 2 - rho + frac * 2 * rho
    got = sens.earth_view_factor(phi, r)
    np.testing.assert_allclose(got, f_spec(phi, r), rtol=0.0, atol=1e-12)
    # branch 1 region too
    phi1 = np.linspace(0.0, np.pi / 2 - rho, 20)
    np.testing.assert_allclose(sens.earth_view_factor(phi1, r), np.cos(phi1) / H**2, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("r", [R400, R1000])
def test_view_factor_exactly_zero_beyond_horizon(r):
    H = r / RE
    rho = np.arcsin(1.0 / H)
    phi_b = np.pi / 2 + rho
    assert float(sens.earth_view_factor(phi_b, r)) == pytest.approx(0.0, abs=1e-12)
    beyond = np.array([phi_b + 1e-6, phi_b + 0.1, 2.5, np.pi - 1e-9, np.pi])
    beyond = beyond[beyond >= phi_b]
    np.testing.assert_array_equal(sens.earth_view_factor(beyond, r), np.zeros(beyond.shape))


@pytest.mark.parametrize("r", [R400, R1000])
def test_view_factor_is_non_increasing_and_bounded(r):
    phi = np.linspace(0.0, np.pi, 3001)
    F = sens.earth_view_factor(phi, r)
    assert F.shape == phi.shape
    assert np.all(np.isfinite(F))
    assert np.all(F >= -1e-15) and np.all(F <= (RE / r) ** 2 + 1e-15)
    # round-off scale of the formula ~1e-16 per evaluation
    assert np.max(np.diff(F)) <= 1e-13
    assert F[0] > F[1500] > F[-1] - 1e-15


def test_view_factor_closed_form_at_90_degrees():
    for r in (R400, R1000, 3 * RE):
        assert float(sens.earth_view_factor(np.pi / 2, r)) == pytest.approx(f_at_90deg(r), abs=1e-12)


def test_view_factor_published_prototype_values_400km():
    # spec prototype values (4 decimals) at 400 km: rounding error <= 5e-5, so 1e-4
    phi = np.deg2rad([0.0, 30.0, 60.0, 90.0, 110.0])
    expected = [0.8855, 0.7712, 0.5372, 0.2887, 0.1495]
    np.testing.assert_allclose(sens.earth_view_factor(phi, R400), expected, rtol=0.0, atol=1e-4)


def test_view_factor_broadcasting():
    phi = np.deg2rad(np.array([0.0, 30.0, 90.0, 120.0]))[:, None]
    r = np.array([R400, R1000, 2 * RE])[None, :]
    F = sens.earth_view_factor(phi, r)
    assert F.shape == (4, 3)
    for i in range(4):
        for j in range(3):
            assert F[i, j] == pytest.approx(float(sens.earth_view_factor(phi[i, 0], r[0, j])), abs=1e-15)
    assert np.ndim(sens.earth_view_factor(0.3, R400)) == 0 or np.shape(sens.earth_view_factor(0.3, R400)) == ()


@pytest.mark.parametrize("bad", [RE, RE - 1.0, 0.5 * RE, 0.0, -7e6])
def test_view_factor_invalid_r_norm_raises(bad):
    with pytest.raises(ValueError):
        sens.earth_view_factor(0.1, bad)
    with pytest.raises(ValueError):
        sens.earth_view_factor(np.array([0.1, 0.2]), np.array([R400, bad]))


@pytest.fixture(scope="module")
def cosine_rays():
    """2e6 cosine-weighted (pdf cos(theta)/pi) directions in the hemisphere about the plate normal (z)."""
    rng = np.random.default_rng(20260202)
    n = 2_000_000
    u1 = rng.uniform(size=n)
    u2 = rng.uniform(size=n)
    st = np.sqrt(u1)
    ct = np.sqrt(1.0 - u1)
    az = 2 * np.pi * u2
    return np.stack([st * np.cos(az), st * np.sin(az), ct], axis=1)


@pytest.mark.parametrize("r", [R400, R1000])
@pytest.mark.parametrize("phi_deg", [0.0, 20.0, 45.0, 70.0, 90.0, 105.0])
def test_view_factor_matches_cosine_weighted_monte_carlo(cosine_rays, r, phi_deg):
    phi = np.deg2rad(phi_deg)
    d = np.array([np.sin(phi), 0.0, np.cos(phi)])  # direction from the plate to the sphere centre
    half_angle = np.arcsin(RE / r)  # a ray hits iff it is within this angle of d
    hit = cosine_rays @ d > np.cos(half_angle)
    F_mc = hit.mean()
    # binomial std <= 0.5/sqrt(2e6) = 3.5e-4; spec tolerance 0.003 is > 8 sigma
    assert float(sens.earth_view_factor(phi, r)) == pytest.approx(F_mc, abs=0.003)


# ============================================================ 6. albedo_currents
def _geometry(rng, n, r_norm=A_LEO):
    rhat = random_unit(rng, n)
    q = random_unit(rng, n, dim=4)
    return q, rhat * r_norm, rhat


def _sun_with_cos(rng, rhat, cos_zen):
    """Unit sun vectors with s . rhat = cos_zen (array of shape (n,))."""
    cz = np.asarray(cos_zen, float)[:, None]
    t = random_unit(rng, rhat.shape[0])
    t = t - np.sum(t * rhat, axis=1, keepdims=True) * rhat
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return cz * rhat + np.sqrt(1.0 - cz**2) * t


def test_albedo_zero_when_sun_not_above_horizon():
    rng = np.random.default_rng(601)
    n = 300
    q, r, rhat = _geometry(rng, n)
    cz = rng.uniform(-1.0, 0.0, n)
    s = _sun_with_cos(rng, rhat, cz)
    out = sens.albedo_currents(q, r, s, 0.3)
    assert out.shape == (n, 6)
    np.testing.assert_array_equal(out, np.zeros((n, 6)))
    # exactly perpendicular (s . rhat = 0 up to round-off)
    s0 = _sun_with_cos(rng, rhat, np.zeros(n))
    out0 = sens.albedo_currents(q, r, s0, 0.3)
    np.testing.assert_allclose(out0, 0.0, atol=1e-14)
    # and positive somewhere when the sun is above the horizon (the zero above is not vacuous)
    s1 = _sun_with_cos(rng, rhat, np.full(n, 0.5))
    assert np.max(sens.albedo_currents(q, r, s1, 0.3)) > 0.05


def _aligned_body(rng, rhat):
    """Quaternion (body-to-ECI) whose body +z points to nadir (-rhat), random roll about z."""
    align, _ = Rot.align_vectors(-rhat[None, :], np.array([[0.0, 0.0, 1.0]]))
    roll = Rot.from_rotvec([0.0, 0.0, rng.uniform(0, 2 * np.pi)])
    rot = align * roll
    np.testing.assert_allclose(rot.apply([0, 0, 1.0]), -rhat, atol=1e-12)
    return wxyz_from_rot(rot)


@pytest.mark.parametrize("r_norm", [A_LEO, R1000])
def test_albedo_sun_at_zenith_body_z_to_nadir_closed_forms(r_norm):
    rng = np.random.default_rng(602)
    albedo = 0.3
    for _ in range(5):
        rhat = random_unit(rng, 1)[0]
        q = _aligned_body(rng, rhat)
        out = sens.albedo_currents(q, r_norm * rhat, rhat, albedo)
        assert out.shape == (6,)
        # order +x, -x, +y, -y, +z, -z
        assert out[4] == pytest.approx(albedo * (RE / r_norm) ** 2, rel=1e-12)
        assert out[5] == pytest.approx(0.0, abs=1e-14)
        np.testing.assert_allclose(out[:4], albedo * f_at_90deg(r_norm), rtol=1e-11)


def test_albedo_custom_re_is_used():
    rng = np.random.default_rng(603)
    rhat = random_unit(rng, 1)[0]
    q = _aligned_body(rng, rhat)
    out = sens.albedo_currents(q, 9.0e6 * rhat, rhat, 0.4, re=6.0e6)
    assert out[4] == pytest.approx(0.4 * (6.0e6 / 9.0e6) ** 2, rel=1e-12)
    np.testing.assert_allclose(out[:4], 0.4 * f_at_90deg(9.0e6, re=6.0e6), rtol=1e-11)


def test_albedo_matches_scipy_nadir_and_spec_formula_random_batch():
    rng = np.random.default_rng(604)
    n = 120
    rhat = random_unit(rng, n)
    r_norm = rng.uniform(RE + 300e3, RE + 1200e3, n)
    r = rhat * r_norm[:, None]
    q = random_unit(rng, n, dim=4)
    s = _sun_with_cos(rng, rhat, rng.uniform(-0.3, 1.0, n))
    albedo = rng.uniform(0.05, 0.6, n)
    got = sens.albedo_currents(q, r, s, albedo)
    assert got.shape == (n, 6)
    # nadir in body axes: R(q)^T (-rhat), via scipy (body-to-ECI rotation)
    nadir_b = rot_from_wxyz(q).inv().apply(-rhat)
    np.testing.assert_allclose(np.linalg.norm(nadir_b, axis=1), 1.0, atol=1e-13)
    cos_phi = np.clip(nadir_b @ CSS_NORMALS.T, -1.0, 1.0)
    phi = np.arccos(cos_phi)
    illum = np.maximum(0.0, np.sum(s * rhat, axis=1))
    expected = albedo[:, None] * f_spec(phi, r_norm[:, None]) * illum[:, None]
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)
    assert np.all(got >= 0.0)
    # the sun-below-horizon samples are exactly zero, the others generally not
    below = np.sum(s * rhat, axis=1) <= 0
    assert below.any() and (~below).any()
    assert np.all(got[below] == 0.0)
    assert np.all(got[~below].max(axis=1) > 0.0)


def test_albedo_diode_faces_towards_earth_see_more_than_faces_away():
    # a body-axis-aligned attitude with identity quaternion: nadir_b = -rhat ECI components
    rhat = np.array([0.0, 0.0, 1.0])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    out = sens.albedo_currents(q, A_LEO * rhat, rhat, 0.3)  # nadir = -z body
    assert out[5] == pytest.approx(0.3 * (RE / A_LEO) ** 2, rel=1e-12)
    assert out[4] == pytest.approx(0.0, abs=1e-14)
    # 180 deg about x: +z body -> -z ECI = nadir
    q2 = np.array([0.0, 1.0, 0.0, 0.0])
    out2 = sens.albedo_currents(q2, A_LEO * rhat, rhat, 0.3)
    assert out2[4] == pytest.approx(0.3 * (RE / A_LEO) ** 2, rel=1e-12)
    assert out2[5] == pytest.approx(0.0, abs=1e-14)
    # a 90 deg rotation about +y (body x -> ECI -z): R(q) x_b = -z, so +x sees nadir
    q3 = wxyz_from_rot(Rot.from_rotvec([0.0, np.pi / 2, 0.0]))
    out3 = sens.albedo_currents(q3, A_LEO * rhat, rhat, 0.3)
    assert out3[0] == pytest.approx(0.3 * (RE / A_LEO) ** 2, rel=1e-12)
    assert out3[1] == pytest.approx(0.0, abs=1e-14)


def test_albedo_linear_in_albedo_and_array_albedo():
    rng = np.random.default_rng(605)
    n = 40
    q, r, rhat = _geometry(rng, n)
    s = _sun_with_cos(rng, rhat, rng.uniform(0.1, 1.0, n))
    a1 = sens.albedo_currents(q, r, s, 0.1)
    a3 = sens.albedo_currents(q, r, s, 0.3)
    np.testing.assert_allclose(a3, 3.0 * a1, rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(sens.albedo_currents(q, r, s, 0.0), np.zeros((n, 6)))
    al = rng.uniform(0.05, 0.9, n)
    arr = sens.albedo_currents(q, r, s, al)
    assert arr.shape == (n, 6)
    np.testing.assert_allclose(arr, al[:, None] * a1 / 0.1, rtol=1e-12, atol=1e-15)
    # additivity in albedo
    np.testing.assert_allclose(
        sens.albedo_currents(q, r, s, 0.1 + 0.25), a1 + sens.albedo_currents(q, r, s, 0.25), rtol=1e-12, atol=1e-15
    )


def test_albedo_batch_shapes_and_single_sample_consistency():
    rng = np.random.default_rng(606)
    n = 8
    q, r, rhat = _geometry(rng, n)
    s = _sun_with_cos(rng, rhat, rng.uniform(0.2, 1.0, n))
    batch = sens.albedo_currents(q, r, s, 0.3)
    assert batch.shape == (n, 6)
    for i in range(n):
        one = sens.albedo_currents(q[i], r[i], s[i], 0.3)
        assert one.shape == (6,)
        np.testing.assert_allclose(one, batch[i], rtol=1e-13, atol=1e-15)
    # leading batch dims (2, 4)
    b2 = sens.albedo_currents(q.reshape(2, 4, 4), r.reshape(2, 4, 3), s.reshape(2, 4, 3), 0.3)
    assert b2.shape == (2, 4, 6)
    np.testing.assert_allclose(b2.reshape(8, 6), batch, rtol=1e-13, atol=1e-15)


def test_albedo_does_not_mutate_inputs():
    rng = np.random.default_rng(607)
    q, r, rhat = _geometry(rng, 5)
    s = _sun_with_cos(rng, rhat, np.full(5, 0.7))
    qc, rc, sc = q.copy(), r.copy(), s.copy()
    sens.albedo_currents(q, r, s, 0.3)
    np.testing.assert_array_equal(q, qc)
    np.testing.assert_array_equal(r, rc)
    np.testing.assert_array_equal(s, sc)


def test_albedo_eclipse_geometry_is_zero():
    # satellite directly behind the Earth relative to the Sun: s . rhat = -1
    rng = np.random.default_rng(608)
    q = random_unit(rng, 10, dim=4)
    rhat = np.tile([-1.0, 0.0, 0.0], (10, 1))
    s = np.tile([1.0, 0.0, 0.0], (10, 1))
    np.testing.assert_array_equal(sens.albedo_currents(q, A_LEO * rhat, s, 0.3), np.zeros((10, 6)))


# ================================================ 7. sun_sensor_currents(albedo=...)
def test_css_albedo_noise_free_equals_direct_plus_albedo():
    rng = np.random.default_rng(701)
    n = 500
    s = random_unit(rng, n)
    lit = rng.uniform(size=n) < 0.6
    alb = rng.uniform(0.0, 0.3, size=(n, 6))
    for fov in (None, np.deg2rad(60.0)):
        cfg = SunSensorConfig(sigma_diode=0.0, fov=fov)
        got = sens.sun_sensor_currents(s, lit, np.random.default_rng(0), cfg, albedo=alb)
        np.testing.assert_allclose(got, _direct_currents(s, lit, fov) + alb, rtol=0.0, atol=1e-15)
        # albedo is added regardless of lit and fov: eclipsed samples return exactly the albedo
        np.testing.assert_allclose(got[~lit], alb[~lit], rtol=0.0, atol=1e-15)


def test_css_albedo_none_is_unchanged_behaviour():
    rng = np.random.default_rng(702)
    s = random_unit(rng, 100)
    lit = rng.uniform(size=100) < 0.5
    a = sens.sun_sensor_currents(s, lit, np.random.default_rng(4), albedo=None)
    b = sens.sun_sensor_currents(s, lit, np.random.default_rng(4))
    np.testing.assert_array_equal(a, b)


def test_css_albedo_with_noise_same_seed_difference_is_albedo():
    rng = np.random.default_rng(703)
    n = 400
    s = random_unit(rng, n)
    lit = rng.uniform(size=n) < 0.6
    alb = rng.uniform(0.0, 0.3, size=(n, 6))
    for fov in (None, np.deg2rad(60.0)):
        cfg = SunSensorConfig(fov=fov)
        with_a = sens.sun_sensor_currents(s, lit, np.random.default_rng(21), cfg, albedo=alb)
        without = sens.sun_sensor_currents(s, lit, np.random.default_rng(21), cfg)
        # values are O(0.1), so rounding is ~1e-16
        np.testing.assert_allclose(with_a - without, alb, rtol=0.0, atol=1e-14)


def test_css_albedo_integration_with_albedo_currents_in_eclipse_free_geometry():
    # end-to-end of the two new pieces: a lit sample with the sun at the zenith and body +z to nadir
    rng = np.random.default_rng(704)
    rhat = random_unit(rng, 1)[0]
    q = _aligned_body(rng, rhat)
    alb = sens.albedo_currents(q, A_LEO * rhat, rhat, 0.3)
    s_body = rot_from_wxyz(q).inv().apply(rhat)  # sun at zenith -> -z body (away from the Earth-facing +z face)
    cfg = SunSensorConfig(sigma_diode=0.0, fov=np.deg2rad(60.0))
    out = sens.sun_sensor_currents(s_body, np.array(True), np.random.default_rng(0), cfg, albedo=alb)
    expected = np.zeros(6)
    expected[5] = 1.0  # direct sun on -z
    np.testing.assert_allclose(out, expected + alb, rtol=0.0, atol=1e-12)
