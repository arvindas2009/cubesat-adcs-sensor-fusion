"""Tests for ``sensor_fusion.dead_reckoning`` written from
``docs/specs/sensor_fusion_data.md`` section 4 (tests 1-6) and the
``initial_attitude_from_accel_mag`` helper description.

Independent oracles: ``scipy.spatial.transform.Rotation`` and closed-form
kinematics.  Conventions (CONVENTIONS.md): scalar-first Hamilton quaternions,
body-to-NED, specific force ``f_b = R(q)^T (a_n - g_n) + b_a``,
``g_n = [0, 0, +G0]``.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from common.frames import G0
from sensor_fusion.dead_reckoning import (
    DeadReckoningResult,
    initial_attitude_from_accel_mag,
    strapdown,
)

G_NED = np.array([0.0, 0.0, G0])


# --------------------------------------------------------------------------
# helpers (all built on scipy, not on the code under test)
# --------------------------------------------------------------------------
def to_scipy(q):
    """Scalar-first [w,x,y,z] (batched) -> scipy Rotation."""
    q = np.asarray(q, dtype=float)
    return Rotation.from_quat(q[..., [1, 2, 3, 0]])


def from_scipy(r):
    """scipy Rotation -> scalar-first quaternion array."""
    return r.as_quat()[..., [3, 0, 1, 2]]


def quat_err(a, b):
    """Sign-invariant max-norm distance between two (batched) quaternions."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.minimum(np.linalg.norm(a - b, axis=-1), np.linalg.norm(a + b, axis=-1))


def tile(v, n):
    return np.tile(np.asarray(v, dtype=float), (n, 1))


def specific_force(q_series, a_n, g_n=G_NED):
    """f_b = R(q)^T (a_n - g_n), via scipy (reference -> body)."""
    a_n = np.broadcast_to(np.asarray(a_n, dtype=float), (len(q_series), 3))
    return to_scipy(q_series).inv().apply(a_n - g_n)


def random_unit_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


# --------------------------------------------------------------------------
# result container / shapes
# --------------------------------------------------------------------------
def test_result_shapes_and_initial_values():
    n = 50
    t = np.linspace(0.0, 1.0, n)
    q0 = np.array([0.5, 0.5, -0.5, 0.5])
    v0 = np.array([1.0, -2.0, 0.5])
    res = strapdown(t, np.zeros((n, 3)), np.zeros((n, 3)), q0, v0=v0, g_ned=(0, 0, 0))
    assert isinstance(res, DeadReckoningResult)
    assert res.t.shape == (n,)
    assert res.q.shape == (n, 4)
    assert res.v.shape == (n, 3)
    assert res.p.shape == (n, 3)
    np.testing.assert_allclose(res.t, t, atol=0, rtol=0)
    assert quat_err(res.q[0], q0) < 1e-12
    np.testing.assert_allclose(res.v[0], v0, atol=1e-12)
    # unit quaternions
    np.testing.assert_allclose(np.linalg.norm(res.q, axis=1), 1.0, atol=1e-12)


def test_inputs_not_mutated():
    rng = np.random.default_rng(0)
    n = 20
    t = np.linspace(0, 1, n)
    gyro = rng.normal(size=(n, 3))
    accel = rng.normal(size=(n, 3))
    q0 = random_unit_quat(rng)
    gb = rng.normal(size=3) * 1e-2
    ab = rng.normal(size=3) * 1e-2
    copies = [x.copy() for x in (t, gyro, accel, q0, gb, ab)]
    strapdown(t, gyro, accel, q0, gyro_bias=gb, accel_bias=ab)
    for orig, cp in zip((t, gyro, accel, q0, gb, ab), copies):
        np.testing.assert_array_equal(orig, cp)


# --------------------------------------------------------------------------
# Test 1: constant body rate, no gravity effect
# --------------------------------------------------------------------------
@pytest.mark.parametrize("omega", [
    [0.3, -0.5, 0.7],
    [0.0, 0.0, 1.2],
    [-2.0, 0.1, 0.4],
])
def test_constant_body_rate_matches_scipy_rotvec(omega):
    omega = np.array(omega)
    dt = 0.01
    t = np.arange(0.0, 10.0 + dt / 2, dt)   # up to 10 s; angle > 2 pi for some cases
    n = len(t)
    res = strapdown(t, tile(omega, n), np.zeros((n, 3)), np.array([1.0, 0, 0, 0]),
                    g_ned=(0, 0, 0))
    expected = from_scipy(Rotation.from_rotvec(np.outer(t, omega)))
    # Exact algebra (constant rate => exact exponential), ~1000 steps of roundoff.
    assert np.max(quat_err(res.q, expected)) < 1e-10
    # Zero accel and zero gravity => no translation
    np.testing.assert_allclose(res.v, 0.0, atol=1e-12)
    np.testing.assert_allclose(res.p, 0.0, atol=1e-12)


def test_constant_body_rate_with_nonidentity_q0():
    rng = np.random.default_rng(1)
    q0 = random_unit_quat(rng)
    omega = np.array([0.4, 0.9, -0.3])
    t = np.linspace(0.0, 8.0, 801)
    n = len(t)
    res = strapdown(t, tile(omega, n), np.zeros((n, 3)), q0, g_ned=(0, 0, 0))
    # body-rate (right) composition: R = R0 * Rot(omega t)
    expected = from_scipy(to_scipy(q0) * Rotation.from_rotvec(np.outer(t, omega)))
    assert np.max(quat_err(res.q, expected)) < 1e-10


def test_full_turn_returns_to_start():
    t = np.linspace(0.0, 1.0, 1001)
    n = len(t)
    omega = np.array([0.0, 0.0, 2 * np.pi])
    q0 = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])
    res = strapdown(t, tile(omega, n), np.zeros((n, 3)), q0, g_ned=(0, 0, 0))
    assert quat_err(res.q[-1], q0) < 1e-10


# --------------------------------------------------------------------------
# Attitude integration scheme (spec: trapezoid or interval-start; document it)
# --------------------------------------------------------------------------
def test_attitude_scheme_is_trapezoid_or_interval_start():
    """Rate about a fixed axis varying linearly in time: the accumulated angle
    is a plain sum, so each documented scheme has an exact closed form.
    Trapezoid reproduces the integral exactly; interval-start gives the
    left Riemann sum.  The implementation must match one of them."""
    dt = 0.05
    t = np.arange(0.0, 4.0 + dt / 2, dt)
    a0, a1 = 0.2, 0.5                       # omega_z(t) = a0 + a1 t  [rad/s]
    w = a0 + a1 * t
    gyro = np.column_stack([np.zeros_like(t), np.zeros_like(t), w])
    res = strapdown(t, gyro, np.zeros((len(t), 3)), np.array([1.0, 0, 0, 0]),
                    g_ned=(0, 0, 0))
    angle_trap = np.concatenate([[0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(t))])
    angle_left = np.concatenate([[0], np.cumsum(w[:-1] * np.diff(t))])
    qz = lambda a: np.column_stack([np.cos(a / 2), 0 * a, 0 * a, np.sin(a / 2)])
    err_trap = np.max(quat_err(res.q, qz(angle_trap)))
    err_left = np.max(quat_err(res.q, qz(angle_left)))
    assert min(err_trap, err_left) < 1e-10, (err_trap, err_left)
    # trapezoid is exact for linear rate: equals analytic integral a0 t + a1 t^2/2
    assert np.max(quat_err(qz(angle_trap), qz(a0 * t + 0.5 * a1 * t**2))) < 1e-12


# --------------------------------------------------------------------------
# Test 2: static, level (and static at arbitrary attitude)
# --------------------------------------------------------------------------
def test_static_level_stays_at_rest():
    n = 1000
    t = np.arange(n) * 0.005
    accel = tile([0.0, 0.0, -G0], n)
    res = strapdown(t, np.zeros((n, 3)), accel, np.array([1.0, 0, 0, 0]))
    assert np.max(np.abs(res.v)) < 1e-9
    assert np.max(np.abs(res.p)) < 1e-9
    assert np.max(quat_err(res.q, tile([1, 0, 0, 0], n))) < 1e-12


def test_static_arbitrary_attitude_stays_at_rest():
    rng = np.random.default_rng(2)
    for _ in range(5):
        q0 = random_unit_quat(rng)
        n = 400
        t = np.arange(n) * 0.01
        f_b = to_scipy(q0).inv().apply(-G_NED)          # R^T (0 - g_n)
        res = strapdown(t, np.zeros((n, 3)), tile(f_b, n), q0)
        assert np.max(np.abs(res.v)) < 1e-9
        assert np.max(np.abs(res.p)) < 1e-9


def test_custom_and_default_gravity():
    n = 201
    t = np.linspace(0, 4.0, n)
    f = tile([0.0, 0.0, -9.7], n)
    q0 = np.array([1.0, 0, 0, 0])
    # custom g matching the measurement => at rest
    res = strapdown(t, np.zeros((n, 3)), f, q0, g_ned=(0, 0, 9.7))
    assert np.max(np.abs(res.p)) < 1e-9
    # default g = G0 (down): a_n,z = -9.7 + G0 (positive = downward)
    res = strapdown(t, np.zeros((n, 3)), f, q0)
    a_z = G0 - 9.7
    np.testing.assert_allclose(res.v[:, 2], a_z * t, atol=1e-9)
    np.testing.assert_allclose(res.p[:, 2], 0.5 * a_z * t**2, atol=1e-9)
    np.testing.assert_allclose(res.p[:, :2], 0.0, atol=1e-12)


# --------------------------------------------------------------------------
# Test 3: constant acceleration, rotated attitude
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [10, 11, 12])
def test_constant_acceleration_arbitrary_attitude(seed):
    rng = np.random.default_rng(seed)
    q0 = random_unit_quat(rng)
    a_n = rng.uniform(-3, 3, size=3)
    n = 1001
    t = np.linspace(0.0, 10.0, n)
    f_b = specific_force(tile(q0, n), a_n)
    res = strapdown(t, np.zeros((n, 3)), f_b, q0)
    expected_v = np.outer(t, a_n)
    expected_p = 0.5 * np.outer(t**2, a_n)
    # trapezoid is exact for constant acceleration; only roundoff remains
    assert np.max(np.abs(res.v - expected_v)) < 1e-9
    assert np.max(np.abs(res.p - expected_p)) < 1e-9


def test_initial_velocity_propagates():
    rng = np.random.default_rng(13)
    q0 = random_unit_quat(rng)
    a_n = np.array([0.5, -1.0, 2.0])
    v0 = np.array([3.0, -1.0, 0.25])
    n = 501
    t = np.linspace(0.0, 5.0, n)
    f_b = specific_force(tile(q0, n), a_n)
    res = strapdown(t, np.zeros((n, 3)), f_b, q0, v0=v0)
    np.testing.assert_allclose(res.v, v0 + np.outer(t, a_n), atol=1e-9)
    np.testing.assert_allclose(res.p, np.outer(t, v0) + 0.5 * np.outer(t**2, a_n), atol=1e-9)


def test_initial_position_is_a_constant_offset():
    """p is reported relative to p0: with p0 given the trajectory is identical
    up to a constant offset (0 or p0 depending on the reporting convention)."""
    rng = np.random.default_rng(14)
    q0 = random_unit_quat(rng)
    n = 301
    t = np.linspace(0.0, 3.0, n)
    f_b = specific_force(tile(q0, n), [0.3, 0.2, -0.1])
    p0 = np.array([100.0, -50.0, 7.0])
    a = strapdown(t, np.zeros((n, 3)), f_b, q0)
    b = strapdown(t, np.zeros((n, 3)), f_b, q0, p0=p0)
    diff = b.p - a.p
    np.testing.assert_allclose(diff, np.broadcast_to(diff[0], diff.shape), atol=1e-9)
    np.testing.assert_allclose(b.v, a.v, atol=1e-12)


# --------------------------------------------------------------------------
# Test 4: circular motion with a tilting/rotating body (convergence order)
# --------------------------------------------------------------------------
def _circular_case(dt, t_end=5.0):
    """Inertial circle in the NED x-y plane; body rotating at a constant body
    rate about a tilted axis (so the specific force in body axes rotates)."""
    radius, w_orb = 5.0, 0.7
    omega_b = np.array([0.3, -0.2, 0.5])
    q0 = from_scipy(Rotation.from_euler("ZYX", [0.4, -0.3, 0.2]))
    n = int(round(t_end / dt)) + 1
    t = np.arange(n) * dt
    # analytic inertial kinematics (relative to p(0))
    p_true = radius * np.column_stack([np.cos(w_orb * t) - 1.0, np.sin(w_orb * t), 0 * t])
    v_true = radius * w_orb * np.column_stack([-np.sin(w_orb * t), np.cos(w_orb * t), 0 * t])
    a_true = -radius * w_orb**2 * np.column_stack([np.cos(w_orb * t), np.sin(w_orb * t), 0 * t])
    # exact body attitude for constant body rate: q0 * exp(omega_b t)
    r_true = to_scipy(q0) * Rotation.from_rotvec(np.outer(t, omega_b))
    f_b = r_true.inv().apply(a_true - G_NED)
    res = strapdown(t, tile(omega_b, n), f_b, q0, v0=v_true[0])
    return res, p_true, v_true, r_true


def test_circular_motion_converges_second_order():
    errs = []
    dts = [0.02, 0.01, 0.005, 0.0025]
    for dt in dts:
        res, p_true, _, _ = _circular_case(dt)
        errs.append(np.max(np.linalg.norm(res.p - p_true, axis=1)))
    errs = np.array(errs)
    # position error decreases with dt
    assert np.all(np.diff(errs) < 0), errs
    # order >= 1.8 on every halving (trapezoid/midpoint schemes are O(dt^2))
    orders = np.log2(errs[:-1] / errs[1:])
    assert np.all(orders >= 1.8), (errs, orders)
    # bounded absolute error (5 s, radius 5 m): coarse grid within 1 cm,
    # finest within 0.1 mm (O(dt^2) estimate: error ~ C dt^2, C ~ O(10) m/s^2 * s)
    assert errs[0] < 1e-2
    assert errs[-1] < 1e-4


def test_circular_motion_velocity_and_attitude():
    res, _, v_true, r_true = _circular_case(0.005)
    assert np.max(np.linalg.norm(res.v - v_true, axis=1)) < 1e-3
    # attitude is exact for a constant body rate
    assert np.max(quat_err(res.q, from_scipy(r_true))) < 1e-10


# --------------------------------------------------------------------------
# Test 5: bias drift, analytic
# --------------------------------------------------------------------------
def test_accel_bias_position_error_half_b_t_squared():
    b = np.array([0.02, -0.03, 0.01])
    n = 2001
    t = np.linspace(0.0, 20.0, n)
    f_true = tile([0.0, 0.0, -G0], n)
    # unbiased truth: stays at rest; biased measurement without compensation
    res = strapdown(t, np.zeros((n, 3)), f_true + b, np.array([1.0, 0, 0, 0]))
    np.testing.assert_allclose(res.v, np.outer(t, b), atol=1e-9)
    np.testing.assert_allclose(res.p, 0.5 * np.outer(t**2, b), atol=1e-9)
    # compensating with accel_bias removes it exactly
    res = strapdown(t, np.zeros((n, 3)), f_true + b, np.array([1.0, 0, 0, 0]),
                    accel_bias=b)
    assert np.max(np.abs(res.v)) < 1e-9
    assert np.max(np.abs(res.p)) < 1e-9


def test_accel_bias_rotated_by_attitude():
    """a_n = R (f - b_a) + g: a bias error is rotated into NED by R(q)."""
    rng = np.random.default_rng(20)
    q0 = random_unit_quat(rng)
    b = np.array([0.05, 0.02, -0.04])
    n = 501
    t = np.linspace(0, 10, n)
    f_true = to_scipy(q0).inv().apply(-G_NED)
    res = strapdown(t, np.zeros((n, 3)), tile(f_true, n) + b, q0)
    err_n = to_scipy(q0).apply(b)
    np.testing.assert_allclose(res.p, 0.5 * np.outer(t**2, err_n), atol=1e-9)


@pytest.mark.parametrize("bias", [[0.001, 0.0, 0.0], [0.0, -0.002, 0.0], [0.0007, 0.0004, -0.0011]])
def test_gyro_bias_attitude_error_is_b_times_t(bias):
    bias = np.array(bias)
    n = 2001
    t = np.linspace(0.0, 20.0, n)
    accel = tile([0.0, 0.0, -G0], n)
    # static vehicle: gyro reads the bias only
    res = strapdown(t, tile(bias, n), accel, np.array([1.0, 0, 0, 0]))
    angle = to_scipy(res.q).magnitude()
    np.testing.assert_allclose(angle, np.linalg.norm(bias) * t, atol=1e-10)
    np.testing.assert_allclose(to_scipy(res.q).as_rotvec(), np.outer(t, bias), atol=1e-10)
    # compensating the bias keeps the attitude fixed and the vehicle at rest
    res = strapdown(t, tile(bias, n), accel, np.array([1.0, 0, 0, 0]), gyro_bias=bias)
    assert np.max(quat_err(res.q, tile([1, 0, 0, 0], n))) < 1e-12
    assert np.max(np.abs(res.p)) < 1e-9


@pytest.mark.parametrize("axis, sign", [(0, +1.0), (1, -1.0)])
def test_gyro_bias_position_growth_closed_form(axis, sign):
    """Static level vehicle, gyro bias b about body x (or y).
    Body x: R = Rx(theta), a_n = g [0, sin theta, 1 - cos theta]
    Body y: R = Ry(theta), a_n = g [-sin theta, 0, 1 - cos theta]
    with theta = b t.  Integrating twice:
      p_h = sign * g (theta - sin theta)/b^2   (horizontal, ~ g b t^3 / 6)
      p_z = g (t^2/2 + (cos theta - 1)/b^2)    (vertical, ~ g b^2 t^4 / 24)
    """
    bm = 1e-3
    dt = 0.005
    t = np.arange(0.0, 20.0 + dt / 2, dt)
    n = len(t)
    bias = np.zeros(3)
    bias[axis] = bm
    res = strapdown(t, tile(bias, n), tile([0, 0, -G0], n), np.array([1.0, 0, 0, 0]))
    th = bm * t
    p_h = sign * G0 * (th - np.sin(th)) / bm**2
    p_z = G0 * (t**2 / 2 + (np.cos(th) - 1) / bm**2)
    h_axis = 1 if axis == 0 else 0
    # The trapezoid/midpoint scheme has an O(dt^2) discretisation error that is
    # not small *relative* to p at very early times (p ~ g b t^3/6 is tiny there),
    # so use rtol 1e-4 plus atol 1e-7 m (~ dt^2 * g * b-scale error budget; the
    # final p is ~13 m).  A first-order (left-rectangle) scheme would be off by
    # ~5e-3 m at the end, i.e. ~4e-4 relative, and is caught by rtol.
    np.testing.assert_allclose(res.p[:, h_axis], p_h, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(res.p[:, 2], p_z, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(res.p[:, axis], 0.0, atol=1e-9)
    # leading-order expression from the spec: |p| ~ g |b| t^3 / 6, theta_max = 0.02 rad
    # => the relative correction is theta^2/20 ~ 2e-5; 1e-3 tolerance is generous.
    lead = G0 * bm * t[-1] ** 3 / 6
    assert abs(res.p[-1, h_axis]) == pytest.approx(lead, rel=1e-3)
    # cubic growth: p(2t)/p(t) ~ 8
    k = n // 2
    ratio = res.p[-1, h_axis] / res.p[k, h_axis]
    assert ratio == pytest.approx(8.0, rel=1e-2)


# --------------------------------------------------------------------------
# Test 6: non-uniform t, validation
# --------------------------------------------------------------------------
def test_nonuniform_time_constant_acceleration_exact():
    rng = np.random.default_rng(30)
    n = 400
    t = np.concatenate([[0.0], np.cumsum(rng.uniform(0.001, 0.05, size=n - 1))])
    q0 = random_unit_quat(rng)
    a_n = np.array([0.8, -1.1, 0.3])
    f_b = specific_force(tile(q0, n), a_n)
    res = strapdown(t, np.zeros((n, 3)), f_b, q0)
    np.testing.assert_allclose(res.v, np.outer(t, a_n), atol=1e-9)
    np.testing.assert_allclose(res.p, 0.5 * np.outer(t**2, a_n), atol=1e-9)


def test_nonuniform_time_constant_rate_exact():
    rng = np.random.default_rng(31)
    n = 400
    t = np.concatenate([[0.0], np.cumsum(rng.uniform(0.001, 0.05, size=n - 1))])
    omega = np.array([0.5, -0.8, 0.3])
    res = strapdown(t, tile(omega, n), np.zeros((n, 3)), np.array([1.0, 0, 0, 0]),
                    g_ned=(0, 0, 0))
    expected = from_scipy(Rotation.from_rotvec(np.outer(t, omega)))
    assert np.max(quat_err(res.q, expected)) < 1e-10


def test_nonuniform_time_uses_actual_dt():
    """A uniform grid and a non-uniform grid over the same span give the same
    answer for constant acceleration; wrongly assuming a fixed dt would not."""
    t_u = np.linspace(0.0, 4.0, 5)
    t_n = np.array([0.0, 0.5, 1.0, 3.0, 4.0])
    a_n = np.array([1.0, 0.0, -2.0])
    q0 = np.array([1.0, 0, 0, 0])
    for t in (t_u, t_n):
        n = len(t)
        res = strapdown(t, np.zeros((n, 3)), tile(a_n - G_NED, n), q0)
        np.testing.assert_allclose(res.p[-1], 0.5 * a_n * 16.0, atol=1e-9)
        np.testing.assert_allclose(res.v[-1], 4.0 * a_n, atol=1e-9)


def test_mismatched_lengths_raise_value_error():
    n = 10
    t = np.linspace(0, 1, n)
    q0 = np.array([1.0, 0, 0, 0])
    z = np.zeros((n, 3))
    with pytest.raises(ValueError):
        strapdown(t, np.zeros((n - 1, 3)), z, q0)
    with pytest.raises(ValueError):
        strapdown(t, z, np.zeros((n + 1, 3)), q0)
    with pytest.raises(ValueError):
        strapdown(t[:-1], z, z, q0)
    with pytest.raises(ValueError):
        strapdown(np.linspace(0, 1, n + 3), z, z, q0)


# --------------------------------------------------------------------------
# initial_attitude_from_accel_mag
# --------------------------------------------------------------------------
B_FIELD = 50e-6   # T, typical magnitude


def _synth(roll, pitch, yaw, inclination=np.deg2rad(60.0), b=B_FIELD):
    """Noise-free static specific force and NED-tilt-consistent magnetic field
    (declination 0 => no east component) in body axes for a known attitude."""
    rot = Rotation.from_euler("ZYX", [yaw, pitch, roll])        # body -> NED
    f_b = rot.inv().apply(-G_NED)                               # a_n = 0
    b_n = b * np.array([np.cos(inclination), 0.0, np.sin(inclination)])
    m_b = rot.inv().apply(b_n)
    return rot, f_b, m_b


def _angle_wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def _euler_from_result(q):
    yaw, pitch, roll = to_scipy(q).as_euler("ZYX")
    return roll, pitch, yaw


def test_initial_attitude_level_known_headings():
    # Level, body x pointing east (yaw = +90 deg): hand calculation.
    # B_n = [B, 0, 0] (north); body x = east, body y = south, so m_b = [0, -B, 0].
    q = initial_attitude_from_accel_mag(tile([0.0, 0.0, -G0], 5), tile([0.0, -B_FIELD, 0.0], 5))
    q = np.asarray(q)
    assert q.shape == (4,)
    assert quat_err(q, from_scipy(Rotation.from_euler("Z", np.pi / 2))) < 1e-9
    # Level, body x pointing north: identity
    q = initial_attitude_from_accel_mag(tile([0.0, 0.0, -G0], 5), tile([B_FIELD, 0.0, 0.0], 5))
    assert quat_err(q, [1, 0, 0, 0]) < 1e-9


def test_initial_attitude_recovers_random_attitudes():
    rng = np.random.default_rng(40)
    for _ in range(50):
        roll = rng.uniform(-np.deg2rad(170), np.deg2rad(170))
        pitch = rng.uniform(-np.deg2rad(80), np.deg2rad(80))
        yaw = rng.uniform(-np.pi, np.pi)
        incl = rng.uniform(np.deg2rad(-80), np.deg2rad(80))
        rot, f_b, m_b = _synth(roll, pitch, yaw, inclination=incl, b=rng.uniform(20e-6, 60e-6))
        n = 20
        q = np.asarray(initial_attitude_from_accel_mag(tile(f_b, n), tile(m_b, n)))
        assert q.shape == (4,)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-12
        r, p, y = _euler_from_result(q)
        assert abs(_angle_wrap(r - roll)) < 1e-9
        assert abs(p - pitch) < 1e-9
        assert abs(_angle_wrap(y - yaw)) < 1e-9
        # whole rotation agrees too
        assert (rot.inv() * to_scipy(q)).magnitude() < 1e-9


def test_initial_attitude_averages_window_noise():
    """Antithetic (zero-mean, exactly cancelling) accel perturbations: the mean
    over the window equals the noise-free specific force."""
    rng = np.random.default_rng(41)
    roll, pitch, yaw = 0.3, -0.5, 2.0
    rot, f_b, m_b = _synth(roll, pitch, yaw)
    e = rng.normal(scale=0.5, size=(100, 3))
    accel = np.vstack([f_b + e, f_b - e])
    mag = tile(m_b, 200)
    q = np.asarray(initial_attitude_from_accel_mag(accel, mag))
    assert (rot.inv() * to_scipy(q)).magnitude() < 1e-9


def test_initial_attitude_independent_of_field_magnitude_and_gravity_scale():
    roll, pitch, yaw = -0.7, 0.4, -1.3
    rot, f_b, m_b = _synth(roll, pitch, yaw)
    q1 = np.asarray(initial_attitude_from_accel_mag(tile(f_b, 4), tile(m_b, 4)))
    q2 = np.asarray(initial_attitude_from_accel_mag(tile(3.0 * f_b, 4), tile(0.01 * m_b, 4)))
    assert (rot.inv() * to_scipy(q1)).magnitude() < 1e-9
    assert (rot.inv() * to_scipy(q2)).magnitude() < 1e-9


def test_initial_attitude_feeds_strapdown_static_rest():
    """The recovered attitude is consistent with the strapdown model: a static
    vehicle started from it stays at rest."""
    roll, pitch, yaw = 0.2, 0.1, -2.5
    rot, f_b, m_b = _synth(roll, pitch, yaw)
    q0 = np.asarray(initial_attitude_from_accel_mag(tile(f_b, 10), tile(m_b, 10)))
    n = 300
    t = np.arange(n) * 0.01
    res = strapdown(t, np.zeros((n, 3)), tile(f_b, n), q0)
    assert np.max(np.abs(res.v)) < 1e-8
    assert np.max(np.abs(res.p)) < 1e-8
