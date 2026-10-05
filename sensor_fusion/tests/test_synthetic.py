"""Self-checks of ``sensor_fusion.synthetic`` (spec ``docs/specs/ekf_drone.md`` sections 6-7).

Independent oracles: finite differences of the generator's own truth (quaternion -> body rate,
velocity -> acceleration), ``scipy.spatial.transform.Rotation`` for rotation algebra,
``sensor_fusion.dead_reckoning.strapdown`` (a separately tested integrator) for the
noise-free IMU round trip, and closed-form statistics of white noise / random walks.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from common.frames import G0
from sensor_fusion.dead_reckoning import strapdown
from sensor_fusion.ekf_drone import SensorStreams
from sensor_fusion.synthetic import SyntheticNoise, make_synthetic_flight

TRAJ = ["figure8", "hover", "static"]


def rot(q):
    """Scalar-first quaternion(s) -> scipy Rotation."""
    q = np.asarray(q)
    return Rotation.from_quat(np.concatenate([q[..., 1:], q[..., :1]], axis=-1))


def qmul(p, q):
    """Hamilton product, written out independently of common.quaternion (batched, scalar-first)."""
    pw, px, py, pz = (p[..., i] for i in range(4))
    qw, qx, qy, qz = (q[..., i] for i in range(4))
    return np.stack([pw * qw - px * qx - py * qy - pz * qz,
                     pw * qx + px * qw + py * qz - pz * qy,
                     pw * qy - px * qz + py * qw + pz * qx,
                     pw * qz + px * qy - py * qx + pz * qw], axis=-1)


def qconj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


# --------------------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("traj", TRAJ)
def test_streams_structure_and_shapes(traj):
    s, tr = make_synthetic_flight(traj, duration=10.0, imu_rate=100.0, gps_rate=5.0, baro_rate=10.0,
                                  mag_rate=10.0, gps_delay=0.2)
    assert isinstance(s, SensorStreams)
    n = 1001
    assert s.imu_t.shape == (n,) and s.gyro.shape == (n, 3) and s.accel.shape == (n, 3)
    assert np.allclose(np.diff(s.imu_t), 0.01, atol=1e-12)          # uniform grid
    assert s.gps_t.shape == (51,) and s.gps_pos_ned.shape == (51, 3) and s.gps_vel_ned.shape == (51, 3)
    assert s.gps_pos_sigma.shape == (51, 3) and s.gps_vel_sigma.shape == (51,)
    assert s.baro_t.shape == (101,) and s.baro_d.shape == (101,)
    assert s.mag_t.shape == (101,) and s.mag_field.shape == (101, 3)
    assert s.g == G0 and s.declination == 0.1 and s.origin_lla == (0.0, 0.0, 0.0)
    assert s.meta["synthetic"] is True
    for name in ("p", "v", "a", "omega", "b_g", "b_a"):
        assert getattr(tr, name).shape == (n, 3)
    assert tr.q.shape == (n, 4) and tr.t.shape == (n,)
    assert np.allclose(np.linalg.norm(tr.q, axis=1), 1.0, atol=1e-14)
    # GPS stamped t + delay, noise-free GPS equals truth at the *un-delayed* time
    assert np.allclose(s.gps_t, np.arange(51) / 5.0 + 0.2, atol=1e-12)
    assert np.allclose(s.gps_pos_ned, tr.p[::20], atol=1e-12)
    assert np.allclose(s.gps_vel_ned, tr.v[::20], atol=1e-12)
    assert np.allclose(s.baro_d, tr.p[::10, 2], atol=1e-12)


def test_unknown_trajectory_raises():
    with pytest.raises(ValueError):
        make_synthetic_flight("loop-the-loop", duration=1.0)


def test_seed_reproducible_and_different():
    noise = SyntheticNoise(accel_noise=0.03, gyro_noise=1e-3, gps_pos_sigma=(0.5, 0.5, 0.8),
                           gps_vel_sigma=0.2, baro_sigma=1.0, mag_sigma=1e-6)
    a, _ = make_synthetic_flight("hover", duration=5.0, noise=noise, seed=7)
    b, _ = make_synthetic_flight("hover", duration=5.0, noise=noise, seed=7)
    c, _ = make_synthetic_flight("hover", duration=5.0, noise=noise, seed=8)
    assert np.array_equal(a.gyro, b.gyro) and np.array_equal(a.gps_pos_ned, b.gps_pos_ned)
    assert not np.allclose(a.gyro, c.gyro)


def test_trajectory_character():
    """Spec section 6: size/period of the figure-eight, attitude excursions, constant hover yaw."""
    _, f8 = make_synthetic_flight("figure8", duration=60.0)
    horiz = f8.p[:, :2]
    assert 15.0 <= np.ptp(horiz[:, 0]) <= 45.0 and 10.0 <= np.ptp(horiz[:, 1]) <= 45.0
    # periodic horizontal motion with period ~20 s
    i20 = 2000
    assert np.allclose(f8.p[0, :2], f8.p[i20, :2], atol=1e-9)
    assert np.allclose(f8.v[0, :2], f8.v[i20, :2], atol=1e-9)
    eul = rot(f8.q).as_euler("ZYX")                        # yaw, pitch, roll
    assert 5.0 <= np.degrees(np.abs(eul[:, 2]).max()) <= 15.0
    assert 5.0 <= np.degrees(np.abs(eul[:, 1]).max()) <= 15.0
    assert np.ptp(np.unwrap(eul[:, 0])) > np.radians(90.0)   # yaw really rotates
    # hover: constant yaw, small tilt
    _, hv = make_synthetic_flight("hover", duration=20.0)
    eul = rot(hv.q).as_euler("ZYX")
    assert np.ptp(eul[:, 0]) < 1e-12
    assert np.degrees(np.abs(eul[:, 1:]).max()) <= 3.0
    assert np.all(hv.p == hv.p[0]) and np.all(hv.v == 0.0) and np.all(hv.a == 0.0)
    # static: fixed attitude 5/-3/40 deg
    _, st = make_synthetic_flight("static", duration=2.0)
    eul = rot(st.q).as_euler("ZYX")
    assert np.allclose(np.degrees(eul[:, 2]), 5.0, atol=1e-9)
    assert np.allclose(np.degrees(eul[:, 1]), -3.0, atol=1e-9)
    assert np.allclose(np.degrees(eul[:, 0]), 40.0, atol=1e-9)
    assert np.all(st.omega == 0.0)


# --------------------------------------------------------------------------------------
# kinematic consistency of the truth
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("traj", TRAJ)
def test_omega_matches_quaternion_derivative(traj):
    # central difference of q on a 1 kHz grid: truncation error ~ h^2/3 |q'''| ~ 1e-7 (observed 6e-9)
    _, tr = make_synthetic_flight(traj, duration=20.0, imu_rate=1000.0)
    h = 1.0 / 1000.0
    qdot = (tr.q[2:] - tr.q[:-2]) / (2 * h)
    w_fd = 2.0 * qmul(qconj(tr.q[1:-1]), qdot)
    assert np.allclose(w_fd[:, 0], 0.0, atol=1e-6)         # scalar part of 2 q* (x) qdot is ~0 for unit q
    assert np.allclose(w_fd[:, 1:], tr.omega[1:-1], atol=1e-7)


def test_omega_euler_rate_relation_known_values():
    """Closed-form check of the Euler-rate relation at hand-computed points (independent of the code path)."""
    # hover at t = 0: phi=0, theta=0.035 sin(1)=0.0294.., yaw rate 0, roll rate 0.04*1.3
    _, tr = make_synthetic_flight("hover", duration=1.0, imu_rate=10.0)
    phi_d = 0.04 * 1.3
    theta = 0.035 * np.sin(1.0)
    theta_d = 0.035 * 0.9 * np.cos(1.0)
    assert np.allclose(tr.omega[0], [phi_d, theta_d, 0.0], atol=1e-14)
    # figure8, t = 0: phi = 0.2 sin(.3), theta = 0.15 sin(1), psi_d = 0.1 + 0.6 w
    _, f8 = make_synthetic_flight("figure8", duration=1.0, imu_rate=10.0)
    w0 = 2 * np.pi / 20
    phi, theta = 0.2 * np.sin(0.3), 0.15 * np.sin(1.0)
    phi_d, theta_d, psi_d = 0.2 * w0 * np.cos(0.3), 0.3 * w0 * np.cos(1.0), 0.1 + 0.6 * w0
    expect = [phi_d - psi_d * np.sin(theta),
              theta_d * np.cos(phi) + psi_d * np.cos(theta) * np.sin(phi),
              -theta_d * np.sin(phi) + psi_d * np.cos(theta) * np.cos(phi)]
    assert np.allclose(f8.omega[0], expect, atol=1e-14)


@pytest.mark.parametrize("traj", TRAJ)
def test_velocity_acceleration_position_consistency(traj):
    _, tr = make_synthetic_flight(traj, duration=20.0, imu_rate=1000.0)
    h = 1e-3
    # central differences: truncation error h^2/6 |x'''| with |x'''| <~ 40 w^3 ~ 12 -> ~ 2e-6 (observed 3e-7, 4e-7)
    assert np.allclose((tr.v[2:] - tr.v[:-2]) / (2 * h), tr.a[1:-1], atol=2e-6)
    assert np.allclose((tr.p[2:] - tr.p[:-2]) / (2 * h), tr.v[1:-1], atol=2e-6)


@pytest.mark.parametrize("traj", TRAJ)
def test_specific_force_definition(traj):
    """Noise-free accel = R^T (a - g_n) (scipy rotation); at rest it is -g along body-down."""
    g = 9.7859
    s, tr = make_synthetic_flight(traj, duration=10.0, g=g)
    f = rot(tr.q).inv().apply(tr.a - np.array([0.0, 0.0, g]))
    assert np.allclose(s.accel, f, atol=1e-12)
    assert np.allclose(s.gyro, tr.omega, atol=1e-15)
    if traj != "figure8":
        assert np.allclose(np.linalg.norm(s.accel, axis=1), g, atol=1e-12)


def test_bias_added_and_constant_without_random_walk():
    s0, _ = make_synthetic_flight("figure8", duration=5.0)
    s1, tr = make_synthetic_flight("figure8", duration=5.0, gyro_bias=(0.01, -0.008, 0.005),
                                   accel_bias=(0.1, -0.15, 0.2))
    assert np.allclose(s1.gyro - s0.gyro, [0.01, -0.008, 0.005], atol=1e-14)
    assert np.allclose(s1.accel - s0.accel, [0.1, -0.15, 0.2], atol=1e-14)
    assert np.allclose(tr.b_g, [0.01, -0.008, 0.005]) and np.allclose(tr.b_a, [0.1, -0.15, 0.2])


def test_mag_field_default_and_rotation():
    decl = 0.25
    s, tr = make_synthetic_flight("figure8", duration=10.0, declination=decl)
    m_n = np.array([50e-6 * np.cos(np.radians(65)) * np.cos(decl),
                    50e-6 * np.cos(np.radians(65)) * np.sin(decl),
                    50e-6 * np.sin(np.radians(65))])
    expect = rot(tr.q[::10]).inv().apply(m_n)
    assert np.allclose(s.mag_field, expect, atol=1e-15)
    assert np.allclose(np.linalg.norm(s.mag_field, axis=1), 50e-6, rtol=1e-12)
    # custom field
    s2, _ = make_synthetic_flight("static", duration=1.0, field_ned=[30e-6, 0.0, 0.0])
    assert np.allclose(np.linalg.norm(s2.mag_field, axis=1), 30e-6, rtol=1e-12)


# --------------------------------------------------------------------------------------
# noise-free IMU through strapdown reproduces truth, second-order convergence
# --------------------------------------------------------------------------------------
def _strapdown_errors(rate, duration=20.0):
    s, tr = make_synthetic_flight("figure8", duration=duration, imu_rate=rate)
    r = strapdown(s.imu_t, s.gyro, s.accel, tr.q[0], tr.v[0], tr.p[0], g_ned=(0, 0, s.g))
    ep = np.max(np.linalg.norm(r.p - tr.p, axis=1))
    ev = np.max(np.linalg.norm(r.v - tr.v, axis=1))
    eq = np.max((rot(tr.q).inv() * rot(r.q)).magnitude())
    return ep, ev, eq


def test_noise_free_imu_reproduces_truth_through_strapdown():
    ep, ev, eq = _strapdown_errors(400.0)
    # 400 Hz, 20 s: observed max errors 2.3e-5 m, 5.7e-6 m/s, 7.9e-8 rad (second-order scheme); 4x guard
    assert ep < 1e-4 and ev < 2.5e-5 and eq < 4e-7


def test_strapdown_second_order_convergence():
    e = np.array([_strapdown_errors(r) for r in (50.0, 100.0, 200.0, 400.0)])   # rows: rate, cols: p, v, q
    ratios = e[:-1] / e[1:]
    # second order => ratio 4 when halving dt (observed 4.000 to 5 digits); +-5 % guard
    assert np.all(ratios > 3.8) and np.all(ratios < 4.2), ratios
    assert np.all(e[-1] < e[0] / 40.0)


# --------------------------------------------------------------------------------------
# noise statistics
# --------------------------------------------------------------------------------------
def test_white_noise_statistics_match_densities():
    noise = SyntheticNoise(accel_noise=0.035, gyro_noise=1.5e-3, gps_pos_sigma=(0.5, 0.6, 0.9),
                           gps_vel_sigma=0.2, baro_sigma=1.2, mag_sigma=2e-6)
    rate = 100.0
    s, tr = make_synthetic_flight("figure8", duration=400.0, imu_rate=rate, gps_rate=20.0,
                                  baro_rate=20.0, mag_rate=20.0, noise=noise, seed=11)
    # per-sample sigma = density * sqrt(rate); N = 40001 per axis -> std error of sigma ~ 0.2 % (x3 axes) ; 3 % tol
    r_g = s.gyro - tr.omega
    r_a = s.accel - np.einsum("nji,nj->ni", rot(tr.q).as_matrix(), tr.a - np.array([0, 0, G0]))
    assert np.allclose(r_g.std(axis=0), 1.5e-3 * np.sqrt(rate), rtol=0.03)
    assert np.allclose(r_a.std(axis=0), 0.035 * np.sqrt(rate), rtol=0.03)
    assert np.all(np.abs(r_g.mean(axis=0)) < 5 * 1.5e-3 * np.sqrt(rate) / np.sqrt(r_g.shape[0]))
    # other sensors: N = 8001 samples -> std error ~ 0.8 % ; 5 % tol
    t_g = s.gps_t
    pg = s.gps_pos_ned - np.stack([20 * np.sin(2 * np.pi / 20 * t_g), 10 * np.sin(4 * np.pi / 20 * t_g),
                                   -10 - 2 * np.sin(np.pi / 20 * t_g)], axis=1)
    assert np.allclose(pg.std(axis=0), [0.5, 0.6, 0.9], rtol=0.05)
    assert np.allclose(s.gps_pos_sigma[0], [0.5, 0.6, 0.9]) and np.allclose(s.gps_vel_sigma, 0.2)
    vg = s.gps_vel_ned - tr.v[::5]
    assert np.allclose(vg.std(axis=0), 0.2, rtol=0.05)
    bd = s.baro_d - tr.p[::5, 2]
    assert np.isclose(bd.std(), 1.2, rtol=0.05)
    m_clean, _ = make_synthetic_flight("figure8", duration=400.0, imu_rate=rate, gps_rate=20.0,
                                       baro_rate=20.0, mag_rate=20.0, seed=11)
    assert np.allclose((s.mag_field - m_clean.mag_field).std(axis=0), 2e-6, rtol=0.05)


def test_bias_random_walk_statistics():
    noise = SyntheticNoise(gyro_bias_rw=1e-4, accel_bias_rw=3e-4)
    # 20 independent seeds, 100 s each: Var[b(T) - b(0)] = density^2 T per axis (60 samples -> 9 % std error on sigma)
    dg, da = [], []
    for sd in range(20):
        _, tr = make_synthetic_flight("static", duration=100.0, imu_rate=50.0, noise=noise, seed=sd,
                                      gyro_bias=(0.01, 0.0, 0.0))
        dg.append(tr.b_g[-1] - tr.b_g[0]); da.append(tr.b_a[-1] - tr.b_a[0])
        # increments per step
    dg, da = np.array(dg), np.array(da)
    assert np.isclose(dg.std(), 1e-4 * 10.0, rtol=0.25)
    assert np.isclose(da.std(), 3e-4 * 10.0, rtol=0.25)
    # per-step increments: sigma = density * sqrt(dt)
    _, tr = make_synthetic_flight("static", duration=2000.0, imu_rate=10.0, noise=noise, seed=99)
    inc = np.diff(tr.b_g, axis=0)
    assert np.isclose(inc.std(), 1e-4 * np.sqrt(0.1), rtol=0.03)
    assert np.isclose(np.diff(tr.b_a, axis=0).std(), 3e-4 * np.sqrt(0.1), rtol=0.03)
    # IMU gyro really contains the walking bias
    s, tr2 = make_synthetic_flight("static", duration=50.0, imu_rate=50.0, noise=noise, seed=3)
    assert np.allclose(s.gyro, tr2.b_g, atol=1e-15)
