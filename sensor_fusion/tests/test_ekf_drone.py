"""Tests for ``sensor_fusion.ekf_drone`` (spec ``docs/specs/ekf_drone.md`` section 7, tests 1-11).

Independent oracles (never the filter's own ``F``/``H``/propagation as "truth"):

* the **true nonlinear error dynamics** rebuilt in the test from the error definitions of spec
  section 1 (``q_true = q (x) exp(dtheta)``, additive others) with ``scipy.spatial.transform.Rotation``;
* finite differences of ``scipy`` ZYX yaw;
* synthetic truth from ``sensor_fusion.synthetic`` (closed-form trajectories);
* chi-square bounds from ``scipy.stats``; Somigliana gravity and ``ppigrf`` for the real-log checks.
"""
from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scipy.stats import chi2

from common.ekf import is_spd
from sensor_fusion import ekf_drone as ek
from sensor_fusion.ekf_drone import (DroneEKF, EkfConfig, InitialState, error_state_jacobians,
                                     mag_heading_jacobian, mag_heading_measurement, process_noise_psd, run_filter,
                                     static_initial_state, yaw_jacobian)
from sensor_fusion.synthetic import SyntheticNoise, default_field_ned, make_synthetic_flight

G = 9.80665
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
HOVER_LOG = DATA_DIR / "f4e81624-7ccc-4859-bb45-d02fa0246b77.ulg"


# --------------------------------------------------------------------------------------
# helpers (scipy based, independent of common.quaternion / the filter)
# --------------------------------------------------------------------------------------
def rot(q):
    q = np.asarray(q, float)
    return Rotation.from_quat(np.concatenate([q[..., 1:], q[..., :1]], axis=-1))


def quat_sf(r: Rotation):
    x = r.as_quat()
    return np.concatenate([x[..., 3:], x[..., :3]], axis=-1)


def att_err_deg(q_est, q_true):
    """Angle of the relative rotation [deg] per sample."""
    return np.degrees((rot(q_est).inv() * rot(q_true)).magnitude())


def true_error_rate(q, f_m, w_m, b_g, b_a, dx, noise=None, g=G):
    """Time derivative of the error state ``delta x`` obtained from the TRUE nonlinear model.

    Truth: ``q_t = q (x) exp(dth)``, ``p_t = p + dp``, ... ; measured ``f_m``, ``w_m`` fixed.
    ``noise = [n_a, n_g, n_bg, n_ba]`` enters as in spec section 1 (``f_m = R_t^T(a-g)+b_a+n_a``,
    ``w_m = w + b_g + n_g``, ``db_g/dt = n_bg``). Position/velocity rates are analytic; the attitude
    error rate is a finite difference of the exact error rotation vector over a tiny dt.
    """
    noise = np.zeros(12) if noise is None else noise
    n_a, n_g, n_bg, n_ba = noise[0:3], noise[3:6], noise[6:9], noise[9:12]
    dp, dv, dth, dbg, dba = dx[0:3], dx[3:6], dx[6:9], dx[9:12], dx[12:15]
    R = rot(q)
    Rt = R * Rotation.from_rotvec(dth)
    gn = np.array([0.0, 0.0, g])
    a_nom = R.apply(f_m - b_a) + gn
    a_true = Rt.apply(f_m - (b_a + dba) - n_a) + gn
    # attitude: exact one-step propagation of both rotations (body-rate exponential)
    h = 1e-6
    w_nom = w_m - b_g
    w_true = w_m - (b_g + dbg) - n_g
    R1 = R * Rotation.from_rotvec(w_nom * h)
    Rt1 = Rt * Rotation.from_rotvec(w_true * h)
    dth1 = (R1.inv() * Rt1).as_rotvec()
    dth0 = (R.inv() * Rt).as_rotvec()
    out = np.zeros(15)
    out[0:3] = dv
    out[3:6] = a_true - a_nom
    out[6:9] = (dth1 - dth0) / h
    out[9:12] = n_bg
    out[12:15] = n_ba
    return out


# --------------------------------------------------------------------------------------
# 1. F and G Jacobians against the true nonlinear model
# --------------------------------------------------------------------------------------
def test_state_layout():
    assert len(ek.STATE_NAMES) == 15
    assert (ek.IDX_P, ek.IDX_V, ek.IDX_TH, ek.IDX_BG, ek.IDX_BA) == (
        slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15))
    assert ek.FROZEN_VAR == 1e-18


def test_F_jacobian_matches_true_nonlinear_error_dynamics():
    rng = np.random.default_rng(1)
    for _ in range(25):
        q = quat_sf(Rotation.random(random_state=int(rng.integers(1 << 30))))
        f_m = rng.normal(size=3) * 5.0
        w_m = rng.normal(size=3) * 1.0
        b_g = rng.normal(size=3) * 0.05
        b_a = rng.normal(size=3) * 0.2
        dx = rng.normal(size=15) * 1e-4          # small enough that second-order terms are ~1e-8 * |f|
        rate = true_error_rate(q, f_m, w_m, b_g, b_a, dx)
        F, _ = error_state_jacobians(q, f_m - b_a, w_m - b_g)
        # |F dx| ~ 1e-3; truncation (|dx|^2 |f|) ~ 1e-7; FD round-off 1e-16/1e-6 = 1e-10 -> atol 3e-6
        assert np.allclose(F @ dx, rate, atol=3e-6, rtol=0), np.abs(F @ dx - rate).max()
        # the test must be sensitive: neglecting the F entries gives a far larger error
        assert np.abs(rate).max() > 5e-5


def test_F_blocks_individually_and_structure():
    """Each column of F recovered by unit perturbations of the true model (checks every block, incl. signs)."""
    rng = np.random.default_rng(2)
    q = quat_sf(Rotation.random(random_state=5))
    f_m, w_m = rng.normal(size=3) * 4, rng.normal(size=3)
    b_g, b_a = rng.normal(size=3) * 0.02, rng.normal(size=3) * 0.1
    F, _ = error_state_jacobians(q, f_m - b_a, w_m - b_g)
    eps = 1e-5
    F_fd = np.zeros((15, 15))
    for j in range(15):
        e = np.zeros(15); e[j] = eps
        # central difference in the perturbation (cancels the O(eps^2) truncation)
        F_fd[:, j] = (true_error_rate(q, f_m, w_m, b_g, b_a, e)
                      - true_error_rate(q, f_m, w_m, b_g, b_a, -e)) / (2 * eps)
    # FD round-off 1e-16/(1e-6*2e-5) ~ 5e-6 on the attitude rows -> atol 1e-4 (entries are O(1..10))
    assert np.allclose(F, F_fd, atol=1e-4), np.abs(F - F_fd).max()
    # known structural zeros
    assert np.all(F[:, 0:3] == 0.0)                                  # nothing depends on position
    assert np.all(F[9:15, :] == 0.0)                                 # biases are random walks
    assert np.allclose(F[0:3, 3:6], np.eye(3))


def test_G_noise_mapping_each_channel():
    rng = np.random.default_rng(3)
    q = quat_sf(Rotation.random(random_state=9))
    f_m, w_m = rng.normal(size=3) * 3, rng.normal(size=3)
    b_g, b_a = np.array([0.01, -0.02, 0.005]), np.array([0.1, 0.05, -0.2])
    _, Gm = error_state_jacobians(q, f_m - b_a, w_m - b_g)
    assert Gm.shape == (15, 12)
    eps = 1e-4
    for j in range(12):
        n = np.zeros(12); n[j] = eps
        # zero error state, single noise channel on: rate = G[:, j] eps (to first order)
        rate = true_error_rate(q, f_m, w_m, b_g, b_a, np.zeros(15), noise=n)
        assert np.allclose(rate / eps, Gm[:, j], atol=1e-5), (j, np.abs(rate / eps - Gm[:, j]).max())
    # Qc ordering [n_a, n_g, n_bg, n_ba] and G Qc G^T diagonal (R R^T = I, isotropic noise)
    cfg = EkfConfig()
    Qc = process_noise_psd(cfg, np.ones(15, bool))
    assert np.allclose(np.diag(Qc), np.repeat([cfg.accel_noise ** 2, cfg.gyro_noise ** 2,
                                               cfg.gyro_bias_rw ** 2, cfg.accel_bias_rw ** 2], 3))
    GQG = Gm @ Qc @ Gm.T
    assert np.allclose(GQG, np.diag(np.diag(GQG)), atol=1e-18)
    assert np.allclose(np.diag(GQG)[3:6], cfg.accel_noise ** 2) and np.allclose(np.diag(GQG)[6:9], cfg.gyro_noise ** 2)


def test_process_noise_zero_for_frozen_biases():
    cfg = EkfConfig()
    mask = np.ones(15, bool); mask[9:15] = False
    Qc = process_noise_psd(cfg, mask)
    assert np.all(np.diag(Qc)[6:12] == 0.0)
    assert np.all(np.diag(Qc)[0:6] > 0.0)


# --------------------------------------------------------------------------------------
# 2. yaw Jacobian and magnetic heading
# --------------------------------------------------------------------------------------
def _yaw_of(q):
    return rot(q).as_euler("ZYX")[0]


def test_yaw_jacobian_vs_finite_difference():
    rng = np.random.default_rng(4)
    eps = 1e-6
    for _ in range(30):
        # |pitch| < 60 deg (the filter skips the update beyond), arbitrary roll / yaw
        yaw, pitch, roll = rng.uniform(-np.pi, np.pi), rng.uniform(-1.0, 1.0), rng.uniform(-np.pi, np.pi)
        q = quat_sf(Rotation.from_euler("ZYX", [yaw, pitch, roll]))
        J = yaw_jacobian(q)
        fd = np.zeros(3)
        for i in range(3):
            e = np.zeros(3); e[i] = eps
            qp = quat_sf(rot(q) * Rotation.from_rotvec(e))
            qm = quat_sf(rot(q) * Rotation.from_rotvec(-e))
            d = _yaw_of(qp) - _yaw_of(qm)
            fd[i] = ((d + np.pi) % (2 * np.pi) - np.pi) / (2 * eps)
        # central difference: truncation ~ eps^2, round-off ~ 1e-16/eps = 1e-10 -> 1e-7
        assert np.allclose(J, fd, atol=1e-7), (J, fd)


def test_mag_heading_jacobian_vs_finite_difference():
    """H of the heading residual r(dth) = z(q, R(q (x) exp(dth))^T m_n) - psi(q): z uses the ESTIMATED tilt
    for compensation, the field is the TRUE body field (depends on dth). Evaluated at dth = 0."""
    rng = np.random.default_rng(6)
    eps = 1e-6
    for _ in range(40):
        yaw = rng.uniform(-np.pi, np.pi)
        pitch, roll = np.radians(rng.uniform(-40, 40, size=2))
        decl = rng.uniform(-0.4, 0.4)
        q = quat_sf(Rotation.from_euler("ZYX", [yaw, pitch, roll]))
        m_n = default_field_ned(decl, inclination_deg=rng.uniform(-67, 67))
        psi_hat = _yaw_of(q)

        def resid(dth):
            m_b = (rot(q) * Rotation.from_rotvec(dth)).inv().apply(m_n)
            d = mag_heading_measurement(q, m_b, decl) - psi_hat
            return (d + np.pi) % (2 * np.pi) - np.pi

        fd = np.zeros(3)
        for i in range(3):
            e = np.zeros(3); e[i] = eps
            fd[i] = (resid(e) - resid(-e)) / (2 * eps)
        H = mag_heading_jacobian(q, rot(q).inv().apply(m_n))
        assert H.shape == (3,)
        # central difference: truncation ~ eps^2 * (strong nonlinearity, O(10)) ~ 1e-11, round-off ~ 1e-10 -> 1e-7
        assert np.allclose(H, fd, atol=1e-7), (H, fd)
        assert abs(resid(np.zeros(3))) < 1e-12      # consistent field -> zero residual


def test_mag_heading_jacobian_level_case_closed_form():
    """Level attitude: a true yaw error dth_z about the vertical rotates the measured heading by +dth_z."""
    q = quat_sf(Rotation.from_euler("ZYX", [0.4, 0.0, 0.0]))
    m_n = default_field_ned(0.1, inclination_deg=60.0)
    H = mag_heading_jacobian(q, rot(q).inv().apply(m_n))
    # true heading = psi + dth_z -> z - psi_hat = dth_z => dr/ddth_z = +1
    assert np.isclose(H[2], 1.0, atol=1e-12)


def test_yaw_jacobian_level_attitude_is_unit_about_z():
    q = quat_sf(Rotation.from_euler("ZYX", [0.7, 0.0, 0.0]))
    assert np.allclose(yaw_jacobian(q), [0.0, 0.0, 1.0], atol=1e-14)


def test_mag_heading_returns_true_yaw_noise_free():
    rng = np.random.default_rng(5)
    for _ in range(50):
        yaw = rng.uniform(-np.pi, np.pi)
        pitch, roll = np.radians(rng.uniform(-50, 50, size=2))
        decl = rng.uniform(-0.5, 0.5)
        q = quat_sf(Rotation.from_euler("ZYX", [yaw, pitch, roll]))
        m_n = default_field_ned(decl, total=rng.uniform(25e-6, 65e-6), inclination_deg=rng.uniform(-70, 70))
        m_b = rot(q).inv().apply(m_n)
        z = mag_heading_measurement(q, m_b, decl)
        diff = (z - yaw + np.pi) % (2 * np.pi) - np.pi
        assert abs(diff) < 1e-12
        assert -np.pi < z <= np.pi


def test_wrap_angle():
    w = ek.wrap_angle
    assert np.isclose(w(3 * np.pi / 2), -np.pi / 2)
    assert np.isclose(w(-np.pi), np.pi)          # (-pi, pi]
    assert np.isclose(w(2 * np.pi + 0.1), 0.1)
    assert np.allclose(w(np.array([0.0, 7.0])), [0.0, 7.0 - 2 * np.pi])


def test_heading_update_skipped_beyond_60deg_pitch():
    q = quat_sf(Rotation.from_euler("ZYX", [0.3, np.radians(70.0), 0.1]))
    f = DroneEKF(EkfConfig(), InitialState(np.zeros(3), np.zeros(3), q), G)
    m = rot(q).inv().apply(default_field_ned(0.1))
    assert f.update_mag_heading(m, 0.1, 0.07) is None
    q2 = quat_sf(Rotation.from_euler("ZYX", [0.3, np.radians(50.0), 0.1]))
    f2 = DroneEKF(EkfConfig(), InitialState(np.zeros(3), np.zeros(3), q2), G)
    assert f2.update_mag_heading(rot(q2).inv().apply(default_field_ned(0.1)), 0.1, 0.07) is not None
    # batch run counts skips
    s, tr = make_synthetic_flight("static", duration=3.0)
    s.mag_field[:] = rot(q).inv().apply(default_field_ned(0.1))
    res = run_filter(s, EkfConfig(increment=2),
                     init=InitialState(np.zeros(3), np.zeros(3), q))
    # the (inconsistent) static IMU/GPS data pull the tilt back below 60 deg after a few updates, so only
    # the first few headings are skipped; every mag sample is either skipped or applied
    assert res.meta["mag_skipped_tilt"] >= 1
    assert res.meta["mag_skipped_tilt"] + res.meta["n_updates"]["mag_heading"] == s.mag_t.size


# --------------------------------------------------------------------------------------
# 3. discretisation
# --------------------------------------------------------------------------------------
def test_first_order_vs_van_loan_covariance_after_10s():
    s, tr = make_synthetic_flight("figure8", duration=10.0, imu_rate=100.0)
    init = InitialState(tr.p[0], tr.v[0], tr.q[0])
    out = {}
    for disc in ("first_order", "van_loan"):
        f = DroneEKF(EkfConfig(discretization=disc), init, G)
        for k in range(len(s.imu_t) - 1):
            f.predict(s.gyro[k], s.gyro[k + 1], s.accel[k], s.accel[k + 1], s.imu_t[k + 1] - s.imu_t[k])
        out[disc] = f.ekf.P
    d1, d2 = np.diag(out["first_order"]), np.diag(out["van_loan"])
    rel = np.abs(d1 - d2) / np.abs(d2)
    assert np.all(rel <= 0.02), rel
    # sanity: the covariance really changed (the comparison is not trivially between two P0)
    P0 = np.diag(np.repeat([1.0, 0.5, 0.1, 0.1, 0.2], 3) ** 2)
    assert np.abs(np.diag(out["van_loan"]) / np.diag(P0) - 1).max() > 0.01
    assert is_spd(out["first_order"]) and is_spd(out["van_loan"])


# --------------------------------------------------------------------------------------
# 4. noise-free truth tracking
# --------------------------------------------------------------------------------------
def test_noise_free_truth_tracking_figure8():
    s, tr = make_synthetic_flight("figure8", duration=60.0, imu_rate=100.0)
    init = InitialState(tr.p[0], tr.v[0], tr.q[0])
    res = run_filter(s, EkfConfig(increment=3, gps_delay_s=0.0), init=init)
    ep = np.linalg.norm(res.p - tr.p, axis=1)
    ev = np.linalg.norm(res.v - tr.v, axis=1)
    ea = att_err_deg(res.q, tr.q)
    # spec thresholds: 0.05 m, 0.05 m/s, 0.05 deg
    assert ep.max() < 0.05 and ev.max() < 0.05 and ea.max() < 0.05, (ep.max(), ev.max(), ea.max())
    assert np.all(np.isfinite(res.std))
    # biases stay (near) zero with noise-free consistent data
    assert np.abs(res.b_g).max() < 1e-3 and np.abs(res.b_a).max() < 1e-2


# --------------------------------------------------------------------------------------
# 5. convergence with noise and biases
# --------------------------------------------------------------------------------------
NOISE = SyntheticNoise(accel_noise=0.02, gyro_noise=1.0e-3, gyro_bias_rw=1.0e-4, accel_bias_rw=3.0e-4,
                       gps_pos_sigma=(0.8, 0.8, 1.2), gps_vel_sigma=0.3, baro_sigma=1.0, mag_sigma=1.0e-6)


def matched_cfg(**kw):
    """Filter noise = generator noise (gps floors are below the generator sigmas; no delay)."""
    base = dict(accel_noise=NOISE.accel_noise, gyro_noise=NOISE.gyro_noise,
                gyro_bias_rw=NOISE.gyro_bias_rw, accel_bias_rw=NOISE.accel_bias_rw,
                baro_sigma=NOISE.baro_sigma, mag_heading_sigma=0.05, gps_delay_s=0.0)
    base.update(kw)
    return EkfConfig(**base)


def test_convergence_with_noise_and_biases():
    b_g, b_a = np.array([0.01, -0.008, 0.005]), np.array([0.1, -0.15, 0.2])
    s, tr = make_synthetic_flight("figure8", duration=60.0, noise=NOISE, gyro_bias=b_g, accel_bias=b_a, seed=21)
    init = InitialState(tr.p[0] + [0.3, -0.2, 0.4], tr.v[0] + [0.1, -0.1, 0.05], tr.q[0])
    res = run_filter(s, matched_cfg(increment=3), init=init)
    err_bg = tr.b_g[-1] - res.b_g[-1]
    err_ba = tr.b_a[-1] - res.b_a[-1]
    sig = res.std[-1]
    assert np.all(np.abs(err_bg) < 3 * sig[9:12]), (err_bg, sig[9:12])
    assert np.all(np.abs(err_ba) < 3 * sig[12:15]), (err_ba, sig[12:15])
    assert np.abs(err_bg).max() < 0.002
    late = tr.t >= 30.0
    assert att_err_deg(res.q, tr.q)[late].max() < 1.0
    # sanity: biases really were learned (init was zero)
    assert np.linalg.norm(res.b_g[-1] - b_g) < 0.3 * np.linalg.norm(b_g)


# --------------------------------------------------------------------------------------
# 6. consistency (NEES / NIS)
# --------------------------------------------------------------------------------------
def _nees_run(seed, cfg_kw):
    """One Monte-Carlo run; initial error drawn from P0 (and biases = error + 0 estimate)."""
    rng = np.random.default_rng(1000 + seed)
    cfg = matched_cfg(increment=3, cov_every=100, **cfg_kw)
    d_p = rng.normal(size=3) * cfg.sigma_p0
    d_v = rng.normal(size=3) * cfg.sigma_v0
    d_th = rng.normal(size=3) * cfg.sigma_att0
    d_bg = rng.normal(size=3) * cfg.sigma_bg0
    d_ba = rng.normal(size=3) * cfg.sigma_ba0
    s, tr = make_synthetic_flight("figure8", duration=45.0, noise=NOISE, gyro_bias=d_bg, accel_bias=d_ba,
                                  seed=seed)
    # estimate = truth - error (p_true = p + dp, q_true = q (x) exp(dth)); estimated biases start at 0
    q0 = quat_sf(rot(tr.q[0]) * Rotation.from_rotvec(-d_th))
    init = InitialState(tr.p[0] - d_p, tr.v[0] - d_v, q0)
    res = run_filter(s, cfg, init=init)
    idx = np.round(res.P_hist_t * 100.0).astype(int)
    e = np.empty((idx.size, 9))
    e[:, 0:3] = tr.p[idx] - res.p[idx]
    e[:, 3:6] = tr.v[idx] - res.v[idx]
    e[:, 6:9] = (rot(res.q[idx]).inv() * rot(tr.q[idx])).as_rotvec()
    nees = np.array([e[k] @ np.linalg.solve(res.P_hist[k][:9, :9], e[k]) for k in range(idx.size)])
    return res.P_hist_t, nees, res.innovations["gps_pos"]


def test_nees_nis_consistency_monte_carlo():
    runs = 10
    # initial gyro bias sigma reduced from the 0.1 rad/s default to 0.02 so that the first seconds
    # stay linear; the initial errors are drawn from exactly this P0, so the test is a fair NEES test.
    kw = dict(sigma_bg0=0.02)
    nees_all, nis_all = [], []
    for sd in range(runs):
        t, nees, inn = _nees_run(sd, kw)
        nees_all.append(nees[t >= 20.0])
        m = inn.accepted & (inn.t >= 5.0)
        nis_all.append(inn.nis[m])
    nees_all = np.array(nees_all)                     # (runs, times)
    mean_nees = nees_all.mean()
    # two-sided 99 % band for the run-averaged NEES: runs*9 dof (spec); time averaging only tightens it
    lo = chi2.ppf(0.005, 9 * runs) / runs
    hi = chi2.ppf(0.995, 9 * runs) / runs
    assert lo <= mean_nees <= hi, (lo, mean_nees, hi)
    nis = np.concatenate(nis_all)
    c_lo, c_hi = chi2.ppf([0.025, 0.975], 3)
    frac = np.mean((nis >= c_lo) & (nis <= c_hi))
    assert frac >= 0.85, frac


# --------------------------------------------------------------------------------------
# 7. increments
# --------------------------------------------------------------------------------------
def _hover(seed=5, gb=(0.0, 0.0, 0.0), ab=(0.0, 0.0, 0.0), duration=40.0, noise=NOISE):
    return make_synthetic_flight("hover", duration=duration, noise=noise, gyro_bias=gb, accel_bias=ab, seed=seed)


def test_increment_1_yaw_unobservable_and_biases_frozen():
    b0_g, b0_a = np.array([0.003, 0.002, -0.001]), np.array([0.02, -0.01, 0.03])
    # frozen biases equal to the truth biases (a consistent increment-1 scenario)
    s, tr = _hover(gb=b0_g, ab=b0_a)
    init = InitialState(tr.p[0], tr.v[0], tr.q[0], b_g=b0_g, b_a=b0_a)
    res = run_filter(s, matched_cfg(increment=1), init=init)
    # biases identically equal to the initial values, and stds stay at sqrt(FROZEN_VAR)
    assert np.array_equal(res.b_g, np.tile(b0_g, (len(res.t), 1)))
    assert np.array_equal(res.b_a, np.tile(b0_a, (len(res.t), 1)))
    assert np.all(res.std[:, 9:15] < 1e-6)
    # yaw is not observable in hover without a heading sensor: sigma stays at the 0.1 rad prior.
    # (Noisy accelerometer samples enter F through f and create a small spurious observability,
    # observed -5 %; we require no more than a 15 % reduction, vs 96 % reduction with the magnetometer.)
    sy = res.std[:, 8]
    assert sy[-1] > 0.85 * 0.1
    # tilt remains well determined by GPS velocity
    assert res.std[-1, 6] < 0.01 and res.std[-1, 7] < 0.01
    # noise-free sensors: no spurious information, process noise makes the yaw sigma grow
    s0, tr0 = _hover(noise=SyntheticNoise())
    r0 = run_filter(s0, matched_cfg(increment=1), init=InitialState(tr0.p[0], tr0.v[0], tr0.q[0]))
    assert r0.std[-1, 8] > r0.std[100, 8] > 0.0999 * 0.999 and r0.std[-1, 8] > 0.1


def test_increment_2_yaw_bounded_by_mag():
    s, tr = _hover()
    init = InitialState(tr.p[0], tr.v[0], tr.q[0])
    cfg = matched_cfg(increment=2)
    res = run_filter(s, cfg, init=init)
    late = res.t >= 20.0
    assert np.all(res.std[late, 8] < 3 * cfg.mag_heading_sigma)
    # and in fact far below the open-loop (increment 1) yaw uncertainty
    res1 = run_filter(s, matched_cfg(increment=1), init=init)
    assert res.std[-1, 8] < 0.2 * res1.std[-1, 8]
    # yaw is actually estimated: error well inside 3 degrees (mag sigma is 2.9 deg, filter averages ~400 samples)
    yaw_err = (rot(res.q).inv() * rot(tr.q)).as_rotvec()[:, 2]
    assert np.abs(yaw_err[late]).max() < np.radians(3.0)
    assert np.all(res.b_g == 0.0) and np.all(res.b_a == 0.0)      # still frozen at zero


def test_increment_3_biases_move():
    s, tr = _hover(gb=(0.01, -0.008, 0.005), ab=(0.1, -0.15, 0.2))
    init = InitialState(tr.p[0], tr.v[0], tr.q[0])
    res = run_filter(s, matched_cfg(increment=3), init=init)
    assert np.abs(res.b_g[-1]).max() > 1e-3 and np.abs(res.b_a[-1]).max() > 1e-2
    assert res.std[-1, 9] < 0.1 and res.std[-1, 12] < 0.2     # variance reduced from the prior
    assert "bg_x" in res.meta["active_states"]
    r1 = run_filter(s, matched_cfg(increment=1), init=init)
    assert "bg_x" not in r1.meta["active_states"]


def test_config_flags_override_increment_defaults():
    assert EkfConfig(increment=1).mag_enabled is False and EkfConfig(increment=2).mag_enabled is True
    assert EkfConfig(increment=2).biases_enabled is False and EkfConfig(increment=3).biases_enabled is True
    assert EkfConfig(increment=1, use_mag=True).mag_enabled is True
    assert EkfConfig(increment=3, estimate_biases=False).biases_enabled is False
    with pytest.raises(ValueError):
        EkfConfig(increment=4)
    with pytest.raises(ValueError):
        EkfConfig(discretization="euler")


# --------------------------------------------------------------------------------------
# 8. SPD and unit norm
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("inc", [1, 2, 3])
def test_covariance_spd_and_quaternion_unit_norm(inc):
    s, tr = make_synthetic_flight("figure8", duration=30.0, noise=NOISE, gyro_bias=(0.01, -0.008, 0.005),
                                  accel_bias=(0.1, -0.15, 0.2), seed=31)
    res = run_filter(s, matched_cfg(increment=inc, cov_every=25), init=None)
    assert res.P_hist.shape == (len(res.P_hist_t), 15, 15) and res.P_hist.shape[0] >= 100
    for P in res.P_hist:
        assert is_spd(P)
    assert is_spd(res.P_final)
    assert np.allclose(res.P_final, res.P_final.T, atol=0, rtol=1e-12) or is_spd(res.P_final)
    assert np.abs(np.linalg.norm(res.q, axis=1) - 1.0).max() < 1e-12
    assert np.all(np.isfinite(res.p)) and np.all(np.isfinite(res.std))
    # std output equals sqrt(diag(P)) at stored instants
    k = np.round(res.P_hist_t / 0.01).astype(int)
    assert np.allclose(res.std[k], np.sqrt(np.diagonal(res.P_hist, axis1=1, axis2=2)), rtol=1e-12, atol=0)


def test_cov_every_zero_stores_only_final():
    s, tr = make_synthetic_flight("hover", duration=5.0)
    res = run_filter(s, EkfConfig(), init=InitialState(tr.p[0], tr.v[0], tr.q[0]))
    assert res.P_hist.shape[0] == 0 and res.P_final.shape == (15, 15)


# --------------------------------------------------------------------------------------
# 9. GPS delay
# --------------------------------------------------------------------------------------
def test_gps_delay_compensation_reduces_position_error():
    delay = 0.2
    s, tr = make_synthetic_flight("figure8", duration=60.0, noise=NOISE, gps_delay=delay, seed=41,
                                  gyro_bias=(0.01, -0.008, 0.005), accel_bias=(0.1, -0.15, 0.2))
    init = InitialState(tr.p[0], tr.v[0], tr.q[0])
    rmse = {}
    for d in (delay, 0.0):
        res = run_filter(s, matched_cfg(increment=3, gps_delay_s=d), init=init)
        m = tr.t >= 10.0
        rmse[d] = math.sqrt(np.mean(np.sum((res.p[m] - tr.p[m]) ** 2, axis=1)))
    assert rmse[delay] < rmse[0.0], rmse
    # the effect is large (0.2 s x 6 m/s = 1.2 m lag) so require a clear margin, not a tie
    assert rmse[delay] < 0.8 * rmse[0.0], rmse


# --------------------------------------------------------------------------------------
# 10. static initialisation
# --------------------------------------------------------------------------------------
def test_static_initial_state_attitude():
    s, tr = make_synthetic_flight("static", duration=3.0, declination=0.1)
    init = static_initial_state(s, 1.0)
    # truth: roll 5, pitch -3, yaw 40 deg (yaw in true north, i.e. includes the declination)
    q_true = quat_sf(Rotation.from_euler("ZYX", np.radians([40.0, -3.0, 5.0])))
    assert att_err_deg(init.q, q_true) < 0.01
    e = rot(init.q).as_euler("ZYX", degrees=True)
    assert np.allclose(e, [40.0, -3.0, 5.0], atol=0.01)
    assert np.allclose(init.v, 0.0) and np.allclose(init.b_g, 0.0) and np.allclose(init.b_a, 0.0)
    assert np.allclose(init.p, s.gps_pos_ned[0])
    assert abs(np.linalg.norm(init.q) - 1.0) < 1e-12


@pytest.mark.parametrize("decl", [-0.3, 0.0, 0.1, 0.45])
def test_static_initial_state_declination_enters_yaw_one_to_one(decl):
    s, _ = make_synthetic_flight("static", duration=2.0, declination=decl)
    init = static_initial_state(s, 1.0)
    assert np.isclose(rot(init.q).as_euler("ZYX", degrees=True)[0], 40.0, atol=0.01)


def test_run_filter_default_init_is_static_init_and_stays_put():
    s, tr = make_synthetic_flight("static", duration=20.0)
    res = run_filter(s, EkfConfig(increment=3, gps_delay_s=0.0))
    assert np.abs(res.p - tr.p).max() < 1e-3 and np.abs(res.v).max() < 1e-3
    assert att_err_deg(res.q, tr.q).max() < 0.02


# --------------------------------------------------------------------------------------
# measurement scheduling / API behaviour
# --------------------------------------------------------------------------------------
def test_measurements_outside_imu_span_are_dropped_and_counts():
    s, tr = make_synthetic_flight("hover", duration=10.0, gps_delay=0.0)
    res = run_filter(s, matched_cfg(increment=2, gps_delay_s=0.0),
                     init=InitialState(tr.p[0], tr.v[0], tr.q[0]))
    n = res.meta["n_updates"]
    assert n["gps_pos"] == s.gps_t.size == 51 and n["gps_vel"] == 51
    assert n["baro"] == s.baro_t.size and n["mag_heading"] == s.mag_t.size
    # with a 0.11 s delay the first GPS fix (t = 0) lands before the IMU start and is dropped
    res2 = run_filter(s, matched_cfg(increment=1, gps_delay_s=0.11),
                      init=InitialState(tr.p[0], tr.v[0], tr.q[0]))
    assert res2.meta["n_updates"]["gps_pos"] == 50
    assert res2.meta["n_updates"]["mag_heading"] == 0     # increment 1 has no mag


def test_gps_update_against_independent_kalman_formula():
    """One position update from a known P: dx = P H^T (H P H^T + R)^-1 r (independent linear algebra)."""
    cfg = EkfConfig(increment=3)
    q = quat_sf(Rotation.from_euler("ZYX", [0.2, 0.1, -0.1]))
    f = DroneEKF(cfg, InitialState(np.array([1.0, 2.0, 3.0]), np.zeros(3), q), G)
    P0 = f.ekf.P
    z = np.array([1.5, 1.0, 4.0])
    sig = np.array([0.5, 0.5, 0.8])
    H = np.zeros((3, 15)); H[:, :3] = np.eye(3)
    S = H @ P0 @ H.T + np.diag(sig ** 2)
    K = P0 @ H.T @ np.linalg.inv(S)
    dx = K @ (z - np.array([1.0, 2.0, 3.0]))
    P_exp = (np.eye(15) - K @ H) @ P0 @ (np.eye(15) - K @ H).T + K @ np.diag(sig ** 2) @ K.T
    f.update_gps_pos(z, sig)
    assert np.allclose(f.p, np.array([1.0, 2.0, 3.0]) + dx[:3], atol=1e-12)
    assert np.allclose(f.v, dx[3:6], atol=1e-12)
    q_exp = quat_sf(rot(q) * Rotation.from_rotvec(dx[6:9]))
    assert att_err_deg(f.q, q_exp) < 1e-9
    assert np.allclose(f.b_g, dx[9:12], atol=1e-12) and np.allclose(f.b_a, dx[12:15], atol=1e-12)
    # reset: attitude block mapped with (I - 0.5 [dth x])
    Gr = np.eye(15)
    sk = np.array([[0, -dx[8], dx[7]], [dx[8], 0, -dx[6]], [-dx[7], dx[6], 0]])
    Gr[6:9, 6:9] = np.eye(3) - 0.5 * sk
    assert np.allclose(f.ekf.P, Gr @ P_exp @ Gr.T, atol=1e-14)


def test_gating_rejects_outlier():
    cfg = EkfConfig(gate_prob=0.99)
    q = np.array([1.0, 0, 0, 0])
    f = DroneEKF(cfg, InitialState(np.zeros(3), np.zeros(3), q), G)
    res = f.update_gps_pos(np.array([500.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.5]))
    assert not res.accepted and np.all(f.p == 0.0)
    assert f.update_gps_pos(np.array([0.5, 0.0, 0.0]), np.array([0.5, 0.5, 0.5])).accepted


# --------------------------------------------------------------------------------------
# 11. real-log smoke test
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def hover_log():
    if not HOVER_LOG.exists():
        pytest.skip("pinned hover .ulg missing")
    from sensor_fusion.loader import load_log
    return load_log(HOVER_LOG)


@pytest.fixture(scope="module")
def hover_streams(hover_log):
    return ek.streams_from_log(hover_log)


def test_streams_from_log_hover(hover_log, hover_streams):
    ppigrf = pytest.importorskip("ppigrf")
    s = hover_streams
    lat0, lon0, alt0 = s.origin_lla
    assert np.allclose(s.gps_pos_ned[0], 0.0, atol=1e-6)
    assert (lat0, lon0, alt0) == (hover_log.gps.lat[0], hover_log.gps.lon[0], hover_log.gps.alt[0])
    # baro height-change datum: first-second mean maps to ~0 down position
    m = s.baro_t <= s.baro_t[0] + 1.0
    assert np.abs(s.baro_d[m]).max() < 0.5
    assert abs(s.baro_d[m].mean() - s.gps_pos_ned[0, 2]) < 1e-9
    # gravity: Somigliana closed form (WGS-84) + free-air correction
    sin2 = math.sin(lat0) ** 2
    g_som = 9.7803253359 * (1 + 0.00193185265241 * sin2) / math.sqrt(1 - 0.00669437999013 * sin2) - 3.086e-6 * alt0
    assert abs(s.g - g_som) < 1e-4          # constants of the closed form quoted to ~1e-10; tolerance for rounding of coefficients
    assert 9.76 < s.g < 9.84
    # declination: ppigrf at the origin and the log's UTC date
    when = hover_log.info["gps_utc_start"]
    when = when.astimezone(dt.timezone.utc).replace(tzinfo=None) if when.tzinfo else when
    be, bn, bu = ppigrf.igrf(math.degrees(lon0), math.degrees(lat0), alt0 / 1000.0, when)
    decl = math.atan2(float(np.ravel(be)[0]), float(np.ravel(bn)[0]))
    assert abs(s.declination - decl) < 1e-9
    assert abs(s.declination) < math.radians(30)
    # structure
    assert s.gps_pos_sigma.shape == (s.gps_t.size, 3) and s.gps_vel_sigma.shape == (s.gps_t.size,)
    assert np.all(np.diff(s.imu_t) > 0)
    # mag in tesla (|B| ~ 25-65 uT)
    nb = np.linalg.norm(s.mag_field, axis=1)
    assert 15e-6 < np.median(nb) < 80e-6


def test_run_filter_on_hover_log_40s(hover_log, hover_streams):
    s = hover_streams.sliced(40.0)
    res = run_filter(s, EkfConfig(increment=3, cov_every=500))
    assert np.all(np.isfinite(res.p)) and np.all(np.isfinite(res.v)) and np.all(np.isfinite(res.q))
    assert np.all(np.isfinite(res.b_g)) and np.all(np.isfinite(res.b_a)) and np.all(np.isfinite(res.std))
    assert is_spd(res.P_final)
    for P in res.P_hist:
        assert is_spd(P)
    assert np.abs(np.linalg.norm(res.q, axis=1) - 1.0).max() < 1e-12
    ref = ek.ekf2_reference(hover_log, s.origin_lla)
    sel = (ref["t_pos"] >= 10.0) & (ref["t_pos"] <= 40.0)
    assert sel.sum() > 10
    p_i = np.column_stack([np.interp(ref["t_pos"][sel], res.t, res.p[:, i]) for i in range(3)])
    dh = np.linalg.norm(p_i[:, :2] - ref["pos_ned"][sel][:, :2], axis=1)
    assert dh.max() < 3.0, dh.max()


def test_ekf2_reference_origin_guard(hover_log, hover_streams):
    lat0, lon0, alt0 = hover_streams.origin_lla
    ref = ek.ekf2_reference(hover_log, hover_streams.origin_lla)
    assert ref["pos_ned"].shape == (ref["t_pos"].size, 3) and ref["q"].shape[1] == 4
    # an origin that is not the first GPS fix must be rejected unless its MSL height is supplied
    with pytest.raises(ValueError):
        ek.ekf2_reference(hover_log, (lat0 + 1e-4, lon0, alt0))
    ref2 = ek.ekf2_reference(hover_log, (lat0 + 1e-4, lon0, alt0), alt_msl0=float(hover_log.gps.alt_msl[0]))
    # moving the origin 1e-4 rad (~637 m) north shifts EKF2 positions by about -637 m in north (translation only)
    dn = ref2["pos_ned"][:, 0] - ref["pos_ned"][:, 0]
    assert np.allclose(dn, -1e-4 * 6.371e6, rtol=0.01)
    assert np.allclose(ref2["pos_ned"][:, 2], ref["pos_ned"][:, 2])
