"""Tests for the Group 8 MEKF extensions in ``adcs.estimation`` (M9).

Written from ``docs/specs/adcs_closed_loop.md`` section 2 (tests 1-9) and
``docs/specs/adcs_estimation.md`` sections 0 and 3, not from the implementation.
Oracles:

* ``common.ekf.van_loan`` on the augmented 9-state continuous model (propagation);
* central finite differences of ``h(q (x) exp(dtheta), bm + dbm) = R^T v + bm`` and of
  ``n_i^T R(q (x) exp(dtheta))^T s`` built with ``scipy.spatial.transform.Rotation``,
  combined with a textbook (Joseph form) EKF update written in this file;
* scipy ``Rotation`` for the quaternion injection;
* chi-square / NIS / NEES statistics on constructed truth (``adcs.simulate.run``
  pointing truth, ``Ephemeris`` + ``FieldTable`` reference vectors) with known sensor
  noise: matched models give NIS per dof ~ 1 and attitude NEES ~ 3;
* hand-built noise-free data for the TRIAD-guard tests (angles known by construction).

Speed: the pointing truth (2 orbits of ``adcs.simulate.run``) is computed once in a
worker process that is started when the module starts (autouse fixture), so it
overlaps with the fast tests; the 18 statistical filter runs (6 seeds x 3 scenarios)
are then spread over the process pool.  All randomness is seeded, so results do not
depend on scheduling.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from common import quaternion as quat
from common.ekf import van_loan
from adcs.estimation import (
    MEKF,
    STATE_NAMES_MAG_BIAS,
    MekfConfig,
    attitude_errors,
    css_vector_complete,
    nees,
    run_mekf,
    triad,
    triad_covariance,
    triad_init,
    vector_angle,
)
from adcs.sensors import (
    CSS_NORMALS,
    GyroConfig,
    MagnetometerConfig,
    SunSensorConfig,
    albedo_currents,
    measure_series,
    sun_sensor_currents,
    sun_vector_from_currents,
)

# --------------------------------------------------------------------------- helpers
ORBITS_STAT = 2.0                 # orbits of pointing truth for the statistical tests
SEEDS_STAT = tuple(range(6))      # spec: >= 6 seeds
MAG_BIAS_TRUE = np.array([200.0, -150.0, 100.0]) * 1e-9     # T, prototype setup
SIGMA_MB0_STAT = 300e-9           # T, prototype setup
FOV_TRUE = np.deg2rad(60.0)
OFFSET_DEG = np.array([3.0, -2.0, 2.0])                     # prototype diode-mode start error
SETTLE = 300                      # s, skipped before computing statistics (k_init = 0)


def _sf(q):
    """scalar-first quaternion -> scipy Rotation (own conversion, no repo code)."""
    return Rotation.from_quat(np.roll(np.asarray(q, float), -1, axis=-1))


def _skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _rand_q(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def _rand_unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def _spd(rng, scales):
    """Random SPD matrix with correlated entries and the given per-state scales."""
    n = len(scales)
    A = rng.normal(size=(n, n))
    C = A @ A.T + n * np.eye(n)
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    s = np.asarray(scales, float)
    return np.outer(s, s) * C


def _scaled_max_err(A, B, scales):
    """max |A - B| in units of the per-state scales (matrix with entries of very
    different physical units: rad^2, (rad/s)^2, T^2)."""
    s = np.asarray(scales, float)
    return np.max(np.abs(A - B) / np.outer(s, s))


def _ekf_oracle(P, H, Rm, r):
    """Textbook EKF update (Joseph form).  Returns ``dx = K r`` and the updated P."""
    S = H @ P @ H.T + Rm
    K = P @ H.T @ np.linalg.inv(S)
    A = np.eye(P.shape[0]) - K @ H
    return K @ r, A @ P @ A.T + K @ Rm @ K.T, S


def _reset_oracle(P, dth):
    """Covariance reset G P G^T with G = blockdiag(I - 1/2 [dth x], I, ...)."""
    G = np.eye(P.shape[0])
    G[:3, :3] = np.eye(3) - 0.5 * _skew(dth)
    return G @ P @ G.T


def _inject_oracle(q, dth):
    """Rotation matrix of q (x) exp(dth) from scipy."""
    return (_sf(q) * Rotation.from_rotvec(dth)).as_matrix()


def _fd_att_jacobian(fun, q, h=1e-7):
    """Central-difference d fun(q (x) exp(d)) / d d at d = 0; ``fun(R)`` takes the
    scipy Rotation of the perturbed attitude and returns a vector."""
    R0 = _sf(q)
    cols = []
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        cols.append((fun(R0 * Rotation.from_rotvec(e)) - fun(R0 * Rotation.from_rotvec(-e))) / (2 * h))
    return np.stack(cols, axis=1)


# =========================================================================== 1. defaults
def test_defaults_and_config_validation():
    assert MekfConfig().n_states == 6
    assert MekfConfig(mag_bias=True).n_states == 9
    assert len(STATE_NAMES_MAG_BIAS) == 9
    # a 9-state filter rejects a 6x6 P0 (and a 6-state filter a 9x9)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        MEKF(q0, np.zeros(3), 1e-4 * np.eye(6), MekfConfig(mag_bias=True))
    with pytest.raises(ValueError):
        MEKF(q0, np.zeros(3), 1e-4 * np.eye(9), MekfConfig())
    # valid shapes are accepted
    assert MEKF(q0, np.zeros(3), 1e-4 * np.eye(9), MekfConfig(mag_bias=True)).P.shape == (9, 9)
    assert MEKF(q0, np.zeros(3), 1e-4 * np.eye(6), MekfConfig()).P.shape == (6, 6)
    # an unknown css_mode raises (at config or at filter construction)
    with pytest.raises(ValueError):
        MEKF(q0, np.zeros(3), 1e-4 * np.eye(6), MekfConfig(css_mode="x"))
    # defaults of the Group 7 filter
    cfg = MekfConfig()
    assert cfg.css_mode == "vector" and cfg.mag_bias is False
    assert cfg.triad_min_angle == 0.0 and cfg.init_att_sigma_floor == 0.0
    assert cfg.albedo_model == 0.0 and cfg.css_fov is None
    assert cfg.diode_min_current == 0.55
    assert cfg.sigma_mb0 == 200e-9 and cfg.mag_bias_rw == 0.0


# =========================================================================== 2. 9-state propagation
@pytest.mark.parametrize("mb_rw", [0.0, 3e-8])
@pytest.mark.parametrize("dt", [0.3, 1.0, 4.0])
def test_nine_state_propagation_matches_van_loan(dt, mb_rw):
    rng = np.random.default_rng(int(1000 * dt) + int(mb_rw * 1e9))
    cfg = MekfConfig(mag_bias=True, mag_bias_rw=mb_rw)
    q0 = _rand_q(rng)
    b0 = 1e-3 * rng.normal(size=3)
    bm0 = 2e-7 * rng.normal(size=3)
    w_meas = 0.05 * rng.normal(size=3)
    scales = [2e-2] * 3 + [1e-3] * 3 + [3e-7] * 3
    P0 = _spd(rng, scales)
    f = MEKF(q0, b0, P0, cfg, bm0=bm0)
    f.propagate(w_meas, dt)

    # independent continuous model: dth' = -[w x] dth - db - n_v, db' = n_u, dbm' = n_m
    w_hat = w_meas - b0
    F = np.zeros((9, 9))
    F[:3, :3] = -_skew(w_hat)
    F[:3, 3:6] = -np.eye(3)
    G = np.diag([-1.0] * 3 + [1.0] * 6)
    Qc = np.diag([cfg.arw ** 2] * 3 + [cfg.rrw ** 2] * 3 + [mb_rw ** 2] * 3)
    Phi, Qd = van_loan(F, Qc, dt, G)
    P_exp = Phi @ P0 @ Phi.T + Qd
    sc = np.sqrt(np.diag(P_exp))
    assert _scaled_max_err(f.P, P_exp, sc) < 1e-10
    # the random-walk term really is in the covariance (otherwise the test is vacuous)
    if mb_rw > 0.0:
        assert np.allclose(Qd[6:, 6:], mb_rw ** 2 * dt * np.eye(3), rtol=1e-12)
        assert Qd[6, 6] > 1e-3 * P_exp[6, 6]

    # nominal state: q (x) exp(w_hat dt), b and bm unchanged
    R_exp = (_sf(q0) * Rotation.from_rotvec(w_hat * dt)).as_matrix()
    assert np.max(np.abs(_sf(f.q).as_matrix() - R_exp)) < 1e-12
    assert abs(np.linalg.norm(f.q) - 1.0) < 1e-14
    assert np.array_equal(f.b, b0)
    assert np.array_equal(f.bm, bm0)


# =========================================================================== 3. mag update, bias states
@pytest.mark.parametrize("seed", range(4))
def test_mag_update_with_bias_states_matches_textbook_ekf(seed):
    rng = np.random.default_rng(100 + seed)
    cfg = MekfConfig(mag_bias=True, sigma_mag=50e-9)
    q_hat = _rand_q(rng)
    bm_hat = 3e-7 * rng.normal(size=3)
    b_gyro = 1e-3 * rng.normal(size=3)
    b_eci = 3e-5 * rng.normal(size=3)
    scales = [1e-2] * 3 + [1e-3] * 3 + [3e-7] * 3
    P0 = _spd(rng, scales)
    f = MEKF(q_hat, b_gyro, P0, cfg, bm0=bm_hat)

    pred = _sf(q_hat).inv().apply(b_eci) + bm_hat
    z = pred + 3e-7 * rng.normal(size=3)
    r = z - pred

    # H by central differences of h(dth, dbm) = R(q (x) exp(dth))^T b_eci + bm + dbm
    h = 1e-7
    H = np.zeros((3, 9))
    H[:, :3] = _fd_att_jacobian(lambda R: R.inv().apply(b_eci), q_hat, h)
    # columns 6..8: central differences of the full function in the bias components
    R_att = _sf(q_hat).inv().apply(b_eci)
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        hp = R_att + (bm_hat + e)
        hm = R_att + (bm_hat - e)
        H[:, 6 + i] = (hp - hm) / (2 * h)

    dx_o, P_o, S_o = _ekf_oracle(P0, H, cfg.sigma_mag ** 2 * np.eye(3), r)
    res = f.update_mag(z, b_eci, t=12.0)

    assert res.accepted and res.name == "mag"
    assert res.dx.shape == (9,)
    assert np.max(np.abs(res.residual - r)) < 1e-12 * np.linalg.norm(b_eci)
    for sl in (slice(0, 3), slice(3, 6), slice(6, 9)):
        assert np.max(np.abs(res.dx[sl] - dx_o[sl])) < 1e-6 * np.max(np.abs(dx_o[sl]))
    assert np.max(np.abs(res.S - S_o)) < 1e-6 * np.max(np.abs(S_o))
    nis_o = float(r @ np.linalg.solve(S_o, r))
    assert abs(res.nis - nis_o) < 1e-6 * nis_o

    # injection (scipy for the attitude, addition for the biases)
    dth = dx_o[:3]
    assert np.max(np.abs(_sf(f.q).as_matrix() - _inject_oracle(q_hat, dx_o[:3]))) < 1e-7
    assert np.max(np.abs(_sf(f.q).as_matrix() - _inject_oracle(q_hat, res.dx[:3]))) < 1e-12
    assert np.max(np.abs(f.b - (b_gyro + res.dx[3:6]))) < 1e-15
    assert np.max(np.abs(f.bm - (bm_hat + res.dx[6:9]))) < 1e-18
    assert np.max(np.abs(f.bm - (bm_hat + dx_o[6:9]))) < 1e-6 * np.max(np.abs(dx_o[6:9]))

    # covariance: Joseph form followed by the reset G = blockdiag(I - 1/2 [dth x], I, I)
    P_final_o = _reset_oracle(P_o, dth)
    assert _scaled_max_err(f.P, P_final_o, np.sqrt(np.diag(P_o))) < 1e-6
    # the reset is not a no-op here (so the check above does test it)
    assert _scaled_max_err(P_o, P_final_o, np.sqrt(np.diag(P_o))) > 1e-4
    assert _scaled_max_err(f.P, P_o, np.sqrt(np.diag(P_o))) > 1e-4
    assert np.allclose(f.P, f.P.T, rtol=0, atol=1e-14 * np.max(np.abs(f.P)))


def test_mag_update_bias_estimate_is_observable_from_a_noise_free_bias():
    """A pure magnetometer offset (no attitude error) is attributed to the bias states
    in proportion to the prior variances: noise-free, attitude prior tiny => dbm = offset."""
    rng = np.random.default_rng(5)
    cfg = MekfConfig(mag_bias=True, sigma_mag=1e-9)
    q_hat = _rand_q(rng)
    b_eci = 3e-5 * rng.normal(size=3)
    P0 = np.diag([1e-14] * 3 + [1e-6] * 3 + [(3e-7) ** 2] * 3)   # attitude known to 1e-7 rad
    f = MEKF(q_hat, np.zeros(3), P0, cfg)
    offset = np.array([2e-7, -1e-7, 1.5e-7])
    z = _sf(q_hat).inv().apply(b_eci) + offset
    f.update_mag(z, b_eci)
    # attitude prior 1e-7 rad x 3e-5 T = 3e-12 T of leverage vs 3e-7 T bias prior -> dbm ~ offset
    assert np.max(np.abs(f.bm - offset)) < 1e-4 * np.linalg.norm(offset)


# =========================================================================== 4. diode update
def _diode_scenario(rng, s_b, n_states, css_fov=None, albedo_model=0.0, sigma_albedo=0.0,
                    sigma_diode=0.05):
    """Filter at a random attitude with the sun at body-axes direction ``s_b``."""
    q_hat = _rand_q(rng)
    s_b = np.asarray(s_b, float) / np.linalg.norm(s_b)
    s_eci = _sf(q_hat).apply(s_b)
    cfg = MekfConfig(css_mode="diodes", mag_bias=(n_states == 9), css_fov=css_fov,
                     albedo_model=albedo_model, sigma_albedo=sigma_albedo,
                     sigma_diode=sigma_diode)
    scales = [1e-2] * 3 + [1e-3] * 3 + ([3e-7] * 3 if n_states == 9 else [])
    P0 = _spd(rng, scales)
    f = MEKF(q_hat, 1e-3 * rng.normal(size=3), P0, cfg)
    cos_pred = CSS_NORMALS @ s_b
    return f, cfg, q_hat, s_eci, s_b, P0, cos_pred


def _check_diode_update(f, cfg, q_hat, s_eci, P0, currents, idx, r_eci=None, alb=None):
    """Compare ``update_css_diodes`` with a textbook update over diodes ``idx``."""
    n = P0.shape[0]
    alb = np.zeros(6) if alb is None else alb
    b_before = f.b
    # finite-difference H of n_i^T R(q (x) exp(dth))^T s over the selected diodes
    H = np.zeros((len(idx), n))
    for row, i in enumerate(idx):
        H[row, :3] = _fd_att_jacobian(
            lambda R: np.array([CSS_NORMALS[i] @ R.inv().apply(s_eci)]), q_hat, 1e-7)[0]
    cos_pred = np.array([CSS_NORMALS[i] @ _sf(q_hat).inv().apply(s_eci) for i in idx])
    r = currents[idx] - (cos_pred + alb[idx])
    var = cfg.sigma_diode ** 2 * np.ones(len(idx))
    if cfg.albedo_model > 0.0:
        var = var + (cfg.sigma_albedo * alb[idx] / cfg.albedo_model) ** 2
    Rm = np.diag(var)
    dx_o, P_o, S_o = _ekf_oracle(P0, H, Rm, r)

    res = f.update_css_diodes(currents, s_eci, r_eci=r_eci, t=3.0)
    assert res is not None and res.accepted and res.name == "css"
    assert res.residual.shape == (len(idx),)
    assert np.max(np.abs(res.residual - r)) < 1e-9
    assert np.max(np.abs(res.S - S_o)) < 1e-6 * np.max(np.abs(S_o))
    for sl in ((slice(0, 3), slice(3, 6), slice(6, 9)) if n == 9 else (slice(0, 3), slice(3, 6))):
        assert np.max(np.abs(res.dx[sl] - dx_o[sl])) < 1e-6 * np.max(np.abs(dx_o[sl]))
    assert np.max(np.abs(_sf(f.q).as_matrix() - _inject_oracle(q_hat, res.dx[:3]))) < 1e-12
    assert np.max(np.abs(f.b - (b_before + res.dx[3:6]))) < 1e-15
    assert _scaled_max_err(f.P, _reset_oracle(P_o, dx_o[:3]), np.sqrt(np.diag(P0))) < 1e-6
    return res


@pytest.mark.parametrize("n_states", [6, 9])
def test_diode_update_matches_textbook_ekf(n_states):
    rng = np.random.default_rng(31 + n_states)
    f, cfg, q_hat, s_eci, s_b, P0, cos_pred = _diode_scenario(
        rng, [0.62, -0.58, 0.60], n_states)
    assert np.sum(cos_pred > 0.55) == 3
    currents = np.maximum(cos_pred, 0.0) + 0.03 * rng.normal(size=6)
    idx = np.nonzero(cos_pred > 0.55)[0]
    _check_diode_update(f, cfg, q_hat, s_eci, P0, currents, idx)


def test_diode_update_selection_rules():
    rng = np.random.default_rng(77)

    # (a) predicted cosine between 0.5 and 0.55 is not used, even if the current is lit
    f, cfg, q_hat, s_eci, s_b, P0, cos = _diode_scenario(rng, [0.8, 0.53, 0.2813], 6)
    assert 0.5 < cos[2] < 0.55 and cos[0] > 0.55
    cur = np.maximum(cos, 0.0)
    res = f.update_css_diodes(cur, s_eci)
    assert res.residual.size == 1                      # only +x

    # (b) measured current below 0.55 / 2 is not used although predicted lit
    f, cfg, q_hat, s_eci, s_b, P0, cos = _diode_scenario(rng, [0.8, 0.6, 0.0], 6)
    assert cos[0] > 0.55 and cos[2] > 0.55
    cur = np.maximum(cos, 0.0)
    cur[2] = 0.2
    assert cur[2] < 0.55 / 2
    res = f.update_css_diodes(cur, s_eci)
    assert res.residual.size == 1
    # same, with the +y current just above the limit -> it is used
    f, cfg, q_hat, s_eci, s_b, P0, cos = _diode_scenario(rng, [0.8, 0.6, 0.0], 6)
    cur = np.maximum(cos, 0.0)
    cur[2] = 0.28
    res = f.update_css_diodes(cur, s_eci)
    assert res.residual.size == 2

    # (c) css_fov removes diodes whose predicted cosine is below cos(fov)
    for fov_deg, n_used in ((None, 2), (50.0, 1), (40.0, 1), (60.0, 2)):
        fov = None if fov_deg is None else np.deg2rad(fov_deg)
        f, cfg, q_hat, s_eci, s_b, P0, cos = _diode_scenario(
            np.random.default_rng(5), [0.8, 0.6, 0.0], 6, css_fov=fov)
        cur = np.maximum(cos, 0.0)
        res = f.update_css_diodes(cur, s_eci)
        assert res.residual.size == n_used, (fov_deg, res.residual.size)
        # the surviving diode is +x (largest cosine), whose residual is zero (noise-free)
        assert abs(res.residual[0]) < 1e-12


def test_diode_update_fov_selection_oracle_against_textbook():
    """With css_fov = 50 deg (cos 0.643) the +y diode (cos 0.6) is dropped: the update
    equals a textbook update over +x only."""
    rng = np.random.default_rng(8)
    f, cfg, q_hat, s_eci, s_b, P0, cos = _diode_scenario(
        rng, [0.8, 0.6, 0.0], 6, css_fov=np.deg2rad(50.0))
    currents = np.maximum(cos, 0.0) + 0.02 * rng.normal(size=6)
    currents[1:] = np.maximum(currents[1:], 0.0)
    _check_diode_update(f, cfg, q_hat, s_eci, P0, currents, np.array([0]))


def test_diode_update_none_when_nothing_usable_and_state_untouched():
    rng = np.random.default_rng(9)
    # (a) eclipse: all measured currents ~ 0 -> None
    f, cfg, q_hat, s_eci, s_b, P0, cos = _diode_scenario(rng, [0.62, -0.58, 0.60], 6)
    q_before, P_before, b_before = f.q, f.P, f.b
    assert f.update_css_diodes(np.zeros(6), s_eci) is None
    assert f.update_css_diodes(0.01 * rng.normal(size=6), s_eci) is None
    # (b) lit, but no predicted cosine above cos(40 deg): s = (1,1,1)/sqrt3 has cosines 0.577 < 0.766
    # (for a unit vector max_i |c_i| >= 0.577 > 0.55, so the fov is what removes every diode here)
    f4, cfg4, q4, s_eci4, s_b4, P04, cos4 = _diode_scenario(
        rng, [1.0, 1.0, 1.0], 6, css_fov=np.deg2rad(40.0))
    cur4 = np.maximum(cos4, 0.0)
    q4b, P4b = f4.q, f4.P
    assert f4.update_css_diodes(cur4, s_eci4) is None
    assert np.array_equal(f4.q, q4b)
    assert np.array_equal(f4.P, P4b)
    # (c) states untouched in the eclipse case
    assert np.array_equal(f.q, q_before)
    assert np.array_equal(f.P, P_before)
    assert np.array_equal(f.b, b_before)
    # no innovation was logged by the empty updates
    assert f.ekf.innovations("css").nis.size == 0


def test_diode_update_with_albedo_model():
    rng = np.random.default_rng(12)
    alb_coeff, sig_alb, sig_d = 0.3, 0.5, 0.04
    f, cfg, q_hat, s_eci, s_b, P0, cos = _diode_scenario(
        rng, [0.62, -0.58, 0.60], 6, albedo_model=alb_coeff, sigma_albedo=sig_alb,
        sigma_diode=sig_d)
    # spacecraft above the surface with the Sun 53 deg from the zenith (illumination 0.6)
    u = np.cross(s_eci, _rand_unit(rng))
    u /= np.linalg.norm(u)
    r_hat = 0.6 * s_eci + 0.8 * u
    r_eci = 6.778e6 * r_hat
    alb = albedo_currents(q_hat, r_eci, s_eci, alb_coeff)
    assert alb.shape == (6,) and np.all(alb >= 0.0) and np.max(alb) > 0.01
    idx = np.nonzero(cos > 0.55)[0]
    assert np.all(alb[idx] > 0.0)
    currents = np.maximum(cos, 0.0) + alb + 0.02 * rng.normal(size=6)
    res = _check_diode_update(f, cfg, q_hat, s_eci, P0, currents, idx, r_eci=r_eci, alb=alb)
    # R_ii = sigma_d^2 + (sigma_albedo a_i / albedo_model)^2 : read back from S
    Hm = np.zeros((idx.size, 6))
    for row, i in enumerate(idx):
        Hm[row, :3] = _fd_att_jacobian(
            lambda R: np.array([CSS_NORMALS[i] @ R.inv().apply(s_eci)]), q_hat, 1e-7)[0]
    R_exp = np.diag(sig_d ** 2 + (sig_alb * alb[idx] / alb_coeff) ** 2)
    assert np.max(np.abs((res.S - Hm @ P0 @ Hm.T) - R_exp)) < 1e-6 * np.max(np.diag(R_exp))
    # the albedo inflation is non-trivial: it exceeds sigma_d^2 for the used diodes
    assert np.all(np.diag(R_exp) > sig_d ** 2)
    # and the residual equals I - (cos_pred + albedo): compare with the albedo-free residual
    f0, cfg0, q0_, s0_, sb0_, P00, cos0 = _diode_scenario(
        np.random.default_rng(12), [0.62, -0.58, 0.60], 6, sigma_diode=sig_d)
    res0 = f0.update_css_diodes(currents, s_eci)
    assert np.max(np.abs((res0.residual - res.residual) - alb[idx])) < 1e-9


# =========================================================================== 7. triad_init, helpers
@pytest.mark.parametrize("inflate", [1.0, 3.0])
def test_triad_init_nine_state_blocks_and_floor(inflate):
    rng = np.random.default_rng(21)
    cfg = MekfConfig(mag_bias=True, sigma_mb0=300e-9, init_att_sigma_floor=np.deg2rad(5.0),
                     init_inflate=inflate)
    R_true = Rotation.random(random_state=3)
    b_eci = 30e-6 * _rand_unit(rng)
    s_eci = _rand_unit(rng)
    while abs(b_eci @ s_eci / np.linalg.norm(b_eci)) > 0.8:
        s_eci = _rand_unit(rng)
    mag_b = R_true.inv().apply(b_eci) + 15e-9 * rng.normal(size=3)
    sun_b = R_true.inv().apply(s_eci) + 0.06 * rng.normal(size=3)

    q0, P0 = triad_init(mag_b, sun_b, b_eci, s_eci, cfg)
    assert P0.shape == (9, 9)
    assert np.allclose(P0, P0.T, rtol=0, atol=1e-30)
    assert np.all(P0[:3, 3:] == 0.0) and np.all(P0[3:6, 6:] == 0.0)
    assert np.allclose(P0[3:6, 3:6], cfg.sigma_b0 ** 2 * np.eye(3), rtol=1e-12, atol=0)
    assert np.allclose(P0[6:, 6:], (300e-9) ** 2 * np.eye(3), rtol=1e-12, atol=0)

    # attitude block: inflate * triad_covariance(sigma_mag_eff / |b|, sigma_sun) + floor^2 I,
    # with sigma_mag_eff = hypot(sigma_mag, sigma_mb0); magnetometer primary here
    s_mag = np.hypot(cfg.sigma_mag, cfg.sigma_mb0) / np.linalg.norm(mag_b)
    assert s_mag < cfg.sigma_sun
    P_tri = triad_covariance(mag_b, sun_b, s_mag, cfg.sigma_sun)
    floor2 = np.deg2rad(5.0) ** 2
    assert np.max(np.abs(P0[:3, :3] - inflate * P_tri - floor2 * np.eye(3))) < 1e-14
    assert np.max(np.abs((P0[:3, :3] - floor2 * np.eye(3)) / inflate - P_tri)) < 1e-14
    assert quat.angle_between(q0, triad(mag_b, sun_b, b_eci, s_eci)) < 1e-12


def test_triad_init_nine_state_primary_follows_inflated_mag_sigma():
    """A weak field with the bias uncertainty added makes the sun the primary vector,
    whereas sigma_mag alone would keep the magnetometer primary."""
    rng = np.random.default_rng(22)
    b_eci = 3e-6 * _rand_unit(rng)                      # 3 uT
    s_eci = _rand_unit(rng)
    R_true = Rotation.random(random_state=4)
    mag_b = R_true.inv().apply(b_eci)
    sun_b = R_true.inv().apply(s_eci)
    cfg6 = MekfConfig()
    cfg9 = MekfConfig(mag_bias=True, sigma_mb0=300e-9)
    nb = np.linalg.norm(mag_b)
    assert cfg6.sigma_mag / nb < cfg6.sigma_sun < np.hypot(cfg9.sigma_mag, cfg9.sigma_mb0) / nb
    q6, _ = triad_init(mag_b, sun_b, b_eci, s_eci, cfg6)
    q9, P9 = triad_init(mag_b, sun_b, b_eci, s_eci, cfg9)
    assert quat.angle_between(q6, triad(mag_b, sun_b, b_eci, s_eci)) < 1e-12
    assert quat.angle_between(q9, triad(sun_b, mag_b, s_eci, b_eci)) < 1e-12
    s_mag9 = np.hypot(cfg9.sigma_mag, cfg9.sigma_mb0) / nb
    assert np.max(np.abs(P9[:3, :3] - triad_covariance(sun_b, mag_b, cfg9.sigma_sun, s_mag9))) < 1e-14


def test_triad_init_six_state_floor_only_adds_to_attitude_block():
    rng = np.random.default_rng(23)
    b_eci, s_eci = 30e-6 * _rand_unit(rng), _rand_unit(rng)
    R_true = Rotation.random(random_state=5)
    mag_b, sun_b = R_true.inv().apply(b_eci), R_true.inv().apply(s_eci)
    _, P0 = triad_init(mag_b, sun_b, b_eci, s_eci, MekfConfig())
    _, Pf = triad_init(mag_b, sun_b, b_eci, s_eci, MekfConfig(init_att_sigma_floor=0.05))
    assert Pf.shape == (6, 6)
    assert np.max(np.abs(Pf[:3, :3] - P0[:3, :3] - 0.05 ** 2 * np.eye(3))) < 1e-15
    assert np.array_equal(Pf[3:, 3:], P0[3:, 3:])


def test_css_vector_complete_truth_table():
    m = 0.55
    table = [
        ([0.8, 0, 0.7, 0, 0.9, 0], True),
        ([0, 0.8, 0, 0.7, 0, 0.9], True),
        ([0, 0.8, 0.7, 0, 0, 0.9], True),
        ([0.8, 0, 0.7, 0, 0.5, 0], False),               # z axis only 0.5 < 0.55
        ([0.8, 0, 0, 0, 0.9, 0], False),                 # y axis dark
        ([0, 0, 0.7, 0, 0.9, 0], False),                 # x axis dark
        ([0.0] * 6, False),
        ([0.5501, 0, 0.5501, 0, 0.5501, 0], True),
        ([0.5499, 0, 0.5501, 0, 0.5501, 0], False),
        ([0.3, 0.5, 0.6, 0.0, 0.0, 0.7], False),         # x axis: max(0.3, 0.5) < 0.55
        ([-0.1, 0.8, 0.7, -0.2, 0.9, 0.1], True),        # noisy negative dark diodes
    ]
    for cur, expected in table:
        assert css_vector_complete(np.array(cur, float), m) is expected or \
            css_vector_complete(np.array(cur, float), m) == expected, cur
    # brute-force oracle on random currents and thresholds
    rng = np.random.default_rng(24)
    for _ in range(500):
        c = rng.uniform(-0.2, 1.0, size=6)
        thr = rng.uniform(0.3, 0.8)
        exp = all((c[2 * a] > thr) or (c[2 * a + 1] > thr) for a in range(3))
        assert bool(css_vector_complete(c, thr)) == exp


def test_vector_angle_against_arccos():
    rng = np.random.default_rng(25)
    for _ in range(300):
        a = rng.normal(size=3) * 10.0 ** rng.uniform(-6, 2)
        b = rng.normal(size=3) * 10.0 ** rng.uniform(-6, 2)
        exp = np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))
        assert abs(vector_angle(a, b) - exp) < 1e-12
    x = np.array([1.0, 0, 0])
    assert vector_angle(x, 7.0 * x) == pytest.approx(0.0, abs=1e-15)
    assert vector_angle(x, -3.0 * x) == pytest.approx(np.pi, abs=1e-12)
    assert vector_angle(x, np.array([0, 2.0, 0])) == pytest.approx(np.pi / 2, abs=1e-14)
    # tiny angles (where arccos of the dot product is useless): rotate v about a perpendicular axis
    v = _rand_unit(rng)
    ax = np.cross(v, _rand_unit(rng))
    ax /= np.linalg.norm(ax)
    for ang in (1e-9, 1e-6, 1e-3):
        w = Rotation.from_rotvec(ang * ax).apply(v)
        assert vector_angle(v, 5.0 * w) == pytest.approx(ang, rel=1e-6)


# =========================================================================== 8. triad_min_angle
def _guard_data(s_hat, alpha_deg, sensors_diodes=False):
    """Noise-free sensor record: identity attitude, fixed sun ``s_hat``, field making
    angle ``alpha_deg[k]`` with the sun at sample k (known by construction)."""
    s_hat = np.asarray(s_hat, float) / np.linalg.norm(s_hat)
    n = len(alpha_deg)
    perp = np.cross(s_hat, [0.3, -0.5, 0.8])
    perp /= np.linalg.norm(perp)
    a = np.deg2rad(alpha_deg)
    b_eci = 30e-6 * (np.cos(a)[:, None] * s_hat + np.sin(a)[:, None] * perp)
    s_eci = np.tile(s_hat, (n, 1))
    t = np.arange(n, dtype=float)
    q = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    lit = np.ones(n, bool)
    data = measure_series(t, q, b_eci, s_eci, lit, seed=0,
                          mag=MagnetometerConfig(sigma=0.0), css=SunSensorConfig(sigma_diode=0.0),
                          gyro=GyroConfig(arw=0.0, rrw=0.0, bias0_sigma=0.0))
    return data, b_eci, s_eci


def test_run_mekf_triad_min_angle_picks_first_sample_above_guard():
    n = 600
    alpha = 5.0 + 170.0 * np.arange(n) / (n - 1)            # 5 deg ... 175 deg
    data, b_eci, s_eci = _guard_data([1.0, 0.0, 0.0], alpha)
    assert np.all(data.sun_valid)
    guard = np.deg2rad(20.0)

    # default config: no guard -> initialises at the very first valid sample
    res0 = run_mekf(data, b_eci, s_eci, MekfConfig())
    assert res0.k_init == 0
    assert vector_angle(data.mag[0], data.sun[0]) < guard    # the guard would have rejected it

    res = run_mekf(data, b_eci, s_eci, MekfConfig(triad_min_angle=guard))
    ang = np.array([np.arccos(np.clip(m @ s / (np.linalg.norm(m) * np.linalg.norm(s)), -1, 1))
                    for m, s in zip(data.mag, data.sun)])
    k_exp = int(np.nonzero(ang >= guard)[0][0])
    assert k_exp > 20                                       # construction: first valid samples are < guard
    assert res.k_init == k_exp
    assert ang[res.k_init] >= guard and np.all(ang[:res.k_init] < guard)
    # rows before the initialisation are NaN, rows from it are finite
    for arr in (res.q, res.b, res.std, res.P_att):
        assert np.all(np.isnan(arr[:k_exp]))
        assert np.all(np.isfinite(arr[k_exp:]))
    # noise-free data: the TRIAD attitude is the true (identity) attitude
    assert np.max(quat.angle_between(res.q[k_exp:], np.tile([1.0, 0, 0, 0], (n - k_exp, 1)))) < 1e-9

    # a guard exactly between two samples picks the later one; vacuous guard picks sample 0
    res_hi = run_mekf(data, b_eci, s_eci, MekfConfig(triad_min_angle=np.deg2rad(100.0)))
    k_hi = int(np.nonzero(ang >= np.deg2rad(100.0))[0][0])
    assert res_hi.k_init == k_hi


def test_run_mekf_impossible_triad_guard_never_initialises():
    n = 600
    alpha = 5.0 + 170.0 * np.arange(n) / (n - 1)            # never reaches 179 deg
    data, b_eci, s_eci = _guard_data([1.0, 0.0, 0.0], alpha)
    res = run_mekf(data, b_eci, s_eci, MekfConfig(triad_min_angle=np.deg2rad(179.0)))
    assert res.k_init == -1
    for arr in (res.q, res.b, res.std, res.P_att):
        assert np.all(np.isnan(arr))
    # 9-state / diode configuration also returns an all-NaN record
    res9 = run_mekf(data, b_eci, s_eci,
                    MekfConfig(triad_min_angle=np.deg2rad(179.0), mag_bias=True,
                               css_mode="diodes"))
    assert res9.k_init == -1
    assert np.all(np.isnan(res9.q)) and np.all(np.isnan(res9.std))


def test_run_mekf_diodes_mode_needs_complete_css_vector_to_initialise():
    n = 400
    alpha = 5.0 + 170.0 * np.arange(n) / (n - 1)
    guard = np.deg2rad(20.0)
    # (a) sun with all three axes above 0.55 -> complete from k = 0; guard still applies
    data, b_eci, s_eci = _guard_data([0.62, -0.58, 0.60], alpha)
    assert css_vector_complete(data.css_currents[0], 0.55)
    cfg = MekfConfig(css_mode="diodes", triad_min_angle=guard)
    res = run_mekf(data, b_eci, s_eci, cfg)
    ang = np.array([vector_angle(m, s) for m, s in zip(data.mag, data.sun)])
    assert res.k_init == int(np.nonzero(ang >= guard)[0][0])
    assert np.all(np.isfinite(res.q[res.k_init:]))
    # (b) sun in the body x-y plane: the z axis has no lit diode, vector valid but incomplete
    data2, b2, s2 = _guard_data([0.8, 0.6, 0.0], alpha)
    assert np.all(data2.sun_valid)
    assert not css_vector_complete(data2.css_currents[0], 0.55)
    res2 = run_mekf(data2, b2, s2, MekfConfig(css_mode="diodes"))
    assert res2.k_init == -1 and np.all(np.isnan(res2.q))
    # the vector-mode filter does initialise on the same data
    assert run_mekf(data2, b2, s2, MekfConfig()).k_init == 0


# =========================================================================== 9. step equivalence
def _step_record(n=80, with_albedo=False, seed=3):
    """Slowly rotating truth with default-noise sensors; some samples in eclipse."""
    dt = 1.0
    t = np.arange(n) * dt
    q = np.empty((n, 4))
    q[0] = quat.from_rotvec(np.array([0.3, -0.5, 0.2]))
    for k in range(n - 1):
        q[k + 1] = quat.propagate(q[k], np.array([0.004, -0.006, 0.003]), dt)
    ax = np.array([0.0, 0.6, 0.8])
    b_eci = 30e-6 * Rotation.from_rotvec(np.outer(0.01 * t, ax)).apply([1.0, 0.0, 0.0])
    s_hat = np.array([0.5, 0.7, 0.51])
    s_hat /= np.linalg.norm(s_hat)
    s_eci = np.tile(s_hat, (n, 1))
    lit = np.ones(n, bool)
    lit[30:40] = False
    data = measure_series(t, q, b_eci, s_eci, lit, seed=seed)
    r_hat = 0.4 * s_hat + np.sqrt(1 - 0.16) * np.cross(s_hat, [0.0, 0.0, 1.0]) / \
        np.linalg.norm(np.cross(s_hat, [0.0, 0.0, 1.0]))
    r_eci = np.tile(6.9e6 * r_hat, (n, 1))
    # diode currents for the diodes-mode comparison (albedo 0.3 truth)
    s_body = np.array([quat.rotate_inverse(qk, s_hat) for qk in q])
    alb = albedo_currents(q, r_eci, s_eci, 0.3)
    cur = sun_sensor_currents(s_body, lit, np.random.default_rng(seed + 100), SunSensorConfig(),
                              albedo=alb)
    return data, b_eci, s_eci, r_eci, cur, lit


def _init_pair(data, b_eci, s_eci, cfg):
    q0, P0 = triad_init(data.mag[0], data.sun[0], b_eci[0], s_eci[0], cfg)
    return q0, P0


def test_step_equals_manual_sequence_vector_mode():
    data, b_eci, s_eci, r_eci, cur, lit = _step_record()
    cfg = MekfConfig()
    q0, P0 = _init_pair(data, b_eci, s_eci, cfg)
    f_man, f_step = MEKF(q0, np.zeros(3), P0, cfg), MEKF(q0, np.zeros(3), P0, cfg)
    n_sun_used = 0
    for k in range(1, len(data.t)):
        dt = data.t[k] - data.t[k - 1]
        f_man.propagate(data.gyro[k], dt)
        f_man.update_mag(data.mag[k], b_eci[k], data.t[k])
        if data.sun_valid[k]:
            f_man.update_sun(data.sun[k], s_eci[k], data.t[k])
            n_sun_used += 1
        f_step.step(data.gyro[k], dt, mag=data.mag[k], b_eci=b_eci[k], sun=data.sun[k],
                    s_eci=s_eci[k], sun_valid=bool(data.sun_valid[k]), t=data.t[k])
        assert np.array_equal(f_man.q, f_step.q)
        assert np.array_equal(f_man.b, f_step.b)
        assert np.array_equal(f_man.P, f_step.P)
    assert 0 < n_sun_used < len(data.t) - 1                  # both branches exercised
    for name in ("mag", "sun"):
        a, b = f_man.ekf.innovations(name), f_step.ekf.innovations(name)
        assert np.array_equal(a.nis, b.nis) and np.array_equal(a.t, b.t)


def test_step_respects_use_flags_and_missing_measurements():
    data, b_eci, s_eci, r_eci, cur, lit = _step_record(n=30)
    q0, P0 = _init_pair(data, b_eci, s_eci, MekfConfig())
    # use_mag=False and use_sun=False: step == propagate
    cfg = MekfConfig(use_mag=False, use_sun=False)
    f1, f2 = MEKF(q0, np.zeros(3), P0, cfg), MEKF(q0, np.zeros(3), P0, cfg)
    for k in range(1, 30):
        f1.propagate(data.gyro[k], 1.0)
        f2.step(data.gyro[k], 1.0, mag=data.mag[k], b_eci=b_eci[k], sun=data.sun[k],
                s_eci=s_eci[k], sun_valid=True, currents=cur[k], r_eci=r_eci[k])
    assert np.array_equal(f1.q, f2.q) and np.array_equal(f1.P, f2.P)
    assert f2.ekf.innovations("mag").nis.size == 0 and f2.ekf.innovations("sun").nis.size == 0
    # mag=None skips the magnetometer; sun_valid False skips the sun
    cfg = MekfConfig()
    f3, f4 = MEKF(q0, np.zeros(3), P0, cfg), MEKF(q0, np.zeros(3), P0, cfg)
    for k in range(1, 10):
        f3.propagate(data.gyro[k], 1.0)
        f4.step(data.gyro[k], 1.0, mag=None, sun=data.sun[k], s_eci=s_eci[k], sun_valid=False)
    assert np.array_equal(f3.q, f4.q) and np.array_equal(f3.P, f4.P)


@pytest.mark.parametrize("n_states", [6, 9])
def test_step_equals_manual_sequence_diodes_mode(n_states):
    data, b_eci, s_eci, r_eci, cur, lit = _step_record(n=80)
    cfg = MekfConfig(css_mode="diodes", mag_bias=(n_states == 9), albedo_model=0.3,
                     sigma_albedo=0.5, css_fov=np.deg2rad(60.0), mag_bias_rw=1e-11)
    q0, P0 = triad_init(data.mag[0], data.sun[0], b_eci[0], s_eci[0], cfg)
    f_man, f_step = MEKF(q0, np.zeros(3), P0, cfg), MEKF(q0, np.zeros(3), P0, cfg)
    n_css = 0
    for k in range(1, len(data.t)):
        dt = data.t[k] - data.t[k - 1]
        f_man.propagate(data.gyro[k], dt)
        f_man.update_mag(data.mag[k], b_eci[k], data.t[k])
        if f_man.update_css_diodes(cur[k], s_eci[k], r_eci[k], data.t[k]) is not None:
            n_css += 1
        f_step.step(data.gyro[k], dt, mag=data.mag[k], b_eci=b_eci[k], sun=data.sun[k],
                    s_eci=s_eci[k], sun_valid=bool(data.sun_valid[k]), currents=cur[k],
                    r_eci=r_eci[k], t=data.t[k])
        assert np.array_equal(f_man.q, f_step.q)
        assert np.array_equal(f_man.b, f_step.b)
        assert np.array_equal(f_man.bm, f_step.bm)
        assert np.array_equal(f_man.P, f_step.P)
    assert 20 < n_css < len(data.t) - 1                       # diode updates in lit samples only
    a, b = f_man.ekf.innovations("css"), f_step.ekf.innovations("css")
    assert a.nis.size == n_css
    assert np.array_equal(a.nis, b.nis) and np.array_equal(a.dof, b.dof)
    # the vector-mode sun update is not used in diodes mode
    assert f_step.ekf.innovations("sun").nis.size == 0


# =========================================================================== 5, 6. statistics
def _truth_worker(orbits):
    """Pointing truth of the spec prototype plus the filter's reference vectors."""
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    from adcs.config import DEFAULT
    from adcs.environment import FieldTable, sunlit
    from adcs.orbit import Ephemeris, initial_state, orbital_period
    from adcs.simulate import lvlh_state, run

    r0, v0 = initial_state(DEFAULT)
    period = float(orbital_period(DEFAULT.orbit.a))
    t_end = float(round(orbits * period))
    sim = run(lvlh_state(r0, v0, offset=np.deg2rad([2.0, -1.0, 3.0])), t_end, "pointing",
              log_every=1)
    eph = Ephemeris(r0, v0, t_end + 10.0)
    table = FieldTable(eph, DEFAULT, t_end + 10.0)
    t = np.asarray(sim.t)
    r_eci = table.r_eci(t)
    s_eci = table.sun_dir_eci(t)
    return dict(t=t, q=np.asarray(sim.q), b_eci=table.b_eci(t), s_eci=s_eci, r_eci=r_eci,
                lit=np.asarray(sunlit(r_eci, s_eci), bool), period=period)


def _bias_worker(seed, tr):
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    data = measure_series(tr["t"], tr["q"], tr["b_eci"], tr["s_eci"], tr["lit"], seed=seed)
    data.mag = data.mag + MAG_BIAS_TRUE                        # constant body-frame bias
    cfg = MekfConfig(mag_bias=True, sigma_mb0=SIGMA_MB0_STAT, mag_bias_rw=0.0)
    res = run_mekf(data, tr["b_eci"], tr["s_eci"], cfg)
    k0 = res.k_init
    err = attitude_errors(tr["q"], res.q)
    sl = slice(k0 + SETTLE, None)
    n = nees(err[sl], res.P_att[sl])
    acc = np.asarray(res.nis["mag"][2], bool)
    return dict(
        seed=seed, k0=k0, nees=n,
        bm_err=MAG_BIAS_TRUE - res.bm[-1], bm_std=res.std[-1, 6:],
        bm_err_1orb=MAG_BIAS_TRUE - res.bm[int(tr["period"])],
        nis_mag=np.asarray(res.nis["mag"][1])[acc], n_states=res.std.shape[1],
    )


def _diode_worker(seed, tr, albedo):
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    css_cfg = SunSensorConfig(fov=FOV_TRUE)
    data = measure_series(tr["t"], tr["q"], tr["b_eci"], tr["s_eci"], tr["lit"], seed=seed,
                          css=css_cfg)
    # re-draw the diode currents with the sun-sensor stream (same stream as measure_series),
    # adding the albedo of the variant
    s_body = quat.rotate_inverse(tr["q"], tr["s_eci"])
    alb = None if albedo == 0.0 else albedo_currents(tr["q"], tr["r_eci"], tr["s_eci"], albedo)
    rng_s = np.random.default_rng(seed).spawn(3)[2]
    cur = sun_sensor_currents(s_body, tr["lit"], rng_s, css_cfg, albedo=alb)
    data.css_currents = cur
    data.sun, data.sun_valid = sun_vector_from_currents(cur, css_cfg)
    cfg = MekfConfig(css_mode="diodes", css_fov=FOV_TRUE, albedo_model=albedo, sigma_albedo=0.0)
    off = quat.from_rotvec(np.deg2rad(OFFSET_DEG))
    q0 = quat.multiply(tr["q"][0], off)
    P0 = np.diag([np.deg2rad(5.0) ** 2] * 3 + [cfg.sigma_b0 ** 2] * 3)
    res = run_mekf(data, tr["b_eci"], tr["s_eci"], cfg, q0=q0, P0=P0, k_init=0,
                   r_eci=tr["r_eci"])
    err = attitude_errors(tr["q"], res.q)
    sl = slice(SETTLE, None)
    n = nees(err[sl], res.P_att[sl])
    t_c, nis_c, acc_c = res.nis["css"]
    dof = np.asarray(res.meta["nis_dof"]["css"])
    sel = np.asarray(acc_c, bool) & (np.asarray(t_c) >= SETTLE)
    e_norm = np.linalg.norm(err, axis=1)
    lit_sel = tr["lit"].copy()
    lit_sel[:SETTLE] = False
    return dict(
        seed=seed, nees=n, nis_sum=float(np.sum(np.asarray(nis_c)[sel])),
        dof_sum=int(np.sum(dof[sel])), n_upd=int(sel.sum()),
        mean_diodes=float(np.mean(dof[sel])),
        rms_sunlit=float(np.sqrt(np.mean(e_norm[lit_sel] ** 2))),
        n_lit=int(lit_sel.sum()),
    )


@pytest.fixture(scope="module", autouse=True)
def _pool(request):
    """Process pool; the truth job is submitted at module start so that it overlaps
    with the fast tests (only when a statistical test is selected)."""
    if not any("statistical" in it.name for it in request.session.items):
        yield None
        return
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ex = ProcessPoolExecutor(max_workers=max(2, min(18, os.cpu_count() or 2)))
    jobs = {"truth": ex.submit(_truth_worker, ORBITS_STAT)}
    yield {"ex": ex, "jobs": jobs}
    ex.shutdown(wait=True, cancel_futures=True)


@pytest.fixture(scope="module")
def stat_runs(_pool):
    ex, jobs = _pool["ex"], _pool["jobs"]
    tr = jobs["truth"].result()
    futs = dict(
        bias=[ex.submit(_bias_worker, s, tr) for s in SEEDS_STAT],
        diode=[ex.submit(_diode_worker, s, tr, 0.0) for s in SEEDS_STAT],
        diode_alb=[ex.submit(_diode_worker, s, tr, 0.3) for s in SEEDS_STAT],
    )
    out = {k: [f.result() for f in v] for k, v in futs.items()}
    out["truth"] = tr
    return out


def test_statistical_truth_fixture_is_the_spec_scenario(stat_runs):
    tr = stat_runs["truth"]
    assert abs(tr["t"][-1] - round(ORBITS_STAT * tr["period"])) < 1e-9
    assert np.allclose(np.diff(tr["t"]), 1.0, atol=1e-9)
    # pointing truth started 2/-1/3 deg off nadir: sunlit and eclipse samples both present
    assert 0.3 < np.mean(tr["lit"]) < 0.8


def test_statistical_mag_bias_convergence_and_consistency(stat_runs):
    runs = stat_runs["bias"]
    assert all(r["n_states"] == 9 for r in runs)
    nees_all = np.concatenate([r["nees"] for r in runs])
    per_seed = [float(np.mean(r["nees"])) for r in runs]
    bm_err_end = np.array([np.linalg.norm(r["bm_err"]) for r in runs]) * 1e9
    bm_err_1 = np.array([np.linalg.norm(r["bm_err_1orb"]) for r in runs]) * 1e9
    covered = [bool(np.all(np.abs(r["bm_err"]) < 3.0 * r["bm_std"])) for r in runs]
    nis_mag = np.concatenate([r["nis_mag"] for r in runs])
    print(f"\n[bias] NEES pooled {nees_all.mean():.3f} (per seed {np.round(per_seed, 2)}); "
          f"bm err end [nT] {np.round(bm_err_end, 2)}; after 1 orbit {np.round(bm_err_1, 2)}; "
          f"3-sigma covered {covered}; sigma_end [nT] {np.round(runs[0]['bm_std'] * 1e9, 3)}; "
          f"NIS mag/3 {nis_mag.mean() / 3:.3f}")
    assert 2.4 <= nees_all.mean() <= 3.6
    assert np.all(bm_err_end < 3.0)                            # nT, every seed
    assert sum(covered) >= 5
    # the filter really estimated a bias that was far from its prior mean (b_m0 = 0)
    assert np.all(np.linalg.norm(MAG_BIAS_TRUE) * 1e9 > 10 * bm_err_end)


@pytest.mark.parametrize("key", ["diode", "diode_alb"])
def test_statistical_diodes_consistency(stat_runs, key):
    runs = stat_runs[key]
    nis_pd = sum(r["nis_sum"] for r in runs) / sum(r["dof_sum"] for r in runs)
    nees_all = np.concatenate([r["nees"] for r in runs])
    rms = [np.rad2deg(r["rms_sunlit"]) for r in runs]
    print(f"\n[{key}] NIS/dof pooled {nis_pd:.4f} (per seed "
          f"{np.round([r['nis_sum'] / r['dof_sum'] for r in runs], 3)}); NEES {nees_all.mean():.3f} "
          f"(per seed {np.round([np.mean(r['nees']) for r in runs], 2)}); sunlit RMS [deg] "
          f"{np.round(rms, 3)}; mean diodes/update {np.mean([r['mean_diodes'] for r in runs]):.2f}")
    assert all(r["n_upd"] > 1000 for r in runs)                # enough updates for the statistic
    assert 0.95 <= nis_pd <= 1.05
    assert 2.4 <= nees_all.mean() <= 3.8
    assert all(x < 1.5 for x in rms)
