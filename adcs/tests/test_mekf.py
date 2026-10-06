"""Tests for the multiplicative EKF in ``adcs.estimation`` (Group 7, M8).

Written from ``docs/specs/adcs_estimation.md`` section 0 (sampling model), section 3
(MEKF) and section 5 (tests 1-11), not from the implementation.  Oracles:

* ``common.ekf.van_loan`` (matrix exponential of the continuous error model) for the
  closed-form ``discrete_model``;
* central finite differences of ``R(q (x) exp(dtheta))^T v`` built with
  ``scipy.spatial.transform.Rotation`` for the measurement Jacobian;
* constructed truth: synthetic attitude trajectories with known sensor noise, so that
  NIS / NEES have known chi-square expectations (3 degrees of freedom -> mean 3);
* ``adcs.estimation.triad`` / ``triad_covariance`` (tested separately in test_triad.py)
  for the initialisation;
* a hand-written loop over the public ``MEKF`` class for the gating comparison.

Statistical thresholds are the ones in spec section 5 (set from a main-session
prototype); none has been loosened.  Speed: the Monte Carlo of test 7 (10 x one orbit
of 5554 steps, ~2.4 s each) and the real-environment run (one orbit of
``adcs.simulate.run``, ~5 s, plus the filter) are executed concurrently in a
``ProcessPoolExecutor`` that is started by the first test that needs it, so they
overlap with the other (main-process) tests.  The truth/filter/sensor numbers are
identical to a serial run (all randomness is seeded).
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from common import quaternion as quat
from common.ekf import van_loan
from adcs.estimation import (
    MEKF,
    MekfConfig,
    attitude_errors,
    discrete_model,
    nees,
    run_mekf,
    triad,
    triad_covariance,
    triad_init,
)
from adcs.sensors import (
    GyroConfig,
    MagnetometerConfig,
    SunSensorConfig,
    measure_series,
)

# --------------------------------------------------------------------------- constants
P_ORB = 5554.0                    # s, orbital period quoted in the spec
N_ORB = 1.13e-3                   # rad/s, orbit-rate pitch (spec section 5)
LIT_FRACTION = 0.61
B_MAG = 30e-6                     # T, field magnitude of the synthetic reference
BIAS_DEG = 0.1                    # deg/s, spec test 6 turn-on bias on every axis
DEFAULT_SEEDS = tuple(range(10))
SIGMA_M = MagnetometerConfig().sigma
SIGMA_SUN = SunSensorConfig().sigma_vector


def _sf(q_scalar_first):
    """scalar-first (w, x, y, z) -> scipy scalar-last Rotation."""
    q = np.asarray(q_scalar_first, float)
    return Rotation.from_quat(np.roll(q, -1, axis=-1))


def _skew(v):
    """Own cross-product matrix (independent of common.quaternion.skew)."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


# --------------------------------------------------------------------------- synthetic truth
@lru_cache(maxsize=8)
def synthetic_truth(n_steps: int, phase0: float = 0.35):
    """Slowly varying body-rate truth, 1 s grid, with reference vectors and sun/eclipse flag.

    Body rate = orbit-rate pitch about body y + a small sinusoidal wobble about x and z.
    Field: 30 uT unit vector rotating at 2n about an axis tilted 30 deg from the
    rotation (x-z) plane; sun: fixed unit vector at 50 deg from that axis, so the angle
    between field and sun stays within [40, 140] deg (TRIAD never degenerate).
    ``lit`` = sunlit for the first 61 % of every period; ``phase0`` is the orbital
    phase (fraction of a period) at t = 0.
    """
    dt = 1.0
    t = np.arange(n_steps + 1) * dt
    q = np.empty((n_steps + 1, 4))
    q[0] = quat.from_rotvec(np.array([0.3, -0.5, 0.2]))
    amp = 3e-4
    for k in range(n_steps):
        tm = t[k] + 0.5 * dt
        w = np.array([amp * np.sin(2 * np.pi * tm / 310.0), N_ORB,
                      amp * np.cos(2 * np.pi * tm / 470.0)])
        q[k + 1] = quat.propagate(q[k], w, dt)
    tilt = np.deg2rad(30.0)
    axis = np.array([np.cos(tilt), np.sin(tilt), 0.0])
    v0 = np.array([0.0, 0.0, 1.0])                     # perpendicular to the axis
    b_eci = B_MAG * Rotation.from_rotvec(np.outer(2.0 * N_ORB * t, axis)).apply(v0)
    ang = np.deg2rad(50.0)
    s_dir = np.cos(ang) * axis + np.sin(ang) * v0
    s_eci = np.tile(s_dir / np.linalg.norm(s_dir), (n_steps + 1, 1))
    lit = ((t / P_ORB + phase0) % 1.0) < LIT_FRACTION
    return dict(t=t, q=q, b_eci=b_eci, s_eci=s_eci, lit=lit)


def sensors_for(truth, seed, gyro=GyroConfig(), bias0=None, css=SunSensorConfig(),
                mag=MagnetometerConfig()):
    return measure_series(truth["t"], truth["q"], truth["b_eci"], truth["s_eci"], truth["lit"],
                          seed=seed, mag=mag, css=css, gyro=gyro, bias0=bias0)


def _bias_vec(deg=BIAS_DEG):
    return np.deg2rad(deg) * np.ones(3)


def filter_run(truth, seed, cfg=MekfConfig(), bias0=None, **kw):
    data = sensors_for(truth, seed, bias0=bias0, **kw)
    res = run_mekf(data, truth["b_eci"], truth["s_eci"], cfg)
    return data, res


def _valid_nis(res, name):
    t, nis, acc = res.nis[name]
    return np.asarray(nis)[np.asarray(acc, bool)]


# --------------------------------------------------------------------------- pool workers
def _mc_worker(seed):
    """One Monte Carlo run (synthetic truth, default sensors/filter, TRIAD start)."""
    truth = synthetic_truth(int(P_ORB))
    data, res = filter_run(truth, seed)
    return dict(seed=seed, q_true=truth["q"], res=res, lit=truth["lit"],
                bias=data.gyro_bias)


def _real_worker():
    """1 orbit of the real simulation (pointing, perfect-knowledge control) + MEKF."""
    from adcs.config import DEFAULT
    from adcs.environment import FieldTable
    from adcs.orbit import Ephemeris, initial_state
    from adcs.simulate import lvlh_state, run

    r0, v0 = initial_state(DEFAULT)
    t_end = float(round(P_ORB))
    sim = run(lvlh_state(r0, v0, offset=np.deg2rad([2.0, -1.0, 3.0])), t_end, "pointing",
              log_every=1)
    eph = Ephemeris(r0, v0, t_end + 10.0)
    table = FieldTable(eph, DEFAULT, t_end + 10.0)
    b_eci = table.b_eci(sim.t)
    s_eci = table.sun_dir_eci(sim.t)
    lit = table.sunlit(sim.t)
    data = measure_series(sim.t, sim.q, b_eci, s_eci, lit, seed=0)
    res = run_mekf(data, b_eci, s_eci, MekfConfig())
    return dict(q_true=sim.q, t=sim.t, res=res, lit=lit, bias=data.gyro_bias)


@pytest.fixture(scope="module")
def pool_jobs():
    """Start the heavy runs (10 MC seeds + the real-environment run) in worker processes."""
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # one BLAS thread per worker
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ex = ProcessPoolExecutor(max_workers=min(11, os.cpu_count() or 2))
    jobs = dict(real=ex.submit(_real_worker),
                mc=[ex.submit(_mc_worker, s) for s in DEFAULT_SEEDS])
    yield jobs
    ex.shutdown(wait=True, cancel_futures=True)


@pytest.fixture(scope="module")
def base_run(pool_jobs):
    """Shared one-orbit synthetic run: default noise, 0.1 deg/s turn-on bias on every axis.

    Starts sunlit, eclipse in the middle (phase0 = 0.35): lit 0-1444 s, eclipse
    1444-3610 s, lit again to the end.  ``pool_jobs`` is requested only so that the
    worker processes start before this (main-process) run and overlap with it.
    """
    truth = synthetic_truth(int(P_ORB))
    data, res = filter_run(truth, seed=0, bias0=_bias_vec())
    return truth, data, res


# =========================================================================== 1. discrete_model
THETAS = [0.0, 1e-6, 5e-3, 9.9e-3, 1.01e-2, 0.2, 1.0, 3.0]
DTS = [0.1, 1.0, 10.0]


def _oracle(w_hat, dt, arw, rrw):
    F = np.zeros((6, 6))
    F[:3, :3] = -_skew(w_hat)
    F[:3, 3:] = -np.eye(3)
    G = np.diag([-1.0] * 3 + [1.0] * 3)
    Qc = np.diag([arw**2] * 3 + [rrw**2] * 3)
    return van_loan(F, Qc, dt, G)


@pytest.mark.parametrize("dt", DTS)
@pytest.mark.parametrize("theta", THETAS)
def test_discrete_model_matches_van_loan(theta, dt):
    rng = np.random.default_rng(1234)
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    w_hat = d * (theta / dt)
    assert np.isclose(np.linalg.norm(w_hat) * dt, theta, rtol=1e-12, atol=0.0)

    # transition matrix
    Phi, _ = discrete_model(w_hat, dt, 0.0, 1.0)
    Phi_o, Q_o = _oracle(w_hat, dt, 0.0, 1.0)
    assert np.max(np.abs(Phi - Phi_o)) < 1e-12

    # bias-random-walk-only noise: abs error < 1e-9 x largest off-diagonal element
    _, Qd = discrete_model(w_hat, dt, 0.0, 1.0)
    off = Q_o - np.diag(np.diag(Q_o))
    assert np.max(np.abs(Qd - Q_o)) < 1e-9 * np.max(np.abs(off))

    # default sensor noise: relative error < 1e-9 (relative to the largest entry)
    arw, rrw = GyroConfig().arw, GyroConfig().rrw
    Phi_d, Qd_d = discrete_model(w_hat, dt, arw, rrw)
    _, Q_do = _oracle(w_hat, dt, arw, rrw)
    assert np.max(np.abs(Phi_d - Phi_o)) < 1e-12
    assert np.max(np.abs(Qd_d - Q_do)) < 1e-9 * np.max(np.abs(Q_do))
    # symmetric and positive semi-definite
    assert np.allclose(Qd_d, Qd_d.T, rtol=0, atol=1e-30)
    assert np.min(np.linalg.eigvalsh(Qd)) > -1e-9 * np.max(np.linalg.eigvalsh(Qd))


def test_discrete_model_arw_only_is_white_noise_integral():
    # With rrw = 0: Qd = diag(arw^2 dt I, 0) exactly (Phi11 is orthogonal), any rate.
    w_hat = np.array([0.3, -0.2, 0.5])
    for dt in DTS:
        _, Qd = discrete_model(w_hat, dt, 1.9e-4, 0.0)
        expected = np.zeros((6, 6))
        expected[:3, :3] = (1.9e-4) ** 2 * dt * np.eye(3)
        assert np.max(np.abs(Qd - expected)) < 1e-12 * (1.9e-4) ** 2 * dt


# =========================================================================== 2. Jacobian
def _fd_jacobian(q_hat, v, h=1e-6):
    """d/d(dtheta) of R(q_hat (x) exp(dtheta))^T v by central differences (scipy oracle)."""
    R0 = _sf(q_hat)
    J = np.zeros((3, 3))
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        Rp = (R0 * Rotation.from_rotvec(e)).inv().apply(v)
        Rm = (R0 * Rotation.from_rotvec(-e)).inv().apply(v)
        J[:, i] = (Rp - Rm) / (2 * h)
    return J


@pytest.mark.parametrize("scale", [1.0, 30e-6])
def test_measurement_jacobian_matches_finite_differences(scale):
    rng = np.random.default_rng(7)
    for _ in range(5):
        q_hat = quat.canonicalize(quat.normalize(rng.normal(size=4)))
        v = scale * rng.normal(size=3)
        v_b = _sf(q_hat).inv().apply(v)
        J = _fd_jacobian(q_hat, v)
        # the formula of the spec: H_att = skew(R(q_hat)^T v)
        assert np.max(np.abs(J - _skew(v_b))) < 1e-8 * np.linalg.norm(v)

        # the filter's H, seen through its public update: S = H P H^T + sigma^2 I and
        # dx = P H^T S^-1 r with H = [J, 0]; random full 6x6 SPD P (attitude-bias
        # cross-terms included) so that nonzero bias columns would show.
        A = rng.normal(size=(6, 6))
        P0 = 1e-4 * (A @ A.T + 6 * np.eye(6))
        # sigma comparable to |H P H^T|^(1/2) keeps S well conditioned (S has a small
        # eigenvalue sigma^2 along v_b, where H has a null direction, so a tiny sigma
        # would amplify the 1e-10 finite-difference error of J in dx).
        sigma = 0.05 * scale
        f = MEKF(q_hat, np.zeros(3), P0)
        r_true = _skew(v_b) @ rng.normal(scale=1e-2, size=3) + sigma * rng.normal(size=3)
        res = f.update_vector(v_b + r_true, v, sigma, "x")
        H = np.hstack([J, np.zeros((3, 3))])
        S = H @ P0 @ H.T + sigma**2 * np.eye(3)
        dx = P0 @ H.T @ np.linalg.solve(S, r_true)
        assert np.max(np.abs(res.S - S)) < 1e-7 * np.max(np.abs(S))
        assert np.max(np.abs(res.dx - dx)) < 1e-7 * np.max(np.abs(dx))
        assert np.max(np.abs(res.residual - r_true)) < 1e-12 * scale


def test_vector_update_corrects_toward_truth_sign_and_injection():
    """Noise-free vector measurement of a rotated truth: the correction has the sign
    of q_true = q_hat (x) exp(dtheta) on the observable (perpendicular) components and
    the injected quaternion moves toward q_true."""
    rng = np.random.default_rng(11)
    for _ in range(5):
        q_hat = quat.canonicalize(quat.normalize(rng.normal(size=4)))
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        v_b = _sf(q_hat).inv().apply(v)
        dth = rng.normal(size=3)
        dth -= (dth @ v_b) * v_b                       # observable part only
        dth *= 1e-4 / np.linalg.norm(dth)
        q_true = (_sf(q_hat) * Rotation.from_rotvec(dth)).as_quat()
        q_true = np.roll(q_true, 1)
        z = _sf(q_true).inv().apply(v)
        f = MEKF(q_hat, np.zeros(3), 1e-2 * np.eye(6))
        res = f.update_vector(z, v, 1e-9, "x")
        assert res.accepted
        # |dth|^2 ~ 1e-8 second-order terms: tolerance 1e-7 on a 1e-4 quantity
        assert np.max(np.abs(res.dx[:3] - dth)) < 1e-7
        assert np.max(np.abs(res.dx[3:])) < 1e-7
        q_new = f.q
        assert quat.angle_between(q_new, q_true) < 1e-7 < quat.angle_between(q_hat, q_true)
        # reset Jacobian keeps P symmetric and shrinks the observable directions
        assert np.allclose(f.P, f.P.T, atol=1e-18)


# =========================================================================== 3. exact propagation
def test_exact_propagation_noise_free_over_2000_steps():
    truth = synthetic_truth(2000)
    zero = GyroConfig(arw=0.0, rrw=0.0, bias0_sigma=0.0)
    data = sensors_for(truth, seed=0, gyro=zero)
    assert np.all(data.gyro_bias == 0.0)
    cfg = MekfConfig(use_mag=False, use_sun=False)
    res = run_mekf(data, truth["b_eci"], truth["s_eci"], cfg,
                   q0=truth["q"][0], P0=1e-4 * np.eye(6), k_init=0)
    err = quat.angle_between(res.q, truth["q"])
    assert res.q.shape == (2001, 4)
    assert np.max(err) < 1e-9, f"max attitude error {np.max(err):.3e} rad"
    assert np.max(np.abs(res.b)) == 0.0              # bias estimate unchanged


def test_exact_propagation_detects_sampling_model_error():
    """Negative control: feeding the gyro one sample late must NOT reproduce the truth,
    so the 1e-9 check above really tests the section-0 sampling model."""
    truth = synthetic_truth(2000)
    zero = GyroConfig(arw=0.0, rrw=0.0, bias0_sigma=0.0)
    data = sensors_for(truth, seed=0, gyro=zero)
    g = data.gyro.copy()
    g[2:] = data.gyro[1:-1]                          # delayed by one sample
    data.gyro = g
    res = run_mekf(data, truth["b_eci"], truth["s_eci"],
                   MekfConfig(use_mag=False, use_sun=False),
                   q0=truth["q"][0], P0=1e-4 * np.eye(6), k_init=0)
    assert np.max(quat.angle_between(res.q, truth["q"])) > 1e-6


# =========================================================================== 4. housekeeping
def test_housekeeping_unit_norm_spd_and_finite(base_run):
    truth, data, res = base_run
    k0 = res.k_init
    assert k0 == 0
    assert res.q.shape == (len(truth["t"]), 4)
    norms = np.linalg.norm(res.q, axis=1)
    assert np.max(np.abs(norms - 1.0)) < 1e-12
    P = res.P_att
    assert P.shape == (len(truth["t"]), 3, 3)
    assert np.allclose(P, np.swapaxes(P, 1, 2), rtol=0, atol=1e-18 + 1e-12 * np.max(np.abs(P)))
    ev = np.linalg.eigvalsh(0.5 * (P + np.swapaxes(P, 1, 2)))
    assert np.min(ev) > 0.0
    assert np.all(np.isfinite(res.std[k0:]))
    assert np.all(res.std[k0:] > 0.0)
    assert np.all(np.isfinite(res.b[k0:]))
    # std is consistent with the logged attitude covariance
    assert np.allclose(res.std[:, :3] ** 2, np.einsum("nii->ni", P), rtol=1e-9)


def test_housekeeping_eclipse_start_nan_rows_and_k_init():
    truth = synthetic_truth(2500, 0.70)              # phase 0.70: eclipse until t = 1666 s
    assert not truth["lit"][0]
    data, res = filter_run(truth, seed=3)
    first_valid = int(np.nonzero(data.sun_valid)[0][0])
    assert res.k_init == first_valid
    assert first_valid == int(np.nonzero(truth["lit"])[0][0])   # noise-free first lit index
    assert first_valid > 1000
    for arr in (res.q, res.b, res.std, res.P_att):
        assert np.all(np.isnan(arr[:first_valid]))
        assert np.all(np.isfinite(arr[first_valid:]))
    assert np.allclose(np.linalg.norm(res.q[first_valid:], axis=1), 1.0, atol=1e-12)
    ev = np.linalg.eigvalsh(0.5 * (res.P_att[first_valid:]
                                   + np.swapaxes(res.P_att[first_valid:], 1, 2)))
    assert np.min(ev) > 0.0


def test_housekeeping_never_valid_sun_returns_nan():
    truth = synthetic_truth(100, 0.70)               # eclipse for the whole run
    data, res = filter_run(truth, seed=0)
    assert not np.any(data.sun_valid)
    assert res.k_init == -1
    assert np.all(np.isnan(res.q))


# =========================================================================== 5. TRIAD initialisation
def test_triad_initialisation_default_primary_is_magnetometer():
    truth = synthetic_truth(40)
    data, res = filter_run(truth, seed=5)
    k = res.k_init
    assert k == 0
    mag_b, sun_b = data.mag[k], data.sun[k]
    q_oracle = triad(mag_b, sun_b, truth["b_eci"][k], truth["s_eci"][k])
    assert quat.angle_between(res.q[k], q_oracle) < 1e-12
    s_m = SIGMA_M / np.linalg.norm(mag_b)
    assert s_m < SIGMA_SUN                           # magnetometer is the more accurate vector
    P_att = triad_covariance(mag_b, sun_b, s_m, SIGMA_SUN)
    assert np.allclose(res.P_att[k], P_att, rtol=1e-9, atol=0.0)
    # bias block sigma_b0^2 I, bias estimate zero
    assert np.allclose(res.std[k, 3:], GyroConfig().bias0_sigma, rtol=1e-12)
    assert np.all(res.b[k] == 0.0)
    # and the public initialiser gives the same pair
    q0, P0 = triad_init(mag_b, sun_b, truth["b_eci"][k], truth["s_eci"][k], MekfConfig())
    assert quat.angle_between(q0, q_oracle) < 1e-12
    assert np.allclose(P0[:3, :3], P_att, rtol=1e-9, atol=0.0)
    assert np.allclose(P0[3:, 3:], GyroConfig().bias0_sigma ** 2 * np.eye(3), rtol=1e-12)
    assert np.all(P0[:3, 3:] == 0.0)


def test_triad_initialisation_primary_switches_to_sun_for_huge_mag_sigma():
    truth = synthetic_truth(40)
    cfg = MekfConfig(sigma_mag=1e-4)
    data, res = filter_run(truth, seed=5, cfg=cfg)
    k = res.k_init
    mag_b, sun_b = data.mag[k], data.sun[k]
    s_m = 1e-4 / np.linalg.norm(mag_b)
    assert s_m > SIGMA_SUN
    q_oracle = triad(sun_b, mag_b, truth["s_eci"][k], truth["b_eci"][k])
    assert quat.angle_between(res.q[k], q_oracle) < 1e-12
    # the sun vector is matched exactly by the primary
    assert np.allclose(_sf(res.q[k]).apply(sun_b / np.linalg.norm(sun_b)),
                       truth["s_eci"][k], atol=1e-12)
    P_att = triad_covariance(sun_b, mag_b, SIGMA_SUN, s_m)
    assert np.allclose(res.P_att[k], P_att, rtol=1e-9, atol=0.0)
    # differs from the magnetometer-primary attitude of the same measurements
    q_mag = triad(mag_b, sun_b, truth["b_eci"][k], truth["s_eci"][k])
    assert quat.angle_between(res.q[k], q_mag) > 1e-4


def test_triad_initialisation_inflation_scales_attitude_block_only():
    truth = synthetic_truth(40)
    _, r1 = filter_run(truth, seed=5)
    _, r4 = filter_run(truth, seed=5, cfg=MekfConfig(init_inflate=4.0))
    assert np.allclose(r4.P_att[0], 4.0 * r1.P_att[0], rtol=1e-9)
    assert np.allclose(r4.std[0, 3:], r1.std[0, 3:], rtol=1e-12)
    assert quat.angle_between(r4.q[0], r1.q[0]) < 1e-12


# =========================================================================== 6. bias convergence
def test_bias_convergence(base_run):
    truth, data, res = base_run
    b_err = data.gyro_bias - res.b                     # true minus estimated, rad/s
    norm_deg = np.rad2deg(np.linalg.norm(b_err, axis=1))
    k25 = int(0.25 * P_ORB)
    print(f"\n[bias] |db| at t=0: {norm_deg[0]:.4f} deg/s; at 0.25 orbit: {norm_deg[k25]:.5f} deg/s; "
          f"max over 0.25..1 orbit: {norm_deg[k25:].max():.5f}")
    assert norm_deg[0] > 0.15                          # starts from b_hat = 0, true |b| = 0.173
    assert norm_deg[k25] < 0.01
    # end of one orbit: each axis within 3 sigma and sigma_b below 0.005 deg/s per axis
    sig_end = res.std[-1, 3:]
    print(f"[bias] end: err = {np.rad2deg(b_err[-1])} deg/s, sigma_b = {np.rad2deg(sig_end)} deg/s")
    assert np.all(np.abs(b_err[-1]) < 3.0 * sig_end)
    assert np.all(np.rad2deg(sig_end) < 0.005)


# =========================================================================== 7. consistency Monte Carlo
def test_consistency_monte_carlo(pool_jobs):
    nis_pool = {"mag": [], "sun": []}
    nees_pool, inside, inside_b = [], [], []
    per_run_nees = []
    for fut in pool_jobs["mc"]:
        out = fut.result()
        res = out["res"]
        k0 = res.k_init
        for name in nis_pool:
            nis_pool[name].append(_valid_nis(res, name))
        err = attitude_errors(out["q_true"], res.q)
        sl = slice(k0 + 300, None)
        n = nees(err[sl], res.P_att[sl])
        nees_pool.append(n)
        per_run_nees.append(n.mean())
        inside.append(np.abs(err[sl]) < 3.0 * res.std[sl, :3])
        # bias: sample k uses true bias gyro_bias[k]; estimate after step k is res.b[k]
        inside_b.append(np.abs(out["bias"][sl] - res.b[sl]) < 3.0 * res.std[sl, 3:])
    mean_nis = {k: float(np.mean(np.concatenate(v))) for k, v in nis_pool.items()}
    n_nis = {k: int(np.concatenate(v).size) for k, v in nis_pool.items()}
    mean_nees = float(np.mean(np.concatenate(nees_pool)))
    frac_inside = np.mean(np.concatenate(inside), axis=0)
    frac_inside_b = np.mean(np.concatenate(inside_b), axis=0)
    print(f"\n[MC 10 seeds] NIS mag = {mean_nis['mag']:.4f} (n={n_nis['mag']}), "
          f"NIS sun = {mean_nis['sun']:.4f} (n={n_nis['sun']}), NEES = {mean_nees:.4f} "
          f"(per-run {np.round(per_run_nees, 2)}), frac |dth_i|<3sigma_i = {frac_inside}")
    for name in ("mag", "sun"):
        assert 3.0 * 0.95 <= mean_nis[name] <= 3.0 * 1.05, (name, mean_nis[name])
    assert 2.55 <= mean_nees <= 3.45, mean_nees
    assert np.all(frac_inside >= 0.98), frac_inside
    print(f"[MC 10 seeds] frac |db_i|<3sigma_i = {frac_inside_b}")
    assert np.all(frac_inside_b >= 0.98), frac_inside_b


# =========================================================================== 8. eclipse degradation
def test_eclipse_degradation_and_recovery(base_run):
    truth, data, res = base_run
    lit = truth["lit"]
    t = truth["t"]
    i_in = int(np.nonzero(~lit)[0][0])                 # first eclipse sample
    i_out = int(np.nonzero(~lit)[0][-1])               # last eclipse sample
    assert lit[:i_in].all() and (~lit[i_in:i_out + 1]).all() and lit[i_out + 1:].all()
    sig = np.sqrt(np.trace(res.P_att, axis1=1, axis2=2))
    entry, exit_ = sig[i_in - 1], sig[i_out]           # last sunlit update / last eclipse step
    k_rec = i_out + 300                                # 300 s after the last eclipse sample
    assert k_rec < len(t)
    recovered = sig[k_rec]
    print(f"\n[eclipse] sigma_att entry = {np.rad2deg(entry):.4f} deg, end = {np.rad2deg(exit_):.4f} deg "
          f"(ratio {exit_ / entry:.2f}); +300 s after sun = {np.rad2deg(recovered):.4f} deg "
          f"(ratio to end {recovered / exit_:.2f})")
    assert exit_ >= 1.5 * entry
    assert recovered <= 0.7 * exit_

    err = np.linalg.norm(attitude_errors(truth["q"], res.q), axis=1)
    k25 = int(0.25 * P_ORB)
    idx = np.arange(len(t))
    sel_ecl = (idx >= max(k25, i_in)) & (idx <= i_out)
    sel_sun = (idx >= k25) & lit
    rms_ecl = np.sqrt(np.mean(err[sel_ecl] ** 2))
    rms_sun = np.sqrt(np.mean(err[sel_sun] ** 2))
    print(f"[eclipse] RMS |dth| sunlit = {np.rad2deg(rms_sun):.3f} deg, eclipse = {np.rad2deg(rms_ecl):.3f} deg")
    assert rms_ecl > rms_sun
    # during eclipse no sun update was logged (sun_valid false), outside it they were
    t_sun = res.nis["sun"][0]
    assert not np.any((t_sun > t[i_in]) & (t_sun <= t[i_out]))
    assert np.sum(data.sun_valid[~lit]) == 0


# =========================================================================== 9. magnetometer only
def test_mag_only_runs_and_is_weaker_about_field_direction(base_run):
    truth, data, res_both = base_run
    cfg = MekfConfig(use_sun=False)
    # start from the same TRIAD solution as the two-sensor run (k_init = 0)
    res_mag = run_mekf(data, truth["b_eci"], truth["s_eci"], cfg)
    assert res_mag.k_init == res_both.k_init
    assert np.allclose(res_mag.q[0], res_both.q[0], atol=1e-14)
    assert np.all(np.isfinite(res_mag.std[res_mag.k_init:]))
    P = res_mag.P_att
    ev = np.linalg.eigvalsh(0.5 * (P + np.swapaxes(P, 1, 2)))
    assert np.min(ev[res_mag.k_init:]) > 0.0
    assert np.allclose(np.linalg.norm(res_mag.q, axis=1), 1.0, atol=1e-12)
    assert len(res_mag.nis.get("sun", (np.array([]),))[0]) == 0   # no sun update was made

    # attitude sigma about the instantaneous field direction at the last eclipse sample
    # (the two-sensor run has had no sun update for the whole eclipse there)
    k = int(np.nonzero(~truth["lit"])[0][-1])
    u = quat.rotate_inverse(truth["q"][k], truth["b_eci"][k])
    u /= np.linalg.norm(u)
    s_mag = np.sqrt(u @ res_mag.P_att[k] @ u)
    s_both = np.sqrt(u @ res_both.P_att[k] @ u)
    # mag-only: the about-field axis is the weak one (perpendicular axes are fixed by the field)
    e1 = np.cross(u, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    s_perp = np.sqrt(e1 @ res_mag.P_att[k] @ e1)
    print(f"\n[mag-only] sigma about field dir at last eclipse sample: mag-only {np.rad2deg(s_mag):.4f} deg, "
          f"both {np.rad2deg(s_both):.4f} deg (ratio {s_mag / s_both:.2f}); "
          f"mag-only perpendicular {np.rad2deg(s_perp):.4f} deg")
    assert s_mag > s_both
    assert s_mag > 3.0 * s_perp


# =========================================================================== 10. gating
def test_gating_rejects_outlier_and_estimate_changes_only_by_propagation():
    n = 300
    kk = 150
    truth = synthetic_truth(n)
    data = sensors_for(truth, seed=2, bias0=_bias_vec())
    cfg = MekfConfig(gate_prob=0.9973)

    # clean gated run: nearly all updates accepted (gate = chi2 99.73 %)
    clean = run_mekf(data, truth["b_eci"], truth["s_eci"], cfg)
    acc_clean = np.asarray(clean.nis["mag"][2], bool)
    assert acc_clean.mean() > 0.97

    # outlier: +100 sigma_m along a direction perpendicular to the (body) field
    b_body = quat.rotate_inverse(truth["q"][kk], truth["b_eci"][kk])
    d = np.cross(b_body, [1.0, 0.0, 0.0])
    d /= np.linalg.norm(d)
    assert abs(d @ b_body) < 1e-12 * np.linalg.norm(b_body)
    data_out = sensors_for(truth, seed=2, bias0=_bias_vec())
    data_out.mag[kk] = data.mag[kk] + 100.0 * SIGMA_M * d

    gated = run_mekf(data_out, truth["b_eci"], truth["s_eci"], cfg)
    t_log, nis_log, acc_log = gated.nis["mag"]
    i = int(np.nonzero(np.isclose(t_log, truth["t"][kk]))[0][0])
    assert not acc_log[i]
    assert nis_log[i] > 1e3                       # (~100^2 / 1 sigma^2 scale) clearly an outlier
    assert np.all(acc_log[np.arange(len(acc_log)) != i] == acc_clean[np.arange(len(acc_log)) != i])

    # reference run through the public MEKF class that simply skips that mag update
    q0, P0 = triad_init(data_out.mag[0], data_out.sun[0], truth["b_eci"][0], truth["s_eci"][0], cfg)
    f = MEKF(q0, np.zeros(3), P0, cfg)
    q_skip = np.full((n + 1, 4), np.nan)
    b_skip = np.full((n + 1, 3), np.nan)
    q_skip[0], b_skip[0] = f.q, f.b
    t = truth["t"]
    for k in range(1, n + 1):
        f.propagate(data_out.gyro[k], t[k] - t[k - 1])
        if k != kk:
            f.update_mag(data_out.mag[k], truth["b_eci"][k], t[k])
        if data_out.sun_valid[k]:
            f.update_sun(data_out.sun[k], truth["s_eci"][k], t[k])
        q_skip[k], b_skip[k] = f.q, f.b
    # same quaternion sign convention along the path -> compare elementwise
    sgn = np.sign(np.sum(gated.q[kk] * q_skip[kk]))
    assert np.max(np.abs(gated.q[kk] - sgn * q_skip[kk])) < 1e-12
    assert np.max(np.abs(gated.b[kk] - b_skip[kk])) < 1e-12
    # and for the whole remaining run (history identical)
    assert np.max(np.abs(quat.angle_between(gated.q, q_skip))) < 1e-10
    assert np.max(np.abs(gated.b - b_skip)) < 1e-12

    # contrast: without gating the outlier drags the estimate away (>> 1e-4 rad)
    ungated = run_mekf(data_out, truth["b_eci"], truth["s_eci"], MekfConfig())
    pulled = quat.angle_between(ungated.q[kk], q_skip[kk])
    assert pulled > 1e-4
    print(f"\n[gating] outlier NIS = {nis_log[i]:.3e}; ungated estimate moved {pulled:.3e} rad "
          f"vs gated max |dq| ~ 0; clean accepted fraction {acc_clean.mean():.4f}")


# =========================================================================== 11. real environment
def test_real_environment_smoke(pool_jobs):
    out = pool_jobs["real"].result()
    res = out["res"]
    t = out["t"]
    k0 = res.k_init
    assert k0 == 0                                  # pointing run starts in sunlight
    err = attitude_errors(out["q_true"], res.q)
    n = nees(err[k0 + 300:], res.P_att[k0 + 300:])
    nis_mag = _valid_nis(res, "mag")
    nis_sun = _valid_nis(res, "sun")
    k25 = int(0.25 * P_ORB)
    max_err = np.rad2deg(np.max(np.linalg.norm(err[k25:], axis=1)))
    lit = out["lit"]
    e = np.rad2deg(np.linalg.norm(err, axis=1))
    print(f"\n[real env] NEES = {n.mean():.3f}, NIS mag = {nis_mag.mean():.3f} (n={nis_mag.size}), "
          f"NIS sun = {nis_sun.mean():.3f} (n={nis_sun.size}), max |dth| after 0.25 orbit = "
          f"{max_err:.3f} deg, RMS sunlit {np.sqrt(np.mean(e[k25:][lit[k25:]]**2)):.3f} / "
          f"eclipse {np.sqrt(np.mean(e[k25:][~lit[k25:]]**2)):.3f} deg, "
          f"bias err at 0.25 orbit {np.rad2deg(np.linalg.norm(out['bias'][k25] - res.b[k25])):.4f} deg/s")
    assert len(t) == int(round(P_ORB)) + 1
    assert 2.0 <= n.mean() <= 4.0
    assert max_err < 3.0
    assert 2.7 <= nis_mag.mean() <= 3.3
    assert 2.7 <= nis_sun.mean() <= 3.3
