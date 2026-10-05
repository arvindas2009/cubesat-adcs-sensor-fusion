"""Tests for ``common.ekf`` (generic error-state EKF core).

Spec: module docstring of ``common/ekf.py`` and CONVENTIONS.md section 3.
Expected values come from independent oracles written in this file (textbook
Kalman filter, scipy DARE / Lyapunov solvers, closed-form Van Loan results,
chi-square closed forms) -- never from the module's own internals.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import solve_continuous_lyapunov, solve_discrete_are
from scipy.stats import chi2

from common import ekf


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def random_spd(rng, n, scale=1.0):
    A = rng.standard_normal((n, n))
    return scale * (A @ A.T + n * np.eye(n))


# 1-D constant-velocity model: state [pos, vel], position measurements.
DT = 0.1
Q_ACC = 0.5          # white-acceleration spectral density
R_POS = 0.25         # position measurement variance
F_CV = np.array([[1.0, DT], [0.0, 1.0]])
H_CV = np.array([[1.0, 0.0]])
QD_CV = Q_ACC * np.array([[DT**3 / 3, DT**2 / 2], [DT**2 / 2, DT]])
P0_CV = np.diag([4.0, 1.0])


def textbook_kf_step(x, P, z, F, H, Q, R):
    """One predict+update of a textbook KF (standard form, explicit inverse)."""
    x = F @ x
    P = F @ P @ F.T + Q
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ (z - H @ x)
    P = (np.eye(len(x)) - K @ H) @ P
    return x, P


# ---------------------------------------------------------------------------
# 1. Linear constant-velocity tracking
# ---------------------------------------------------------------------------

def test_cv_matches_textbook_kf_step_by_step():
    rng = np.random.default_rng(1)
    L = np.linalg.cholesky(QD_CV)
    x_true = np.array([1.0, 0.5])
    x_nom = np.array([0.0, 0.0])           # error-state filter: nominal state
    x_kf = x_nom.copy()
    P_kf = P0_CV.copy()
    f = ekf.ErrorStateEKF(P0_CV)
    for k in range(200):
        x_true = F_CV @ x_true + L @ rng.standard_normal(2)
        z = H_CV @ x_true + np.sqrt(R_POS) * rng.standard_normal(1)

        x_nom = F_CV @ x_nom
        f.predict(F_CV, QD_CV)
        res = f.update(z - H_CV @ x_nom, H_CV, R_POS, name="pos", t=k * DT)
        x_nom = x_nom + res.dx             # inject; error state is zero again

        x_kf, P_kf = textbook_kf_step(x_kf, P_kf, z, F_CV, H_CV, QD_CV, np.array([[R_POS]]))

        # Same algebra, different arithmetic order: 1e-9 is generous (O(1) values).
        np.testing.assert_allclose(x_nom, x_kf, atol=1e-9, rtol=0)
        np.testing.assert_allclose(f.P, P_kf, atol=1e-9, rtol=0)


def test_cv_predicted_covariance_converges_to_dare():
    rng = np.random.default_rng(2)
    f = ekf.ErrorStateEKF(P0_CV)
    x_nom = np.zeros(2)
    x_true = np.zeros(2)
    L = np.linalg.cholesky(QD_CV)
    for _ in range(600):
        x_true = F_CV @ x_true + L @ rng.standard_normal(2)
        z = H_CV @ x_true + np.sqrt(R_POS) * rng.standard_normal(1)
        x_nom = F_CV @ x_nom
        f.predict(F_CV, QD_CV)
        P_pred = f.P
        res = f.update(z - H_CV @ x_nom, H_CV, R_POS)
        x_nom = x_nom + res.dx
    # Filtering DARE: X = F X F' + Q - F X H'(H X H' + R)^-1 H X F'; X = predicted P.
    P_dare = solve_discrete_are(F_CV.T, H_CV.T, QD_CV, np.array([[R_POS]]))
    # Convergence is geometric and complete after 600 steps; limit is rounding.
    np.testing.assert_allclose(P_pred, P_dare, rtol=1e-8, atol=1e-12)


def test_cv_monte_carlo_consistency():
    n_runs, n_steps = 150, 200
    rng = np.random.default_rng(3)
    L = np.linalg.cholesky(QD_CV)
    Lp0 = np.linalg.cholesky(P0_CV)
    nis_all = np.empty((n_runs, n_steps))
    nees = np.empty((n_runs, n_steps))
    for run in range(n_runs):
        x_true = Lp0 @ rng.standard_normal(2)      # truth drawn from N(0, P0)
        x_nom = np.zeros(2)
        f = ekf.ErrorStateEKF(P0_CV)
        w = rng.standard_normal((n_steps, 2))
        v = np.sqrt(R_POS) * rng.standard_normal(n_steps)
        for k in range(n_steps):
            x_true = F_CV @ x_true + L @ w[k]
            z = H_CV @ x_true + v[k]
            x_nom = F_CV @ x_nom
            f.predict(F_CV, QD_CV)
            res = f.update(z - H_CV @ x_nom, H_CV, R_POS, name="pos")
            x_nom = x_nom + res.dx
            e = x_true - x_nom
            nees[run, k] = e @ np.linalg.solve(f.P, e)
            nis_all[run, k] = res.nis
        np.testing.assert_allclose(f.innovations("pos").nis, nis_all[run])

    # NIS samples are independent for a consistent KF (white innovations), so the
    # pooled mean is chi2(N*dof)/N. Use 99% bounds: a correct filter fails
    # this with probability 1% (the seed is fixed so the outcome is deterministic).
    lo, hi = ekf.nis_bounds(1, n_runs * n_steps, prob=0.99)
    assert lo < nis_all.mean() < hi

    # NEES at a fixed time is chi2(2R)/R across the R independent runs; the
    # time-average has no larger variance, so the 99% bounds for one time step
    # are a conservative interval for the grand mean (expected value 2).
    lo2, hi2 = ekf.nis_bounds(2, n_runs, prob=0.99)
    assert lo2 < nees.mean() < hi2
    assert abs(nees.mean() - 2.0) < 0.3

    # Fraction of individual NIS in the 95% interval: binomial sd =
    # sqrt(.95*.05/30000)=0.0013, so 0.01 is a ~7-sigma tolerance.
    frac = ekf.nis_fraction_inside(nis_all.ravel(), 1)
    assert frac == pytest.approx(0.95, abs=0.01)


# ---------------------------------------------------------------------------
# 2. Joseph form: symmetry, positive definiteness, numerical robustness
# ---------------------------------------------------------------------------

JOSEPH_SEEDS = tuple(range(20))     # fixed; Joseph never fails on these, short form does
JOSEPH_R = 1e-10
JOSEPH_STEPS = 30


def _joseph_problem(seed):
    """6-state problem: P0 = U diag(10**u) U^T with a 14-decade spectrum
    (u ~ U(-8, 6)), U a random rotation (QR of a Gaussian, sign-fixed so it is
    Haar distributed), and 30 random 2x6 measurement matrices / residuals."""
    rng = np.random.default_rng(seed)
    Qm, Rm = np.linalg.qr(rng.standard_normal((6, 6)))
    U = Qm * np.sign(np.diag(Rm))
    P0 = U @ np.diag(10.0 ** rng.uniform(-8.0, 6.0, 6)) @ U.T
    P0 = 0.5 * (P0 + P0.T)
    Hs = rng.standard_normal((JOSEPH_STEPS, 2, 6))
    rs = rng.standard_normal((JOSEPH_STEPS, 2))
    return P0, Hs, rs


def _is_healthy(P, S):
    """S has a Cholesky factor and P has no eigenvalue below -1e-12 * max eig.
    (1e-12 relative is ~4 orders above double-precision eigen-solver noise for
    a symmetric matrix, so only a genuinely indefinite P trips it.)"""
    try:
        np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        return False
    w = np.linalg.eigvalsh(0.5 * (P + P.T))
    return bool(w[0] >= -1e-12 * w[-1])


def _short_form_fails(seed):
    """Control: textbook short-form update P = (I-KH)P, written inline (does
    NOT use ``ekf``), symmetrised after each step as in ``ekf``. True if S loses
    its Cholesky factor or P goes indefinite at any of the 30 steps."""
    P, Hs, _ = _joseph_problem(seed)
    R = JOSEPH_R * np.eye(2)
    for H in Hs:
        S = H @ P @ H.T + R
        try:
            np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            return True
        K = P @ H.T @ np.linalg.inv(S)
        P = (np.eye(6) - K @ H) @ P
        P = 0.5 * (P + P.T)
        if not _is_healthy(P, S):
            return True
    return False


@pytest.mark.parametrize("seed", JOSEPH_SEEDS)
def test_joseph_form_stays_spd_ill_conditioned(seed):
    # 14 decades of eigenvalue spread, near noise-free measurements (R = 1e-10),
    # H unaligned with the eigenvectors, 30 consecutive updates: the Joseph
    # form keeps S Cholesky-factorable and P positive semi-definite at every step.
    P0, Hs, rs = _joseph_problem(seed)
    f = ekf.ErrorStateEKF(P0)
    for k, (H, r) in enumerate(zip(Hs, rs)):
        res = f.update(r, H, JOSEPH_R, name="m", t=float(k))
        assert res.accepted
        assert _is_healthy(f.P, res.S), f"seed {seed}, step {k}"
        assert np.array_equal(f.P, f.P.T)


def test_joseph_control_short_form_fails_on_same_seeds():
    # Control proving the test above has teeth: the short form (I-KH)P fails
    # (S not Cholesky-factorable or P indefinite) on the same problems. Seeds
    # 0..19 fail for the short form on about half of them (a review found ~29/300
    # failures with a different generator and ~1/300 for Joseph); require at
    # least 5 to leave margin against platform BLAS differences.
    fails = [s for s in JOSEPH_SEEDS if _short_form_fails(s)]
    assert len(fails) >= 5, f"short form failed only on seeds {fails}"


def test_p_exactly_symmetric_after_predict_update_reset():
    rng = np.random.default_rng(5)
    n = 5
    f = ekf.ErrorStateEKF(random_spd(rng, n))
    for _ in range(10):
        Phi = np.eye(n) + 0.1 * rng.standard_normal((n, n))
        f.predict(Phi, random_spd(rng, n, 0.01))
        assert np.array_equal(f.P, f.P.T)
        H = rng.standard_normal((2, n))
        f.update(rng.standard_normal(2), H, np.array([[1.0, 0.2], [0.2, 2.0]]))
        assert np.array_equal(f.P, f.P.T)
        f.reset(np.eye(n) + 0.01 * rng.standard_normal((n, n)))
        assert np.array_equal(f.P, f.P.T)
        assert ekf.is_spd(f.P)


def test_near_singular_innovation_covariance_gives_no_nan():
    # Two identical measurement rows with tiny noise: S = [[a+R, a],[a, a+R]]
    # with R = 1e-14 a (condition number ~ 2e14).
    f = ekf.ErrorStateEKF(np.eye(3))
    H = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    res = f.update(np.array([1.0, 1.0 + 1e-7]), H, 1e-14)
    assert np.all(np.isfinite(res.dx))
    assert np.isfinite(res.nis)
    P = f.P
    assert np.all(np.isfinite(P))
    assert np.array_equal(P, P.T)
    assert np.all(np.diag(P) >= 0)
    # Both rows saw the same state; the state estimate must be ~ the residual.
    assert res.dx[0] == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.isfinite(f.std))


# ---------------------------------------------------------------------------
# 3. update_sequential == update for diagonal R
# ---------------------------------------------------------------------------

def test_update_sequential_equals_block_update():
    rng = np.random.default_rng(6)
    n, m = 6, 3
    P0 = random_spd(rng, n)
    a = ekf.ErrorStateEKF(P0)
    b = ekf.ErrorStateEKF(P0)
    for k in range(5):
        H = rng.standard_normal((m, n))
        r = rng.standard_normal(m)
        Rd = rng.uniform(0.2, 2.0, m)
        ra = a.update(r, H, np.diag(Rd), name="blk", t=float(k))
        rb = b.update_sequential(r, H, Rd, name="seq", t=float(k))
        np.testing.assert_allclose(rb.dx, ra.dx, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(b.P, a.P, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(rb.S, ra.S, atol=1e-10, rtol=1e-10)
        assert rb.nis == pytest.approx(ra.nis, rel=1e-10)
        assert rb.accepted and ra.accepted
        # Predict between updates so P is not trivially the same path.
        Phi = np.eye(n) + 0.05 * rng.standard_normal((n, n))
        Qd = random_spd(rng, n, 0.01)
        a.predict(Phi, Qd)
        b.predict(Phi, Qd)
    np.testing.assert_allclose(b.innovations("seq").nis, a.innovations("blk").nis, rtol=1e-10)


def test_update_sequential_matches_independent_formula():
    # Independent oracle: block KF with explicit inverse, standard form.
    rng = np.random.default_rng(7)
    n, m = 6, 3
    P0 = random_spd(rng, n)
    H = rng.standard_normal((m, n))
    r = rng.standard_normal(m)
    Rd = np.array([0.5, 1.0, 2.0])
    S = H @ P0 @ H.T + np.diag(Rd)
    K = P0 @ H.T @ np.linalg.inv(S)
    f = ekf.ErrorStateEKF(P0)
    res = f.update_sequential(r, H, Rd)
    np.testing.assert_allclose(res.dx, K @ r, atol=1e-10)
    np.testing.assert_allclose(f.P, (np.eye(n) - K @ H) @ P0, atol=1e-10)
    assert res.nis == pytest.approx(r @ np.linalg.solve(S, r), rel=1e-10)


# ---------------------------------------------------------------------------
# 4. Gating
# ---------------------------------------------------------------------------

def test_gate_rejects_gross_outlier_but_logs_it():
    f = ekf.ErrorStateEKF(np.eye(2))
    H = np.array([[1.0, 0.0]])
    gate = ekf.chi2_gate(1)
    P_before = f.P
    res = f.update(np.array([100.0]), H, 1.0, name="gps", t=3.0, gate=gate)
    assert res.accepted is False
    np.testing.assert_array_equal(res.dx, np.zeros(2))
    np.testing.assert_array_equal(f.P, P_before)
    assert res.nis == pytest.approx(100.0**2 / 2.0)     # S = 1 + 1 = 2
    log = f.innovations("gps")
    assert log.nis.shape == (1,)
    assert not log.accepted[0]
    assert log.t[0] == 3.0


def test_gate_accepts_normal_measurement():
    f = ekf.ErrorStateEKF(np.eye(2))
    H = np.array([[1.0, 0.0]])
    res = f.update(np.array([0.5]), H, 1.0, name="gps", gate=ekf.chi2_gate(1))
    assert res.accepted is True
    assert res.dx[0] == pytest.approx(0.25)            # K = 1/2
    assert f.P[0, 0] == pytest.approx(0.5)
    assert f.innovations("gps").accepted[0]


def test_gate_sequential_rejects_whole_measurement():
    rng = np.random.default_rng(8)
    n, m = 4, 3
    P0 = random_spd(rng, n)
    f = ekf.ErrorStateEKF(P0)
    H = rng.standard_normal((m, n))
    res = f.update_sequential(np.array([1e3, 0.0, 0.0]), H, np.ones(m),
                              gate=ekf.chi2_gate(m))
    assert not res.accepted
    np.testing.assert_array_equal(res.dx, 0.0)
    np.testing.assert_array_equal(f.P, P0)
    assert len(f.history["meas"]) == 1


def test_chi2_gate_and_bounds_closed_forms():
    # chi2 with 2 dof has CDF 1 - exp(-x/2) -> ppf(p) = -2 ln(1-p).
    assert ekf.chi2_gate(2, 0.95) == pytest.approx(-2 * np.log(0.05), rel=1e-9)
    assert ekf.chi2_gate(1) == pytest.approx(9.0, abs=1e-2)  # (3 sigma)^2, p=0.9973
    lo, hi = ekf.nis_bounds(2, 1, 0.95)
    assert lo == pytest.approx(-2 * np.log(0.975), rel=1e-9)
    assert hi == pytest.approx(-2 * np.log(0.025), rel=1e-9)
    # Averaging narrows the interval around dof.
    lo50, hi50 = ekf.nis_bounds(3, 50)
    assert lo50 < 3 < hi50 and (hi50 - lo50) < (ekf.nis_bounds(3, 1)[1] - ekf.nis_bounds(3, 1)[0])


def test_nis_fraction_inside_synthetic():
    rng = np.random.default_rng(9)
    s = rng.chisquare(3, 200000)
    assert ekf.nis_fraction_inside(s, 3) == pytest.approx(0.95, abs=0.005)
    assert ekf.nis_fraction_inside(np.array([1e-9, 1e3]), 2) == 0.0
    assert ekf.nis_fraction_inside(np.array([2.0]), 2) == 1.0   # inside [0.0506, 7.378]
    assert np.isnan(ekf.nis_fraction_inside([], 2))


# ---------------------------------------------------------------------------
# 5. Van Loan discretisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dt", [0.01, 0.1, 2.5])
def test_van_loan_cv_white_acceleration(dt):
    q = 0.7
    F = np.array([[0.0, 1.0], [0.0, 0.0]])
    Qc = np.array([[0.0, 0.0], [0.0, q]])
    Phi, Qd = ekf.van_loan(F, Qc, dt)
    Qd_exact = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    np.testing.assert_allclose(Phi, [[1.0, dt], [0.0, 1.0]], atol=1e-12)
    np.testing.assert_allclose(Qd, Qd_exact, atol=1e-12 * max(1.0, dt**3), rtol=1e-12)


def test_van_loan_with_G_matrix_and_vector_Qc():
    q, dt = 0.7, 0.2
    F = np.array([[0.0, 1.0], [0.0, 0.0]])
    G = np.array([[0.0], [1.0]])
    Qd_exact = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    _, Qd1 = ekf.van_loan(F, np.array([[q]]), dt, G)
    _, Qd2 = ekf.van_loan(F, np.array([q]), dt, G)          # 1-D Qc = diagonal
    np.testing.assert_allclose(Qd1, Qd_exact, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(Qd2, Qd_exact, atol=1e-12, rtol=1e-12)
    # G scaling: noise gain g multiplies the spectral density by g^2.
    _, Qd3 = ekf.van_loan(F, np.array([[q]]), dt, 3.0 * G)
    np.testing.assert_allclose(Qd3, 9.0 * Qd_exact, atol=1e-12, rtol=1e-12)


def test_van_loan_shape_mismatch_raises():
    with pytest.raises(ValueError):
        ekf.van_loan(np.zeros((2, 2)), np.eye(3), 0.1)


def test_van_loan_first_order_gauss_markov():
    tau, q, dt = 2.0, 0.3, 0.5
    Phi, Qd = ekf.van_loan(np.array([[-1.0 / tau]]), np.array([[q]]), dt)
    assert Phi[0, 0] == pytest.approx(np.exp(-dt / tau), rel=1e-12)
    assert Qd[0, 0] == pytest.approx(q * tau / 2 * (1 - np.exp(-2 * dt / tau)), rel=1e-12)
    # With a noise gain g
    g = 1.7
    _, Qd_g = ekf.van_loan(np.array([[-1.0 / tau]]), np.array([[q]]), dt, np.array([[g]]))
    assert Qd_g[0, 0] == pytest.approx(g**2 * q * tau / 2 * (1 - np.exp(-2 * dt / tau)), rel=1e-12)


def test_van_loan_stable_system_matches_lyapunov_identity():
    # For stable F: Pinf solves F P + P F' + Q = 0, and Qd = Pinf - Phi Pinf Phi'.
    rng = np.random.default_rng(10)
    n, dt = 4, 0.3
    A = rng.standard_normal((n, n))
    F = A - (np.max(np.linalg.eigvals(A).real) + 0.5) * np.eye(n)   # all eig(F) real part < -0.5
    Q = random_spd(rng, n, 0.1)
    Phi, Qd = ekf.van_loan(F, Q, dt)
    Pinf = solve_continuous_lyapunov(F, -Q)
    np.testing.assert_allclose(Qd, Pinf - Phi @ Pinf @ Phi.T, atol=1e-10, rtol=1e-10)
    assert np.array_equal(Qd, Qd.T)
    assert ekf.is_spd(Qd)


# ---------------------------------------------------------------------------
# 6. Nonlinear pendulum, error-state path
# ---------------------------------------------------------------------------

G_ACC, L_PEND = 9.81, 1.0
W2 = G_ACC / L_PEND
P_DT = 0.1
P_Q = 0.05           # white torque noise density on theta-ddot [rad^2/s^3]
P_SIGMA_Z = 0.02     # measurement noise std [m]
P_QC = np.array([[0.0, 0.0], [0.0, P_Q]])


def pend_f(x):
    return np.array([x[1], -W2 * np.sin(x[0])])


def rk4(x, dt, substeps=10):
    h = dt / substeps
    for _ in range(substeps):
        k1 = pend_f(x)
        k2 = pend_f(x + 0.5 * h * k1)
        k3 = pend_f(x + 0.5 * h * k2)
        k4 = pend_f(x + h * k3)
        x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def pend_F(theta):
    return np.array([[0.0, 1.0], [-W2 * np.cos(theta), 0.0]])


def run_pendulum(rng, n_steps, offset):
    """One closed-loop run; returns (nis, error, std, NEES) arrays."""
    x_true = np.array([0.5, 0.0])
    x_nom = x_true + offset
    f = ekf.ErrorStateEKF(np.diag([0.3**2, 0.3**2]), state_names=["theta", "omega"])
    w = rng.standard_normal((n_steps, 2))
    v = P_SIGMA_Z * rng.standard_normal(n_steps)
    err = np.empty((n_steps, 2))
    std = np.empty((n_steps, 2))
    nees = np.empty(n_steps)
    for k in range(n_steps):
        # Truth: nonlinear flow plus discrete process noise consistent with Qc.
        th_old = x_true[0]
        x_new = rk4(x_true, P_DT)
        _, Qt = ekf.van_loan(pend_F(0.5 * (th_old + x_new[0])), P_QC, P_DT)
        x_true = x_new + np.linalg.cholesky(Qt) @ w[k]
        z = L_PEND * np.sin(x_true[0]) + v[k]

        # Filter: nominal propagated nonlinearly, error covariance via linearisation.
        th_old = x_nom[0]
        x_nom = rk4(x_nom, P_DT)
        Phi, Qd = ekf.van_loan(pend_F(0.5 * (th_old + x_nom[0])), P_QC, P_DT)
        f.predict(Phi, Qd)

        H = np.array([[L_PEND * np.cos(x_nom[0]), 0.0]])
        res = f.update(np.array([z - L_PEND * np.sin(x_nom[0])]), H, P_SIGMA_Z**2,
                       name="pos", t=(k + 1) * P_DT)
        x_nom = x_nom + res.dx
        err[k] = x_true - x_nom
        std[k] = f.std
        nees[k] = err[k] @ np.linalg.solve(f.P, err[k])
    return f.innovations("pos").nis, err, std, nees


def test_pendulum_error_state_ekf_converges_and_is_consistent():
    n_runs, n_steps, burn_in = 40, 300, 30
    rng = np.random.default_rng(11)
    offset = np.array([0.3, -0.2])          # wrong initial estimate: 0.3 rad, -0.2 rad/s
    nis = np.empty((n_runs, n_steps))
    err = np.empty((n_runs, n_steps, 2))
    std = np.empty((n_runs, n_steps, 2))
    nees = np.empty((n_runs, n_steps))
    for i in range(n_runs):
        nis[i], err[i], std[i], nees[i] = run_pendulum(rng, n_steps, offset)

    # Convergence: uncertainty shrinks a lot and the actual errors sit inside 3 sigma.
    # (steady-state theta std is ~0.015 rad, set by the 0.02 m measurement noise; the
    # initial std was 0.3, so require at least a 10x reduction).
    assert np.all(std[:, -1, 0] < 0.1 * 0.3)
    z = np.abs(err[:, burn_in:, :]) / std[:, burn_in:, :]
    # Gaussian: P(|e| < 3 sigma) = 0.9973 per component; allow nonlinearity/finite-sample slack.
    assert np.mean(z < 3.0) > 0.99
    # Typical final error far below the initial 0.3 rad offset.
    assert np.sqrt(np.mean(err[:, -50:, 0] ** 2)) < 0.3 / 10

    # Time-averaged NIS after burn-in. Pooled over runs the samples are independent,
    # so the grand mean ~ chi2(N*dof)/N. Bounds are 99.9% (deliberately widened from
    # 99%): a review showed the seed-11 pooled mean 0.969 is a -2.3 sigma sampling
    # draw, not a bias -- the grand mean over 30 seeds is 0.996 +/- 0.016. A 99%
    # test fails ~1% of seeds by construction; 99.9% still catches real
    # inconsistency (a 10% bias in S would be > 5 sigma away).
    post = nis[:, burn_in:]
    lo, hi = ekf.nis_bounds(1, post.size, prob=0.999)
    assert lo < post.mean() < hi
    # Per-run time averages (n_post samples each) should lie in the 99% bounds for
    # nearly all runs (expected miss rate 1%; allow 10% for finite-sample slack).
    lo_r, hi_r = ekf.nis_bounds(1, post.shape[1], prob=0.99)
    inside = (post.mean(axis=1) > lo_r) & (post.mean(axis=1) < hi_r)
    assert inside.mean() >= 0.9
    assert ekf.nis_fraction_inside(post.ravel(), 1) == pytest.approx(0.95, abs=0.02)

    # NEES (dof 2) averaged over runs and time after burn-in: same conservative
    # single-time-step chi2(2R)/R bounds as in the linear test, at 99%.
    lo2, hi2 = ekf.nis_bounds(2, n_runs, prob=0.99)
    assert lo2 < nees[:, burn_in:].mean() < hi2


# ---------------------------------------------------------------------------
# 7. API / validation
# ---------------------------------------------------------------------------

def test_constructor_validation():
    with pytest.raises(ValueError):
        ekf.ErrorStateEKF(np.diag([1.0, -1.0]))                 # not positive definite
    with pytest.raises(ValueError):
        ekf.ErrorStateEKF(np.array([[1.0, 0.5], [0.0, 1.0]]))   # not symmetric
    with pytest.raises(ValueError):
        ekf.ErrorStateEKF(np.zeros((2, 2)))                     # singular
    with pytest.raises(ValueError):
        ekf.ErrorStateEKF(np.ones((2, 3)))                      # not square
    with pytest.raises(ValueError):
        ekf.ErrorStateEKF(np.eye(2), state_names=["a"])         # wrong name count
    f = ekf.ErrorStateEKF(np.eye(2), state_names=["a", "b"])
    assert f.n == 2 and f.state_names == ["a", "b"]
    assert ekf.ErrorStateEKF(np.eye(2)).state_names is None


def test_shape_mismatches_raise_value_error():
    f = ekf.ErrorStateEKF(np.eye(3))
    with pytest.raises(ValueError):
        f.predict(np.eye(2), np.eye(2))
    with pytest.raises(ValueError):
        f.predict(np.eye(3), np.eye(2))
    with pytest.raises(ValueError):
        f.update(np.zeros(2), np.zeros((2, 4)), 1.0)            # H wrong columns
    with pytest.raises(ValueError):
        f.update(np.zeros(2), np.zeros((3, 3)), 1.0)            # H wrong rows
    with pytest.raises(ValueError):
        f.update(np.zeros(2), np.zeros((2, 3)), np.ones(3))     # R vector wrong length
    with pytest.raises(ValueError):
        f.update(np.zeros(2), np.zeros((2, 3)), np.eye(3))      # R matrix wrong shape
    with pytest.raises(ValueError):
        f.update_sequential(np.zeros(2), np.zeros((2, 4)), np.ones(2))
    with pytest.raises(ValueError):
        f.reset(np.eye(2))
    with pytest.raises(ValueError):
        f.set_P(np.eye(2))
    with pytest.raises(ValueError):
        f.set_P(np.diag([1.0, 1.0, -1.0]))
    # filter state untouched by failed calls
    np.testing.assert_array_equal(f.P, np.eye(3))


def test_R_accepted_as_scalar_vector_and_matrix():
    rng = np.random.default_rng(12)
    n, m = 4, 2
    P0 = random_spd(rng, n)
    H = rng.standard_normal((m, n))
    r = rng.standard_normal(m)
    outs = []
    for R in (0.3, np.array([0.3, 0.3]), 0.3 * np.eye(m)):
        f = ekf.ErrorStateEKF(P0)
        res = f.update(r, H, R)
        outs.append((res.dx, f.P, res.S, res.nis))
    for o in outs[1:]:
        for a, b in zip(outs[0], o):
            np.testing.assert_allclose(a, b, atol=1e-14, rtol=1e-14)
    # Independent oracle for the scalar case.
    S = H @ P0 @ H.T + 0.3 * np.eye(m)
    K = P0 @ H.T @ np.linalg.inv(S)
    np.testing.assert_allclose(outs[0][0], K @ r, atol=1e-12)

    # Non-trivial vector vs diag matrix
    Rd = np.array([0.1, 0.9])
    fa, fb = ekf.ErrorStateEKF(P0), ekf.ErrorStateEKF(P0)
    ra, rb = fa.update(r, H, Rd), fb.update(r, H, np.diag(Rd))
    np.testing.assert_allclose(ra.dx, rb.dx, atol=1e-14)
    np.testing.assert_allclose(fa.P, fb.P, atol=1e-14)


def test_scalar_residual_inputs_accepted():
    f = ekf.ErrorStateEKF(np.eye(2))
    res = f.update(0.5, [1.0, 0.0], 1.0)           # scalar residual, 1-D H
    assert res.dx.shape == (2,)
    assert res.dx[0] == pytest.approx(0.25)


def test_innovations_stacking_shapes_and_values():
    rng = np.random.default_rng(13)
    n, m, N = 3, 2, 5
    P = random_spd(rng, n)
    f = ekf.ErrorStateEKF(P)
    H = rng.standard_normal((m, n))
    R = np.diag([0.5, 1.5])
    rs, ts, S_expected = [], [], []
    for k in range(N):
        r = rng.standard_normal(m)
        S_expected.append(H @ f.P @ H.T + R)
        f.update(r, H, R, name="a", t=0.1 * k)
        rs.append(r)
        ts.append(0.1 * k)
    f.update(np.zeros(1), np.ones((1, n)), 1.0, name="b")        # other name, no time

    log = f.innovations("a")
    assert log.t.shape == (N,) and log.residual.shape == (N, m)
    assert log.sigma.shape == (N, m) and log.nis.shape == (N,) and log.accepted.shape == (N,)
    assert log.accepted.dtype == bool and log.accepted.all()
    np.testing.assert_allclose(log.t, ts)
    np.testing.assert_allclose(log.residual, np.array(rs))
    np.testing.assert_allclose(log.sigma, np.sqrt([np.diag(S) for S in S_expected]), rtol=1e-12)
    np.testing.assert_allclose(
        log.nis, [r @ np.linalg.solve(S, r) for r, S in zip(rs, S_expected)], rtol=1e-10)

    logb = f.innovations("b")
    assert logb.nis.shape == (1,) and np.isnan(logb.t[0]) and logb.residual.shape == (1, 1)
    assert set(f.history) == {"a", "b"}
    assert len(f.history["a"]) == N

    empty = f.innovations("nonexistent")
    assert empty.nis.shape == (0,) and empty.t.shape == (0,)


def test_innovations_ragged_dof_nan_padded_and_rejections_logged():
    n = 6
    f = ekf.ErrorStateEKF(np.diag(np.arange(1.0, n + 1.0)))
    H3 = np.hstack([np.eye(3), np.zeros((3, 3))])
    H6 = np.eye(n)
    gate3 = ekf.chi2_gate(3)
    # m=3, m=6, m=3 gross outlier (rejected by the gate), m=6
    f.update(np.array([0.1, -0.2, 0.3]), H3, 0.5, name="x", t=0.0)
    f.update(np.linspace(-0.3, 0.3, 6), H6, 0.5, name="x", t=1.0)
    rej = f.update(np.array([1e3, 0.0, 0.0]), H3, 0.5, name="x", t=2.0, gate=gate3)
    f.update(np.linspace(0.3, -0.3, 6), H6, 0.5, name="x", t=3.0)
    assert rej.accepted is False

    log = f.innovations("x")
    assert log.residual.shape == (4, 6) and log.sigma.shape == (4, 6)
    np.testing.assert_array_equal(log.dof, [3, 6, 3, 6])
    np.testing.assert_array_equal(log.accepted, [True, True, False, True])
    np.testing.assert_allclose(log.t, [0.0, 1.0, 2.0, 3.0])
    assert log.nis.shape == (4,)
    # m=3 rows are NaN-padded in columns 3..5 and finite in 0..2; m=6 rows are full.
    for i in (0, 2):
        assert np.all(np.isnan(log.residual[i, 3:])) and np.all(np.isnan(log.sigma[i, 3:]))
        assert np.all(np.isfinite(log.residual[i, :3])) and np.all(np.isfinite(log.sigma[i, :3]))
    for i in (1, 3):
        assert np.all(np.isfinite(log.residual[i])) and np.all(np.isfinite(log.sigma[i]))
    np.testing.assert_allclose(log.residual[0, :3], [0.1, -0.2, 0.3])
    np.testing.assert_allclose(log.residual[2, :3], [1e3, 0.0, 0.0])
    # NIS of the rejected update is still recorded and is huge.
    assert log.nis[2] > gate3


def test_record_false_keeps_no_history():
    f = ekf.ErrorStateEKF(np.eye(2), record=False)
    res = f.update(np.array([1.0]), np.array([[1.0, 0.0]]), 1.0, name="x")
    f.update_sequential(np.array([1.0]), np.array([[1.0, 0.0]]), [1.0], name="x")
    f.update(np.array([1e3]), np.array([[1.0, 0.0]]), 1.0, name="x", gate=5.0)
    assert f.history == {}
    assert f.innovations("x").nis.size == 0
    assert res.accepted and res.name == "x"          # results are still returned


def test_update_result_fields_and_frozen():
    f = ekf.ErrorStateEKF(np.eye(2))
    res = f.update(np.array([2.0]), np.array([[1.0, 0.0]]), 1.0, name="m", t=7.5)
    assert res.name == "m" and res.t == 7.5 and res.accepted
    np.testing.assert_allclose(res.residual, [2.0])
    np.testing.assert_allclose(res.S, [[2.0]])
    assert res.nis == pytest.approx(2.0)             # r^2 / S = 4 / 2
    with pytest.raises(Exception):
        res.nis = 0.0                                 # frozen dataclass


def test_reset_applies_G_P_Gt():
    rng = np.random.default_rng(14)
    n = 4
    P0 = random_spd(rng, n)
    G = np.eye(n) + 0.2 * rng.standard_normal((n, n))
    f = ekf.ErrorStateEKF(P0)
    f.reset(G)
    np.testing.assert_allclose(f.P, G @ P0 @ G.T, atol=1e-12, rtol=1e-12)


def test_predict_formula_and_diagonal_Qd():
    rng = np.random.default_rng(15)
    n = 3
    P0 = random_spd(rng, n)
    Phi = np.eye(n) + 0.3 * rng.standard_normal((n, n))
    q = np.array([0.1, 0.2, 0.3])
    f = ekf.ErrorStateEKF(P0)
    f.predict(Phi, q)
    np.testing.assert_allclose(f.P, Phi @ P0 @ Phi.T + np.diag(q), atol=1e-12)


def test_std_is_sqrt_diag_P():
    rng = np.random.default_rng(16)
    P0 = random_spd(rng, 5)
    f = ekf.ErrorStateEKF(P0)
    np.testing.assert_allclose(f.std, np.sqrt(np.diag(P0)), rtol=1e-14)
    f.update(rng.standard_normal(2), rng.standard_normal((2, 5)), 0.5)
    np.testing.assert_allclose(f.std, np.sqrt(np.diag(f.P)), rtol=1e-14)


def test_P_returns_copy_and_P0_not_aliased():
    P0 = np.diag([1.0, 2.0])
    f = ekf.ErrorStateEKF(P0)
    P0[0, 0] = 99.0                                   # caller's array mutated
    got = f.P
    got[0, 0] = -5.0                                  # returned copy mutated
    got[0, 1] = 7.0
    np.testing.assert_array_equal(f.P, np.diag([1.0, 2.0]))
    assert f.std.tolist() == pytest.approx([1.0, np.sqrt(2.0)])


def test_set_P_replaces_covariance_and_copies():
    f = ekf.ErrorStateEKF(np.eye(2))
    newP = np.diag([3.0, 4.0])
    f.set_P(newP)
    newP[0, 0] = 0.0
    np.testing.assert_array_equal(f.P, np.diag([3.0, 4.0]))


def test_symmetrize_and_is_spd_helpers():
    A = np.array([[1.0, 2.0], [0.0, 3.0]])
    np.testing.assert_allclose(ekf.symmetrize(A), [[1.0, 1.0], [1.0, 3.0]])
    assert ekf.is_spd(np.eye(3))
    assert not ekf.is_spd(np.diag([1.0, 0.0]))
    assert not ekf.is_spd(np.diag([1.0, -1.0]))
    assert not ekf.is_spd(np.array([[1.0, 0.5], [0.0, 1.0]]))
    assert not ekf.is_spd(np.array([[1.0, np.nan], [np.nan, 1.0]]))
    assert not ekf.is_spd(np.ones((2, 3)))
