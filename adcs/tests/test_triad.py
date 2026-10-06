"""Tests for ``triad`` and ``triad_covariance`` in ``adcs.estimation``.

Written from ``docs/specs/adcs_estimation.md`` section 2 (tests 1-6), not from the
implementation.  Oracles: ``scipy.spatial.transform.Rotation`` (random attitudes, matrix
and angle algebra; scipy is scalar-last, the repo scalar-first body-to-reference), a
textbook TRIAD written independently in this file, and Monte Carlo statistics with fixed
seeds.

Monte Carlo thresholds: N = 20 000 samples; an element of the sample covariance has
standard error <= sqrt(2) * lambda_max / sqrt(N) = 0.7 % of lambda_max, so the spec band of
5 % of the largest eigenvalue is > 7 standard errors.  The mean-error check uses 4 standard
errors per component (std / sqrt(N)).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from adcs.estimation import triad, triad_covariance
from common.quaternion import error_rotvec


# --------------------------------------------------------------------------- helpers
def _to_sf(rot: Rotation) -> np.ndarray:
    q = np.roll(rot.as_quat(), 1, axis=-1)
    return np.where((q[..., :1] < 0), -q, q)


def _to_rot(q) -> Rotation:
    return Rotation.from_quat(np.asarray(q)[..., [1, 2, 3, 0]])


def _angle_err(q_est, rot_true: Rotation) -> np.ndarray:
    """Rotation angle [rad] between estimate and truth, computed entirely in scipy."""
    return (_to_rot(q_est).inv() * rot_true).magnitude()


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _pair(rng, n, amin_deg=10.0, amax_deg=170.0):
    """n unit-vector pairs with angle uniformly in [amin, amax] deg."""
    v1 = _unit(rng.normal(size=(n, 3)))
    u = rng.normal(size=(n, 3))
    u = _unit(u - np.einsum("ij,ij->i", u, v1)[:, None] * v1)
    a = np.radians(rng.uniform(amin_deg, amax_deg, n))
    v2 = np.cos(a)[:, None] * v1 + np.sin(a)[:, None] * u
    return v1, v2


def _triad_ref(v1b, v2b, v1r, v2r) -> Rotation:
    """Independent textbook TRIAD (single case) via scipy; v1 primary."""
    def frame(a, b):
        t1 = a / np.linalg.norm(a)
        t2 = np.cross(a, b)
        t2 /= np.linalg.norm(t2)
        t3 = np.cross(t1, t2)
        return np.column_stack([t1, t2, t3])

    r = frame(v1r, v2r) @ frame(v1b, v2b).T
    return Rotation.from_matrix(r)


def _noisy_body(rng, v, sigma, n):
    """n noisy copies of unit vector v: add perpendicular noise (std sigma per axis), renormalise."""
    a = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(v, a)
    u /= np.linalg.norm(u)
    w = np.cross(v, u)
    noise = sigma * (rng.normal(size=(n, 1)) * u + rng.normal(size=(n, 1)) * w)
    return _unit(v + noise)


# --------------------------------------------------------------------------- 1. noise-free
class TestNoiseFree:
    def setup_method(self):
        rng = np.random.default_rng(1)
        n = 500
        self.n = n
        self.rot = Rotation.random(n, rng)
        self.v1r, self.v2r = _pair(rng, n, 10.0, 170.0)
        # body vectors: v_b = R^T v_r
        self.v1b = self.rot.inv().apply(self.v1r)
        self.v2b = self.rot.inv().apply(self.v2r)
        self.rng = rng

    def test_recovers_true_attitude_batched(self):
        q = triad(self.v1b, self.v2b, self.v1r, self.v2r)
        assert q.shape == (self.n, 4)
        # exact algebra; worst-case conditioning at 10 deg is ~6, so 1e-9 is very conservative
        assert np.all(_angle_err(q, self.rot) < 1e-9)
        np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-14)
        assert np.all(q[:, 0] >= 0.0)  # canonical w >= 0

    def test_batched_equals_unbatched(self):
        qb = triad(self.v1b, self.v2b, self.v1r, self.v2r)
        for k in range(self.n):
            qk = triad(self.v1b[k], self.v2b[k], self.v1r[k], self.v2r[k])
            assert qk.shape == (4,)
            assert _angle_err(qk, _to_rot(qb[k])) < 1e-12

    def test_arbitrary_vector_lengths(self):
        rng = np.random.default_rng(2)
        sc = lambda: 10.0 ** rng.uniform(-6, 6, size=(self.n, 1))  # noqa: E731  (T-scale ... 1e6)
        q_ref = triad(self.v1b, self.v2b, self.v1r, self.v2r)
        q_sc = triad(sc() * self.v1b, sc() * self.v2b, sc() * self.v1r, sc() * self.v2r)
        assert np.all(_angle_err(q_sc, _to_rot(q_ref)) < 1e-9)
        assert np.all(_angle_err(q_sc, self.rot) < 1e-9)
        # magnetometer-like magnitude (3e-5 T) and a unit sun vector
        q_mag = triad(3e-5 * self.v1b, self.v2b, 4.5e-5 * self.v1r, self.v2r)
        assert np.all(_angle_err(q_mag, self.rot) < 1e-9)

    def test_matches_independent_textbook_triad(self):
        q = triad(self.v1b, self.v2b, self.v1r, self.v2r)
        for k in range(0, self.n, 25):
            ref = _triad_ref(self.v1b[k], self.v2b[k], self.v1r[k], self.v2r[k])
            assert _angle_err(q[k], ref) < 1e-12

    def test_multi_dimensional_batch_and_array_likes(self):
        shp = (5, 4)
        sel = slice(0, 20)
        q = triad(
            self.v1b[sel].reshape(*shp, 3),
            self.v2b[sel].reshape(*shp, 3),
            self.v1r[sel].reshape(*shp, 3),
            self.v2r[sel].reshape(*shp, 3),
        )
        assert q.shape == (5, 4, 4)
        flat = _to_rot(q.reshape(20, 4))
        assert np.all((flat.inv() * self.rot[sel]).magnitude() < 1e-9)
        # plain lists are accepted
        ql = triad(self.v1b[0].tolist(), self.v2b[0].tolist(), self.v1r[0].tolist(), self.v2r[0].tolist())
        assert isinstance(ql, np.ndarray) and ql.shape == (4,)

    def test_inputs_not_mutated(self):
        args = [a.copy() for a in (self.v1b, self.v2b, self.v1r, self.v2r)]
        triad(*args)
        for a, b in zip(args, (self.v1b, self.v2b, self.v1r, self.v2r)):
            np.testing.assert_array_equal(a, b)

    def test_identity_and_known_rotation(self):
        # identity: body == reference
        q = triad([1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0])
        np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-15)
        # 90 deg about z (body-to-reference): body x -> ref y, body y -> ref -x
        q = triad([1, 0, 0], [0, 1, 0], [0, 1, 0], [-1, 0, 0])
        np.testing.assert_allclose(q, [math.sqrt(0.5), 0, 0, math.sqrt(0.5)], atol=1e-14)
        # 180 deg about x: w = 0 -> unit norm output still returned, rotation correct
        q = triad([1, 0, 0], [0, 1, 0], [1, 0, 0], [0, -1, 0])
        assert _angle_err(q, Rotation.from_rotvec([math.pi, 0, 0])) < 1e-12


# --------------------------------------------------------------------------- 2. primary exactness
class TestPrimaryExactness:
    def setup_method(self):
        rng = np.random.default_rng(3)
        n = 300
        rot = Rotation.random(n, rng)
        v1r, v2r = _pair(rng, n, 20.0, 160.0)
        v1b = rot.inv().apply(v1r)
        v2b = rot.inv().apply(v2r)
        # noise 1e-2 rad (per axis, perpendicular) on both BODY vectors
        self.v1b_n = _unit(v1b + 1e-2 * rng.normal(size=(n, 3)))
        self.v2b_n = _unit(v2b + 1e-2 * rng.normal(size=(n, 3)))
        self.v1r, self.v2r = v1r, v2r
        self.q = triad(self.v1b_n, self.v2b_n, v1r, v2r)

    def test_primary_vector_matched_exactly(self):
        got = _to_rot(self.q).apply(self.v1b_n)
        np.testing.assert_allclose(got, self.v1r, rtol=0, atol=1e-12)

    def test_secondary_generally_not_matched(self):
        got = _to_rot(self.q).apply(self.v2b_n)
        miss = np.linalg.norm(got - self.v2r, axis=1)
        # noise 1e-2 rad: residual typically ~1e-2, never all below 1e-4
        assert np.median(miss) > 1e-3
        assert np.mean(miss > 1e-4) > 0.95

    def test_plane_normal_matched(self):
        # R t2_b = t2_r with t2 = v1 x v2 / |v1 x v2| (spec definition of the triad)
        nb = _unit(np.cross(self.v1b_n, self.v2b_n))
        nr = _unit(np.cross(self.v1r, self.v2r))
        np.testing.assert_allclose(_to_rot(self.q).apply(nb), nr, rtol=0, atol=1e-12)

    def test_matches_independent_textbook_triad_with_noise(self):
        for k in range(0, 300, 10):
            ref = _triad_ref(self.v1b_n[k], self.v2b_n[k], self.v1r[k], self.v2r[k])
            assert _angle_err(self.q[k], ref) < 1e-12


# --------------------------------------------------------------------------- 3. swap
class TestSwap:
    def test_swap_irrelevant_when_noise_free(self):
        rng = np.random.default_rng(4)
        n = 200
        rot = Rotation.random(n, rng)
        v1r, v2r = _pair(rng, n, 15.0, 165.0)
        v1b, v2b = rot.inv().apply(v1r), rot.inv().apply(v2r)
        qa = triad(v1b, v2b, v1r, v2r)
        qb = triad(v2b, v1b, v2r, v1r)
        assert np.all(_angle_err(qa, _to_rot(qb)) < 1e-9)
        assert np.all(_angle_err(qb, rot) < 1e-9)

    def test_swap_changes_result_with_noise(self):
        rng = np.random.default_rng(5)
        n = 200
        rot = Rotation.random(n, rng)
        v1r, v2r = _pair(rng, n, 30.0, 150.0)
        v1b = _unit(rot.inv().apply(v1r) + 1e-2 * rng.normal(size=(n, 3)))
        v2b = _unit(rot.inv().apply(v2r) + 1e-2 * rng.normal(size=(n, 3)))
        qa = triad(v1b, v2b, v1r, v2r)
        qb = triad(v2b, v1b, v2r, v1r)
        diff = _angle_err(qa, _to_rot(qb))
        assert np.median(diff) > 1e-3  # ~1e-2 rad noise -> mm-rad scale differences
        assert np.all(diff > 1e-6)
        # swapped call equals the independent reference with v2 as primary
        for k in range(0, n, 20):
            ref = _triad_ref(v2b[k], v1b[k], v2r[k], v1r[k])
            assert _angle_err(qb[k], ref) < 1e-12

    def test_difference_scales_with_noise(self):
        rng = np.random.default_rng(6)
        n = 300
        rot = Rotation.random(n, rng)
        v1r, v2r = _pair(rng, n, 40.0, 140.0)
        v1c, v2c = rot.inv().apply(v1r), rot.inv().apply(v2r)
        e1, e2 = rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
        med = []
        for s in (1e-3, 1e-2):
            v1b, v2b = _unit(v1c + s * e1), _unit(v2c + s * e2)
            d = _angle_err(triad(v1b, v2b, v1r, v2r), _to_rot(triad(v2b, v1b, v2r, v1r)))
            med.append(np.median(d))
        assert med[1] / med[0] == pytest.approx(10.0, rel=0.1)  # linear in noise (small-angle)


# --------------------------------------------------------------------------- 4. degenerate
class TestDegenerate:
    X = np.array([1.0, 0.0, 0.0])
    Y = np.array([0.0, 1.0, 0.0])

    def test_parallel_and_antiparallel_raise(self):
        for v2 in (self.X, -self.X, 7.0 * self.X, -1e-5 * self.X):
            with pytest.raises(ValueError):
                triad(self.X, v2, self.X, self.Y)  # degenerate in the body frame
            with pytest.raises(ValueError):
                triad(self.X, self.Y, self.X, v2)  # degenerate in the reference frame
            with pytest.raises(ValueError):
                triad(self.X, v2, self.X, v2)  # both frames

    def test_threshold_one_micro(self):
        def vec(sin_a):
            return np.array([math.sqrt(1 - sin_a**2), sin_a, 0.0])

        with pytest.raises(ValueError):
            triad(self.X, vec(5e-7), self.X, self.Y)
        with pytest.raises(ValueError):
            triad(self.X, self.Y, self.X, vec(5e-7))
        # just above 1e-6: accepted and finite
        q = triad(self.X, vec(5e-6), self.X, vec(5e-6))
        assert np.all(np.isfinite(q))

    def test_batch_with_one_degenerate_element_raises(self):
        rng = np.random.default_rng(7)
        v1, v2 = _pair(rng, 10, 20.0, 160.0)
        v2b = v2.copy()
        v2b[6] = 2.5 * v1[6]
        with pytest.raises(ValueError):
            triad(v1, v2b, v1, v2)
        with pytest.raises(ValueError):
            triad(v1, v2, v1, v2b)
        # unchanged batch is fine
        assert triad(v1, v2, v1, v2).shape == (10, 4)


# --------------------------------------------------------------------------- covariance
def _geometry(alpha_deg, seed):
    """True attitude + noise-free unit vector pair at angle alpha; returns body/ref vectors."""
    rng = np.random.default_rng(seed)
    rot = Rotation.random(1, rng)[0]
    v1r = _unit(rng.normal(size=3))
    u = rng.normal(size=3)
    u = _unit(u - u @ v1r * v1r)
    a = math.radians(alpha_deg)
    v2r = math.cos(a) * v1r + math.sin(a) * u
    return rot, rot.inv().apply(v1r), rot.inv().apply(v2r), v1r, v2r


def _frame_e(v1b, v2b):
    """e1, e2, e3 basis defined in the spec, body frame."""
    ca = float(v1b @ v2b)
    sa = math.sqrt(1 - ca**2)
    e1 = v1b
    e2 = np.cross(v1b, v2b) / sa
    e3 = np.cross(e1, e2)
    return e1, e2, e3, ca, sa


def _mc_dtheta(alpha_deg, s1, s2, n=20_000, seed=0):
    rot, v1b, v2b, v1r, v2r = _geometry(alpha_deg, seed)
    rng = np.random.default_rng(1000 + seed)
    v1n = _noisy_body(rng, v1b, s1, n)
    v2n = _noisy_body(rng, v2b, s2, n)
    q = triad(v1n, v2n, np.tile(v1r, (n, 1)), np.tile(v2r, (n, 1)))
    q_true = np.tile(np.roll(rot.as_quat(), 1), (n, 1))
    # q_true = q_triad (x) exp(dtheta)  ->  dtheta = error_rotvec(q_triad, q_true)
    dth = error_rotvec(q, q_true)
    return dth, v1b, v2b


GEOMS = [(a, s1, s2) for a in (20.0, 60.0, 90.0, 150.0) for (s1, s2) in ((1e-3, 3e-3), (3e-3, 1e-3))]


class TestCovarianceFormula:
    def test_shape_symmetry_psd(self):
        for a in (20.0, 60.0, 90.0, 150.0):
            _, v1b, v2b, _, _ = _geometry(a, 11)
            p = triad_covariance(v1b, v2b, 1e-3, 3e-3)
            assert p.shape == (3, 3)
            np.testing.assert_allclose(p, p.T, rtol=0, atol=1e-18)
            assert np.linalg.eigvalsh(p).min() > 0.0

    def test_right_angle_closed_form(self):
        # cos(alpha) = 0: P = s1^2 (I - e1 e1^T) + s2^2 e1 e1^T   (hand evaluation of the spec formula)
        _, v1b, v2b, _, _ = _geometry(90.0, 12)
        e1, _, _, ca, _ = _frame_e(v1b, v2b)
        assert abs(ca) < 1e-12
        s1, s2 = 1e-3, 3e-3
        p = triad_covariance(v1b, v2b, s1, s2)
        expected = s1**2 * (np.eye(3) - np.outer(e1, e1)) + s2**2 * np.outer(e1, e1)
        np.testing.assert_allclose(p, expected, rtol=0, atol=1e-15)

    def test_input_length_invariance(self):
        _, v1b, v2b, _, _ = _geometry(60.0, 13)
        p = triad_covariance(v1b, v2b, 1e-3, 3e-3)
        p2 = triad_covariance(3e-5 * v1b, 5.0 * v2b, 1e-3, 3e-3)
        np.testing.assert_allclose(p2, p, rtol=0, atol=1e-18)

    def test_rotation_covariance(self):
        # body-frame covariance rotates as P' = Q P Q^T when the body axes are rotated by Q
        _, v1b, v2b, _, _ = _geometry(60.0, 14)
        qm = Rotation.random(1, np.random.default_rng(15))[0].as_matrix()
        p = triad_covariance(v1b, v2b, 1e-3, 3e-3)
        p_rot = triad_covariance(qm @ v1b, qm @ v2b, 1e-3, 3e-3)
        np.testing.assert_allclose(p_rot, qm @ p @ qm.T, rtol=0, atol=1e-15)

    def test_secondary_noise_only_enters_e1e1_term(self):
        # sigma2 -> large inflates only the e1e1 term (rotation about the primary axis)
        _, v1b, v2b, _, _ = _geometry(60.0, 16)
        e1 = v1b
        p_lo = triad_covariance(v1b, v2b, 1e-3, 1e-3)
        p_hi = triad_covariance(v1b, v2b, 1e-3, 1e-2)
        d = p_hi - p_lo
        sa2 = 1 - float(v1b @ v2b) ** 2
        np.testing.assert_allclose(d, (1e-4 - 1e-6) / sa2 * np.outer(e1, e1), rtol=1e-9, atol=1e-15)


class TestCovarianceMonteCarlo:
    @pytest.mark.parametrize("alpha,s1,s2", GEOMS)
    def test_matches_triad_covariance(self, alpha, s1, s2):
        dth, v1b, v2b = _mc_dtheta(alpha, s1, s2, seed=int(alpha))
        n = dth.shape[0]
        p_model = triad_covariance(v1b, v2b, s1, s2)
        p_mc = np.cov(dth.T, ddof=1)
        lam_max = np.linalg.eigvalsh(p_model).max()
        # spec: element-wise within 5 % of the largest eigenvalue
        assert np.max(np.abs(p_mc - p_model)) < 0.05 * lam_max
        # mean error below 4 standard errors, per component
        se = dth.std(axis=0, ddof=1) / math.sqrt(n)
        assert np.all(np.abs(dth.mean(axis=0)) < 4.0 * se)

    @pytest.mark.parametrize("alpha,s1,s2", [(60.0, 3e-3, 1e-3), (150.0, 3e-3, 1e-3)])
    def test_covariance_agrees_with_stronger_relative_tolerance_on_diagonal(self, alpha, s1, s2):
        # sanity beyond the spec: diagonal entries (in the e-basis) agree to 3 % of themselves
        dth, v1b, v2b = _mc_dtheta(alpha, s1, s2, seed=int(alpha) + 7)
        e1, e2, e3, _, _ = _frame_e(v1b, v2b)
        basis = np.column_stack([e1, e2, e3])
        p_model = basis.T @ triad_covariance(v1b, v2b, s1, s2) @ basis
        p_mc = np.cov((dth @ basis).T, ddof=1)
        # std error of a variance estimate is sqrt(2/N) = 1.0 %: 3 % is 3 sigma
        np.testing.assert_allclose(np.diag(p_mc), np.diag(p_model), rtol=0.04)


class TestCovarianceSign:
    @pytest.mark.parametrize("alpha", [60.0, 150.0])
    @pytest.mark.parametrize("s1,s2", [(3e-3, 1e-3), (1e-3, 1e-3)])
    def test_e1_e3_correlation_has_sign_of_minus_cos_alpha(self, alpha, s1, s2):
        dth, v1b, v2b = _mc_dtheta(alpha, s1, s2, seed=int(alpha) + 31)
        e1, _, e3, ca, _ = _frame_e(v1b, v2b)
        c1, c3 = dth @ e1, dth @ e3
        corr = np.corrcoef(c1, c3)[0, 1]
        assert np.sign(corr) == np.sign(-ca)
        # magnitude matches the model correlation (std error of a correlation ~ 1/sqrt(N) = 0.7 %)
        p = triad_covariance(v1b, v2b, s1, s2)
        model = (e1 @ p @ e3) / math.sqrt((e1 @ p @ e1) * (e3 @ p @ e3))
        assert np.sign(model) == np.sign(-ca)
        assert abs(corr) > 0.2
        assert corr == pytest.approx(model, abs=0.03)

    def test_uncorrelated_at_right_angle(self):
        dth, v1b, v2b = _mc_dtheta(90.0, 3e-3, 1e-3, seed=41)
        e1, _, e3, ca, _ = _frame_e(v1b, v2b)
        corr = np.corrcoef(dth @ e1, dth @ e3)[0, 1]
        assert abs(corr) < 0.03  # 4 standard errors at N = 20000
