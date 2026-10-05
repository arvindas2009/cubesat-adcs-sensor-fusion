"""Tests for ``common.quaternion``.

Written from ``docs/specs/common_api.md`` (common/quaternion.py section) and
``CONVENTIONS.md`` sections 2-3, *not* from the implementation.

Oracles
-------
* ``scipy.spatial.transform.Rotation`` (scalar-first quaternions via
  ``scalar_first=True``) for DCM, rotate, rotvec, Euler ZYX, slerp, composition.
* Hand-derived values (90/180 degree rotations about the axes, i*j = k, ...).
* Analytic solutions (constant body rate => q (x) exp(w dt)).

Convention reminders: scalar-first ``[w, x, y, z]``, Hamilton product,
``v_ref = R(q) v_body``, ``q_AC = q_AB (x) q_BC``, ``q_dot = 1/2 q (x) [0, w_b]``.

Tolerances: ``ATOL = 1e-12`` for exact algebra on unit quaternions (typical
errors are ~1e-16; 1e-12 leaves room for accumulation over products/batches).
Anything looser carries its own justification in a comment.
"""

import warnings

import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from common import quaternion as Q

ATOL = 1e-12
S = np.sqrt(0.5)  # cos(45 deg) = sin(45 deg)

X_HAT = np.array([1.0, 0.0, 0.0])
Y_HAT = np.array([0.0, 1.0, 0.0])
Z_HAT = np.array([0.0, 0.0, 1.0])

# Hand-built quaternions [cos(a/2), sin(a/2) * axis] for +90 deg about each axis.
Q90X = np.array([S, S, 0.0, 0.0])
Q90Y = np.array([S, 0.0, S, 0.0])
Q90Z = np.array([S, 0.0, 0.0, S])
Q180X = np.array([0.0, 1.0, 0.0, 0.0])
Q180Y = np.array([0.0, 0.0, 1.0, 0.0])
Q180Z = np.array([0.0, 0.0, 0.0, 1.0])
IDENT = np.array([1.0, 0.0, 0.0, 0.0])


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def sci(q):
    """scipy Rotation from scalar-first quaternion(s)."""
    return Rotation.from_quat(np.asarray(q, dtype=float), scalar_first=True)


def sci_quat(rot):
    return rot.as_quat(scalar_first=True)


def rand_quats(rng, n):
    """Random unit quaternions (NOT canonicalised: w may be negative)."""
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def rand_unit_vecs(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def sign_invariant_diff(a, b):
    """Per-sample max-abs difference between quaternions a and b modulo sign."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d1 = np.max(np.abs(a - b), axis=-1)
    d2 = np.max(np.abs(a + b), axis=-1)
    return np.minimum(d1, d2)


def assert_same_rotation(a, b, atol=ATOL):
    diff = sign_invariant_diff(a, b)
    assert np.max(diff) <= atol, f"max quaternion diff (mod sign) = {np.max(diff):.3e}"


def wrap(d):
    """Wrap angle differences to (-pi, pi]."""
    return np.angle(np.exp(1j * np.asarray(d)))


class strict:
    """No floating-point warnings/errors allowed inside the block."""

    def __enter__(self):
        self._err = np.errstate(divide="raise", invalid="raise", over="raise")
        self._err.__enter__()
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("error")

    def __exit__(self, *exc):
        self._w.__exit__(*exc)
        self._err.__exit__(*exc)
        return False


def rot_about(axis, angle):
    """Reference quaternion built by the textbook formula (independent of Q)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


# ==========================================================================
# identity / basic constructors
# ==========================================================================
class TestIdentityAndBasics:
    def test_identity_value_shape(self):
        q = Q.identity()
        assert q.shape == (4,)
        np.testing.assert_array_equal(q, IDENT)

    def test_identity_returns_fresh_array(self):
        a = Q.identity()
        a[0] = 99.0
        np.testing.assert_array_equal(Q.identity(), IDENT)

    def test_identity_is_no_rotation(self):
        v = np.array([0.3, -1.2, 2.5])
        np.testing.assert_allclose(Q.rotate(Q.identity(), v), v, atol=ATOL)
        np.testing.assert_allclose(Q.to_dcm(Q.identity()), np.eye(3), atol=ATOL)

    def test_conjugate(self):
        q = np.array([0.1, 0.2, -0.3, 0.4])
        np.testing.assert_array_equal(Q.conjugate(q), [0.1, -0.2, 0.3, -0.4])

    def test_norm_values_and_shape(self):
        assert np.shape(Q.norm(np.array([1.0, 2.0, 2.0, 4.0]))) == ()
        np.testing.assert_allclose(Q.norm(np.array([1.0, 2.0, 2.0, 4.0])), 5.0, atol=ATOL)
        rng = np.random.default_rng(1)
        q = rng.normal(size=(7, 4))
        n = Q.norm(q)
        assert n.shape == (7,)
        np.testing.assert_allclose(n, np.linalg.norm(q, axis=-1), atol=ATOL)

    def test_inverse_of_non_unit(self):
        q = np.array([1.0, 2.0, -1.0, 0.5])
        inv = Q.inverse(q)
        np.testing.assert_allclose(inv, np.array([1.0, -2.0, 1.0, -0.5]) / np.dot(q, q), atol=ATOL)
        np.testing.assert_allclose(Q.multiply(q, inv), IDENT, atol=ATOL)
        np.testing.assert_allclose(Q.multiply(inv, q), IDENT, atol=ATOL)

    def test_inverse_equals_conjugate_for_unit(self):
        q = rand_quats(np.random.default_rng(2), 100)
        np.testing.assert_allclose(Q.inverse(q), Q.conjugate(q), atol=ATOL)

    def test_array_like_inputs_and_no_mutation(self):
        p = [1.0, 0.0, 0.0, 0.0]
        q = [0.0, 1.0, 0.0, 0.0]
        out = Q.multiply(p, q)
        assert isinstance(out, np.ndarray) and out.dtype == float
        assert p == [1.0, 0.0, 0.0, 0.0] and q == [0.0, 1.0, 0.0, 0.0]

        a = np.array([0.3, -0.5, 0.7, 0.1])
        v = np.array([1.0, 2.0, 3.0])
        a0, v0 = a.copy(), v.copy()
        for f in (
            lambda: Q.normalize(a),
            lambda: Q.canonicalize(a),
            lambda: Q.conjugate(a),
            lambda: Q.inverse(a),
            lambda: Q.to_dcm(a),
            lambda: Q.rotate(a, v),
            lambda: Q.rotate_inverse(a, v),
            lambda: Q.to_rotvec(a),
            lambda: Q.to_euler_zyx(a),
            lambda: Q.derivative(a, v),
            lambda: Q.propagate(a, v, 0.1),
            lambda: Q.multiply(a, a),
            lambda: Q.slerp(a, -a, 0.3),
            lambda: Q.error_rotvec(a, a),
            lambda: Q.angle_between(a, a),
            lambda: Q.from_rotvec(v),
            lambda: Q.skew(v),
        ):
            f()
        np.testing.assert_array_equal(a, a0)
        np.testing.assert_array_equal(v, v0)


# ==========================================================================
# skew
# ==========================================================================
class TestSkew:
    def test_hand_value(self):
        np.testing.assert_array_equal(
            Q.skew([1.0, 2.0, 3.0]),
            [[0.0, -3.0, 2.0], [3.0, 0.0, -1.0], [-2.0, 1.0, 0.0]],
        )

    def test_unit_axes_cross_products(self):
        # x cross y = z, y cross z = x, z cross x = y (right-handed)
        np.testing.assert_allclose(Q.skew(X_HAT) @ Y_HAT, Z_HAT, atol=ATOL)
        np.testing.assert_allclose(Q.skew(Y_HAT) @ Z_HAT, X_HAT, atol=ATOL)
        np.testing.assert_allclose(Q.skew(Z_HAT) @ X_HAT, Y_HAT, atol=ATOL)

    def test_equals_cross_random_batched(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(1000, 3))
        b = rng.normal(size=(1000, 3))
        S_ = Q.skew(a)
        assert S_.shape == (1000, 3, 3)
        got = np.einsum("nij,nj->ni", S_, b)
        np.testing.assert_allclose(got, np.cross(a, b), atol=ATOL)

    def test_antisymmetric(self):
        a = np.random.default_rng(4).normal(size=(50, 3))
        S_ = Q.skew(a)
        np.testing.assert_allclose(S_, -np.swapaxes(S_, -1, -2), atol=0)


# ==========================================================================
# normalize / canonicalize
# ==========================================================================
class TestNormalizeCanonicalize:
    def test_normalize_unit_norm(self):
        rng = np.random.default_rng(5)
        q = rng.normal(size=(1000, 4)) * rng.uniform(0.01, 100, size=(1000, 1))
        n = Q.normalize(q)
        np.testing.assert_allclose(np.linalg.norm(n, axis=-1), 1.0, atol=ATOL)
        np.testing.assert_allclose(n, q / np.linalg.norm(q, axis=-1, keepdims=True), atol=ATOL)

    def test_normalize_zero_raises(self):
        with pytest.raises(ValueError):
            Q.normalize(np.zeros(4))

    def test_normalize_tiny_norm_raises(self):
        # spec: raises if any norm < 1e-12
        with pytest.raises(ValueError):
            Q.normalize(np.array([1e-13, 0.0, 0.0, 0.0]))

    def test_normalize_batch_with_one_zero_row_raises(self):
        q = np.tile(IDENT, (5, 1))
        q[3] = 0.0
        with pytest.raises(ValueError):
            Q.normalize(q)

    def test_normalize_small_but_valid_ok(self):
        out = Q.normalize(np.array([0.0, 1e-6, 0.0, 0.0]))
        np.testing.assert_allclose(out, [0.0, 1.0, 0.0, 0.0], atol=ATOL)

    def test_canonicalize_basic(self):
        np.testing.assert_array_equal(
            Q.canonicalize(np.array([-0.5, 0.5, -0.5, 0.5])), [0.5, -0.5, 0.5, -0.5]
        )
        np.testing.assert_array_equal(
            Q.canonicalize(np.array([0.5, 0.5, -0.5, 0.5])), [0.5, 0.5, -0.5, 0.5]
        )

    @pytest.mark.parametrize(
        "q, expected",
        [
            ([0.0, -1.0, 2.0, 3.0], [0.0, 1.0, -2.0, -3.0]),  # first non-zero is x < 0 -> flip
            ([0.0, 1.0, -2.0, 3.0], [0.0, 1.0, -2.0, 3.0]),  # x > 0 -> keep
            ([0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 1.0, 0.0]),  # first non-zero is y < 0 -> flip
            ([0.0, 0.0, 1.0, -3.0], [0.0, 0.0, 1.0, -3.0]),
            ([0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 1.0]),  # first non-zero is z < 0 -> flip
            ([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]),
        ],
    )
    def test_canonicalize_w_zero_rule(self, q, expected):
        np.testing.assert_array_equal(Q.canonicalize(np.array(q)), expected)

    def test_canonicalize_batched_random(self):
        q = rand_quats(np.random.default_rng(6), 1000)
        c = Q.canonicalize(q)
        assert c.shape == q.shape
        assert np.all(c[:, 0] >= 0)
        # same rotation, sign only
        np.testing.assert_allclose(np.abs(c), np.abs(q), atol=0)
        assert np.all(sign_invariant_diff(c, q) == 0)
        # idempotent
        np.testing.assert_array_equal(Q.canonicalize(c), c)
        # q and -q canonicalise to the same representative
        np.testing.assert_allclose(Q.canonicalize(-q), c, atol=0)


# ==========================================================================
# multiply
# ==========================================================================
class TestMultiply:
    def test_ijk_hamilton_rules(self):
        i = np.array([0.0, 1.0, 0.0, 0.0])
        j = np.array([0.0, 0.0, 1.0, 0.0])
        k = np.array([0.0, 0.0, 0.0, 1.0])
        minus_one = np.array([-1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(Q.multiply(i, j), k, atol=ATOL)
        np.testing.assert_allclose(Q.multiply(j, k), i, atol=ATOL)
        np.testing.assert_allclose(Q.multiply(k, i), j, atol=ATOL)
        np.testing.assert_allclose(Q.multiply(j, i), -k, atol=ATOL)
        np.testing.assert_allclose(Q.multiply(k, j), -i, atol=ATOL)
        np.testing.assert_allclose(Q.multiply(i, k), -j, atol=ATOL)
        for e in (i, j, k):
            np.testing.assert_allclose(Q.multiply(e, e), minus_one, atol=ATOL)
        np.testing.assert_allclose(Q.multiply(Q.multiply(i, j), k), minus_one, atol=ATOL)

    def test_hand_general_product(self):
        # (1 + 2i + 3j + 4k)(5 + 6i + 7j + 8k) = -60 + 12i + 30j + 24k (Hamilton)
        p = np.array([1.0, 2.0, 3.0, 4.0])
        q = np.array([5.0, 6.0, 7.0, 8.0])
        np.testing.assert_allclose(Q.multiply(p, q), [-60.0, 12.0, 30.0, 24.0], atol=ATOL)

    def test_identity_both_sides(self):
        q = rand_quats(np.random.default_rng(7), 100)
        np.testing.assert_allclose(Q.multiply(IDENT, q), q, atol=ATOL)
        np.testing.assert_allclose(Q.multiply(q, IDENT), q, atol=ATOL)

    def test_q_times_conjugate_is_identity(self):
        q = rand_quats(np.random.default_rng(8), 1000)
        expected = np.tile(IDENT, (1000, 1))
        np.testing.assert_allclose(Q.multiply(q, Q.conjugate(q)), expected, atol=ATOL)
        np.testing.assert_allclose(Q.multiply(Q.conjugate(q), q), expected, atol=ATOL)

    def test_associativity(self):
        rng = np.random.default_rng(9)
        a, b, c = (rand_quats(rng, 1000) for _ in range(3))
        lhs = Q.multiply(Q.multiply(a, b), c)
        rhs = Q.multiply(a, Q.multiply(b, c))
        np.testing.assert_allclose(lhs, rhs, atol=ATOL)

    def test_not_commutative(self):
        assert not np.allclose(Q.multiply(Q90X, Q90Y), Q.multiply(Q90Y, Q90X))

    def test_norm_multiplicative(self):
        rng = np.random.default_rng(10)
        p = rng.normal(size=(500, 4))
        q = rng.normal(size=(500, 4))
        np.testing.assert_allclose(
            np.linalg.norm(Q.multiply(p, q), axis=-1),
            np.linalg.norm(p, axis=-1) * np.linalg.norm(q, axis=-1),
            rtol=1e-12,
        )

    def test_matches_scipy_composition(self):
        # R(p (x) q) = R(p) R(q)
        rng = np.random.default_rng(11)
        p = rand_quats(rng, 1000)
        q = rand_quats(rng, 1000)
        pq = Q.multiply(p, q)
        assert_same_rotation(pq, sci_quat(sci(p) * sci(q)))
        np.testing.assert_allclose(
            Q.to_dcm(pq), sci(p).as_matrix() @ sci(q).as_matrix(), atol=ATOL
        )

    def test_composition_order_hand(self):
        # q_AC = q_AB (x) q_BC.  Apply 90 deg about z (inner, body) then ... :
        # R(Q90Z (x) Q90X) v = R(Q90Z) R(Q90X) v.  x -> (Rx) x -> (Rz) y?? no:
        # R(Q90X) y = z, R(Q90Z) z = z  => y maps to z.
        q = Q.multiply(Q90Z, Q90X)
        np.testing.assert_allclose(Q.rotate(q, Y_HAT), Z_HAT, atol=ATOL)
        # and x: R(Q90X) x = x, R(Q90Z) x = y  => x maps to y
        np.testing.assert_allclose(Q.rotate(q, X_HAT), Y_HAT, atol=ATOL)

    @pytest.mark.parametrize(
        "shape_p, shape_q, shape_out",
        [
            ((10, 4), (10, 4), (10, 4)),
            ((10, 4), (4,), (10, 4)),
            ((4,), (10, 4), (10, 4)),
            ((2, 3, 4), (2, 3, 4), (2, 3, 4)),
            ((2, 1, 4), (1, 3, 4), (2, 3, 4)),
            ((4,), (4,), (4,)),
        ],
    )
    def test_batched_shapes(self, shape_p, shape_q, shape_out):
        rng = np.random.default_rng(12)
        p = rng.normal(size=shape_p)
        q = rng.normal(size=shape_q)
        out = Q.multiply(p, q)
        assert out.shape == shape_out
        # element-wise agreement with the single-item product
        pb, qb = np.broadcast_arrays(p, q)
        pf, qf = pb.reshape(-1, 4), qb.reshape(-1, 4)
        ref = np.stack([Q.multiply(a, b) for a, b in zip(pf, qf)]).reshape(shape_out)
        np.testing.assert_allclose(out, ref, atol=ATOL)


# ==========================================================================
# to_dcm / rotate / rotate_inverse
# ==========================================================================
class TestDcmAndRotate:
    def test_to_dcm_hand_90_degree(self):
        np.testing.assert_allclose(
            Q.to_dcm(Q90X), [[1, 0, 0], [0, 0, -1], [0, 1, 0]], atol=ATOL
        )
        np.testing.assert_allclose(
            Q.to_dcm(Q90Y), [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], atol=ATOL
        )
        np.testing.assert_allclose(
            Q.to_dcm(Q90Z), [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=ATOL
        )

    def test_to_dcm_hand_180_degree(self):
        np.testing.assert_allclose(Q.to_dcm(Q180X), np.diag([1.0, -1.0, -1.0]), atol=ATOL)
        np.testing.assert_allclose(Q.to_dcm(Q180Y), np.diag([-1.0, 1.0, -1.0]), atol=ATOL)
        np.testing.assert_allclose(Q.to_dcm(Q180Z), np.diag([-1.0, -1.0, 1.0]), atol=ATOL)

    def test_to_dcm_matches_scipy(self):
        q = rand_quats(np.random.default_rng(13), 1000)
        np.testing.assert_allclose(Q.to_dcm(q), sci(q).as_matrix(), atol=ATOL)

    def test_to_dcm_batched_shape(self):
        q = rand_quats(np.random.default_rng(14), 6).reshape(2, 3, 4)
        assert Q.to_dcm(q).shape == (2, 3, 3, 3)
        assert Q.to_dcm(q[0, 0]).shape == (3, 3)

    def test_to_dcm_orthonormal_det_plus_one(self):
        q = rand_quats(np.random.default_rng(15), 1000)
        R = Q.to_dcm(q)
        eye = np.broadcast_to(np.eye(3), R.shape)
        np.testing.assert_allclose(R @ np.swapaxes(R, -1, -2), eye, atol=ATOL)
        np.testing.assert_allclose(np.swapaxes(R, -1, -2) @ R, eye, atol=ATOL)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=ATOL)

    def test_to_dcm_sign_invariant(self):
        q = rand_quats(np.random.default_rng(16), 200)
        np.testing.assert_allclose(Q.to_dcm(q), Q.to_dcm(-q), atol=ATOL)

    def test_to_dcm_normalises_input(self):
        rng = np.random.default_rng(17)
        q = rand_quats(rng, 100)
        scale = rng.uniform(0.1, 10.0, size=(100, 1))
        np.testing.assert_allclose(Q.to_dcm(q * scale), Q.to_dcm(q), atol=ATOL)

    @pytest.mark.parametrize(
        "q, v, expected",
        [
            # +90 deg about z: x->y, y->-x, z->z
            (Q90Z, X_HAT, Y_HAT),
            (Q90Z, Y_HAT, -X_HAT),
            (Q90Z, Z_HAT, Z_HAT),
            # +90 deg about x: y->z, z->-y, x->x
            (Q90X, Y_HAT, Z_HAT),
            (Q90X, Z_HAT, -Y_HAT),
            (Q90X, X_HAT, X_HAT),
            # +90 deg about y: z->x, x->-z, y->y
            (Q90Y, Z_HAT, X_HAT),
            (Q90Y, X_HAT, -Z_HAT),
            (Q90Y, Y_HAT, Y_HAT),
            # 180 deg
            (Q180Z, X_HAT, -X_HAT),
            (Q180Z, Y_HAT, -Y_HAT),
            (Q180Z, Z_HAT, Z_HAT),
            (Q180X, Y_HAT, -Y_HAT),
            (Q180X, Z_HAT, -Z_HAT),
            (Q180X, X_HAT, X_HAT),
            (Q180Y, X_HAT, -X_HAT),
            (Q180Y, Z_HAT, -Z_HAT),
            (Q180Y, Y_HAT, Y_HAT),
            # identity
            (IDENT, X_HAT, X_HAT),
        ],
    )
    def test_rotate_hand_derived(self, q, v, expected):
        np.testing.assert_allclose(Q.rotate(q, v), expected, atol=ATOL)

    def test_rotate_hand_derived_non_axis_vector(self):
        # +90 deg about z maps (1, 2, 3) -> (-2, 1, 3)
        np.testing.assert_allclose(
            Q.rotate(Q90Z, np.array([1.0, 2.0, 3.0])), [-2.0, 1.0, 3.0], atol=ATOL
        )
        # 45 deg about z maps x -> (cos45, sin45, 0)
        np.testing.assert_allclose(
            Q.rotate(rot_about(Z_HAT, np.pi / 4), X_HAT), [S, S, 0.0], atol=ATOL
        )

    def test_rotate_matches_scipy_apply(self):
        rng = np.random.default_rng(18)
        q = rand_quats(rng, 1000)
        v = rng.normal(size=(1000, 3)) * 10.0
        np.testing.assert_allclose(Q.rotate(q, v), sci(q).apply(v), atol=1e-11)  # |v|~10

    def test_rotate_equals_dcm_matvec(self):
        rng = np.random.default_rng(19)
        q = rand_quats(rng, 1000)
        v = rng.normal(size=(1000, 3))
        got = Q.rotate(q, v)
        ref = np.einsum("nij,nj->ni", Q.to_dcm(q), v)
        np.testing.assert_allclose(got, ref, atol=ATOL)

    def test_rotate_single_q_many_vectors(self):
        rng = np.random.default_rng(20)
        q = rand_quats(rng, 1)[0]
        v = rng.normal(size=(50, 3))
        out = Q.rotate(q, v)
        assert out.shape == (50, 3)
        np.testing.assert_allclose(out, v @ Q.to_dcm(q).T, atol=ATOL)

    def test_rotate_many_q_single_vector(self):
        rng = np.random.default_rng(21)
        q = rand_quats(rng, 50)
        v = np.array([0.2, -0.4, 0.9])
        out = Q.rotate(q, v)
        assert out.shape == (50, 3)
        np.testing.assert_allclose(out, sci(q).apply(v), atol=ATOL)

    def test_rotate_single_shape(self):
        assert Q.rotate(Q90Z, X_HAT).shape == (3,)
        assert Q.rotate_inverse(Q90Z, X_HAT).shape == (3,)

    def test_rotate_preserves_norm(self):
        rng = np.random.default_rng(22)
        q = rand_quats(rng, 1000)
        v = rng.normal(size=(1000, 3))
        np.testing.assert_allclose(
            np.linalg.norm(Q.rotate(q, v), axis=-1), np.linalg.norm(v, axis=-1), atol=ATOL
        )

    def test_rotate_sign_invariant(self):
        rng = np.random.default_rng(23)
        q = rand_quats(rng, 200)
        v = rng.normal(size=(200, 3))
        np.testing.assert_allclose(Q.rotate(q, v), Q.rotate(-q, v), atol=ATOL)

    def test_rotate_composition(self):
        # R(p q) v = R(p) (R(q) v)
        rng = np.random.default_rng(24)
        p, q = rand_quats(rng, 1000), rand_quats(rng, 1000)
        v = rng.normal(size=(1000, 3))
        np.testing.assert_allclose(
            Q.rotate(Q.multiply(p, q), v), Q.rotate(p, Q.rotate(q, v)), atol=ATOL
        )

    def test_rotate_inverse_hand(self):
        # reference->body: +90 about z maps y_ref to x_body
        np.testing.assert_allclose(Q.rotate_inverse(Q90Z, Y_HAT), X_HAT, atol=ATOL)
        np.testing.assert_allclose(Q.rotate_inverse(Q90Z, X_HAT), -Y_HAT, atol=ATOL)

    def test_rotate_inverse_matches_scipy_and_dcm_transpose(self):
        rng = np.random.default_rng(25)
        q = rand_quats(rng, 1000)
        v = rng.normal(size=(1000, 3))
        got = Q.rotate_inverse(q, v)
        np.testing.assert_allclose(got, sci(q).apply(v, inverse=True), atol=ATOL)
        ref = np.einsum("nji,nj->ni", Q.to_dcm(q), v)  # R^T v
        np.testing.assert_allclose(got, ref, atol=ATOL)

    def test_rotate_inverse_undoes_rotate(self):
        rng = np.random.default_rng(26)
        q = rand_quats(rng, 1000)
        v = rng.normal(size=(1000, 3))
        np.testing.assert_allclose(Q.rotate_inverse(q, Q.rotate(q, v)), v, atol=ATOL)
        np.testing.assert_allclose(Q.rotate(q, Q.rotate_inverse(q, v)), v, atol=ATOL)


# ==========================================================================
# from_dcm
# ==========================================================================
class TestFromDcm:
    def test_identity(self):
        np.testing.assert_allclose(Q.from_dcm(np.eye(3)), IDENT, atol=ATOL)

    @pytest.mark.parametrize("q", [Q90X, Q90Y, Q90Z])
    def test_90_degree_hand(self, q):
        # from the hand-derived matrices of TestDcmAndRotate
        np.testing.assert_allclose(Q.from_dcm(sci(q).as_matrix()), q, atol=ATOL)

    @pytest.mark.parametrize(
        "R, expected",
        [
            (np.diag([1.0, -1.0, -1.0]), Q180X),
            (np.diag([-1.0, 1.0, -1.0]), Q180Y),
            (np.diag([-1.0, -1.0, 1.0]), Q180Z),
        ],
    )
    def test_180_degrees_about_axes(self, R, expected):
        q = Q.from_dcm(R)
        assert q.shape == (4,)
        assert np.all(np.isfinite(q))
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=ATOL)
        assert q[0] >= 0.0
        # w = 0 here, so the quaternion is defined up to sign; rotation must match
        assert_same_rotation(q, expected)
        np.testing.assert_allclose(Q.to_dcm(q), R, atol=ATOL)

    def test_180_degrees_random_axes(self):
        rng = np.random.default_rng(27)
        axes = rand_unit_vecs(rng, 200)
        R = Rotation.from_rotvec(np.pi * axes).as_matrix()
        q = Q.from_dcm(R)
        assert q.shape == (200, 4)
        assert np.all(np.isfinite(q))
        assert np.all(q[:, 0] >= 0.0)
        np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1.0, atol=ATOL)
        np.testing.assert_allclose(Q.to_dcm(q), R, atol=ATOL)
        # vector part is +-axis
        expected = np.concatenate([np.zeros((200, 1)), axes], axis=-1)
        assert_same_rotation(q, expected, atol=1e-9)  # w ~ 1e-17, axis from sqrt of diag => 1e-9

    def test_180_degrees_single_random_axis_stable(self):
        axis = np.array([0.36, -0.48, 0.8])  # unit
        R = Rotation.from_rotvec(np.pi * axis).as_matrix()
        q = Q.from_dcm(R)
        np.testing.assert_allclose(Q.to_dcm(q), R, atol=ATOL)
        assert q[0] >= 0.0

    def test_round_trip_random_canonical(self):
        q = rand_quats(np.random.default_rng(28), 1000)
        back = Q.from_dcm(Q.to_dcm(q))
        assert back.shape == (1000, 4)
        assert np.all(back[:, 0] >= 0.0)
        np.testing.assert_allclose(back, Q.canonicalize(q), atol=ATOL)

    def test_matches_scipy_from_matrix(self):
        rng = np.random.default_rng(29)
        R = Rotation.random(1000, random_state=rng).as_matrix()
        q = Q.from_dcm(R)
        ref = sci_quat(Rotation.from_matrix(R))
        assert_same_rotation(q, ref)
        assert np.all(q[:, 0] >= 0.0)

    def test_unit_norm_output(self):
        R = Rotation.random(500, random_state=30).as_matrix()
        np.testing.assert_allclose(np.linalg.norm(Q.from_dcm(R), axis=-1), 1.0, atol=ATOL)

    # ---- all four Shepperd branches --------------------------------------
    @pytest.mark.parametrize(
        "q_in, dominant",
        [
            ([0.9, 0.1, 0.2, 0.3], 0),
            ([0.1, 0.9, 0.3, -0.2], 1),
            ([0.2, -0.3, 0.9, 0.1], 2),
            ([0.15, 0.25, -0.1, 0.9], 3),
            ([-0.9, 0.1, 0.2, 0.3], 0),  # negative w
            ([-0.1, -0.9, 0.3, 0.2], 1),
            ([-0.2, 0.3, -0.9, 0.1], 2),
            ([-0.15, 0.25, 0.1, -0.9], 3),
        ],
    )
    def test_each_branch_single(self, q_in, dominant):
        q_in = np.asarray(q_in, dtype=float)
        q_in /= np.linalg.norm(q_in)
        assert np.argmax(np.abs(q_in)) == dominant  # sanity of the test data
        R = sci(q_in).as_matrix()
        q = Q.from_dcm(R)
        assert q[0] >= 0.0
        assert_same_rotation(q, q_in)
        np.testing.assert_allclose(Q.to_dcm(q), R, atol=ATOL)

    @pytest.mark.parametrize("dominant", [0, 1, 2, 3])
    def test_each_branch_vectorised(self, dominant):
        rng = np.random.default_rng(31 + dominant)
        q = rng.normal(size=(2000, 4)) * 0.3
        q[:, dominant] = rng.choice([-1.0, 1.0], size=2000) * rng.uniform(1.0, 2.0, size=2000)
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        assert np.mean(np.argmax(np.abs(q), axis=-1) == dominant) > 0.99  # test-data sanity
        R = sci(q).as_matrix()
        out = Q.from_dcm(R)
        assert np.all(out[:, 0] >= 0.0)
        assert_same_rotation(out, q, atol=ATOL)
        np.testing.assert_allclose(Q.to_dcm(out), R, atol=ATOL)

    @pytest.mark.parametrize(
        "q_in",
        [
            [0.5, 0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5, -0.5],
            [-0.5, 0.5, 0.5, -0.5],
            [0.5 + 1e-9, 0.5, 0.5, 0.5 - 1e-9],
            [0.5, 0.5 + 1e-9, 0.5 - 1e-9, 0.5],
            [0.5, 0.5, 0.5 + 1e-9, 0.5 - 1e-9],
            [0.5, 0.5 - 1e-9, 0.5, 0.5 + 1e-9],
        ],
    )
    def test_near_ties_between_branches(self, q_in):
        q_in = np.asarray(q_in, dtype=float)
        q_in /= np.linalg.norm(q_in)
        R = sci(q_in).as_matrix()
        q = Q.from_dcm(R)
        assert q[0] >= 0.0
        assert_same_rotation(q, q_in)

    def test_all_branches_covered_in_random_set(self):
        # guard: the random set used below really exercises every branch
        q = rand_quats(np.random.default_rng(35), 1000)
        assert set(np.argmax(np.abs(q), axis=-1)) == {0, 1, 2, 3}

    def test_near_180_degrees_each_axis_perturbed(self):
        for axis in (X_HAT, Y_HAT, Z_HAT, np.array([1.0, 1.0, 1.0]) / np.sqrt(3)):
            for eps in (-1e-6, 0.0, 1e-6, 1e-3):
                R = Rotation.from_rotvec((np.pi - eps) * axis).as_matrix()
                q = Q.from_dcm(R)
                assert q[0] >= 0.0
                np.testing.assert_allclose(Q.to_dcm(q), R, atol=ATOL)

    def test_small_angle(self):
        for ang in (1e-10, 1e-6, 1e-3):
            q_in = rot_about([1.0, -2.0, 0.5], ang)
            q = Q.from_dcm(sci(q_in).as_matrix())
            assert q[0] >= 0.0
            # DCM only carries ~1e-16 absolute accuracy, so a 1e-10 angle's vector part
            # (5e-11) is recovered with ~1e-16 absolute error.
            np.testing.assert_allclose(q, q_in, atol=1e-15)

    def test_batched_shape(self):
        R = Rotation.random(6, random_state=36).as_matrix().reshape(2, 3, 3, 3)
        q = Q.from_dcm(R)
        assert q.shape == (2, 3, 4)
        np.testing.assert_allclose(Q.to_dcm(q), R, atol=ATOL)


# ==========================================================================
# from_axis_angle
# ==========================================================================
class TestAxisAngle:
    @pytest.mark.parametrize(
        "axis, angle, expected",
        [
            (X_HAT, np.pi / 2, Q90X),
            (Y_HAT, np.pi / 2, Q90Y),
            (Z_HAT, np.pi / 2, Q90Z),
            (X_HAT, np.pi, Q180X),
            (Y_HAT, np.pi, Q180Y),
            (Z_HAT, np.pi, Q180Z),
            (Z_HAT, 0.0, IDENT),
        ],
    )
    def test_hand(self, axis, angle, expected):
        np.testing.assert_allclose(Q.from_axis_angle(axis, angle), expected, atol=ATOL)

    def test_axis_is_normalised(self):
        np.testing.assert_allclose(
            Q.from_axis_angle([0.0, 0.0, 5.0], np.pi / 2), Q90Z, atol=ATOL
        )
        np.testing.assert_allclose(
            Q.from_axis_angle([3.0, 4.0, 0.0], 1.0), rot_about([0.6, 0.8, 0.0], 1.0), atol=ATOL
        )

    def test_matches_scipy_batched(self):
        rng = np.random.default_rng(37)
        axes = rng.normal(size=(1000, 3)) * rng.uniform(0.1, 5.0, size=(1000, 1))
        ang = rng.uniform(-np.pi, np.pi, size=1000)
        q = Q.from_axis_angle(axes, ang)
        assert q.shape == (1000, 4)
        unit = axes / np.linalg.norm(axes, axis=-1, keepdims=True)
        assert_same_rotation(q, sci_quat(Rotation.from_rotvec(unit * ang[:, None])))
        np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1.0, atol=ATOL)


# ==========================================================================
# rotvec
# ==========================================================================
class TestRotvec:
    def test_from_rotvec_hand(self):
        np.testing.assert_allclose(Q.from_rotvec([0.0, 0.0, np.pi / 2]), Q90Z, atol=ATOL)
        np.testing.assert_allclose(Q.from_rotvec([np.pi / 2, 0.0, 0.0]), Q90X, atol=ATOL)
        np.testing.assert_allclose(Q.from_rotvec([0.0, np.pi / 2, 0.0]), Q90Y, atol=ATOL)
        np.testing.assert_allclose(Q.from_rotvec([0.0, 0.0, np.pi]), Q180Z, atol=ATOL)

    def test_from_rotvec_matches_scipy(self):
        rng = np.random.default_rng(38)
        phi = rand_unit_vecs(rng, 1000) * rng.uniform(0.0, 3.0, size=(1000, 1))
        q = Q.from_rotvec(phi)
        assert q.shape == (1000, 4)
        assert_same_rotation(q, sci_quat(Rotation.from_rotvec(phi)))
        np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1.0, atol=ATOL)

    def test_from_rotvec_formula_exact_sign(self):
        # spec: exp(phi) = [cos(|phi|/2), sin(|phi|/2) phi/|phi|]  (no canonicalisation)
        rng = np.random.default_rng(39)
        phi = rand_unit_vecs(rng, 500) * rng.uniform(0.0, 6.0, size=(500, 1))
        n = np.linalg.norm(phi, axis=-1, keepdims=True)
        expected = np.concatenate([np.cos(n / 2), np.sin(n / 2) * phi / n], axis=-1)
        np.testing.assert_allclose(Q.from_rotvec(phi), expected, atol=ATOL)

    def test_from_rotvec_zero_is_exact_identity_no_warnings(self):
        with strict():
            q = Q.from_rotvec(np.zeros(3))
        np.testing.assert_array_equal(q, IDENT)

    def test_to_rotvec_identity_is_exact_zero_no_warnings(self):
        with strict():
            phi = Q.to_rotvec(IDENT)
        np.testing.assert_array_equal(phi, np.zeros(3))

    def test_zero_in_batch_no_warnings(self):
        phi = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [0.0, 0.0, 0.0], [1e-12, 0.0, 0.0]])
        with strict():
            q = Q.from_rotvec(phi)
            back = Q.to_rotvec(q)
        np.testing.assert_array_equal(q[0], IDENT)
        np.testing.assert_array_equal(q[2], IDENT)
        np.testing.assert_allclose(back, phi, atol=1e-15)

    def test_neg_identity_to_rotvec_is_zero(self):
        with strict():
            phi = Q.to_rotvec(-IDENT)
        np.testing.assert_allclose(phi, np.zeros(3), atol=0)

    @pytest.mark.parametrize("mag", [1e-14, 1e-12, 1e-10, 1e-9, 0.9e-8, 1e-8, 1.1e-8, 1e-7, 1e-5])
    def test_tiny_angles(self, mag):
        axis = np.array([2.0, -1.0, 2.0]) / 3.0
        phi = mag * axis
        with strict():
            q = Q.from_rotvec(phi)
            back = Q.to_rotvec(q)
        # agreement with the exact formula sin(mag/2) * axis to full relative accuracy
        np.testing.assert_allclose(q[1:], np.sin(0.5 * mag) * axis, rtol=1e-13, atol=0)
        np.testing.assert_allclose(q[0], 1.0, atol=1e-15)
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-15)
        # round trip must keep full *relative* accuracy at tiny angles (atan2, not arccos)
        np.testing.assert_allclose(back, phi, rtol=1e-10, atol=0)

    @pytest.mark.parametrize("mag", [1e-10, 1e-6])
    def test_tiny_angles_vs_scipy(self, mag):
        phi = mag * np.array([0.6, 0.0, -0.8])
        assert_same_rotation(Q.from_rotvec(phi), sci_quat(Rotation.from_rotvec(phi)), atol=1e-15)
        q = sci_quat(Rotation.from_rotvec(phi))
        np.testing.assert_allclose(Q.to_rotvec(q), phi, rtol=1e-10, atol=0)

    def test_continuity_across_taylor_threshold(self):
        axis = np.array([0.0, 0.6, 0.8])
        lo = Q.from_rotvec(0.999999e-8 * axis)
        hi = Q.from_rotvec(1.000001e-8 * axis)
        np.testing.assert_allclose(lo, hi, atol=1e-13)
        assert np.all(np.diff([lo[3], hi[3]]) >= 0)  # monotonic in the vector part

    def test_to_rotvec_hand(self):
        np.testing.assert_allclose(Q.to_rotvec(Q90Z), [0.0, 0.0, np.pi / 2], atol=ATOL)
        np.testing.assert_allclose(Q.to_rotvec(Q90X), [np.pi / 2, 0.0, 0.0], atol=ATOL)
        np.testing.assert_allclose(Q.to_rotvec(Q90Y), [0.0, np.pi / 2, 0.0], atol=ATOL)

    def test_to_rotvec_matches_scipy(self):
        # exclude angles within 1e-3 of pi where as_rotvec's sign convention is arbitrary
        rng = np.random.default_rng(40)
        phi = rand_unit_vecs(rng, 1000) * rng.uniform(0.0, np.pi - 1e-3, size=(1000, 1))
        q = sci_quat(Rotation.from_rotvec(phi))
        got = Q.to_rotvec(q)
        assert got.shape == (1000, 3)
        np.testing.assert_allclose(got, sci(q).as_rotvec(), atol=ATOL)
        np.testing.assert_allclose(got, phi, atol=ATOL)

    def test_to_rotvec_sign_invariant_and_bounded(self):
        q = rand_quats(np.random.default_rng(41), 1000)
        a = Q.to_rotvec(q)
        b = Q.to_rotvec(-q)
        np.testing.assert_allclose(a, b, atol=ATOL)
        assert np.all(np.linalg.norm(a, axis=-1) <= np.pi + 1e-12)

    def test_to_rotvec_matches_scipy_random_rotations(self):
        rng = np.random.default_rng(42)
        rot = Rotation.random(1000, random_state=rng)
        q = sci_quat(rot)
        got = Q.to_rotvec(q)
        # angle agrees everywhere; axis agrees except in the measure-zero pi case
        np.testing.assert_allclose(np.linalg.norm(got, axis=-1), rot.magnitude(), atol=1e-12)
        np.testing.assert_allclose(got, rot.as_rotvec(), atol=1e-9)  # near-pi rows are ill-conditioned

    def test_round_trip_q_random(self):
        q = rand_quats(np.random.default_rng(43), 1000)
        back = Q.from_rotvec(Q.to_rotvec(q))
        assert_same_rotation(back, q)

    def test_round_trip_rotvec_random(self):
        rng = np.random.default_rng(44)
        phi = rand_unit_vecs(rng, 1000) * rng.uniform(0.0, np.pi - 1e-6, size=(1000, 1))
        np.testing.assert_allclose(Q.to_rotvec(Q.from_rotvec(phi)), phi, atol=ATOL)

    def test_large_rotvec_maps_into_ball(self):
        # |phi| = 4 > pi: equivalent rotation by (4 - 2 pi) about the same axis, |phi'| <= pi
        axis = np.array([1.0, 2.0, -2.0]) / 3.0
        for mag in (4.0, 5.5, 6.0, 7.0):
            phi = mag * axis
            got = Q.to_rotvec(Q.from_rotvec(phi))
            assert np.linalg.norm(got) <= np.pi + 1e-12
            np.testing.assert_allclose(got, sci(sci_quat(Rotation.from_rotvec(phi))).as_rotvec(), atol=1e-12)
            assert_same_rotation(Q.from_rotvec(got), Q.from_rotvec(phi))

    @pytest.mark.parametrize("delta", [1e-3, 1e-6, 1e-9, 0.0, -1e-9, -1e-6])
    def test_near_pi(self, delta):
        axis = np.array([2.0, 3.0, 6.0]) / 7.0
        angle = np.pi - delta
        q = rot_about(axis, angle)  # w = sin(delta/2) (negative for delta < 0)
        with strict():
            phi = Q.to_rotvec(q)
        mag = np.linalg.norm(phi)
        assert mag <= np.pi + 1e-12
        # minimal-angle representation: |phi| = pi - |delta|
        np.testing.assert_allclose(mag, np.pi - abs(delta), atol=1e-12)
        assert_same_rotation(Q.from_rotvec(phi), q, atol=1e-12)
        if delta > 0:
            np.testing.assert_allclose(phi, angle * axis, atol=1e-12)
        elif delta < 0:
            # w < 0: canonicalisation gives the equivalent short way round, about -axis
            np.testing.assert_allclose(phi, -(2 * np.pi - angle) * axis, atol=1e-12)

    def test_exactly_pi_each_axis(self):
        for q, axis in ((Q180X, X_HAT), (Q180Y, Y_HAT), (Q180Z, Z_HAT)):
            for sgn in (1.0, -1.0):
                with strict():
                    phi = Q.to_rotvec(sgn * q)
                np.testing.assert_allclose(np.linalg.norm(phi), np.pi, atol=1e-12)
                # +-pi about the axis (sign ambiguous at exactly pi)
                assert abs(np.dot(phi, axis)) == pytest.approx(np.pi, abs=1e-12)

    def test_non_unit_quaternion_to_rotvec(self):
        # to_rotvec should be insensitive to a positive overall scale of q (atan2 form)
        q = rot_about([0.0, 1.0, 1.0], 1.2)
        np.testing.assert_allclose(Q.to_rotvec(3.0 * q), Q.to_rotvec(q), atol=1e-12)

    def test_batched_shapes(self):
        phi = np.random.default_rng(45).normal(size=(2, 3, 3)) * 0.3
        q = Q.from_rotvec(phi)
        assert q.shape == (2, 3, 4)
        assert Q.to_rotvec(q).shape == (2, 3, 3)
        np.testing.assert_allclose(Q.to_rotvec(q), phi, atol=ATOL)


# ==========================================================================
# Euler ZYX
# ==========================================================================
class TestEulerZYX:
    def test_from_euler_hand(self):
        np.testing.assert_allclose(Q.from_euler_zyx(np.pi / 2, 0.0, 0.0), Q90X, atol=ATOL)
        np.testing.assert_allclose(Q.from_euler_zyx(0.0, np.pi / 2, 0.0), Q90Y, atol=ATOL)
        np.testing.assert_allclose(Q.from_euler_zyx(0.0, 0.0, np.pi / 2), Q90Z, atol=ATOL)
        np.testing.assert_allclose(Q.from_euler_zyx(0.0, 0.0, 0.0), IDENT, atol=ATOL)

    def test_from_euler_composition_order_hand(self):
        # R = Rz(yaw) Ry(pitch) Rx(roll): roll applied first (to the vector).
        # yaw 90, roll 90: y -> (Rx) z -> (Rz) z ; x -> x -> y
        q = Q.from_euler_zyx(np.pi / 2, 0.0, np.pi / 2)
        np.testing.assert_allclose(Q.rotate(q, Y_HAT), Z_HAT, atol=ATOL)
        np.testing.assert_allclose(Q.rotate(q, X_HAT), Y_HAT, atol=ATOL)
        # pitch 90, yaw 90: x -> (Ry90) -z -> (Rz) -z ; z -> x -> y
        q = Q.from_euler_zyx(0.0, np.pi / 2, np.pi / 2)
        np.testing.assert_allclose(Q.rotate(q, X_HAT), -Z_HAT, atol=ATOL)
        np.testing.assert_allclose(Q.rotate(q, Z_HAT), Y_HAT, atol=ATOL)

    def test_from_euler_matches_scipy(self):
        rng = np.random.default_rng(46)
        roll = rng.uniform(-np.pi, np.pi, 1000)
        pitch = rng.uniform(-np.pi / 2, np.pi / 2, 1000)
        yaw = rng.uniform(-np.pi, np.pi, 1000)
        q = Q.from_euler_zyx(roll, pitch, yaw)
        assert q.shape == (1000, 4)
        ref = sci_quat(Rotation.from_euler("ZYX", np.stack([yaw, pitch, roll], axis=-1)))
        assert_same_rotation(q, ref)
        np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1.0, atol=ATOL)

    def test_from_euler_dcm_is_rz_ry_rx(self):
        r, p, y = 0.3, -0.7, 2.1
        Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
        Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
        Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
        np.testing.assert_allclose(Q.to_dcm(Q.from_euler_zyx(r, p, y)), Rz @ Ry @ Rx, atol=ATOL)

    def test_from_euler_broadcast(self):
        pitch = np.linspace(-1.0, 1.0, 9)
        q = Q.from_euler_zyx(0.4, pitch, -0.2)
        assert q.shape == (9, 4)
        for k in range(9):
            np.testing.assert_allclose(q[k], Q.from_euler_zyx(0.4, pitch[k], -0.2), atol=ATOL)
        q2 = Q.from_euler_zyx(np.zeros((2, 1)), np.zeros((1, 3)), 0.5)
        assert q2.shape == (2, 3, 4)

    def test_from_euler_scalar_shape(self):
        assert Q.from_euler_zyx(0.1, 0.2, 0.3).shape == (4,)

    def test_to_euler_hand(self):
        # (Q90Y is a gimbal-lock case; roll/yaw split is ambiguous, covered separately.)
        for q, expected in ((Q90X, (np.pi / 2, 0.0, 0.0)), (Q90Z, (0.0, 0.0, np.pi / 2))):
            r, p, y = Q.to_euler_zyx(q)
            np.testing.assert_allclose([r, p, y], expected, atol=ATOL)
        r, p, y = Q.to_euler_zyx(rot_about(Y_HAT, 0.5))
        np.testing.assert_allclose([r, p, y], [0.0, 0.5, 0.0], atol=ATOL)

    def test_to_euler_returns_triple(self):
        out = Q.to_euler_zyx(IDENT)
        assert len(out) == 3
        np.testing.assert_allclose(out, [0.0, 0.0, 0.0], atol=ATOL)

    def test_round_trip_random(self):
        rng = np.random.default_rng(47)
        roll = rng.uniform(-3.0, 3.0, 1000)  # keep away from the +-pi wrap
        pitch = rng.uniform(-1.5, 1.5, 1000)  # away from gimbal lock (|pitch| < pi/2 - 0.07)
        yaw = rng.uniform(-3.0, 3.0, 1000)
        r, p, y = Q.to_euler_zyx(Q.from_euler_zyx(roll, pitch, yaw))
        for got, want in ((r, roll), (p, pitch), (y, yaw)):
            assert got.shape == (1000,)
            # error amplification ~1/cos(pitch) <= 14 on 1e-16 input => stay far below 1e-12
            np.testing.assert_allclose(got, want, atol=ATOL)

    def test_round_trip_from_quaternion(self):
        q = rand_quats(np.random.default_rng(48), 1000)
        r, p, y = Q.to_euler_zyx(q)
        assert_same_rotation(Q.from_euler_zyx(r, p, y), q, atol=1e-9)  # near-lock rows amplify rounding

    def test_ranges(self):
        q = rand_quats(np.random.default_rng(49), 2000)
        r, p, y = Q.to_euler_zyx(q)
        assert np.all(np.isfinite(r) & np.isfinite(p) & np.isfinite(y))
        assert np.all(r > -np.pi - 1e-15) and np.all(r <= np.pi + 1e-15)
        assert np.all(y > -np.pi - 1e-15) and np.all(y <= np.pi + 1e-15)
        assert np.all(np.abs(p) <= np.pi / 2 + 1e-15)

    def test_matches_scipy_as_euler(self):
        rng = np.random.default_rng(50)
        rot = Rotation.random(1000, random_state=rng)
        eul = rot.as_euler("ZYX")  # [yaw, pitch, roll]
        away = np.abs(eul[:, 1]) < np.pi / 2 - 1e-3  # exclude near gimbal lock
        q = sci_quat(rot)[away]
        r, p, y = Q.to_euler_zyx(q)
        np.testing.assert_allclose(wrap(r - eul[away, 2]), 0.0, atol=1e-10)
        np.testing.assert_allclose(p, eul[away, 1], atol=1e-10)
        np.testing.assert_allclose(wrap(y - eul[away, 0]), 0.0, atol=1e-10)

    def test_sign_invariant(self):
        q = rand_quats(np.random.default_rng(51), 500)
        a = Q.to_euler_zyx(q)
        b = Q.to_euler_zyx(-q)
        for u, v in zip(a, b):
            np.testing.assert_allclose(wrap(u - v), 0.0, atol=1e-10)

    @pytest.mark.parametrize("sign", [1.0, -1.0])
    def test_gimbal_lock_no_nan(self, sign):
        rng = np.random.default_rng(52)
        roll = rng.uniform(-np.pi, np.pi, 500)
        yaw = rng.uniform(-np.pi, np.pi, 500)
        pitch = sign * np.pi / 2
        eul = np.stack([yaw, np.full(500, pitch), roll], axis=-1)
        q = sci_quat(Rotation.from_euler("ZYX", eul))  # independent construction
        with strict():  # arcsin(1 + eps) would raise 'invalid' unless clipped
            r, p, y = Q.to_euler_zyx(q)
        assert np.all(np.isfinite(r)) and np.all(np.isfinite(p)) and np.all(np.isfinite(y))
        # arcsin is ill-conditioned at +-1: sqrt(2 * 1e-16) ~ 1.5e-8 worst-case error
        np.testing.assert_allclose(p, pitch, atol=1e-7)

    @pytest.mark.parametrize("sign", [1.0, -1.0])
    def test_gimbal_lock_exact_quaternions_and_overshoot(self, sign):
        # q whose pitch argument 2(wy - xz) rounds to just above 1
        a = np.pi / 4 * sign  # half of pitch = pi/2
        q = np.array([np.cos(a), 0.0, np.sin(a), 0.0])
        for scale in (1.0, 1.0 + 1e-12):  # slightly non-unit input pushes arg beyond 1
            with strict():
                r, p, y = Q.to_euler_zyx(q * scale)
            assert np.all(np.isfinite([r, p, y]))
            assert abs(p) <= np.pi / 2 + 1e-15
            np.testing.assert_allclose(p, sign * np.pi / 2, atol=1e-6)  # non-unit => arg error ~1e-12 => 1.4e-6

    def test_near_gimbal_lock_round_trip(self):
        for pitch in (np.pi / 2 - 1e-4, -np.pi / 2 + 1e-4):
            r0, y0 = 0.4, -1.1
            r, p, y = Q.to_euler_zyx(Q.from_euler_zyx(r0, pitch, y0))
            # conditioning ~ 1/cos(pitch) = 1e4 on 1e-16 inputs
            np.testing.assert_allclose([r, p, y], [r0, pitch, y0], atol=1e-10)


# ==========================================================================
# small_angle_quat
# ==========================================================================
class TestSmallAngleQuat:
    def test_formula(self):
        dth = np.array([0.02, -0.04, 0.06])
        raw = np.array([1.0, 0.01, -0.02, 0.03])
        np.testing.assert_allclose(Q.small_angle_quat(dth), raw / np.linalg.norm(raw), atol=ATOL)

    def test_zero_is_identity(self):
        np.testing.assert_allclose(Q.small_angle_quat(np.zeros(3)), IDENT, atol=ATOL)

    def test_unit_norm_and_batched(self):
        d = np.random.default_rng(53).normal(size=(1000, 3)) * 0.1
        q = Q.small_angle_quat(d)
        assert q.shape == (1000, 4)
        np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1.0, atol=ATOL)

    def test_close_to_exact_exp_for_small_angles(self):
        # difference between normalise([1, d/2]) and exp(d) is O(|d|^3 / 24)
        d = np.random.default_rng(54).normal(size=(500, 3))
        d = d / np.linalg.norm(d, axis=-1, keepdims=True) * 1e-4
        assert_same_rotation(Q.small_angle_quat(d), sci_quat(Rotation.from_rotvec(d)), atol=1e-13)


# ==========================================================================
# error_rotvec / angle_between
# ==========================================================================
class TestErrorAndAngle:
    def test_error_identical_is_zero(self):
        q = rand_quats(np.random.default_rng(55), 100)
        np.testing.assert_allclose(Q.error_rotvec(q, q), 0.0, atol=ATOL)
        np.testing.assert_allclose(Q.error_rotvec(q, -q), 0.0, atol=ATOL)

    def test_error_hand_body_frame(self):
        # q_ref = 90 deg about z; q = q_ref (x) (0.1 rad about BODY x).
        # Body-frame (right) error => e = [0.1, 0, 0], not rotated into inertial axes.
        q_ref = Q90Z
        q = sci_quat(sci(q_ref) * Rotation.from_rotvec([0.1, 0.0, 0.0]))
        np.testing.assert_allclose(Q.error_rotvec(q_ref, q), [0.1, 0.0, 0.0], atol=ATOL)

    def test_error_recovers_injected_rotvec(self):
        rng = np.random.default_rng(56)
        q_ref = rand_quats(rng, 1000)
        delta = rand_unit_vecs(rng, 1000) * rng.uniform(0.0, 3.0, size=(1000, 1))
        q = sci_quat(sci(q_ref) * Rotation.from_rotvec(delta))  # q = q_ref (x) exp(delta)
        e = Q.error_rotvec(q_ref, q)
        assert e.shape == (1000, 3)
        np.testing.assert_allclose(e, delta, atol=ATOL)
        np.testing.assert_allclose(Q.error_rotvec(q_ref, -q), delta, atol=ATOL)

    def test_error_reconstructs_q(self):
        # multiply(q_ref, from_rotvec(e)) == +-q
        rng = np.random.default_rng(57)
        q_ref, q = rand_quats(rng, 1000), rand_quats(rng, 1000)
        e = Q.error_rotvec(q_ref, q)
        assert np.all(np.linalg.norm(e, axis=-1) <= np.pi + 1e-12)
        assert_same_rotation(Q.multiply(q_ref, Q.from_rotvec(e)), q)

    def test_error_small_matches_two_times_vector_part(self):
        rng = np.random.default_rng(58)
        q_ref = rand_quats(rng, 100)
        delta = rng.normal(size=(100, 3)) * 1e-6
        q = Q.multiply(q_ref, Q.from_rotvec(delta))
        np.testing.assert_allclose(Q.error_rotvec(q_ref, q), delta, atol=1e-14)

    def test_angle_between_hand(self):
        np.testing.assert_allclose(Q.angle_between(IDENT, Q90Z), np.pi / 2, atol=ATOL)
        np.testing.assert_allclose(Q.angle_between(IDENT, Q180X), np.pi, atol=ATOL)
        np.testing.assert_allclose(Q.angle_between(Q90X, Q90X), 0.0, atol=1e-7)  # arccos resolution
        np.testing.assert_allclose(Q.angle_between(Q90Z, Q180Z), np.pi / 2, atol=ATOL)
        np.testing.assert_allclose(Q.angle_between(Q90Z, Q.conjugate(Q90Z)), np.pi, atol=ATOL)

    def test_angle_between_same_quaternion_nonnegative_finite(self):
        # identical (and sign-flipped) quaternions: 0, never NaN even if a dot product
        # rounds to 1 + eps.
        q = rand_quats(np.random.default_rng(59), 1000)
        with strict():
            a = Q.angle_between(q, q)
            b = Q.angle_between(q, -q)
        assert np.all(a >= 0) and np.all(b >= 0)
        # arccos-based implementations have sqrt(eps) ~ 1.5e-8 resolution at 0
        np.testing.assert_allclose(a, 0.0, atol=1e-7)
        np.testing.assert_allclose(b, 0.0, atol=1e-7)

    def test_angle_between_shapes(self):
        assert np.shape(Q.angle_between(IDENT, Q90Z)) == ()
        q = rand_quats(np.random.default_rng(60), 12)
        assert Q.angle_between(q, Q90Z).shape == (12,)
        assert Q.angle_between(q[:6].reshape(2, 3, 4), q[6:].reshape(2, 3, 4)).shape == (2, 3)

    def test_angle_between_sign_invariance(self):
        rng = np.random.default_rng(61)
        a, b = rand_quats(rng, 1000), rand_quats(rng, 1000)
        ref = Q.angle_between(a, b)
        for sa, sb in ((1, -1), (-1, 1), (-1, -1)):
            np.testing.assert_allclose(Q.angle_between(sa * a, sb * b), ref, atol=ATOL)

    def test_angle_between_range_and_symmetry(self):
        rng = np.random.default_rng(62)
        a, b = rand_quats(rng, 1000), rand_quats(rng, 1000)
        ang = Q.angle_between(a, b)
        assert np.all(ang >= 0) and np.all(ang <= np.pi + 1e-15)
        np.testing.assert_allclose(Q.angle_between(b, a), ang, atol=ATOL)

    def test_angle_between_matches_scipy_and_error_rotvec(self):
        rng = np.random.default_rng(63)
        a, b = rand_quats(rng, 1000), rand_quats(rng, 1000)
        ang = Q.angle_between(a, b)
        # scipy: magnitude of the relative rotation
        np.testing.assert_allclose(ang, (sci(a).inv() * sci(b)).magnitude(), atol=1e-12)
        np.testing.assert_allclose(ang, np.linalg.norm(Q.error_rotvec(a, b), axis=-1), atol=1e-12)

    def test_angle_between_known_small_angle(self):
        # arccos(|dot|) loses half the digits near 0 (atan2 forms keep them).  Accept either:
        # at 1e-6 rad, 1 - |dot| ~ 1.25e-13 carries ~1e-3 relative rounding => ~1e-9 angle error.
        base = rot_about([1.0, 1.0, 0.0], 0.7)
        for ang in (1e-6, 1e-4, 1e-2):
            other = sci_quat(sci(base) * Rotation.from_rotvec(ang * np.array([0.0, 0.0, 1.0])))
            np.testing.assert_allclose(Q.angle_between(base, other), ang, atol=3e-9)


# ==========================================================================
# kinematics: derivative / omega_matrix / propagate
# ==========================================================================
class TestKinematics:
    def test_derivative_of_identity_hand(self):
        w = np.array([0.2, -0.4, 0.6])
        np.testing.assert_allclose(Q.derivative(IDENT, w), [0.0, 0.1, -0.2, 0.3], atol=ATOL)

    def test_derivative_hand_non_trivial(self):
        # q = 90 deg about z, w_b = [2, 0, 0]: q_dot = 1/2 q (x) [0,2,0,0] = q (x) i
        # Hamilton: (S + S k) (x) i = S i + S k i = S i + S j
        np.testing.assert_allclose(
            Q.derivative(Q90Z, np.array([2.0, 0.0, 0.0])), [0.0, S, S, 0.0], atol=ATOL
        )

    def test_derivative_equals_half_omega_matrix_q(self):
        rng = np.random.default_rng(64)
        q = rand_quats(rng, 1000)
        w = rng.normal(size=(1000, 3))
        d = Q.derivative(q, w)
        assert d.shape == (1000, 4)
        Om = Q.omega_matrix(w)
        assert Om.shape == (1000, 4, 4)
        np.testing.assert_allclose(d, 0.5 * np.einsum("nij,nj->ni", Om, q), atol=ATOL)

    def test_derivative_equals_half_q_times_pure_w(self):
        rng = np.random.default_rng(65)
        q = rand_quats(rng, 1000)
        w = rng.normal(size=(1000, 3))
        pure = np.concatenate([np.zeros((1000, 1)), w], axis=-1)
        # Reference product through scipy composition: q (x) [0, w]  (not unit => use rotvec trick
        # is unavailable) so compare against the explicit Hamilton formula instead.
        w0, v = q[:, :1], q[:, 1:]
        ref_w = -np.sum(v * w, axis=-1, keepdims=True)
        ref_v = w0 * w + np.cross(v, w)
        ref = 0.5 * np.concatenate([ref_w, ref_v], axis=-1)
        np.testing.assert_allclose(Q.derivative(q, w), ref, atol=ATOL)
        np.testing.assert_allclose(Q.derivative(q, w), 0.5 * Q.multiply(q, pure), atol=ATOL)

    def test_derivative_orthogonal_to_unit_q(self):
        # d/dt |q|^2 = 0  =>  q . q_dot = 0
        rng = np.random.default_rng(66)
        q = rand_quats(rng, 1000)
        w = rng.normal(size=(1000, 3))
        np.testing.assert_allclose(np.sum(q * Q.derivative(q, w), axis=-1), 0.0, atol=ATOL)

    def test_omega_matrix_antisymmetric(self):
        w = np.random.default_rng(67).normal(size=(200, 3))
        Om = Q.omega_matrix(w)
        np.testing.assert_allclose(Om, -np.swapaxes(Om, -1, -2), atol=ATOL)
        assert Q.omega_matrix(w[0]).shape == (4, 4)

    def test_derivative_zero_rate(self):
        q = rand_quats(np.random.default_rng(68), 10)
        np.testing.assert_allclose(Q.derivative(q, np.zeros(3)), 0.0, atol=0)

    # ---- propagate -------------------------------------------------------
    def test_propagate_hand_quarter_turn(self):
        # w = [0, 0, pi/2] rad/s for 1 s from identity -> +90 deg about z
        np.testing.assert_allclose(
            Q.propagate(IDENT, np.array([0.0, 0.0, np.pi / 2]), 1.0), Q90Z, atol=ATOL
        )
        # half the time -> 45 deg
        np.testing.assert_allclose(
            Q.propagate(IDENT, np.array([0.0, 0.0, np.pi / 2]), 0.5),
            rot_about(Z_HAT, np.pi / 4),
            atol=ATOL,
        )

    def test_propagate_matches_q_times_exp(self):
        rng = np.random.default_rng(69)
        q = rand_quats(rng, 1000)
        w = rng.normal(size=(1000, 3)) * 2.0
        dt = 0.37
        out = Q.propagate(q, w, dt)
        assert out.shape == (1000, 4)
        ref = sci_quat(sci(q) * Rotation.from_rotvec(w * dt))  # q (x) exp(w dt) via scipy
        assert_same_rotation(out, ref)
        np.testing.assert_allclose(np.linalg.norm(out, axis=-1), 1.0, atol=ATOL)

    def test_propagate_rotates_about_body_axis(self):
        # Constant body rate about body x: the body x axis stays fixed in the reference frame.
        q = rot_about([1.0, 2.0, -0.5], 1.1)
        x_before = Q.rotate(q, X_HAT)
        out = Q.propagate(q, np.array([0.7, 0.0, 0.0]), 1.3)
        np.testing.assert_allclose(Q.rotate(out, X_HAT), x_before, atol=ATOL)
        # And the angle moved equals |w| dt.
        np.testing.assert_allclose(Q.angle_between(q, out), 0.7 * 1.3, atol=1e-12)

    def test_propagate_body_not_inertial_hand(self):
        # q = +90 deg about z (body x = reference y). Spin 90 deg about BODY x:
        # reference-frame y axis (body x) unchanged; body y axis (reference -x) -> body z axis.
        out = Q.propagate(Q90Z, np.array([np.pi / 2, 0.0, 0.0]), 1.0)
        np.testing.assert_allclose(Q.rotate(out, X_HAT), Y_HAT, atol=ATOL)
        np.testing.assert_allclose(Q.rotate(out, Y_HAT), Z_HAT, atol=ATOL)

    def test_propagate_zero_rate_normalises(self):
        q = np.array([2.0, 0.0, 0.0, 0.0])
        with strict():
            out = Q.propagate(q, np.zeros(3), 0.5)
        np.testing.assert_allclose(out, IDENT, atol=ATOL)
        qq = rand_quats(np.random.default_rng(70), 50) * 3.0
        np.testing.assert_allclose(
            Q.propagate(qq, np.zeros(3), 1.0), qq / np.linalg.norm(qq, axis=-1, keepdims=True), atol=ATOL
        )

    def test_propagate_zero_dt(self):
        q = rand_quats(np.random.default_rng(71), 50)
        np.testing.assert_allclose(Q.propagate(q, np.array([1.0, 2.0, 3.0]), 0.0), q, atol=ATOL)

    def test_propagate_many_small_steps_equals_one_large_step(self):
        # constant body rate => exact, so 1000 steps == 1 step (only rounding accumulates)
        rng = np.random.default_rng(72)
        q0 = rand_quats(rng, 20)
        w = rng.normal(size=(20, 3))  # |w| ~ 1.7 rad/s; total angle ~ 8.5 rad (> 2 pi)
        T, n = 5.0, 1000
        q = q0.copy()
        for _ in range(n):
            q = Q.propagate(q, w, T / n)
        big = Q.propagate(q0, w, T)
        assert_same_rotation(q, big, atol=1e-12)  # 1000 steps x ~1e-16 rounding

    def test_propagate_unit_norm_after_many_steps(self):
        q = rand_quats(np.random.default_rng(73), 1)[0]
        w = np.array([0.3, -0.2, 0.9])
        for _ in range(5000):
            q = Q.propagate(q, w, 0.01)
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-13)

    def test_propagate_vs_rk4_of_derivative(self):
        # Independent route: integrate q_dot = derivative(q, w) with classical RK4.
        rng = np.random.default_rng(74)
        q0 = rand_quats(rng, 50)
        w = rng.normal(size=(50, 3))
        T, n = 1.0, 200
        h = T / n
        q = q0.copy()
        for _ in range(n):
            k1 = Q.derivative(q, w)
            k2 = Q.derivative(q + 0.5 * h * k1, w)
            k3 = Q.derivative(q + 0.5 * h * k2, w)
            k4 = Q.derivative(q + h * k3, w)
            q = q + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # RK4 global error ~ (|w| h)^4 * T ~ (1.7*5e-3)^4 ~ 5e-9 worst case; plus norm drift
        assert_same_rotation(q / np.linalg.norm(q, axis=-1, keepdims=True), Q.propagate(q0, w, T), atol=1e-8)

    def test_finite_difference_matches_derivative(self):
        rng = np.random.default_rng(75)
        q = rand_quats(rng, 500)
        w = rng.normal(size=(500, 3))
        d = Q.derivative(q, w)
        # forward difference, error ~ h |q_ddot| / 2 ~ h |w|^2 / 8 ~ 4e-7 for h = 1e-7
        h = 1e-7
        fwd = (Q.propagate(q, w, h) - q) / h
        np.testing.assert_allclose(fwd, d, atol=2e-6)
        # central difference, error ~ h^2 |q_dddot| / 6 ~ 1e-10; rounding ~ 1e-16/h = 1e-11
        h = 1e-5
        cen = (Q.propagate(q, w, h) - Q.propagate(q, w, -h)) / (2 * h)
        np.testing.assert_allclose(cen, d, atol=1e-8)

    def test_finite_difference_forward_error_is_first_order(self):
        rng = np.random.default_rng(76)
        q = rand_quats(rng, 100)
        w = rng.normal(size=(100, 3))
        d = Q.derivative(q, w)
        errs = []
        for h in (1e-3, 1e-4):
            errs.append(np.max(np.abs((Q.propagate(q, w, h) - q) / h - d)))
        ratio = errs[0] / errs[1]
        assert 7.0 < ratio < 13.0, ratio  # O(h) truncation error => ratio ~ 10

    def test_propagate_broadcast_single_rate(self):
        q = rand_quats(np.random.default_rng(77), 30)
        w = np.array([0.1, -0.2, 0.3])
        out = Q.propagate(q, w, 0.2)
        assert out.shape == (30, 4)
        for k in (0, 7, 29):
            np.testing.assert_allclose(out[k], Q.propagate(q[k], w, 0.2), atol=ATOL)


# ==========================================================================
# slerp
# ==========================================================================
class TestSlerp:
    def test_endpoints(self):
        rng = np.random.default_rng(78)
        q0, q1 = rand_quats(rng, 200), rand_quats(rng, 200)
        np.testing.assert_allclose(Q.slerp(q0, q1, 0.0), q0, atol=ATOL)
        assert_same_rotation(Q.slerp(q0, q1, 1.0), q1)

    def test_midpoint_fixed_axis_is_half_angle(self):
        axis = np.array([1.0, -2.0, 2.0]) / 3.0
        th0, th1 = 0.3, 2.1  # differ by < pi: no shortest-arc flip
        q0, q1 = rot_about(axis, th0), rot_about(axis, th1)
        np.testing.assert_allclose(
            Q.slerp(q0, q1, 0.5), rot_about(axis, 0.5 * (th0 + th1)), atol=ATOL
        )

    def test_fixed_axis_from_identity_grid(self):
        axis = np.array([0.0, 0.6, 0.8])
        q1 = rot_about(axis, 2.4)
        for t in np.linspace(0.0, 1.0, 11):
            assert_same_rotation(Q.slerp(IDENT, q1, t), rot_about(axis, 2.4 * t))

    def test_matches_scipy_slerp(self):
        rng = np.random.default_rng(79)
        q0, q1 = rand_quats(rng, 100), rand_quats(rng, 100)
        t = 0.3
        got = Q.slerp(q0, q1, t)
        assert got.shape == (100, 4)
        ref = np.stack(
            [
                sci_quat(Slerp([0.0, 1.0], sci(np.stack([a, b])))([t]))[0]
                for a, b in zip(q0, q1)
            ]
        )
        assert_same_rotation(got, ref)
        np.testing.assert_allclose(np.linalg.norm(got, axis=-1), 1.0, atol=ATOL)

    def test_negative_q1_gives_same_rotation(self):
        rng = np.random.default_rng(80)
        q0, q1 = rand_quats(rng, 500), rand_quats(rng, 500)
        for t in (0.0, 0.1, 0.5, 0.9, 1.0):
            a = Q.slerp(q0, q1, t)
            b = Q.slerp(q0, -q1, t)
            assert_same_rotation(a, b)

    def test_shortest_arc_hand(self):
        # q1 = 270 deg about z has q0.q1 < 0; the shortest path is -90 deg about z
        q1 = rot_about(Z_HAT, 1.5 * np.pi)
        assert q1[0] < 0
        mid = Q.slerp(IDENT, q1, 0.5)
        assert_same_rotation(mid, rot_about(Z_HAT, -np.pi / 4))
        np.testing.assert_allclose(Q.angle_between(IDENT, mid), np.pi / 4, atol=1e-12)
        # and with the opposite-sign representation of the same q1
        assert_same_rotation(Q.slerp(IDENT, -q1, 0.5), rot_about(Z_HAT, -np.pi / 4))

    def test_never_travels_more_than_pi(self):
        rng = np.random.default_rng(81)
        q0, q1 = rand_quats(rng, 1000), rand_quats(rng, 1000)
        total = Q.angle_between(q0, q1)
        for t in (0.25, 0.5, 0.75):
            mid = Q.slerp(q0, q1, t)
            np.testing.assert_allclose(Q.angle_between(q0, mid), t * total, atol=1e-9)  # arccos resolution

    def test_constant_angular_speed(self):
        rng = np.random.default_rng(82)
        q0, q1 = rand_quats(rng, 100), rand_quats(rng, 100)
        total = Q.angle_between(q0, q1)
        a = Q.angle_between(Q.slerp(q0, q1, 0.2), Q.slerp(q0, q1, 0.6))
        np.testing.assert_allclose(a, 0.4 * total, atol=1e-9)  # arccos resolution

    def test_array_t_broadcasts(self):
        rng = np.random.default_rng(83)
        q0, q1 = rand_quats(rng, 2)
        t = np.linspace(0.0, 1.0, 7)
        out = Q.slerp(q0, q1, t)
        assert out.shape == (7, 4)
        for k, tk in enumerate(t):
            np.testing.assert_allclose(out[k], Q.slerp(q0, q1, tk), atol=ATOL)

    def test_identical_endpoints_no_nan(self):
        q = rand_quats(np.random.default_rng(84), 20)
        with strict():
            for t in (0.0, 0.3, 1.0):
                assert_same_rotation(Q.slerp(q, q, t), q)
                assert_same_rotation(Q.slerp(q, -q, t), q)
            near = Q.multiply(q, Q.from_rotvec(np.array([1e-9, 0.0, 0.0])))
            out = Q.slerp(q, near, 0.5)
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(np.linalg.norm(out, axis=-1), 1.0, atol=1e-12)
        assert np.max(Q.angle_between(out, q)) < 1e-7


# ==========================================================================
# randomised property tests (fixed seeds, vectorised, ~1000 samples)
# ==========================================================================
class TestProperties:
    N = 1000

    def test_product_of_unit_quaternions_is_unit(self):
        rng = np.random.default_rng(100)
        a, b = rand_quats(rng, self.N), rand_quats(rng, self.N)
        np.testing.assert_allclose(np.linalg.norm(Q.multiply(a, b), axis=-1), 1.0, atol=ATOL)

    def test_dcm_is_homomorphism(self):
        rng = np.random.default_rng(101)
        a, b = rand_quats(rng, self.N), rand_quats(rng, self.N)
        np.testing.assert_allclose(
            Q.to_dcm(Q.multiply(a, b)), Q.to_dcm(a) @ Q.to_dcm(b), atol=ATOL
        )
        np.testing.assert_allclose(Q.to_dcm(Q.conjugate(a)), np.swapaxes(Q.to_dcm(a), -1, -2), atol=ATOL)

    def test_from_dcm_to_dcm_round_trip_all_samples(self):
        rng = np.random.default_rng(102)
        R = Rotation.random(self.N, random_state=rng).as_matrix()
        np.testing.assert_allclose(Q.to_dcm(Q.from_dcm(R)), R, atol=ATOL)

    def test_rotvec_exp_log_identities(self):
        rng = np.random.default_rng(103)
        q = rand_quats(rng, self.N)
        phi = Q.to_rotvec(q)
        assert np.all(np.linalg.norm(phi, axis=-1) <= np.pi + 1e-12)
        assert_same_rotation(Q.from_rotvec(phi), q)
        # inverse rotation <-> negated rotvec
        np.testing.assert_allclose(Q.to_rotvec(Q.conjugate(q)), -phi, atol=ATOL)

    def test_euler_round_trip_random_in_range(self):
        rng = np.random.default_rng(104)
        r = rng.uniform(-3.1, 3.1, self.N)
        p = rng.uniform(-1.55, 1.55, self.N)
        y = rng.uniform(-3.1, 3.1, self.N)
        r2, p2, y2 = Q.to_euler_zyx(Q.from_euler_zyx(r, p, y))
        # amplification 1/cos(1.55) ~ 48 on 1e-16
        np.testing.assert_allclose(r2, r, atol=1e-12)
        np.testing.assert_allclose(p2, p, atol=1e-12)
        np.testing.assert_allclose(y2, y, atol=1e-12)

    def test_error_state_injection_consistency(self):
        # CONVENTIONS section 3: q_true = q_est (x) exp(dtheta); e = error_rotvec(q_est, q_true)
        rng = np.random.default_rng(105)
        q_est = rand_quats(rng, self.N)
        dth = rng.normal(size=(self.N, 3)) * 0.05
        q_true = Q.multiply(q_est, Q.from_rotvec(dth))
        np.testing.assert_allclose(Q.error_rotvec(q_est, q_true), dth, atol=ATOL)
        # small-angle approximation exp(dth) ~ [1, dth/2] to O(|dth|^3)
        # Per-sample bound |dth|^3 / 8 (truncation error of the first-order map is
        # ~|dth|^3/24 in the vector part and ~|dth|^3/16 via the renormalisation).
        diff = sign_invariant_diff(Q.small_angle_quat(dth), Q.from_rotvec(dth))
        assert np.all(diff <= np.linalg.norm(dth, axis=-1) ** 3 / 8 + 1e-15)

    def test_propagate_is_group_action(self):
        # propagate(propagate(q, w, a), w, b) == propagate(q, w, a + b)
        rng = np.random.default_rng(106)
        q = rand_quats(rng, self.N)
        w = rng.normal(size=(self.N, 3))
        a, b = 0.13, 0.29
        assert_same_rotation(Q.propagate(Q.propagate(q, w, a), w, b), Q.propagate(q, w, a + b))

    def test_slerp_midpoint_symmetric(self):
        rng = np.random.default_rng(107)
        q0, q1 = rand_quats(rng, self.N), rand_quats(rng, self.N)
        assert_same_rotation(Q.slerp(q0, q1, 0.5), Q.slerp(q1, q0, 0.5))

    def test_angle_triangle_inequality(self):
        rng = np.random.default_rng(108)
        a, b, c = (rand_quats(rng, self.N) for _ in range(3))
        assert np.all(
            Q.angle_between(a, c) <= Q.angle_between(a, b) + Q.angle_between(b, c) + 1e-9
        )  # arccos resolution

    def test_body_rate_inertial_rate_consistency(self):
        # R(q) w_b is the inertial rate; propagating with w_b about body axes equals
        # pre-multiplying by exp(R(q) w_b dt) (left perturbation), by scipy composition algebra.
        rng = np.random.default_rng(109)
        q = rand_quats(rng, self.N)
        w_b = rng.normal(size=(self.N, 3))
        dt = 0.21
        w_i = Q.rotate(q, w_b)
        left = sci_quat(Rotation.from_rotvec(w_i * dt) * sci(q))
        assert_same_rotation(Q.propagate(q, w_b, dt), left)


# ==========================================================================
# error injection / reset Jacobian (CONVENTIONS section 3)
# ==========================================================================
class TestInjectAttitudeError:
    N = 40

    def _random_dtheta(self, rng, n, max_angle=3.1):
        d = rand_unit_vecs(rng, n)
        return d * rng.uniform(0.0, max_angle, size=(n, 1))

    def test_equals_normalize_multiply_from_rotvec(self):
        rng = np.random.default_rng(200)
        q, dth = rand_quats(rng, self.N), self._random_dtheta(rng, self.N)
        out = Q.inject_attitude_error(q, dth)
        np.testing.assert_allclose(out, Q.normalize(Q.multiply(q, Q.from_rotvec(dth))), atol=ATOL)

    def test_matches_scipy_composition(self):
        # q (x) exp(dtheta) is the right (body-frame) perturbation: scipy R * Rotvec.
        rng = np.random.default_rng(201)
        q, dth = rand_quats(rng, self.N), self._random_dtheta(rng, self.N)
        ref = sci_quat(sci(q) * Rotation.from_rotvec(dth))
        assert_same_rotation(Q.inject_attitude_error(q, dth), ref)

    def test_error_rotvec_round_trip(self):
        # error_rotvec(q, inject(q, d)) == d for |d| < pi (principal rotation vector).
        rng = np.random.default_rng(202)
        q, dth = rand_quats(rng, self.N), self._random_dtheta(rng, self.N)
        np.testing.assert_allclose(
            Q.error_rotvec(q, Q.inject_attitude_error(q, dth)), dth, atol=ATOL
        )

    def test_output_unit_norm_and_shape(self):
        rng = np.random.default_rng(203)
        q, dth = rand_quats(rng, self.N), self._random_dtheta(rng, self.N)
        out = Q.inject_attitude_error(q, dth)
        assert out.shape == (self.N, 4)
        np.testing.assert_allclose(np.linalg.norm(out, axis=-1), 1.0, atol=ATOL)

    def test_zero_error_is_identity_operation(self):
        rng = np.random.default_rng(204)
        q = rand_quats(rng, self.N)
        assert_same_rotation(Q.inject_attitude_error(q, np.zeros((self.N, 3))), q)

    def test_single_and_broadcast(self):
        rng = np.random.default_rng(205)
        q = rand_quats(rng, 1)[0]
        dth = self._random_dtheta(rng, 5)
        out = Q.inject_attitude_error(q, dth)             # (4,) with (5,3) -> (5,4)
        assert out.shape == (5, 4)
        for i in range(5):
            single = Q.inject_attitude_error(q, dth[i])
            assert single.shape == (4,)
            np.testing.assert_allclose(out[i], single, atol=ATOL)

    def test_sequential_injection_composes(self):
        # inject(inject(q, a), b) = q (x) exp(a) (x) exp(b): body-frame (right) composition.
        rng = np.random.default_rng(206)
        q = rand_quats(rng, self.N)
        a, b = self._random_dtheta(rng, self.N, 1.0), self._random_dtheta(rng, self.N, 1.0)
        out = Q.inject_attitude_error(Q.inject_attitude_error(q, a), b)
        ref = sci_quat(sci(q) * Rotation.from_rotvec(a) * Rotation.from_rotvec(b))
        assert_same_rotation(out, ref)


def _skew_oracle(v):
    """Explicit cross-product matrix (independent of Q.skew)."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


class TestAttitudeResetJacobian:
    H_FD = 1e-6   # central-difference step; truncation O(h^2) ~ 1e-12, rounding ~ 1e-10

    @staticmethod
    def _post_reset_error(q, dth_hat, dth_true):
        """Error of the post-injection estimate w.r.t. the truth q (x) exp(dth_true)."""
        q_true = Q.multiply(q, Q.from_rotvec(dth_true))
        return Q.error_rotvec(Q.inject_attitude_error(q, dth_hat), q_true)

    def _numeric_jacobian(self, q, dth_hat):
        """d(delta_theta_plus)/d(delta_theta) at delta_theta = dth_hat (central differences)."""
        J = np.empty((3, 3))
        for j in range(3):
            e = np.zeros(3)
            e[j] = self.H_FD
            fp = self._post_reset_error(q, dth_hat, dth_hat + e)
            fm = self._post_reset_error(q, dth_hat, dth_hat - e)
            J[:, j] = (fp - fm) / (2.0 * self.H_FD)
        return J

    def test_value_is_identity_minus_half_skew(self):
        rng = np.random.default_rng(210)
        dth = rng.normal(size=(10, 3))
        for d in dth:
            np.testing.assert_allclose(
                Q.attitude_reset_jacobian(d), np.eye(3) - 0.5 * _skew_oracle(d), atol=ATOL
            )

    def test_zero_gives_identity(self):
        np.testing.assert_array_equal(Q.attitude_reset_jacobian(np.zeros(3)), np.eye(3))

    def test_batched_shape_and_values(self):
        rng = np.random.default_rng(211)
        dth = rng.normal(size=(17, 3))
        G = Q.attitude_reset_jacobian(dth)
        assert G.shape == (17, 3, 3)
        for i in range(17):
            np.testing.assert_allclose(G[i], Q.attitude_reset_jacobian(dth[i]), atol=ATOL)
        assert Q.attitude_reset_jacobian(dth[0]).shape == (3, 3)

    @pytest.mark.parametrize("angle", [0.1, 0.01])
    def test_matches_finite_difference_oracle(self, angle):
        # Exact derivative of log(exp(-a) exp(x)) at x = a is the inverse right
        # Jacobian of -a: I - 1/2 [a x] + 1/12 [a x]^2 + O(|a|^4), so the first-order
        # G differs from it by at most |a|^2/12. Tolerance |a|^2/6 (2x margin on the
        # second-order term) plus 1e-7 for finite-difference noise.
        rng = np.random.default_rng(212)
        for _ in range(8):
            q = rand_quats(rng, 1)[0]
            dth_hat = angle * rand_unit_vecs(rng, 1)[0]
            J = self._numeric_jacobian(q, dth_hat)
            G = Q.attitude_reset_jacobian(dth_hat)
            tol = angle**2 / 6.0 + 1e-7
            assert np.max(np.abs(J - G)) < tol
            # the post-reset error is zero when the estimate is exact
            np.testing.assert_allclose(
                self._post_reset_error(q, dth_hat, dth_hat), 0.0, atol=1e-12
            )

    def test_wrong_sign_does_not_match(self):
        rng = np.random.default_rng(213)
        angle = 0.1
        for _ in range(8):
            q = rand_quats(rng, 1)[0]
            dth_hat = angle * rand_unit_vecs(rng, 1)[0]
            J = self._numeric_jacobian(q, dth_hat)
            G_wrong = np.eye(3) + 0.5 * _skew_oracle(dth_hat)
            # |J - G_wrong| ~ |[a x]| = |a| (max-abs entry ~ |a| * entry of unit skew),
            # while the correct G is within |a|^2/6. Demand >10x separation.
            err_right = np.max(np.abs(J - Q.attitude_reset_jacobian(dth_hat)))
            err_wrong = np.max(np.abs(J - G_wrong))
            assert err_wrong > 10.0 * err_right
            assert err_wrong > 0.2 * angle

    def test_error_shrinks_quadratically(self):
        # Residual of the first-order Jacobian is O(|a|^2): factor 10 in |a| -> ~100 in error.
        rng = np.random.default_rng(214)
        q = rand_quats(rng, 1)[0]
        u = rand_unit_vecs(rng, 1)[0]
        errs = []
        for a in (0.1, 0.01):
            J = self._numeric_jacobian(q, a * u)
            errs.append(np.max(np.abs(J - Q.attitude_reset_jacobian(a * u))))
        assert 50.0 < errs[0] / errs[1] < 200.0
