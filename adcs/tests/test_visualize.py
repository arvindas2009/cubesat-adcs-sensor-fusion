"""Tests for ``adcs.visualize`` (3D attitude animation, M11).

Written from ``docs/specs/visualizer.md`` (sections 1-4), not from the implementation.

Oracles
-------
* LVLH axes are rebuilt here from the definition of ``CONVENTIONS.md`` section 4
  (``z = -r/|r|``, ``y = -(r x v)/|r x v|``, ``x = y x z``) and never via
  ``common.frames.dcm_eci_to_lvlh``; the reference quaternion is
  ``Rotation.from_matrix([x y z])`` (body axes aligned with LVLH);
* attitude matrices come from ``scipy.spatial.transform.Rotation`` (same convention as the
  repository quaternions, ``CONVENTIONS.md`` section 2);
* the rotation angle of ``C`` for a perfect-knowledge run must equal the logged nadir error, and
  the angle of ``C_est^T C`` must equal ``|know_err|`` (``q_true = q_est (x) exp(know_err)``);
* eclipse flags are recomputed with the cylindrical-shadow definition written out by hand;
* ``frame_schedule`` is checked against the guarantees of spec section 2 on synthetic constant-rate
  and piecewise-rate rotations whose frame counts follow from closed-form arithmetic.

Matplotlib is forced to the Agg backend; MP4 tests are skipped without ``ffmpeg``.
"""
from __future__ import annotations

import copy
import dataclasses
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation as Rot

from adcs import visualize as viz
from adcs.config import DEFAULT
from adcs.environment import sun_direction_eci
from adcs.orbit import Ephemeris, initial_state
from adcs.simulate import (ClosedLoopResult, SimResult, lvlh_state, run, run_closed_loop,
                           tumble_state)
from common.frames import R_EARTH

REPO_ROOT = Path(__file__).resolve().parents[2]
MU = 3.986004418e14                       # m^3/s^2, WGS-84 GM
PERIOD = 2.0 * np.pi * np.sqrt(DEFAULT.orbit.a ** 3 / MU)

HAS_FFMPEG = shutil.which("ffmpeg") is not None


# ---------------------------------------------------------------------------
# independent helpers
# ---------------------------------------------------------------------------
def q_sf(rot: Rot) -> np.ndarray:
    """scipy Rotation -> scalar-first quaternion(s)."""
    return np.asarray(rot.as_quat())[..., [3, 0, 1, 2]]


def rot_of(q) -> Rot:
    """Scalar-first quaternion(s) -> scipy Rotation (same convention as CONVENTIONS section 2)."""
    q = np.asarray(q, float)
    return Rot.from_quat(q[..., [1, 2, 3, 0]])


def lvlh_axes(r, v) -> np.ndarray:
    """Matrix M (..., 3, 3) whose columns are the LVLH x, y, z axes in ECI (CONVENTIONS section 4)."""
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    z = -r / np.linalg.norm(r, axis=-1, keepdims=True)
    h = np.cross(r, v)
    y = -h / np.linalg.norm(h, axis=-1, keepdims=True)
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=-1)


def to_lvlh(M, a) -> np.ndarray:
    """Components of ECI vectors ``a`` along the LVLH axes: ``[a.x, a.y, a.z]``."""
    return np.einsum("...ji,...j->...i", M, np.asarray(a, float))


def random_orbit_state(rng, n):
    """n random (r, v) pairs, LEO-like magnitudes, v not exactly perpendicular to r."""
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = 6.9e6 * d
    w = rng.normal(size=(n, 3))
    w -= np.sum(w * d, axis=1, keepdims=True) * d * 0.9            # keep a radial part
    v = 7.5e3 * w / np.linalg.norm(w, axis=1, keepdims=True)
    return r, v


def random_quats(rng, n, scale=1.0):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return scale * q


def angle_of(C) -> np.ndarray:
    return Rot.from_matrix(C).magnitude()


def hand_sunlit(r, s) -> np.ndarray:
    """Cylindrical shadow written out by hand: dark iff r.s < 0 and distance to the axis < R_E."""
    along = np.sum(r * s, axis=-1)
    perp = np.linalg.norm(r - along[..., None] * s, axis=-1)
    return ~((along < 0.0) & (perp < R_EARTH))


def rot_series(t, theta, axis=(1.0, 2.0, -0.5)):
    """(N,3,3) rotation matrices about a fixed axis by angles ``theta`` (N,)."""
    ax = np.asarray(axis, float)
    ax = ax / np.linalg.norm(ax)
    return Rot.from_rotvec(np.asarray(theta)[:, None] * ax).as_matrix()


# ---------------------------------------------------------------------------
# 1. body_dcm_lvlh
# ---------------------------------------------------------------------------
class TestBodyDcmLvlh:
    def test_nadir_pointing_is_identity(self):
        rng = np.random.default_rng(1)
        r, v = random_orbit_state(rng, 6)
        q_ref = q_sf(Rot.from_matrix(lvlh_axes(r, v)))
        C = viz.body_dcm_lvlh(q_ref, r, v)
        assert C.shape == (6, 3, 3)
        np.testing.assert_allclose(C, np.broadcast_to(np.eye(3), (6, 3, 3)), atol=1e-12)

    def test_single_sample_shape_and_identity(self):
        rng = np.random.default_rng(2)
        r, v = random_orbit_state(rng, 1)
        r, v = r[0], v[0]
        q_ref = q_sf(Rot.from_matrix(lvlh_axes(r, v)))
        C = viz.body_dcm_lvlh(q_ref, r, v)
        assert C.shape == (3, 3)
        np.testing.assert_allclose(C, np.eye(3), atol=1e-12)

    def test_offset_from_reference_equals_rotvec_matrix(self):
        # q = q_ref (x) exp(phi)  =>  C = R(exp(phi))
        rng = np.random.default_rng(3)
        n = 25
        r, v = random_orbit_state(rng, n)
        phi = rng.normal(size=(n, 3)) * rng.uniform(0.05, 1.5, size=(n, 1))
        R_q = Rot.from_matrix(lvlh_axes(r, v)) * Rot.from_rotvec(phi)
        C = viz.body_dcm_lvlh(q_sf(R_q), r, v)
        np.testing.assert_allclose(C, Rot.from_rotvec(phi).as_matrix(), atol=1e-12)

    def test_columns_are_body_axes_in_lvlh(self):
        # body z axis in ECI = R(q) e_z; its LVLH components must be column 2 of C
        rng = np.random.default_rng(4)
        n = 10
        r, v = random_orbit_state(rng, n)
        q = random_quats(rng, n)
        C = viz.body_dcm_lvlh(q, r, v)
        M = lvlh_axes(r, v)
        R = rot_of(q).as_matrix()
        for k in range(3):
            axis_eci = R[:, :, k]
            np.testing.assert_allclose(C[:, :, k], to_lvlh(M, axis_eci), atol=1e-12)

    def test_batch_shapes(self):
        rng = np.random.default_rng(5)
        r, v = random_orbit_state(rng, 12)
        q = random_quats(rng, 12)
        C = viz.body_dcm_lvlh(q, r, v)
        assert C.shape == (12, 3, 3)
        # per-row evaluation equals the batched one
        for i in (0, 5, 11):
            np.testing.assert_allclose(viz.body_dcm_lvlh(q[i], r[i], v[i]), C[i], atol=1e-14)

    def test_nan_in_nan_out(self):
        rng = np.random.default_rng(6)
        r, v = random_orbit_state(rng, 5)
        q = random_quats(rng, 5)
        q[2] = np.nan
        C = viz.body_dcm_lvlh(q, r, v)          # must not raise
        assert C.shape == (5, 3, 3)
        assert np.all(np.isnan(C[2]))
        good = [0, 1, 3, 4]
        assert np.all(np.isfinite(C[good]))
        M = lvlh_axes(r[good], v[good])
        np.testing.assert_allclose(C[good], np.einsum("nji,njk->nik", M, rot_of(q[good]).as_matrix()),
                                   atol=1e-12)

    def test_orthonormal_det_plus_one_random(self):
        rng = np.random.default_rng(7)
        n = 200
        r, v = random_orbit_state(rng, n)
        q = random_quats(rng, n)
        C = viz.body_dcm_lvlh(q, r, v)
        np.testing.assert_allclose(np.einsum("nji,njk->nik", C, C), np.broadcast_to(np.eye(3), (n, 3, 3)),
                                   atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(C), 1.0, atol=1e-12)

    def test_quaternion_normalised_on_entry(self):
        rng = np.random.default_rng(8)
        r, v = random_orbit_state(rng, 8)
        q = random_quats(rng, 8)
        np.testing.assert_allclose(viz.body_dcm_lvlh(3.7 * q, r, v), viz.body_dcm_lvlh(q, r, v), atol=1e-12)

    def test_sign_of_quaternion_irrelevant(self):
        rng = np.random.default_rng(9)
        r, v = random_orbit_state(rng, 8)
        q = random_quats(rng, 8)
        np.testing.assert_allclose(viz.body_dcm_lvlh(-q, r, v), viz.body_dcm_lvlh(q, r, v), atol=1e-12)

    def test_inputs_not_modified(self):
        rng = np.random.default_rng(10)
        r, v = random_orbit_state(rng, 4)
        q = random_quats(rng, 4) * 2.0
        q0, r0, v0 = q.copy(), r.copy(), v.copy()
        viz.body_dcm_lvlh(q, r, v)
        assert np.array_equal(q, q0) and np.array_equal(r, r0) and np.array_equal(v, v0)


# ---------------------------------------------------------------------------
# 2. sun_lvlh
# ---------------------------------------------------------------------------
class TestSunLvlh:
    def test_equals_dot_products_with_lvlh_axes(self):
        rng = np.random.default_rng(11)
        n = 40
        r, v = random_orbit_state(rng, n)
        s = rng.normal(size=(n, 3))
        s /= np.linalg.norm(s, axis=1, keepdims=True)
        M = lvlh_axes(r, v)
        expect = np.stack([np.sum(s * M[:, :, k], axis=1) for k in range(3)], axis=1)
        out = viz.sun_lvlh(s, r, v)
        assert out.shape == (n, 3)
        np.testing.assert_allclose(out, expect, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-12)

    def test_single_sample(self):
        rng = np.random.default_rng(12)
        r, v = random_orbit_state(rng, 1)
        s = np.array([0.3, -0.5, 0.8])
        s /= np.linalg.norm(s)
        out = viz.sun_lvlh(s, r[0], v[0])
        assert out.shape == (3,)
        M = lvlh_axes(r[0], v[0])
        np.testing.assert_allclose(out, [s @ M[:, 0], s @ M[:, 1], s @ M[:, 2]], atol=1e-12)

    def test_special_directions_circular_orbit(self):
        # circular orbit r = a x^, v = V y^  =>  h^ = z^ (ECI), nadir = -x^, LVLH z = -r^, y = -h^, x = v^
        a, V = 7.0e6, 7.5e3
        r = np.array([a, 0.0, 0.0])
        v = np.array([0.0, V, 0.0])
        np.testing.assert_allclose(viz.sun_lvlh(-r / a, r, v), [0, 0, 1], atol=1e-12)        # nadir
        np.testing.assert_allclose(viz.sun_lvlh(v / V, r, v), [1, 0, 0], atol=1e-12)         # along track
        np.testing.assert_allclose(viz.sun_lvlh(np.array([0, 0, 1.0]), r, v), [0, -1, 0], atol=1e-12)
        np.testing.assert_allclose(viz.sun_lvlh(r / a, r, v), [0, 0, -1], atol=1e-12)        # zenith

    def test_unnormalised_input_is_normalised(self):
        rng = np.random.default_rng(13)
        r, v = random_orbit_state(rng, 5)
        s = rng.normal(size=(5, 3))
        out = viz.sun_lvlh(5.0 * s, r, v)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(out, viz.sun_lvlh(s / np.linalg.norm(s, axis=1, keepdims=True), r, v),
                                   atol=1e-12)


# ---------------------------------------------------------------------------
# 3. display frame
# ---------------------------------------------------------------------------
class TestDisplayFrame:
    def test_flip_is_proper_rotation(self):
        F = viz.DISPLAY_FLIP
        np.testing.assert_allclose(F, np.diag([1.0, -1.0, -1.0]), atol=0)
        assert np.linalg.det(F) == pytest.approx(1.0, abs=1e-14)
        np.testing.assert_allclose(F @ F.T, np.eye(3), atol=1e-15)
        # equals a 180 degree rotation about x
        np.testing.assert_allclose(F, Rot.from_rotvec([np.pi, 0, 0]).as_matrix(), atol=1e-15)

    def test_identity_attitude_body_z_points_down_and_equals_nadir(self):
        C = np.eye(3)
        body_z_display = viz.DISPLAY_FLIP @ C[:, 2]
        np.testing.assert_allclose(body_z_display, [0, 0, -1], atol=1e-15)
        nadir_lvlh = np.array([0.0, 0.0, 1.0])               # LVLH z = nadir
        np.testing.assert_allclose(viz.DISPLAY_FLIP @ nadir_lvlh, [0, 0, -1], atol=1e-15)
        np.testing.assert_allclose(viz.DISPLAY_FLIP @ C[:, 0], [1, 0, 0], atol=1e-15)
        # zenith (LVLH -z) is up in the display frame
        np.testing.assert_allclose(viz.DISPLAY_FLIP @ np.array([0.0, 0.0, -1.0]), [0, 0, 1], atol=1e-15)

    def test_perfect_pointing_end_to_end_in_display_frame(self):
        # q = q_ref  =>  C = I  =>  body z (long axis) is the display nadir (0,0,-1)
        rng = np.random.default_rng(14)
        r, v = random_orbit_state(rng, 1)
        q_ref = q_sf(Rot.from_matrix(lvlh_axes(r[0], v[0])))
        C = viz.body_dcm_lvlh(q_ref, r[0], v[0])
        np.testing.assert_allclose(viz.DISPLAY_FLIP @ C[:, 2], [0, 0, -1], atol=1e-12)
        # and the Sun at zenith shows up as display +Z
        s = r[0] / np.linalg.norm(r[0])
        np.testing.assert_allclose(viz.DISPLAY_FLIP @ viz.sun_lvlh(s, r[0], v[0]), [0, 0, 1], atol=1e-12)


# ---------------------------------------------------------------------------
# 4. frame_schedule
# ---------------------------------------------------------------------------
def _check_basic(idx, N, max_frames):
    idx = np.asarray(idx)
    assert np.issubdtype(idx.dtype, np.integer)
    assert idx.ndim == 1
    assert idx[0] == 0 and idx[-1] == N - 1
    assert np.all(np.diff(idx) > 0)
    assert idx.size <= max_frames
    assert idx.size >= 2


def _const_sun(N):
    return np.tile([1.0, 0.0, 0.0], (N, 1))


class TestFrameSchedule:
    def test_basic_guarantees_random_walk(self):
        rng = np.random.default_rng(20)
        N = 3000
        t = np.arange(N, dtype=float)
        theta = np.cumsum(np.abs(rng.normal(0, 0.02, N)))
        C = rot_series(t, theta)
        sun = _const_sun(N)
        for mf in (10, 50, 200, 600):
            idx = viz.frame_schedule(t, C, sun, max_frames=mf)
            _check_basic(idx, N, mf)

    def test_constant_rate_gives_uniform_frames(self):
        N = 2000
        t = np.arange(N, dtype=float)
        C = rot_series(t, 0.01 * t)                       # 0.01 rad/s about a fixed axis
        sun = _const_sun(N)
        idx = viz.frame_schedule(t, C, sun, max_frames=200)
        _check_basic(idx, N, 200)
        d = np.diff(idx)[:-1]                              # last interval may be shorter
        assert d.max() - d.min() <= 1                      # uniform within one sample
        assert idx.size >= 100                             # bisection uses (most of) the budget

    def test_constant_sun_rate_gives_uniform_frames(self):
        # constant attitude, Sun moving at a constant rate: motion comes from the Sun term only
        N = 1500
        t = np.arange(N, dtype=float)
        C = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
        ang = 0.005 * t
        sun = np.stack([np.cos(ang), np.sin(ang), np.zeros(N)], axis=1)
        idx = viz.frame_schedule(t, C, sun, max_frames=100)
        _check_basic(idx, N, 100)
        d = np.diff(idx)[:-1]
        assert d.max() - d.min() <= 1

    def test_all_indices_when_every_step_is_large(self):
        N = 100
        t = np.arange(N, dtype=float)
        C = rot_series(t, 0.5 * t)                         # 28.6 deg per step
        idx = viz.frame_schedule(t, C, _const_sun(N), max_frames=600)
        np.testing.assert_array_equal(idx, np.arange(N))

    def test_dt_max_respected_for_static_attitude(self):
        N = 1000
        t = np.arange(N, dtype=float)
        C = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
        for dt_max in (60.0, 30.0):
            idx = viz.frame_schedule(t, C, _const_sun(N), max_frames=600, dt_max=dt_max)
            _check_basic(idx, N, 600)
            assert np.diff(t[idx]).max() <= dt_max
            # no motion at all: frames come only from dt_max (every dt_max seconds) plus the last index
            np.testing.assert_array_equal(idx[:-1], np.arange(0, N, int(dt_max))[: idx.size - 1])

    def test_dt_max_respected_for_slow_motion(self):
        N = 4000
        t = np.arange(N, dtype=float)
        C = rot_series(t, 1e-6 * t)                        # almost static
        idx = viz.frame_schedule(t, C, _const_sun(N), max_frames=600, dt_max=45.0)
        _check_basic(idx, N, 600)
        assert np.diff(t[idx]).max() <= 45.0

    def test_dt_min_respected_when_motion_is_large(self):
        N = 200
        t = np.arange(N, dtype=float)
        C = rot_series(t, 0.5 * t)                         # every step exceeds any trial angle
        idx = viz.frame_schedule(t, C, _const_sun(N), max_frames=600, dt_min=5.0)
        _check_basic(idx, N, 600)
        d = np.diff(idx)
        assert np.all(d[:-1] >= 5)                         # last frame is forced and may come earlier
        np.testing.assert_array_equal(idx[:-1], np.arange(0, 200, 5)[: idx.size - 1])

    def test_fast_segment_gets_more_frames_than_slow_segment(self):
        N = 2000
        t = np.arange(N, dtype=float)
        rate = np.where(t < 1000.0, 0.02, 0.002)           # rad/s, 10x slower in the second half
        theta = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))])
        C = rot_series(t, theta)
        idx = viz.frame_schedule(t, C, _const_sun(N), max_frames=100)
        _check_basic(idx, N, 100)
        n_fast = int(np.sum(t[idx] < 1000.0))
        n_slow = int(np.sum(t[idx] >= 1000.0))
        # dt_max = 60 s gives the slow half >= 16 frames, so at most ~84 remain for the 10x faster half:
        # the fast half is sampled several times more densely per second than the slow half.
        assert n_fast > 2 * n_slow
        assert n_slow >= 15

    def test_fast_sun_segment_gets_more_frames(self):
        N = 2000
        t = np.arange(N, dtype=float)
        C = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
        rate = np.where(t < 1000.0, 0.02, 0.001)
        ang = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))])
        sun = np.stack([np.cos(ang), np.sin(ang), np.zeros(N)], axis=1)
        idx = viz.frame_schedule(t, C, sun, max_frames=100)
        _check_basic(idx, N, 100)
        assert np.sum(t[idx] < 1000.0) > 2 * np.sum(t[idx] >= 1000.0)

    def test_monotone_in_max_frames(self):
        rng = np.random.default_rng(21)
        N = 5000
        t = np.arange(N, dtype=float)
        theta = np.cumsum(np.abs(rng.normal(0, 0.01, N)) * (1 + 5 * (np.arange(N) % 1000 < 200)))
        C = rot_series(t, theta)
        sun = _const_sun(N)
        counts = []
        for mf in (20, 50, 100, 200, 400, 800):
            idx = viz.frame_schedule(t, C, sun, max_frames=mf)
            _check_basic(idx, N, mf)
            counts.append(idx.size)
        assert counts == sorted(counts)
        assert counts[-1] > counts[0]

    def test_subsampling_when_even_90_deg_gives_too_many_frames(self):
        N = 100
        t = np.arange(N, dtype=float)
        C = rot_series(t, 2.0 * t)                         # 114.6 deg per step > 90 deg
        idx = viz.frame_schedule(t, C, _const_sun(N), max_frames=10)
        _check_basic(idx, N, 10)
        assert idx.size == 10
        d = np.diff(idx)
        assert d.max() - d.min() <= 1                      # uniform by index

    def test_two_samples(self):
        t = np.array([0.0, 5.0])
        C = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
        idx = viz.frame_schedule(t, C, _const_sun(2), max_frames=600)
        np.testing.assert_array_equal(idx, [0, 1])

    def test_nan_motion_counts_as_zero(self):
        N = 500
        t = np.arange(N, dtype=float)
        C = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
        C[100:150] = np.nan
        idx = viz.frame_schedule(t, C, _const_sun(N), max_frames=600, dt_max=60.0)   # must not raise
        _check_basic(idx, N, 600)
        assert np.diff(t[idx]).max() <= 60.0

    def test_inputs_unmodified_and_deterministic(self):
        rng = np.random.default_rng(22)
        N = 800
        t = np.arange(N, dtype=float)
        C = rot_series(t, np.cumsum(np.abs(rng.normal(0, 0.03, N))))
        sun = _const_sun(N) + 0.0
        t0, C0, s0 = t.copy(), C.copy(), sun.copy()
        a = viz.frame_schedule(t, C, sun, max_frames=60)
        b = viz.frame_schedule(t, C, sun, max_frames=60)
        np.testing.assert_array_equal(a, b)
        assert np.array_equal(t, t0) and np.array_equal(C, C0) and np.array_equal(sun, s0)

    def test_large_grid_is_fast(self):
        # spec section 3: N ~ 4e4 with ~30 bisection iterations in a few seconds
        import time
        N = 40000
        t = np.arange(N, dtype=float)
        C = rot_series(t, 0.003 * t)
        t_start = time.perf_counter()
        idx = viz.frame_schedule(t, C, _const_sun(N), max_frames=600)
        elapsed = time.perf_counter() - t_start
        _check_basic(idx, N, 600)
        assert elapsed < 10.0


# ---------------------------------------------------------------------------
# 5. render
# ---------------------------------------------------------------------------
FPS = 2          # small fps/dpi keep the render tests fast (each render ~4 s)
DPI = 30


def synthetic_data(with_est=True, with_know=True, all_nan_est=False, n=12) -> "viz.AnimationData":
    t = np.linspace(0.0, 6000.0, n)
    rv = np.array([0.3, -0.2, 0.9])
    theta = np.linspace(0.0, 2.5, n)
    C = Rot.from_rotvec(theta[:, None] * rv / np.linalg.norm(rv)).as_matrix()
    ang = np.linspace(0.0, 1.0, n)
    sun = np.stack([np.cos(ang), np.sin(ang), 0.2 * np.ones(n)], axis=1)
    sun /= np.linalg.norm(sun, axis=1, keepdims=True)
    lit = np.arange(n) < n // 2                                  # lit -> eclipse change
    mode = (np.arange(n) >= n // 3).astype(int)                  # detumble -> pointing
    nadir_err = np.geomspace(0.8, 1e-3, n)
    C_est = know_err = None
    if with_est:
        dC = Rot.from_rotvec(np.array([0.02, -0.01, 0.015])).as_matrix()
        C_est = np.einsum("nij,jk->nik", C, dC.T)
        C_est[:3] = np.nan                                       # before MEKF initialisation
        if all_nan_est:
            C_est[:] = np.nan
    if with_know:
        know_err = np.full(n, 0.0333)
        know_err[:3] = np.nan
        if all_nan_est:
            know_err[:] = np.nan
    return viz.AnimationData(t=t, C=C, sun=sun, lit=lit, mode=mode, nadir_err=nadir_err,
                             C_est=C_est, know_err=know_err, period=float(PERIOD))


def gif_trailing_hold_ms(path) -> float:
    """Total display time of the trailing run of frames identical to the final frame."""
    with Image.open(path) as im:
        frames, durs = [], []
        for i in range(im.n_frames):
            im.seek(i)
            frames.append(np.asarray(im.convert("RGB")).copy())
            durs.append(float(im.info.get("duration", 0)))
    last = frames[-1]
    total = 0.0
    for f, d in zip(reversed(frames), reversed(durs)):
        if not np.array_equal(f, last):
            break
        total += d
    return total


@pytest.fixture(scope="module")
def full_gif(tmp_path_factory):
    out = tmp_path_factory.mktemp("viz") / "sub" / "dir" / "full.gif"      # parents must be created
    data = synthetic_data()
    ret = viz.render(data, out, fps=FPS, dpi=DPI, title="synthetic")
    return out, ret


class TestRender:
    def test_gif_written_with_frames(self, full_gif):
        out, ret = full_gif
        assert Path(ret) == out
        assert isinstance(ret, Path)
        assert out.exists() and out.stat().st_size > 0
        with Image.open(out) as im:
            assert im.n_frames >= 12

    def test_gif_loops_forever_and_has_expected_geometry(self, full_gif):
        out, _ = full_gif
        with Image.open(out) as im:
            w, h = im.size
            assert w % 2 == 0 and h % 2 == 0                     # required by h264
            assert w / h == pytest.approx(2.0, rel=0.03)         # ~ 12 x 6 in
            assert w == pytest.approx(12 * DPI, rel=0.05)
            assert im.info.get("loop", None) == 0                # loop forever

    def test_gif_last_frame_held_at_least_one_second(self, full_gif):
        out, _ = full_gif
        hold = gif_trailing_hold_ms(out)
        # GIF delays are stored in centiseconds: allow a rounding slack of a few cs
        assert hold >= 1000.0 - 40.0

    def test_gif_without_estimate(self, tmp_path):
        out = tmp_path / "noest.gif"
        viz.render(synthetic_data(with_est=False, with_know=False), out, fps=FPS, dpi=DPI)
        assert out.stat().st_size > 0
        with Image.open(out) as im:
            assert im.n_frames >= 12

    def test_render_does_not_modify_data_and_is_deterministic(self, full_gif, tmp_path):
        out1, _ = full_gif
        data = synthetic_data()
        snapshot = copy.deepcopy(data)
        out2 = tmp_path / "again.gif"
        viz.render(data, out2, fps=FPS, dpi=DPI, title="synthetic")
        for f in dataclasses.fields(data):
            a, b = getattr(data, f.name), getattr(snapshot, f.name)
            if isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b)        # NaN-aware equality
            else:
                assert a == b
        with Image.open(out1) as a, Image.open(out2) as b:
            assert a.n_frames == b.n_frames
            assert a.size == b.size

    @pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not on PATH")
    def test_mp4_written(self, tmp_path):
        out = tmp_path / "anim.mp4"
        # all-NaN estimate (never initialised) and no title: nothing to draw for the dashed axes
        ret = viz.render(synthetic_data(all_nan_est=True), out, fps=FPS, dpi=DPI, title=None)
        assert Path(ret) == out
        assert out.stat().st_size > 0
        with open(out, "rb") as fh:
            head = fh.read(12)
        assert head[4:8] == b"ftyp"                           # ISO base media (MP4) container

    def test_unknown_extension_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError):
            viz.render(synthetic_data(), tmp_path / "anim.avi", fps=FPS, dpi=DPI)

    def test_missing_ffmpeg_raises_runtime_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda *a, **k: None)
        with pytest.raises(RuntimeError) as exc:
            viz.render(synthetic_data(), tmp_path / "anim.mp4", fps=FPS, dpi=DPI)
        msg = str(exc.value).lower()
        assert "ffmpeg" in msg
        assert "gif" in msg                                   # suggests the .gif alternative


# ---------------------------------------------------------------------------
# 6. build_animation_data
# ---------------------------------------------------------------------------
T_SHORT = 300.0
OFFSET = np.deg2rad([5.0, -3.0, 4.0])


@pytest.fixture(scope="module")
def sim_res() -> SimResult:
    r0, v0 = initial_state(DEFAULT)
    return run(lvlh_state(r0, v0, offset=OFFSET), T_SHORT, "pointing")


@pytest.fixture(scope="module")
def cl_res() -> ClosedLoopResult:
    return run_closed_loop(tumble_state(np.deg2rad(2.0), 0), T_SHORT, seed=0)


def _frame_index(res, data) -> np.ndarray:
    idx = np.searchsorted(res.t, data.t)
    assert np.all(idx < res.t.size)
    np.testing.assert_array_equal(res.t[idx], data.t)         # frames are samples of the full grid
    return idx


def _oracle_rv_sun(t):
    eph = Ephemeris(*initial_state(DEFAULT), t[-1] + 10.0)
    r, v = eph.at(t)
    s = sun_direction_eci(DEFAULT.jd_epoch + t / 86400.0)
    return np.asarray(r), np.asarray(v), np.asarray(s)


def _check_common(res, data, max_frames):
    N = res.t.size
    M = data.t.size
    assert 2 <= M <= max_frames
    assert data.C.shape == (M, 3, 3)
    assert data.sun.shape == (M, 3)
    assert data.lit.shape == (M,) and data.lit.dtype == bool
    assert data.mode.shape == (M,)
    assert data.nadir_err.shape == (M,)
    assert data.t[0] == res.t[0] and data.t[-1] == res.t[-1] and np.all(np.diff(data.t) > 0)
    assert N >= M
    idx = _frame_index(res, data)
    np.testing.assert_array_equal(data.mode, res.mode[idx])
    np.testing.assert_allclose(data.nadir_err, res.nadir_err[idx], atol=1e-12)
    assert data.period == pytest.approx(PERIOD, rel=1e-9)

    # C = R(q_ref)^T R(q), independent LVLH axes
    r, v, s = _oracle_rv_sun(res.t[idx])
    Mx = lvlh_axes(r, v)
    C_expect = np.einsum("nji,njk->nik", Mx, rot_of(res.q[idx]).as_matrix())
    np.testing.assert_allclose(data.C, C_expect, atol=1e-9)
    np.testing.assert_allclose(np.einsum("nji,njk->nik", data.C, data.C),
                               np.broadcast_to(np.eye(3), (M, 3, 3)), atol=1e-12)
    # rotation angle of C is the nadir error
    np.testing.assert_allclose(angle_of(data.C), res.nadir_err[idx], atol=1e-7)
    # Sun in LVLH axes
    np.testing.assert_allclose(data.sun, to_lvlh(Mx, s), atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(data.sun, axis=1), 1.0, atol=1e-12)
    return idx, r, v, s


class TestBuildAnimationData:
    def test_sim_result_perfect_pointing(self, sim_res):
        data = viz.build_animation_data(sim_res, max_frames=50)
        idx, r, v, s = _check_common(sim_res, data, 50)
        assert data.C_est is None and data.know_err is None
        # lit is computed from ephemeris + Sun for a SimResult (hand-written cylindrical shadow)
        np.testing.assert_array_equal(data.lit, hand_sunlit(r, s))
        # initial offset (5, -3, 4) deg => C(0) is the rotation exp(OFFSET) (nadir_err[0] = |OFFSET|)
        np.testing.assert_allclose(data.C[0], Rot.from_rotvec(OFFSET).as_matrix(), atol=1e-9)
        assert data.nadir_err[0] == pytest.approx(np.linalg.norm(OFFSET), abs=1e-9)
        # pointing run: C stays close to I (rotation angle = nadir error, bounded by the offset)
        assert np.all(angle_of(data.C) <= np.linalg.norm(OFFSET) + 1e-6)
        assert np.all(np.linalg.norm(data.C - np.eye(3), ord=2, axis=(1, 2)) <= data.nadir_err + 1e-9)

    def test_closed_loop_result(self, cl_res):
        data = viz.build_animation_data(cl_res, max_frames=50)
        idx, r, v, s = _check_common(cl_res, data, 50)
        np.testing.assert_array_equal(data.lit, cl_res.lit[idx])
        np.testing.assert_array_equal(data.lit, hand_sunlit(r, s))
        M = data.t.size
        assert data.C_est is not None and data.C_est.shape == (M, 3, 3)
        assert data.know_err is not None and data.know_err.shape == (M,)
        ok = np.isfinite(cl_res.q_est[idx, 0])
        assert ok.any()
        # C_est = R(q_ref)^T R(q_est) on finite rows, NaN elsewhere
        Mx = lvlh_axes(r, v)
        Ce = np.einsum("nji,njk->nik", Mx[ok], rot_of(cl_res.q_est[idx][ok]).as_matrix())
        np.testing.assert_allclose(data.C_est[ok], Ce, atol=1e-9)
        assert np.all(np.isnan(data.C_est[~ok]))
        # know_err is the *angle* of the logged rotation-vector error
        ke = np.linalg.norm(cl_res.know_err[idx], axis=1)
        np.testing.assert_allclose(data.know_err, ke, atol=1e-12, equal_nan=True)
        # independent: q_true = q_est (x) exp(know_err)  =>  angle(C_est^T C) = |know_err|
        rel = np.einsum("nji,njk->nik", data.C_est[ok], data.C[ok])
        np.testing.assert_allclose(angle_of(rel), data.know_err[ok], atol=1e-6)

    def test_max_frames_is_honoured_and_schedule_is_used(self, sim_res):
        for mf in (10, 40):
            data = viz.build_animation_data(sim_res, max_frames=mf)
            assert 2 <= data.t.size <= mf
            assert data.t[0] == sim_res.t[0] and data.t[-1] == sim_res.t[-1]
        # the selected frames are those of frame_schedule on the full grid
        full = viz.build_animation_data(sim_res, max_frames=10)
        r, v, s = _oracle_rv_sun(sim_res.t)
        Mx = lvlh_axes(r, v)
        C_all = np.einsum("nji,njk->nik", Mx, rot_of(sim_res.q).as_matrix())
        idx_expect = viz.frame_schedule(sim_res.t, C_all, to_lvlh(Mx, s), max_frames=10)
        np.testing.assert_array_equal(full.t, sim_res.t[idx_expect])

    def test_deterministic_and_inputs_untouched(self, cl_res):
        t0, q0, qe0 = cl_res.t.copy(), cl_res.q.copy(), cl_res.q_est.copy()
        a = viz.build_animation_data(cl_res, max_frames=30)
        b = viz.build_animation_data(cl_res, max_frames=30)
        for f in dataclasses.fields(a):
            x, y = getattr(a, f.name), getattr(b, f.name)
            if isinstance(x, np.ndarray):
                np.testing.assert_array_equal(x, y)
            else:
                assert x == y
        assert np.array_equal(cl_res.t, t0) and np.array_equal(cl_res.q, q0)
        assert np.array_equal(cl_res.q_est, qe0, equal_nan=True)


def _fake_sim(t, q, nadir_err=None, mode=None) -> dict:
    """Keyword arguments for a hand-built SimResult/ClosedLoopResult with the base fields only."""
    N = t.size
    z3 = np.zeros((N, 3))
    return dict(t=t, q=q, w=z3.copy(), h_w=z3.copy(), wheel_speed=z3.copy(), m=z3.copy(), u_w=z3.copy(),
                tau_dist=z3.copy(), tau_mag=z3.copy(),
                nadir_err=np.zeros(N) if nadir_err is None else nadir_err,
                rate=np.zeros(N), mode=np.zeros(N, int) if mode is None else mode,
                dump=np.zeros(N, bool), meta={})


class TestBuildAnimationDataFullOrbit:
    """Hand-built results spanning more than one orbit (contains an eclipse)."""

    def test_sim_result_eclipse_flag_and_attitude(self):
        rng = np.random.default_rng(30)
        t = np.arange(0.0, 6000.0, 60.0)
        q = random_quats(rng, t.size)
        nadir_err = rng.uniform(0.0, 3.0, t.size)
        mode = (np.arange(t.size) > 40).astype(int)
        res = SimResult(**_fake_sim(t, q, nadir_err, mode))
        data = viz.build_animation_data(res, max_frames=600)
        idx = _frame_index(res, data)
        r, v, s = _oracle_rv_sun(t[idx])
        lit_expect = hand_sunlit(r, s)
        np.testing.assert_array_equal(data.lit, lit_expect)
        assert lit_expect.any() and (~lit_expect).any()       # the orbit has sunlit and eclipse parts
        np.testing.assert_array_equal(data.mode, mode[idx])
        np.testing.assert_allclose(data.nadir_err, nadir_err[idx], atol=1e-14)
        Mx = lvlh_axes(r, v)
        np.testing.assert_allclose(data.C, np.einsum("nji,njk->nik", Mx, rot_of(q[idx]).as_matrix()), atol=1e-9)
        np.testing.assert_allclose(data.sun, to_lvlh(Mx, s), atol=1e-9)

    def test_closed_loop_result_nan_estimate_before_init(self):
        rng = np.random.default_rng(31)
        t = np.arange(0.0, 6000.0, 60.0)
        N = t.size
        q = random_quats(rng, N)
        ke = rng.normal(0.0, 0.01, (N, 3))                     # rad, q_true = q_est (x) exp(ke)
        q_est = q_sf(rot_of(q) * Rot.from_rotvec(ke).inv())
        q_est[:10] = np.nan
        ke_log = ke.copy()
        ke_log[:10] = np.nan
        r, v, s = _oracle_rv_sun(t)
        lit_log = hand_sunlit(r, s)
        base = _fake_sim(t, q)
        res = ClosedLoopResult(
            **base, q_est=q_est, b_est=np.zeros((N, 3)), bm_est=np.zeros((N, 3)), att_sigma=np.zeros(N),
            P_att=np.zeros((N, 3, 3)), know_err=ke_log, est_nadir_err=np.zeros(N), w_est=np.zeros((N, 3)),
            gyro_bias=np.zeros((N, 3)), sun_used=np.zeros(N, int), lit=lit_log)
        data = viz.build_animation_data(res, max_frames=600)
        idx = _frame_index(res, data)
        pre = idx < 10
        assert pre.any() and (~pre).any()
        assert np.all(np.isnan(data.C_est[pre]))
        assert np.all(np.isnan(data.know_err[pre]))
        assert np.all(np.isfinite(data.C_est[~pre]))
        np.testing.assert_allclose(data.know_err[~pre], np.linalg.norm(ke[idx][~pre], axis=1), atol=1e-12)
        np.testing.assert_array_equal(data.lit, lit_log[idx])
        Mx = lvlh_axes(r[idx], v[idx])
        np.testing.assert_allclose(data.C_est[~pre],
                                   np.einsum("nji,njk->nik", Mx[~pre], rot_of(q_est[idx][~pre]).as_matrix()),
                                   atol=1e-9)


# ---------------------------------------------------------------------------
# 7. module hygiene
# ---------------------------------------------------------------------------
class TestModuleHygiene:
    def test_all_lists_public_api(self):
        for name in ("DISPLAY_FLIP", "body_dcm_lvlh", "sun_lvlh", "frame_schedule", "AnimationData",
                     "build_animation_data", "render", "main"):
            assert name in viz.__all__, name
            assert hasattr(viz, name)

    def test_import_has_no_side_effects(self):
        # a fresh interpreter: importing the module must not import ffmpeg bindings or show a window
        code = ("import sys, adcs.visualize as v; "
                "bad = [m for m in sys.modules if 'ffmpeg' in m.lower()]; "
                "assert not bad, bad; print('ok')")
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True,
                             text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip().endswith("ok")
