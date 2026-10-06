"""Tests for ``adcs.control`` (B-dot, nadir reference, rate-limited PD, momentum dumping).

Written from ``docs/specs/adcs_control.md`` sections 2 and 4 (tests 6-11), not from
the implementation.  Oracles: hand arithmetic and closed forms derived in the
comments, ``scipy.spatial.transform.Rotation`` (scalar-last quaternions, converted
here), ``adcs.orbit.propagate`` with central finite differences, an independent
RK4 loop built on ``adcs.dynamics.rk4_step``, and energy / momentum balances.
Spec thresholds are used as written.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from adcs import dynamics as dyn
from adcs.actuators import WheelConfig
from adcs.config import DEFAULT
from adcs.control import (
    PointingGains,
    bdot_dipole,
    bdot_gain,
    dump_active,
    dump_dipole,
    nadir_reference,
    pointing_torque,
)
from adcs.orbit import initial_state, orbital_period, propagate
from common.frames import GM_EARTH

MU = GM_EARTH
A_ORBIT = DEFAULT.orbit.a
INC = DEFAULT.orbit.inc
I_BOX = DEFAULT.spacecraft.inertia
IZZ = 4.0 * (0.10**2 + 0.10**2) / 12.0                  # exact Izz of the 4 kg 10x10x34 cm box
N_ORBIT = math.sqrt(MU / A_ORBIT**3)


# --------------------------------------------------------------------------
# helpers: scipy is the rotation oracle (scipy quaternions are [x, y, z, w])
# --------------------------------------------------------------------------
def to_scipy(q):
    q = np.asarray(q, float)
    return Rot.from_quat(np.concatenate([q[..., 1:], q[..., :1]], axis=-1))


def from_scipy(r):
    xyzw = r.as_quat()
    q = np.concatenate([xyzw[..., 3:], xyzw[..., :3]], axis=-1)
    return q if q[..., 0] >= 0 else -q


def rand_unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def rand_spd(rng, lo=0.005, hi=0.05):
    """Random symmetric positive definite 3x3 (a non-diagonal 'inertia')."""
    r = Rot.random(random_state=int(rng.integers(1 << 30))).as_matrix()
    return r @ np.diag(rng.uniform(lo, hi, 3)) @ r.T


def orbit_point(frac, j2=False):
    r0, v0 = initial_state(DEFAULT)
    t = frac * float(orbital_period(A_ORBIT))
    if t == 0.0:
        return r0, v0
    r, v = propagate(r0, v0, np.array([0.0, t]), j2=j2)
    return r[1], v[1]


# --------------------------------------------------------------------------
# 6. bdot_gain
# --------------------------------------------------------------------------
class TestBdotGain:
    def test_default_inertia_hand_value(self):
        k = bdot_gain(I_BOX, A_ORBIT, math.radians(51.6))
        exp = 2.0 * math.sqrt(MU / A_ORBIT**3) * (1.0 + math.sin(math.radians(51.6))) * IZZ
        assert k == pytest.approx(exp, rel=1e-12)
        # sanity on magnitude: n = 1.1e-3 rad/s, (1+sin) = 1.78, Izz = 6.67e-3 -> 2.6e-5 N m s
        assert 2.5e-5 < k < 2.7e-5

    def test_uses_smallest_eigenvalue_of_nondiagonal_inertia(self):
        rng = np.random.default_rng(6)
        for _ in range(20):
            lam = np.sort(rng.uniform(0.004, 0.05, 3))
            r = Rot.random(random_state=int(rng.integers(1 << 30))).as_matrix()
            inertia = r @ np.diag(lam) @ r.T
            inc = rng.uniform(0.1, 1.5)
            exp = 2.0 * N_ORBIT * (1.0 + math.sin(inc)) * lam[0]
            assert bdot_gain(inertia, A_ORBIT, inc) == pytest.approx(exp, rel=1e-12)

    def test_mu_and_a_dependence(self):
        a2 = 2.0 * A_ORBIT
        exp = 2.0 * math.sqrt(MU / a2**3) * (1.0 + math.sin(INC)) * IZZ
        assert bdot_gain(I_BOX, a2, INC) == pytest.approx(exp, rel=1e-12)
        mu2 = 3.0e14
        exp = 2.0 * math.sqrt(mu2 / A_ORBIT**3) * (1.0 + math.sin(INC)) * IZZ
        assert bdot_gain(I_BOX, A_ORBIT, INC, mu=mu2) == pytest.approx(exp, rel=1e-12)

    def test_equatorial_and_polar_inclination(self):
        k0 = bdot_gain(I_BOX, A_ORBIT, 0.0)
        k90 = bdot_gain(I_BOX, A_ORBIT, math.pi / 2)
        assert k0 == pytest.approx(2.0 * N_ORBIT * IZZ, rel=1e-12)
        assert k90 == pytest.approx(4.0 * N_ORBIT * IZZ, rel=1e-12)


# --------------------------------------------------------------------------
# 7. bdot_dipole
# --------------------------------------------------------------------------
class TestBdotDipole:
    def _rotating_pair(self, rng, w, dt):
        """b_prev random; b_now = body rotated by exp(w dt) in a constant inertial field."""
        b_prev = rand_unit(rng) * rng.uniform(2e-5, 6e-5)
        b_now = Rot.from_rotvec(w * dt).inv().apply(b_prev)       # R(q q_step)^T B_eci
        return b_now, b_prev

    def test_dissipation_exact_for_finite_dt_1000_cases(self):
        rng = np.random.default_rng(7)
        worst = -np.inf
        for _ in range(1000):
            wmag = 10.0 ** rng.uniform(-3.0, 0.0)                  # rad/s
            w = wmag * rand_unit(rng)
            dt = rng.uniform(0.01, 3.0) / wmag                      # |w| dt in (0.01, 3)
            k = 10.0 ** rng.uniform(-6.0, -3.0)
            m_max = 10.0 ** rng.uniform(-3.0, 0.0)
            b_now, b_prev = self._rotating_pair(rng, w, dt)
            m = bdot_dipole(b_now, b_prev, dt, k, m_max)
            power = float(w @ np.cross(m, b_now))                  # w . tau
            worst = max(worst, power)
            assert power <= 1e-25
        assert worst <= 1e-25

    def test_unsaturated_power_closed_form(self):
        # With b_now = exp(-w dt) b_prev (rotation about w by -|w| dt), derived by hand:
        #   w . (m x b_now) = -(k/dt) |w| |b_hat_perp|^2 sin(|w| dt),
        # b_hat_perp = component of b_hat_prev perpendicular to w.
        rng = np.random.default_rng(71)
        for _ in range(200):
            wmag = 10.0 ** rng.uniform(-3.0, 0.0)
            wh = rand_unit(rng)
            dt = rng.uniform(0.01, 3.0) / wmag
            k = 10.0 ** rng.uniform(-6.0, -3.0)
            b_now, b_prev = self._rotating_pair(rng, wmag * wh, dt)
            m = bdot_dipole(b_now, b_prev, dt, k, 1e9)              # unsaturated
            bh = b_prev / np.linalg.norm(b_prev)
            perp2 = 1.0 - (bh @ wh) ** 2
            exp = -(k / dt) * wmag * perp2 * math.sin(wmag * dt)
            got = float((wmag * wh) @ np.cross(m, b_now))
            assert got == pytest.approx(exp, rel=1e-8, abs=1e-30)

    def test_small_dt_limit_matches_cross_product_law(self):
        rng = np.random.default_rng(72)
        for _ in range(50):
            wmag = 10.0 ** rng.uniform(-2.0, 0.0)
            w = wmag * rand_unit(rng)
            dt = 1e-4 / wmag                                       # |w| dt = 1e-4
            k = 10.0 ** rng.uniform(-6.0, -3.0)
            b_now, b_prev = self._rotating_pair(rng, w, dt)
            m = bdot_dipole(b_now, b_prev, dt, k, 1e9)
            bhat = b_now / np.linalg.norm(b_now)
            exp = (k / np.linalg.norm(b_now)) * np.cross(w, bhat)   # m -> (k/|B|)(w x b_hat)
            assert np.linalg.norm(m - exp) <= 1e-3 * np.linalg.norm(exp)

    def test_torque_is_minus_k_w_perp_small_dt(self):
        rng = np.random.default_rng(73)
        w = 0.1 * rand_unit(rng)
        dt = 1e-5
        b_now, b_prev = self._rotating_pair(rng, w, dt)
        k = 2.6e-5
        m = bdot_dipole(b_now, b_prev, dt, k, 1e9)
        bh = b_now / np.linalg.norm(b_now)
        w_perp = w - (w @ bh) * bh
        tau = np.cross(m, b_now)
        np.testing.assert_allclose(tau, -k * w_perp, rtol=1e-3, atol=1e-3 * k * np.linalg.norm(w_perp))

    def test_saturation_scales_and_preserves_direction(self):
        rng = np.random.default_rng(74)
        w = 0.3 * rand_unit(rng)
        dt = 0.5
        b_now, b_prev = self._rotating_pair(rng, w, dt)
        k = 1e-3                                                    # large gain -> saturates
        m_free = bdot_dipole(b_now, b_prev, dt, k, 1e9)
        assert np.max(np.abs(m_free)) > 0.2
        m = bdot_dipole(b_now, b_prev, dt, k, 0.2)
        assert np.max(np.abs(m)) == pytest.approx(0.2, rel=1e-12)
        assert np.linalg.norm(np.cross(m, m_free)) <= 1e-12 * np.linalg.norm(m) * np.linalg.norm(m_free)
        assert m @ m_free > 0.0

    def test_unsaturated_below_limit_equals_free(self):
        rng = np.random.default_rng(75)
        w = 0.01 * rand_unit(rng)
        dt = 1.0
        b_now, b_prev = self._rotating_pair(rng, w, dt)
        m_free = bdot_dipole(b_now, b_prev, dt, 2.6e-5, 1e9)
        assert np.max(np.abs(m_free)) < 0.2
        np.testing.assert_array_equal(bdot_dipole(b_now, b_prev, dt, 2.6e-5, 0.2), m_free)

    def test_hand_formula_and_zero_for_constant_field(self):
        b_prev = np.array([3e-5, 0.0, 0.0])
        b_now = np.array([0.0, 4e-5, 0.0])
        # b_hat = [0, 1, 0], b_hat_prev = [1, 0, 0], |B| = 4e-5, dt = 2, k = 1e-5
        # m = -(1e-5 / 4e-5) * ([0,1,0] - [1,0,0]) / 2 = -0.125 * [-1, 1, 0] = [0.125, -0.125, 0]
        m = bdot_dipole(b_now, b_prev, 2.0, 1e-5, 1.0)
        np.testing.assert_allclose(m, [0.125, -0.125, 0.0], rtol=1e-13, atol=1e-16)
        # field unchanged in direction (even with different magnitude) -> no dipole
        m0 = bdot_dipole(2.0 * b_prev, b_prev, 1.0, 1e-5, 0.2)
        np.testing.assert_allclose(m0, np.zeros(3), atol=1e-15)

    def test_depends_on_direction_not_magnitude(self):
        rng = np.random.default_rng(76)
        b_prev = rand_unit(rng) * 3e-5
        b_now = rand_unit(rng) * 5e-5
        m1 = bdot_dipole(b_now, b_prev, 1.0, 2.6e-5, 1e9)
        # scaling the previous sample's magnitude changes nothing, scaling the current one scales 1/|B| only
        m2 = bdot_dipole(b_now, 7.0 * b_prev, 1.0, 2.6e-5, 1e9)
        np.testing.assert_allclose(m2, m1, rtol=1e-12)
        m3 = bdot_dipole(2.0 * b_now, b_prev, 1.0, 2.6e-5, 1e9)
        np.testing.assert_allclose(m3, m1 / 2.0, rtol=1e-12)


# --------------------------------------------------------------------------
# 8. B-dot kinetic energy (own RK4 loop, independent of adcs.simulate)
# --------------------------------------------------------------------------
class TestBdotEnergy:
    def test_kinetic_energy_non_increasing_and_h_along_b_conserved(self):
        rng = np.random.default_rng(8)
        inertia = I_BOX
        b_eci = 30e-6 * rand_unit(rng)                              # constant inertial field, 30 uT
        bhat = b_eci / np.linalg.norm(b_eci)
        q0 = from_scipy(Rot.random(random_state=8))
        w0 = math.radians(10.0) * rand_unit(rng)
        k = bdot_gain(inertia, A_ORBIT, INC)
        dt = 0.1
        n_steps = 2000

        def body_field(q):
            return to_scipy(q).inv().apply(b_eci)                   # R(q)^T B (scipy normalises q)

        y = np.concatenate([q0, w0, np.zeros(3)])
        b_prev = None
        m = np.zeros(3)
        energy = np.empty(n_steps + 1)
        h_dot_b = np.empty(n_steps + 1)
        for n in range(n_steps + 1):
            q, w = y[:4], y[4:7]
            energy[n] = 0.5 * w @ inertia @ w
            h_inertial = to_scipy(q).apply(inertia @ w)             # R(q)(I w), h_w = 0
            h_dot_b[n] = h_inertial @ bhat
            if n == n_steps:
                break
            b_now = body_field(q)
            if b_prev is not None:
                m = bdot_dipole(b_now, b_prev, dt, k, 0.2)          # held over the step
            b_prev = b_now

            def ext(_t, qq, _w, _h, m=m):
                return np.cross(m, body_field(qq))                  # re-evaluated per RK4 stage

            y = dyn.rk4_step(n * dt, y, dt, inertia, None, ext)

        t0 = energy[0]
        d_e = np.diff(energy)
        assert np.max(d_e) <= 1e-12 * t0, f"energy rose by {np.max(d_e) / t0:.3e} T0 in one step"
        assert energy[-1] < energy[0]                                # and it did dissipate
        h0_norm = np.linalg.norm(to_scipy(q0).apply(inertia @ w0))
        assert np.max(np.abs(h_dot_b - h_dot_b[0])) <= 1e-9 * h0_norm


# --------------------------------------------------------------------------
# 9. nadir_reference
# --------------------------------------------------------------------------
class TestNadirReference:
    @pytest.mark.parametrize("frac", [0.0, 0.1, 0.3, 0.55, 0.8])
    def test_q_ref_is_lvlh_aligned(self, frac):
        r, v = orbit_point(frac)
        q_ref, w_ref = nadir_reference(r, v)
        assert q_ref.shape == (4,) and w_ref.shape == (3,)
        assert np.linalg.norm(dyn.lvlh_error(q_ref, r, v)) <= 1e-12
        assert q_ref[0] >= 0.0
        assert np.linalg.norm(q_ref) == pytest.approx(1.0, abs=1e-14)
        # independent frame check (CONVENTIONS section 4): body axes in ECI are the LVLH axes
        z = -r / np.linalg.norm(r)
        h = np.cross(r, v)
        y = -h / np.linalg.norm(h)
        x = np.cross(y, z)
        rot = to_scipy(q_ref).as_matrix()
        np.testing.assert_allclose(rot[:, 0], x, atol=1e-13)
        np.testing.assert_allclose(rot[:, 1], y, atol=1e-13)
        np.testing.assert_allclose(rot[:, 2], z, atol=1e-13)

    def test_w_ref_closed_form_circular(self):
        r, v = orbit_point(0.37)
        _, w_ref = nadir_reference(r, v)
        # [0, -|r x v|/|r|^2, 0]; circular orbit: |r x v|/|r|^2 = n
        np.testing.assert_allclose(w_ref, [0.0, -N_ORBIT, 0.0], rtol=1e-9, atol=1e-15)
        exp_y = -np.linalg.norm(np.cross(r, v)) / (r @ r)
        assert w_ref[1] == pytest.approx(exp_y, rel=1e-14)

    def test_w_ref_is_plus_n_about_orbit_normal_in_eci(self):
        r, v = orbit_point(0.21)
        q_ref, w_ref = nadir_reference(r, v)
        w_eci = to_scipy(q_ref).apply(w_ref)
        h_hat = np.cross(r, v) / np.linalg.norm(np.cross(r, v))
        np.testing.assert_allclose(w_eci, N_ORBIT * h_hat, rtol=1e-9, atol=1e-15)

    def test_batched_inputs(self):
        pts = [orbit_point(f) for f in (0.0, 0.2, 0.4)]
        r = np.array([p[0] for p in pts])
        v = np.array([p[1] for p in pts])
        q_ref, w_ref = nadir_reference(r, v)
        assert q_ref.shape == (3, 4) and w_ref.shape == (3, 3)
        for i in range(3):
            qi, wi = nadir_reference(r[i], v[i])
            np.testing.assert_allclose(q_ref[i], qi, atol=1e-14)
            np.testing.assert_allclose(w_ref[i], wi, atol=1e-18)

    @staticmethod
    def _fd_rate_error(j2):
        r0, v0 = initial_state(DEFAULT)
        h = 0.5
        t_c = 100.0 + 600.0 * np.arange(10)                         # 10 points over ~1 orbit
        t = np.sort(np.concatenate([t_c - h, t_c, t_c + h]))
        r, v = propagate(r0, v0, np.concatenate([[0.0], t]), j2=j2)
        r, v = r[1:], v[1:]
        q, w_ref = nadir_reference(r, v)
        errs = []
        for i in range(10):
            lo, mid, hi = 3 * i, 3 * i + 1, 3 * i + 2
            q_lo, q_mid, q_hi = q[lo], q[mid], q[hi]
            # align quaternion signs with the centre sample (the reference is w >= 0 canonical)
            q_lo = q_lo if q_lo @ q_mid >= 0 else -q_lo
            q_hi = q_hi if q_hi @ q_mid >= 0 else -q_hi
            q_dot = (q_hi - q_lo) / (2.0 * h)
            # w = 2 vec(q* (x) q_dot), explicit Hamilton product of conj(q_mid) with q_dot
            a = np.array([q_mid[0], -q_mid[1], -q_mid[2], -q_mid[3]])
            b = q_dot
            vec = np.array([a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
                            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]])
            w_fd = 2.0 * vec
            errs.append(np.max(np.abs(w_fd - w_ref[mid])))
        return np.array(errs)

    def test_w_ref_matches_finite_difference_two_body(self):
        errs = self._fd_rate_error(j2=False)
        assert np.max(errs) < 1e-9

    def test_w_ref_matches_finite_difference_j2(self):
        # J2 adds an out-of-plane LVLH rotation (~1e-6 rad/s) that is not in w_ref
        errs = self._fd_rate_error(j2=True)
        assert np.max(errs) < 5e-6


# --------------------------------------------------------------------------
# 10. pointing_torque
# --------------------------------------------------------------------------
class TestPointingGains:
    def test_defaults(self):
        g = PointingGains()
        assert g.wn == pytest.approx(0.1, rel=1e-15)
        assert g.zeta == pytest.approx(0.7, rel=1e-15)
        assert g.rate_limit == pytest.approx(math.radians(0.5), rel=1e-14)

    def test_frozen_and_validation(self):
        with pytest.raises(Exception):
            PointingGains().wn = 1.0  # type: ignore[misc]


class TestPointingTorque:
    def test_a_exact_tracking_gives_pure_gyroscopic_feedforward(self):
        rng = np.random.default_rng(10)
        gains = PointingGains()
        for _ in range(5):
            q_ref = from_scipy(Rot.random(random_state=int(rng.integers(1 << 30))))
            w_ref = np.array([1.0e-3, -1.1e-3, 5.0e-4]) * rng.uniform(0.5, 1.5, 3)
            # q = q_ref -> delta q = identity -> w_ref_b = w_ref; w = w_ref_b -> dw = 0
            tau, err = pointing_torque(q_ref, w_ref, np.zeros(3), q_ref, w_ref, I_BOX, gains)
            np.testing.assert_allclose(err, np.zeros(3), atol=1e-15)
            expected = np.cross(w_ref, I_BOX @ w_ref)
            assert np.max(np.abs(tau - expected)) <= 1e-15
            assert np.linalg.norm(expected) > 1e-9               # not a vacuous zero comparison

    def test_b_small_angle_proportional_law(self):
        rng = np.random.default_rng(11)
        gains = PointingGains()
        for inertia in (I_BOX, rand_spd(rng)):
            d_theta = 1e-3 * rand_unit(rng)
            q = from_scipy(Rot.from_rotvec(d_theta))
            tau, err = pointing_torque(q, np.zeros(3), np.zeros(3),
                                       np.array([1.0, 0, 0, 0]), np.zeros(3), inertia, gains)
            np.testing.assert_allclose(err, d_theta, rtol=1e-12)
            expected = -gains.wn**2 * inertia @ d_theta
            np.testing.assert_allclose(tau, expected, rtol=1e-6, atol=0)

    def test_c_rate_saturation_90_deg(self):
        rng = np.random.default_rng(12)
        gains = PointingGains()
        for inertia in (I_BOX, rand_spd(rng)):
            e_hat = rand_unit(rng)
            q = from_scipy(Rot.from_rotvec(0.5 * math.pi * e_hat))
            tau, err = pointing_torque(q, np.zeros(3), np.zeros(3),
                                       np.array([1.0, 0, 0, 0]), np.zeros(3), inertia, gains)
            np.testing.assert_allclose(err, 0.5 * math.pi * e_hat, rtol=1e-12)
            # (wn / 2 zeta) * (pi/2) = 0.112 rad/s >> rate_limit = 0.00873 rad/s, so
            # w_cmd = -rate_limit e_hat and tau = Kd w_cmd = -2 zeta wn I rate_limit e_hat
            expected = -2.0 * gains.zeta * gains.wn * (inertia @ e_hat) * gains.rate_limit
            np.testing.assert_allclose(tau, expected, rtol=1e-12, atol=0)

    def test_c2_rate_limit_is_a_norm_saturation_and_changeable(self):
        # custom limit: the saturated command magnitude follows rate_limit, direction kept
        gains = PointingGains(wn=0.2, zeta=1.0, rate_limit=0.05)
        e_hat = np.array([1.0, 2.0, -2.0]) / 3.0
        q = from_scipy(Rot.from_rotvec(1.0 * e_hat))                  # 1 rad error: 0.1 rad/s > 0.05
        tau, _ = pointing_torque(q, np.zeros(3), np.zeros(3), np.array([1.0, 0, 0, 0]),
                                 np.zeros(3), I_BOX, gains)
        np.testing.assert_allclose(tau, -2.0 * 1.0 * 0.2 * (I_BOX @ e_hat) * 0.05, rtol=1e-12)
        # just below the limit: pure proportional law, no saturation
        q = from_scipy(Rot.from_rotvec(0.4 * e_hat))                  # 0.04 rad/s < 0.05
        tau, _ = pointing_torque(q, np.zeros(3), np.zeros(3), np.array([1.0, 0, 0, 0]),
                                 np.zeros(3), I_BOX, gains)
        np.testing.assert_allclose(tau, -(0.2**2) * (I_BOX @ (0.4 * e_hat)), rtol=1e-12)

    def test_d_err_matches_lvlh_error_and_scipy(self):
        rng = np.random.default_rng(13)
        gains = PointingGains()
        for i in range(20):
            r, v = orbit_point(rng.uniform(0.0, 1.0))
            q_ref, w_ref = nadir_reference(r, v)
            q = from_scipy(Rot.random(random_state=int(rng.integers(1 << 30))))
            w = 0.01 * rng.normal(size=3)
            _, err = pointing_torque(q, w, np.zeros(3), q_ref, w_ref, I_BOX, gains)
            np.testing.assert_allclose(err, dyn.lvlh_error(q, r, v), atol=1e-12)
            # scipy oracle: err = rotvec(R_ref^-1 R_q), angle <= pi (shortest rotation)
            err_sp = (to_scipy(q_ref).inv() * to_scipy(q)).as_rotvec()
            np.testing.assert_allclose(err, err_sp, atol=1e-12)
            assert np.linalg.norm(err) <= math.pi + 1e-12

    def test_d2_err_is_shortest_rotation_for_negated_quaternion(self):
        q_ref = np.array([1.0, 0.0, 0.0, 0.0])
        q = from_scipy(Rot.from_rotvec([0.0, 0.3, 0.0]))
        for sign in (1.0, -1.0):
            _, err = pointing_torque(sign * q, np.zeros(3), np.zeros(3), q_ref, np.zeros(3),
                                     I_BOX, PointingGains())
            np.testing.assert_allclose(err, [0.0, 0.3, 0.0], atol=1e-14)

    def test_e_wheel_momentum_enters_only_through_w_cross_h(self):
        rng = np.random.default_rng(14)
        gains = PointingGains()
        for _ in range(10):
            r, v = orbit_point(rng.uniform(0.0, 1.0))
            q_ref, w_ref = nadir_reference(r, v)
            q = from_scipy(Rot.random(random_state=int(rng.integers(1 << 30))))
            w = 0.02 * rng.normal(size=3)
            h = 1.5e-3 * rng.uniform(-1, 1, 3)
            tau0, err0 = pointing_torque(q, w, np.zeros(3), q_ref, w_ref, I_BOX, gains)
            tau1, err1 = pointing_torque(q, w, h, q_ref, w_ref, I_BOX, gains)
            np.testing.assert_array_equal(err1, err0)
            assert np.max(np.abs((tau1 - tau0) - np.cross(w, h))) <= 1e-15
            assert np.linalg.norm(np.cross(w, h)) > 1e-8

    def test_rate_term_hand_value(self):
        # q = q_ref = identity, w_ref = 0, w = [0.01, 0, 0], h = 0, err = 0, w_cmd = 0:
        # tau = -Kd w + w x (I w) = -2 zeta wn I w + 0 (w along a principal axis)
        gains = PointingGains()
        w = np.array([0.01, 0.0, 0.0])
        tau, _ = pointing_torque(np.array([1.0, 0, 0, 0]), w, np.zeros(3),
                                 np.array([1.0, 0, 0, 0]), np.zeros(3), I_BOX, gains)
        exp = np.array([-2.0 * 0.7 * 0.1 * I_BOX[0, 0] * 0.01, 0.0, 0.0])
        np.testing.assert_allclose(tau, exp, rtol=1e-12, atol=1e-18)

    def test_reference_rate_feedforward_hand_value(self):
        # q = q_ref (dq = I), w_ref_b = w_ref = [0, -n, 0], w = 0:
        #   dw = -w_ref, w_cmd = 0, tau = Kd w_ref + 0 - I (0 x w_ref) = 2 zeta wn I w_ref
        gains = PointingGains()
        w_ref = np.array([0.0, -1.1e-3, 0.0])
        q = np.array([1.0, 0, 0, 0])
        tau, _ = pointing_torque(q, np.zeros(3), np.zeros(3), q, w_ref, I_BOX, gains)
        np.testing.assert_allclose(tau, 2.0 * 0.7 * 0.1 * (I_BOX @ w_ref), rtol=1e-12)

    def test_unsaturated_closed_loop_error_dynamics_static_reference(self):
        # Static reference (q_ref = I, w_ref = 0): with the body feeling tau (u_w = -tau,
        # h_w = h0 arbitrary), I w_dot = -Kp err - Kd w exactly while unsaturated.
        rng = np.random.default_rng(15)
        gains = PointingGains()
        inertia = rand_spd(rng)
        d_theta = 0.02 * rand_unit(rng)
        q = from_scipy(Rot.from_rotvec(d_theta))
        w = 0.003 * rng.normal(size=3)
        h = 1e-3 * rng.uniform(-1, 1, 3)
        tau, err = pointing_torque(q, w, h, np.array([1.0, 0, 0, 0]), np.zeros(3), inertia, gains)
        y = np.concatenate([q, w, h])
        ydot = dyn.derivative(0.0, y, inertia, wheel_torque=-tau)
        kp = gains.wn**2 * inertia
        kd = 2.0 * gains.zeta * gains.wn * inertia
        expected = np.linalg.solve(inertia, -kp @ err - kd @ w)
        np.testing.assert_allclose(ydot[4:7], expected, rtol=1e-10, atol=1e-14)

    def test_closed_loop_tracks_rotating_reference(self):
        # Own reference: constant w_ref in reference axes, q_ref(t) = q0 (x) exp(w_ref t).
        # Continuous-time control through RK4 stages; the error must decay (time constant
        # 1/(zeta wn) = 14 s) to well below the initial 1e-2 rad.
        gains = PointingGains()
        inertia = I_BOX
        q_ref0 = from_scipy(Rot.random(random_state=16))
        w_ref = np.array([2.0e-4, -1.1e-3, 6.0e-4])

        def q_ref_at(t):
            return from_scipy(to_scipy(q_ref0) * Rot.from_rotvec(w_ref * t))

        offset = Rot.from_rotvec(np.array([0.6, -0.5, 0.4]) * 0.01)
        q0 = from_scipy(to_scipy(q_ref0) * offset)
        w0 = offset.inv().apply(w_ref)
        y = np.concatenate([q0, w0, np.zeros(3)])

        def u_fn(t, q, w, h):
            tau, _ = pointing_torque(q, w, h, q_ref_at(t), w_ref, inertia, gains)
            return -tau

        errs = []
        dt = 0.2
        for n in range(900):                                         # 180 s
            t = n * dt
            _, e = pointing_torque(y[:4], y[4:7], y[7:], q_ref_at(t), w_ref, inertia, gains)
            errs.append(np.linalg.norm(e))
            y = dyn.rk4_step(t, y, dt, inertia, u_fn, None)
        _, e = pointing_torque(y[:4], y[4:7], y[7:], q_ref_at(900 * dt), w_ref, inertia, gains)
        errs.append(np.linalg.norm(e))
        assert errs[0] == pytest.approx(0.01 * np.linalg.norm([0.6, -0.5, 0.4]), rel=1e-9)
        assert max(errs) < 2.0 * errs[0]                              # no wild transient
        assert errs[-1] < 2e-4 * errs[0]                              # e^{-0.07*180} = 3e-6 (+ margin)


# --------------------------------------------------------------------------
# 11. dump_dipole / dump_active
# --------------------------------------------------------------------------
class TestDumpDipole:
    def test_unsaturated_torque_is_minus_k_h_perp(self):
        rng = np.random.default_rng(17)
        for _ in range(100):
            b = rand_unit(rng) * rng.uniform(1.5e-5, 6e-5)
            h = rng.uniform(-1.77e-3, 1.77e-3, 3)
            k = 1e-3
            m = dump_dipole(h, b, k, 1e9)
            bh = b / np.linalg.norm(b)
            h_perp = h - (h @ bh) * bh
            tau = np.cross(m, b)
            np.testing.assert_allclose(tau, -k * h_perp, rtol=1e-12, atol=1e-16 * np.linalg.norm(h) * k)
            # explicit: m = (k/|B|^2) (h x B)
            np.testing.assert_allclose(m, (k / (b @ b)) * np.cross(h, b), rtol=1e-12, atol=1e-18)

    def test_m_perpendicular_to_b(self):
        rng = np.random.default_rng(18)
        for _ in range(100):
            b = rand_unit(rng) * 4e-5
            h = rng.uniform(-1.77e-3, 1.77e-3, 3)
            m = dump_dipole(h, b, 1e-3, 0.2)
            assert abs(m @ b) <= 1e-14 * np.linalg.norm(m) * np.linalg.norm(b)

    def test_parallel_h_gives_zero_dipole(self):
        b = np.array([1.0e-5, -2.0e-5, 3.0e-5])
        for c in (2.0e2, -1.0e2, 0.0):
            m = dump_dipole(c * b, b, 1e-3, 0.2)
            np.testing.assert_allclose(m, np.zeros(3), atol=1e-15)

    def test_h_target(self):
        rng = np.random.default_rng(19)
        b = rand_unit(rng) * 3e-5
        bh = b / np.linalg.norm(b)
        h = rng.uniform(-1e-3, 1e-3, 3)
        ht = rng.uniform(-5e-4, 5e-4, 3)
        k = 2e-3
        m = dump_dipole(h, b, k, 1e9, h_target=ht)
        d = h - ht
        np.testing.assert_allclose(np.cross(m, b), -k * (d - (d @ bh) * bh), rtol=1e-12, atol=1e-18)
        # target == h  ->  nothing to dump
        np.testing.assert_allclose(dump_dipole(h, b, k, 0.2, h_target=h), np.zeros(3), atol=1e-18)

    def test_saturation_via_scale(self):
        rng = np.random.default_rng(20)
        b = rand_unit(rng) * 3e-5
        h = rng.uniform(-1.5e-3, 1.5e-3, 3)
        m_free = dump_dipole(h, b, 10.0, 1e9)                       # absurd gain -> saturates
        assert np.max(np.abs(m_free)) > 0.2
        m = dump_dipole(h, b, 10.0, 0.2)
        assert np.max(np.abs(m)) == pytest.approx(0.2, rel=1e-12)
        assert np.linalg.norm(np.cross(m, m_free)) <= 1e-12 * np.linalg.norm(m) * np.linalg.norm(m_free)
        assert m @ m_free > 0.0
        assert abs(m @ b) <= 1e-14 * np.linalg.norm(m) * np.linalg.norm(b)


class TestDumpActive:
    @pytest.mark.parametrize(
        "prev, frac, expected",
        [
            (False, 0.31, True),      # above on -> on
            (True, 0.31, True),
            (False, 0.30, True),      # frac >= on_frac includes equality
            (True, 0.30, True),
            (False, 0.20, False),     # between: hold previous
            (True, 0.20, True),
            (False, 0.0501, False),
            (True, 0.0501, True),
            (True, 0.05, False),      # frac <= off_frac includes equality
            (False, 0.05, False),
            (True, 0.04, False),      # below off -> off
            (False, 0.04, False),
            (True, 0.0, False),
            (False, 5.0, True),
        ],
    )
    def test_truth_table(self, prev, frac, expected):
        out = dump_active(prev, frac, 0.3, 0.05)
        assert out is expected or out == expected
        assert bool(out) is expected

    def test_hysteresis_sequence(self):
        state = False
        seq = [0.1, 0.2, 0.35, 0.25, 0.1, 0.06, 0.04, 0.2, 0.1]
        expected = [False, False, True, True, True, True, False, False, False]
        got = []
        for f in seq:
            state = dump_active(state, f, 0.3, 0.05)
            got.append(bool(state))
        assert got == expected
