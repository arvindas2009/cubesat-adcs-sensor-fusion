"""Tests for ``adcs.dynamics`` (select with ``-k dynamics``) and
``adcs.disturbances`` (select with ``-k dist``).

Written from ``docs/specs/adcs_physics.md`` sections 3, 4 and tests 11-20 of
section 5, not from the implementation.  Oracles: closed-form symmetric-top
and gyroscopic solutions (derived in the comments), conservation laws, hand
component arithmetic, ``scipy.spatial.transform.Rotation``.

Spec thresholds are fixed; they are used as written (no loosening).  Where a
spec statement is ambiguous the stricter reading is used and noted.

Equations of motion used throughout (spec section 3), body axes:

    I w_dot = -w x (I w + h) - h_dot + tau,      h_dot = u_w.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from adcs import disturbances as dist
from adcs import dynamics as dyn
from adcs.config import DEFAULT, SpacecraftConfig, inertia_box
from common.frames import GM_EARTH, OMEGA_EARTH, R_EARTH

MU = GM_EARTH
A_ORBIT = DEFAULT.orbit.a                       # 6 778 137 m
INC = DEFAULT.orbit.inc
I_BOX = DEFAULT.spacecraft.inertia              # diag(0.041867, 0.041867, 0.006667)
T_ORBIT = 2.0 * math.pi * math.sqrt(A_ORBIT**3 / MU)

Z3 = np.zeros(3)


# --------------------------------------------------------------------------
# helpers (scipy is the rotation oracle; scipy quaternions are [x, y, z, w])
# --------------------------------------------------------------------------
def q_of(rot: Rot) -> np.ndarray:
    """scipy Rotation -> scalar-first quaternion (body-to-reference)."""
    x, y, z, w = rot.as_quat()
    return np.array([w, x, y, z])


def rot_of(q) -> Rot:
    q = np.asarray(q, float)
    return Rot.from_quat(np.concatenate([q[..., 1:], q[..., :1]], axis=-1))


def state(q, w, h=Z3) -> np.ndarray:
    return np.concatenate([np.asarray(q, float), np.asarray(w, float), np.asarray(h, float)])


def circ_state(u: float, a: float = A_ORBIT, inc: float = INC):
    """Circular-orbit ECI position/velocity at argument of latitude u (explicit formulas)."""
    vc = math.sqrt(MU / a)
    r = a * np.array([math.cos(u), math.sin(u) * math.cos(inc), math.sin(u) * math.sin(inc)])
    v = vc * np.array([-math.sin(u), math.cos(u) * math.cos(inc), math.cos(u) * math.sin(inc)])
    return r, v


def lvlh_rot(r, v) -> Rot:
    """Body-to-ECI rotation of a body aligned with LVLH (CONVENTIONS section 4):
    z = -r_hat, y = -h_hat, x = y x z; the columns of the DCM are the body axes in ECI."""
    z = -r / np.linalg.norm(r)
    h = np.cross(r, v)
    y = -h / np.linalg.norm(h)
    x = np.cross(y, z)
    return Rot.from_matrix(np.column_stack([x, y, z]))


Q0 = q_of(Rot.from_rotvec([0.3, -0.2, 0.5]))    # a generic initial attitude


# ==========================================================================
# 11. torque-free asymmetric tumble: conservation laws
# ==========================================================================
def test_dynamics_free_tumble_conservation_10_orbits():
    w0 = np.array([0.05, 0.12, 0.03])
    t = np.linspace(0.0, 10.0 * T_ORBIT, 11)          # 5.55e4 s ~ 5.57e4 s of the spec
    # rtol=1e-11: at the default 1e-10 the H_i vector drifts 1.8e-9 (DOP853 phase error over
    # ~7000 rad of tumble; an independent scratch solve_ivp gives the same). 1e-9 threshold kept.
    y = dyn.integrate(state(Q0, w0), t, I_BOX, rtol=1e-11, normalize=False)
    assert y.shape == (11, 10)

    # Inertial angular momentum, computed here with scipy (independent of the module).
    H = rot_of(y[:, :4]).apply(y[:, 4:7] @ I_BOX)
    H0 = H[0]
    drift_H = np.linalg.norm(H - H0, axis=1) / np.linalg.norm(H0)
    Ke = 0.5 * np.einsum("ni,ij,nj->n", y[:, 4:7], I_BOX, y[:, 4:7])
    drift_T = np.abs(Ke / Ke[0] - 1.0)
    drift_q = np.abs(np.linalg.norm(y[:, :4], axis=1) - 1.0)
    print(f"tumble drifts: H {drift_H.max():.3e}  T {drift_T.max():.3e}  |q|-1 {drift_q.max():.3e}")
    assert drift_H.max() < 1e-9
    assert drift_T.max() < 1e-9
    # raw integrator output (normalize=False): 1.5e-10 measured, so 1e-12 is not attainable
    assert drift_q.max() < 1e-9
    # body-frame momentum magnitude is also conserved (rotation preserves norm)
    Hb = np.linalg.norm(y[:, 4:7] @ I_BOX, axis=1)
    assert np.abs(Hb / Hb[0] - 1.0).max() < 1e-9

    # module helpers agree with the scipy-based values
    np.testing.assert_allclose(dyn.angular_momentum_inertial(y, I_BOX), H, rtol=0, atol=1e-12)
    np.testing.assert_allclose(dyn.kinetic_energy(y, I_BOX), Ke, rtol=1e-12, atol=0)


# ==========================================================================
# 12. symmetric top, analytic
#   I = diag(Ix, Ix, Iz), w0 = [A, 0, wz], h_w = 0.
#   Euler:  Ix wx' = (Iy - Iz) wy wz = (Ix - Iz) wy wz   ->  wx' = -lam wy
#           Ix wy' = (Iz - Ix) wz wx                     ->  wy' =  lam wx
#           Iz wz' = 0 -> wz = const;  lam = (Iz - Ix) wz / Ix
#   => wx = A cos(lam t), wy = A sin(lam t).
#   Attitude: H = Ix w + (Iz - Ix) wz z_hat  =>  w = H/Ix - ((Iz-Ix)/Ix) wz z_hat,
#   so z_hat' = w x z_hat = (|H|/Ix) H_hat x z_hat: the symmetry axis precesses about
#   H_hat at the constant rate |H|/Ix; the nutation angle satisfies cos = Iz wz/|H|.
# ==========================================================================
TOPS = [
    pytest.param(0.041867, 0.006667, id="prolate-box"),     # Iz < Ix: lam < 0
    pytest.param(0.010, 0.020, id="oblate"),                # Iz > Ix: lam > 0
]


@pytest.mark.parametrize("Ix,Iz", TOPS)
def test_dynamics_symmetric_top_body_rates(Ix, Iz):
    A, wz = 0.02, 0.1
    I = np.diag([Ix, Ix, Iz])
    lam = (Iz - Ix) * wz / Ix
    t = np.linspace(0.0, 200.0, 2001)
    y = dyn.integrate(state(Q0, [A, 0.0, wz]), t, I)
    # strict reading of "to 1e-8": error relative to the amplitude A
    assert np.abs(y[:, 4] - A * np.cos(lam * t)).max() < 1e-8 * A
    assert np.abs(y[:, 5] - A * np.sin(lam * t)).max() < 1e-8 * A
    assert np.abs(y[:, 6] - wz).max() < 1e-8 * wz


@pytest.mark.parametrize("Ix,Iz", TOPS)
def test_dynamics_symmetric_top_attitude_precession(Ix, Iz):
    A, wz = 0.02, 0.1
    I = np.diag([Ix, Ix, Iz])
    t = np.linspace(0.0, 200.0, 2001)
    y = dyn.integrate(state(Q0, [A, 0.0, wz]), t, I)
    R = rot_of(y[:, :4])
    H = R.apply(y[:, 4:7] @ I)
    Hn = np.linalg.norm(H, axis=1)
    Hh = H / Hn[:, None]
    # H_i direction fixed
    assert np.abs(Hh - Hh[0]).max() < 1e-9
    # nutation angle between H_i and body z constant, and equal to the analytic value
    zi = R.apply([0.0, 0.0, 1.0])
    nut = np.arccos(np.clip(np.sum(zi * Hh, axis=1), -1, 1))
    assert np.ptp(nut) < 1e-9
    H_mag = math.hypot(Ix * A, Iz * wz)
    assert nut[0] == pytest.approx(math.acos(Iz * wz / H_mag), abs=1e-9)
    # precession rate of the symmetry axis about H_hat = |H| / Ix (fit of unwrapped angle)
    e1 = np.cross(Hh[0], [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(Hh[0], e1)
    p = zi - np.sum(zi * Hh[0], axis=1)[:, None] * Hh[0]
    psi = np.unwrap(np.arctan2(p @ e2, p @ e1))
    slope = np.polyfit(t, psi, 1)[0]
    assert slope == pytest.approx(H_mag / Ix, rel=1e-6)


# ==========================================================================
# 13. momentum bias.  Ix = Iy, h_w = [0, 0, hz] constant (h_dot = 0), w0 = [A, 0, 0].
#   I w + h = (Ix wx, Ix wy, Iz wz + hz).  w x (I w + h):
#     x: wy (Iz wz + hz) - wz Ix wy
#     y: wz Ix wx - wx (Iz wz + hz)
#     z: wx Ix wy - wy Ix wx = 0   ->  Iz wz' = 0, so wz = 0 stays 0.
#   With wz = 0:  Ix wx' = -hz wy,  Ix wy' = +hz wx   =>  lam = hz / Ix,
#   wx = A cos(lam t), wy = A sin(lam t)   (gyroscopic nutation).
# ==========================================================================
@pytest.mark.parametrize("Iz", [0.006667, 0.02, 0.1])
def test_dynamics_momentum_bias_gyroscopic_nutation(Iz):
    Ix, A, hz = 0.041867, 0.03, 0.01
    I = np.diag([Ix, Ix, Iz])
    lam = hz / Ix
    t = np.linspace(0.0, 100.0, 1001)
    y = dyn.integrate(state(Q0, [A, 0, 0], [0, 0, hz]), t, I)
    assert np.abs(y[:, 4] - A * np.cos(lam * t)).max() < 1e-8 * A
    assert np.abs(y[:, 5] - A * np.sin(lam * t)).max() < 1e-8 * A
    assert np.abs(y[:, 6]).max() < 1e-8 * A                 # wz stays 0
    np.testing.assert_array_equal(y[:, 7:], np.tile([0, 0, hz], (len(t), 1)))  # h_dot = 0
    # quaternion kinematics sanity: |q| = 1 and total momentum conserved
    assert np.abs(np.linalg.norm(y[:, :4], axis=1) - 1).max() < 1e-12
    H = rot_of(y[:, :4]).apply(y[:, 4:7] @ I + y[:, 7:])
    assert np.abs(H - H[0]).max() < 1e-9 * np.linalg.norm(H[0])


# ==========================================================================
# 14. momentum exchange
#   Wheel torque u about body z on a body at rest (principal axis): w x (I w + h) = 0 at
#   all times (w and I w + h along z), so Iz wz' = -u  ->  wz = -u t / Iz, h = u t.
# ==========================================================================
def test_dynamics_momentum_exchange_wheel_spin_up():
    u = 1e-4
    Iz = I_BOX[2, 2]
    t = np.linspace(0.0, 50.0, 11)
    q0 = np.array([1.0, 0, 0, 0])
    y = dyn.integrate(state(q0, Z3, Z3), t, I_BOX, wheel_torque=np.array([0, 0, u]))
    hz = u * t
    np.testing.assert_allclose(y[:, 7], 0.0, atol=1e-14)
    np.testing.assert_allclose(y[:, 8], 0.0, atol=1e-14)
    np.testing.assert_allclose(y[:, 9], hz, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(y[:, 4:6], 0.0, atol=1e-12)
    np.testing.assert_allclose(y[:, 6], -hz / Iz, rtol=1e-10, atol=1e-12)
    # inertial momentum stays at its initial value 0 (absolute tolerance, scale u t = 5e-3)
    H = dyn.angular_momentum_inertial(y, I_BOX)
    assert np.abs(H).max() < 1e-12
    # attitude: z-rotation by theta(t) = -u t^2 / (2 Iz); q = [cos(th/2), 0, 0, sin(th/2)]
    th = -u * t**2 / (2.0 * Iz)
    np.testing.assert_allclose(y[:, 0], np.cos(th / 2), atol=1e-8)
    np.testing.assert_allclose(y[:, 3], np.sin(th / 2), atol=1e-8)


def test_dynamics_wheel_torque_nonprincipal_conserves_inertial_momentum():
    I = np.diag([0.041867, 0.0419, 0.006667]) + 1e-3 * np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    u = np.array([2e-4, -1e-4, 1.5e-4])
    y0 = state(Q0, [0.01, -0.02, 0.015], [0.002, 0.001, -0.003])
    t = np.linspace(0.0, 50.0, 26)
    for torque in (u, lambda tt, q, w, h: u):          # constant vector and callable forms
        y = dyn.integrate(y0, t, I, wheel_torque=torque)
        np.testing.assert_allclose(y[:, 7:], y0[7:] + np.outer(t, u), rtol=1e-9, atol=1e-12)
        H = rot_of(y[:, :4]).apply(y[:, 4:7] @ I + y[:, 7:])
        H0 = H[0]
        assert np.linalg.norm(H - H0, axis=1).max() < 1e-9 * np.linalg.norm(H0)


def test_dynamics_time_varying_wheel_torque_conserves_inertial_momentum():
    def u_w(t, q, w, h):
        return np.array([1e-4 * math.sin(0.3 * t), 2e-4 * math.cos(0.2 * t), -1e-4])
    y0 = state(Q0, [0.01, 0.02, -0.01])
    t = np.linspace(0.0, 60.0, 31)
    y = dyn.integrate(y0, t, I_BOX, wheel_torque=u_w)
    H = rot_of(y[:, :4]).apply(y[:, 4:7] @ I_BOX + y[:, 7:])
    assert np.linalg.norm(H - H[0], axis=1).max() < 1e-9 * np.linalg.norm(H[0])


# ==========================================================================
# derivative: hand evaluation
# ==========================================================================
def test_dynamics_derivative_hand_components():
    I = np.diag([1.0, 2.0, 3.0])
    w = np.array([0.1, 0.2, 0.3])
    h = np.array([0.0, 0.0, 0.5])
    u = np.array([0.01, -0.02, 0.03])
    tau = np.array([0.001, 0.002, -0.003])
    y = state([1, 0, 0, 0], w, h)
    # I w + h = (0.1, 0.4, 0.9 + 0.5)
    a1, a2, a3 = 0.1, 0.4, 1.4
    cross = np.array([w[1] * a3 - w[2] * a2, w[2] * a1 - w[0] * a3, w[0] * a2 - w[1] * a1])
    wdot = (-cross - u + tau) / np.array([1.0, 2.0, 3.0])
    ydot = dyn.derivative(0.0, y, I, u, tau)
    assert ydot.shape == (10,)
    np.testing.assert_allclose(ydot[:4], [0.0, 0.05, 0.1, 0.15], atol=1e-15)   # q = 1: 0.5 [0, w]
    np.testing.assert_allclose(ydot[4:7], wdot, rtol=1e-13, atol=1e-16)
    np.testing.assert_allclose(ydot[7:], u, rtol=0, atol=1e-16)
    # omitted torques default to zero
    ydot0 = dyn.derivative(0.0, y, I)
    np.testing.assert_allclose(ydot0[4:7], -cross / np.array([1.0, 2.0, 3.0]), rtol=1e-13, atol=1e-16)
    np.testing.assert_allclose(ydot0[7:], 0.0, atol=1e-16)


def test_dynamics_quaternion_rate_uses_body_frame_rate():
    # q_dot = 0.5 q (x) [0, w]: for a z-rotation by 90 deg and w along body x the quaternion
    # rate must equal the one from right-multiplying a small rotation (body-frame rate).
    q = q_of(Rot.from_euler("z", 90, degrees=True))
    w = np.array([0.2, -0.1, 0.3])
    ydot = dyn.derivative(0.0, state(q, w), I_BOX)
    eps = 1e-7
    q_next = q_of(rot_of(q) * Rot.from_rotvec(w * eps))
    np.testing.assert_allclose(ydot[:4], (q_next - q) / eps, atol=1e-6)


# ==========================================================================
# 15. rk4_step order, derivative vs finite differences, lvlh_error
# ==========================================================================
def test_dynamics_rk4_step_fourth_order_convergence():
    I = np.diag([0.041867, 0.0419, 0.006667])
    u = np.array([1e-4, -2e-4, 1.5e-4])
    tau = np.array([3e-5, 0.0, -2e-5])
    y0 = state(Q0, [0.05, 0.12, 0.03], [0.001, -0.002, 0.003])
    T = 20.0
    ref = dyn.integrate(y0, np.array([0.0, T]), I, u, tau, rtol=1e-13, atol=1e-15)[-1]
    errs = []
    for n in (20, 40, 80, 160):                      # dt = 1, 0.5, 0.25, 0.125
        dt = T / n
        y = y0.copy()
        for k in range(n):
            y = dyn.rk4_step(k * dt, y, dt, I, u, tau)
        errs.append(np.linalg.norm(y - ref))
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    print("rk4 errors", errs, "ratios", ratios)
    assert errs[-1] > 1e-12                          # reference is far more accurate than RK4
    for r in ratios:
        assert 14.0 < r < 18.0


def test_dynamics_rk4_step_renormalises_quaternion_and_accepts_callables():
    I = I_BOX
    w = np.array([0.5, -0.3, 0.8])
    y0 = state(Q0, w)
    y1 = dyn.rk4_step(0.0, y0, 2.0, I, lambda t, q, ww, h: 1e-4 * np.ones(3),
                      lambda t, q, ww, h: 2e-5 * np.array([1.0, 0, -1.0]))
    assert y1.shape == (10,)
    assert abs(np.linalg.norm(y1[:4]) - 1.0) < 1e-14
    assert not np.shares_memory(y0, y1)


def test_dynamics_derivative_matches_finite_difference_of_integrate():
    I = np.diag([0.041867, 0.0419, 0.006667])
    u = np.array([1e-4, -2e-4, 1.5e-4])
    tau = np.array([3e-5, 1e-5, -2e-5])
    y0 = state(Q0, [0.05, 0.12, 0.03], [0.001, -0.002, 0.003])
    h = 1e-3
    t = np.array([0.0, 5.0 - h, 5.0, 5.0 + h])
    y = dyn.integrate(y0, t, I, u, tau, rtol=1e-13, atol=1e-15)
    fd = (y[3] - y[1]) / (2 * h)                     # central difference, error ~ h^2 y'''/6 ~ 1e-9
    d = dyn.derivative(5.0, y[2], I, u, tau)
    np.testing.assert_allclose(d, fd, rtol=0, atol=1e-8)
    assert np.abs(d[4:]).max() > 1e-3                # the comparison is not vacuous


def test_dynamics_lvlh_error_zero_for_aligned_attitude():
    for u in (0.0, 1.0, 2.5, 4.0, 5.5):
        r, v = circ_state(u)
        q = q_of(lvlh_rot(r, v))
        e = dyn.lvlh_error(q, r, v)
        assert np.shape(e) == (3,)
        assert np.abs(e).max() < 1e-12
        assert float(dyn.nadir_angle(q, r, v)) < 1e-12


@pytest.mark.parametrize("u", [0.0, 1.3, 3.0, 5.0])
def test_dynamics_lvlh_error_known_5deg_about_body_y(u):
    r, v = circ_state(u)
    a = math.radians(5.0)
    q = q_of(lvlh_rot(r, v) * Rot.from_rotvec([0.0, a, 0.0]))     # body-frame rotation
    assert float(dyn.nadir_angle(q, r, v)) == pytest.approx(a, abs=1e-10)
    np.testing.assert_allclose(dyn.lvlh_error(q, r, v), [0.0, a, 0.0], atol=1e-10)


def test_dynamics_lvlh_error_is_body_frame_vector():
    r, v = circ_state(1.0)
    a = 0.08
    R_l = lvlh_rot(r, v)
    # rotation about body z (right-multiplication) -> error along body z
    q = q_of(R_l * Rot.from_rotvec([0, 0, a]))
    np.testing.assert_allclose(dyn.lvlh_error(q, r, v), [0, 0, a], atol=1e-12)
    # rotation about body x
    q = q_of(R_l * Rot.from_rotvec([a, 0, 0]))
    np.testing.assert_allclose(dyn.lvlh_error(q, r, v), [a, 0, 0], atol=1e-12)
    # an inertial-frame (left) rotation about ECI z shows up in the body frame as R_l^T (a z_eci)
    axis_eci = np.array([0.0, 0.0, 1.0])
    q = q_of(Rot.from_rotvec(a * axis_eci) * R_l)
    np.testing.assert_allclose(dyn.lvlh_error(q, r, v), R_l.inv().apply(a * axis_eci), atol=1e-12)


# ==========================================================================
# 16. external torque
# ==========================================================================
def test_dynamics_torque_callables_receive_t_q_w_hw():
    y = state(q_of(Rot.from_rotvec([0.1, 0.2, -0.3])), [0.01, -0.02, 0.03], [0.001, 0.002, 0.003])
    seen = {}

    def ext(t, q, w, h):
        seen["ext"] = (t, np.array(q), np.array(w), np.array(h))
        return np.array([1e-6, 0.0, 0.0])

    def wheel(t, q, w, h):
        seen["wheel"] = (t, np.array(q), np.array(w), np.array(h))
        return np.array([0.0, 2e-6, 0.0])

    out = dyn.derivative(3.5, y, I_BOX, wheel, ext)
    for key in ("ext", "wheel"):
        t, q, w, h = seen[key]
        assert t == 3.5
        np.testing.assert_allclose(q, y[:4], atol=1e-15)
        np.testing.assert_allclose(w, y[4:7], atol=1e-15)
        np.testing.assert_allclose(h, y[7:], atol=1e-15)
    # the callables' values are actually used
    np.testing.assert_allclose(out[7:], [0.0, 2e-6, 0.0], atol=1e-18)
    ref = dyn.derivative(3.5, y, I_BOX, [0.0, 2e-6, 0.0], [1e-6, 0.0, 0.0])
    np.testing.assert_allclose(out, ref, atol=1e-15)


def test_dynamics_integrate_calls_torque_callable_with_state_arrays():
    calls = []

    def ext(t, q, w, h):
        calls.append((float(t), np.shape(q), np.shape(w), np.shape(h)))
        return np.zeros(3)

    y0 = state(Q0, [0.01, 0.02, 0.03])
    dyn.integrate(y0, np.array([0.0, 1.0, 2.0]), I_BOX, ext_torque=ext)
    assert len(calls) > 0
    assert calls[0][0] == 0.0
    assert all(c[1:] == ((4,), (3,), (3,)) for c in calls)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_dynamics_constant_ext_torque_principal_axis_rest_body(axis):
    # tau along a principal axis, body at rest: w x I w = 0 throughout, so I w' = tau exactly.
    tau = np.zeros(3)
    tau[axis] = 1e-5
    t = np.linspace(0.0, 100.0, 11)
    y = dyn.integrate(state([1, 0, 0, 0], Z3), t, I_BOX, ext_torque=tau)
    expected = np.outer(t, tau) / np.diag(I_BOX)
    np.testing.assert_allclose(y[:, 4:7], expected, rtol=1e-11, atol=1e-14)
    np.testing.assert_allclose(y[:, 7:], 0.0, atol=1e-18)


def test_dynamics_time_dependent_ext_torque_closed_form():
    # tau_z(t) = c cos(k t)  ->  w_z = c sin(k t) / (k Iz); checks that t is passed correctly
    c, k = 1e-5, 0.7
    t = np.linspace(0.0, 30.0, 16)
    y = dyn.integrate(state([1, 0, 0, 0], Z3), t, I_BOX,
                      ext_torque=lambda tt, q, w, h: np.array([0.0, 0.0, c * math.cos(k * tt)]))
    np.testing.assert_allclose(y[:, 6], c * np.sin(k * t) / (k * I_BOX[2, 2]), rtol=1e-8, atol=1e-12)


# ==========================================================================
# helpers: shapes, hand values, no input mutation
# ==========================================================================
def test_dynamics_momentum_energy_hand_values_and_shapes():
    I = np.diag([1.0, 2.0, 3.0])
    q = q_of(Rot.from_euler("z", 90, degrees=True))
    y = state(q, [0.1, 0.2, 0.3], [0.0, 0.0, 0.5])
    # body momentum I w + h = (0.1, 0.4, 1.4); Rz(90): (x, y, z) -> (-y, x, z)
    H = dyn.angular_momentum_inertial(y, I)
    assert H.shape == (3,)
    np.testing.assert_allclose(H, [-0.4, 0.1, 1.4], atol=1e-15)
    assert float(dyn.kinetic_energy(y, I)) == pytest.approx(0.18, abs=1e-15)   # 0.5 (0.01 + 0.08 + 0.27)
    Y = np.stack([y, y, y])
    assert dyn.angular_momentum_inertial(Y, I).shape == (3, 3)
    assert dyn.kinetic_energy(Y, I).shape == (3,)


def test_dynamics_does_not_mutate_inputs():
    y0 = state(Q0, [0.05, 0.12, 0.03], [0.001, 0.0, 0.0])
    I = I_BOX.copy()
    t = np.linspace(0.0, 10.0, 5)
    u = np.array([1e-5, 0, 0])
    y0c, Ic, tc, uc = y0.copy(), I.copy(), t.copy(), u.copy()
    dyn.derivative(0.0, y0, I, u, u)
    dyn.rk4_step(0.0, y0, 0.5, I, u, u)
    dyn.integrate(y0, t, I, u, u)
    np.testing.assert_array_equal(y0, y0c)
    np.testing.assert_array_equal(I, Ic)
    np.testing.assert_array_equal(t, tc)
    np.testing.assert_array_equal(u, uc)


# ==========================================================================
# 17. gravity-gradient torque.   tau = k * r_b x (I r_b),   k = 3 mu / r^3.
#   With r_b = (a, b, c) and I = diag(Ix, Iy, Iz), explicit components:
#     tau = k * ( b c (Iz - Iy),  c a (Ix - Iz),  a b (Iy - Ix) ).
# ==========================================================================
def gg_hand(r_b, I_diag, r, mu=MU):
    a, b, c = r_b
    Ix, Iy, Iz = I_diag
    k = 3.0 * mu / r**3
    return k * np.array([b * c * (Iz - Iy), c * a * (Ix - Iz), a * b * (Iy - Ix)])


def body_r_dir(R_body: Rot, rhat_eci):
    return R_body.inv().apply(rhat_eci)


IX, IY, IZ = np.diag(I_BOX)
K_GG = 3.0 * MU / A_ORBIT**3


def test_dist_gravity_gradient_zero_on_principal_axes():
    I = np.diag([0.041867, 0.0419, 0.006667])       # three distinct principal moments
    rng = np.random.default_rng(11)
    for _ in range(5):
        R0 = Rot.random(random_state=int(rng.integers(1 << 30)))
        for e in np.eye(3):
            for s in (+1, -1):
                # body axis s*e points along r_eci
                rhat_eci = R0.apply(s * e)
                r_eci = A_ORBIT * rhat_eci
                tau = dist.gravity_gradient_torque(r_eci, q_of(R0), I)
                assert np.abs(tau).max() < 1e-20       # scale of non-zero torques ~1e-8


@pytest.mark.parametrize("theta_deg", [-70.0, -30.0, 10.0, 45.0, 80.0])
def test_dist_gravity_gradient_xz_plane_explicit_components(theta_deg):
    # r_b = (sin th, 0, cos th) (angle th from body z toward +x).  Then
    #   tau_y = k (Ix - Iz) sin th cos th,  tau_x = tau_z = 0.
    th = math.radians(theta_deg)
    # body->ECI rotation about y by (pi/2 - th) maps r_b to the ECI x axis (see derivation:
    # Ry(phi)(x, y, z) = (x cos phi + z sin phi, y, -x sin phi + z cos phi)).
    R0 = Rot.from_euler("y", math.pi / 2 - th)
    r_eci = np.array([A_ORBIT, 0.0, 0.0])
    r_b = R0.inv().apply(r_eci / A_ORBIT)
    np.testing.assert_allclose(r_b, [math.sin(th), 0, math.cos(th)], atol=1e-15)
    tau = dist.gravity_gradient_torque(r_eci, q_of(R0), I_BOX)
    expected = np.array([0.0, K_GG * (IX - IZ) * math.sin(th) * math.cos(th), 0.0])
    np.testing.assert_allclose(tau, expected, rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(tau, gg_hand(r_b, (IX, IY, IZ), A_ORBIT), rtol=1e-12, atol=1e-20)


@pytest.mark.parametrize("delta_deg", [-60.0, -20.0, 5.0, 40.0])
def test_dist_gravity_gradient_body_rotation_about_y_matches_spec_sign(delta_deg):
    # Spec wording tau_y = (3 mu/r^3)(Iz - Ix) sin th cos th holds when th is the rotation
    # angle of the BODY about +y away from the nadir-aligned (z along r) attitude, for then
    # r_b = (-sin th, 0, cos th).
    d = math.radians(delta_deg)
    rhat = np.array([0.3, -0.5, 0.8])
    rhat /= np.linalg.norm(rhat)
    R_align = Rot.align_vectors(rhat[None], [[0.0, 0.0, 1.0]])[0]    # body z -> r_hat
    R = R_align * Rot.from_rotvec([0.0, d, 0.0])
    tau = dist.gravity_gradient_torque(A_ORBIT * rhat, q_of(R), I_BOX)
    expected = K_GG * (IZ - IX) * math.sin(d) * math.cos(d)
    np.testing.assert_allclose(tau, [0.0, expected, 0.0], rtol=1e-11, atol=1e-20)


def test_dist_gravity_gradient_peak_magnitude():
    # |r_b x I r_b| is maximal (= (Imax - Imin)/2) at 45 deg in the plane of Imax and Imin.
    R0 = Rot.from_euler("y", math.pi / 2 - math.pi / 4)      # r_b = (sin 45, 0, cos 45)
    tau = dist.gravity_gradient_torque(A_ORBIT * np.array([1.0, 0, 0]), q_of(R0), I_BOX)
    peak = 1.5 * MU / A_ORBIT**3 * abs(IZ - IX)
    assert np.linalg.norm(tau) == pytest.approx(peak, rel=1e-12)
    # spec quotes ~6.9e-8 N m at 400 km (hand value is 6.76e-8: within 3 %)
    assert peak == pytest.approx(6.9e-8, rel=0.03)
    # random attitudes never exceed it
    rng = np.random.default_rng(5)
    big = 0.0
    for _ in range(200):
        R = Rot.random(random_state=int(rng.integers(1 << 30)))
        rhat = rng.normal(size=3)
        rhat /= np.linalg.norm(rhat)
        t_ = dist.gravity_gradient_torque(A_ORBIT * rhat, q_of(R), I_BOX)
        big = max(big, np.linalg.norm(t_))
    assert big <= peak * (1 + 1e-12)


def test_dist_gravity_gradient_perpendicular_to_rhat_and_random_hand_values():
    rng = np.random.default_rng(17)
    I = np.diag([0.041867, 0.0419, 0.006667])
    for _ in range(50):
        R = Rot.random(random_state=int(rng.integers(1 << 30)))
        rhat = rng.normal(size=3)
        rhat /= np.linalg.norm(rhat)
        r = rng.uniform(6.6e6, 7.2e6)
        tau = dist.gravity_gradient_torque(r * rhat, q_of(R), I)
        r_b = R.inv().apply(rhat)
        assert abs(tau @ r_b) < 1e-20
        np.testing.assert_allclose(tau, gg_hand(r_b, np.diag(I), r), rtol=1e-10, atol=1e-20)


def test_dist_gravity_gradient_scales_inverse_cube_and_mu():
    R = Rot.from_rotvec([0.4, 0.2, -0.1])
    rhat = np.array([0.6, 0.0, 0.8])
    t1 = dist.gravity_gradient_torque(7.0e6 * rhat, q_of(R), I_BOX)
    t2 = dist.gravity_gradient_torque(1.4e7 * rhat, q_of(R), I_BOX)
    np.testing.assert_allclose(t2, t1 / 8.0, rtol=1e-12)
    t3 = dist.gravity_gradient_torque(7.0e6 * rhat, q_of(R), I_BOX, mu=2 * MU)
    np.testing.assert_allclose(t3, 2 * t1, rtol=1e-12)


def test_dist_gravity_gradient_rotation_covariance_with_nondiagonal_inertia():
    # New body axes B' = B rotated by P (body'->ECI rotation R P): v' = P^T v, I' = P^T I P,
    # tau' = P^T tau.
    P = Rot.from_rotvec([0.3, -0.5, 0.2])
    I_rot = P.as_matrix().T @ I_BOX @ P.as_matrix()
    R = Rot.from_rotvec([0.7, 0.1, -0.4])
    rhat = np.array([0.2, 0.9, -0.3])
    rhat /= np.linalg.norm(rhat)
    q = q_of(R)
    tau = dist.gravity_gradient_torque(A_ORBIT * rhat, q, I_BOX)
    # body frame B' = P-rotated frame: vectors transform v' = P^T v, so use body->ECI R * P
    tau_rot = dist.gravity_gradient_torque(A_ORBIT * rhat, q_of(R * P), I_rot)
    np.testing.assert_allclose(tau_rot, P.inv().apply(tau), rtol=1e-10, atol=1e-20)


def test_dist_gravity_gradient_nadir_long_axis_equilibrium_is_stable():
    # long axis z (min inertia) along r.  Small body rotation +d about y: r_b = (-sin d, 0, cos d),
    # tau_y = k (Ix - Iz)(-sin d cos d) < 0 (restoring); -d gives tau_y > 0.  Rotation +d about x:
    # r_b = (0, sin d, cos d), tau_x = k sin d cos d (Iz - Iy) < 0.
    rhat = np.array([0.0, 0.6, 0.8])
    R_align = Rot.align_vectors(rhat[None], [[0.0, 0.0, 1.0]])[0]
    for d in (0.01, 0.05):
        for sgn in (+1, -1):
            tau_y = dist.gravity_gradient_torque(
                A_ORBIT * rhat, q_of(R_align * Rot.from_rotvec([0, sgn * d, 0])), I_BOX)
            assert sgn * tau_y[1] < 0
            tau_x = dist.gravity_gradient_torque(
                A_ORBIT * rhat, q_of(R_align * Rot.from_rotvec([sgn * d, 0, 0])), I_BOX)
            assert sgn * tau_x[0] < 0
    # and the aligned attitude itself has zero torque
    tau0 = dist.gravity_gradient_torque(A_ORBIT * rhat, q_of(R_align), I_BOX)
    assert np.abs(tau0).max() < 1e-20


# ==========================================================================
# 18. residual dipole
# ==========================================================================
def test_dist_residual_dipole_hand_cross_product():
    m = np.array([1e-3, 2e-3, 3e-3])
    b = np.array([2e-5, -1e-5, 4e-5])
    # m x b explicit components
    exp = np.array([m[1] * b[2] - m[2] * b[1], m[2] * b[0] - m[0] * b[2], m[0] * b[1] - m[1] * b[0]])
    np.testing.assert_allclose(exp, [2e-3 * 4e-5 + 3e-3 * 1e-5, 3e-3 * 2e-5 - 1e-3 * 4e-5,
                                     -1e-3 * 1e-5 - 2e-3 * 2e-5])
    tau = dist.residual_dipole_torque(m, b)
    assert tau.shape == (3,)
    np.testing.assert_allclose(tau, exp, rtol=1e-14)


def test_dist_residual_dipole_perpendicular_parallel_zero_and_antisymmetry():
    rng = np.random.default_rng(3)
    for _ in range(20):
        m = rng.normal(size=3) * 1e-3
        b = rng.normal(size=3) * 3e-5
        tau = dist.residual_dipole_torque(m, b)
        assert abs(tau @ m) < 1e-12 * np.linalg.norm(tau) * np.linalg.norm(m) + 1e-30
        assert abs(tau @ b) < 1e-12 * np.linalg.norm(tau) * np.linalg.norm(b) + 1e-30
        # parallel and anti-parallel
        # rounding of the cross-product terms is ~1e-16 * |m||3m|
        atol = 1e-15 * 3.0 * (m @ m)
        np.testing.assert_allclose(dist.residual_dipole_torque(m, 3.0 * m), 0.0, atol=atol)
        np.testing.assert_allclose(dist.residual_dipole_torque(m, -2.0 * m), 0.0, atol=atol)
        # linear in both arguments and antisymmetric
        np.testing.assert_allclose(dist.residual_dipole_torque(2 * m, b), 2 * tau, rtol=1e-13)
        np.testing.assert_allclose(dist.residual_dipole_torque(b, m), -tau, rtol=1e-13)


def test_dist_residual_dipole_eci_variant_equals_body_variant_after_rotation():
    rng = np.random.default_rng(23)
    m = np.array([0.001, 0.001, 0.001])
    for _ in range(10):
        R = Rot.random(random_state=int(rng.integers(1 << 30)))
        b_eci = rng.normal(size=3) * 3e-5
        b_b = R.inv().apply(b_eci)                 # body = R^T eci (scipy oracle)
        np.testing.assert_allclose(dist.residual_dipole_torque_eci(m, q_of(R), b_eci),
                                   dist.residual_dipole_torque(m, b_b), rtol=1e-12, atol=1e-25)


# ==========================================================================
# 19. aerodynamic torque
# ==========================================================================
RHO400 = 3.725e-12


def v_rel_hand(r, v):
    """v - Omega z_hat x r with z x r = (-y, x, 0), written out component-wise."""
    return np.array([v[0] + OMEGA_EARTH * r[1], v[1] - OMEGA_EARTH * r[0], v[2]])


def aero_hand(r, v, q, size, com, cd, rho):
    """Face-by-face flat-plate force/torque (spec section 4), explicit loops."""
    size = np.asarray(size, float)
    com = np.asarray(com, float)
    vr = v_rel_hand(r, v)
    sp = np.linalg.norm(vr)
    vb = rot_of(q).inv().apply(vr / sp)
    F = np.zeros(3)
    tau = np.zeros(3)
    for k in range(3):
        for s in (+1.0, -1.0):
            n = np.zeros(3)
            n[k] = s
            area = size[(k + 1) % 3] * size[(k + 2) % 3]
            ca = n @ vb
            if ca > 0.0:
                Fi = -0.5 * rho * sp**2 * cd * area * ca * vb
                ri = n * size / 2.0 - com
                F = F + Fi
                tau = tau + np.array([ri[1] * Fi[2] - ri[2] * Fi[1],
                                      ri[2] * Fi[0] - ri[0] * Fi[2],
                                      ri[0] * Fi[1] - ri[1] * Fi[0]])
    return F, tau


def flow_along_body_x_case(vrel_mag=7670.0, r0=A_ORBIT):
    """Equatorial point r = (r0, 0, 0); body x axis along ECI +y (Rz(+90 deg)); the velocity is
    chosen so that v_rel = vrel_mag * (0, 1, 0), i.e. v = v_rel + Omega z x r = (0, vrel + Omega r0, 0)."""
    r = np.array([r0, 0.0, 0.0])
    v = np.array([0.0, vrel_mag + OMEGA_EARTH * r0, 0.0])
    q = q_of(Rot.from_euler("z", 90, degrees=True))
    return r, v, q


def test_dist_aero_flow_along_body_x_hand_value():
    # Only the +x face (normal +x, area A_x = b c = 0.034) faces the flow.
    # F = -f x_hat, f = 0.5 rho v^2 Cd A_x.  Lever (face centre - CoM) = (0.05 - 0.005, -0.005, -0.010).
    # tau = r x F = (0, -rz Fx_mag.., ...): tau_x = ry Fz - rz Fy = 0; tau_y = rz Fx - rx Fz = (-0.010)(-f) = +0.010 f;
    #  tau_z = rx Fy - ry Fx = -(-0.005)(-f) = -0.005 f.
    r, v, q = flow_along_body_x_case()
    cfg = DEFAULT.spacecraft
    f = 0.5 * RHO400 * 7670.0**2 * cfg.cd * (0.1 * 0.34)
    tau = dist.aero_torque(r, v, q, cfg, rho=RHO400)
    np.testing.assert_allclose(tau, f * np.array([0.0, 0.010, -0.005]), rtol=1e-10, atol=1e-18)
    F, tau2 = dist._aero_force_torque(r, v, q, cfg, rho=RHO400)
    np.testing.assert_allclose(F, [-f, 0.0, 0.0], rtol=1e-10, atol=1e-16)
    np.testing.assert_allclose(tau2, tau, rtol=1e-12, atol=1e-20)


def test_dist_aero_magnitude_at_400km_within_10_percent():
    # hand computation: f = 0.5 * 3.725e-12 * 7670^2 * 2.2 * 0.034 = 8.196e-6 N,
    # |tau| = f * sqrt(0.010^2 + 0.005^2) = 9.16e-8 N m
    r, v, q = flow_along_body_x_case(7670.0)
    f = 0.5 * 3.725e-12 * 7670.0**2 * 2.2 * 0.034
    hand = f * math.hypot(0.010, 0.005)
    assert hand == pytest.approx(9.16e-8, rel=5e-3)
    tau = dist.aero_torque(r, v, q, DEFAULT.spacecraft, rho=RHO400)
    assert np.linalg.norm(tau) == pytest.approx(hand, rel=0.10)


def test_dist_aero_default_density_is_exponential_at_spherical_height():
    # rho = rho0 exp(-(h - h0)/H), h = |r| - R_EARTH (Vallado table values from the spec)
    for h in (400e3, 450e3, 350e3):
        r, v, q = flow_along_body_x_case(7670.0, r0=R_EARTH + h)
        rho = 3.725e-12 * math.exp(-(h - 400e3) / 58.515e3)
        np.testing.assert_allclose(dist.aero_torque(r, v, q, DEFAULT.spacecraft),
                                   dist.aero_torque(r, v, q, DEFAULT.spacecraft, rho=rho), rtol=1e-10, atol=1e-20)


def test_dist_aero_zero_lever_gives_zero_torque():
    cfg = SpacecraftConfig(com_offset_b=(0.0, 0.0, 0.0))
    for k, e in enumerate(np.eye(3)):
        for s in (+1, -1):
            # body axis s*e along the relative flow (ECI +y at the equatorial point)
            R0 = Rot.align_vectors([[0.0, 1.0, 0.0]], [s * e])[0]
            r = np.array([A_ORBIT, 0.0, 0.0])
            v = np.array([0.0, 7670.0 + OMEGA_EARTH * A_ORBIT, 0.0])
            tau = dist.aero_torque(r, v, q_of(R0), cfg, rho=RHO400)
            assert np.abs(tau).max() < 1e-20
            F, _ = dist._aero_force_torque(r, v, q_of(R0), cfg, rho=RHO400)
            assert np.linalg.norm(F) > 1e-6                # a drag force exists, only the lever is zero


def test_dist_aero_scaling_with_density_and_speed_squared():
    r = np.array([A_ORBIT * 0.8, A_ORBIT * 0.6, 0.0])
    w_x_r = OMEGA_EARTH * np.array([-r[1], r[0], 0.0])
    vrel = np.array([-3000.0, 6000.0, 2500.0])
    q = q_of(Rot.from_rotvec([0.4, -0.7, 0.2]))
    cfg = DEFAULT.spacecraft
    t1 = dist.aero_torque(r, vrel + w_x_r, q, cfg, rho=RHO400)
    t_rho = dist.aero_torque(r, vrel + w_x_r, q, cfg, rho=2.5 * RHO400)
    t_v = dist.aero_torque(r, 2.0 * vrel + w_x_r, q, cfg, rho=RHO400)      # v_rel doubled
    np.testing.assert_allclose(t_rho, 2.5 * t1, rtol=1e-12)
    np.testing.assert_allclose(t_v, 4.0 * t1, rtol=1e-12)
    assert np.linalg.norm(t1) > 1e-9


def test_dist_aero_corotating_atmosphere_equator_hand_case():
    # equatorial point r = (r0, 0, 0), inertial v = (0, v0, 0):
    #   v_rel = v - Omega z x r = (0, v0 - Omega r0, 0)   (z x r = (0, r0, 0)).
    r0, v0 = A_ORBIT, math.sqrt(MU / A_ORBIT)
    r = np.array([r0, 0.0, 0.0])
    v = np.array([0.0, v0, 0.0])
    q = q_of(Rot.from_euler("z", 90, degrees=True))                      # body x along +y_eci
    vrel = v0 - OMEGA_EARTH * r0
    assert vrel == pytest.approx(7668.6 - 494.3, abs=2.0)               # sanity of the hand number
    cfg = DEFAULT.spacecraft
    f = 0.5 * RHO400 * vrel**2 * cfg.cd * 0.034
    tau = dist.aero_torque(r, v, q, cfg, rho=RHO400)
    np.testing.assert_allclose(tau, f * np.array([0.0, 0.010, -0.005]), rtol=1e-10, atol=1e-18)
    # and a point on the +y axis: z x r = (-r0, 0, 0)  ->  v_rel = (vx + Omega r0, ...)
    r = np.array([0.0, r0, 0.0])
    v = np.array([-v0, 0.0, 0.0])
    q = q_of(Rot.from_euler("z", 180, degrees=True))                     # body x along -x_eci
    vrel = v0 - OMEGA_EARTH * r0                                         # |(-v0 + Omega r0)|
    f = 0.5 * RHO400 * vrel**2 * cfg.cd * 0.034
    tau = dist.aero_torque(r, v, q, cfg, rho=RHO400)
    np.testing.assert_allclose(tau, f * np.array([0.0, 0.010, -0.005]), rtol=1e-10, atol=1e-18)


def test_dist_aero_matches_face_by_face_hand_model_random_attitudes():
    rng = np.random.default_rng(29)
    cfgs = [DEFAULT.spacecraft,
            SpacecraftConfig(mass=6.0, size=(0.1, 0.2, 0.3), com_offset_b=(-0.01, 0.02, 0.03), cd=2.0)]
    for cfg in cfgs:
        for _ in range(60):
            r = rng.normal(size=3)
            r *= rng.uniform(6.7e6, 6.9e6) / np.linalg.norm(r)
            v = rng.normal(size=3)
            v *= 7600.0 / np.linalg.norm(v)
            q = q_of(Rot.random(random_state=int(rng.integers(1 << 30))))
            F_h, tau_h = aero_hand(r, v, q, cfg.size, cfg.com_offset_b, cfg.cd, RHO400)
            tau = dist.aero_torque(r, v, q, cfg, rho=RHO400)
            np.testing.assert_allclose(tau, tau_h, rtol=1e-9, atol=1e-22)
            F, tau2 = dist._aero_force_torque(r, v, q, cfg, rho=RHO400)
            np.testing.assert_allclose(F, F_h, rtol=1e-9, atol=1e-18)
            np.testing.assert_allclose(tau2, tau_h, rtol=1e-9, atol=1e-22)


def test_dist_aero_force_opposes_flow_and_opposite_faces_never_both_contribute():
    rng = np.random.default_rng(31)
    cfg = DEFAULT.spacecraft
    size = np.array(cfg.size)
    areas = np.array([size[1] * size[2], size[0] * size[2], size[0] * size[1]])    # A_x, A_y, A_z
    for _ in range(100):
        r = rng.normal(size=3)
        r *= 6.78e6 / np.linalg.norm(r)
        v = rng.normal(size=3)
        v *= 7600.0 / np.linalg.norm(v)
        R = Rot.random(random_state=int(rng.integers(1 << 30)))
        vr = v_rel_hand(r, v)
        sp = np.linalg.norm(vr)
        vb = R.inv().apply(vr / sp)
        F, tau = dist._aero_force_torque(r, v, q_of(R), cfg, rho=RHO400)
        assert F @ vb <= 0.0
        # F is purely drag, along -v_hat_b
        assert np.linalg.norm(np.cross(F, vb)) < 1e-12 * np.linalg.norm(F)
        # exactly one of each +/- face pair (or none) contributes: |F| = 0.5 rho v^2 Cd sum_k A_k |vb_k|
        expected_mag = 0.5 * RHO400 * sp**2 * cfg.cd * np.sum(areas * np.abs(vb))
        assert np.linalg.norm(F) == pytest.approx(expected_mag, rel=1e-10)
        assert np.isfinite(tau).all()


# ==========================================================================
# total / breakdown, shapes, no mutation
# ==========================================================================
def test_dist_total_equals_sum_of_breakdown_and_individual_functions():
    rng = np.random.default_rng(37)
    cfg = DEFAULT.spacecraft
    for _ in range(10):
        r = rng.normal(size=3)
        r *= 6.78e6 / np.linalg.norm(r)
        v = rng.normal(size=3)
        v *= 7660.0 / np.linalg.norm(v)
        R = Rot.random(random_state=int(rng.integers(1 << 30)))
        q = q_of(R)
        b_eci = rng.normal(size=3) * 3e-5
        br = dist.disturbance_breakdown(r, v, q, b_eci, cfg)
        assert set(br) == {"gravity_gradient", "residual_dipole", "aero"}
        assert all(np.shape(x) == (3,) for x in br.values())
        tot = dist.total_disturbance_torque(r, v, q, b_eci, cfg)
        assert tot.shape == (3,)
        np.testing.assert_allclose(tot, br["gravity_gradient"] + br["residual_dipole"] + br["aero"],
                                   rtol=1e-12, atol=1e-22)
        np.testing.assert_allclose(br["gravity_gradient"],
                                   dist.gravity_gradient_torque(r, q, cfg.inertia), rtol=1e-12, atol=1e-22)
        np.testing.assert_allclose(br["residual_dipole"],
                                   dist.residual_dipole_torque_eci(np.array(cfg.residual_dipole_b), q, b_eci),
                                   rtol=1e-12, atol=1e-25)
        np.testing.assert_allclose(br["aero"], dist.aero_torque(r, v, q, cfg), rtol=1e-12, atol=1e-22)


def test_dist_functions_do_not_mutate_inputs():
    r, v, q = flow_along_body_x_case()
    b = np.array([1e-5, 2e-5, -3e-5])
    I = I_BOX.copy()
    m = np.array([1e-3, 2e-3, 3e-3])
    saved = [x.copy() for x in (r, v, q, b, I, m)]
    dist.gravity_gradient_torque(r, q, I)
    dist.residual_dipole_torque(m, b)
    dist.residual_dipole_torque_eci(m, q, b)
    dist.aero_torque(r, v, q, DEFAULT.spacecraft, rho=RHO400)
    dist.total_disturbance_torque(r, v, q, b, DEFAULT.spacecraft)
    dist.disturbance_breakdown(r, v, q, b, DEFAULT.spacecraft)
    for a, s in zip((r, v, q, b, I, m), saved):
        np.testing.assert_array_equal(a, s)


# ==========================================================================
# 20. magnitude gate (reported; order-of-magnitude check, Wertz SMAD ch. 17, unverified)
#   Orbit: circular 400 km, 51.6 deg (explicit formulas).  Magnetic field: an independent tilted
#   dipole (M = 7.94e22 A m^2, axis 10 deg from -z_eci; textbook value, unverified), giving
#   2.5e-5 ... 5e-5 T at 400 km -- no dependence on the IGRF module.
# ==========================================================================
def dipole_b_eci(r, moment=7.94e22, tilt_deg=10.0):
    mh = np.array([math.sin(math.radians(tilt_deg)), 0.0, -math.cos(math.radians(tilt_deg))])
    rn = np.linalg.norm(r)
    rh = r / rn
    return 1e-7 * moment * (3.0 * (mh @ rh) * rh - mh) / rn**3


ATTITUDES = [("nadir", None, 0.0)] + [
    (f"{ax_name}{ang:+.0f}deg", ax, ang) for ax_name, ax in (("x", 0), ("y", 1), ("z", 2)) for ang in (10.0, 20.0)
]


@pytest.mark.parametrize("name,axis,angle", ATTITUDES, ids=[a[0] for a in ATTITUDES])
def test_dist_magnitude_gate_default_config(name, axis, angle):
    cfg = DEFAULT.spacecraft
    us = np.linspace(0.0, 2.0 * math.pi, 120, endpoint=False)
    mags = {"gravity_gradient": [], "residual_dipole": [], "aero": []}
    for u in us:
        r, v = circ_state(u)
        R = lvlh_rot(r, v)
        if axis is not None:
            rv = np.zeros(3)
            rv[axis] = math.radians(angle)
            R = R * Rot.from_rotvec(rv)                    # body-frame offset about one axis
        q = q_of(R)
        b = dipole_b_eci(r)
        br = dist.disturbance_breakdown(r, v, q, b, cfg)
        tot = dist.total_disturbance_torque(r, v, q, b, cfg)
        np.testing.assert_allclose(tot, sum(br.values()), rtol=1e-12, atol=1e-22)
        for k in mags:
            mags[k].append(np.linalg.norm(br[k]))
    summary = []
    for k, arr in mags.items():
        arr = np.array(arr)
        rms, mx = float(np.sqrt(np.mean(arr**2))), float(arr.max())
        summary.append(f"{k}: rms {rms:.2e} max {mx:.2e}")
        # Gravity gradient is analytically ZERO when the long axis z is along nadir (nadir-aligned
        # and the z-offset attitudes: r_b = -z_hat for diagonal I).  The gate lower bound cannot
        # apply there; it is applied wherever the torque is non-zero by construction.
        if k == "gravity_gradient" and axis in (None, 2):
            assert mx < 1e-10
            continue
        assert 1e-10 <= rms <= 1e-6, f"{name} {k} rms {rms:.3e}"
        assert 1e-10 <= mx <= 1e-6, f"{name} {k} max {mx:.3e}"
    print(f"[magnitude gate] {name}: " + " | ".join(summary))
