"""Environmental disturbance torques on the 3U CubeSat (Project B, Group 5 / M6).

Conventions (CONVENTIONS.md): SI units, scalar-first Hamilton quaternion ``q``
= body-to-ECI rotation, positions/velocities in ECI (m, m/s), every torque
returned in **body axes** (N m) about the centre of mass.

Models: spherical-Earth gravity gradient, constant residual magnetic dipole,
flat-plate aerodynamic drag on the six faces of the box with a co-rotating
atmosphere. Solar radiation pressure is neglected (assumption B-12).
All functions are batched over leading dimensions of ``r_eci``/``v_eci``/``q``.
"""
from __future__ import annotations

import numpy as np

from common.frames import GM_EARTH, OMEGA_EARTH, R_EARTH
from common.quaternion import rotate_inverse
from adcs.config import DEFAULT
from adcs.environment import atmospheric_density

__all__ = [
    "gravity_gradient_torque",
    "residual_dipole_torque",
    "residual_dipole_torque_eci",
    "aero_torque",
    "total_disturbance_torque",
    "disturbance_breakdown",
]

#: Outward face normals (body axes) of the six box faces: +x, -x, +y, -y, +z, -z.
_FACE_NORMALS = np.array(
    [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
     [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
)


def _vec(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def gravity_gradient_torque(r_eci, q, inertia, mu: float = GM_EARTH) -> np.ndarray:
    """Gravity-gradient torque (N m, body axes) about the CoM, spherical Earth.

    ``tau = 3 mu / |r|^3 * r_b x (I r_b)`` with ``r_b`` the unit vector from the
    Earth centre to the spacecraft expressed in body axes.

    Parameters
    ----------
    r_eci : (..., 3) spacecraft position in ECI (m).
    q : (..., 4) body-to-ECI quaternion, scalar-first.
    inertia : (3, 3) inertia tensor about the CoM in body axes (kg m^2).
    mu : gravitational parameter (m^3/s^2).
    """
    r = _vec(r_eci)
    rn = np.linalg.norm(r, axis=-1, keepdims=True)
    r_b = rotate_inverse(q, r / rn)
    I_r = np.einsum("ij,...j->...i", _vec(inertia), r_b)
    return 3.0 * mu / rn**3 * np.cross(r_b, I_r)


def residual_dipole_torque(m_b, b_b) -> np.ndarray:
    """Residual-dipole torque ``m x B`` (N m, body axes).

    ``m_b`` residual dipole (A m^2, body axes), ``b_b`` magnetic flux density
    (T, body axes); broadcast over leading dimensions.
    """
    return np.cross(_vec(m_b), _vec(b_b))


def residual_dipole_torque_eci(m_b, q, b_eci) -> np.ndarray:
    """Residual-dipole torque (N m, body axes) from the ECI field.

    ``b_eci`` (T, ECI) is rotated into the body with ``q`` (body-to-ECI,
    scalar-first) and then ``m_b (A m^2, body axes) x B_b`` is returned.
    """
    return residual_dipole_torque(m_b, rotate_inverse(q, _vec(b_eci)))


def _aero_force_torque(r_eci, v_eci, q, cfg=DEFAULT.spacecraft, rho=None):
    """Flat-plate drag force and torque about the CoM, both in body axes.

    Parameters
    ----------
    r_eci, v_eci : (..., 3) position (m) and velocity (m/s) in ECI.
    q : (..., 4) body-to-ECI quaternion, scalar-first.
    cfg : ``SpacecraftConfig`` (size, com_offset_b, cd).
    rho : atmospheric density (kg/m^3), scalar or broadcastable to the leading
        dimensions; ``None`` -> ``atmospheric_density(|r| - R_EARTH)``.

    Returns
    -------
    F : (..., 3) total drag force (N, body axes).
    tau : (..., 3) drag torque about the CoM (N m, body axes).

    Each face (normal n, area A) facing the flow (``cos a = n . v_hat > 0``)
    gets ``F_i = -1/2 rho |v_rel|^2 Cd A cos a v_hat``, applied at the face
    centre ``n * size/2 - com_offset_b`` relative to the CoM. The atmosphere
    co-rotates with Earth: ``v_rel = v - Omega_E z x r`` (ECI).
    """
    r = _vec(r_eci)
    v = _vec(v_eci)
    if rho is None:
        rho = atmospheric_density(np.linalg.norm(r, axis=-1) - R_EARTH)
    rho = _vec(rho)

    omega_e = np.array([0.0, 0.0, OMEGA_EARTH])
    v_rel = v - np.cross(omega_e, r)
    speed = np.linalg.norm(v_rel, axis=-1, keepdims=True)
    v_hat = rotate_inverse(q, v_rel / speed)  # unit flow direction, body axes

    size = _vec(cfg.size)
    com = _vec(cfg.com_offset_b)
    areas = np.array([size[1] * size[2], size[0] * size[2], size[0] * size[1]])
    q_dyn = 0.5 * rho[..., None] * speed**2  # (..., 1) dynamic pressure, Pa

    F = np.zeros(np.broadcast(v_hat, q_dyn).shape)
    tau = np.zeros_like(F)
    for k, n in enumerate(_FACE_NORMALS):
        axis = k // 2
        cos_a = v_hat @ n  # (...,)
        cos_a = np.maximum(cos_a, 0.0)  # only faces facing the flow
        F_i = -(q_dyn * cfg.cd * areas[axis] * cos_a[..., None]) * v_hat
        lever = n * 0.5 * size - com  # centre of pressure relative to CoM (m)
        F = F + F_i
        tau = tau + np.cross(lever, F_i)
    return F, tau


def aero_torque(r_eci, v_eci, q, cfg=DEFAULT.spacecraft, rho=None) -> np.ndarray:
    """Aerodynamic drag torque (N m, body axes) about the CoM, flat-plate model.

    Six faces of the box (normals +-x, +-y, +-z, areas ``b c, a c, a b`` for
    ``size = (a, b, c)``), drag coefficient ``cfg.cd``, convex body (no
    shadowing, no lift). The atmosphere co-rotates with Earth:
    ``v_rel = v_eci - OMEGA_EARTH z x r_eci``. The centre of pressure of each face
    is the face centre relative to the CoM (``cfg.com_offset_b`` displaces the CoM
    from the geometric centre).

    Parameters
    ----------
    r_eci (m), v_eci (m/s) : (..., 3) ECI position / velocity.
    q : (..., 4) body-to-ECI quaternion, scalar-first.
    rho : density (kg/m^3) or ``None`` for ``atmospheric_density(|r| - R_EARTH)``.
        The default uses the spherical-Earth height, ignoring oblateness: the
        height error is up to ~21 km at the poles (~13 km at 51.6 deg latitude), i.e.
        up to exp(21/58.5) - 1 ~ 44 % (~25 % at 51.6 deg) in density.
    """
    return _aero_force_torque(r_eci, v_eci, q, cfg, rho)[1]


def disturbance_breakdown(r_eci, v_eci, q, b_eci, cfg=DEFAULT.spacecraft,
                          mu: float = GM_EARTH) -> dict:
    """Individual disturbance torques (N m, body axes) about the CoM.

    Returns ``{"gravity_gradient", "residual_dipole", "aero"}``, each ``(..., 3)``.
    ``b_eci`` is the magnetic flux density in ECI (T).
    """
    return {
        "gravity_gradient": gravity_gradient_torque(r_eci, q, cfg.inertia, mu),
        "residual_dipole": residual_dipole_torque_eci(cfg.residual_dipole_b, q, b_eci),
        "aero": aero_torque(r_eci, v_eci, q, cfg),
    }


def total_disturbance_torque(r_eci, v_eci, q, b_eci, cfg=DEFAULT.spacecraft,
                             mu: float = GM_EARTH) -> np.ndarray:
    """Sum of gravity-gradient, residual-dipole and aero torques (N m, body axes)."""
    d = disturbance_breakdown(r_eci, v_eci, q, b_eci, cfg, mu)
    return d["gravity_gradient"] + d["residual_dipole"] + d["aero"]
