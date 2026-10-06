"""Attitude determination for the 3U CubeSat ADCS (Project B).

Frames and units (CONVENTIONS.md): quaternions scalar-first Hamilton, body-to-ECI;
vectors in the frame named by the argument (``*_b`` spacecraft body axes, ``*_r``
reference = ECI); angles in rad.  Error convention ``q_true = q_est (x) exp(dtheta)``
(right / body-frame error).

Contents
--------
* **TRIAD** (this part): deterministic two-vector attitude, ``triad`` and its
  small-noise covariance ``triad_covariance``.
* **MEKF**: multiplicative EKF on :class:`common.ekf.ErrorStateEKF`, error state
  ``[dtheta (3), db_g (3)]``, gyro propagation with closed-form ``(Phi, Qd)``
  (:func:`discrete_model`), magnetometer and coarse-sun-sensor vector updates,
  TRIAD initialisation (:func:`triad_init`), batch runner :func:`run_mekf`.
  Spec: ``docs/specs/adcs_estimation.md`` section 3; derivation in the comment
  block above :func:`discrete_model`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from common import quaternion as quat
from common.ekf import ErrorStateEKF, chi2_gate
from adcs.sensors import GyroConfig, MagnetometerConfig, SunSensorConfig

__all__ = [
    "triad",
    "triad_covariance",
    "discrete_model",
    "MekfConfig",
    "MEKF",
    "STATE_NAMES",
    "triad_init",
    "EstimationResult",
    "run_mekf",
    "attitude_errors",
    "nees",
]

_MIN_SIN = 1e-6


def _unit(v, name: str) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(n == 0.0) or not np.all(np.isfinite(n)):
        raise ValueError(f"{name} must be finite and non-zero")
    return v / n


def _triad_frame(v1, v2, label: str) -> np.ndarray:
    """Columns ``[t1 t2 t3] (..., 3, 3)`` of the triad built from unit vectors ``v1, v2``."""
    c = np.cross(v1, v2)
    s = np.linalg.norm(c, axis=-1, keepdims=True)
    if np.any(s < _MIN_SIN):
        raise ValueError(f"{label} vectors are (anti)parallel: |v1 x v2| < {_MIN_SIN}")
    t2 = c / s
    t3 = np.cross(v1, t2)
    return np.stack([v1, t2, t3], axis=-1)


def triad(v1_b, v2_b, v1_r, v2_r) -> np.ndarray:
    """TRIAD attitude ``(..., 4)`` (canonical, w >= 0), body-to-reference quaternion.

    ``v1_b, v2_b`` are two vectors in body axes, ``v1_r, v2_r`` the same vectors in
    the reference (ECI) frame, each ``(..., 3)`` with any non-zero length (units
    arbitrary; normalised internally).  ``v1`` is the **primary**: ``R(q) v1_b``
    equals ``v1_r`` exactly, ``v2`` only enters through the plane it defines.

    Triads ``t1 = v1, t2 = v1 x v2 / |v1 x v2|, t3 = t1 x t2`` in each frame and
    ``R = [t1r t2r t3r] [t1b t2b t3b]^T``.

    Raises
    ------
    ValueError
        If a vector has zero length, or ``|v1 x v2| < 1e-6`` (unit vectors) in either
        frame for any element.
    """
    v1b, v2b = _unit(v1_b, "v1_b"), _unit(v2_b, "v2_b")
    v1r, v2r = _unit(v1_r, "v1_r"), _unit(v2_r, "v2_r")
    Tb = _triad_frame(v1b, v2b, "body")
    Tr = _triad_frame(v1r, v2r, "reference")
    R = Tr @ np.swapaxes(Tb, -1, -2)
    return quat.canonicalize(quat.from_dcm(R))


def triad_covariance(v1_b, v2_b, sigma1: float, sigma2: float) -> np.ndarray:
    """Body-frame covariance ``(3, 3)`` [rad^2] of the TRIAD attitude error ``dtheta``.

    Single (unbatched) case, small-noise approximation, ``q_true = q_triad (x) exp(dtheta)``.
    ``v1_b, v2_b`` (3,) are the body-frame vectors (any non-zero length; v1 primary);
    ``sigma_i`` [rad] is the per-axis std of the unit-vector noise perpendicular to
    ``v_i`` (magnetometer: ``sigma_m / |b|``; sun sensor: ``sigma_vector``).

    With ``e1 = v1``, ``e2 = v1 x v2 / sin(a)``, ``e3 = e1 x e2``, ``cos(a) = v1 . v2``::

        P = s1^2 (I - e1 e1^T) + ((s2^2 + s1^2 cos^2 a) / sin^2 a) e1 e1^T
            - (s1^2 cos a / sin a) (e1 e3^T + e3 e1^T)

    Raises
    ------
    ValueError
        If the vectors are (anti)parallel (``sin a < 1e-6``) or zero.
    """
    v1 = _unit(np.asarray(v1_b, dtype=float).reshape(3), "v1_b")
    v2 = _unit(np.asarray(v2_b, dtype=float).reshape(3), "v2_b")
    c = np.cross(v1, v2)
    sin_a = float(np.linalg.norm(c))
    if sin_a < _MIN_SIN:
        raise ValueError(f"vectors are (anti)parallel: sin(alpha) < {_MIN_SIN}")
    cos_a = float(v1 @ v2)
    e1 = v1
    e2 = c / sin_a
    e3 = np.cross(e1, e2)
    s1sq, s2sq = float(sigma1) ** 2, float(sigma2) ** 2
    E11 = np.outer(e1, e1)
    P = (
        s1sq * (np.eye(3) - E11)
        + ((s2sq + s1sq * cos_a**2) / sin_a**2) * E11
        - (s1sq * cos_a / sin_a) * (np.outer(e1, e3) + np.outer(e3, e1))
    )
    return 0.5 * (P + P.T)

# ---------------------------------------------------------------------------
# Multiplicative EKF (MEKF)
# ---------------------------------------------------------------------------
#
# Nominal state: q (body -> ECI, unit), b (gyro bias, rad/s, body axes).
# Error state (6): x = [dθ, db] with q_true = q ⊗ exp(dθ), b_true = b + db.
#
# Gyro model (adcs.sensors): w_m = w + b_true + n_v, b_true' = n_u.
# Nominal propagation with w_hat = w_m - b:  q <- q ⊗ exp(w_hat dt).
#
# Error dynamics. With w = w_m - b_true - n_v = w_hat - db - n_v and the
# right error, d/dt exp(dθ) = ... gives to first order
#     dθ' = -[w_hat x] dθ + (w - w_hat) = -W dθ - db - n_v,      W = [w_hat x]
#     db' = n_u
# so F = [[-W, -I], [0, 0]], noise G = [[-I, 0], [0, I]], Qc = diag(σv² I, σu² I).
#
# Closed-form discretisation (constant w_hat over dt; u = |w_hat|, θ = u dt):
#     Phi11 = exp(-W dt) = I - (sin θ / u) W + ((1 - cos θ) / u²) W²
#     Phi12 = -∫0^dt exp(-W s) ds = -[dt I - ((1 - cos θ)/u²) W + ((θ - sin θ)/u³) W²]
#     Q11 = σv² dt I + σu² [dt³/3 I + ((θ³/3 + 2 sin θ - 2θ)/u⁵) W²]
#     Q12 = -σu² [dt²/2 I - ((θ - sin θ)/u³) W + ((θ²/2 + cos θ - 1)/u⁴) W²]
#     Q22 = σu² dt I
# (Phi11 is orthogonal, so ∫ Phi11 Phi11ᵀ = dt I; ∫ Phi12 Phi12ᵀ uses
# W³ = -u² W.) Same result as Crassidis & Junkins, *Optimal Estimation of
# Dynamic Systems*, 2nd ed., §7.2 (equation numbers unverified); checked
# against common.ekf.van_loan in the tests.
# The simulated gyro (adcs.sensors.Gyro) steps its bias once per sample, so its
# exact discrete noise is Q11 = σv² dt + σu² dt³, Q12 = -σu² dt² (dt = 1 s)
# instead of the continuous-model σu² dt³/3 and -σu² dt²/2 used here; the
# difference is ~2e-5 of Q11 and 1e-12 absolute in Q12 (★ review), negligible.
#
# Vector measurement z = R(q_true)ᵀ v + n, E[n nᵀ] = σ² I. With
# R(q ⊗ exp(dθ))ᵀ v = exp(-[dθ x]) R(q)ᵀ v ≈ v_b - dθ x v_b = v_b + [v_b x] dθ,
# where v_b = R(q)ᵀ v:  r = z - v_b,  H = [[v_b x], 0].


def _series_coeffs(u: float, dt: float):
    """Coefficients of W and W² in Phi and Qd (see comment above), stable for small u."""
    th = u * dt
    if th < 1e-2:
        # Taylor series in θ (relative truncation error < 1e-12 for θ < 1e-2)
        t2 = th * th
        a1 = dt * (1 - t2 / 6 + t2 * t2 / 120)                      # sin θ / u
        a2 = dt**2 * (0.5 - t2 / 24 + t2 * t2 / 720)                # (1 - cos θ) / u²
        a3 = dt**3 * (1 / 6 - t2 / 120 + t2 * t2 / 5040)            # (θ - sin θ) / u³
        a4 = dt**4 * (1 / 24 - t2 / 720 + t2 * t2 / 40320)          # (θ²/2 + cos θ - 1) / u⁴
        a5 = dt**5 * (1 / 60 - t2 / 2520 + t2 * t2 / 181440)         # (θ³/3 + 2 sin θ - 2θ) / u⁵
    else:
        s, c = np.sin(th), np.cos(th)
        a1 = s / u
        a2 = (1 - c) / u**2
        a3 = (th - s) / u**3
        a4 = (th * th / 2 + c - 1) / u**4
        a5 = (th**3 / 3 + 2 * s - 2 * th) / u**5
    return a1, a2, a3, a4, a5


def discrete_model(w_hat, dt: float, arw: float, rrw: float):
    """Closed-form ``(Phi, Qd)`` (6 x 6) of the MEKF error model over ``dt`` [s].

    ``w_hat`` (3,) bias-corrected body rate [rad/s], held constant over ``dt``;
    ``arw`` = σ_v [rad/√s], ``rrw`` = σ_u [rad/s/√s]. Error state ``[dθ, db]``.
    Equal to ``van_loan(F, diag(σv², σu²), dt, G)`` to round-off.
    """
    w_hat = np.asarray(w_hat, dtype=float)
    dt = float(dt)
    u = float(np.linalg.norm(w_hat))
    W = quat.skew(w_hat)
    W2 = W @ W
    I3 = np.eye(3)
    a1, a2, a3, a4, a5 = _series_coeffs(u, dt)
    sv2, su2 = float(arw) ** 2, float(rrw) ** 2
    Phi = np.eye(6)
    Phi[:3, :3] = I3 - a1 * W + a2 * W2
    Phi[:3, 3:] = -(dt * I3 - a2 * W + a3 * W2)
    Q11 = sv2 * dt * I3 + su2 * (dt**3 / 3 * I3 + a5 * W2)
    Q12 = -su2 * (dt**2 / 2 * I3 - a3 * W + a4 * W2)
    Qd = np.zeros((6, 6))
    Qd[:3, :3] = Q11
    Qd[:3, 3:] = Q12
    Qd[3:, :3] = Q12.T
    Qd[3:, 3:] = su2 * dt * I3
    return Phi, 0.5 * (Qd + Qd.T)


@dataclass(frozen=True)
class MekfConfig:
    """MEKF noise model and initialisation; defaults match ``adcs.sensors`` [B-07, B-19]."""

    arw: float = GyroConfig().arw                    # rad/√s
    rrw: float = GyroConfig().rrw                    # rad/s/√s
    sigma_mag: float = MagnetometerConfig().sigma    # T, per axis
    sigma_sun: float = SunSensorConfig().sigma_vector  # per axis of the CSS vector
    sigma_b0: float = GyroConfig().bias0_sigma       # rad/s, initial bias std (b0 = 0)
    init_inflate: float = 1.0                        # factor on the TRIAD covariance
    gate_prob: float | None = None                   # chi-square gate probability; None = off
    use_mag: bool = True
    use_sun: bool = True


STATE_NAMES = ["dth_x", "dth_y", "dth_z", "db_x", "db_y", "db_z"]


class MEKF:
    """Multiplicative EKF for attitude + gyro bias (error state ``[dθ, db]``).

    ``q0`` body->ECI quaternion, ``b0`` gyro bias [rad/s], ``P0`` (6, 6).
    Wraps :class:`common.ekf.ErrorStateEKF`; every accepted update is
    injected into ``(q, b)`` and followed by the covariance reset.
    """

    def __init__(self, q0, b0, P0, cfg: MekfConfig = MekfConfig()):
        self.cfg = cfg
        self._q = quat.normalize(np.asarray(q0, dtype=float))
        self._b = np.asarray(b0, dtype=float).copy()
        self.ekf = ErrorStateEKF(P0, state_names=STATE_NAMES)
        self._gate = {3: None if cfg.gate_prob is None else chi2_gate(3, cfg.gate_prob)}

    @property
    def q(self) -> np.ndarray:
        return self._q.copy()

    @property
    def b(self) -> np.ndarray:
        return self._b.copy()

    @property
    def P(self) -> np.ndarray:
        return self.ekf.P

    @property
    def std(self) -> np.ndarray:
        return self.ekf.std

    def propagate(self, w_meas, dt: float) -> None:
        """Propagate nominal state and covariance with one gyro sample (mean rate over ``dt``)."""
        w_hat = np.asarray(w_meas, dtype=float) - self._b
        self._q = quat.normalize(quat.multiply(self._q, quat.from_rotvec(w_hat * dt)))
        Phi, Qd = discrete_model(w_hat, dt, self.cfg.arw, self.cfg.rrw)
        self.ekf.predict(Phi, Qd)

    def update_vector(self, z, v_eci, sigma: float, name: str, t: float | None = None):
        """Update with ``z = R(q)ᵀ v_eci + n``, ``n ~ N(0, σ² I)``; returns the ``UpdateResult``."""
        v_b = quat.rotate_inverse(self._q, np.asarray(v_eci, dtype=float))
        r = np.asarray(z, dtype=float) - v_b
        H = np.zeros((3, 6))
        H[:, :3] = quat.skew(v_b)
        res = self.ekf.update(r, H, float(sigma) ** 2, name=name, t=t, gate=self._gate[3])
        if res.accepted:
            dth, db = res.dx[:3], res.dx[3:]
            self._q = quat.inject_attitude_error(self._q, dth)
            self._b = self._b + db
            G = np.eye(6)
            G[:3, :3] = quat.attitude_reset_jacobian(dth)
            self.ekf.reset(G)
        return res

    def update_mag(self, z, b_eci, t: float | None = None):
        """Magnetometer update; ``z`` and ``b_eci`` in T."""
        return self.update_vector(z, b_eci, self.cfg.sigma_mag, "mag", t)

    def update_sun(self, z, s_eci, t: float | None = None):
        """Coarse-sun-sensor update; ``z`` = reconstructed CSS vector, ``s_eci`` unit."""
        return self.update_vector(z, s_eci, self.cfg.sigma_sun, "sun", t)


def triad_init(mag_b, sun_b, b_eci, s_eci, cfg: MekfConfig = MekfConfig()):
    """TRIAD attitude ``q0`` and initial covariance ``P0`` (6, 6) for the MEKF.

    The primary vector is the one with the smaller unit-vector noise
    (magnetometer: ``sigma_mag/|b|``; sun: ``sigma_sun``). ``b0 = 0``.
    """
    mag_b, sun_b = np.asarray(mag_b, float), np.asarray(sun_b, float)
    s_mag = cfg.sigma_mag / np.linalg.norm(mag_b)
    s_sun = cfg.sigma_sun
    if s_mag <= s_sun:
        q0 = triad(mag_b, sun_b, b_eci, s_eci)
        P_att = triad_covariance(mag_b, sun_b, s_mag, s_sun)
    else:
        q0 = triad(sun_b, mag_b, s_eci, b_eci)
        P_att = triad_covariance(sun_b, mag_b, s_sun, s_mag)
    P0 = np.zeros((6, 6))
    P0[:3, :3] = cfg.init_inflate * P_att
    P0[3:, 3:] = cfg.sigma_b0 ** 2 * np.eye(3)
    return q0, P0


@dataclass
class EstimationResult:
    """MEKF output on the sensor time grid (NaN before initialisation)."""

    t: np.ndarray          # (N,) s
    q: np.ndarray          # (N, 4) estimated body->ECI
    b: np.ndarray          # (N, 3) estimated gyro bias, rad/s
    std: np.ndarray        # (N, 6) 1-sigma of [dθ (rad), db (rad/s)]
    P_att: np.ndarray      # (N, 3, 3) attitude-error covariance, rad²
    k_init: int            # index of the TRIAD initialisation (-1 if never)
    nis: dict              # name -> (t, nis, accepted) arrays
    meta: dict


def run_mekf(data, b_eci, s_eci, cfg: MekfConfig = MekfConfig(), q0=None, P0=None,
             b0=None, k_init: int | None = None) -> EstimationResult:
    """Run the MEKF over a :class:`adcs.sensors.SensorData` record.

    ``b_eci (N, 3)`` [T] and ``s_eci (N, 3)`` (unit) are the filter's reference
    vectors at ``data.t``. By default the filter starts at the first sample
    with a valid sun measurement, from TRIAD (:func:`triad_init`); passing
    ``q0`` and ``P0`` (and optionally ``b0``, ``k_init``, default 0) starts it
    there instead. At each later sample k: propagate with ``gyro[k]``, then
    magnetometer update, then sun update if ``sun_valid[k]``.
    """
    t = np.asarray(data.t, dtype=float)
    n = t.size
    b_eci = np.asarray(b_eci, dtype=float)
    s_eci = np.asarray(s_eci, dtype=float)
    out_q = np.full((n, 4), np.nan)
    out_b = np.full((n, 3), np.nan)
    out_std = np.full((n, 6), np.nan)
    out_P = np.full((n, 3, 3), np.nan)

    if q0 is not None:
        if P0 is None:
            raise ValueError("P0 required with q0")
        k0 = 0 if k_init is None else int(k_init)
        f = MEKF(q0, np.zeros(3) if b0 is None else b0, P0, cfg)
    else:
        valid = np.nonzero(np.asarray(data.sun_valid, bool))[0]
        if valid.size == 0:
            return EstimationResult(t, out_q, out_b, out_std, out_P, -1, {}, {"init": "none"})
        k0 = int(valid[0])
        q_init, P_init = triad_init(data.mag[k0], data.sun[k0], b_eci[k0], s_eci[k0], cfg)
        f = MEKF(q_init, np.zeros(3) if b0 is None else b0, P_init, cfg)

    def store(k):
        out_q[k] = f._q
        out_b[k] = f._b
        P = f.ekf._P
        out_std[k] = np.sqrt(np.clip(np.diag(P), 0.0, None))
        out_P[k] = P[:3, :3]

    store(k0)
    for k in range(k0 + 1, n):
        dt = t[k] - t[k - 1]
        f.propagate(data.gyro[k], dt)
        if cfg.use_mag:
            f.update_mag(data.mag[k], b_eci[k], t[k])
        if cfg.use_sun and data.sun_valid[k]:
            f.update_sun(data.sun[k], s_eci[k], t[k])
        store(k)

    nis = {}
    for name in ("mag", "sun"):
        log = f.ekf.innovations(name)
        nis[name] = (log.t, log.nis, log.accepted)
    meta = {"cfg": asdict(cfg), "init": "triad" if q0 is None else "given"}
    return EstimationResult(t, out_q, out_b, out_std, out_P, k0, nis, meta)


def attitude_errors(q_true, q_est) -> np.ndarray:
    """Body-frame attitude error ``dθ (N, 3)`` [rad] with ``q_true = q_est ⊗ exp(dθ)``."""
    return quat.error_rotvec(q_est, q_true)


def nees(err, P) -> np.ndarray:
    """Normalised estimation error squared ``eᵀ P⁻¹ e`` per row; ``err (N, m)``, ``P (N, m, m)``."""
    err = np.asarray(err, float)
    P = np.asarray(P, float)
    return np.einsum("ni,ni->n", err, np.linalg.solve(P, err[..., None])[..., 0])
