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
* **Group 8 options** (all off in ``MekfConfig()``; spec
  ``docs/specs/adcs_closed_loop.md`` section 2): magnetometer-bias states
  (error state ``[dtheta, db_g, db_m]``, 9), per-diode coarse-sun-sensor updates
  with field-of-view gating and onboard albedo compensation
  (:meth:`MEKF.update_css_diodes`), TRIAD minimum field-sun angle, public
  per-period cycle :meth:`MEKF.step` shared by :func:`run_mekf` and the closed
  loop (``adcs.simulate.run_closed_loop``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from common import quaternion as quat
from common.ekf import ErrorStateEKF, chi2_gate
from adcs.sensors import CSS_NORMALS, GyroConfig, MagnetometerConfig, SunSensorConfig

__all__ = [
    "triad",
    "triad_covariance",
    "discrete_model",
    "MekfConfig",
    "MEKF",
    "STATE_NAMES",
    "STATE_NAMES_MAG_BIAS",
    "vector_angle",
    "css_vector_complete",
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
    """MEKF noise model, options and initialisation [B-07, B-19, B-22].

    The defaults reproduce the Group 7 filter (6 states, CSS vector update,
    no TRIAD angle guard); ``adcs.simulate.MEKF_DESIGN`` is the Group 8 design.
    """

    arw: float = GyroConfig().arw                    # rad/√s
    rrw: float = GyroConfig().rrw                    # rad/s/√s
    sigma_mag: float = MagnetometerConfig().sigma    # T, per axis (white noise + model error)
    sigma_sun: float = SunSensorConfig().sigma_vector  # per axis of the CSS vector
    sigma_b0: float = GyroConfig().bias0_sigma       # rad/s, initial bias std (b0 = 0)
    init_inflate: float = 1.0                        # factor on the TRIAD covariance
    gate_prob: float | None = None                   # chi-square gate probability; None = off
    use_mag: bool = True
    use_sun: bool = True
    # --- Group 8 options ---------------------------------------------------
    mag_bias: bool = False                           # estimate a magnetometer bias (states 6-8)
    sigma_mb0: float = 200e-9                        # T, initial mag-bias std per axis (bm0 = 0)
    mag_bias_rw: float = 0.0                         # T/√s, mag-bias random walk
    css_mode: str = "vector"                         # "vector" (Group 7) or "diodes"
    sigma_diode: float = SunSensorConfig().sigma_diode  # per-diode current noise (diodes mode)
    diode_min_current: float = 0.55                  # predicted cosine needed to use a diode
    css_fov: float | None = None                     # rad; also require predicted cos > cos(fov)
    albedo_model: float = 0.0                        # onboard albedo coefficient (diodes mode)
    sigma_albedo: float = 0.0                        # albedo-coefficient uncertainty -> R inflation
    triad_min_angle: float = 0.0                     # rad, minimum field-sun angle for TRIAD
    init_att_sigma_floor: float = 0.0                # rad, added in quadrature to TRIAD P (per axis)

    def __post_init__(self):
        if self.css_mode not in ("vector", "diodes"):
            raise ValueError(f"css_mode must be 'vector' or 'diodes', got {self.css_mode!r}")

    @property
    def n_states(self) -> int:
        return 9 if self.mag_bias else 6


STATE_NAMES = ["dth_x", "dth_y", "dth_z", "db_x", "db_y", "db_z"]


def _cross3(a, b) -> np.ndarray:
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _rot_inv(q, v) -> np.ndarray:
    """``R(q)^T v`` for one unit quaternion and one 3-vector (fast path of ``quat.rotate_inverse``)."""
    u = q[1:4]
    uv = _cross3(u, v)
    return v - 2.0 * q[0] * uv + 2.0 * _cross3(u, uv)


STATE_NAMES_MAG_BIAS = STATE_NAMES + ["dbm_x", "dbm_y", "dbm_z"]


class MEKF:
    """Multiplicative EKF for attitude + gyro bias (+ optional magnetometer bias).

    Error state ``[dθ, db]`` (6) or ``[dθ, db, dbm]`` (9, ``cfg.mag_bias``) with
    ``q_true = q ⊗ exp(dθ)``, ``b_true = b + db``, ``bm_true = bm + dbm``.
    ``q0`` body->ECI quaternion, ``b0`` gyro bias [rad/s], ``P0`` (n, n),
    ``bm0`` magnetometer bias [T] (9-state only, default 0).
    Wraps :class:`common.ekf.ErrorStateEKF`; every accepted update is
    injected into the nominal state and followed by the covariance reset.
    """

    def __init__(self, q0, b0, P0, cfg: MekfConfig = MekfConfig(), bm0=None):
        self.cfg = cfg
        self.n = cfg.n_states
        self._q = quat.normalize(np.asarray(q0, dtype=float))
        self._b = np.asarray(b0, dtype=float).copy()
        self._bm = np.zeros(3) if bm0 is None else np.asarray(bm0, dtype=float).copy()
        P0 = np.asarray(P0, dtype=float)
        if P0.shape != (self.n, self.n):
            raise ValueError(f"P0 must be ({self.n}, {self.n}) for this config")
        names = STATE_NAMES_MAG_BIAS if cfg.mag_bias else STATE_NAMES
        self.ekf = ErrorStateEKF(P0, state_names=names)
        self._gates: dict[int, float | None] = {}

    # -- state access ------------------------------------------------------

    @property
    def q(self) -> np.ndarray:
        return self._q.copy()

    @property
    def b(self) -> np.ndarray:
        return self._b.copy()

    @property
    def bm(self) -> np.ndarray:
        """Estimated magnetometer bias [T] (zeros for the 6-state filter)."""
        return self._bm.copy()

    @property
    def P(self) -> np.ndarray:
        return self.ekf.P

    @property
    def std(self) -> np.ndarray:
        return self.ekf.std

    @property
    def att_sigma(self) -> float:
        """``sqrt(trace(P_att))`` [rad], total attitude 1-sigma."""
        return float(np.sqrt(max(np.trace(self.ekf._P[:3, :3]), 0.0)))

    def _gate(self, dof: int):
        if self.cfg.gate_prob is None:
            return None
        if dof not in self._gates:
            self._gates[dof] = chi2_gate(dof, self.cfg.gate_prob)
        return self._gates[dof]

    # -- propagation -------------------------------------------------------

    def propagate(self, w_meas, dt: float) -> None:
        """Propagate nominal state and covariance with one gyro sample (mean rate over ``dt``)."""
        w_hat = np.asarray(w_meas, dtype=float) - self._b
        self._q = quat.normalize(quat.multiply(self._q, quat.from_rotvec(w_hat * dt)))
        Phi6, Qd6 = discrete_model(w_hat, dt, self.cfg.arw, self.cfg.rrw)
        if self.n == 6:
            self.ekf.predict(Phi6, Qd6)
            return
        Phi = np.eye(9)
        Phi[:6, :6] = Phi6
        Qd = np.zeros((9, 9))
        Qd[:6, :6] = Qd6
        Qd[6:, 6:] = (self.cfg.mag_bias_rw ** 2 * float(dt)) * np.eye(3)
        self.ekf.predict(Phi, Qd)

    # -- updates -----------------------------------------------------------

    def _apply(self, res) -> None:
        """Inject an accepted correction and reset the covariance."""
        if not res.accepted:
            return
        dth = res.dx[:3]
        self._q = quat.inject_attitude_error(self._q, dth)
        self._b = self._b + res.dx[3:6]
        if self.n == 9:
            self._bm = self._bm + res.dx[6:9]
        G = np.eye(self.n)
        G[:3, :3] = quat.attitude_reset_jacobian(dth)
        self.ekf.reset(G)

    def update_vector(self, z, v_eci, sigma: float, name: str, t: float | None = None,
                      offset=None, offset_jacobian=None):
        """Update with ``z = R(q)ᵀ v_eci + offset + n``, ``n ~ N(0, σ² I)``; returns the ``UpdateResult``.

        ``offset`` (3,) is added to the prediction; ``offset_jacobian`` (3, n−6)
        fills the H columns after the gyro-bias block (used for the magnetometer bias).
        """
        v_b = _rot_inv(self._q, np.asarray(v_eci, dtype=float))
        pred = v_b if offset is None else v_b + offset
        r = np.asarray(z, dtype=float) - pred
        H = np.zeros((3, self.n))
        H[:, :3] = quat.skew(v_b)
        if offset_jacobian is not None:
            H[:, 6:] = offset_jacobian
        res = self.ekf.update(r, H, float(sigma) ** 2, name=name, t=t, gate=self._gate(3))
        self._apply(res)
        return res

    def update_mag(self, z, b_eci, t: float | None = None):
        """Magnetometer update; ``z`` and ``b_eci`` in T (``z = R(q)ᵀ b_eci + bm + n``)."""
        if self.n == 9:
            return self.update_vector(z, b_eci, self.cfg.sigma_mag, "mag", t,
                                      offset=self._bm, offset_jacobian=np.eye(3))
        return self.update_vector(z, b_eci, self.cfg.sigma_mag, "mag", t)

    def update_sun(self, z, s_eci, t: float | None = None):
        """Coarse-sun-sensor vector update; ``z`` = reconstructed CSS vector, ``s_eci`` unit."""
        return self.update_vector(z, s_eci, self.cfg.sigma_sun, "sun", t)

    def predicted_albedo(self, s_eci, r_eci) -> np.ndarray:
        """Onboard albedo prediction (6,) [fraction of full scale] at the current estimate."""
        if self.cfg.albedo_model <= 0.0 or r_eci is None:
            return np.zeros(6)
        from adcs.sensors import albedo_currents
        return albedo_currents(self._q, r_eci, s_eci, self.cfg.albedo_model)

    def update_css_diodes(self, currents, s_eci, r_eci=None, t: float | None = None):
        """Per-diode sun update (block of the usable diodes); returns the ``UpdateResult`` or ``None``.

        Model for diode i: ``I_i = n_i · R(q)ᵀ s_eci + a_i + n``, ``a_i`` = onboard
        albedo prediction (:meth:`predicted_albedo`, needs ``r_eci``; its attitude
        dependence is neglected in H), ``H_i = [n_iᵀ [s_b×], 0]``,
        ``R_ii = σ_d² + (sigma_albedo · a_i / albedo_model)²`` (albedo-coefficient
        uncertainty times the predicted geometry factor). Diode i is used when the
        *predicted* direct cosine exceeds ``diode_min_current`` (and ``cos(css_fov)``
        if set) and the measured current exceeds ``diode_min_current / 2`` (rejects
        dark diodes, e.g. in eclipse). Gating on the prediction, not on the noisy
        current, avoids selecting samples by their noise.
        """
        c = np.asarray(currents, dtype=float)
        c_min = self.cfg.diode_min_current
        lit = c > 0.5 * c_min
        if not lit.any():
            return None
        s_eci = np.asarray(s_eci, dtype=float)
        s_b = _rot_inv(self._q, s_eci)
        cos_pred = CSS_NORMALS @ s_b
        use = lit & (cos_pred > c_min)
        if self.cfg.css_fov is not None:
            use &= cos_pred > np.cos(self.cfg.css_fov)
        idx = np.nonzero(use)[0]
        if idx.size == 0:
            return None
        alb = self.predicted_albedo(s_eci, r_eci)
        nrm = CSS_NORMALS[idx]
        r = c[idx] - (cos_pred[idx] + alb[idx])
        H = np.zeros((idx.size, self.n))
        H[:, :3] = nrm @ quat.skew(s_b)
        var = np.full(idx.size, self.cfg.sigma_diode ** 2)
        if self.cfg.albedo_model > 0.0 and self.cfg.sigma_albedo > 0.0:
            var = var + (self.cfg.sigma_albedo * alb[idx] / self.cfg.albedo_model) ** 2
        res = self.ekf.update(r, H, np.diag(var), name="css", t=t, gate=self._gate(idx.size))
        self._apply(res)
        return res

    def step(self, gyro, dt: float, mag=None, b_eci=None, sun=None, s_eci=None,
             sun_valid: bool = False, currents=None, r_eci=None, t: float | None = None) -> None:
        """One filter cycle (sampling model, ``docs/specs/adcs_estimation.md`` §0).

        Propagate with ``gyro`` over ``dt``; then the magnetometer update (if
        ``cfg.use_mag`` and ``mag`` is given); then the sun update if
        ``cfg.use_sun``: ``css_mode="vector"`` uses ``sun`` when ``sun_valid``,
        ``"diodes"`` uses ``currents`` (and ``r_eci`` for the albedo model).
        """
        self.propagate(gyro, dt)
        if self.cfg.use_mag and mag is not None:
            self.update_mag(mag, b_eci, t)
        if not self.cfg.use_sun:
            return
        if self.cfg.css_mode == "vector":
            if sun_valid and sun is not None:
                self.update_sun(sun, s_eci, t)
        elif currents is not None:
            self.update_css_diodes(currents, s_eci, r_eci, t)


def vector_angle(a, b) -> float:
    """Angle [rad] between two non-zero 3-vectors (any length)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b))


def css_vector_complete(currents, min_current: float) -> bool:
    """True when each body axis has one lit diode above ``min_current`` (all three
    components of the reconstructed CSS vector are then direct measurements)."""
    c = np.asarray(currents, float)
    return bool(np.all(np.maximum(c[0::2], c[1::2]) > min_current))


def triad_init(mag_b, sun_b, b_eci, s_eci, cfg: MekfConfig = MekfConfig()):
    """TRIAD attitude ``q0`` and initial covariance ``P0`` (n, n) for the MEKF.

    The primary vector is the one with the smaller unit-vector noise
    (magnetometer: ``sigma_mag/|b|``, with ``sigma_mb0`` added in quadrature for
    the 9-state filter; sun: ``sigma_sun``). ``b0 = 0``, ``bm0 = 0``. The
    attitude block is ``init_inflate · P_triad + init_att_sigma_floor² I``.
    """
    mag_b, sun_b = np.asarray(mag_b, float), np.asarray(sun_b, float)
    sig_m = np.hypot(cfg.sigma_mag, cfg.sigma_mb0) if cfg.mag_bias else cfg.sigma_mag
    s_mag = sig_m / np.linalg.norm(mag_b)
    s_sun = cfg.sigma_sun
    if s_mag <= s_sun:
        q0 = triad(mag_b, sun_b, b_eci, s_eci)
        P_att = triad_covariance(mag_b, sun_b, s_mag, s_sun)
    else:
        q0 = triad(sun_b, mag_b, s_eci, b_eci)
        P_att = triad_covariance(sun_b, mag_b, s_sun, s_mag)
    P0 = np.zeros((cfg.n_states, cfg.n_states))
    P0[:3, :3] = cfg.init_inflate * P_att + cfg.init_att_sigma_floor ** 2 * np.eye(3)
    P0[3:6, 3:6] = cfg.sigma_b0 ** 2 * np.eye(3)
    if cfg.mag_bias:
        P0[6:, 6:] = cfg.sigma_mb0 ** 2 * np.eye(3)
    return q0, P0


@dataclass
class EstimationResult:
    """MEKF output on the sensor time grid (NaN before initialisation)."""

    t: np.ndarray          # (N,) s
    q: np.ndarray          # (N, 4) estimated body->ECI
    b: np.ndarray          # (N, 3) estimated gyro bias, rad/s
    std: np.ndarray        # (N, n) 1-sigma of the error state [dθ (rad), db (rad/s), dbm (T)]
    P_att: np.ndarray      # (N, 3, 3) attitude-error covariance, rad²
    k_init: int            # index of the TRIAD initialisation (-1 if never)
    nis: dict              # name -> (t, nis, accepted) arrays; dof per row in meta["nis_dof"]
    meta: dict
    bm: np.ndarray | None = None   # (N, 3) estimated magnetometer bias, T (9-state only)


def run_mekf(data, b_eci, s_eci, cfg: MekfConfig = MekfConfig(), q0=None, P0=None,
             b0=None, k_init: int | None = None, r_eci=None) -> EstimationResult:
    """Run the MEKF over a :class:`adcs.sensors.SensorData` record.

    ``b_eci (N, 3)`` [T] and ``s_eci (N, 3)`` (unit) are the filter's reference
    vectors at ``data.t``; ``r_eci (N, 3)`` [m] is only needed for the onboard
    albedo model in ``css_mode="diodes"``. By default the filter starts at the
    first sample with a valid sun measurement (``sun_valid``; in diodes mode
    additionally :func:`css_vector_complete`) whose field-sun angle is at least
    ``cfg.triad_min_angle``, from TRIAD (:func:`triad_init`); passing ``q0`` and
    ``P0`` (and optionally ``b0``, ``k_init``, default 0) starts it there
    instead. At each later sample k: :meth:`MEKF.step`.
    """
    t = np.asarray(data.t, dtype=float)
    n = t.size
    b_eci = np.asarray(b_eci, dtype=float)
    s_eci = np.asarray(s_eci, dtype=float)
    ns = cfg.n_states
    out_q = np.full((n, 4), np.nan)
    out_b = np.full((n, 3), np.nan)
    out_bm = np.full((n, 3), np.nan)
    out_std = np.full((n, ns), np.nan)
    out_P = np.full((n, 3, 3), np.nan)
    currents = getattr(data, "css_currents", None)

    def result(k0, f, init):
        nis = {}
        dof = {}
        if f is not None:
            for name in ("mag", "sun", "css"):
                log = f.ekf.innovations(name)
                if log.nis.size or name != "css":
                    nis[name] = (log.t, log.nis, log.accepted)
                    dof[name] = log.dof
        meta = {"cfg": asdict(cfg), "init": init, "nis_dof": dof}
        return EstimationResult(t, out_q, out_b, out_std, out_P, k0, nis, meta,
                                out_bm if cfg.mag_bias else None)

    if q0 is not None:
        if P0 is None:
            raise ValueError("P0 required with q0")
        k0 = 0 if k_init is None else int(k_init)
        f = MEKF(q0, np.zeros(3) if b0 is None else b0, P0, cfg)
        init = "given"
    else:
        ok = np.asarray(data.sun_valid, bool).copy()
        if cfg.css_mode == "diodes" and currents is not None:
            ok &= np.array([css_vector_complete(c, cfg.diode_min_current) for c in currents])
        cand = np.nonzero(ok)[0]
        if cfg.triad_min_angle > 0.0:
            cand = [k for k in cand if vector_angle(data.mag[k], data.sun[k]) >= cfg.triad_min_angle]
        if len(cand) == 0:
            return result(-1, None, "none")
        k0 = int(cand[0])
        q_init, P_init = triad_init(data.mag[k0], data.sun[k0], b_eci[k0], s_eci[k0], cfg)
        f = MEKF(q_init, np.zeros(3) if b0 is None else b0, P_init, cfg)
        init = "triad"

    def store(k):
        out_q[k] = f.q
        out_b[k] = f.b
        out_bm[k] = f.bm
        out_std[k] = f.std
        out_P[k] = f.ekf._P[:3, :3]

    store(k0)
    for k in range(k0 + 1, n):
        f.step(data.gyro[k], t[k] - t[k - 1], data.mag[k], b_eci[k], data.sun[k], s_eci[k],
               bool(data.sun_valid[k]), None if currents is None else currents[k],
               None if r_eci is None else r_eci[k], t[k])
        store(k)
    return result(k0, f, init)


def attitude_errors(q_true, q_est) -> np.ndarray:
    """Body-frame attitude error ``dθ (N, 3)`` [rad] with ``q_true = q_est ⊗ exp(dθ)``."""
    return quat.error_rotvec(q_est, q_true)


def nees(err, P) -> np.ndarray:
    """Normalised estimation error squared ``eᵀ P⁻¹ e`` per row; ``err (N, m)``, ``P (N, m, m)``."""
    err = np.asarray(err, float)
    P = np.asarray(P, float)
    return np.einsum("ni,ni->n", err, np.linalg.solve(P, err[..., None])[..., 0])
