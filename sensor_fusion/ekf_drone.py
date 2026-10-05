"""15-state error-state EKF for PX4 multirotor logs (Project A, M3).

Spec: ``docs/specs/ekf_drone.md``. Conventions: CONVENTIONS.md (§2 quaternions,
§3 error state, §4 frames and IMU model, §6 PX4 mapping). Assumptions: A-01 … A-12.

Frames and state
----------------
Navigation frame: local NED about the first GPS fix, treated as inertial (A-01).
Body: PX4 FRD. Nominal state ``x = (p, v, q, b_g, b_a)``: NED position [m],
NED velocity [m/s], FRD→NED quaternion (scalar-first Hamilton), gyro bias
[rad/s], accelerometer bias [m/s²]. Error state, fixed order::

    δx = [δp(0:3), δv(3:6), δθ(6:9), δb_g(9:12), δb_a(12:15)]

with ``p_true = p + δp`` (likewise v, b_g, b_a) and the **body-frame right
perturbation** ``q_true = q ⊗ exp(δθ)``, i.e. ``R_true = R (I + [δθ×])`` to
first order.

Error-state derivation
----------------------
IMU model (CONVENTIONS §4): ``ω_m = ω_true + b_g + n_g``,
``f_m = R_trueᵀ (a − g_n) + b_a + n_a``, ``ḃ_g = n_bg``, ``ḃ_a = n_ba``,
``g_n = [0, 0, g]``. Nominal kinematics use ``ω = ω_m − b_g``, ``f = f_m − b_a``::

    ṗ = v,   v̇ = R f + g_n,   q̇ = ½ q ⊗ [0, ω],   ḃ_g = 0,   ḃ_a = 0.

Velocity error. ``v̇_true = R (I + [δθ×]) (f − δb_a − n_a) + g_n``; subtracting
the nominal and keeping first-order terms, with ``[δθ×] f = −[f×] δθ``::

    δv̇ = −R [f×] δθ − R δb_a − R n_a.

Attitude error. From ``q_true = q ⊗ exp(δθ)`` and both quaternions obeying
``q̇ = ½ q ⊗ [0, ω]`` with ``ω_true = ω − δb_g − n_g``, the standard
local-error result (Solà 2017, "Quaternion kinematics for the error-state
Kalman filter", Eq. 237) is::

    δθ̇ = −[ω×] δθ − δb_g − n_g.

Hence, with noise vector ``w = [n_a, n_g, n_bg, n_ba]`` (12)::

    F[δp, δv]   = I            G[δv, n_a]   = −R
    F[δv, δθ]   = −R [f×]      G[δθ, n_g]   = −I
    F[δv, δb_a] = −R           G[δb_g, n_bg] = I
    F[δθ, δθ]   = −[ω×]        G[δb_a, n_ba] = I
    F[δθ, δb_g] = −I

``Qc = diag(σ_a² I, σ_g² I, σ_bg² I, σ_ba² I)`` (spectral densities, A-03).
Because ``R Rᵀ = I`` and ``Qc`` is isotropic per sensor, ``G Qc Gᵀ`` is diagonal.

Discretisation (per IMU interval ``[t_k, t_{k+1}]``, ``dt = t_{k+1} − t_k``).
The nominal state is integrated exactly like ``dead_reckoning.strapdown``:
trapezoid ``ω = ½(ω_m,k + ω_m,k+1) − b_g``, ``f = ½(f_m,k + f_m,k+1) − b_a``,
``q_{k+1} = q_k ⊗ exp(ω dt)``, ``q_mid = q_k ⊗ exp(ω dt/2)``,
``v_{k+1} = v_k + (R(q_mid) f + g_n) dt``, ``p_{k+1} = p_k + ½(v_k + v_{k+1}) dt``.
``F`` is evaluated with ``R(q_mid)``, ``ω``, ``f``. Default first-order
covariance discretisation ``Φ = I + F dt + ½ (F dt)²``, ``Qd = G Qc Gᵀ dt``
(``"van_loan"`` available for checks; ‖F‖dt ≈ 0.04 at 220 Hz from either 7 rad/s
or 1 g, ≈ 0.1 in high-g manoeuvres; agrees with Van Loan to ~1e-5 relative over 10 s).
PX4 IMU samples are means over the interval *ending* at their stamp (CONVENTIONS
§6), so the trapezoid carries a dt/2 (≈ 2.3 ms) lag; on the aggressive log it
changes tilt RMS vs EKF2 by 0.01° compared with the interval-end scheme (★ review).

Measurements (``r = z − h(x)``)
-------------------------------
* ``gps_pos``: ``h = p``, ``H = [I 0 0 0 0]``, ``R = diag(σ_h², σ_h², σ_v²)``,
  ``σ_h = max(eph, floor_h)``, ``σ_v = max(epv, floor_v)``. No lever arm
  (``EKF2_GPS_POS_* = EKF2_IMU_POS_* = 0`` in all pinned logs).
* ``gps_vel``: ``h = v``, ``H = [0 I 0 0 0]``, ``R = max(s_acc, floor)² I``.
* ``baro``: ``h = p_D``, ``H = e_3ᵀ`` on δp; ``z`` is the baro height change
  turned into a down position (A-10).
* ``mag_heading``: tilt-compensated heading with the **estimated** roll φ and
  pitch θ: ``m_h = Ry(θ) Rx(φ) m_b``, ``z = decl − atan2(m_h,y, m_h,x)``
  (since ``Ry Rx m_b = Rz(ψ)ᵀ m_n`` and ``m_n ∝ [cos decl, sin decl, ·]``);
  ``h = ψ(q) = atan2(R10, R00)``; residual wrapped to (−π, π].

  Because the tilt compensation uses the *estimated* tilt ``T̂ = Ry(θ̂) Rx(φ̂)``,
  a tilt error leaks into ``z`` (amplified by tan(inclination); ★ review), so the
  residual is linearised as a whole, not just ``ψ``. With
  ``m_b = R_trueᵀ m_n = (I − [δθ×]) R̂ᵀ m_n`` and ``T̂ R̂ᵀ = Rz(ψ̂)ᵀ``::

      m_h = Rz(ψ̂)ᵀ m_n + T̂ [m_b×] δθ  ⇒  r = z − ψ̂ ≈ h_m δθ,
      h_m = −(m_h,x e_yᵀ − m_h,y e_xᵀ) T̂ [m_b×] / (m_h,x² + m_h,y²)

  evaluated with the *measured* ``m_b`` (equal to ``R̂ᵀ m_n`` to first order).
  The residual is strongly curved at high inclination (the horizontal field is
  only cos I of |B|): the Jacobian changes by ~15 per rad of attitude error at
  I = 67° (verified by finite differences), so heading updates are less linear
  than the tan I factor alone suggests. For a level vehicle and ``δθ = ε e_z``
  this gives ``r = ε``. The pure yaw Jacobian ``yaw_jacobian`` (``∂ψ/∂δθ =
  [0, R10 R02 − R00 R12, R00 R11 − R10 R01] / (R00² + R10²)``) is kept for
  tests/diagnostics. Skipped when |pitch| > 60°.

After every accepted update the correction is injected (``p += δp``, …,
``q ← inject_attitude_error(q, δθ)``) and the covariance reset
``P ← G_r P G_rᵀ`` is applied, ``G_r = I`` except the attitude block
``I − ½[δθ×]`` (``common.quaternion.attitude_reset_jacobian``). Measurements
are applied at the IMU sample nearest their (GPS: delay-corrected) time stamp,
in the order gps_pos, gps_vel, baro, mag_heading.

Increments: 1 = GPS + baro (biases frozen), 2 = + mag heading, 3 = + bias
estimation. Frozen states have ``P0 = FROZEN_VAR``, no process noise and their
``dx`` is zeroed before injection, so they keep their initial values exactly.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from common import quaternion as quat
from common.ekf import ErrorStateEKF, InnovationLog, UpdateResult, chi2_gate, symmetrize, van_loan
from common.frames import lla_to_ned, normal_gravity
from sensor_fusion.dead_reckoning import initial_attitude_from_accel_mag

__all__ = [
    "STATE_NAMES", "IDX_P", "IDX_V", "IDX_TH", "IDX_BG", "IDX_BA", "FROZEN_VAR",
    "EkfConfig", "SensorStreams", "InitialState", "EkfResult", "DroneEKF",
    "error_state_jacobians", "process_noise_psd", "mag_heading_measurement", "mag_heading_jacobian",
    "yaw_jacobian", "igrf_declination",
    "propagate_nominal", "static_initial_state", "run_filter", "streams_from_log",
    "ekf2_reference", "wrap_angle",
]

STATE_NAMES = ("p_n", "p_e", "p_d", "v_n", "v_e", "v_d", "th_x", "th_y", "th_z",
               "bg_x", "bg_y", "bg_z", "ba_x", "ba_y", "ba_z")
IDX_P, IDX_V, IDX_TH, IDX_BG, IDX_BA = (slice(0, 3), slice(3, 6), slice(6, 9),
                                        slice(9, 12), slice(12, 15))
FROZEN_VAR = 1e-18
_N = 15
_I3 = np.eye(3)
_E3 = np.array([0.0, 0.0, 1.0])
_MIN_COS2_PITCH = math.cos(math.radians(60.0)) ** 2


# ---------------------------------------------------------------------------
# Configuration and containers
# ---------------------------------------------------------------------------

@dataclass
class EkfConfig:
    """Filter settings. Defaults are starting values from PX4 EKF2 parameters (A-03, A-04)."""
    increment: int = 3
    # process noise densities (A-03)
    accel_noise: float = 0.035        # m/s^2/sqrt(Hz)
    gyro_noise: float = 1.5e-3        # rad/s/sqrt(Hz)
    gyro_bias_rw: float = 1.0e-4      # rad/s^2/sqrt(Hz)
    accel_bias_rw: float = 3.0e-4     # m/s^3/sqrt(Hz)
    # initial 1-sigma (A-03)
    sigma_p0: float = 1.0             # m
    sigma_v0: float = 0.5             # m/s
    sigma_att0: float = 0.1           # rad
    sigma_bg0: float = 0.1            # rad/s
    sigma_ba0: float = 0.2            # m/s^2
    # measurements (A-04, A-09, A-10, A-11)
    gps_delay_s: float = 0.11
    gps_pos_floor_h: float = 0.5      # m
    gps_pos_floor_v: float = 0.5      # m
    gps_vel_floor: float = 0.3        # m/s
    baro_sigma: float = 1.5           # m
    mag_heading_sigma: float = 0.07   # rad (4 deg)
    gate_prob: float | None = None
    init_window_s: float = 1.0
    discretization: str = "first_order"
    cov_every: int = 0
    use_mag: bool | None = None
    estimate_biases: bool | None = None

    def __post_init__(self):
        if self.increment not in (1, 2, 3):
            raise ValueError("increment must be 1, 2 or 3")
        if self.discretization not in ("first_order", "van_loan"):
            raise ValueError("discretization must be 'first_order' or 'van_loan'")

    @property
    def mag_enabled(self) -> bool:
        return self.increment >= 2 if self.use_mag is None else bool(self.use_mag)

    @property
    def biases_enabled(self) -> bool:
        return self.increment >= 3 if self.estimate_biases is None else bool(self.estimate_biases)


@dataclass
class SensorStreams:
    """Filter input (real logs via ``streams_from_log``, synthetic via ``sensor_fusion.synthetic``)."""
    imu_t: np.ndarray
    gyro: np.ndarray
    accel: np.ndarray
    gps_t: np.ndarray                 # receiver stamps (not delay-corrected)
    gps_pos_ned: np.ndarray
    gps_vel_ned: np.ndarray
    gps_pos_sigma: np.ndarray         # (M,3) [σ_n, σ_e, σ_d]
    gps_vel_sigma: np.ndarray         # (M,)
    baro_t: np.ndarray
    baro_d: np.ndarray
    mag_t: np.ndarray
    mag_field: np.ndarray             # (L,3) tesla, FRD
    g: float
    declination: float
    origin_lla: tuple = (0.0, 0.0, 0.0)
    meta: dict = field(default_factory=dict)

    def sliced(self, t_end: float) -> "SensorStreams":
        """Copy restricted to ``t <= t_end`` on every stream (GPS by receiver stamp)."""
        def m(t):
            return np.asarray(t) <= t_end
        mi, mg, mb, mm = m(self.imu_t), m(self.gps_t), m(self.baro_t), m(self.mag_t)
        return SensorStreams(self.imu_t[mi], self.gyro[mi], self.accel[mi], self.gps_t[mg],
                             self.gps_pos_ned[mg], self.gps_vel_ned[mg], self.gps_pos_sigma[mg],
                             self.gps_vel_sigma[mg], self.baro_t[mb], self.baro_d[mb],
                             self.mag_t[mm], self.mag_field[mm], self.g, self.declination,
                             self.origin_lla, dict(self.meta))


@dataclass
class InitialState:
    p: np.ndarray
    v: np.ndarray
    q: np.ndarray
    b_g: np.ndarray = field(default_factory=lambda: np.zeros(3))
    b_a: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class EkfResult:
    t: np.ndarray
    p: np.ndarray
    v: np.ndarray
    q: np.ndarray
    b_g: np.ndarray
    b_a: np.ndarray
    std: np.ndarray
    P_hist_t: np.ndarray
    P_hist: np.ndarray
    P_final: np.ndarray
    innovations: dict
    config: EkfConfig
    meta: dict


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------

def wrap_angle(a):
    """Wrap angle(s) to (−π, π]."""
    a = np.asarray(a, dtype=float)
    w = np.mod(a + np.pi, 2.0 * np.pi) - np.pi
    w = np.where(w == -np.pi, np.pi, w)
    return float(w) if w.ndim == 0 else w


def error_state_jacobians(q, f_body, omega_body) -> tuple[np.ndarray, np.ndarray]:
    """Continuous-time error-state ``F`` (15×15) and noise ``G`` (15×12), see module docstring.

    ``f_body`` / ``omega_body`` are the bias-corrected specific force and body rate.
    """
    R = quat.to_dcm(q)
    F = np.zeros((_N, _N))
    F[IDX_P, IDX_V] = _I3
    F[IDX_V, IDX_TH] = -R @ quat.skew(f_body)
    F[IDX_V, IDX_BA] = -R
    F[IDX_TH, IDX_TH] = -quat.skew(omega_body)
    F[IDX_TH, IDX_BG] = -_I3
    G = np.zeros((_N, 12))
    G[IDX_V, 0:3] = -R
    G[IDX_TH, 3:6] = -_I3
    G[IDX_BG, 6:9] = _I3
    G[IDX_BA, 9:12] = _I3
    return F, G


def process_noise_psd(cfg: EkfConfig, active_mask=None) -> np.ndarray:
    """``Qc`` (12×12) for ``w = [n_a, n_g, n_bg, n_ba]``; bias channels zero if frozen.

    ``active_mask`` (15,) bool: the bias random walks are zeroed where the
    corresponding bias states are inactive.
    """
    qbg, qba = cfg.gyro_bias_rw ** 2, cfg.accel_bias_rw ** 2
    if active_mask is not None:
        active_mask = np.asarray(active_mask, dtype=bool)
        if not active_mask[IDX_BG].all():
            qbg = 0.0
        if not active_mask[IDX_BA].all():
            qba = 0.0
    return np.diag(np.repeat([cfg.accel_noise ** 2, cfg.gyro_noise ** 2, qbg, qba], 3))


def yaw_jacobian(q) -> np.ndarray:
    """``∂ψ/∂δθ`` (3,) for ``q_true = q ⊗ exp(δθ)``, ``ψ = atan2(R10, R00)``."""
    R = quat.to_dcm(q)
    d = R[0, 0] ** 2 + R[1, 0] ** 2
    return np.array([0.0, R[1, 0] * R[0, 2] - R[0, 0] * R[1, 2],
                     R[0, 0] * R[1, 1] - R[1, 0] * R[0, 1]]) / d


def _yaw(q) -> float:
    R = quat.to_dcm(q)
    return math.atan2(R[1, 0], R[0, 0])


def _tilt_matrix(q) -> np.ndarray:
    """``T = Ry(pitch) Rx(roll)`` of ``q`` (body → yaw-only heading frame)."""
    roll, pitch, _ = quat.to_euler_zyx(q)
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    return np.array([[cp, sp * sr, sp * cr],
                     [0.0, cr, -sr],
                     [-sp, cp * sr, cp * cr]])


def mag_heading_measurement(q, mag_body, declination) -> float:
    """Tilt-compensated true heading (rad, wrapped) from a body-frame field using the tilt of ``q``."""
    hx, hy, _ = _tilt_matrix(q) @ np.asarray(mag_body, float)
    return wrap_angle(declination - math.atan2(hy, hx))


def mag_heading_jacobian(q, mag_body) -> np.ndarray:
    """``∂(z − ψ̂)/∂δθ`` (3,) of the tilt-compensated heading residual (module docstring).

    ``mag_body`` is the measured field (only its direction matters).
    """
    m = np.asarray(mag_body, float)
    T = _tilt_matrix(q)
    hx, hy, _ = T @ m
    return -(hx * np.array([0.0, 1.0, 0.0]) - hy * np.array([1.0, 0.0, 0.0])) @ T @ quat.skew(m) \
        / (hx * hx + hy * hy)


def propagate_nominal(p, v, q, b_g, b_a, gyro0, gyro1, acc0, acc1, dt, g):
    """One strapdown step (trapezoid inputs, mid-interval attitude), see module docstring.

    Returns ``(p1, v1, q1, q_mid, omega, f)`` where ``omega``/``f`` are the
    bias-corrected interval-mean rate and specific force.
    """
    omega = 0.5 * (np.asarray(gyro0, float) + np.asarray(gyro1, float)) - b_g
    f = 0.5 * (np.asarray(acc0, float) + np.asarray(acc1, float)) - b_a
    phi = omega * dt
    q1 = quat.normalize(quat.multiply(q, quat.from_rotvec(phi)))
    q_mid = quat.normalize(quat.multiply(q, quat.from_rotvec(0.5 * phi)))
    a = quat.rotate(q_mid, f)
    a[2] += g
    v1 = v + a * dt
    p1 = p + 0.5 * (v + v1) * dt
    return p1, v1, q1, q_mid, omega, f


def static_initial_state(streams: SensorStreams, window_s: float = 1.0) -> InitialState:
    """Static initialisation over ``[t0, t0 + window_s]``.

    Roll/pitch from the mean specific force, heading from the tilt-compensated
    mean magnetic field plus the declination (rotation about NED down), position
    from the first GPS sample, velocity zero, biases zero.
    """
    t0 = streams.imu_t[0]
    mi = streams.imu_t <= t0 + window_s
    mm = streams.mag_t <= t0 + window_s
    if not mm.any():
        mm = np.zeros(streams.mag_t.size, bool)
        mm[0] = True
    q_mag = initial_attitude_from_accel_mag(streams.accel[mi], streams.mag_field[mm])
    q0 = quat.canonicalize(quat.multiply(quat.from_axis_angle([0, 0, 1], streams.declination), q_mag))
    p0 = np.asarray(streams.gps_pos_ned[0], float).copy() if streams.gps_t.size else np.zeros(3)
    return InitialState(p=p0, v=np.zeros(3), q=q0)


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

class DroneEKF:
    """15-state ES-EKF: nominal state + ``common.ekf.ErrorStateEKF`` for the covariance."""

    def __init__(self, cfg: EkfConfig, init: InitialState, g: float):
        self.cfg = cfg
        self.g = float(g)
        self.p = np.asarray(init.p, float).copy()
        self.v = np.asarray(init.v, float).copy()
        self.q = quat.normalize(np.asarray(init.q, float))
        self.b_g = np.asarray(init.b_g, float).copy()
        self.b_a = np.asarray(init.b_a, float).copy()
        self.active = np.ones(_N, bool)
        if not cfg.biases_enabled:
            self.active[IDX_BG] = False
            self.active[IDX_BA] = False
        sig = np.repeat([cfg.sigma_p0, cfg.sigma_v0, cfg.sigma_att0, cfg.sigma_bg0, cfg.sigma_ba0], 3)
        var = sig ** 2
        var[~self.active] = FROZEN_VAR
        self.ekf = ErrorStateEKF(np.diag(var), state_names=STATE_NAMES)
        self._Qc = process_noise_psd(cfg, self.active)
        self._gate = {}
        if cfg.gate_prob is not None:
            self._gate = {m: chi2_gate(m, cfg.gate_prob) for m in (1, 3)}

    # -- prediction ---------------------------------------------------------

    def predict(self, gyro0, gyro1, acc0, acc1, dt):
        """Propagate nominal state and covariance over one IMU interval."""
        p1, v1, q1, q_mid, omega, f = propagate_nominal(
            self.p, self.v, self.q, self.b_g, self.b_a, gyro0, gyro1, acc0, acc1, dt, self.g)
        F, G = error_state_jacobians(q_mid, f, omega)
        if self.cfg.discretization == "van_loan":
            Phi, Qd = van_loan(F, self._Qc, dt, G)
        else:
            Fdt = F * dt
            Phi = np.eye(_N) + Fdt + 0.5 * (Fdt @ Fdt)
            Qd = (G @ self._Qc @ G.T) * dt
        self.ekf.predict(Phi, Qd)
        self.p, self.v, self.q = p1, v1, q1

    # -- correction ---------------------------------------------------------

    def _inject(self, dx):
        dx = np.where(self.active, dx, 0.0)
        self.p = self.p + dx[IDX_P]
        self.v = self.v + dx[IDX_V]
        self.q = quat.inject_attitude_error(self.q, dx[IDX_TH])
        self.b_g = self.b_g + dx[IDX_BG]
        self.b_a = self.b_a + dx[IDX_BA]
        Gr = np.eye(_N)
        Gr[IDX_TH, IDX_TH] = quat.attitude_reset_jacobian(dx[IDX_TH])
        self.ekf.reset(Gr)

    def _update(self, r, H, Rdiag, name, t):
        res = self.ekf.update(r, H, Rdiag, name=name, t=t, gate=self._gate.get(len(r)))
        if res.accepted:
            self._inject(res.dx)
        return res

    def update_gps_pos(self, z, sigma3, t=None) -> UpdateResult:
        H = np.zeros((3, _N)); H[:, IDX_P] = _I3
        return self._update(np.asarray(z, float) - self.p, H, np.asarray(sigma3, float) ** 2, "gps_pos", t)

    def update_gps_vel(self, z, sigma, t=None) -> UpdateResult:
        H = np.zeros((3, _N)); H[:, IDX_V] = _I3
        return self._update(np.asarray(z, float) - self.v, H, np.full(3, float(sigma) ** 2), "gps_vel", t)

    def update_baro(self, z_d, sigma, t=None) -> UpdateResult:
        H = np.zeros((1, _N)); H[0, 2] = 1.0
        return self._update(np.array([float(z_d) - self.p[2]]), H, np.array([float(sigma) ** 2]), "baro", t)

    def update_mag_heading(self, mag_body, declination, sigma, t=None) -> UpdateResult | None:
        """Heading update; returns None (no update) if |pitch| > 60°."""
        R = quat.to_dcm(self.q)
        if R[0, 0] ** 2 + R[1, 0] ** 2 < _MIN_COS2_PITCH:
            return None
        z = mag_heading_measurement(self.q, mag_body, declination)
        r = wrap_angle(z - math.atan2(R[1, 0], R[0, 0]))
        H = np.zeros((1, _N)); H[0, IDX_TH] = mag_heading_jacobian(self.q, mag_body)
        return self._update(np.array([r]), H, np.array([float(sigma) ** 2]), "mag_heading", t)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def _nearest_index(t_imu: np.ndarray, t_meas: np.ndarray) -> np.ndarray:
    """Nearest IMU sample index for each measurement time; −1 outside the IMU span."""
    t_meas = np.asarray(t_meas, float)
    k = np.searchsorted(t_imu, t_meas)
    k = np.clip(k, 1, t_imu.size - 1)
    left_closer = (t_meas - t_imu[k - 1]) <= (t_imu[k] - t_meas)
    k = np.where(left_closer, k - 1, k)
    out = (t_meas < t_imu[0]) | (t_meas > t_imu[-1])
    return np.where(out, -1, k)


def run_filter(streams: SensorStreams, cfg: EkfConfig | None = None,
               init: InitialState | None = None) -> EkfResult:
    """Run the ES-EKF over a whole stream set (offline, time ordered). See module docstring."""
    cfg = EkfConfig() if cfg is None else cfg
    t_wall = time.perf_counter()
    static_init = init is None
    if static_init:
        init = static_initial_state(streams, cfg.init_window_s)
    filt = DroneEKF(cfg, init, streams.g)

    t = np.asarray(streams.imu_t, float)
    gyro = np.asarray(streams.gyro, float)
    acc = np.asarray(streams.accel, float)
    n = t.size
    if n < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("imu_t must have >= 2 strictly increasing samples")

    # events: index -> list of (order, payload)
    events: dict[int, list] = {}

    def add(kind_order, idx, payloads):
        for k, pl in zip(idx, payloads):
            if k >= 0:
                events.setdefault(int(k), []).append((kind_order, pl))

    t_gps = np.asarray(streams.gps_t, float) - cfg.gps_delay_s
    k_gps = _nearest_index(t, t_gps) if t_gps.size else np.empty(0, int)
    floors = np.array([cfg.gps_pos_floor_h, cfg.gps_pos_floor_h, cfg.gps_pos_floor_v])
    add(0, k_gps, [("gps_pos", t_gps[j], streams.gps_pos_ned[j],
                    np.maximum(streams.gps_pos_sigma[j], floors)) for j in range(t_gps.size)])
    add(1, k_gps, [("gps_vel", t_gps[j], streams.gps_vel_ned[j],
                    max(float(streams.gps_vel_sigma[j]), cfg.gps_vel_floor)) for j in range(t_gps.size)])
    if streams.baro_t.size:
        add(2, _nearest_index(t, streams.baro_t),
            [("baro", streams.baro_t[j], streams.baro_d[j], cfg.baro_sigma) for j in range(streams.baro_t.size)])
    if cfg.mag_enabled and streams.mag_t.size:
        add(3, _nearest_index(t, streams.mag_t),
            [("mag", streams.mag_t[j], streams.mag_field[j], cfg.mag_heading_sigma)
             for j in range(streams.mag_t.size)])

    P_out = np.empty((n, 3)); V_out = np.empty((n, 3)); Q_out = np.empty((n, 4))
    BG_out = np.empty((n, 3)); BA_out = np.empty((n, 3)); STD_out = np.empty((n, _N))
    P_hist_t, P_hist = [], []
    mag_skipped = 0
    n_updates = {"gps_pos": 0, "gps_vel": 0, "baro": 0, "mag_heading": 0}

    def apply_events(k):
        nonlocal mag_skipped
        for _, (kind, tm, z, s) in sorted(events.get(k, ()), key=lambda e: e[0]):
            if kind == "gps_pos":
                res = filt.update_gps_pos(z, s, t=float(tm))
            elif kind == "gps_vel":
                res = filt.update_gps_vel(z, s, t=float(tm))
            elif kind == "baro":
                res = filt.update_baro(z, s, t=float(tm))
            else:
                kind = "mag_heading"
                res = filt.update_mag_heading(z, streams.declination, s, t=float(tm))
                if res is None:
                    mag_skipped += 1
                    continue
            if res.accepted:
                n_updates[kind] += 1

    def record(k):
        P_out[k], V_out[k], Q_out[k] = filt.p, filt.v, filt.q
        BG_out[k], BA_out[k] = filt.b_g, filt.b_a
        STD_out[k] = filt.ekf.std
        if cfg.cov_every and k % cfg.cov_every == 0:
            P_hist_t.append(t[k]); P_hist.append(filt.ekf.P)

    apply_events(0)
    record(0)
    for k in range(n - 1):
        filt.predict(gyro[k], gyro[k + 1], acc[k], acc[k + 1], t[k + 1] - t[k])
        apply_events(k + 1)
        record(k + 1)

    innov = {name: filt.ekf.innovations(name) for name in ("gps_pos", "gps_vel", "baro", "mag_heading")
             if name in filt.ekf.history}
    meta = {
        "init": {"p": np.asarray(init.p).tolist(), "v": np.asarray(init.v).tolist(),
                 "q": np.asarray(init.q).tolist(), "b_g": np.asarray(init.b_g).tolist(),
                 "b_a": np.asarray(init.b_a).tolist(), "static": static_init},
        "g": float(streams.g), "declination": float(streams.declination),
        "origin_lla": list(streams.origin_lla), "n_imu": int(n),
        "n_updates": n_updates, "mag_skipped_tilt": mag_skipped,
        "active_states": [s for s, a in zip(STATE_NAMES, filt.active) if a],
        "runtime_s": time.perf_counter() - t_wall, "config": asdict(cfg),
    }
    return EkfResult(t=t.copy(), p=P_out, v=V_out, q=Q_out, b_g=BG_out, b_a=BA_out, std=STD_out,
                     P_hist_t=np.asarray(P_hist_t, float),
                     P_hist=np.asarray(P_hist, float).reshape(-1, _N, _N),
                     P_final=filt.ekf.P, innovations=innov, config=cfg, meta=meta)


# ---------------------------------------------------------------------------
# Real-log adapters
# ---------------------------------------------------------------------------

def igrf_declination(lat: float, lon: float, alt: float, when) -> float:
    """Magnetic declination [rad, east positive] from IGRF (``ppigrf``) at geodetic lat/lon [rad], alt [m]."""
    import ppigrf
    if getattr(when, "tzinfo", None) is not None:
        when = (when - when.utcoffset()).replace(tzinfo=None)   # aware -> naive UTC
    be, bn, _ = ppigrf.igrf(math.degrees(lon), math.degrees(lat), alt / 1000.0, when)
    return math.atan2(float(np.ravel(be)[0]), float(np.ravel(bn)[0]))


def streams_from_log(log) -> SensorStreams:
    """Build ``SensorStreams`` from a ``loader.FlightLog`` (spec §5)."""
    gps, baro, mag, imu = log.gps, log.baro, log.mag, log.imu
    lat0, lon0, alt0 = float(gps.lat[0]), float(gps.lon[0]), float(gps.alt[0])
    pos = lla_to_ned(gps.lat, gps.lon, gps.alt, lat0, lon0, alt0)
    g = float(normal_gravity(lat0, alt0))
    when = log.info.get("gps_utc_start")
    if when is None:
        raise ValueError("log has no GPS UTC time; cannot date the IGRF model")
    decl = igrf_declination(lat0, lon0, alt0, when)
    m_ref = baro.t <= baro.t[0] + 1.0
    alt_ref = float(np.mean(baro.alt_baro[m_ref]))
    d0 = float(pos[0, 2])
    baro_d = d0 - (baro.alt_baro - alt_ref)
    return SensorStreams(
        imu_t=imu.t, gyro=imu.gyro, accel=imu.accel,
        gps_t=gps.t, gps_pos_ned=pos, gps_vel_ned=gps.vel_ned,
        gps_pos_sigma=np.column_stack([gps.eph, gps.eph, gps.epv]),
        gps_vel_sigma=np.asarray(gps.s_variance_m_s, float),
        baro_t=baro.t, baro_d=baro_d, mag_t=mag.t, mag_field=mag.field,
        g=g, declination=decl, origin_lla=(lat0, lon0, alt0),
        meta={"log_id": log.log_id, "alt_msl0": float(gps.alt_msl[0]), "baro_alt_ref": alt_ref,
              "gps_utc_start": str(when)})


def ekf2_reference(log, origin_lla, alt_msl0: float | None = None) -> dict:
    """EKF2 outputs translated into the filter's NED frame (CONVENTIONS §6). Reference, not truth.

    ``alt_msl0`` is the MSL height of the origin; by default the origin must be
    the log's first GPS fix (as in ``streams_from_log``) and its ``alt_msl`` is used.
    """
    ek, gps = log.ekf2, log.gps
    lat0, lon0, alt0 = origin_lla
    if alt_msl0 is None:
        if not np.allclose([lat0, lon0, alt0], [gps.lat[0], gps.lon[0], gps.alt[0]], rtol=0, atol=1e-9):
            raise ValueError("origin_lla is not the first GPS fix; pass alt_msl0 explicitly")
        alt_msl0 = float(gps.alt_msl[0])
    if not np.isfinite(alt_msl0):
        raise ValueError("MSL height of the origin is not available")
    off = lla_to_ned(lat0, lon0, alt0, ek.ref_lat, ek.ref_lon, alt0)   # our origin in EKF2's frame (horizontal)
    pos = np.asarray(ek.pos_ned, float).copy()
    pos[:, 0] -= off[0]
    pos[:, 1] -= off[1]
    pos[:, 2] -= ek.ref_alt - alt_msl0
    return {"t_pos": ek.t_pos, "pos_ned": pos, "vel_ned": ek.vel_ned, "t_att": ek.t_att, "q": ek.q,
            "reset_times": list(log.info.get("ekf2_quat_reset_times_s", [])),
            "origin_offset_ne": off[:2].tolist(), "vertical_offset": float(ek.ref_alt - alt_msl0)}
