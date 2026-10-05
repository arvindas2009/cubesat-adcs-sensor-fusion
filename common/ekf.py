"""Generic error-state extended Kalman filter (ES-EKF) core.

The filter owns only the **error-state covariance** ``P`` (n x n) and an
innovation log. The nominal state (positions, velocities, quaternions, biases)
is owned by the caller, because only the caller knows how to propagate it and
how to inject a correction into it (e.g. ``q <- q ⊗ exp(dθ)``). One cycle::

    # --- prediction (every IMU / gyro sample) ---
    x_nom = propagate_nominal(x_nom, u, dt)        # caller, nonlinear
    Phi, Qd = van_loan(F, Qc, dt, G)               # or caller's own discretisation
    ekf.predict(Phi, Qd)                           # P <- Phi P Phi^T + Qd

    # --- correction (whenever a measurement arrives) ---
    r = z - h(x_nom)                               # residual, caller
    H = dh/d(dx) evaluated at x_nom                # error-state Jacobian, caller
    res = ekf.update(r, H, R, name="gps", t=t)     # Joseph-form update
    x_nom = inject(x_nom, res.dx)                  # caller; error state is now 0
    ekf.reset(G_reset)                             # optional: P <- G P G^T

**Contract:** the filter assumes the error-state mean is zero before every
predict and update. The caller must inject ``dx`` (and optionally reset) after
*each* accepted update, before the next update -- including several sensors
at the same timestamp (e.g. GPS, then baro, then mag), each of which must see
the nominal state already corrected by the previous one. For attitude use
``quaternion.inject_attitude_error`` and ``quaternion.attitude_reset_jacobian``.

Conventions (see CONVENTIONS.md §3):

- Residual ``r = z - h(x_nominal)``; correction ``dx = K r``.
- Covariance update in **Joseph form** ``P = (I-KH) P (I-KH)^T + K R K^T``,
  followed by explicit symmetrisation. ``P`` is symmetrised after every
  predict/update/reset.
- ``S = H P H^T + R``; gains and NIS use a Cholesky solve of ``S``
  (never an explicit inverse).
- **NIS** ``r^T S^-1 r`` (a scalar, ~ chi-square with ``m = len(r)`` degrees of
  freedom for a consistent filter) is recorded for every update, accepted or
  not.
- **Gating** (optional): if ``gate`` is given and ``NIS > gate`` the update is
  rejected: ``P`` is unchanged and ``dx`` is zero. Use :func:`chi2_gate`
  to turn a probability into a threshold.

Public API
----------
``ErrorStateEKF(P0, state_names=None, record=True)``
    ``.n``, ``.P`` (copy), ``.std`` (sqrt of diagonal), ``.state_names``
    ``.predict(Phi, Qd)``
    ``.update(residual, H, R, name="meas", t=None, gate=None) -> UpdateResult``
    ``.update_sequential(residual, H, R_diag, name="meas", t=None, gate=None) -> UpdateResult``
    ``.reset(G)``
    ``.history`` (dict name -> list[UpdateResult]), ``.innovations(name) -> InnovationLog``
    ``.set_P(P)``

``UpdateResult`` (frozen dataclass): ``dx, residual, S, nis, accepted, name, t``

Module functions: ``symmetrize(P)``, ``is_spd(P)``, ``van_loan(F, Qc, dt, G=None)``,
``chi2_gate(dof, prob=0.9973)``, ``nis_bounds(dof, n, prob=0.95)``,
``nis_fraction_inside(nis, dof, prob=0.95)``.

``R`` may be given as an (m, m) matrix, an (m,) vector of variances
(diagonal R), or a scalar variance (``R = r * I``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve, expm
from scipy.stats import chi2

__all__ = [
    "ErrorStateEKF",
    "UpdateResult",
    "InnovationLog",
    "symmetrize",
    "is_spd",
    "van_loan",
    "chi2_gate",
    "nis_bounds",
    "nis_fraction_inside",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def symmetrize(P: np.ndarray) -> np.ndarray:
    """Return ``(P + P^T) / 2``."""
    P = np.asarray(P, dtype=float)
    return 0.5 * (P + P.T)


def is_spd(P: np.ndarray, tol: float = 1e-9) -> bool:
    """True if ``P`` is symmetric (relative tolerance ``tol``) and Cholesky-factorisable."""
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1] or not np.all(np.isfinite(P)):
        return False
    scale = max(np.max(np.abs(P)), 1e-300)
    if np.max(np.abs(P - P.T)) > tol * scale:
        return False
    try:
        np.linalg.cholesky(P)
    except np.linalg.LinAlgError:
        return False
    return True


def _as_noise_matrix(R, m: int) -> np.ndarray:
    """Convert scalar / (m,) / (m, m) measurement noise into an (m, m) matrix."""
    R = np.asarray(R, dtype=float)
    if R.ndim == 0:
        return float(R) * np.eye(m)
    if R.ndim == 1:
        if R.shape != (m,):
            raise ValueError(f"R vector has shape {R.shape}, expected ({m},)")
        return np.diag(R)
    if R.shape != (m, m):
        raise ValueError(f"R has shape {R.shape}, expected ({m}, {m})")
    return symmetrize(R)


def van_loan(F: np.ndarray, Qc: np.ndarray, dt: float,
             G: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Discretise a continuous linear error model with the Van Loan method.

    Continuous model: ``d(dx)/dt = F dx + G w``, ``E[w w^T] = Qc δ(t-τ)``.
    Returns ``(Phi, Qd)`` with ``Phi = expm(F dt)`` and
    ``Qd = ∫_0^dt expm(F s) G Qc G^T expm(F s)^T ds`` (symmetrised).
    If ``G`` is None, ``Qc`` is already the (n, n) noise spectral density.
    """
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    Qc = np.asarray(Qc, dtype=float)
    if Qc.ndim == 1:
        Qc = np.diag(Qc)
    GQGt = Qc if G is None else np.asarray(G, dtype=float) @ Qc @ np.asarray(G, dtype=float).T
    if GQGt.shape != (n, n):
        raise ValueError(f"G Qc G^T has shape {GQGt.shape}, expected ({n}, {n})")
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -F
    M[:n, n:] = GQGt
    M[n:, n:] = F.T
    E = expm(M * dt)
    Phi = E[n:, n:].T
    Qd = Phi @ E[:n, n:]
    return Phi, symmetrize(Qd)


def chi2_gate(dof: int, prob: float = 0.9973) -> float:
    """NIS threshold below which a consistent innovation falls with probability ``prob``."""
    return float(chi2.ppf(prob, dof))


def nis_bounds(dof: int, n: int, prob: float = 0.95) -> tuple[float, float]:
    """Two-sided ``prob`` bounds for the **time-averaged** NIS of ``n`` samples.

    For a consistent filter ``n * mean(NIS) ~ chi2(n * dof)`` (Bar-Shalom,
    Li & Kirubarajan 2001, §5.4), so the average NIS lies in the returned
    interval (centred near ``dof``) with probability ``prob``.
    """
    lo, hi = chi2.ppf([(1 - prob) / 2, (1 + prob) / 2], n * dof)
    return float(lo) / n, float(hi) / n


def nis_fraction_inside(nis, dof: int, prob: float = 0.95) -> float:
    """Fraction of individual NIS samples inside the two-sided ``prob`` chi2(dof) interval.

    Should be close to ``prob`` for a consistent filter.
    """
    nis = np.asarray(nis, dtype=float)
    if nis.size == 0:
        return float("nan")
    lo, hi = chi2.ppf([(1 - prob) / 2, (1 + prob) / 2], dof)
    return float(np.mean((nis >= lo) & (nis <= hi)))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UpdateResult:
    """Outcome of one measurement update.

    ``dx`` is the error-state correction to inject (zeros if rejected),
    ``residual`` the innovation ``z - h(x_nom)``, ``S`` the prior innovation
    covariance, ``nis`` = ``r^T S^-1 r``.
    """
    dx: np.ndarray
    residual: np.ndarray
    S: np.ndarray
    nis: float
    accepted: bool
    name: str
    t: float | None


class InnovationLog(NamedTuple):
    """Stacked innovation history for one measurement name.

    Includes rejected (gated) updates; filter on ``accepted`` for consistency
    statistics of the applied updates. If the measurement dimension varied
    under one name, ``residual``/``sigma`` have ``m = max`` columns, NaN-padded,
    and ``dof`` gives each row's dimension.
    """
    t: np.ndarray          # (N,)  NaN where no time was given
    residual: np.ndarray   # (N, m)
    sigma: np.ndarray      # (N, m) sqrt(diag(S))
    nis: np.ndarray        # (N,)
    accepted: np.ndarray   # (N,) bool
    dof: np.ndarray        # (N,) int, measurement dimension of each row


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

class ErrorStateEKF:
    """Error-state covariance propagation and correction (see module docstring)."""

    def __init__(self, P0, state_names: Sequence[str] | None = None, record: bool = True):
        P0 = np.asarray(P0, dtype=float)
        if P0.ndim != 2 or P0.shape[0] != P0.shape[1]:
            raise ValueError(f"P0 must be square, got shape {P0.shape}")
        if not is_spd(P0):
            raise ValueError("P0 must be symmetric positive definite")
        self.n = P0.shape[0]
        self._P = symmetrize(P0.copy())
        if state_names is not None and len(state_names) != self.n:
            raise ValueError("state_names length does not match P0")
        self.state_names = list(state_names) if state_names is not None else None
        self.record = record
        self.history: dict[str, list[UpdateResult]] = {}

    # -- state access ------------------------------------------------------

    @property
    def P(self) -> np.ndarray:
        """Copy of the error-state covariance."""
        return self._P.copy()

    @property
    def std(self) -> np.ndarray:
        """1-sigma error-state standard deviations ``sqrt(diag(P))``."""
        return np.sqrt(np.clip(np.diag(self._P), 0.0, None))

    def set_P(self, P) -> None:
        """Overwrite the covariance (e.g. re-initialisation). Must be SPD."""
        P = np.asarray(P, dtype=float)
        if P.shape != (self.n, self.n) or not is_spd(P):
            raise ValueError("P must be an SPD matrix of the filter's dimension")
        self._P = symmetrize(P.copy())

    # -- prediction --------------------------------------------------------

    def predict(self, Phi, Qd) -> None:
        """``P <- Phi P Phi^T + Qd`` (``Qd`` may be (n,) diagonal)."""
        Phi = np.asarray(Phi, dtype=float)
        Qd = np.asarray(Qd, dtype=float)
        if Qd.ndim == 1:
            Qd = np.diag(Qd)
        if Phi.shape != (self.n, self.n) or Qd.shape != (self.n, self.n):
            raise ValueError("Phi and Qd must be (n, n)")
        self._P = symmetrize(Phi @ self._P @ Phi.T + Qd)

    # -- correction --------------------------------------------------------

    def _check_meas(self, residual, H):
        r = np.atleast_1d(np.asarray(residual, dtype=float))
        H = np.atleast_2d(np.asarray(H, dtype=float))
        if r.ndim != 1:
            raise ValueError("residual must be 1-D")
        if H.shape != (r.size, self.n):
            raise ValueError(f"H has shape {H.shape}, expected ({r.size}, {self.n})")
        return r, H

    def _log(self, res: UpdateResult) -> UpdateResult:
        if self.record:
            self.history.setdefault(res.name, []).append(res)
        return res

    def update(self, residual, H, R, name: str = "meas", t: float | None = None,
               gate: float | None = None) -> UpdateResult:
        """Block measurement update with Joseph-form covariance.

        Returns an :class:`UpdateResult`; the caller injects ``result.dx``
        into the nominal state.
        """
        r, H = self._check_meas(residual, H)
        m = r.size
        Rm = _as_noise_matrix(R, m)
        P = self._P
        PHt = P @ H.T
        S = symmetrize(H @ PHt + Rm)
        cS = cho_factor(S)
        nis = float(r @ cho_solve(cS, r))
        if gate is not None and nis > gate:
            return self._log(UpdateResult(np.zeros(self.n), r, S, nis, False, name, t))
        K = cho_solve(cS, PHt.T).T                    # K = P H^T S^-1
        dx = K @ r
        A = np.eye(self.n) - K @ H
        self._P = symmetrize(A @ P @ A.T + K @ Rm @ K.T)
        return self._log(UpdateResult(dx, r, S, nis, True, name, t))

    def update_sequential(self, residual, H, R_diag, name: str = "meas",
                          t: float | None = None, gate: float | None = None) -> UpdateResult:
        """Update with mutually uncorrelated measurement components, one scalar at a time.

        Mathematically equivalent to :meth:`update` with ``R = diag(R_diag)``
        (each later component's residual is corrected by the error already
        estimated from earlier components: ``r_i' = r_i - H_i dx``). Gating,
        if requested, is decided on the block NIS before any component is
        applied, so a measurement is accepted or rejected as a whole.
        """
        r, H = self._check_meas(residual, H)
        m = r.size
        R_diag = np.broadcast_to(np.asarray(R_diag, dtype=float), (m,))
        P = self._P
        S = symmetrize(H @ P @ H.T + np.diag(R_diag))
        nis = float(r @ cho_solve(cho_factor(S), r))
        if gate is not None and nis > gate:
            return self._log(UpdateResult(np.zeros(self.n), r, S, nis, False, name, t))
        dx = np.zeros(self.n)
        I = np.eye(self.n)
        for i in range(m):
            h = H[i]
            Ph = P @ h
            s = float(h @ Ph + R_diag[i])
            k = Ph / s
            dx = dx + k * (r[i] - h @ dx)
            A = I - np.outer(k, h)
            P = symmetrize(A @ P @ A.T + R_diag[i] * np.outer(k, k))
        self._P = P
        return self._log(UpdateResult(dx, r, S, nis, True, name, t))

    # -- reset -------------------------------------------------------------

    def reset(self, G) -> None:
        """Error-state reset after injection: ``P <- G P G^T``."""
        G = np.asarray(G, dtype=float)
        if G.shape != (self.n, self.n):
            raise ValueError("G must be (n, n)")
        self._P = symmetrize(G @ self._P @ G.T)

    # -- logs --------------------------------------------------------------

    def innovations(self, name: str) -> InnovationLog:
        """Stack the recorded updates for ``name`` into arrays."""
        items = self.history.get(name, [])
        if not items:
            return InnovationLog(np.empty(0), np.empty((0, 0)), np.empty((0, 0)),
                                 np.empty(0), np.empty(0, dtype=bool),
                                 np.empty(0, dtype=int))
        t = np.array([np.nan if it.t is None else it.t for it in items], dtype=float)
        dof = np.array([it.residual.size for it in items], dtype=int)
        m = int(dof.max())
        resid = np.full((len(items), m), np.nan)
        sigma = np.full((len(items), m), np.nan)
        for i, it in enumerate(items):
            resid[i, :dof[i]] = it.residual
            sigma[i, :dof[i]] = np.sqrt(np.diag(it.S))
        nis = np.array([it.nis for it in items])
        acc = np.array([it.accepted for it in items], dtype=bool)
        return InnovationLog(t, resid, sigma, nis, acc, dof)
