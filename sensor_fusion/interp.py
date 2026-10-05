"""Time-series interpolation helper for Project A consumers.

Lives here (not in ``loader.py``) so that the loader stays free of resampling
(spec ``docs/specs/sensor_fusion_data.md`` section 2, rule 1).
"""
from __future__ import annotations

import numpy as np

from common import quaternion as quat


def interp_to(t_target, t, x, *, quaternion: bool = False) -> np.ndarray:
    """Interpolate a time series onto new time stamps.

    Parameters
    ----------
    t_target : (M,) array-like, float seconds -- query times (any order).
    t : (N,) array-like, float seconds -- strictly increasing sample times, N >= 1.
    x : (N,), (N, k) array of samples, or (N, 4) scalar-first quaternions
        ``[w, x, y, z]`` (body-to-reference, CONVENTIONS section 2) if
        ``quaternion=True``. Time is on axis 0.
    quaternion : if True, interpolate with ``common.quaternion.slerp`` along the
        shortest arc (output unit norm, sign follows the shortest arc from the
        bracketing left sample); otherwise element-wise linear interpolation.

    Returns
    -------
    ``(M,) + x.shape[1:]`` float array. Queries outside ``[t[0], t[-1]]`` are
    clamped (hold the end value); no extrapolation.

    Raises ``ValueError`` on length mismatch, non-increasing ``t`` or bad shapes.
    """
    t_target = np.atleast_1d(np.asarray(t_target, dtype=float))
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if t.ndim != 1 or t.size < 1:
        raise ValueError("t must be a non-empty 1-D array")
    if x.shape[0] != t.shape[0]:
        raise ValueError(f"x has {x.shape[0]} samples but t has {t.shape[0]}")
    if t_target.ndim != 1:
        raise ValueError("t_target must be 1-D")
    if t.size > 1 and not np.all(np.diff(t) > 0):
        raise ValueError("t must be strictly increasing")
    if quaternion and (x.ndim != 2 or x.shape[1] != 4):
        raise ValueError("quaternion=True requires x of shape (N, 4)")

    if t.size == 1:
        return np.repeat(x[:1], t_target.size, axis=0).copy() if not quaternion \
            else quat.normalize(np.repeat(x[:1], t_target.size, axis=0))

    # left bracketing index i in [0, N-2]; fraction tau in [0, 1] (clamped)
    i = np.clip(np.searchsorted(t, t_target, side="right") - 1, 0, t.size - 2)
    tau = np.clip((t_target - t[i]) / (t[i + 1] - t[i]), 0.0, 1.0)

    if quaternion:
        return quat.slerp(x[i], x[i + 1], tau)
    tau = tau.reshape((-1,) + (1,) * (x.ndim - 1))
    return (1.0 - tau) * x[i] + tau * x[i + 1]
