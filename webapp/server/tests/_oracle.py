"""Independent oracles shared by the webapp server tests (not a test module).

Everything here is computed from a ``ClosedLoopSim.snapshot()`` with plain numpy / scipy, never with
the code under test (``webapp.server.adcs_session.frame``).

Tolerance note: transmitted floats carry 7 significant digits (``float(f"{x:.7g}")``), so the
relative rounding error is at most 5e-7 per number. Comparisons of transmitted numbers therefore use
``rtol = RTOL = 1e-6``. Entries of a rotation matrix that are zero up to round-off (~1e-16) are
compared with a small absolute tolerance ``ATOL_MAT`` instead of a relative one.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

RTOL = 1e-6
ATOL_MAT = 1e-12

#: LVLH -> display: 180 degrees about x (spec section 3.2), written out literally.
FLIP = np.diag([1.0, -1.0, -1.0])


def lvlh_rows(r, v) -> np.ndarray:
    """ECI -> LVLH DCM from CONVENTIONS.md section 4: z = -r/|r|, y = -(r x v)/|r x v|, x = y x z."""
    r, v = np.asarray(r, float), np.asarray(v, float)
    z = -r / np.linalg.norm(r)
    y = -np.cross(r, v)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.vstack([x, y, z])


def body_to_eci(q) -> np.ndarray:
    """R(q) for a scalar-first quaternion, via scipy (xyzw ordering)."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def display_C(q, r, v) -> np.ndarray:
    return FLIP @ lvlh_rows(r, v) @ body_to_eci(q)


def display_vec(x_eci, r, v) -> np.ndarray:
    x = np.asarray(x_eci, float)
    return FLIP @ lvlh_rows(r, v) @ (x / np.linalg.norm(x))


def _close(actual, expected, rtol=RTOL, atol=0.0, what=""):
    np.testing.assert_allclose(np.asarray(actual, float), np.asarray(expected, float), rtol=rtol,
                               atol=atol, err_msg=what)


def assert_frame_matches(f: dict, snap: dict, period: float, *, wheel_max_rpm: float = 8000.0) -> None:
    """Assert a transmitted ``frame`` equals the snapshot converted independently (7-digit rounding).

    ``wheel_max_rpm`` is the rated wheel speed of ``WheelConfig`` (8000 rpm).
    """
    r, v, q = snap["r_eci"], snap["v_eci"], snap["q"]
    assert f["type"] == "frame"
    assert f["k"] == snap["k"]
    _close(f["t"], snap["t"], what="t")
    _close(f["orbit"], snap["t"] / period, what="orbit")
    assert f["mode"] == snap["mode"]
    assert f["mode_name"] == {0: "DETUMBLE", 1: "POINTING"}[snap["mode"]]
    assert f["dump"] is bool(snap["dump"])
    assert f["lit"] is bool(snap["lit"])

    e = f["eci"]
    _close(e["q"], q, what="q")
    _close(e["r_m"], r, what="r")
    _close(e["v_m_s"], v, what="v")
    _close(e["sun"], snap["sun_eci"], what="sun_eci")
    _close(e["b_T"], snap["b_eci"], what="b_eci")
    _close(e["gmst_rad"], snap["gmst"], what="gmst")
    d = f["display"]
    _close(d["C"], display_C(q, r, v), atol=ATOL_MAT, what="C")
    _close(d["sun"], display_vec(snap["sun_eci"], r, v), atol=ATOL_MAT, what="display sun")
    _close(d["b"], display_vec(snap["b_eci"], r, v), atol=ATOL_MAT, what="display b")

    m = f["metrics"]
    ws = np.asarray(snap["wheel_speed"], float)
    _close(m["nadir_err_deg"], np.rad2deg(snap["nadir_err"]), what="nadir_err")
    _close(m["rate_deg_s"], np.rad2deg(np.linalg.norm(snap["w"])), what="rate")
    _close(m["w_deg_s"], np.rad2deg(snap["w"]), what="w")
    _close(m["wheel_speed_rpm"], ws * 60.0 / (2.0 * math.pi), what="wheel rpm")
    _close(m["wheel_frac"], np.max(np.abs(ws)) * 60.0 / (2.0 * math.pi) / wheel_max_rpm, what="wheel_frac")
    _close(m["m_Am2"], snap["m"], what="m")
    _close(m["u_w_Nm"], snap["u_w"], what="u_w")
    assert m["sun_used"] == snap["sun_used"]

    ev = f["events"]
    for key in ("switch_times", "fallback_times"):
        assert len(ev[key]) == len(snap[key]), key
        if snap[key]:
            _close(ev[key], snap[key], what=key)

    # estimator fields: null exactly when the snapshot has None / NaN
    if snap["q_est"] is None:
        assert e["q_est"] is None and d["C_est"] is None
    else:
        _close(e["q_est"], snap["q_est"], what="q_est")
        _close(d["C_est"], display_C(snap["q_est"], r, v), atol=ATOL_MAT, what="C_est")
    for key, val in (("est_nadir_err_deg", snap["est_nadir_err"]), ("att_sigma_deg", snap["att_sigma"])):
        if math.isnan(val):
            assert m[key] is None, key
        else:
            _close(m[key], np.rad2deg(val), what=key)
    if snap["know_err"] is None:
        assert m["know_err_deg"] is None
    else:
        _close(m["know_err_deg"], np.rad2deg(np.linalg.norm(snap["know_err"])), what="know_err")
    if snap["w_est"] is None:
        assert m["w_est_deg_s"] is None
    else:
        _close(m["w_est_deg_s"], np.rad2deg(snap["w_est"]), what="w_est")
    if math.isnan(snap["t_init"]):
        assert ev["t_init"] is None
    else:
        _close(ev["t_init"], snap["t_init"], what="t_init")


def numbers(obj):
    """Yield every int/float leaf of a JSON-like structure (bools excluded)."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return
    if isinstance(obj, (int, float)):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from numbers(v)


def without_seq(f: dict) -> dict:
    return {k: v for k, v in f.items() if k != "seq"}
