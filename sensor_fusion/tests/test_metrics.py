"""Tests for ``sensor_fusion.metrics`` (spec ``docs/specs/tuning_writeup.md`` section 1 and section 4 items 1-7).

Independent oracles only: hand-computed values, ``scipy.spatial.transform.Rotation`` (tilt / yaw),
``scipy.stats.chi2`` (NIS statistics) and closed-form constant-offset scenarios for ``compare_to_ekf2``.
Nothing here calls the metric functions "in a different way" to produce expected values.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scipy.stats import chi2

from common.ekf import InnovationLog
from sensor_fusion import metrics as M

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
REPORT_DIR = Path(__file__).resolve().parents[1] / "report"
HOVER_ID = "f4e81624-7ccc-4859-bb45-d02fa0246b77"
HOVER_LOG = DATA_DIR / f"{HOVER_ID}.ulg"
HOVER_JSON = REPORT_DIR / f"ekf_metrics_{HOVER_ID}.json"


# --------------------------------------------------------------------------------------
# helpers (scipy based)
# --------------------------------------------------------------------------------------
def to_sf(r: Rotation) -> np.ndarray:
    """scipy (x, y, z, w) -> repository scalar-first [w, x, y, z]."""
    x = r.as_quat()
    return np.concatenate([x[..., 3:], x[..., :3]], axis=-1)


def from_sf(q) -> Rotation:
    q = np.asarray(q, float)
    return Rotation.from_quat(np.concatenate([q[..., 1:], q[..., :1]], axis=-1))


def qz(angle) -> np.ndarray:
    """Rotation about NED down by ``angle`` [rad] as scalar-first quaternions, shape (N, 4) (closed form)."""
    a = np.atleast_1d(np.asarray(angle, float))
    return np.column_stack([np.cos(a / 2), np.zeros_like(a), np.zeros_like(a), np.sin(a / 2)])


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


@dataclass
class FakeRes:
    """Minimal stand-in for ``EkfResult`` (only the attributes the metrics read)."""
    t: np.ndarray = None
    p: np.ndarray = None
    v: np.ndarray = None
    q: np.ndarray = None
    b_g: np.ndarray = None
    b_a: np.ndarray = None
    std: np.ndarray = None
    innovations: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# module constants
# --------------------------------------------------------------------------------------
def test_constants():
    assert M.SKIP_INIT_S == 5.0
    assert M.RESET_EXCL_S == 1.0
    assert tuple(M.SENSOR_ORDER) == ("gps_pos", "gps_vel", "baro", "mag_heading")


# --------------------------------------------------------------------------------------
# 1. error_stats / clean_json
# --------------------------------------------------------------------------------------
def test_error_stats_known_vector():
    s = M.error_stats(np.array([3.0, 4.0]))
    assert s["rms"] == pytest.approx(math.sqrt(12.5), rel=1e-14)
    assert s["max_abs"] == pytest.approx(4.0, rel=1e-14)
    assert s["final"] == pytest.approx(4.0, rel=1e-14)
    assert s["n"] == 2


def test_error_stats_signed_final_and_max_abs():
    x = np.array([-5.0, 2.0, 1.0, -2.0])
    s = M.error_stats(x)
    assert s["max_abs"] == pytest.approx(5.0, rel=1e-14)        # absolute value
    assert s["final"] == pytest.approx(-2.0, rel=1e-14)         # last sample, signed
    assert s["rms"] == pytest.approx(math.sqrt((25 + 4 + 1 + 4) / 4), rel=1e-14)
    assert s["n"] == 4


def test_error_stats_scale():
    x = np.array([3.0, -4.0])
    s = M.error_stats(x, scale=180.0 / math.pi)
    k = 180.0 / math.pi
    assert s["rms"] == pytest.approx(math.sqrt(12.5) * k, rel=1e-14)
    assert s["max_abs"] == pytest.approx(4.0 * k, rel=1e-14)
    assert s["final"] == pytest.approx(-4.0 * k, rel=1e-14)
    assert s["n"] == 2


def test_error_stats_empty():
    s = M.error_stats(np.array([]))
    assert s["rms"] is None and s["max_abs"] is None and s["final"] is None
    assert s["n"] == 0


def test_clean_json_makes_json_safe():
    obj = {1: np.array([1.0, np.nan, np.inf]), "a": (np.float64(2.5), np.int64(3), np.bool_(True)),
           "b": {"x": float("nan"), "y": [np.array([[1, 2]])]}, "s": "text", "n": None}
    out = M.clean_json(obj)
    text = json.dumps(out)                   # must not raise (no numpy types, no NaN literal needed)
    assert "NaN" not in text and "Infinity" not in text
    assert out["1"] == [1.0, None, None]
    assert out["a"] == [2.5, 3, True]
    assert type(out["a"][1]) is int and type(out["a"][2]) is bool
    assert out["b"]["x"] is None
    assert out["b"]["y"] == [[[1, 2]]]
    assert out["s"] == "text" and out["n"] is None


# --------------------------------------------------------------------------------------
# 2. tilt_angle / yaw_of
# --------------------------------------------------------------------------------------
def test_tilt_identity_is_zero():
    q = np.tile([1.0, 0, 0, 0], (5, 1))
    assert np.allclose(M.tilt_angle(q, q), 0.0, atol=1e-14)
    # also for generic attitudes
    qr = to_sf(Rotation.random(7, random_state=1))
    assert np.allclose(M.tilt_angle(qr, qr), 0.0, atol=1e-12)


def test_tilt_independent_of_yaw():
    # (a) level attitudes with arbitrary headings: both body down axes are +z_NED.
    rng = np.random.default_rng(10)
    yaw1, yaw2 = rng.uniform(-np.pi, np.pi, 20), rng.uniform(-np.pi, np.pi, 20)
    zeros = np.zeros(20)
    q1 = to_sf(Rotation.from_euler("ZYX", np.column_stack([yaw1, zeros, zeros])))
    q2 = to_sf(Rotation.from_euler("ZYX", np.column_stack([yaw2, zeros, zeros])))
    assert np.allclose(M.tilt_angle(q1, q2), 0.0, atol=1e-12)
    # (b) generic attitudes differing by a pure yaw about the BODY down axis, q2 = q1 (x) Rz(d):
    #     R2 e_z = R1 Rz(d) e_z = R1 e_z, so the tilt difference is 0 for any d.
    qg = to_sf(Rotation.random(20, random_state=2))
    d = rng.uniform(-np.pi, np.pi, 20)
    qy = to_sf(from_sf(qg) * Rotation.from_rotvec(d[:, None] * np.array([0.0, 0.0, 1.0])))
    assert np.allclose(M.tilt_angle(qg, qy), 0.0, atol=1e-12)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_tilt_equals_rotation_angle_between_body_down_axes(axis):
    # q2 = q1 (x) R_axis(delta) (body-frame perturbation): the body down axes differ by exactly delta.
    rng = np.random.default_rng(11)
    q1 = to_sf(Rotation.random(30, random_state=5))
    delta = rng.uniform(0.05, 3.0, 30)
    unit = np.array([1.0, 0.0, 0.0]) if axis == "x" else np.array([0.0, 1.0, 0.0])
    q2 = to_sf(from_sf(q1) * Rotation.from_rotvec(delta[:, None] * unit))
    assert np.allclose(M.tilt_angle(q1, q2), delta, atol=1e-12)


def test_tilt_matches_scipy_down_axis_angle_random():
    q1 = to_sf(Rotation.random(200, random_state=21))
    q2 = to_sf(Rotation.random(200, random_state=22))
    ez = np.array([0.0, 0.0, 1.0])
    a, b = from_sf(q1).apply(ez), from_sf(q2).apply(ez)
    expected = np.arccos(np.clip(np.sum(a * b, axis=1), -1, 1))
    # arccos is ill-conditioned at 0 and pi: random samples stay away from both.
    assert np.allclose(M.tilt_angle(q1, q2), expected, atol=1e-9)


def test_tilt_sign_ambiguity_and_range():
    q1 = to_sf(Rotation.random(20, random_state=3))
    q2 = to_sf(Rotation.random(20, random_state=4))
    t = M.tilt_angle(q1, q2)
    assert t.shape == (20,)
    assert np.all(t >= 0) and np.all(t <= np.pi + 1e-12)
    assert np.allclose(t, M.tilt_angle(q1, -q2), atol=1e-12)           # q and -q are the same rotation
    assert np.allclose(t, M.tilt_angle(q2, q1), atol=1e-12)           # symmetric


def test_tilt_known_roll():
    # 30 deg roll vs level: the down axis of a roll of phi is [0, -sin(phi), cos(phi)] -> angle phi.
    q1 = np.array([[1.0, 0, 0, 0]])
    q2 = to_sf(Rotation.from_euler("x", np.radians(30.0)))[None]
    assert M.tilt_angle(q1, q2)[0] == pytest.approx(math.radians(30.0), abs=1e-14)


def test_yaw_of_matches_scipy_zyx():
    rng = np.random.default_rng(12)
    yaw = rng.uniform(-np.pi + 0.01, np.pi - 0.01, 100)
    pitch = rng.uniform(-1.2, 1.2, 100)            # away from gimbal lock at +/- pi/2
    roll = rng.uniform(-3.0, 3.0, 100)
    r = Rotation.from_euler("ZYX", np.column_stack([yaw, pitch, roll]))
    out = M.yaw_of(to_sf(r))
    assert out.shape == (100,)
    assert np.allclose(wrap(out - r.as_euler("ZYX")[:, 0]), 0.0, atol=1e-12)
    assert np.allclose(wrap(out - yaw), 0.0, atol=1e-12)


def test_yaw_of_pure_yaw_and_sign_invariance():
    ang = np.radians([-170.0, -90.0, 0.0, 45.0, 179.0])
    q = qz(ang)
    assert np.allclose(M.yaw_of(q), ang, atol=1e-14)
    assert np.allclose(M.yaw_of(-q), ang, atol=1e-14)


# --------------------------------------------------------------------------------------
# 3. reset_mask
# --------------------------------------------------------------------------------------
def test_reset_mask_boundary_exact():
    # values exactly representable in binary so |t - t_r| == half is hit exactly.
    t = np.arange(0.0, 40.0001, 0.25)
    m = M.reset_mask(t, [20.0], half=1.0)
    assert m.shape == t.shape and m.dtype == bool
    inside = np.abs(t - 20.0) <= 1.0                         # the spec: masks exactly |t - t_r| <= half
    # The mask is a partition into exactly this set and its complement (which side is True is checked
    # through compare_to_ekf2 below, where the mask is used to *exclude* samples).
    assert np.array_equal(m, inside) or np.array_equal(m, ~inside)
    assert inside.sum() == 9          # 19.0, 19.25, ..., 21.0


def test_reset_mask_multiple_resets_and_default_half():
    t = np.arange(0.0, 100.0001, 0.5)
    resets = [10.0, 55.5]
    m = M.reset_mask(t, resets)                              # default half = RESET_EXCL_S = 1
    inside = (np.abs(t - 10.0) <= 1.0) | (np.abs(t - 55.5) <= 1.0)
    assert np.array_equal(m, inside) or np.array_equal(m, ~inside)
    assert inside.sum() == 10        # 5 samples per reset
    m2 = M.reset_mask(t, resets, half=2.5)
    inside2 = (np.abs(t - 10.0) <= 2.5) | (np.abs(t - 55.5) <= 2.5)
    assert np.array_equal(m2, inside2) or np.array_equal(m2, ~inside2)
    # same polarity for every call
    assert np.array_equal(m, inside) == np.array_equal(m2, inside2)


def test_reset_mask_no_resets():
    t = np.linspace(0, 10, 11)
    m = M.reset_mask(t, [])
    assert m.shape == t.shape
    assert m.all() or (~m).all()


# --------------------------------------------------------------------------------------
# 4. compare_to_ekf2 on fabricated res / ref
# --------------------------------------------------------------------------------------
T_RES = np.arange(0.0, 60.0001, 0.5)             # filter grid (exact binary fractions)
T_POS = np.arange(0.25, 59.9, 0.5)               # EKF2 position grid, half-way between filter samples
T_ATT = np.arange(0.0, 60.0001, 0.25)            # EKF2 attitude grid: on and between filter samples
YAW0, YAW_RATE = math.radians(150.0), math.radians(1.0)   # crosses +/-180 deg at t = 30 s


def truth_pos(t):
    t = np.asarray(t, float)
    return np.column_stack([0.5 * t, -0.2 * t, 0.1 * t])    # linear: linear interpolation is exact


def truth_yaw(t):
    return YAW0 + YAW_RATE * np.asarray(t, float)


def make_pair(pos_off=(0, 0, 0), vel_off=(0, 0, 0), yaw_off_deg=0.0, yaw_extra=None, resets=()):
    """res/ref with constant offsets ours - EKF2; ``yaw_extra(ta)`` [rad] adds a time dependent yaw difference."""
    ta = T_ATT
    extra = np.zeros_like(ta) if yaw_extra is None else yaw_extra(ta)
    ref = {
        "t_pos": T_POS, "pos_ned": truth_pos(T_POS), "vel_ned": np.tile([1.0, 0.0, -0.5], (T_POS.size, 1)),
        "t_att": ta, "q": qz(truth_yaw(ta) - math.radians(yaw_off_deg) - extra),
        "reset_times": list(resets),
    }
    res = FakeRes(t=T_RES, p=truth_pos(T_RES) + np.asarray(pos_off, float),
                  v=np.tile([1.0, 0.0, -0.5], (T_RES.size, 1)) + np.asarray(vel_off, float),
                  q=qz(truth_yaw(T_RES)))
    return res, ref


def test_compare_constant_position_offset():
    res, ref = make_pair(pos_off=(3.0, 4.0, 5.0))
    out = M.compare_to_ekf2(res, ref)
    assert set(out) >= {"cmp", "arr"}
    c = out["cmp"]
    n_expected = int(np.sum(T_POS >= 5.0))
    assert c["pos_horizontal_m"]["rms"] == pytest.approx(5.0, rel=1e-12)
    assert c["pos_horizontal_m"]["max_abs"] == pytest.approx(5.0, rel=1e-12)
    assert c["pos_horizontal_m"]["final"] == pytest.approx(5.0, rel=1e-12)
    assert c["pos_horizontal_m"]["n"] == n_expected
    assert c["pos_vertical_m"]["rms"] == pytest.approx(5.0, rel=1e-12)
    assert c["pos_vertical_m"]["final"] == pytest.approx(5.0, rel=1e-12)
    assert c["vertical_signed_mean_m"] == pytest.approx(5.0, rel=1e-12)
    # velocities agree and the attitudes agree
    assert c["vel_m_s"]["rms"] == pytest.approx(0.0, abs=1e-12)
    assert c["tilt_deg"]["rms"] == pytest.approx(0.0, abs=1e-6)
    assert c["yaw_deg"]["rms"] == pytest.approx(0.0, abs=1e-9)


def test_compare_vertical_sign_is_ours_minus_ekf2():
    res, ref = make_pair(pos_off=(0.0, 0.0, -2.0))
    c = M.compare_to_ekf2(res, ref)["cmp"]
    assert c["vertical_signed_mean_m"] == pytest.approx(-2.0, rel=1e-12)
    assert c["pos_vertical_m"]["rms"] == pytest.approx(2.0, rel=1e-12)       # RMS is unsigned
    assert c["pos_vertical_m"]["final"] == pytest.approx(-2.0, rel=1e-12)    # final is signed
    assert c["pos_horizontal_m"]["rms"] == pytest.approx(0.0, abs=1e-12)


def test_compare_constant_velocity_offset():
    res, ref = make_pair(vel_off=(1.0, 2.0, 2.0))        # norm 3
    c = M.compare_to_ekf2(res, ref)["cmp"]
    assert c["vel_m_s"]["rms"] == pytest.approx(3.0, rel=1e-12)
    assert c["vel_m_s"]["max_abs"] == pytest.approx(3.0, rel=1e-12)
    assert c["vel_m_s"]["n"] == int(np.sum(T_POS >= 5.0))
    assert c["pos_horizontal_m"]["rms"] == pytest.approx(0.0, abs=1e-12)


def test_compare_constant_yaw_offset_10_deg_with_wrap():
    # yaw crosses +/-180 deg at t = 30 s, the wrapped difference must stay 10 deg.
    res, ref = make_pair(yaw_off_deg=10.0)
    c = M.compare_to_ekf2(res, ref)["cmp"]
    assert c["yaw_deg"]["rms"] == pytest.approx(10.0, rel=1e-9)
    assert c["yaw_deg"]["max_abs"] == pytest.approx(10.0, rel=1e-9)
    assert c["yaw_deg"]["final"] == pytest.approx(10.0, rel=1e-9)
    assert c["yaw_signed_mean_deg"] == pytest.approx(10.0, rel=1e-9)
    assert c["tilt_deg"]["rms"] == pytest.approx(0.0, abs=1e-6)
    assert c["n_att_samples_excluded_resets"] == 0


def test_compare_negative_yaw_offset_signed_mean():
    res, ref = make_pair(yaw_off_deg=-7.0)
    c = M.compare_to_ekf2(res, ref)["cmp"]
    assert c["yaw_deg"]["rms"] == pytest.approx(7.0, rel=1e-9)
    assert c["yaw_signed_mean_deg"] == pytest.approx(-7.0, rel=1e-9)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_compare_constant_tilt_offset(axis):
    # constant attitudes: ours = ref (x) R_axis(2 deg)  ->  tilt difference exactly 2 deg (scipy-built).
    r0 = Rotation.from_euler("ZYX", [0.7, 0.2, -0.3])
    q_ref = to_sf(r0)
    q_ours = to_sf(r0 * Rotation.from_euler(axis, np.radians(2.0)))
    ref = {"t_pos": T_POS, "pos_ned": truth_pos(T_POS), "vel_ned": np.zeros((T_POS.size, 3)),
           "t_att": T_ATT, "q": np.tile(q_ref, (T_ATT.size, 1)), "reset_times": []}
    res = FakeRes(t=T_RES, p=truth_pos(T_RES), v=np.zeros((T_RES.size, 3)), q=np.tile(q_ours, (T_RES.size, 1)))
    c = M.compare_to_ekf2(res, ref)["cmp"]
    assert c["tilt_deg"]["rms"] == pytest.approx(2.0, rel=1e-9)
    assert c["tilt_deg"]["max_abs"] == pytest.approx(2.0, rel=1e-9)


def _outlier_extra(ta):
    """+70 deg in the first 5 s, +50 deg within +/-1 s of a reset at 20 s (on top of the 10 deg offset)."""
    e = np.zeros_like(ta)
    e[ta < 5.0] = math.radians(70.0)
    e[np.abs(ta - 20.0) <= 1.0] = math.radians(50.0)
    return e


def test_compare_skips_first_5s_and_excludes_resets():
    res, ref = make_pair(yaw_off_deg=10.0, yaw_extra=_outlier_extra, resets=[20.0])
    c = M.compare_to_ekf2(res, ref)["cmp"]
    # with the outliers excluded the yaw difference is exactly the 10 deg offset
    assert c["yaw_deg"]["rms"] == pytest.approx(10.0, rel=1e-9)
    assert c["yaw_deg"]["max_abs"] == pytest.approx(10.0, rel=1e-9)
    after = T_ATT >= 5.0
    in_window = np.abs(T_ATT - 20.0) <= 1.0
    assert in_window.sum() == 9                                        # 19.0 ... 21.0 inclusive
    n_excl = int((after & in_window).sum())
    assert c["n_att_samples_excluded_resets"] == n_excl == 9
    assert c["yaw_deg"]["n"] == int(after.sum()) - n_excl
    assert c["tilt_deg"]["n"] == int(after.sum()) - n_excl
    assert c["yaw_signed_mean_deg"] == pytest.approx(10.0, rel=1e-9)


def test_compare_without_reset_outliers_are_counted():
    # same data but no reset declared -> the window outliers enter the statistics (and nothing is excluded)
    res, ref = make_pair(yaw_off_deg=10.0, yaw_extra=_outlier_extra, resets=[])
    c = M.compare_to_ekf2(res, ref)["cmp"]
    after = T_ATT >= 5.0
    d = np.full(T_ATT.size, 10.0)
    d[np.abs(T_ATT - 20.0) <= 1.0] = 60.0
    expected_rms = math.sqrt(np.mean(d[after] ** 2))
    assert c["n_att_samples_excluded_resets"] == 0
    assert c["yaw_deg"]["n"] == int(after.sum())
    assert c["yaw_deg"]["rms"] == pytest.approx(expected_rms, rel=1e-9)
    assert c["yaw_deg"]["max_abs"] == pytest.approx(60.0, rel=1e-9)


def test_compare_two_resets_and_custom_window():
    res, ref = make_pair(yaw_off_deg=10.0, yaw_extra=_outlier_extra, resets=[20.0, 40.0])
    c = M.compare_to_ekf2(res, ref, reset_excl_s=2.0)["cmp"]
    after = T_ATT >= 5.0
    excl = (np.abs(T_ATT - 20.0) <= 2.0) | (np.abs(T_ATT - 40.0) <= 2.0)
    n_excl = int((after & excl).sum())
    assert n_excl == 34                                   # 2 windows x 17 samples (18 ... 22 s, 38 ... 42 s)
    assert c["n_att_samples_excluded_resets"] == n_excl
    assert c["yaw_deg"]["n"] == int(after.sum()) - n_excl
    assert c["yaw_deg"]["rms"] == pytest.approx(10.0, rel=1e-9)     # the +50 deg samples sit within +/-1 s of 20 s


def test_compare_custom_skip_init():
    res, ref = make_pair(pos_off=(3.0, 4.0, 5.0), yaw_off_deg=10.0)
    c = M.compare_to_ekf2(res, ref, skip_init_s=20.0)["cmp"]
    assert c["pos_horizontal_m"]["n"] == int(np.sum(T_POS >= 20.0))
    assert c["yaw_deg"]["n"] == int(np.sum(T_ATT >= 20.0))
    assert c["pos_horizontal_m"]["rms"] == pytest.approx(5.0, rel=1e-12)


def test_compare_init_samples_only_in_max_incl_init():
    # a big position error in the first 4 s must not enter RMS/max_abs but must show in max_incl_init.
    res, ref = make_pair(pos_off=(3.0, 4.0, 5.0))
    res.p = res.p.copy()
    res.p[T_RES < 4.0, 0] += 100.0           # x error 103 vs 3 before t = 4 s
    out = M.compare_to_ekf2(res, ref)["cmp"]
    assert out["pos_horizontal_m"]["rms"] == pytest.approx(5.0, rel=1e-12)
    assert out["pos_horizontal_m"]["max_abs"] == pytest.approx(5.0, rel=1e-12)
    assert out["max_incl_init"]["pos_horizontal_m"] > 100.0
    assert out["max_incl_init"]["pos_vertical_m"] == pytest.approx(5.0, rel=1e-12)


def test_compare_samples_outside_filter_span_dropped():
    # EKF2 samples outside [t0, t1] of the filter result cannot be compared and are not counted.
    res, ref = make_pair(pos_off=(3.0, 4.0, 5.0))
    keep = T_RES <= 30.0
    res.t, res.p, res.v, res.q = res.t[keep], res.p[keep], res.v[keep], res.q[keep]
    c = M.compare_to_ekf2(res, ref)["cmp"]
    assert c["pos_horizontal_m"]["n"] == int(np.sum((T_POS >= 5.0) & (T_POS <= 30.0)))
    assert c["yaw_deg"]["n"] == int(np.sum((T_ATT >= 5.0) & (T_ATT <= 30.0)))


# --------------------------------------------------------------------------------------
# 5. nis_summary
# --------------------------------------------------------------------------------------
def make_innov(nis, dof, t, accepted=None):
    nis = np.asarray(nis, float)
    n = nis.size
    acc = np.ones(n, bool) if accepted is None else np.asarray(accepted, bool)
    return InnovationLog(t=np.asarray(t, float), residual=np.zeros((n, dof)), sigma=np.ones((n, dof)),
                         nis=nis, accepted=acc, dof=np.full(n, dof, int))


@pytest.mark.parametrize("dof", [1, 3])
def test_nis_summary_chi2_samples(dof):
    rng = np.random.default_rng(100 + dof)
    n = 6000
    nis = chi2.rvs(dof, size=n, random_state=rng)
    t = np.linspace(0.0, 120.0, n)
    res = FakeRes(innovations={"gps_pos": make_innov(nis, dof, t)})
    s = M.nis_summary(res, 5.0)["gps_pos"]
    se = math.sqrt(2.0 * dof / n)
    assert s["dof"] == dof
    assert s["n_total"] == n and s["n_accepted"] == n and s["n_rejected"] == 0
    assert abs(s["mean_nis"] - dof) < 4 * se                  # mean of chi2(dof) is dof
    assert s["mean_nis"] == pytest.approx(np.mean(nis), rel=1e-12)
    assert s["median_nis"] == pytest.approx(np.median(nis), rel=1e-12)
    # fraction of samples inside the two-sided 95 % chi2 interval
    lo, hi = chi2.ppf([0.025, 0.975], dof)
    assert 0.93 <= s["fraction_inside_95"] <= 0.97
    assert s["fraction_inside_95"] == pytest.approx(np.mean((nis >= lo) & (nis <= hi)), abs=1e-12)
    # 95 % band of the mean of n NIS values: sum of n chi2(dof) is chi2(n dof)
    blo, bhi = chi2.ppf([0.025, 0.975], n * dof) / n
    assert s["mean_bounds_95"][0] == pytest.approx(blo, rel=1e-8)
    assert s["mean_bounds_95"][1] == pytest.approx(bhi, rel=1e-8)
    assert s["mean_inside_bounds"] is True or s["mean_inside_bounds"] == True  # noqa: E712
    # after-5-s sub-statistics
    late = t >= 5.0
    a = s["after_5s"]
    assert a["n"] == int(late.sum())
    assert a["mean_nis"] == pytest.approx(np.mean(nis[late]), rel=1e-12)
    assert 0.93 <= a["fraction_inside_95"] <= 0.97


def test_nis_summary_inflated_innovations_flag_inconsistency():
    dof, n = 3, 4000
    rng = np.random.default_rng(7)
    nis = chi2.rvs(dof, size=n, random_state=rng) * 9.0       # innovations x3  ->  quadratic form x9
    res = FakeRes(innovations={"gps_vel": make_innov(nis, dof, np.linspace(0, 100, n))})
    s = M.nis_summary(res, 5.0)["gps_vel"]
    assert s["mean_inside_bounds"] is False or s["mean_inside_bounds"] == False  # noqa: E712
    assert s["mean_nis"] == pytest.approx(9.0 * dof, rel=0.1)
    assert s["fraction_inside_95"] < 0.5


def test_nis_summary_rejected_updates_not_counted():
    dof, n = 1, 3000
    rng = np.random.default_rng(8)
    nis = chi2.rvs(dof, size=n, random_state=rng)
    acc = np.ones(n, bool)
    acc[::10] = False
    nis_with_junk = nis.copy()
    nis_with_junk[~acc] = 1.0e6                                # would dominate any statistic if counted
    t = np.linspace(0, 100, n)
    res = FakeRes(innovations={"baro": make_innov(nis_with_junk, dof, t, acc)})
    s = M.nis_summary(res, 5.0)["baro"]
    assert s["n_total"] == n
    assert s["n_accepted"] == int(acc.sum())
    assert s["n_rejected"] == int((~acc).sum())
    assert s["mean_nis"] == pytest.approx(np.mean(nis[acc]), rel=1e-12)
    assert s["mean_nis"] < 2.0
    assert 0.93 <= s["fraction_inside_95"] <= 0.97
    late = acc & (t >= 5.0)
    assert s["after_5s"]["n"] == int(late.sum())
    assert s["after_5s"]["mean_nis"] == pytest.approx(np.mean(nis[late]), rel=1e-12)


def test_nis_summary_only_sensors_present_and_sensor_order():
    rng = np.random.default_rng(9)
    n = 200
    t = np.linspace(0, 50, n)
    res = FakeRes(innovations={
        "baro": make_innov(chi2.rvs(1, size=n, random_state=rng), 1, t),
        "gps_pos": make_innov(chi2.rvs(3, size=n, random_state=rng), 3, t),
    })
    out = M.nis_summary(res, 5.0)
    assert set(out) == {"gps_pos", "baro"}
    assert list(out) == ["gps_pos", "baro"]                    # SENSOR_ORDER, not insertion order
    assert out["gps_pos"]["dof"] == 3 and out["baro"]["dof"] == 1


def test_nis_summary_no_late_samples_has_no_after_5s():
    n = 50
    t = np.linspace(0.0, 4.0, n)                                # all before t_eval0
    res = FakeRes(innovations={"baro": make_innov(np.ones(n), 1, t)})
    s = M.nis_summary(res, 5.0)["baro"]
    assert s["n_accepted"] == n
    assert "after_5s" not in s


# --------------------------------------------------------------------------------------
# 6. consistency_score
# --------------------------------------------------------------------------------------
def entry(mean_nis, dof, n=100, frac=0.9):
    return {"dof": dof, "n_accepted": n, "mean_nis": mean_nis,
            "after_5s": {"n": n, "mean_nis": mean_nis, "fraction_inside_95": frac, "mean_bounds_95": [0.0, 9.0]}}


def test_consistency_ratio_one_scores_zero():
    out = M.consistency_score({"gps_pos": entry(3.0, 3), "baro": entry(1.0, 1)})
    assert out["score"] == pytest.approx(0.0, abs=1e-12)
    assert out["n_sensors"] == 2
    for name in ("gps_pos", "baro"):
        p = out["per_sensor"][name]
        assert p["ratio"] == pytest.approx(1.0, abs=1e-12)
        assert p["log_ratio"] == pytest.approx(0.0, abs=1e-12)


def test_consistency_ratio_two_and_half_same_score():
    s2 = M.consistency_score({"gps_pos": entry(6.0, 3)})            # ratio 2
    sh = M.consistency_score({"gps_pos": entry(1.5, 3)})            # ratio 0.5
    assert s2["score"] == pytest.approx(math.log(2.0), rel=1e-12)
    assert sh["score"] == pytest.approx(math.log(2.0), rel=1e-12)
    assert s2["per_sensor"]["gps_pos"]["ratio"] == pytest.approx(2.0, rel=1e-12)
    assert sh["per_sensor"]["gps_pos"]["ratio"] == pytest.approx(0.5, rel=1e-12)
    assert s2["per_sensor"]["gps_pos"]["log_ratio"] == pytest.approx(math.log(2.0), rel=1e-12)
    assert sh["per_sensor"]["gps_pos"]["log_ratio"] == pytest.approx(-math.log(2.0), rel=1e-12)


def test_consistency_score_is_rms_of_log_ratios():
    nis = {"gps_pos": entry(3.0, 3), "gps_vel": entry(12.0, 3), "baro": entry(0.25, 1)}   # ratios 1, 4, 0.25
    out = M.consistency_score(nis)
    expected = math.sqrt((0.0 + math.log(4.0) ** 2 + math.log(0.25) ** 2) / 3.0)
    assert out["score"] == pytest.approx(expected, rel=1e-12)
    assert out["n_sensors"] == 3
    # two sensors at ratio 2 and 0.5 -> score ln 2 as well
    out2 = M.consistency_score({"a": entry(2.0, 1), "b": entry(0.5, 1)})
    assert out2["score"] == pytest.approx(math.log(2.0), rel=1e-12)


def test_consistency_passes_through_fraction_and_n():
    out = M.consistency_score({"gps_pos": entry(3.0, 3, n=77, frac=0.81)})
    p = out["per_sensor"]["gps_pos"]
    assert p["fraction_inside_95"] == pytest.approx(0.81, rel=1e-12)
    assert p["n"] == 77


def test_consistency_min_updates_filters_sensors():
    nis = {"gps_pos": entry(9.0, 3, n=4),        # ratio 3 but only 4 updates -> ignored (default min 5)
           "baro": entry(1.0, 1, n=5)}           # exactly min_updates -> used (n >= min_updates)
    out = M.consistency_score(nis)
    assert out["n_sensors"] == 1
    assert set(out["per_sensor"]) == {"baro"}
    assert out["score"] == pytest.approx(0.0, abs=1e-12)
    out10 = M.consistency_score(nis, min_updates=10)
    assert out10["n_sensors"] == 0 and out10["score"] is None
    out1 = M.consistency_score(nis, min_updates=1)
    assert out1["n_sensors"] == 2


def test_consistency_missing_after_5s_ignored():
    e = entry(3.0, 3)
    del e["after_5s"]
    out = M.consistency_score({"gps_pos": e, "baro": entry(2.0, 1)})
    assert set(out["per_sensor"]) == {"baro"}
    assert out["score"] == pytest.approx(math.log(2.0), rel=1e-12)


def test_consistency_empty_is_none():
    out = M.consistency_score({})
    assert out["score"] is None
    assert out["n_sensors"] == 0
    assert out["per_sensor"] == {}


def test_consistency_on_chi2_nis_summary_is_small():
    # integration with nis_summary: consistent synthetic innovations -> ratio ~ 1 -> small score
    rng = np.random.default_rng(55)
    n = 5000
    t = np.linspace(0, 100, n)
    res = FakeRes(innovations={"gps_pos": make_innov(chi2.rvs(3, size=n, random_state=rng), 3, t),
                               "baro": make_innov(chi2.rvs(1, size=n, random_state=rng), 1, t)})
    out = M.consistency_score(M.nis_summary(res, 5.0))
    assert out["n_sensors"] == 2
    assert out["score"] < 0.06                 # ratio std ~ sqrt(2/(dof n)) <= 0.02 per sensor
    inflated = FakeRes(innovations={"gps_pos": make_innov(9.0 * chi2.rvs(3, size=n, random_state=rng), 3, t)})
    big = M.consistency_score(M.nis_summary(inflated, 5.0))
    assert big["score"] == pytest.approx(math.log(9.0), rel=0.05)


# --------------------------------------------------------------------------------------
# 7. bias_summary
# --------------------------------------------------------------------------------------
def test_bias_summary_keys_and_values():
    rng = np.random.default_rng(3)
    n = 6
    b_g, b_a = rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
    std = np.abs(rng.normal(size=(n, 15))) + 0.1
    res = FakeRes(b_g=b_g, b_a=b_a, std=std,
                  meta={"init": {"b_g": [0.01, 0.02, 0.03], "b_a": [0.1, 0.2, 0.3]}})
    out = M.bias_summary(res)
    assert set(out) == {"b_g_rad_s", "b_g_sigma_rad_s", "b_a_m_s2", "b_a_sigma_m_s2",
                        "b_g_init_rad_s", "b_a_init_m_s2"}
    # error-state order [dp, dv, dtheta, db_g, db_a]: gyro-bias sigmas are columns 9:12, accel 12:15
    assert np.allclose(out["b_g_rad_s"], b_g[-1], rtol=1e-14, atol=0)
    assert np.allclose(out["b_a_m_s2"], b_a[-1], rtol=1e-14, atol=0)
    assert np.allclose(out["b_g_sigma_rad_s"], std[-1, 9:12], rtol=1e-14, atol=0)
    assert np.allclose(out["b_a_sigma_m_s2"], std[-1, 12:15], rtol=1e-14, atol=0)
    assert list(out["b_g_init_rad_s"]) == [0.01, 0.02, 0.03]
    assert list(out["b_a_init_m_s2"]) == [0.1, 0.2, 0.3]
    json.dumps(out)                                             # JSON-serialisable (plain lists / floats)


# --------------------------------------------------------------------------------------
# real log (skipped without the .ulg / stored metrics)
# --------------------------------------------------------------------------------------
@pytest.mark.skipif(not (HOVER_LOG.exists() and HOVER_JSON.exists()),
                    reason="hover .ulg or stored ekf_metrics JSON missing")
def test_ekf2_bias_reference_matches_stored_json():
    from sensor_fusion.loader import load_log
    ref = M.ekf2_bias_reference(load_log(HOVER_LOG))
    stored = json.loads(HOVER_JSON.read_text())["ekf2_bias_reference"]
    assert ref["available"] is True and stored["available"] is True
    for k in ("selector_accel_device_id", "selector_gyro_device_id", "multi_id", "n_samples"):
        assert ref[k] == stored[k]
    for k in ("accel_bias_median", "accel_bias_start", "accel_bias_end", "gyro_bias_median",
              "gyro_bias_start", "gyro_bias_end", "t_start_s", "t_end_s"):
        assert np.allclose(ref[k], stored[k], rtol=1e-9, atol=0), k
    assert ref["t"].shape[0] == ref["accel_bias"].shape[0] == ref["gyro_bias"].shape[0] == stored["n_samples"]
