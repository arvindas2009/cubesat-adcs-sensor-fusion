"""Tests for the PX4 replay server (spec: docs/specs/px4_replay.md sections 1, 2 and 4).

Written from the spec. Oracles are independent of the code under test: a separate ``run_filter`` run
(deterministic), ``sensor_fusion.metrics.compare_to_ekf2``, ``numpy.interp``, ``scipy.spatial.transform``
(``Rotation`` / ``Slerp``; scipy is scalar-last ``[x, y, z, w]``, the repository is scalar-first
``[w, x, y, z]``) and ``scipy.stats.chi2``.

Synthetic data: a 30 s noisy figure-8 flight with the "EKF2 reference" taken from the truth. The grid
(50 Hz) contains every second IMU sample (100 Hz), so grid values can be compared with filter samples
directly. Transmitted JSON numbers carry 7 significant digits, i.e. a relative rounding error of at
most 5e-7 (compared with ``rtol = 5e-7`` plus an absolute floor of 1e-12 for zeros).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("pydantic")

from fastapi.testclient import TestClient  # noqa: E402
from scipy.spatial.transform import Rotation, Slerp  # noqa: E402
from scipy.stats import chi2  # noqa: E402

from common import quaternion as quat  # noqa: E402
from sensor_fusion import metrics  # noqa: E402
from sensor_fusion.download_logs import PINNED_LOGS  # noqa: E402
from sensor_fusion.ekf_drone import EkfConfig, run_filter  # noqa: E402
from sensor_fusion.synthetic import SyntheticNoise, make_synthetic_flight  # noqa: E402
from webapp.server import px4_replay  # noqa: E402
from webapp.server.app import create_app  # noqa: E402
from webapp.server.px4_replay import (CACHE_VERSION, CAVEAT, DOWNLOAD_HINT, LogMissing,  # noqa: E402
                                      ReplayStore, UnknownLog, build_replay, replay_to_json)

REPO = Path(__file__).resolve().parents[3]
RTOL_JSON = 5e-7 * (1 + 1e-6)     # 7 significant digits: worst-case relative rounding error is 5e-7
ATOL_JSON = 1e-12                 # for values that are exactly (or numerically) zero
RATE = 50.0
NOISE = SyntheticNoise(accel_noise=0.02, gyro_noise=1e-3, gps_pos_sigma=(0.5, 0.5, 0.8),
                       gps_vel_sigma=0.1, baro_sigma=0.5, mag_sigma=1e-6)
#: gate_prob < 1 so that a few updates are rejected (``accepted`` contains both values).
CFG = EkfConfig(increment=3, gps_delay_s=0.0, gate_prob=0.9)
META = {"id": "synthetic", "role": "test", "notes": "unit test", "duration_s": 30.0}
TOP_KEYS = {"schema_version", "caveat", "rate_hz", "log", "filter", "t", "ours", "ekf2", "diff", "nis", "events"}
OURS_KEYS = {"p_ned_m", "q", "euler_deg", "sigma3_att_deg", "sigma3_pos_m"}
EKF2_KEYS = {"p_ned_m", "q", "euler_deg"}
DIFF_KEYS = {"pos_h_m", "pos_d_m", "tilt_deg", "yaw_deg"}
NIS_KEYS = {"t", "nis", "accepted", "dof", "band95"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_ref(truth, *, pos_span=None, att_span=None, resets=()):
    """The ``ekf2_reference``-shaped dict built from the synthetic truth (optionally cropped in time)."""
    def crop(span):
        if span is None:
            return np.ones(truth.t.shape, bool)
        return (truth.t >= span[0]) & (truth.t <= span[1])
    mp, ma = crop(pos_span), crop(att_span)
    return {"t_pos": truth.t[mp], "pos_ned": truth.p[mp], "vel_ned": truth.v[mp],
            "t_att": truth.t[ma], "q": truth.q[ma], "reset_times": list(resets)}


def qdist(a, b):
    """Largest component difference between quaternion rows, modulo the q / -q ambiguity."""
    a, b = np.asarray(a), np.asarray(b)
    return np.minimum(np.abs(a - b).max(axis=-1), np.abs(a + b).max(axis=-1))


def scipy_euler_deg(q):
    """[roll, pitch, yaw] in degrees from scalar-first body->NED quaternions, via scipy (ZYX = yaw, pitch, roll)."""
    q = np.asarray(q, float)
    rot = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    return rot.as_euler("ZYX", degrees=True)[:, ::-1]


def wrap_deg(d):
    return (np.asarray(d) + 180.0) % 360.0 - 180.0


def to_array(x):
    """Parsed JSON (None -> NaN) to a float array."""
    return np.array(x, dtype=float)


def assert_json_close(parsed, arr):
    got, want = to_array(parsed), np.asarray(arr, float)
    assert got.shape == want.shape
    assert np.array_equal(np.isnan(got), np.isnan(want) | np.isinf(want))
    ok = ~np.isnan(got)
    err = np.abs(got[ok] - want[ok])
    assert np.all(err <= RTOL_JSON * np.abs(want[ok]) + ATOL_JSON), float(err.max())


def make_streams_ctx(duration=10.0, seed=2):
    """A short synthetic flight with a fake geodetic origin and GPS start time, plus its reference."""
    streams, truth = make_synthetic_flight("figure8", duration=duration, imu_rate=100, seed=seed, noise=NOISE)
    streams = dataclasses.replace(
        streams, origin_lla=(math.radians(47.4), math.radians(8.5), 500.0),
        meta={**streams.meta, "gps_utc_start": "2024-05-01 12:00:00+00:00"})
    return streams, make_ref(truth)


class Loader:
    """Counting ``context_loader`` (optionally slow or failing)."""

    def __init__(self, streams, ref, delay=0.0, fail=None):
        self.streams, self.ref, self.delay, self.fail = streams, ref, delay, fail
        self.calls = []
        self._lock = threading.Lock()

    def __call__(self, entry):
        with self._lock:
            self.calls.append(entry["id"])
        if self.delay:
            time.sleep(self.delay)
        if self.fail is not None:
            raise self.fail
        return self.streams, self.ref


def make_cfg():
    return EkfConfig(increment=3, gps_delay_s=0.0)


def make_store(tmp_path, loader, *, present=("hover", "aggressive"), cfg_factory=make_cfg, **kw):
    """Store on a tmp data dir holding fake ``.ulg`` files for the given roles and a not-yet-created cache dir."""
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    for e in PINNED_LOGS:
        if e["role"] in present and not (data / f"{e['id']}.ulg").exists():   # never touch an existing file: mtime is part of the cache key
            (data / f"{e['id']}.ulg").write_bytes(b"ULog\x01\x12\x35" + bytes(100))
    return ReplayStore(cache_dir=tmp_path / "cache" / "nested", data_dir=data, context_loader=loader,
                       cfg_factory=cfg_factory, **kw)


def entry_of(role):
    return next(e for e in PINNED_LOGS if e["role"] == role)


# ---------------------------------------------------------------------------
# module-scoped data (built once)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def flight():
    return make_synthetic_flight("figure8", duration=30, imu_rate=100, seed=11, noise=NOISE)


@pytest.fixture(scope="module")
def streams(flight):
    return flight[0]


@pytest.fixture(scope="module")
def truth(flight):
    return flight[1]


@pytest.fixture(scope="module")
def ref(truth):
    return make_ref(truth)


@pytest.fixture(scope="module")
def res(streams):
    """Independent filter run (deterministic) used as the oracle for the replay's ``ours`` block."""
    return run_filter(streams, CFG)


@pytest.fixture(scope="module")
def rep(streams, ref, res):
    return build_replay(streams, ref, CFG, log_meta=META, rate_hz=RATE)


# ---------------------------------------------------------------------------
# module constants and import hygiene
# ---------------------------------------------------------------------------
def test_constants_and_exceptions():
    assert CACHE_VERSION == 1
    assert px4_replay.DEFAULT_RATE_HZ == 50.0
    assert DOWNLOAD_HINT == "python -m sensor_fusion.download_logs"
    assert CAVEAT == "EKF2 is a reference, not truth"
    assert Path(px4_replay.DEFAULT_CACHE_DIR) == REPO / "webapp" / ".cache"
    assert issubclass(UnknownLog, LookupError)
    assert issubclass(LogMissing, FileNotFoundError)


def test_import_does_not_import_fastapi():
    code = ("import sys; import webapp.server.px4_replay; "
            "bad = [m for m in sys.modules if m == 'fastapi' or m.startswith('fastapi.')]; "
            "sys.exit(1 if bad else 0)")
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr


# ---------------------------------------------------------------------------
# 1. grid
# ---------------------------------------------------------------------------
def test_structure_and_shapes(rep, res, streams):
    assert set(rep) == TOP_KEYS
    assert rep["schema_version"] == 1
    assert rep["caveat"] == "EKF2 is a reference, not truth"
    assert rep["rate_hz"] == RATE
    assert rep["log"] == META
    assert set(rep["ours"]) == OURS_KEYS and set(rep["ekf2"]) == EKF2_KEYS and set(rep["diff"]) == DIFF_KEYS
    assert set(rep["events"]) == {"ekf2_resets_s", "skip_init_s"}
    n = rep["t"].size
    assert n == 1501
    for key, width in [("p_ned_m", 3), ("q", 4), ("euler_deg", 3), ("sigma3_att_deg", 3), ("sigma3_pos_m", 3)]:
        assert rep["ours"][key].shape == (n, width), key
        assert np.all(np.isfinite(rep["ours"][key])), key
    for key, width in [("p_ned_m", 3), ("q", 4), ("euler_deg", 3)]:
        assert rep["ekf2"][key].shape == (n, width), key
    for key in DIFF_KEYS:
        assert rep["diff"][key].shape == (n,), key
    f = rep["filter"]
    assert f["increment"] == CFG.increment
    assert f["n_imu"] == streams.imu_t.size
    assert f["n_updates"] == res.meta["n_updates"]
    assert f["runtime_s"] > 0.0


@pytest.mark.parametrize("rate", [50.0, 30.0, 4.3])
def test_grid(streams, ref, res, rate):
    r = build_replay(streams, ref, CFG, log_meta=META, rate_hz=rate)
    t = r["t"]
    assert t[0] == res.t[0]
    np.testing.assert_allclose(np.diff(t), 1.0 / rate, rtol=0, atol=1e-9)
    # t[-1] <= t1 < t[-1] + 1/rate; 1e-9 slack: the grid is arange(n)/rate, an ulp can exceed t1 for rates like 4.3
    assert t[-1] <= res.t[-1] + 1e-9
    assert res.t[-1] < t[-1] + 1.0 / rate
    assert r["rate_hz"] == rate


def test_build_runs_filter_exactly_once(streams, ref, monkeypatch):
    import sensor_fusion.ekf_drone as ed
    calls = []
    real = ed.run_filter

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(ed, "run_filter", counting)
    monkeypatch.setattr(px4_replay, "run_filter", counting, raising=False)
    build_replay(streams, ref, CFG, log_meta=META, rate_hz=10.0)
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# 2. ours == run_filter
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def k_imu(rep):
    """IMU sample index of each grid sample (100 Hz IMU, 50 Hz grid)."""
    return 2 * np.arange(rep["t"].size)


def test_ours_matches_run_filter(rep, res, k_imu):
    np.testing.assert_allclose(rep["t"], res.t[k_imu], rtol=0, atol=1e-12)
    ours = rep["ours"]
    np.testing.assert_allclose(ours["p_ned_m"], res.p[k_imu], rtol=0, atol=1e-9)
    assert qdist(ours["q"], res.q[k_imu]).max() < 1e-9
    # states: p = 0:3, th = 6:9 (STATE_NAMES in sensor_fusion.ekf_drone)
    np.testing.assert_allclose(ours["sigma3_pos_m"], 3.0 * res.std[k_imu][:, 0:3], rtol=0, atol=1e-9)
    np.testing.assert_allclose(ours["sigma3_att_deg"], 3.0 * np.rad2deg(res.std[k_imu][:, 6:9]), rtol=0, atol=1e-9)
    assert np.all(ours["sigma3_att_deg"] > 0) and np.all(ours["sigma3_pos_m"] > 0)


def test_ours_euler_matches_repo_and_scipy(rep, res, k_imu):
    e = rep["ours"]["euler_deg"]
    roll, pitch, yaw = quat.to_euler_zyx(rep["ours"]["q"])
    np.testing.assert_allclose(e, np.rad2deg(np.stack([roll, pitch, yaw], axis=-1)), rtol=0, atol=1e-9)
    assert np.abs(e[:, 1]).max() < 80.0, "test flight must stay away from gimbal lock"
    # scipy oracle on the independent filter quaternions; yaw compared modulo 360 deg
    sp = scipy_euler_deg(res.q[k_imu])
    assert np.abs(e[:, :2] - sp[:, :2]).max() < 1e-6
    assert np.abs(wrap_deg(e[:, 2] - sp[:, 2])).max() < 1e-6


def test_ours_interpolation_off_sample_times(streams, ref, res):
    """A grid that does not coincide with the 100 Hz samples: linear for p/std, slerp for q (scipy oracle)."""
    rate = 30.0
    r = build_replay(streams, ref, CFG, log_meta=META, rate_hz=rate)
    t = r["t"]
    assert np.abs(t * 100 - np.round(t * 100)).max() > 0.1      # genuinely off the IMU samples
    for j in range(3):
        np.testing.assert_allclose(r["ours"]["p_ned_m"][:, j], np.interp(t, res.t, res.p[:, j]), rtol=0, atol=1e-9)
        np.testing.assert_allclose(r["ours"]["sigma3_pos_m"][:, j], 3 * np.interp(t, res.t, res.std[:, j]),
                                   rtol=0, atol=1e-9)
        np.testing.assert_allclose(r["ours"]["sigma3_att_deg"][:, j],
                                   3 * np.rad2deg(np.interp(t, res.t, res.std[:, 6 + j])), rtol=0, atol=1e-9)
    slerp = Slerp(res.t, Rotation.from_quat(res.q[:, [1, 2, 3, 0]]))
    want = slerp(t).as_quat()[:, [3, 0, 1, 2]]
    # nlerp vs slerp over a 10 ms step differ by O(theta^3) ~ 1e-10; 1e-7 leaves room for either implementation
    assert qdist(r["ours"]["q"], want).max() < 1e-7
    np.testing.assert_allclose(np.linalg.norm(r["ours"]["q"], axis=1), 1.0, atol=1e-9)
    # EKF2 side: reference sampled at 100 Hz, linear interpolation of its positions
    for j in range(3):
        np.testing.assert_allclose(r["ekf2"]["p_ned_m"][:, j], np.interp(t, ref["t_pos"], ref["pos_ned"][:, j]),
                                   rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# 3. differences
# ---------------------------------------------------------------------------
def test_diff_equals_compare_to_ekf2(rep, res, ref, k_imu):
    arr = metrics.compare_to_ekf2(res, ref)["arr"]
    # the reference is sampled at 100 Hz inside [t0, t1], so arr[...][2k] is the grid sample k
    assert arr["tp"].size == res.t.size
    np.testing.assert_allclose(arr["tp"][k_imu], rep["t"], rtol=0, atol=1e-12)
    d = rep["diff"]
    np.testing.assert_allclose(d["pos_h_m"], arr["pos_h"][k_imu], rtol=0, atol=1e-6)
    np.testing.assert_allclose(d["pos_d_m"], arr["pos_d"][k_imu], rtol=0, atol=1e-6)
    np.testing.assert_allclose(d["tilt_deg"], np.rad2deg(arr["tilt"][k_imu]), rtol=0, atol=1e-6)
    np.testing.assert_allclose(d["yaw_deg"], np.rad2deg(arr["yaw"][k_imu]), rtol=0, atol=1e-6)
    assert np.all(np.isfinite(d["tilt_deg"])) and np.all(np.isfinite(d["pos_h_m"]))


def test_diff_matches_scipy_and_plain_numpy(rep, res, truth, k_imu):
    """Definitions re-derived from scratch: horizontal norm, signed down difference, tilt of the body
    down axes, wrapped yaw difference (all ours - EKF2, with the truth standing in for EKF2)."""
    d = rep["diff"]
    dp = res.p[k_imu] - truth.p[k_imu]
    np.testing.assert_allclose(d["pos_h_m"], np.hypot(dp[:, 0], dp[:, 1]), rtol=0, atol=1e-8)
    np.testing.assert_allclose(d["pos_d_m"], dp[:, 2], rtol=0, atol=1e-8)
    assert d["pos_h_m"].min() >= 0.0
    ra = Rotation.from_quat(res.q[k_imu][:, [1, 2, 3, 0]]).as_matrix()[:, :, 2]
    rb = Rotation.from_quat(truth.q[k_imu][:, [1, 2, 3, 0]]).as_matrix()[:, :, 2]
    tilt = np.degrees(np.arctan2(np.linalg.norm(np.cross(ra, rb), axis=1), np.sum(ra * rb, axis=1)))
    np.testing.assert_allclose(d["tilt_deg"], tilt, rtol=0, atol=1e-6)
    yaw = wrap_deg(scipy_euler_deg(res.q[k_imu])[:, 2] - scipy_euler_deg(truth.q[k_imu])[:, 2])
    assert np.abs(wrap_deg(d["yaw_deg"] - yaw)).max() < 1e-6
    # EKF2 (truth) block is the reference itself at the grid times
    np.testing.assert_allclose(rep["ekf2"]["p_ned_m"], truth.p[k_imu], rtol=0, atol=1e-9)
    assert qdist(rep["ekf2"]["q"], truth.q[k_imu]).max() < 1e-9
    np.testing.assert_allclose(rep["ekf2"]["euler_deg"][:, :2], scipy_euler_deg(truth.q[k_imu])[:, :2], atol=1e-6)
    assert np.abs(wrap_deg(rep["ekf2"]["euler_deg"][:, 2] - scipy_euler_deg(truth.q[k_imu])[:, 2])).max() < 1e-6


def test_diff_sign_conventions(streams, truth, rep):
    """Sanity of the signs with a deliberately offset reference: EKF2 0.5 m higher (smaller d) than the
    truth makes 'ours lower', i.e. pos_d_m larger by 0.5 relative to the truth reference."""
    ref_up = make_ref(truth)
    ref_up["pos_ned"] = truth.p + np.array([0.0, 0.0, -0.5])
    r = build_replay(streams, ref_up, CFG, log_meta=META, rate_hz=RATE)
    np.testing.assert_allclose(r["diff"]["pos_d_m"] - rep["diff"]["pos_d_m"], 0.5, atol=1e-9)
    np.testing.assert_allclose(r["diff"]["pos_h_m"], rep["diff"]["pos_h_m"], atol=1e-9)


# ---------------------------------------------------------------------------
# 4. coverage
# ---------------------------------------------------------------------------
def test_ekf2_nan_outside_reference_span(streams, truth):
    """Position reference only in [5, 25] s and attitude reference only in [8, 20] s: NaN exactly outside, separately."""
    ref = make_ref(truth, pos_span=(5.0, 25.0), att_span=(8.0, 20.0))
    r = build_replay(streams, ref, CFG, log_meta=META, rate_hz=RATE)
    t, eps = r["t"], 1e-9
    pos_out, pos_in = (t < 5.0 - eps) | (t > 25.0 + eps), (t > 5.0 + eps) & (t < 25.0 - eps)
    att_out, att_in = (t < 8.0 - eps) | (t > 20.0 + eps), (t > 8.0 + eps) & (t < 20.0 - eps)
    for key in ("p_ned_m",):
        nan_rows = np.isnan(r["ekf2"][key]).all(axis=1)
        assert nan_rows[pos_out].all()
        assert np.isfinite(r["ekf2"][key][pos_in]).all()
    for key in ("q", "euler_deg"):
        nan_rows = np.isnan(r["ekf2"][key]).all(axis=1)
        assert nan_rows[att_out].all(), key
        assert np.isfinite(r["ekf2"][key][att_in]).all(), key
    d = r["diff"]
    for key in ("pos_h_m", "pos_d_m"):
        assert np.isnan(d[key][pos_out]).all() and np.isfinite(d[key][pos_in]).all(), key
    for key in ("tilt_deg", "yaw_deg"):
        assert np.isnan(d[key][att_out]).all() and np.isfinite(d[key][att_in]).all(), key
    # ours is never NaN, and the values inside the span match the full-coverage run
    for v in r["ours"].values():
        assert np.isfinite(v).all()


def test_coverage_values_inside_span_match_full_reference(streams, truth, rep):
    ref = make_ref(truth, pos_span=(5.0, 25.0), att_span=(5.0, 25.0))
    r = build_replay(streams, ref, CFG, log_meta=META, rate_hz=RATE)
    inside = (r["t"] > 5.0 + 1e-9) & (r["t"] < 25.0 - 1e-9)
    for key in DIFF_KEYS:
        np.testing.assert_allclose(r["diff"][key][inside], rep["diff"][key][inside], rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# 5. NIS
# ---------------------------------------------------------------------------
def expected_nis(res, rate_hz):
    """NIS block re-derived from ``res.innovations`` as the spec words it."""
    span = res.t[-1] - res.t[0]
    out = {}
    for name in metrics.SENSOR_ORDER:
        il = res.innovations.get(name)
        if il is None or il.t.size == 0:
            continue
        m = np.isfinite(il.t) & np.isfinite(il.nis)
        t, nis, acc, dof = il.t[m], il.nis[m], il.accepted[m], il.dof[m]
        if t.size > rate_hz * span:
            stride = math.ceil(t.size / (rate_hz * span))
            t, nis, acc, dof = t[::stride], nis[::stride], acc[::stride], dof[::stride]
        out[name] = (t, nis, acc, dof)
    return out


def test_nis_entries_match_innovations(rep, res):
    want = expected_nis(res, RATE)
    assert set(want) == {"gps_pos", "gps_vel", "baro", "mag_heading"}
    assert set(rep["nis"]) == set(want)
    for name, (t, nis, acc, dof) in want.items():
        e = rep["nis"][name]
        assert set(e) == NIS_KEYS
        np.testing.assert_array_equal(e["t"], t)
        np.testing.assert_array_equal(e["nis"], nis)
        np.testing.assert_array_equal(np.asarray(e["accepted"]), acc)
        assert np.asarray(e["accepted"]).dtype == bool
        d = np.bincount(dof).argmax()
        assert int(e["dof"]) == d
        np.testing.assert_allclose(e["band95"], chi2.ppf([0.025, 0.975], d), rtol=1e-12)
    assert {n: int(rep["nis"][n]["dof"]) for n in rep["nis"]} == {"gps_pos": 3, "gps_vel": 3, "baro": 1,
                                                                   "mag_heading": 1}
    # the gate rejects some updates, so the accepted flag is exercised in both states
    assert any(not np.all(rep["nis"][n]["accepted"]) for n in rep["nis"])
    assert any(np.any(rep["nis"][n]["accepted"]) for n in rep["nis"])


def test_nis_decimation(streams, ref, res):
    rate = 0.5                      # budget 0.5 Hz * 30 s = 15 samples per sensor -> strides of 11 (gps) / 21 (baro, mag)
    r = build_replay(streams, ref, CFG, log_meta=META, rate_hz=rate)
    want = expected_nis(res, rate)
    assert set(r["nis"]) == set(want)
    for name, (t, nis, acc, dof) in want.items():
        e = r["nis"][name]
        np.testing.assert_array_equal(e["t"], t)
        np.testing.assert_array_equal(e["nis"], nis)
        np.testing.assert_array_equal(np.asarray(e["accepted"]), acc)
    # explicit hand-computed expectation for baro: 301 samples, budget 15 -> stride 21, indices 0, 21, 42, ...
    il = res.innovations["baro"]
    assert il.t.size == 301
    np.testing.assert_array_equal(r["nis"]["baro"]["t"], il.t[::21])
    assert len(r["nis"]["baro"]["t"]) == math.ceil(301 / 21)
    assert len(r["nis"]["gps_pos"]["t"]) == math.ceil(151 / 11)


# ---------------------------------------------------------------------------
# 6. events
# ---------------------------------------------------------------------------
def test_events_filter_and_sort(streams, truth):
    ref = make_ref(truth, resets=[28.0, 45.0, 12.0, -2.0, 3.5, 100.0])
    r = build_replay(streams, ref, CFG, log_meta=META, rate_hz=RATE)
    assert list(r["events"]["ekf2_resets_s"]) == [3.5, 12.0, 28.0]
    assert r["events"]["skip_init_s"] == 5.0 == metrics.SKIP_INIT_S


def test_events_interval_is_closed_and_empty_ok(streams, truth):
    r = build_replay(streams, make_ref(truth, resets=[30.0, 0.0, 30.0001, -0.0001]), CFG, log_meta=META, rate_hz=RATE)
    assert list(r["events"]["ekf2_resets_s"]) == [0.0, 30.0]
    r = build_replay(streams, make_ref(truth), CFG, log_meta=META, rate_hz=RATE)
    assert list(r["events"]["ekf2_resets_s"]) == []


# ---------------------------------------------------------------------------
# 7. JSON
# ---------------------------------------------------------------------------
def sig_digits(literal: str) -> int:
    mant = literal.lower().split("e")[0].lstrip("+-").replace(".", "").strip("0")
    return len(mant)


def test_json_structure_and_values(rep):
    s = replay_to_json(rep)
    assert isinstance(s, str)
    bad = []
    p = json.loads(s, parse_float=lambda x: (bad.append(x) if sig_digits(x) > 7 else None) or float(x))
    assert not bad, f"numbers with more than 7 significant digits: {bad[:5]}"
    json.dumps(p, allow_nan=False)
    assert set(p) == TOP_KEYS
    assert p["schema_version"] == 1 and p["caveat"] == "EKF2 is a reference, not truth"
    assert p["rate_hz"] == RATE and p["log"] == META
    assert p["filter"]["increment"] == CFG.increment and p["filter"]["n_imu"] == rep["filter"]["n_imu"]
    assert isinstance(p["filter"]["n_imu"], int) and not isinstance(p["filter"]["n_imu"], bool)
    assert p["filter"]["n_updates"] == rep["filter"]["n_updates"]
    assert_json_close(p["filter"]["runtime_s"], rep["filter"]["runtime_s"])
    n = rep["t"].size
    assert len(p["t"]) == n
    assert_json_close(p["t"], rep["t"])
    for grp in ("ours", "ekf2"):
        assert set(p[grp]) == set(rep[grp])
        for k, arr in rep[grp].items():
            assert isinstance(p[grp][k], list) and len(p[grp][k]) == n
            assert isinstance(p[grp][k][0], list) and len(p[grp][k][0]) == arr.shape[1]     # (n, k) -> list of lists
            assert_json_close(p[grp][k], arr)
    for k, arr in rep["diff"].items():
        assert_json_close(p["diff"][k], arr)
    assert set(p["nis"]) == set(rep["nis"])
    for name, e in rep["nis"].items():
        pe = p["nis"][name]
        assert set(pe) == NIS_KEYS
        assert_json_close(pe["t"], e["t"])
        assert_json_close(pe["nis"], e["nis"])
        assert_json_close(pe["band95"], e["band95"])
        assert pe["accepted"] == [bool(a) for a in e["accepted"]]
        assert all(type(a) is bool for a in pe["accepted"])
        assert pe["dof"] == int(e["dof"]) and type(pe["dof"]) is int
    assert p["events"] == {"ekf2_resets_s": list(rep["events"]["ekf2_resets_s"]), "skip_init_s": 5.0}


def test_json_nan_and_inf_become_null(streams, truth):
    ref = make_ref(truth, pos_span=(5.0, 25.0), att_span=(8.0, 20.0))
    r = build_replay(streams, ref, CFG, log_meta=META, rate_hz=RATE)
    assert np.isnan(r["ekf2"]["p_ned_m"]).any()
    r["ours"]["p_ned_m"][3, 0] = np.inf
    r["ours"]["p_ned_m"][4, 1] = -np.inf
    r["ours"]["p_ned_m"][5, 2] = np.nan
    s = replay_to_json(r)
    assert "NaN" not in s and "Infinity" not in s
    p = json.loads(s)
    json.dumps(p, allow_nan=False)
    assert p["ours"]["p_ned_m"][3][0] is None and p["ours"]["p_ned_m"][4][1] is None
    assert p["ours"]["p_ned_m"][5][2] is None
    assert p["ours"]["p_ned_m"][3][1] is not None
    got = to_array(p["ekf2"]["p_ned_m"])
    want = r["ekf2"]["p_ned_m"]
    assert np.array_equal(np.isnan(got), np.isnan(want))
    assert p["ekf2"]["q"][0] == [None] * 4                         # t = 0 s lies outside the [8, 20] s attitude span
    assert np.array_equal(np.isnan(to_array(p["diff"]["tilt_deg"])), np.isnan(r["diff"]["tilt_deg"]))


def fake_replay(n, rng):
    """A spec-shaped replay dict with random content (no filter involved) for the speed test."""
    def a(*shape):
        x = rng.normal(size=shape) * 10.0 ** rng.integers(-6, 6, size=shape)
        x[rng.random(shape) < 0.05] = np.nan
        return x
    m = 3000
    return {
        "schema_version": 1, "caveat": CAVEAT, "rate_hz": 50.0, "log": dict(META),
        "filter": {"increment": 3, "runtime_s": 12.3456789, "n_imu": 150000, "n_updates": {"gps_pos": 1}},
        "t": np.arange(n) / 50.0,
        "ours": {"p_ned_m": a(n, 3), "q": a(n, 4), "euler_deg": a(n, 3), "sigma3_att_deg": a(n, 3),
                 "sigma3_pos_m": a(n, 3)},
        "ekf2": {"p_ned_m": a(n, 3), "q": a(n, 4), "euler_deg": a(n, 3)},
        "diff": {"pos_h_m": a(n), "pos_d_m": a(n), "tilt_deg": a(n), "yaw_deg": a(n)},
        "nis": {name: {"t": np.sort(rng.random(m)) * n / 50.0, "nis": a(m), "accepted": rng.random(m) < 0.9,
                       "dof": 3, "band95": chi2.ppf([0.025, 0.975], 3)} for name in metrics.SENSOR_ORDER},
        "events": {"ekf2_resets_s": [12.5, 99.0], "skip_init_s": 5.0},
    }


def test_json_speed_34k_samples():
    """Target from the spec: < 2 s for the 675 s log (34 k grid samples). Best of two runs."""
    big = fake_replay(34000, np.random.default_rng(1234))
    best, s = np.inf, None
    for _ in range(2):
        t0 = time.perf_counter()
        s = replay_to_json(big)
        best = min(best, time.perf_counter() - t0)
    assert best < 2.0, f"replay_to_json took {best:.2f} s"
    p = json.loads(s)
    json.dumps(p, allow_nan=False)
    assert len(p["t"]) == 34000 and len(p["ours"]["q"][0]) == 4
    assert_json_close(p["ours"]["p_ned_m"], big["ours"]["p_ned_m"])
    assert_json_close(p["diff"]["tilt_deg"], big["diff"]["tilt_deg"])
    assert_json_close(p["nis"]["baro"]["nis"], big["nis"]["baro"]["nis"])
    assert pe_bool_ok(p["nis"]["baro"]["accepted"], big["nis"]["baro"]["accepted"])


def pe_bool_ok(got, want):
    return got == [bool(x) for x in want] and all(type(x) is bool for x in got)


# ---------------------------------------------------------------------------
# 8. store
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def short_ctx():
    return make_streams_ctx(duration=10.0)


@pytest.fixture()
def loader(short_ctx):
    return Loader(*short_ctx)


def test_list_logs_present_and_cached(tmp_path, loader):
    store = make_store(tmp_path, loader)
    logs = store.list_logs()
    assert [(e["id"], e["role"]) for e in logs] == [(e["id"], e["role"]) for e in PINNED_LOGS]
    for e, pinned in zip(logs, PINNED_LOGS):
        assert set(e) == {"id", "role", "notes", "present", "cached"}
        assert e["notes"] == pinned["notes"]
    by = {e["role"]: e for e in store.list_logs()}
    assert by["hover"]["present"] and by["aggressive"]["present"] and not by["other"]["present"]
    assert not any(e["cached"] for e in by.values())
    store.get_json("hover")
    by = {e["role"]: e for e in store.list_logs()}
    assert by["hover"]["cached"] is True
    assert by["aggressive"]["cached"] is False and by["other"]["cached"] is False
    assert all(type(e["present"]) is bool and type(e["cached"]) is bool for e in by.values())
    # a fresh store on the same cache sees the same state
    store2 = make_store(tmp_path, Loader(*loader_ctx(loader)))
    assert {e["role"]: e["cached"] for e in store2.list_logs()} == {"hover": True, "aggressive": False, "other": False}


def loader_ctx(loader):
    return loader.streams, loader.ref


def test_path(tmp_path, loader):
    store = make_store(tmp_path, loader)
    e = entry_of("hover")
    assert Path(store.path(e)) == tmp_path / "data" / f"{e['id']}.ulg"


def test_resolve_role_id_prefix(tmp_path, loader):
    store = make_store(tmp_path, loader)
    for e in PINNED_LOGS:
        assert store.resolve(e["role"])["id"] == e["id"]
        assert store.resolve(e["id"])["id"] == e["id"]
        assert store.resolve(e["id"][:4])["id"] == e["id"]
        assert store.resolve(e["id"][:8])["id"] == e["id"]
        assert store.resolve(e["id"][:13])["id"] == e["id"]
    assert store.resolve("hover")["role"] == "hover"


@pytest.mark.parametrize("token", ["all", "", "nope", "f4e", "zzzzzz", "hoverx", "f4e8x"])
def test_resolve_rejects(tmp_path, loader, token):
    store = make_store(tmp_path, loader)
    with pytest.raises(UnknownLog):
        store.resolve(token)


def test_resolve_ambiguous_prefix(tmp_path, loader):
    logs = ({"id": "aaaa1111-0000-0000-0000-000000000001", "role": "one", "notes": "a"},
            {"id": "aaaa2222-0000-0000-0000-000000000002", "role": "two", "notes": "b"})
    store = ReplayStore(cache_dir=tmp_path / "c", logs=logs, data_dir=tmp_path, context_loader=loader,
                        cfg_factory=make_cfg)
    for tok in ("aaaa", "aaa"):
        with pytest.raises(UnknownLog):
            store.resolve(tok)
    assert store.resolve("aaaa1")["role"] == "one"
    assert store.resolve("aaaa2222")["role"] == "two"


def test_missing_file_raises_log_missing(tmp_path, loader):
    store = make_store(tmp_path, loader)
    with pytest.raises(LogMissing) as ei:
        store.get_json("other")
    msg = str(ei.value)
    e = entry_of("other")
    assert "python -m sensor_fusion.download_logs" in msg
    assert e["id"] in msg and "other" in msg and "not downloaded" in msg
    assert isinstance(ei.value, FileNotFoundError)
    assert loader.calls == [] and store.n_computed == 0
    with pytest.raises(UnknownLog):
        store.get_json("all")
    assert not (tmp_path / "cache").exists(), "nothing is written for a missing log"


def test_cache_hit_and_files(tmp_path, loader):
    store = make_store(tmp_path, loader)
    cache = tmp_path / "cache" / "nested"
    assert not cache.exists()
    s1 = store.get_json("hover")
    assert cache.is_dir()
    s2 = store.get_json("hover")
    s3 = store.get_json(entry_of("hover")["id"][:6])
    assert s1 == s2 == s3
    assert loader.calls == [entry_of("hover")["id"]] and store.n_computed == 1
    # the cache key is re-derived from the spec: first 16 hex of sha256 over the sorted JSON of the key fields
    e = entry_of("hover")
    st = (tmp_path / "data" / f"{e['id']}.ulg").stat()
    key = hashlib.sha256(json.dumps(
        {"cache_version": 1, "id": e["id"], "rate_hz": 50.0, "config": dataclasses.asdict(make_cfg()),
         "ulg_size": st.st_size, "ulg_mtime_ns": st.st_mtime_ns}, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    jf, nf = cache / f"px4_{e['id'][:8]}_{key}.json", cache / f"px4_{e['id'][:8]}_{key}.npz"
    assert jf.is_file() and nf.is_file()
    assert jf.read_text(encoding="utf-8") == s1
    assert sorted(p.name for p in cache.iterdir()) == sorted([jf.name, nf.name]), "no temporary files left behind"
    with np.load(nf) as z:
        assert len(z.files) > 0
    # content of the replay itself
    p = json.loads(s1)
    assert set(p) == TOP_KEYS
    assert p["rate_hz"] == 50.0 and p["filter"]["increment"] == 3
    meta = p["log"]
    assert meta["id"] == e["id"] and meta["role"] == "hover" and meta["notes"] == e["notes"]
    assert meta["duration_s"] == pytest.approx(10.0, rel=1e-6)
    assert meta["origin_lat_deg"] == pytest.approx(47.4, rel=1e-6)
    assert meta["origin_lon_deg"] == pytest.approx(8.5, rel=1e-6)
    assert meta["gps_utc_start"] == "2024-05-01 12:00:00+00:00"
    assert meta["declination_deg"] == pytest.approx(math.degrees(0.1), rel=1e-6)
    assert meta["g_m_s2"] == pytest.approx(9.80665, rel=1e-6)
    assert set(meta) == {"id", "role", "notes", "duration_s", "origin_lat_deg", "origin_lon_deg",
                         "gps_utc_start", "declination_deg", "g_m_s2"}


def test_cache_miss_without_json_recomputes(tmp_path, loader):
    store = make_store(tmp_path, loader)
    store.get_json("hover")
    cache = tmp_path / "cache" / "nested"
    for f in cache.glob("*.json"):
        f.unlink()
    assert store.list_logs()[0]["cached"] is False       # the .npz alone is not a hit
    store.get_json("hover")
    assert store.n_computed == 2 and len(loader.calls) == 2


def test_new_store_same_cache_dir_hits(tmp_path, loader, short_ctx):
    s1 = make_store(tmp_path, loader).get_json("hover")
    loader2 = Loader(*short_ctx)
    store2 = make_store(tmp_path, loader2)
    assert store2.get_json("hover") == s1
    assert loader2.calls == [] and store2.n_computed == 0


def test_config_change_recomputes(tmp_path, loader, short_ctx):
    s1 = make_store(tmp_path, loader).get_json("hover")
    loader2 = Loader(*short_ctx)
    store2 = make_store(tmp_path, loader2,
                        cfg_factory=lambda: EkfConfig(increment=3, gps_delay_s=0.0, baro_sigma=2.0))
    s2 = store2.get_json("hover")
    assert loader2.calls == [entry_of("hover")["id"]] and store2.n_computed == 1
    assert s2 != s1
    assert len(list((tmp_path / "cache" / "nested").glob("px4_*.json"))) == 2
    # and the new entry is itself cached
    store2.get_json("hover")
    assert store2.n_computed == 1


def test_rate_change_recomputes(tmp_path, loader, short_ctx):
    make_store(tmp_path, loader).get_json("hover")
    loader2 = Loader(*short_ctx)
    store2 = make_store(tmp_path, loader2, rate_hz=25.0)
    p = json.loads(store2.get_json("hover"))
    assert p["rate_hz"] == 25.0 and store2.n_computed == 1
    assert len(p["t"]) == int(math.floor(10 * 25 + 1e-9)) + 1
    assert abs((p["t"][1] - p["t"][0]) - 1 / 25.0) < 1e-6


def test_ulg_size_change_recomputes(tmp_path, loader):
    store = make_store(tmp_path, loader)
    store.get_json("hover")
    store.get_json("hover")
    assert store.n_computed == 1
    f = tmp_path / "data" / f"{entry_of('hover')['id']}.ulg"
    f.write_bytes(b"ULog\x01\x12\x35" + bytes(5000))               # different size
    assert store.list_logs()[0]["cached"] is False
    store.get_json("hover")
    assert store.n_computed == 2 and len(loader.calls) == 2
    store.get_json("hover")
    assert store.n_computed == 2


def test_ulg_mtime_change_recomputes(tmp_path, loader):
    store = make_store(tmp_path, loader)
    store.get_json("hover")
    f = tmp_path / "data" / f"{entry_of('hover')['id']}.ulg"
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))      # same size, mtime +5 s
    store.get_json("hover")
    assert store.n_computed == 2


def test_concurrent_get_json_builds_once(tmp_path, short_ctx):
    loader = Loader(*short_ctx, delay=0.4)
    store = make_store(tmp_path, loader)
    barrier = threading.Barrier(2)
    out, errs = {}, []

    def worker(i):
        try:
            barrier.wait(timeout=10)
            out[i] = store.get_json("hover")
        except Exception as e:                          # noqa: BLE001
            errs.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=60)
    assert not errs and not any(th.is_alive() for th in threads)
    assert len(loader.calls) == 1 and store.n_computed == 1
    assert out[0] == out[1]
    json.loads(out[0])


def test_loader_exception_propagates_and_nothing_cached(tmp_path, short_ctx):
    loader = Loader(*short_ctx, fail=RuntimeError("boom"))
    store = make_store(tmp_path, loader)
    with pytest.raises(RuntimeError, match="boom"):
        store.get_json("hover")
    assert store.list_logs()[0]["cached"] is False
    assert not list((tmp_path / "cache").rglob("*.json"))
    loader.fail = None
    store.get_json("hover")                             # recovers (the per-id lock was released)
    assert store.list_logs()[0]["cached"] is True


# ---------------------------------------------------------------------------
# 9. API
# ---------------------------------------------------------------------------
@pytest.fixture()
def api(tmp_path, loader):
    store = make_store(tmp_path, loader)
    app = create_app(frontend_dist=None, px4_store=store)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c, store, loader


def test_app_state_holds_store(tmp_path, loader):
    store = make_store(tmp_path, loader)
    assert create_app(frontend_dist=None, px4_store=store).state.px4 is store


def test_api_logs(api):
    c, store, _ = api
    r = c.get("/api/px4/logs")
    assert r.status_code == 200
    b = r.json()
    assert set(b) == {"logs", "download_hint", "caveat"}
    assert b["download_hint"] == "python -m sensor_fusion.download_logs"
    assert b["caveat"] == "EKF2 is a reference, not truth"
    assert b["logs"] == store.list_logs()
    assert [e["role"] for e in b["logs"]] == ["hover", "aggressive", "other"]
    assert [e["present"] for e in b["logs"]] == [True, True, False]
    assert not any(e["cached"] for e in b["logs"])
    c.post("/api/px4/replay/hover")
    assert [e["cached"] for e in c.get("/api/px4/logs").json()["logs"]] == [True, False, False]


def test_api_replay_ok(api):
    c, store, loader = api
    r = c.post("/api/px4/replay/hover")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    b = r.json()
    assert set(b) == TOP_KEYS
    assert b["caveat"] == CAVEAT and b["log"]["role"] == "hover"
    assert r.text == store.get_json("hover")                   # body is the cached text, verbatim
    assert len(loader.calls) == 1
    # by id prefix, and a second request is a cache hit
    r2 = c.post(f"/api/px4/replay/{entry_of('hover')['id'][:6]}")
    assert r2.status_code == 200 and r2.text == r.text and len(loader.calls) == 1


def test_api_unknown_log(api):
    c, _, loader = api
    r = c.post("/api/px4/replay/nope")
    assert r.status_code == 404
    assert r.json() == {"detail": "unknown log 'nope'"}
    r = c.post("/api/px4/replay/all")
    assert r.status_code == 404 and r.json() == {"detail": "unknown log 'all'"}
    assert loader.calls == []


def test_api_missing_log(api):
    c, _, loader = api
    r = c.post("/api/px4/replay/other")
    assert r.status_code == 404
    d = r.json()["detail"]
    assert "python -m sensor_fusion.download_logs" in d and entry_of("other")["id"] in d
    assert loader.calls == []


def test_api_replay_failure_is_500(tmp_path, short_ctx):
    loader = Loader(*short_ctx, fail=RuntimeError("boom"))
    app = create_app(frontend_dist=None, px4_store=make_store(tmp_path, loader))
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/api/px4/replay/hover")
        assert r.status_code == 500
        assert r.json() == {"detail": "replay failed: boom"}
        assert c.get("/api/health").json() == {"ok": True}          # the server survives


def test_api_gzip(api):
    c, _, _ = api
    r = c.post("/api/px4/replay/hover", headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200
    assert r.headers.get("content-encoding") == "gzip"
    assert set(r.json()) == TOP_KEYS                                # httpx decoded it transparently
    r = c.post("/api/px4/replay/hover", headers={"Accept-Encoding": "identity"})
    assert r.status_code == 200 and "content-encoding" not in r.headers
    assert set(r.json()) == TOP_KEYS


def test_api_other_endpoints_still_work(api):
    c, _, _ = api
    assert c.get("/api/health").json() == {"ok": True}
    r = c.get("/api/adcs/presets")
    assert r.status_code == 200 and "defaults" in r.json()


def test_api_build_does_not_block_other_requests(tmp_path, short_ctx):
    """get_json runs in a worker thread: /api/health answers while a slow build is in progress."""
    loader = Loader(*short_ctx, delay=2.0)
    app = create_app(frontend_dist=None, px4_store=make_store(tmp_path, loader))
    result = {}
    with TestClient(app, raise_server_exceptions=False) as c:
        def slow():
            result["r"] = c.post("/api/px4/replay/hover")
        th = threading.Thread(target=slow)
        th.start()
        deadline = time.time() + 10
        while not loader.calls and time.time() < deadline:
            time.sleep(0.01)
        assert loader.calls, "build did not start"
        t0 = time.perf_counter()
        h = c.get("/api/health")
        elapsed = time.perf_counter() - t0
        assert h.status_code == 200
        assert th.is_alive(), "the slow build finished before the probe request"
        assert elapsed < 1.0, f"/api/health blocked for {elapsed:.2f} s during a build"
        th.join(timeout=60)
    assert result["r"].status_code == 200


# ---------------------------------------------------------------------------
# 10. real log (optional)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(os.environ.get("PX4_REPLAY_REAL") != "1", reason="set PX4_REPLAY_REAL=1 to replay the real hover log")
def test_real_hover_log_tilt_rms(tmp_path):
    from sensor_fusion import run_flight
    from sensor_fusion.download_logs import DATA_DIR
    e = entry_of("hover")
    if not (Path(DATA_DIR) / f"{e['id']}.ulg").is_file():
        pytest.skip("hover .ulg not downloaded")
    p = json.loads(ReplayStore(cache_dir=tmp_path / "cache").get_json("hover"))
    t = np.array(p["t"])
    tilt = to_array(p["diff"]["tilt_deg"])
    resets = np.array(p["events"]["ekf2_resets_s"], float)
    keep = (t >= t[0] + 5.0) & np.isfinite(tilt)
    for tr in resets:
        keep &= np.abs(t - tr) > 1.0
    rms_replay = float(np.sqrt(np.mean(tilt[keep] ** 2)))

    ctx = run_flight.load_context(e)
    res = run_filter(ctx["S"], run_flight.build_config(3))
    rms_ref = metrics.compare_to_ekf2(res, ctx["ref"])["cmp"]["tilt_deg"]["rms"]
    print(f"\nREAL-LOG tilt RMS: replay grid = {rms_replay:.5f} deg, compare_to_ekf2 = {rms_ref:.5f} deg "
          f"(ratio {rms_replay / rms_ref:.4f}); {keep.sum()} grid samples, {resets.size} resets")
    assert t.size > 1000
    assert rms_replay == pytest.approx(rms_ref, rel=0.10)
    assert p["caveat"] == CAVEAT and p["log"]["role"] == "hover"
