"""Tests for the step-by-step closed-loop engine ``adcs.stepper.ClosedLoopSim``.

Written from ``docs/specs/stepper.md`` (sections 0-5), not from the implementation.

Oracles
-------
* golden equivalence: ``adcs/tests/data/closed_loop_golden.npz`` was captured from the unmodified
  pre-refactor ``run_closed_loop`` (loaded with ``allow_pickle=False``); both the wrapper and a
  runner that drives ``ClosedLoopSim`` in irregular chunks must reproduce it byte for byte
  (``golden_closed_loop.compare``);
* snapshot = logged row: the logger of the same run, bit for bit; gmst from ``common.frames.gmst``;
  the LVLH reference attitude from the CONVENTIONS.md LVLH definition + scipy ``Rotation``; the
  magnetorquer torque ``m x R(q)^T b`` with scipy; the knowledge error ``q = q_est (x) exp(e)``
  with scipy;
* free-rotation integration: ``scipy.integrate.solve_ivp`` (DOP853, tight tolerance) of Euler's
  equations with a wheel, ``I w' = -w x (I w + h)``, ``h' = 0``, and ``q' = 1/2 q (x) [0, w]``;
* kick: exact floating-point algebra of the spec (``y[4:7] + dw``), the angular momentum
  ``H = R(q)(I w + h_w)`` with scipy ``Rotation``, ``wheel_speed`` of ``adcs.actuators``;
* mode machine: thresholds of ``ModeConfig`` (``safe_hold``, ``safe_rate``) of the spec.

The golden runs (14 runs, mostly 2 orbits each) execute in parallel worker processes (BLAS threads
limited to 1, as when the golden file was captured); everything else is short. The golden file is
bit-exact only on its capture platform: the equivalence tests skip, with an explicit reason, when
``golden_closed_loop.environment_matches`` reports a different interpreter / library / machine.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as Rot

from adcs.actuators import MagnetorquerConfig, WheelConfig, wheel_speed
from adcs.config import DEFAULT, OrbitConfig, SimConfig
from adcs.orbit import initial_state, orbital_period
from adcs.sensors import GyroConfig, MagnetometerConfig, SunSensorConfig
from adcs.simulate import (
    IDEAL_SENSORS,
    MEKF_DESIGN,
    MEKF_G7,
    MODE_DETUMBLE,
    MODE_POINTING,
    REALISTIC_SENSORS,
    ControlConfig,
    ModeConfig,
    lvlh_state,
    run_closed_loop,
    tumble_state,
)
import adcs.stepper
from adcs.stepper import MAX_KICK_RATE, ClosedLoopSim  # no skip: a missing module must fail loudly
from adcs.tests import golden_closed_loop as gcl
from adcs.tests.golden_closed_loop import CASES, compare, flatten, run_case
from common.frames import gmst

R0, V0 = initial_state(DEFAULT)
P_ORBIT = float(orbital_period(DEFAULT.orbit.a))
I_NOM = DEFAULT.spacecraft.inertia
GOLDEN_PATH = gcl.DEFAULT_OUT

# vector / scalar fields of ClosedLoopResult that have a snapshot counterpart of the same name
ROW_VEC = ("q", "w", "h_w", "wheel_speed", "m", "u_w", "tau_dist", "tau_mag",
           "q_est", "b_est", "bm_est", "know_err", "w_est")
ROW_SCA = ("t", "nadir_err", "rate", "att_sigma", "est_nadir_err")
ROW_INT = ("mode", "sun_used")
ROW_BOOL = ("dump", "lit")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _bits_equal(a, b) -> bool:
    """Identical bytes in every non-NaN entry and NaN in exactly the same places."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return False
    na, nb = np.isnan(a), np.isnan(b)
    return bool(np.array_equal(na, nb) and a[~na].tobytes() == b[~nb].tobytes())


def _same_result(ra, rb) -> bool:
    """Every array and every meta leaf of two results has the same bytes (kicks ignored)."""
    fa, fb = flatten(ra), flatten(rb)
    return set(fa) == set(fb) and all(
        fa[k].dtype == fb[k].dtype and fa[k].shape == fb[k].shape and fa[k].tobytes() == fb[k].tobytes()
        for k in fa)


class _Golden:
    """Minimal stand-in for an opened npz (``.files`` and ``[key]``) built from a flatten dict."""

    def __init__(self, case, flat):
        self._d = {f"{case}/{k}": v for k, v in flat.items()}
        self.files = list(self._d)

    def __getitem__(self, key):
        return self._d[key]


def _y0(rate=0.05, seed=1, h_w=None):
    y = tumble_state(rate, seed)
    if h_w is not None:
        y[7:10] = h_w
    return y


def _short(t_end=30.0, y0=None, **kw):
    return ClosedLoopSim(_y0() if y0 is None else y0, t_end, **kw)


def _hamilton_rate(q, w):
    """q_dot = 1/2 q (x) [0, w] written out (scalar-first Hamilton)."""
    w0, x, y, z = q
    wx, wy, wz = w
    return 0.5 * np.array([-x * wx - y * wy - z * wz,
                           w0 * wx + y * wz - z * wy,
                           w0 * wy - x * wz + z * wx,
                           w0 * wz + x * wy - y * wx])


def _free_rotation(y0, inertia, tspan):
    """Truth of ``I w' = -w x (I w + h)``, ``h' = 0``, ``q' = 1/2 q (x) [0, w]`` (scipy DOP853)."""
    inv = np.linalg.inv(inertia)

    def f(_t, y):
        q, w, h = y[0:4], y[4:7], y[7:10]
        return np.concatenate([_hamilton_rate(q, w), inv @ (-np.cross(w, inertia @ w + h)), np.zeros(3)])

    sol = solve_ivp(f, (0.0, tspan), y0, method="DOP853", rtol=1e-12, atol=1e-14)
    return sol.y[:, -1]


def _angular_momentum_inertial(y, inertia):
    """R(q) (I w + h_w) with scipy Rotation (scalar-first -> scipy's [x, y, z, w])."""
    q = y[0:4]
    rot = Rot.from_quat([q[1], q[2], q[3], q[0]])
    return rot.apply(inertia @ y[4:7] + y[7:10])


# --------------------------------------------------------------------------
# 1. golden equivalence (pre-refactor bytes)
# --------------------------------------------------------------------------
def _chunked_runner(y0, t_end, **kw):
    """Drive ClosedLoopSim in irregular chunks (1, 7, 333, 1000, 1, 7, ...) until done."""
    sim = ClosedLoopSim(y0, t_end, **kw)
    for n in itertools.cycle((1, 7, 333, 1000)):
        if sim.done:
            break
        sim.step(n)
    return sim.result()


def _golden_job(name, kind):
    return run_case(name, None if kind == "wrapper" else _chunked_runner)


@pytest.fixture(scope="module")
def golden():
    g = np.load(GOLDEN_PATH, allow_pickle=False)
    try:
        yield g
    finally:
        g.close()


@pytest.fixture(scope="module")
def golden_env(golden):
    """Skip (with the reason) when the golden file was captured on a different platform / library stack.

    Declared before ``golden_runs`` depends on it, so no worker process is started on a mismatch.
    """
    ok, reason = gcl.environment_matches(golden)
    if not ok:
        pytest.skip(reason)
    return golden


@pytest.fixture(scope="module")
def golden_runs(golden_env):
    """(case, kind) -> Future of the case result; kind is 'wrapper' or 'chunked'."""
    keys = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    old = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ[k] = "1"
    ex = ProcessPoolExecutor(max_workers=min(14, os.cpu_count() or 1))
    try:
        # the 2-orbit cases are the slow ones: submit them first
        order = sorted(CASES, key=lambda n: -CASES[n]["t_end"])
        yield {(n, kind): ex.submit(_golden_job, n, kind) for n in order for kind in ("wrapper", "chunked")}
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_golden_file_covers_all_cases(golden):
    cases_in_file = {k.split("/", 1)[0] for k in golden.files if "/" in k}
    assert cases_in_file == set(CASES)
    assert "lvlh_wheels_fallback" in CASES and len(CASES) == 7


def test_environment_matches_contract(golden):
    """environment_matches(golden) -> (bool, reason); a changed library version is reported as a mismatch."""
    ok, reason = gcl.environment_matches(golden)
    assert isinstance(ok, (bool, np.bool_)) and isinstance(reason, str)
    info_txt = str(golden["info_json"])
    assert np.__version__ in info_txt  # the file records the numpy version it was captured with
    tampered = {"info_json": np.asarray(info_txt.replace(np.__version__, "0.0.0"))}

    class _Fake:
        files = list(tampered)

        def __getitem__(self, key):
            return tampered[key]

    bad_ok, bad_reason = gcl.environment_matches(_Fake())
    assert not bad_ok and isinstance(bad_reason, str) and bad_reason


def test_golden_lvlh_wheels_fallback_covers_fallback_and_dumping(golden):
    """Coverage claim of spec 5.11, checked in the stored (pre-refactor) bytes: this case reaches the
    fallback and the dumping branches that the other cases never exercise."""
    case = "lvlh_wheels_fallback"
    meta = json.loads(str(golden[f"{case}/meta_json"]))
    assert len(meta["fallback_times"]) >= 1
    assert len(meta["switch_times"]) >= 1
    dump = golden[f"{case}/res/dump"]
    mode = golden[f"{case}/res/mode"]
    assert dump.dtype == bool and np.count_nonzero(dump) >= 1
    # the logged mode really went POINTING -> DETUMBLE
    first_pointing = int(np.argmax(mode == MODE_POINTING))
    assert (mode == MODE_POINTING).any() and np.any(mode[first_pointing:] == MODE_DETUMBLE)
    # and the dumping flag was set while pointing (the dumping controller only runs in POINTING)
    assert np.any(dump & (mode == MODE_POINTING))


@pytest.mark.parametrize("kind", ["wrapper", "chunked"])
@pytest.mark.parametrize("case", list(CASES))
def test_golden_equivalence(golden_runs, golden, case, kind):
    res = golden_runs[(case, kind)].result()
    assert compare(res, golden, case) == []
    assert res.meta["kicks"] == []


# --------------------------------------------------------------------------
# 2. snapshot == logged row (realistic + design, 600 s, log_every = 1, 1 sample per step)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def snap_run():
    """Golden case 'real_design_s0_le1' (TRIAD initialises at t = 269 s), stepped sample by sample."""
    box = {}

    def runner(y0, t_end, **kw):
        sim = ClosedLoopSim(y0, t_end, **kw)
        snaps = []
        while not sim.done:
            assert sim.step(1) == 1
            snaps.append(sim.snapshot())
        box["sim"], box["snaps"] = sim, snaps
        return sim.result()

    res = run_case("real_design_s0_le1", runner)
    return box["sim"], box["snaps"], res


def _column(snaps, key, shape=()):
    """Stack a snapshot key over samples; None -> NaN."""
    out = np.full((len(snaps),) + shape, np.nan)
    for i, s in enumerate(snaps):
        if s[key] is not None:
            out[i] = s[key]
    return out


def test_snapshot_count_and_index(snap_run):
    sim, snaps, res = snap_run
    assert len(snaps) == sim.n_steps + 1 == len(res.t) == 601
    assert [s["k"] for s in snaps] == list(range(601))
    assert sim.done


@pytest.mark.parametrize("key", ROW_VEC)
def test_snapshot_vector_equals_row_bitwise(snap_run, key):
    _, snaps, res = snap_run
    width = getattr(res, key).shape[1]
    assert _bits_equal(_column(snaps, key, (width,)), getattr(res, key)), key


@pytest.mark.parametrize("key", ROW_SCA)
def test_snapshot_scalar_equals_row_bitwise(snap_run, key):
    _, snaps, res = snap_run
    assert _bits_equal(_column(snaps, key), getattr(res, key)), key


@pytest.mark.parametrize("key", ROW_INT + ROW_BOOL)
def test_snapshot_discrete_equals_row(snap_run, key):
    _, snaps, res = snap_run
    got = np.array([s[key] for s in snaps])
    assert np.array_equal(got, getattr(res, key)), key


def test_snapshot_P_att_equals_row_bitwise(snap_run):
    _, snaps, res = snap_run
    assert _bits_equal(_column(snaps, "P_att", (3, 3)), res.P_att)


def test_snapshot_none_iff_row_nan_and_init_flags(snap_run):
    """Estimator fields are None before TRIAD (all-NaN rows) and arrays after; t_init = 269 s."""
    sim, snaps, res = snap_run
    t_init = res.meta["t_init"]
    assert t_init == 269.0  # golden case, pinned in the golden file
    for i, s in enumerate(snaps):
        ready = i >= 269
        assert s["filter_ready"] is ready
        for key in ("q_est", "b_est", "bm_est", "know_err", "P_att"):
            assert (s[key] is not None) == ready, (i, key)
        assert (s["w_est"] is None) == (i == 0), i  # raw gyro before init, None only at k = 0
        assert math.isnan(s["att_sigma"]) == (not ready)
        assert math.isnan(s["est_nadir_err"]) == (not ready)
        assert (s["t_init"] == 269.0) if ready else math.isnan(s["t_init"])
    # rows: NaN before init exactly where the snapshot is None
    assert np.all(np.isnan(res.q_est[:269])) and not np.any(np.isnan(res.q_est[269:]))
    assert np.all(np.isnan(res.bm_est[:269])) and not np.any(np.isnan(res.bm_est[269:]))


def test_snapshot_environment_equals_sim_arrays(snap_run):
    sim, snaps, _ = snap_run
    for key, arr in (("r_eci", sim.r_eci), ("v_eci", sim.v_eci), ("sun_eci", sim.sun_eci)):
        assert _bits_equal(_column(snaps, key, (3,)), arr), key
    assert np.array_equal([s["lit"] for s in snaps], sim.lit)
    assert np.array_equal([s["t"] for s in snaps], sim.t_grid)


def test_snapshot_gmst_equals_frames_gmst(snap_run):
    _, snaps, _ = snap_run
    for s in (snaps[0], snaps[1], snaps[269], snaps[300], snaps[-1]):
        assert s["gmst"] == gmst(DEFAULT.jd_epoch + s["t"] / 86400.0)
    # Earth rotation advances 7.292115e-5 rad/s (sidereal rate): independent sanity check
    dg = (snaps[-1]["gmst"] - snaps[0]["gmst"]) % (2 * math.pi)
    assert dg == pytest.approx(7.2921159e-5 * 600.0, abs=1e-6)


def test_snapshot_quaternions_are_unit(snap_run):
    _, snaps, _ = snap_run
    for key in ("q", "q_est", "q_ref"):
        col = _column(snaps, key, (4,))
        ok = ~np.isnan(col[:, 0])
        assert ok.any()
        assert np.all(np.abs(np.linalg.norm(col[ok], axis=1) - 1.0) < 1e-12), key


def test_snapshot_q_ref_is_lvlh_of_conventions(snap_run):
    """q_ref: z = -r/|r|, y = -(r x v)/|r x v|, x = y x z (CONVENTIONS.md section 4); nadir_err consistent."""
    sim, snaps, res = snap_run
    for i in (0, 100, 269, 450, 600):
        r, v = sim.r_eci[i], sim.v_eci[i]
        z = -r / np.linalg.norm(r)
        y = -np.cross(r, v) / np.linalg.norm(np.cross(r, v))
        x = np.cross(y, z)
        dcm = np.column_stack([x, y, z])
        qr = snaps[i]["q_ref"]
        r_ref = Rot.from_quat([qr[1], qr[2], qr[3], qr[0]])
        # 1e-12: float64 DCM from a unit quaternion
        assert np.max(np.abs(r_ref.as_matrix() - dcm)) < 1e-12
        q = snaps[i]["q"]
        r_q = Rot.from_quat([q[1], q[2], q[3], q[0]])
        assert (r_ref.inv() * r_q).magnitude() == pytest.approx(res.nadir_err[i], abs=1e-12, rel=1e-12)


def test_snapshot_know_err_and_att_sigma_consistent(snap_run):
    """q_true = q_est (x) exp(know_err) (scipy), att_sigma = sqrt(trace P_att)."""
    _, snaps, _ = snap_run
    for i in (269, 270, 400, 600):
        s = snaps[i]
        qe, q, e = s["q_est"], s["q"], s["know_err"]
        rot = Rot.from_quat([qe[1], qe[2], qe[3], qe[0]]) * Rot.from_rotvec(e)
        assert (rot.inv() * Rot.from_quat([q[1], q[2], q[3], q[0]])).magnitude() < 1e-12
        assert s["att_sigma"] == pytest.approx(math.sqrt(np.trace(s["P_att"])), rel=1e-12)
        assert np.allclose(s["P_att"], s["P_att"].T, rtol=0, atol=1e-18)


def test_snapshot_tau_mag_is_m_cross_body_field(snap_run):
    """tau_mag = m x R(q)^T b_eci with the snapshot's own b_eci (independent arithmetic via scipy)."""
    _, snaps, _ = snap_run
    n_checked = 0
    for s in snaps[1:200:7]:
        q = s["q"]
        b_body = Rot.from_quat([q[1], q[2], q[3], q[0]]).inv().apply(s["b_eci"])
        expect = np.cross(s["m"], b_body)
        # 1e-12 relative to the torque scale: different operation order, both ~1e-5 T x 0.2 A m^2
        assert np.allclose(s["tau_mag"], expect, rtol=1e-12, atol=1e-12 * 1e-5 * 0.2)
        assert 1e-5 < np.linalg.norm(s["b_eci"]) < 7e-5  # LEO field magnitude, T
        n_checked += int(np.any(s["m"] != 0))
    assert n_checked > 5  # B-dot is actually commanding during the detumble


def test_snapshot_environment_vectors_physical(snap_run):
    _, snaps, _ = snap_run
    for s in snaps[::60]:
        assert np.linalg.norm(s["sun_eci"]) == pytest.approx(1.0, abs=1e-12)
        assert abs(np.linalg.norm(s["r_eci"]) / DEFAULT.orbit.a - 1.0) < 0.02
        assert s["rate"] == pytest.approx(np.linalg.norm(s["w"]), rel=1e-14)


def test_snapshot_types_and_copies(snap_run):
    sim, snaps, _ = snap_run
    s = snaps[-1]
    assert s["k"] == 600 and type(s["k"]) is int
    for key in ("t", "rate", "gmst", "nadir_err", "att_sigma", "est_nadir_err", "t_init"):
        assert type(s[key]) is float, key
    for key in ("mode", "sun_used", "n_kicks"):
        assert type(s[key]) is int, key
    for key in ("dump", "lit", "filter_ready"):
        assert type(s[key]) is bool, key
    for key in ("q", "w", "h_w", "wheel_speed", "m", "u_w", "tau_dist", "tau_mag", "r_eci", "v_eci",
                "sun_eci", "b_eci", "q_ref", "q_est", "b_est", "bm_est", "w_est", "know_err"):
        assert isinstance(s[key], np.ndarray) and s[key].dtype == np.float64, key
    assert s["P_att"].shape == (3, 3) and s["P_att"].dtype == np.float64
    assert isinstance(s["switch_times"], list) and isinstance(s["fallback_times"], list)
    json.dumps({k: v for k, v in s.items() if not isinstance(v, np.ndarray)})  # pure Python scalars


def test_snapshot_returns_independent_copies(snap_run):
    sim, snaps, _ = snap_run
    a = sim.snapshot()
    ref = {k: (v.copy() if isinstance(v, np.ndarray) else copy.deepcopy(v)) for k, v in a.items()}
    for v in a.values():
        if isinstance(v, np.ndarray):
            v[...] = 123.0
        elif isinstance(v, list):
            v.append(-1.0)
    b = sim.snapshot()
    for k, v in ref.items():
        if isinstance(v, np.ndarray):
            assert _bits_equal(b[k], v), k
        else:
            assert (b[k] == v) or (isinstance(v, float) and math.isnan(v) and math.isnan(b[k])), k
    assert _bits_equal(sim.state[0:4], snaps[-1]["q"])  # state after the last sample = logged truth


# --------------------------------------------------------------------------
# 3. indexing and errors
# --------------------------------------------------------------------------
def test_new_sim_indexing_and_properties():
    y0 = _y0()
    y0[0:4] *= 2.5  # un-normalised on purpose
    y0_before = y0.copy()
    sim = ClosedLoopSim(y0, 30.0, seed=2)
    assert sim.T == 1.0 and type(sim.n_steps) is int and sim.n_steps == 30
    assert sim.t_end == 30.0
    assert sim.period == pytest.approx(P_ORBIT, rel=1e-14)
    assert sim.k == 0 and sim.t == 0.0 and sim.done is False and sim.remaining == 31
    with pytest.raises(RuntimeError):
        sim.snapshot()
    s = sim.state
    assert s.shape == (10,) and s.dtype == np.float64
    assert np.max(np.abs(s[0:4] - y0_before[0:4] / np.linalg.norm(y0_before[0:4]))) < 1e-15
    assert abs(np.linalg.norm(s[0:4]) - 1.0) < 1e-15
    assert np.array_equal(s[4:10], y0_before[4:10])
    assert np.array_equal(y0, y0_before)  # the caller's array is not modified
    y0[:] = 0.0  # ... and the sim does not alias it
    assert np.array_equal(sim.state, s)


def test_environment_arrays_shapes_and_values():
    sim = _short(40.0)
    n = sim.n_steps + 1
    assert np.array_equal(sim.t_grid, np.arange(n) * 1.0)
    assert sim.r_eci.shape == sim.v_eci.shape == sim.sun_eci.shape == (n, 3)
    assert sim.lit.shape == (n,) and sim.lit.dtype == bool
    assert np.all(np.abs(np.linalg.norm(sim.sun_eci, axis=1) - 1.0) < 1e-12)
    # circular LEO: |r| ~ a (loose, orbit is not exactly circular under the ephemeris model), |v| ~ sqrt(mu/a)
    assert np.all(np.abs(np.linalg.norm(sim.r_eci, axis=1) / DEFAULT.orbit.a - 1.0) < 0.02)
    assert np.all(np.abs(np.linalg.norm(sim.v_eci, axis=1) / math.sqrt(3.986004418e14 / DEFAULT.orbit.a) - 1.0) < 0.02)


def test_lit_flag_matches_cylindrical_shadow_over_one_orbit():
    """Independent cylinder-shadow test over a full orbit (construction only, nothing stepped)."""
    sim = ClosedLoopSim(_y0(), P_ORBIT, seed=0)
    r, s = sim.r_eci, sim.sun_eci
    along = np.sum(r * s, axis=1)
    perp = np.linalg.norm(r - along[:, None] * s, axis=1)
    expect = ~((along < 0.0) & (perp < 6378137.0))
    # R_earth of the repo may differ from 6378137.0 by a few m (WGS-84 vs mean): allow boundary flips only
    assert np.count_nonzero(sim.lit != expect) <= 2
    assert not sim.lit.all()  # an LEO orbit has an eclipse at this epoch/inclination
    assert sim.lit.any()


def test_step_after_one_sample():
    sim = _short(30.0)
    assert sim.step(1) == 1
    assert sim.k == 1 and sim.t == 1.0 and sim.remaining == 30 and not sim.done
    assert sim.snapshot()["t"] == 0.0 and sim.snapshot()["k"] == 0
    assert sim.step(3) == 3
    assert sim.k == 4 and sim.t == 4.0 and sim.snapshot()["k"] == 3 and sim.snapshot()["t"] == 3.0


def test_step_to_horizon_and_done_errors():
    sim = _short(30.0)
    assert sim.step(10 ** 9) == 31
    assert sim.done and sim.k == 31 and sim.remaining == 0
    assert sim.t == sim.t_end == 30.0
    assert sim.snapshot()["k"] == 30 and sim.snapshot()["t"] == 30.0
    with pytest.raises(RuntimeError, match="horizon reached"):
        sim.step()
    with pytest.raises(RuntimeError, match="horizon reached"):
        sim.step(5)
    with pytest.raises(RuntimeError, match="horizon reached"):
        sim.kick([0.0, 0.0, 0.01])
    assert len(sim.result().t) == 31  # still readable


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, False, "3", None, 2.0])
def test_step_rejects_bad_n(bad):
    sim = _short(10.0)
    with pytest.raises(ValueError):
        sim.step(bad)
    assert sim.k == 0  # nothing was processed


def test_step_return_is_min_of_n_and_remaining():
    sim = _short(10.0)
    assert sim.step(4) == 4
    assert sim.step(100) == 7  # 11 samples in total
    assert sim.done


def test_last_sample_does_not_integrate():
    """After the last sample the pending state stays at t_end: the state no longer changes."""
    sim = _short(5.0)
    sim.step(5)
    s_before = sim.state
    assert sim.k == 5 and sim.t == 5.0
    sim.step(1)  # sample n_steps: log only
    assert sim.done and sim.t == 5.0
    assert np.array_equal(sim.state, s_before)
    snap = sim.snapshot()
    assert snap["k"] == 5 and snap["t"] == 5.0
    assert np.array_equal(snap["q"], sim.state[0:4])  # the logged truth is the pending state at t_end


def test_minimal_horizon_one_period():
    sim = _short(1.0)  # n_steps = 1: two samples
    assert sim.n_steps == 1 and sim.remaining == 2
    assert sim.step(2) == 2 and sim.done and sim.t == 1.0
    assert len(sim.result().t) == 2


def test_constructor_errors():
    with pytest.raises(ValueError, match="t_end shorter than one control period"):
        ClosedLoopSim(_y0(), 0.4)
    with pytest.raises(ValueError):
        ClosedLoopSim(_y0(), 0.0)
    for bad in (0, -3):
        with pytest.raises(ValueError):
            ClosedLoopSim(_y0(), 10.0, log_every=bad)


def test_non_unit_control_period():
    sim = ClosedLoopSim(_y0(), 20.0, mtq=MagnetorquerConfig(control_period=2.0))
    assert sim.T == 2.0 and sim.n_steps == 10 and sim.t_end == 20.0
    assert np.array_equal(sim.t_grid, np.arange(11) * 2.0)
    sim.step(3)
    assert sim.t == 6.0 and sim.snapshot()["t"] == 4.0 and sim.remaining == 8


def test_free_rotation_integration_matches_scipy():
    """State after sample 0 is the truth at t_1 = T: free rotation (no disturbance, m = 0, u_w = 0 at k = 0)."""
    y0 = _y0(rate=0.05, seed=3, h_w=[2e-4, -1e-4, 3e-4])
    sim = ClosedLoopSim(y0, 10.0, ctrl=ControlConfig(disturbances=False))
    sim.step(1)
    sim_y0 = y0.copy()
    expect = _free_rotation(sim_y0, I_NOM, 1.0)
    got = sim.state
    # RK4 with 0.8 s + 0.2 s sub-steps at 0.05 rad/s: truncation error ~ (0.05 * 0.8)^5 / 120 ~ 1e-9
    assert np.max(np.abs(got[0:4] - expect[0:4] * np.sign(expect[0] * got[0]))) < 1e-8
    assert np.max(np.abs(got[4:7] - expect[4:7])) < 1e-8
    assert np.array_equal(got[7:10], sim_y0[7:10])  # h_w constant: u_w = 0 in the first detumble sample


# --------------------------------------------------------------------------
# 4. step splitting is invisible; state is a copy
# --------------------------------------------------------------------------
def test_step_splitting_bit_identical():
    kw = dict(seed=4, sensors=REALISTIC_SENSORS, mekf_cfg=MEKF_DESIGN, log_every=3)
    a = ClosedLoopSim(_y0(), 150.0, **kw)
    a.step(a.remaining)
    b = ClosedLoopSim(_y0(), 150.0, **kw)
    for n in (1, 147):
        b.step(n)
    b.step(b.remaining)
    c = ClosedLoopSim(_y0(), 150.0, **kw)
    while not c.done:
        c.step(1)
    ra, rb, rc = a.result(), b.result(), c.result()
    assert _same_result(ra, rb) and _same_result(ra, rc)
    assert np.array_equal(a.state, b.state) and np.array_equal(a.state, c.state)


def test_state_is_a_copy():
    sim_a, sim_b = _short(20.0, seed=5), _short(20.0, seed=5)
    s = sim_a.state
    s[:] = 99.0
    assert not np.any(sim_a.state == 99.0)
    sim_a.state[4:7] = 7.0  # also a throw-away copy
    sim_a.step(sim_a.remaining)
    sim_b.step(sim_b.remaining)
    assert _same_result(sim_a.result(), sim_b.result())
    s2 = sim_a.state
    s2[0] = -5.0
    assert sim_a.state[0] != -5.0


# --------------------------------------------------------------------------
# 5. kick = exact delta-omega
# --------------------------------------------------------------------------
def test_kick_is_exact_delta_omega():
    sim = _short(30.0, y0=_y0(h_w=[1e-4, 2e-4, -3e-4]))
    sim.step(4)
    s0 = sim.state
    dw = np.array([0.011, -0.007, 0.0234])
    sim.kick(dw)
    s1 = sim.state
    assert np.array_equal(s1[4:7], s0[4:7] + dw)
    assert np.array_equal(s1[0:4], s0[0:4])
    assert np.array_equal(s1[7:10], s0[7:10])
    assert sim.k == 4 and sim.t == 4.0 and sim.remaining == 27  # kick processes no sample


def test_kick_accepts_sequences_and_int_arrays():
    sim = _short(10.0)
    s0 = sim.state
    # list of ints, tuple of ints, int ndarray (kept far below MAX_KICK_RATE = 30 deg/s = 0.52 rad/s)
    sim.kick([1e-1, 0, 0])
    sim.kick((0, 5e-2, 0))
    sim.kick(np.array([0, 0, 2]) * 1e-1)
    sim_int = _short(10.0)
    sim_int.kick(np.array([0, 0, 0]))  # integer zero array is a valid (3,) finite input
    assert np.array_equal(sim_int.state, s0)
    assert np.array_equal(sim.state[4:7], ((s0[4:7] + np.array([0.1, 0, 0])) + np.array([0, 0.05, 0]))
                          + np.array([0, 0, 0.2]))


def test_two_kicks_add_and_are_recorded():
    sim = _short(30.0)
    sim.step(6)
    s0 = sim.state
    d1, d2 = np.array([0.01, 0.02, -0.03]), np.array([-0.004, 0.0, 0.05])
    sim.kick(d1)
    sim.kick(d2)
    assert np.array_equal(sim.state[4:7], (s0[4:7] + d1) + d2)
    kicks = sim.result().meta["kicks"]
    assert len(kicks) == 2
    for rec, d in zip(kicks, (d1, d2)):
        assert set(rec) == {"t", "k", "dw"}
        assert rec["t"] == 6.0 and rec["k"] == 6
        assert rec["dw"] == [float(x) for x in d]
    json.dumps(kicks)  # plain Python types only
    assert type(kicks[0]["k"]) is int and type(kicks[0]["t"]) is float
    assert all(type(x) is float for x in kicks[0]["dw"])
    sim.step(1)
    assert sim.snapshot()["n_kicks"] == 2


def test_kick_records_time_of_each_kick():
    sim = _short(30.0)
    sim.kick([0, 0, 0.01])
    sim.step(5)
    sim.kick([0.01, 0, 0])
    kicks = sim.result().meta["kicks"]
    assert [(r["k"], r["t"]) for r in kicks] == [(0, 0.0), (5, 5.0)]


def test_kick_before_first_sample_acts_like_modified_y0_but_is_recorded():
    y0 = _y0(seed=6)
    dw = np.array([0.02, -0.01, 0.015])
    a = ClosedLoopSim(y0, 40.0, seed=1)
    a.kick(dw)
    y0b = y0.copy()
    y0b[4:7] = y0b[4:7] + dw
    b = ClosedLoopSim(y0b, 40.0, seed=1)
    a.step(a.remaining)
    b.step(b.remaining)
    assert _same_result(a.result(), b.result())
    assert a.result().meta["kicks"] == [{"t": 0.0, "k": 0, "dw": [0.02, -0.01, 0.015]}]
    assert b.result().meta["kicks"] == []


@pytest.mark.parametrize("bad", [[1.0, 2.0], [0.0, 0.0, 0.0, 0.0], 0.1, [[0.0], [0.0], [0.0]],
                                 [np.nan, 0.0, 0.0], [0.0, np.inf, 0.0], [0.0, 0.0, -np.inf]])
def test_kick_rejects_bad_input(bad):
    sim = _short(10.0)
    sim.step(2)
    s0 = sim.state
    with pytest.raises(ValueError):
        sim.kick(bad)
    assert np.array_equal(sim.state, s0)
    assert sim.result().meta["kicks"] == []


def test_kicks_list_is_a_copy():
    sim = _short(10.0)
    sim.kick([0.0, 0.0, 0.01])
    res = sim.result()
    res.meta["kicks"].append("junk")
    res.meta["kicks"][0]["dw"][0] = 99.0
    assert sim.result().meta["kicks"] == [{"t": 0.0, "k": 0, "dw": [0.0, 0.0, 0.01]}]


# --------------------------------------------------------------------------
# 6. kick physics and timing
# --------------------------------------------------------------------------
def test_kick_wheel_speed_change():
    """Wheel i spins about body axis i: Omega = h/J - w, so the kick changes it by -dw_i; h_w and q are untouched."""
    h0 = [3e-4, -2e-4, 5e-4]
    sim = ClosedLoopSim(_y0(rate=0.1, seed=7, h_w=h0), 40.0, seed=3)
    sim.step(20)
    s0 = sim.state
    dw = np.array([0.09, -0.05, 0.12])
    sim.kick(dw)
    s1 = sim.state
    d_omega = wheel_speed(s1[7:10], s1[4:7]) - wheel_speed(s0[7:10], s0[4:7])
    # |Omega| ~ 150 rad/s -> ulp 3e-14; 1e-12 leaves margin while dw is O(0.1)
    assert np.max(np.abs(d_omega + dw)) < 1e-12
    assert np.array_equal(s1[7:10], s0[7:10]) and np.array_equal(s1[0:4], s0[0:4])


def test_kick_next_snapshot_logs_kicked_rate_and_gyro_does_not_see_it():
    """Sample k logs the kicked truth; its gyro (mean over [t_{k-1}, t_k]) and controller do not see the kick."""
    kw = dict(seed=0, gyro_cfg=GyroConfig(arw=1e-9, rrw=1e-12, bias0_sigma=1e-9))
    y0 = lvlh_state(R0, V0)
    dw = np.array([0.05, -0.03, 0.04])
    ref_state_at_5 = ClosedLoopSim(y0, 30.0, **kw)
    ref_state_at_5.step(5)
    ref_state = ref_state_at_5.state  # truth at t_5 before the kick
    ref2, kicked2 = ClosedLoopSim(y0, 30.0, **kw), ClosedLoopSim(y0, 30.0, **kw)
    ref2.step(5)
    kicked2.step(5)
    kicked2.kick(dw)
    ref2.step(1)
    kicked2.step(1)
    r5, k5 = ref2.snapshot(), kicked2.snapshot()
    ref2.step(1)
    kicked2.step(1)
    r6, k6 = ref2.snapshot(), kicked2.snapshot()
    assert r5["k"] == k5["k"] == 5
    # truth at t_5 is the kicked rate, exactly
    assert np.array_equal(k5["w"], r5["w"] + dw)
    assert np.array_equal(k5["w"], np.asarray(ref_state[4:7]) + dw)
    assert np.array_equal(k5["q"], r5["q"]) and np.array_equal(k5["h_w"], r5["h_w"])
    # sensors, estimator and controller of sample 5 are untouched by the kick, to the bit
    for key in ("w_est", "q_est", "b_est", "bm_est", "att_sigma", "m", "u_w", "mode"):
        a, b = r5[key], k5[key]
        assert _bits_equal(a, b), key
    # sample 6: the gyro mean over [t_5, t_6] now contains the kick (noise-free gyro: error is dynamics only)
    # tolerance 5e-3 rad/s: rate change over 1 s from the B-dot torque / gyroscopic terms at 0.05 rad/s
    # is O(1e-3) rad/s, 10x below |dw|
    assert np.max(np.abs((k6["w_est"] - r6["w_est"]) - dw)) < 5e-3
    assert np.max(np.abs(k6["w_est"] - k6["w"])) < 5e-2  # sanity: tracks the truth after the kick


# non-diagonal, symmetric positive definite (eigenvalues checked in the test), clearly unlike I_NOM
I_FULL = np.array([[0.050, 0.004, -0.003], [0.004, 0.045, 0.002], [-0.003, 0.002, 0.009]])


@pytest.mark.parametrize("inertia_true", [None, I_FULL], ids=["nominal", "full-matrix"])
def test_kick_enters_integration_from_the_kicked_rate(inertia_true):
    """Free rotation after a kick at t_0: truth at t_1 equals the scipy integration from y0 + dw with I_true.

    Controller output at k = 0 (verified here from the snapshot, per spec 1): the sim starts in DETUMBLE,
    B-dot has no previous magnetometer sample at k = 0, so m = 0, and the wheels are idle in DETUMBLE
    (u_w = 0). With disturbances off there is no external torque, so the integration is exactly the
    torque-free rotation with a constant wheel momentum and the TRUE inertia (the kick itself never
    touches the inertia; a wrong inertia in the integrator would show up here).
    """
    inertia = I_NOM if inertia_true is None else inertia_true
    assert np.all(np.linalg.eigvalsh(inertia) > 0.0) and np.allclose(inertia, inertia.T)
    y0 = _y0(rate=0.05, seed=8, h_w=[1e-4, 2e-4, -2e-4])
    dw = np.array([0.12, -0.08, 0.15])  # |w + dw| ~ 0.2 rad/s, well below MAX_KICK_RATE = 0.52 rad/s
    sim = ClosedLoopSim(y0, 10.0, ctrl=ControlConfig(disturbances=False), inertia_true=inertia_true)
    sim.kick(dw)
    sim.step(1)
    snap = sim.snapshot()
    assert snap["mode"] == MODE_DETUMBLE and snap["k"] == 0
    assert np.array_equal(snap["m"], np.zeros(3)) and np.array_equal(snap["u_w"], np.zeros(3))
    assert np.array_equal(snap["tau_dist"], np.zeros(3))
    y_k = y0.copy()
    y_k[4:7] += dw
    expect = _free_rotation(y_k, inertia, 1.0)
    got = sim.state
    sgn = np.sign(expect[0] * got[0])
    # RK4 with 0.8 s + 0.2 s sub-steps at |w| ~ 0.2 rad/s: truncation ~ (0.2 * 0.8)^5 / 120 ~ 9e-7
    # (measured ~7e-8); 1e-6 is therefore a principled bound, not a fitted one
    tol = 1e-6
    assert np.max(np.abs(got[0:4] - expect[0:4] * sgn)) < tol
    assert np.max(np.abs(got[4:7] - expect[4:7])) < tol
    assert np.array_equal(got[7:10], y_k[7:10])  # wheel momentum constant (u_w = 0)
    # angular momentum H = R(q)(I w + h_w) with the TRUE inertia is conserved across the integration
    H0 = _angular_momentum_inertial(y_k, inertia)
    H1 = _angular_momentum_inertial(got, inertia)
    assert np.max(np.abs(H1 - H0)) < 1e-6 * np.linalg.norm(H0)
    if inertia_true is not None:
        # the same initial state integrated with the NOMINAL inertia is clearly different: the kicked
        # rate evolves with I_true.  Observed ~2e-3 rad/s (w) and ~5e-4 (q) vs 1e-6 agreement above.
        wrong = _free_rotation(y_k, I_NOM, 1.0)
        sgn_w = np.sign(wrong[0] * got[0])
        assert np.max(np.abs(got[4:7] - wrong[4:7])) > 1e-4
        assert np.max(np.abs(got[0:4] - wrong[0:4] * sgn_w)) > 1e-4


# --------------------------------------------------------------------------
# 7. kick response: POINTING -> DETUMBLE fallback
# --------------------------------------------------------------------------
# Pointing start. The golden 'real_design_s0' case needs ~6500 s to reach POINTING; instead start at
# the nadir attitude with lvlh_state and IDEAL_SENSORS + MEKF_G7: TRIAD initialises at k = 0, B-dot
# runs while the MEKF converges, and the mode machine switches at t = 360 s (switch_min_filter_time
# 300 s + detumble_hold 60 s; checked in the test).  Each run simulates 480 s (about 1.5 s of CPU);
# the kick is applied at t = 420 s, i.e. 60 s after the switch, and followed for 60 s.
# (A realistic-sensor start from lvlh_state does not run TRIAD within 900 s, hence ideal sensors.)
KICK_T = 420
KICK_DW = np.array([np.deg2rad(5.0), 0.0, 0.0])  # 5 deg/s about body x
# Why this kick forces a fallback: it is a SATURATION test, not a generic "fast rate" test.  The impulse
# is dL_x = I_xx dw_x = 0.0419 * 0.0873 = 3.65e-3 N m s, more than twice the wheel capacity
# h_max = 1.77e-3 N m s (WheelConfig.rated_momentum), so the x wheel hits its speed limit and cannot
# absorb it; the rate is left stuck at ~2.56 deg/s > safe_rate = 2 deg/s for far longer than safe_hold.
# A kick whose dL is below h_max (KICK_SMALL_DW below) is absorbed and gives no fallback.
KICK_SMALL_DW = np.array([np.deg2rad(2.25), 0.0, 0.0])  # dL_x = 1.64e-3 N m s = 0.93 h_max


@pytest.fixture(scope="module")
def kick_runs():
    y0 = lvlh_state(R0, V0)
    kw = dict(seed=0, sensors=IDEAL_SENSORS, mekf_cfg=MEKF_G7, log_every=1)
    base = ClosedLoopSim(y0, 480.0, **kw)
    base.step(base.remaining)
    kicked = ClosedLoopSim(y0, 480.0, **kw)
    kicked.step(KICK_T)
    pre_snap = kicked.snapshot()
    kicked.kick(KICK_DW)
    kicked.step(kicked.remaining)
    return base.result(), kicked.result(), pre_snap, kicked


@pytest.fixture(scope="module")
def kick_small_run():
    y0 = lvlh_state(R0, V0)
    kw = dict(seed=0, sensors=IDEAL_SENSORS, mekf_cfg=MEKF_G7, log_every=1)
    sim = ClosedLoopSim(y0, 480.0, **kw)
    sim.step(KICK_T)
    sim.kick(KICK_SMALL_DW)
    sim.step(sim.remaining)
    return sim.result()


def test_kick_response_starts_from_pointing(kick_runs):
    base, kicked, pre_snap, _ = kick_runs
    assert pre_snap["k"] == KICK_T - 1 and pre_snap["mode"] == MODE_POINTING
    assert base.meta["switch_times"] == kicked.meta["switch_times"] == [360.0]
    assert np.all(base.mode[360:] == MODE_POINTING)


def test_kick_before_the_kick_nothing_differs(kick_runs):
    base, kicked, _, _ = kick_runs
    for name in ROW_VEC + ROW_SCA + ROW_INT + ROW_BOOL + ("P_att", "gyro_bias"):
        a, b = getattr(base, name)[:KICK_T], getattr(kicked, name)[:KICK_T]
        assert _bits_equal(a, b), name


def test_kick_row_at_kick_time_logs_kicked_truth_but_unkicked_estimates(kick_runs):
    base, kicked, _, _ = kick_runs
    k = KICK_T
    assert np.array_equal(kicked.w[k], base.w[k] + KICK_DW)
    assert np.array_equal(kicked.q[k], base.q[k]) and np.array_equal(kicked.h_w[k], base.h_w[k])
    # wheel speed relative to the body drops by dw (a_i = e_i); relative 1e-12 on O(1) rad/s values
    assert np.allclose(kicked.wheel_speed[k] - base.wheel_speed[k], -KICK_DW, rtol=0, atol=1e-12)
    # the gyro of sample k covers [t_{k-1}, t_k] and does not see the kick: estimator outputs are identical
    for name in ("w_est", "q_est", "b_est", "bm_est", "att_sigma", "P_att", "know_err"):
        assert _bits_equal(getattr(kicked, name)[k], getattr(base, name)[k]), name
    assert kicked.mode[k] == base.mode[k] == MODE_POINTING
    # sample k+1 is the first that sees it: gyro mean over 1 s differs by ~dw (controller removes <= 3e-3)
    dx = kicked.w_est[k + 1, 0] - base.w_est[k + 1, 0]
    assert abs(dx - KICK_DW[0]) < 0.01


def test_kick_makes_mode_machine_fall_back(kick_runs):
    base, kicked, _, _ = kick_runs
    T, hold = 1.0, ModeConfig().safe_hold
    fb = kicked.meta["fallback_times"]
    assert len(fb) == 1
    assert fb[0] >= KICK_T
    # spec: fallback within safe_hold + 2T of the kick ...
    assert fb[0] <= KICK_T + hold + 2 * T
    # ... and not before the estimated rate has been above safe_rate for safe_hold: the first sample
    # whose gyro sees the kick is k+1, so above_since >= t_{k+1} and the fallback is >= t_{k+1} + safe_hold
    assert fb[0] >= KICK_T + T + hold
    i = int(round(fb[0] / T))
    assert np.all(kicked.mode[360:i] == MODE_POINTING)
    assert np.all(kicked.mode[i:] == MODE_DETUMBLE)
    assert kicked.mode[-1] == MODE_DETUMBLE


def test_run_without_kick_has_no_fallback(kick_runs):
    base, _, _, _ = kick_runs
    assert base.meta["fallback_times"] == []
    assert base.meta["kicks"] == []
    assert np.all(base.mode[360:] == MODE_POINTING)


def test_kick_fallback_is_caused_by_x_wheel_saturation(kick_runs):
    """dL_x > h_max, the x wheel reaches max_speed, and after that the rate stays above safe_rate."""
    base, kicked, _, _ = kick_runs
    wheel, modes = WheelConfig(), ModeConfig()
    d_l = I_NOM[0, 0] * KICK_DW[0]
    assert d_l > 2.0 * wheel.h_max  # 3.65e-3 vs 1.77e-3 N m s: cannot be absorbed (derived from the spec data)
    ws = np.abs(kicked.wheel_speed[KICK_T:, 0])
    # the wheel reaches its speed limit (1e-3 margin: the limiter acts on the next sample's torque)
    assert ws.max() >= 0.999 * wheel.max_speed
    i_sat = KICK_T + int(np.argmax(ws >= 0.999 * wheel.max_speed))
    assert i_sat < KICK_T + 20  # saturates within seconds of the kick
    # without the kick the wheel never gets near the limit
    assert np.abs(base.wheel_speed[:, 0]).max() < 0.5 * wheel.max_speed
    # once saturated, the controller cannot remove the rate any more: truth rate stays above safe_rate
    assert np.all(kicked.rate[i_sat:] > modes.safe_rate)
    assert i_sat + modes.safe_hold < len(kicked.t)  # long enough for the hold timer to expire


def test_kick_below_wheel_capacity_gives_no_fallback(kick_small_run):
    """Counter-case: dL_x = 0.93 h_max, above safe_rate at first (so the test is not vacuous), but the
    wheel removes the rate within a few seconds, long before safe_hold (10 s) expires: no fallback."""
    kicked = kick_small_run
    wheel, modes = WheelConfig(), ModeConfig()
    assert I_NOM[0, 0] * KICK_SMALL_DW[0] < wheel.h_max
    k1 = KICK_T + 1
    assert np.linalg.norm(kicked.w_est[k1]) > modes.safe_rate  # the estimate sees a rate above the threshold
    assert kicked.rate[KICK_T] > modes.safe_rate
    assert np.all(kicked.rate[KICK_T + 6:] < modes.safe_rate)  # but it is removed (observed within ~2 s)
    assert kicked.meta["fallback_times"] == []
    assert np.all(kicked.mode[360:] == MODE_POINTING)
    assert kicked.meta["kicks"][0]["dw"] == [float(x) for x in KICK_SMALL_DW]


def test_kick_snapshot_event_lists_follow_result(kick_runs):
    _, kicked, _, sim = kick_runs
    s = sim.snapshot()
    assert s["fallback_times"] == kicked.meta["fallback_times"]
    assert s["switch_times"] == kicked.meta["switch_times"]
    assert s["n_kicks"] == 1
    assert kicked.meta["kicks"] == [{"t": float(KICK_T), "k": KICK_T, "dw": [float(x) for x in KICK_DW]}]


# --------------------------------------------------------------------------
# 8. result() mid-run
# --------------------------------------------------------------------------
def test_result_midrun_prefix_of_full_run_and_unchanged_afterwards():
    kw = dict(seed=0, sensors=IDEAL_SENSORS, mekf_cfg=MEKF_G7, log_every=10)
    y0 = lvlh_state(R0, V0)
    sim = ClosedLoopSim(y0, 400.0, **kw)
    sim.step(100)
    r1 = sim.result()
    r1_copy = copy.deepcopy(r1)
    assert len(r1.t) == 10
    assert np.array_equal(r1.t, np.arange(10) * 10 * 1.0)
    assert r1.meta["t_end"] == 400.0  # horizon, not the current time
    assert r1.meta["switch_times"] == [] and np.isnan(r1.meta["switch_time"])
    sim.step(260)  # samples 0..359 done: the switch (sample t = 360 s) has not happened yet
    sim.step(40)   # samples 360..399: the switch happens, 40 logged rows in total
    r2 = sim.result()
    assert len(r2.t) == 40 and r2.meta["switch_times"] == [360.0]
    # the first result is untouched by later steps (arrays and event lists)
    for fld in ("t", "q", "w", "h_w", "wheel_speed", "m", "u_w", "mode", "dump", "q_est", "P_att", "lit"):
        assert _bits_equal(getattr(r1, fld), getattr(r1_copy, fld)), fld
    assert r1.meta["switch_times"] == [] and np.isnan(r1.meta["switch_time"])
    assert r1.meta["fallback_times"] == r1_copy.meta["fallback_times"]
    # rows [0:10] of the later result are the first result
    for fld in ("t", "q", "w", "h_w", "m", "u_w", "mode", "q_est", "att_sigma", "know_err"):
        assert _bits_equal(getattr(r2, fld)[:10], getattr(r1, fld)), fld
    # and the whole thing equals an uninterrupted run
    full = ClosedLoopSim(y0, 400.0, **kw)
    full.step(full.remaining)
    # sim has processed 400 of 401 samples; finish it and compare
    sim.step(sim.remaining)
    assert _same_result(sim.result(), full.result())
    assert _same_result(sim.result(), sim.result())  # repeatable


def test_result_event_lists_and_kicks_are_copies():
    sim = _short(30.0, seed=0)
    sim.kick([0.0, 0.0, 0.001])
    res = sim.result()
    sim.kick([0.0, 0.001, 0.0])
    assert len(res.meta["kicks"]) == 1 and len(sim.result().meta["kicks"]) == 2
    res.meta["switch_times"].append(1.0)
    res.meta["fallback_times"].append(1.0)
    again = sim.result()
    assert again.meta["switch_times"] == [] and again.meta["fallback_times"] == []


def test_result_on_new_sim_is_empty():
    sim = ClosedLoopSim(_y0(), 30.0, seed=0, log_every=5)
    res = sim.result()
    assert len(res.t) == 0
    assert res.q.shape == (0, 4) and res.w.shape == (0, 3) and res.P_att.shape == (0, 3, 3)
    assert res.q_est.shape == (0, 4) and res.mode.shape == (0,)
    assert np.isnan(res.meta["t_init"]) and np.isnan(res.meta["switch_time"])
    assert res.meta["t_end"] == 30.0 and res.meta["kicks"] == []
    assert res.meta["switch_times"] == [] and res.meta["fallback_times"] == []
    assert res.meta["nis"] == {}


def test_result_t_init_nan_before_triad_and_value_after():
    """Golden case real_design_s0: TRIAD initialises at t = 269 s (pinned in the golden file)."""
    y0 = tumble_state(np.deg2rad(10.0), 0)
    sim = ClosedLoopSim(y0, 600.0, seed=0, sensors=REALISTIC_SENSORS, mekf_cfg=MEKF_DESIGN, log_every=1)
    sim.step(100)
    assert np.isnan(sim.result().meta["t_init"])
    assert np.isnan(sim.snapshot()["t_init"]) and not sim.snapshot()["filter_ready"]
    assert sim.result().meta["nis"] == {}
    sim.step(200)  # samples 0..299
    assert sim.result().meta["t_init"] == 269.0
    assert sim.snapshot()["t_init"] == 269.0 and sim.snapshot()["filter_ready"]


# --------------------------------------------------------------------------
# 9. wrapper identity
# --------------------------------------------------------------------------
def test_wrapper_equals_sim_default_arguments():
    y0 = _y0(rate=0.1, seed=5)
    kw = dict(seed=3, sensors=REALISTIC_SENSORS, mekf_cfg=MEKF_DESIGN, log_every=2)
    wrapped = run_closed_loop(y0, 120.0, **kw)
    sim = ClosedLoopSim(y0, 120.0, **kw)
    sim.step(sim.remaining)
    direct = sim.result()
    assert compare(wrapped, _Golden("w", flatten(direct)), "w") == []
    assert wrapped.meta["kicks"] == [] and direct.meta["kicks"] == []


def test_wrapper_forwards_every_argument():
    """Every argument has a non-default value (cfg: a 550 km orbit instead of 400 km); a dropped or swapped
    one makes the results differ."""
    y0 = _y0(rate=0.08, seed=9)
    cfg = SimConfig(orbit=OrbitConfig(altitude=550e3))
    assert cfg != DEFAULT
    kw = dict(seed=11, cfg=cfg, ctrl=ControlConfig(dumping=False, disturbances=False),
              mtq=MagnetorquerConfig(m_max=0.15, duty=0.6), wheel=WheelConfig(max_torque=1.5e-4),
              sensors=REALISTIC_SENSORS, mekf_cfg=MEKF_DESIGN, modes=ModeConfig(safe_hold=5.0),
              mag_cfg=MagnetometerConfig(sigma=300e-9), css_cfg=SunSensorConfig(sigma_diode=0.01),
              gyro_cfg=GyroConfig(arw=3e-4), inertia_true=1.05 * I_NOM, log_every=3)
    wrapped = run_closed_loop(y0, 90.0, **kw)
    sim = ClosedLoopSim(y0, 90.0, **kw)
    sim.step(sim.remaining)
    direct = sim.result()
    assert compare(wrapped, _Golden("w", flatten(direct)), "w") == []
    # the arguments really reached the model (not just the same defaults twice)
    assert wrapped.meta["mtq"]["m_max"] == 0.15 and wrapped.meta["modes"]["safe_hold"] == 5.0
    assert wrapped.meta["wheel"]["max_torque"] == 1.5e-4
    assert np.array_equal(wrapped.meta["inertia_true"], 1.05 * I_NOM)
    assert wrapped.meta["seed"] == 11 and len(wrapped.t) == 30 + 1
    # cfg reached the model: the orbital period is the 550 km one (Kepler: T = 2 pi sqrt(a^3/mu)), not 400 km
    assert wrapped.meta["orbital_period"] == pytest.approx(orbital_period(cfg.orbit.a), rel=1e-14)
    assert wrapped.meta["orbital_period"] > P_ORBIT + 100.0
    assert direct.meta["orbital_period"] == wrapped.meta["orbital_period"]
    plain = run_closed_loop(y0, 90.0, seed=11, sensors=REALISTIC_SENSORS, mekf_cfg=MEKF_DESIGN, log_every=3)
    assert not _same_result(wrapped, plain)
    # dropping ONLY cfg (default 400 km orbit) changes the result: a wrapper that ignored cfg would fail
    assert not _same_result(wrapped, run_closed_loop(y0, 90.0, **dict(kw, cfg=DEFAULT)))


# --------------------------------------------------------------------------
# 10. kick limits (spec 5.12, [B-25])
# --------------------------------------------------------------------------
def _sim_at_rate(w, t_end=20.0, **kw):
    """Sim whose initial truth body rate is exactly ``w`` (rad/s)."""
    y = _y0()
    y[4:7] = w
    return ClosedLoopSim(y, t_end, **kw)


def test_max_kick_rate_is_30_deg_per_s():
    assert MAX_KICK_RATE == np.deg2rad(30.0)
    assert MAX_KICK_RATE == pytest.approx(0.5235987755982988, rel=1e-15)
    assert float(MAX_KICK_RATE) < math.pi  # below the wrap of the gyro's mean-rate rotation vector


@pytest.mark.parametrize("w0, dw", [
    ([0.0, 0.05, 0.0], [0.6, 0.0, 0.0]),                      # |w + dw| = 0.60 > cap
    ([0.0, 0.3, 0.3], [0.0, 0.3, 0.3]),                       # |dw| = 0.42 < cap, |w + dw| = 0.85 > cap
    ([0.0, 0.0, 0.0], [-MAX_KICK_RATE * 1.001, 0.0, 0.0]),    # just above the cap, negative direction
])
def test_kick_above_cap_raises_and_leaves_state_unchanged(w0, dw):
    sim = _sim_at_rate(w0)
    s0 = sim.state
    assert np.linalg.norm(np.asarray(w0) + np.asarray(dw)) > MAX_KICK_RATE  # premise of the case
    with pytest.raises(ValueError):
        sim.kick(dw)
    assert np.array_equal(sim.state, s0)
    assert sim.result().meta["kicks"] == [] and sim.k == 0
    assert sim.failed is False  # argument validation does not latch
    assert sim.step(1) == 1  # and the sim keeps working


def test_kick_cap_is_on_the_resulting_rate_not_on_dw():
    """w = (0.4, 0, 0): a kick of -0.8 rad/s (|dw| > cap) lands at -0.4 rad/s (< cap) and is accepted."""
    sim = _sim_at_rate([0.4, 0.0, 0.0])
    sim.kick([-0.8, 0.0, 0.0])
    assert np.array_equal(sim.state[4:7], np.array([0.4, 0.0, 0.0]) + np.array([-0.8, 0.0, 0.0]))
    assert np.linalg.norm(sim.state[4:7]) < MAX_KICK_RATE < 0.8


def test_kick_just_below_cap_accepted_just_above_rejected():
    direction = np.array([1.0, 2.0, 2.0]) / 3.0  # exact unit vector (1 + 4 + 4 = 9)
    below = _sim_at_rate([0.0, 0.0, 0.0])
    dw = direction * MAX_KICK_RATE * (1.0 - 1e-9)
    below.kick(dw)
    assert np.array_equal(below.state[4:7], dw)
    assert np.linalg.norm(below.state[4:7]) < MAX_KICK_RATE
    above = _sim_at_rate([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        above.kick(direction * MAX_KICK_RATE * (1.0 + 1e-9))
    assert np.array_equal(above.state[4:7], np.zeros(3))


def test_second_kick_cap_uses_accumulated_rate():
    sim = _sim_at_rate([0.0, 0.0, 0.0])
    sim.kick([0.3, 0.0, 0.0])
    s1 = sim.state
    with pytest.raises(ValueError):
        sim.kick([0.3, 0.0, 0.0])  # each is below the cap, together 0.6 rad/s > 0.52
    assert np.array_equal(sim.state, s1)
    assert len(sim.result().meta["kicks"]) == 1
    sim.kick([-0.3, 0.1, 0.0])  # opposite direction is fine
    assert len(sim.result().meta["kicks"]) == 2


def test_kick_at_last_period_raises_horizon_reached():
    """k == n_steps: no integration is left, a kick could only alter the last logged w."""
    sim = _short(5.0)
    sim.step(5)  # samples 0..4 -> k = 5 = n_steps
    assert sim.k == sim.n_steps == 5 and not sim.done
    s0 = sim.state
    with pytest.raises(RuntimeError, match="horizon reached"):
        sim.kick([0.0, 0.0, 0.01])
    assert np.array_equal(sim.state, s0)
    assert sim.result().meta["kicks"] == []
    # one sample earlier a kick is still allowed
    other = _short(5.0)
    other.step(4)
    other.kick([0.0, 0.0, 0.01])
    assert len(other.result().meta["kicks"]) == 1
    # the last sample can still be processed after the rejected kick
    assert sim.step(1) == 1 and sim.done


# --------------------------------------------------------------------------
# 11. failure latch (spec "Failure latch")
# --------------------------------------------------------------------------
class _Boom(Exception):
    pass


def _make_raiser(real, n_ok):
    """Wrapper that calls ``real`` ``n_ok`` times and raises _Boom on the next call."""
    calls = {"n": 0}

    def wrapper(*a, **kw):
        calls["n"] += 1
        if calls["n"] > n_ok:
            raise _Boom(f"injected failure on call {calls['n']}")
        return real(*a, **kw)

    wrapper.calls = calls
    return wrapper


def _assert_latched(sim):
    assert sim.failed is True
    with pytest.raises(RuntimeError, match="simulation failed"):
        sim.step(1)
    with pytest.raises(RuntimeError, match="simulation failed"):
        sim.step(10)
    with pytest.raises(RuntimeError, match="simulation failed"):
        sim.kick([0.0, 0.0, 0.001])
    with pytest.raises(RuntimeError, match="simulation failed"):
        sim.snapshot()


@pytest.mark.parametrize("name", ["total_disturbance_torque", "rk4_step", "magnetometer"])
def test_failure_latch_mid_chunk(monkeypatch, name):
    """A dependency of the per-sample loop raises on its 5th call inside step(10)."""
    sim = _short(30.0)
    assert sim.failed is False
    raiser = _make_raiser(getattr(adcs.stepper, name), n_ok=4)
    monkeypatch.setattr(adcs.stepper, name, raiser)
    with pytest.raises(_Boom):
        sim.step(10)
    _assert_latched(sim)
    assert raiser.calls["n"] == 5  # failed on the 5th call; nothing was retried afterwards
    # result() still works and is internally consistent: whatever was logged (at most the 4 samples before)
    res = sim.result()
    n = len(res.t)
    assert 0 <= n <= 4
    for fld in ("q", "w", "h_w", "m", "u_w", "mode", "q_est", "P_att", "lit"):
        assert len(getattr(res, fld)) == n, fld
    assert res.meta["kicks"] == [] and res.meta["t_end"] == 30.0
    assert sim.result().t.shape == res.t.shape  # repeatable
    # the latch is permanent, also once the faulty dependency is healthy again
    monkeypatch.undo()
    _assert_latched(sim)


def test_failure_latch_after_a_good_chunk(monkeypatch):
    """The first chunk succeeds (and its snapshot works); the second one fails; the first rows are kept."""
    sim = _short(30.0, log_every=1)
    sim.step(3)
    before = sim.snapshot()
    res_before = sim.result()
    assert len(res_before.t) == 3 and before["k"] == 2
    raiser = _make_raiser(adcs.stepper.total_disturbance_torque, n_ok=2)  # 2 more good samples, then fails
    monkeypatch.setattr(adcs.stepper, "total_disturbance_torque", raiser)
    with pytest.raises(_Boom):
        sim.step(10)
    _assert_latched(sim)
    res = sim.result()
    # rows that were committed before the failing chunk are never lost
    assert len(res.t) >= 3
    assert _bits_equal(res.q[:3], res_before.q) and _bits_equal(res.w[:3], res_before.w)


def test_failure_does_not_corrupt_a_previously_returned_result(monkeypatch):
    sim = _short(30.0)
    sim.step(4)
    res = sim.result()
    ref = copy.deepcopy(res)
    monkeypatch.setattr(adcs.stepper, "rk4_step", _make_raiser(adcs.stepper.rk4_step, n_ok=0))
    with pytest.raises(_Boom):
        sim.step(5)
    assert sim.failed
    for fld in ("t", "q", "w", "h_w", "m", "u_w", "mode"):
        assert _bits_equal(getattr(res, fld), getattr(ref, fld)), fld


def test_validation_errors_do_not_latch():
    sim = _short(10.0)
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            sim.step(bad)
        assert sim.failed is False
    for bad in ([1.0, 2.0], [np.nan, 0.0, 0.0], [0.0, 0.0, np.inf], [0.7, 0.0, 0.0]):
        with pytest.raises(ValueError):
            sim.kick(bad)
        assert sim.failed is False
    # state untouched: the sim is exactly as good as a fresh one
    fresh = _short(10.0)
    assert sim.step(sim.remaining) == fresh.step(fresh.remaining) == 11
    assert _same_result(sim.result(), fresh.result())
    assert sim.failed is False and fresh.failed is False


def test_failed_is_false_on_new_and_finished_sims_and_after_horizon_errors():
    sim = _short(5.0)
    assert sim.failed is False
    sim.step(sim.remaining)
    assert sim.failed is False
    with pytest.raises(RuntimeError, match="horizon reached"):
        sim.step()  # past the horizon: a plain error, not a latch
    assert sim.failed is False
    assert sim.snapshot()["k"] == 5  # still readable
