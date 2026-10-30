"""Tests for ``webapp.server.adcs_session`` (spec: docs/specs/webapp_protocol.md sections 2, 3, 5.2).

Part 1 tests the pure frame builder ``frame(snapshot, sim, session_state)`` against independent numpy
/ scipy conversions of ``ClosedLoopSim.snapshot()``. Part 2 drives ``AdcsSession`` and
``SessionRegistry`` as plain asyncio code through a fake connection (no socket, no fastapi calls).

Wall-clock tests use generous timeouts: they run on Windows next to a busy suite.
"""
from __future__ import annotations

import asyncio
import copy
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from adcs.config import DEFAULT  # noqa: E402
from adcs.orbit import initial_state, orbital_period  # noqa: E402
from adcs.simulate import IDEAL_SENSORS, MEKF_G7, lvlh_state  # noqa: E402
from adcs.stepper import ClosedLoopSim  # noqa: E402
from adcs.visualize import DISPLAY_FLIP  # noqa: E402
from webapp.server.adcs_session import (AdcsSession, ConnectionClosed, SessionRegistry,  # noqa: E402
                                        SessionState, TooManySessions, frame)
from webapp.server.schemas import SessionParams, build_sim_kwargs  # noqa: E402
from webapp.server.tests._oracle import (FLIP, RTOL, assert_frame_matches, body_to_eci,  # noqa: E402
                                         lvlh_rows, numbers, without_seq)

REPO = Path(__file__).resolve().parents[3]
PERIOD = float(orbital_period(DEFAULT.orbit.a))
SHORT = SessionParams(orbits=0.05)                      # 278 control periods (T = 1 s)
SHORT_IDEAL = SessionParams(orbits=0.05, sensors="ideal", filter="g7")
STATE = SessionState(seq=3, playing=True, speed_set=250.0, speed_actual=187.25)


def _sim(params: SessionParams, n: int = 1) -> ClosedLoopSim:
    sim = ClosedLoopSim(**build_sim_kwargs(params))
    sim.step(n)
    return sim


# ---------------------------------------------------------------------------
# fixtures (module scoped: each sim costs ~1 s)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real0():
    """Realistic + design after sample 0: TRIAD has not initialised (estimator fields unavailable)."""
    sim = _sim(SHORT)
    return sim, sim.snapshot()


@pytest.fixture(scope="module")
def ideal():
    """Ideal + G7: snapshot at k = 0 (filter ready, no w_est yet) and at k = 60."""
    sim = _sim(SHORT_IDEAL)
    s0 = sim.snapshot()
    sim.step(60)
    return sim, s0, sim.snapshot()


@pytest.fixture(scope="module")
def at_q_ref():
    """One sample from exactly the nadir reference attitude (zero offset, zero relative rate)."""
    r0, v0 = initial_state(DEFAULT)
    y0 = lvlh_state(r0, v0)
    sim = ClosedLoopSim(y0, 0.05 * PERIOD, seed=0, sensors=IDEAL_SENSORS, mekf_cfg=MEKF_G7, log_every=10)
    sim.step(1)
    return sim, sim.snapshot()


# ===========================================================================
# 1. frame builder
# ===========================================================================
class TestFrameBuilder:
    def test_C_is_a_proper_rotation(self, real0, ideal):
        for sim, snap in ((real0[0], real0[1]), (ideal[0], ideal[1]), (ideal[0], ideal[2])):
            f = frame(snap, sim, STATE)
            keys = ("C", "C_est") if snap["q_est"] is not None else ("C",)
            for key in keys:
                C = np.array(f["display"][key])
                assert C.shape == (3, 3)
                # 1e-6: entries are rounded to 7 significant digits
                np.testing.assert_allclose(C @ C.T, np.eye(3), atol=1e-6)
                assert np.linalg.det(C) == pytest.approx(1.0, abs=1e-6)

    def test_C_orthonormal_when_estimator_unavailable(self, real0):
        sim, snap = real0
        C = np.array(frame(snap, sim, STATE)["display"]["C"])
        np.testing.assert_allclose(C @ C.T, np.eye(3), atol=1e-6)
        assert np.linalg.det(C) == pytest.approx(1.0, abs=1e-6)

    def test_q_equals_q_ref_gives_display_flip(self, at_q_ref):
        sim, snap = at_q_ref
        # the truth quaternion is the nadir reference (sanity of the fixture)
        np.testing.assert_allclose(snap["q"], snap["q_ref"], atol=1e-9)
        f = frame(snap, sim, STATE)
        C = np.array(f["display"]["C"])
        np.testing.assert_allclose(C, DISPLAY_FLIP, atol=1e-6)
        np.testing.assert_allclose(C, np.diag([1.0, -1.0, -1.0]), atol=1e-6)
        # body z (long axis) points down in the display frame: (0, 0, -1)
        np.testing.assert_allclose(C @ [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], atol=1e-6)
        # nominal pointing metrics
        assert f["metrics"]["nadir_err_deg"] == pytest.approx(0.0, abs=1e-4)

    def test_display_vectors_are_unit(self, real0, ideal, at_q_ref):
        for sim, snap in ((real0[0], real0[1]), (ideal[0], ideal[2]), (at_q_ref[0], at_q_ref[1])):
            d = frame(snap, sim, STATE)["display"]
            assert np.linalg.norm(d["sun"]) == pytest.approx(1.0, abs=1e-6)
            assert np.linalg.norm(d["b"]) == pytest.approx(1.0, abs=1e-6)

    def test_sun_at_zenith_is_display_up(self):
        # Sun along the position vector => zenith => display +z; along -r => nadir => display -z.
        r = np.array([6.778137e6, 0.0, 0.0])
        v = np.array([0.0, 4763.3, 6009.8])
        snap = _fake_snapshot(r=r, v=v, sun=r / np.linalg.norm(r))
        up = frame(snap, _fake_sim(), STATE)["display"]["sun"]
        np.testing.assert_allclose(up, [0.0, 0.0, 1.0], atol=1e-6)
        snap = _fake_snapshot(r=r, v=v, sun=-r / np.linalg.norm(r))
        np.testing.assert_allclose(frame(snap, _fake_sim(), STATE)["display"]["sun"], [0.0, 0.0, -1.0],
                                   atol=1e-6)

    def test_display_vectors_are_the_body_vectors_rotated_by_C(self, real0, ideal):
        """Convention-free consistency: C^T maps display vectors back to the body-frame vector R(q)^T x."""
        for sim, snap in ((real0[0], real0[1]), (ideal[0], ideal[2])):
            f = frame(snap, sim, STATE)
            C = np.array(f["display"]["C"])
            R = body_to_eci(snap["q"])
            r_hat = snap["r_eci"] / np.linalg.norm(snap["r_eci"])
            s_hat = snap["sun_eci"] / np.linalg.norm(snap["sun_eci"])
            b_hat = snap["b_eci"] / np.linalg.norm(snap["b_eci"])
            np.testing.assert_allclose(C.T @ f["display"]["sun"], R.T @ s_hat, atol=1e-6)
            np.testing.assert_allclose(C.T @ f["display"]["b"], R.T @ b_hat, atol=1e-6)
            # nadir in the display frame is (0, 0, -1): in body axes that is R^T (-r_hat)
            np.testing.assert_allclose(C.T @ [0.0, 0.0, -1.0], R.T @ (-r_hat), atol=1e-6)

    def test_flip_is_applied_not_a_mirror(self, real0):
        sim, snap = real0
        C = np.array(frame(snap, sim, STATE)["display"]["C"])
        assert np.linalg.det(C) > 0
        # C = FLIP @ C_lvlh, so FLIP @ C is the LVLH-axes DCM, also a proper rotation
        C_lvlh = FLIP @ C
        np.testing.assert_allclose(C_lvlh, lvlh_rows(snap["r_eci"], snap["v_eci"]) @ body_to_eci(snap["q"]),
                                   atol=1e-6)

    def test_estimator_fields_null_before_triad(self, real0):
        sim, snap = real0
        assert snap["q_est"] is None     # fixture sanity: estimator genuinely unavailable
        f = frame(snap, sim, STATE)
        assert f["eci"]["q_est"] is None
        assert f["display"]["C_est"] is None
        m = f["metrics"]
        for key in ("est_nadir_err_deg", "know_err_deg", "att_sigma_deg", "w_est_deg_s"):
            assert m[key] is None, key
        assert f["events"]["t_init"] is None
        # the truth-side fields are still present
        assert f["eci"]["q"] is not None and f["display"]["C"] is not None
        assert m["nadir_err_deg"] is not None and m["rate_deg_s"] > 0.0

    def test_estimator_fields_present_after_init(self, ideal):
        sim, s0, s60 = ideal
        f0 = frame(s0, sim, STATE)
        # TRIAD initialises at the first sample for ideal sensors; no gyro bias-corrected rate yet
        assert f0["eci"]["q_est"] is not None and f0["display"]["C_est"] is not None
        assert f0["metrics"]["w_est_deg_s"] is None
        assert f0["events"]["t_init"] == 0.0
        f60 = frame(s60, sim, STATE)
        m = f60["metrics"]
        for key in ("est_nadir_err_deg", "know_err_deg", "att_sigma_deg"):
            assert isinstance(m[key], float) and m[key] >= 0.0, key
        assert len(m["w_est_deg_s"]) == 3
        assert len(f60["eci"]["q_est"]) == 4

    @pytest.mark.parametrize("which", ["real0", "ideal0", "ideal60"])
    def test_values_match_snapshot_converted_independently(self, request, which):
        if which == "real0":
            sim, snap = request.getfixturevalue("real0")
        else:
            sim, s0, s60 = request.getfixturevalue("ideal")
            snap = s0 if which == "ideal0" else s60
        f = frame(snap, sim, STATE)
        assert_frame_matches(f, snap, PERIOD)

    def test_header_fields_come_from_session_state(self, real0):
        sim, snap = real0
        f = frame(snap, sim, STATE)
        assert f["type"] == "frame"
        assert f["seq"] == 3
        assert f["playing"] is True
        assert f["speed_set"] == 250.0
        assert f["speed_actual"] == pytest.approx(187.25, rel=RTOL)
        assert f["done"] is False
        assert f["k"] == 0 and f["t"] == 0.0 and f["orbit"] == 0.0
        f2 = frame(snap, sim, SessionState(seq=0, playing=False, speed_set=2000.0, speed_actual=0.0))
        assert f2["playing"] is False and f2["seq"] == 0 and f2["speed_set"] == 2000.0

    def test_done_flag_follows_the_sim(self, reference):
        sim, snaps = reference.sim, reference.snaps
        assert sim.done
        f = frame(snaps["end"], sim, SessionState(playing=False))
        assert f["done"] is True
        assert f["k"] == sim.n_steps and f["t"] == sim.t_end
        assert f["orbit"] == pytest.approx(sim.t_end / PERIOD, rel=RTOL)

    def test_mode_dump_lit_and_event_mapping(self, ideal):
        sim, _s0, s60 = ideal
        snap = dict(s60, mode=1, dump=True, lit=False, sun_used=2,
                    switch_times=[100.0, 250.123456789], fallback_times=[300.0])
        f = frame(snap, sim, STATE)
        assert (f["mode"], f["mode_name"], f["dump"], f["lit"]) == (1, "POINTING", True, False)
        assert f["metrics"]["sun_used"] == 2
        assert f["events"]["switch_times"] == [100.0, 250.1235]
        assert f["events"]["fallback_times"] == [300.0]
        snap0 = dict(s60, mode=0, dump=False, lit=True)
        f0 = frame(snap0, sim, STATE)
        assert (f0["mode"], f0["mode_name"], f0["dump"], f0["lit"]) == (0, "DETUMBLE", False, True)

    def test_wheel_and_unit_conversions(self, ideal):
        sim, _s0, s60 = ideal
        wmax = 8000.0 * 2.0 * math.pi / 60.0            # rated wheel speed 8000 rpm, rad/s
        ws = np.array([0.25 * wmax, -0.5 * wmax, 0.1])
        snap = dict(s60, wheel_speed=ws, w=np.array([0.1, 0.0, -0.2]), nadir_err=np.deg2rad(12.5))
        m = frame(snap, sim, STATE)["metrics"]
        np.testing.assert_allclose(m["wheel_speed_rpm"], [2000.0, -4000.0, 0.1 * 60 / (2 * math.pi)], rtol=RTOL)
        assert m["wheel_frac"] == pytest.approx(0.5, rel=RTOL)
        assert m["rate_deg_s"] == pytest.approx(math.degrees(math.hypot(0.1, 0.2)), rel=RTOL)
        np.testing.assert_allclose(m["w_deg_s"], [5.729578, 0.0, -11.459156], rtol=RTOL, atol=0)
        assert m["nadir_err_deg"] == pytest.approx(12.5, rel=RTOL)

    def test_know_err_is_norm_of_the_vector(self, ideal):
        sim, _s0, s60 = ideal
        snap = dict(s60, know_err=np.deg2rad([3.0, 4.0, 0.0]))
        assert frame(snap, sim, STATE)["metrics"]["know_err_deg"] == pytest.approx(5.0, rel=RTOL)

    @pytest.mark.parametrize("which", ["real0", "ideal0", "ideal60"])
    def test_strict_json_finite_and_seven_digits(self, request, which):
        if which == "real0":
            sim, snap = request.getfixturevalue("real0")
        else:
            sim, s0, s60 = request.getfixturevalue("ideal")
            snap = s0 if which == "ideal0" else s60
        f = frame(snap, sim, STATE)
        text = json.dumps(f, allow_nan=False)           # raises on NaN / inf
        assert json.loads(text) == f
        nums = list(numbers(f))
        assert nums and all(math.isfinite(x) for x in nums)
        for x in nums:
            if isinstance(x, float):
                assert float(f"{x:.7g}") == x, x

    def test_nan_scalars_become_null_not_nan(self, ideal):
        sim, _s0, s60 = ideal
        snap = dict(s60, est_nadir_err=float("nan"), att_sigma=float("nan"), t_init=float("nan"),
                    know_err=None, w_est=None, q_est=None)
        f = frame(snap, sim, STATE)
        json.dumps(f, allow_nan=False)
        assert f["metrics"]["est_nadir_err_deg"] is None and f["metrics"]["att_sigma_deg"] is None
        assert f["events"]["t_init"] is None and f["eci"]["q_est"] is None

    def test_frame_is_pure(self, ideal):
        sim, _s0, s60 = ideal
        before = copy.deepcopy(s60)
        f1 = frame(s60, sim, STATE)
        f2 = frame(s60, sim, STATE)
        assert f1 == f2
        assert set(before) == set(s60)
        for k, v in before.items():
            if isinstance(v, np.ndarray):
                np.testing.assert_array_equal(s60[k], v)
            else:
                assert (s60[k] == v) or (isinstance(v, float) and math.isnan(v)), k

    def test_shapes(self, ideal):
        sim, _s0, s60 = ideal
        f = frame(s60, sim, STATE)
        assert [len(f["eci"][k]) for k in ("q", "q_est", "r_m", "v_m_s", "sun", "b_T")] == [4, 4, 3, 3, 3, 3]
        assert all(len(row) == 3 for row in f["display"]["C"]) and len(f["display"]["C"]) == 3
        assert len(f["display"]["C_est"]) == 3
        assert [len(f["metrics"][k]) for k in ("w_deg_s", "w_est_deg_s", "wheel_speed_rpm", "m_Am2", "u_w_Nm")] == [3] * 5
        assert f["events"]["kicks"] == []
        # the Earth is at the right distance and the satellite speed is orbital (independent: ~7.67 km/s)
        assert np.linalg.norm(f["eci"]["r_m"]) == pytest.approx(DEFAULT.orbit.a, rel=1e-3)
        assert np.linalg.norm(f["eci"]["v_m_s"]) == pytest.approx(7668.6, rel=1e-3)


def _fake_sim():
    return SimpleNamespace(period=PERIOD, done=False, wheel=None,
                           result=lambda: SimpleNamespace(meta={"kicks": []}))


def _fake_snapshot(r, v, sun) -> dict:
    """A hand-made snapshot with the body aligned to nothing in particular (only geometry matters)."""
    z3 = np.zeros(3)
    return {
        "k": 0, "t": 0.0, "q": np.array([1.0, 0.0, 0.0, 0.0]), "w": z3.copy(), "h_w": z3.copy(),
        "wheel_speed": z3.copy(), "rate": 0.0, "m": z3.copy(), "u_w": z3.copy(), "tau_dist": z3.copy(),
        "tau_mag": z3.copy(), "mode": 0, "dump": False, "lit": True, "sun_used": 0,
        "r_eci": np.asarray(r, float), "v_eci": np.asarray(v, float), "sun_eci": np.asarray(sun, float),
        "b_eci": np.array([1.0e-5, 2.0e-5, -3.0e-5]), "q_ref": np.array([1.0, 0.0, 0.0, 0.0]),
        "gmst": 1.0, "nadir_err": 0.0, "filter_ready": False, "q_est": None, "b_est": None,
        "bm_est": None, "w_est": None, "know_err": None, "att_sigma": float("nan"), "P_att": None,
        "est_nadir_err": float("nan"), "t_init": float("nan"), "switch_times": [], "fallback_times": [],
        "n_kicks": 0,
    }


# ===========================================================================
# 2. AdcsSession through a fake connection
# ===========================================================================
_GONE = object()


class FakeConn:
    """The connection shape ``AdcsSession.serve`` expects (send_text / receive_text / close)."""

    def __init__(self):
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.outbox: asyncio.Queue = asyncio.Queue()
        self.closed: tuple | None = None

    async def send_text(self, text):
        if self.closed is not None:
            raise ConnectionClosed("closed")
        self.outbox.put_nowait(text)

    async def receive_text(self):
        item = await self.inbox.get()
        if item is _GONE:
            raise ConnectionClosed("gone")
        return item

    async def close(self, code, reason=""):
        self.closed = (code, reason)
        self.inbox.put_nowait(_GONE)


class Peer:
    """A test client: runs ``session.serve`` on a fake connection and parses what it sends."""

    def __init__(self, session):
        self.conn = FakeConn()
        self.task = asyncio.create_task(session.serve(self.conn))

    def send(self, obj):
        self.conn.inbox.put_nowait(obj if isinstance(obj, str) else json.dumps(obj))

    async def recv(self, timeout=20.0):
        text = await asyncio.wait_for(self.conn.outbox.get(), timeout)
        assert "NaN" not in text and "Infinity" not in text
        return json.loads(text)

    async def recv_until(self, pred, timeout=20.0):
        end = time.monotonic() + timeout
        seen = []
        while True:
            msg = await self.recv(max(0.01, end - time.monotonic()))
            seen.append(msg)
            if pred(msg):
                return msg, seen

    async def collect(self, seconds):
        out = []
        end = time.monotonic() + seconds
        while (left := end - time.monotonic()) > 0:
            try:
                out.append(await self.recv(left))
            except asyncio.TimeoutError:
                break
        return out

    async def hello_and_first_frame(self):
        hello = await self.recv()
        first = await self.recv()
        return hello, first

    async def disconnect(self):
        self.conn.inbox.put_nowait(_GONE)
        await asyncio.wait_for(self.task, 20.0)


def run(coro, timeout=180.0):
    return asyncio.run(asyncio.wait_for(coro, timeout))


class TestSessionBasics:
    def test_connect_sends_hello_then_frame0_and_steps_deterministically(self, reference):
        async def main():
            session = await AdcsSession.create(SHORT)
            assert len(session.id) == 32 and int(session.id, 16) >= 0
            assert session.status()["k"] == 0 and session.status()["t"] == 0.0
            assert session.status()["connected"] is False
            peer = Peer(session)
            hello, first = await peer.hello_and_first_frame()
            assert hello["type"] == "hello" and hello["id"] == session.id
            assert hello["params"] == SHORT.model_dump()
            assert hello["info"]["n_steps"] == 278
            assert hello["status"]["k"] == 0 and hello["status"]["playing"] is False
            assert first["type"] == "frame" and first["seq"] == 0
            assert first["k"] == 0 and first["t"] == 0.0
            assert first["playing"] is False and first["done"] is False
            assert session.status()["connected"] is True
            assert_frame_matches(first, reference.snaps[0], PERIOD)

            peer.send({"cmd": "step", "n": 100})
            f = await peer.recv()
            assert f["type"] == "frame" and f["seq"] == 1
            assert f["k"] == 100 and f["t"] == 100.0
            assert_frame_matches(f, reference.snaps[100], PERIOD)
            assert session.status()["k"] == 100 and session.status()["t"] == 100.0

            # step 1 more: single-step and 100-step chunking are bit-identical in the engine, so
            # the frame at k = 101 equals what a second independent step gives
            peer.send({"cmd": "step", "n": 1})
            g = await peer.recv()
            assert g["k"] == 101 and g["seq"] == 2
            await peer.disconnect()
            assert session.status()["connected"] is False

        run(main())

    def test_play_streams_increasing_frames_until_done(self, reference):
        async def main():
            session = await AdcsSession.create(SHORT)
            peer = Peer(session)
            await peer.hello_and_first_frame()
            peer.send({"cmd": "speed", "value": 2000})
            peer.send({"cmd": "play"})
            done_frame, seen = await peer.recv_until(lambda m: m["type"] == "frame" and m["done"], timeout=30)
            frames = [m for m in seen if m["type"] == "frame"]
            assert not [m for m in seen if m["type"] == "error"]
            assert len(frames) >= 3
            ts = [m["t"] for m in frames]
            assert all(b > a for a, b in zip(ts, ts[1:])), ts
            assert [m["seq"] for m in frames] == list(range(1, len(frames) + 1))
            assert all(m["speed_set"] == 2000.0 for m in frames)
            assert all(m["playing"] for m in frames[:-1])
            # final frame: horizon reached, no longer playing, identical to the independent last sample
            assert done_frame["done"] is True and done_frame["playing"] is False
            assert done_frame["k"] == 278 and done_frame["t"] == 278.0
            assert_frame_matches(done_frame, reference.snaps["end"], PERIOD)
            # nothing more arrives, and the status agrees
            assert await peer.collect(0.4) == []
            st = session.status()
            assert st["done"] is True and st["playing"] is False and st["t"] == 278.0
            # play on a finished sim: a done frame is still sent
            peer.send({"cmd": "play"})
            again = await peer.recv()
            assert again["type"] == "frame" and again["done"] is True and again["playing"] is False
            assert again["t"] == 278.0
            await peer.disconnect()

        run(main())

    def test_pause_stops_stepping(self):
        async def main():
            session = await AdcsSession.create(SHORT)
            peer = Peer(session)
            await peer.hello_and_first_frame()
            peer.send({"cmd": "speed", "value": 60})
            peer.send({"cmd": "play"})
            m, _ = await peer.recv_until(lambda m: m["type"] == "frame" and m["t"] > 3.0, timeout=30)
            assert m["playing"] is True
            peer.send({"cmd": "pause"})
            p, _ = await peer.recv_until(lambda m: m["type"] == "frame" and m["playing"] is False)
            assert p["done"] is False
            t_pause = p["t"]
            assert t_pause >= m["t"]
            late = await peer.collect(0.3)
            assert not [x for x in late if x["type"] == "frame" and x["t"] > t_pause]
            st = session.status()
            assert st["playing"] is False and st["t"] == t_pause
            # resume from where it stopped
            peer.send({"cmd": "play"})
            m2, _ = await peer.recv_until(lambda m: m["type"] == "frame" and m["t"] > t_pause)
            assert m2["playing"] is True
            peer.send({"cmd": "pause"})
            await peer.recv_until(lambda m: m["type"] == "frame" and m["playing"] is False)
            await peer.disconnect()

        run(main())

    def test_speed_actual_is_reported(self):
        async def main():
            session = await AdcsSession.create(SHORT)
            peer = Peer(session)
            await peer.hello_and_first_frame()
            peer.send({"cmd": "speed", "value": 40})
            peer.send({"cmd": "play"})
            frames = [m for m in await peer.collect(2.5) if m["type"] == "frame"]
            peer.send({"cmd": "pause"})
            assert len(frames) >= 5
            for m in frames:
                assert m["speed_set"] == 40.0
                assert m["speed_actual"] is not None and m["speed_actual"] >= 0.0
            # the sim cannot run faster than commanded (plus the integer-sample granularity and the
            # window start-up effect), and 40x is trivially affordable, so it should not lag far behind
            assert max(m["speed_actual"] for m in frames) < 2.0 * 40.0
            assert frames[-1]["speed_actual"] > 0.25 * 40.0
            # advanced sim time agrees with the wall clock to within a factor 2
            assert 0.5 * 40 * 1.0 < frames[-1]["t"] < 2.0 * 40 * 3.5
            await peer.disconnect()

        run(main())

    def test_kick_is_logged_applied_and_visible_in_frames(self, reference):
        async def main():
            session = await AdcsSession.create(SHORT)
            peer = Peer(session)
            await peer.hello_and_first_frame()
            peer.send({"cmd": "kick", "axis": "y", "deg_s": -4.0})
            f = await peer.recv()
            assert f["type"] == "frame" and f["k"] == 0 and f["seq"] == 1
            ks = f["events"]["kicks"]
            assert len(ks) == 1
            np.testing.assert_allclose(ks[0]["dw_deg_s"], [0.0, -4.0, 0.0], rtol=RTOL, atol=1e-12)
            assert ks[0]["t"] == pytest.approx(1.0)       # applies to the pending state at t_1
            meta_kicks = session.sim.result().meta["kicks"]
            assert len(meta_kicks) == 1
            np.testing.assert_allclose(meta_kicks[0]["dw"], np.deg2rad([0.0, -4.0, 0.0]), atol=1e-15)

            # the next sample logs the kicked truth rate: unkicked reference + dw (exact algebra)
            peer.send({"cmd": "kick", "axis": "x", "deg_s": 2.5})
            f2 = await peer.recv()
            assert len(f2["events"]["kicks"]) == 2
            np.testing.assert_allclose(f2["events"]["kicks"][1]["dw_deg_s"], [2.5, 0.0, 0.0], rtol=RTOL)
            peer.send({"cmd": "step", "n": 1})
            g = await peer.recv()
            assert g["k"] == 1
            w_expected = np.rad2deg(reference.snaps[1]["w"]) + np.array([2.5, -4.0, 0.0])
            # 7 significant digits of a ~10 deg/s number: absolute 1e-5 deg/s is ample
            np.testing.assert_allclose(g["metrics"]["w_deg_s"], w_expected, atol=2e-5, rtol=0)
            assert len(g["events"]["kicks"]) == 2
            assert len(session.sim.result().meta["kicks"]) == 2
            await peer.disconnect()

        run(main())

    def test_bad_commands_give_errors_and_keep_the_socket_usable(self):
        bad = [
            "this is not json", "[1, 2, 3]", "42", "null", "{}", '{"cmd": "warp"}', '{"cmd": 5}',
            {"cmd": "play", "extra": 1}, {"cmd": "pause", "now": True},
            {"cmd": "speed"}, {"cmd": "speed", "value": 0.5}, {"cmd": "speed", "value": 2001},
            {"cmd": "speed", "value": "fast"}, {"cmd": "speed", "value": True},
            {"cmd": "kick", "axis": "x", "deg_s": 0}, {"cmd": "kick", "axis": "x", "deg_s": 20.5},
            {"cmd": "kick", "axis": "x", "deg_s": -20.5}, {"cmd": "kick", "axis": "w", "deg_s": 1},
            {"cmd": "kick", "deg_s": 1}, {"cmd": "kick", "axis": "x"},
            {"cmd": "step", "n": 0}, {"cmd": "step", "n": -3}, {"cmd": "step", "n": 100001},
            {"cmd": "step", "n": 1.5}, {"cmd": "step", "n": "5"}, {"cmd": "step", "n": True},
            {"cmd": "step"}, {"cmd": "reset", "hard": True},
        ]

        async def main():
            session = await AdcsSession.create(SHORT)
            peer = Peer(session)
            await peer.hello_and_first_frame()
            for msg in bad:
                peer.send(msg)
                r = await peer.recv()
                assert r["type"] == "error", (msg, r)
                assert isinstance(r["message"], str) and r["message"]
            # nothing else was queued, the sim did not move and the socket still works
            assert session.status()["k"] == 0
            peer.send({"cmd": "pause"})
            f = await peer.recv()
            assert f["type"] == "frame" and f["playing"] is False and f["k"] == 0
            assert f["seq"] == 1                      # errors do not consume frame sequence numbers
            assert await peer.collect(0.2) == []
            await peer.disconnect()

        run(main())

    def test_valid_boundary_commands_are_accepted(self):
        async def main():
            session = await AdcsSession.create(SHORT)
            peer = Peer(session)
            await peer.hello_and_first_frame()
            for cmd in ({"cmd": "speed", "value": 1}, {"cmd": "speed", "value": 2000},
                        {"cmd": "speed", "value": 1234.5}):
                peer.send(cmd)
            peer.send({"cmd": "kick", "axis": "z", "deg_s": 20})
            r = await peer.recv()
            assert r["type"] == "frame" and r["speed_set"] == 1234.5
            peer.send({"cmd": "kick", "axis": "z", "deg_s": -20})
            assert (await peer.recv())["type"] == "frame"
            peer.send({"cmd": "kick", "axis": "x", "deg_s": 0.001})
            assert (await peer.recv())["type"] == "frame"
            await peer.disconnect()

        run(main())

    def test_step_rejected_while_playing_and_commands_during_streaming(self):
        async def main():
            session = await AdcsSession.create(SHORT)
            peer = Peer(session)
            await peer.hello_and_first_frame()
            peer.send({"cmd": "speed", "value": 30})
            peer.send({"cmd": "play"})
            peer.send({"cmd": "step", "n": 5})
            err, seen = await peer.recv_until(lambda m: m["type"] == "error")
            assert "pause" in err["message"].lower() or "play" in err["message"].lower() or err["message"]
            # kick and speed changes while streaming (lock discipline): no error, frames keep coming
            peer.send({"cmd": "kick", "axis": "z", "deg_s": 1.0})
            peer.send({"cmd": "speed", "value": 90})
            msgs = await peer.collect(1.2)
            assert not [m for m in msgs if m["type"] == "error"], msgs
            frames = [m for m in msgs if m["type"] == "frame"]
            assert any(len(m["events"]["kicks"]) == 1 for m in frames)
            assert frames[-1]["speed_set"] == 90.0
            ts = [m["t"] for m in frames]
            assert all(b >= a for a, b in zip(ts, ts[1:]))
            peer.send({"cmd": "pause"})
            p, _ = await peer.recv_until(lambda m: m["type"] == "frame" and m["playing"] is False)
            # a stream frame already in flight may add a second playing:false frame; drain it
            await peer.collect(0.3)
            k_paused = session.status()["k"]
            assert k_paused >= p["k"]
            # after the pause stepping is allowed again
            peer.send({"cmd": "step", "n": 2})
            s = await peer.recv()
            assert s["type"] == "frame" and s["k"] == k_paused + 2
            await peer.disconnect()

        run(main())

    def test_run_to_end_then_play_kick_and_step(self, reference):
        async def main():
            session = await AdcsSession.create(SHORT)
            peer = Peer(session)
            await peer.hello_and_first_frame()
            peer.send({"cmd": "step", "n": 100000})
            f = await peer.recv(timeout=60)
            assert f["type"] == "frame" and f["done"] is True and f["playing"] is False
            assert f["k"] == 278 and f["t"] == 278.0
            assert_frame_matches(f, reference.snaps["end"], PERIOD)
            assert session.sim.done
            peer.send({"cmd": "play"})
            g = await peer.recv()
            assert g["type"] == "frame" and g["done"] is True and g["playing"] is False
            peer.send({"cmd": "kick", "axis": "x", "deg_s": 1.0})
            e = await peer.recv()
            assert e["type"] == "error"
            peer.send({"cmd": "step", "n": 1})
            e2 = await peer.recv()
            assert e2["type"] == "error"
            assert session.sim.result().meta["kicks"] == []
            assert await peer.collect(0.2) == []
            await peer.disconnect()

        run(main())

    def test_reset_rebuilds_the_same_run(self):
        async def main():
            session = await AdcsSession.create(SHORT)
            peer = Peer(session)
            await peer.hello_and_first_frame()
            peer.send({"cmd": "step", "n": 100})
            first = await peer.recv()
            assert first["k"] == 100
            peer.send({"cmd": "kick", "axis": "x", "deg_s": 3.0})
            kicked = await peer.recv()
            assert len(kicked["events"]["kicks"]) == 1

            # reset while playing: pauses, then hello + frame k = 0 with the kick history cleared
            peer.send({"cmd": "speed", "value": 50})
            peer.send({"cmd": "play"})
            await peer.recv_until(lambda m: m["type"] == "frame" and m["playing"] is True)
            peer.send({"cmd": "reset"})
            hello, _ = await peer.recv_until(lambda m: m["type"] == "hello")
            assert hello["status"]["k"] == 0 and hello["status"]["t"] == 0.0
            assert hello["status"]["playing"] is False
            assert hello["params"] == SHORT.model_dump()
            f0 = await peer.recv()
            assert f0["type"] == "frame" and f0["k"] == 0 and f0["t"] == 0.0
            assert f0["playing"] is False and f0["events"]["kicks"] == []
            assert await peer.collect(0.3) == []         # paused: nothing else arrives
            assert session.sim.result().meta["kicks"] == []

            peer.send({"cmd": "step", "n": 100})
            again = await peer.recv()
            assert again["k"] == 100
            assert without_seq(again) == without_seq(first)      # exact reproduction
            await peer.disconnect()

        run(main())

    def test_second_client_replaces_the_first_and_sees_the_same_state(self):
        async def main():
            session = await AdcsSession.create(SHORT)
            a = Peer(session)
            await a.hello_and_first_frame()
            a.send({"cmd": "step", "n": 50})
            fa = await a.recv()
            assert fa["k"] == 50
            b = Peer(session)
            hello, first = await b.hello_and_first_frame()
            assert hello["status"]["k"] == 50 and first["k"] == 50 and first["seq"] == 0
            await asyncio.wait_for(a.task, 10)
            assert a.conn.closed is not None and a.conn.closed[0] == 4001
            assert b.conn.closed is None
            assert session.status()["connected"] is True
            b.send({"cmd": "step", "n": 1})
            assert (await b.recv())["k"] == 51
            await b.disconnect()
            assert session.status()["connected"] is False

        run(main())

    def test_disconnect_pauses_the_session(self):
        async def main():
            session = await AdcsSession.create(SHORT)
            peer = Peer(session)
            await peer.hello_and_first_frame()
            peer.send({"cmd": "speed", "value": 60})
            peer.send({"cmd": "play"})
            await peer.recv_until(lambda m: m["type"] == "frame" and m["t"] > 2.0, timeout=30)
            await peer.disconnect()
            st = session.status()
            assert st["connected"] is False and st["playing"] is False
            t1 = st["t"]
            await asyncio.sleep(0.4)
            assert session.status()["t"] == t1 and session.sim.t == pytest.approx(t1 + 1.0)
            # a reconnecting client gets the paused state
            again = Peer(session)
            hello, frm = await again.hello_and_first_frame()
            assert hello["status"]["playing"] is False and frm["playing"] is False and frm["t"] == t1
            await again.disconnect()

        run(main())

    def test_replaced_client_does_not_keep_the_session_streaming(self):
        async def main():
            session = await AdcsSession.create(SHORT)
            a = Peer(session)
            await a.hello_and_first_frame()
            a.send({"cmd": "speed", "value": 60})
            a.send({"cmd": "play"})
            await a.recv_until(lambda m: m["type"] == "frame" and m["t"] > 2.0, timeout=30)
            b = Peer(session)
            hello, _ = await b.hello_and_first_frame()
            assert hello["status"]["playing"] is False        # a new connection starts paused
            t1 = session.status()["t"]
            assert await b.collect(0.4) == []
            assert session.status()["t"] == t1
            await b.disconnect()

        run(main())


# ===========================================================================
# 3. registry
# ===========================================================================
class TestRegistry:
    def test_create_get_delete_and_limit_bookkeeping(self):
        async def main():
            reg = SessionRegistry(max_sessions=2)
            assert len(reg) == 0
            s = await reg.create(SHORT)
            assert reg.get(s.id) is s and len(reg) == 1
            assert reg.get("0" * 32) is None
            assert await reg.delete(s.id) is True
            assert await reg.delete(s.id) is False
            assert reg.get(s.id) is None and len(reg) == 0

        run(main())

    def test_eviction_is_least_recently_used_idle_session(self):
        async def main():
            reg = SessionRegistry(max_sessions=2)
            a = await reg.create(SHORT_IDEAL)
            b = await reg.create(SHORT_IDEAL)
            assert reg.get(a.id) is a                      # touch a: b is now the LRU session
            c = await reg.create(SHORT_IDEAL)
            assert reg.get(b.id) is None
            assert reg.get(a.id) is a and reg.get(c.id) is c and len(reg) == 2

        run(main())

    def test_sessions_with_a_client_are_never_evicted_and_429_when_all_busy(self):
        async def main():
            reg = SessionRegistry(max_sessions=2)
            a = await reg.create(SHORT_IDEAL)
            pa = Peer(a)
            await pa.hello_and_first_frame()              # a is the *older* session but has a client
            b = await reg.create(SHORT_IDEAL)
            assert reg.get(b.id) is b                     # b is now the most recently used
            c = await reg.create(SHORT_IDEAL)             # must evict the idle b, not the busy a
            assert reg.get(a.id) is a and reg.get(b.id) is None and reg.get(c.id) is c

            pc = Peer(c)
            await pc.hello_and_first_frame()
            with pytest.raises(TooManySessions):
                await reg.create(SHORT_IDEAL)
            assert len(reg) == 2 and reg.get(a.id) is a and reg.get(c.id) is c

            await pa.disconnect()                         # a becomes idle again => evictable
            d = await reg.create(SHORT_IDEAL)
            assert reg.get(a.id) is None and reg.get(d.id) is d
            await pc.disconnect()

        run(main())

    def test_delete_closes_the_connected_client(self):
        async def main():
            reg = SessionRegistry(max_sessions=2)
            s = await reg.create(SHORT_IDEAL)
            peer = Peer(s)
            await peer.hello_and_first_frame()
            peer.send({"cmd": "speed", "value": 50})
            peer.send({"cmd": "play"})
            await peer.recv_until(lambda m: m["type"] == "frame" and m["t"] > 1.0, timeout=30)
            assert await reg.delete(s.id)
            await asyncio.wait_for(peer.task, 10)
            assert peer.conn.closed is not None
            assert s.status()["playing"] is False

        run(main())

    def test_creation_does_not_block_the_event_loop(self):
        """6 orbits of precompute takes about a second; a heartbeat coroutine must keep ticking."""
        async def main():
            gaps = []
            stop = False

            async def heartbeat():
                last = time.perf_counter()
                while not stop:
                    await asyncio.sleep(0.005)
                    now = time.perf_counter()
                    gaps.append(now - last)
                    last = now

            reg = SessionRegistry(max_sessions=1)
            hb = asyncio.create_task(heartbeat())
            await asyncio.sleep(0.05)
            t0 = time.perf_counter()
            s = await reg.create(SessionParams(orbits=6.0))
            elapsed = time.perf_counter() - t0
            stop = True
            await hb
            assert s.info["n_steps"] == round(6 * PERIOD)
            # inline construction would produce one gap ~ elapsed; a worker thread leaves only small gaps
            assert max(gaps) < 0.25 * elapsed + 0.15, (max(gaps), elapsed)
            assert len(gaps) > 10

        run(main())


# ===========================================================================
# 4. module hygiene
# ===========================================================================
def test_session_and_schema_modules_do_not_import_fastapi():
    code = ("import sys, webapp.server.schemas, webapp.server.adcs_session; "
            "bad = [m for m in sys.modules if m == 'fastapi' or m.startswith('fastapi.') or m == 'starlette']; "
            "assert not bad, bad")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
