"""REST and WebSocket tests for ``webapp.server.app`` (spec: docs/specs/webapp_protocol.md sections 2-5).

Uses Starlette's ``TestClient`` as a context manager so that the whole test shares one event loop
(sessions hold asyncio state). WebSocket reads go through a pump thread with timeouts, so a server
bug cannot hang the suite. Wall-clock expectations are generous: the suite runs on Windows next to
other work.
"""
from __future__ import annotations

import contextlib
import json
import math
import queue
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
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from adcs.actuators import MagnetorquerConfig  # noqa: E402
from adcs.config import DEFAULT  # noqa: E402
from adcs.orbit import initial_state, orbital_period  # noqa: E402
from adcs.simulate import ControlConfig, ModeConfig  # noqa: E402
from common.frames import R_EARTH  # noqa: E402
from webapp.server.app import create_app  # noqa: E402
from webapp.server.schemas import SessionParams  # noqa: E402
from webapp.server.tests._oracle import RTOL, assert_frame_matches, numbers, without_seq  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
PERIOD = float(orbital_period(DEFAULT.orbit.a))
SHORT_BODY = {"orbits": 0.05}
DEFAULTS = {"orbits": 6.0, "rate_deg_s": 10.0, "seed": 0, "sensors": "realistic", "filter": "design",
            "dumping": True, "inertia_scale": 1.0}
HEX32 = re.compile(r"^[0-9a-f]{32}$")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class Closed(Exception):
    def __init__(self, code):
        super().__init__(f"websocket closed with code {code}")
        self.code = code


class Ws:
    """A TestClient websocket with timeout-protected reads (pump thread -> queue)."""

    def __init__(self, session):
        self.s = session
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        while True:
            try:
                text = self.s.receive_text()
            except WebSocketDisconnect as e:
                self.q.put(("close", e.code))
                return
            except Exception as e:                      # noqa: BLE001 - surface anything as a close
                self.q.put(("close", repr(e)))
                return
            self.q.put(("msg", text))

    def send(self, obj):
        self.s.send_text(obj if isinstance(obj, str) else json.dumps(obj))

    def recv(self, timeout=20.0):
        kind, val = self.q.get(timeout=timeout)
        if kind == "close":
            raise Closed(val)
        assert "NaN" not in val and "Infinity" not in val
        return json.loads(val)

    def recv_until(self, pred, timeout=20.0):
        end = time.monotonic() + timeout
        seen = []
        while True:
            m = self.recv(max(0.05, end - time.monotonic()))
            seen.append(m)
            if pred(m):
                return m, seen

    def collect(self, seconds):
        out, end = [], time.monotonic() + seconds
        while (left := end - time.monotonic()) > 0:
            try:
                out.append(self.recv(left))
            except queue.Empty:
                break
        return out

    def expect_close(self, timeout=20.0):
        """Read until the socket closes; return its close code."""
        end = time.monotonic() + timeout
        while True:
            try:
                self.recv(max(0.05, end - time.monotonic()))
            except Closed as c:
                return c.code


def open_ws(client, sid):
    return client.websocket_connect(f"/api/adcs/sessions/{sid}/ws")


def create(client, **body):
    r = client.post("/api/adcs/sessions", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def wait_for(cond, timeout=10.0, step=0.05):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(step)
    return cond()


# Class scope, not module scope: a TestClient context that stays open while a second TestClient
# opens a WebSocket (the TestLimit tests build their own apps) makes the second one fail on exit.
@pytest.fixture(scope="class")
def client():
    with TestClient(create_app(max_sessions=4)) as c:
        yield c


@pytest.fixture
def short(client):
    """A fresh 0.05-orbit session (default realistic + design), deleted afterwards."""
    s = create(client, **SHORT_BODY)
    yield s
    client.delete(f"/api/adcs/sessions/{s['id']}")


@pytest.fixture(scope="class")
def default_session(client):
    s = create(client)                                  # POST {} : the six-orbit default
    yield s
    client.delete(f"/api/adcs/sessions/{s['id']}")


# ===========================================================================
# REST
# ===========================================================================
class TestRest:
    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200 and r.json() == {"ok": True}

    def test_presets(self, client):
        r = client.get("/api/adcs/presets")
        assert r.status_code == 200
        body = r.json()
        assert body["defaults"] == DEFAULTS
        assert body["params_schema"] == SessionParams.model_json_schema()
        props = body["params_schema"]["properties"]
        assert set(props) == set(DEFAULTS)
        assert props["orbits"]["minimum"] == 0.05 and props["orbits"]["maximum"] == 15.0
        assert props["seed"]["maximum"] == 2**31 - 1
        assert body["params_schema"].get("additionalProperties") is False

    def test_create_default_session(self, default_session):
        s = default_session
        assert set(s) == {"id", "params", "info", "status"}
        assert HEX32.match(s["id"])
        assert s["params"] == DEFAULTS
        assert s["status"] == {"k": 0, "t": 0.0, "done": False, "playing": False, "speed": 50.0,
                               "connected": False}

    def test_info_constants(self, default_session):
        info = default_session["info"]
        kepler = 2.0 * math.pi * math.sqrt(DEFAULT.orbit.a ** 3 / 3.986004418e14)
        assert info["period_s"] == pytest.approx(kepler, rel=1e-6)
        assert info["period_s"] == pytest.approx(PERIOD, rel=1e-12)
        assert info["control_period_s"] == 1.0
        assert info["n_steps"] == round(6 * PERIOD)
        assert info["t_end_s"] == pytest.approx(info["n_steps"] * info["control_period_s"], abs=1e-9)
        assert info["epoch_utc"] == "2026-03-20T12:00:00"
        assert info["body_size_m"] == [0.10, 0.10, 0.34]
        assert info["wheel_max_speed_rad_s"] == pytest.approx(8000.0 * 2.0 * math.pi / 60.0, rel=1e-12)
        assert info["m_max_Am2"] == pytest.approx(MagnetorquerConfig().m_max)
        assert info["safe_rate_deg_s"] == pytest.approx(np.rad2deg(ModeConfig().safe_rate))
        assert info["detumble_threshold_deg_s"] == pytest.approx(np.rad2deg(ControlConfig().detumble_threshold))
        assert info["safe_rate_deg_s"] == pytest.approx(2.0) and info["detumble_threshold_deg_s"] == pytest.approx(0.5)
        assert set(info) == {"period_s", "t_end_s", "control_period_s", "n_steps", "epoch_utc", "body_size_m",
                             "wheel_max_speed_rad_s", "m_max_Am2", "safe_rate_deg_s",
                             "detumble_threshold_deg_s", "orbit_track_eci_m", "eclipses_s"}

    def test_orbit_track_closes_the_ring(self, client, default_session):
        info = default_session["info"]
        track = np.array(info["orbit_track_eci_m"])
        assert track.ndim == 2 and track.shape[1] == 3
        # first orbit every 30 s: ceil(5553 / 30) = 186 samples, plus the closing point
        assert abs(len(track) - (math.ceil(PERIOD / 30.0) + 1)) <= 1
        assert np.linalg.norm(track[-1] - track[0]) < 1000.0                      # closes to 1 km
        r0, _ = initial_state(DEFAULT)
        np.testing.assert_allclose(track[0], r0, atol=1000.0)
        # circular orbit of radius a (400 km altitude); J2 ripple is a few km
        assert np.all(np.abs(np.linalg.norm(track, axis=1) - DEFAULT.orbit.a) < 30e3)
        # spacing 30 s apart: chord = 2 a sin(n dt / 2) with the Kepler mean motion
        n = 2.0 * math.pi / PERIOD
        chord = 2.0 * DEFAULT.orbit.a * math.sin(n * 30.0 / 2.0)
        steps = np.linalg.norm(np.diff(track[:-1], axis=0), axis=1)
        np.testing.assert_allclose(steps, chord, rtol=0.01)
        # the track is planar (inclination 51.6 deg): all points orthogonal to one normal
        normal = np.cross(track[0], track[40])
        normal /= np.linalg.norm(normal)
        assert np.max(np.abs(track @ normal)) < 0.02 * DEFAULT.orbit.a
        ang = np.degrees(np.arccos(abs(normal[2])))
        assert ang == pytest.approx(51.6, abs=0.5)

    def test_eclipse_intervals(self, client, default_session):
        info = default_session["info"]
        ecl = info["eclipses_s"]
        assert 4 <= len(ecl) <= 8                          # ~one per orbit over six orbits
        flat = []
        for a, b in ecl:
            assert 0.0 <= a <= b <= info["t_end_s"]
            flat += [a, b]
        assert all(y > x for x, y in zip(flat, flat[1:]))   # ordered and disjoint (gaps between intervals)
        # independent recomputation of the maximal shadow intervals from the cylindrical shadow model
        sim = client.app.state.registry.get(default_session["id"]).sim
        r, s = np.asarray(sim.r_eci), np.asarray(sim.sun_eci)
        along = np.sum(r * s, axis=1)
        perp = np.linalg.norm(r - along[:, None] * s, axis=1)
        shadow = (along < 0.0) & (perp < R_EARTH)
        expected = []
        i, n = 0, len(shadow)
        while i < n:
            if shadow[i]:
                j = i
                while j + 1 < n and shadow[j + 1]:
                    j += 1
                expected.append([float(sim.t_grid[i]), float(sim.t_grid[j])])
                i = j + 1
            else:
                i += 1
        assert ecl == expected
        # beta ~ 0 at the March equinox: eclipses last roughly 35-40 % of the orbit
        frac = sum(b - a for a, b in ecl) / info["t_end_s"]
        assert 0.28 < frac < 0.45

    def test_get_session(self, client, default_session):
        r = client.get(f"/api/adcs/sessions/{default_session['id']}")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == default_session["id"]
        assert body["params"] == default_session["params"] and body["info"] == default_session["info"]
        assert body["status"]["k"] == 0 and body["status"]["connected"] is False

    def test_unknown_id_is_404(self, client):
        for sid in ("0" * 32, "nope", "x" * 5):
            assert client.get(f"/api/adcs/sessions/{sid}").status_code == 404
            assert client.delete(f"/api/adcs/sessions/{sid}").status_code == 404

    def test_delete_then_get(self, client):
        s = create(client, **SHORT_BODY)
        r = client.delete(f"/api/adcs/sessions/{s['id']}")
        assert r.status_code == 204 and r.content == b""
        assert client.get(f"/api/adcs/sessions/{s['id']}").status_code == 404
        assert client.delete(f"/api/adcs/sessions/{s['id']}").status_code == 404

    def test_custom_params_are_echoed(self, client):
        body = {"orbits": 0.05, "rate_deg_s": 3.5, "seed": 7, "sensors": "ideal", "filter": "g7",
                "dumping": False, "inertia_scale": 1.25}
        s = create(client, **body)
        try:
            assert s["params"] == body
            assert s["info"]["n_steps"] == 278
        finally:
            client.delete(f"/api/adcs/sessions/{s['id']}")

    @pytest.mark.parametrize("body", [
        {"orbits": 0.04}, {"orbits": 0}, {"orbits": 15.5}, {"orbits": -1},
        {"rate_deg_s": -1}, {"rate_deg_s": 30.5},
        {"seed": -1}, {"seed": 2**31}, {"seed": 1.5},
        {"inertia_scale": 0.4}, {"inertia_scale": 1.6},
        {"sensors": "perfect"}, {"filter": "kalman"}, {"dumping": "maybe"},
        {"orbits": "six"}, {"orbits": True}, {"rate_deg_s": True}, {"orbits": None},
        {"bogus": 1}, {"orbits": 1.0, "extra": "x"},
    ])
    def test_bad_params_are_422(self, client, body):
        n_before = len(client.app.state.registry)
        r = client.post("/api/adcs/sessions", json=body)
        assert r.status_code == 422, r.text
        assert len(client.app.state.registry) == n_before

    def test_non_object_bodies_are_422(self, client):
        assert client.post("/api/adcs/sessions", json=[1, 2]).status_code == 422
        assert client.post("/api/adcs/sessions", content=b"not json",
                           headers={"content-type": "application/json"}).status_code == 422

    def test_event_loop_stays_responsive_while_a_session_is_built(self, client):
        """A health request must be answerable while a six-orbit POST is being processed."""
        results = {}

        def post():
            results["post"] = client.post("/api/adcs/sessions", json={"orbits": 6.0})

        th = threading.Thread(target=post)
        th.start()
        try:
            time.sleep(0.15)                                     # the POST is inside its precompute now
            worst = 0.0
            for _ in range(5):
                t0 = time.perf_counter()
                assert client.get("/api/health").status_code == 200
                worst = max(worst, time.perf_counter() - t0)
        finally:
            th.join(timeout=60)
        assert results["post"].status_code == 201
        client.delete(f"/api/adcs/sessions/{results['post'].json()['id']}")
        assert worst < 1.0, worst


# ===========================================================================
# session limit and eviction
# ===========================================================================
class TestLimit:
    def test_429_when_every_session_has_a_client_then_slot_frees_up(self):
        with TestClient(create_app(max_sessions=1)) as c:
            a = create(c, **SHORT_BODY)
            with open_ws(c, a["id"]) as s:
                ws = Ws(s)
                assert ws.recv()["type"] == "hello"
                ws.recv()
                r = c.post("/api/adcs/sessions", json=SHORT_BODY)
                assert r.status_code == 429, r.text
                assert c.get(f"/api/adcs/sessions/{a['id']}").status_code == 200   # a was not evicted
                assert c.get(f"/api/adcs/sessions/{a['id']}").json()["status"]["connected"] is True
            # client gone: the session becomes idle and may be evicted for a new one
            assert wait_for(lambda: c.get(f"/api/adcs/sessions/{a['id']}").json()["status"]["connected"] is False)
            b = create(c, **SHORT_BODY)
            assert b["id"] != a["id"]
            assert c.get(f"/api/adcs/sessions/{a['id']}").status_code == 404
            assert c.get(f"/api/adcs/sessions/{b['id']}").status_code == 200

    def test_idle_session_is_evicted_least_recently_used_first(self):
        with TestClient(create_app(max_sessions=2)) as c:
            a = create(c, **SHORT_BODY)
            b = create(c, **SHORT_BODY)
            assert c.get(f"/api/adcs/sessions/{a['id']}").status_code == 200   # touch a: b is now the LRU
            d = create(c, **SHORT_BODY)
            assert c.get(f"/api/adcs/sessions/{b['id']}").status_code == 404
            assert c.get(f"/api/adcs/sessions/{a['id']}").status_code == 200
            assert c.get(f"/api/adcs/sessions/{d['id']}").status_code == 200

    def test_default_limit_is_four(self):
        with TestClient(create_app()) as c:
            ids = [create(c, orbits=0.05, sensors="ideal", filter="g7")["id"] for _ in range(4)]
            with contextlib.ExitStack() as stack:             # every socket is closed even if one exit raises
                for sid in ids:
                    ws = Ws(stack.enter_context(open_ws(c, sid)))
                    ws.recv()
                    ws.recv()
                assert c.post("/api/adcs/sessions", json=SHORT_BODY).status_code == 429


# ===========================================================================
# WebSocket
# ===========================================================================
class TestWebSocket:
    def test_unknown_id_closes_with_4404(self, client):
        with open_ws(client, "f" * 32) as s:
            ws = Ws(s)
            assert ws.expect_close() == 4404

    def test_connect_gives_hello_then_frame_zero(self, client, short, reference):
        with open_ws(client, short["id"]) as s:
            ws = Ws(s)
            hello = ws.recv()
            assert hello["type"] == "hello"
            assert set(hello) == {"type", "id", "params", "info", "status"}
            assert hello["id"] == short["id"] and hello["params"] == short["params"]
            assert hello["info"] == short["info"]
            assert hello["status"]["connected"] is True and hello["status"]["k"] == 0
            f = ws.recv()
            assert f["type"] == "frame" and f["seq"] == 0
            assert f["k"] == 0 and f["t"] == 0
            assert f["playing"] is False and f["done"] is False and f["speed_set"] == 50.0
            assert_frame_matches(f, reference.snaps[0], PERIOD)
            assert client.get(f"/api/adcs/sessions/{short['id']}").json()["status"]["connected"] is True
            assert ws.collect(0.3) == []                     # connected but paused: nothing streams

    def test_step_100_matches_independent_engine(self, client, short, reference):
        with open_ws(client, short["id"]) as s:
            ws = Ws(s)
            ws.recv()
            ws.recv()
            ws.send({"cmd": "step", "n": 100})
            f = ws.recv()
            assert f["type"] == "frame" and f["seq"] == 1
            assert f["k"] == 100 and f["t"] == 100
            # 7-significant-digit transmission => relative 1e-6 against the independent engine run
            assert_frame_matches(f, reference.snaps[100], PERIOD)
            assert client.get(f"/api/adcs/sessions/{short['id']}").json()["status"]["k"] == 100

    def test_play_at_2000x_streams_increasing_frames(self, client, short):
        with open_ws(client, short["id"]) as s:
            ws = Ws(s)
            ws.recv()
            ws.recv()
            ws.send({"cmd": "speed", "value": 2000})
            ws.send({"cmd": "play"})
            frames = []
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and len(frames) < 3:
                m = ws.recv(max(0.05, deadline - time.monotonic()))
                assert m["type"] == "frame", m
                frames.append(m)
            assert len(frames) >= 3
            ts = [m["t"] for m in frames]
            assert all(b > a for a, b in zip(ts, ts[1:])), ts
            assert [m["seq"] for m in frames] == list(range(1, len(frames) + 1))
            assert all(m["speed_set"] == 2000 for m in frames)
            ws.send({"cmd": "pause"})
            ws.recv_until(lambda m: m["type"] == "frame" and m["playing"] is False)

    def test_pause_freezes_time(self, client, short):
        with open_ws(client, short["id"]) as s:
            ws = Ws(s)
            ws.recv()
            ws.recv()
            ws.send({"cmd": "speed", "value": 60})
            ws.send({"cmd": "play"})
            m, _ = ws.recv_until(lambda m: m["type"] == "frame" and m["t"] > 3.0, timeout=30)
            assert m["playing"] is True
            ws.send({"cmd": "pause"})
            p, _ = ws.recv_until(lambda m: m["type"] == "frame" and m["playing"] is False)
            t_pause = p["t"]
            late = ws.collect(0.3)
            assert not [x for x in late if x["type"] == "frame" and x["t"] > t_pause]
            st = client.get(f"/api/adcs/sessions/{short['id']}").json()["status"]
            assert st["playing"] is False and st["t"] == t_pause
            time.sleep(0.3)
            assert client.get(f"/api/adcs/sessions/{short['id']}").json()["status"]["t"] == t_pause

    def test_kick_is_reported_applied_and_logged(self, client, short, reference):
        with open_ws(client, short["id"]) as s:
            ws = Ws(s)
            ws.recv()
            ws.recv()
            ws.send({"cmd": "kick", "axis": "y", "deg_s": -4.0})
            f = ws.recv()
            assert f["type"] == "frame" and f["k"] == 0
            assert len(f["events"]["kicks"]) == 1
            np.testing.assert_allclose(f["events"]["kicks"][0]["dw_deg_s"], [0.0, -4.0, 0.0], rtol=RTOL, atol=1e-12)
            sim = client.app.state.registry.get(short["id"]).sim
            logged = sim.result().meta["kicks"]
            assert len(logged) == 1
            np.testing.assert_allclose(logged[0]["dw"], np.deg2rad([0.0, -4.0, 0.0]), atol=1e-15)
            # the next sample sees the kicked truth rate: independent unkicked rate + dw
            ws.send({"cmd": "step", "n": 1})
            g = ws.recv()
            w_expected = np.rad2deg(reference.snaps[1]["w"]) + np.array([0.0, -4.0, 0.0])
            np.testing.assert_allclose(g["metrics"]["w_deg_s"], w_expected, atol=2e-5, rtol=0)
            assert len(g["events"]["kicks"]) == 1

    def test_bad_commands_return_errors_and_the_socket_survives(self, client, short):
        bad = ["not json at all", "[]", "{}", '{"cmd": "launch"}', {"cmd": "play", "x": 1},
               {"cmd": "speed", "value": 0}, {"cmd": "speed", "value": 5000},
               {"cmd": "kick", "axis": "q", "deg_s": 1}, {"cmd": "kick", "axis": "x", "deg_s": 0},
               {"cmd": "kick", "axis": "x", "deg_s": 25}, {"cmd": "step", "n": 0},
               {"cmd": "step", "n": 100001}, {"cmd": "step", "n": "ten"}]
        with open_ws(client, short["id"]) as s:
            ws = Ws(s)
            ws.recv()
            ws.recv()
            for msg in bad:
                ws.send(msg)
                r = ws.recv()
                assert r["type"] == "error", (msg, r)
                assert isinstance(r["message"], str) and r["message"]
            ws.send({"cmd": "step", "n": 3})
            f = ws.recv()
            assert f["type"] == "frame" and f["k"] == 3

    def test_step_while_playing_is_an_error(self, client, short):
        with open_ws(client, short["id"]) as s:
            ws = Ws(s)
            ws.recv()
            ws.recv()
            ws.send({"cmd": "speed", "value": 30})
            ws.send({"cmd": "play"})
            ws.send({"cmd": "step", "n": 10})
            err, _ = ws.recv_until(lambda m: m["type"] == "error")
            assert err["message"]
            ws.send({"cmd": "pause"})
            ws.recv_until(lambda m: m["type"] == "frame" and m["playing"] is False)

    def test_run_to_end_done_play_and_kick(self, client, short, reference):
        with open_ws(client, short["id"]) as s:
            ws = Ws(s)
            ws.recv()
            ws.recv()
            ws.send({"cmd": "step", "n": 100000})
            f = ws.recv(timeout=60)
            assert f["type"] == "frame" and f["done"] is True and f["playing"] is False
            assert f["k"] == 278 and f["t"] == 278
            assert_frame_matches(f, reference.snaps["end"], PERIOD)
            ws.send({"cmd": "play"})
            g = ws.recv()
            assert g["type"] == "frame" and g["done"] is True and g["playing"] is False
            ws.send({"cmd": "kick", "axis": "x", "deg_s": 1})
            assert ws.recv()["type"] == "error"
            ws.send({"cmd": "step", "n": 1})
            assert ws.recv()["type"] == "error"
            st = client.get(f"/api/adcs/sessions/{short['id']}").json()["status"]
            assert st["done"] is True and st["k"] == 278 and st["playing"] is False

    def test_reset_then_step_reproduces_the_first_run(self, client, short):
        with open_ws(client, short["id"]) as s:
            ws = Ws(s)
            ws.recv()
            ws.recv()
            ws.send({"cmd": "step", "n": 100})
            first = ws.recv()
            ws.send({"cmd": "kick", "axis": "z", "deg_s": 5})
            assert len(ws.recv()["events"]["kicks"]) == 1
            ws.send({"cmd": "reset"})
            hello = ws.recv()
            assert hello["type"] == "hello" and hello["status"]["k"] == 0
            f0 = ws.recv()
            assert f0["type"] == "frame" and f0["k"] == 0 and f0["t"] == 0
            assert f0["playing"] is False and f0["events"]["kicks"] == []
            ws.send({"cmd": "step", "n": 100})
            again = ws.recv()
            assert without_seq(again) == without_seq(first)

    def test_second_connection_replaces_the_first_with_4001(self, client, short):
        with open_ws(client, short["id"]) as s1:
            ws1 = Ws(s1)
            ws1.recv()
            ws1.recv()
            ws1.send({"cmd": "step", "n": 10})
            assert ws1.recv()["k"] == 10
            with open_ws(client, short["id"]) as s2:
                ws2 = Ws(s2)
                hello = ws2.recv()
                assert hello["type"] == "hello" and hello["status"]["k"] == 10
                f = ws2.recv()
                assert f["type"] == "frame" and f["k"] == 10 and f["seq"] == 0
                assert ws1.expect_close() == 4001
                ws2.send({"cmd": "step", "n": 1})
                assert ws2.recv()["k"] == 11
                assert client.get(f"/api/adcs/sessions/{short['id']}").json()["status"]["connected"] is True

    def test_disconnect_pauses_the_session(self, client, short):
        with open_ws(client, short["id"]) as s:
            ws = Ws(s)
            ws.recv()
            ws.recv()
            ws.send({"cmd": "speed", "value": 80})
            ws.send({"cmd": "play"})
            ws.recv_until(lambda m: m["type"] == "frame" and m["t"] > 2.0, timeout=30)
        url = f"/api/adcs/sessions/{short['id']}"
        assert wait_for(lambda: client.get(url).json()["status"]["connected"] is False)
        assert wait_for(lambda: client.get(url).json()["status"]["playing"] is False)
        t1 = client.get(url).json()["status"]["t"]
        time.sleep(0.4)
        assert client.get(url).json()["status"]["t"] == t1
        # reconnect: the session is still there, paused, at the same time
        with open_ws(client, short["id"]) as s:
            ws = Ws(s)
            hello = ws.recv()
            assert hello["status"]["playing"] is False and hello["status"]["t"] == t1
            assert ws.recv()["t"] == t1

    def test_delete_closes_a_connected_socket(self, client):
        sess = create(client, **SHORT_BODY)
        with open_ws(client, sess["id"]) as s:
            ws = Ws(s)
            ws.recv()
            ws.recv()
            assert client.delete(f"/api/adcs/sessions/{sess['id']}").status_code == 204
            ws.expect_close()
        assert client.get(f"/api/adcs/sessions/{sess['id']}").status_code == 404

    def test_all_streamed_frames_are_strict_json(self, client, short):
        with open_ws(client, short["id"]) as s:
            ws = Ws(s)
            ws.recv()
            ws.recv()
            ws.send({"cmd": "speed", "value": 500})
            ws.send({"cmd": "play"})
            m, seen = ws.recv_until(lambda m: m["type"] == "frame" and m["done"], timeout=30)
            for f in seen:
                assert f["type"] == "frame"
                assert all(math.isfinite(x) for x in numbers(f))
                # TestClient re-serialising with allow_nan=False must succeed
                json.dumps(f, allow_nan=False)


# ===========================================================================
# static frontend, launcher, scripted client
# ===========================================================================
class TestStaticAndLauncher:
    def test_root_without_a_built_frontend(self, tmp_path):
        for dist in (None, tmp_path / "missing", tmp_path):          # None, absent, empty directory
            with TestClient(create_app(frontend_dist=dist)) as c:
                r = c.get("/")
                assert r.status_code == 200
                assert r.headers["content-type"].startswith("text/html")
                assert "frontend not built" in r.text
                assert "npm run build" in r.text
                assert c.get("/api/health").json() == {"ok": True}

    def test_built_frontend_is_served_after_the_api_routes(self, tmp_path):
        (tmp_path / "index.html").write_text("<html><body>BUILT-FRONTEND-MARKER</body></html>")
        (tmp_path / "app.js").write_text("console.log('x');")
        with TestClient(create_app(frontend_dist=tmp_path)) as c:
            r = c.get("/")
            assert r.status_code == 200 and "BUILT-FRONTEND-MARKER" in r.text
            assert "frontend not built" not in r.text
            assert c.get("/app.js").status_code == 200
            # API routes win over the static mount
            assert c.get("/api/health").json() == {"ok": True}
            assert c.get("/api/adcs/presets").status_code == 200
            assert c.get("/api/adcs/sessions/" + "0" * 32).status_code == 404

    def test_launcher_arguments_and_defaults(self, monkeypatch, capsys):
        import uvicorn
        from webapp.__main__ import main
        calls = []
        monkeypatch.setattr(uvicorn, "run", lambda app, host, port, **kw: calls.append((app, host, port)))
        main([])
        app, host, port = calls[-1]
        assert (host, port) == ("127.0.0.1", 8000)               # local-only by default
        assert hasattr(app, "routes")                            # a FastAPI app
        assert "http://127.0.0.1:8000" in capsys.readouterr().out
        main(["--host", "0.0.0.0", "--port", "9123"])
        assert calls[-1][1:] == ("0.0.0.0", 9123)
        assert "9123" in capsys.readouterr().out

    def test_launcher_help_runs(self):
        out = subprocess.run([sys.executable, "-m", "webapp", "--help"], cwd=REPO, capture_output=True,
                              text=True, timeout=120)
        assert out.returncode == 0, out.stderr[-1500:]
        assert "--host" in out.stdout and "--port" in out.stdout and "127.0.0.1" in out.stdout

    def test_scripted_websocket_client_over_real_uvicorn(self):
        """Exit criterion: ``python -m webapp.server.tests.ws_smoke`` (uvicorn + websockets client)."""
        pytest.importorskip("uvicorn")
        pytest.importorskip("websockets")
        out = subprocess.run([sys.executable, "-m", "webapp.server.tests.ws_smoke"], cwd=REPO,
                             capture_output=True, text=True, timeout=240)
        assert out.returncode == 0, out.stdout[-1500:] + out.stderr[-1500:]
        m = re.search(r"frames:\s*(\d+)", out.stdout)
        assert m and int(m.group(1)) >= 10, out.stdout
        assert "t strictly increasing: True" in out.stdout
