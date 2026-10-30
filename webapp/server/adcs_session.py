"""ADCS session: one ``ClosedLoopSim`` plus its streaming loop, and the session registry.

Spec: ``docs/specs/webapp_protocol.md`` sections 2 and 3. A visual aid only; it adds no numbers to
any report and does no physics of its own: every display quantity comes from ``adcs.visualize`` and
``common.frames``. Units: SI and radians internally; the wire format uses degrees only in fields
named ``*_deg`` / ``*_deg_s`` (and rpm for wheel speeds).

This module imports no web framework. The WebSocket is any object with ``send_text(str)``,
``receive_text()`` and ``close(code, reason)`` coroutines that raises :class:`ConnectionClosed` when
the peer is gone (``app.py`` adapts Starlette's ``WebSocket`` to that shape), so the session logic is
testable as plain asyncio code.

Concurrency: the ``ClosedLoopSim`` is not thread-safe. Every sim access (step, kick, snapshot,
frame build) and every send happens under one ``asyncio.Lock``; stepping and construction run in
``asyncio.to_thread``.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from adcs.actuators import MagnetorquerConfig, WheelConfig
from adcs.config import DEFAULT
from adcs.simulate import ControlConfig, ModeConfig
from adcs.stepper import ClosedLoopSim
from adcs.visualize import DISPLAY_FLIP, body_dcm_lvlh, sun_lvlh
from common.frames import dcm_eci_to_lvlh
from webapp.server.schemas import (CommandError, KickCmd, PauseCmd, PlayCmd, ResetCmd, SessionParams,
                                   SpeedCmd, StepCmd, build_sim_kwargs, parse_command)

__all__ = ["AdcsSession", "SessionRegistry", "SessionState", "ConnectionClosed", "TooManySessions",
           "frame", "build_info", "TICK_S", "DEFAULT_SPEED"]

TICK_S = 1.0 / 30.0          # target streaming period, s (wall)
MAX_DT_WALL = 0.1            # s, clamp on the wall time credited per tick (no catch-up bursts)
BATCH_WALL_S = 0.1           # s of compute per batch the step cap aims for
RATE_EMA = 0.3               # weight of the newest measurement in the step-rate average
INITIAL_N_MAX = 50           # samples per batch before any step rate has been measured
SPEED_WINDOW_S = 1.0         # s, window of the achieved-speed estimate
DEFAULT_SPEED = 50.0         # sim-seconds per wall-second
TRACK_DT_S = 30.0            # s between orbit-track points
_RAD2RPM = 60.0 / (2.0 * math.pi)
_MODE_NAMES = {0: "DETUMBLE", 1: "POINTING"}
_AXES = {"x": 0, "y": 1, "z": 2}


class ConnectionClosed(Exception):
    """The peer closed (or lost) the WebSocket."""


class TooManySessions(Exception):
    """``max_sessions`` sessions exist and every one has a connected client."""


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------
def _r(x) -> float | None:
    """Round to 7 significant digits; NaN and infinities become ``None`` (JSON null)."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return float(f"{x:.7g}")


def _rv(a) -> list:
    """7-significant-digit nested list of a numpy array (NaN -> None), row-major."""
    arr = np.asarray(a, float)
    if arr.ndim == 0:
        return _r(arr)
    if arr.ndim == 1:
        return [_r(v) for v in arr]
    return [_rv(row) for row in arr]


def _opt(a) -> list | None:
    return None if a is None else _rv(a)


def _opt_deg(x) -> float | None:
    """Degrees of an angle in rad, or ``None`` for ``None`` / NaN."""
    if x is None:
        return None
    x = float(x)
    return None if not math.isfinite(x) else _r(np.rad2deg(x))


# ---------------------------------------------------------------------------
# Frame builder (pure)
# ---------------------------------------------------------------------------
@dataclass
class SessionState:
    """The per-session numbers a frame needs besides the sim snapshot."""

    seq: int = 0
    playing: bool = False
    speed_set: float = DEFAULT_SPEED
    speed_actual: float = 0.0
    kicks: list = field(default_factory=list)   # [{"t", "dw_deg_s"}]; None -> read from the sim


def _state_get(state, name: str, default, *aliases: str):
    """Read ``name`` (or an alias) from a mapping or an attribute object."""
    for key in (name, *aliases):
        if isinstance(state, Mapping):
            if key in state:
                return state[key]
        elif hasattr(state, key):
            return getattr(state, key)
    return default


def _kicks_from_sim(sim) -> list:
    return [{"t": _r(k["t"]), "dw_deg_s": _rv(np.rad2deg(np.asarray(k["dw"], float)))}
            for k in sim.result().meta.get("kicks", [])]


def frame(snapshot: dict, sim, session_state) -> dict:
    """Build the JSON-ready ``frame`` message for ``snapshot`` (spec section 3.2).

    ``snapshot`` is ``sim.snapshot()``; ``sim`` supplies ``period``, ``done`` and (only when the
    state carries no kick list) the kick records; ``session_state`` is a :class:`SessionState` or a
    mapping with ``seq``, ``playing``, ``speed_set`` (alias ``speed``), ``speed_actual`` and
    optionally ``kicks``. Display frame: ``p_display = DISPLAY_FLIP @ p_lvlh`` (zenith up, nadir
    ``(0, 0, -1)``); ``C = DISPLAY_FLIP @ body_dcm_lvlh(q, r, v)`` is the local-to-world rotation
    of a body-fixed mesh. Angles in degrees only in ``*_deg*`` fields; floats carry 7 significant
    digits and non-finite values are ``null``.
    """
    s = snapshot
    r, v = s["r_eci"], s["v_eci"]
    q, q_est = s["q"], s["q_est"]
    C = DISPLAY_FLIP @ body_dcm_lvlh(q, r, v)
    C_est = None if q_est is None else DISPLAY_FLIP @ body_dcm_lvlh(q_est, r, v)
    sun_d = DISPLAY_FLIP @ sun_lvlh(s["sun_eci"], r, v)
    b = np.asarray(s["b_eci"], float)
    b_d = DISPLAY_FLIP @ (dcm_eci_to_lvlh(r, v) @ (b / np.linalg.norm(b)))

    wheel = getattr(sim, "wheel", None)
    wheel_max = float(getattr(wheel, "max_speed", WheelConfig().max_speed))
    ws = np.asarray(s["wheel_speed"], float)
    know = s["know_err"]
    w_est = s["w_est"]
    mode = int(s["mode"])
    t_init = float(s["t_init"])

    kicks = _state_get(session_state, "kicks", None)
    if kicks is None:
        kicks = _kicks_from_sim(sim) if s.get("n_kicks", 0) else []

    return {
        "type": "frame",
        "seq": int(_state_get(session_state, "seq", 0)),
        "k": int(s["k"]),
        "t": _r(s["t"]),
        "orbit": _r(float(s["t"]) / float(sim.period)),
        "mode": mode,
        "mode_name": _MODE_NAMES.get(mode, str(mode)),
        "dump": bool(s["dump"]),
        "lit": bool(s["lit"]),
        "done": bool(sim.done),
        "playing": bool(_state_get(session_state, "playing", False)),
        "speed_set": _r(_state_get(session_state, "speed_set", DEFAULT_SPEED, "speed")),
        "speed_actual": _r(_state_get(session_state, "speed_actual", 0.0)),
        "eci": {
            "q": _rv(q), "q_est": _opt(q_est),
            "r_m": _rv(r), "v_m_s": _rv(v), "sun": _rv(s["sun_eci"]), "b_T": _rv(b),
            "gmst_rad": _r(s["gmst"]),
        },
        "display": {"C": _rv(C), "C_est": _opt(C_est), "sun": _rv(sun_d), "b": _rv(b_d)},
        "metrics": {
            "nadir_err_deg": _opt_deg(s["nadir_err"]),
            "est_nadir_err_deg": _opt_deg(s["est_nadir_err"]),
            "know_err_deg": None if know is None else _opt_deg(np.linalg.norm(know)),
            "att_sigma_deg": _opt_deg(s["att_sigma"]),
            "rate_deg_s": _opt_deg(np.linalg.norm(s["w"])),
            "w_deg_s": _rv(np.rad2deg(s["w"])),
            "w_est_deg_s": None if w_est is None else _rv(np.rad2deg(w_est)),
            "wheel_speed_rpm": _rv(ws * _RAD2RPM),
            "wheel_frac": _r(np.max(np.abs(ws)) / wheel_max),
            "m_Am2": _rv(s["m"]),
            "u_w_Nm": _rv(s["u_w"]),
            "sun_used": int(s["sun_used"]),
        },
        "events": {
            "t_init": None if not math.isfinite(t_init) else _r(t_init),
            "switch_times": [_r(x) for x in s["switch_times"]],
            "fallback_times": [_r(x) for x in s["fallback_times"]],
            "kicks": [dict(k) for k in kicks],
        },
    }


# ---------------------------------------------------------------------------
# Static session info
# ---------------------------------------------------------------------------
def build_info(sim: ClosedLoopSim) -> dict:
    """Static per-session description (spec section 2): horizon, constants, orbit track, eclipses.

    ``orbit_track_eci_m`` is ``sim.r_eci`` over the first orbit every 30 s (m, ECI), closed by
    repeating the first point; ``eclipses_s`` lists the first and last shadow sample time (s) of
    each maximal shadow interval over the horizon.
    """
    t_grid = np.asarray(sim.t_grid, float)
    n = len(t_grid)
    idx = np.round(np.arange(0.0, sim.period, TRACK_DT_S) / sim.T).astype(int)
    idx = np.unique(idx[idx < n])
    track = np.asarray(sim.r_eci, float)[idx]
    track = np.vstack([track, track[:1]])

    shadow = ~np.asarray(sim.lit, bool)
    d = np.diff(np.concatenate([[0], shadow.astype(int), [0]]))
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1
    eclipses = [[float(t_grid[a]), float(t_grid[b])] for a, b in zip(starts, ends)]

    return {
        "period_s": float(sim.period),
        "t_end_s": float(sim.t_end),
        "control_period_s": float(sim.T),
        "n_steps": int(sim.n_steps),
        "epoch_utc": DEFAULT.epoch.isoformat(),
        "body_size_m": [float(x) for x in DEFAULT.spacecraft.size],
        "wheel_max_speed_rad_s": float(WheelConfig().max_speed),
        "m_max_Am2": float(MagnetorquerConfig().m_max),
        "safe_rate_deg_s": float(np.rad2deg(ModeConfig().safe_rate)),
        "detumble_threshold_deg_s": float(np.rad2deg(ControlConfig().detumble_threshold)),
        "orbit_track_eci_m": _rv(track),
        "eclipses_s": eclipses,
    }


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------
def _make_sim(params: SessionParams) -> ClosedLoopSim:
    """Build the sim and process sample 0 so that a snapshot always exists (runs in a thread)."""
    sim = ClosedLoopSim(**build_sim_kwargs(params))
    sim.step(1)
    return sim


def _timed_step(sim: ClosedLoopSim, n: int) -> tuple[int, float]:
    """``sim.step(n)`` returning ``(samples processed, wall seconds spent)`` (runs in a thread)."""
    t0 = time.perf_counter()
    done = sim.step(n)
    return int(done), time.perf_counter() - t0


class _Client:
    """One connected peer: the connection, its frame counter and a close signal."""

    def __init__(self, conn) -> None:
        self.conn = conn
        self.seq = 0
        self.closed = asyncio.Event()


class AdcsSession:
    """One ``ClosedLoopSim`` with its streaming loop and (at most one) connected client."""

    def __init__(self, id: str, params: SessionParams, sim: ClosedLoopSim) -> None:
        self.id = id
        self.params = params
        self.sim = sim
        self.info = build_info(sim)
        self.playing = False
        self.speed = DEFAULT_SPEED
        self.speed_actual = 0.0
        self.last_used = time.monotonic()
        self._client: _Client | None = None
        self._kicks: list = []
        self._lock: asyncio.Lock | None = None
        self._lock_loop = None
        self._task: asyncio.Task | None = None
        self._wake: asyncio.Event | None = None
        self._rate: float | None = None           # EMA of measured samples per wall second
        self._window: deque = deque()             # (wall s, cumulative sim s) for speed_actual
        self._advanced = 0.0                      # cumulative sim seconds stepped while playing
        self._snap: dict | None = None
        self._k, self._t, self._done = self._progress()

    # ------------------------------------------------------------------ construction

    @classmethod
    async def create(cls, params: SessionParams, id: str | None = None) -> "AdcsSession":
        """Build the sim in a thread and process sample 0; ``id`` defaults to ``uuid4().hex``."""
        sim = await asyncio.to_thread(_make_sim, params)
        return cls(id or uuid.uuid4().hex, params, sim)

    # ------------------------------------------------------------------ status

    def _progress(self) -> tuple[int, float, bool]:
        snap = self.sim.snapshot()
        self._snap = snap                    # kept so frames still work once the sim has failed
        return int(snap["k"]), float(snap["t"]), bool(self.sim.done)

    @property
    def connected(self) -> bool:
        return self._client is not None

    def status(self) -> dict:
        """Dynamic status (spec section 2); read from a cache refreshed under the lock."""
        return {"k": self._k, "t": self._t, "done": self._done, "playing": self.playing,
                "speed": float(self.speed), "connected": self.connected}

    def touch(self) -> None:
        self.last_used = time.monotonic()

    def _get_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:   # tolerate a new event loop (tests)
            self._lock, self._lock_loop = asyncio.Lock(), loop
        return self._lock

    # ------------------------------------------------------------------ frames

    def _state(self, client: _Client | None) -> SessionState:
        return SessionState(seq=0 if client is None else client.seq, playing=self.playing,
                            speed_set=self.speed,
                            speed_actual=self.speed_actual if self.playing else 0.0,
                            kicks=list(self._kicks))

    def _frame(self, client: _Client | None) -> dict:
        """Frame for the current snapshot (the last good one if the sim has failed). Caller holds the lock."""
        snap = self._snap if self.sim.failed and self._snap is not None else self.sim.snapshot()
        return frame(snap, self.sim, self._state(client))

    def _hello(self) -> dict:
        return {"type": "hello", "id": self.id, "params": self.params.model_dump(), "info": self.info,
                "status": self.status()}

    @staticmethod
    async def _send(client: _Client, msg: dict) -> None:
        """Send one message; frames get this connection's next ``seq``. Caller holds the lock."""
        if msg.get("type") == "frame":
            msg["seq"] = client.seq
            client.seq += 1
        await client.conn.send_text(json.dumps(msg, allow_nan=False, separators=(",", ":")))

    async def _send_error(self, client: _Client, message: str) -> None:
        async with self._get_lock():
            await self._send(client, {"type": "error", "message": message})

    # ------------------------------------------------------------------ streaming

    def _note_rate(self, n: int, elapsed: float) -> None:
        if n > 0 and elapsed > 0.0:
            rate = n / elapsed
            self._rate = rate if self._rate is None else (1.0 - RATE_EMA) * self._rate + RATE_EMA * rate

    def _n_max(self) -> int:
        if self._rate is None:
            return INITIAL_N_MAX
        return max(1, math.ceil(BATCH_WALL_S * self._rate))

    def _speed_actual(self, now: float) -> float:
        """Sim seconds per wall second over the last ~1 s (the oldest point inside the window and
        one point before it as baseline)."""
        w = self._window
        w.append((now, self._advanced))
        while len(w) > 2 and w[1][0] <= now - SPEED_WINDOW_S:
            w.popleft()
        dt = now - w[0][0]
        return (self._advanced - w[0][1]) / dt if dt > 0.0 else 0.0

    async def _sleep(self, delay: float) -> None:
        if delay <= 0.0:
            await asyncio.sleep(0)
            return
        try:
            await asyncio.wait_for(self._wake.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass

    def _start_stream(self, client: _Client) -> None:
        self.playing = True
        self._wake = asyncio.Event()
        self._task = asyncio.create_task(self._stream(client))

    async def _halt_stream(self) -> None:
        """Stop the streaming task (it finishes its current tick first); sets ``playing = False``."""
        self.playing = False
        task, self._task = self._task, None
        if task is not None and not task.done():
            self._wake.set()
            await task

    async def _stream(self, client: _Client) -> None:
        sim = self.sim
        T = sim.T
        budget = 0.0
        now = time.perf_counter()
        last, next_tick = now, now + TICK_S
        self._window.clear()
        self._window.append((now, self._advanced))
        try:
            while True:
                await self._sleep(next_tick - time.perf_counter())
                if not self.playing:
                    return
                now = time.perf_counter()
                dt_wall = min(now - last, MAX_DT_WALL)
                last = now
                next_tick = max(next_tick + TICK_S, now)
                budget += self.speed * dt_wall
                n = int(budget // T)
                if n < 1:
                    continue
                capped = n > self._n_max()
                n = min(n, self._n_max())
                async with self._get_lock():
                    if not self.playing:
                        return
                    count, elapsed = await asyncio.to_thread(_timed_step, sim, min(n, sim.remaining))
                    self._note_rate(count, elapsed)
                    self._advanced += count * T
                    budget = 0.0 if capped else budget - n * T
                    self._k, self._t, self._done = self._progress()
                    if self._done:
                        self.playing = False
                    self.speed_actual = self._speed_actual(time.perf_counter())
                    await self._send(client, self._frame(client))
                    if self._done:
                        return
        except ConnectionClosed:
            self.playing = False
        except asyncio.CancelledError:
            self.playing = False
            raise
        except Exception as e:                                   # keep the server alive
            self.playing = False
            try:
                msg = ("simulation failed; reset to build a new one" if self.sim.failed
                       else f"stream stopped: {e!r}")
                await self._send_error(client, msg)
            except Exception:
                pass

    # ------------------------------------------------------------------ connection

    async def serve(self, conn) -> None:
        """Serve one client until it disconnects or is replaced; pauses the session on exit."""
        client = _Client(conn)
        old, self._client = self._client, client
        if old is not None:                                     # a second client replaces the first
            await self._halt_stream()
            old.closed.set()
            try:
                await old.conn.close(4001, "replaced")
            except Exception:
                pass
        await self._halt_stream()
        self.touch()
        try:
            async with self._get_lock():
                await self._send(client, self._hello())
                await self._send(client, self._frame(client))
            while True:
                raw = await self._receive(client)
                self.touch()
                await self._handle(client, raw)
        except ConnectionClosed:
            pass
        finally:
            # Clear the client before any await: a cancellation during _halt_stream must not
            # leave the session marked connected (unevictable).
            client.closed.set()
            if self._client is client:
                self._client = None
                self.playing = False
                await self._halt_stream()

    @staticmethod
    async def _receive(client: _Client):
        """Next client message, or :class:`ConnectionClosed` if the peer left or was replaced."""
        recv = asyncio.ensure_future(client.conn.receive_text())
        closed = asyncio.ensure_future(client.closed.wait())
        try:
            await asyncio.wait({recv, closed}, return_when=asyncio.FIRST_COMPLETED)
            if recv.done():
                return recv.result()                            # raises ConnectionClosed on disconnect
            raise ConnectionClosed("replaced")
        finally:
            # Cancel only; awaiting the cancelled helpers lets a CancelledError escape when the
            # ASGI server cancels the scope right after the disconnect.
            for f in (recv, closed):
                if not f.done():
                    f.cancel()

    async def shutdown(self) -> None:
        """Stop streaming and close the client (session deleted or server stopping)."""
        await self._halt_stream()
        client, self._client = self._client, None
        if client is not None:
            client.closed.set()
            try:
                await client.conn.close(1000, "session closed")
            except Exception:
                pass

    # ------------------------------------------------------------------ commands

    async def _handle(self, client: _Client, raw: str) -> None:
        try:
            cmd = parse_command(raw)
        except CommandError as e:
            await self._send_error(client, str(e))
            return
        if isinstance(cmd, SpeedCmd):
            self.speed = float(cmd.value)
        elif isinstance(cmd, PlayCmd):
            await self._cmd_play(client)
        elif isinstance(cmd, PauseCmd):
            await self._halt_stream()
            async with self._get_lock():
                await self._send(client, self._frame(client))
        elif isinstance(cmd, KickCmd):
            await self._cmd_kick(client, cmd)
        elif isinstance(cmd, StepCmd):
            await self._cmd_step(client, cmd)
        elif isinstance(cmd, ResetCmd):
            await self._cmd_reset(client)

    async def _cmd_play(self, client: _Client) -> None:
        if self.playing:
            return
        async with self._get_lock():
            if self.sim.done:
                self.playing = False
                await self._send(client, self._frame(client))
                return
        self._start_stream(client)

    async def _cmd_kick(self, client: _Client, cmd: KickCmd) -> None:
        async with self._get_lock():
            if self.sim.done:
                await self._send(client, {"type": "error", "message": "horizon reached: cannot kick"})
                return
            dw = np.zeros(3)
            dw[_AXES[cmd.axis]] = np.deg2rad(cmd.deg_s)
            try:
                self.sim.kick(dw)
            except (ValueError, RuntimeError) as e:       # rate cap, last sample, failed sim
                await self._send(client, {"type": "error", "message": str(e)})
                return
            self._kicks = _kicks_from_sim(self.sim)
            await self._send(client, self._frame(client))

    async def _cmd_step(self, client: _Client, cmd: StepCmd) -> None:
        if self.playing:
            await self._send_error(client, "step is only allowed while paused")
            return
        async with self._get_lock():
            if self.sim.done:
                await self._send(client, {"type": "error", "message": "horizon reached: cannot step"})
                return
            if self.sim.failed:
                await self._send(client, {"type": "error", "message": "simulation failed; reset to build a new one"})
                return
            try:
                count, elapsed = await asyncio.to_thread(_timed_step, self.sim, min(cmd.n, self.sim.remaining))
            except Exception as e:                        # the sim latches `failed`; keep the socket open
                msg = ("simulation failed; reset to build a new one" if self.sim.failed
                       else f"step failed: {e!r}")
                await self._send(client, {"type": "error", "message": msg})
                return
            self._note_rate(count, elapsed)
            self._k, self._t, self._done = self._progress()
            await self._send(client, self._frame(client))

    async def _cmd_reset(self, client: _Client) -> None:
        await self._halt_stream()
        async with self._get_lock():
            self.sim = await asyncio.to_thread(_make_sim, self.params)
            self._kicks = []
            self.speed_actual, self._advanced = 0.0, 0.0
            self._window.clear()
            self._k, self._t, self._done = self._progress()
            await self._send(client, self._hello())
            await self._send(client, self._frame(client))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class SessionRegistry:
    """At most ``max_sessions`` sessions; creating more evicts the least recently used idle one."""

    def __init__(self, max_sessions: int = 4) -> None:
        self.max_sessions = int(max_sessions)
        self._sessions: dict[str, AdcsSession] = {}
        self._pending = 0

    def __len__(self) -> int:
        return len(self._sessions)

    def get(self, id: str) -> AdcsSession | None:
        s = self._sessions.get(id)
        if s is not None:
            s.touch()
        return s

    def ids(self) -> list[str]:
        return list(self._sessions)

    async def create(self, params: SessionParams) -> AdcsSession:
        """Create a session (sim built in a thread). Raises :class:`TooManySessions` (-> HTTP 429)."""
        victim = None
        if len(self._sessions) + self._pending >= self.max_sessions:
            idle = [s for s in self._sessions.values() if not s.connected]
            if not idle:
                raise TooManySessions(f"{self.max_sessions} sessions all have a connected client")
            victim = min(idle, key=lambda s: s.last_used)
            del self._sessions[victim.id]
        self._pending += 1                      # reserve the slot while the sim is built
        try:
            if victim is not None:
                await victim.shutdown()
            session = await AdcsSession.create(params)
        finally:
            self._pending -= 1
        self._sessions[session.id] = session
        return session

    async def delete(self, id: str) -> bool:
        s = self._sessions.pop(id, None)
        if s is None:
            return False
        await s.shutdown()
        return True

    async def close_all(self) -> None:
        for id in list(self._sessions):
            await self.delete(id)
