"""Scripted WebSocket client (spec section 5.5): ``python -m webapp.server.tests.ws_smoke``.

Starts uvicorn on a free port in a background thread, creates a default session over REST (httpx),
connects with the ``websockets`` client, sends ``speed 500`` and ``play``, collects frames for 2 s,
and prints the frame count, first/last ``t``, the last ``speed_actual`` and the mode. Exits nonzero
if fewer than 10 frames arrived or ``t`` was not strictly increasing. Not a pytest module.
"""
from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
import time

import httpx
import uvicorn
import websockets

from webapp.server.app import create_app

COLLECT_S = 2.0
MIN_FRAMES = 10
SPEED = 500


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _collect(port: int, sid: str) -> list[dict]:
    frames: list[dict] = []
    async with websockets.connect(f"ws://127.0.0.1:{port}/api/adcs/sessions/{sid}/ws") as ws:
        hello = json.loads(await ws.recv())
        assert hello["type"] == "hello", hello
        first = json.loads(await ws.recv())
        assert first["type"] == "frame", first
        await ws.send(json.dumps({"cmd": "speed", "value": SPEED}))
        await ws.send(json.dumps({"cmd": "play"}))
        deadline = time.monotonic() + COLLECT_S
        while (left := deadline - time.monotonic()) > 0:
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=left))
            except asyncio.TimeoutError:
                break
            if msg["type"] == "frame":
                frames.append(msg)
            elif msg["type"] == "error":
                print("server error:", msg["message"])
        await ws.send(json.dumps({"cmd": "pause"}))
    return frames


def main() -> int:
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 15.0
        while not server.started:
            if time.monotonic() > deadline or not thread.is_alive():
                print("server failed to start")
                return 2
            time.sleep(0.05)

        base = f"http://127.0.0.1:{port}"
        r = httpx.post(f"{base}/api/adcs/sessions", json={}, timeout=120.0)
        r.raise_for_status()
        sid = r.json()["id"]
        frames = asyncio.run(_collect(port, sid))
    finally:
        server.should_exit = True
        thread.join(timeout=10.0)

    ts = [f["t"] for f in frames]
    increasing = all(b > a for a, b in zip(ts, ts[1:]))
    print(f"frames: {len(frames)}")
    if frames:
        print(f"first t: {ts[0]:.1f} s, last t: {ts[-1]:.1f} s")
        print(f"last speed_actual: {frames[-1]['speed_actual']:.1f}x (set {SPEED}x)")
        print(f"mode: {frames[-1]['mode_name']}")
    print(f"t strictly increasing: {increasing}")
    ok = len(frames) >= MIN_FRAMES and increasing
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
