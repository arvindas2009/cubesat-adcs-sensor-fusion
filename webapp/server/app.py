"""FastAPI application for the browser simulator (spec: docs/specs/webapp_protocol.md sections 2-4).

A visual aid only. REST under ``/api``, one WebSocket per ADCS session, and the built frontend (if
present) mounted at ``/`` after the API routes. Binds to localhost by default (``python -m webapp``);
there is no authentication.
"""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response, WebSocket
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from webapp.server.adcs_session import ConnectionClosed, SessionRegistry, TooManySessions
from webapp.server.px4_replay import CAVEAT, DOWNLOAD_HINT, LogMissing, ReplayStore, UnknownLog
from webapp.server.schemas import SessionParams

__all__ = ["create_app", "DEFAULT_FRONTEND_DIST"]

DEFAULT_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

_NOT_BUILT = (
    "<!doctype html><html><head><meta charset='utf-8'><title>ADCS simulator</title></head><body>"
    "<h1>ADCS simulator server</h1>"
    "<p>frontend not built: cd webapp/frontend &amp;&amp; npm ci &amp;&amp; npm run build</p>"
    "<p>The API is running: <a href='/api/health'>/api/health</a>, "
    "<a href='/docs'>/docs</a>.</p></body></html>"
)


class _FrontendFiles(StaticFiles):
    """Built frontend with explicit caching: Vite's content-hashed ``assets/*`` are immutable; every
    other file (``index.html``, textures) must be revalidated, so a rebuild is picked up on reload."""

    async def get_response(self, path: str, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = ("public, max-age=31536000, immutable"
                                         if path.replace("\\", "/").startswith("assets/") else "no-cache")
        return resp


class _WsConn:
    """Adapt a Starlette ``WebSocket`` to the shape ``AdcsSession.serve`` expects."""

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws

    async def send_text(self, text: str) -> None:
        try:
            await self.ws.send_text(text)
        except Exception as e:                      # disconnected peer: Starlette/uvicorn raise several types
            raise ConnectionClosed(str(e)) from e

    async def receive_text(self) -> str:
        try:
            msg = await self.ws.receive()
        except Exception as e:
            raise ConnectionClosed(str(e)) from e
        if msg["type"] == "websocket.disconnect":
            raise ConnectionClosed(f"disconnect {msg.get('code')}")
        if msg.get("text") is not None:
            return msg["text"]
        return (msg.get("bytes") or b"").decode("utf-8", errors="replace")

    async def close(self, code: int, reason: str = "") -> None:
        try:
            await self.ws.close(code=code, reason=reason)
        except Exception:
            pass


def create_app(frontend_dist: Path | None = DEFAULT_FRONTEND_DIST, max_sessions: int = 4,
               px4_store: ReplayStore | None = None) -> FastAPI:
    """Build the application.

    ``frontend_dist``: directory of the built frontend (mounted at ``/`` when it contains
    ``index.html``; otherwise ``GET /`` returns a short note). ``max_sessions``: concurrent ADCS
    sessions; the registry is ``app.state.registry``. ``px4_store``: the PX4 replay store
    (``None`` -> ``ReplayStore()``; ``app.state.px4``); spec ``docs/specs/px4_replay.md`` section 2.
    """
    registry = SessionRegistry(max_sessions=max_sessions)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await registry.close_all()

    app = FastAPI(title="ADCS browser simulator", lifespan=lifespan)
    app.state.registry = registry
    app.state.px4 = ReplayStore() if px4_store is None else px4_store
    app.add_middleware(GZipMiddleware, minimum_size=1024)       # HTTP only; WebSockets are untouched

    def _session_json(s) -> dict:
        return {"id": s.id, "params": s.params.model_dump(), "info": s.info, "status": s.status()}

    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True}

    @app.get("/api/adcs/presets")
    async def presets() -> dict:
        return {"params_schema": SessionParams.model_json_schema(), "defaults": SessionParams().model_dump()}

    @app.post("/api/adcs/sessions", status_code=201)
    async def create_session(params: SessionParams) -> JSONResponse:
        try:
            s = await registry.create(params)
        except TooManySessions as e:
            raise HTTPException(status_code=429, detail=str(e)) from None
        return JSONResponse(_session_json(s), status_code=201)

    @app.get("/api/adcs/sessions/{sid}")
    async def get_session(sid: str) -> JSONResponse:
        s = registry.get(sid)
        if s is None:
            raise HTTPException(status_code=404, detail="unknown session")
        return JSONResponse(_session_json(s))

    @app.delete("/api/adcs/sessions/{sid}", status_code=204)
    async def delete_session(sid: str) -> Response:
        if not await registry.delete(sid):
            raise HTTPException(status_code=404, detail="unknown session")
        return Response(status_code=204)

    @app.websocket("/api/adcs/sessions/{sid}/ws")
    async def session_ws(websocket: WebSocket, sid: str) -> None:
        await websocket.accept()
        s = registry.get(sid)
        if s is None:
            await websocket.close(code=4404, reason="unknown session")
            return
        await s.serve(_WsConn(websocket))

    @app.get("/api/px4/logs")
    async def px4_logs() -> dict:
        store = app.state.px4
        return {"logs": await asyncio.to_thread(store.list_logs), "download_hint": DOWNLOAD_HINT,
                "caveat": CAVEAT}

    @app.post("/api/px4/replay/{log}")
    async def px4_replay(log: str) -> Response:
        store = app.state.px4
        try:
            text = await asyncio.to_thread(store.get_json, log)
        except UnknownLog:
            return JSONResponse({"detail": f"unknown log '{log}'"}, status_code=404)
        except LogMissing as e:
            return JSONResponse({"detail": str(e)}, status_code=404)
        except Exception as e:                       # loader / filter failure: report, keep serving
            return JSONResponse({"detail": f"replay failed: {e}"}, status_code=500)
        return Response(content=text, media_type="application/json")

    dist = None if frontend_dist is None else Path(frontend_dist)
    if dist is not None and (dist / "index.html").is_file():
        app.mount("/", _FrontendFiles(directory=dist, html=True), name="frontend")
    else:
        @app.get("/", response_class=HTMLResponse)
        async def frontend_missing() -> HTMLResponse:
            return HTMLResponse(_NOT_BUILT)

    return app
