# Spec: browser simulator server and wire protocol (`webapp/`, M12)

This file is the contract between `py-implementer` (server), `test-author` (server tests), and the Group 11–12 frontend (`web-implementer`). Read `docs/specs/stepper.md` first: the server only drives `adcs.stepper.ClosedLoopSim`. Conventions follow `CONVENTIONS.md`: scalar-first `[w, x, y, z]` quaternions, body-to-ECI, SI units. Degrees appear only in fields whose names end in `_deg` or `_deg_s`.

The app is a **visual aid only**. It adds no numbers to any report, and it does no physics of its own. The frontend does no physics either: every display quantity is computed in Python by existing, tested helpers.

## 0. Layout and dependencies

```
webapp/
  __init__.py
  __main__.py            python -m webapp [--host 127.0.0.1] [--port 8000]
  server/
    __init__.py
    app.py               create_app(frontend_dist: Path | None = <webapp/frontend/dist>) -> FastAPI
    schemas.py           pydantic v2 models: SessionParams, Command variants; build_sim_kwargs()
    adcs_session.py      AdcsSession (owns one ClosedLoopSim + the streaming loop), SessionRegistry, frame()
    tests/               test_schemas.py, test_adcs_session.py, test_app.py (TestClient)
  frontend/              (Group 11)
```

* **Dependencies.** `pyproject.toml` gets `[project.optional-dependencies] web = ["fastapi>=0.110", "uvicorn[standard]>=0.29", "httpx>=0.27"]`, and `webapp*` is added to `[tool.setuptools.packages.find] include`. `webapp` is also added to `[tool.pytest.ini_options] testpaths`. `requirements.txt` is unchanged.
* **Install:** `.venv/Scripts/python -m pip install -e .[web]`.
* **Missing extra.** Every test module in `webapp/server/tests/` starts with `pytest.importorskip("fastapi")`, so the core suite stays green without the extra.
* **Imports.** Importing `webapp.server.adcs_session` and `webapp.server.schemas` must not import fastapi. The session logic stays testable as plain asyncio code; `schemas` uses pydantic only, and pydantic is installed with fastapi, so `test_schemas` also importorskips it.
* **Binding.** The server binds to `127.0.0.1` by default. There is no authentication, and it is a local tool.

## 1. Session parameters (`SessionParams`, pydantic, `extra="forbid"`)

| Field | Type | Default | Constraint | Maps to |
|---|---|---|---|---|
| `orbits` | float | 6.0 | 0.05 ≤ x ≤ 15 | `t_end = orbits × orbital_period(DEFAULT.orbit.a)` |
| `rate_deg_s` | float | 10.0 | 0 ≤ x ≤ 30 | `y0 = tumble_state(np.deg2rad(rate_deg_s), seed)` |
| `seed` | int | 0 | 0 ≤ x ≤ 2³¹−1 | `seed=` (sim) and the `tumble_state` seed |
| `sensors` | `"realistic"` \| `"ideal"` | `"realistic"` | | `REALISTIC_SENSORS` / `IDEAL_SENSORS` |
| `filter` | `"design"` \| `"g7"` | `"design"` | | `MEKF_DESIGN` / `MEKF_G7` |
| `dumping` | bool | true | | `ctrl=ControlConfig(dumping=…)` |
| `inertia_scale` | float | 1.0 | 0.5 ≤ x ≤ 1.5 | `inertia_true = None if scale == 1.0 else scale × DEFAULT.spacecraft.inertia` (the controller always uses the nominal inertia) |

* `build_sim_kwargs(p: SessionParams) -> dict` returns the exact keyword arguments for `ClosedLoopSim(**kw)`. The server always passes `log_every=10`; it never needs the full log.
* With the defaults, the kwargs equal what `python -m adcs.simulate` (realistic, design, 10 °/s, seed 0, 6 orbits) passes to `run_closed_loop`. A test asserts this by comparing a short run (orbits = 0.05): the server's sim must produce bytes identical to `run_closed_loop` with the CLI's arguments.
* Violations return HTTP 422 (FastAPI's default validation response). Examples: unknown keys, out-of-range values, wrong types, booleans given for floats.

## 2. REST API (all JSON, prefix `/api`)

| Method + path | Body | Success | Errors |
|---|---|---|---|
| `GET /api/health` | | 200 `{"ok": true}` | |
| `GET /api/adcs/presets` | | 200 `{"params_schema": SessionParams.model_json_schema(), "defaults": SessionParams().model_dump()}` | |
| `POST /api/adcs/sessions` | `SessionParams` (`{}` = defaults) | 201 `{"id", "params", "info", "status"}` | 422, 429 |
| `GET /api/adcs/sessions/{id}` | | 200 `{"id", "params", "info", "status"}` | 404 |
| `DELETE /api/adcs/sessions/{id}` | | 204 | 404 |

**Creation.**
* `POST` builds the `ClosedLoopSim` in `asyncio.to_thread`. The precompute can take seconds over 15 orbits, and the event loop must stay responsive.
* It then immediately processes **sample 0** (`sim.step(1)`), so that a snapshot always exists. The `POST` returns once the session is ready; the frontend shows its own "preparing" spinner while it waits.
* `id` is `uuid4().hex`.

**Limit.**
* At most `max_sessions` sessions exist at once: 4 by default, and a `create_app` argument.
* Creating one more first evicts the least-recently-used session that has **no** connected client.
* If every session has a client, the response is 429.

`info` (static per session):

```json
{"period_s": float, "t_end_s": float, "control_period_s": float, "n_steps": int,
 "epoch_utc": "2026-03-20T12:00:00", "body_size_m": [0.10, 0.10, 0.34],
 "wheel_max_speed_rad_s": float, "m_max_Am2": float, "safe_rate_deg_s": float,
 "detumble_threshold_deg_s": float,
 "orbit_track_eci_m": [[x, y, z], ...],
 "eclipses_s": [[t_start, t_end], ...]}
```

* `orbit_track_eci_m` is `sim.r_eci` over the **first orbit**, every 30 s, with the last point closing the ring.
* `eclipses_s` lists the maximal intervals of `~sim.lit` over the whole horizon, using `t_grid` values: the first and last shadow sample of each interval.

`status` (dynamic):

```json
{"k": int, "t": float, "done": bool, "playing": bool, "speed": float, "connected": bool}
```

Here `k` and `t` belong to the last processed sample, which is `snapshot()["k"]` and `["t"]`.

## 3. WebSocket `WS /api/adcs/sessions/{id}/ws`

**Connection.**
* An unknown id: the server accepts the socket, then closes it with code **4404**.
* One client per session: a new connection closes the previous one with code **4001** ("replaced").
* When the client disconnects, the session pauses. It does not step with nobody watching.

On connect, the server sends, in this order:
1. `{"type": "hello", "id", "params", "info", "status"}`;
2. one `frame` (§3.2) for the current snapshot.

### 3.1 Commands (client → server, JSON text)

A message that fails validation gets `{"type": "error", "message": str}` back, and the socket stays open. This covers bad JSON, an unknown `cmd`, extra keys, and out-of-range values.

| Command | Fields | Effect |
|---|---|---|
| `{"cmd": "play"}` | | start streaming. No-op if done (a frame with `done: true` is still sent) |
| `{"cmd": "pause"}` | | stop stepping; send one frame with `playing: false` |
| `{"cmd": "speed", "value": x}` | 1 ≤ x ≤ 2000 | target sim-seconds per wall-second. Default 50 |
| `{"cmd": "kick", "axis": "x"\|"y"\|"z", "deg_s": d}` | 0 < \|d\| ≤ 20 | `sim.kick(np.deg2rad(d) × e_axis)` (body axes). Error if done. Then send one frame |
| `{"cmd": "step", "n": n}` | 1 ≤ n ≤ 100000, only while paused | `sim.step(n)` in a thread, then one frame. Error if playing or done. Deterministic: tests use this |
| `{"cmd": "reset"}` | | rebuild the sim from the same params (thread), process sample 0, pause, then send a new `hello` and a frame |

### 3.2 Frame (server → client)

All floats are serialised with 7 significant digits (`float(f"{x:.7g}")`). `null` is used for unavailable estimator values. Display-frame matrices are row-major 3×3 nested lists.

```json
{
 "type": "frame", "seq": int, "k": int, "t": float, "orbit": t / period,
 "mode": 0 | 1, "mode_name": "DETUMBLE" | "POINTING", "dump": bool, "lit": bool,
 "done": bool, "playing": bool, "speed_set": float, "speed_actual": float,
 "eci": {"q": [4], "q_est": [4] | null, "r_m": [3], "v_m_s": [3], "sun": [3], "b_T": [3],
         "gmst_rad": float},
 "display": {"C": [[3],[3],[3]], "C_est": [[3],[3],[3]] | null, "sun": [3], "b": [3]},
 "metrics": {"nadir_err_deg": float, "est_nadir_err_deg": float | null,
             "know_err_deg": float | null, "att_sigma_deg": float | null,
             "rate_deg_s": float, "w_deg_s": [3], "w_est_deg_s": [3] | null,
             "wheel_speed_rpm": [3], "wheel_frac": float, "m_Am2": [3], "u_w_Nm": [3],
             "sun_used": int},
 "events": {"t_init": float | null, "switch_times": [..], "fallback_times": [..],
            "kicks": [{"t": float, "dw_deg_s": [3]}]}
}
```

**Display frame.** This is the same frame as the Group 9 visualizer (`docs/specs/visualizer.md` §1): LVLH rotated 180° about x, zenith up, nadir `(0, 0, −1)`. The display fields are:
* `C = DISPLAY_FLIP @ body_dcm_lvlh(q, r, v)`. Column `k` is body axis `k` in display coordinates, so it is exactly the local-to-world rotation of a body-fixed mesh in a display-frame scene. There is no right-hand factor. For `q = q_ref`, `C = DISPLAY_FLIP`, and body z points down.
* `C_est` uses the same formula with `q_est`.
* `sun = DISPLAY_FLIP @ sun_lvlh(sun_eci, r, v)`.
* `b = DISPLAY_FLIP @ dcm_eci_to_lvlh(r, v) @ b_eci / ‖b_eci‖`.

All helpers come from `adcs.visualize` and `common.frames`.

**Other frame fields.**
* `know_err_deg` is `‖know_err‖`.
* `wheel_speed_rpm` is `wheel_speed × 60/(2π)`.
* `wheel_frac` is `max|wheel_speed| / wheel.max_speed`.
* `rate_deg_s` is `‖w‖`. All metrics come from `snapshot()`.
* `seq` counts the frames sent on this connection.

### 3.3 Streaming loop (`AdcsSession`)

* One asyncio task per connected client. Every tick, with target period `1/30` s:
  * Let `dt_wall` be the wall time since the last tick, clamped to ≤ 0.1 s, so that a stalled loop does not trigger a catch-up burst.
  * Add `speed × dt_wall` to a sim-time budget.
  * `n = floor(budget / T)`, capped at `n_max`. If `n ≥ 1`: `await asyncio.to_thread(sim.step, min(n, remaining))`, subtract `n × T` from the budget, and send one frame.
  * If the cap was hit, the leftover budget is **dropped**. The sim falls behind real time rather than piling up debt.
  * `n_max = max(1, ceil(0.1 s × measured step rate))`, i.e. about 0.1 s of compute per batch, keeping commands responsive. The measured step rate is an exponential moving average of samples per second, measured inside `to_thread`.
* `speed_actual` is the sim time advanced divided by the wall time over a sliding window of about 1 s. It reports how fast the sim really runs when the CPU cannot keep up.
* **Thread safety.** `ClosedLoopSim` is not thread-safe. Only one coroutine touches it at a time, under an `asyncio.Lock`. Commands that touch the sim (`kick`, `step`, `reset`) take the lock, so they never run during a batch.
* **Done.** When `sim.done`, set `playing = false` and send a final frame with `done: true`.
* **Frame builder.** `frame(snapshot, sim, session_state) -> dict` is a pure function in `adcs_session.py`, unit-testable without a socket.

## 4. Static frontend

* If `frontend_dist` exists and contains `index.html`, mount it at `/` with `StaticFiles(html=True)`, after the API routes.
* Otherwise `GET /` returns 200 `text/html` with a short note: "frontend not built: cd webapp/frontend && npm ci && npm run build".
* `python -m webapp`: argparse `--host` (default `127.0.0.1`) and `--port` (default 8000), then `uvicorn.run(create_app(), host, port)`. It prints the URL.

## 5. Tests (`webapp/server/tests/`, written from this spec)

1. **Schemas.**
   * The defaults are as in §1.
   * Every out-of-range value, extra key, or wrong type raises `ValidationError`.
   * `build_sim_kwargs` maps the presets and `inertia_scale` as specified: `None` at 1.0, and an exact `scale × nominal` otherwise.
   * The CLI-equivalence check from §1 passes on a short run.
2. **Frame builder.**
   * `C` is orthonormal with det +1 (1e-6, because of the 7-digit rounding).
   * With `q = q_ref` (build a sim from `lvlh_state` with zero offset; one sample), `C ≈ DISPLAY_FLIP` to 1e-6, so body z points down `(0, 0, −1)`.
   * `sun` and `b` are unit vectors.
   * Before MEKF init (ideal sensors, a sample with no TRIAD; or construct by stepping 0 samples), the estimator fields are `null`.
   * All numbers are finite, or `null`.
   * The values match `snapshot()` converted independently (degrees by `np.rad2deg`) to 7 significant digits.
   * `json.dumps(frame, allow_nan=False)` succeeds.
3. **REST** (`TestClient`).
   * health and presets work.
   * `POST {}` → 201 with a 32-character hex id; `info.n_steps` equals `round(6 × period)`.
   * `orbit_track` closes the ring to 1 km.
   * `eclipses_s` intervals lie within `[0, t_end]` and are ordered.
   * `GET` the id works; an unknown id gives 404.
   * `DELETE` gives 204, and a `GET` afterwards gives 404.
   * Bad params give 422 (one case per constraint).
   * 429 when `max_sessions` sessions all have clients. Use `create_app(max_sessions=1)` with one WebSocket connected.
   * Eviction of an idle session works.
4. **WebSocket** (`TestClient.websocket_connect`; use `orbits=0.05` to keep it fast).
   * Connect: `hello` then a `frame` with `k == 0`, `t == 0`.
   * `step n=100`: a frame with `k == 100`, `t == 100`, and its metrics equal to an independent `ClosedLoopSim(**build_sim_kwargs(p))` stepped 101 samples, to 7 significant digits.
   * `play` at speed 2000: at least 3 frames arrive within 3 s, with strictly increasing `t`.
   * `pause`: a frame with `playing: false`; then for 0.3 s no frame with a larger `t` arrives, and `GET` status `t` is unchanged.
   * `kick`: the next frame's `events.kicks` has one entry with the right `dw_deg_s`, and the session's `sim.result().meta["kicks"]` matches.
   * Error messages for bad commands, with the socket still usable afterwards.
   * `step` while playing gives an error.
   * Running to the end (`step n=100000` on the short session) gives `done: true`; then `play` sends a done frame, and `kick` gives an error.
   * `reset` sends `hello` + frame with `k == 0`, and the following `step` reproduces the first run's frame exactly.
   * An unknown id closes with code 4404.
   * A second connection closes the first with 4001.
5. **Scripted client (exit criterion).** A test or the command `python -m webapp.server.tests.ws_smoke` starts uvicorn on a free port in a thread, connects with the `websockets` client, plays at 500× for 2 s, and prints the frame count and `speed_actual`. `websockets` comes with `uvicorn[standard]`.
