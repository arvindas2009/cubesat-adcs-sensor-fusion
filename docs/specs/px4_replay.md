# Spec: PX4 log replay (server `webapp/server/px4_replay.py` + frontend `src/px4/`, M14)

This file is the contract between `py-implementer` (server), `test-author` (server tests) and `web-implementer` (frontend). It extends `docs/specs/webapp_protocol.md` (server, REST conventions) and `docs/specs/frontend.md` (page, colours, `frames.ts`). Conventions follow `CONVENTIONS.md`: scalar-first `[w, x, y, z]` quaternions, **body (FRD) to NED**, SI units. Degrees appear only in fields whose names end in `_deg`.

The replay is a **visual aid only**. It adds no numbers to any report, and the frontend does no estimation and no physics: every series is computed in Python by the existing, tested Project A code (`sensor_fusion.ekf_drone.run_filter`, `ekf2_reference`, `sensor_fusion.metrics` helpers, `sensor_fusion.interp.interp_to`, `common.quaternion.to_euler_zyx`). **EKF2 is a reference, not truth**, and every replay screen says so permanently.

Unlike the ADCS page there is no WebSocket: the server precomputes the whole run once (cached on disk), sends it as one JSON document, and the browser plays and scrubs it locally.

## 1. Module `webapp/server/px4_replay.py`

Importing it must not import fastapi (same rule as `adcs_session`). Heavy imports (`sensor_fusion.loader`/pyulog) may be lazy.

```python
CACHE_VERSION = 1
DEFAULT_RATE_HZ = 50.0
DEFAULT_CACHE_DIR = <repo>/webapp/.cache          # gitignored
DOWNLOAD_HINT = "python -m sensor_fusion.download_logs"
CAVEAT = "EKF2 is a reference, not truth"

class UnknownLog(LookupError): ...
class LogMissing(FileNotFoundError): ...          # str(e) contains DOWNLOAD_HINT

def build_replay(streams, ref, cfg, *, log_meta: dict, rate_hz: float = DEFAULT_RATE_HZ) -> dict
def replay_to_json(rep: dict) -> str
class ReplayStore: ...
```

### 1.1 `build_replay(streams, ref, cfg, *, log_meta, rate_hz)` (pure numpy; arrays, not rounded)

* `streams`: `SensorStreams`; `ref`: an `ekf2_reference`-shaped dict (`t_pos`, `pos_ned`, `t_att`, `q`, `reset_times`; `vel_ned` unused); `cfg`: `EkfConfig`.
* Runs `res = run_filter(streams, cfg)` exactly once.
* **Grid:** `t0, t1 = res.t[0], res.t[-1]`; `t = t0 + np.arange(n) / rate_hz` with `n = floor((t1 − t0) · rate_hz + 1e-9) + 1` (so the spacing is exactly `1/rate_hz` and `t[-1] ≤ t1`).
* **Ours** (all interpolated onto `t` with `interp_to`; quaternions with `quaternion=True`):
  * `p_ned_m` (n,3) from `res.p`;
  * `q` (n,4) from `res.q`;
  * `euler_deg` (n,3) = `rad2deg(to_euler_zyx(q))` stacked as `[roll, pitch, yaw]`;
  * `sigma3_att_deg` (n,3) = `3 · rad2deg(std[:, IDX_TH])` (the attitude-error states `th_x, th_y, th_z`; `ekf_drone.IDX_TH`);
  * `sigma3_pos_m` (n,3) = `3 · std[:, IDX_P]`.
* **EKF2** (interpolated onto `t`; **NaN outside the reference's own time span**, i.e. where `t < t_ref[0]` or `t > t_ref[-1]`, separately for position and attitude):
  * `p_ned_m` (n,3) from `ref["pos_ned"]` on `ref["t_pos"]`;
  * `q` (n,4) from `ref["q"]` on `ref["t_att"]`;
  * `euler_deg` (n,3).
* **Differences, ours − EKF2** (on the grid, with the metrics helpers so the definitions match the reports; NaN where EKF2 is NaN):
  * `pos_h_m` = `‖Δp[:, :2]‖`; `pos_d_m` = `Δp[:, 2]` (signed, positive = ours lower);
  * `tilt_deg` = `rad2deg(metrics.tilt_angle(q_ours, q_ekf2))`;
  * `yaw_deg` = `rad2deg(wrap_angle(metrics.yaw_of(q_ours) − metrics.yaw_of(q_ekf2)))`.
* **NIS** (`res.innovations`), for each name in `metrics.SENSOR_ORDER` that is present and has ≥ 1 sample: keep samples with finite `t` and `nis`; if there are more than `rate_hz · (t1 − t0)` of them, keep every `stride`-th (`stride = ceil(count / (rate_hz · (t1 − t0)))`, starting at index 0). Entry: `t`, `nis`, `accepted` (bool), `dof` (the most common `dof` among the kept samples), `band95 = chi2.ppf([0.025, 0.975], dof)` (per-sample two-sided 95 % interval, as in `common.ekf.nis_fraction_inside`).
* **Events:** `ekf2_resets_s` = the `ref["reset_times"]` inside `[t0, t1]`, sorted; `skip_init_s = metrics.SKIP_INIT_S` (the reports exclude the first 5 s; the page shades it).
* Returns:

```python
{"schema_version": 1, "caveat": CAVEAT, "rate_hz": rate_hz,
 "log": log_meta,                       # passed through unchanged (see 1.3)
 "filter": {"increment": cfg.increment, "runtime_s": res.meta["runtime_s"], "n_imu": res.meta["n_imu"],
            "n_updates": res.meta["n_updates"]},
 "t": t,
 "ours": {"p_ned_m", "q", "euler_deg", "sigma3_att_deg", "sigma3_pos_m"},
 "ekf2": {"p_ned_m", "q", "euler_deg"},
 "diff": {"pos_h_m", "pos_d_m", "tilt_deg", "yaw_deg"},
 "nis": {name: {"t", "nis", "accepted", "dof", "band95"}},
 "events": {"ekf2_resets_s": [...], "skip_init_s": 5.0}}
```

### 1.2 `replay_to_json(rep) -> str`

* Same structure; every float written with **7 significant digits** (`%.7g` semantics), NaN/±inf → `null`, booleans as JSON booleans, ints as ints. Arrays (n,k) become lists of lists, (n,) become lists.
* Must be fast enough for 34 k grid samples (the 675 s log): target < 2 s. Vectorised formatting (e.g. `np.char.mod` / building the text per column) is fine; `json.loads` of the output must work and `json.dumps(json.loads(s), allow_nan=False)` must succeed.

### 1.3 `ReplayStore`

```python
ReplayStore(cache_dir=DEFAULT_CACHE_DIR, logs=PINNED_LOGS, data_dir=sensor_fusion.download_logs.DATA_DIR,
            context_loader=None, cfg_factory=None, rate_hz=DEFAULT_RATE_HZ)
```

* `context_loader(entry) -> (streams, ref)`; default uses `sensor_fusion.run_flight.load_context(entry)` → `(ctx["S"], ctx["ref"])`. `cfg_factory() -> EkfConfig`; default `run_flight.build_config(3)` (the tuned defaults, as `run_flight` uses). Tests inject a synthetic loader.
* `path(entry)` = `data_dir / f"{id}.ulg"`.
* `resolve(token) -> entry`: role, full id, or id prefix of length ≥ 4, matching exactly one entry; anything else (including `"all"`, empty, ambiguous) raises `UnknownLog`.
* `list_logs() -> list[dict]`: per entry `{"id", "role", "notes", "present": bool, "cached": bool}` (`cached` = a valid cache file for the current key exists; `False` when not present).
* **Cache key** = first 16 hex of sha256 over `json.dumps({"cache_version", "id", "rate_hz", "config": dataclasses.asdict(cfg), "ulg_size", "ulg_mtime_ns"}, sort_keys=True, default=str)`. Files: `cache_dir / f"px4_{id[:8]}_{key}.json"` (the `replay_to_json` text) and `… .npz` (the arrays of `build_replay`, `np.savez_compressed`, for offline inspection). A hit needs the `.json` file. Writes go to a temporary name then `os.replace` (no torn files if interrupted). `cache_dir` is created on first write.
* `get_json(token) -> str`: resolve (→ `UnknownLog`), check the file (→ `LogMissing(f"log {id} ({role}) is not downloaded; run: {DOWNLOAD_HINT}")`), return the cached text on a hit, otherwise load, build, write, return. `n_computed` counts builds (for tests).
* `log_meta` for a real log: `{"id", "role", "notes", "duration_s": imu span, "origin_lat_deg", "origin_lon_deg", "gps_utc_start", "declination_deg", "g_m_s2"}` from the streams (`origin_lla`, `declination`, `g`, `meta["gps_utc_start"]`). Missing values are `None`.
* Thread safety: `get_json` may be called from worker threads; at most one build per log id at a time (a per-id `threading.Lock`; a second caller waits and then gets the cache hit).

## 2. REST (added to `create_app`)

`create_app(frontend_dist=…, max_sessions=4, px4_store: ReplayStore | None = None)`; `None` → `ReplayStore()`. Stored as `app.state.px4`.

| Method + path | Success | Errors |
|---|---|---|
| `GET /api/px4/logs` | 200 `{"logs": list_logs(), "download_hint": DOWNLOAD_HINT, "caveat": CAVEAT}` | |
| `POST /api/px4/replay/{log}` | 200, body = `get_json(log)` text, `application/json` | 404 `{"detail": "unknown log '<log>'"}`; 404 `{"detail": str(LogMissing)}` (contains the hint); 500 `{"detail": "replay failed: <message>"}` on any other exception |

* `get_json` runs in `asyncio.to_thread` (a first build takes 10–35 s; other endpoints and ADCS WebSockets must keep working meanwhile).
* Add Starlette's `GZipMiddleware(minimum_size=1024)` to the app (the 675 s replay is several MB of JSON). It only touches HTTP responses; WebSockets are unaffected. Existing tests must still pass.

## 3. Frontend

### 3.1 Files

```
src/math/ned.ts          NED -> Three.js conversions (pure, tested)
src/px4/types.ts         Replay, LogsResponse (mirror section 1.1 / 2; nullable numbers are `number | null`)
src/px4/api.ts           getLogs(), loadReplay(log) (reuse ApiError/describeError from src/api.ts)
src/px4/clock.ts         PlaybackClock + indexAt (pure, tested)
src/px4/series.ts        replay -> uPlot columns, path polylines, reset markers, readout values (pure, tested)
src/px4/DroneView.ts     attitude scene: ours solid, EKF2 ghost
src/px4/PathView.ts      3D NED path scene
src/px4/ReplayPanels.ts  log picker, transport, readouts, timeline, plots
src/px4/ReplayPage.ts    page controller (wires the above; created once, lazily on first visit)
tests/ned.test.ts  tests/clock.test.ts  tests/series.test.ts
tests/fixtures/make_ned_fixtures.py -> tests/fixtures/ned_fixtures.json   (Python oracle, committed)
```

### 3.2 NED conversion (`src/math/ned.ts`)

One fixed proper rotation `N` maps NED into Three.js world (Y-up): **`N · (n, e, d) = (n, −d, e)`**, `N = [[1,0,0],[0,0,−1],[0,1,0]]`, `det N = +1`. North is world +X, up is world +Y, east is world +Z.

| Function | Definition |
|---|---|
| `nedToWorld(v, scale = 1) -> THREE.Vector3` | `N v · scale` |
| `bodyNedToWorld(q) -> THREE.Quaternion` | `quat(N) · toThreeQuaternion(q)` (reuse `frames.ts`). A mesh modelled in **FRD body coordinates** (x forward, y right, z down) and given this quaternion shows `N · R(q) · p_body`. |

`make_ned_fixtures.py` (run from the repo root with `.venv`) writes, from `common.quaternion` (`to_dcm`, `from_euler_zyx`, `to_euler_zyx`) and numpy only:
* about 20 seeded random unit quaternions with `N @ to_dcm(q)` (world body axes, columns);
* named attitudes from aerospace Euler angles (roll, pitch, yaw): level north `(0,0,0)` → nose world `(1,0,0)`, body z world `(0,−1,0)`; yaw 90° → nose world `(0,0,1)` (east); pitch +30° → nose has positive world y (nose up); roll +30° → body y (right wing) has negative world y (right wing down); and the matching `to_euler_zyx` values in degrees;
* a few NED vectors with `N v`.

### 3.3 Page

* **Tabs.** The "PX4 replay" tab becomes active. Tabs switch between `#page-adcs` and `#page-px4` and set `location.hash` (`#adcs` / `#px4`); loading the page with `#px4` opens the replay tab. Hidden views skip rendering; on showing a tab, its renderers resize. The ADCS session keeps running while hidden. The header title follows the tab ("CubeSat ADCS live simulator" / "PX4 flight replay"); the footer on the replay tab reads "Replay — visual aid only; reported numbers come from `sensor_fusion/report/`. EKF2 is a reference, not truth."
* **Caveat.** A permanent, non-dismissible badge "EKF2 is a reference, not truth" (text = `replay.caveat` once loaded, the constant before). It now lives in the PX4 status slot of the top bar (`docs/specs/ui_console.md` §2): the main shell writes it, so it is visible before the replay page is created, and the page header row of the earlier layout no longer exists. The EKF2 ghost's legend label is "EKF2 (reference)". Difference plots are titled "ours − EKF2".
* **Layout** (same grid as the ADCS page): left column two 3D viewports (DroneView top, PathView bottom); right column: Log, Transport, Readouts panels; bottom band: timeline + 2×2 plots.
* **Log panel.** From `GET /api/px4/logs`: one row per log (role, short id, notes on hover), a status chip (`not downloaded` / `cached` / `ready`), and a Load button. Missing logs show the hint as copyable `<code>python -m sensor_fusion.download_logs</code>` and their Load button is disabled. While `POST /api/px4/replay/{role}` is pending: "Running the filter… (first load of a log takes 10–35 s; cached afterwards)". Errors (`detail`) go to the banner. Retry the logs request every 2 s while the server is unreachable (as the presets request does).
* **DroneView.** A quadcopter in FRD body coordinates: X-frame arms, four rotor discs, a nose marker on +x, small body triad (x red, y green, z blue). Ours: solid, light grey with a red nose. EKF2: same geometry, translucent amber wireframe ghost, hidden while the EKF2 quaternion is `null`. Both at the origin (attitude only), oriented with `bodyNedToWorld`. A horizontal ground grid with "N" and "E" labels (world +X / +Z). OrbitControls.
* **PathView.** Ours as a solid line (teal), EKF2 as a dashed amber line, 1 world unit = 1 m via `nedToWorld`. Full path faint, the part up to the current time bright; current-position markers for both; start marker. A ground grid at the height of the first sample. The camera auto-fits the bounding box of both paths on load. OrbitControls. Null EKF2 samples split the line (no segments through gaps).
* **Transport.** Play/Pause; speed select 0.25, 0.5, 1, 2, 4, 8, 16× (default 1×); a scrub slider over `[t[0], t[-1]]` with the time readout; "previous/next EKF2 reset" buttons; Space toggles play. Playing past the end stops at the end.
* **Readouts** (current grid sample `i = indexAt(t, clock.t)`): t and fraction of the log; roll/pitch/yaw ours vs EKF2 (deg); tilt and yaw difference (deg); horizontal and vertical position difference (m); ours 3σ attitude (deg, max of the 3); height above the start (m, `−(d − d[0])`) ours vs EKF2. `null` shows as "—".
* **Timeline.** A bar over `[t[0], t[-1]]`: shaded `skip_init_s` region (label "excluded from report statistics"), EKF2 reset markers, the cursor; click/drag seeks.
* **Plots** (uPlot, built once per replay, full log; a vertical cursor line at the current time; reset times as thin vertical lines; clicking a plot seeks):
  1. tilt and yaw difference, ours − EKF2 (deg; yaw signed);
  2. horizontal and vertical position difference, ours − EKF2 (m);
  3. NIS by sensor (log y; points; accepted only, rejected as hollow/faint), with the 95 % band (`band95`) lines for each distinct `dof`;
  4. height above the start, ours vs EKF2 (m).
* No console errors in normal use; a replay can be reloaded or another log loaded without leaking renderers (dispose old geometry).

### 3.4 Pure modules and tests (vitest)

* `clock.ts`: `class PlaybackClock(t0, t1)` with `t`, `playing`, `speed`, `play()`, `pause()`, `toggle()`, `seek(t)` (clamped), `tick(dtWallSeconds) -> t` (advances `speed · dt` while playing, clamps at `t1` and pauses there; a `play()` at `t1` restarts from `t0`). `indexAt(tArr, t)`: index of the last grid sample `≤ t` (binary search; 0 below the start, last index above the end).
* `series.ts`: `diffColumns(rep)`, `heightColumns(rep)`, `nisSeries(rep)`, `pathSegments(points)` (splits at `null`), `nextReset(rep, t)` / `prevReset(rep, t)`, `readoutsAt(rep, i)`. Tested on a small hand-built `Replay` object that follows section 1.1.
* `ned.test.ts`: against `ned_fixtures.json`: `bodyNedToWorld(q)` applied to the unit axes equals `N @ to_dcm(q)` columns to 1e-12; the named attitudes point the nose/wing as stated; `nedToWorld` equals `N v`; `det N = +1`.

## 4. Server tests (`webapp/server/tests/test_px4_replay.py`, from this spec)

Use a synthetic flight (`sensor_fusion.synthetic.make_synthetic_flight("figure8", duration=30, imu_rate=100, seed=…)`) and a reference built from its truth: `{"t_pos": truth.t, "pos_ned": truth.p, "vel_ned": truth.v, "t_att": truth.t, "q": truth.q, "reset_times": [...]}`. Keep the whole module under about 20 s.

1. **Grid:** spacing exactly `1/rate_hz` (to 1e-9), `t[0] == res.t[0]`, `t[-1] ≤ res.t[-1] < t[-1] + 1/rate_hz`.
2. **Ours = `run_filter`:** run `run_filter` independently with the same cfg (it is deterministic). At grid times that coincide with filter samples (100 Hz IMU, 50 Hz grid), `ours.p_ned_m` equals `res.p` to 1e-9, `ours.q` equals `res.q` up to sign to 1e-9, `sigma3_*` equal `3·std` of the right states, `euler_deg` equals `rad2deg(to_euler_zyx)`, and an independent check with `scipy.spatial.transform.Rotation` (`from_quat` with scalar-last order, `as_euler("ZYX")` reversed) agrees to 1e-6 deg away from gimbal lock.
3. **Differences = `compare_to_ekf2`:** with the truth reference sampled at 100 Hz, the grid times are reference sample times, so `diff.tilt_deg`, `diff.yaw_deg`, `diff.pos_h_m`, `diff.pos_d_m` equal `compare_to_ekf2(res, ref)["arr"]` (converted to degrees) at those times to 1e-6.
4. **Coverage:** a reference restricted to `[5, 25] s` gives NaN EKF2 and diff entries exactly outside that span and finite inside.
5. **NIS:** entries equal the `res.innovations` arrays (accepted, nis, t, dof); `band95` equals `scipy.stats.chi2.ppf([0.025, 0.975], dof)`; with `rate_hz = 0.5` the decimation keeps every `stride`-th sample as specified.
6. **Events:** reset times outside `[t0, t1]` are dropped; `skip_init_s == 5.0`.
7. **JSON:** `replay_to_json` parses; NaN → `null`; numbers equal the arrays to relative 5e-7 (absolute 1e-12 for zero); `json.dumps(…, allow_nan=False)` of the parsed object succeeds; keys and `schema_version`, `caveat == "EKF2 is a reference, not truth"` as in section 1.1.
8. **Store:** with a tmp `data_dir` holding fake `<id>.ulg` files for some ids, a tmp `cache_dir`, and an injected synthetic `context_loader` (count its calls):
   * `list_logs` reports `present` correctly and `cached` false, then true after a build;
   * `resolve` by role, full id, 4+ character prefix; `"all"`, `""`, an unknown and a too-short/ambiguous prefix raise `UnknownLog`;
   * a missing file raises `LogMissing` whose message contains `python -m sensor_fusion.download_logs`;
   * the second `get_json` is a cache hit (loader called once, `n_computed == 1`, identical text); a **new** store on the same `cache_dir` also hits; a different `cfg_factory` result, or rewriting the `.ulg` with a different size, recomputes; the `.json` and `.npz` exist in `cache_dir`.
   * two threads calling `get_json` for the same log concurrently build once.
9. **API** (`TestClient(create_app(frontend_dist=None, px4_store=<synthetic store>))`): `GET /api/px4/logs` shape; `POST /api/px4/replay/<role>` 200 with the keys of section 1.1; unknown → 404; missing → 404 with the hint in `detail`; a loader that raises → 500 `"replay failed: …"`; a request with `Accept-Encoding: gzip` gets `content-encoding: gzip` on the replay; `/api/health` still works.
10. **Optional real-log check** (skipped unless the env var `PX4_REPLAY_REAL=1` and the hover `.ulg` is present): the RMS of `diff.tilt_deg` over grid samples with `t ≥ t0 + 5 s` and more than 1 s from every reset agrees with `compare_to_ekf2(...)["cmp"]["tilt_deg"]["rms"]` within 10 % (different sample grids, same definition).

## 5. Exit (Group 12)

* `pytest` green (the replay tests included); `npm test` and `npm run build` clean.
* Browser check (main session, claude-in-chrome) of all three pinned logs: each loads, plays and scrubs; the EKF2 ghost tracks ours; the caveat badge is visible; reset markers appear for logs that have resets; no console errors.
* README section "Interactive simulator" gives a working launch: `pip install -e .[web]`, `npm ci && npm run build`, `python -m webapp`.
