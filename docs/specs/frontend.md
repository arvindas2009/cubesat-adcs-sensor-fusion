# Spec: browser simulator frontend, ADCS page (`webapp/frontend/`, M13)

This file is the contract between `web-implementer` (code) and the tests. The server side and wire format are in `docs/specs/webapp_protocol.md`; that file is the source of truth for every message field. Conventions follow `CONVENTIONS.md`: quaternions are scalar-first `[w, x, y, z]`, body-to-ECI.

The page is a **visual aid only**. It does no physics and adds no numbers to any report. Every physical quantity comes from server frames; the frontend only maps frames into Three.js coordinates and draws them.

## 0. Stack and layout

* **Stack:** Vite + TypeScript (strict), `three`, `uplot`. Dev dependencies: `vite`, `typescript`, `vitest`, `@types/three`. Use the current stable versions at install time. Commit `package.json` and `package-lock.json`, since Group 12 and the README use `npm ci`.
* **`vite.config.ts`:**
  * `root` = `webapp/frontend`;
  * `build.outDir` = `dist`. `dist/` and `node_modules/` are gitignored. FastAPI serves `dist/` at `/`; see protocol §4;
  * the dev server proxies `/api` to `http://127.0.0.1:8000`, with `ws: true`;
  * vitest uses the `node` environment. The pure modules must not touch the DOM.
* **Scripts:** `npm run dev`, `npm run build` (`tsc --noEmit && vite build`), `npm test` (`vitest run`).
* **Static assets:** `public/textures/earth_bluemarble_3840.jpg` already exists. It is NASA Blue Marble (public domain, provenance in `public/textures/manifest.json`), equirectangular, longitude −180..180 left to right, north up. If it fails to load, fall back to a plain shaded sphere; never block on it.

```
webapp/frontend/
  index.html  package.json  package-lock.json  tsconfig.json  vite.config.ts
  public/textures/...                      (exists)
  src/main.ts                              app bootstrap, page shell
  src/api.ts                               REST: presets, create/get/delete session
  src/ws.ts                                WebSocket client: typed messages, commands, reconnect-free
  src/types.ts                             Hello, Frame, Info, Status, SessionParams (mirror protocol §1–§3)
  src/math/frames.ts                       ALL coordinate conversions (pure, tested)
  src/adcs/history.ts                      ring buffer of frame samples for plots + mode/event timeline (pure, tested)
  src/adcs/OrbitView.ts                    ECI scene
  src/adcs/SatView.ts                      display-frame scene
  src/adcs/Panels.ts                       form, transport, kick, readouts, timeline, uPlot strips
  src/style.css
  tests/frames.test.ts  tests/history.test.ts
  tests/fixtures/make_fixtures.py          Python oracle generator (run with the repo .venv)
  tests/fixtures/frames_fixtures.json      generated, committed
```

## 1. Coordinate conversions (`src/math/frames.ts`)

Three.js is **Y-up**. One fixed proper rotation, `M`, maps a right-handed "z-up" frame into Three.js world coordinates:

`M · (x, y, z) = (x, z, −y)`, i.e. `M = [[1,0,0],[0,0,1],[0,−1,0]]`, with `det M = +1`. It is a −90° rotation about x.

Exports (all pure; inputs are plain `number[]` or arrays of arrays, as on the wire):

| Function | Definition |
|---|---|
| `toThreeQuaternion(q: [w,x,y,z]) -> THREE.Quaternion` | `new THREE.Quaternion(x, y, z, w)`. Gives the same rotation as `common.quaternion.to_dcm(q)` (body to ECI). |
| `zUpToWorld(v) -> THREE.Vector3` | `M v` |
| `eciToWorld(r_m, scale) -> THREE.Vector3` | `M r × scale` |
| `earthRotationY(gmst_rad) -> number` | the Earth mesh's `rotation.y`. It equals `gmst_rad`. Derivation below. |
| `displayMatrixToWorld(C: number[][]) -> THREE.Matrix4` | rotation part `M · C`. Columns are the body axes in world coordinates, for a body-fixed mesh in the display scene. |
| `bodyInEciToWorld(q) -> THREE.Quaternion` | `quat(M) · toThreeQuaternion(q)`, the attitude of a body-fixed mesh in the ECI scene. |
| `latLonToEcef(lat, lon, R) -> [x, y, z]` | spherical, for tests and ground markers |

**Earth texture.** `THREE.SphereGeometry` puts its poles on local ±Y. The vertex for UV `u` is at azimuth `φ = 2πu`, at local `(−cos φ sin θ, cos θ, sin φ sin θ)`. So `u = 0.5`, which is the texture's longitude 0, sits at local +X, and `u = 0.75` (90°E) sits at local −Z. In ECEF terms, the sphere's local coordinates are exactly `M · p_ecef`.
* The ECEF→ECI relation is `p_eci = Rz(gmst)ᵀ p_ecef` (CONVENTIONS / `common.frames.eci_to_ecef` is the passive `Rz(gmst)`).
* Rotating an ECI vector actively by `+gmst` about ECI z is, in world coordinates, a rotation by `+gmst` about world +Y.
* **Hence `earth.rotation.y = gmst_rad`, with no offset**, and the Earth mesh sits directly in world coordinates. The fixtures in §4 must confirm this. If they do not, fix the formula, not the test.

**Display scene.** The display frame is LVLH rotated 180° about x (X ≈ along-track, Y = orbit normal, Z = zenith). It is mapped to world coordinates with the same `M`, so zenith is world-up and nadir is world `(0, −1, 0)`. The frame's display vectors `sun` and `b` are drawn as `zUpToWorld(v)`.

## 2. Page (single page, ADCS tab; a "PX4 replay" tab placeholder for Group 12)

* **Layout.** Use a desktop layout of at least 1280 px wide, as a CSS grid:
  * left about 62 % holds two 3D viewports stacked, OrbitView on top (about 55 %) and SatView below;
  * the right column holds the controls and readouts;
  * a full-width bottom band holds the mode timeline and the plots.
* **Look.** A dark "mission console": a deep navy/near-black background, hairline panel borders, monospace tabular numerals for every readout, and colour roles that are consistent everywhere:
  * body axes x/y/z red/green/blue;
  * the estimate is cyan and dashed/translucent;
  * Sun is warm yellow;
  * the magnetic field is violet;
  * nadir is grey-white;
  * DETUMBLE is amber, POINTING is green, eclipse is a dark band, kicks are magenta.
* **Footer.** "Simulation — visual aid only; reported numbers come from `adcs/report/`."
* **OrbitView (ECI scene):**
  * scale: 1 world unit = 1000 km;
  * the textured Earth rotating with `gmst`, plus a faint equator line and a polar axis;
  * the first-orbit ring from `info.orbit_track_eci_m`, and a trail of the last N frame positions;
  * the satellite as a small marker exaggerated in size, with a tiny body-axes triad using `bodyInEciToWorld`;
  * a Sun direction arrow from the Earth centre, plus a directional light along `eci.sun`, so the day/night terminator is real;
  * eclipse state shown by dimming the marker;
  * OrbitControls, with the camera following the satellite by default and a toggle for a free camera.
* **SatView (display-frame scene):**
  * the 3U box drawn to scale (`info.body_size_m`) with `displayMatrixToWorld(display.C)`;
  * body axes as RGB arrows;
  * a translucent cyan ghost box and dashed axes from `display.C_est` (hidden when `null`);
  * arrows for nadir (fixed, down), Sun (`display.sun`, grey and labelled ECLIPSE when `!lit`) and B (`display.b`);
  * three small wheel bars, height proportional to `|wheel_speed_rpm|` relative to the max speed, with the sign shown by colour;
  * a grid plane under the body as the local horizontal;
  * OrbitControls.
* **Panels:**
  * **Parameter form:** built from `GET /api/adcs/presets` (defaults and constraints); field order as in protocol §1, with inline validation. "Start session" (`POST`), then the WebSocket connects. While POST is pending, show "Preparing…" (a 15-orbit build takes about 1–2 s).
  * **Transport:** Play/Pause; a speed slider on a log scale from 1 to 2000×, showing set vs actual speed; step +1 / +60 / +600 (enabled only while paused); Reset.
  * **Kick:** axis x/y/z, sign, magnitude in °/s (default 5, max 20), Apply. Errors come from the server and are shown as a toast. They include the 30 °/s cap from `assumptions.md` [B-25].
  * **Readouts:** t [s] and orbits; mode badge; Sunlit/Eclipse; dumping; nadir error; knowledge error; 3σ (`3 × att_sigma_deg`); rate; ω̂; wheel rpm and fraction; dipole; TRIAD init time.
  * **Timeline:** a bar over `[0, t_end]` showing eclipse intervals (`info.eclipses_s`), mode segments built from received frames, and markers for `t_init`, switches, fallbacks and kicks, plus a current-time cursor.
  * **uPlot strips** (sharing the time axis, last up to 20 000 samples):
    1. nadir error and estimated nadir error in degrees (log y);
    2. knowledge error and 3σ in degrees (log y);
    3. rate in °/s (log y), with the 0.5 and 2.0 °/s thresholds from `info`;
    4. wheel speeds in rpm (3 lines).
  * `null` values are gaps, not zeros.
* **Robustness.**
  * Handle `error` messages as a toast; the socket stays open.
  * On close code 4001 ("replaced") or 4404, show a banner, with no auto-reconnect.
  * If frames arrive faster than the display rate, render only the newest per animation frame, but add **every** frame to the history.
  * Window resize must re-layout and resize the renderers.
  * No console errors in normal use.

## 3. `src/adcs/history.ts` (pure)

* `class History`: a bounded ring buffer (capacity set in the constructor) of `{t, nadir, estNadir, know, sigma3, rate, wheel0..2}`, with `null` allowed. Methods:
  * `push(frame)`;
  * `columns()` returns arrays in uPlot's aligned-data format;
  * `clear()`.
* `modeSegments(samples: {t, mode}[]) -> {mode, t0, t1}[]` merges consecutive equal modes.
* `eclipseFraction(eclipses, t_end)`, used for a readout.

## 4. Tests (vitest) and fixtures

`tests/fixtures/make_fixtures.py` is run as `.venv/Scripts/python webapp/frontend/tests/fixtures/make_fixtures.py` from the repo root. It writes `frames_fixtures.json` from the **Python** implementations, which are the oracle:
* about 20 random unit quaternions (seeded), with `common.quaternion.to_dcm(q)`;
* about 10 frames from a real `ClosedLoopSim` session built with `webapp.server.schemas.build_sim_kwargs` and `webapp.server.adcs_session.frame`. They come from different phases (before init, detumble, pointing), and include their `display.C`, `sun`, `b`, `eci.q`, `r_m`, `gmst_rad`;
* for each such frame, the expected world-coordinate body axes for the display scene (`M @ C`) and for the ECI scene (`M @ to_dcm(q)`), computed in numpy;
* Earth-texture checks: for (lat, lon) ∈ {(0, 0), (0, 90°E), (45°N, −120°), (−30°, 150°)} and 3 GMST values, the expected world position of that ground point. Compute it as `M @ ecef_to_eci(latlon → ecef, jd)` using `common.frames.eci_to_ecef` (invert by transpose, or by solving). Also give the matching UV `(u, v)` on an equirectangular texture: `u = (lon + 180°)/360°`, `v = (90° − lat)/180°`, in Three.js UV convention with v measured from the top.

`tests/frames.test.ts`:
1. `toThreeQuaternion(q)` → `Matrix4.makeRotationFromQuaternion` equals `to_dcm(q)` to 1e-12, for every fixture quaternion.
2. `displayMatrixToWorld(C)` columns equal the fixture's world body axes to 1e-6. The frames are 7-digit rounded, so the tolerance is 1e-6. The matrix is a proper rotation (det +1 within 1e-5). `C = DISPLAY_FLIP` gives body z = world (0, −1, 0).
3. `bodyInEciToWorld(q)` applied to the unit axes equals the fixture's ECI-scene world axes to 1e-12.
4. **Earth texture.** Build an actual `THREE.SphereGeometry(1, 64, 32)` mesh with `rotation.y = earthRotationY(gmst)`. For each fixture ground point, find the vertex whose UV is nearest the fixture UV. Its world position (after `mesh.updateMatrixWorld()`) must equal the fixture world position, to within the mesh's grid spacing (about 0.1 rad; compare angles < 0.06 rad). This test proves the texture orientation, so it must not be skipped.
5. `zUpToWorld` / `eciToWorld` match `M` on the fixture vectors.

`tests/history.test.ts` covers ring-buffer wraparound, `null` handling, `modeSegments` merging, and `columns()` alignment.

## 5. Exit (Group 11)

* `npm test` is green.
* `npm run build` is clean, with no TypeScript errors.
* `pytest` is green. The Python suite is unaffected, except for an optional `webapp/server/tests/test_frontend_fixtures.py` that re-runs `make_fixtures.py` into a temporary file and checks the committed JSON is up to date (allowed, not required).
* `python -m webapp` serves the built page at `http://127.0.0.1:8000`.
* Browser check (main session, claude-in-chrome): a 6-orbit realistic+design session at 500×. Expect detumble → TRIAD init → POINTING. A kick causes a fallback and recovery. Record a GIF.
