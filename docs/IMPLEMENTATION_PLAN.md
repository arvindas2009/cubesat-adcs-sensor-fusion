# Build Plan v2: Space Estimation & Control (PX4 Sensor Fusion + CubeSat ADCS)

> **Version history**
> - v1 (original, unchanged copy): `C:\Users\arvin\.claude\plans\plan-cubesat-adcs-rustling-sunbeam.v1-original.md`
> - v2 (this file): adds execution **groups**, a **model assignment (Opus vs Sonnet)** per group/task, **custom subagent definitions**, and a **stop-and-wait gate** after every group. Technical scope of M0–M11 is unchanged.
> - At the start of Group 1 this file is copied into the repo as `docs/IMPLEMENTATION_PLAN.md` and kept in sync from then on.
> - v2.1: adds the status board and the "Resuming in a new chat" section below. **The repo copy (`docs/IMPLEMENTATION_PLAN.md`) is the source of truth from here on.**

## Status board

| Group | Milestones | Main session | Status |
|---|---|---|---|
| 1 Foundations | M0 | Opus | ✅ done (commit `8d78ae7`, 445 tests) |
| 2 Project A data + dead reckoning | M1, M2 | Sonnet | ✅ done (commit `ec49fa7`, 605 tests) |
| 3 Project A 15-state EKF | M3 | Opus | ✅ done (commit see `git log`, 664 tests) |
| 4 Project A tuning + writeup | M4 | Sonnet | ✅ done (commit see `git log`, 722 tests) — Project A complete |
| 5 Project B physics (orbit, environment, dynamics, disturbances) | M5, M6 | Sonnet | ✅ done (commit see `git log`, 898 tests) |
| 6 Project B control (perfect knowledge) | M7 | Opus | ✅ done (commit see `git log`, 1054 tests) |
| 7 Project B attitude determination | M8 | Opus | ✅ done (commit see `git log`, 1177 tests) |
| 8 Project B closed loop + Monte Carlo + design doc | M9 | Opus | ✅ done (commit see `git log`, 1388 tests) — Project B complete |
| 9 Packaging + 3D visualizer | M10, M11 | Sonnet | ✅ done (commit see `git log`, 1436 tests) — Monte Carlo regeneration in a fresh venv not completed (see `PROGRESS.md`) |
| 10 Browser sim: engine stepper + server API | M12 | Opus | ✅ done (commit see `git log`, 1742 tests) — engine ≈ 595× real time when the CPU is at full speed, 300–390× at other times (see `PROGRESS.md`) |
| 11 Browser sim: ADCS frontend (Vite + TS + Three.js) | M13 | Sonnet | ⏭ next |
| 12 Browser sim: PX4 log replay + docs | M14 | Sonnet | planned |

Details of what each completed group delivered, measured numbers, and open decisions live in `PROGRESS.md`.

## Resuming in a new chat

1. Open Claude Code in `D:\Projects\Coding\ubeSat ADCS + Sensor Fusion` (the project `CLAUDE.md` loads automatically and points here).
2. Switch to the main-session model listed for the next group in the status board (`/model sonnet` or `/model opus`).
3. Prompt: **"Continue with Group N per docs/IMPLEMENTATION_PLAN.md"** (N = the group marked ⏭ next).
4. Claude then: reads `PROGRESS.md` + this file + `CONVENTIONS.md` → runs the group per its task table, delegating to the named subagents (`py-implementer`, `test-author`, `data-wrangler`, `figures-runner`, `doc-writer` on Sonnet; `math-reviewer` on Opus — they load by name in a fresh session) → ★ review → exit criteria + green `pytest` → updates `PROGRESS.md` and this status board → one git commit → **stops and waits** for the next prompt.

## Context

The user has a detailed specification for two linked pure-Python projects and wants Claude to build all of it. v1 reordered the scope into dependency-ordered milestones M0–M11. The user now wants:

1. Milestones **grouped** into chunks that are naturally done together.
2. Each group / task tagged **Opus** (judgement- and math-heavy) or **Sonnet** (well-specified implementation, data plumbing, plots, docs) to save tokens.
3. **Custom subagents**, mostly Sonnet, to do the delegable work.
4. Strict order: **Project A first, then Project B**.
5. **Group-by-group execution**: after each group Claude reports and **waits for the user's next prompt** before starting the next group.
6. No code yet — this plan only.

- **Project A — Sensor fusion on real PX4 drone logs:** 15-state error-state EKF (position, velocity, attitude, gyro bias, accel bias) fusing IMU, GPS, baro, mag from real `.ulg` logs, validated against onboard EKF2.
- **Project B — 3U CubeSat ADCS simulation:** 400 km / 51.6° orbit, B-dot detumble, reaction-wheel nadir pointing with momentum dumping, TRIAD + MEKF, closed-loop Monte Carlo.

Shared spine: error-state EKF + quaternion machinery in `common/`, built once, used by both.

**User decisions (confirmed, v1):** auto-download public PX4 logs from review.px4.io with pinned IDs; Claude writes both reports from actual results; 3D visualizer included only as the final milestone.

**Environment (verified):** Windows 11, Python 3.14.2 default, **Python 3.12 also installed** (`py -3.12`) as the wheel fallback. Working dir `D:\Projects\Coding\ubeSat ADCS + Sensor Fusion\` contains only `.kilo\`. Not yet a git repo.

**Naming rule:** **attitude** determination and control, never "altitude" (except for genuine altitude, e.g. baro altitude / orbit altitude).

---

## Execution model

### How model selection works

- **Main session (orchestrator)** — the model the user has selected with `/model`. Each group below states the **recommended main-session model**; the user switches with `/model sonnet` or `/model opus` before sending the "continue" prompt. Opus is recommended only where the main session itself does the hard math/design work.
- **Subagents** — model fixed in each agent's frontmatter (`model: sonnet` / `model: opus`), independent of the main session. Sonnet agents do the bulk of the writing; the single Opus agent is a **read-only reviewer**, invoked only at the math-critical checkpoints listed below.
- **Rule of thumb used for tagging:** *Opus* = derivations, Jacobians, frame/sign conventions, filter/controller tuning judgement, diagnosing why validation fails, final technical writing. *Sonnet* = scaffolding, I/O and data plumbing, modules fully specified by an equation in this plan, test suites against an oracle, plotting, CLI, sweeps, README/packaging.

### Custom subagents (created in Group 1 as `.claude/agents/*.md` in the repo)

| Agent | Model | Tools | Purpose |
|---|---|---|---|
| `py-implementer` | sonnet | Read, Write, Edit, Bash, Glob, Grep | Implements one fully-specified module from a written spec (equations, signatures, conventions). Must read `CONVENTIONS.md` first, run `pytest` on its module before returning, return a short summary + test result. Never changes conventions or other modules' public APIs. |
| `test-author` | sonnet | Read, Write, Edit, Bash, Glob, Grep | Writes pytest suites **from the spec, not from the implementation** (independent oracles: scipy `Rotation`, `sgp4`, analytic solutions, conservation laws). Gives independence from the implementer. |
| `data-wrangler` | sonnet | Read, Write, Edit, Bash, Glob, Grep, WebFetch | PX4 Flight Review API, log screening/download, `pyulog` extraction, raw-channel inspection plots, data notes. |
| `figures-runner` | sonnet | Read, Write, Edit, Bash, Glob, Grep | Runs simulations/sweeps/CLIs, generates figures and metrics JSON, collects numbers into tables. Edits only scripts under `report/`, run/CLI files, `plotting.py`; never algorithm code. Reports anomalies instead of "fixing" them. |
| `doc-writer` | sonnet | Read, Write, Edit, Glob, Grep | README, `assumptions.md` entries with citations, `PROGRESS.md`, first drafts of report sections **from metrics JSON / figures only** (no invented numbers), "altitude" sweep. |
| `math-reviewer` | **opus** | Read, Glob, Grep, Bash (read-only use: run pytest/scripts) | Reviews math-critical code: quaternion conventions, Jacobians, frame transforms (FRD/NED, ECI/ECEF/LVLH), covariance handling, control law signs, validation results. Returns a findings list; does not edit. Invoked sparingly (marked ★ below). |

Common rules written into every agent prompt: read `CONVENTIONS.md` + the relevant milestone section of `docs/IMPLEMENTATION_PLAN.md`; SI units; scalar-first body-to-inertial quaternions; no "altitude" for attitude; keep `pytest` green; no git commits (main session commits).

No Explore/Plan agents are needed: the repo is empty and the design is already in this plan.

### Group workflow (every group)

1. Main session reads `PROGRESS.md` + this plan, briefs subagents with precise specs.
2. Independent tasks within a group run as **parallel subagents**; dependent ones run in sequence.
3. ★ checkpoints → `math-reviewer` (Opus) pass; main session applies or delegates fixes.
4. Exit criteria checked: full `pytest` green, required figures exist, numbers recorded.
5. Main session updates `PROGRESS.md` (done, numbers, open issues, next group) and the status board at the top of this file, makes **one git commit for the group**.
6. **STOP.** Report a short summary and wait for the user's next prompt. Do not start the next group.

---

## Groups (Project A → Project B)

### Group 1 — Foundations (M0) · main session: **Opus** · ✅ done
Why grouped: everything else depends on conventions + `common/`; doing it as one block fixes the API once.

| Task | Who | Model |
|---|---|---|
| Create `.claude/agents/*.md` (6 agents above); copy this plan to `docs/IMPLEMENTATION_PLAN.md`; create `PROGRESS.md` | main | Opus |
| `git init`, `.gitignore`, `pyproject.toml`, `requirements.txt`, venv, install + import check on 3.14 (fallback `py -3.12`) | `py-implementer` | Sonnet |
| `CONVENTIONS.md` (frames, quaternion, units, PX4 FRD/NED mapping placeholder), skeleton `assumptions.md`, `README.md` | main writes CONVENTIONS; `doc-writer` the rest | Opus / Sonnet |
| `common/quaternion.py` | `py-implementer` from main's exact spec | Sonnet |
| `common/tests/test_quaternion.py` (scipy oracle) | `test-author` (parallel to above) | Sonnet |
| `common/ekf.py` generic error-state EKF (Joseph form, sequential updates, NIS) | main | **Opus** |
| `common/tests/test_ekf.py` (1D CV KF, pendulum) | `test-author` | Sonnet |
| `common/frames.py` + tests (LLA↔ECEF↔NED, GMST ECI↔ECEF) | `py-implementer` + `test-author` | Sonnet |
| ★ Review quaternion + ekf + frames conventions | `math-reviewer` | Opus |

**Exit:** v1 M0 exit; agents exist; `PROGRESS.md` + `docs/IMPLEMENTATION_PLAN.md` committed. → **STOP, wait.**

### Group 2 — Project A data + dead-reckoning baseline (M1 + M2) · main session: **Sonnet**
Why grouped: both are "get real data in and see it"; the baseline is the first consumer of the loader and exposes loader bugs immediately.

| Task | Who | Model |
|---|---|---|
| `download_logs.py`: query Flight Review, screen (multirotor, GPS, duration), pick 3 logs (hover + aggressive), pin IDs | `data-wrangler` | Sonnet |
| `loader.py` (pyulog → time-aligned per-sensor arrays, gaps, units) + inspection plots + `data/NOTES.md` | `data-wrangler` | Sonnet |
| Loader tests (monotonic time, rates, no NaN) | `test-author` | Sonnet |
| ★ FRD/NED + unit mapping review; finalize that section of `CONVENTIONS.md` | `math-reviewer` | Opus |
| Strapdown dead reckoning + synthetic constant-rate test (analytic) | `py-implementer` | Sonnet |
| Drift-vs-EKF2 plots for 3 flights | `figures-runner` | Sonnet |

**Exit:** v1 M1 + M2 exits. → **STOP, wait.**

### Group 3 — Project A 15-state error-state EKF (M3) · main session: **Opus** · ✅ done
Why grouped: one filter, three increments; the derivations must be consistent across all three, so one owner.

| Task | Who | Model |
|---|---|---|
| Error-state derivation (F, G, H for GPS/baro/mag), Q/R structure, written into `ekf_drone.py` docstring | main | **Opus** |
| Increment 1 GPS+baro, 2 attitude+mag, 3 biases — implementation | main | **Opus** |
| Synthetic-trajectory truth generator + tests (convergence, SPD covariance, unit quaternion) | `test-author` | Sonnet |
| Run all 3 flights after each increment; quick plots | `figures-runner` | Sonnet |
| ★ Jacobian / sign review before increment 3 | `math-reviewer` | Opus |

**Exit:** v1 M3 exit. → **STOP, wait.**

### Group 4 — Project A tuning, validation, writeup (M4) · main session: **Sonnet**
Why grouped: all post-filter work; mostly running, plotting and writing from numbers.

| Task | Who | Model |
|---|---|---|
| Time-boxed covariance sweep script + results table | `figures-runner` | Sonnet |
| Pick final tuning values from sweep (judgement) | `math-reviewer` advises, main records | Opus (advice) |
| RMSE + NIS metrics with chi-square bounds; `run_flight.py` CLI → plots + metrics JSON | `py-implementer` → `figures-runner` | Sonnet |
| `sensor_fusion/report/writeup.md` draft from metrics JSON | `doc-writer` | Sonnet |
| ★ Technical accuracy pass on writeup (failure-mode explanations, no unsupported claims) | `math-reviewer` | Opus |

**Exit:** v1 M4 exit — **Project A done.** → **STOP, wait.**

### Group 5 — Project B physics: orbit, environment, dynamics, disturbances (M5 + M6) · main session: **Sonnet**
Why grouped: all "truth model" physics, no feedback loop yet; modules are fully specified by standard equations and independently testable, so they parallelize well.

| Task | Who | Model |
|---|---|---|
| `config.py` + inertia/orbit assumptions in `assumptions.md` | `py-implementer` + `doc-writer` | Sonnet |
| `orbit.py` (two-body + J2) ‖ `environment.py` (ppigrf IGRF, sun, eclipse, density) — **parallel** | 2× `py-implementer` | Sonnet |
| Validation tests: ISS TLE vs `sgp4`, IGRF 25–45 µT, sun at equinox/solstice, eclipse 35–40% | `test-author` | Sonnet |
| `dynamics.py` (quaternion kinematics + Euler with wheel momentum) ‖ `disturbances.py` — **parallel** | 2× `py-implementer` | Sonnet |
| Conservation tests (H, T over 10 orbits; symmetric-top precession) | `test-author` | Sonnet |
| Orbit-vs-SGP4 and disturbance-magnitude plots | `figures-runner` | Sonnet |
| ★ Review ECI↔ECEF for IGRF, Euler-with-wheels equation, gravity-gradient sanity gate | `math-reviewer` | Opus |

**Exit:** v1 M5 + M6 exits. → **STOP, wait.**

### Group 6 — Project B control, perfect knowledge (M7) · main session: **Opus** · ✅ done
Why grouped: detumble, pointing and dumping share the actuator models and interact (dumping uses torquers while wheels point).

| Task | Who | Model |
|---|---|---|
| `actuators.py` (wheel torque/speed limits from cited datasheet; torquer 0.2 A·m², duty cycle) | `py-implementer`; datasheet citations by `doc-writer` | Sonnet |
| B-dot law, quaternion-error PD gains, cross-product dump law + threshold — design and tuning | main | **Opus** |
| Tests (B-dot removes energy, saturation clamps, dumping bounds wheel speed) | `test-author` | Sonnet |
| Sensitivity sweeps (initial rate, torquer strength), 15-orbit pointing/wheel plots | `figures-runner` | Sonnet |

**Exit:** v1 M7 exit (<0.5 deg/s detumble in 1–3 orbits, <1° nadir with perfect knowledge). → **STOP, wait.**

### Group 7 — Project B attitude determination (M8) · main session: **Opus** · ✅ done
Why grouped: sensors, TRIAD and MEKF form one estimation stack validated together (TRIAD seeds MEKF).

| Task | Who | Model |
|---|---|---|
| `sensors.py` (mag, CSS dead in eclipse, gyro ARW + bias RW; cited params) | `py-implementer` + `doc-writer` | Sonnet |
| TRIAD + random-attitude tests | `py-implementer` + `test-author` | Sonnet |
| MEKF on `common/ekf.py` (6-state error, mag + sun updates, init from TRIAD) | main | **Opus** |
| MEKF tests (bias convergence, NIS, eclipse error growth) | `test-author` | Sonnet |
| Estimation-error / NIS / eclipse plots over several orbits | `figures-runner` | Sonnet |
| ★ MEKF review | `math-reviewer` | Opus |

**Exit:** v1 M8 exit. → **STOP, wait.**

### Group 8 — Project B closed loop, Monte Carlo, design document (M9) · main session: **Opus** · ✅ done
Why grouped: Monte Carlo and the design document are both direct outputs of the closed-loop sim.

**Before writing the Group 8 spec:** read `PROGRESS.md` → Group 7 → "Findings / open items for Group 8" (onboard field model ≠ truth, magnetometer bias, CSS FOV/albedo, in-loop gyro sampling, public MEKF stepper, TRIAD minimum angle, gating, detumble→pointing switch on the estimate) and `assumptions.md` B-20 "Scope and optimism". Review scratch scripts behind those numbers: `docs/review_scratch/group7/`.

| Task | Who | Model |
|---|---|---|
| `simulate.py` mode machine (detumble → TRIAD init → MEKF pointing), controller on **estimated** attitude; debug integration issues | main | **Opus** |
| Closed-loop vs perfect-knowledge comparison plots | `figures-runner` | Sonnet |
| `monte_carlo.py` (100 runs, multiprocessing, randomized tumble/seeds/±10% inertia) + histograms | `py-implementer` → `figures-runner` | Sonnet |
| `adcs/report/design_document.md` (6–10 pp) — section drafts from results | `doc-writer` | Sonnet |
| Design document: requirements, architecture rationale, failure-mode analysis, final edit | main | **Opus** |

**Exit:** v1 M9 exit — **Project B done.** → **STOP, wait.**

### Group 9 — Packaging, reproducibility, visualizer (M10 + M11) · main session: **Sonnet**

**Before starting:** read `PROGRESS.md` → Group 8 "Decisions and findings". One-command reproduction for Project B: `python -m adcs.simulate` (closed loop, default realistic sensors + design filter), `python -m adcs.report.closed_loop_figures` (~12 min), `python -m adcs.monte_carlo` (~45 min on 20 processes), `python -m adcs.report.mc_worst_runs`, plus the Group 5–7 figure scripts. Headline numbers: `adcs/report/design_document.md` (Summary) and `adcs/report/monte_carlo_summary.md`.
Why grouped: both are polish on finished work; M11 still gated on M10 passing within the group.

| Task | Who | Model |
|---|---|---|
| Final README (headline numbers, one-command reproduction) | `doc-writer` | Sonnet |
| Fresh-venv smoke test, regenerate every figure | `figures-runner` | Sonnet |
| "altitude"/"attitude" sweep across all text | `doc-writer` | Sonnet |
| `visualize.py` matplotlib 3D animation → MP4/GIF (only after smoke test passes) | `py-implementer` | Sonnet |

**Exit:** v1 M10 + M11 + end-to-end verification below. → **Done.**

---

## Extension: browser 3D simulator (Groups 10–12, M12–M14)

Added 2026-10-30 after Group 9. Full approved plan: `C:\Users\arvin\.claude\plans\lexical-chasing-ocean.md`. User decisions:
- **live stepper** (stream state step by step: play/pause, speed, mid-run "kick" disturbances);
- **Vite + TypeScript** frontend with Three.js and uPlot;
- **Project B live sim plus a Project A PX4 log replay page**.

The app is a **visual aid only** and adds no numbers to any report. Project A screens always say "EKF2 is a reference, not truth".

**Layout:**
- `adcs/stepper.py`: `ClosedLoopSim`, the step-by-step engine.
- `webapp/server/`: FastAPI app, schemas, ADCS session, PX4 replay; tests in `webapp/server/tests/`.
- `webapp/frontend/`: Vite + TS. Contains `src/math/frames.ts`, `src/adcs/{OrbitView,SatView,Panels}.ts` and `src/px4/ReplayView.ts`, with vitest tests.
- Launch: `python -m webapp`.
- Web dependencies are an optional extra, `pip install -e .[web]` (fastapi, uvicorn[standard], httpx); `requirements.txt` stays unchanged.
- `.gitignore` gets `webapp/frontend/node_modules/` and `webapp/.cache/`.

**New subagent:** `.claude/agents/web-implementer.md` (Sonnet) covers TS, Three.js, Vite and FastAPI, with the same project rules as `py-implementer`.

### Group 10 — Engine stepper + server API (M12) · main session: **Opus**

**Engine refactor:**
- The body of `run_closed_loop` (`adcs/simulate.py`) moves into `ClosedLoopSim(…, t_end)`. The horizon is needed because ephemeris, field table and Gauss-Markov truth series are precomputed over it.
- Methods: `step(n=1)`, `snapshot()`, `kick(dw_body)` (adds Δω to the truth body rate between steps and logs it to `meta["kicks"]`), `done`, `result()`.
- `run_closed_loop` becomes a thin wrapper around it, so there is one code path.

**Equivalence gate:**
- Golden outputs are captured from the **current** code *before* any edit, into `adcs/tests/data/closed_loop_golden.npz`: 1.5 orbits, `log_every=10`, ideal+G7 and realistic+design, seeds 0 and 1.
- After the refactor, every array and `meta` must match bit for bit.
- Existing closed-loop tests must pass unchanged.

**Server:**
- `POST /api/adcs/sessions` takes orbits ≤ 15, rate, seed, sensor and filter presets, dumping and inertia scale.
- `WS /api/adcs/sessions/{id}/ws`:
  - frames at about 30 Hz;
  - commands `play`, `pause`, `speed` (1–2000×), `kick`, `reset`;
  - stepping runs in `asyncio.to_thread`, and frames report the achieved speed.
- Protocol spec: `docs/specs/webapp_protocol.md`.

| Task | Who | Model |
|---|---|---|
| Specs `docs/specs/stepper.md`, `docs/specs/webapp_protocol.md` | main | Opus |
| Capture golden npz from current `run_closed_loop` (before any edit) | `figures-runner` | Sonnet |
| `adcs/stepper.py` + `run_closed_loop` wrapper | `py-implementer` | Sonnet |
| Stepper tests: golden equivalence, snapshot fields, kick = exact Δω, `done`/horizon errors | `test-author` | Sonnet |
| ★ Review refactor + kick semantics | `math-reviewer` | Opus |
| `webapp/server/` (app, schemas, adcs_session), `python -m webapp`, `[web]` extra | `py-implementer` | Sonnet |
| Server tests: TestClient REST + WebSocket (frames arrive, t monotonic, pause stops t, kick logged, bad params → 422) | `test-author` | Sonnet |

**Exit:**
- Full `pytest` green, including golden equivalence.
- Measured step rate recorded in `PROGRESS.md` (target ≥ 500× real time on one core).
- A scripted WebSocket client receives frames.

### Group 11 — ADCS frontend (M13) · main session: **Sonnet**

Three.js stores quaternions as `(x,y,z,w)`. Every conversion goes through `src/math/frames.ts`, which is tested against JSON fixtures generated by Python (`common.quaternion`, `adcs.visualize.body_dcm_lvlh`). These fixtures are the cross-language oracle.

The Earth texture is NASA Blue Marble (public domain, license logged). If that fails, the fallback is a procedural sphere.

| Task | Who | Model |
|---|---|---|
| `web-implementer` agent file; Vite/TS scaffold; `frames.ts` + Python fixture script + vitest | `web-implementer` | Sonnet |
| OrbitView (Earth rotating with GMST, trail, eclipse, Sun), SatView (3U body in LVLH display frame, estimate ghost, axes, nadir/Sun/B arrows, wheel bars), Panels (parameter form, transport, kick, mode timeline, uPlot strips) | `web-implementer` | Sonnet |
| Earth texture via asset-fetch | main | Sonnet |
| Browser check (claude-in-chrome): 6-orbit realistic+design at 500×, detumble → TRIAD → POINTING, kick → fallback/recovery, GIF | main | Sonnet |

**Exit:** `npx vitest run` green, `npm run build` clean, `pytest` green, browser-check GIF saved.

### Group 12 — PX4 log replay + docs (M14) · main session: **Sonnet**

`GET /api/px4/logs` lists the pinned logs and whether each is present. Missing logs get a hint: `python -m sensor_fusion.download_logs`.

`POST /api/px4/replay/{log}`:
- runs `run_filter` and `ekf2_reference` through `run_flight.load_context`/`build_config`;
- caches the result in `webapp/.cache/<id>.npz`;
- decimates it to 50 Hz;
- the browser then plays and scrubs it.

| Task | Who | Model |
|---|---|---|
| `webapp/server/px4_replay.py` + endpoints + cache | `py-implementer` | Sonnet |
| Replay tests (synthetic log from `sensor_fusion/synthetic.py` matches `run_filter`; missing-log hint; cache reuse) | `test-author` | Sonnet |
| `ReplayView.ts`: drone attitude ours vs EKF2, NED path, scrub/speed, difference + NIS plots, permanent EKF2 caveat, yaw-reset markers | `web-implementer` | Sonnet |
| README "Interactive simulator", CLAUDE.md commands, PROGRESS.md | `doc-writer` | Sonnet |
| Browser check of all three pinned logs | main | Sonnet |

**Exit:** all suites green, browser check passes, and the README gives a one-command launch.

**End-to-end verification (after Group 12):**
1. `pip install -e .[web]`, then `npm ci && npm run build` in `webapp/frontend`.
2. `pytest` and `npx vitest run` green.
3. `python -m webapp` → `http://localhost:8000`.
4. A realistic+design session (10 °/s) detumbles, initialises TRIAD and switches to POINTING within about 1.2–1.9 orbits. The nadir error settles to about 1°. A kick triggers fallback and recovery.
5. The PX4 `hover` replay ghost tracks EKF2.

### Model summary

| Group | Milestones | Main session | Opus usage |
|---|---|---|---|
| 1 Foundations | M0 | Opus | CONVENTIONS, `ekf.py`, ★ review |
| 2 A data + DR | M1, M2 | Sonnet | ★ FRD/NED review |
| 3 A EKF | M3 | Opus | derivation + implementation, ★ review |
| 4 A tuning + writeup | M4 | Sonnet | tuning advice, ★ writeup review |
| 5 B physics | M5, M6 | Sonnet | ★ frames/dynamics review |
| 6 B control | M7 | Opus | control design + tuning |
| 7 B estimation | M8 | Opus | MEKF, ★ review |
| 8 B closed loop + MC + doc | M9 | Opus | integration, design doc |
| 9 Packaging + viz | M10, M11 | Sonnet | none |
| 10 Browser sim engine + server | M12 | Opus | specs, ★ refactor review |
| 11 Browser sim ADCS frontend | M13 | Sonnet | none |
| 12 Browser sim PX4 replay + docs | M14 | Sonnet | none |

---

## Repository structure (target)

Unchanged from v1, plus:

```
  .claude/agents/           # py-implementer, test-author, data-wrangler, figures-runner, doc-writer, math-reviewer
  docs/IMPLEMENTATION_PLAN.md
  PROGRESS.md               # per-group status, numbers, open issues; read at start of every group
```

```
ubeSat ADCS + Sensor Fusion/
  README.md  CONVENTIONS.md  assumptions.md  requirements.txt  pyproject.toml
  common/        quaternion.py  ekf.py  frames.py  plotting.py  tests/
  sensor_fusion/ download_logs.py  loader.py  ekf_drone.py  run_flight.py  tests/  report/  data/ (gitignored)
  adcs/          config.py  orbit.py  environment.py  dynamics.py  disturbances.py  actuators.py
                 sensors.py  estimation.py  control.py  simulate.py  monte_carlo.py  visualize.py  tests/  report/
```

## Global conventions (fixed in Group 1, never changed)

- Quaternions **scalar-first** `[w, x, y, z]`, unit norm, **body-to-inertial**.
- Inertial ECI (true-of-date via GMST rotation only, ≈ J2000 within 0.5°; see CONVENTIONS.md §4); NED for drone, LVLH for nadir pointing; PX4 FRD body mapped explicitly.
- SI units; radians internally, degrees only in plots/reports.
- Error-state EKF everywhere: 3-element attitude error in covariance.
- pytest tests per module before the next module; `pytest` green at every group exit.

## Dependencies

`numpy`, `scipy`, `matplotlib`, `pytest`, `pyulog`, `ppigrf`, `sgp4`, `requests`. Checked on 3.14 in Group 1; fallback `py -3.12` venv (confirmed installed).

---

## Milestone technical detail (unchanged from v1)

- **M0** quaternion (scipy oracle tests), generic error-state EKF (Joseph form, sequential updates, NIS; 1D CV + pendulum tests), frames (LLA↔ECEF↔NED, GMST).
- **M1** pinned public logs (3: hover + aggressive), pyulog loader for `sensor_combined`/`vehicle_imu`, `vehicle_magnetometer`, `vehicle_air_data`, `vehicle_gps_position`, EKF2 `vehicle_local_position`/`vehicle_attitude`; inspection plots; loader tests.
- **M2** IMU-only strapdown; drift vs EKF2; analytic constant-rate test.
- **M3** 15-state ES-EKF in 3 increments (GPS+baro → attitude+mag → biases); synthetic-truth tests; SPD covariance; normalized quaternion.
- **M4** time-boxed covariance sweep; RMSE + NIS with chi-square bounds; `run_flight.py` CLI; ~2-page writeup incl. failure modes.
- **M5** config (3U, 4 kg, box inertia), two-body+J2 `solve_ivp`, IGRF via ppigrf, sun vector, cylindrical eclipse, exponential density; ISS TLE vs sgp4 1 day; IGRF 25–45 µT; eclipse 35–40%.
- **M6** dynamics with wheel momentum; gravity gradient / residual dipole / aero torque; GG-dominates sanity gate; conservation over 10 orbits; symmetric-top precession.
- **M7** B-dot (0.2 A·m², duty cycling) 10→<0.5 deg/s, 1–3 orbits; PD on cited wheels; cross-product dumping, 15 orbits on/off; <1° perfect-knowledge nadir.
- **M8** sensors with cited params; TRIAD; MEKF (6-state error) on `common/ekf.py`; eclipse degradation; NIS; bias convergence.
- **M9** closed loop on estimated attitude with mode machine; cost of estimation; 100-run MC (multiprocessing); 6–10 pp design document.
- **M10** README with real numbers + one-command reproduction; fresh-venv smoke test; "altitude" sweep.
- **M11** matplotlib 3D animation (body axes, nadir, sun) → MP4/GIF.

## Validation doctrine

Every reported number is from a conservation law, an independent library (scipy Rotation, sgp4, ppigrf), the log's EKF2, or a cited published figure. All assumptions in `assumptions.md`. `doc-writer` drafts may only use numbers present in metrics JSON / test output.

## Risks

- Python 3.14 wheel gaps → fallback 3.12 (installed).
- Flight Review API awkward → hand-picked pinned IDs.
- PX4 frames/units → ★ review in Group 2, tests.
- IGRF needs ECEF → GMST rotation in `frames.py`, ★ review in Group 5.
- Tuning rabbit hole → time-boxed.
- Monte Carlo wall time → multiprocessing, coarser tolerance for MC only (stated).
- **Sonnet agent drift** (silently changing conventions/APIs, "fixing" failing validation by loosening tests) → agents forbidden from editing conventions/other APIs/test thresholds; `figures-runner` reports anomalies instead of fixing; ★ Opus review at math-critical points.
- **Context loss between groups** → `PROGRESS.md` + `docs/IMPLEMENTATION_PLAN.md` read at the start of each group.

## Verification (end-to-end, after Group 9)

1. `pytest` from repo root: all suites green.
2. `python -m sensor_fusion.download_logs` then `python -m sensor_fusion.run_flight <id>` per pinned log: plots + metrics JSON match the writeup.
3. `python -m adcs.simulate`: detumble <0.5 deg/s, closed-loop nadir target met.
4. `python -m adcs.monte_carlo`: 100 runs, histograms regenerate.
5. Fresh-venv reproduction confirms README one-command claims.

Per-group verification: the group's exit criteria + green `pytest` + updated `PROGRESS.md`, then stop and wait for the user.
