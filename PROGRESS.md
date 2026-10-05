# Progress

Read this file and `docs/IMPLEMENTATION_PLAN.md` at the start of every group. Updated by the main session at every group exit.

| Group | Milestones | Status |
|---|---|---|
| 1 Foundations | M0 | **done** |
| 2 A data + dead reckoning | M1, M2 | next |
| 3 A 15-state EKF | M3 | not started |
| 4 A tuning + writeup | M4 | not started |
| 5 B physics | M5, M6 | not started |
| 6 B control | M7 | not started |
| 7 B estimation | M8 | not started |
| 8 B closed loop + MC + doc | M9 | not started |
| 9 Packaging + visualizer | M10, M11 | not started |

## Group 1 — Foundations (M0) — done

**Delivered**
- Repo scaffold: git, `.gitignore`, `.gitattributes` (LF), `pyproject.toml` (editable install), `requirements.txt`, packages `common/`, `sensor_fusion/`, `adcs/`.
- Environment: `.venv` on **Python 3.14.2** — all deps have 3.14 wheels (no 3.12 fallback needed). numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2, pytest 9.1.1, pyulog 1.2.4, ppigrf 2.1.0 (pulls pandas 3.0.6), sgp4 2.27, requests 2.34.2.
- `CONVENTIONS.md`, `docs/specs/common_api.md`, skeleton `README.md`, `assumptions.md` (G-01…G-09 filled).
- `common/quaternion.py` (+ `inject_attitude_error`, `attitude_reset_jacobian` — the single shared ES-EKF injection/reset), `common/frames.py`, `common/ekf.py` (generic ES-EKF core: Joseph form, block + sequential updates, gating, NIS log, Van Loan, chi² bounds).
- Tests: 445 passing (`pytest`, ~10 s). Quaternion verified against scipy `Rotation`; frames against closed-form WGS-84 and Vallado GMST examples; EKF against a textbook KF, DARE steady state, analytic Van Loan, Monte-Carlo NIS/NEES, nonlinear pendulum.
- Subagents defined in `.claude/agents/` (py-implementer, test-author, data-wrangler, figures-runner, doc-writer = sonnet; math-reviewer = opus).

**★ Opus math review:** no blockers. A mutation test (26 deliberate bugs) was caught 22/26; the 4 escapes are float-equivalent or immaterial. Follow-ups applied: Joseph-form test now actually discriminates vs short form; shared attitude inject/reset helpers; LVLH/NED DCM ↔ quaternion link tests (`from_dcm(C.T)` is the body-aligned quaternion); ragged `innovations()` logs with `dof`; ES-EKF contract (inject after *every* update) in `ekf.py` docstring.

**Notes for later groups**
- Custom agents in `.claude/agents/` only load at session start. In the session that created them they were run as general-purpose agents with the model set (sonnet/opus) and told to follow the agent file; from a fresh session use them by name.
- `common/plotting.py` deferred to Group 2 (owned by figures-runner).
- ppigrf 2.1.0: `ppigrf.igrf(lon_deg, lat_deg, h_km, datetime)` → `(Be, Bn, Bu)` in **nT**, local geodetic **ENU**; scalar input returns shape `(1,)` arrays (use `[0]`). Needed in Group 5.
- sgp4: `Satrec.twoline2rv` + `sat.sgp4(jd, fr)` → TEME km, km/s (convert to m).
- Group 3 decision pending (from review): PX4 GPS arrives ~100–200 ms delayed — buffer-and-replay vs ignore + inflate R. `van_loan` costs ~160 µs/step at n=15 (fine; first-order discretisation optional).
- GMST oracle (Vallado Ex. 3-5) matches to 7.6e-8°; tests use 1e-6°.

**Next: Group 2** (M1 + M2: PX4 log download/loader, inspection plots, dead-reckoning baseline). Recommended main-session model: **Sonnet** (`/model sonnet`).
