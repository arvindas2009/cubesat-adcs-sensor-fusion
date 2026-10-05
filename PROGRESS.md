# Progress

Read this file and `docs/IMPLEMENTATION_PLAN.md` at the start of every group. Updated by the main session at every group exit.

| Group | Milestones | Status |
|---|---|---|
| 1 Foundations | M0 | **done** |
| 2 A data + dead reckoning | M1, M2 | **done** |
| 3 A 15-state EKF | M3 | next |
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

## Group 2 — Project A data + dead reckoning (M1, M2) — done

**Delivered**
- `sensor_fusion/download_logs.py` (3 pinned Flight Review logs, retries, ULog-magic check), `loader.py` (pyulog → time-aligned SI arrays; `FlightLog` with `imu/mag/baro/gps/ekf2/info`; cleaning, gap report, armed-window crop), `interp.py` (`interp_to`, slerp for quaternions), `inspect_logs.py` (+ `--smoke`), `common/plotting.py`, 15 inspection figures, `sensor_fusion/data/NOTES.md`.
- `sensor_fusion/dead_reckoning.py`: `strapdown` (trapezoid gyro, mid-interval attitude for accel rotation, trapezoid v/p; vectorised prefix-scan quaternion product), `initial_attitude_from_accel_mag`.
- `common/frames.py::normal_gravity` (Somigliana + free-air) added after the ★ review.
- `CONVENTIONS.md` §6 finalised (PX4 → repo mapping; three height datums; local gravity; EKF2 = reference not truth); `assumptions.md` A-07…A-12; `docs/specs/sensor_fusion_data.md`.
- Tests: **605 passing** (`pytest`, ~17 s) — new: 34 dead-reckoning (scipy/analytic oracles, order-2 convergence), ~94 loader/download (checked against raw pyulog reads, ppigrf inclination/declination, gyro-vs-EKF2 attitude rate, accel-vs-gravity), 11 `normal_gravity`.

**Pinned logs** (all gitignored `.ulg`, 46 MB in `sensor_fusion/data/`)

| ID | Role | HW / FW | Window | IMU rate |
|---|---|---|---|---|
| `f4e81624-7ccc-4859-bb45-d02fa0246b77` | hover (slow position-mode flight) | FMU_V6C, v1.17.0 | 280 s | 221.7 Hz |
| `7cd0a812-d6d6-4c67-9da8-1f3a9cc460b0` | aggressive (stabilized, gyro to 386 °/s, ends in impact) | FMU_V6C, v1.17.0 | 169 s | 221.6 Hz |
| `78808b7c-d9e9-4350-b311-3bb510f29a7b` | other (long, 9 m/s, RTK GPS 2 Hz, old int message format) | FMU_V5, v1.14.0 | 674 s | 199.4 Hz |

Other stream rates: mag ≈ 2 Hz, baro ≈ 5–6 Hz, GPS 2–10 Hz, EKF2 attitude 20 Hz / position 10 Hz.

**★ Opus review (PX4 mapping)** — core mapping confirmed independently (R(q_EKF2)·f_b ≈ −g down; gyro vs EKF2 attitude rate rms 0.013–0.022 rad/s, zero lag; GPS vel sign; GPS→NED vs EKF2 0.05–0.3 m rms). Findings fixed in this group: red baro test (EKF2 height is GPS-referenced on 2 of 3 logs and GPS height drifts ~9 m in the aggressive log → baro test criterion changed per spec §6), wrong EKF2-origin claim in NOTES, loader dropped GPS MSL height (`gps.alt_msl` added), per-log local gravity (G0 would add up to 0.028 m/s²), mag/baro now stamped with `timestamp_sample`, GPS latency is measurable (≈0.10–0.18 s; `EKF2_GPS_DELAY` = 110 ms).

**Measured dead-reckoning drift** (raw IMU, no bias removal, seeded from EKF2 at t=0, local g; EKF2 is the reference, not ground truth; `report/dr_metrics_<id>.json`)

| Log | Pos err @10 s | @30 s | @60 s | Att err @10 s | @60 s | Growth exponent (2–30 s) |
|---|---|---|---|---|---|---|
| hover | 10.8 m | 97 m | 695 m | 1.1° | 13.3° | 1.93 |
| aggressive | 9.1 m | 109 m | 320 m | 1.5° | 5.3° | 2.20 |
| other | 2.4 m | 49.7 m | 256 m | 1.0° | 7.7° | 2.77 |

Whole-flight end errors reach 3 km – 172 km (e.g. hover 78 km after 280 s). Re-seeding after the EKF2 yaw reset (`post_reset` in the JSON) gives similar growth (exponent 1.74–2.14). Using G0 instead of local g shifts position by exactly ½|Δg|t² (hover 51 m at 60 s) — real but small next to bias-driven error. Exponents near 2 point to accel-bias-dominated error and 2.77 on the long log to gyro-bias gravity coupling; **the split was not tested** — Group 3 estimates the biases.

**Open decisions / notes for Group 3**
- Mag: fuse **tilt-compensated heading only**, σψ ≈ 3–5° (measured heading offset vs IGRF 1.2–6.6°, likely uncorrected hard-iron on the hover log; IGRF date from `time_utc_usec`). Mag is only ~2 Hz in the default log profile.
- Baro: use as height change vs mean of the first 1 s on the ground, σ ≈ 1.5 m (to be confirmed by NIS); baro offset drifts by metres and its scale differs from GPS by ~10 %.
- GPS delay: re-stamp as `t − 0.11 s` and process in time order (offline, exact); sensitivity sweep τ ∈ {0, 0.11, 0.2} s. (This resolves the Group 1 buffer-and-replay vs inflate-R question.)
- Local gravity per log via `normal_gravity` (9.7784 / 9.7859 / 9.8119 m/s²); NED origin = first valid GPS fix (ellipsoidal height); compare with EKF2 using the constant origin offset (CONVENTIONS §6).
- Each log has one EKF2 pure-yaw quaternion reset (−4.92° @16.1 s, −0.98° @4.2 s, −3.53° @7.4 s): mark or exclude in filter-vs-EKF2 yaw statistics.
- Aggressive log: baro vs GPS 10 s-increment correlation is only r = 0.34 (non-overlapping windows) because of GPS vertical drift; and the log ends in an impact at t ≈ 167 s.
- `strapdown` gyro scheme (trapezoid) carries a ≈ dt/2 lag vs PX4's interval-end means; within 0.01° of the best scheme over 1 s windows — accepted.
- Citations in `assumptions.md` A-07…A-12 marked "unverified" (Hofmann-Wellenhof & Moritz section, PX4 doc pages, IGRF details, Snyder) need checking before the writeup. Recomputed IGRF inclinations using the per-log UTC date are TBD.
- `ppigrf` dates: 78808b7c is from 2024-12-19.

**Next: Group 3** (M3: 15-state error-state EKF, three increments GPS+baro → attitude+mag → biases). Recommended main-session model: **Opus** (`/model opus`).
