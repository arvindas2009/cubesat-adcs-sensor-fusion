# Spec: Project A tuning, validation metrics, CLI, writeup (Group 4, M4)

Source of truth for `sensor_fusion/metrics.py`, `sensor_fusion/run_flight.py`, `sensor_fusion/report/ekf_figures.py`, the sweep (`sensor_fusion/report/tuning_sweep.py`), the writeup and the ★ review. Conventions per `CONVENTIONS.md`; filter spec in `docs/specs/ekf_drone.md`. EKF2 is a **reference, not truth** everywhere (A-08).

## 1. `sensor_fusion/metrics.py` (py-implementer)

Pure functions, no plotting, no file I/O except `ekf2_bias_reference` (pyulog). Extracted **behaviour-identically** from `sensor_fusion/report/ekf_study.py` (names without the leading underscore); `ekf_study.py` is then changed to import from here.

```python
SKIP_INIT_S = 5.0; RESET_EXCL_S = 1.0; SENSOR_ORDER = ("gps_pos", "gps_vel", "baro", "mag_heading")

def clean_json(o)                                      # was _clean
def error_stats(x, scale=1.0) -> dict                  # was _stats(t, x, scale): drop the unused t argument
def tilt_angle(q1, q2) -> (N,) rad                     # as _tilt_angle
def yaw_of(q) -> (N,) rad                              # as _yaw
def reset_mask(t, resets, half=RESET_EXCL_S) -> bool (N,)
def ekf2_bias_reference(log) -> dict                   # unchanged
def compare_to_ekf2(res, ref, skip_init_s=SKIP_INIT_S, reset_excl_s=RESET_EXCL_S) -> {"cmp": ..., "arr": ...}   # was compare_run(res, ref, S)
def nis_summary(res, t_eval0) -> dict                  # unchanged output keys
def bias_summary(res) -> dict                          # unchanged
def consistency_score(nis: dict, min_updates=5) -> dict   # NEW, see below
```

`consistency_score(nis_summary_dict)`: for each sensor with `after_5s` present and `after_5s["n"] >= min_updates`, `ratio_s = after_5s["mean_nis"] / dof`, `log_ratio_s = ln(ratio_s)`. Returns `{"per_sensor": {name: {"ratio": ..., "log_ratio": ..., "fraction_inside_95": ..., "n": ...}}, "score": sqrt(mean(log_ratio_s²)), "n_sensors": k}`; `score = None` if `k == 0`. A perfectly consistent filter has ratio ≈ 1 (score ≈ 0); ratio 2 and ratio 0.5 score the same (ln 2 = 0.693).

Regression requirement: on the hover log, increment 3, default `EkfConfig`, `compare_to_ekf2` / `nis_summary` / `bias_summary` reproduce the numbers stored in `sensor_fusion/report/ekf_metrics_f4e81624-….json` (increments["3"]) to ≤ 1e-9 relative. (Do this check **before** the `EkfConfig` defaults are changed; `ekf_study.py` keeps the old defaults afterwards through an explicit `G3_DEFAULTS` dict handled by the main session.)

## 2. `sensor_fusion/report/ekf_figures.py` (py-implementer)

Move `fig_pos`, `fig_att`, `fig_bias`, `fig_nis` (and helpers `_title`, `_decim`, constants they need) from `ekf_study.py` unchanged, adding a keyword `prefix: str = "ekf"` and `out_dir: Path | None = None` so that the file name is `f"{prefix}_{log_id}_{kind}.png"` (default output identical to today). `ekf_study.py` imports them from here.

## 3. `sensor_fusion/run_flight.py` — CLI (py-implementer, then run by figures-runner)

`.venv/Scripts/python -m sensor_fusion.run_flight <log> [--increment 3] [--set key=value ...] [--no-figures] [--out-dir DIR] [--prefix final]`

- `<log>`: role (`hover`, `aggressive`, `other`), full id or id prefix of `PINNED_LOGS`; `all` runs the three.
- Runs `streams_from_log` → `run_filter(streams, EkfConfig(increment=…, **overrides))` (current `EkfConfig` defaults = the tuned values) → `ekf2_reference` → `metrics.compare_to_ekf2`, `nis_summary`, `consistency_score`, `bias_summary`, `ekf2_bias_reference`.
- Writes `<out-dir>/<prefix>_metrics_<id>.json` (default out-dir `sensor_fusion/report`, prefix `final`; same layout as the `ekf_metrics_<id>.json` of `ekf_study.py` but for the single chosen increment, plus the full `EkfConfig` and `consistency`), and, unless `--no-figures`, `figures/<prefix>_<id>_{pos,att,bias,nis}.png` through `ekf_figures` (bias/NIS only meaningful for increment 3; skip them otherwise with a printed note).
- `--set` overrides any `EkfConfig` field (`float`/`int`/`bool`/`None` parsed with `ast.literal_eval`); unknown fields → `SystemExit` with the list of valid names.
- Prints a short summary table (RMS posH/posD/vel/tilt/yaw, NIS mean/dof per sensor, consistency score, runtime). Exit code 0.
- Importable: `run_one(log_entry, increment, overrides, out_dir, prefix, figures) -> dict` (used by the sweep with `figures=False`, `write=False`).

## 4. Tests (test-author, from this spec; `sensor_fusion/tests/test_metrics.py`, `test_run_flight.py`)

Independent oracles only (hand-computed values, scipy `Rotation`, `scipy.stats.chi2`).

1. `error_stats`: known vectors (RMS of `[3,4]` = √12.5, max_abs, final, scale), empty input.
2. `tilt_angle`: identity → 0; pure yaw difference → 0 for any yaw; known roll/pitch difference equals the rotation angle between the two body-down axes (scipy). `yaw_of`: scipy `as_euler("ZYX")`.
3. `reset_mask`: returns a **keep**-mask (True = sample kept); samples with `|t − t_r| ≤ half` are False (as in the extracted `_reset_mask`, which `compare_to_ekf2` relies on).
4. `compare_to_ekf2` on a fabricated `res`/`ref` pair (small dataclass stand-ins or `EkfResult` built by hand): constant position offset `[3,4,5]` m → horizontal RMS 5, vertical RMS 5 and signed mean 5; constant velocity offset; constant yaw offset 10° → `yaw_deg.rms = 10`, tilt 0; samples within ±1 s of a reset excluded from attitude statistics and counted in `n_att_samples_excluded_resets`; first 5 s skipped.
5. `nis_summary`: synthetic innovations drawn from χ²(dof) (seeded, n large) → mean ≈ dof, `fraction_inside_95` within [0.93, 0.97], `mean_inside_bounds` True; innovations scaled ×3 → `mean_inside_bounds` False; rejected updates (`accepted=False`) not counted.
6. `consistency_score`: ratio 1 → score 0; ratios 2 and 0.5 give the same score `ln 2`; sensors with fewer than `min_updates` ignored; empty → `score is None`.
7. `bias_summary` keys and values from a hand-built result.
8. `run_flight` / `run_one`: synthetic path not available through the CLI, so (a) argument parsing: `--set gyro_noise=0.002 increment`-style overrides parsed into the right types, unknown field rejected; (b) real-log smoke (skip if the `.ulg` is missing): `run_one(hover, 3, {}, figures=False, write=False)` returns finite metrics with the keys `comparison_vs_ekf2`, `nis`, `consistency`, `final_biases`, `runtime_s`; a regression test that its horizontal-position RMS equals the value in `ekf_metrics_<hover>.json` **only while `EkfConfig` defaults are the G3 defaults** is NOT written (defaults will change) — instead test that `--set` of the G3 values (`G3_DEFAULTS` dict exported from `sensor_fusion/report/ekf_study.py`) reproduces that JSON value to 1e-9 relative (skip if the JSON/`.ulg` is missing).

Runtime budget for the new tests ≤ 40 s.

## 5. Covariance sweep (`sensor_fusion/report/tuning_sweep.py`, figures-runner)

Time-boxed, deterministic, increment 3 only (the final filter), all three pinned logs, `multiprocessing` (22 cores available; each worker loads its own log once).

- **Knobs** (all `EkfConfig` fields): `accel_noise`, `gyro_noise`, `gyro_bias_rw`, `accel_bias_rw`, `gps_pos_floor_h`, `gps_pos_floor_v`, `gps_vel_floor`, `baro_sigma`, `mag_heading_sigma`.
- **Stage 1 — one at a time:** baseline plus each knob × factor ∈ {1/4, 1/2, 2, 4} (1 + 9·4 = 37 configs × 3 logs).
- **Stage 2 — joint:** for every knob take the factor with the best stage-1 *total* score (rule in "Scoring"; ties → factor closest to 1), apply all together (candidate `joint`), then also each knob's selected factor moved one step toward 1 (`joint_halfstep`); also `joint_no_mag` (mag σ unchanged) — 3 more configs × 3 logs.
- **Time box:** `--budget-min` (default 30) wall-clock limit; stage 2 is skipped (and the report says so) if stage 1 exceeded it. Report total wall time.
- **Per (config, log) record:** `consistency_score` (+ per-sensor ratio and fraction inside 95 %), RMS posH / posD / vel / tilt / yaw vs EKF2, final bias values ±σ, runtime, finite flag, `n_rejected` per sensor.
- **Scoring (fixed here, not tuned):** per log, `score_nis` = `consistency_score["score"]`; `total_score` = mean over the three logs of `score_nis`. A configuration is **admissible** only if, on every log, posH RMS, vel RMS and tilt RMS vs EKF2 are each ≤ 1.15 × the baseline value on that log (guard against buying consistency with divergence from EKF2; EKF2 is a reference only). Among admissible configs the lowest `total_score` wins. The sweep reports the winner, the full table and the runner-up per knob; it never edits `EkfConfig`.
- **Outputs:** `sensor_fusion/report/tuning_sweep.json` (everything above, plus the knob list, factors, baseline values and git-independent metadata), `sensor_fusion/report/tuning_sweep.md` (table: config × log → score and RMS, markdown) and `figures/tuning_sweep_<knob>.png` (x = factor, log axis; two panels: consistency score per log, posH RMS per log; baseline marked) — one figure per knob, plus `figures/tuning_sweep_nis_ratio.png` (per-sensor NIS ratio for baseline vs `joint`, per log).
- Anomalies (divergence, NaN, NIS > 100) are reported, not fixed.

## 6. Final tuning (main session, with `math-reviewer` advice)

The reviewer gets `tuning_sweep.json`/`.md` and the A-03/A-04 rationale and advises which values to adopt and which NIS ratios are structurally unfixable (e.g. aggressive-log baro NIS from GPS vertical drift). The main session records the choice in `assumptions.md` (A-03/A-04 updated; new entry for the tuning procedure and the admissibility rule), changes the `EkfConfig` defaults, keeps the pre-tuning values as `G3_DEFAULTS` in `ekf_study.py`, and re-runs `run_flight all` for the final numbers.

## 7. Writeup (`sensor_fusion/report/writeup.md`, doc-writer; ~2 pages + tables)

From metrics JSON / sweep JSON / figures only (every number traceable; cite the file). Sections: (1) goal and data (3 pinned logs, EKF2 as reference, not truth); (2) filter summary (15 states, measurements, increments; one paragraph, pointing to the `ekf_drone.py` docstring); (3) results — increments table (G3 baseline, `ekf_metrics_<id>.json`) and final tuned table (`final_metrics_<id>.json`) with RMS and NIS-with-χ²-bounds per sensor; (4) tuning study — procedure, table, what moved, what did not; (5) failure modes and limits — **only those with evidence**: hover horizontal accel-bias unobservability, hover yaw offset vs EKF2 with uncorrected hard iron (A-11), baro vs GPS vertical drift on the aggressive log, GPS-delay sensitivity, mag nonlinearity at high inclination, EKF2 reset exclusion, vertical datum (A-08/§6); (6) reproduction commands. No claim about causes unless a test, a sweep row or a cited source supports it; otherwise wording "consistent with" / "not tested". "Attitude", never "altitude" except baro altitude. Figures referenced by relative path.
