# Project A data notes (PX4 logs, Group 2 / M1)

Everything below was measured on the three pinned logs with `sensor_fusion/loader.py` and
pyulog; reproduce with `.venv/Scripts/python -m sensor_fusion.inspect_logs --smoke` and the
plots with `.venv/Scripts/python -m sensor_fusion.inspect_logs`
(`sensor_fusion/report/figures/data_<id>_{imu,mag,alt,track,attitude}.png`).
Logs are downloaded by `python -m sensor_fusion.download_logs` into this directory (`*.ulg` gitignored).

Revised after the Group 2 ★ review: items marked "(Group 2 ★ review)" are results of the review's
independent checks and supersede the earlier loader-only statements. The fixed PX4 -> repository mapping
is in `CONVENTIONS.md` section 6 (not repeated here); this file holds the evidence. Order of values
is always hover / aggressive / other unless stated.

## 1. Screening (one-off, results hard-coded in `PINNED_LOGS`)

* One request to `review.px4.io/browse_data_retrieval` (DataTables endpoint; needs `draw`,
  `start`, `length`, `order[0][*]` and `columns[i][*]` params, plain `length=500` gives HTTP 400)
  returned the 500 most recent public logs. Filter: Quadrotor, real hardware (not `PX4_SITL`),
  2-15 min -> 80 rows -> 13 candidates chosen by flight mode / hardware diversity and downloaded
  to a scratch directory (1 s between files); the 3 pinned logs were then re-downloaded through
  `download_pinned()` so the pinned path is what produced `sensor_fusion/data/*.ulg`.
* Criteria: multirotor, GPS 3D fix for the whole flight, 2-15 min, topics `sensor_combined`,
  `vehicle_magnetometer`, `vehicle_air_data`, `vehicle_gps_position`, `vehicle_local_position`,
  `vehicle_attitude`, `vehicle_status` present, < 40 MB, IMU >= ~200 Hz preferred, plausible
  magnetometer norm (35-60 uT).
* Rejected after download (examples): `ee7064eb` (very gentle hover but |B| = 12.5 +/- 11 uT, i.e.
  badly calibrated/disturbed magnetometer); `16c1acde` (GPS fix_type 0, 183 Hz IMU);
  `3330d636` (armed only 52 % of the log); `7da4a8e9` (no magnetometer / GPS topics);
  `0e3b91fd` (GPS fix 0 rows, same vehicle as the "other" log).

## 2. Pinned logs

| ID | Role | Hardware / FW | Airframe (SYS_AUTOSTART) | Log / window | Flight character |
|---|---|---|---|---|---|
| `f4e81624-7ccc-4859-bb45-d02fa0246b77` | hover | PX4_FMU_V6C, v1.17.0 (git d6f12ad) | 4001 (generic quad X) | 280.9 s log, 280.0 s window | Position mode, slow: GPS speed <= 2.2 m/s, climb <= 0.8 m/s, gyro rms 11 deg/s (p99 26, max 63 deg/s); site at ~4200 m MSL (27.51 N, 92.11 E); GPS fix_type 3, 19-20 sats |
| `7cd0a812-d6d6-4c67-9da8-1f3a9cc460b0` | aggressive | PX4_FMU_V6C, v1.17.0 (git d6f12ad) | 4015 (Holybro S500) | 170.0 s log, 169.4 s window | Stabilized (manual) with repeated roll/pitch/yaw rate bursts: gyro rms 29.5 deg/s, p99 115, max 386 deg/s; 3.9 m/s horizontal, 2.1 m/s vertical; ends with a hard landing/impact at t = 167 s (accel z to -140 m/s^2 -- not a data error); Mumbai (19.13 N, 72.92 E); fix_type 4, 24-32 sats |
| `78808b7c-d9e9-4350-b311-3bb510f29a7b` | other | PX4_FMU_V5, v1.14.0 (git 56f05a5) | 4002 | 675.4 s log, 674.5 s window | Position mode, long; up to 9.1 m/s and 28 m height change, gyro rms 49 deg/s (p99 116, max 199 deg/s); RTK (fix_type 6 for 98 %, 4 for 2 %), eph 0.02 m, GPS only 2 Hz; OLD message format (see 4); Oxford area (51.77 N, 1.38 W) |

"Hover" = gentle slow flight, not a stationary hover (the track covers ~14 m).
Both v1.17.0 logs are from the same board family (V6C); the third differs in firmware
generation, GPS rate/format and field names, which is the point of "other".

Sample rates actually present in the logs (median 1/dt, after cleaning, in the flight window):

| Stream (topic) | hover | aggressive | other |
|---|---|---|---|
| IMU (`sensor_combined`) | 221.7 Hz | 221.6 Hz | 199.4 Hz |
| Magnetometer (`vehicle_magnetometer`) | 2.04 Hz | 2.23 Hz | 2.03 Hz |
| Baro (`vehicle_air_data`) | 5.78 Hz | 5.78 Hz | 4.98 Hz |
| GPS (`vehicle_gps_position`) | 5.00 Hz | 9.80 Hz | 2.00 Hz |
| EKF2 attitude (`vehicle_attitude`) | 20.2 Hz | 20.1 Hz | 19.9 Hz |
| EKF2 position (`vehicle_local_position`) | 10.1 Hz | 10.1 Hz | 10.0 Hz |

**Important:** the default PX4 logging profile (`SDLOG_PROFILE = 1`, all three logs) records the
magnetometer at ~2 Hz, `vehicle_imu` at 2 Hz, `sensor_mag` at 1 Hz, EKF2 attitude at 20 Hz.
The "native" mag rate for the filter is therefore only ~2 Hz (not 50-100 Hz).
IMU interval is 4.5 ms (221 Hz) on V6C (`IMU_INTEG_RATE = 200` is the target, actual dt is set by
the driver sample rate) and 5.0 ms on V5; `gyro_integral_dt`/`accelerometer_integral_dt`
in `sensor_combined` match the log spacing (e.g. 4512 us).

## 3. Topics used per stream

| Stream | Topic (instance) | Fields -> repository quantity |
|---|---|---|
| IMU | `sensor_combined` (only instance 0 exists) | `gyro_rad[3]` rad/s -> `gyro`; `accelerometer_m_s2[3]` m/s^2 -> `accel`; time = `timestamp` |
| (fallback IMU) | `vehicle_imu` (instance 0 used; 2-3 instances logged) | `delta_angle[3]/delta_angle_dt`, `delta_velocity[3]/delta_velocity_dt` (dt in us); only 2 Hz in these logs, so **not usable for strapdown** -- implemented and checked (`load_log(..., imu_source="vehicle_imu")` gives the same gyro rms and accel median as `sensor_combined`) |
| Mag | `vehicle_magnetometer` (instance 0) | `magnetometer_ga[3]` gauss -> x1e-4 -> tesla |
| Baro | `vehicle_air_data` | `baro_pressure_pa` Pa; `baro_alt_meter` m (up); temperature `ambient_temperature` (v1.17) or `baro_temp_celcius` (v1.14), degC |
| GPS | `vehicle_gps_position` | new: `latitude_deg`, `longitude_deg` deg, `altitude_ellipsoid_m`, `altitude_msl_m` m; old: `lat`, `lon` int deg*1e-7, `alt_ellipsoid`, `alt` int mm; both: `vel_n/e/d_m_s`, `eph`, `epv`, `s_variance_m_s`, `fix_type`, `satellites_used`, `timestamp_sample` |
| EKF2 att | `vehicle_attitude` | `q[4]` (w,x,y,z), FRD->NED |
| EKF2 pos/vel | `vehicle_local_position` | `x,y,z` (NED m), `vx,vy,vz` (NED m/s), `ref_lat`, `ref_lon` (deg), `ref_alt` (m AMSL) |
| Flight window | `vehicle_status.arming_state == 2` (armed) | fallbacks `actuator_armed.armed`, `vehicle_land_detected.landed == 0` (not needed here) |

## 4. Units and frames found (per log unless stated)

* **Accelerometer**: m/s^2, FRD, **specific force**. At rest/hover the z axis reads about -9.8.
  Check: `R(q_EKF2) f_b` has median [0.24, -0.02, -9.79] (hover), [-0.006, -0.005, -9.77]
  (aggressive), [0.01, 0.05, -9.84] (other) m/s^2, i.e. `[0, 0, -g]` in NED, consistent with
  `f_b = R^T(a_n - g_n)` and `g_n = [0, 0, +g]`. High vibration: mean |f_b| is 12.7 / 10.1 / 12.8 m/s^2
  (norm of a noisy signal is biased high; the medians above are fine).
* **Local gravity** (Group 2 ★ review): normal gravity at the NED origin is 9.7785 / 9.7859 / 9.8119 m/s^2
  (`common.frames.normal_gravity`); the measured `R(q_EKF2) f_b` down component is 9.786 / 9.782 / 9.814 m/s^2.
  `G0` = 9.80665 m/s^2 is therefore wrong by up to 0.028 m/s^2 on the hover log (4.2 km site). The per-log
  value is used for real logs; see `assumptions.md` A-07. (The measured values include accelerometer bias
  and vibration rectification, so they are a plausibility check, not a gravity measurement.)
* **IMU sample semantics** (Group 2 ★ review): the `sensor_combined` gyro and accelerometer values are means
  over the integration interval ending at `timestamp`, not instantaneous samples (the loader uses
  `timestamp` as the sample time; the half-interval offset, half of the 4.5 ms interval on V6C, is far below the 0.11 s GPS delay).
* **Gyro**: rad/s, FRD, mean rate over `gyro_integral_dt`. Integrating the gyro over 1 s windows
  starting from the EKF2 quaternion reproduces the EKF2 quaternion 1 s later to a median error of
  0.17 / 0.20 / 0.27 deg (p90 0.30 / 0.38 / 0.53 deg), so gyro axes/signs and q convention agree
  (scalar-first Hamilton, FRD->NED, `q_{k+1} = q_k (x) exp(w dt)`).
* **Magnetometer**: gauss in the log (`magnetometer_ga`). Converted to tesla. |B| = 48.2 +/- 0.8,
  45.1 +/- 0.6, 49.3 +/- 1.1 uT (hover/aggressive/other). Frame: **body FRD** (after PX4's board-rotation
  and calibration handling). Evidence: the body-frame field has std of 1.2-30 uT per axis while
  `R(q_EKF2) B_b` has std 0.5-1.8 uT, i.e. it is a nearly constant NED vector with the right
  (down-positive) sign; a frame error would not give a constant NED vector at all.
  `CAL_MAG0_ROT` = -1/0/-1 and `SENS_BOARD_ROT` = 0. EKF2 mag fusion type `EKF2_MAG_TYPE = 0` (automatic).
  * Comparison with IGRF (Group 2 ★ review): declination of the measured field rotated to NED is
    6.28 / 1.22 / 3.46 deg against IGRF -0.31 / 0.00 / 0.45 deg, i.e. a heading disagreement of
    6.6 / 1.2 / 3.0 deg; the median angle between the measured NED field and the IGRF vector is
    6.0 / 2.7 / 1.6 deg.
  * Likely cause on the hover log: uncorrected hard-iron, mean body-frame bias about [4.2, 1.4, 1.7] uT.
    It is unobservable on that log because the yaw range is only 8.4 deg. The other two logs have yaw
    ranges of 1229 / 3431 deg (aggressive / other) and body-frame bias < 1 uT, so their calibration is adequate.
  * IGRF date: the model date must come from `vehicle_gps_position.time_utc_usec` (e.g. `78808b7c` was
    flown on 2024-12-19, not in 2026). The inclination figures in the first revision of this file
    (47.5 / 26.3 / 66.7 deg measured vs 43.7 / 28.4 / 66.7 deg IGRF) used the wrong model date
    (2026-09-15) and are superseded by the numbers above; recomputed inclinations: TBD(rerun with IGRF
    dated from `time_utc_usec`; the UTC dates of the other two logs are also TBD).
* **Baro**: Pa and degC; `baro_alt_meter` is height **up** (increases with climb: level
  correlation with GPS altitude 0.998 / 0.015 / 0.973; increments positively correlated with EKF2 `-z`).
  It is **not** zero at take-off nor equal to GPS height: it is the ISA altitude for the
  measured pressure with `SENS_BARO_QNH = 1013.25`, so baro - GPS MSL = +16.3 / +6.8 / +27.8 m
  on the three logs. Use only changes in baro altitude, or estimate the offset.
  * Offset is not constant (Group 2 ★ review): std of (baro - EKF2 height) is 1.0 / 3.2 / 1.3 m.
    Scale of baro height change versus GPS height change: 0.91 on hover and 1.11 on other (aggressive: not
    reported, GPS height drifts, see below).
  * `ambient_temperature` is a 15 degC placeholder on v1.17 (not a measurement); the v1.14 log has a
    real `baro_temp_celcius`. Do not use the v1.17 temperature for density/ISA corrections.
  * `EKF2_HGT_REF` = 1 (GPS height reference) on both v1.17 logs and 0 (baro) on the v1.14 log,
    so EKF2 height follows GPS on hover/aggressive and baro on other.
  * The low baro-GPS correlation on the aggressive log (0.015) is caused by **GPS vertical drift**, not by
    baro noise or the final impact: GPS height relative to take-off is -9.2 / -11.6 / -8.9 m at
    t = 40 / 100 / 169 s, while baro stays within 0.6-3.0 m and ends at -0.15 m
    (Group 2 ★ review). GPS height is therefore unreliable over minutes on that log.
* **GPS**: lat/lon, new format in degrees (float64), old format int32 degrees x 1e-7 (confirmed on
  `78808b7c`: lat = 517744331 -> 51.7744331 N). The loader stores radians. Height: `alt` = **WGS-84 ellipsoid**
  height (new `altitude_ellipsoid_m`; old `alt_ellipsoid` mm), the MSL value is not used
  (geoid separation is large: hover site 4223.1 m MSL vs 4184.2 m ellipsoid; Mumbai 52.9 vs -12.7 m;
  Oxford 75.2 vs 122.4 m). `vel_n/e/d_m_s` NED m/s; `eph`/`epv` m 1-sigma-like accuracy;
  `s_variance_m_s` speed accuracy m/s. `timestamp_sample == timestamp` in all three logs (median latency 0 us),
  so PX4 does not stamp the receiver latency. Rows with `fix_type < 3` do not occur (fix types 3 / 4 / 4+6).
  * GPS latency **is measurable** (Group 2 ★ review): the best lag is 0.10 / 0.14 / 0.18 s from GPS velocity vs
    IMU-integrated velocity, and 0.10 / 0.11 / 0.15 s from GPS velocity vs EKF2 velocity.
    `EKF2_GPS_DELAY` = 110 ms in the logs. The earlier statement that latency is not observable is withdrawn.
* **EKF2 position/velocity**: NED m, m/s about `(ref_lat, ref_lon)`; `ref_alt` is **MSL** (4220.5 / 61.4 / 64.7 m).
  * Origin (Group 2 ★ review): EKF2 sets its origin at the first GPS fix after boot, **before logging starts**,
    so the origin is not our first logged GPS sample and EKF2 position at loader t = 0 is not zero:
    (x, y, z) = (0.97, -1.26, 11.70) m hover, (-3.91, 0.24, 1.86) m aggressive, (0.01, 0.00, 1.22) m other.
  * Height datum: `ref_alt` is MSL, e.g. 4220.51 m on hover against a GPS MSL of 4208.90 m at t = 0
    (the difference is the 11.70 m vertical offset above, within GPS noise). Over the flight, the median of
    GPS MSL - (`ref_alt` - z) is 0.07 / -0.09 / 0.60 m.
  * Agreement: GPS (NED about the EKF2 reference) minus EKF2 horizontal position:
    rms 0.05 / 0.15-0.21 / 0.24-0.31 m; velocity rms 0.05-0.09 / 0.14-0.27 / 0.17-0.26 m/s.
  * Projection (Group 2 ★ review): PX4 uses a spherical azimuthal-equidistant projection (R = 6371 km);
    `common.frames.lla_to_ned` is an ellipsoidal tangent plane. They differ by <= 0.32 m on the 99 m RTK
    track. Small against the 0.05-0.3 m GPS/EKF2 agreement above but not zero (see `assumptions.md` A-12).
* **EKF2 attitude**: unit norm to 1.5e-7; canonicalised (w >= 0) by the loader.
  **Each log contains one EKF2 quaternion reset in flight** (Group 2 ★ review: pure yaw, -4.92 deg at
  t = 16.06 s (hover), -0.98 deg at 4.21 s (aggressive), -3.53 deg at 7.38 s (other), loader time);
  there are **no position or velocity resets**. The reference attitude jumps in yaw only (tilt is unaffected).
  The loader reports the times in `info["ekf2_quat_reset_times_s"]` and the plots mark them.
  Initialising from the EKF2 attitude at t = 0 predates the reset, so drift/filter comparisons must start
  after the reset or report tilt and yaw separately.
* **Time**: all `timestamp`s are us since boot (uint64) -> int64 -> seconds minus `t0_us`.
  `vehicle_gps_position.time_utc_usec` provides UTC and is used only to date the IGRF model (see Magnetometer).

## 5. Removals, gaps, odd things

* Removed rows per stream are in `info["removed"]` (non-finite, GPS `fix_type < 3`, duplicate/backward
  timestamps, outside the flight window). On all three logs: **0 non-finite, 0 duplicate, 0 backward
  timestamps, 0 `fix_type < 3` rows**; the only removals are the 141-193 IMU samples (and a few slow-stream
  samples) outside the window (armed window +-2 s intersected with the span common to all streams).
* Flight windows (boot time): hover 3250.30-3530.31 s, aggressive 337.07-506.43 s, other 60.81-735.29 s.
  The window is limited by the log ends (the logs start at arming), so all three logs start
  essentially at arming and the 2 s pre-arm margin is not available; the window starts at the first
  sample common to all streams (< 0.8 s after arming).
* Gaps > 5x median interval: hover has one IMU gap 130.921-131.012 s (0.09 s, pyulog logger dropout,
  70 ms + 13 ms); none in the other two logs.
* `accelerometer_clipping` is 0 on all logs. `gyro_clipping` is 0 on the v1.17.0 logs; on the v1.14.0 log it holds
  arbitrary uint8 values on 1606 samples although max |gyro| is 199 deg/s (way below saturation):
  treat the v1.14 field as meaningless.
* `accelerometer_timestamp_relative` = 0 in all logs: accel and gyro share the sample time stamp.
* Multi-instance: `vehicle_imu` 1-3 instances, `sensor_mag` 1-2, `estimator_status` 1-3 logged; the loader
  uses the lowest instance (0). `sensor_combined`, `vehicle_magnetometer`, `vehicle_air_data`,
  `vehicle_gps_position` have a single instance.
* `info` also records SHA-256 of each file, PX4 git hash/release, parameters of interest
  (`EKF2_*`, `CAL_MAG0_ROT`, `SENS_BOARD_ROT`, `IMU_INTEG_RATE`, `SENS_BARO_QNH`, `SYS_AUTOSTART`)
  and the logger dropouts.

## 6. Resolved questions (Group 2 ★ review)

The questions raised before the review are answered below. The binding statements are in
`CONVENTIONS.md` section 6 (finalised); this list records the reasoning and the evidence in sections 2-5.

1. **Quaternion**: `vehicle_attitude.q` is scalar-first Hamilton FRD -> NED and matches CONVENTIONS section 2
   (gyro-integration check in section 4). No conjugation and no FRD<->FLU / NED<->ENU constant anywhere.
2. **Magnetometer units and frame**: PX4 logs gauss, the repository uses tesla (x1e-4). The frame is body FRD
   (PX4 applies the board rotation and calibration), confirmed by the near-constant NED field, not by source reading.
3. **Mag model and rate**: the magnetometer is fused as a **tilt-compensated heading only**, with heading
   standard deviation sigma_psi of about 3-5 deg (recorded as the `R_mag` note and in `assumptions.md` A-11),
   at the logged rate of about 2 Hz. Rationale: the measured heading disagreement with IGRF is 6.6 / 1.2 / 3.0 deg
   and the field-vector angle error is 6.0 / 2.7 / 1.6 deg (section 4), so a field-vector or |B|-based R
   (|B| scatter 0.6-1.1 uT) would not represent the heading error, and the hover log has an uncorrected
   hard-iron bias that cannot be observed (yaw range 8.4 deg). Using heading only keeps the tilt out of the mag update.
4. **Baro**: fused as a **height change** relative to the mean over the first 1 s of the window (vehicle on
   the ground), `z_D,baro = -(alt_baro - alt_baro_ref)`, with sigma of about 1.5 m (`assumptions.md` A-10).
   Rationale: the baro-EKF2 height std is 1.0 / 3.2 / 1.3 m, the datum is ISA/QNH and the offset drifts.
5. **GPS scaling and height**: loader normalises to rad / m (new format deg and m, old format 1e-7 deg and mm).
   Height is WGS-84 ellipsoid (CONVENTIONS section 4); EKF2 `ref_alt` is MSL. Never mix the datums.
6. **NED reference**: our origin is the first valid GPS fix in the loader window. EKF2's origin is the first fix
   after boot, before logging starts, so EKF2 position at t = 0 is non-zero. Comparisons translate EKF2 by the
   constant origin offset (CONVENTIONS section 6, "Local frame"); no subtraction of first samples.
7. **Gravity**: per-log normal gravity 9.7785 / 9.7859 / 9.8119 m/s^2 instead of `G0`; recorded in
   `assumptions.md` A-07.
8. **EKF2 reference is not truth**: it contains one pure-yaw quaternion reset per log (-4.92 / -0.98 / -3.53 deg)
   and its own bias estimates (`estimator_sensor_bias` is not logged). Drift and filter comparisons must start
   after the EKF2 yaw reset or report tilt and yaw errors separately (`assumptions.md` A-08).
9. **Time and GPS latency**: `tau_gps` = 0.11 s (`EKF2_GPS_DELAY`); the independent estimates 0.10-0.18 s
   confirm the order of magnitude. Filter results are reported with a sensitivity sweep over
   tau_gps in {0, 0.11, 0.2} s (`assumptions.md` A-09).
10. **CONVENTIONS.md section 6**: finalised (units, frames, datums, local frame, gravity, EKF2 caveats). Nothing
    further is flagged from Group 2 documentation.
