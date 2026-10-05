"""Tests for ``sensor_fusion.loader`` and the ``interp_to`` helper.

Source of truth: ``docs/specs/sensor_fusion_data.md`` sections 2-3 and ``CONVENTIONS.md``.
Facts about the three pinned logs (rates, sites) come from ``sensor_fusion/data/NOTES.md``.

Independent oracles used (never the loader itself):
* ``pyulog`` read directly from the .ulg (raw values, units, timestamps, arming state, UTC date);
* ``ppigrf`` (IGRF-13/14 field model) for the magnetic inclination / declination at the log's
  location and date;
* ``scipy.spatial.transform`` (Rotation, Slerp) for all rotation arithmetic;
* WGS-84 closed-form meridian / prime-vertical radii for GPS lat/lon -> local metres;
* gravity (specific force of a resting/slow vehicle is ``-g`` along NED down).

Tests that need the pinned ``.ulg`` files skip if the files are absent (no network in tests).
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import functools
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from sensor_fusion import loader as ldr
from sensor_fusion.loader import interp_to

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

HOVER = "f4e81624-7ccc-4859-bb45-d02fa0246b77"
AGGR = "7cd0a812-d6d6-4c67-9da8-1f3a9cc460b0"
OTHER = "78808b7c-d9e9-4350-b311-3bb510f29a7b"
LOG_IDS = [HOVER, AGGR, OTHER]

# From NOTES.md section 2 (measured median 1/dt in the flight window).
IMU_RATE_HZ = {HOVER: 221.7, AGGR: 221.6, OTHER: 199.4}
# Approximate flight sites (NOTES.md section 2): (lat deg, lon deg).
SITE_DEG = {HOVER: (27.51, 92.11), AGGR: (19.13, 72.92), OTHER: (51.77, -1.38)}

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def _path(log_id):
    p = DATA_DIR / f"{log_id}.ulg"
    if not p.exists():
        pytest.skip(f"{p.name} not present (run python -m sensor_fusion.download_logs)")
    return p


@functools.lru_cache(maxsize=None)
def _load(log_id, crop=True):
    return ldr.load_log(_path(log_id), crop_to_flight=crop)


@functools.lru_cache(maxsize=None)
def _ulog(log_id):
    from pyulog import ULog
    return ULog(str(_path(log_id)))


def _raw(log_id, topic):
    """Raw field dict of the lowest instance of ``topic``, read straight with pyulog."""
    ds = sorted((d for d in _ulog(log_id).data_list if d.name == topic), key=lambda d: d.multi_id)
    assert ds, f"topic {topic} missing"
    return ds[0].data


def _raw_vec(d, base):
    return np.column_stack([np.asarray(d[f"{base}[{i}]"], dtype=float) for i in range(3)])


def _armed_intervals_boot_s(log_id):
    """Armed intervals (boot seconds) from vehicle_status.arming_state == 2 (2 = ARMED)."""
    d = _raw(log_id, "vehicle_status")
    t = d["timestamp"].astype(np.int64) * 1e-6
    armed = d["arming_state"] == 2
    edges = np.diff(armed.astype(int))
    starts = list(t[1:][edges == 1])
    ends = list(t[1:][edges == -1])
    if armed[0]:
        starts.insert(0, t[0])
    if armed[-1]:
        ends.append(t[-1])
    return list(zip(starts, ends))


def _stream_times(log):
    return {"imu": log.imu.t, "mag": log.mag.t, "baro": log.baro.t, "gps": log.gps.t,
            "ekf2_att": log.ekf2.t_att, "ekf2_pos": log.ekf2.t_pos}


def _rot_slerp(t_knots, q_wxyz):
    return Slerp(t_knots, R.from_quat(q_wxyz[:, [1, 2, 3, 0]]))


def _raw_idx(log, t_loader, raw_t_us):
    """Indices into the raw array whose timestamps equal the loader's (t0_us offset)."""
    t_us = np.rint(np.asarray(t_loader) * 1e6).astype(np.int64) + log.t0_us
    idx = np.searchsorted(raw_t_us, t_us)
    idx = np.clip(idx, 0, len(raw_t_us) - 1)
    assert np.array_equal(raw_t_us[idx], t_us), "loader timestamps are not raw log timestamps"
    return idx


def _stamp_us(d):
    """Validity time of each raw row: ``timestamp_sample`` where the topic carries it (spec section 6:
    mag, baro), else ``timestamp``."""
    ts = d["timestamp"].astype(np.int64)
    if "timestamp_sample" in d:
        tss = d["timestamp_sample"].astype(np.int64)
        return np.where(tss > 0, tss, ts)
    return ts


pytestmark_logs = pytest.mark.parametrize("log_id", LOG_IDS)


# ============================================================================ structure / time
@pytestmark_logs
def test_flightlog_fields_and_ids(log_id):
    log = _load(log_id)
    assert log.log_id == log_id
    assert Path(log.path).name == f"{log_id}.ulg"
    assert isinstance(log.t0_us, (int, np.integer)) and log.t0_us > 0
    for name in ("imu", "mag", "baro", "gps", "ekf2", "info"):
        assert hasattr(log, name)
    for key in ("removed", "gaps"):
        assert key in log.info
    for name in ("imu", "mag", "baro", "gps"):
        assert "t" in {f.name for f in dataclasses.fields(getattr(log, name))}
    assert {"t_att", "q", "t_pos", "pos_ned", "vel_ned", "ref_lat", "ref_lon", "ref_alt"} <= \
        {f.name for f in dataclasses.fields(log.ekf2)}


@pytestmark_logs
def test_loads_by_bare_uuid(log_id):
    _path(log_id)
    a = ldr.load_log(log_id)
    b = _load(log_id)
    assert a.t0_us == b.t0_us and np.array_equal(a.imu.t, b.imu.t)


@pytestmark_logs
def test_time_strictly_increasing_float64_nonnegative(log_id):
    log = _load(log_id)
    times = _stream_times(log)
    times["gps_t_sample"] = log.gps.t_sample
    for name, t in times.items():
        assert t.dtype == np.float64, name
        assert t.ndim == 1 and len(t) > 10, name
        assert t[0] >= 0.0, f"{name} starts before 0: {t[0]}"
        if name != "gps_t_sample":
            assert np.all(np.diff(t) > 0), f"{name} not strictly increasing"
    # the common window starts at 0 (t0 = first sample of the window)
    assert min(t[0] for n, t in times.items() if n != "gps_t_sample") == pytest.approx(0.0, abs=1e-9)


@pytestmark_logs
def test_gps_sample_time_not_after_receive_time(log_id):
    g = _load(log_id).gps
    assert g.t_sample.shape == g.t.shape
    assert np.all(g.t_sample <= g.t + 1e-9)
    assert np.all(g.t - g.t_sample < 1.0)  # latency, if any, is far below a second


@pytestmark_logs
def test_all_arrays_finite_and_shapes(log_id):
    log = _load(log_id)
    for cont in (log.imu, log.mag, log.baro, log.gps, log.ekf2):
        for f in dataclasses.fields(cont):
            v = getattr(cont, f.name)
            if isinstance(v, np.ndarray) and v.dtype.kind == "f":
                assert np.all(np.isfinite(v)), f"{type(cont).__name__}.{f.name} has NaN/inf"
    for scalar in (log.imu.rate_hz, log.ekf2.ref_lat, log.ekf2.ref_lon, log.ekf2.ref_alt):
        assert np.isfinite(scalar)
    n = len(log.imu.t)
    assert log.imu.gyro.shape == (n, 3) and log.imu.accel.shape == (n, 3) and log.imu.dt.shape == (n,)
    n = len(log.mag.t)
    assert log.mag.field.shape == (n, 3)
    n = len(log.baro.t)
    assert log.baro.pressure.shape == log.baro.alt_baro.shape == log.baro.temperature.shape == (n,)
    n = len(log.gps.t)
    assert log.gps.vel_ned.shape == (n, 3)
    for name in ("lat", "lon", "alt", "eph", "epv", "s_variance_m_s", "fix_type", "n_sat", "t_sample"):
        assert getattr(log.gps, name).shape == (n,), name
    n = len(log.ekf2.t_att)
    assert log.ekf2.q.shape == (n, 4)
    n = len(log.ekf2.t_pos)
    assert log.ekf2.pos_ned.shape == (n, 3) and log.ekf2.vel_ned.shape == (n, 3)


@pytestmark_logs
def test_ekf2_quaternions_unit_and_canonical(log_id):
    q = _load(log_id).ekf2.q
    assert np.max(np.abs(np.linalg.norm(q, axis=1) - 1.0)) < 1e-12
    assert np.all(q[:, 0] >= 0.0)


@pytestmark_logs
def test_ekf2_quaternion_matches_raw_up_to_sign(log_id):
    """Loader q equals raw vehicle_attitude.q (already scalar-first FRD->NED) up to sign/normalisation."""
    log = _load(log_id)
    raw = _raw(log_id, "vehicle_attitude")
    idx = _raw_idx(log, log.ekf2.t_att, raw["timestamp"].astype(np.int64))
    qr = np.column_stack([raw[f"q[{i}]"] for i in range(4)])[idx].astype(float)
    qr /= np.linalg.norm(qr, axis=1, keepdims=True)
    dot = np.abs(np.sum(qr * log.ekf2.q, axis=1))
    assert np.all(dot > 1.0 - 1e-12)


# ============================================================================ IMU
@pytestmark_logs
def test_imu_rate_and_dt_consistency(log_id):
    imu = _load(log_id).imu
    assert imu.rate_hz == pytest.approx(IMU_RATE_HZ[log_id], rel=0.05)
    assert imu.rate_hz >= 190.0  # spec: prefer >= 200 Hz; V5 log is 199.4
    d = np.diff(imu.t)
    assert np.allclose(imu.dt[1:], d, rtol=0, atol=1e-12)
    assert imu.dt[0] == pytest.approx(np.median(d), abs=1e-12)  # spec: dt[0] = median
    assert imu.rate_hz == pytest.approx(1.0 / np.median(d), rel=1e-9)
    assert np.all(imu.dt > 0)


@pytestmark_logs
def test_imu_values_match_raw_log(log_id):
    """Units/axes: gyro rad/s and accel m/s^2 straight from sensor_combined, no scaling or axis change."""
    log = _load(log_id)
    raw = _raw(log_id, "sensor_combined")
    idx = _raw_idx(log, log.imu.t, raw["timestamp"].astype(np.int64))
    np.testing.assert_allclose(log.imu.gyro, _raw_vec(raw, "gyro_rad")[idx], rtol=0, atol=1e-12)
    np.testing.assert_allclose(log.imu.accel, _raw_vec(raw, "accelerometer_m_s2")[idx], rtol=0, atol=1e-12)


@pytestmark_logs
def test_gyro_integration_matches_ekf2_attitude_change(log_id):
    """Independent frame/sign check: integrating the gyro over 1 s starting from EKF2's attitude
    reproduces EKF2's attitude 1 s later (median error < 1 deg); a sign-flipped gyro does not.

    Integration: R_{k+1} = R_k exp(w_k dt_k) (body-frame right multiplication, Hamilton FRD->NED),
    done with scipy Rotation, not common.quaternion.  1 deg tolerance: EKF2 estimates gyro bias and
    resets yaw once per log (1-5 deg) so this is a median, not a max, statement; NOTES.md measured
    0.17-0.27 deg.
    """
    log = _load(log_id)
    imu, ekf = log.imu, log.ekf2
    sl = _rot_slerp(ekf.t_att, ekf.q)
    starts = np.linspace(ekf.t_att[0] + 0.5, ekf.t_att[-1] - 1.5, 60)

    def window_errors(sign):
        errs = []
        for ts in starts:
            i0 = int(np.searchsorted(imu.t, ts))
            i1 = int(np.searchsorted(imu.t, imu.t[i0] + 1.0))
            if i1 >= len(imu.t) or imu.t[i1] > ekf.t_att[-1]:
                continue
            k = np.arange(i0 + 1, i1 + 1)
            steps = R.from_rotvec(sign * imu.gyro[k] * imu.dt[k, None])
            Rk = sl([imu.t[i0]])[0]
            for s in steps:
                Rk = Rk * s
            Rend = sl([imu.t[i1]])[0]
            errs.append(np.degrees((Rk.inv() * Rend).magnitude()))
        return np.array(errs)

    good, flipped = window_errors(+1.0), window_errors(-1.0)
    assert len(good) >= 40
    assert np.median(good) < 1.0, f"median 1 s gyro-vs-EKF2 attitude error {np.median(good):.2f} deg"
    assert np.median(flipped) > 3.0 * np.median(good)


@pytestmark_logs
def test_accel_is_specific_force_down_reads_minus_g(log_id):
    """R(q_ekf2) f_b has median z ~ -9.8 m/s^2 while armed (specific force of a vehicle that is on
    average not accelerating: -g along NED down), and ~0 horizontally.  +/-0.3 as in the task text;
    NOTES.md measured -9.77 ... -9.84."""
    log = _load(log_id)
    imu, ekf = log.imu, log.ekf2
    boot = imu.t + log.t0_us * 1e-6
    armed = np.zeros(len(boot), bool)
    for a, b in _armed_intervals_boot_s(log_id):
        armed |= (boot >= a) & (boot <= b)
    assert armed.sum() > 0.8 * len(boot)
    m = armed & (imu.t >= ekf.t_att[0]) & (imu.t <= ekf.t_att[-1])
    f_ned = _rot_slerp(ekf.t_att, ekf.q)(imu.t[m]).apply(imu.accel[m])
    med = np.median(f_ned, axis=0)
    assert med[2] == pytest.approx(-9.8, abs=0.3)
    assert abs(med[0]) < 0.5 and abs(med[1]) < 0.5


@pytestmark_logs
def test_local_gravity_matches_measured_specific_force_at_start(log_id):
    """normal_gravity at the NED origin (first GPS fix: lat, lon, ellipsoidal height) equals the measured
    -down component of R(q_ekf2) f_b while the vehicle is still on the ground, within 0.03 m/s^2.

    Window: from the start of the loader window (within ~0.03 s of arming on all pinned logs) until the
    vehicle departs the ground -- first of |EKF2 height change| > 0.3 m or |gyro| > 0.2 rad/s -- capped
    at 3 s ('first armed seconds').  On the aggressive log that is only ~0.6 s (immediate take-off), where
    thrust and EKF2 attitude error would otherwise bias the median.  Median -> robust to vibration.
    """
    from common.frames import normal_gravity
    log = _load(log_id)
    imu, ekf, gps = log.imu, log.ekf2, log.gps
    t_lo = max(ekf.t_att[0], imu.t[0])
    z = ekf.pos_ned[:, 2]
    leave = [ekf.t_pos[np.argmax(np.abs(z - z[0]) > 0.3)],
             imu.t[np.argmax(np.linalg.norm(imu.gyro, axis=1) > 0.2)]]
    t_hi = min(t_lo + 3.0, *leave)
    m = (imu.t >= t_lo) & (imu.t <= t_hi)
    assert m.sum() > 100, f"only {m.sum()} static IMU samples ({t_hi - t_lo:.2f} s)"
    f_ned = _rot_slerp(ekf.t_att, ekf.q)(imu.t[m]).apply(imu.accel[m])
    g_meas = -np.median(f_ned[:, 2])
    g_local = float(normal_gravity(gps.lat[0], gps.alt[0]))
    assert abs(g_meas - g_local) < 0.03, (
        f"measured {g_meas:.4f} vs normal_gravity {g_local:.4f} m/s^2 (diff {g_meas - g_local:+.4f}, "
        f"window {t_lo:.2f}-{t_hi:.2f} s, {m.sum()} samples)")


# ============================================================================ mag
@pytestmark_logs
def test_mag_units_tesla_and_match_raw_gauss(log_id):
    log = _load(log_id)
    raw = _raw(log_id, "vehicle_magnetometer")
    idx = _raw_idx(log, log.mag.t, _stamp_us(raw))
    ga = _raw_vec(raw, "magnetometer_ga")[idx]
    np.testing.assert_allclose(log.mag.field, ga * 1e-4, rtol=1e-12, atol=0)  # 1 G = 1e-4 T
    nrm = np.linalg.norm(log.mag.field, axis=1)
    assert nrm.min() > 20e-6 and nrm.max() < 70e-6, (nrm.min(), nrm.max())
    assert np.mean((nrm > 30e-6) & (nrm < 60e-6)) > 0.95  # mostly 30-60 uT (Earth's field)


def _igrf_ned(log_id, log):
    import ppigrf
    utc = _raw(log_id, "vehicle_gps_position")["time_utc_usec"]
    utc = utc[utc > 0]
    date = dt.datetime(1970, 1, 1) + dt.timedelta(microseconds=int(utc[len(utc) // 2]))
    lat, lon, h = (float(np.mean(x)) for x in (np.degrees(log.gps.lat), np.degrees(log.gps.lon), log.gps.alt))
    be, bn, bu = (float(np.ravel(x)[0]) for x in ppigrf.igrf(lon, lat, h * 1e-3, date))  # nT
    return np.array([bn, be, -bu]) * 1e-9, date


@pytestmark_logs
def test_mag_rotated_by_ekf2_matches_igrf_inclination_and_declination(log_id):
    """B_ned = R(q_ekf2) B_body must be a nearly constant vector whose inclination matches IGRF at
    the log's location/date within 10 deg (spec).  It also confirms body FRD axes + signs: a wrong
    axis permutation/sign would not give a constant NED vector with positive-down inclination.
    Declination (extra check, 15 deg): EKF2 yaw is itself mag-aided, so heading must agree with IGRF
    to within local magnetic disturbance."""
    log = _load(log_id)
    ekf = log.ekf2
    m = (log.mag.t >= ekf.t_att[0]) & (log.mag.t <= ekf.t_att[-1])
    b_ned = _rot_slerp(ekf.t_att, ekf.q)(log.mag.t[m]).apply(log.mag.field[m])
    mean = b_ned.mean(axis=0)
    incl = np.degrees(np.arctan2(mean[2], np.hypot(mean[0], mean[1])))
    decl = np.degrees(np.arctan2(mean[1], mean[0]))

    ref, _ = _igrf_ned(log_id, log)
    incl_ref = np.degrees(np.arctan2(ref[2], np.hypot(ref[0], ref[1])))
    decl_ref = np.degrees(np.arctan2(ref[1], ref[0]))
    assert abs(incl - incl_ref) < 10.0, f"inclination {incl:.1f} vs IGRF {incl_ref:.1f} deg"
    ddecl = (decl - decl_ref + 180.0) % 360.0 - 180.0
    assert abs(ddecl) < 15.0, f"declination {decl:.1f} vs IGRF {decl_ref:.1f} deg"
    # NED field is (nearly) constant: scatter much smaller than the field itself
    assert np.all(b_ned.std(axis=0) < 0.1 * np.linalg.norm(mean))


# ============================================================================ baro
@pytestmark_logs
def test_baro_matches_raw_and_sign_is_up(log_id):
    log = _load(log_id)
    raw = _raw(log_id, "vehicle_air_data")
    idx = _raw_idx(log, log.baro.t, _stamp_us(raw))
    np.testing.assert_allclose(log.baro.pressure, raw["baro_pressure_pa"][idx], rtol=1e-12)
    np.testing.assert_allclose(log.baro.alt_baro, raw["baro_alt_meter"][idx], rtol=1e-12)
    assert 50e3 < log.baro.pressure.min() and log.baro.pressure.max() < 110e3  # Pa, not hPa/kPa
    assert -40 < log.baro.temperature.min() and log.baro.temperature.max() < 70  # degC


@pytestmark_logs
def test_baro_and_mag_t_equal_raw_timestamp_sample(log_id):
    """Spec section 6: mag/baro ``t`` is the raw ``timestamp_sample`` (measurement validity time), in
    seconds relative to t0, not the (later) publication ``timestamp``."""
    log = _load(log_id)
    for topic, t_loader in (("vehicle_magnetometer", log.mag.t), ("vehicle_air_data", log.baro.t)):
        raw = _raw(log_id, topic)
        assert "timestamp_sample" in raw, f"{topic} has no timestamp_sample in this log"
        ts = raw["timestamp_sample"].astype(np.int64)
        assert np.all(ts > 0)
        idx = _raw_idx(log, t_loader, ts)  # exact microsecond equality asserted inside
        np.testing.assert_allclose(t_loader, (ts[idx] - log.t0_us) * 1e-6, rtol=0, atol=1e-9)
        # the sample time precedes publication by the sensor latency (>= 0, strictly > 0 on average)
        lag = raw["timestamp"].astype(np.int64)[idx] - ts[idx]
        assert np.all(lag >= 0) and np.median(lag) > 0, topic


@pytestmark_logs
def test_info_ekf2_hgt_ref_matches_raw_parameter(log_id):
    raw_val = _ulog(log_id).initial_parameters["EKF2_HGT_REF"]
    assert "ekf2_hgt_ref" in _load(log_id).info
    assert _load(log_id).info["ekf2_hgt_ref"] == raw_val
    assert raw_val in (0, 1, 2, 3)


@pytestmark_logs
def test_info_gps_utc_start_matches_raw(log_id):
    """info['gps_utc_start']: first non-zero ``time_utc_usec`` among the GPS rows kept in the window, as a
    UTC datetime (used to date the IGRF model)."""
    log = _load(log_id)
    raw = _raw(log_id, "vehicle_gps_position")
    utc = raw["time_utc_usec"].astype(np.int64)[_raw_idx(log, log.gps.t, raw["timestamp"].astype(np.int64))]
    utc = utc[utc > 0]
    assert len(utc) > 0
    got = log.info["gps_utc_start"]
    assert got.tzinfo is not None and got.utcoffset() == dt.timedelta(0)
    exp = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(microseconds=int(utc[0]))
    assert got == exp
    assert dt.datetime(2019, 1, 1, tzinfo=dt.timezone.utc) < got < dt.datetime(2027, 1, 1, tzinfo=dt.timezone.utc)


@pytestmark_logs
def test_baro_altitude_changes_vs_ekf2_height_or_gps(log_id):
    """Baro height (m up) sanity, per the spec section 6 'Baro test criterion'.

    * ``info['ekf2_hgt_ref'] == 0`` (EKF2 height is baro-driven): alt_baro changes ~ -(change in EKF2 NED
      z), 95th percentile of the difference < 5 m over the whole flight (individual baro samples spike
      with prop-wash / dynamic pressure, 5.2 m max on the v1.14 log), and for a real climb (> 10 m) slope
      in (0.8, 1.2) and r > 0.9 (sign test).
    * otherwise (EKF2 height follows GPS and inherits GPS vertical drift, ~9 m on the aggressive log):
      (a) baro returns to take-off level within 5 m by the end of the flight (mean over the last 1 s vs
      the first 1 s of the window, both on the ground); (b) baro increments over non-overlapping 10 s
      windows correlate with GPS (ellipsoidal) height increments, r > 0.3 over the whole flight.
    """
    log = _load(log_id)
    b, g, ekf = log.baro, log.gps, log.ekf2
    if log.info["ekf2_hgt_ref"] == 0:
        m = (b.t >= ekf.t_pos[0]) & (b.t <= ekf.t_pos[-1])
        up_ekf = -interp_to(b.t[m], ekf.t_pos, ekf.pos_ned[:, 2])
        up_baro = b.alt_baro[m]
        d_ekf = up_ekf - up_ekf[0]
        d_baro = up_baro - up_baro[0]
        assert np.percentile(np.abs(d_baro - d_ekf), 95) < 5.0
        if np.ptp(d_ekf) > 10.0:
            slope = np.polyfit(d_ekf, d_baro, 1)[0]
            assert 0.8 < slope < 1.2, f"baro-vs-EKF2 height slope {slope:.2f} (negative => sign error)"
            assert np.corrcoef(d_ekf, d_baro)[0, 1] > 0.9
        return
    start = b.alt_baro[b.t < b.t[0] + 1.0].mean()
    end = b.alt_baro[b.t > b.t[-1] - 1.0].mean()
    assert abs(end - start) < 5.0, f"baro take-off/landing level differ by {end - start:.2f} m"
    grid = np.arange(max(b.t[0], g.t[0]), min(b.t[-1], g.t[-1]), 10.0)
    assert len(grid) >= 10
    d_baro = np.diff(np.interp(grid, b.t, b.alt_baro))
    d_gps = np.diff(np.interp(grid, g.t, g.alt))
    r = np.corrcoef(d_baro, d_gps)[0, 1]
    assert r > 0.3, f"10 s baro-vs-GPS height increment correlation r={r:.3f} over {len(d_baro)} windows"


# ============================================================================ GPS
def _to_local_ned(lat, lon, alt, lat0, lon0, alt0):
    """Small-offset WGS-84 geodetic -> local NED (closed-form radii of curvature), metres."""
    s2 = np.sin(lat0) ** 2
    n_rad = WGS84_A / np.sqrt(1 - WGS84_E2 * s2)
    m_rad = WGS84_A * (1 - WGS84_E2) / (1 - WGS84_E2 * s2) ** 1.5
    return np.column_stack([(lat - lat0) * (m_rad + alt0),
                            (lon - lon0) * (n_rad + alt0) * np.cos(lat0),
                            -(alt - alt0)])


@pytestmark_logs
def test_gps_radians_site_fix_and_matches_raw(log_id):
    log = _load(log_id)
    g = log.gps
    assert np.all(np.abs(g.lat) <= np.pi / 2) and np.all(np.abs(g.lon) <= np.pi)
    lat_deg, lon_deg = SITE_DEG[log_id]
    assert np.degrees(np.median(g.lat)) == pytest.approx(lat_deg, abs=0.2)
    assert np.degrees(np.median(g.lon)) == pytest.approx(lon_deg, abs=0.2)
    assert np.ptp(g.lat) < np.radians(0.05) and np.ptp(g.lon) < np.radians(0.05)  # flight stays local
    assert np.all(g.fix_type >= 3)
    assert np.all(g.eph > 0) and np.all(g.epv > 0)

    raw = _raw(log_id, "vehicle_gps_position")
    idx = _raw_idx(log, g.t, raw["timestamp"].astype(np.int64))
    if "latitude_deg" in raw:
        lat_r, lon_r = raw["latitude_deg"][idx], raw["longitude_deg"][idx]
        ell, msl = raw["altitude_ellipsoid_m"][idx], raw["altitude_msl_m"][idx]
    else:  # old format: integer degrees * 1e-7, altitudes in integer mm
        lat_r, lon_r = raw["lat"][idx] * 1e-7, raw["lon"][idx] * 1e-7
        ell, msl = raw["alt_ellipsoid"][idx] * 1e-3, raw["alt"][idx] * 1e-3
    np.testing.assert_allclose(g.lat, np.radians(lat_r), rtol=0, atol=1e-12)
    np.testing.assert_allclose(g.lon, np.radians(lon_r), rtol=0, atol=1e-12)
    np.testing.assert_allclose(g.alt, ell, rtol=0, atol=1e-6)
    assert np.max(np.abs(g.alt - msl)) > 1.0  # ellipsoid, not MSL (geoid separation is metres..tens of m)
    vel = np.column_stack([raw["vel_n_m_s"], raw["vel_e_m_s"], raw["vel_d_m_s"]])[idx]
    np.testing.assert_allclose(g.vel_ned, vel, rtol=0, atol=1e-6)


@pytestmark_logs
def test_gps_alt_msl_matches_raw_in_metres(log_id):
    """alt_msl equals the raw MSL height in metres: ``altitude_msl_m`` [m] (new format) or ``alt`` [int mm]
    (<= v1.14) -- the mm->m scaling is the point of this test."""
    log = _load(log_id)
    g = log.gps
    raw = _raw(log_id, "vehicle_gps_position")
    idx = _raw_idx(log, g.t, raw["timestamp"].astype(np.int64))
    msl = raw["altitude_msl_m"][idx] if "altitude_msl_m" in raw else raw["alt"][idx] * 1e-3
    assert g.alt_msl.shape == g.t.shape
    np.testing.assert_allclose(g.alt_msl, msl, rtol=0, atol=1e-6)
    assert np.all(np.isfinite(g.alt_msl))
    # metres, not mm: within a few km of sea level worldwide, and close to EKF2 ref_alt (MSL)
    assert np.all(np.abs(g.alt_msl) < 9000.0)
    assert abs(np.median(g.alt_msl) - log.ekf2.ref_alt) < 50.0


@pytestmark_logs
def test_geoid_separation_plausible_and_constant(log_id):
    g = _load(log_id).gps
    n = g.alt - g.alt_msl
    assert np.all(np.abs(n) < 100.0), (n.min(), n.max())
    assert np.std(n) < 1.0, f"geoid separation varies over the flight: std {np.std(n):.3f} m"


@pytestmark_logs
def test_ekf2_origin_height_offset_gps_msl_minus_ekf2(log_id):
    """GPS MSL height minus EKF2 MSL height (``ref_alt - pos_ned_z``) is a small constant: |median| < 1 m.
    Confirms ref_alt is MSL and z is NED-down about it, so that the CONVENTIONS section 6 vertical origin
    offset ``ref_alt - alt_msl(origin)`` is correct."""
    log = _load(log_id)
    g, ekf = log.gps, log.ekf2
    m = (g.t >= ekf.t_pos[0]) & (g.t <= ekf.t_pos[-1])
    ekf_msl = ekf.ref_alt - interp_to(g.t[m], ekf.t_pos, ekf.pos_ned[:, 2])
    off = g.alt_msl[m] - ekf_msl
    assert abs(np.median(off)) < 1.0, f"median(GPS MSL - EKF2 MSL) = {np.median(off):.3f} m"


@pytestmark_logs
def test_gps_velocity_consistent_with_differentiated_position(log_id):
    """GPS vel_ned (NED m/s, down positive) vs finite-differenced geodetic position over ~2 s spans.
    A span of >= 2 s keeps the position-noise contribution (~0.3 m / 2 s) well below the 0.5 m/s bound;
    the GPS velocity is averaged over the same span so that the comparison is exact for any motion."""
    g = _load(log_id).gps
    pos = _to_local_ned(g.lat, g.lon, g.alt, g.lat[0], g.lon[0], g.alt[0])
    errs = []
    i = 0
    n = len(g.t)
    while i < n:
        j = int(np.searchsorted(g.t, g.t[i] + 2.0))
        if j >= n:
            break
        v_pos = (pos[j] - pos[i]) / (g.t[j] - g.t[i])
        seg = slice(i, j + 1)
        v_gps = np.trapezoid(g.vel_ned[seg], g.t[seg], axis=0) / (g.t[j] - g.t[i])
        errs.append(np.linalg.norm(v_pos - v_gps))
        i = j
    errs = np.array(errs)
    assert len(errs) > 30
    assert np.median(errs) < 0.5, f"median |v_gps - dp/dt| = {np.median(errs):.3f} m/s"


@pytestmark_logs
def test_ekf2_position_vs_gps_derived_ned_bounded(log_id):
    """EKF2 pos_ned (about its own ref lat/lon) vs GPS geodetic -> NED about the same origin.
    Horizontal: EKF2 is GPS-aided so median < 0.5 m, 99th pct < 3 m.  Vertical: EKF2 reference
    height is MSL while GPS height is ellipsoidal, so a constant offset is removed first; the
    remaining variation (baro-driven EKF2 height vs GPS) must stay < 3 m (95th pct)."""
    log = _load(log_id)
    g, ekf = log.gps, log.ekf2
    m = (g.t >= ekf.t_pos[0]) & (g.t <= ekf.t_pos[-1])
    gps_ned = _to_local_ned(g.lat[m], g.lon[m], g.alt[m], ekf.ref_lat, ekf.ref_lon, 0.0)
    e = interp_to(g.t[m], ekf.t_pos, ekf.pos_ned) - gps_ned
    horiz = np.linalg.norm(e[:, :2], axis=1)
    assert np.median(horiz) < 0.5, f"median horizontal diff {np.median(horiz):.2f} m"
    assert np.percentile(horiz, 99) < 3.0, f"p99 horizontal diff {np.percentile(horiz, 99):.2f} m"
    ez = e[:, 2] - np.median(e[:, 2])
    assert np.percentile(np.abs(ez), 95) < 3.0, f"p95 vertical diff {np.percentile(np.abs(ez), 95):.2f} m"


@pytestmark_logs
def test_ekf2_position_velocity_match_raw(log_id):
    log = _load(log_id)
    raw = _raw(log_id, "vehicle_local_position")
    idx = _raw_idx(log, log.ekf2.t_pos, raw["timestamp"].astype(np.int64))
    np.testing.assert_allclose(log.ekf2.pos_ned,
                               np.column_stack([raw["x"], raw["y"], raw["z"]])[idx], rtol=0, atol=1e-9)
    np.testing.assert_allclose(log.ekf2.vel_ned,
                               np.column_stack([raw["vx"], raw["vy"], raw["vz"]])[idx], rtol=0, atol=1e-9)
    assert log.ekf2.ref_lat == pytest.approx(np.radians(raw["ref_lat"][idx][0]), abs=1e-12)
    assert log.ekf2.ref_lon == pytest.approx(np.radians(raw["ref_lon"][idx][0]), abs=1e-12)


# ============================================================================ cleaning, gaps, crop
@pytestmark_logs
def test_gaps_reported_exactly(log_id):
    """Every inter-sample gap > 5x median is in info['gaps'] as (stream, t_start, t_end), and vice versa."""
    log = _load(log_id)
    listed = {(s, round(a, 6), round(b, 6)) for s, a, b in log.info["gaps"]}
    found = set()
    for name, t in _stream_times(log).items():
        d = np.diff(t)
        for i in np.flatnonzero(d > 5.0 * np.median(d)):
            found.add((name, round(float(t[i]), 6), round(float(t[i + 1]), 6)))
    assert listed == found
    # spec: no gaps > 5x median except those listed -- restated per stream
    for name, t in _stream_times(log).items():
        d = np.diff(t)
        bad = d > 5.0 * np.median(d)
        assert bad.sum() == sum(1 for s, *_ in log.info["gaps"] if s == name)


def test_hover_log_has_known_imu_dropout():
    """NOTES.md: hover log has one logger dropout of 0.091 s in the IMU stream.  Compared as a duration
    and as raw boot-time stamps (independent of where t0 falls, which depends on mag/baro stamping)."""
    log = _load(HOVER)
    imu_gaps = [(a, b) for s, a, b in log.info["gaps"] if s == "imu"]
    assert len(imu_gaps) == 1
    a, b = imu_gaps[0]
    assert b - a == pytest.approx(0.091, abs=0.005)
    # independent confirmation from the raw timestamps, in boot time
    raw_t = _raw(HOVER, "sensor_combined")["timestamp"].astype(np.int64)
    d = np.diff(raw_t)
    k = int(np.argmax(d))
    assert d[k] * 1e-6 == pytest.approx(b - a, abs=1e-5)
    t0 = log.t0_us
    assert (raw_t[k] - t0) * 1e-6 == pytest.approx(a, abs=1e-6)
    assert (raw_t[k + 1] - t0) * 1e-6 == pytest.approx(b, abs=1e-6)


@pytestmark_logs
def test_removed_counts_recorded_and_consistent(log_id):
    log = _load(log_id)
    rem = log.info["removed"]
    for name, n_kept in (("imu", len(log.imu.t)), ("mag", len(log.mag.t)), ("baro", len(log.baro.t)),
                         ("gps", len(log.gps.t)), ("ekf2_att", len(log.ekf2.t_att)),
                         ("ekf2_pos", len(log.ekf2.t_pos))):
        assert name in rem, name
        assert isinstance(rem[name], dict) and rem[name]
        assert rem[name].get("n_kept", n_kept) == n_kept
    # NOTES.md: no GPS rows with fix < 3 and no non-finite rows in these logs
    raw_fix = _raw(log_id, "vehicle_gps_position")["fix_type"]
    assert int((raw_fix < 3).sum()) == 0
    assert all(v == 0 for k, v in rem["gps"].items() if "fix" in k)


@pytestmark_logs
def test_crop_window_matches_armed_window(log_id):
    log = _load(log_id)
    uncropped = _load(log_id, False)
    arm = _armed_intervals_boot_s(log_id)
    assert arm
    a0, a1 = min(a for a, _ in arm), max(b for _, b in arm)
    margin = 2.0
    t0 = log.t0_us * 1e-6
    for name, t in _stream_times(log).items():
        boot = t + t0
        assert boot[0] >= a0 - margin - 1e-6, f"{name} starts {a0 - boot[0]:.3f} s before arming"
        assert boot[-1] <= a1 + margin + 1e-6, f"{name} ends {boot[-1] - a1:.3f} s after disarming"
    # reported window (boot seconds) equals the independent armed window within the margin
    info_arm = log.info["arming_windows_boot_s"]
    assert len(info_arm) == len(arm)
    # vehicle_status is event/low-rate logged: the disarm instant is only known to within one status
    # message period, so the end tolerance is the largest status spacing (the loader reports the last
    # armed message, the independent reading above the first disarmed one).
    status_dt = np.max(np.diff(_raw(log_id, "vehicle_status")["timestamp"].astype(np.int64))) * 1e-6
    for (ia, ib), (a, b) in zip(info_arm, arm):
        assert ia == pytest.approx(a, abs=1e-3)
        assert ib == pytest.approx(b, abs=status_dt + 1e-3)
    # not cropping can only keep more data
    assert uncropped.t0_us <= log.t0_us
    assert len(uncropped.imu.t) >= len(log.imu.t)
    assert uncropped.imu.t[-1] + uncropped.t0_us * 1e-6 >= log.imu.t[-1] + t0 - 1e-9


# ============================================================================ interp_to
def test_interp_linear_vector_and_scalar_exact_for_affine_data():
    t = np.array([0.0, 0.3, 1.0, 1.7, 3.0])
    c = np.array([1.0, -2.0, 0.5])
    x = 4.0 + t[:, None] * c  # affine in t -> linear interpolation exact
    tq = np.array([0.0, 0.1, 0.3, 0.65, 1.7, 2.9, 3.0])
    np.testing.assert_allclose(interp_to(tq, t, x), 4.0 + tq[:, None] * c, atol=1e-12)
    np.testing.assert_allclose(interp_to(tq, t, x[:, 0]), 4.0 + tq * c[0], atol=1e-12)
    assert interp_to(tq, t, x).shape == (len(tq), 3)
    assert interp_to(tq, t, x[:, 0]).shape == (len(tq),)


def test_interp_matches_numpy_on_random_data_and_returns_samples_at_knots():
    rng = np.random.default_rng(20240607)
    t = np.cumsum(rng.uniform(0.01, 0.5, 40))
    x = rng.normal(size=(40, 3))
    tq = rng.uniform(t[0], t[-1], 200)
    ref = np.column_stack([np.interp(tq, t, x[:, k]) for k in range(3)])
    np.testing.assert_allclose(interp_to(tq, t, x), ref, atol=1e-12)
    np.testing.assert_allclose(interp_to(t, t, x), x, atol=1e-12)


def test_interp_clamps_outside_range():
    t = np.array([1.0, 2.0, 4.0])
    x = np.array([[1.0, 10.0], [3.0, 20.0], [7.0, 0.0]])
    out = interp_to(np.array([-5.0, 0.99, 4.01, 100.0]), t, x)
    np.testing.assert_allclose(out, [x[0], x[0], x[-1], x[-1]], atol=1e-12)


def test_interp_quaternion_midpoint_is_half_angle_rotation():
    th = 1.2
    q0 = np.array([1.0, 0, 0, 0])
    q1 = R.from_rotvec([0, 0, th]).as_quat()[[3, 0, 1, 2]]
    out = interp_to(np.array([0.5, 0.25]), np.array([0.0, 1.0]), np.stack([q0, q1]), quaternion=True)
    for row, frac in zip(out, (0.5, 0.25)):
        exp = R.from_rotvec([0, 0, frac * th]).as_quat()[[3, 0, 1, 2]]
        np.testing.assert_allclose(row, exp, atol=1e-12)


def test_interp_quaternion_matches_scipy_slerp_unit_norm_and_endpoints():
    rng = np.random.default_rng(7)
    rots = R.random(12, random_state=rng)
    t = np.sort(rng.uniform(0, 10, 12))
    q = rots.as_quat()[:, [3, 0, 1, 2]]
    tq = rng.uniform(t[0], t[-1], 300)
    out = interp_to(tq, t, q, quaternion=True)
    exp = Slerp(t, rots)(tq).as_quat()[:, [3, 0, 1, 2]]
    assert np.max(np.abs(np.linalg.norm(out, axis=1) - 1.0)) < 1e-12
    # same rotation (sign-insensitive)
    assert np.all(np.abs(np.sum(out * exp, axis=1)) > 1 - 1e-12)
    # at knots returns the sample rotation
    at = interp_to(t, t, q, quaternion=True)
    assert np.all(np.abs(np.sum(at * q, axis=1)) > 1 - 1e-12)


def test_interp_quaternion_takes_shortest_arc_for_antipodal_signs():
    """q and -q are the same rotation: interpolating must not sweep the long way round."""
    q0 = R.from_rotvec([0.1, 0.0, 0.0]).as_quat()[[3, 0, 1, 2]]
    q1 = R.from_rotvec([0.5, 0.0, 0.0]).as_quat()[[3, 0, 1, 2]]
    out = interp_to(np.array([0.5]), np.array([0.0, 1.0]), np.stack([q0, -q1]), quaternion=True)[0]
    exp = R.from_rotvec([0.3, 0.0, 0.0]).as_quat()[[3, 0, 1, 2]]
    assert abs(float(np.dot(out, exp))) > 1 - 1e-12


def test_interp_quaternion_clamps():
    q = R.random(2, random_state=3).as_quat()[:, [3, 0, 1, 2]]
    out = interp_to(np.array([-1.0, 5.0]), np.array([0.0, 1.0]), q, quaternion=True)
    assert abs(np.dot(out[0], q[0])) > 1 - 1e-12
    assert abs(np.dot(out[1], q[1])) > 1 - 1e-12


def test_interp_length_mismatch_raises():
    with pytest.raises(ValueError):
        interp_to(np.array([0.5]), np.array([0.0, 1.0, 2.0]), np.zeros((2, 3)))


@pytestmark_logs
def test_interp_ekf2_attitude_onto_imu_grid_is_unit_and_close_to_nearest(log_id):
    """Integration check on real data: slerped EKF2 attitude on the IMU grid stays unit and within the
    maximum per-sample EKF2 attitude step of the nearest EKF2 sample."""
    log = _load(log_id)
    ekf = log.ekf2
    m = (log.imu.t >= ekf.t_att[0]) & (log.imu.t <= ekf.t_att[-1])
    out = interp_to(log.imu.t[m], ekf.t_att, ekf.q, quaternion=True)
    assert np.max(np.abs(np.linalg.norm(out, axis=1) - 1.0)) < 1e-12
    exp = _rot_slerp(ekf.t_att, ekf.q)(log.imu.t[m]).as_quat()[:, [3, 0, 1, 2]]
    assert np.all(np.abs(np.sum(out * exp, axis=1)) > 1 - 1e-12)
