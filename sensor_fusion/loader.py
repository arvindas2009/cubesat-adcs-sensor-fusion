"""PX4 ULog loader for Project A (spec ``docs/specs/sensor_fusion_data.md`` sections 2-3).

``load_log(path_or_id)`` reads a ``.ulg`` with pyulog and returns a ``FlightLog`` whose
containers hold numpy arrays in SI units and repository frames (CONVENTIONS sections
1, 2, 4, 5): body FRD, local NED, time on axis 0 in float seconds since ``t0_us``
(boot-time microseconds), quaternions scalar-first Hamilton FRD->NED, ``w >= 0``.

Native sample rates are preserved (no resampling here; see ``sensor_fusion.interp``).

Topics used (logged field names differ between PX4 versions; both handled)
------------------------------------------------------------------------
IMU   ``sensor_combined`` (instance 0): ``gyro_rad[3]`` [rad/s], ``accelerometer_m_s2[3]``
      [m/s^2, specific force], both FRD.  Fallback ``vehicle_imu`` (instance 0):
      ``delta_angle / delta_angle_dt`` and ``delta_velocity / delta_velocity_dt``
      (dt in us).  ``imu_source="auto"`` (default) prefers ``sensor_combined`` because the
      default PX4 log profile records ``vehicle_imu`` at only 2 Hz.
MAG   ``vehicle_magnetometer.magnetometer_ga[3]`` gauss -> tesla (x 1e-4); stamped with
      ``timestamp_sample`` when the topic has it (else ``timestamp``).
BARO  ``vehicle_air_data``: ``baro_pressure_pa`` [Pa], ``baro_alt_meter`` [m, up],
      temperature ``ambient_temperature`` (new) or ``baro_temp_celcius`` (old) [degC];
      stamped with ``timestamp_sample`` when present (else ``timestamp``).
GPS   ``vehicle_gps_position``: new format ``latitude_deg/longitude_deg`` [deg],
      ``altitude_ellipsoid_m``/``altitude_msl_m`` [m]; old format ``lat/lon`` [deg 1e-7],
      ``alt_ellipsoid/alt`` [mm].  Velocity ``vel_n/e/d_m_s`` [m/s] (NED).
      ``GpsData.alt`` is the ellipsoidal height, ``GpsData.alt_msl`` the height above the
      geoid (NaN if the log carries none); ``alt - alt_msl`` is the geoid separation.
EKF2  ``vehicle_attitude.q`` (FRD->NED) and ``vehicle_local_position`` (x, y, z, vx, vy,
      vz in NED [m, m/s], ``ref_lat/ref_lon`` [deg], ``ref_alt`` [m AMSL]).
Flight window: ``vehicle_status.arming_state == 2`` -> ``actuator_armed.armed`` ->
      ``vehicle_land_detected.landed == 0`` (first available), +-``CROP_MARGIN_S``.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from common import quaternion as quat
from sensor_fusion.download_logs import DATA_DIR
from sensor_fusion.interp import interp_to  # re-exported helper for consumers

__all__ = ["FlightLog", "ImuData", "MagData", "BaroData", "GpsData", "Ekf2Data",
           "load_log", "interp_to", "sample_rates"]

CROP_MARGIN_S = 2.0
GAP_FACTOR = 5.0
MIN_GPS_FIX = 3
ARMING_STATE_ARMED = 2
GAUSS_TO_TESLA = 1e-4

_TOPICS = ["sensor_combined", "vehicle_imu", "vehicle_magnetometer", "vehicle_air_data",
           "vehicle_gps_position", "vehicle_attitude", "vehicle_local_position",
           "vehicle_status", "actuator_armed", "vehicle_land_detected"]
_PARAMS = ["SYS_AUTOSTART", "EKF2_MAG_TYPE", "EKF2_HGT_REF", "EKF2_IMU_CTRL", "SDLOG_PROFILE",
           "CAL_MAG0_ROT", "SENS_BOARD_ROT", "IMU_INTEG_RATE", "SENS_BARO_QNH"]


# ----------------------------------------------------------------------------- containers
@dataclass
class ImuData:
    t: np.ndarray            # (N,) s
    gyro: np.ndarray         # (N,3) rad/s, FRD
    accel: np.ndarray        # (N,3) m/s^2 specific force, FRD
    dt: np.ndarray           # (N,) s; dt[0] = median
    rate_hz: float


@dataclass
class MagData:
    t: np.ndarray
    field: np.ndarray        # (N,3) tesla, FRD


@dataclass
class BaroData:
    t: np.ndarray
    pressure: np.ndarray     # Pa
    alt_baro: np.ndarray     # m, height UP
    temperature: np.ndarray  # degC


@dataclass
class GpsData:
    t: np.ndarray
    lat: np.ndarray          # rad
    lon: np.ndarray          # rad
    alt: np.ndarray          # m above ellipsoid (MSL if the log has no ellipsoid height; see info)
    alt_msl: np.ndarray      # m above the geoid (NaN if not logged); alt - alt_msl = geoid separation
    vel_ned: np.ndarray      # (N,3) m/s
    eph: np.ndarray          # m
    epv: np.ndarray          # m
    s_variance_m_s: np.ndarray
    fix_type: np.ndarray
    n_sat: np.ndarray
    t_sample: np.ndarray     # s; time of the measurement if logged, else t


@dataclass
class Ekf2Data:
    t_att: np.ndarray
    q: np.ndarray            # (N,4) FRD->NED, scalar-first, w >= 0
    t_pos: np.ndarray
    pos_ned: np.ndarray      # (M,3) m, relative to the (ref_lat, ref_lon) origin; z is down
    vel_ned: np.ndarray      # (M,3) m/s
    ref_lat: float           # rad
    ref_lon: float           # rad
    ref_alt: float           # m AMSL (as logged by PX4)


@dataclass
class FlightLog:
    log_id: str
    path: Path
    t0_us: int
    imu: ImuData
    mag: MagData
    baro: BaroData
    gps: GpsData
    ekf2: Ekf2Data
    info: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- helpers
def _f64(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def _dataset(ulog, name, info, *, required=True):
    """Lowest-``multi_id`` instance of topic ``name`` (None / ValueError if absent)."""
    ds = sorted((d for d in ulog.data_list if d.name == name), key=lambda d: d.multi_id)
    if not ds:
        if required:
            raise ValueError(f"log has no '{name}' topic")
        return None
    if len(ds) > 1:
        info.setdefault("multi_instance", {})[name] = [d.multi_id for d in ds]
    return ds[0].data


def _vec(data, base, n=3):
    return np.column_stack([_f64(data[f"{base}[{i}]"]) for i in range(n)])


def _clean(name, t_us, cols, removed, drop_masks=None):
    """Explicit cleaning of one stream; returns ``(t_us, cols)`` and fills ``removed[name]``.

    Order: non-finite rows -> stream-specific masks (e.g. GPS fix < 3) -> duplicate /
    backward timestamps (keep the first of equal stamps, drop any stamp <= running max).
    """
    t_us = np.asarray(t_us, dtype=np.int64)
    n_raw = len(t_us)
    bad = np.zeros(n_raw, bool)
    for a in cols.values():
        if np.issubdtype(a.dtype, np.floating):
            bad |= ~np.isfinite(a).reshape(n_raw, -1).all(axis=1)
    rec = {"n_raw": n_raw, "nonfinite": int(bad.sum())}
    keep = ~bad
    for key, mask in (drop_masks or {}).items():
        m = mask & keep
        rec[key] = int(m.sum())
        keep &= ~m
    idx = np.flatnonzero(keep)
    t = t_us[idx]
    ok = np.ones(len(t), bool)
    rec["duplicate_t"] = rec["backward_t"] = 0
    if len(t) > 1:
        prev_max = np.maximum.accumulate(t)[:-1]
        rec["duplicate_t"] = int((t[1:] == prev_max).sum())
        rec["backward_t"] = int((t[1:] < prev_max).sum())
        ok[1:] = t[1:] > prev_max
    idx = idx[ok]
    rec["n_after_clean"] = int(len(idx))
    removed[name] = rec
    return t_us[idx], {k: v[idx] for k, v in cols.items()}


def _median_rate(t) -> float:
    return float(1.0 / np.median(np.diff(t))) if len(t) > 1 else float("nan")


def _gaps(name, t, out):
    if len(t) < 3:
        return
    d = np.diff(t)
    med = np.median(d)
    for i in np.flatnonzero(d > GAP_FACTOR * med):
        out.append((name, float(t[i]), float(t[i + 1])))


def _arming_intervals(ulog, info):
    """Armed (or airborne) intervals in boot-time us, plus the source used."""
    def runs(t, flag):
        flag = np.asarray(flag, bool)
        if not flag.any():
            return []
        edges = np.diff(flag.astype(np.int8))
        starts = list(np.flatnonzero(edges == 1) + 1)
        ends = list(np.flatnonzero(edges == -1))
        if flag[0]:
            starts.insert(0, 0)
        if flag[-1]:
            ends.append(len(flag) - 1)
        return [(int(t[s]), int(t[e])) for s, e in zip(starts, ends)]

    for topic, key, test, label in (
        ("vehicle_status", "arming_state", lambda a: a == ARMING_STATE_ARMED, "vehicle_status.arming_state"),
        ("actuator_armed", "armed", lambda a: a != 0, "actuator_armed.armed"),
        ("vehicle_land_detected", "landed", lambda a: a == 0, "vehicle_land_detected.landed==0 (airborne)"),
    ):
        d = _dataset(ulog, topic, info, required=False)
        if d is None or key not in d:
            continue
        iv = runs(d["timestamp"].astype(np.int64), test(d[key]))
        if iv:
            return iv, label
    return [], "none"


def _decode_release(r) -> str:
    r = int(r)
    return f"v{(r >> 24) & 255}.{(r >> 16) & 255}.{(r >> 8) & 255}" + ("" if (r & 255) == 255 else f"-dev{r & 255}")


# ----------------------------------------------------------------------------- stream readers
def _read_imu(ulog, info, source):
    sc = _dataset(ulog, "sensor_combined", info, required=False)
    if source == "auto":
        source = "sensor_combined" if sc is not None else "vehicle_imu"
    info["imu_source"] = source
    if source == "sensor_combined":
        if sc is None:
            raise ValueError("log has no 'sensor_combined' topic")
        cols = {"gyro": _vec(sc, "gyro_rad"), "accel": _vec(sc, "accelerometer_m_s2")}
        for k in ("accelerometer_clipping", "gyro_clipping"):
            if k in sc:
                info.setdefault("imu_clipped_samples", {})[k] = int((sc[k] != 0).sum())
        if "accelerometer_timestamp_relative" in sc:
            rel = sc["accelerometer_timestamp_relative"].astype(np.int64)
            valid = np.abs(rel) < 1_000_000  # INT32_MAX marks "accel sample not in this message"
            info["imu_accel_timestamp_relative_us_median"] = float(np.median(rel[valid])) if valid.any() else None
            info["imu_accel_timestamp_relative_invalid"] = int((~valid).sum())
        return sc["timestamp"].astype(np.int64), cols
    # vehicle_imu: integrate-and-dump deltas -> mean rate / specific force over the interval
    vi = _dataset(ulog, "vehicle_imu", info)
    dta = _f64(vi["delta_angle_dt"]) * 1e-6
    dtv = _f64(vi["delta_velocity_dt"]) * 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        cols = {"gyro": _vec(vi, "delta_angle") / dta[:, None],
                "accel": _vec(vi, "delta_velocity") / dtv[:, None]}
    return vi["timestamp"].astype(np.int64), cols


def _sample_stamp(d, name, info) -> np.ndarray:
    """Time stamps [boot us] of a topic: ``timestamp_sample`` where present and non-zero, else ``timestamp``.

    Records the stamp source and the median publish latency ``timestamp - timestamp_sample``
    in ``info[f"{name}_time_source"]`` / ``info[f"{name}_latency_us_median"]``.
    """
    t = d["timestamp"].astype(np.int64)
    if "timestamp_sample" not in d:
        info[f"{name}_time_source"] = "timestamp (topic has no timestamp_sample)"
        return t
    ts = d["timestamp_sample"].astype(np.int64)
    ok = ts > 0
    info[f"{name}_time_source"] = "timestamp_sample" if ok.all() else         f"timestamp_sample (fallback to timestamp for {int((~ok).sum())} rows)"
    info[f"{name}_latency_us_median"] = float(np.median(t - np.where(ok, ts, t)))
    return np.where(ok, ts, t)


def _read_mag(ulog, info):
    m = _dataset(ulog, "vehicle_magnetometer", info)
    info["mag_raw_unit"] = "gauss (magnetometer_ga), converted x1e-4 -> tesla"
    t = _sample_stamp(m, "mag", info)
    return t, {"field": _vec(m, "magnetometer_ga") * GAUSS_TO_TESLA}


def _read_baro(ulog, info):
    b = _dataset(ulog, "vehicle_air_data", info)
    temp_key = next((k for k in ("ambient_temperature", "baro_temp_celcius") if k in b), None)
    info["baro_temperature_field"] = temp_key
    temp = _f64(b[temp_key]) if temp_key else np.full(len(b["timestamp"]), np.nan)
    return _sample_stamp(b, "baro", info), {
        "pressure": _f64(b["baro_pressure_pa"]), "alt_baro": _f64(b["baro_alt_meter"]), "temperature": temp}


def _read_gps(ulog, info):
    g = _dataset(ulog, "vehicle_gps_position", info)
    if "latitude_deg" in g:
        info["gps_format"] = "new (latitude_deg/longitude_deg in deg, altitude_*_m in m)"
        lat, lon = _f64(g["latitude_deg"]), _f64(g["longitude_deg"])
        ell, msl = (_f64(g[k]) if k in g else None for k in ("altitude_ellipsoid_m", "altitude_msl_m"))
    else:
        info["gps_format"] = "old (lat/lon integer deg*1e-7, alt/alt_ellipsoid integer mm)"
        lat, lon = _f64(g["lat"]) * 1e-7, _f64(g["lon"]) * 1e-7
        ell = _f64(g["alt_ellipsoid"]) * 1e-3 if "alt_ellipsoid" in g else None
        msl = _f64(g["alt"]) * 1e-3 if "alt" in g else None
    if np.nanmax(np.abs(lat)) > 90 or np.nanmax(np.abs(lon)) > 180:
        raise ValueError("GPS lat/lon out of range after scaling; unknown message format")
    if ell is not None and np.any(ell != 0):
        alt, info["gps_alt_source"] = ell, "ellipsoid"
    else:
        alt, info["gps_alt_source"] = msl, "MSL (log carries no ellipsoid height)"
    if msl is None:
        msl = np.full(len(lat), np.nan)
    if alt is None:
        raise ValueError("GPS message carries neither ellipsoid nor MSL height")
    utc = g["time_utc_usec"].astype(np.int64) if "time_utc_usec" in g else np.zeros(len(lat), np.int64)
    t = g["timestamp"].astype(np.int64)
    ts = g["timestamp_sample"].astype(np.int64) if "timestamp_sample" in g else t
    ts = np.where(ts > 0, ts, t)
    info["gps_latency_us_median"] = float(np.median(t - ts))
    vel_valid = g["vel_ned_valid"] if "vel_ned_valid" in g else np.ones(len(t), bool)
    info["gps_vel_ned_invalid_rows_raw"] = int((vel_valid == 0).sum())
    cols = {
        "lat": np.radians(lat), "lon": np.radians(lon), "alt": alt, "alt_msl": msl,
        "vel_ned": np.column_stack([_f64(g["vel_n_m_s"]), _f64(g["vel_e_m_s"]), _f64(g["vel_d_m_s"])]),
        "eph": _f64(g["eph"]), "epv": _f64(g["epv"]), "s_variance_m_s": _f64(g["s_variance_m_s"]),
        "fix_type": g["fix_type"].astype(np.int64), "n_sat": g["satellites_used"].astype(np.int64),
        "ts_us": ts, "utc_us": utc,
    }
    return t, cols, {"fix_lt3": g["fix_type"].astype(np.int64) < MIN_GPS_FIX}


def _read_ekf2(ulog, info):
    a = _dataset(ulog, "vehicle_attitude", info)
    q = np.column_stack([_f64(a[f"q[{i}]"]) for i in range(4)])
    n = np.linalg.norm(q, axis=1)
    info["ekf2_q_norm_dev_max_raw"] = float(np.nanmax(np.abs(n - 1.0)))
    if "quat_reset_counter" in a:
        ch = np.flatnonzero(np.diff(a["quat_reset_counter"].astype(np.int64)) != 0) + 1
        info["ekf2_quat_resets"] = int(len(ch))
        info["_ekf2_quat_reset_boot_us"] = [int(a["timestamp"][i]) for i in ch]
    att = (a["timestamp"].astype(np.int64), {"q": q}, {"zero_norm_q": ~(n > 1e-6)})
    p = _dataset(ulog, "vehicle_local_position", info)
    for k in ("xy_reset_counter", "z_reset_counter", "vxy_reset_counter", "vz_reset_counter"):
        if k in p:
            info.setdefault("ekf2_local_pos_resets", {})[k] = int((np.diff(p[k].astype(np.int64)) != 0).sum())
    for k in ("xy_valid", "z_valid", "v_xy_valid", "v_z_valid"):
        if k in p:
            info.setdefault("ekf2_local_pos_invalid_rows_raw", {})[k] = int((p[k] == 0).sum())
    cols = {"pos_ned": np.column_stack([_f64(p["x"]), _f64(p["y"]), _f64(p["z"])]),
            "vel_ned": np.column_stack([_f64(p["vx"]), _f64(p["vy"]), _f64(p["vz"])]),
            "ref_lat_deg": _f64(p["ref_lat"]), "ref_lon_deg": _f64(p["ref_lon"]), "ref_alt": _f64(p["ref_alt"])}
    return att, (p["timestamp"].astype(np.int64), cols)


# ----------------------------------------------------------------------------- public API
def _resolve_path(path_or_id) -> Path:
    p = Path(str(path_or_id))
    if p.suffix == ".ulg" or p.exists():
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    cand = DATA_DIR / f"{path_or_id}.ulg"
    if not cand.exists():
        raise FileNotFoundError(f"{cand} not found; run `python -m sensor_fusion.download_logs`")
    return cand


def sample_rates(log: FlightLog) -> dict:
    """Median sample rate [Hz] of every stream of ``log``."""
    return {"imu": log.imu.rate_hz, "mag": _median_rate(log.mag.t), "baro": _median_rate(log.baro.t),
            "gps": _median_rate(log.gps.t), "ekf2_att": _median_rate(log.ekf2.t_att),
            "ekf2_pos": _median_rate(log.ekf2.t_pos)}


def load_log(path_or_id, *, crop_to_flight: bool = True, imu_source: str = "auto") -> FlightLog:
    """Load a PX4 ULog into a ``FlightLog`` (see module docstring and the spec).

    ``imu_source`` is ``"auto"`` (sensor_combined, else vehicle_imu), ``"sensor_combined"``
    or ``"vehicle_imu"`` (an extension to the spec signature, used to test the fallback).
    """
    from pyulog import ULog

    if imu_source not in ("auto", "sensor_combined", "vehicle_imu"):
        raise ValueError(f"imu_source must be auto/sensor_combined/vehicle_imu, got {imu_source!r}")
    path = _resolve_path(path_or_id)
    ulog = ULog(str(path), message_name_filter_list=_TOPICS)
    info: dict = {"removed": {}, "gaps": []}
    removed = info["removed"]

    # ---- metadata
    mi = ulog.msg_info_dict
    info["size_bytes"] = path.stat().st_size
    info["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    info["firmware"] = {"ver_sw_git": mi.get("ver_sw"), "ver_sw_release": _decode_release(mi["ver_sw_release"])
                        if "ver_sw_release" in mi else mi.get("ver_sw_release_str"),
                        "ver_hw": mi.get("ver_hw"), "sys_name": mi.get("sys_name"),
                        "branch": mi.get("ver_sw_branch")}
    info["airframe_sys_autostart"] = ulog.initial_parameters.get("SYS_AUTOSTART")
    info["params"] = {k: ulog.initial_parameters[k] for k in _PARAMS if k in ulog.initial_parameters}
    info["params_changed_in_flight"] = len(ulog.changed_parameters)
    info["log_duration_s"] = (ulog.last_timestamp - ulog.start_timestamp) * 1e-6
    info["logger_dropouts"] = [(d.timestamp * 1e-6, d.duration * 1e-3) for d in ulog.dropouts]

    # ---- raw streams
    imu_t, imu_c = _read_imu(ulog, info, imu_source)
    mag_t, mag_c = _read_mag(ulog, info)
    baro_t, baro_c = _read_baro(ulog, info)
    gps_t, gps_c, gps_masks = _read_gps(ulog, info)
    (att_t, att_c, att_masks), (pos_t, pos_c) = _read_ekf2(ulog, info)

    streams = {}
    streams["imu"] = _clean("imu", imu_t, imu_c, removed)
    streams["mag"] = _clean("mag", mag_t, mag_c, removed)
    streams["baro"] = _clean("baro", baro_t, baro_c, removed)
    streams["gps"] = _clean("gps", gps_t, gps_c, removed, gps_masks)
    streams["ekf2_att"] = _clean("ekf2_att", att_t, att_c, removed, att_masks)
    streams["ekf2_pos"] = _clean("ekf2_pos", pos_t, pos_c, removed)
    for name, (t, _) in streams.items():
        if len(t) < 2:
            raise ValueError(f"stream '{name}' has fewer than 2 valid samples after cleaning")

    # ---- common window (+ optional crop to the armed window)
    lo = max(int(t[0]) for t, _ in streams.values())
    hi = min(int(t[-1]) for t, _ in streams.values())
    info["common_window_us"] = (lo, hi)
    iv, src = _arming_intervals(ulog, info)
    info["arming_source"] = src
    info["arming_windows_boot_s"] = [(a * 1e-6, b * 1e-6) for a, b in iv]
    info["crop_to_flight"] = bool(crop_to_flight)
    if crop_to_flight and iv:
        m = int(CROP_MARGIN_S * 1e6)
        lo, hi = max(lo, min(a for a, _ in iv) - m), min(hi, max(b for _, b in iv) + m)
    elif crop_to_flight:
        info["crop_note"] = "no arming information found; not cropped"
    if hi <= lo:
        raise ValueError("empty common window after cropping")
    sliced = {}
    for name, (t, c) in streams.items():
        sel = (t >= lo) & (t <= hi)
        removed[name]["outside_window"] = int((~sel).sum())
        removed[name]["n_kept"] = int(sel.sum())
        sliced[name] = (t[sel], {k: v[sel] for k, v in c.items()})
        if sel.sum() < 2:
            raise ValueError(f"stream '{name}' has < 2 samples in the window")
    t0_us = int(min(t[0] for t, _ in sliced.values()))
    info["flight_window_boot_s"] = (lo * 1e-6, hi * 1e-6)
    info["t0_us"] = t0_us

    def sec(t_us):
        return (np.asarray(t_us, dtype=np.int64) - t0_us) * 1e-6

    # ---- containers
    t_imu, c = sliced["imu"]
    t = sec(t_imu)
    dt = np.empty_like(t)
    dt[1:] = np.diff(t)
    dt[0] = np.median(dt[1:])
    imu = ImuData(t=t, gyro=c["gyro"], accel=c["accel"], dt=dt, rate_hz=float(1.0 / np.median(dt[1:])))

    t_m, c = sliced["mag"]
    mag = MagData(t=sec(t_m), field=c["field"])
    t_b, c = sliced["baro"]
    baro = BaroData(t=sec(t_b), pressure=c["pressure"], alt_baro=c["alt_baro"], temperature=c["temperature"])
    t_g, c = sliced["gps"]
    c_gps_utc = c["utc_us"].astype(np.int64)
    gps = GpsData(t=sec(t_g), lat=c["lat"], lon=c["lon"], alt=c["alt"], alt_msl=c["alt_msl"], vel_ned=c["vel_ned"], eph=c["eph"],
                  epv=c["epv"], s_variance_m_s=c["s_variance_m_s"], fix_type=c["fix_type"], n_sat=c["n_sat"],
                  t_sample=sec(c["ts_us"]))
    t_a, ca = sliced["ekf2_att"]
    t_p, cp = sliced["ekf2_pos"]
    refs = np.column_stack([cp["ref_lat_deg"], cp["ref_lon_deg"], cp["ref_alt"]])
    info["ekf2_ref_changes"] = int((np.abs(np.diff(refs, axis=0)).max(axis=1) > 0).sum())
    ekf2 = Ekf2Data(t_att=sec(t_a), q=quat.canonicalize(quat.normalize(ca["q"])),
                    t_pos=sec(t_p), pos_ned=cp["pos_ned"], vel_ned=cp["vel_ned"],
                    ref_lat=float(np.radians(refs[0, 0])), ref_lon=float(np.radians(refs[0, 1])),
                    ref_alt=float(refs[0, 2]))

    log = FlightLog(log_id=path.stem, path=path, t0_us=t0_us, imu=imu, mag=mag, baro=baro, gps=gps,
                    ekf2=ekf2, info=info)
    for name, tt in (("imu", imu.t), ("mag", mag.t), ("baro", baro.t), ("gps", gps.t),
                     ("ekf2_att", ekf2.t_att), ("ekf2_pos", ekf2.t_pos)):
        _gaps(name, tt, info["gaps"])
    info["ekf2_quat_reset_times_s"] = [float(sec(x)) for x in info.pop("_ekf2_quat_reset_boot_us", [])]
    info["rates_hz"] = sample_rates(log)
    # Time of the last sample of ANY of the six streams (IMU, mag, baro, GPS, EKF2 attitude, EKF2
    # position), in seconds after t0 (t0 = earliest first sample in the window). It is the span of
    # the retained data, not of the armed period alone (crop margin included).
    info["window_duration_s"] = float(max(imu.t[-1], mag.t[-1], baro.t[-1], gps.t[-1],
                                          ekf2.t_att[-1], ekf2.t_pos[-1]))
    info["ekf2_hgt_ref"] = info["params"].get("EKF2_HGT_REF")
    utc = c_gps_utc[c_gps_utc > 0]
    info["gps_utc_start"] = (datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=int(utc[0]))
                             if len(utc) else None)
    return log
