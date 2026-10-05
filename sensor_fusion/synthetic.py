"""Synthetic truth + sensor generator for the Project A filter tests (Group 3, M3).

Spec: ``docs/specs/ekf_drone.md`` section 6. Conventions: CONVENTIONS.md (scalar-first
Hamilton quaternions, FRD body -> NED, specific force ``f_b = R(q)^T (a - g_n) + b_a``,
``g_n = [0, 0, g]``, SI units, radians).

Truth trajectories are closed-form functions of time ``t`` (``w0 = 2 pi / 20`` rad/s):

``"figure8"`` (position NED [m], Euler angles ZYX [rad])::

    n = 20 sin(w0 t)            e = 10 sin(2 w0 t)          d = -10 - 2 sin(w0 t / 2)
    roll  phi   = 0.20 sin(w0 t + 0.3)         (11.5 deg)
    pitch theta = 0.15 sin(2 w0 t + 1.0)       (8.6 deg)
    yaw   psi   = 0.3 + 0.1 t + 0.6 sin(w0 t)  (rotates ~ 6 rad per minute)

velocity and acceleration are the analytic first and second derivatives of ``p(t)``
(horizontal speed <= 6.3 m/s, horizontal acceleration <= 4 m/s^2).

``"hover"``::

    p = [0, 0, -5],  v = a = 0
    roll = 0.04 sin(1.3 t), pitch = 0.035 sin(0.9 t + 1.0) (<= 2.3 deg), yaw = 0.6 constant

``"static"``: ``p = 0``, ``v = a = 0``, constant roll 5 deg, pitch -3 deg, yaw 40 deg.

Body rates come from the analytic ZYX Euler-rate relation (never finite differences)::

    w_b = [ phi' - psi' sin(theta),
            theta' cos(phi) + psi' cos(theta) sin(phi),
           -theta' sin(phi) + psi' cos(theta) cos(phi) ]

Sensors: IMU samples are **instantaneous** values on the uniform grid
``t_k = k / imu_rate``; gyro ``= w_b + b_g(t) + n`` and accel
``= R(q)^T (a - g_n) + b_a(t) + n`` with white per-sample sigma
``density * sqrt(imu_rate)``; the biases start at ``gyro_bias`` / ``accel_bias`` and random-walk
with increments ``density * sqrt(dt) * N(0, 1)``. GPS (true ``p``, ``v`` + white noise) is sampled
on its own grid and **stamped ``t + gps_delay``**. Baro is ``p_D + noise``; magnetometer is
``R(q)^T m_n + noise`` (tesla, per-axis white noise).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common import quaternion as quat
from common.frames import G0
from sensor_fusion.ekf_drone import SensorStreams

__all__ = ["SyntheticNoise", "SyntheticTruth", "make_synthetic_flight", "default_field_ned",
           "trajectory_state"]

_W0 = 2.0 * np.pi / 20.0


@dataclass
class SyntheticNoise:
    """Noise densities (same units as ``EkfConfig``) and white measurement sigmas."""
    accel_noise: float = 0.0          # m/s^2/sqrt(Hz)
    gyro_noise: float = 0.0           # rad/s/sqrt(Hz)
    gyro_bias_rw: float = 0.0         # rad/s^2/sqrt(Hz)
    accel_bias_rw: float = 0.0        # m/s^3/sqrt(Hz)
    gps_pos_sigma: tuple = (0.0, 0.0, 0.0)   # m, NED
    gps_vel_sigma: float = 0.0        # m/s per axis
    baro_sigma: float = 0.0           # m
    mag_sigma: float = 0.0            # tesla per axis


@dataclass
class SyntheticTruth:
    t: np.ndarray
    p: np.ndarray
    v: np.ndarray
    a: np.ndarray
    q: np.ndarray
    omega: np.ndarray                 # body rate, FRD
    b_g: np.ndarray
    b_a: np.ndarray
    meta: dict = field(default_factory=dict)


def default_field_ned(declination: float, total: float = 50e-6, inclination_deg: float = 65.0):
    """NED magnetic field [T]: horizontal part rotated east of true north by ``declination``."""
    inc = np.deg2rad(inclination_deg)
    h = total * np.cos(inc)
    return np.array([h * np.cos(declination), h * np.sin(declination), total * np.sin(inc)])


def trajectory_state(trajectory: str, t) -> dict:
    """Closed-form truth at times ``t`` (N,): p, v, a (N,3), Euler angles (N,), q (N,4), omega (N,3)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    z = np.zeros_like(t)
    w = _W0
    if trajectory == "figure8":
        p = np.stack([20.0 * np.sin(w * t), 10.0 * np.sin(2 * w * t), -10.0 - 2.0 * np.sin(0.5 * w * t)], -1)
        v = np.stack([20.0 * w * np.cos(w * t), 20.0 * w * np.cos(2 * w * t), -w * np.cos(0.5 * w * t)], -1)
        a = np.stack([-20.0 * w ** 2 * np.sin(w * t), -40.0 * w ** 2 * np.sin(2 * w * t),
                      0.5 * w ** 2 * np.sin(0.5 * w * t)], -1)
        roll = 0.20 * np.sin(w * t + 0.3)
        roll_d = 0.20 * w * np.cos(w * t + 0.3)
        pitch = 0.15 * np.sin(2 * w * t + 1.0)
        pitch_d = 0.30 * w * np.cos(2 * w * t + 1.0)
        yaw = 0.3 + 0.1 * t + 0.6 * np.sin(w * t)
        yaw_d = 0.1 + 0.6 * w * np.cos(w * t)
    elif trajectory == "hover":
        p = np.stack([z, z, z - 5.0], -1)
        v = np.zeros((t.size, 3))
        a = np.zeros((t.size, 3))
        roll = 0.04 * np.sin(1.3 * t)
        roll_d = 0.04 * 1.3 * np.cos(1.3 * t)
        pitch = 0.035 * np.sin(0.9 * t + 1.0)
        pitch_d = 0.035 * 0.9 * np.cos(0.9 * t + 1.0)
        yaw = z + 0.6
        yaw_d = z
    elif trajectory == "static":
        p = np.zeros((t.size, 3))
        v = np.zeros((t.size, 3))
        a = np.zeros((t.size, 3))
        roll = z + np.deg2rad(5.0)
        pitch = z + np.deg2rad(-3.0)
        yaw = z + np.deg2rad(40.0)
        roll_d = pitch_d = yaw_d = z
    else:
        raise ValueError(f"unknown trajectory {trajectory!r}; use 'figure8', 'hover' or 'static'")
    sp, cp = np.sin(pitch), np.cos(pitch)
    sr, cr = np.sin(roll), np.cos(roll)
    omega = np.stack([roll_d - yaw_d * sp,
                      pitch_d * cr + yaw_d * cp * sr,
                      -pitch_d * sr + yaw_d * cp * cr], -1)
    q = quat.from_euler_zyx(roll, pitch, yaw)
    return {"p": p, "v": v, "a": a, "q": q, "omega": omega, "roll": roll, "pitch": pitch, "yaw": yaw}


def make_synthetic_flight(trajectory: str = "figure8", duration: float = 60.0, imu_rate: float = 100.0,
                          gps_rate: float = 5.0, baro_rate: float = 10.0, mag_rate: float = 10.0, *,
                          noise: SyntheticNoise | None = None, gyro_bias=(0.0, 0.0, 0.0),
                          accel_bias=(0.0, 0.0, 0.0), gps_delay: float = 0.0, declination: float = 0.1,
                          field_ned=None, g: float = G0, seed: int = 0):
    """Generate ``(SensorStreams, SyntheticTruth)``; see the module docstring for the models."""
    noise = SyntheticNoise() if noise is None else noise
    rng = np.random.default_rng(seed)
    n_imu = int(round(duration * imu_rate)) + 1
    t = np.arange(n_imu) / imu_rate
    dt = 1.0 / imu_rate
    tr = trajectory_state(trajectory, t)
    R = quat.to_dcm(tr["q"])                                   # (N,3,3) body -> NED
    g_n = np.array([0.0, 0.0, g])

    # biases: constant start value + random walk
    b_g = np.empty((n_imu, 3)); b_a = np.empty((n_imu, 3))
    b_g[0] = np.asarray(gyro_bias, float)
    b_a[0] = np.asarray(accel_bias, float)
    if n_imu > 1:
        b_g[1:] = b_g[0] + np.cumsum(noise.gyro_bias_rw * np.sqrt(dt) * rng.standard_normal((n_imu - 1, 3)), 0)
        b_a[1:] = b_a[0] + np.cumsum(noise.accel_bias_rw * np.sqrt(dt) * rng.standard_normal((n_imu - 1, 3)), 0)

    f_true = np.einsum("nji,nj->ni", R, tr["a"] - g_n)         # R^T (a - g_n)
    gyro = tr["omega"] + b_g + noise.gyro_noise * np.sqrt(imu_rate) * rng.standard_normal((n_imu, 3))
    accel = f_true + b_a + noise.accel_noise * np.sqrt(imu_rate) * rng.standard_normal((n_imu, 3))

    # GPS
    n_gps = int(np.floor(duration * gps_rate + 1e-9)) + 1
    t_g = np.arange(n_gps) / gps_rate
    sg = trajectory_state(trajectory, t_g)
    sig_p = np.asarray(noise.gps_pos_sigma, float) * np.ones(3)
    gps_pos = sg["p"] + sig_p * rng.standard_normal((n_gps, 3))
    gps_vel = sg["v"] + noise.gps_vel_sigma * rng.standard_normal((n_gps, 3))

    # baro
    n_b = int(np.floor(duration * baro_rate + 1e-9)) + 1
    t_b = np.arange(n_b) / baro_rate
    sb = trajectory_state(trajectory, t_b)
    baro_d = sb["p"][:, 2] + noise.baro_sigma * rng.standard_normal(n_b)

    # magnetometer
    m_n = default_field_ned(declination) if field_ned is None else np.asarray(field_ned, float)
    n_m = int(np.floor(duration * mag_rate + 1e-9)) + 1
    t_m = np.arange(n_m) / mag_rate
    sm = trajectory_state(trajectory, t_m)
    Rm = quat.to_dcm(sm["q"])
    mag = np.einsum("nji,j->ni", Rm, m_n) + noise.mag_sigma * rng.standard_normal((n_m, 3))

    meta = {"synthetic": True, "trajectory": trajectory, "duration": duration, "imu_rate": imu_rate,
            "gps_delay": gps_delay, "seed": seed, "gyro_bias": list(map(float, gyro_bias)),
            "accel_bias": list(map(float, accel_bias)), "field_ned": m_n.tolist()}
    streams = SensorStreams(
        imu_t=t, gyro=gyro, accel=accel,
        gps_t=t_g + gps_delay, gps_pos_ned=gps_pos, gps_vel_ned=gps_vel,
        gps_pos_sigma=np.tile(sig_p, (n_gps, 1)), gps_vel_sigma=np.full(n_gps, float(noise.gps_vel_sigma)),
        baro_t=t_b, baro_d=baro_d, mag_t=t_m, mag_field=mag,
        g=float(g), declination=float(declination), origin_lla=(0.0, 0.0, 0.0), meta=meta)
    truth = SyntheticTruth(t=t, p=tr["p"], v=tr["v"], a=tr["a"], q=tr["q"], omega=tr["omega"],
                           b_g=b_g, b_a=b_a, meta={"trajectory": trajectory})
    return streams, truth
