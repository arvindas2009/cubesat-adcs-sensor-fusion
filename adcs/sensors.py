"""Sensor models for the 3U CubeSat ADCS (Project B): magnetometer, coarse sun sensors, gyro.

Frames and units (CONVENTIONS.md): SI throughout, vectors in **spacecraft body axes**
unless named ``*_eci``; quaternions scalar-first Hamilton, body-to-ECI; body rates
are relative to ECI, in body axes (rad/s); magnetic field in tesla; time on axis 0.

Sampling model (``docs/specs/adcs_estimation.md`` section 0)
-----------------------------------------------------------
All sensors are sampled once per control period ``T = 1 s`` at ``t_k = k T`` (the
magnetometer with the coils off, Group 6 duty cycle).

* **Gyro** sample ``k >= 1`` is the *mean body rate over [t_{k-1}, t_k]* plus bias and
  noise.  The noise-free mean rate is the constant rate that maps ``q_{k-1}`` to
  ``q_k`` exactly, ``w_k = to_rotvec(q_{k-1}* (x) q_k) / T`` (ideal, coning-compensated
  rate-integrating gyro).  Sample 0 does not exist (NaN row).
* **Magnetometer** and **sun sensor** samples ``k >= 0`` are instantaneous at ``t_k``
  with the true ``q_k``.
* The estimator at step ``k`` propagates with gyro sample ``k`` from ``t_{k-1}`` to
  ``t_k``, then updates with the magnetometer, then the sun sensor.  With this
  definition ``q_k = q_{k-1} (x) exp(w_k T)`` holds exactly for the truth, so all
  estimation error comes from sensor noise and bias.

Parameters (assumptions.md [B-07]; vendor pages read 2026-10-06)
----------------------------------------------------------------
==============  ==================  ==========================  =====================================
Sensor          Part                Datasheet value             Model parameter
==============  ==================  ==========================  =====================================
Magnetometer    PNI RM3100          noise 15 nT                 white noise sigma_m = 15e-9 T per
                                                                axis; no bias, scale or misalignment,
                                                                no coil/spacecraft field
Gyro            Analog Devices      ARW 0.65 deg/sqrt(h);       ARW sigma_v = 0.65 (pi/180)/60
                ADXRS453            in-run bias stability       = 1.8908e-4 rad/sqrt(s); bias random
                                    16 deg/h; rate noise        walk sigma_u = b_inst/sqrt(P_orbit)
                                    density 0.015 deg/s/rtHz    = 7.7570e-5/sqrt(5553.6)
                                                                = 1.0409e-6 rad/s/sqrt(s) (ASSUMPTION:
                                                                1-sigma bias drift over one orbit =
                                                                in-run bias stability); turn-on bias
                                                                N(0, 0.1 deg/s = 1.7453e-3 rad/s)
                                                                per axis (design choice)
Coarse sun      Tensor Tech         "< 5 deg (1-sigma) sun      6 ideal cosine photodiodes (faces
sensor          CSS-10, x6          determination accuracy"     +-x, +-y, +-z), hemispherical response,
                                    after calibration; +-60 deg FOV/albedo/Earth IR not modelled;
                                    FOV                         per-diode white noise sigma_d = 0.0436
                                                                of full-scale current -> reconstructed
                                                                vector per-axis noise sqrt(2) sigma_d
                                                                = 0.0617, total angle error
                                                                sqrt(2)*0.0617 rad = 5.0 deg (1-sigma)
==============  ==================  ==========================  =====================================

Randomness only enters through ``numpy.random.Generator`` objects passed in or created
from ``seed`` (no global state).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from common import quaternion as quat

__all__ = [
    "CSS_NORMALS",
    "MagnetometerConfig",
    "SunSensorConfig",
    "GyroConfig",
    "magnetometer",
    "sun_sensor_currents",
    "sun_vector_from_currents",
    "Gyro",
    "mean_rates",
    "SensorData",
    "measure_series",
]

#: Sun-sensor face normals in body axes, rows ``+x, -x, +y, -y, +z, -z``, shape (6, 3).
CSS_NORMALS = np.array(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ]
)
CSS_NORMALS.setflags(write=False)


@dataclass(frozen=True)
class MagnetometerConfig:
    """Magnetometer noise: ``sigma`` is the white-noise std per axis in T (RM3100, 15 nT)."""

    sigma: float = 15e-9


@dataclass(frozen=True)
class SunSensorConfig:
    """Coarse sun sensor model.

    ``sigma_diode``: per-diode white noise std as a fraction of full-scale current
    (dimensionless).  ``threshold``: minimum summed current (fraction of full scale)
    for a valid sun measurement.  0.6 puts eclipse false detections at 5.6 sigma of
    the summed noise (sqrt(6) sigma_d = 0.107) and lit misses (sum >= 1) at 3.7 sigma.
    """

    sigma_diode: float = 0.0436
    threshold: float = 0.6

    @property
    def sigma_vector(self) -> float:
        """Per-axis std of the reconstructed sun vector, ``sqrt(2) sigma_diode`` (rad, unit vector)."""
        return float(np.sqrt(2.0) * self.sigma_diode)


@dataclass(frozen=True)
class GyroConfig:
    """Gyro noise: ``arw`` angle random walk (rad/sqrt(s)), ``rrw`` bias random walk
    (rad/s/sqrt(s)), ``bias0_sigma`` per-axis std of the turn-on bias (rad/s)."""

    arw: float = 1.8908e-4
    rrw: float = 1.0409e-6
    bias0_sigma: float = 1.7453e-3


def _f(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def magnetometer(b_body, rng: np.random.Generator, cfg: MagnetometerConfig = MagnetometerConfig()) -> np.ndarray:
    """Magnetometer reading ``(..., 3)`` [T] in body axes: ``b_body + N(0, sigma^2 I)``.

    ``b_body (..., 3)`` is the true field in body axes [T].  Not mutated.
    """
    b = _f(b_body)
    return b + rng.normal(0.0, cfg.sigma, size=b.shape)


def sun_sensor_currents(
    s_body, lit, rng: np.random.Generator, cfg: SunSensorConfig = SunSensorConfig()
) -> np.ndarray:
    """Six photodiode currents ``(..., 6)`` [fraction of full scale], order of ``CSS_NORMALS``.

    ``s_body (..., 3)`` unit sun direction in body axes, ``lit (...)`` bool (False in
    eclipse).  Current ``= where(lit, max(0, n_i . s_body), 0) + N(0, sigma_d^2)`` on all
    six diodes (also in eclipse); noisy currents may be slightly negative.
    """
    s = _f(s_body)
    lit = np.asarray(lit, dtype=bool)
    cos_inc = s @ CSS_NORMALS.T
    clean = np.where(lit[..., None], np.maximum(cos_inc, 0.0), 0.0)
    return clean + rng.normal(0.0, cfg.sigma_diode, size=clean.shape)


def sun_vector_from_currents(currents, cfg: SunSensorConfig = SunSensorConfig()):
    """Reconstruct the sun vector from diode currents ``(..., 6)``.

    Returns ``(s_meas (..., 3), valid (...) bool)``.  ``s_meas`` [-] (body axes) is
    ``I[+x] - I[-x]`` etc. and is **not** normalised; noise-free and lit it equals the
    true unit vector exactly.  ``valid = sum(currents) > cfg.threshold``.
    """
    c = _f(currents)
    s_meas = c[..., 0::2] - c[..., 1::2]
    valid = c.sum(axis=-1) > cfg.threshold
    return s_meas, valid


class Gyro:
    """Stateful rate-integrating gyro with bias random walk and angle random walk.

    Parameters
    ----------
    cfg : GyroConfig
    rng : numpy Generator, optional
        Noise source; ``None`` creates an unseeded generator.
    bias0 : (3,) array, optional
        Turn-on bias [rad/s].  ``None`` draws ``N(0, bias0_sigma^2 I)`` from ``rng``.
    """

    def __init__(self, cfg: GyroConfig = GyroConfig(), rng: np.random.Generator | None = None, bias0=None):
        self.cfg = cfg
        self.rng = np.random.default_rng() if rng is None else rng
        if bias0 is None:
            self._bias = self.rng.normal(0.0, cfg.bias0_sigma, size=3)
        else:
            self._bias = np.array(bias0, dtype=float).reshape(3)

    @property
    def bias(self) -> np.ndarray:
        """Current true gyro bias (3,) [rad/s], body axes (copy)."""
        return self._bias.copy()

    def sample(self, w_mean, dt: float) -> np.ndarray:
        """Gyro reading (3,) [rad/s] for a period ``dt`` [s] with mean body rate ``w_mean`` [rad/s].

        First steps the bias ``b += rrw sqrt(dt) N(0, I)``, then returns
        ``w_mean + b + (arw / sqrt(dt)) N(0, I)``.
        """
        sdt = np.sqrt(dt)
        self._bias = self._bias + self.cfg.rrw * sdt * self.rng.standard_normal(3)
        return _f(w_mean) + self._bias + (self.cfg.arw / sdt) * self.rng.standard_normal(3)


def mean_rates(q, dt: float) -> np.ndarray:
    """Noise-free mean body rate over each sample interval.

    ``q (N, 4)`` body-to-ECI attitudes at ``t_k = k dt``; returns ``(N, 3)`` [rad/s],
    body axes: row 0 NaN, row k = ``to_rotvec(q_{k-1}* (x) q_k) / dt``.
    """
    q = _f(q)
    out = np.full((q.shape[0], 3), np.nan)
    if q.shape[0] > 1:
        dq = quat.multiply(quat.conjugate(q[:-1]), q[1:])
        out[1:] = quat.to_rotvec(dq) / dt
    return out


@dataclass
class SensorData:
    """Sensor series on a uniform grid ``t (N,)`` [s]; time on axis 0.

    ``gyro`` (N, 3) rad/s (row 0 NaN); ``gyro_bias`` (N, 3) rad/s true bias used in
    sample k (row 0 = turn-on bias); ``mag`` (N, 3) T body axes; ``css_currents``
    (N, 6); ``sun`` (N, 3) reconstructed sun vector (body axes); ``sun_valid`` (N,)
    bool; ``lit`` (N,) bool; ``meta`` configs as dicts and the seed.
    """

    t: np.ndarray
    gyro: np.ndarray
    gyro_bias: np.ndarray
    mag: np.ndarray
    css_currents: np.ndarray
    sun: np.ndarray
    sun_valid: np.ndarray
    lit: np.ndarray
    meta: dict = field(default_factory=dict)


def measure_series(
    t,
    q,
    b_eci,
    s_eci,
    lit,
    seed=None,
    mag: MagnetometerConfig = MagnetometerConfig(),
    css: SunSensorConfig = SunSensorConfig(),
    gyro: GyroConfig = GyroConfig(),
    bias0=None,
) -> SensorData:
    """Simulate all sensors along a truth trajectory (sampling model in the module docstring).

    Parameters
    ----------
    t : (N,) s, uniform grid (``ValueError`` if ``|diff(t) - dt| > 1e-9`` anywhere).
    q : (N, 4) true body-to-ECI quaternions.
    b_eci : (N, 3) magnetic field in ECI [T].
    s_eci : (N, 3) unit sun direction in ECI.
    lit : (N,) bool, True when the sun is visible.
    seed : int or None, root of ``default_rng(seed).spawn(3)`` -> (gyro, magnetometer,
        sun sensor) streams, so one sensor's config does not change another's noise.
    bias0 : (3,) optional gyro turn-on bias [rad/s].
    """
    t = _f(t)
    q = _f(q)
    b_eci = _f(b_eci)
    s_eci = _f(s_eci)
    lit = np.asarray(lit, dtype=bool)
    n = t.shape[0]
    if n < 2:
        raise ValueError("need at least two samples")
    dt = float(t[1] - t[0])
    if dt <= 0 or np.any(np.abs(np.diff(t) - dt) > 1e-9):
        raise ValueError("time grid must be uniform and increasing (tolerance 1e-9 s)")
    if not (q.shape == (n, 4) and b_eci.shape == (n, 3) and s_eci.shape == (n, 3) and lit.shape == (n,)):
        raise ValueError("q, b_eci, s_eci, lit must have N rows matching t")

    rng_g, rng_m, rng_s = np.random.default_rng(seed).spawn(3)

    # Gyro
    g = Gyro(gyro, rng_g, bias0)
    w_mean = mean_rates(q, dt)
    gyro_out = np.full((n, 3), np.nan)
    bias_out = np.empty((n, 3))
    bias_out[0] = g.bias
    for k in range(1, n):
        gyro_out[k] = g.sample(w_mean[k], dt)
        bias_out[k] = g.bias

    # Magnetometer and sun sensor (body-frame truth R(q)^T v_eci)
    b_body = quat.rotate_inverse(q, b_eci)
    s_body = quat.rotate_inverse(q, s_eci)
    mag_out = magnetometer(b_body, rng_m, mag)
    currents = sun_sensor_currents(s_body, lit, rng_s, css)
    sun, valid = sun_vector_from_currents(currents, css)

    meta = {
        "seed": seed,
        "dt": dt,
        "mag": asdict(mag),
        "css": asdict(css),
        "gyro": asdict(gyro),
    }
    return SensorData(
        t=t.copy(),
        gyro=gyro_out,
        gyro_bias=bias_out,
        mag=mag_out,
        css_currents=currents,
        sun=sun,
        sun_valid=valid,
        lit=lit.copy(),
        meta=meta,
    )
