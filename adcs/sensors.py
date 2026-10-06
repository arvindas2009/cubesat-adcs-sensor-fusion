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
                                                                axis; optional constant body-frame
                                                                ``bias`` (default none); no scale or
                                                                misalignment, no coil/spacecraft field
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
sensor          CSS-10, x6          determination accuracy"     +-x, +-y, +-z), hemispherical response
                                    after calibration; +-60 deg FOV only if ``SunSensorConfig.fov`` is
                                    FOV                         set (default None = full hemisphere);
                                                                albedo only if passed (``albedo=``);
                                                                Earth IR not modelled; per-diode
                                                                white noise sigma_d = 0.0436
                                                                of full-scale current -> reconstructed
                                                                vector per-axis noise sqrt(2) sigma_d
                                                                = 0.0617, total angle error
                                                                sqrt(2)*0.0617 rad = 5.0 deg (1-sigma)
==============  ==================  ==========================  =====================================

Optional realism (Group 8, ``docs/specs/adcs_closed_loop.md`` section 1): magnetometer
``bias``, CSS field-of-view cutoff ``SunSensorConfig.fov``, Earth-albedo currents
(:func:`albedo_currents`, :func:`earth_view_factor`) and the :func:`gauss_markov` series
generator.  All default to off, i.e. the Group 7 behaviour, and none of them changes the
noise draws of the existing sensors.

Randomness only enters through ``numpy.random.Generator`` objects passed in or created
from ``seed`` (no global state).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.signal import lfilter

from common import quaternion as quat
from common.frames import R_EARTH

__all__ = [
    "CSS_NORMALS",
    "MagnetometerConfig",
    "SunSensorConfig",
    "GyroConfig",
    "magnetometer",
    "gauss_markov",
    "sun_sensor_currents",
    "earth_view_factor",
    "albedo_currents",
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
    ``fov``: half-angle of each diode's field of view (rad); ``None`` = full hemisphere.
    """

    sigma_diode: float = 0.0436
    threshold: float = 0.6
    fov: float | None = None

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


def magnetometer(
    b_body, rng: np.random.Generator, cfg: MagnetometerConfig = MagnetometerConfig(), bias=None
) -> np.ndarray:
    """Magnetometer reading ``(..., 3)`` [T] in body axes: ``b_body + bias + N(0, sigma^2 I)``.

    ``b_body (..., 3)`` is the true field in body axes [T].  ``bias`` (3,) [T], body axes,
    broadcast over the leading dimensions; ``None`` = 0.  The noise draw is identical with
    or without ``bias``.  Inputs are not mutated.
    """
    b = _f(b_body)
    noise = rng.normal(0.0, cfg.sigma, size=b.shape)
    if bias is None:
        return b + noise
    return b + _f(bias) + noise


def gauss_markov(n: int, dt: float, sigma: float, tau: float, rng: np.random.Generator, dim: int = 3) -> np.ndarray:
    """Stationary first-order Gauss-Markov series, shape ``(n, dim)``, unit of ``sigma``.

    ``x_0 = sigma w_0``, ``x_k = a x_{k-1} + sigma sqrt(1 - a^2) w_k``, ``a = exp(-dt/tau)``,
    ``w = rng.standard_normal((n, dim))`` drawn in one call (row k <-> ``w_k``).  ``dt``
    sample period [s], ``tau`` correlation time [s].  ``sigma == 0`` returns zeros without
    drawing.  ``ValueError`` if ``n < 1``, ``dt <= 0``, ``tau <= 0`` or ``sigma < 0``.
    """
    n = int(n)
    if n < 1:
        raise ValueError("gauss_markov: n must be >= 1")
    if not dt > 0.0:
        raise ValueError("gauss_markov: dt must be > 0")
    if not tau > 0.0:
        raise ValueError("gauss_markov: tau must be > 0")
    if not sigma >= 0.0:
        raise ValueError("gauss_markov: sigma must be >= 0")
    if sigma == 0.0:
        return np.zeros((n, dim))
    a = float(np.exp(-dt / tau))
    w = rng.standard_normal((n, dim))
    u = (sigma * np.sqrt(1.0 - a * a)) * w
    u[0] = sigma * w[0]
    return lfilter([1.0], [1.0, -a], u, axis=0)


def sun_sensor_currents(
    s_body, lit, rng: np.random.Generator, cfg: SunSensorConfig = SunSensorConfig(), albedo=None
) -> np.ndarray:
    """Six photodiode currents ``(..., 6)`` [fraction of full scale], order of ``CSS_NORMALS``.

    ``s_body (..., 3)`` unit sun direction in body axes, ``lit (...)`` bool (False in
    eclipse).  With ``c_i = n_i . s_body`` the direct term is ``where(lit, max(0, c_i), 0)``
    (``cfg.fov is None``) or ``where(lit, c_i if c_i > cos(fov) else 0, 0)`` (hard field-of-view
    cutoff; for ``fov > pi/2`` the threshold is limited to 0, i.e. never beyond the
    hemisphere).  ``albedo (..., 6)`` [fraction of full scale] (``None`` = 0) is then added
    regardless of ``lit`` and ``fov``, and ``N(0, sigma_d^2)`` on all six diodes (also in
    eclipse; noisy currents may be slightly negative).  The noise draw does not depend on
    ``albedo`` or ``fov``.
    """
    s = _f(s_body)
    lit = np.asarray(lit, dtype=bool)
    cos_inc = s @ CSS_NORMALS.T
    if cfg.fov is None:
        clean = np.where(lit[..., None], np.maximum(cos_inc, 0.0), 0.0)
    else:
        cos_fov = max(float(np.cos(cfg.fov)), 0.0)
        clean = np.where(lit[..., None] & (cos_inc > cos_fov), cos_inc, 0.0)
    if albedo is not None:
        clean = clean + _f(albedo)
    return clean + rng.normal(0.0, cfg.sigma_diode, size=clean.shape)


def earth_view_factor(phi, r_norm, re: float = R_EARTH) -> np.ndarray:
    """View factor ``F`` from a differential plate to a sphere (Earth), dimensionless.

    ``phi`` [rad]: angle between the plate normal and the direction from the plate to the
    sphere centre; ``r_norm`` [m]: distance of the plate from the sphere centre; ``re`` [m]:
    sphere radius.  Broadcasting over ``phi`` and ``r_norm``.  With ``H = r/re``,
    ``rho = asin(1/H)``, ``x = sqrt(H^2 - 1)``: ``F = cos(phi)/H^2`` for
    ``phi <= pi/2 - rho``; ``F = 1/2 - asin(x/(H sin phi))/pi + (cos(phi) acos(-x cot phi)
    - x sqrt(1 - H^2 cos^2 phi))/(pi H^2)`` for ``pi/2 - rho < phi < pi/2 + rho``; 0 beyond
    (standard "differential planar element to sphere" configuration factor, Howell catalog;
    entry unverified).  ``ValueError`` if any ``r_norm <= re``.
    """
    phi_a, r_a = np.broadcast_arrays(_f(phi), _f(r_norm))
    if np.any(r_a <= re):
        raise ValueError("earth_view_factor: r_norm must be > re")
    shape = phi_a.shape
    ph = phi_a.reshape(-1)
    h = r_a.reshape(-1) / re
    rho = np.arcsin(1.0 / h)
    out = np.zeros(ph.shape)

    full = ph <= 0.5 * np.pi - rho
    out[full] = np.cos(ph[full]) / h[full] ** 2

    part = (ph > 0.5 * np.pi - rho) & (ph < 0.5 * np.pi + rho)
    if np.any(part):
        p, hh, rh = ph[part], h[part], rho[part]
        x = np.sqrt(hh * hh - 1.0)
        cp = np.cos(p)
        # w = sqrt(1 - H^2 cos^2 phi) = sqrt((1 - u)(1 + u)), u = H cos(phi), evaluated without
        # cancellation near both ends of the interval (d1, d2 = distance to either end, rad):
        # 1 - u = 2 sin^2(d1/2) + x sin d1, 1 + u = 2 sin^2(d2/2) + x sin d2.  The same w gives
        # asin(x/(H sin phi)) = atan2(x, w) since H^2 sin^2 phi - x^2 = w^2.
        d1 = p - (0.5 * np.pi - rh)
        d2 = (0.5 * np.pi + rh) - p
        w = np.sqrt((2.0 * np.sin(0.5 * d1) ** 2 + x * np.sin(d1))
                    * (2.0 * np.sin(0.5 * d2) ** 2 + x * np.sin(d2)))
        a1 = np.arctan2(x, w)
        a2 = np.arctan2(w, -x * cp)          # = acos(-x cot phi), since 1 - x^2 cot^2 phi = w^2/sin^2 phi
        out[part] = 0.5 - a1 / np.pi + (cp * a2 - x * w) / (np.pi * hh * hh)
    return out.reshape(shape)


def albedo_currents(q, r_eci, s_eci, albedo, re: float = R_EARTH) -> np.ndarray:
    """Extra CSS currents ``(..., 6)`` [fraction of full scale] from Earth albedo.

    ``q (..., 4)`` body-to-ECI attitude, ``r_eci (..., 3)`` spacecraft position [m, ECI],
    ``s_eci (..., 3)`` unit Earth-to-Sun vector (ECI), ``albedo`` scalar or ``(...)``
    albedo coefficient [-], ``re`` Earth radius [m].  Order of ``CSS_NORMALS``.

    ``nadir_b = R(q)^T (-r_hat)`` (body axes), ``phi_i = acos(n_i . nadir_b)``,
    ``I_i = albedo F(phi_i, |r|) max(0, s_hat . r_hat)``.  Assumptions: diode full scale =
    one solar constant at normal incidence; Lambertian Earth with uniform albedo; the whole
    visible cap is lit as the cosine of the solar zenith angle at the sub-satellite point
    (so zero whenever ``s_hat . r_hat <= 0``, including eclipse); diode cosine response over
    the full view factor (no field-of-view limit on albedo).
    """
    q = _f(q)
    r = _f(r_eci)
    s = _f(s_eci)
    rn = np.linalg.norm(r, axis=-1)
    r_hat = r / rn[..., None]
    nadir_b = quat.rotate_inverse(q, -r_hat)
    phi = np.arccos(np.clip(nadir_b @ CSS_NORMALS.T, -1.0, 1.0))
    illum = np.maximum(np.sum(s * r_hat, axis=-1), 0.0)
    f = earth_view_factor(phi, rn[..., None], re)
    return _f(albedo)[..., None] * f * illum[..., None]


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
