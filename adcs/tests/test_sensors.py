"""Tests for ``adcs.sensors`` (magnetometer, coarse sun sensor, gyro, series generator).

Written from ``docs/specs/adcs_estimation.md`` section 0 and section 1 (tests 1-6), not
from the implementation.  Oracles: closed-form statistics (Gaussian / Rayleigh moments,
random-walk variance), the datasheet arithmetic quoted in the spec, explicit cosine-law
component formulas, and ``scipy.spatial.transform.Rotation`` for all rotation algebra
(scipy is scalar-last ``[x, y, z, w]``; the repo is scalar-first ``[w, x, y, z]``).

Statistical thresholds: the spec values are used as written.  Where a spec band is too
tight for the sample size it is noted at the test (only the RRW variance check is
affected: +-7 % is 2.2 sigma for 2000 values of one axis, so the three axes of the 2000
gyros are pooled, giving 6000 values and a 3.8 sigma band; the per-axis check uses 13 %).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from adcs import sensors
from adcs.sensors import (
    CSS_NORMALS,
    Gyro,
    GyroConfig,
    MagnetometerConfig,
    SensorData,
    SunSensorConfig,
    magnetometer,
    mean_rates,
    measure_series,
    sun_sensor_currents,
    sun_vector_from_currents,
)


# --------------------------------------------------------------------------- helpers
def _to_sf(rot: Rotation) -> np.ndarray:
    """scipy Rotation -> scalar-first quaternion with w >= 0."""
    q = np.roll(rot.as_quat(), 1, axis=-1)
    return np.where((q[..., :1] < 0), -q, q)


def _to_rot(q: np.ndarray) -> Rotation:
    """scalar-first quaternion -> scipy Rotation."""
    return Rotation.from_quat(np.asarray(q)[..., [1, 2, 3, 0]])


def _random_unit(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _scipy_mean_rates(rot: Rotation, dt: float) -> np.ndarray:
    """Oracle: rotvec(R_{k-1}^T R_k)/dt for k >= 1 (scipy, no repo code)."""
    return (rot[:-1].inv() * rot[1:]).as_rotvec() / dt


def _series(rng, n=200, dt=0.5):
    """Smooth random attitude history + fields; returns (t, q, rot, b_eci, s_eci, lit)."""
    rv = np.cumsum(rng.normal(0.0, 0.02, size=(n, 3)), axis=0)
    rot = Rotation.from_rotvec(rv)
    q = _to_sf(rot)
    t = dt * np.arange(n)
    b = 3e-5 * rng.normal(size=(n, 3))
    s = _random_unit(rng, n)
    lit = rng.random(n) < 0.7
    return t, q, rot, b, s, lit


NOISE_OFF = dict(
    mag=MagnetometerConfig(sigma=0.0),
    css=SunSensorConfig(sigma_diode=0.0),
    gyro=GyroConfig(arw=0.0, rrw=0.0, bias0_sigma=0.0),
)


# --------------------------------------------------------------------------- configs
class TestConfigs:
    def test_css_normals(self):
        expected = np.array(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float
        )
        assert np.asarray(CSS_NORMALS).shape == (6, 3)
        np.testing.assert_array_equal(CSS_NORMALS, expected)

    def test_defaults_match_datasheet_arithmetic(self):
        assert MagnetometerConfig().sigma == 15e-9
        # ADXRS453 ARW 0.65 deg/sqrt(h) -> rad/sqrt(s): /60 converts 1/sqrt(h) to 1/sqrt(s)
        arw = 0.65 * math.pi / 180.0 / 60.0
        # bias drift over one orbit equals 16 deg/h in-run stability
        rrw = 16.0 * math.pi / 180.0 / 3600.0 / math.sqrt(5553.6)
        g = GyroConfig()
        assert g.arw == pytest.approx(arw, rel=1e-4)
        assert g.rrw == pytest.approx(rrw, rel=1e-3)
        assert g.bias0_sigma == pytest.approx(0.1 * math.pi / 180.0, rel=1e-4)
        s = SunSensorConfig()
        assert s.sigma_diode == 0.0436
        assert s.threshold == 0.6
        assert s.sigma_vector == pytest.approx(math.sqrt(2.0) * 0.0436, rel=1e-12)
        # 1-sigma total angle error sqrt(2)*sigma_vector = 5.0 deg (CSS-10 datasheet "< 5 deg")
        assert math.degrees(math.sqrt(2.0) * s.sigma_vector) == pytest.approx(5.0, abs=0.05)

    def test_configs_are_frozen(self):
        for cfg, field in [
            (MagnetometerConfig(), "sigma"),
            (SunSensorConfig(), "sigma_diode"),
            (GyroConfig(), "arw"),
        ]:
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(cfg, field, 1.0)

    def test_sigma_vector_follows_sigma_diode(self):
        assert SunSensorConfig(sigma_diode=0.01).sigma_vector == pytest.approx(
            math.sqrt(2.0) * 0.01, rel=1e-12
        )


# --------------------------------------------------------------------------- 1. magnetometer
class TestMagnetometer:
    def test_noise_statistics(self):
        n = 20_000
        sigma = 15e-9
        b = np.array([2.0e-5, -3.0e-5, 4.0e-5])
        rng = np.random.default_rng(101)
        out = magnetometer(np.tile(b, (n, 1)), rng)
        assert out.shape == (n, 3)
        err = out - b
        # std of the sample std is sigma/sqrt(2N) = 0.5 %, so 3 % is a 6-sigma band
        np.testing.assert_allclose(err.std(axis=0, ddof=1), sigma, rtol=0.03)
        assert np.all(np.abs(err.mean(axis=0)) < 4.0 * sigma / math.sqrt(n))
        corr = np.corrcoef(err.T)
        off = corr[~np.eye(3, dtype=bool)]
        assert np.all(np.abs(off) < 0.03)

    def test_custom_sigma_and_default_argument(self):
        n = 20_000
        rng = np.random.default_rng(102)
        cfg = MagnetometerConfig(sigma=1e-6)
        out = magnetometer(np.zeros((n, 3)), rng, cfg)
        np.testing.assert_allclose(out.std(axis=0, ddof=1), 1e-6, rtol=0.03)

    def test_zero_sigma_is_identity(self):
        b = np.array([[1e-5, 2e-5, -3e-5], [0.0, 1e-5, 0.0]])
        out = magnetometer(b, np.random.default_rng(0), MagnetometerConfig(sigma=0.0))
        np.testing.assert_array_equal(out, b)

    def test_shapes_no_mutation_and_reproducibility(self):
        b = np.full((4, 5, 3), 3e-5)
        b0 = b.copy()
        a = magnetometer(b, np.random.default_rng(7))
        a2 = magnetometer(b, np.random.default_rng(7))
        c = magnetometer(b, np.random.default_rng(8))
        assert a.shape == (4, 5, 3)
        np.testing.assert_array_equal(b, b0)
        np.testing.assert_array_equal(a, a2)
        assert not np.array_equal(a, c)
        assert magnetometer(np.array([1e-5, 0.0, 0.0]), np.random.default_rng(1)).shape == (3,)
        # list input accepted (array-like in, float array out)
        out = magnetometer([1e-5, 0.0, 0.0], np.random.default_rng(1))
        assert isinstance(out, np.ndarray) and out.dtype == float


# --------------------------------------------------------------------------- 2. CSS noise-free
class TestCssNoiseFree:
    cfg0 = SunSensorConfig(sigma_diode=0.0)

    def test_exact_reconstruction_and_cosine_law(self):
        rng = np.random.default_rng(201)
        s = _random_unit(rng, 1000)
        lit = np.ones(1000, dtype=bool)
        cur = sun_sensor_currents(s, lit, rng, self.cfg0)
        assert cur.shape == (1000, 6)
        s_meas, valid = sun_vector_from_currents(cur, self.cfg0)
        assert s_meas.shape == (1000, 3) and valid.shape == (1000,)
        np.testing.assert_allclose(s_meas, s, rtol=0, atol=1e-15)
        assert valid.all()
        assert (cur >= 0.0).all()
        # explicit cosine law, face order +x -x +y -y +z -z
        expected = np.stack(
            [
                np.maximum(0, s[:, 0]),
                np.maximum(0, -s[:, 0]),
                np.maximum(0, s[:, 1]),
                np.maximum(0, -s[:, 1]),
                np.maximum(0, s[:, 2]),
                np.maximum(0, -s[:, 2]),
            ],
            axis=1,
        )
        np.testing.assert_allclose(cur, expected, rtol=0, atol=1e-15)
        # for each axis exactly one of the two opposite faces is illuminated
        assert np.all((cur[:, 0::2] == 0.0) | (cur[:, 1::2] == 0.0))
        # three faces lit, three dark (generic vector)
        assert np.all((cur > 0).sum(axis=1) == 3)

    def test_axis_aligned_sun(self):
        for k, (axis, sign) in enumerate([(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]):
            s = np.zeros(3)
            s[axis] = sign
            cur = sun_sensor_currents(s, np.True_, np.random.default_rng(0), self.cfg0)
            expected = np.zeros(6)
            expected[k] = 1.0
            np.testing.assert_allclose(cur, expected, atol=1e-15)

    def test_eclipse_all_zero_and_invalid(self):
        rng = np.random.default_rng(202)
        s = _random_unit(rng, 500)
        cur = sun_sensor_currents(s, np.zeros(500, dtype=bool), rng, self.cfg0)
        np.testing.assert_array_equal(cur, 0.0)
        s_meas, valid = sun_vector_from_currents(cur, self.cfg0)
        assert not valid.any()
        np.testing.assert_array_equal(s_meas, 0.0)

    def test_mixed_lit_mask_and_batch_shape(self):
        rng = np.random.default_rng(203)
        s = _random_unit(rng, 60).reshape(3, 20, 3)
        lit = rng.random((3, 20)) < 0.5
        cur = sun_sensor_currents(s, lit, rng, self.cfg0)
        assert cur.shape == (3, 20, 6)
        s_meas, valid = sun_vector_from_currents(cur, self.cfg0)
        assert s_meas.shape == (3, 20, 3) and valid.shape == (3, 20)
        np.testing.assert_array_equal(valid, lit)
        np.testing.assert_allclose(s_meas[lit], s[lit], rtol=0, atol=1e-15)
        np.testing.assert_array_equal(cur[~lit], 0.0)
        # inputs are not mutated
        assert s.shape == (3, 20, 3)

    def test_validity_threshold_is_sum_greater_than_threshold(self):
        cfg = SunSensorConfig(sigma_diode=0.0, threshold=0.5)
        cur = np.zeros((3, 6))
        cur[0, 0] = 0.49
        cur[1, 0] = 0.51
        cur[2, 0], cur[2, 3] = 0.3, 0.3
        _, valid = sun_vector_from_currents(cur, cfg)
        np.testing.assert_array_equal(valid, [False, True, True])

    def test_s_meas_is_difference_of_opposite_faces_not_normalised(self):
        cur = np.array([0.3, 0.1, 0.0, 0.2, 0.05, 0.0])
        s_meas, _ = sun_vector_from_currents(cur, self.cfg0)
        np.testing.assert_allclose(s_meas, [0.2, -0.2, 0.05], atol=1e-15)


# --------------------------------------------------------------------------- 3. CSS noisy
class TestCssNoisy:
    def test_vector_noise_and_angle_error(self):
        n = 20_000
        cfg = SunSensorConfig()
        rng = np.random.default_rng(301)
        s = _random_unit(rng, n)
        cur = sun_sensor_currents(s, np.ones(n, dtype=bool), rng, cfg)
        s_meas, valid = sun_vector_from_currents(cur, cfg)
        err = s_meas - s
        sig_v = math.sqrt(2.0) * 0.0436
        assert cfg.sigma_vector == pytest.approx(sig_v, rel=1e-12)
        # per-axis std error of the sample std is 0.5 % at N = 20000
        np.testing.assert_allclose(err.std(axis=0, ddof=1), sig_v, rtol=0.03)
        assert np.all(np.abs(err.mean(axis=0)) < 4.0 * sig_v / math.sqrt(n))
        # angle between measured and true direction (atan2 form, accurate for small angles)
        ang = np.arctan2(np.linalg.norm(np.cross(s_meas, s), axis=1), np.einsum("ij,ij->i", s_meas, s))
        # perpendicular error is 2-D Gaussian with per-axis sigma_v -> Rayleigh, mean sigma sqrt(pi/2)
        rayleigh_mean_deg = math.degrees(math.sqrt(math.pi / 2.0) * sig_v)
        assert np.degrees(ang).mean() == pytest.approx(rayleigh_mean_deg, rel=0.03)
        assert valid.all()

    def test_eclipse_never_valid_and_lit_always_valid(self):
        n = 10_000
        cfg = SunSensorConfig()
        rng = np.random.default_rng(302)
        s = _random_unit(rng, n)
        cur_e = sun_sensor_currents(s, np.zeros(n, dtype=bool), rng, cfg)
        # noise only in eclipse: sum of six currents ~ N(0, 6 sigma_d^2), threshold 0.6 = 5.6 sigma
        # (P ~ 1e-8 per sample; seed fixed, deterministic)
        _, valid_e = sun_vector_from_currents(cur_e, cfg)
        assert not valid_e.any()
        np.testing.assert_allclose(cur_e.std(), 0.0436, rtol=0.03)
        assert abs(cur_e.mean()) < 4 * 0.0436 / math.sqrt(cur_e.size)
        # lit: sum of positive parts = |sx|+|sy|+|sz| >= 1 for unit s
        cur_l = sun_sensor_currents(s, np.ones(n, dtype=bool), rng, cfg)
        _, valid_l = sun_vector_from_currents(cur_l, cfg)
        assert valid_l.all()

    def test_noise_present_on_all_six_faces_including_dark_ones(self):
        n = 10_000
        cfg = SunSensorConfig()
        s = np.tile([1.0, 0.0, 0.0], (n, 1))
        cur = sun_sensor_currents(s, np.ones(n, dtype=bool), np.random.default_rng(303), cfg)
        # dark faces (-x, +-y, +-z) carry zero-mean noise with std sigma_d (currents may be < 0)
        dark = cur[:, 1:]
        assert (dark < 0).any()
        np.testing.assert_allclose(dark.std(axis=0, ddof=1), 0.0436, rtol=0.03)
        # lit face: mean 1, std sigma_d
        assert cur[:, 0].mean() == pytest.approx(1.0, abs=4 * 0.0436 / math.sqrt(n))

    def test_reproducible_with_same_rng_seed(self):
        s = _random_unit(np.random.default_rng(0), 50)
        lit = np.ones(50, dtype=bool)
        a = sun_sensor_currents(s, lit, np.random.default_rng(5))
        b = sun_sensor_currents(s, lit, np.random.default_rng(5))
        np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- 4. gyro
class TestGyro:
    def test_zero_noise_returns_mean_plus_bias0_exactly(self):
        cfg = GyroConfig(arw=0.0, rrw=0.0)
        bias0 = np.array([1e-3, -2e-3, 3e-3])
        g = Gyro(cfg, np.random.default_rng(0), bias0=bias0)
        w = np.array([0.01, -0.02, 0.03])
        for dt in (0.1, 1.0, 10.0):
            np.testing.assert_allclose(g.sample(w, dt), w + bias0, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(g.bias, bias0)

    def test_bias_property_is_a_copy_and_bias0_not_mutated(self):
        cfg = GyroConfig(arw=0.0, rrw=1e-3)
        bias0 = np.array([1e-3, -2e-3, 3e-3])
        keep = bias0.copy()
        g = Gyro(cfg, np.random.default_rng(1), bias0=bias0)
        g.sample(np.zeros(3), 1.0)
        np.testing.assert_array_equal(bias0, keep)
        b = g.bias
        b[:] = 99.0
        assert not np.any(g.bias == 99.0)

    @pytest.mark.parametrize("dt", [0.1, 1.0, 10.0])
    def test_arw_only_sample_std(self, dt):
        n = 20_000
        cfg = GyroConfig(arw=1.8908e-4, rrw=0.0, bias0_sigma=0.0)
        g = Gyro(cfg, np.random.default_rng(401), bias0=np.zeros(3))
        w = np.array([0.02, 0.0, -0.01])
        out = np.array([g.sample(w, dt) for _ in range(n)])
        err = out - w
        # 3 % band is 6 sigma for N = 20000 per axis
        np.testing.assert_allclose(err.std(axis=0, ddof=1), cfg.arw / math.sqrt(dt), rtol=0.03)
        assert np.all(np.abs(err.mean(axis=0)) < 4.0 * cfg.arw / math.sqrt(dt) / math.sqrt(n))
        np.testing.assert_array_equal(g.bias, 0.0)  # no bias motion when rrw = 0

    def test_rrw_only_bias_variance(self):
        n_gyro = 2000
        dt = 2.0
        cfg = GyroConfig(arw=0.0, rrw=1.0409e-6)
        rngs = np.random.default_rng(402).spawn(n_gyro)
        gyros = [Gyro(cfg, r, bias0=np.zeros(3)) for r in rngs]
        w = np.zeros(3)
        steps_done = 0
        for n_steps in (5, 20):
            for _ in range(n_steps - steps_done):
                last = np.array([g.sample(w, dt) for g in gyros])
            steps_done = n_steps
            bias = np.array([g.bias for g in gyros])
            np.testing.assert_allclose(last, bias, rtol=0, atol=1e-18)  # arw = 0, w = 0
            expected = cfg.rrw**2 * n_steps * dt
            # pooled over 3 axes: 6000 values -> std error of variance 1.8 %, 7 % = 3.8 sigma
            assert (bias**2).mean() == pytest.approx(expected, rel=0.07)
            # per axis 2000 values -> 3.2 % std error; 13 % = 4 sigma
            np.testing.assert_allclose(bias.var(axis=0, ddof=0), expected, rtol=0.13)
            assert np.all(np.abs(bias.mean(axis=0)) < 4.0 * math.sqrt(expected / n_gyro))

    def test_default_bias0_distribution(self):
        n_gyro = 2000
        cfg = GyroConfig()
        rngs = np.random.default_rng(403).spawn(n_gyro)
        b0 = np.array([Gyro(cfg, r).bias for r in rngs])
        assert b0.shape == (n_gyro, 3)
        # std error of std for 2000 values is 1.6 % -> 7 % is 4.4 sigma
        np.testing.assert_allclose(b0.std(axis=0, ddof=1), cfg.bias0_sigma, rtol=0.07)
        assert np.all(np.abs(b0.mean(axis=0)) < 4.0 * cfg.bias0_sigma / math.sqrt(n_gyro))

    def test_bias_steps_before_sample_and_sample_includes_new_bias(self):
        # arw = 0: sample = w + bias after the step (spec: step first, then return)
        cfg = GyroConfig(arw=0.0, rrw=1e-3)
        g = Gyro(cfg, np.random.default_rng(404), bias0=np.zeros(3))
        out = g.sample(np.zeros(3), 1.0)
        np.testing.assert_allclose(out, g.bias, atol=1e-18)
        assert np.any(out != 0.0)

    def test_reproducible(self):
        cfg = GyroConfig()
        a = Gyro(cfg, np.random.default_rng(9))
        b = Gyro(cfg, np.random.default_rng(9))
        w = np.array([0.1, 0.0, 0.0])
        for _ in range(5):
            np.testing.assert_array_equal(a.sample(w, 1.0), b.sample(w, 1.0))


# --------------------------------------------------------------------------- 5. mean_rates
class TestMeanRates:
    @pytest.mark.parametrize("dt", [0.5, 1.0])
    def test_constant_body_rate(self, dt):
        w = np.array([0.011, -0.023, 0.034])
        n = 60
        # truth by scipy: R_k = R_0 * Exp(w k dt)  (body rate in body axes -> right multiplication)
        r0 = Rotation.from_rotvec([0.3, -0.5, 0.2])
        rot = Rotation.from_rotvec(np.outer(np.arange(n) * dt, w))
        q = _to_sf(Rotation.from_quat(r0.as_quat()) * rot)
        out = mean_rates(q, dt)
        assert out.shape == (n, 3)
        assert np.isnan(out[0]).all()
        np.testing.assert_allclose(out[1:], np.tile(w, (n - 1, 1)), rtol=0, atol=1e-12)

    def test_fast_constant_rate_below_pi_per_step(self):
        w = np.array([0.0, 0.0, 2.9])  # 2.9 rad per 1 s step, below pi
        rot = Rotation.from_rotvec(np.outer(np.arange(10), w))
        out = mean_rates(_to_sf(rot), 1.0)
        np.testing.assert_allclose(out[1:], np.tile(w, (9, 1)), rtol=0, atol=1e-12)

    def test_quaternion_sign_flips_do_not_change_result(self):
        w = np.array([0.02, 0.01, -0.03])
        rot = Rotation.from_rotvec(np.outer(np.arange(30), w))
        q = _to_sf(rot)
        q_flip = q.copy()
        q_flip[1::2] *= -1.0
        np.testing.assert_allclose(mean_rates(q_flip, 1.0)[1:], np.tile(w, (29, 1)), atol=1e-12)

    def test_random_attitude_sequence_roundtrip(self):
        rng = np.random.default_rng(501)
        n = 300
        for dt in (1.0, 0.25):
            rot = Rotation.random(n, rng)
            q = _to_sf(rot)
            wbar = mean_rates(q, dt)
            assert wbar.shape == (n, 3)
            assert np.isnan(wbar[0]).all() and np.isfinite(wbar[1:]).all()
            # oracle 1: q_{k-1} (x) exp(w_k dt) = q_k, composed with scipy
            rec = rot[:-1] * Rotation.from_rotvec(wbar[1:] * dt)
            np.testing.assert_allclose(rec.as_matrix(), rot[1:].as_matrix(), rtol=0, atol=1e-12)
            # oracle 2: direct scipy rotvec of the relative rotation / dt
            np.testing.assert_allclose(wbar[1:], _scipy_mean_rates(rot, dt), rtol=0, atol=1e-12)
            # |w dt| <= pi (principal rotation)
            assert np.all(np.linalg.norm(wbar[1:], axis=1) * dt <= math.pi + 1e-12)

    def test_stationary_attitude_gives_zero_rate(self):
        q = np.tile(_to_sf(Rotation.from_rotvec([0.1, 0.2, 0.3])), (5, 1))
        out = mean_rates(q, 1.0)
        np.testing.assert_allclose(out[1:], 0.0, atol=1e-12)

    def test_input_not_mutated(self):
        q = _to_sf(Rotation.random(10, np.random.default_rng(3)))
        q0 = q.copy()
        mean_rates(q, 1.0)
        np.testing.assert_array_equal(q, q0)


# --------------------------------------------------------------------------- 6. measure_series
class TestMeasureSeries:
    def setup_method(self):
        rng = np.random.default_rng(601)
        self.dt = 0.5
        self.t, self.q, self.rot, self.b, self.s, self.lit = _series(rng, n=200, dt=self.dt)
        self.n = 200

    def _run(self, **kw):
        return measure_series(self.t, self.q, self.b, self.s, self.lit, **kw)

    def test_shapes_and_types(self):
        d = self._run(seed=1)
        n = self.n
        assert isinstance(d, SensorData)
        assert d.t.shape == (n,)
        assert d.gyro.shape == (n, 3)
        assert d.gyro_bias.shape == (n, 3)
        assert d.mag.shape == (n, 3)
        assert d.css_currents.shape == (n, 6)
        assert d.sun.shape == (n, 3)
        assert d.sun_valid.shape == (n,) and d.sun_valid.dtype == bool
        assert d.lit.shape == (n,) and d.lit.dtype == bool
        assert isinstance(d.meta, dict)
        np.testing.assert_array_equal(d.t, self.t)
        np.testing.assert_array_equal(d.lit, self.lit)
        assert np.isnan(d.gyro[0]).all()
        assert np.isfinite(d.gyro[1:]).all()
        assert np.isfinite(d.gyro_bias).all()

    def test_inputs_not_mutated(self):
        copies = [a.copy() for a in (self.t, self.q, self.b, self.s, self.lit)]
        self._run(seed=2)
        for a, c in zip((self.t, self.q, self.b, self.s, self.lit), copies):
            np.testing.assert_array_equal(a, c)

    def test_reproducible_for_equal_seeds_and_differs_otherwise(self):
        a, b, c = self._run(seed=3), self._run(seed=3), self._run(seed=4)
        for name in ("gyro", "gyro_bias", "mag", "css_currents", "sun", "sun_valid"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
        assert not np.array_equal(a.mag, c.mag)
        assert not np.array_equal(a.css_currents, c.css_currents)
        assert not np.array_equal(a.gyro[1:], c.gyro[1:])

    def test_changing_css_config_leaves_gyro_and_mag_identical(self):
        a = self._run(seed=5)
        b = self._run(seed=5, css=SunSensorConfig(sigma_diode=0.1))
        np.testing.assert_array_equal(a.gyro, b.gyro)  # NaN row compares equal
        np.testing.assert_array_equal(a.gyro_bias, b.gyro_bias)
        np.testing.assert_array_equal(a.mag, b.mag)
        assert not np.array_equal(a.css_currents, b.css_currents)

    def test_changing_mag_config_leaves_gyro_and_css_identical(self):
        a = self._run(seed=6)
        b = self._run(seed=6, mag=MagnetometerConfig(sigma=1e-6))
        np.testing.assert_array_equal(a.gyro, b.gyro)
        np.testing.assert_array_equal(a.css_currents, b.css_currents)
        np.testing.assert_array_equal(a.sun, b.sun)
        assert not np.array_equal(a.mag, b.mag)

    def test_changing_gyro_config_leaves_mag_and_css_identical(self):
        a = self._run(seed=7)
        b = self._run(seed=7, gyro=GyroConfig(arw=1e-3, rrw=1e-5, bias0_sigma=1e-2))
        np.testing.assert_array_equal(a.mag, b.mag)
        np.testing.assert_array_equal(a.css_currents, b.css_currents)
        assert not np.array_equal(a.gyro[1:], b.gyro[1:])

    def test_noise_free_matches_scipy_oracle(self):
        bias0 = np.array([1e-3, -2e-3, 3e-3])
        d = self._run(seed=8, bias0=bias0, **NOISE_OFF)
        rot = self.rot
        # body vectors R(q)^T v via scipy
        b_body = rot.inv().apply(self.b)
        s_body = rot.inv().apply(self.s)
        np.testing.assert_allclose(d.mag, b_body, rtol=0, atol=1e-15)
        # gyro: sample - bias = mean rate from the true attitudes
        wbar = _scipy_mean_rates(rot, self.dt)
        np.testing.assert_allclose(d.gyro[1:] - d.gyro_bias[1:], wbar, rtol=0, atol=1e-12)
        np.testing.assert_allclose(d.gyro_bias, np.tile(bias0, (self.n, 1)), rtol=0, atol=1e-15)
        # sun: lit rows -> exact body sun vector, valid; unlit -> zero currents, invalid
        np.testing.assert_array_equal(d.sun_valid, self.lit)
        np.testing.assert_allclose(d.sun[self.lit], s_body[self.lit], rtol=0, atol=1e-14)
        np.testing.assert_array_equal(d.css_currents[~self.lit], 0.0)
        np.testing.assert_allclose(d.css_currents[self.lit].min(), 0.0, atol=0)
        assert (d.css_currents >= 0).all()

    def test_noisy_gyro_statistics_and_bias_walk(self):
        n = 4000
        dt = 0.5
        rng = np.random.default_rng(602)
        t, q, rot, b, s, lit = _series(rng, n=n, dt=dt)
        d = measure_series(t, q, b, s, lit, seed=11)
        cfg = GyroConfig()
        wbar = _scipy_mean_rates(rot, dt)
        white = d.gyro[1:] - d.gyro_bias[1:] - wbar
        # 12000 values -> 0.65 % std error; 5 % band
        np.testing.assert_allclose(white.std(axis=0, ddof=1), cfg.arw / math.sqrt(dt), rtol=0.05)
        steps = np.diff(d.gyro_bias, axis=0)
        np.testing.assert_allclose(steps.std(axis=0, ddof=1), cfg.rrw * math.sqrt(dt), rtol=0.05)
        # magnetometer noise
        merr = d.mag - rot.inv().apply(b)
        np.testing.assert_allclose(merr.std(axis=0, ddof=1), 15e-9, rtol=0.05)

    def test_nonuniform_time_raises(self):
        t = self.t.copy()
        t[50] += 1e-6
        with pytest.raises(ValueError):
            measure_series(t, self.q, self.b, self.s, self.lit, seed=1)
        t2 = self.t.copy()
        t2[-1] += 0.1
        with pytest.raises(ValueError):
            measure_series(t2, self.q, self.b, self.s, self.lit, seed=1)

    def test_tiny_time_jitter_within_tolerance_is_accepted(self):
        t = self.t.copy()
        t[50] += 1e-11  # well inside the 1e-9 s uniformity tolerance
        d = measure_series(t, self.q, self.b, self.s, self.lit, seed=1)
        assert d.gyro.shape == (self.n, 3)
