"""Tests for ``adcs.actuators`` (magnetorquers and reaction wheels).

Written from ``docs/specs/adcs_control.md`` sections 1 and 4 (tests 1-5), not from
the implementation.  Oracles: hand arithmetic (explicit component formulas), the
datasheet numbers quoted in the spec, and property checks that restate the spec
text for the wheel-torque limiter.  Spec thresholds are used as written.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from adcs.actuators import (
    RPM,
    MagnetorquerConfig,
    WheelConfig,
    limit_wheel_torque,
    magnetorquer_torque,
    saturate_dipole,
    wheel_momentum,
    wheel_speed,
)

# Hand values (spec test 1): 8000 rpm in rad/s, derived spin inertia.
MAX_SPEED_HAND = 8000.0 * 2.0 * math.pi / 60.0          # 837.758 rad/s
JS_HAND = 1.77e-3 / MAX_SPEED_HAND                      # 2.1128e-6 kg m^2


# --------------------------------------------------------------------------
# 1. configs
# --------------------------------------------------------------------------
class TestConfigs:
    def test_rpm_constant(self):
        assert RPM == pytest.approx(2.0 * math.pi / 60.0, rel=1e-15)

    def test_wheel_defaults_hand_values(self):
        cfg = WheelConfig()
        assert cfg.max_torque == pytest.approx(0.23e-3, rel=1e-15)
        assert cfg.rated_momentum == pytest.approx(1.77e-3, rel=1e-15)
        assert cfg.rated_speed_rpm == pytest.approx(8000.0, rel=1e-15)
        assert cfg.max_speed == pytest.approx(837.758, rel=1e-4)
        assert cfg.max_speed == pytest.approx(MAX_SPEED_HAND, rel=1e-12)
        assert cfg.spin_inertia == pytest.approx(2.1128e-6, rel=1e-4)
        assert cfg.spin_inertia == pytest.approx(JS_HAND, rel=1e-12)
        assert cfg.h_max == pytest.approx(1.77e-3, rel=1e-15)
        # consistency: J_s * max_speed is the rated momentum
        assert cfg.spin_inertia * cfg.max_speed == pytest.approx(cfg.h_max, rel=1e-12)

    def test_wheel_derived_properties_follow_custom_values(self):
        cfg = WheelConfig(max_torque=1e-3, rated_momentum=5e-3, rated_speed_rpm=6000.0)
        ms = 6000.0 * 2.0 * math.pi / 60.0
        assert cfg.max_speed == pytest.approx(ms, rel=1e-12)
        assert cfg.spin_inertia == pytest.approx(5e-3 / ms, rel=1e-12)
        assert cfg.h_max == pytest.approx(5e-3, rel=1e-15)

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(max_torque=0.0),
            dict(max_torque=-1e-4),
            dict(rated_momentum=0.0),
            dict(rated_momentum=-1.0),
            dict(rated_speed_rpm=0.0),
            dict(rated_speed_rpm=-8000.0),
        ],
    )
    def test_wheel_invalid_raises(self, kwargs):
        with pytest.raises(ValueError):
            WheelConfig(**kwargs)

    def test_magnetorquer_defaults(self):
        cfg = MagnetorquerConfig()
        assert cfg.m_max == pytest.approx(0.2, rel=1e-15)
        assert cfg.control_period == pytest.approx(1.0, rel=1e-15)
        assert cfg.duty == pytest.approx(0.8, rel=1e-15)
        assert cfg.on_time == pytest.approx(0.8, rel=1e-15)

    def test_magnetorquer_on_time_scales_with_period_and_duty(self):
        cfg = MagnetorquerConfig(control_period=2.5, duty=0.4)
        assert cfg.on_time == pytest.approx(1.0, rel=1e-15)
        assert MagnetorquerConfig(duty=1.0).on_time == pytest.approx(1.0, rel=1e-15)

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(duty=0.0),
            dict(duty=1.0000001),
            dict(duty=2.0),
            dict(duty=-0.5),
            dict(m_max=0.0),
            dict(m_max=-0.2),
            dict(control_period=0.0),
            dict(control_period=-1.0),
        ],
    )
    def test_magnetorquer_invalid_raises(self, kwargs):
        with pytest.raises(ValueError):
            MagnetorquerConfig(**kwargs)

    def test_configs_are_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            MagnetorquerConfig().m_max = 1.0  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            WheelConfig().max_torque = 1.0  # type: ignore[misc]


# --------------------------------------------------------------------------
# 2. saturate_dipole
# --------------------------------------------------------------------------
class TestSaturateDipole:
    @pytest.mark.parametrize("mode", ["scale", "clip"])
    def test_within_limits_unchanged(self, mode):
        m = np.array([0.1, -0.2, 0.05])               # |m_i| <= 0.2 (boundary included)
        out = saturate_dipole(m, 0.2, mode)
        np.testing.assert_array_equal(out, m)

    def test_scale_hand_value(self):
        # max |m_i| = 0.8 -> factor 0.2 / 0.8 = 0.25
        out = saturate_dipole([0.4, -0.8, 0.2], 0.2, "scale")
        np.testing.assert_allclose(out, [0.1, -0.2, 0.05], rtol=1e-14, atol=0)

    def test_scale_is_default_mode(self):
        m = [0.4, -0.8, 0.2]
        np.testing.assert_array_equal(saturate_dipole(m, 0.2), saturate_dipole(m, 0.2, "scale"))

    def test_scale_preserves_direction_and_limits_peak(self):
        rng = np.random.default_rng(101)
        for _ in range(200):
            m = rng.normal(size=3) * rng.uniform(0.05, 5.0)
            m_max = rng.uniform(0.05, 0.5)
            out = saturate_dipole(m, m_max, "scale")
            if np.max(np.abs(m)) > m_max:
                assert np.max(np.abs(out)) == pytest.approx(m_max, rel=1e-14)
                # direction preserved: out x m = 0 (relative to the product of norms)
                cross = np.cross(out, m)
                assert np.linalg.norm(cross) <= 1e-14 * np.linalg.norm(out) * np.linalg.norm(m)
                assert np.dot(out, m) > 0
            else:
                np.testing.assert_array_equal(out, m)

    def test_clip_hand_value_changes_direction(self):
        out = saturate_dipole([0.4, -0.8, 0.1], 0.2, "clip")
        np.testing.assert_allclose(out, [0.2, -0.2, 0.1], rtol=0, atol=1e-15)

    def test_clip_per_component_random(self):
        rng = np.random.default_rng(102)
        m = rng.normal(size=(300, 3))
        out = saturate_dipole(m, 0.3, "clip")
        np.testing.assert_allclose(out, np.clip(m, -0.3, 0.3), rtol=0, atol=1e-15)

    def test_batched_rows_handled_independently(self):
        rng = np.random.default_rng(103)
        m = rng.normal(size=(50, 3)) * rng.uniform(0.01, 3.0, size=(50, 1))
        m_max = 0.2
        out = saturate_dipole(m, m_max, "scale")
        assert out.shape == (50, 3)
        peak = np.max(np.abs(m), axis=1)
        factor = np.minimum(1.0, m_max / peak)           # explicit per-row oracle
        np.testing.assert_allclose(out, m * factor[:, None], rtol=1e-14, atol=0)
        assert np.all(np.max(np.abs(out), axis=1) <= m_max * (1 + 1e-14))
        # clip, batched
        out_c = saturate_dipole(m, m_max, "clip")
        np.testing.assert_allclose(out_c, np.clip(m, -m_max, m_max), rtol=0, atol=1e-15)

    def test_zero_vector_stays_zero(self):
        np.testing.assert_array_equal(saturate_dipole(np.zeros(3), 0.2, "scale"), np.zeros(3))
        np.testing.assert_array_equal(saturate_dipole(np.zeros((4, 3)), 0.2, "clip"), np.zeros((4, 3)))

    def test_input_not_mutated(self):
        m = np.array([1.0, -2.0, 3.0])
        m0 = m.copy()
        saturate_dipole(m, 0.2, "scale")
        saturate_dipole(m, 0.2, "clip")
        np.testing.assert_array_equal(m, m0)

    @pytest.mark.parametrize("mode", ["bogus", "", "SCALE", "limit"])
    def test_bad_mode_raises(self, mode):
        with pytest.raises(ValueError):
            saturate_dipole([0.1, 0.2, 0.3], 0.2, mode)


# --------------------------------------------------------------------------
# 3. magnetorquer_torque
# --------------------------------------------------------------------------
class TestMagnetorquerTorque:
    def test_hand_cross_product(self):
        # m = [1, 2, 3] A m^2, B = [4, 5, 6] e-5 T
        # m x B = [2*6-3*5, 3*4-1*6, 1*5-2*4] e-5 = [-3, 6, -3] e-5
        tau = magnetorquer_torque([1.0, 2.0, 3.0], [4e-5, 5e-5, 6e-5])
        np.testing.assert_allclose(tau, [-3e-5, 6e-5, -3e-5], rtol=1e-14, atol=0)

    def test_axis_pairs(self):
        # x_hat x y_hat = z_hat etc. (right-handed)
        B = 3e-5
        np.testing.assert_allclose(magnetorquer_torque([0.2, 0, 0], [0, B, 0]), [0, 0, 0.2 * B], atol=1e-20)
        np.testing.assert_allclose(magnetorquer_torque([0, 0.2, 0], [B, 0, 0]), [0, 0, -0.2 * B], atol=1e-20)
        np.testing.assert_allclose(magnetorquer_torque([0, 0, 0.2], [0, B, 0]), [-0.2 * B, 0, 0], atol=1e-20)

    def test_perpendicular_to_both_and_antisymmetric(self):
        rng = np.random.default_rng(104)
        for _ in range(100):
            m = rng.uniform(-0.2, 0.2, 3)
            b = rng.uniform(-6e-5, 6e-5, 3)
            tau = magnetorquer_torque(m, b)
            scale = np.linalg.norm(m) * np.linalg.norm(b)
            assert abs(tau @ m) <= 1e-15 * scale * np.linalg.norm(m)
            assert abs(tau @ b) <= 1e-15 * scale * np.linalg.norm(b)
            np.testing.assert_allclose(magnetorquer_torque(b, m), -tau, rtol=1e-14, atol=1e-30)

    def test_parallel_gives_zero(self):
        b = np.array([1e-5, -2e-5, 3e-5])
        np.testing.assert_allclose(magnetorquer_torque(0.5 * b / 1e-5, b), np.zeros(3), atol=1e-20)

    def test_batch_and_broadcast(self):
        rng = np.random.default_rng(105)
        m = rng.uniform(-0.2, 0.2, (20, 3))
        b = rng.uniform(-5e-5, 5e-5, (20, 3))
        tau = magnetorquer_torque(m, b)
        assert tau.shape == (20, 3)
        # explicit component formula
        exp = np.stack([m[:, 1] * b[:, 2] - m[:, 2] * b[:, 1],
                        m[:, 2] * b[:, 0] - m[:, 0] * b[:, 2],
                        m[:, 0] * b[:, 1] - m[:, 1] * b[:, 0]], axis=1)
        np.testing.assert_allclose(tau, exp, rtol=1e-14, atol=1e-30)
        # one field vector broadcast against N dipoles
        b0 = np.array([2e-5, -1e-5, 4e-5])
        tau0 = magnetorquer_torque(m, b0)
        assert tau0.shape == (20, 3)
        np.testing.assert_allclose(tau0[7], magnetorquer_torque(m[7], b0), rtol=1e-14, atol=1e-30)


# --------------------------------------------------------------------------
# 4. wheel_speed / wheel_momentum
# --------------------------------------------------------------------------
class TestWheelSpeedMomentum:
    def test_hand_values(self):
        h = np.array([1.0e-3, -5.0e-4, 0.0])
        w = np.array([0.01, -0.02, 0.03])
        om = wheel_speed(h, w)
        exp = np.array([1.0e-3 / JS_HAND - 0.01, -5.0e-4 / JS_HAND + 0.02, 0.0 - 0.03])
        np.testing.assert_allclose(om, exp, rtol=1e-12, atol=1e-9)

    def test_rated_momentum_gives_rated_speed_for_zero_body_rate(self):
        om = wheel_speed([1.77e-3, -1.77e-3, 0.0], np.zeros(3))
        np.testing.assert_allclose(om, [8000.0 * RPM, -8000.0 * RPM, 0.0], rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(om[:2] / RPM, [8000.0, -8000.0], rtol=1e-12)

    def test_momentum_hand_value(self):
        speed = np.array([100.0, -200.0, 0.0])
        w = np.array([0.1, 0.2, -0.3])
        h = wheel_momentum(speed, w)
        np.testing.assert_allclose(h, JS_HAND * (speed + w), rtol=1e-12, atol=0)

    def test_round_trip_both_directions(self):
        rng = np.random.default_rng(106)
        h = rng.uniform(-1.8e-3, 1.8e-3, (100, 3))
        w = rng.uniform(-0.2, 0.2, (100, 3))
        h_back = wheel_momentum(wheel_speed(h, w), w)
        np.testing.assert_allclose(h_back, h, rtol=1e-12, atol=1e-17)
        speed = rng.uniform(-900.0, 900.0, (100, 3))
        sp_back = wheel_speed(wheel_momentum(speed, w), w)
        np.testing.assert_allclose(sp_back, speed, rtol=1e-12, atol=1e-9)

    def test_body_rate_shifts_speed_by_minus_w(self):
        h = np.array([3e-4, 1e-4, -2e-4])
        w = np.array([0.05, -0.1, 0.2])
        np.testing.assert_allclose(wheel_speed(h, np.zeros(3)) - wheel_speed(h, w), w, rtol=0, atol=1e-9)

    def test_custom_config_used(self):
        cfg = WheelConfig(rated_momentum=4e-3, rated_speed_rpm=4000.0)
        js = 4e-3 / (4000.0 * 2.0 * math.pi / 60.0)
        np.testing.assert_allclose(wheel_speed([2e-3, 0, 0], [0.1, 0, 0], cfg),
                                   [2e-3 / js - 0.1, 0, 0], rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(wheel_momentum([100.0, 0, 0], [0.1, 0, 0], cfg),
                                   [js * 100.1, 0, 0], rtol=1e-12, atol=0)

    def test_batch_shapes(self):
        h = np.zeros((7, 3))
        w = np.zeros((7, 3))
        assert wheel_speed(h, w).shape == (7, 3)
        assert wheel_momentum(h, w).shape == (7, 3)


# --------------------------------------------------------------------------
# 5. limit_wheel_torque
# --------------------------------------------------------------------------
CFG = WheelConfig()
TMAX = CFG.max_torque
JS = JS_HAND


class TestLimitWheelTorque:
    def test_returns_three_vector_and_does_not_mutate(self):
        u = np.array([1e-3, -1e-3, 1e-4])
        h = np.array([0.0, 0.0, 0.0])
        u0, h0 = u.copy(), h.copy()
        out = limit_wheel_torque(u, h, np.zeros(3), CFG, dt=1.0)
        assert out.shape == (3,)
        np.testing.assert_array_equal(u, u0)
        np.testing.assert_array_equal(h, h0)

    @pytest.mark.parametrize("dt", [None, 0.1, 1.0])
    def test_torque_clip_only_when_far_from_limits(self, dt):
        out = limit_wheel_torque([1e-3, -1e-3, 1e-4], np.zeros(3), np.zeros(3), CFG, dt=dt)
        np.testing.assert_allclose(out, [TMAX, -TMAX, 1e-4], rtol=1e-15, atol=0)

    def test_none_dt_blocks_push_into_limit(self):
        w = np.zeros(3)
        h_hi = 1.77e-3 * (1.0 + 1e-9)                     # just beyond +-limit (avoids ulp ties)
        h = np.array([h_hi, -h_hi, h_hi])
        u = np.array([1e-4, -1e-4, -1e-4])               # axis 0, 1 pushed outward; axis 2 inward
        out = limit_wheel_torque(u, h, w, CFG, dt=None)
        np.testing.assert_allclose(out, [0.0, 0.0, -1e-4], rtol=1e-15, atol=0)

    def test_none_dt_beyond_limit_blocks_outward_allows_inward(self):
        w = np.zeros(3)
        h = np.array([1.2 * 1.77e-3, -1.2 * 1.77e-3, 0.0])
        out = limit_wheel_torque([1e-4, -1e-4, 5e-5], h, w, CFG, dt=None)
        np.testing.assert_allclose(out, [0.0, 0.0, 5e-5], atol=0)
        out = limit_wheel_torque([-1e-4, 1e-4, 5e-5], h, w, CFG, dt=None)
        np.testing.assert_allclose(out, [-1e-4, 1e-4, 5e-5], rtol=1e-15, atol=0)

    def test_limits_depend_on_body_rate(self):
        # w shifts the momentum limits: h_hi = J_s (max_speed + w).  With w = 100 rad/s,
        # h_hi = 1.77e-3 + 100 J_s; a wheel at 1.77e-3 is 100 J_s below the limit.
        w = np.array([100.0, 0.0, 0.0])
        h_hi = JS * (MAX_SPEED_HAND + 100.0)
        h = np.array([1.77e-3, 0.0, 0.0])
        out = limit_wheel_torque([TMAX, 0, 0], h, w, CFG, dt=1.0)
        # room = h_hi - h = 100 J_s = 2.1128e-4 < TMAX*1 s: held command limited to the room
        assert 100.0 * JS < TMAX
        np.testing.assert_allclose(out[0], (h_hi - h[0]) / 1.0, rtol=1e-9)
        # with w = 0 the same wheel is beyond its limit (h > 1.77e-3 (1 + 1e-9)): blocked
        h2 = np.array([1.77e-3 * (1.0 + 1e-9), 0.0, 0.0])
        out0 = limit_wheel_torque([TMAX, 0, 0], h2, np.zeros(3), CFG, dt=None)
        np.testing.assert_allclose(out0[0], 0.0, atol=0)

    def test_dt_hand_cases(self):
        w = np.zeros(3)
        h_hi = 1.77e-3
        # inside: 1e-4 N m s below the upper limit, held 1 s -> at most 1e-4 N m
        out = limit_wheel_torque([TMAX, 0, 0], [h_hi - 1e-4, 0, 0], w, CFG, dt=1.0)
        np.testing.assert_allclose(out[0], 1e-4, rtol=1e-6)
        # same room, dt = 0.1 -> room allows 1e-3 N m > TMAX, so just the torque clip
        out = limit_wheel_torque([TMAX, 0, 0], [h_hi - 1e-4, 0, 0], w, CFG, dt=0.1)
        np.testing.assert_allclose(out[0], TMAX, rtol=1e-15)
        # beyond the upper limit: outward blocked (and never forced inward)
        out = limit_wheel_torque([1e-4, 0, 0], [h_hi + 1e-5, 0, 0], w, CFG, dt=1.0)
        assert out[0] == 0.0
        # beyond the upper limit: inward passes untouched
        out = limit_wheel_torque([-1e-4, 0, 0], [h_hi + 1e-5, 0, 0], w, CFG, dt=1.0)
        np.testing.assert_allclose(out[0], -1e-4, rtol=1e-15)
        # lower limit mirror
        out = limit_wheel_torque([-TMAX, 0, 0], [-(h_hi - 1e-4), 0, 0], w, CFG, dt=1.0)
        np.testing.assert_allclose(out[0], -1e-4, rtol=1e-6)

    def test_random_property_1000_cases(self):
        rng = np.random.default_rng(20240607)
        cnt = dict(inside_binding=0, beyond_hi=0, beyond_lo=0, free=0)
        for case in range(1000):
            dt = (None, 0.1, 1.0)[case % 3]
            u = rng.uniform(-3.0, 3.0, 3) * TMAX                  # beyond +-TMAX for ~2/3 of entries
            h = rng.uniform(-1.2, 1.2, 3) * CFG.h_max
            w = rng.uniform(-0.05, 0.05, 3)                       # small body rate, rad/s
            out = limit_wheel_torque(u, h, w, CFG, dt=dt)
            uc = np.clip(u, -TMAX, TMAX)
            h_hi = JS * (MAX_SPEED_HAND + w)
            h_lo = JS * (-MAX_SPEED_HAND + w)
            assert out.shape == (3,)
            for i in range(3):
                oi = out[i]
                # (a) torque limit
                assert abs(oi) <= TMAX * (1 + 1e-15)
                # (b) never reverses the command, never amplifies it
                assert oi * u[i] >= 0.0
                assert abs(oi) <= abs(uc[i]) * (1 + 1e-15)
                # (c) a wheel beyond a limit never gets torque pushing it further out
                if h[i] > h_hi[i]:
                    assert oi <= 0.0
                    cnt["beyond_hi"] += 1
                    # (d) ... but inward torque is not reduced by the speed limit
                    if u[i] < 0:
                        assert oi == pytest.approx(uc[i], rel=1e-15)
                if h[i] < h_lo[i]:
                    assert oi >= 0.0
                    cnt["beyond_lo"] += 1
                    if u[i] > 0:
                        assert oi == pytest.approx(uc[i], rel=1e-15)
                inside = h_lo[i] <= h[i] <= h_hi[i]
                if dt is None:
                    # no hold: only the torque clip, except outward at/beyond a limit
                    if inside:
                        assert oi == pytest.approx(uc[i], rel=1e-15)
                else:
                    if inside:
                        # (e) a held command cannot carry the wheel past a limit
                        h_end = h[i] + oi * dt
                        assert h_lo[i] - 1e-15 <= h_end <= h_hi[i] + 1e-15
                        # (f) and it is not reduced more than needed
                        end_unlimited = h[i] + uc[i] * dt
                        if h_lo[i] <= end_unlimited <= h_hi[i]:
                            assert oi == pytest.approx(uc[i], rel=1e-15)
                        else:
                            bound = h_hi[i] if end_unlimited > h_hi[i] else h_lo[i]
                            assert h_end == pytest.approx(bound, abs=1e-15)
                            cnt["inside_binding"] += 1
                        # (g) away from the nearest limit the command passes untouched
                        if (h_hi[i] - h[i]) / dt >= TMAX and (h[i] - h_lo[i]) / dt >= TMAX:
                            assert oi == pytest.approx(uc[i], rel=1e-15)
                            cnt["free"] += 1
        # the random draw must have exercised each branch (guards against a vacuous test)
        assert cnt["inside_binding"] > 20
        assert cnt["beyond_hi"] > 50 and cnt["beyond_lo"] > 50
        assert cnt["free"] > 100

    def test_dt_none_zero_when_blocked_exact(self):
        rng = np.random.default_rng(107)
        for _ in range(200):
            w = rng.uniform(-0.05, 0.05, 3)
            h_hi = JS * (MAX_SPEED_HAND + w)
            h_lo = JS * (-MAX_SPEED_HAND + w)
            h = np.where(rng.random(3) < 0.5, h_hi + rng.uniform(0, 1e-4, 3), h_lo - rng.uniform(0, 1e-4, 3))
            u = rng.uniform(-1.0, 1.0, 3) * 2 * TMAX
            out = limit_wheel_torque(u, h, w, CFG, dt=None)
            for i in range(3):
                outward = (h[i] >= h_hi[i] and u[i] > 0) or (h[i] <= h_lo[i] and u[i] < 0)
                if outward:
                    assert out[i] == 0.0
                else:
                    assert out[i] == pytest.approx(np.clip(u[i], -TMAX, TMAX), rel=1e-15)
