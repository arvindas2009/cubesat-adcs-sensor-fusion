"""Tests for adcs/config.py (spec docs/specs/adcs_physics.md, section 5 "Config").

Oracles: hand-computed box inertias, the standard cube inertia m a^2 / 6, and a
Julian-date count made from Python's own datetime arithmetic (not from
common.frames.julian_date).
"""
from __future__ import annotations

import dataclasses
import datetime as dt

import numpy as np
import pytest

from adcs.config import (
    DEFAULT,
    EPOCH,
    OrbitConfig,
    SimConfig,
    SpacecraftConfig,
    inertia_box,
)
from common.frames import R_EARTH


# ---------------------------------------------------------------- inertia_box
def test_inertia_box_hand_values_3u():
    # 4 kg, 0.1 x 0.1 x 0.34 m: Ixx = Iyy = 4/12 (0.1^2 + 0.34^2), Izz = 4/12 (0.1^2 + 0.1^2)
    i = inertia_box(4.0, (0.1, 0.1, 0.34))
    ixx = 4.0 / 12.0 * (0.1**2 + 0.34**2)
    izz = 4.0 / 12.0 * (0.1**2 + 0.1**2)
    assert ixx == pytest.approx(0.041867, abs=5e-7)  # hand value quoted in the spec
    assert izz == pytest.approx(0.006667, abs=5e-7)
    np.testing.assert_allclose(i, np.diag([ixx, ixx, izz]), rtol=1e-14, atol=0)


def test_inertia_box_shape_diagonal_symmetric():
    i = inertia_box(2.5, (0.2, 0.3, 0.5))
    assert i.shape == (3, 3)
    np.testing.assert_array_equal(i, np.diag(np.diag(i)))
    assert np.all(np.diag(i) > 0)


def test_inertia_box_cube_is_m_a2_over_6():
    m, a = 3.7, 0.23
    i = inertia_box(m, (a, a, a))
    np.testing.assert_allclose(i, np.eye(3) * m * a**2 / 6.0, rtol=1e-14)


def test_inertia_box_asymmetric_axes_are_not_mixed_up():
    # Each principal moment uses the two edges perpendicular to its axis.
    m = 6.0
    a, b, c = 0.1, 0.2, 0.4
    i = np.diag(inertia_box(m, (a, b, c)))
    np.testing.assert_allclose(
        i, [m * (b**2 + c**2) / 12, m * (a**2 + c**2) / 12, m * (a**2 + b**2) / 12], rtol=1e-14
    )
    # Longest edge along z -> smallest moment about z; longest edge along x -> smallest about x.
    assert np.argmin(i) == 2
    j = np.diag(inertia_box(m, (c, b, a)))
    assert np.argmin(j) == 0


def test_inertia_box_linear_in_mass_and_scales_with_length_squared():
    i1 = inertia_box(1.0, (0.1, 0.2, 0.3))
    np.testing.assert_allclose(inertia_box(5.0, (0.1, 0.2, 0.3)), 5.0 * i1, rtol=1e-14)
    np.testing.assert_allclose(inertia_box(1.0, (0.2, 0.4, 0.6)), 4.0 * i1, rtol=1e-14)


def test_inertia_box_trace_identity():
    # Ixx + Iyy + Izz = m (a^2 + b^2 + c^2) / 6
    rng = np.random.default_rng(1)
    for _ in range(10):
        m = rng.uniform(0.5, 10)
        s = rng.uniform(0.05, 1.0, 3)
        assert np.trace(inertia_box(m, s)) == pytest.approx(m * np.sum(s**2) / 6.0, rel=1e-13)


def test_default_spacecraft_inertia():
    i = DEFAULT.spacecraft.inertia
    ixx = 4.0 / 12.0 * (0.1**2 + 0.34**2)
    izz = 4.0 / 12.0 * (0.1**2 + 0.1**2)
    np.testing.assert_allclose(i, np.diag([ixx, ixx, izz]), rtol=1e-14)
    assert i[0, 0] == pytest.approx(0.041867, abs=5e-7)
    assert i[2, 2] == pytest.approx(0.006667, abs=5e-7)


# --------------------------------------------------------------------- frozen
@pytest.mark.parametrize(
    "obj, attr, value",
    [
        (DEFAULT, "epoch", EPOCH),
        (DEFAULT, "orbit", OrbitConfig()),
        (DEFAULT.spacecraft, "mass", 5.0),
        (DEFAULT.spacecraft, "cd", 2.0),
        (DEFAULT.orbit, "altitude", 500e3),
        (DEFAULT.orbit, "inc", 0.0),
    ],
)
def test_configs_are_frozen(obj, attr, value):
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(obj, attr, value)


def test_config_types_are_frozen_dataclasses():
    for cls in (SimConfig, SpacecraftConfig, OrbitConfig):
        assert dataclasses.is_dataclass(cls)
        assert cls.__dataclass_params__.frozen


def test_replace_creates_new_object_and_leaves_default_untouched():
    other = dataclasses.replace(DEFAULT, orbit=dataclasses.replace(DEFAULT.orbit, altitude=500e3))
    assert other.orbit.altitude == 500e3
    assert DEFAULT.orbit.altitude == 400e3
    assert other.orbit.a == pytest.approx(R_EARTH + 500e3)


# ----------------------------------------------------------------------- epoch
def test_jd_epoch_is_2461120():
    assert DEFAULT.jd_epoch == 2461120.0


def test_epoch_is_20260320_1200_utc_and_jd_matches_independent_count():
    assert DEFAULT.epoch == dt.datetime(2026, 3, 20, 12, 0, 0)
    # Independent: J2000.0 = 2000-01-01 12:00 = JD 2451545.0
    delta = DEFAULT.epoch - dt.datetime(2000, 1, 1, 12, 0, 0)
    assert 2451545.0 + delta.days + delta.seconds / 86400.0 == 2461120.0


def test_jd_epoch_follows_custom_epoch():
    cfg = SimConfig(epoch=dt.datetime(2026, 3, 21, 18, 0, 0))
    assert cfg.jd_epoch == pytest.approx(2461121.25, abs=1e-9)


# ---------------------------------------------------------------- orbit config
def test_orbit_semi_major_axis_is_radius_plus_altitude():
    assert DEFAULT.orbit.a == pytest.approx(R_EARTH + 400e3, abs=1e-6)
    assert DEFAULT.orbit.ecc == 0.0
    assert DEFAULT.orbit.inc == pytest.approx(np.deg2rad(51.6), abs=1e-14)
    assert DEFAULT.orbit.raan == 0.0  # B-14: RAAN 0 at the equinox epoch => beta ~ 0
