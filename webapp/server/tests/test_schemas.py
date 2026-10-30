"""Tests for ``webapp.server.schemas`` (spec: docs/specs/webapp_protocol.md section 1 and 5.1).

Oracles: the spec's constants and constraints, ``adcs.simulate`` presets, and an independent
``run_closed_loop`` call with the arguments ``python -m adcs.simulate`` passes (CLI equivalence).
"""
from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

pytest.importorskip("fastapi")
pydantic = pytest.importorskip("pydantic")

from pydantic import ValidationError  # noqa: E402

from adcs.config import DEFAULT  # noqa: E402
from adcs.orbit import orbital_period  # noqa: E402
from adcs.simulate import (IDEAL_SENSORS, MEKF_DESIGN, MEKF_G7, REALISTIC_SENSORS,  # noqa: E402
                           ControlConfig, run_closed_loop, tumble_state)
from webapp.server.schemas import SessionParams, build_sim_kwargs  # noqa: E402

P_ORBIT = float(orbital_period(DEFAULT.orbit.a))
FIELDS = {"orbits", "rate_deg_s", "seed", "sensors", "filter", "dumping", "inertia_scale"}


# ---------------------------------------------------------------------------
# defaults and validation
# ---------------------------------------------------------------------------
class TestDefaults:
    def test_default_values(self):
        p = SessionParams()
        assert p.orbits == 6.0
        assert p.rate_deg_s == 10.0
        assert p.seed == 0
        assert p.sensors == "realistic"
        assert p.filter == "design"
        assert p.dumping is True
        assert p.inertia_scale == 1.0

    def test_field_set_is_exactly_the_spec_table(self):
        assert set(SessionParams().model_dump()) == FIELDS
        assert set(SessionParams.model_json_schema()["properties"]) == FIELDS

    def test_empty_dict_equals_defaults(self):
        assert SessionParams.model_validate({}) == SessionParams()

    def test_roundtrip_through_dict(self):
        p = SessionParams(orbits=1.5, rate_deg_s=3.0, seed=7, sensors="ideal", filter="g7",
                          dumping=False, inertia_scale=0.8)
        assert SessionParams.model_validate(p.model_dump()) == p
        assert SessionParams.model_validate_json(p.model_dump_json()) == p


@pytest.mark.parametrize("kw", [
    dict(orbits=0.05), dict(orbits=15), dict(orbits=15.0),
    dict(rate_deg_s=0), dict(rate_deg_s=0.0), dict(rate_deg_s=30), dict(rate_deg_s=30.0),
    dict(seed=0), dict(seed=2**31 - 1),
    dict(inertia_scale=0.5), dict(inertia_scale=1.5),
    dict(sensors="ideal"), dict(sensors="realistic"), dict(filter="g7"), dict(filter="design"),
    dict(dumping=False), dict(dumping=True),
    dict(orbits=1),  # an int is a valid float
])
def test_boundary_and_valid_values_accepted(kw):
    SessionParams(**kw)


@pytest.mark.parametrize("kw", [
    # out of range, one case per constraint side
    dict(orbits=0.049), dict(orbits=0.0), dict(orbits=-1.0), dict(orbits=15.001),
    dict(rate_deg_s=-0.001), dict(rate_deg_s=30.001),
    dict(seed=-1), dict(seed=2**31),
    dict(inertia_scale=0.499), dict(inertia_scale=1.501), dict(inertia_scale=0.0),
    # non-finite floats are outside every interval
    dict(orbits=float("nan")), dict(orbits=float("inf")), dict(rate_deg_s=float("nan")),
    dict(rate_deg_s=float("inf")), dict(inertia_scale=float("nan")),
    # unknown enum values
    dict(sensors="perfect"), dict(sensors=""), dict(filter="g8"), dict(filter="DESIGN"),
    # wrong types
    dict(orbits="abc"), dict(rate_deg_s="fast"), dict(seed="x"), dict(seed=1.5),
    dict(seed=None), dict(orbits=None), dict(dumping="maybe"), dict(dumping=None),
    dict(inertia_scale=[1.0]), dict(sensors=1), dict(filter=None),
    # booleans given for floats (spec section 1)
    dict(orbits=True), dict(rate_deg_s=True), dict(rate_deg_s=False), dict(inertia_scale=True),
    # unknown keys
    dict(extra_key=1), dict(orbit=3.0), dict(Orbits=3.0),
])
def test_invalid_values_raise(kw):
    with pytest.raises(ValidationError):
        SessionParams(**kw)


def test_extra_key_alongside_valid_fields_raises():
    with pytest.raises(ValidationError):
        SessionParams.model_validate({"orbits": 1.0, "bogus": True})


def test_non_dict_input_raises():
    with pytest.raises(ValidationError):
        SessionParams.model_validate([1, 2, 3])


# ---------------------------------------------------------------------------
# build_sim_kwargs
# ---------------------------------------------------------------------------
class TestBuildSimKwargs:
    def test_keys_are_closedloopsim_arguments(self):
        from adcs.stepper import ClosedLoopSim
        kw = build_sim_kwargs(SessionParams())
        allowed = set(inspect.signature(ClosedLoopSim.__init__).parameters) - {"self"}
        assert set(kw) <= allowed
        assert {"y0", "t_end", "seed", "sensors", "mekf_cfg", "ctrl", "inertia_true",
                "log_every"} <= set(kw)

    def test_log_every_is_ten(self):
        assert build_sim_kwargs(SessionParams())["log_every"] == 10
        assert build_sim_kwargs(SessionParams(sensors="ideal", filter="g7"))["log_every"] == 10

    def test_defaults_match_cli_arguments(self):
        kw = build_sim_kwargs(SessionParams())
        assert kw["seed"] == 0
        assert kw["sensors"] == REALISTIC_SENSORS
        assert kw["mekf_cfg"] == MEKF_DESIGN
        assert kw["ctrl"] == ControlConfig(dumping=True)
        assert kw["inertia_true"] is None
        assert kw["t_end"] == 6.0 * P_ORBIT
        np.testing.assert_array_equal(kw["y0"], tumble_state(np.deg2rad(10.0), 0))

    @pytest.mark.parametrize("sensors,expected", [("realistic", REALISTIC_SENSORS),
                                                  ("ideal", IDEAL_SENSORS)])
    def test_sensor_presets(self, sensors, expected):
        assert build_sim_kwargs(SessionParams(sensors=sensors))["sensors"] == expected

    @pytest.mark.parametrize("flt,expected", [("design", MEKF_DESIGN), ("g7", MEKF_G7)])
    def test_filter_presets(self, flt, expected):
        assert build_sim_kwargs(SessionParams(filter=flt))["mekf_cfg"] == expected

    @pytest.mark.parametrize("dumping", [True, False])
    def test_dumping_maps_to_control_config(self, dumping):
        ctrl = build_sim_kwargs(SessionParams(dumping=dumping))["ctrl"]
        assert ctrl == ControlConfig(dumping=dumping)
        assert ctrl.dumping is dumping

    def test_inertia_scale_one_is_none(self):
        assert build_sim_kwargs(SessionParams(inertia_scale=1.0))["inertia_true"] is None
        assert build_sim_kwargs(SessionParams(inertia_scale=1))["inertia_true"] is None

    @pytest.mark.parametrize("scale", [0.5, 0.8, 0.999, 1.001, 1.1, 1.5])
    def test_inertia_scale_is_exact_multiple_of_nominal(self, scale):
        got = build_sim_kwargs(SessionParams(inertia_scale=scale))["inertia_true"]
        # exact algebra: one multiplication by the nominal inertia, so bit equality
        np.testing.assert_array_equal(got, scale * DEFAULT.spacecraft.inertia)
        assert got.shape == (3, 3)

    def test_controller_keeps_nominal_inertia(self):
        # the scaled inertia is the *truth* inertia only; the nominal config is untouched
        kw = build_sim_kwargs(SessionParams(inertia_scale=1.3))
        assert kw.get("cfg", DEFAULT) == DEFAULT

    @pytest.mark.parametrize("orbits", [0.05, 0.5, 1.0, 6.0, 15.0])
    def test_t_end_is_orbits_times_period(self, orbits):
        kw = build_sim_kwargs(SessionParams(orbits=orbits))
        assert kw["t_end"] == orbits * P_ORBIT
        # independent period: Kepler's third law with the standard gravitational parameter
        mu = 3.986004418e14
        kepler = 2.0 * math.pi * math.sqrt(DEFAULT.orbit.a ** 3 / mu)
        assert kw["t_end"] == pytest.approx(orbits * kepler, rel=1e-6)

    @pytest.mark.parametrize("rate,seed", [(10.0, 0), (0.0, 1), (30.0, 5), (2.5, 2**31 - 1)])
    def test_y0_is_tumble_state(self, rate, seed):
        kw = build_sim_kwargs(SessionParams(rate_deg_s=rate, seed=seed))
        y0 = kw["y0"]
        assert kw["seed"] == seed
        assert y0.shape == (10,)
        assert np.linalg.norm(y0[:4]) == pytest.approx(1.0, abs=1e-12)
        # independent check of the rate magnitude and of h_w = 0
        assert np.linalg.norm(y0[4:7]) == pytest.approx(np.deg2rad(rate), abs=1e-14)
        np.testing.assert_array_equal(y0[7:], 0.0)
        np.testing.assert_array_equal(y0, tumble_state(np.deg2rad(rate), seed))

    def test_seed_changes_y0(self):
        a = build_sim_kwargs(SessionParams(seed=0))["y0"]
        b = build_sim_kwargs(SessionParams(seed=1))["y0"]
        assert not np.array_equal(a, b)

    def test_does_not_mutate_params_and_returns_fresh_arrays(self):
        p = SessionParams(inertia_scale=1.2)
        before = p.model_dump()
        k1, k2 = build_sim_kwargs(p), build_sim_kwargs(p)
        assert p.model_dump() == before
        k1["y0"][:] = 0.0
        k1["inertia_true"][:] = 0.0
        assert np.linalg.norm(k2["y0"][:4]) == pytest.approx(1.0)
        assert k2["inertia_true"][0, 0] > 0.0


# ---------------------------------------------------------------------------
# CLI equivalence (spec section 1): server sim == run_closed_loop with the CLI's arguments
# ---------------------------------------------------------------------------
def _flat_bytes_equal(a, b):
    from adcs.tests.golden_closed_loop import flatten
    fa, fb = flatten(a), flatten(b)
    assert set(fa) == set(fb)
    bad = [k for k in fa if fa[k].dtype != fb[k].dtype or fa[k].shape != fb[k].shape
           or fa[k].tobytes() != fb[k].tobytes()]
    return bad


def test_default_session_equals_cli_run_bit_for_bit():
    """``python -m adcs.simulate`` defaults (realistic, design, 10 deg/s, seed 0), 0.05 orbits.

    The CLI does not pass ``log_every``; the server always uses 10, so the reference call gets
    the same ``log_every=10`` (the logged rows are subsampled views of the same trajectory).
    """
    from adcs.stepper import ClosedLoopSim
    p = SessionParams(orbits=0.05)
    kw = build_sim_kwargs(p)
    sim = ClosedLoopSim(**kw)
    sim.step(sim.remaining)
    res_server = sim.result()

    # CLI arguments, written out independently of build_sim_kwargs
    y0 = tumble_state(np.deg2rad(10.0), 0)
    res_cli = run_closed_loop(y0, 0.05 * P_ORBIT, seed=0, ctrl=ControlConfig(dumping=True),
                              sensors=REALISTIC_SENSORS, mekf_cfg=MEKF_DESIGN, log_every=10)
    assert _flat_bytes_equal(res_server, res_cli) == []
    assert res_server.meta["kicks"] == []


def test_nondefault_session_equals_direct_call_bit_for_bit():
    from adcs.stepper import ClosedLoopSim
    p = SessionParams(orbits=0.05, rate_deg_s=5.0, seed=3, sensors="ideal", filter="g7",
                      dumping=False, inertia_scale=1.2)
    sim = ClosedLoopSim(**build_sim_kwargs(p))
    sim.step(sim.remaining)
    ref = run_closed_loop(tumble_state(np.deg2rad(5.0), 3), 0.05 * P_ORBIT, seed=3,
                          ctrl=ControlConfig(dumping=False), sensors=IDEAL_SENSORS,
                          mekf_cfg=MEKF_G7, inertia_true=1.2 * DEFAULT.spacecraft.inertia,
                          log_every=10)
    assert _flat_bytes_equal(sim.result(), ref) == []
