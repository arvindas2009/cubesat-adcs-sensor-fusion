"""Tests for ``adcs.monte_carlo`` (spec ``docs/specs/adcs_closed_loop.md`` section 4, tests 1-4).

Written from the spec, not from the implementation.  Oracles:

* ``draw_run``: a generator seeded manually as ``default_rng([base_seed, index, 0])``
  with the documented draw order (normal(4), normal(3), uniform, uniform(3)),
  ``scipy.stats.kstest`` for uniformity, plain numpy for unit-norm checks.
* ``aggregate`` / ``check_requirements``: hand-built rows with hand-computed
  statistics and requirement fractions (NaN, boundary and failing rows).
* ``run_one``: a direct ``run_closed_loop`` + ``summarize_closed_loop`` call with
  the arguments given in the spec.
* ``run_campaign`` / ``save_json``: serial ``run_one`` results, ``json.load``.

All simulations use ``orbits = 0.03`` (about 170 s of simulated time).
"""
from __future__ import annotations

import dataclasses
import json
import math
import os

import numpy as np
import pytest
from scipy import stats

from adcs import monte_carlo as mc
from adcs.config import DEFAULT
from adcs.orbit import orbital_period
from adcs.simulate import (
    IDEAL_SENSORS,
    MEKF_DESIGN,
    MEKF_G7,
    REALISTIC_SENSORS,
    run_closed_loop,
    summarize_closed_loop,
)

P_ORBIT = float(orbital_period(DEFAULT.orbit.a))
ORBITS_SHORT = 0.03
RATE_LO, RATE_HI = np.deg2rad(2.0), np.deg2rad(10.0)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def same(a, b, rtol=0.0, atol=0.0):
    """Recursive equality with NaN == NaN; floats compared to rtol/atol."""
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(same(a[k], b[k], rtol, atol) for k in a)
    if isinstance(a, (list, tuple, np.ndarray)):
        return (isinstance(b, (list, tuple, np.ndarray)) and len(a) == len(b)
                and all(same(x, y, rtol, atol) for x, y in zip(a, b)))
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return bool(a) == bool(b)
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
        if math.isnan(a) and math.isnan(b):
            return True
        return bool(np.isclose(a, b, rtol=rtol, atol=atol)) if (rtol or atol) else a == b
    return a == b


def manual_draw(index, base_seed=2026, rate_range=(RATE_LO, RATE_HI), tol=0.10):
    """The documented draw order, written out independently of ``adcs.monte_carlo``."""
    rng = np.random.default_rng([base_seed, index, 0])
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)
    if q[0] < 0:                     # canonical sign, w >= 0
        q = -q
    axis = rng.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    rate = rng.uniform(*rate_range)
    scale = rng.uniform(1.0 - tol, 1.0 + tol, size=3)
    y0 = np.concatenate([q, rate * axis, np.zeros(3)])
    return y0, rate, scale


# ----------------------------------------------------------------------------
# 1. draw_run
# ----------------------------------------------------------------------------
class TestDrawRun:
    def test_deterministic(self):
        a = mc.draw_run(7, base_seed=2026)
        b = mc.draw_run(7, base_seed=2026)
        assert a == b
        assert a.y0 == b.y0 and a.inertia_scale == b.inertia_scale

    def test_default_base_seed_is_2026(self):
        assert mc.draw_run(3) == mc.draw_run(3, base_seed=2026)

    def test_different_indices_and_seeds_differ(self):
        a, b = mc.draw_run(0), mc.draw_run(1)
        assert a.y0 != b.y0
        assert a.rate != b.rate
        assert a.inertia_scale != b.inertia_scale
        c = mc.draw_run(0, base_seed=2027)
        assert c.y0 != a.y0

    def test_fields_and_sim_seed(self):
        s = mc.draw_run(5, base_seed=11)
        assert s.index == 5 and s.base_seed == 11
        assert s.sim_seed == (11, 5, 1)
        assert len(s.y0) == 10
        assert len(s.inertia_scale) == 3
        assert all(isinstance(x, float) for x in s.y0)

    def test_runspec_is_frozen(self):
        s = mc.draw_run(0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.index = 3
        assert dataclasses.is_dataclass(mc.RunSpec)

    @pytest.mark.parametrize("index,base_seed", [(0, 2026), (1, 2026), (37, 2026), (5, 99), (1999, 7)])
    def test_documented_draw_order_reproduces_spec(self, index, base_seed):
        y0, rate, scale = manual_draw(index, base_seed)
        s = mc.draw_run(index, base_seed=base_seed)
        # identical operations up to the normalisation arithmetic: 1e-14 is a few ulp
        np.testing.assert_allclose(np.array(s.y0), y0, rtol=0, atol=1e-14)
        assert s.rate == pytest.approx(rate, rel=1e-15, abs=0)
        np.testing.assert_allclose(np.array(s.inertia_scale), scale, rtol=0, atol=1e-15)
        # rate field is in rad/s and equals the norm of the angular-rate part of y0
        assert np.linalg.norm(s.y0[4:7]) == pytest.approx(s.rate, rel=1e-12)
        # wheel momentum starts at zero
        assert s.y0[7:] == (0.0, 0.0, 0.0)

    def test_custom_range_and_tolerance_follow_the_draw_order(self):
        rr = (np.deg2rad(1.0), np.deg2rad(3.0))
        s = mc.draw_run(4, base_seed=5, rate_range=rr, inertia_tol=0.25)
        y0, rate, scale = manual_draw(4, 5, rate_range=rr, tol=0.25)
        np.testing.assert_allclose(np.array(s.y0), y0, rtol=0, atol=1e-14)
        np.testing.assert_allclose(np.array(s.inertia_scale), scale, rtol=0, atol=1e-15)
        assert rr[0] <= s.rate <= rr[1]

    def test_statistics_over_2000_draws(self):
        n = 2000
        specs = [mc.draw_run(i) for i in range(n)]
        rates = np.array([s.rate for s in specs])
        scales = np.array([s.inertia_scale for s in specs])
        y0 = np.array([s.y0 for s in specs])

        # rate: inside the range, uniform (KS p > 0.01; seeds are fixed so this is deterministic)
        assert rates.min() >= RATE_LO and rates.max() <= RATE_HI
        res = stats.kstest(rates, "uniform", args=(RATE_LO, RATE_HI - RATE_LO))
        assert res.pvalue > 0.01
        # inertia scales: within +-10 % and uniform on [0.9, 1.1]
        assert scales.min() >= 0.9 and scales.max() <= 1.1
        for k in range(3):
            assert stats.kstest(scales[:, k], "uniform", args=(0.9, 0.2)).pvalue > 0.01
        # quaternion: unit norm, w >= 0
        q = y0[:, :4]
        np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, rtol=0, atol=1e-12)
        assert np.all(q[:, 0] >= 0.0)
        # rotation axis: unit, uniform on the sphere -> component mean 0 (std of the mean is
        # 1/sqrt(3 n) = 0.013, so 0.05 is a ~3.9 sigma bound) and E[a a^T] = I/3
        axes = y0[:, 4:7] / rates[:, None]
        np.testing.assert_allclose(np.linalg.norm(axes, axis=1), 1.0, rtol=0, atol=1e-12)
        assert np.all(np.abs(axes.mean(axis=0)) < 0.05)
        np.testing.assert_allclose(axes.T @ axes / n, np.eye(3) / 3.0, rtol=0, atol=0.05)
        # wheel momentum zero
        assert np.all(y0[:, 7:] == 0.0)
        # quaternion is not degenerate: Haar measure on S^3 -> E[w^2] = 1/4 (w sign-folded)
        assert np.mean(q[:, 0] ** 2) == pytest.approx(0.25, abs=0.03)


# ----------------------------------------------------------------------------
# 2. aggregate / check_requirements on hand-built rows
# ----------------------------------------------------------------------------
def make_row(index, **kw):
    """A run passing every requirement; ``kw`` overrides."""
    row = {
        "index": index,
        "detumble_time_orbits": 1.0,
        "switch_time_orbits": 2.0,
        "nadir_max_deg": 2.0,
        "know_rms_deg": 0.5,
        "nadir_rms_deg": 0.6,
        "wheel_max_frac_pointing": 0.2,
        "success": True,
    }
    row.update(kw)
    return row


NAN = float("nan")


def requirement_rows():
    """Eight rows; indices deliberately non-contiguous (failing lists use ``row["index"]``).

    Expected pass/fail per row is worked out by hand in ``EXPECTED_FAIL`` below
    (limits: detumble <= 3, switch <= 4, nadir max <= 5, know rms <= 2,
    nadir rms <= 1 (goal), wheel < 1 (strict), success).
    """
    return [
        make_row(10),                                                    # passes all
        make_row(11, detumble_time_orbits=3.0),                          # boundary, meets (<=)
        make_row(12, detumble_time_orbits=3.5),                          # fails detumble
        make_row(13, switch_time_orbits=NAN, nadir_max_deg=NAN, know_rms_deg=NAN,
                 nadir_rms_deg=NAN, wheel_max_frac_pointing=NAN, success=False),   # never switched
        make_row(14, switch_time_orbits=4.0, nadir_max_deg=5.0, know_rms_deg=2.0,
                 nadir_rms_deg=1.2, wheel_max_frac_pointing=1.0),        # boundaries; goal and wheel fail
        make_row(15, nadir_max_deg=5.01, nadir_rms_deg=1.0),             # nadir max fails; goal boundary meets
        make_row(16, know_rms_deg=2.5, success=False),                   # knowledge and success fail
        make_row(17, switch_time_orbits=4.5),                            # acquisition fails
    ]


EXPECTED_FAIL = {
    "detumble_orbits_max": [12],
    "acquire_orbits_max": [13, 17],
    "pointing_max_deg": [13, 15],
    "knowledge_rms_deg": [13, 16],
    "pointing_rms_goal_deg": [13, 14],
    "wheel_frac_max": [13, 14],
    "success": [13, 16],
}
# all_requirements = meeting everything except the goal: only rows 10 and 11 pass
EXPECTED_ALL_FAIL = [12, 13, 14, 15, 16, 17]


class TestAggregate:
    X = [1.0, 2.0, 3.0, 4.0, 10.0]

    def rows(self):
        xs = self.X + [NAN]
        succ = [True, True, False, True, False, True]
        return [{"index": i, "x": xs[i], "y": NAN, "z": 5.0, "success": succ[i]} for i in range(6)]

    def test_hand_computed_statistics(self):
        agg = mc.aggregate(self.rows())
        x = agg["x"]
        assert x["n_finite"] == 5
        assert x["mean"] == pytest.approx(4.0, abs=1e-12)          # (1+2+3+4+10)/5
        assert x["median"] == pytest.approx(3.0, abs=1e-12)
        assert x["min"] == 1.0 and x["max"] == 10.0
        # linear-interpolation percentiles: position 0.05*4 = 0.2 -> 1.2 ; 0.95*4 = 3.8 -> 4 + 0.8*6 = 8.8
        assert x["p5"] == pytest.approx(1.2, abs=1e-12)
        assert x["p95"] == pytest.approx(8.8, abs=1e-12)
        # The spec does not say ddof; deviations from the mean are (-3,-2,-1,0,6): sum sq = 50.
        assert any(x["std"] == pytest.approx(v, abs=1e-12) for v in (math.sqrt(50 / 5), math.sqrt(50 / 4)))
        assert set(x) >= {"mean", "std", "median", "p5", "p95", "min", "max", "n_finite"}

    def test_all_nan_key(self):
        y = mc.aggregate(self.rows())["y"]
        assert y["n_finite"] == 0
        for k in ("mean", "std", "median", "p5", "p95", "min", "max"):
            assert math.isnan(y[k]), k

    def test_constant_key(self):
        z = mc.aggregate(self.rows())["z"]
        assert z["n_finite"] == 6
        for k in ("mean", "median", "p5", "p95", "min", "max"):
            assert z[k] == pytest.approx(5.0, abs=1e-12), k
        assert z["std"] == pytest.approx(0.0, abs=1e-12)

    def test_success_fraction(self):
        assert mc.aggregate(self.rows())["success_fraction"] == pytest.approx(4 / 6, abs=1e-15)

    def test_all_success(self):
        rows = [make_row(i) for i in range(3)]
        assert mc.aggregate(rows)["success_fraction"] == 1.0

    def test_does_not_mutate_input_and_ignores_list_values(self):
        rows = self.rows()
        for r in rows:
            r["inertia_scale"] = [1.0, 1.0, 1.0]
        before = json.dumps(rows)
        agg = mc.aggregate(rows)
        assert json.dumps(rows) == before
        assert agg["x"]["n_finite"] == 5


class TestCheckRequirements:
    def result(self):
        return mc.check_requirements(requirement_rows())

    @staticmethod
    def fraction_of(entry):
        """Accept either a bare fraction or a dict holding it."""
        if isinstance(entry, dict):
            for k in ("fraction", "frac", "met", "fraction_met"):
                if k in entry:
                    return entry[k]
            raise AssertionError(f"no fraction key in {entry}")
        return entry

    @staticmethod
    def failing_of(entry):
        if isinstance(entry, dict):
            for k in ("failing", "failing_indices", "failed", "fail"):
                if k in entry:
                    return entry[k]
        raise AssertionError(f"no failing-index list in {entry}")

    def test_requirement_constants(self):
        r = mc.REQUIREMENTS
        assert r["detumble_orbits_max"] == 3.0
        assert r["acquire_orbits_max"] == 4.0
        assert r["pointing_max_deg"] == 5.0
        assert r["knowledge_rms_deg"] == 2.0
        assert r["pointing_rms_goal_deg"] == 1.0
        assert r["wheel_frac_max"] == 1.0

    def test_fractions_per_item(self):
        res = self.result()
        for name, failing in EXPECTED_FAIL.items():
            assert self.fraction_of(res[name]) == pytest.approx((8 - len(failing)) / 8, abs=1e-15), name

    def test_failing_indices_per_item(self):
        res = self.result()
        for name, failing in EXPECTED_FAIL.items():
            assert sorted(int(i) for i in self.failing_of(res[name])) == failing, name

    def test_all_requirements_excludes_the_goal(self):
        res = self.result()
        assert self.fraction_of(res["all_requirements"]) == pytest.approx(2 / 8, abs=1e-15)
        assert sorted(int(i) for i in self.failing_of(res["all_requirements"])) == EXPECTED_ALL_FAIL

    def test_nan_counts_as_not_met(self):
        # row 13 is NaN in every metric: it must be in every failing list except detumble
        res = self.result()
        for name in EXPECTED_FAIL:
            if name == "detumble_orbits_max":
                assert 13 not in [int(i) for i in self.failing_of(res[name])]
            else:
                assert 13 in [int(i) for i in self.failing_of(res[name])], name

    def test_custom_requirements_are_used(self):
        req = dict(mc.REQUIREMENTS, detumble_orbits_max=0.5, pointing_rms_goal_deg=10.0)
        res = mc.check_requirements(requirement_rows(), req=req)
        assert self.fraction_of(res["detumble_orbits_max"]) == 0.0
        assert sorted(int(i) for i in self.failing_of(res["detumble_orbits_max"])) == list(range(10, 18))
        # goal relaxed to 10 deg: only the NaN row still fails
        assert self.fraction_of(res["pointing_rms_goal_deg"]) == pytest.approx(7 / 8, abs=1e-15)
        assert self.failing_of(res["pointing_rms_goal_deg"]) == [13]
        assert self.fraction_of(res["all_requirements"]) == 0.0

    def test_all_runs_passing(self):
        res = mc.check_requirements([make_row(i) for i in range(4)])
        for name in list(EXPECTED_FAIL) + ["all_requirements"]:
            assert self.fraction_of(res[name]) == 1.0, name
            assert list(self.failing_of(res[name])) == [], name


# ----------------------------------------------------------------------------
# 3. run_one
# ----------------------------------------------------------------------------
def direct_summary(spec, sensors, mekf, inertia_scale=None):
    scale = np.asarray(spec.inertia_scale if inertia_scale is None else inertia_scale, float)
    inertia_true = np.diag(np.diag(DEFAULT.spacecraft.inertia) * scale)
    res = run_closed_loop(list(spec.y0), ORBITS_SHORT * P_ORBIT, seed=list(spec.sim_seed),
                          sensors=sensors, mekf_cfg=mekf, inertia_true=inertia_true, log_every=1)
    return summarize_closed_loop(res)


@pytest.fixture(scope="module")
def run_one_case():
    spec = mc.draw_run(0)
    row = mc.run_one(spec, campaign="ideal", orbits=ORBITS_SHORT)
    expected = direct_summary(spec, IDEAL_SENSORS, MEKF_G7)
    nominal = direct_summary(spec, IDEAL_SENSORS, MEKF_G7, inertia_scale=(1.0, 1.0, 1.0))
    return spec, row, expected, nominal


class TestRunOne:
    def test_campaigns_table(self):
        assert set(mc.CAMPAIGNS) == {"design", "ideal"}
        assert mc.CAMPAIGNS["design"] == (REALISTIC_SENSORS, MEKF_DESIGN)
        assert mc.CAMPAIGNS["ideal"] == (IDEAL_SENSORS, MEKF_G7)

    def test_keys_and_metadata(self, run_one_case):
        spec, row, expected, _ = run_one_case
        assert row["index"] == spec.index
        assert row["rate_degps"] == pytest.approx(np.rad2deg(spec.rate), rel=1e-12)
        assert row["inertia_scale"] == pytest.approx(list(spec.inertia_scale), rel=0, abs=1e-15)
        assert isinstance(row["inertia_scale"], list)
        assert isinstance(row["runtime_s"], float) and row["runtime_s"] > 0.0
        for k in expected:
            assert k in row, k
        assert set(row) == {"index", "rate_degps", "inertia_scale", "runtime_s", *expected}

    def test_json_ready(self, run_one_case):
        _, row, _, _ = run_one_case
        text = json.dumps(row)                # NaN allowed, numpy bool/array would raise
        assert isinstance(json.loads(text), dict)
        assert type(row["success"]) is bool

    def test_summary_equals_direct_closed_loop_call(self, run_one_case):
        _, row, expected, _ = run_one_case
        got = {k: v for k, v in row.items() if k in expected}
        # same code path, same seed, same process -> bit-identical
        assert same(got, expected), {k: (got[k], expected[k]) for k in expected if not same(got[k], expected[k])}

    def test_inertia_scale_is_applied(self, run_one_case):
        # Guard against a vacuous comparison: running the same case with the nominal
        # inertia must change the summary; ``run_one`` follows the *scaled* run.
        spec, row, expected, nominal = run_one_case
        assert max(abs(s - 1.0) for s in spec.inertia_scale) > 0.01
        differing = [k for k in expected if not same(expected[k], nominal[k])]
        assert differing, "scaled and nominal inertia gave identical summaries; test is not discriminating"
        for k in differing:
            assert same(row[k], expected[k]) and not same(row[k], nominal[k]), k

    def test_sim_seed_is_used(self):
        spec = mc.draw_run(0)
        other = dataclasses.replace(spec, base_seed=spec.base_seed + 1)
        a = mc.run_one(spec, campaign="ideal", orbits=ORBITS_SHORT)
        b = mc.run_one(other, campaign="ideal", orbits=ORBITS_SHORT)
        # different sim_seed (and identical y0/inertia, since the dataclass replace keeps them)
        assert other.sim_seed != spec.sim_seed and other.y0 == spec.y0
        assert not same({k: v for k, v in a.items() if k != "runtime_s"},
                        {k: v for k, v in b.items() if k != "runtime_s"})


# ----------------------------------------------------------------------------
# 4. run_campaign / save_json
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def campaign_result():
    return mc.run_campaign("ideal", n=2, orbits=ORBITS_SHORT, workers=2)


class TestRunCampaign:
    def test_structure(self, campaign_result):
        r = campaign_result
        assert set(r) >= {"campaign", "config", "runs", "aggregate", "requirements", "wall_time_s"}
        assert r["campaign"] == "ideal"
        assert r["wall_time_s"] > 0.0
        cfg = r["config"]
        assert set(cfg) >= {"n", "orbits", "base_seed", "rate_range_degps", "inertia_tol",
                            "sensors", "mekf", "requirements"}
        assert cfg["n"] == 2
        assert cfg["orbits"] == ORBITS_SHORT
        assert cfg["base_seed"] == 2026
        np.testing.assert_allclose(cfg["rate_range_degps"], [2.0, 10.0], rtol=1e-12)
        assert cfg["inertia_tol"] == 0.10
        assert isinstance(cfg["sensors"], dict) and isinstance(cfg["mekf"], dict)
        assert cfg["requirements"] == mc.REQUIREMENTS
        # whole result is serialisable once NaN is allowed
        json.dumps(r)

    def test_rows_sorted_by_index(self, campaign_result):
        runs = campaign_result["runs"]
        assert [r["index"] for r in runs] == [0, 1]

    def test_rows_equal_serial_run_one(self, campaign_result):
        for row in campaign_result["runs"]:
            serial = mc.run_one(mc.draw_run(row["index"]), campaign="ideal", orbits=ORBITS_SHORT)
            a = {k: v for k, v in row.items() if k != "runtime_s"}
            b = {k: v for k, v in serial.items() if k != "runtime_s"}
            # Worker processes run with OPENBLAS_NUM_THREADS=1 and the parent may not;
            # the matrices are <= 9x9 so results should still be bit-identical, but a
            # few-ulp relative slack is allowed for BLAS summation order.
            assert same(a, b, rtol=1e-12, atol=1e-300), {k: (a[k], b[k]) for k in b if not same(a[k], b[k], 1e-12)}

    def test_aggregate_and_requirements_match_rows(self, campaign_result):
        runs = campaign_result["runs"]
        assert same(campaign_result["aggregate"], mc.aggregate(runs))
        assert same(campaign_result["requirements"], mc.check_requirements(runs))
        assert campaign_result["aggregate"]["success_fraction"] == pytest.approx(
            np.mean([r["success"] for r in runs]), abs=1e-15)

    def test_openblas_threads_set(self, campaign_result):
        assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"

    def test_save_json_of_campaign_has_null_for_nan(self, campaign_result, tmp_path):
        path = tmp_path / "mc.json"
        mc.save_json(campaign_result, path)
        text = path.read_text()
        assert "NaN" not in text and "Infinity" not in text
        loaded = json.loads(text)                     # strict parse, no NaN constants
        # 0.03 orbits is far too short to reach pointing: metrics are NaN -> null
        nan_keys = [k for k, v in campaign_result["runs"][0].items()
                    if isinstance(v, float) and math.isnan(v)]
        assert nan_keys, "precondition: a 0.03-orbit run has undefined (NaN) metrics"
        assert "null" in text
        for run_in, run_out in zip(campaign_result["runs"], loaded["runs"]):
            for k in nan_keys:
                if isinstance(run_in[k], float) and math.isnan(run_in[k]):
                    assert run_out[k] is None, k
            assert run_out["index"] == run_in["index"]
            assert run_out["success"] is run_in["success"]
        assert loaded["campaign"] == "ideal"
        assert loaded["config"]["n"] == 2
        assert loaded["aggregate"]["success_fraction"] == campaign_result["aggregate"]["success_fraction"]


class TestSaveJson:
    def test_numpy_and_nan_conversion(self, tmp_path):
        obj = {
            "f": np.float64(1.5),
            "i": np.int64(4),
            "b": np.bool_(True),
            "arr": np.arange(3),
            "mat": np.array([[1.0, np.nan], [3.0, 4.0]]),
            "nan": float("nan"),
            "nested": {"list": [1.0, float("nan"), np.float64(2.5)], "tuple": (1, 2)},
        }
        path = tmp_path / "x.json"
        mc.save_json(obj, path)
        text = path.read_text()
        assert "NaN" not in text
        out = json.loads(text)
        assert out == {
            "f": 1.5, "i": 4, "b": True, "arr": [0, 1, 2], "mat": [[1.0, None], [3.0, 4.0]],
            "nan": None, "nested": {"list": [1.0, None, 2.5], "tuple": [1, 2]},
        }

    def test_indent_is_one(self, tmp_path):
        path = tmp_path / "y.json"
        mc.save_json({"a": 1, "b": [1, 2]}, path)
        lines = path.read_text().splitlines()
        assert lines[0] == "{"
        assert lines[1].startswith(' "a"') and not lines[1].startswith("  ")
        assert lines[2].startswith(' "b"') and not lines[2].startswith("  ")
        assert lines[3] == "  1,"          # list items indented by two spaces (nesting depth 2 x indent 1)

    def test_accepts_str_path(self, tmp_path):
        path = str(tmp_path / "z.json")
        mc.save_json({"a": float("nan")}, path)
        assert json.load(open(path)) == {"a": None}
