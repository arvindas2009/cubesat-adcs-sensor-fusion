"""Tests for ``sensor_fusion.run_flight`` (spec ``docs/specs/tuning_writeup.md`` section 3 and section 4 item 8).

* (a) ``--set`` override parsing (types via ``ast.literal_eval``, unknown field rejected with the valid names);
* (b) real-log smoke test of ``run_one`` on the hover log and the G3 regression: ``--set`` of the pre-tuning
  values (``G3_DEFAULTS`` of ``sensor_fusion.report.ekf_study``) must reproduce the numbers stored in
  ``ekf_metrics_<hover>.json`` (increment 3) to 1e-9 relative. No test pins the current ``EkfConfig`` defaults
  to the stored JSON because the defaults are tuned in this group.

The real-log tests skip if the ``.ulg`` or the stored metrics JSON is missing. Filter runs are cached per
effective configuration so that, while the tuned defaults equal the G3 values, the whole file needs one run.
"""
from __future__ import annotations

import dataclasses
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from sensor_fusion import run_flight as rf
from sensor_fusion.download_logs import PINNED_LOGS
from sensor_fusion.ekf_drone import EkfConfig

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "sensor_fusion" / "data"
REPORT_DIR = ROOT / "sensor_fusion" / "report"
HOVER = next(e for e in PINNED_LOGS if e["role"] == "hover")
HOVER_LOG = DATA_DIR / f"{HOVER['id']}.ulg"
HOVER_JSON = REPORT_DIR / f"ekf_metrics_{HOVER['id']}.json"

needs_hover = pytest.mark.skipif(not (HOVER_LOG.exists() and HOVER_JSON.exists()),
                                 reason="hover .ulg or stored ekf_metrics JSON missing")

FIELDS = [f.name for f in dataclasses.fields(EkfConfig)]


# --------------------------------------------------------------------------------------
# (a) argument parsing
# --------------------------------------------------------------------------------------
def test_parse_overrides_types():
    out = rf.parse_overrides(["gyro_noise=0.002", "cov_every=10", "use_mag=True", "estimate_biases=False",
                              "gate_prob=None", "baro_sigma=1e-1", "accel_noise=2", "discretization='van_loan'"])
    assert out == {"gyro_noise": 0.002, "cov_every": 10, "use_mag": True, "estimate_biases": False,
                   "gate_prob": None, "baro_sigma": 0.1, "accel_noise": 2, "discretization": "van_loan"}
    assert type(out["gyro_noise"]) is float
    assert type(out["cov_every"]) is int
    assert type(out["use_mag"]) is bool and type(out["estimate_biases"]) is bool
    assert out["gate_prob"] is None
    assert type(out["baro_sigma"]) is float
    assert type(out["accel_noise"]) is int            # literal_eval keeps "2" an int; the dataclass does not coerce
    assert type(out["discretization"]) is str


def test_parse_overrides_increment_and_negative_and_empty():
    out = rf.parse_overrides(["increment=2", "sigma_p0=-1.5"])
    assert out == {"increment": 2, "sigma_p0": -1.5}
    assert type(out["increment"]) is int
    assert rf.parse_overrides([]) == {}


def test_parse_overrides_accepts_every_config_field():
    out = rf.parse_overrides([f"{name}=1" for name in FIELDS])
    assert set(out) == set(FIELDS)


def test_parse_overrides_unknown_field_rejected_with_valid_names():
    with pytest.raises(SystemExit) as ei:
        rf.parse_overrides(["gyro_noize=0.002"])
    msg = str(ei.value.code)
    for name in ("gyro_noise", "accel_noise", "baro_sigma", "mag_heading_sigma", "gps_pos_floor_h"):
        assert name in msg


def test_parse_overrides_unknown_among_valid_still_rejected():
    with pytest.raises(SystemExit):
        rf.parse_overrides(["gyro_noise=0.002", "not_a_field=1"])


def test_cli_unknown_field_exits_with_valid_names():
    # through the real command line; validation must not need a filter run (generous timeout nevertheless)
    p = subprocess.run([sys.executable, "-m", "sensor_fusion.run_flight", "hover", "--set", "bogus_field=1",
                        "--no-figures"], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert p.returncode != 0
    assert "gyro_noise" in p.stderr and "baro_sigma" in p.stderr


def test_g3_defaults_are_valid_config_fields():
    from sensor_fusion.report.ekf_study import G3_DEFAULTS
    assert isinstance(G3_DEFAULTS, dict) and G3_DEFAULTS
    assert set(G3_DEFAULTS) <= set(FIELDS)
    # the documented pre-tuning starting values (EkfConfig docstring / A-03, A-04 of the Group 3 filter)
    assert G3_DEFAULTS["accel_noise"] == 0.035
    assert G3_DEFAULTS["gyro_noise"] == 1.5e-3
    assert G3_DEFAULTS["gyro_bias_rw"] == 1.0e-4
    assert G3_DEFAULTS["accel_bias_rw"] == 3.0e-4
    assert G3_DEFAULTS["baro_sigma"] == 1.5
    assert G3_DEFAULTS["mag_heading_sigma"] == 0.07
    assert G3_DEFAULTS["gps_vel_floor"] == 0.3
    assert G3_DEFAULTS["gps_pos_floor_h"] == 0.5 and G3_DEFAULTS["gps_pos_floor_v"] == 0.5
    # and EkfConfig accepts them
    EkfConfig(**{k: v for k, v in G3_DEFAULTS.items()})


def test_g3_defaults_roundtrip_through_set_strings():
    from sensor_fusion.report.ekf_study import G3_DEFAULTS
    items = [f"{k}={v!r}" for k, v in G3_DEFAULTS.items()]
    assert rf.parse_overrides(items) == G3_DEFAULTS


# --------------------------------------------------------------------------------------
# (b) real-log runs (cached per effective configuration)
# --------------------------------------------------------------------------------------
_CACHE: dict = {}
_OUT_DIR = Path(tempfile.mkdtemp(prefix="run_flight_test_"))


def hover_run(overrides: dict) -> dict:
    key = json.dumps(dataclasses.asdict(EkfConfig(increment=3, **overrides)), sort_keys=True)
    if key not in _CACHE:
        _CACHE[key] = rf.run_one(HOVER, 3, dict(overrides), out_dir=_OUT_DIR, prefix="pytest",
                                 figures=False, write=False)
    return _CACHE[key]


def g3_overrides() -> dict:
    from sensor_fusion.report.ekf_study import G3_DEFAULTS
    return {k: v for k, v in G3_DEFAULTS.items() if k != "increment"}


def assert_close_tree(actual, expected, path="", rel=1e-9):
    """Recursive comparison; numbers to ``rel`` relative (plus a negligible absolute floor for exact zeros)."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert set(actual) == set(expected), f"{path}: keys {sorted(set(actual) ^ set(expected))}"
        for k in expected:
            assert_close_tree(actual[k], expected[k], f"{path}/{k}", rel)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_close_tree(a, e, f"{path}[{i}]", rel)
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        assert actual == expected, path
    else:
        assert actual == pytest.approx(expected, rel=rel, abs=1e-15), path


@needs_hover
def test_run_one_hover_smoke_default_config():
    m = hover_run({})
    for key in ("comparison_vs_ekf2", "nis", "consistency", "final_biases", "runtime_s"):
        assert key in m, key
    c = m["comparison_vs_ekf2"]
    for name in ("pos_horizontal_m", "pos_vertical_m", "vel_m_s", "tilt_deg", "yaw_deg"):
        for stat in ("rms", "max_abs", "final"):
            assert c[name][stat] is not None and math.isfinite(c[name][stat]), (name, stat)
        assert c[name]["rms"] > 0 and c[name]["n"] > 100
    assert m["runtime_s"] > 0 and math.isfinite(m["runtime_s"])
    # all four sensors are present on the hover log, each with finite statistics and dof 3/3/1/1
    dofs = {"gps_pos": 3, "gps_vel": 3, "baro": 1, "mag_heading": 1}
    for name, dof in dofs.items():
        e = m["nis"][name]
        assert e["dof"] == dof
        assert math.isfinite(e["mean_nis"]) and e["mean_nis"] > 0
        assert 0.0 <= e["fraction_inside_95"] <= 1.0
    cons = m["consistency"]
    assert cons["n_sensors"] == 4
    assert cons["score"] is not None and math.isfinite(cons["score"]) and cons["score"] >= 0
    # the score is the RMS of ln(mean NIS after 5 s / dof) over the sensors (re-derived here from the NIS block)
    logs = [math.log(m["nis"][n]["after_5s"]["mean_nis"] / d) for n, d in dofs.items()]
    assert cons["score"] == pytest.approx(math.sqrt(sum(x * x for x in logs) / 4), rel=1e-9)
    fb = m["final_biases"]
    assert all(math.isfinite(x) for k in ("b_g_rad_s", "b_a_m_s2", "b_g_sigma_rad_s", "b_a_sigma_m_s2") for x in fb[k])
    # the result is JSON-serialisable (no numpy types / NaN)
    json.dumps(m, allow_nan=False)


@needs_hover
def test_run_one_without_write_creates_no_files():
    hover_run({})
    assert list(_OUT_DIR.iterdir()) == []


@needs_hover
def test_run_one_g3_values_via_set_reproduce_stored_json():
    stored = json.loads(HOVER_JSON.read_text())["increments"]["3"]
    m = hover_run(g3_overrides())
    # horizontal position RMS (the headline value) and every stored metric block, to 1e-9 relative
    assert m["comparison_vs_ekf2"]["pos_horizontal_m"]["rms"] == pytest.approx(
        stored["comparison_vs_ekf2"]["pos_horizontal_m"]["rms"], rel=1e-9)
    assert_close_tree(m["comparison_vs_ekf2"], stored["comparison_vs_ekf2"], "comparison_vs_ekf2")
    assert_close_tree(m["nis"], stored["nis"], "nis")
    assert_close_tree(m["final_biases"], stored["final_biases"], "final_biases")
    assert m["n_updates"] == stored["n_updates"]
    assert m["mag_skipped_tilt"] == stored["mag_skipped_tilt"]
    assert m["all_finite"] == stored["all_finite"]
