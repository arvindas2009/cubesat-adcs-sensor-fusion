"""Tests for the closed-loop runner ``adcs.simulate.run_closed_loop`` and ``summarize_closed_loop``.

Written from ``docs/specs/adcs_closed_loop.md`` section 3 ("Tests", items 1-9), section 0 of
``docs/specs/adcs_estimation.md`` (sampling model) and section 3 of ``docs/specs/adcs_control.md``
(Group 6 perfect-knowledge runner ``run``), not from the implementation.

Oracles
-------
* test 1: the Group 6 perfect-knowledge runner ``run(y0, ., "auto")`` is the reference for the
  low-noise limit (all sensor sigma x 0.01, matched filter);
* tests 2-4: spec thresholds (never loosened); the PK runner supplies the 3 x PK-RMS floor;
* test 5: a geometric triangle inequality for rotation angles
  ``angle(q_true, q_ref) <= angle(q_true, q_est) + angle(q_est, q_ref)``, evaluated from the logged
  quaternions with scipy ``Rotation`` and with ``dynamics.nadir_angle`` + ``Ephemeris`` (not the
  runner's own bookkeeping);
* test 7: the random streams are rebuilt in the test from ``default_rng(seed).spawn(4)``;
* test 9: hand computation (plain Python loops, hand-made percentile) on a small hand-built
  ``ClosedLoopResult``.

The long scenarios (tests 1-5) and the short ones (6-8) run once, in parallel worker processes, in
a session-scoped fixture (pattern of ``test_simulate_pk.py``; BLAS threads limited to 1 in workers).
"""
from __future__ import annotations

import dataclasses
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from adcs import dynamics as dyn
from adcs.config import DEFAULT
from adcs.estimation import MekfConfig
from adcs.orbit import Ephemeris, initial_state, orbital_period
from adcs.sensors import Gyro, GyroConfig, MagnetometerConfig, SunSensorConfig
from adcs.simulate import (
    IDEAL_SENSORS,
    MEKF_DESIGN,
    MEKF_G7,
    MODE_DETUMBLE,
    MODE_POINTING,
    REALISTIC_SENSORS,
    ClosedLoopResult,
    ModeConfig,
    SensorErrors,
    run,
    run_closed_loop,
    summarize_closed_loop,
    tumble_state,
)

P_ORBIT = float(orbital_period(DEFAULT.orbit.a))
R0, V0 = initial_state(DEFAULT)
T_LONG = 2.5 * P_ORBIT
T_SHORT = 0.4 * P_ORBIT
Y0 = tumble_state(math.radians(2.0), seed=0)

# low-noise case of spec section 3 (all sensor sigma x 0.01), exactly as specified
MAG_LN = MagnetometerConfig(sigma=0.15e-9)
CSS_LN = SunSensorConfig(sigma_diode=0.000436)
GYRO_LN = GyroConfig(arw=1.8908e-6, rrw=1.0409e-8, bias0_sigma=1.7453e-3)
MEKF_LN = dataclasses.replace(MEKF_G7, arw=GYRO_LN.arw, rrw=GYRO_LN.rrw, sigma_mag=MAG_LN.sigma,
                              sigma_sun=CSS_LN.sigma_vector)

BIAS_EXPLICIT = (1.0e-7, -2.0e-7, 3.0e-7)


# --------------------------------------------------------------------------
# session fixture: every simulation of the file, run once in a process pool
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def cl_jobs():
    """name -> Future of the simulation (heaviest first so that they start immediately).

    Each simulation is independent and deterministic.  BLAS threads are limited to 1 in the
    workers (set before they start) to avoid oversubscription on many-core machines.
    """
    keys = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    old = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ[k] = "1"
    ex = ProcessPoolExecutor(max_workers=min(15, os.cpu_count() or 1))
    try:
        sub = ex.submit
        jobs = {
            # --- long: 2.5 orbits (tests 1-5) ---
            "real_design": sub(run_closed_loop, Y0, T_LONG, 0, sensors=REALISTIC_SENSORS,
                               mekf_cfg=MEKF_DESIGN),
            "real_g7": sub(run_closed_loop, Y0, T_LONG, 0, sensors=REALISTIC_SENSORS,
                           mekf_cfg=MEKF_G7),
            "ideal": sub(run_closed_loop, Y0, T_LONG, 0, sensors=IDEAL_SENSORS, mekf_cfg=MEKF_G7),
            "lownoise": sub(run_closed_loop, Y0, T_LONG, 0, sensors=IDEAL_SENSORS, mekf_cfg=MEKF_LN,
                            mag_cfg=MAG_LN, css_cfg=CSS_LN, gyro_cfg=GYRO_LN),
            "pk": sub(run, Y0, T_LONG, "auto"),
            # --- mode machine (test 6), 0.4 orbit ---
            "no_switch": sub(run_closed_loop, Y0, T_SHORT, 0, modes=ModeConfig(switch_att_sigma=0.0)),
            "safe": sub(run_closed_loop, Y0, T_SHORT, 0, modes=ModeConfig(safe_rate=1e-6)),
            "triad179": sub(run_closed_loop, Y0, T_SHORT, 0,
                            mekf_cfg=dataclasses.replace(MEKF_G7, triad_min_angle=math.radians(179.0))),
            # --- reproducibility and streams (test 7) ---
            "rep_a": sub(run_closed_loop, Y0, 150.0, 3, sensors=REALISTIC_SENSORS),
            "rep_b": sub(run_closed_loop, Y0, 150.0, 3, sensors=REALISTIC_SENSORS),
            "rep_c": sub(run_closed_loop, Y0, 150.0, 4, sensors=REALISTIC_SENSORS),
            "bias_explicit": sub(run_closed_loop, Y0, 20.0, 5,
                                 sensors=SensorErrors(mag_bias_sigma=200e-9, mag_bias=BIAS_EXPLICIT)),
            # --- inertia (test 8) ---
            "inertia_none": sub(run_closed_loop, Y0, 200.0, 0, inertia_true=None),
            "inertia_nom": sub(run_closed_loop, Y0, 200.0, 0,
                               inertia_true=DEFAULT.spacecraft.inertia.copy()),
            "inertia_110": sub(run_closed_loop, Y0, 200.0, 0,
                               inertia_true=1.1 * DEFAULT.spacecraft.inertia),
        }
        yield jobs
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _res(jobs, name):
    return jobs[name].result()


@pytest.fixture(scope="session")
def pk(cl_jobs):
    return _res(cl_jobs, "pk")


@pytest.fixture(scope="session")
def lownoise(cl_jobs):
    return _res(cl_jobs, "lownoise")


@pytest.fixture(scope="session")
def ideal(cl_jobs):
    return _res(cl_jobs, "ideal")


@pytest.fixture(scope="session")
def real_g7(cl_jobs):
    return _res(cl_jobs, "real_g7")


@pytest.fixture(scope="session")
def design(cl_jobs):
    return _res(cl_jobs, "real_design")


@pytest.fixture(scope="session")
def eph():
    return Ephemeris(R0, V0, T_LONG + 60.0)


def _rms(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x ** 2)))


def pk_window_rms_deg(pk_res):
    """PK nadir RMS over the window pointing and t >= PK switch + 1 orbit (spec test 1)."""
    sw = pk_res.meta["switch_time"]
    win = (np.asarray(pk_res.mode) == 1) & (pk_res.t >= sw + P_ORBIT)
    assert win.any()
    return float(np.degrees(_rms(pk_res.nadir_err[win])))


def know_angle(res):
    """|know_err| from the logged quaternions with scipy (independent of ``know_err``)."""
    ok = np.isfinite(res.q_est[:, 0])
    out = np.full(res.t.size, np.nan)
    rel = Rot.from_quat(res.q_est[ok][:, [1, 2, 3, 0]]).inv() * Rot.from_quat(res.q[ok][:, [1, 2, 3, 0]])
    out[ok] = rel.magnitude()
    return out


# --------------------------------------------------------------------------
# 1. low-noise equivalence with the perfect-knowledge runner
# --------------------------------------------------------------------------
class TestLowNoiseEquivalence:
    def test_pk_reference_is_sane(self, pk):
        # sanity of the oracle itself: PK auto mode switches within the first orbit and points well
        sw = pk.meta["switch_time"]
        assert np.isfinite(sw) and 0.2 * P_ORBIT < sw < 0.6 * P_ORBIT
        assert pk_window_rms_deg(pk) < 0.2

    def test_switch_time_matches_pk(self, lownoise, pk):
        sw = lownoise.meta["switch_time"]
        assert np.isfinite(sw)
        assert abs(sw - pk.meta["switch_time"]) <= 30.0

    def test_steady_nadir_rms_matches_pk(self, lownoise, pk):
        # same window for both runs: pointing and t >= PK switch + 1 orbit
        sw = pk.meta["switch_time"]
        t0 = sw + P_ORBIT
        win_pk = (np.asarray(pk.mode) == 1) & (pk.t >= t0)
        win_cl = (np.asarray(lownoise.mode) == MODE_POINTING) & (lownoise.t >= t0)
        assert win_pk.sum() > 1000 and win_cl.sum() > 1000
        rms_pk = np.degrees(_rms(pk.nadir_err[win_pk]))
        rms_cl = np.degrees(_rms(lownoise.nadir_err[win_cl]))
        assert abs(rms_cl - rms_pk) < 0.01

    def test_summary_rms_consistent_with_hand_window(self, lownoise):
        # the runner's own summary window is (first switch + 1 orbit); recompute it by hand
        s = summarize_closed_loop(lownoise)
        win = (np.asarray(lownoise.mode) == 1) & (lownoise.t >= lownoise.meta["switch_time"] + P_ORBIT)
        assert s["nadir_rms_deg"] == pytest.approx(np.degrees(_rms(lownoise.nadir_err[win])), rel=1e-12)

    def test_knowledge_rms_below_0p02_deg(self, lownoise, pk):
        t0 = pk.meta["switch_time"] + P_ORBIT
        win = (np.asarray(lownoise.mode) == 1) & (lownoise.t >= t0)
        ka = know_angle(lownoise)
        assert np.all(np.isfinite(ka[win]))
        assert np.degrees(_rms(ka[win])) < 0.02
        assert summarize_closed_loop(lownoise)["know_rms_deg"] < 0.02

    def test_first_sample_is_initial_state(self, lownoise):
        np.testing.assert_allclose(lownoise.q[0], Y0[:4], atol=1e-15)
        np.testing.assert_allclose(lownoise.w[0], Y0[4:7], atol=1e-18)
        np.testing.assert_array_equal(lownoise.h_w[0], np.zeros(3))
        assert lownoise.t[0] == 0.0
        assert lownoise.t.size == int(round(T_LONG)) + 1


# --------------------------------------------------------------------------
# 2. ideal sensors
# --------------------------------------------------------------------------
class TestIdealSensors:
    def test_success(self, ideal):
        s = summarize_closed_loop(ideal)
        assert s["success"] is True
        assert s["n_switches"] >= 1 and s["n_fallbacks"] == 0 and s["n_steady"] > 0

    def test_nadir_rms_between_three_pk_and_one_degree(self, ideal, pk):
        s = summarize_closed_loop(ideal)
        assert 3.0 * pk_window_rms_deg(pk) < s["nadir_rms_deg"] < 1.0

    def test_nees_consistent(self, ideal):
        assert 2.2 <= summarize_closed_loop(ideal)["nees_mean"] <= 4.0

    def test_nis_per_dof(self, ideal):
        s = summarize_closed_loop(ideal)
        assert 0.9 <= s["nis_per_dof_mag"] <= 1.1
        assert 0.9 <= s["nis_per_dof_sun"] <= 1.1

    def test_summary_keys_present(self, ideal):
        s = summarize_closed_loop(ideal)
        keys = ["t_init_s", "switch_time_s", "switch_time_orbits", "detumble_time_s",
                "detumble_time_orbits", "n_switches", "n_fallbacks", "n_steady",
                "nadir_rms_deg", "nadir_p95_deg", "nadir_p99_deg", "nadir_max_deg",
                "know_rms_deg", "know_rms_sunlit_deg", "know_rms_eclipse_deg", "know_max_deg",
                "ctrl_rms_deg", "ctrl_max_deg", "nees_mean", "wheel_max_frac",
                "wheel_max_frac_pointing", "dump_frac", "dipole_max", "gyro_bias_err_end_degps",
                "mag_bias_true_nT", "mag_bias_err_end_nT", "nis_per_dof_mag", "nis_per_dof_sun",
                "success"]
        for k in keys:
            assert k in s, k
        # ideal sensors: no magnetometer bias in the truth
        assert s["mag_bias_true_nT"] == 0.0

    def test_result_shapes(self, ideal):
        n = ideal.t.size
        assert ideal.q.shape == (n, 4) and ideal.q_est.shape == (n, 4)
        assert ideal.know_err.shape == (n, 3) and ideal.P_att.shape == (n, 3, 3)
        assert ideal.att_sigma.shape == (n,) and ideal.lit.dtype == bool
        np.testing.assert_allclose(np.linalg.norm(ideal.q, axis=1), 1.0, atol=1e-12)
        ok = np.isfinite(ideal.q_est[:, 0])
        np.testing.assert_allclose(np.linalg.norm(ideal.q_est[ok], axis=1), 1.0, atol=1e-12)


# --------------------------------------------------------------------------
# 3. realistic sensors: the Group 7 filter fails
# --------------------------------------------------------------------------
class TestRealisticG7Fails:
    def test_knowledge_rms_above_5_deg(self, real_g7):
        s = summarize_closed_loop(real_g7)
        assert s["know_rms_deg"] > 5.0

    def test_mag_nis_above_10(self, real_g7):
        s = summarize_closed_loop(real_g7)
        assert s["nis_per_dof_mag"] > 10.0


# --------------------------------------------------------------------------
# 4. realistic sensors: the design filter works
# --------------------------------------------------------------------------
class TestRealisticDesign:
    def test_success(self, design):
        assert summarize_closed_loop(design)["success"] is True

    def test_pointing_accuracy(self, design):
        s = summarize_closed_loop(design)
        assert s["nadir_max_deg"] < 5.0
        assert s["nadir_rms_deg"] < 1.5

    def test_mag_bias_estimated(self, design):
        s = summarize_closed_loop(design)
        assert s["mag_bias_true_nT"] > 10.0                 # the draw is not degenerate
        assert s["mag_bias_err_end_nT"] < 0.3 * s["mag_bias_true_nT"]

    def test_gyro_bias_estimated(self, design):
        assert summarize_closed_loop(design)["gyro_bias_err_end_degps"] < 0.02

    def test_logged_truth_bias_matches_meta(self, design):
        # the logged estimate is in tesla and compared with the truth draw of the same run
        end = np.isfinite(design.bm_est[:, 0])
        err = np.linalg.norm(design.bm_est[end][-1] - design.meta["mag_bias_true"]) * 1e9
        assert err == pytest.approx(summarize_closed_loop(design)["mag_bias_err_end_nT"], rel=1e-12)


# --------------------------------------------------------------------------
# 5. the controller acts on the estimate
# --------------------------------------------------------------------------
class TestControllerActsOnEstimate:
    def test_ctrl_error_much_smaller_than_knowledge_error(self, design):
        s = summarize_closed_loop(design)
        assert s["ctrl_rms_deg"] < 0.5 * s["know_rms_deg"]

    def test_pointing_error_dominated_by_knowledge(self, design):
        s = summarize_closed_loop(design)
        assert abs(s["nadir_rms_deg"] - s["know_rms_deg"]) < 0.3 * s["know_rms_deg"]

    def test_triangle_inequality_logged(self, design):
        ok = np.isfinite(design.q_est[:, 0])
        assert ok.sum() > 10000
        lhs = design.nadir_err[ok]
        rhs = np.linalg.norm(design.know_err[ok], axis=1) + design.est_nadir_err[ok]
        assert np.all(lhs <= rhs + 1e-9)

    def test_triangle_inequality_independent_angles(self, design, eph):
        # angles rebuilt from the logged quaternions: scipy for the truth/estimate angle,
        # dynamics.nadir_angle (own LVLH build from the ephemeris) for the two nadir angles
        ok = np.isfinite(design.q_est[:, 0])
        r, v = eph.at(design.t[ok])
        a_true = dyn.nadir_angle(design.q[ok], r, v)
        a_est = dyn.nadir_angle(design.q_est[ok], r, v)
        a_know = know_angle(design)[ok]
        assert np.all(a_true <= a_know + a_est + 1e-9)
        # the runner's own log agrees with the independent angles
        # (1e-7 rad: ephemeris rebuilt with another t_end, dense-output differences ~1e-9 m)
        np.testing.assert_allclose(design.nadir_err[ok], a_true, atol=1e-7)
        np.testing.assert_allclose(design.est_nadir_err[ok], a_est, atol=1e-7)
        np.testing.assert_allclose(np.linalg.norm(design.know_err[ok], axis=1), a_know, rtol=1e-9,
                                   atol=1e-12)

    def test_attitude_known_error_definition(self, design):
        # q_true = q_est (x) exp(know_err), right/body-frame error (CONVENTIONS.md section 3)
        ok = np.isfinite(design.q_est[:, 0])
        idx = np.nonzero(ok)[0][::500]
        for i in idx:
            qe = Rot.from_quat(design.q_est[i][[1, 2, 3, 0]])
            qt = qe * Rot.from_rotvec(design.know_err[i])
            ref = Rot.from_quat(design.q[i][[1, 2, 3, 0]])
            assert (qt.inv() * ref).magnitude() < 1e-10


# --------------------------------------------------------------------------
# 6. mode machine
# --------------------------------------------------------------------------
class TestModeMachine:
    @pytest.mark.parametrize("name", ["ideal", "real_design", "safe"])
    def test_no_pointing_before_filter_settled(self, cl_jobs, name):
        res = _res(cl_jobs, name)
        mode = np.asarray(res.mode)
        t_init = res.meta["t_init"]
        assert np.isfinite(t_init)
        assert not np.any(mode[res.t < t_init + 300.0] == MODE_POINTING)
        assert res.meta["switch_time"] >= t_init + 300.0

    @pytest.mark.parametrize("name", ["ideal", "real_design", "safe"])
    def test_att_sigma_below_2_deg_at_switch(self, cl_jobs, name):
        res = _res(cl_jobs, name)
        mode = np.asarray(res.mode)
        first = int(np.argmax(mode == MODE_POINTING))
        assert mode[first] == MODE_POINTING and (first == 0 or mode[first - 1] == MODE_DETUMBLE)
        assert res.t[first] == pytest.approx(res.meta["switch_time"], abs=1e-9)
        assert res.att_sigma[first] < math.radians(2.0)
        # att_sigma is sqrt(trace P_att)
        assert res.att_sigma[first] == pytest.approx(math.sqrt(np.trace(res.P_att[first])), rel=1e-9)

    def test_rate_condition_at_switch(self, cl_jobs):
        # |gyro - b_hat| < 0.5 deg/s held for 60 s before the switch (logged w_est)
        res = _res(cl_jobs, "ideal")
        first = int(np.argmax(np.asarray(res.mode) == MODE_POINTING))
        w = np.linalg.norm(res.w_est[first - 60:first + 1], axis=1)
        assert np.all(w < math.radians(0.5))

    def test_switch_att_sigma_zero_never_points(self, cl_jobs):
        res = _res(cl_jobs, "no_switch")
        assert np.all(np.asarray(res.mode) == MODE_DETUMBLE)
        assert math.isnan(res.meta["switch_time"])
        assert res.meta["switch_times"] == []
        assert np.isfinite(res.meta["t_init"])                     # the filter itself did run
        s = summarize_closed_loop(res)
        assert s["success"] is False
        assert s["n_switches"] == 0 and s["n_steady"] == 0
        assert math.isnan(s["switch_time_s"]) and math.isnan(s["nadir_rms_deg"])
        # detumble only: wheels untouched
        assert np.all(res.u_w == 0.0) and np.all(res.h_w == 0.0)

    def test_safe_rate_triggers_fallback(self, cl_jobs):
        res = _res(cl_jobs, "safe")
        mode = np.asarray(res.mode)
        sw = res.meta["switch_times"][0]
        assert np.isfinite(sw)
        assert len(res.meta["fallback_times"]) >= 1
        fb = res.meta["fallback_times"][0]
        safe_hold = ModeConfig().safe_hold
        assert fb > sw
        assert fb - sw <= safe_hold + 2.0
        i_sw = int(np.argmin(np.abs(res.t - sw)))
        i_fb = int(np.argmin(np.abs(res.t - fb)))
        assert np.all(mode[i_sw:i_fb] == MODE_POINTING)            # pointing until the fallback
        assert mode[i_fb] == MODE_DETUMBLE                         # then back in detumble
        s = summarize_closed_loop(res)
        assert s["n_fallbacks"] >= 1
        assert s["success"] is False

    def test_wheels_idle_after_fallback_until_next_switch(self, cl_jobs):
        # in detumble the wheel torque is zero; the dipole is allowed again
        res = _res(cl_jobs, "safe")
        mode = np.asarray(res.mode)
        det = mode == MODE_DETUMBLE
        assert np.all(res.u_w[det] == 0.0)

    def test_triad_guard_179_never_initialises(self, cl_jobs):
        res = _res(cl_jobs, "triad179")
        assert math.isnan(res.meta["t_init"])
        assert np.all(np.isnan(res.q_est))
        assert np.all(np.isnan(res.att_sigma))
        assert np.all(np.asarray(res.mode) == MODE_DETUMBLE)
        assert math.isnan(res.meta["switch_time"])
        s = summarize_closed_loop(res)
        assert s["success"] is False and math.isnan(s["t_init_s"])
        assert math.isnan(s["gyro_bias_err_end_degps"]) and math.isnan(s["mag_bias_err_end_nT"])


# --------------------------------------------------------------------------
# 7. reproducibility and random streams
# --------------------------------------------------------------------------
class TestReproducibilityAndStreams:
    def test_equal_seeds_identical(self, cl_jobs):
        a, b = _res(cl_jobs, "rep_a"), _res(cl_jobs, "rep_b")
        for name in ("q", "w", "m", "gyro_bias", "tau_mag", "nadir_err"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
        np.testing.assert_array_equal(a.q_est, b.q_est)            # NaN pattern identical too
        np.testing.assert_array_equal(a.meta["mag_bias_true"], b.meta["mag_bias_true"])

    def test_different_seeds_differ(self, cl_jobs):
        a, c = _res(cl_jobs, "rep_a"), _res(cl_jobs, "rep_c")
        assert not np.allclose(a.gyro_bias[0], c.gyro_bias[0], rtol=0, atol=0)
        assert not np.allclose(a.meta["mag_bias_true"], c.meta["mag_bias_true"], rtol=0, atol=0)
        assert np.max(np.abs(a.q[-1] - c.q[-1])) > 1e-9           # trajectories diverge (B-dot sees other noise)

    @pytest.mark.parametrize("name,seed", [("rep_a", 3), ("rep_c", 4)])
    def test_gyro_turn_on_bias_stream(self, cl_jobs, name, seed):
        res = _res(cl_jobs, name)
        g = Gyro(GyroConfig(), np.random.default_rng(seed).spawn(4)[0])   # stream 0 = gyro
        np.testing.assert_array_equal(res.gyro_bias[0], g.bias)
        assert np.linalg.norm(res.gyro_bias[0]) > 0.0

    @pytest.mark.parametrize("name,seed", [("rep_a", 3), ("rep_c", 4)])
    def test_mag_bias_draw_stream(self, cl_jobs, name, seed):
        res = _res(cl_jobs, name)
        draw = np.random.default_rng(seed).spawn(4)[3].spawn(3)[0].standard_normal(3)
        expected = REALISTIC_SENSORS.mag_bias_sigma * draw
        np.testing.assert_allclose(res.meta["mag_bias_true"], expected, rtol=1e-14, atol=0)
        s = summarize_closed_loop(res)
        assert s["mag_bias_true_nT"] == pytest.approx(np.linalg.norm(expected) * 1e9, rel=1e-12)

    def test_explicit_mag_bias_used_verbatim(self, cl_jobs):
        res = _res(cl_jobs, "bias_explicit")
        np.testing.assert_array_equal(res.meta["mag_bias_true"], np.asarray(BIAS_EXPLICIT))
        # the explicit bias wins over the (non-zero) sigma draw
        draw = np.random.default_rng(5).spawn(4)[3].spawn(3)[0].standard_normal(3)
        assert not np.allclose(res.meta["mag_bias_true"], 200e-9 * draw, rtol=0, atol=1e-12)
        assert summarize_closed_loop(res)["mag_bias_true_nT"] == pytest.approx(
            np.linalg.norm(BIAS_EXPLICIT) * 1e9, rel=1e-12)


# --------------------------------------------------------------------------
# 8. true inertia
# --------------------------------------------------------------------------
class TestInertia:
    def test_none_equals_nominal_bit_identical(self, cl_jobs):
        a, b = _res(cl_jobs, "inertia_none"), _res(cl_jobs, "inertia_nom")
        for name in ("q", "w", "h_w", "m", "u_w", "gyro_bias", "nadir_err"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
        np.testing.assert_array_equal(a.q_est, b.q_est)
        np.testing.assert_array_equal(a.meta["inertia_true"], DEFAULT.spacecraft.inertia)

    def test_ten_percent_inertia_changes_truth(self, cl_jobs):
        a, c = _res(cl_jobs, "inertia_none"), _res(cl_jobs, "inertia_110")
        np.testing.assert_array_equal(c.meta["inertia_true"], 1.1 * DEFAULT.spacecraft.inertia)
        # first sample is the common initial state; afterwards the body responds differently to the
        # same magnetorquer torque (omega_dot = I^-1 tau): hand estimate >= 1e-5 rad/s after 200 s
        np.testing.assert_array_equal(a.q[0], c.q[0])
        assert np.max(np.abs(a.w[-1] - c.w[-1])) > 1e-7
        assert np.max(np.abs(a.q[-1] - c.q[-1])) > 1e-6

    def test_controller_uses_nominal_inertia_in_bdot_gain(self, cl_jobs):
        # the B-dot gain is computed from the nominal inertia in both runs
        a, c = _res(cl_jobs, "inertia_none"), _res(cl_jobs, "inertia_110")
        assert a.meta["bdot_k"] == c.meta["bdot_k"]


# --------------------------------------------------------------------------
# 9. summarize_closed_loop on hand-built results
# --------------------------------------------------------------------------
def _hand_percentile(vals, p):
    """Linear-interpolation percentile (numpy 'linear' definition) by hand."""
    s = sorted(vals)
    pos = p / 100.0 * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _hand_rms(vals):
    return math.sqrt(sum(v * v for v in vals) / len(vals))


N_H = 20
P_H = 8.0
WMAX_H = 800.0
THR_H = 0.5


def _build(mode, switch_times, fallback_times, t_init=1.0):
    n = N_H
    t = np.arange(n, dtype=float)
    rng = np.random.default_rng(9)
    nadir = 1e-3 * (1.0 + 0.1 * np.arange(n)) ** 1.0                    # rad, strictly increasing
    know = rng.normal(0.0, 2e-3, size=(n, 3))
    diag = rng.uniform(1e-6, 4e-6, size=(n, 3))
    P = np.zeros((n, 3, 3))
    for i in range(n):
        P[i] = np.diag(diag[i])
    est_nadir = 0.2e-3 * (1.0 + np.arange(n) % 5)
    ws = rng.uniform(-300.0, 300.0, size=(n, 3))
    ws[5, 1] = -700.0                                                    # in pointing, outside the window
    ws[15, 2] = 500.0                                                    # in the window
    dump = np.zeros(n, bool)
    dump[[4, 13, 14, 18]] = True
    m = rng.uniform(-0.1, 0.1, size=(n, 3))
    m[3, 0] = 0.19                                                       # outside the window
    m[16, 2] = -0.15                                                     # inside the window
    lit = (np.arange(n) % 3) != 0
    rate = np.ones(n)
    rate[5], rate[6] = 0.3, 0.7                                          # dips below, then above again
    rate[7:] = 0.2
    b_est = rng.normal(0.0, 1e-4, size=(n, 3))
    b_est[:3] = np.nan                                                   # filter not yet initialised
    gyro_bias = rng.normal(0.0, 1e-3, size=(n, 3))
    bm_est = rng.normal(0.0, 1e-7, size=(n, 3))
    bm_est[:3] = np.nan
    mag_bias_true = np.array([1e-7, -2e-7, 2e-7])
    nis = {
        "mag": {"t": np.array([0.0, 100.0, 300.0, 301.0, 350.0, 400.0, 500.0]),
                "nis": np.array([30.0, 9.0, 8.0, 3.0, 6.0, 12.0, 9.0]),
                "dof": np.full(7, 3.0),
                "accepted": np.array([True, True, True, True, False, True, True])},
        "sun": {"t": np.array([302.0, 303.0]), "nis": np.array([2.0, 4.0]), "dof": np.array([2.0, 2.0]),
                "accepted": np.array([True, True])},
    }
    meta = {"orbital_period": P_H, "switch_time": switch_times[0] if switch_times else float("nan"),
            "switch_times": list(switch_times), "fallback_times": list(fallback_times),
            "t_init": t_init, "t_end": float(t[-1]), "ctrl": {"detumble_threshold": THR_H},
            "wheel": {"max_speed": WMAX_H}, "mag_bias_true": mag_bias_true, "nis": nis}
    z3 = np.zeros((n, 3))
    res = ClosedLoopResult(
        t=t, q=np.tile([1.0, 0, 0, 0], (n, 1)), w=z3, h_w=z3, wheel_speed=ws, m=m, u_w=z3,
        tau_dist=z3, tau_mag=z3, nadir_err=nadir, rate=rate, mode=np.asarray(mode, int), dump=dump,
        meta=meta, q_est=np.tile([1.0, 0, 0, 0], (n, 1)), b_est=b_est, bm_est=bm_est,
        att_sigma=np.sqrt(diag.sum(axis=1)), P_att=P, know_err=know, est_nadir_err=est_nadir,
        w_est=z3, gyro_bias=gyro_bias, sun_used=np.zeros(n, int), lit=lit)
    return res, dict(nadir=nadir, know=know, diag=diag, est_nadir=est_nadir, ws=ws, dump=dump, m=m,
                     lit=lit, b_est=b_est, gyro_bias=gyro_bias, bm_est=bm_est, mag_bias_true=mag_bias_true,
                     nis=nis, t=t)


def _mode_with_fallback():
    mode = np.zeros(N_H, int)
    mode[2:11] = 1          # pointing t = 2..10
    mode[11:13] = 0         # fallback at 11, back in detumble for t = 11, 12
    mode[13:] = 1           # pointing again from t = 13
    return mode


class TestSummarizeHandBuilt:
    def test_window_stats_default_settle(self):
        res, d = _build(_mode_with_fallback(), [2.0, 13.0], [11.0])
        s = summarize_closed_loop(res)                     # settle = 1 orbit = 8 s
        # window: pointing and t >= 2 + 8 = 10  ->  t = 10, 13, 14, ..., 19
        idx = [10] + list(range(13, 20))
        assert s["n_steady"] == len(idx) == 8
        e = [math.degrees(d["nadir"][i]) for i in idx]
        assert s["nadir_rms_deg"] == pytest.approx(_hand_rms(e), rel=1e-12)
        assert s["nadir_p95_deg"] == pytest.approx(_hand_percentile(e, 95), rel=1e-12)
        assert s["nadir_p99_deg"] == pytest.approx(_hand_percentile(e, 99), rel=1e-12)
        assert s["nadir_max_deg"] == pytest.approx(max(e), rel=1e-12)
        ke = [math.degrees(math.sqrt(sum(c * c for c in d["know"][i]))) for i in idx]
        assert s["know_rms_deg"] == pytest.approx(_hand_rms(ke), rel=1e-12)
        assert s["know_max_deg"] == pytest.approx(max(ke), rel=1e-12)
        ke_sun = [k for k, i in zip(ke, idx) if d["lit"][i]]
        ke_ecl = [k for k, i in zip(ke, idx) if not d["lit"][i]]
        assert len(ke_sun) > 0 and len(ke_ecl) > 0
        assert s["know_rms_sunlit_deg"] == pytest.approx(_hand_rms(ke_sun), rel=1e-12)
        assert s["know_rms_eclipse_deg"] == pytest.approx(_hand_rms(ke_ecl), rel=1e-12)
        ce = [math.degrees(d["est_nadir"][i]) for i in idx]
        assert s["ctrl_rms_deg"] == pytest.approx(_hand_rms(ce), rel=1e-12)
        assert s["ctrl_max_deg"] == pytest.approx(max(ce), rel=1e-12)

    def test_nees_hand(self):
        res, d = _build(_mode_with_fallback(), [2.0, 13.0], [11.0])
        s = summarize_closed_loop(res)
        idx = [10] + list(range(13, 20))
        vals = [sum(d["know"][i][a] ** 2 / d["diag"][i][a] for a in range(3)) for i in idx]
        assert s["nees_mean"] == pytest.approx(sum(vals) / len(vals), rel=1e-10)

    def test_wheel_dump_dipole(self):
        res, d = _build(_mode_with_fallback(), [2.0, 13.0], [11.0])
        s = summarize_closed_loop(res)
        idx = [10] + list(range(13, 20))
        assert s["wheel_max_frac"] == pytest.approx(max(abs(x) for i in idx for x in d["ws"][i]) / WMAX_H,
                                                    rel=1e-12)
        assert s["wheel_max_frac"] == pytest.approx(500.0 / WMAX_H, rel=1e-12)        # planted maximum
        # over all pointing samples (t = 2..10 and 13..19) the planted -700 at t = 5 dominates
        assert s["wheel_max_frac_pointing"] == pytest.approx(700.0 / WMAX_H, rel=1e-12)
        assert s["dump_frac"] == pytest.approx(sum(1 for i in idx if d["dump"][i]) / len(idx), rel=1e-12)
        assert s["dump_frac"] == pytest.approx(3.0 / 8.0, rel=1e-12)                  # t = 13, 14, 18
        assert s["dipole_max"] == pytest.approx(max(abs(x) for i in idx for x in d["m"][i]), rel=1e-12)
        assert s["dipole_max"] == pytest.approx(0.15, rel=1e-12)                      # m[3] = 0.19 is outside

    def test_times_counts_and_detumble(self):
        res, _ = _build(_mode_with_fallback(), [2.0, 13.0], [11.0])
        s = summarize_closed_loop(res)
        assert s["t_init_s"] == 1.0
        assert s["switch_time_s"] == 2.0 and s["switch_time_orbits"] == pytest.approx(2.0 / P_H)
        assert s["n_switches"] == 2 and s["n_fallbacks"] == 1
        # rate < 0.5 from t = 7 on (t = 5 dips, t = 6 is above): detumble time 7 s
        assert s["detumble_time_s"] == 7.0
        assert s["detumble_time_orbits"] == pytest.approx(7.0 / P_H)
        assert s["success"] is False                                # a fallback happened

    def test_bias_errors_and_nis(self):
        res, d = _build(_mode_with_fallback(), [2.0, 13.0], [11.0])
        s = summarize_closed_loop(res)
        db = d["b_est"][-1] - d["gyro_bias"][-1]
        assert s["gyro_bias_err_end_degps"] == pytest.approx(math.degrees(math.sqrt(sum(x * x for x in db))),
                                                              rel=1e-12)
        dm = d["bm_est"][-1] - d["mag_bias_true"]
        assert s["mag_bias_err_end_nT"] == pytest.approx(1e9 * math.sqrt(sum(x * x for x in dm)), rel=1e-12)
        assert s["mag_bias_true_nT"] == pytest.approx(1e9 * math.sqrt(1 + 4 + 4) * 1e-7, rel=1e-12)   # = 300
        # NIS: accepted and t >= t_init + 300 = 301: samples at 301 (3.0), 400 (12.0), 500 (9.0)
        assert s["nis_per_dof_mag"] == pytest.approx((3.0 + 12.0 + 9.0) / 3.0 / 3.0, rel=1e-12)
        assert s["nis_per_dof_sun"] == pytest.approx((2.0 / 2.0 + 4.0 / 2.0) / 2.0, rel=1e-12)
        assert "nis_per_dof_css" not in s

    def test_custom_settle(self):
        res, d = _build(_mode_with_fallback(), [2.0, 13.0], [11.0])
        s = summarize_closed_loop(res, settle=3.0)                  # window t >= 5, pointing
        idx = list(range(5, 11)) + list(range(13, 20))
        assert s["n_steady"] == len(idx)
        e = [math.degrees(d["nadir"][i]) for i in idx]
        assert s["nadir_rms_deg"] == pytest.approx(_hand_rms(e), rel=1e-12)
        assert s["wheel_max_frac"] == pytest.approx(700.0 / WMAX_H, rel=1e-12)        # t = 5 now inside

    def test_success_true_without_fallback(self):
        mode = np.zeros(N_H, int)
        mode[2:] = 1
        res, d = _build(mode, [2.0], [])
        s = summarize_closed_loop(res)
        assert s["n_switches"] == 1 and s["n_fallbacks"] == 0
        assert s["n_steady"] == 10                                  # t = 10..19
        assert s["success"] is True
        e = [math.degrees(d["nadir"][i]) for i in range(10, 20)]
        assert s["nadir_rms_deg"] == pytest.approx(_hand_rms(e), rel=1e-12)

    def test_no_switch_gives_nan_and_failure(self):
        res, _ = _build(np.zeros(N_H, int), [], [])
        s = summarize_closed_loop(res)
        assert s["n_switches"] == 0 and s["n_steady"] == 0
        assert s["success"] is False
        for k in ("switch_time_s", "switch_time_orbits", "nadir_rms_deg", "nadir_p95_deg",
                  "nadir_p99_deg", "nadir_max_deg", "know_rms_deg", "know_rms_sunlit_deg",
                  "know_rms_eclipse_deg", "know_max_deg", "ctrl_rms_deg", "ctrl_max_deg", "nees_mean",
                  "wheel_max_frac", "dump_frac", "dipole_max", "wheel_max_frac_pointing"):
            assert math.isnan(s[k]), k
        assert math.isfinite(s["gyro_bias_err_end_degps"])         # filter data exist regardless

    def test_switch_but_empty_window_is_failure(self):
        # switch at t = 15, settle 1 orbit (8 s) -> window starts at 23 > t_end = 19
        mode = np.zeros(N_H, int)
        mode[15:] = 1
        res, _ = _build(mode, [15.0], [])
        s = summarize_closed_loop(res)
        assert s["n_switches"] == 1 and s["n_steady"] == 0
        assert s["success"] is False
        assert math.isnan(s["nadir_rms_deg"])
        # wheel metric over all pointing samples still defined
        assert math.isfinite(s["wheel_max_frac_pointing"])
