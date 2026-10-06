"""Tests for ``adcs.simulate`` (perfect-knowledge runner), spec tests 12-16.

Written from ``docs/specs/adcs_control.md`` sections 3 and 4, not from the
implementation.  Oracles: hand values (wheel limits, thresholds), an independent
nadir error from ``adcs.orbit.Ephemeris`` + ``adcs.dynamics.nadir_angle`` (the
runner's own ``nadir_err`` is only cross-checked against it), the inertial
angular-momentum conservation law for internal wheel torques, explicit kinetic
energy, and scipy ``Rotation`` for the ``lvlh_state`` rate.  Spec thresholds are
used as written.

Each scenario is run once in a module-scoped fixture (a ``run`` costs ~4 s setup
plus ~1 ms per control period).
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from adcs import dynamics as dyn
from adcs.actuators import WheelConfig
from adcs.config import DEFAULT
from adcs.orbit import Ephemeris, initial_state, orbital_period
from common.frames import GM_EARTH
from adcs.simulate import (
    ControlConfig,
    SimResult,
    detumble_time,
    lvlh_state,
    run,
    tumble_state,
)

I_BOX = DEFAULT.spacecraft.inertia
A_ORBIT = DEFAULT.orbit.a
P_ORBIT = float(orbital_period(A_ORBIT))
R0, V0 = initial_state(DEFAULT)
WHEEL = WheelConfig()
H_MAX = 1.77e-3
MAX_SPEED = 8000.0 * 2.0 * math.pi / 60.0              # hand value, rad/s
JS = H_MAX / MAX_SPEED                                  # hand value, kg m^2
THRESH = math.radians(0.5)


def kinetic_energy(w):
    w = np.asarray(w, float)
    return 0.5 * np.einsum("...i,ij,...j->...", w, I_BOX, w)


def wheel_fraction(res):
    """max_i |Omega_i| / max_speed, with Omega = h/J_s - w from the logged h_w and w (hand J_s)."""
    om = res.h_w / JS - res.w
    return np.max(np.abs(om), axis=1) / MAX_SPEED


def independent_nadir_deg(res, eph):
    r, v = eph.at(res.t)
    return np.degrees(dyn.nadir_angle(res.q, r, v))


def last_time_above(t, rate, thr):
    """Own oracle for the detumble time: first sample after the last sample with rate >= thr."""
    above = np.nonzero(rate >= thr)[0]
    if above.size == 0:
        return float(t[0])
    if above[-1] == len(rate) - 1:
        return float("nan")
    return float(t[above[-1] + 1])


@pytest.fixture(scope="module")
def eph():
    return Ephemeris(R0, V0, P_ORBIT + 60.0)


def _dump_y0():
    y0 = lvlh_state(R0, V0)
    y0[8] = 0.9 * H_MAX                                              # y wheel preloaded to 0.9 h_max
    return y0


@pytest.fixture(scope="module")
def sim_jobs():
    """Start the five long scenarios once, in parallel worker processes (each ``run`` is an
    independent, deterministic simulation; running them serially would take ~2 min).

    Yields a dict name -> (y0, Future[SimResult]).  BLAS threads are limited to 1 in the
    workers (set before they start) to avoid oversubscription on many-core machines.
    """
    keys = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    old = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ[k] = "1"
    ex = ProcessPoolExecutor(max_workers=min(5, os.cpu_count() or 1))
    try:
        y_det = tumble_state(math.radians(10.0), seed=0)
        y_pt = lvlh_state(R0, V0, offset=np.deg2rad([10.0, 0.0, 0.0]))
        y_dmp = _dump_y0()
        y_auto = tumble_state(math.radians(0.3), seed=5)
        jobs = {
            "detumble": (y_det, ex.submit(run, y_det, 2.0 * P_ORBIT, "detumble")),
            "pointing_clean": (y_pt, ex.submit(
                run, y_pt, P_ORBIT, "pointing",
                ctrl=ControlConfig(disturbances=False, dumping=False))),
            "dump_off": (y_dmp, ex.submit(
                run, y_dmp, P_ORBIT, "pointing", ctrl=ControlConfig(dumping=False))),
            "dump_on": (y_dmp, ex.submit(
                run, y_dmp, P_ORBIT, "pointing", ctrl=ControlConfig(dumping=True))),
            "auto": (y_auto, ex.submit(run, y_auto, 600.0, "auto")),
        }
        yield jobs
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _get(jobs, name):
    y0, fut = jobs[name]
    return y0, fut.result()


# --------------------------------------------------------------------------
# 12. detumble
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def detumble_result(sim_jobs):
    return _get(sim_jobs, "detumble")


class TestDetumble:
    def test_initial_state(self, detumble_result):
        y0, res = detumble_result
        assert res.t[0] == 0.0
        np.testing.assert_allclose(res.w[0], y0[4:7], atol=0)
        assert res.rate[0] == pytest.approx(math.radians(10.0), rel=1e-12)

    def test_detumbles_within_two_orbits(self, detumble_result):
        _, res = detumble_result
        assert res.t[-1] == pytest.approx(2.0 * P_ORBIT, abs=1.0)
        td = last_time_above(res.t, res.rate, THRESH)                # own oracle
        assert math.isfinite(td)
        assert td < 2.0 * P_ORBIT
        # the runner's helper agrees with the oracle
        assert detumble_time(res.t, res.rate, THRESH) == pytest.approx(td, abs=1e-9)
        # M7 exit criterion (1-3 orbits) is not met trivially early
        assert td > 0.0

    def test_final_kinetic_energy_below_one_percent(self, detumble_result):
        _, res = detumble_result
        ratio = kinetic_energy(res.w[-1]) / kinetic_energy(res.w[0])
        assert ratio < 0.01

    def test_wheels_untouched(self, detumble_result):
        _, res = detumble_result
        assert np.all(res.u_w == 0.0)
        assert np.all(res.h_w == 0.0)
        np.testing.assert_allclose(res.wheel_speed, -res.w, rtol=1e-12, atol=1e-15)   # h_w = 0

    def test_dipole_limit_and_mode(self, detumble_result):
        _, res = detumble_result
        assert np.max(np.abs(res.m)) <= 0.2 * (1 + 1e-12)
        assert np.max(np.abs(res.m)) > 0.05                          # the coils really worked
        assert np.all(res.mode == 0)
        assert not np.any(res.dump)

    def test_logged_torque_consistent_with_dipole(self, detumble_result):
        # tau_mag = m x b_b is perpendicular to the dipole
        _, res = detumble_result
        dot = np.einsum("ij,ij->i", res.tau_mag, res.m)
        scale = np.linalg.norm(res.tau_mag, axis=1) * np.linalg.norm(res.m, axis=1)
        assert np.all(np.abs(dot) <= 1e-12 * scale + 1e-30)
        assert np.max(np.linalg.norm(res.tau_mag, axis=1)) < 0.2 * math.sqrt(3.0) * 7e-5      # |m| |B| bound, |B| < 70 uT

    def test_quaternion_unit_norm(self, detumble_result):
        _, res = detumble_result
        np.testing.assert_allclose(np.linalg.norm(res.q, axis=1), 1.0, atol=1e-12)


# --------------------------------------------------------------------------
# 13. pointing without disturbances or dumping
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pointing_clean_result(sim_jobs):
    return _get(sim_jobs, "pointing_clean")


class TestPointingClean:
    def test_nadir_error_below_hundredth_degree_after_1000_s(self, pointing_clean_result, eph):
        _, res = pointing_clean_result
        err = independent_nadir_deg(res, eph)
        assert err[0] == pytest.approx(10.0, rel=1e-9)               # started 10 deg off
        late = res.t > 1000.0
        assert late.any()
        assert np.max(err[late]) < 0.01
        # the runner's own log agrees with the independent evaluation
        np.testing.assert_allclose(np.degrees(res.nadir_err), err, atol=1e-6)

    def test_inertial_angular_momentum_conserved(self, pointing_clean_result):
        # no disturbances, no magnetorquer: only internal wheel torques act
        y0, res = pointing_clean_result
        y = np.concatenate([res.q, res.w, res.h_w], axis=1)
        h_i = dyn.angular_momentum_inertial(y, I_BOX)
        h0 = np.linalg.norm(h_i[0])
        assert h0 > 1e-6
        drift = np.max(np.abs(h_i - h_i[0]))
        assert drift < 1e-9 * h0

    def test_no_dipole_and_pointing_mode(self, pointing_clean_result):
        _, res = pointing_clean_result
        assert np.all(res.mode == 1)
        assert np.all(res.m == 0.0)
        assert not np.any(res.dump)
        assert np.all(res.tau_dist == 0.0)

    def test_wheels_within_limits(self, pointing_clean_result):
        _, res = pointing_clean_result
        assert np.max(np.abs(res.u_w)) <= WHEEL.max_torque * (1 + 1e-12)
        assert np.max(wheel_fraction(res)) < 0.5                     # small initial offset, no dumping needed

    def test_logged_wheel_speed_matches_hand_formula(self, pointing_clean_result):
        _, res = pointing_clean_result
        np.testing.assert_allclose(res.wheel_speed, res.h_w / JS - res.w, rtol=1e-9, atol=1e-6)


# --------------------------------------------------------------------------
# 14. momentum dumping
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dump_off_result(sim_jobs):
    return _get(sim_jobs, "dump_off")


@pytest.fixture(scope="module")
def dump_on_result(sim_jobs):
    return _get(sim_jobs, "dump_on")


class TestMomentumDumping:
    def test_preload_is_90_percent(self, dump_off_result, dump_on_result):
        for y0, res in (dump_off_result, dump_on_result):
            frac0 = wheel_fraction(res)[0]
            assert frac0 == pytest.approx(0.9, abs=1e-3)
            assert frac0 <= 0.9 + 1e-3

    def test_dumping_off_saturates_and_pointing_is_lost(self, dump_off_result, eph):
        _, res = dump_off_result
        frac = wheel_fraction(res)
        assert np.max(frac) >= 0.99
        i_sat = int(np.argmax(frac >= 0.99))
        err = independent_nadir_deg(res, eph)
        assert np.max(err[i_sat:]) > 10.0
        # without dumping the magnetorquers stay idle
        assert np.all(res.m == 0.0)
        assert not np.any(res.dump)

    def test_dumping_on_keeps_wheels_below_start_level_and_unloads(self, dump_on_result):
        _, res = dump_on_result
        frac = wheel_fraction(res)
        assert np.max(frac) <= 0.9 + 1e-3
        assert frac[-1] <= 0.3
        assert np.any(res.dump)
        assert np.max(np.abs(res.m)) > 0.0
        assert np.max(np.abs(res.m)) <= 0.2 * (1 + 1e-12)

    def test_dumping_on_keeps_pointing(self, dump_on_result, eph):
        _, res = dump_on_result
        err = independent_nadir_deg(res, eph)
        late = res.t > 2000.0
        assert np.max(err[late]) < 1.0
        np.testing.assert_allclose(np.degrees(res.nadir_err), err, atol=1e-6)

    @pytest.mark.parametrize("which", ["off", "on"])
    def test_wheel_speed_never_beyond_limit(self, which, dump_off_result, dump_on_result):
        _, res = dump_off_result if which == "off" else dump_on_result
        om = res.h_w / JS - res.w                                    # hand formula
        assert np.max(np.abs(om)) <= MAX_SPEED * (1.0 + 1e-5)
        assert np.max(np.abs(res.wheel_speed)) <= MAX_SPEED * (1.0 + 1e-5)

    @pytest.mark.parametrize("which", ["off", "on"])
    def test_wheel_torque_limit(self, which, dump_off_result, dump_on_result):
        _, res = dump_off_result if which == "off" else dump_on_result
        assert np.max(np.abs(res.u_w)) <= WHEEL.max_torque * (1 + 1e-12)

    def test_dump_flag_follows_hysteresis(self, dump_on_result):
        # on at >= 0.3 of max speed, off at <= 0.05 (spec ControlConfig defaults); the flag at
        # sample k is decided from the wheel speed at sample k (hand-recomputed here).
        _, res = dump_on_result
        frac = wheel_fraction(res)
        for k in range(len(frac)):
            # only the unambiguous parts: a sample above the on level has dumping, below off none
            if frac[k] > 0.3 + 1e-6:
                assert res.dump[k]
            if frac[k] < 0.05 - 1e-6:
                assert not res.dump[k]


# --------------------------------------------------------------------------
# 15. auto mode
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def auto_result(sim_jobs):
    return _get(sim_jobs, "auto")


class TestAutoMode:
    def test_switches_exactly_once_from_detumble_to_pointing(self, auto_result):
        _, res = auto_result
        mode = np.asarray(res.mode)
        assert mode[0] == 0
        assert mode[-1] == 1
        assert np.count_nonzero(np.diff(mode) != 0) == 1
        assert np.all(np.diff(mode) >= 0)

    def test_switch_time_is_hold_time(self, auto_result):
        # |w| = 0.3 deg/s < 0.5 deg/s threshold from t = 0, detumble_hold = 60 s
        _, res = auto_result
        assert res.meta["switch_time"] == pytest.approx(60.0, abs=1e-9)
        first_pointing = res.t[int(np.argmax(np.asarray(res.mode) == 1))]
        assert first_pointing == pytest.approx(60.0, abs=1e-9)
        assert np.all(np.asarray(res.mode)[res.t < 60.0] == 0)

    def test_wheels_idle_during_detumble_phase(self, auto_result):
        _, res = auto_result
        det = np.asarray(res.mode) == 0
        assert np.all(res.u_w[det] == 0.0)
        assert np.all(res.h_w[det] == 0.0)
        assert np.max(np.abs(res.m[det])) <= 0.2 * (1 + 1e-12)

    def test_wheels_active_and_magnetorquers_idle_after_switch(self, auto_result):
        _, res = auto_result
        pt = np.asarray(res.mode) == 1
        assert np.max(np.abs(res.u_w[pt])) > 0.0
        assert np.max(np.abs(res.u_w[pt])) <= WHEEL.max_torque * (1 + 1e-12)
        # wheel fraction is far below the dump-on level (0.3) in 540 s from a slow tumble
        assert not np.any(res.dump[pt])
        assert np.all(res.m[pt] == 0.0)


# --------------------------------------------------------------------------
# 16. plumbing
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def plumbing_results():
    y0 = lvlh_state(R0, V0, offset=np.deg2rad([5.0, -3.0, 2.0]))
    full = run(y0, 40.0, "pointing", log_every=1)
    sub = run(y0, 40.0, "pointing", log_every=10)
    return y0, full, sub


class TestPlumbing:
    def test_result_type_and_shapes_log_every_10(self, plumbing_results):
        _, full, sub = plumbing_results
        assert isinstance(sub, SimResult)
        n = 40 // 10 + 1
        assert sub.t.shape == (n,)
        np.testing.assert_allclose(sub.t, 10.0 * np.arange(n), atol=1e-9)
        assert sub.q.shape == (n, 4)
        for name in ("w", "h_w", "wheel_speed", "m", "u_w", "tau_dist", "tau_mag"):
            assert getattr(sub, name).shape == (n, 3), name
        for name in ("nadir_err", "rate", "mode", "dump"):
            assert getattr(sub, name).shape == (n,), name
        assert isinstance(sub.meta, dict) and "switch_time" in sub.meta
        assert sub.dump.dtype == bool
        # full-rate log: 40 control periods -> 41 samples
        assert full.t.shape == (41,)
        np.testing.assert_allclose(full.t, np.arange(41.0), atol=1e-9)
        assert full.q.shape == (41, 4)

    def test_log_every_is_a_pure_subsample(self, plumbing_results):
        _, full, sub = plumbing_results
        np.testing.assert_allclose(sub.t, full.t[::10], atol=1e-9)
        np.testing.assert_allclose(sub.q, full.q[::10], atol=1e-12)
        np.testing.assert_allclose(sub.w, full.w[::10], atol=1e-12)
        np.testing.assert_allclose(sub.h_w, full.h_w[::10], atol=1e-15)
        np.testing.assert_allclose(sub.rate, full.rate[::10], atol=1e-12)

    def test_rate_is_norm_of_w_and_nadir_err_consistent(self, plumbing_results, eph):
        _, full, _ = plumbing_results
        np.testing.assert_allclose(full.rate, np.linalg.norm(full.w, axis=1), rtol=1e-14, atol=0)
        err = np.degrees(dyn.nadir_angle(full.q, *eph.at(full.t)))
        np.testing.assert_allclose(np.degrees(full.nadir_err), err, atol=1e-6)

    def test_first_sample_is_initial_state(self, plumbing_results):
        y0, full, _ = plumbing_results
        np.testing.assert_allclose(full.q[0], y0[:4], atol=1e-15)
        np.testing.assert_allclose(full.w[0], y0[4:7], atol=1e-18)
        np.testing.assert_allclose(full.h_w[0], y0[7:], atol=1e-18)

    def test_bad_mode_raises(self):
        y0 = lvlh_state(R0, V0)
        with pytest.raises(ValueError):
            run(y0, 10.0, "bogus")

    def test_run_does_not_mutate_initial_state(self):
        y0 = lvlh_state(R0, V0, offset=np.deg2rad([1.0, 2.0, 3.0]))
        y_copy = y0.copy()
        run(y0, 3.0, "pointing")
        np.testing.assert_array_equal(y0, y_copy)

    # ---- detumble_time hand cases -----------------------------------------------------------
    def test_detumble_time_never_below_is_nan(self):
        t = np.arange(5.0) + 5.0
        assert math.isnan(detumble_time(t, np.ones(5), 0.5))

    def test_detumble_time_last_sample_not_below_is_nan(self):
        t = np.arange(5.0) + 5.0
        assert math.isnan(detumble_time(t, [0.1, 0.1, 0.1, 0.1, 0.9], 0.5))

    def test_detumble_time_always_below_is_first_time(self):
        t = np.arange(5.0) + 5.0
        assert detumble_time(t, 0.1 * np.ones(5), 0.5) == 5.0

    def test_detumble_time_dips_below_then_above(self):
        t = np.arange(5.0) + 5.0
        # below at samples 1, 2, above at 3, below at 4 -> only the last sample stays below
        assert detumble_time(t, [1.0, 0.1, 0.1, 1.0, 0.1], 0.5) == 9.0
        # monotonic decay crossing between samples 1 and 2
        assert detumble_time(t, [1.0, 0.9, 0.1, 0.05, 0.01], 0.5) == 7.0

    def test_detumble_time_threshold_is_strict(self):
        t = np.arange(4.0)
        # rate == threshold is not "below"
        assert detumble_time(t, [1.0, 0.5, 0.2, 0.1], 0.5) == 2.0
        assert math.isnan(detumble_time(t, [1.0, 0.7, 0.6, 0.5], 0.5))

    def test_detumble_time_random_vs_oracle(self):
        rng = np.random.default_rng(16)
        for _ in range(200):
            n = int(rng.integers(2, 40))
            t = np.sort(rng.uniform(0.0, 100.0, n))
            rate = rng.uniform(0.0, 1.0, n)
            if rng.random() < 0.5:
                rate[-int(rng.integers(1, n)):] *= 0.2               # force a below-threshold tail
            exp = last_time_above(t, rate, 0.5)
            got = detumble_time(t, rate, 0.5)
            if math.isnan(exp):
                assert math.isnan(got)
            else:
                assert got == exp

    # ---- lvlh_state ----------------------------------------------------------------------
    def test_lvlh_state_zero_offset_has_zero_nadir_error(self):
        y0 = lvlh_state(R0, V0)
        assert y0.shape == (10,)
        assert np.linalg.norm(y0[:4]) == pytest.approx(1.0, abs=1e-14)
        assert float(dyn.nadir_angle(y0[:4], R0, V0)) <= 1e-12
        n = math.sqrt(GM_EARTH / A_ORBIT**3)
        np.testing.assert_allclose(y0[4:7], [0.0, -n, 0.0], rtol=1e-6, atol=1e-12)   # w_ref in LVLH axes
        np.testing.assert_array_equal(y0[7:], np.zeros(3))

    def test_lvlh_state_offset_and_rates(self):
        off = np.deg2rad([20.0, -15.0, 30.0])
        w_rel = np.deg2rad([0.1, -0.1, 0.2])
        y0 = lvlh_state(R0, V0, offset=off, w_rel=w_rel)
        err = dyn.lvlh_error(y0[:4], R0, V0)
        np.testing.assert_allclose(err, off, atol=1e-12)
        assert float(dyn.nadir_angle(y0[:4], R0, V0)) == pytest.approx(np.linalg.norm(off), abs=1e-12)
        w0 = y0[4:7] - w_rel                                         # = R(exp(off))^T w_ref
        w_ref = lvlh_state(R0, V0)[4:7]
        np.testing.assert_allclose(w0, Rot.from_rotvec(off).inv().apply(w_ref), atol=1e-14)
        np.testing.assert_array_equal(y0[7:], np.zeros(3))

    # ---- tumble_state --------------------------------------------------------------------
    def test_tumble_state_norms_and_zero_wheels(self):
        for seed in range(20):
            y = tumble_state(math.radians(10.0), seed=seed)
            assert y.shape == (10,)
            assert np.linalg.norm(y[:4]) == pytest.approx(1.0, abs=1e-14)
            assert np.linalg.norm(y[4:7]) == pytest.approx(math.radians(10.0), rel=1e-14)
            np.testing.assert_array_equal(y[7:], np.zeros(3))

    def test_tumble_state_reproducible_per_seed(self):
        a = tumble_state(0.1, seed=3)
        b = tumble_state(0.1, seed=3)
        c = tumble_state(0.1, seed=4)
        np.testing.assert_array_equal(a, b)
        assert not np.allclose(a[:4], c[:4])
        assert not np.allclose(a[4:7], c[4:7])

    def test_tumble_state_rate_scales_same_axis_and_attitude(self):
        a = tumble_state(0.1, seed=7)
        b = tumble_state(0.4, seed=7)
        np.testing.assert_allclose(a[:4], b[:4], atol=0)
        np.testing.assert_allclose(4.0 * a[4:7], b[4:7], rtol=1e-14)

    def test_tumble_state_axis_and_attitude_are_uniform(self):
        # 2000 fixed seeds -> deterministic; sample mean of an isotropic unit vector has
        # sd = sqrt(1/3 / 2000) = 0.013 per component, so 0.06 is > 4.5 sigma.
        axes = np.empty((2000, 3))
        zs = np.empty((2000, 3))
        for s in range(2000):
            y = tumble_state(1.0, seed=s)
            axes[s] = y[4:7]
            zs[s] = Rot.from_quat(np.concatenate([y[1:4], y[:1]])).apply([0.0, 0.0, 1.0])
        assert np.max(np.abs(axes.mean(axis=0))) < 0.06
        assert np.max(np.abs(zs.mean(axis=0))) < 0.06
        # second moments of an isotropic unit vector: E[x_i x_j] = delta_ij / 3
        cov = axes.T @ axes / 2000.0
        np.testing.assert_allclose(cov, np.eye(3) / 3.0, atol=0.04)

    def test_control_config_defaults(self):
        c = ControlConfig()
        assert c.bdot_k is None
        assert c.dumping is True
        assert c.disturbances is True
        assert c.dump_k == pytest.approx(1e-3, rel=1e-15)
        assert c.dump_on == pytest.approx(0.3, rel=1e-15)
        assert c.dump_off == pytest.approx(0.05, rel=1e-15)
        assert c.detumble_threshold == pytest.approx(math.radians(0.5), rel=1e-14)
        assert c.detumble_hold == pytest.approx(60.0, rel=1e-15)
