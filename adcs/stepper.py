"""Step-by-step closed-loop ADCS engine (Group 10, M12).

:class:`ClosedLoopSim` is the body of :func:`adcs.simulate.run_closed_loop` re-hosted as a
stateful object: the constructor does the precomputation over the fixed horizon ``[0, t_end]``,
:meth:`~ClosedLoopSim.step` processes samples, :meth:`~ClosedLoopSim.kick` injects an
instantaneous angular-rate impulse on the truth, :meth:`~ClosedLoopSim.snapshot` describes the
last processed sample and :meth:`~ClosedLoopSim.result` returns the logged rows.
``run_closed_loop`` is a thin wrapper, so there is one code path and the results are
bit-for-bit those of the pre-refactor function. Spec: ``docs/specs/stepper.md``.

Frames and units as in ``CONVENTIONS.md``: quaternions are scalar-first Hamilton, body-to-ECI;
SI units, radians, body-axes rates.

Sample ``k`` (``0 <= k <= n_steps``) is the work done at ``t_k = k T``: sensors -> estimator ->
mode machine -> control -> logging, then (if ``k < n_steps``) the RK4 integration of the truth
from ``t_k`` to ``t_{k+1}``. The pending truth state therefore sits at ``t_k`` before sample
``k`` is processed.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from common import quaternion as quat
from common.frames import gmst
from adcs.actuators import (MagnetorquerConfig, WheelConfig, limit_wheel_torque,
                            wheel_speed)
from adcs.config import DEFAULT, SimConfig
from adcs.control import (bdot_dipole, bdot_gain, dump_active, dump_dipole,
                          nadir_reference, pointing_torque)
from adcs.disturbances import total_disturbance_torque
from adcs.dynamics import _cross, rk4_step
from adcs.environment import FieldTable, sunlit
from adcs.estimation import MEKF, MekfConfig, css_vector_complete, triad_init, vector_angle
from adcs.orbit import Ephemeris, initial_state, orbital_period
from adcs.sensors import (Gyro, GyroConfig, MagnetometerConfig, SunSensorConfig, albedo_currents,
                          gauss_markov, magnetometer, sun_sensor_currents, sun_vector_from_currents)
from adcs.simulate import (IDEAL_SENSORS, MEKF_G7, MODE_DETUMBLE, MODE_POINTING, ClosedLoopResult,
                           ControlConfig, ModeConfig, SensorErrors, _rot_inv)

__all__ = ["ClosedLoopSim", "MAX_KICK_RATE"]

#: Largest total body-rate norm a kick may produce, rad/s (30 deg/s). Assumption [B-25]: RK4 with
#: the 0.8 s + 0.2 s sub-steps loses energy non-physically at high rates, and the gyro mean-rate
#: model wraps above pi rad/s.
MAX_KICK_RATE = float(np.deg2rad(30.0))

_FAILED_MSG = "simulation failed; build a new one"


_FAILED_MSG = "simulation failed; build a new one"


class ClosedLoopSim:
    """Closed-loop ADCS simulation that advances sample by sample.

    ``y0 = [q, w, h_w]`` (10,): body-to-ECI quaternion, body rate in rad/s (body axes) and total
    wheel momentum in N m s (body axes). Parameters are those of
    :func:`adcs.simulate.run_closed_loop`. The constructor precomputes the environment over
    ``[0, t_end]`` and processes no samples; the horizon cannot be extended.
    """

    def __init__(self, y0, t_end: float, seed: int | None = 0, cfg: SimConfig = DEFAULT,
                 ctrl: ControlConfig = ControlConfig(), mtq: MagnetorquerConfig = MagnetorquerConfig(),
                 wheel: WheelConfig = WheelConfig(), sensors: SensorErrors = IDEAL_SENSORS,
                 mekf_cfg: MekfConfig = MEKF_G7, modes: ModeConfig = ModeConfig(),
                 mag_cfg: MagnetometerConfig = MagnetometerConfig(),
                 css_cfg: SunSensorConfig = SunSensorConfig(), gyro_cfg: GyroConfig = GyroConfig(),
                 inertia_true=None, log_every: int = 1):
        T = float(mtq.control_period)
        n_steps = int(round(float(t_end) / T))
        if n_steps < 1:
            raise ValueError("t_end shorter than one control period")
        if log_every < 1:
            raise ValueError("log_every must be >= 1")
        t_grid = np.arange(n_steps + 1) * T
        inertia_nom = cfg.spacecraft.inertia
        inertia = inertia_nom if inertia_true is None else np.asarray(inertia_true, float)
        k_bdot = (bdot_gain(inertia_nom, cfg.orbit.a, cfg.orbit.inc)
                  if ctrl.bdot_k is None else float(ctrl.bdot_k))

        rng_g, rng_m, rng_s, rng_truth = np.random.default_rng(seed).spawn(4)
        rng_bias, rng_field, rng_alb = rng_truth.spawn(3)

        # --- truth environment and onboard models -----------------------------
        r0, v0 = initial_state(cfg)
        eph = Ephemeris(r0, v0, t_grid[-1] + 10.0)
        table = FieldTable(eph, cfg, t_grid[-1] + 10.0)
        if sensors.onboard_field_degree == table.max_degree:
            table_ob = table
        else:
            table_ob = FieldTable(eph, cfg, t_grid[-1] + 10.0, max_degree=sensors.onboard_field_degree)
        r_all, v_all = eph.at(t_grid)
        b_ob_all = table_ob.b_eci(t_grid)                    # filter reference field
        b_all = table.b_eci(t_grid) + gauss_markov(n_steps + 1, T, sensors.field_residual_sigma,
                                                    sensors.field_residual_tau, rng_field)
        s_all = table.sun_dir_eci(t_grid)
        lit_all = sunlit(r_all, s_all)
        qref_all, wref_all = nadir_reference(r_all, v_all)
        if sensors.albedo > 0.0:
            a_all = np.clip(sensors.albedo + gauss_markov(n_steps + 1, T, sensors.albedo_sigma,
                                                          sensors.albedo_tau, rng_alb, dim=1)[:, 0],
                            *sensors.albedo_clip)
        else:
            a_all = None
        bias_draw = rng_bias.standard_normal(3)
        mag_bias = (np.asarray(sensors.mag_bias, float) if sensors.mag_bias is not None
                    else sensors.mag_bias_sigma * bias_draw)
        css_true = SunSensorConfig(sigma_diode=css_cfg.sigma_diode, threshold=css_cfg.threshold,
                                   fov=sensors.css_fov)
        gyro = Gyro(gyro_cfg, rng_g)

        on_t = float(mtq.on_time)
        off_t = T - on_t
        y = np.asarray(y0, dtype=float).copy()
        y[0:4] = quat.normalize(y[0:4])

        # --- immutable configuration and precomputed environment --------------
        self._T = T
        self._n_steps = n_steps
        self._t_grid = t_grid
        self._cfg = cfg
        self._ctrl = ctrl
        self._mtq = mtq
        self._wheel = wheel
        self._sensors = sensors
        self._mekf_cfg = mekf_cfg
        self._modes = modes
        self._mag_cfg = mag_cfg
        self._seed = seed
        self._log_every = log_every
        self._inertia_nom = inertia_nom
        self._inertia = inertia
        self._k_bdot = k_bdot
        self._rng_m = rng_m
        self._rng_s = rng_s
        self._r_all, self._v_all, self._b_all, self._s_all = r_all, v_all, b_all, s_all
        self._b_ob_all = b_ob_all
        self._lit_all = lit_all
        self._qref_all, self._wref_all = qref_all, wref_all
        self._a_all = a_all
        self._mag_bias = mag_bias
        self._css_true = css_true
        self._gyro = gyro
        self._on_t = on_t
        self._off_t = off_t
        for arr in (t_grid, r_all, v_all, s_all, lit_all):
            arr.flags.writeable = False

        # --- loop state --------------------------------------------------------
        self._k = 0
        self._y = y
        self._f: MEKF | None = None
        self._k_init = -1
        self._cur_mode = MODE_DETUMBLE
        self._below_since = None
        self._above_since = None
        self._switch_time = float("nan")
        self._switch_times: list = []
        self._fallback_times: list = []
        self._dumping = False
        self._mag_prev = None
        self._q_prev = None
        self._kicks: list = []
        self._last = None          # references to the most recently processed sample
        self._failed = False       # latched when an exception escapes the per-sample loop
        self._failed = False       # latched when an exception escapes the per-sample loop

        # --- log buffers -------------------------------------------------------
        n_log = n_steps // log_every + 1
        self._vec = {name: np.full((n_log, d), np.nan) for name, d in
                     (("q", 4), ("w", 3), ("h_w", 3), ("wheel_speed", 3), ("m", 3), ("u_w", 3),
                      ("tau_dist", 3), ("tau_mag", 3), ("q_est", 4), ("b_est", 3), ("bm_est", 3),
                      ("know_err", 3), ("w_est", 3), ("gyro_bias", 3))}
        self._sca = {name: np.full(n_log, np.nan) for name in
                     ("t", "nadir_err", "est_nadir_err", "att_sigma")}
        self._log_mode = np.zeros(n_log, dtype=int)
        self._log_dump = np.zeros(n_log, dtype=bool)
        self._log_used = np.zeros(n_log, dtype=int)
        self._log_lit = np.zeros(n_log, dtype=bool)
        self._log_P = np.full((n_log, 3, 3), np.nan)
        self._j = 0

    # ------------------------------------------------------------------ properties

    @property
    def T(self) -> float:
        """Control period, s."""
        return self._T

    @property
    def n_steps(self) -> int:
        """Number of integration periods (samples = ``n_steps + 1``)."""
        return self._n_steps

    @property
    def t_end(self) -> float:
        """Horizon ``t_grid[-1]``, s."""
        return float(self._t_grid[-1])

    @property
    def period(self) -> float:
        """Orbital period of the configured orbit, s."""
        return float(orbital_period(self._cfg.orbit.a))

    @property
    def k(self) -> int:
        """Index of the next sample to process (``n_steps + 1`` when done)."""
        return self._k

    @property
    def t(self) -> float:
        """Time of the pending truth state, ``t_grid[min(k, n_steps)]``, s."""
        return float(self._t_grid[min(self._k, self._n_steps)])

    @property
    def failed(self) -> bool:
        """True once an exception escaped the per-sample loop of :meth:`step` (latched)."""
        return self._failed

    @property
    def failed(self) -> bool:
        """True once an exception escaped the per-sample loop of :meth:`step` (latched)."""
        return self._failed

    @property
    def done(self) -> bool:
        """True once the last sample (``k = n_steps``) has been processed."""
        return self._k > self._n_steps

    @property
    def remaining(self) -> int:
        """Samples still to process, ``n_steps + 1 - k``."""
        return self._n_steps + 1 - self._k

    @property
    def state(self) -> np.ndarray:
        """Copy of the pending truth state ``[q, w, h_w]`` (10,): -, rad/s, N m s (body axes)."""
        return self._y.copy()

    @property
    def t_grid(self) -> np.ndarray:
        """Sample times, s, shape (n_steps + 1,). Read-only."""
        return self._t_grid

    @property
    def r_eci(self) -> np.ndarray:
        """Spacecraft position in ECI, m, shape (n_steps + 1, 3). Read-only."""
        return self._r_all

    @property
    def v_eci(self) -> np.ndarray:
        """Spacecraft velocity in ECI, m/s, shape (n_steps + 1, 3). Read-only."""
        return self._v_all

    @property
    def sun_eci(self) -> np.ndarray:
        """Unit Sun direction in ECI, shape (n_steps + 1, 3). Read-only."""
        return self._s_all

    @property
    def lit(self) -> np.ndarray:
        """Truth shadow flag (True = sunlit), shape (n_steps + 1,). Read-only."""
        return self._lit_all

    # ------------------------------------------------------------------ stepping

    def step(self, n: int = 1) -> int:
        """Process up to ``n`` samples; return how many were processed (``min(n, remaining)``).

        Raises ``ValueError`` unless ``n`` is an int >= 1 (bool rejected; no state is touched and
        nothing latches), ``RuntimeError("simulation failed; build a new one")`` once
        :attr:`failed`, and ``RuntimeError("horizon reached")`` if the sim is done. Any exception
        escaping the per-sample loop sets :attr:`failed` and is re-raised: the RNG streams, the
        MEKF and the logs have advanced past the committed ``k``, so a retry is not possible.
        """
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError("n must be an integer >= 1")
        if self._failed:
            raise RuntimeError(_FAILED_MSG)
        if self._k > self._n_steps:
            raise RuntimeError("horizon reached")
        try:
            return self._run(int(n))
        except BaseException:
            self._failed = True
            raise

    def _run(self, n: int) -> int:
        """Per-sample loop of :meth:`step` (``n`` already validated); returns samples processed."""
        k0 = self._k
        k1 = k0 + min(n, self._n_steps + 1 - k0)     # exclusive

        T, n_steps, t_grid = self._T, self._n_steps, self._t_grid
        cfg, ctrl, mtq, wheel = self._cfg, self._ctrl, self._mtq, self._wheel
        mekf_cfg, modes, mag_cfg = self._mekf_cfg, self._modes, self._mag_cfg
        inertia_nom, inertia, k_bdot = self._inertia_nom, self._inertia, self._k_bdot
        rng_m, rng_s = self._rng_m, self._rng_s
        r_all, v_all, b_all, s_all = self._r_all, self._v_all, self._b_all, self._s_all
        b_ob_all, lit_all = self._b_ob_all, self._lit_all
        qref_all, wref_all, a_all = self._qref_all, self._wref_all, self._a_all
        mag_bias, css_true, gyro = self._mag_bias, self._css_true, self._gyro
        on_t, off_t, log_every = self._on_t, self._off_t, self._log_every
        switch_times, fallback_times = self._switch_times, self._fallback_times
        vec, sca = self._vec, self._sca
        log_mode, log_dump, log_used = self._log_mode, self._log_dump, self._log_used
        log_lit, log_P = self._log_lit, self._log_P

        y = self._y
        f = self._f
        k_init = self._k_init
        cur_mode = self._cur_mode
        below_since = self._below_since
        above_since = self._above_since
        switch_time = self._switch_time
        dumping = self._dumping
        mag_prev = self._mag_prev
        q_prev = self._q_prev
        j = self._j

        for k in range(k0, k1):
            t = t_grid[k]
            q, w, h = y[0:4], y[4:7], y[7:10]
            r, v, b_eci, s_eci = r_all[k], v_all[k], b_all[k], s_all[k]

            # ---- 1. sensors ---------------------------------------------------
            gyro_k = None
            if q_prev is not None:
                w_mean = quat.to_rotvec(quat.multiply(quat.conjugate(q_prev), q)) / T
                gyro_k = gyro.sample(w_mean, T)
            mag_k = magnetometer(_rot_inv(q, b_eci), rng_m, mag_cfg, bias=mag_bias)
            alb = None if a_all is None else albedo_currents(q, r, s_eci, a_all[k])
            currents = sun_sensor_currents(_rot_inv(q, s_eci), bool(lit_all[k]), rng_s, css_true,
                                           albedo=alb)
            sun_k, sun_valid = sun_vector_from_currents(currents, css_true)
            sun_valid = bool(sun_valid)

            # ---- 2. estimator -------------------------------------------------
            used = 0
            if f is None:
                ok = sun_valid
                if ok and mekf_cfg.css_mode == "diodes":
                    ok = css_vector_complete(currents, mekf_cfg.diode_min_current)
                if ok and mekf_cfg.triad_min_angle > 0.0:
                    ok = vector_angle(mag_k, sun_k) >= mekf_cfg.triad_min_angle
                if ok:
                    q0, P0 = triad_init(mag_k, sun_k, b_ob_all[k], s_eci, mekf_cfg)
                    f = MEKF(q0, np.zeros(3), P0, mekf_cfg)
                    k_init = k
            elif gyro_k is not None:
                n_sun = len(f.ekf.history.get("sun", ()))
                n_css = len(f.ekf.history.get("css", ()))
                f.step(gyro_k, T, mag_k, b_ob_all[k], sun_k, s_eci, sun_valid, currents, r, t)
                if len(f.ekf.history.get("sun", ())) > n_sun:
                    used = 1
                hist = f.ekf.history.get("css", ())
                if len(hist) > n_css:
                    used = int(hist[-1].residual.size)

            # ---- 3. mode machine on the estimate ------------------------------
            if f is not None:
                q_hat, b_hat, bm_hat = f._q, f._b, f._bm
                w_hat = None if gyro_k is None else gyro_k - b_hat
            else:
                q_hat = b_hat = None
                bm_hat = np.zeros(3)
                w_hat = gyro_k
            rate_hat = np.inf if w_hat is None else float(np.linalg.norm(w_hat))
            if cur_mode == MODE_DETUMBLE:
                ready = (f is not None and (k - k_init) * T >= modes.switch_min_filter_time
                         and f.att_sigma < modes.switch_att_sigma)
                if ready and rate_hat < ctrl.detumble_threshold:
                    below_since = t if below_since is None else below_since
                    if t - below_since >= ctrl.detumble_hold:
                        cur_mode = MODE_POINTING
                        switch_times.append(float(t))
                        if np.isnan(switch_time):
                            switch_time = float(t)
                        above_since = None
                else:
                    below_since = None
            else:
                if rate_hat > modes.safe_rate:
                    above_since = t if above_since is None else above_since
                    if t - above_since >= modes.safe_hold:
                        cur_mode = MODE_DETUMBLE
                        fallback_times.append(float(t))
                        below_since = None
                        dumping = False
                else:
                    above_since = None

            # ---- 4. control from measurements / estimates ---------------------
            m = np.zeros(3)
            u_w = np.zeros(3)
            if cur_mode == MODE_DETUMBLE:
                if mag_prev is not None:
                    m = bdot_dipole(mag_k, mag_prev, T, k_bdot, mtq.m_max)
            else:
                b_corr = mag_k - bm_hat
                omega = wheel_speed(h, w, wheel)               # tachometers (truth)
                frac = float(np.max(np.abs(omega))) / wheel.max_speed
                dumping = ctrl.dumping and dump_active(dumping, frac, ctrl.dump_on, ctrl.dump_off)
                if dumping:
                    m = dump_dipole(h, b_corr, ctrl.dump_k, mtq.m_max)
                tau, _ = pointing_torque(q_hat, w_hat, h, qref_all[k], wref_all[k], inertia_nom, ctrl.gains)
                tau_mag_ff = mtq.duty * _cross(m, b_corr)
                u_w = limit_wheel_torque(-(tau - tau_mag_ff), h, w, wheel, dt=T)
            mag_prev = mag_k
            q_prev = q.copy()

            tau_d = (total_disturbance_torque(r, v, q, b_eci, cfg.spacecraft)
                     if ctrl.disturbances else np.zeros(3))

            if k % log_every == 0:
                vec["q"][j], vec["w"][j], vec["h_w"][j] = q, w, h
                vec["wheel_speed"][j] = wheel_speed(h, w, wheel)
                vec["m"][j], vec["u_w"][j], vec["tau_dist"][j] = m, u_w, tau_d
                vec["tau_mag"][j] = _cross(m, _rot_inv(q, b_eci))
                vec["gyro_bias"][j] = gyro.bias
                if w_hat is not None:
                    vec["w_est"][j] = w_hat
                sca["t"][j] = t
                dq = quat.multiply(quat.conjugate(qref_all[k]), q)
                sca["nadir_err"][j] = np.linalg.norm(quat.to_rotvec(dq))
                if f is not None:
                    vec["q_est"][j], vec["b_est"][j], vec["bm_est"][j] = q_hat, b_hat, bm_hat
                    vec["know_err"][j] = quat.error_rotvec(q_hat, q)
                    sca["att_sigma"][j] = f.att_sigma
                    log_P[j] = f.ekf._P[:3, :3]
                    dqe = quat.multiply(quat.conjugate(qref_all[k]), q_hat)
                    sca["est_nadir_err"][j] = np.linalg.norm(quat.to_rotvec(dqe))
                log_mode[j] = cur_mode
                log_dump[j] = dumping
                log_used[j] = used
                log_lit[j] = lit_all[k]
                j += 1
            if k == n_steps:
                break

            def ext_on(_t, qq, _w, _h, m=m, b_eci=b_eci, tau_d=tau_d):
                return _cross(m, _rot_inv(qq, b_eci)) + tau_d

            y = rk4_step(t, y, on_t, inertia, u_w, ext_on)
            if off_t > 0.0:
                y = rk4_step(t + on_t, y, off_t, inertia, u_w, tau_d)

        # ---- write back the loop state; the last sample's values are kept by reference
        # (rk4_step returns new arrays, kick() copies, the MEKF rebinds _q/_b/_bm), so
        # snapshot() can derive everything lazily.
        self._last = (k1 - 1, q, w, h, m, u_w, tau_d, cur_mode, dumping, used,
                      q_hat, b_hat, bm_hat, w_hat)
        self._y = y
        self._f = f
        self._k_init = k_init
        self._cur_mode = cur_mode
        self._below_since = below_since
        self._above_since = above_since
        self._switch_time = switch_time
        self._dumping = dumping
        self._mag_prev = mag_prev
        self._q_prev = q_prev
        self._j = j
        self._k = k1
        return k1 - k0

    # ------------------------------------------------------------------ disturbance

    def kick(self, dw_body) -> None:
        """Instantaneous rate impulse ``dw_body`` (3,), rad/s in body axes, on the truth.

        Adds to the pending state's body rate (``q`` and the total wheel momentum ``h_w`` are
        unchanged, so the impulse is ``I_true dw``). The next sample processed still sees the
        un-kicked gyro mean; the integration from ``t_k`` starts from the kicked rate. Recorded
        in ``result().meta["kicks"]``.

        Raises, in this order, with the state unchanged: ``ValueError`` if ``dw_body`` is not a
        finite (3,) vector; ``RuntimeError("simulation failed; build a new one")`` once
        :attr:`failed`; ``RuntimeError("horizon reached")`` if done or if ``k == n_steps`` (no
        integration is left, so the kick could only alter the last logged rate); ``ValueError``
        if ``||w + dw||`` exceeds :data:`MAX_KICK_RATE` (30 deg/s, assumption [B-25]: RK4 with
        0.8 s + 0.2 s sub-steps is not energy-faithful at high rates and the gyro mean-rate model
        wraps above pi rad/s).
        """
        dw = np.asarray(dw_body, float)
        if dw.shape != (3,) or not np.all(np.isfinite(dw)):
            raise ValueError("dw_body must be a finite vector of shape (3,)")
        if self._failed:
            raise RuntimeError(_FAILED_MSG)
        if self._k >= self._n_steps:
            raise RuntimeError("horizon reached")
        w_new = self._y[4:7] + dw
        if not float(np.linalg.norm(w_new)) <= MAX_KICK_RATE:
            raise ValueError(f"kick would give |w| = {np.rad2deg(np.linalg.norm(w_new)):.2f} deg/s "
                             f"> {np.rad2deg(MAX_KICK_RATE):.0f} deg/s")
        y = self._y.copy()
        y[4:7] = w_new
        self._y = y
        self._kicks.append({"t": self.t, "k": int(self._k), "dw": [float(c) for c in dw]})

    # ------------------------------------------------------------------ outputs

    def snapshot(self) -> dict:
        """Description of the most recently processed sample ``k_last = k - 1`` (copies).

        Values equal what the logger writes for that sample. Estimator fields are ``None``
        (vectors) or ``nan`` (scalars) before the MEKF exists. Raises ``RuntimeError`` if no
        sample has been processed and ``RuntimeError("simulation failed; build a new one")`` once
        :attr:`failed`. Units: s, rad, rad/s, N m, N m s, A m^2, m, m/s, T.
        """
        if self._failed:
            raise RuntimeError(_FAILED_MSG)
        if self._last is None:
            raise RuntimeError("no sample processed yet")
        (k, q, w, h, m, u_w, tau_d, cur_mode, dumping, used,
         q_hat, b_hat, bm_hat, w_hat) = self._last
        wheel, f = self._wheel, self._f
        t = float(self._t_grid[k])
        b_eci = self._b_all[k]
        qref = self._qref_all[k]
        dq = quat.multiply(quat.conjugate(qref), q)
        ready = f is not None
        snap = {
            "k": int(k),
            "t": t,
            "q": q.copy(), "w": w.copy(), "h_w": h.copy(),
            "wheel_speed": wheel_speed(h, w, wheel),
            "rate": float(np.linalg.norm(w)),
            "m": np.array(m, float), "u_w": np.array(u_w, float),
            "tau_dist": np.array(tau_d, float),
            "tau_mag": _cross(m, _rot_inv(q, b_eci)),
            "mode": int(cur_mode),
            "dump": bool(dumping),
            "lit": bool(self._lit_all[k]),
            "sun_used": int(used),
            "r_eci": self._r_all[k].copy(), "v_eci": self._v_all[k].copy(),
            "sun_eci": self._s_all[k].copy(), "b_eci": b_eci.copy(),
            "q_ref": qref.copy(),
            "gmst": float(gmst(self._cfg.jd_epoch + t / 86400.0)),
            "nadir_err": float(np.linalg.norm(quat.to_rotvec(dq))),
            "filter_ready": ready,
            "q_est": None, "b_est": None, "bm_est": None,
            "w_est": None if w_hat is None else np.array(w_hat, float),
            "know_err": None,
            "att_sigma": float("nan"),
            "P_att": None,
            "est_nadir_err": float("nan"),
            "t_init": float(self._t_grid[self._k_init]) if self._k_init >= 0 else float("nan"),
            "switch_times": list(self._switch_times),
            "fallback_times": list(self._fallback_times),
            "n_kicks": len(self._kicks),
        }
        if ready:
            snap["q_est"] = np.array(q_hat, float)
            snap["b_est"] = np.array(b_hat, float)
            snap["bm_est"] = np.array(bm_hat, float)
            snap["know_err"] = quat.error_rotvec(q_hat, q)
            snap["att_sigma"] = float(f.att_sigma)
            snap["P_att"] = np.array(f.ekf._P[:3, :3], float)
            dqe = quat.multiply(quat.conjugate(qref), q_hat)
            snap["est_nadir_err"] = float(np.linalg.norm(quat.to_rotvec(dqe)))
        return snap

    def result(self) -> ClosedLoopResult:
        """Logged rows so far (``[0:j]``) and ``meta``, as ``run_closed_loop`` builds them.

        Callable at any time and repeatedly; arrays are independent copies, so later ``step()``
        calls do not change a result already returned. Still works after a failure (rows logged
        before it). ``meta["t_end"]`` is the horizon, and
        ``meta["kicks"]`` lists the kicks (``t`` s, ``k``, ``dw`` rad/s body axes).
        """
        f, t_grid, vec, sca = self._f, self._t_grid, self._vec, self._sca
        cfg = self._cfg
        nis = {}
        if f is not None:
            for name in ("mag", "sun", "css"):
                lg = f.ekf.innovations(name)
                nis[name] = {"t": lg.t, "nis": lg.nis, "dof": lg.dof, "accepted": lg.accepted}
        wheel = self._wheel
        meta = {
            "mode": "closed_loop",
            "seed": self._seed,
            "t_end": float(t_grid[-1]),
            "control_period": self._T,
            "bdot_k": self._k_bdot,
            "switch_time": self._switch_time,
            "switch_times": list(self._switch_times),
            "fallback_times": list(self._fallback_times),
            "t_init": float(t_grid[self._k_init]) if self._k_init >= 0 else float("nan"),
            "orbital_period": float(orbital_period(cfg.orbit.a)),
            "mag_bias_true": self._mag_bias.copy(),
            "inertia_true": self._inertia.copy(),
            "nis": nis,
            "ctrl": asdict(self._ctrl),
            "sensors": asdict(self._sensors),
            "mekf": asdict(self._mekf_cfg),
            "modes": asdict(self._modes),
            "mtq": asdict(self._mtq),
            "wheel": asdict(wheel) | {"max_speed": wheel.max_speed, "spin_inertia": wheel.spin_inertia},
            "kicks": [{"t": kk["t"], "k": kk["k"], "dw": list(kk["dw"])} for kk in self._kicks],
        }
        sl = slice(0, self._j)
        return ClosedLoopResult(
            t=sca["t"][sl].copy(), q=vec["q"][sl].copy(), w=vec["w"][sl].copy(),
            h_w=vec["h_w"][sl].copy(), wheel_speed=vec["wheel_speed"][sl].copy(),
            m=vec["m"][sl].copy(), u_w=vec["u_w"][sl].copy(),
            tau_dist=vec["tau_dist"][sl].copy(), tau_mag=vec["tau_mag"][sl].copy(),
            nadir_err=sca["nadir_err"][sl].copy(),
            rate=np.linalg.norm(vec["w"][sl], axis=1), mode=self._log_mode[sl].copy(),
            dump=self._log_dump[sl].copy(), meta=meta,
            q_est=vec["q_est"][sl].copy(), b_est=vec["b_est"][sl].copy(),
            bm_est=vec["bm_est"][sl].copy(), att_sigma=sca["att_sigma"][sl].copy(),
            P_att=self._log_P[sl].copy(), know_err=vec["know_err"][sl].copy(),
            est_nadir_err=sca["est_nadir_err"][sl].copy(), w_est=vec["w_est"][sl].copy(),
            gyro_bias=vec["gyro_bias"][sl].copy(), sun_used=self._log_used[sl].copy(),
            lit=self._log_lit[sl].copy())
