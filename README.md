# Space Estimation & Control — PX4 Sensor Fusion + 3U CubeSat ADCS

> Skeleton. Finalised in Group 9 (headline numbers, one-command reproduction).

Two linked pure-Python projects built on a shared error-state EKF and quaternion spine (`common/`).

- **Project A — PX4 sensor fusion.** A 15-state error-state EKF (position, velocity, attitude, gyro bias, accelerometer bias) fuses IMU, GPS, barometer and magnetometer data from real PX4 `.ulg` flight logs and is validated against the onboard EKF2 estimate.
- **Project B — 3U CubeSat ADCS.** A closed-loop attitude determination and control system (ADCS) simulation for a 3U CubeSat in a 400 km, 51.6° orbit: B-dot detumble, reaction-wheel nadir pointing with magnetorquer momentum dumping, TRIAD initialisation and a 6-state multiplicative EKF (MEKF), evaluated with a Monte Carlo campaign.

Both projects use scalar-first Hamilton quaternions and a 3-element attitude error in the filter covariance, implemented once in `common/`.

## Status

See [`PROGRESS.md`](PROGRESS.md) for per-group status, recorded numbers and open issues.

## Repository layout

```
README.md  CONVENTIONS.md  assumptions.md  PROGRESS.md  requirements.txt  pyproject.toml
.claude/agents/            # project subagent definitions
docs/IMPLEMENTATION_PLAN.md
common/        quaternion.py  ekf.py  frames.py  plotting.py  tests/
sensor_fusion/ download_logs.py  loader.py  ekf_drone.py  run_flight.py  tests/  report/  data/ (gitignored)
adcs/          config.py  orbit.py  environment.py  dynamics.py  disturbances.py  actuators.py
               sensors.py  estimation.py  control.py  simulate.py  monte_carlo.py  visualize.py  tests/  report/
```

## Conventions

Full definitions in [`CONVENTIONS.md`](CONVENTIONS.md). Summary:

- SI units internally; angles in radians, degrees only in plots and reports. Time on axis 0 of all arrays.
- Quaternions are scalar-first `[w, x, y, z]`, Hamilton product, unit norm, body-to-reference (matches PX4 `vehicle_attitude.q` and SciPy `Rotation` with reordering).
- Filters carry a 3-element body-frame rotation-vector attitude error, Joseph-form covariance update, NIS consistency checks.
- Frames: ECI/ECEF/LLA (Project B), NED + FRD body (Project A), LVLH for nadir pointing.
- It is **attitude** determination and control; "altitude" means genuine height only.

All physical assumptions are listed in [`assumptions.md`](assumptions.md) and cited by ID in reports.

## Setup (Windows)

```powershell
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest
```

If a dependency lacks a Python 3.14 wheel, use `py -3.12` for the virtual environment instead.

## Reproducing results

### Project A — PX4 sensor fusion

TBD(Group 4): commands to download pinned logs and run `sensor_fusion.run_flight` per log.

### Project B — CubeSat ADCS

TBD(Group 8): commands to run `adcs.simulate` and `adcs.monte_carlo`.

## Headline results

- Project A: TBD(Group 4: position/attitude RMSE vs EKF2 and NIS consistency per flight).
- Project B: TBD(Group 8: detumble time, closed-loop nadir pointing error, Monte Carlo statistics).
