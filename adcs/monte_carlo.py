"""Monte Carlo campaign for the closed-loop CubeSat ADCS (Project B, Group 8 / M9).

Spec: ``docs/specs/adcs_closed_loop.md`` section 4.

Purpose
-------
Run ``adcs.simulate.run_closed_loop`` (sensors -> MEKF -> mode machine -> B-dot /
nadir pointing) from many randomised initial tumbles and report, per run and in
aggregate, the detumble time, acquisition time, steady-state pointing and
knowledge errors and wheel usage, together with the fraction of runs meeting the
mission requirements in :data:`REQUIREMENTS`.

Randomisation (per run ``i``, ``draw_run``)
-------------------------------------------
* initial body rate magnitude uniform in 2-10 deg/s (stored in rad/s);
* initial attitude uniform on SO(3) (normalised Gaussian 4-vector, ``w >= 0``);
* rate axis uniform on the sphere (body axes);
* independent principal-inertia scale factors, uniform in 1 +/- 10 %, applied to
  the **truth** dynamics only (the controller and filter keep the nominal inertia);
* sensor-noise seed ``(base_seed, i, 1)`` passed to ``run_closed_loop``; it drives
  the gyro/magnetometer/sun-sensor noise streams and, in the ``"design"``
  campaign, the realistic error draws (per-run constant magnetometer bias, the
  unmodelled-field Gauss-Markov series and the albedo Gauss-Markov series).
  The initial-condition draw uses ``default_rng([base_seed, i, 0])``.

Campaigns: ``"design"`` = ``REALISTIC_SENSORS`` + ``MEKF_DESIGN`` (truth-side
sensor errors, design filter); ``"ideal"`` = ``IDEAL_SENSORS`` + ``MEKF_G7``
(Group 7 ideal-sensor floor).

Outputs
-------
``run_campaign`` returns a JSON-ready dict (``campaign``, ``config``, ``runs``,
``aggregate``, ``requirements``, ``wall_time_s``); ``save_json`` writes it with
NaN as ``null``. Units in the run rows: degrees for angles, seconds or orbits as
named by the key, rad/s -> deg/s for ``rate_degps``.

Requirements
------------
The thresholds in :data:`REQUIREMENTS` are **assumed mission requirements set
before the campaign was run** (``assumptions.md`` B-24); they were not tuned to
the results. ``pointing_rms_goal_deg`` is a goal, not a requirement, and is
excluded from ``"all_requirements"``.

CLI
---
``python -m adcs.monte_carlo [--campaign design|ideal|all] [--n 100] [--orbits 6]
[--workers N] [--base-seed 2026] [--out DIR] [--no-figures]``
writes ``<out>/monte_carlo_<campaign>.json`` (default ``DIR`` = ``adcs/report`` of
the repository), prints a summary table and, unless ``--no-figures``, calls
``adcs.report.monte_carlo_figures.main`` on the written files when that module
exists. Runs are distributed over a process pool (Windows spawn: workers are
top-level functions; BLAS threads are limited to one per worker).
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from common import quaternion as quat
from adcs.config import DEFAULT, SimConfig
from adcs.orbit import orbital_period
from adcs.simulate import (IDEAL_SENSORS, MEKF_DESIGN, MEKF_G7, REALISTIC_SENSORS, run_closed_loop,
                           summarize_closed_loop)

__all__ = [
    "CAMPAIGNS",
    "REQUIREMENTS",
    "RunSpec",
    "draw_run",
    "run_one",
    "aggregate",
    "check_requirements",
    "run_campaign",
    "save_json",
    "main",
]

#: Campaign name -> (truth-side sensor errors, MEKF configuration).
CAMPAIGNS = {"design": (REALISTIC_SENSORS, MEKF_DESIGN), "ideal": (IDEAL_SENSORS, MEKF_G7)}

#: Assumed mission requirements, fixed before the campaign (assumptions.md B-24).
REQUIREMENTS = {
    "detumble_orbits_max": 3.0,      # orbits until true |w| < 0.5 deg/s
    "acquire_orbits_max": 4.0,       # orbits until the first switch to pointing
    "pointing_max_deg": 5.0,         # deg, steady-state maximum nadir error
    "knowledge_rms_deg": 2.0,        # deg, steady-state attitude-knowledge RMS
    "pointing_rms_goal_deg": 1.0,    # deg, GOAL (not a requirement)
    "wheel_frac_max": 1.0,           # fraction of max wheel speed in pointing; never saturate
}

_DEFAULT_RATE_RANGE = (float(np.deg2rad(2.0)), float(np.deg2rad(10.0)))
_DEFAULT_INERTIA_TOL = 0.10
_NAN = float("nan")


@dataclass(frozen=True)
class RunSpec:
    """One Monte Carlo run: initial state, truth-inertia scales and seeds.

    ``rate`` rad/s (body rate magnitude), ``y0`` = ``[q (w,x,y,z, body-to-ECI),
    w (rad/s, body axes), h_w (N m s)]`` (10 floats), ``inertia_scale`` = three
    dimensionless factors on the principal moments of inertia (truth only).
    """

    index: int
    base_seed: int
    rate: float
    y0: tuple
    inertia_scale: tuple

    @property
    def sim_seed(self) -> tuple:
        """Seed sequence ``(base_seed, index, 1)`` given to ``run_closed_loop``."""
        return (self.base_seed, self.index, 1)


def draw_run(index: int, base_seed: int = 2026, rate_range=_DEFAULT_RATE_RANGE,
             inertia_tol: float = _DEFAULT_INERTIA_TOL) -> RunSpec:
    """Draw the initial condition of run ``index`` (deterministic in ``(index, base_seed)``).

    ``rng = default_rng([base_seed, index, 0])``; draw order: attitude quaternion
    (normalised ``normal(size=4)``, ``w >= 0``), rate axis (normalised
    ``normal(size=3)``), rate magnitude ``uniform(*rate_range)`` (rad/s), inertia
    scales ``uniform(1 - tol, 1 + tol, size=3)``. ``y0 = [q, rate * axis, 0, 0, 0]``.
    """
    rng = np.random.default_rng([int(base_seed), int(index), 0])
    q = quat.canonicalize(quat.normalize(rng.normal(size=4)))
    axis = rng.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    rate = float(rng.uniform(*rate_range))
    scale = rng.uniform(1.0 - inertia_tol, 1.0 + inertia_tol, size=3)
    y0 = np.concatenate([q, rate * axis, np.zeros(3)])
    return RunSpec(index=int(index), base_seed=int(base_seed), rate=rate,
                   y0=tuple(float(v) for v in y0), inertia_scale=tuple(float(v) for v in scale))


def _py(v):
    """numpy scalars/arrays -> plain Python (NaN kept)."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def run_one(spec: RunSpec, campaign: str = "design", orbits: float = 6.0,
            cfg: SimConfig = DEFAULT) -> dict:
    """Run one closed-loop simulation of ``spec`` for ``orbits`` orbits and summarise it.

    The truth inertia is ``diag(diag(I_nominal) * spec.inertia_scale)`` (kg m^2); the
    controller keeps the nominal inertia. Returns a JSON-ready dict: ``index``,
    ``rate_degps`` (initial rate magnitude, deg/s), ``inertia_scale`` (list), all keys of
    ``summarize_closed_loop`` (deg, s, orbits; NaN where undefined) and ``runtime_s``
    (wall time of this run in s).
    """
    if campaign not in CAMPAIGNS:
        raise ValueError(f"unknown campaign {campaign!r}; expected one of {sorted(CAMPAIGNS)}")
    sensors, mekf_cfg = CAMPAIGNS[campaign]
    t0 = time.perf_counter()
    inertia_true = np.diag(np.diag(cfg.spacecraft.inertia) * np.asarray(spec.inertia_scale, float))
    period = float(orbital_period(cfg.orbit.a))
    res = run_closed_loop(list(spec.y0), orbits * period, seed=list(spec.sim_seed), cfg=cfg,
                          sensors=sensors, mekf_cfg=mekf_cfg, inertia_true=inertia_true, log_every=1)
    summary = {k: _py(v) for k, v in summarize_closed_loop(res).items()}
    row = {"index": int(spec.index), "rate_degps": float(np.rad2deg(spec.rate)),
           "inertia_scale": [float(v) for v in spec.inertia_scale], **summary}
    row["runtime_s"] = float(time.perf_counter() - t0)
    return row


def _is_num(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def aggregate(rows) -> dict:
    """Statistics of every numeric key over ``rows`` (list of ``run_one`` dicts).

    For each numeric (non-bool, scalar) key: ``{"mean", "std", "median", "p5", "p95",
    "min", "max", "n_finite"}`` over the finite values (all-NaN -> NaN stats,
    ``n_finite`` 0; ``std`` is the population standard deviation). Plus
    ``"success_fraction"`` (fraction of rows with a true ``success``; NaN for no rows).
    Units are those of the row values.
    """
    rows = list(rows)
    keys = []
    for r in rows:
        for k, v in r.items():
            if _is_num(v) and k not in keys:
                keys.append(k)
    out = {}
    for k in keys:
        vals = np.array([float(r[k]) for r in rows if k in r and _is_num(r[k])], float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[k] = {"mean": float(vals.mean()), "std": float(vals.std()), "median": float(np.median(vals)),
                      "p5": float(np.percentile(vals, 5)), "p95": float(np.percentile(vals, 95)),
                      "min": float(vals.min()), "max": float(vals.max()), "n_finite": int(vals.size)}
        else:
            out[k] = {s: _NAN for s in ("mean", "std", "median", "p5", "p95", "min", "max")}
            out[k]["n_finite"] = 0
    out["success_fraction"] = (float(np.mean([bool(r.get("success", False)) for r in rows]))
                               if rows else _NAN)
    return out


def check_requirements(rows, req: dict = REQUIREMENTS) -> dict:
    """Fraction of runs meeting each requirement, and the failing run indices.

    Returns ``{name: {"fraction": f, "n_met": int, "n": int, "failing": [indices]}}`` with
    ``name`` in ``detumble_orbits_max`` (``detumble_time_orbits <=``), ``acquire_orbits_max``
    (``switch_time_orbits <=``), ``pointing_max_deg`` (``nadir_max_deg <=``),
    ``knowledge_rms_deg`` (``know_rms_deg <=``), ``pointing_rms_goal_deg``
    (``nadir_rms_deg <=``, a goal), ``wheel_frac_max`` (``wheel_max_frac_pointing <``,
    strict), ``success``, and ``all_requirements`` (met all of the above except the goal).
    NaN or missing metrics count as not met. ``fraction`` is NaN for empty ``rows``.
    """
    rows = list(rows)

    def val(r, key):
        v = r.get(key, _NAN)
        return float(v) if _is_num(v) else _NAN

    def le(key, thr):
        return lambda r: bool(val(r, key) <= thr)  # NaN compares False

    tests = {
        "detumble_orbits_max": le("detumble_time_orbits", req["detumble_orbits_max"]),
        "acquire_orbits_max": le("switch_time_orbits", req["acquire_orbits_max"]),
        "pointing_max_deg": le("nadir_max_deg", req["pointing_max_deg"]),
        "knowledge_rms_deg": le("know_rms_deg", req["knowledge_rms_deg"]),
        "pointing_rms_goal_deg": le("nadir_rms_deg", req["pointing_rms_goal_deg"]),
        "wheel_frac_max": lambda r: bool(val(r, "wheel_max_frac_pointing") < req["wheel_frac_max"]),
        "success": lambda r: bool(r.get("success", False)),
    }
    met = {name: [t(r) for r in rows] for name, t in tests.items()}
    met["all_requirements"] = [all(met[n][i] for n in met if n != "pointing_rms_goal_deg")
                               for i in range(len(rows))]
    idx = [int(r.get("index", i)) for i, r in enumerate(rows)]
    out = {}
    for name, flags in met.items():
        n_met = int(sum(flags))
        out[name] = {"fraction": (n_met / len(rows)) if rows else _NAN, "n_met": n_met, "n": len(rows),
                     "failing": [i for i, f in zip(idx, flags) if not f]}
    return out


def _worker(args) -> dict:
    """Process-pool worker (top-level for pickling): draw and run one index.

    A crashing run is returned as a failed row (``success`` False, ``error`` message) so one
    failure does not discard the rest of a long campaign.
    """
    index, base_seed, campaign, orbits = args
    spec = draw_run(index, base_seed)
    try:
        return run_one(spec, campaign, orbits)
    except Exception as exc:  # noqa: BLE001 - recorded in the result, counted as a failed run
        return {"index": int(index), "rate_degps": float(np.rad2deg(spec.rate)),
                "inertia_scale": [float(v) for v in spec.inertia_scale], "success": False,
                "error": f"{type(exc).__name__}: {exc}", "runtime_s": _NAN}


def run_campaign(campaign: str, n: int = 100, orbits: float = 6.0, workers: int | None = None,
                 base_seed: int = 2026) -> dict:
    """Run ``n`` randomised closed-loop runs of ``campaign`` ("design" or "ideal") in a process pool.

    Each run is ``run_one(draw_run(i, base_seed), campaign, orbits)``; ``workers`` defaults to
    ``max(1, cpu_count - 2)``. Returns ``{"campaign", "config", "runs" (sorted by index),
    "aggregate", "requirements", "wall_time_s"}`` (angles in deg, times in s or orbits).
    """
    if campaign not in CAMPAIGNS:
        raise ValueError(f"unknown campaign {campaign!r}; expected one of {sorted(CAMPAIGNS)}")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    workers = workers or max(1, (os.cpu_count() or 1) - 2)
    sensors, mekf_cfg = CAMPAIGNS[campaign]
    t0 = time.perf_counter()
    jobs = [(i, int(base_seed), campaign, float(orbits)) for i in range(int(n))]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_worker, jobs))
    rows.sort(key=lambda r: r["index"])
    for r in rows:
        if "error" in r:
            print(f"WARNING: run {r['index']} failed: {r['error']}")
    config = {"n": int(n), "orbits": float(orbits), "base_seed": int(base_seed),
              "rate_range_degps": [float(np.rad2deg(v)) for v in _DEFAULT_RATE_RANGE],
              "inertia_tol": _DEFAULT_INERTIA_TOL, "workers": int(workers),
              "sensors": asdict(sensors), "mekf": asdict(mekf_cfg), "requirements": dict(REQUIREMENTS)}
    return {"campaign": campaign, "config": config, "runs": rows, "aggregate": aggregate(rows),
            "requirements": check_requirements(rows), "wall_time_s": float(time.perf_counter() - t0)}


def _clean(o):
    """Recursively convert to JSON-safe Python (NaN/inf -> None, numpy -> Python, tuples -> lists)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    o = _py(o)
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def save_json(result: dict, path) -> None:
    """Write ``result`` to ``path`` as JSON (indent 1): NaN/inf -> ``null``, numpy -> Python."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(result), f, indent=1, allow_nan=False)


def _fmt(x, spec=".3g") -> str:
    return "nan" if x is None or not np.isfinite(x) else format(x, spec)


def _print_summary(res: dict) -> None:
    agg, req = res["aggregate"], res["requirements"]
    cfg = res["config"]
    print(f"\n=== campaign '{res['campaign']}': n={cfg['n']}, {cfg['orbits']:g} orbits, "
          f"wall {res['wall_time_s']:.0f} s ===")
    print(f"success fraction: {_fmt(agg['success_fraction'], '.3f')}")
    print("requirements (fraction of runs met; failing run indices):")
    for name, d in req.items():
        fail = d["failing"]
        shown = (str(fail) if len(fail) <= 10 else str(fail[:10])[:-1] + ", ...]")
        goal = " (goal)" if name == "pointing_rms_goal_deg" else ""
        print(f"  {name:<24s}{goal:<7s} {_fmt(d['fraction'], '.3f'):>6s}  fail {shown}")
    print(f"{'metric':<24s}{'median':>10s}{'p95':>10s}{'max':>10s}")
    for key in ("nadir_rms_deg", "nadir_max_deg", "detumble_time_orbits"):
        a = agg.get(key)
        if a:
            print(f"{key:<24s}{_fmt(a['median']):>10s}{_fmt(a['p95']):>10s}{_fmt(a['max']):>10s}")


def main(argv=None) -> int:
    """CLI entry point (see module docstring). Returns 0."""
    p = argparse.ArgumentParser(prog="python -m adcs.monte_carlo", description=__doc__.split("\n")[0])
    p.add_argument("--campaign", choices=["design", "ideal", "all"], default="all")
    p.add_argument("--n", type=int, default=100, help="runs per campaign")
    p.add_argument("--orbits", type=float, default=6.0, help="orbits simulated per run")
    p.add_argument("--workers", type=int, default=None, help="processes (default cpu_count - 2)")
    p.add_argument("--base-seed", type=int, default=2026)
    p.add_argument("--out", default=None, help="output directory (default: adcs/report of the repository)")
    p.add_argument("--no-figures", action="store_true", help="do not call adcs.report.monte_carlo_figures")
    args = p.parse_args(argv)

    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent / "report"
    names = ["design", "ideal"] if args.campaign == "all" else [args.campaign]
    paths = []
    for name in names:
        res = run_campaign(name, n=args.n, orbits=args.orbits, workers=args.workers, base_seed=args.base_seed)
        path = out_dir / f"monte_carlo_{name}.json"
        save_json(res, path)
        paths.append(str(path))
        _print_summary(res)
        print(f"wrote {path}")

    if not args.no_figures:
        spec = importlib.util.find_spec("adcs.report.monte_carlo_figures")
        if spec is None:
            print("note: adcs.report.monte_carlo_figures not found; skipping figures")
        else:
            importlib.import_module("adcs.report.monte_carlo_figures").main(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
