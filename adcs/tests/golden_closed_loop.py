"""Golden closed-loop outputs: capture and bit-for-bit comparison (Group 10 equivalence gate).

``capture`` runs every case in :data:`CASES` with ``adcs.simulate.run_closed_loop`` and stores
the flattened results in one ``.npz`` (loadable with ``allow_pickle=False``). ``compare`` checks a
fresh result against the stored arrays: same key set, same dtype/shape, identical bytes.

The file is bit-exact only on its capture platform: use :func:`environment_matches` (python, numpy,
scipy versions and ``platform.machine()`` from ``info_json``) to skip the equivalence tests with an
explicit reason elsewhere.

RE-CAPTURE ONLY FROM PRE-REFACTOR CODE. The oracle must come from the original
``adcs.simulate.run_closed_loop`` (``git show 137bc1d:adcs/simulate.py``), never from the
stepper-backed wrapper that is under test. Capturing with the current wrapper would make the
equivalence gate vacuous. The case ``lvlh_wheels_fallback`` was added later by loading that
pre-refactor module and passing its ``run_closed_loop`` as ``runner`` to :func:`run_case`.

Run as ``python -m adcs.tests.golden_closed_loop [--out PATH]`` (set ``OPENBLAS_NUM_THREADS=1``);
note that this uses the *current* ``run_closed_loop`` and is therefore only valid on pre-refactor code.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from adcs.actuators import WheelConfig
from adcs.config import DEFAULT
from adcs.orbit import initial_state, orbital_period
from adcs.simulate import (IDEAL_SENSORS, MEKF_DESIGN, MEKF_G7, REALISTIC_SENSORS,
                           ClosedLoopResult, ControlConfig, ModeConfig, lvlh_state,
                           run_closed_loop, tumble_state)

P = orbital_period(DEFAULT.orbit.a)

DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "closed_loop_golden.npz"


def _spec(seed, sensors, mekf_cfg, t_end, log_every, y0=None, **extra):
    """Case definition. ``y0`` is an optional zero-argument builder of the initial state
    (default: ``tumble_state(10 deg/s, seed)``); ``extra`` are further ``run_closed_loop`` kwargs
    (``ctrl``, ``inertia_true``, ``modes``, ``mtq``, ...)."""
    return {"seed": seed, "sensors": sensors, "mekf_cfg": mekf_cfg, "t_end": t_end,
            "log_every": log_every, "y0": y0, "extra": extra}


def _y0_lvlh_wheels() -> np.ndarray:
    """Near-LVLH start with wheels pre-loaded to 0.4-0.6 of rated momentum (N m s, body axes)."""
    r0, v0 = initial_state(DEFAULT)
    y0 = lvlh_state(r0, v0)
    hmax = WheelConfig().rated_momentum
    y0[7:10] = [0.6 * hmax, -0.5 * hmax, 0.4 * hmax]
    return y0


CASES: dict[str, dict] = {
    "ideal_g7_s0": _spec(0, IDEAL_SENSORS, MEKF_G7, 2.0 * P, 10),
    "ideal_g7_s1": _spec(1, IDEAL_SENSORS, MEKF_G7, 2.0 * P, 10),
    "real_design_s0": _spec(0, REALISTIC_SENSORS, MEKF_DESIGN, 2.0 * P, 10),
    "real_design_s1": _spec(1, REALISTIC_SENSORS, MEKF_DESIGN, 2.0 * P, 10),
    "real_design_s0_le1": _spec(0, REALISTIC_SENSORS, MEKF_DESIGN, 600.0, 1),
    "real_design_s2_inertia_nodump": _spec(
        2, REALISTIC_SENSORS, MEKF_DESIGN, 2.0 * P, 10,
        ctrl=ControlConfig(dumping=False), inertia_true=1.1 * DEFAULT.spacecraft.inertia),
    # Covers the POINTING -> DETUMBLE fallback and the momentum-dumping branches (never reached by
    # the cases above). Captured from the pre-refactor run_closed_loop (git show 137bc1d:adcs/simulate.py).
    "lvlh_wheels_fallback": _spec(
        0, IDEAL_SENSORS, MEKF_G7, 1500.0, 1, y0=_y0_lvlh_wheels,
        modes=ModeConfig(safe_rate=float(np.deg2rad(0.05)), safe_hold=3.0)),
}


def n_steps(name: str) -> int:
    """Number of control periods (1 s each) of a case: ``round(t_end/1.0) + 1``."""
    return int(round(CASES[name]["t_end"] / 1.0)) + 1


def run_case(name: str, runner=None) -> ClosedLoopResult:
    """Run one case; ``runner(y0, t_end, seed=..., sensors=..., mekf_cfg=..., log_every=..., **extra)``
    with ``extra`` such as ``ctrl``, ``inertia_true``, ``modes``, ``mtq`` (see :func:`_spec`)."""
    runner = run_closed_loop if runner is None else runner
    spec = CASES[name]
    seed = spec["seed"]
    y0 = tumble_state(np.deg2rad(10.0), seed) if spec["y0"] is None else spec["y0"]()
    kwargs = dict(seed=seed, sensors=spec["sensors"], mekf_cfg=spec["mekf_cfg"],
                  log_every=spec["log_every"], **spec["extra"])
    return runner(y0, spec["t_end"], **kwargs)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"meta leaf of type {type(o).__name__} is not JSON-serialisable")


def _walk(node, path, arrays):
    """Return ``node`` with ndarray leaves removed (collected into ``arrays`` by path)."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            sub = path + (str(k),)
            if isinstance(v, np.ndarray):
                arrays["meta/" + "/".join(sub)] = v
            else:
                out[str(k)] = _walk(v, sub, arrays)
        return out
    return node


def flatten(res: ClosedLoopResult, drop_meta=("kicks",)) -> dict[str, np.ndarray]:
    """Flatten a result into ``{key: ndarray}``: ``res/<field>``, ``meta/<path>`` arrays and ``meta_json``."""
    flat: dict[str, np.ndarray] = {}
    for fld in dataclasses.fields(res):
        if fld.name == "meta":
            continue
        flat[f"res/{fld.name}"] = np.asarray(getattr(res, fld.name))
    meta = {k: v for k, v in res.meta.items() if k not in drop_meta}
    arrays: dict[str, np.ndarray] = {}
    rest = _walk(meta, (), arrays)
    flat.update(arrays)
    flat["meta_json"] = np.asarray(json.dumps(rest, sort_keys=True, allow_nan=True, default=_json_default))
    for key, arr in flat.items():
        if arr.dtype == object:
            raise TypeError(f"{key}: object array would need pickling")
    return flat


def capture(path) -> dict:
    """Run all cases with the default runner and write one compressed npz; returns per-case timing info."""
    import scipy
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out: dict[str, np.ndarray] = {}
    wall = {}
    for name in CASES:
        t0 = time.perf_counter()
        res = run_case(name)
        wall[name] = time.perf_counter() - t0
        for key, arr in flatten(res).items():
            out[f"{name}/{key}"] = arr
    info = {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "machine": platform.machine(), "date": datetime.now(timezone.utc).isoformat(),
            "wall_time_s": wall, "steps": {n: n_steps(n) for n in CASES}}
    out["info_json"] = np.asarray(json.dumps(info, sort_keys=True))
    np.savez_compressed(path, **out)
    return info


def environment_matches(golden) -> tuple[bool, str]:
    """Whether the running interpreter matches the golden file's capture environment.

    Compares python, numpy and scipy versions and ``platform.machine()`` against ``info_json``
    (a field missing from the file, e.g. ``machine`` in older captures, is "unknown" and is not
    compared). Returns ``(ok, reason)``; ``reason`` is ``""`` when ``ok``. Bit-for-bit results are
    only guaranteed on the capture platform, so callers skip rather than fail on a mismatch.
    """
    import scipy
    if "info_json" not in golden.files:
        return False, "golden file has no info_json: capture environment unknown"
    info = json.loads(str(golden["info_json"]))
    now = {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
           "machine": platform.machine()}
    diffs = []
    for key, cur in now.items():
        old = info.get(key)
        if old is None:
            continue
        if key == "python":
            same = old.split()[0] == cur.split()[0]
            old, cur = old.split()[0], cur.split()[0]
        else:
            same = old == cur
        if not same:
            diffs.append(f"{key}: golden {old} != running {cur}")
    if diffs:
        return False, "golden captured in a different environment (" + "; ".join(diffs) + ")"
    return True, ""


def compare(res: ClosedLoopResult, golden, case: str) -> list[str]:
    """Mismatch descriptions between ``res`` and ``golden[case/...]`` (empty when bit-for-bit equal)."""
    new = flatten(res)
    prefix = case + "/"
    old_keys = {k[len(prefix):] for k in golden.files if k.startswith(prefix)}
    problems: list[str] = []
    for k in sorted(old_keys - set(new)):
        problems.append(f"{case}: key {k!r} in golden but missing from result")
    for k in sorted(set(new) - old_keys):
        problems.append(f"{case}: key {k!r} in result but missing from golden")
    for k in sorted(old_keys & set(new)):
        a, b = new[k], golden[prefix + k]
        if a.dtype != b.dtype:
            problems.append(f"{case}/{k}: dtype {a.dtype} != golden {b.dtype}")
        elif a.shape != b.shape:
            problems.append(f"{case}/{k}: shape {a.shape} != golden {b.shape}")
        elif a.tobytes() != b.tobytes():
            if k == "meta_json":
                problems.append(f"{case}/meta_json differs:\n  result: {a.item()[:300]}\n  golden: {b.item()[:300]}")
            else:
                with np.errstate(all="ignore"):
                    nd = int(np.sum(a.view(np.uint8).ravel() != b.view(np.uint8).ravel())) \
                        if a.dtype.kind != "U" else -1
                    try:
                        dmax = float(np.nanmax(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.0
                    except Exception:
                        dmax = float("nan")
                problems.append(f"{case}/{k}: bytes differ ({nd} bytes, max |diff| {dmax:.3e})")
    return problems


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output .npz path")
    args = ap.parse_args(argv)
    info = capture(args.out)
    tot_steps = sum(info["steps"].values())
    tot_wall = sum(info["wall_time_s"].values())
    for n in CASES:
        print(f"{n:32s} {info['wall_time_s'][n]:8.2f} s  {info['steps'][n] / info['wall_time_s'][n]:8.1f} steps/s")
    print(f"{'overall':32s} {tot_wall:8.2f} s  {tot_steps / tot_wall:8.1f} steps/s")
    print(f"wrote {args.out} ({Path(args.out).stat().st_size} bytes)")


if __name__ == "__main__":
    main()
