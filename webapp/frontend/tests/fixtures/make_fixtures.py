"""Generate ``frames_fixtures.json``: the cross-language oracle for ``src/math/frames.ts``.

Run from the repo root::

    .venv/Scripts/python webapp/frontend/tests/fixtures/make_fixtures.py [output.json]

Everything is computed by the Python implementations (the oracle): ``common.quaternion.to_dcm``,
``common.frames`` (``gmst``, ``dcm_eci_to_ecef``, ``julian_date``), the closed-loop engine
``adcs.stepper.ClosedLoopSim`` built through ``webapp.server.schemas.build_sim_kwargs`` and the frame
builder ``webapp.server.adcs_session.frame``. Only numpy does the expected world-axis arithmetic.

Contents (docs/specs/frontend.md section 4):

* ``M``: the z-up to Three.js world matrix, ``(x, y, z) -> (x, z, -y)``.
* ``quats``: seeded random unit quaternions ``q`` (scalar-first) with ``dcm = to_dcm(q)``.
* ``frames``: real server frames from four phases (before TRIAD init, detumble with and without an
  estimate, pointing). For each: the wire ``frame`` (7-significant-digit floats), ``q_unit`` (the
  frame ``q`` normalised in double precision), the expected world-coordinate body axes of the
  display scene ``M @ C`` and of the ECI scene ``M @ to_dcm(q_unit)`` (``*_cols``: entry k is
  body axis k), and the world vectors of ``display.sun`` / ``display.b`` / ``eci.r_m``.
* ``earth``: ground points for several (lat, lon, GMST): the expected world position
  ``M @ p_eci`` (``p_eci = Rz(gmst)^T p_ecef``) and the equirectangular texture coordinates.
* ``display_flip``: ``adcs.visualize.DISPLAY_FLIP``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from adcs.config import DEFAULT  # noqa: E402
from adcs.orbit import initial_state  # noqa: E402
from adcs.simulate import lvlh_state  # noqa: E402
from adcs.stepper import ClosedLoopSim  # noqa: E402
from adcs.visualize import DISPLAY_FLIP  # noqa: E402
from common import frames as cf  # noqa: E402
from common.quaternion import to_dcm  # noqa: E402
from webapp.server.adcs_session import SessionState, frame  # noqa: E402
from webapp.server.schemas import SessionParams, build_sim_kwargs  # noqa: E402

M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
OUT = Path(__file__).resolve().parent / "frames_fixtures.json"


def quat_fixtures(n: int = 20, seed: int = 12345) -> list[dict]:
    rng = np.random.default_rng(seed)
    qs = rng.normal(size=(n, 4))
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    qs = np.vstack([qs, [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])  # identity and a 180 deg turn
    return [{"q": q.tolist(), "dcm": to_dcm(q).tolist()} for q in qs]


def _phase(f: dict) -> str:
    if f["eci"]["q_est"] is None:
        return "pre-init"
    return "pointing" if f["mode"] == 1 else "detumble"


def _collect(sim: ClosedLoopSim, ks: list[int], tag: str) -> list[dict]:
    """Step ``sim`` to each sample index in ``ks`` (ascending) and record the frame there."""
    out = []
    sim.step(1)
    state = SessionState(seq=0, playing=False, speed_set=50.0, speed_actual=0.0)
    for k in ks:
        if k > sim.k - 1:
            sim.step(k - (sim.k - 1))
        f = frame(sim.snapshot(), sim, state)
        assert f["k"] == k, (f["k"], k)
        json.dumps(f, allow_nan=False)
        q_unit = np.asarray(f["eci"]["q"], float)
        q_unit /= np.linalg.norm(q_unit)
        C = np.asarray(f["display"]["C"], float)
        out.append({
            "label": f"{tag}-k{k}",
            "phase": _phase(f),
            "frame": f,
            "q_unit": q_unit.tolist(),
            "display_world_cols": (M @ C).T.tolist(),
            "eci_world_cols": (M @ to_dcm(q_unit)).T.tolist(),
            "display_sun_world": (M @ np.asarray(f["display"]["sun"], float)).tolist(),
            "display_b_world": (M @ np.asarray(f["display"]["b"], float)).tolist(),
            "r_world_m": (M @ np.asarray(f["eci"]["r_m"], float)).tolist(),
        })
    return out


def frame_fixtures() -> list[dict]:
    out: list[dict] = []
    # (a) the default tumble, realistic sensors + design filter: no estimate before TRIAD init
    #     (about t = 269 s for seed 0), then detumble with an estimate.
    kw = build_sim_kwargs(SessionParams(orbits=0.1, seed=0))
    out += _collect(ClosedLoopSim(**kw), [0, 20, 100, 300, 480], "tumble")
    # (b) started near LVLH with ideal sensors + G7 filter: detumble, then POINTING at about t = 360 s.
    kw = build_sim_kwargs(SessionParams(orbits=0.12, sensors="ideal", filter="g7", seed=0))
    r0, v0 = initial_state(DEFAULT)
    kw["y0"] = lvlh_state(r0, v0)
    out += _collect(ClosedLoopSim(**kw), [0, 200, 450, 520], "lvlh")
    phases = {o["phase"] for o in out}
    assert {"pre-init", "detumble", "pointing"} <= phases, phases
    return out


GROUND_POINTS_DEG = [(0.0, 0.0), (0.0, 90.0), (45.0, -120.0), (-30.0, 150.0)]
GMST_OFFSETS_DAYS = [0.0, 0.2, 0.55]


def earth_fixtures() -> list[dict]:
    out = []
    for off in GMST_OFFSETS_DAYS:
        jd = DEFAULT.jd_epoch + off
        gmst = float(cf.gmst(jd))
        A = cf.dcm_eci_to_ecef(jd)  # p_ecef = A p_eci; the inverse is the transpose
        for lat_d, lon_d in GROUND_POINTS_DEG:
            lat, lon = np.deg2rad(lat_d), np.deg2rad(lon_d)
            p_ecef = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
            p_eci = A.T @ p_ecef
            assert np.allclose(p_eci, cf.ecef_to_eci(p_ecef, jd), atol=1e-12)
            assert np.allclose(cf.eci_to_ecef(p_eci, jd), p_ecef, atol=1e-12)
            world = M @ p_eci
            world /= np.linalg.norm(world)
            out.append({
                "jd": jd, "gmst_rad": gmst, "lat_deg": lat_d, "lon_deg": lon_d,
                "lat_rad": float(lat), "lon_rad": float(lon),
                "ecef": p_ecef.tolist(), "world": world.tolist(),
                "u": (lon_d + 180.0) / 360.0, "v": (90.0 - lat_d) / 180.0,
            })
    return out


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    data = {
        "M": M.tolist(),
        "display_flip": DISPLAY_FLIP.tolist(),
        "quats": quat_fixtures(),
        "frames": frame_fixtures(),
        "earth": earth_fixtures(),
    }
    out_path.write_text(json.dumps(data, allow_nan=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out_path} ({len(data['quats'])} quats, {len(data['frames'])} frames, "
          f"{len(data['earth'])} earth points)")


if __name__ == "__main__":
    main()
