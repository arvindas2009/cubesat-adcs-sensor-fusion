"""Generate ``ned_fixtures.json``: the cross-language oracle for ``src/math/ned.ts``.

Run from the repo root::

    .venv/Scripts/python webapp/frontend/tests/fixtures/make_ned_fixtures.py [output.json]

Everything comes from ``common.quaternion`` (``to_dcm``, ``from_euler_zyx``, ``to_euler_zyx``) and numpy
(docs/specs/px4_replay.md section 3.2). Quaternions are scalar-first ``[w, x, y, z]``, body (FRD) to NED.

Contents:

* ``N``: the NED to Three.js world matrix, ``N (n, e, d) = (n, -d, e)`` (north +X, up +Y, east +Z).
* ``quats``: seeded random unit quaternions ``q`` with ``dcm = to_dcm(q)`` and ``world_cols = (N @ dcm).T``
  (entry k = body axis k in world coordinates).
* ``named``: attitudes from aerospace Euler angles (roll, pitch, yaw) with ``q``, the ``to_euler_zyx`` values in
  degrees and the world-coordinate body axes (``nose`` = body x, ``wing`` = body y, ``down`` = body z).
* ``vectors``: NED vectors with ``N v``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from common.quaternion import from_euler_zyx, to_dcm, to_euler_zyx  # noqa: E402

N = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
OUT = Path(__file__).resolve().parent / "ned_fixtures.json"

NAMED_DEG = {  # name: (roll, pitch, yaw)
    "level_north": (0.0, 0.0, 0.0),
    "yaw_90_east": (0.0, 0.0, 90.0),
    "yaw_180_south": (0.0, 0.0, 180.0),
    "pitch_up_30": (0.0, 30.0, 0.0),
    "roll_right_30": (30.0, 0.0, 0.0),
    "mixed": (20.0, -15.0, 135.0),
}


def quat_fixtures(n: int = 20, seed: int = 4242) -> list[dict]:
    rng = np.random.default_rng(seed)
    qs = rng.normal(size=(n, 4))
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    qs = np.vstack([qs, [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    out = []
    for q in qs:
        dcm = to_dcm(q)
        out.append({"q": q.tolist(), "dcm": dcm.tolist(), "world_cols": (N @ dcm).T.tolist()})
    return out


def named_fixtures() -> list[dict]:
    out = []
    for name, (r, p, y) in NAMED_DEG.items():
        q = from_euler_zyx(*np.deg2rad([r, p, y]))
        cols = (N @ to_dcm(q)).T  # entry k = body axis k in world coordinates
        out.append({
            "name": name,
            "euler_in_deg": [r, p, y],
            "q": q.tolist(),
            "euler_deg": np.rad2deg(to_euler_zyx(q)).tolist(),
            "nose_world": cols[0].tolist(),
            "wing_world": cols[1].tolist(),
            "down_world": cols[2].tolist(),
        })
    return out


def vector_fixtures() -> list[dict]:
    vs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [12.5, -3.25, 7.0], [-100.0, 40.0, -2.5]]
    return [{"ned": v, "world": (N @ np.asarray(v)).tolist()} for v in vs]


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    assert abs(np.linalg.det(N) - 1.0) < 1e-12
    data = {"N": N.tolist(), "quats": quat_fixtures(), "named": named_fixtures(), "vectors": vector_fixtures()}
    out_path.write_text(json.dumps(data, allow_nan=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out_path} ({len(data['quats'])} quats, {len(data['named'])} named, {len(data['vectors'])} vectors)")


if __name__ == "__main__":
    main()
