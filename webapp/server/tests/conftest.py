"""Shared fixtures for the webapp server tests."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from adcs.stepper import ClosedLoopSim  # noqa: E402
from webapp.server.schemas import SessionParams, build_sim_kwargs  # noqa: E402

#: The short session every server test uses: 0.05 orbits = 278 control periods (T = 1 s).
SHORT = SessionParams(orbits=0.05)


@pytest.fixture(scope="session")
def reference():
    """Independent run of the SHORT session with the engine directly (no server code involved).

    ``snaps`` holds ``snapshot()`` at sample k = 0, 1, 100 and at the last sample (``"end"``);
    ``sim`` is the finished engine.
    """
    sim = ClosedLoopSim(**build_sim_kwargs(SHORT))
    snaps = {}
    sim.step(1)
    snaps[0] = sim.snapshot()
    sim.step(1)
    snaps[1] = sim.snapshot()
    sim.step(99)
    snaps[100] = sim.snapshot()
    sim.step(sim.remaining)
    snaps["end"] = sim.snapshot()
    assert sim.done
    return SimpleNamespace(sim=sim, snaps=snaps)
