"""Request and command schemas for the browser simulator server (pydantic v2 only).

Spec: ``docs/specs/webapp_protocol.md`` sections 1 and 3.1. A visual aid only; it adds no numbers
to any report. Units: SI and radians internally; ``*_deg_s`` fields are degrees per second at the
wire boundary only. Importing this module imports pydantic and the ADCS presets, never fastapi.
"""
from __future__ import annotations

import json
from typing import Annotated, Any, Literal, Union

import numpy as np
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StrictBool, TypeAdapter, ValidationError, field_validator

from adcs.config import DEFAULT
from adcs.orbit import orbital_period
from adcs.simulate import (IDEAL_SENSORS, MEKF_DESIGN, MEKF_G7, REALISTIC_SENSORS, ControlConfig,
                           tumble_state)

__all__ = [
    "SessionParams", "build_sim_kwargs", "LOG_EVERY",
    "PlayCmd", "PauseCmd", "SpeedCmd", "KickCmd", "StepCmd", "ResetCmd", "Command",
    "CommandError", "parse_command",
]

#: The server never needs the full log; every tenth sample is kept (spec section 1).
LOG_EVERY = 10


def _plain_number(v: Any) -> Any:
    """Reject booleans and strings where a number is expected (pydantic would coerce them)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError("must be a number")
    return v


def _plain_int(v: Any) -> Any:
    """Reject booleans and strings where an integer is expected."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError("must be an integer")
    return v


_Number = BeforeValidator(_plain_number)
_Integer = BeforeValidator(_plain_int)


class SessionParams(BaseModel):
    """Parameters of one simulated session (spec section 1)."""

    model_config = ConfigDict(extra="forbid")

    orbits: Annotated[float, Field(ge=0.05, le=15.0, allow_inf_nan=False), _Number] = 6.0
    rate_deg_s: Annotated[float, Field(ge=0.0, le=30.0, allow_inf_nan=False), _Number] = 10.0
    seed: Annotated[int, Field(ge=0, le=2**31 - 1), _Integer] = 0
    sensors: Literal["realistic", "ideal"] = "realistic"
    filter: Literal["design", "g7"] = "design"
    dumping: StrictBool = True
    inertia_scale: Annotated[float, Field(ge=0.5, le=1.5, allow_inf_nan=False), _Number] = 1.0


def build_sim_kwargs(p: SessionParams) -> dict:
    """Exact keyword arguments for ``adcs.stepper.ClosedLoopSim(**kw)``.

    ``t_end`` is ``orbits x`` the orbital period (s); ``y0`` is the random-tumble state at
    ``rate_deg_s`` (converted to rad/s) with ``seed``. ``inertia_true`` is ``None`` at scale 1.0,
    otherwise ``scale x`` the nominal inertia (kg m^2); the controller always uses the nominal one.
    """
    period = float(orbital_period(DEFAULT.orbit.a))
    scale = float(p.inertia_scale)
    return {
        "y0": tumble_state(np.deg2rad(p.rate_deg_s), p.seed),
        "t_end": p.orbits * period,
        "seed": p.seed,
        "ctrl": ControlConfig(dumping=p.dumping),
        "sensors": REALISTIC_SENSORS if p.sensors == "realistic" else IDEAL_SENSORS,
        "mekf_cfg": MEKF_DESIGN if p.filter == "design" else MEKF_G7,
        "inertia_true": None if scale == 1.0 else scale * DEFAULT.spacecraft.inertia,
        "log_every": LOG_EVERY,
    }


# ---------------------------------------------------------------------------
# WebSocket commands (client -> server)
# ---------------------------------------------------------------------------
class _Cmd(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlayCmd(_Cmd):
    """Start streaming."""

    cmd: Literal["play"]


class PauseCmd(_Cmd):
    """Stop stepping."""

    cmd: Literal["pause"]


class SpeedCmd(_Cmd):
    """Target sim-seconds per wall-second, 1 to 2000."""

    cmd: Literal["speed"]
    value: Annotated[float, Field(ge=1.0, le=2000.0, allow_inf_nan=False), _Number]


class KickCmd(_Cmd):
    """Instantaneous body-rate impulse about one body axis, ``0 < |deg_s| <= 20`` (deg/s)."""

    cmd: Literal["kick"]
    axis: Literal["x", "y", "z"]
    deg_s: Annotated[float, Field(ge=-20.0, le=20.0, allow_inf_nan=False), _Number]

    @field_validator("deg_s")
    @classmethod
    def _nonzero(cls, v: float) -> float:
        if v == 0.0:
            raise ValueError("deg_s must be nonzero")
        return v


class StepCmd(_Cmd):
    """Process ``n`` samples while paused (deterministic stepping)."""

    cmd: Literal["step"]
    n: Annotated[int, Field(ge=1, le=100000), _Integer]


class ResetCmd(_Cmd):
    """Rebuild the sim from the same parameters and pause."""

    cmd: Literal["reset"]


Command = Annotated[Union[PlayCmd, PauseCmd, SpeedCmd, KickCmd, StepCmd, ResetCmd],
                    Field(discriminator="cmd")]
_COMMANDS = TypeAdapter(Command)
_KNOWN = ("play", "pause", "speed", "kick", "step", "reset")


class CommandError(ValueError):
    """A client message that is not a valid command; ``str(e)`` is sent back to the client."""


def parse_command(raw: str | bytes | dict) -> PlayCmd | PauseCmd | SpeedCmd | KickCmd | StepCmd | ResetCmd:
    """Parse and validate one client message (JSON text, or an already-decoded dict).

    Raises :class:`CommandError` for bad JSON, a non-object, an unknown ``cmd``, extra keys or
    out-of-range values.
    """
    obj = raw
    if isinstance(raw, (str, bytes, bytearray)):
        try:
            obj = json.loads(raw)
        except ValueError as e:
            raise CommandError(f"invalid JSON: {e}") from None
    if not isinstance(obj, dict):
        raise CommandError("message must be a JSON object")
    name = obj.get("cmd")
    if name not in _KNOWN:
        raise CommandError(f"unknown cmd {name!r}; expected one of {', '.join(_KNOWN)}")
    try:
        return _COMMANDS.validate_python(obj)
    except ValidationError as e:
        parts = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"][1:]) or "message"
            parts.append(f"{loc}: {err['msg']}")
        raise CommandError(f"bad {name} command: " + "; ".join(parts)) from None
