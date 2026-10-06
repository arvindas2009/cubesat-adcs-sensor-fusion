"""Project B configuration: 3U CubeSat, orbit and epoch (all SI, radians).

Every value is recorded in ``assumptions.md`` (B-01 ... B-12). Actuator and
sensor parameters live in the later modules (Groups 6-7), not here.
Body frame: origin at the CoM, x and y along the two 10 cm sides, z along the
34 cm long axis (CONVENTIONS.md section 4).
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

import numpy as np

from common.frames import R_EARTH, julian_date

#: Simulation epoch (UTC). Near the March equinox 2026, so with RAAN = 0 the
#: beta angle is ~0 and eclipses are at their longest (~39 %). [B-04]
EPOCH = _dt.datetime(2026, 3, 20, 12, 0, 0)


def inertia_box(mass: float, size) -> np.ndarray:
    """Inertia tensor (kg m^2, diagonal 3x3) of a uniform solid box about its centre.

    ``size`` = (a, b, c) edge lengths (m) along body x, y, z:
    ``Ixx = m (b^2 + c^2) / 12`` and cyclic.
    """
    a, b, c = (float(s) for s in size)
    m = float(mass)
    return np.diag([m * (b**2 + c**2) / 12.0, m * (a**2 + c**2) / 12.0, m * (a**2 + b**2) / 12.0])


@dataclass(frozen=True)
class SpacecraftConfig:
    """3U CubeSat. [B-01, B-02, B-03, B-08, B-09]"""

    mass: float = 4.0                                   # kg
    size: tuple = (0.10, 0.10, 0.34)                    # m, body x, y, z
    com_offset_b: tuple = (0.005, 0.005, 0.010)         # m, CoM minus geometric centre, body axes
    residual_dipole_b: tuple = (0.001, 0.001, 0.001)    # A m^2, body axes
    cd: float = 2.2                                     # drag coefficient (flat plates)

    @property
    def inertia(self) -> np.ndarray:
        """Box inertia about the CoM (kg m^2); CoM offset ignored in the tensor [B-03]."""
        return inertia_box(self.mass, self.size)


@dataclass(frozen=True)
class OrbitConfig:
    """Initial osculating orbit. [B-04]"""

    altitude: float = 400e3                             # m above R_EARTH (semi-major axis = R_EARTH + altitude)
    ecc: float = 0.0
    inc: float = float(np.deg2rad(51.6))                # rad
    raan: float = 0.0                                   # rad
    argp: float = 0.0                                   # rad
    nu: float = 0.0                                     # rad, true anomaly at epoch

    @property
    def a(self) -> float:
        return R_EARTH + self.altitude


@dataclass(frozen=True)
class SimConfig:
    epoch: _dt.datetime = EPOCH
    spacecraft: SpacecraftConfig = field(default_factory=SpacecraftConfig)
    orbit: OrbitConfig = field(default_factory=OrbitConfig)

    @property
    def jd_epoch(self) -> float:
        return julian_date(self.epoch)


DEFAULT = SimConfig()
