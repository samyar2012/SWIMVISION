"""Plain data models describing a race (what the user tells us).

These are deliberately simple dataclasses with no computer-vision code so they
are easy to read, test, and save as JSON.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

# --- Race options shown in the UI -------------------------------------------
# Tracking works the same for every stroke. Race analysis supports each listed
# stroke through 200; IM stroke-cycle classification remains explicitly withheld.
STROKES: tuple[str, ...] = ("Freestyle", "Backstroke", "Breaststroke", "Butterfly", "Individual Medley")

# Pool code -> (human label, distance unit, length of one pool length in that unit)
POOLS: dict[str, tuple[str, str, int]] = {
    "SCY": ("Short Course Yards (25 yd)", "yd", 25),
    "SCM": ("Short Course Meters (25 m)", "m", 25),
    "LCM": ("Long Course Meters (50 m)", "m", 50),
}

# Individual events swum in each pool type (USA Swimming / NCAA / high school for
# SCY, World Aquatics for SCM and LCM). 25s are age-group events in short-course pools.
_STROKE_EVENTS = (25, 50, 100, 200)
EVENTS: dict[str, dict[str, tuple[int, ...]]] = {
    "SCY": {
        "Freestyle": (25, 50, 100, 200, 500, 1000, 1650),
        "Backstroke": _STROKE_EVENTS, "Breaststroke": _STROKE_EVENTS, "Butterfly": _STROKE_EVENTS,
        "Individual Medley": (100, 200, 400),
    },
    "SCM": {
        "Freestyle": (25, 50, 100, 200, 400, 800, 1500),
        "Backstroke": _STROKE_EVENTS, "Breaststroke": _STROKE_EVENTS, "Butterfly": _STROKE_EVENTS,
        "Individual Medley": (100, 200, 400),
    },
    "LCM": {
        "Freestyle": (50, 100, 200, 400, 800, 1500),
        "Backstroke": (50, 100, 200), "Breaststroke": (50, 100, 200), "Butterfly": (50, 100, 200),
        "Individual Medley": (200, 400),
    },
}

# Individual medley order: each stroke covers a quarter of the race.
IM_ORDER: tuple[str, ...] = ("Butterfly", "Backstroke", "Breaststroke", "Freestyle")


def distance_unit_for_pool(pool: str) -> str:
    """Return ``"yd"`` or ``"m"`` for a pool code such as ``"SCY"``."""
    return POOLS[pool][1]


def distances_for(stroke: str, pool: str) -> tuple[int, ...]:
    """Official race distances for ``stroke`` in ``pool``, shortest first."""
    return EVENTS[pool][stroke]


@dataclass(frozen=True)
class RaceInfo:
    """Race details chosen by the user before uploading."""

    stroke: str
    distance: int
    pool: str

    def __post_init__(self) -> None:
        # Fail loudly on impossible values instead of storing bad data.
        if self.pool not in POOLS:
            raise ValueError(f"Unknown pool type: {self.pool!r}")
        if self.stroke not in STROKES:
            raise ValueError(f"Unknown stroke: {self.stroke!r}")
        if self.distance not in distances_for(self.stroke, self.pool):
            raise ValueError(f"{self.distance} {self.unit} {self.stroke} is not an event in a {self.pool} pool")

    @property
    def unit(self) -> str:
        return distance_unit_for_pool(self.pool)

    @property
    def pool_length(self) -> int:
        """Length of the pool in ``unit`` (25 or 50)."""
        return POOLS[self.pool][2]

    @property
    def lengths(self) -> int:
        """Number of pool lengths in the race, e.g. 8 for a 200 SCY."""
        return self.distance // self.pool_length

    @property
    def turns(self) -> int:
        return self.lengths - 1

    @property
    def im_segments(self) -> tuple[tuple[str, int], ...]:
        """For an IM: ``(stroke, distance)`` per leg in race order; empty otherwise."""
        if self.stroke != "Individual Medley":
            return ()
        return tuple((stroke, self.distance // 4) for stroke in IM_ORDER)

    @property
    def label(self) -> str:
        """Short human label, e.g. ``"50 yd Freestyle (SCY)"``."""
        return f"{self.distance} {self.unit} {self.stroke} ({self.pool})"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RaceInfo":
        return cls(stroke=data["stroke"], distance=int(data["distance"]), pool=data["pool"])
