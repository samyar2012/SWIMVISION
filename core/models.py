"""Plain data models describing a race (what the user tells us).

These are deliberately simple dataclasses with no computer-vision code so they
are easy to read, test, and save as JSON.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

# --- Race options shown in the UI -------------------------------------------
# Value = support level. Only freestyle is the V1 target.
STROKE_SUPPORT: dict[str, str] = {
    "Freestyle": "supported",
    "Butterfly": "experimental",
    "Breaststroke": "experimental",
    "Backstroke": "experimental",
}

DISTANCES: tuple[int, ...] = (25, 50, 100)

# Pool code -> (human label, distance unit used in that kind of pool)
POOLS: dict[str, tuple[str, str]] = {
    "SCY": ("Short Course Yards (25 yd)", "yd"),
    "SCM": ("Short Course Meters (25 m)", "m"),
    "LCM": ("Long Course Meters (50 m)", "m"),
}


def distance_unit_for_pool(pool: str) -> str:
    """Return ``"yd"`` or ``"m"`` for a pool code such as ``"SCY"``."""
    return POOLS[pool][1]


@dataclass(frozen=True)
class RaceInfo:
    """Race details chosen by the user before uploading."""

    stroke: str
    distance: int
    pool: str

    def __post_init__(self) -> None:
        # Fail loudly on impossible values instead of storing bad data.
        if self.stroke not in STROKE_SUPPORT:
            raise ValueError(f"Unknown stroke: {self.stroke!r}")
        if self.distance not in DISTANCES:
            raise ValueError(f"Unsupported distance: {self.distance!r}")
        if self.pool not in POOLS:
            raise ValueError(f"Unknown pool type: {self.pool!r}")

    @property
    def unit(self) -> str:
        return distance_unit_for_pool(self.pool)

    @property
    def label(self) -> str:
        """Short human label, e.g. ``"50 yd Freestyle (SCY)"``."""
        return f"{self.distance} {self.unit} {self.stroke} ({self.pool})"

    @property
    def support_level(self) -> str:
        return STROKE_SUPPORT[self.stroke]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RaceInfo":
        return cls(stroke=data["stroke"], distance=int(data["distance"]), pool=data["pool"])
