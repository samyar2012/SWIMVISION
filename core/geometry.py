"""Bounding-box geometry shared by the detector, tracker and overlay code.

A box is stored as two corners ``(x1, y1)`` (top-left) and ``(x2, y2)``
(bottom-right) in **pixels of the original video frame**. Pure math, no I/O,
so it is easy to unit-test.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    """An axis-aligned rectangle in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def iou(self, other: "BBox") -> float:
        """Intersection-over-Union: 0 = no overlap, 1 = identical boxes.

        This is the standard way to ask "do these two boxes show the same
        object?". It is the core of how we decide a track is still our swimmer.
        """
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def clamp(self, width: int, height: int) -> "BBox":
        """Return a copy forced to lie inside a ``width`` x ``height`` frame."""
        return BBox(
            min(max(self.x1, 0.0), width), min(max(self.y1, 0.0), height),
            min(max(self.x2, 0.0), width), min(max(self.y2, 0.0), height),
        )

    def to_xywh_int(self) -> tuple[int, int, int, int]:
        """``(x, y, w, h)`` as integers - the format OpenCV trackers expect."""
        return (int(round(self.x1)), int(round(self.y1)),
                int(round(self.width)), int(round(self.height)))

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> "BBox":
        return cls(x, y, x + w, y + h)

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_list(cls, values: list[float]) -> "BBox":
        return cls(*[float(v) for v in values])


def box_from_drag(drag: dict, frame_w: int, frame_h: int, min_side: float = 6.0) -> "BBox | None":
    """Turn a mouse drag on a *scaled* image into a box in real frame pixels.

    ``drag`` holds the press point (``x1, y1``), the release point (``x2, y2``)
    and the size the image was displayed at (``width, height``). The release
    point may lie outside the image, so the box is clamped to the frame.
    Returns ``None`` when the drag is missing or too small to be a real box
    (for example a plain click).
    """
    try:
        shown_w, shown_h = float(drag["width"]), float(drag["height"])
        xs = (float(drag["x1"]), float(drag["x2"]))
        ys = (float(drag["y1"]), float(drag["y2"]))
    except (KeyError, TypeError, ValueError):
        return None
    if shown_w <= 0 or shown_h <= 0:
        return None
    sx, sy = frame_w / shown_w, frame_h / shown_h
    box = BBox(min(xs) * sx, min(ys) * sy, max(xs) * sx, max(ys) * sy).clamp(frame_w, frame_h)
    return box if box.width >= min_side and box.height >= min_side else None
