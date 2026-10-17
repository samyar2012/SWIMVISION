"""Bounding-box geometry and coordinate conversions.

A box is stored as two corners ``(x1, y1)`` (top-left) and ``(x2, y2)``
(bottom-right). Unless a name says otherwise, boxes are in **pixels of the
original video frame**. Three coordinate systems exist in SwimVision:

* original video pixels - what is saved and reported;
* display pixels - the resized frame the user draws on in the browser;
* analysis pixels - the (optionally downscaled) frames SAM 2 sees.

The helpers below convert between them. Pure math, no I/O, unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
        """Intersection-over-Union: 0 = no overlap, 1 = identical boxes."""
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

    def scaled(self, sx: float, sy: float) -> "BBox":
        """Multiply x coordinates by ``sx`` and y coordinates by ``sy``."""
        return BBox(self.x1 * sx, self.y1 * sy, self.x2 * sx, self.y2 * sy)

    def is_inside(self, width: int, height: int) -> bool:
        return 0 <= self.x1 < self.x2 <= width and 0 <= self.y1 < self.y2 <= height

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def to_int_list(self) -> list[int]:
        return [int(round(v)) for v in self.to_list()]

    @classmethod
    def from_list(cls, values: list[float]) -> "BBox":
        return cls(*[float(v) for v in values])


def fit_size(width: int, height: int, target_width: int | None) -> tuple[int, int]:
    """Size to resize a ``width`` x ``height`` frame to so it is ``target_width`` wide.

    Never upscales (``target_width=None`` or larger than the frame keeps the
    original size). Both sides are rounded to even numbers because the H.264
    encoder requires even dimensions.
    """
    factor = 1.0 if not target_width or target_width >= width else target_width / width
    return max(2, int(round(width * factor / 2)) * 2), max(2, int(round(height * factor / 2)) * 2)


def display_size(width: int, height: int, display_width: int) -> tuple[int, int]:
    """Size at which to *show* a frame so it is exactly ``display_width`` wide.

    Unlike ``fit_size`` this may upscale: a small 848 px video is easier to
    draw on when shown larger.
    """
    return display_width, max(1, int(round(height * display_width / width)))


def cropper_rect_to_box(rect: dict[str, Any], display_w: int, display_h: int,
                        frame_w: int, frame_h: int, min_side: float = 4.0) -> BBox | None:
    """Convert the cropper's ``{left, top, width, height}`` (display pixels) to original pixels.

    Returns ``None`` when the rectangle is missing, malformed or smaller than
    ``min_side`` original pixels on either side.
    """
    try:
        left, top = float(rect["left"]), float(rect["top"])
        w, h = float(rect["width"]), float(rect["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if display_w <= 0 or display_h <= 0:
        return None
    box = BBox(left, top, left + w, top + h).scaled(frame_w / display_w, frame_h / display_h)
    box = box.clamp(frame_w, frame_h)
    return box if box.width >= min_side and box.height >= min_side else None


def box_to_cropper_coords(box: BBox, display_w: int, display_h: int,
                          frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Original-pixel box -> the cropper's ``default_coords`` ``(left, right, top, bottom)``."""
    shown = box.scaled(display_w / frame_w, display_h / frame_h)
    return (int(round(shown.x1)), int(round(shown.x2)), int(round(shown.y1)), int(round(shown.y2)))
