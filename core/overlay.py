"""Drawing on frames: the selected box, a zoomed check view, the tracking overlay.

All helpers take and return BGR ``numpy`` images (OpenCV's format) and never
modify the caller's array. Lost frames show "LOST" and no box - nothing is guessed.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.geometry import BBox

# Colours in BGR order (OpenCV).
COLOR_BOX = (0, 200, 255)       # amber: the user's selection
COLOR_TRACKED = (238, 211, 34)  # cyan: SAM 2 mask / box
COLOR_LOST = (68, 68, 239)      # red
COLOR_TEXT_BG = (20, 14, 8)


def _scale(frame: np.ndarray) -> float:
    """Drawing scale so lines/text look similar on small and large frames."""
    return max(0.5, frame.shape[0] / 720.0)


def _put_label(img: np.ndarray, text: str, org: tuple[float, float], color: tuple[int, int, int],
               scale: float) -> None:
    """Draw text on a dark box so it is readable on any background."""
    font, thickness = cv2.FONT_HERSHEY_SIMPLEX, max(1, int(round(scale * 1.5)))
    (tw, th), base = cv2.getTextSize(text, font, 0.6 * scale, thickness)
    x = max(0, min(int(org[0]), img.shape[1] - tw - 6))
    y = max(th + 6, min(int(org[1]), img.shape[0] - 4))
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + base), COLOR_TEXT_BG, -1)
    cv2.putText(img, text, (x + 3, y - 3), font, 0.6 * scale, color, thickness, cv2.LINE_AA)


def _rect(img: np.ndarray, box: BBox, color: tuple[int, int, int], thickness: int) -> None:
    p1 = (int(round(box.x1)), int(round(box.y1)))
    p2 = (int(round(box.x2)), int(round(box.y2)))
    cv2.rectangle(img, p1, p2, color, thickness, cv2.LINE_AA)


def draw_box(frame: np.ndarray, box: BBox, label: str | None = None,
             color: tuple[int, int, int] = COLOR_BOX) -> np.ndarray:
    """Draw one rectangle (optionally labelled) on a copy of ``frame``."""
    out, s = frame.copy(), _scale(frame)
    _rect(out, box, color, max(2, int(round(s * 3))))
    if label:
        _put_label(out, label, (box.x1, box.y1 - 4), color, s)
    return out


def zoomed_crop(frame: np.ndarray, box: BBox, margin: float = 1.0, target_height: int = 300,
                max_width: int = 700) -> np.ndarray:
    """A magnified view around ``box`` so the user can check a small selection.

    ``margin`` is the extra context on every side, as a multiple of the box
    size. The box outline is drawn on the enlarged crop. Never shrinks.
    """
    h, w = frame.shape[:2]
    pad_x, pad_y = max(box.width * margin, 20), max(box.height * margin, 20)
    region = BBox(box.x1 - pad_x, box.y1 - pad_y, box.x2 + pad_x, box.y2 + pad_y).clamp(w, h)
    x1, y1, x2, y2 = region.to_int_list()
    crop = frame[y1:y2, x1:x2]
    zoom = max(1.0, min(target_height / max(crop.shape[0], 1), max_width / max(crop.shape[1], 1)))
    big = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    inner = BBox(box.x1 - x1, box.y1 - y1, box.x2 - x1, box.y2 - y1).scaled(zoom, zoom)
    _rect(big, inner, COLOR_BOX, 2)
    return big


def draw_tracking_overlay(frame: np.ndarray, mask: np.ndarray | None, box: BBox | None,
                          timestamp_s: float, frame_index: int, lost_label: str = "LOST") -> np.ndarray:
    """Overlay for one analysed frame: tinted mask + box when tracked, "LOST" otherwise.

    ``mask`` (boolean, same height/width as ``frame``) and ``box`` must be in
    the pixel coordinates of ``frame``.
    """
    out, s = frame.copy(), _scale(frame)
    tracked = box is not None
    if mask is not None:
        tint = np.zeros_like(out)
        tint[mask] = COLOR_TRACKED
        out = cv2.addWeighted(out, 1.0, tint, 0.45, 0)
    if box is not None:
        _rect(out, box, COLOR_TRACKED, max(1, int(round(s * 2))))
    _put_label(out, f"t = {timestamp_s:6.2f} s   frame {frame_index}", (8, 8 + 28 * s), (255, 255, 255), s)
    _put_label(out, "TRACKED" if tracked else lost_label, (8, 8 + 60 * s),
               COLOR_TRACKED if tracked else COLOR_LOST, s)
    return out
