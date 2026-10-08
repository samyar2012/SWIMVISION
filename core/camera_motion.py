"""Camera motion estimation and the "is the target actually swimming?" check.

Race videos are filmed by hand: the camera pans, shakes and zooms. So a box
that moves across the *image* is not necessarily moving through the *pool*,
and a box that stays still in the image may be a swimmer the camera follows.

We estimate how the whole picture moved between two analysed frames (a
similarity transform: shift + rotation + zoom) from background feature points
tracked with optical flow. Subtracting that camera motion from the box's
motion gives the target's motion *relative to the scene*.

Why it matters: a tracker that drifts off a swimmer often settles on a
spectator or official. Those stand still relative to the scene, while a
swimmer always moves along the pool (sideways in the image, or growing /
shrinking when swimming towards / away from the camera). Long stretches of
"not moving" therefore get flagged so the user can check them.
"""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

from core.geometry import BBox

Affine = np.ndarray  # 2x3 matrix mapping previous-frame points to current-frame points

MOTION_WIDTH = 424  # frames are shrunk to this width for speed; results are scaled back


def to_motion_gray(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """Small greyscale copy of ``frame`` used for motion estimation, and its scale factor."""
    scale = MOTION_WIDTH / frame.shape[1]
    small = cv2.resize(frame, (MOTION_WIDTH, max(1, int(round(frame.shape[0] * scale)))),
                       interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), scale


def estimate_camera_motion(prev_gray: np.ndarray, gray: np.ndarray, scale: float) -> Affine | None:
    """Similarity transform (in full-resolution pixels) from the previous frame to this one.

    Returns ``None`` when there are too few trackable background points
    (e.g. a frame full of plain water), in which case no judgement is made.
    """
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=300, qualityLevel=0.01, minDistance=7)
    if pts is None or len(pts) < 12:
        return None
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, winSize=(21, 21), maxLevel=3)
    good = status.reshape(-1) == 1
    if good.sum() < 12:
        return None
    matrix, inliers = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC,
                                                  ransacReprojThreshold=2.0)
    if matrix is None or inliers is None or inliers.sum() < 10:
        return None
    full = matrix.copy()
    full[:, 2] /= scale  # translation back to full-resolution pixels (rotation/zoom are scale-free)
    return full


def affine_scale(matrix: Affine) -> float:
    """Zoom factor of a similarity transform."""
    return float(math.hypot(matrix[0, 0], matrix[1, 0]))


def apply_affine(matrix: Affine, point: tuple[float, float]) -> tuple[float, float]:
    x, y = point
    return (float(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]),
            float(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]))


def touches_edge(box: BBox, frame_w: int, frame_h: int, margin: float = 2.0) -> bool:
    """True if the box reaches the picture border (so part of the target may be cut off)."""
    return (box.x1 <= margin or box.y1 <= margin
            or box.x2 >= frame_w - margin or box.y2 >= frame_h - margin)


def relative_steps(boxes: Sequence[BBox | None], motions: Sequence[Affine | None],
                   frame_size: tuple[int, int] | None = None) -> list[tuple[float, float, float] | None]:
    """Per analysed frame: the target's own motion since the previous frame.

    ``motions[t]`` maps frame ``t-1`` to frame ``t``. Each entry is
    ``(dx, dy, log_size_change)`` with the camera's shift and zoom removed, or
    ``None`` when it cannot be measured: box or camera motion unknown, or (if
    ``frame_size=(w, h)`` is given) the box is cut off by the picture border,
    which changes its size and centre without the target moving.
    """
    steps: list[tuple[float, float, float] | None] = [None]
    for t in range(1, len(boxes)):
        prev, cur, motion = boxes[t - 1], boxes[t], motions[t]
        if prev is None or cur is None or motion is None or prev.area <= 0 or cur.area <= 0:
            steps.append(None)
            continue
        if frame_size is not None and (touches_edge(prev, *frame_size) or touches_edge(cur, *frame_size)):
            steps.append(None)
            continue
        px, py = apply_affine(motion, prev.center)
        cx, cy = cur.center
        size_change = math.log(math.sqrt(cur.area / prev.area) / affine_scale(motion))
        steps.append((cx - px, cy - py, size_change))
    return steps


def not_moving_flags(boxes: Sequence[BBox | None], steps: Sequence[tuple[float, float, float] | None],
                     window: int, min_shift_boxes: float = 0.6, min_size_change: float = 0.4,
                     min_known: float = 0.6) -> list[bool]:
    """``True`` where the target did not move through the scene over a ``window`` of frames.

    Over a window centred on each frame the target's own steps are *added up*
    (so back-and-forth jitter cancels out). The target counts as moving if its
    net shift is at least ``min_shift_boxes`` box sizes, or its net size
    change is at least ``min_size_change`` (log scale; ``0.4`` = 50 %, i.e.
    swimming towards or away from the camera). The size threshold is high on
    purpose: a segmentation mask of a standing person "breathes" by up to
    ~45 % over two seconds on real footage (measured), a swimmer heading to or
    from the camera changed by 50-100 %. No flag is raised when fewer than ``min_known``
    of the steps in the window are known - we only flag what we can measure.
    """
    n = len(boxes)
    half = max(1, window // 2)
    flags = [False] * n
    for t in range(n):
        if boxes[t] is None:
            continue
        lo, hi = max(1, t - half), min(n - 1, t + half)
        known = [s for s in steps[lo:hi + 1] if s is not None]
        if hi < lo or len(known) < min_known * (hi - lo + 1) or len(known) < 3:
            continue
        sizes = [max(b.width, b.height) for b in boxes[lo - 1:hi + 1] if b is not None]
        ref = float(np.median(sizes)) if sizes else 0.0
        if ref <= 0:
            continue
        net_dx = sum(s[0] for s in known)
        net_dy = sum(s[1] for s in known)
        net_size = abs(sum(s[2] for s in known))
        flags[t] = math.hypot(net_dx, net_dy) < min_shift_boxes * ref and net_size < min_size_change
    return flags
