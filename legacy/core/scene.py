"""Camera-cut detection.

Why this exists: a tracker that "loses" a swimmer because the video cut to a
different camera angle can easily lock on to something else and keep reporting
a healthy score. SwimVision must never silently continue across a cut, so every
tracking pass compares each analysed frame with the previous one and stops the
track if the picture changed completely.

Two cheap signals are compared between consecutive analysed frames:

1. Histogram correlation of grey levels (1.0 = same brightness distribution).
   Measured on real footage: median 0.995 inside a shot, below ~0.4 at cuts.
2. Mean absolute difference of tiny 32x24 grey thumbnails (0-255 scale).
   Median about 3-5 inside a shot; cuts between similar-looking scenes
   (e.g. two shots of grey water, which fool the histogram) reach 35+.

Either signal firing counts as a cut. We deliberately prefer a false alarm
(tracking stops and the user is told) over silently following across shots.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

HIST_CUT_THRESHOLD = 0.60      # correlation below this => cut
THUMB_CUT_THRESHOLD = 35.0     # mean abs thumbnail difference above this => cut


@dataclass(frozen=True)
class FrameSignature:
    """A tiny fingerprint of a frame used only for cut detection."""

    hist: np.ndarray    # 32-bin normalised grey histogram
    thumb: np.ndarray   # 32x24 grey thumbnail (float32)


def frame_signature(frame_bgr: np.ndarray) -> FrameSignature:
    """Compute the fingerprint of one BGR frame."""
    small = cv2.resize(frame_bgr, (160, 120), interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hist = cv2.normalize(cv2.calcHist([grey], [0], None, [32], [0, 256]), None).flatten()
    thumb = cv2.resize(grey, (32, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
    return FrameSignature(hist.astype(np.float32), thumb)


def histogram_similarity(a: FrameSignature, b: FrameSignature) -> float:
    """Histogram correlation in [-1, 1]; higher = more similar pictures."""
    return float(cv2.compareHist(a.hist, b.hist, cv2.HISTCMP_CORREL))


def thumbnail_difference(a: FrameSignature, b: FrameSignature) -> float:
    """Average per-pixel grey difference (0-255) between the thumbnails."""
    return float(np.abs(a.thumb - b.thumb).mean())


def is_scene_cut(a: FrameSignature, b: FrameSignature) -> bool:
    """True if two consecutive frames look like different camera shots."""
    return (histogram_similarity(a, b) < HIST_CUT_THRESHOLD
            or thumbnail_difference(a, b) > THUMB_CUT_THRESHOLD)
