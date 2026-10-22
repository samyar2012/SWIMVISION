"""Automatic pool-lane geometry for swimmer detection and tracking.

Lane ropes are unusually strong identity cues: unlike a swimmer's appearance,
they remain useful through splashes, pose changes, and partial occlusion.  This
module finds long, similarly-oriented line segments in each frame, consolidates
duplicate rope edges, and exposes both a separator mask for heatmaps and the
lane corridor surrounding a selected swimmer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LaneBoundary:
    """A lane-rope centreline represented as ``y = slope * x + intercept``."""

    slope: float
    intercept: float
    confidence: float = 1.0

    def y_at(self, x: float) -> float:
        return self.slope * x + self.intercept

    def scaled(self, scale_x: float, scale_y: float) -> "LaneBoundary":
        if scale_x <= 0 or scale_y <= 0:
            raise ValueError("lane-boundary scales must be positive")
        return LaneBoundary(
            self.slope * scale_y / scale_x,
            self.intercept * scale_y,
            self.confidence,
        )


@dataclass(frozen=True)
class LaneCorridor:
    """The two automatically detected ropes bracketing one swimming lane."""

    upper: LaneBoundary
    lower: LaneBoundary
    frame_width: int
    frame_height: int

    def bounds_at(self, x: float) -> tuple[float, float]:
        first, second = self.upper.y_at(x), self.lower.y_at(x)
        return (first, second) if first <= second else (second, first)

    def coordinate(self, point: tuple[float, float]) -> float:
        """Return cross-lane position: 0=upper rope, 1=lower rope."""
        x, y = point
        upper, lower = self.bounds_at(x)
        width = max(1.0, lower - upper)
        return (y - upper) / width

    def contains(self, point: tuple[float, float], margin: float = 0.12) -> bool:
        coordinate = self.coordinate(point)
        return -margin <= coordinate <= 1.0 + margin

    def overlaps_box(self, box: object, minimum_fraction: float = 0.35) -> bool:
        """Whether enough of an axis-aligned box lies between the lane ropes."""
        center_x = (float(box.x1) + float(box.x2)) / 2.0
        upper, lower = self.bounds_at(center_x)
        overlap = max(0.0, min(float(box.y2), lower) - max(float(box.y1), upper))
        height = max(1.0, float(box.y2) - float(box.y1))
        return overlap / height >= minimum_fraction


@dataclass(frozen=True)
class PoolGeometry:
    """Lane-rope boundaries detected in one video frame."""

    boundaries: tuple[LaneBoundary, ...]
    frame_width: int
    frame_height: int

    def separator_mask(
        self,
        width: int,
        height: int,
        *,
        thickness: int | None = None,
    ) -> np.ndarray:
        """Return a mask whose zero-valued lines separate adjacent lane blobs."""
        mask = np.full((height, width), 255, dtype=np.uint8)
        if not self.boundaries:
            return mask
        scale_x = width / self.frame_width
        scale_y = height / self.frame_height
        line_width = thickness or max(2, int(round(height * 0.008)))
        for boundary in self.boundaries:
            scaled = boundary.scaled(scale_x, scale_y)
            y1 = int(round(scaled.y_at(0.0)))
            y2 = int(round(scaled.y_at(float(width - 1))))
            cv2.line(mask, (0, y1), (width - 1, y2), 0, line_width, cv2.LINE_AA)
        return mask

    def corridor_for(
        self,
        point: tuple[float, float],
        *,
        minimum_width: float = 12.0,
        maximum_width_fraction: float = 0.32,
    ) -> LaneCorridor | None:
        """Return the detected lane containing ``point``, when unambiguous."""
        if len(self.boundaries) < 2:
            return None
        x, y = point
        ordered = sorted(self.boundaries, key=lambda line: line.y_at(x))
        for upper, lower in zip(ordered, ordered[1:]):
            upper_y, lower_y = upper.y_at(x), lower.y_at(x)
            width = lower_y - upper_y
            if (
                upper_y <= y <= lower_y
                and width >= minimum_width
                and width <= self.frame_height * maximum_width_fraction
            ):
                return LaneCorridor(upper, lower, self.frame_width, self.frame_height)
        return None


class PoolLaneDetector:
    """Detect long lane-rope boundaries without requiring any user input."""

    def __init__(
        self,
        *,
        maximum_working_width: int = 960,
        maximum_angle_degrees: float = 35.0,
        minimum_length_fraction: float = 0.22,
    ) -> None:
        self.maximum_working_width = maximum_working_width
        self.maximum_angle_degrees = maximum_angle_degrees
        self.minimum_length_fraction = minimum_length_fraction

    def detect(self, frame: np.ndarray) -> PoolGeometry:
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim < 2:
            raise ValueError("frame must be a non-empty image")
        original_height, original_width = frame.shape[:2]
        if original_height <= 0 or original_width <= 0:
            raise ValueError("frame must have positive dimensions")

        scale = min(1.0, self.maximum_working_width / original_width)
        width = max(2, int(round(original_width * scale)))
        height = max(2, int(round(original_height * scale)))
        working = (
            cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            if scale < 1.0
            else frame
        )

        hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        gray_edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 130)
        colour_edges = cv2.Canny(cv2.GaussianBlur(saturation, (5, 5), 0), 35, 110)
        bright_edges = cv2.Canny(cv2.GaussianBlur(value, (5, 5), 0), 55, 150)
        edges = cv2.bitwise_or(gray_edges, cv2.bitwise_or(colour_edges, bright_edges))

        raw_lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=max(24, width // 28),
            minLineLength=max(30, int(round(width * self.minimum_length_fraction))),
            maxLineGap=max(12, int(round(width * 0.06))),
        )
        if raw_lines is None:
            return PoolGeometry((), original_width, original_height)

        candidates: list[tuple[float, float, float]] = []
        maximum_slope = math.tan(math.radians(self.maximum_angle_degrees))
        for x1, y1, x2, y2 in np.asarray(raw_lines).reshape(-1, 4):
            dx, dy = float(x2 - x1), float(y2 - y1)
            if abs(dx) < 1.0:
                continue
            slope = dy / dx
            length = math.hypot(dx, dy)
            if abs(slope) > maximum_slope or length < width * self.minimum_length_fraction:
                continue
            intercept = float(y1) - slope * float(x1)
            confidence = min(1.0, length / max(1.0, width))
            candidates.append((slope, intercept, confidence))

        if len(candidates) < 2:
            return PoolGeometry((), original_width, original_height)

        slopes = np.asarray([item[0] for item in candidates], dtype=np.float32)
        dominant_slope = float(np.median(slopes))
        candidates = [item for item in candidates if abs(item[0] - dominant_slope) <= 0.22]
        if len(candidates) < 2:
            return PoolGeometry((), original_width, original_height)

        centre_x = width / 2.0
        candidates.sort(key=lambda item: item[0] * centre_x + item[1])
        cluster_distance = max(5.0, height * 0.025)
        clusters: list[list[tuple[float, float, float]]] = []
        for candidate in candidates:
            y = candidate[0] * centre_x + candidate[1]
            if not clusters:
                clusters.append([candidate])
                continue
            previous_y = float(np.mean([s * centre_x + b for s, b, _ in clusters[-1]]))
            if abs(y - previous_y) <= cluster_distance:
                clusters[-1].append(candidate)
            else:
                clusters.append([candidate])

        boundaries: list[LaneBoundary] = []
        for cluster in clusters:
            weights = np.asarray([max(0.05, item[2]) for item in cluster], dtype=np.float64)
            slopes = np.asarray([item[0] for item in cluster], dtype=np.float64)
            intercepts = np.asarray([item[1] for item in cluster], dtype=np.float64)
            slope = float(np.average(slopes, weights=weights))
            intercept = float(np.average(intercepts, weights=weights))
            confidence = float(min(1.0, weights.sum()))
            boundary = LaneBoundary(slope, intercept, confidence).scaled(1.0 / scale, 1.0 / scale)
            centre_y = boundary.y_at(original_width / 2.0)
            if -original_height * 0.10 <= centre_y <= original_height * 1.10:
                boundaries.append(boundary)

        boundaries.sort(key=lambda line: line.y_at(original_width / 2.0))
        return PoolGeometry(tuple(boundaries), original_width, original_height)
