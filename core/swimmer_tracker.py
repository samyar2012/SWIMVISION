"""Identity-aware tracking for one user-selected swimmer.

The hybrid detector proposes every plausible person/swimmer in a frame. This
module keeps the user's swimmer using appearance, a hard inferred lane
corridor, damped motion prediction, and conservative reacquisition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import cv2
import numpy as np

from core.geometry import BBox


class DetectionLike(Protocol):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    source: str


class HybridDetectorLike(Protocol):
    def detect(self, frame: Any) -> list[DetectionLike]: ...


@dataclass(frozen=True)
class LaneConstraint:
    """Allowed image-coordinate band for a lane (normally the y axis)."""

    minimum: float
    maximum: float
    axis: str = "y"

    def __post_init__(self) -> None:
        if self.axis not in ("x", "y"):
            raise ValueError("lane axis must be 'x' or 'y'")
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError("lane bounds must be finite")
        if self.minimum >= self.maximum:
            raise ValueError("lane minimum must be less than lane maximum")

    def score(self, center: tuple[float, float]) -> float | None:
        value = center[0] if self.axis == "x" else center[1]
        if not self.minimum <= value <= self.maximum:
            return None
        midpoint = (self.minimum + self.maximum) / 2.0
        half_width = (self.maximum - self.minimum) / 2.0
        return max(0.0, 1.0 - 0.35 * abs(value - midpoint) / half_width)


@dataclass(frozen=True)
class TrackedSwimmer:
    box: BBox
    confidence: float
    source: str
    score: float
    predicted: bool
    missed_frames: int
    detection: DetectionLike | None = None

    @property
    def x1(self) -> float:
        return self.box.x1

    @property
    def y1(self) -> float:
        return self.box.y1

    @property
    def x2(self) -> float:
        return self.box.x2

    @property
    def y2(self) -> float:
        return self.box.y2

    @property
    def center(self) -> tuple[float, float]:
        return self.box.center


@dataclass(frozen=True)
class _CandidateScore:
    detection: DetectionLike
    box: BBox
    total: float
    appearance: float | None
    descriptor: np.ndarray | None


class PersistentSwimmerTracker:
    """Track exactly one selected swimmer without automatic identity changes."""

    DEFAULT_SOURCE_SCORES = {"person": 0.82, "swimmer": 0.94, "both": 1.0}

    def __init__(
        self,
        detector: HybridDetectorLike | None = None,
        *,
        lane: LaneConstraint | None = None,
        max_missed_frames: int = 12,
        max_reacquire_frames: int = 75,
        max_jump_fraction: float = 0.05,
        lane_tolerance_fraction: float = 0.07,
        velocity_smoothing: float = 0.55,
        prediction_decay: float = 0.86,
        min_candidate_score: float = 0.40,
        min_appearance_similarity: float = 0.30,
        appearance_adaptation: float = 0.06,
        reacquire_confirmations: int = 2,
        source_scores: dict[str, float] | None = None,
    ) -> None:
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative")
        if max_reacquire_frames < max_missed_frames:
            raise ValueError("max_reacquire_frames must be at least max_missed_frames")
        if not 0.0 < max_jump_fraction <= 1.0:
            raise ValueError("max_jump_fraction must be in (0, 1]")
        if not 0.0 < lane_tolerance_fraction <= 1.0:
            raise ValueError("lane_tolerance_fraction must be in (0, 1]")
        if not 0.0 <= velocity_smoothing <= 1.0:
            raise ValueError("velocity_smoothing must be in [0, 1]")
        if not 0.0 < prediction_decay <= 1.0:
            raise ValueError("prediction_decay must be in (0, 1]")
        if not 0.0 <= min_candidate_score <= 1.0:
            raise ValueError("min_candidate_score must be in [0, 1]")
        if not 0.0 <= min_appearance_similarity <= 1.0:
            raise ValueError("min_appearance_similarity must be in [0, 1]")
        if not 0.0 <= appearance_adaptation <= 1.0:
            raise ValueError("appearance_adaptation must be in [0, 1]")
        if reacquire_confirmations < 1:
            raise ValueError("reacquire_confirmations must be positive")

        self.detector = detector
        self.lane = lane
        self.max_missed_frames = max_missed_frames
        self.max_reacquire_frames = max_reacquire_frames
        self.max_jump_fraction = max_jump_fraction
        self.lane_tolerance_fraction = lane_tolerance_fraction
        self.velocity_smoothing = velocity_smoothing
        self.prediction_decay = prediction_decay
        self.min_candidate_score = min_candidate_score
        self.min_appearance_similarity = min_appearance_similarity
        self.appearance_adaptation = appearance_adaptation
        self.reacquire_confirmations = reacquire_confirmations
        self.source_scores = dict(self.DEFAULT_SOURCE_SCORES)
        if source_scores:
            self.source_scores.update({key.lower(): float(value) for key, value in source_scores.items()})

        self._box: BBox | None = None
        self._velocity = (0.0, 0.0)
        self._missed_frames = 0
        self._frame_shape: tuple[int, int] | None = None
        self._lane_center: float | None = None
        self._lane_half_width: float | None = None
        self._last_source = "selection"
        self._appearance_reference: np.ndarray | None = None
        self._appearance_adaptive: np.ndarray | None = None
        self._pending_box: BBox | None = None
        self._pending_count = 0

    @property
    def initialized(self) -> bool:
        return self._box is not None

    @property
    def missed_frames(self) -> int:
        return self._missed_frames

    def reset(self) -> None:
        self._box = None
        self._velocity = (0.0, 0.0)
        self._missed_frames = 0
        self._frame_shape = None
        self._lane_center = None
        self._lane_half_width = None
        self._last_source = "selection"
        self._appearance_reference = None
        self._appearance_adaptive = None
        self._clear_pending()

    def start_from_box(
        self,
        box: BBox | Sequence[float],
        frame_shape: Sequence[int],
        *,
        source: str = "selection",
        frame: Any | None = None,
    ) -> TrackedSwimmer:
        """Initialize from the user's box and optionally remember its pixels."""
        height, width = self._normalise_frame_shape(frame_shape)
        selected = box if isinstance(box, BBox) else BBox.from_list(list(box))
        selected = selected.clamp(width, height)
        if selected.width <= 0 or selected.height <= 0:
            raise ValueError("the initial swimmer box must have positive width and height")

        self._box = selected
        self._velocity = (0.0, 0.0)
        self._missed_frames = 0
        self._frame_shape = (height, width)
        self._last_source = source
        self._clear_pending()
        if self.lane is None:
            self._lane_center = self._cross_lane_value(selected.center)
            cross_size = width if self._lane_axis() == "x" else height
            cross_box_size = selected.width if self._lane_axis() == "x" else selected.height
            self._lane_half_width = max(
                16.0,
                min(cross_size * self.lane_tolerance_fraction,
                    max(cross_size * 0.035, cross_box_size * 1.10)),
            )
        descriptor = self._appearance_descriptor(frame, selected)
        self._appearance_reference = descriptor
        self._appearance_adaptive = descriptor.copy() if descriptor is not None else None
        return TrackedSwimmer(selected, 1.0, source, 1.0, False, 0)

    def start(
        self,
        detections: Sequence[DetectionLike],
        frame_shape: Sequence[int],
        *,
        target_point: tuple[float, float] | None = None,
        target_box: BBox | Sequence[float] | None = None,
        frame: Any | None = None,
    ) -> TrackedSwimmer:
        """Initialize on the detection nearest the user's visual selection."""
        height, width = self._normalise_frame_shape(frame_shape)
        choices = [(d, self._box_for(d, width, height)) for d in detections]
        choices = [(d, b) for d, b in choices if b is not None]
        if not choices:
            raise ValueError("cannot initialize the tracker without a valid detection")

        identity_box: BBox | None = None
        if target_box is not None:
            identity_box = target_box if isinstance(target_box, BBox) else BBox.from_list(list(target_box))
            identity_box = identity_box.clamp(width, height)
            sx, sy = identity_box.center

            def rank(item: tuple[DetectionLike, BBox]) -> tuple[float, float]:
                _, candidate_box = item
                return candidate_box.iou(identity_box), -math.dist(candidate_box.center, (sx, sy))

            chosen, box = max(choices, key=rank)
        elif target_point is not None:
            tx, ty = target_point

            def point_rank(item: tuple[DetectionLike, BBox]) -> tuple[int, float]:
                _, candidate_box = item
                contains = candidate_box.x1 <= tx <= candidate_box.x2 and candidate_box.y1 <= ty <= candidate_box.y2
                return int(contains), -math.dist(candidate_box.center, (tx, ty))

            chosen, box = max(choices, key=point_rank)
        else:
            chosen, box = max(choices, key=lambda item: self._confidence(item[0]) * self._source_score(item[0]))

        result = self.start_from_box(box, (height, width), source=str(chosen.source).lower(), frame=frame)
        if identity_box is not None:
            descriptor = self._appearance_descriptor(frame, identity_box)
            if descriptor is not None:
                self._appearance_reference = descriptor
                self._appearance_adaptive = descriptor.copy()
        self._tighten_lane_from_candidates([candidate_box for _, candidate_box in choices], box)
        return TrackedSwimmer(result.box, self._confidence(chosen), result.source, 1.0, False, 0, chosen)

    def track(self, frame: Any) -> TrackedSwimmer | None:
        if self.detector is None:
            raise RuntimeError("track(frame) requires a detector; use update(...) instead")
        if not hasattr(frame, "shape") or len(frame.shape) < 2:
            raise ValueError("frame must expose a numpy-like (height, width, ...) shape")
        return self.update(self.detector.detect(frame), frame.shape[:2], frame=frame)

    def update(
        self,
        detections: Sequence[DetectionLike],
        frame_shape: Sequence[int],
        *,
        frame: Any | None = None,
    ) -> TrackedSwimmer | None:
        """Accept a trusted observation, predict briefly, or report lost."""
        height, width = self._normalise_frame_shape(frame_shape)
        if self._box is None:
            if not detections:
                return None
            return self.start(detections, (height, width), frame=frame)
        self._frame_shape = (height, width)
        if self._missed_frames >= self.max_reacquire_frames:
            return None

        steps = self._missed_frames + 1
        scored: list[_CandidateScore] = []
        for detection in detections:
            box = self._box_for(detection, width, height)
            if box is None:
                continue
            candidate = self._score_candidate(detection, box, steps, width, height, frame)
            if candidate is not None:
                scored.append(candidate)

        best = max(scored, key=lambda item: item.total, default=None)
        if best is None or best.total < self.min_candidate_score:
            self._clear_pending()
            return self._miss(width, height)

        # After a real gap, require the same candidate twice. A one-frame false
        # positive can therefore never steal the selected identity.
        if self._missed_frames >= 2 and self.reacquire_confirmations > 1:
            if self._pending_matches(best.box, width, height):
                self._pending_count += 1
            else:
                self._pending_box = best.box
                self._pending_count = 1
            if self._pending_count < self.reacquire_confirmations:
                return self._miss(width, height)

        return self._accept(best, steps)

    def _score_candidate(
        self,
        detection: DetectionLike,
        box: BBox,
        steps: int,
        width: int,
        height: int,
        frame: Any | None,
    ) -> _CandidateScore | None:
        assert self._box is not None
        max_step = max(10.0, math.hypot(width, height) * self.max_jump_fraction)
        box_allowance = min(max_step * 0.55, math.hypot(self._box.width, self._box.height) * 0.25)
        from_last = math.dist(box.center, self._box.center)
        from_prediction = math.dist(box.center, self._predicted_center(steps))
        gate = max_step * max(1.0, math.sqrt(steps)) + box_allowance
        if from_last > gate or from_prediction > gate:
            return None

        lane_score = self._lane_score(box.center, width, height)
        if lane_score is None:
            return None
        descriptor = self._appearance_descriptor(frame, box)
        appearance = self._appearance_similarity(descriptor)
        if appearance is not None and appearance < self.min_appearance_similarity:
            return None

        position_score = max(0.0, 1.0 - from_last / gate)
        motion_score = max(0.0, 1.0 - from_prediction / gate)
        confidence_score = self._confidence(detection)
        source_score = self._source_score(detection)
        if appearance is None:
            total = (0.27 * position_score + 0.38 * motion_score + 0.20 * lane_score
                     + 0.10 * confidence_score + 0.05 * source_score)
        else:
            total = (0.08 * position_score + 0.10 * motion_score + 0.12 * lane_score
                     + 0.58 * appearance + 0.07 * confidence_score + 0.05 * source_score)
        return _CandidateScore(detection, box, total, appearance, descriptor)

    def _lane_score(self, center: tuple[float, float], width: int, height: int) -> float | None:
        if self.lane is not None:
            return self.lane.score(center)
        if self._lane_center is None:
            return 1.0
        cross_size = width if self._lane_axis() == "x" else height
        tolerance = self._lane_half_width or max(16.0, cross_size * self.lane_tolerance_fraction)
        distance = abs(self._cross_lane_value(center) - self._lane_center)
        if distance > tolerance:
            return None
        return max(0.0, 1.0 - 0.65 * distance / tolerance)

    def _accept(self, candidate: _CandidateScore, steps: int) -> TrackedSwimmer:
        assert self._box is not None
        old_center, new_center = self._box.center, candidate.box.center
        measured = ((new_center[0] - old_center[0]) / steps, (new_center[1] - old_center[1]) / steps)
        keep = 1.0 - self.velocity_smoothing
        self._velocity = (keep * self._velocity[0] + self.velocity_smoothing * measured[0],
                          keep * self._velocity[1] + self.velocity_smoothing * measured[1])
        self._box = candidate.box
        self._missed_frames = 0
        self._last_source = str(candidate.detection.source).lower()
        self._clear_pending()

        if self.lane is None and self._lane_center is not None:
            value = self._cross_lane_value(new_center)
            self._lane_center = 0.96 * self._lane_center + 0.04 * value
        # Never learn from a weak match; this prevents gradual identity drift.
        if candidate.descriptor is not None and (candidate.appearance or 0.0) >= 0.58 and candidate.total >= 0.58:
            if self._appearance_adaptive is None:
                self._appearance_adaptive = candidate.descriptor.copy()
            else:
                rate = self.appearance_adaptation
                updated = (1.0 - rate) * self._appearance_adaptive + rate * candidate.descriptor
                norm = float(updated.sum())
                self._appearance_adaptive = updated / norm if norm > 0 else updated

        return TrackedSwimmer(candidate.box, self._confidence(candidate.detection), self._last_source,
                              candidate.total, False, 0, candidate.detection)

    def _miss(self, width: int, height: int) -> TrackedSwimmer | None:
        assert self._box is not None
        self._missed_frames += 1
        if self._missed_frames > self.max_reacquire_frames or self._missed_frames > self.max_missed_frames:
            return None
        predicted_center = self._predicted_center(self._missed_frames)
        cx, cy = self._box.center
        predicted = BBox(self._box.x1 + predicted_center[0] - cx,
                         self._box.y1 + predicted_center[1] - cy,
                         self._box.x2 + predicted_center[0] - cx,
                         self._box.y2 + predicted_center[1] - cy).clamp(width, height)
        confidence = max(0.0, 1.0 - self._missed_frames / (self.max_missed_frames + 1))
        return TrackedSwimmer(predicted, confidence, f"predicted:{self._last_source}", 0.0, True,
                              self._missed_frames)

    def _predicted_center(self, steps: int) -> tuple[float, float]:
        assert self._box is not None
        multiplier = (float(steps) if self.prediction_decay == 1.0 else
                      (1.0 - self.prediction_decay ** steps) / (1.0 - self.prediction_decay))
        return (self._box.center[0] + self._velocity[0] * multiplier,
                self._box.center[1] + self._velocity[1] * multiplier)

    def _tighten_lane_from_candidates(self, boxes: Sequence[BBox], selected: BBox) -> None:
        if self.lane is not None or self._lane_center is None or self._lane_half_width is None:
            return
        selected_value = self._cross_lane_value(selected.center)
        cross_box_size = selected.width if self._lane_axis() == "x" else selected.height
        separations = sorted(
            abs(self._cross_lane_value(box.center) - selected_value)
            for box in boxes
            if box != selected
            and abs(self._cross_lane_value(box.center) - selected_value) > max(12.0, cross_box_size * 0.35)
        )
        if separations:
            self._lane_half_width = max(14.0, min(self._lane_half_width, separations[0] * 0.45))

    def _pending_matches(self, box: BBox, width: int, height: int) -> bool:
        if self._pending_box is None:
            return False
        limit = max(12.0, math.hypot(width, height) * self.max_jump_fraction * 0.75,
                    math.hypot(box.width, box.height) * 0.45)
        return math.dist(self._pending_box.center, box.center) <= limit

    def _clear_pending(self) -> None:
        self._pending_box = None
        self._pending_count = 0

    def _appearance_similarity(self, descriptor: np.ndarray | None) -> float | None:
        if descriptor is None or self._appearance_reference is None:
            return None
        fixed = float(np.sqrt(self._appearance_reference * descriptor).sum())
        if self._appearance_adaptive is None:
            return min(1.0, max(0.0, fixed))
        adaptive = float(np.sqrt(self._appearance_adaptive * descriptor).sum())
        return min(1.0, max(0.0, 0.72 * fixed + 0.28 * adaptive))

    @staticmethod
    def _appearance_descriptor(frame: Any | None, box: BBox) -> np.ndarray | None:
        """Return a scale-independent spatial HSV histogram for one crop."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim < 3:
            return None
        height, width = frame.shape[:2]
        x1 = max(0, min(width - 1, int(math.floor(box.x1))))
        y1 = max(0, min(height - 1, int(math.floor(box.y1))))
        x2 = max(x1 + 1, min(width, int(math.ceil(box.x2))))
        y2 = max(y1 + 1, min(height, int(math.ceil(box.y2))))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
        parts: list[np.ndarray] = []
        for row in range(2):
            for col in range(2):
                cell = hsv[row * 24:(row + 1) * 24, col * 24:(col + 1) * 24]
                parts.extend((cv2.calcHist([cell], [0, 1], None, [12, 8], [0, 180, 0, 256]).reshape(-1),
                              cv2.calcHist([cell], [2], None, [8], [0, 256]).reshape(-1)))
        descriptor = np.concatenate(parts).astype(np.float32)
        total = float(descriptor.sum())
        return descriptor / total if total > 0 else None

    def _lane_axis(self) -> str:
        return self.lane.axis if self.lane is not None else "y"

    def _cross_lane_value(self, center: tuple[float, float]) -> float:
        return center[0] if self._lane_axis() == "x" else center[1]

    def _source_score(self, detection: DetectionLike) -> float:
        return min(1.0, max(0.0, self.source_scores.get(str(detection.source).lower(), 0.75)))

    @staticmethod
    def _confidence(detection: DetectionLike) -> float:
        try:
            return min(1.0, max(0.0, float(detection.confidence)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _box_for(detection: DetectionLike, width: int, height: int) -> BBox | None:
        try:
            values = tuple(float(getattr(detection, name)) for name in ("x1", "y1", "x2", "y2"))
        except (AttributeError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in values):
            return None
        box = BBox(*values).clamp(width, height)
        return box if box.width > 0 and box.height > 0 else None

    @staticmethod
    def _normalise_frame_shape(frame_shape: Sequence[int]) -> tuple[int, int]:
        if len(frame_shape) < 2:
            raise ValueError("frame_shape must contain height and width")
        height, width = int(frame_shape[0]), int(frame_shape[1])
        if height <= 0 or width <= 0:
            raise ValueError("frame height and width must be positive")
        return height, width
