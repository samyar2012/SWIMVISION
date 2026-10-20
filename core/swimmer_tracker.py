"""Persistent single-swimmer tracking for hybrid detector candidates.

The hybrid detector remains responsible for finding every plausible person or
swimmer in a frame.  This module keeps the identity of one selected swimmer by
scoring those candidates against the previous position, a constant-velocity
prediction, the selected lane and the detector source.

The tracker deliberately uses a small structural interface instead of
importing ``HybridDetection``.  This keeps the detector and tracker separate:
any candidate with ``x1``, ``y1``, ``x2``, ``y2``, ``confidence`` and
``source`` attributes can be tracked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from core.geometry import BBox


class DetectionLike(Protocol):
    """The part of ``HybridDetection`` used by the tracker."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    source: str


class HybridDetectorLike(Protocol):
    """Public interface implemented by ``HybridSwimmerDetector``."""

    def detect(self, frame: Any) -> list[DetectionLike]: ...


@dataclass(frozen=True)
class LaneConstraint:
    """Allowed image-coordinate band for the selected lane.

    Most side-on pool videos use ``axis="y"`` (the lane lies between two
    horizontal image coordinates).  Use ``axis="x"`` for a rotated/end-on
    view.  Candidate centres outside the band are rejected, not merely
    penalised, which prevents a confident adjacent-lane swimmer taking over.
    """

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
        """Return a centre-weighted lane score, or ``None`` outside the lane."""
        value = center[0] if self.axis == "x" else center[1]
        if not self.minimum <= value <= self.maximum:
            return None
        midpoint = (self.minimum + self.maximum) / 2.0
        half_width = (self.maximum - self.minimum) / 2.0
        return max(0.0, 1.0 - 0.35 * abs(value - midpoint) / half_width)


@dataclass(frozen=True)
class TrackedSwimmer:
    """One tracker output: a detector observation or a short prediction."""

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


class PersistentSwimmerTracker:
    """Track one selected swimmer across detector handoffs and short gaps.

    Call :meth:`start_from_box` with the user's initial selection, then call
    :meth:`track` for each frame when a detector was supplied to ``__init__``.
    Tests and other pipelines can call :meth:`update` with an already-produced
    candidate list instead.
    """

    DEFAULT_SOURCE_SCORES = {
        "person": 0.82,
        "swimmer": 0.94,
        "both": 1.0,
    }

    def __init__(
        self,
        detector: HybridDetectorLike | None = None,
        *,
        lane: LaneConstraint | None = None,
        max_missed_frames: int = 15,
        max_reacquire_frames: int = 90,
        max_jump_fraction: float = 0.08,
        lane_tolerance_fraction: float = 0.15,
        velocity_smoothing: float = 0.65,
        min_candidate_score: float = 0.35,
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
        if not 0.0 <= min_candidate_score <= 1.0:
            raise ValueError("min_candidate_score must be in [0, 1]")

        self.detector = detector
        self.lane = lane
        self.max_missed_frames = max_missed_frames
        self.max_reacquire_frames = max_reacquire_frames
        self.max_jump_fraction = max_jump_fraction
        self.lane_tolerance_fraction = lane_tolerance_fraction
        self.velocity_smoothing = velocity_smoothing
        self.min_candidate_score = min_candidate_score
        self.source_scores = dict(self.DEFAULT_SOURCE_SCORES)
        if source_scores:
            self.source_scores.update({key.lower(): float(value) for key, value in source_scores.items()})

        self._box: BBox | None = None
        self._velocity = (0.0, 0.0)
        self._missed_frames = 0
        self._frame_shape: tuple[int, int] | None = None
        self._lane_center: float | None = None
        self._last_source = "selection"

    @property
    def initialized(self) -> bool:
        return self._box is not None

    @property
    def missed_frames(self) -> int:
        return self._missed_frames

    def reset(self) -> None:
        """Forget the selected swimmer so a new one can be chosen explicitly."""
        self._box = None
        self._velocity = (0.0, 0.0)
        self._missed_frames = 0
        self._frame_shape = None
        self._lane_center = None
        self._last_source = "selection"

    def start_from_box(
        self,
        box: BBox | Sequence[float],
        frame_shape: Sequence[int],
        *,
        source: str = "selection",
    ) -> TrackedSwimmer:
        """Initialize identity from a user-selected box in original pixels."""
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
        if self.lane is None:
            self._lane_center = self._cross_lane_value(selected.center)
        return TrackedSwimmer(selected, 1.0, source, 1.0, False, 0)

    def start(
        self,
        detections: Sequence[DetectionLike],
        frame_shape: Sequence[int],
        *,
        target_point: tuple[float, float] | None = None,
        target_box: BBox | Sequence[float] | None = None,
    ) -> TrackedSwimmer:
        """Initialize from the candidate closest to a user point or box.

        ``target_box`` is preferred when both selectors are supplied.  With no
        selector, the best confidence/source candidate is used; interactive
        applications should normally provide the user's selection.
        """
        height, width = self._normalise_frame_shape(frame_shape)
        choices = [(d, self._box_for(d, width, height)) for d in detections]
        choices = [(d, b) for d, b in choices if b is not None]
        if not choices:
            raise ValueError("cannot initialize the tracker without a valid detection")

        if target_box is not None:
            selection = target_box if isinstance(target_box, BBox) else BBox.from_list(list(target_box))
            sx, sy = selection.center

            def rank(item: tuple[DetectionLike, BBox]) -> tuple[float, float]:
                detection, candidate_box = item
                distance = math.hypot(candidate_box.center[0] - sx, candidate_box.center[1] - sy)
                return (candidate_box.iou(selection), -distance)

            chosen, box = max(choices, key=rank)
        elif target_point is not None:
            tx, ty = target_point

            def point_rank(item: tuple[DetectionLike, BBox]) -> tuple[int, float]:
                _, candidate_box = item
                contains = candidate_box.x1 <= tx <= candidate_box.x2 and candidate_box.y1 <= ty <= candidate_box.y2
                distance = math.hypot(candidate_box.center[0] - tx, candidate_box.center[1] - ty)
                return (int(contains), -distance)

            chosen, box = max(choices, key=point_rank)
        else:
            chosen, box = max(
                choices,
                key=lambda item: self._confidence(item[0]) * self._source_score(item[0]),
            )

        result = self.start_from_box(box, (height, width), source=str(chosen.source).lower())
        return TrackedSwimmer(
            result.box,
            self._confidence(chosen),
            result.source,
            1.0,
            False,
            0,
            chosen,
        )

    def track(self, frame: Any) -> TrackedSwimmer | None:
        """Run the configured hybrid detector and update the persistent track."""
        if self.detector is None:
            raise RuntimeError("track(frame) requires a detector; use update(detections, frame_shape) instead")
        if not hasattr(frame, "shape") or len(frame.shape) < 2:
            raise ValueError("frame must expose a numpy-like (height, width, ...) shape")
        return self.update(self.detector.detect(frame), frame.shape[:2])

    def update(
        self,
        detections: Sequence[DetectionLike],
        frame_shape: Sequence[int],
    ) -> TrackedSwimmer | None:
        """Choose the plausible candidate for this frame or predict through a gap."""
        height, width = self._normalise_frame_shape(frame_shape)
        if self._box is None:
            if not detections:
                return None
            return self.start(detections, (height, width))
        self._frame_shape = (height, width)
        if self._missed_frames >= self.max_reacquire_frames:
            # Identity memory has expired.  Never auto-initialize here because
            # that could silently switch the track to another swimmer.
            return None

        steps = self._missed_frames + 1
        scored: list[_CandidateScore] = []
        for detection in detections:
            box = self._box_for(detection, width, height)
            if box is None:
                continue
            score = self._score_candidate(detection, box, steps, width, height)
            if score is not None:
                scored.append(_CandidateScore(detection, box, score))

        best = max(scored, key=lambda item: item.total, default=None)
        if best is not None and best.total >= self.min_candidate_score:
            return self._accept(best, steps)
        return self._miss(width, height)

    def _score_candidate(
        self,
        detection: DetectionLike,
        box: BBox,
        steps: int,
        width: int,
        height: int,
    ) -> float | None:
        assert self._box is not None
        last_center = self._box.center
        center = box.center
        frame_diagonal = math.hypot(width, height)
        box_allowance = 0.5 * max(math.hypot(self._box.width, self._box.height),
                                  math.hypot(box.width, box.height))
        max_step = max(8.0, frame_diagonal * self.max_jump_fraction)

        predicted_center = (
            last_center[0] + self._velocity[0] * steps,
            last_center[1] + self._velocity[1] * steps,
        )
        from_last = math.dist(center, last_center)
        from_prediction = math.dist(center, predicted_center)
        position_gate = max_step * steps + box_allowance
        prediction_gate = max_step * math.sqrt(steps) + box_allowance

        # Two hard speed gates make high-confidence spectators and distant
        # adjacent swimmers ineligible before confidence is considered.
        if from_last > position_gate or from_prediction > prediction_gate:
            return None

        lane_score = self._lane_score(center, width, height, steps)
        if lane_score is None:
            return None

        position_score = max(0.0, 1.0 - from_last / position_gate)
        motion_score = max(0.0, 1.0 - from_prediction / prediction_gate)
        confidence_score = self._confidence(detection)
        source_score = self._source_score(detection)

        return (
            0.25 * position_score
            + 0.35 * motion_score
            + 0.15 * lane_score
            + 0.15 * confidence_score
            + 0.10 * source_score
        )

    def _lane_score(
        self,
        center: tuple[float, float],
        width: int,
        height: int,
        steps: int,
    ) -> float | None:
        if self.lane is not None:
            return self.lane.score(center)
        if self._lane_center is None:
            return 1.0

        cross_size = width if self._lane_axis() == "x" else height
        tolerance = max(20.0, cross_size * self.lane_tolerance_fraction)
        # A little extra uncertainty is allowed during a detector gap, but the
        # band never expands enough to make lane identity meaningless.
        tolerance *= min(1.5, 1.0 + 0.03 * (steps - 1))
        distance = abs(self._cross_lane_value(center) - self._lane_center)
        if distance > tolerance:
            return None
        return max(0.0, 1.0 - 0.5 * distance / tolerance)

    def _accept(self, candidate: _CandidateScore, steps: int) -> TrackedSwimmer:
        assert self._box is not None
        old_center = self._box.center
        new_center = candidate.box.center
        measured = ((new_center[0] - old_center[0]) / steps,
                    (new_center[1] - old_center[1]) / steps)
        keep = 1.0 - self.velocity_smoothing
        self._velocity = (
            keep * self._velocity[0] + self.velocity_smoothing * measured[0],
            keep * self._velocity[1] + self.velocity_smoothing * measured[1],
        )
        self._box = candidate.box
        self._missed_frames = 0
        self._last_source = str(candidate.detection.source).lower()

        # Slowly adapt an inferred lane centre to perspective/camera motion;
        # explicit lane bounds remain fixed.
        if self.lane is None and self._lane_center is not None:
            value = self._cross_lane_value(new_center)
            self._lane_center = 0.98 * self._lane_center + 0.02 * value

        return TrackedSwimmer(
            candidate.box,
            self._confidence(candidate.detection),
            self._last_source,
            candidate.total,
            False,
            0,
            candidate.detection,
        )

    def _miss(self, width: int, height: int) -> TrackedSwimmer | None:
        assert self._box is not None
        self._missed_frames += 1
        if self._missed_frames > self.max_reacquire_frames:
            # Do not silently select a new high-confidence person.  The user
            # must reset/reselect after identity memory expires.
            return None
        if self._missed_frames > self.max_missed_frames:
            # Keep the last observation internally for a spatially plausible
            # reacquisition, but stop displaying increasingly stale boxes.
            return None

        dx = self._velocity[0] * self._missed_frames
        dy = self._velocity[1] * self._missed_frames
        predicted = BBox(
            self._box.x1 + dx,
            self._box.y1 + dy,
            self._box.x2 + dx,
            self._box.y2 + dy,
        ).clamp(width, height)
        confidence = max(0.0, 1.0 - self._missed_frames / (self.max_missed_frames + 1))
        return TrackedSwimmer(
            predicted,
            confidence,
            f"predicted:{self._last_source}",
            0.0,
            True,
            self._missed_frames,
        )

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
