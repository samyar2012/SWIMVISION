"""Lock one user-selected swimmer to an established BoT-SORT track ID.

The hybrid detector still supplies YOLO observations before water entry and
U-Net heatmap observations in the water. BoT-SORT associates *all* swimmers
over time using a Kalman state, two-stage high/low-confidence matching, and
camera-motion compensation. The box drawn by the user chooses one resulting
track ID; this module never substitutes a different ID later.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Any, Protocol, Sequence

import numpy as np

from core.geometry import BBox
from core.pool_geometry import LaneCorridor, PoolGeometry


class DetectionLike(Protocol):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    source: str


class HybridDetectorLike(Protocol):
    def detect(self, frame: Any) -> list[DetectionLike]: ...


class TrackingPhase(str, Enum):
    """Race phases with meaningfully different motion behaviour."""

    START = "start"
    DIVE = "dive"
    SWIMMING = "swimming"
    FINISH = "finish"


@dataclass(frozen=True)
class LaneConstraint:
    """Legacy explicit lane band, retained for script/API compatibility."""

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
        return max(0.0, 1.0 - abs(value - midpoint) / half_width)


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
class _SelectionDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    source: str = "selection"


class _DetectionBatch:
    """Small NumPy-backed adapter expected by Ultralytics' tracker API."""

    def __init__(self, rows: np.ndarray) -> None:
        rows = np.asarray(rows, dtype=np.float32).reshape(-1, 5)
        self.xyxy = rows[:, :4]
        self.conf = rows[:, 4]
        self.cls = np.zeros(len(rows), dtype=np.float32)

    @property
    def xywh(self) -> np.ndarray:
        result = self.xyxy.copy()
        result[:, 0] = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2.0
        result[:, 1] = (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2.0
        result[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]
        result[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]
        return result

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index: Any) -> "_DetectionBatch":
        xyxy = np.atleast_2d(self.xyxy[index])
        conf = np.atleast_1d(self.conf[index])[:, None]
        return _DetectionBatch(np.concatenate((xyxy, conf), axis=1))


class PersistentSwimmerTracker:
    """Use BoT-SORT for association and expose only the selected track ID."""

    def __init__(
        self,
        detector: HybridDetectorLike | None = None,
        *,
        lane: LaneConstraint | None = None,
        max_missed_frames: int = 12,
        max_reacquire_frames: int = 120,
        **_: Any,
    ) -> None:
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative")
        if max_reacquire_frames < max_missed_frames:
            raise ValueError("max_reacquire_frames must be at least max_missed_frames")
        self.detector = detector
        self.lane = lane
        self.max_missed_frames = max_missed_frames
        self.max_reacquire_frames = max_reacquire_frames
        self._backend = self._make_backend(max_reacquire_frames)
        self._selection: BBox | None = None
        self._target_id: int | None = None
        self._box: BBox | None = None
        self._missed_frames = 0
        self._last_source = "selection"
        self._advanced = False
        self._water_locked = False
        self._lane_corridor: LaneCorridor | None = None
        self._lane_geometry_misses = 0
        self._phase = TrackingPhase.START
        self._water_observations = 0
        self._centres: deque[tuple[float, float]] = deque(maxlen=20)
        self._areas: deque[float] = deque(maxlen=20)
        self._travel_direction = 0

    @staticmethod
    def _make_backend(track_buffer: int) -> Any:
        try:
            from ultralytics.trackers.bot_sort import BOTSORT
        except ImportError as exc:
            raise RuntimeError(
                "BoT-SORT dependencies are missing. Install the project requirements "
                "(including lap>=0.5.12) and restart SwimVision."
            ) from exc
        args = SimpleNamespace(
            # Existing tracks may use weak U-Net heatmap observations. New
            # identities still require 0.30, so background heat cannot spawn
            # a swarm of tracks by itself.
            track_high_thresh=0.20,
            track_low_thresh=0.08,
            new_track_thresh=0.30,
            track_buffer=track_buffer,
            match_thresh=0.70,
            # Heatmap confidence is not calibrated like YOLO confidence; do
            # not multiply it into the IoU association distance.
            fuse_score=False,
            gmc_method="sparseOptFlow",
            proximity_thresh=0.50,
            appearance_thresh=0.80,
            with_reid=False,
            model="auto",
            device="cpu",
        )
        return BOTSORT(args)

    @property
    def initialized(self) -> bool:
        return self._selection is not None

    @property
    def missed_frames(self) -> int:
        return self._missed_frames

    @property
    def target_id(self) -> int | None:
        return self._target_id

    @property
    def water_locked(self) -> bool:
        """Whether the selected ID has handed off to the U-Net heatmap."""
        return self._water_locked

    @property
    def phase(self) -> TrackingPhase:
        return self._phase

    @property
    def lane_locked(self) -> bool:
        return self._lane_corridor is not None or self.lane is not None

    def reset(self) -> None:
        self._backend.reset()
        self._selection = None
        self._target_id = None
        self._box = None
        self._missed_frames = 0
        self._last_source = "selection"
        self._advanced = False
        self._water_locked = False
        self._lane_corridor = None
        self._lane_geometry_misses = 0
        self._phase = TrackingPhase.START
        self._water_observations = 0
        self._centres.clear()
        self._areas.clear()
        self._travel_direction = 0

    def start_from_box(
        self,
        box: BBox | Sequence[float],
        frame_shape: Sequence[int],
        *,
        source: str = "selection",
        frame: Any | None = None,
    ) -> TrackedSwimmer:
        height, width = self._normalise_frame_shape(frame_shape)
        selected = box if isinstance(box, BBox) else BBox.from_list(list(box))
        selected = selected.clamp(width, height)
        if selected.width <= 0 or selected.height <= 0:
            raise ValueError("the initial swimmer box must have positive width and height")
        self._selection = selected
        self._box = selected
        self._last_source = source
        result = self._advance([], (height, width), frame, force_selection=True)
        return result or TrackedSwimmer(selected, 1.0, source, 1.0, False, 0)

    def start(
        self,
        detections: Sequence[DetectionLike],
        frame_shape: Sequence[int],
        *,
        target_point: tuple[float, float] | None = None,
        target_box: BBox | Sequence[float] | None = None,
        frame: Any | None = None,
    ) -> TrackedSwimmer:
        height, width = self._normalise_frame_shape(frame_shape)
        if target_box is not None:
            selection = target_box if isinstance(target_box, BBox) else BBox.from_list(list(target_box))
        elif target_point is not None:
            tx, ty = target_point
            boxes = [(d, self._box_for(d, width, height)) for d in detections]
            valid = [(d, box) for d, box in boxes if box is not None]
            if not valid:
                raise ValueError("cannot initialize the tracker without a valid detection")
            _, selection = max(
                valid,
                key=lambda item: (
                    int(item[1].x1 <= tx <= item[1].x2 and item[1].y1 <= ty <= item[1].y2),
                    -math.dist(item[1].center, (tx, ty)),
                ),
            )
        else:
            valid = [(d, self._box_for(d, width, height)) for d in detections]
            valid = [(d, box) for d, box in valid if box is not None]
            if not valid:
                raise ValueError("cannot initialize the tracker without a valid detection")
            detection, selection = max(valid, key=lambda item: float(item[0].confidence))
            self._last_source = str(detection.source).lower()

        self._selection = selection.clamp(width, height)
        self._box = self._selection
        result = self._advance(detections, (height, width), frame, force_selection=True)
        return result or TrackedSwimmer(self._selection, 1.0, "selection", 1.0, False, 0)

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
        height, width = self._normalise_frame_shape(frame_shape)
        if self._selection is None:
            if not detections:
                return None
            return self.start(detections, (height, width), frame=frame)
        return self._advance(detections, (height, width), frame, force_selection=not self._advanced)

    def _advance(
        self,
        detections: Sequence[DetectionLike],
        frame_shape: tuple[int, int],
        frame: Any | None,
        *,
        force_selection: bool,
    ) -> TrackedSwimmer | None:
        assert self._selection is not None
        height, width = frame_shape
        self._refresh_lane_corridor(width, height)
        valid: list[DetectionLike] = []
        boxes: list[BBox] = []
        for detection in detections:
            # Once the selected swimmer has a real heatmap observation, keep
            # the working in-water detector isolated. Pool-deck people from
            # YOLO are no longer eligible to take part in association.
            if self._water_locked and str(detection.source).lower() == "person":
                continue
            box = self._box_for(detection, width, height)
            if box is None:
                continue
            source = str(detection.source).lower()
            if source in {"swimmer", "both"}:
                if not self._inside_locked_lane(box):
                    continue
                if self._water_locked and not self._plausible_target(box, width, height):
                    continue
            valid.append(detection)
            boxes.append(box)

        # A user selection becomes the initial BoT-SORT observation only when
        # no detector box already represents it. This guarantees a track ID
        # even on a weak detector frame without creating normal duplicates.
        best_overlap = max((box.iou(self._selection) for box in boxes), default=0.0)
        if force_selection and best_overlap < 0.20:
            synthetic = _SelectionDetection(*self._selection.to_list())
            valid.append(synthetic)
            boxes.append(self._selection)

        rows = np.asarray(
            [[box.x1, box.y1, box.x2, box.y2, self._confidence(detection)]
             for detection, box in zip(valid, boxes)],
            dtype=np.float32,
        ).reshape(-1, 5)
        image = frame if isinstance(frame, np.ndarray) else np.zeros((height, width, 3), dtype=np.uint8)
        tracked = self._backend.update(_DetectionBatch(rows), image)
        self._advanced = True

        if self._target_id is None and len(tracked):
            self._target_id = self._choose_initial_id(tracked, self._selection)

        target_row = next((row for row in tracked if int(round(row[4])) == self._target_id), None)
        if target_row is not None:
            box = BBox(*[float(value) for value in target_row[:4]]).clamp(width, height)
            index = int(round(target_row[7]))
            detection = valid[index] if 0 <= index < len(valid) else None
            source = str(detection.source).lower() if detection is not None else self._last_source
            if self.lane is not None and self.lane.score(box.center) is None:
                return self._missing_target(width, height)
            if source in {"swimmer", "both"} and not self._inside_locked_lane(box):
                return self._missing_target(width, height)
            self._box = box
            self._missed_frames = 0
            self._last_source = source
            if source in {"swimmer", "both"}:
                if not self._water_locked:
                    self._phase = TrackingPhase.DIVE
                self._water_locked = True
                self._record_water_observation(box, width, height)
            confidence = float(target_row[5])
            return TrackedSwimmer(box, confidence, source, confidence, False, 0, detection)

        return self._missing_target(width, height)

    def _missing_target(self, width: int, height: int) -> TrackedSwimmer | None:
        self._missed_frames += 1
        if self._target_id is None or self._missed_frames > self.max_missed_frames:
            return None
        # At a finish/turn the swimmer can become stationary at the wall.
        # Continuing a Kalman velocity here is exactly how a box flies into an
        # adjacent lane, so hold the last observation until detection returns.
        predicted_box = (
            self._box
            if self._phase is TrackingPhase.FINISH
            else self._lost_track_box(width, height) or self._box
        )
        if predicted_box is None:
            return None
        self._box = predicted_box
        confidence = max(0.0, 1.0 - self._missed_frames / (self.max_missed_frames + 1))
        return TrackedSwimmer(predicted_box, confidence, f"predicted:{self._last_source}", 0.0, True,
                              self._missed_frames)

    def _lost_track_box(self, width: int, height: int) -> BBox | None:
        for track in [*self._backend.tracked_stracks, *self._backend.lost_stracks]:
            if int(track.track_id) == self._target_id:
                return BBox(*[float(value) for value in track.xyxy]).clamp(width, height)
        return None

    def _refresh_lane_corridor(self, width: int, height: int) -> None:
        geometry = getattr(self.detector, "last_lane_geometry", None)
        if not isinstance(geometry, PoolGeometry):
            self._lane_geometry_misses += 1
            if self._lane_geometry_misses > 10:
                self._lane_corridor = None
            return
        anchor = (self._box or self._selection)
        if anchor is None:
            return
        # An upright swimmer on the block can be taller than the apparent
        # lane width, so lane validity must be relative to the frame rather
        # than to the selected person's temporary box shape.
        minimum_width = max(8.0, height * 0.025)
        corridor = geometry.corridor_for(anchor.center, minimum_width=minimum_width)
        if corridor is not None:
            self._lane_corridor = corridor
            self._lane_geometry_misses = 0

    def _inside_locked_lane(self, box: BBox) -> bool:
        if self._lane_corridor is None:
            return True
        return self._lane_corridor.contains(box.center, margin=0.08) and self._lane_corridor.overlaps_box(
            box, minimum_fraction=0.25
        )

    def _plausible_target(self, box: BBox, width: int, height: int) -> bool:
        """Reject impossible jumps and shape changes before they corrupt the ID."""
        if self._box is None:
            return True
        old_area = max(1.0, self._box.area)
        area_ratio = box.area / old_area
        minimum_ratio, maximum_ratio = (
            (0.12, 6.0) if self._phase in {TrackingPhase.DIVE, TrackingPhase.FINISH} else (0.22, 4.0)
        )
        if not minimum_ratio <= area_ratio <= maximum_ratio:
            return False

        frame_diagonal = math.hypot(width, height)
        box_diagonal = math.hypot(self._box.width, self._box.height)
        phase_fraction = {
            TrackingPhase.START: 0.24,
            TrackingPhase.DIVE: 0.22,
            TrackingPhase.SWIMMING: 0.14,
            TrackingPhase.FINISH: 0.10,
        }[self._phase]
        maximum_distance = max(box_diagonal * 4.0, frame_diagonal * phase_fraction)
        if math.dist(self._box.center, box.center) > maximum_distance:
            return False

        if self._travel_direction and self._phase is TrackingPhase.SWIMMING:
            backwards = (box.center[0] - self._box.center[0]) * self._travel_direction
            if backwards < -max(self._box.width, width * 0.025):
                return False
        return True

    def _record_water_observation(self, box: BBox, width: int, height: int) -> None:
        self._water_observations += 1
        self._centres.append(box.center)
        self._areas.append(box.area)
        if self._water_observations >= 10 and self._phase is TrackingPhase.DIVE:
            self._phase = TrackingPhase.SWIMMING

        if len(self._centres) >= 6:
            displacement = self._centres[-1][0] - self._centres[0][0]
            if abs(displacement) >= width * 0.025:
                self._travel_direction = 1 if displacement > 0 else -1

        if self._phase is not TrackingPhase.SWIMMING or not self._travel_direction:
            return
        x, _ = box.center
        approaching_edge = (
            self._travel_direction > 0 and x >= width * 0.91
        ) or (
            self._travel_direction < 0 and x <= width * 0.09
        )
        if approaching_edge:
            self._phase = TrackingPhase.FINISH

    @staticmethod
    def _choose_initial_id(rows: np.ndarray, selection: BBox) -> int:
        def rank(row: np.ndarray) -> tuple[float, float]:
            box = BBox(*[float(value) for value in row[:4]])
            return box.iou(selection), -math.dist(box.center, selection.center)

        return int(round(max(rows, key=rank)[4]))

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
