"""Lock one user-selected swimmer to a persistent BoT-SORT trajectory.

The hybrid detector still supplies YOLO observations before water entry and
U-Net heatmap observations in the water. BoT-SORT associates *all* swimmers
over time using a Kalman state, two-stage high/low-confidence matching, and
camera-motion compensation. The box drawn by the user chooses the initial
person ID. A source-aware handoff joins that person to the nearest plausible
swimmer heatmap track during the dive. Later relinking is restricted to nearby
swimmer heatmap observations, so pool-deck people cannot take the identity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Protocol, Sequence

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
        max_missed_frames: int = 60,
        max_reacquire_frames: int = 180,
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
        self._observed_box: BBox | None = None
        self._missed_frames = 0
        self._last_source = "selection"
        self._advanced = False
        self._water_locked = False

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
            # identities use the same 0.25 accepted by the YOLO detector.
            track_high_thresh=0.20,
            track_low_thresh=0.08,
            new_track_thresh=0.25,
            track_buffer=track_buffer,
            match_thresh=0.80,
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

    def reset(self) -> None:
        self._backend.reset()
        self._selection = None
        self._target_id = None
        self._box = None
        self._observed_box = None
        self._missed_frames = 0
        self._last_source = "selection"
        self._advanced = False
        self._water_locked = False

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
        self._observed_box = selected
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
        self._observed_box = self._selection
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
        valid: list[DetectionLike] = []
        boxes: list[BBox] = []
        for detection in detections:
            # Once the selected swimmer has a real heatmap observation, keep
            # the working in-water detector isolated. Pool-deck people from
            # YOLO are no longer eligible to take part in association.
            source = str(detection.source).lower()
            if self._water_locked and source == "person":
                continue
            # A `both` box is safe while it is continuously associated, but
            # after a real gap it may be a spectator overlapping a false
            # heatmap region. Require a pure swimmer observation to relink.
            if self._water_locked and self._missed_frames >= 5 and source == "both":
                continue
            box = self._box_for(detection, width, height)
            if box is not None:
                valid.append(detection)
                boxes.append(box)

        # A dive changes the observation from an upright YOLO person box to a
        # horizontal U-Net swimmer blob. Those boxes need not overlap, so
        # BoT-SORT cannot always associate them by IoU alone. After two real
        # person misses, nominate only the closest plausible swimmer blob for
        # a source handoff and give it enough confidence to establish a track.
        # The same gated path can relink a lost water track, but only to a
        # nearby swimmer heatmap observation—not to a pool-deck person.
        handoff_index = self._water_relink_candidate(valid, boxes, width, height)

        # The user's box is authoritative. On the first frame, boost the
        # overlapping YOLO observation to 1.0 so BoT-SORT must create that
        # identity even when YOLO reports the allowed minimum confidence of
        # 0.25. If YOLO has no matching person, seed the exact selected box.
        seed_index: int | None = None
        if force_selection:
            person_indices = [
                index
                for index, detection in enumerate(valid)
                if str(detection.source).lower() in {"person", "both"}
            ]
            candidate_indices = person_indices or list(range(len(boxes)))
            if candidate_indices:
                seed_index = max(
                    candidate_indices,
                    key=lambda index: (
                        int(self._contains(boxes[index], self._selection.center)),
                        boxes[index].iou(self._selection),
                        -math.dist(boxes[index].center, self._selection.center),
                    ),
                )
                seed_box = boxes[seed_index]
                represented = self._contains(seed_box, self._selection.center) or seed_box.iou(self._selection) >= 0.05
                if not represented:
                    seed_index = None
            if seed_index is None:
                synthetic = _SelectionDetection(*self._selection.to_list())
                valid.append(synthetic)
                boxes.append(self._selection)
                seed_index = len(boxes) - 1

        if handoff_index is not None:
            # A reset starts a new BoT-SORT sequence, so feed only the gated
            # swimmer observation on that frame. Keeping unrelated candidates
            # here makes the tracker's returned detection indices ambiguous
            # and can bind the selected identity to a neighbouring lane.
            valid = [valid[handoff_index]]
            boxes = [boxes[handoff_index]]
            handoff_index = 0

        rows = np.asarray(
            [
                [
                    box.x1,
                    box.y1,
                    box.x2,
                    box.y2,
                    1.0 if index in {seed_index, handoff_index} else self._confidence(detection),
                ]
                for index, (detection, box) in enumerate(zip(valid, boxes))
            ],
            dtype=np.float32,
        ).reshape(-1, 5)
        image = frame if isinstance(frame, np.ndarray) else np.zeros((height, width, 3), dtype=np.uint8)
        if handoff_index is not None:
            # Start the water trajectory on its first nominated observation.
            # BoT-SORT otherwise keeps a new mid-sequence ID unconfirmed for
            # one frame, which would create a false LOST result at the dive.
            self._backend.reset()
        tracked = self._backend.update(_DetectionBatch(rows), image)
        self._advanced = True

        if self._target_id is None and len(tracked):
            self._target_id = self._choose_initial_id(tracked, self._selection)

        if handoff_index is not None:
            target_row = next(
                (row for row in tracked if int(round(row[7])) == handoff_index),
                None,
            )
            if target_row is not None:
                self._target_id = int(round(target_row[4]))
        else:
            target_row = next((row for row in tracked if int(round(row[4])) == self._target_id), None)
        if target_row is not None:
            box = BBox(*[float(value) for value in target_row[:4]]).clamp(width, height)
            index = int(round(target_row[7]))
            detection = valid[index] if 0 <= index < len(valid) else None
            source = str(detection.source).lower() if detection is not None else self._last_source
            if self.lane is not None and self.lane.score(box.center) is None:
                return self._missing_target(width, height)
            self._box = box
            self._observed_box = box
            self._missed_frames = 0
            self._last_source = source
            if source in {"swimmer", "both"}:
                self._water_locked = True
            confidence = float(target_row[5])
            return TrackedSwimmer(box, confidence, source, confidence, False, 0, detection)

        return self._missing_target(width, height)

    def _missing_target(self, width: int, height: int) -> TrackedSwimmer | None:
        self._missed_frames += 1
        if self._target_id is None or self._missed_frames > self.max_missed_frames:
            return None
        predicted_box = self._lost_track_box(width, height) or self._box
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

    def _water_relink_candidate(
        self,
        detections: Sequence[DetectionLike],
        boxes: Sequence[BBox],
        width: int,
        height: int,
    ) -> int | None:
        """Return one nearby U-Net candidate for handoff or water relinking."""
        minimum_misses = 5 if self._water_locked else 2
        # While swimming, BoT-SORT's lost-track box includes global camera
        # motion. Use it during a pan instead of freezing relinking around the
        # last raw observation. Before the dive, the raw person observation is
        # the safer anchor.
        anchor = self._box if self._water_locked else (self._observed_box or self._box)
        if not self._advanced or self._missed_frames < minimum_misses or anchor is None:
            return None

        # Do not hand off while YOLO still sees the selected person. This
        # prevents incidental pool heatmaps from stealing the target on the
        # blocks.
        if not self._water_locked:
            for detection, box in zip(detections, boxes):
                if str(detection.source).lower() not in {"person", "both"}:
                    continue
                if box.iou(anchor) >= 0.10 or self._contains(box, anchor.center):
                    return None

        allowed_sources = {"swimmer"} if self._water_locked else {"swimmer", "both"}
        swimmers = [
            index
            for index, detection in enumerate(detections)
            if str(detection.source).lower() in allowed_sources
        ]
        if self._water_locked:
            # In this fixed pool view, adjacent lanes are separated mostly on
            # the image Y axis. During a short gap the selected swimmer cannot
            # move a whole lane vertically, even if a neighbouring blob is
            # closer diagonally to the last box edge.
            maximum_lane_shift = max(anchor.height * 0.9, height * 0.10)
            swimmers = [
                index
                for index in swimmers
                if abs(boxes[index].center[1] - anchor.center[1]) <= maximum_lane_shift
            ]
            # The uploaded race has a yellow caution tape along the bottom
            # edge that the heatmap occasionally labels as a swimmer. A real
            # trajectory that was previously clear of the edge cannot become
            # a border-clipped blob during a short detection gap.
            if anchor.y2 < height * 0.95:
                swimmers = [index for index in swimmers if boxes[index].y2 < height - 1]
        if not swimmers:
            return None

        index = min(
            swimmers,
            key=lambda candidate: (
                self._box_gap(anchor, boxes[candidate]),
                math.dist(anchor.center, boxes[candidate].center),
            ),
        )
        maximum_gap = max(24.0, math.hypot(width, height) * 0.16)
        return index if self._box_gap(anchor, boxes[index]) <= maximum_gap else None

    @staticmethod
    def _box_gap(first: BBox, second: BBox) -> float:
        """Shortest edge-to-edge distance between two boxes (zero if overlapping)."""
        dx = max(first.x1 - second.x2, second.x1 - first.x2, 0.0)
        dy = max(first.y1 - second.y2, second.y1 - first.y2, 0.0)
        return math.hypot(dx, dy)

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
    def _contains(box: BBox, point: tuple[float, float]) -> bool:
        x, y = point
        return box.x1 <= x <= box.x2 and box.y1 <= y <= box.y2

    @staticmethod
    def _normalise_frame_shape(frame_shape: Sequence[int]) -> tuple[int, int]:
        if len(frame_shape) < 2:
            raise ValueError("frame_shape must contain height and width")
        height, width = int(frame_shape[0]), int(frame_shape[1])
        if height <= 0 or width <= 0:
            raise ValueError("frame height and width must be positive")
        return height, width
