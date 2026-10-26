"""Persistent tracking for one user-selected swimmer.

The app-facing tracker follows the pixels inside the user's box with OpenCV's
single-object ViTTrack model before handing off to the existing swimmer
heatmap in the pool. The older BoT-SORT implementation remains available for
compatibility and detector experiments, but is not used by the Streamlit app.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Protocol, Sequence

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
        max_missed_frames: int = 180,
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
        self._lane_axis: tuple[float, float] | None = None
        self._lane_spacing: float | None = None
        self._lane_samples = 0

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
        self._lane_axis = None
        self._lane_spacing = None
        self._lane_samples = 0

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
        self._water_locked = source.lower() in {"swimmer", "both"}
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
        # A box drawn on an already-swimming target must begin in water mode.
        # Previously the first unconfirmed BoT-SORT row fell back to the
        # synthetic selection source, causing every later pure swimmer
        # observation to be rejected as if the athlete were still on deck.
        represented = [
            (detection, box)
            for detection in detections
            if (box := self._box_for(detection, width, height)) is not None
            and (
                self._contains(box, self._selection.center)
                or box.iou(self._selection) >= 0.05
            )
        ]
        if represented:
            selected_detection, _ = max(
                represented,
                key=lambda item: (
                    int(self._contains(item[1], self._selection.center)),
                    item[1].iou(self._selection),
                    -math.dist(item[1].center, self._selection.center),
                ),
            )
            self._last_source = str(selected_detection.source).lower()
            self._water_locked = self._last_source in {"swimmer", "both"}
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
            # Before the dive is semantically verified, pure heatmap regions
            # are inspected only by the guarded handoff below; they are
            # removed before BoT-SORT if that handoff rejects them.
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

        # Preserve the complete water candidate set for learning lane
        # direction even though only the best identity-continuous candidate
        # is allowed into the single selected track below.
        lane_detections = list(valid)
        lane_boxes = list(boxes)

        # BoT-SORT is a multi-object tracker, but this class has a stricter
        # contract: preserve the one identity selected by the user.  Keep a
        # nearby swimmer candidate set while the target is continuously
        # visible so an unrelated heatmap region (lane flag, reflection, or
        # adjacent swimmer) cannot inherit the selected ID through a loose
        # IoU association.  A longer gap uses the guarded relink path below.
        if self._water_locked and self._missed_frames < 5 and self._observed_box is not None:
            anchor = self._observed_box
            anchor_scale = math.hypot(anchor.width, anchor.height)
            continuity_limit = max(
                45.0,
                anchor_scale * 0.60 + (self._missed_frames + 1) * 12.0,
            )
            continuity_limit = min(continuity_limit, math.hypot(width, height) * 0.16)
            keep = [
                index
                for index, box in enumerate(boxes)
                if math.dist(anchor.center, box.center) <= continuity_limit
            ]
            if keep:
                best = min(
                    keep,
                    key=lambda index: (
                        math.dist(anchor.center, boxes[index].center),
                        self._box_gap(anchor, boxes[index]),
                        -self._confidence(valid[index]),
                    ),
                )
                valid = [valid[best]]
                boxes = [boxes[best]]
            else:
                valid = []
                boxes = []

        # A dive changes the observation from an upright YOLO person box to a
        # horizontal U-Net swimmer blob. Those boxes need not overlap, so
        # BoT-SORT cannot always associate them by IoU alone. After a sustained
        # person gap, nominate only the closest plausible swimmer blob for
        # a source handoff and give it enough confidence to establish a track.
        # The same gated path can relink a lost water track, but only to a
        # nearby swimmer heatmap observation—not to a pool-deck person.
        handoff_index = self._water_relink_candidate(valid, boxes, width, height)
        if not self._water_locked and handoff_index is None and not force_selection:
            # Pure heatmap regions may be inspected by the guarded handoff,
            # but never reach BoT-SORT directly while the selected athlete is
            # still a person. This is what prevents a flag or reflection from
            # silently inheriting the ID.
            keep = [
                index
                for index, detection in enumerate(valid)
                if str(detection.source).lower() != "swimmer"
            ]
            valid = [valid[index] for index in keep]
            boxes = [boxes[index] for index in keep]
        if self._water_locked and self._missed_frames >= 5 and handoff_index is None:
            # After a real gap, every water observation must pass the guarded
            # lane/continuity relink above. Feeding rejected detections to
            # BoT-SORT would let its broader association bypass that guard.
            valid = []
            boxes = []

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
                context_index = next(
                    (
                        candidate_index
                        for candidate_index, candidate in enumerate(lane_detections)
                        if candidate is detection
                    ),
                    -1,
                )
                self._update_lane_model(
                    lane_detections,
                    lane_boxes,
                    context_index,
                    width,
                    height,
                )
            confidence = float(target_row[5])
            return TrackedSwimmer(box, confidence, source, confidence, False, 0, detection)

        return self._missing_target(width, height)

    def _missing_target(self, width: int, height: int) -> TrackedSwimmer | None:
        self._missed_frames += 1
        if self._missed_frames > self.max_missed_frames:
            return None
        # Kalman extrapolation is useful for a short detector flicker, but it
        # becomes misleading through a long splash or camera zoom.  After 12
        # frames retain the last verified observation instead of drawing a
        # fast-moving box over a flag, spectator, or neighbouring lane.
        if self._missed_frames <= 12:
            predicted_box = self._lost_track_box(width, height) or self._observed_box or self._box
        else:
            predicted_box = self._observed_box or self._box
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
        # Do not declare a dive during a brief YOLO flicker on the blocks.
        # Real entry gaps in the uploaded races last well beyond eight frames.
        minimum_misses = 5 if self._water_locked else 8
        # Re-identification must be anchored to the last *observed* swimmer,
        # never to a long Kalman-only prediction.  On Video(2), camera motion
        # pushed the predicted box into the next lane while the selected
        # swimmer was underwater; using that drifting box then made the wrong
        # lane look closest.  The expanding jump gate below still allows real
        # camera movement without turning a prediction into evidence.
        anchor = self._observed_box or self._box
        if (
            not self._advanced
            or self._missed_frames < minimum_misses
            or self._missed_frames > self.max_reacquire_frames
            or anchor is None
        ):
            return None

        # Do not hand off while YOLO still sees a person near the selected
        # track.  Requiring literal overlap was too weak on real race video:
        # the upright box can move tens of pixels between frames while a
        # stationary banner or starting-block reflection is already producing
        # a U-Net heatmap.  That false heatmap used to steal the identity
        # before the swimmer entered the water.
        if not self._water_locked:
            for detection, box in zip(detections, boxes):
                if str(detection.source).lower() != "person":
                    continue
                if box.iou(anchor) >= 0.10 or self._contains(box, anchor.center):
                    return None

        # Prefer independently verified YOLO+heatmap regions at entry, while
        # still permitting a pure swimmer heatmap after the selected person
        # has disappeared. The person-presence guard and jump gate above keep
        # flags/reflections from becoming an early handoff.
        allowed_sources = {"swimmer"} if self._water_locked else {"both", "swimmer"}
        swimmers = [
            index
            for index, detection in enumerate(detections)
            if str(detection.source).lower() in allowed_sources
        ]
        if not self._water_locked:
            # A pure heatmap can bridge YOLO's dive dropout, but only on the
            # water side of the last upright person. This rejects lane flags
            # and banners above the blocks. A `both` observation is exempt
            # because YOLO independently confirms that region as a person.
            swimmers = [
                index
                for index in swimmers
                if str(detections[index].source).lower() == "both"
                or boxes[index].center[1] >= anchor.center[1] + height * 0.05
            ]
        if self._water_locked:
            # The uploaded race has a yellow caution tape along the bottom
            # edge that the heatmap occasionally labels as a swimmer. A real
            # trajectory that was previously clear of the edge cannot become
            # a border-clipped blob during a short detection gap.
            if anchor.y2 < height * 0.95:
                swimmers = [index for index in swimmers if boxes[index].y2 < height - 1]
            if self._lane_samples >= 30 and self._lane_axis is not None and self._lane_spacing is not None:
                ax, ay = self._lane_axis
                lane_tolerance = self._lane_spacing * 0.55 + min(
                    self._missed_frames * 1.0,
                    self._lane_spacing * 0.25,
                )
                swimmers = [
                    index
                    for index in swimmers
                    if abs(
                        (boxes[index].center[0] - anchor.center[0]) * ax
                        + (boxes[index].center[1] - anchor.center[1]) * ay
                    ) <= lane_tolerance
                ]
        if not swimmers:
            return None

        index = min(
            swimmers,
            key=lambda candidate: (
                math.dist(anchor.center, boxes[candidate].center),
                self._box_gap(anchor, boxes[candidate]),
            ),
        )
        # Centre continuity is orientation-independent: it works both for the
        # side-on races where lanes separate vertically and for head-on races
        # where lanes separate horizontally.  Ranking by box-edge distance
        # allowed one very wide splash in the next lane to win even though its
        # centre made an impossible jump.  The gate expands gradually through
        # an occlusion but remains tied to the selected swimmer's scale.
        if self._water_locked:
            maximum_jump = max(
                30.0,
                math.hypot(anchor.width, anchor.height) * 0.45 + self._missed_frames * 3.0,
            )
            maximum_jump = min(maximum_jump, math.hypot(width, height) * 0.18)
        else:
            # The initial upright-person to horizontal-swimmer transition is
            # a much larger geometry change than a later in-water gap. Pure
            # heatmaps still reach this point only through the person guard.
            frame_diagonal = math.hypot(width, height)
            if self._missed_frames >= 8:
                # A camera pan during a dive can move the reappearing swimmer
                # much farther than the final upright-person box suggests.
                maximum_jump = frame_diagonal * 0.32
            else:
                maximum_jump = max(
                    40.0,
                    math.hypot(anchor.width, anchor.height) + self._missed_frames * 6.0,
                )
                maximum_jump = min(maximum_jump, frame_diagonal * 0.30)
        return index if math.dist(anchor.center, boxes[index].center) <= maximum_jump else None

    def _update_lane_model(
        self,
        detections: Sequence[DetectionLike],
        boxes: Sequence[BBox],
        target_index: int,
        width: int,
        height: int,
    ) -> None:
        """Learn the image direction and spacing between adjacent lanes.

        Pool views vary: lanes are separated vertically in side-on footage
        and horizontally in head-on footage.  A neighbouring swimmer observed
        alongside the selected one gives that direction without asking the
        user for lane coordinates.  The slowly updated model is used only for
        relinking after a gap; continuous BoT-SORT association is unchanged.
        """
        if not 0 <= target_index < len(boxes):
            return
        target = boxes[target_index]
        neighbours: list[tuple[float, float, float]] = []
        minimum = max(20.0, math.hypot(target.width, target.height) * 0.55)
        maximum = math.hypot(width, height) * 0.50
        tx, ty = target.center
        for index, (detection, box) in enumerate(zip(detections, boxes)):
            if index == target_index or str(detection.source).lower() not in {"swimmer", "both"}:
                continue
            dx, dy = box.center[0] - tx, box.center[1] - ty
            distance = math.hypot(dx, dy)
            if minimum <= distance <= maximum:
                neighbours.append((distance, dx, dy))
        if not neighbours:
            return
        spacing, dx, dy = min(neighbours)
        ux, uy = dx / spacing, dy / spacing
        if self._lane_axis is not None:
            old_x, old_y = self._lane_axis
            if ux * old_x + uy * old_y < 0:
                ux, uy = -ux, -uy
            ux, uy = old_x * 0.85 + ux * 0.15, old_y * 0.85 + uy * 0.15
            length = math.hypot(ux, uy)
            ux, uy = ux / length, uy / length
        self._lane_axis = (ux, uy)
        self._lane_spacing = (
            spacing if self._lane_spacing is None else self._lane_spacing * 0.85 + spacing * 0.15
        )
        self._lane_samples += 1

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


class _VitVisualTracker:
    """Thin wrapper around OpenCV's official one-object ViTTrack API."""

    def __init__(self, model_path: Path, frame: np.ndarray, box: BBox) -> None:
        import cv2

        params = cv2.TrackerVit_Params()
        params.net = str(model_path)
        params.backend = cv2.dnn.DNN_BACKEND_OPENCV
        params.target = cv2.dnn.DNN_TARGET_CPU
        self._tracker = cv2.TrackerVit_create(params)
        self._tracker.init(
            frame,
            (round(box.x1), round(box.y1), round(box.width), round(box.height)),
        )

    def update(self, frame: np.ndarray) -> tuple[BBox | None, float]:
        located, raw = self._tracker.update(frame)
        score = float(self._tracker.getTrackingScore())
        if not located:
            return None, score
        x, y, width, height = (float(value) for value in raw)
        return BBox(x, y, x + width, y + height), score


VisualFactory = Callable[[np.ndarray, BBox], Any]


class PersistentVisualSwimmerTracker:
    """Follow one selected swimmer with visual identity plus detector recovery.

    This deliberately uses a single-object tracker before the dive. A
    multi-object tracker can assign the selected ID to a nearby official or
    lane flag when YOLO briefly loses the partially occluded swimmer. ViTTrack
    instead follows the pixels inside the user's one box. After water entry,
    swimmer-heatmap detections correct the visual track and the visual track
    bridges detector gaps. Only a short, explicitly marked motion estimate is
    returned when neither model has a trustworthy observation.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_estimated_frames: int = 90,
        visual_threshold: float = 0.28,
        visual_factory: VisualFactory | None = None,
    ) -> None:
        if max_estimated_frames < 0:
            raise ValueError("max_estimated_frames must be non-negative")
        self.model_path = Path(model_path)
        self.max_estimated_frames = max_estimated_frames
        self.visual_threshold = visual_threshold
        self._visual_factory = visual_factory or (
            lambda frame, box: _VitVisualTracker(self.model_path, frame, box)
        )
        self.reset()

    @property
    def initialized(self) -> bool:
        return self._selection is not None

    @property
    def missed_frames(self) -> int:
        return self._missed_frames

    @property
    def water_locked(self) -> bool:
        return self._state == "water"

    @property
    def target_id(self) -> int | None:
        """Compatibility property: one selected object always has ID 1."""
        return 1 if self.initialized else None

    def reset(self) -> None:
        self._selection: BBox | None = None
        self._box: BBox | None = None
        self._observed_box: BBox | None = None
        self._visual_box: BBox | None = None
        # Only the selection and detector observations may change this size
        # anchor. Comparing ViTTrack only with its own previous output lets a
        # box inflate a little every frame until it covers the whole video.
        self._reference_box: BBox | None = None
        # The box ViTTrack was last started from; its output keeps that scale.
        self._visual_seed: BBox | None = None
        self._visual: Any | None = None
        self._state = "person"
        self._person_missed = 0
        self._entry_seen = False
        self._missed_frames = 0
        self._last_source = "selection"
        self._velocity = (0.0, 0.0)

    def start(
        self,
        detections: Sequence[DetectionLike],
        frame_shape: Sequence[int],
        *,
        target_point: tuple[float, float] | None = None,
        target_box: BBox | Sequence[float] | None = None,
        frame: Any | None = None,
    ) -> TrackedSwimmer:
        height, width = PersistentSwimmerTracker._normalise_frame_shape(frame_shape)
        if target_box is not None:
            selected = target_box if isinstance(target_box, BBox) else BBox.from_list(list(target_box))
        elif target_point is not None:
            boxes = [self._box_for(detection, width, height) for detection in detections]
            boxes = [box for box in boxes if box is not None]
            if not boxes:
                raise ValueError("cannot initialize the tracker without a valid detection")
            selected = min(boxes, key=lambda box: math.dist(box.center, target_point))
        else:
            boxes = [(detection, self._box_for(detection, width, height)) for detection in detections]
            boxes = [(detection, box) for detection, box in boxes if box is not None]
            if not boxes:
                raise ValueError("cannot initialize the tracker without a valid detection")
            _, selected = max(boxes, key=lambda item: float(item[0].confidence))

        selected = selected.clamp(width, height)
        if selected.width <= 0 or selected.height <= 0:
            raise ValueError("the initial swimmer box must have positive width and height")
        self._selection = selected
        self._box = selected
        self._observed_box = selected
        self._visual_box = selected
        self._reference_box = selected
        if isinstance(frame, np.ndarray):
            self._reset_visual(frame, selected)
        return TrackedSwimmer(selected, 1.0, "selection", 1.0, False, 0)

    def start_from_box(
        self,
        box: BBox | Sequence[float],
        frame_shape: Sequence[int],
        *,
        source: str = "selection",
        frame: Any | None = None,
    ) -> TrackedSwimmer:
        result = self.start([], frame_shape, target_box=box, frame=frame)
        return TrackedSwimmer(result.box, result.confidence, source, result.score, False, 0)

    def update(
        self,
        detections: Sequence[DetectionLike],
        frame_shape: Sequence[int],
        *,
        frame: Any | None = None,
    ) -> TrackedSwimmer | None:
        height, width = PersistentSwimmerTracker._normalise_frame_shape(frame_shape)
        if self._selection is None:
            if not detections:
                return None
            return self.start(detections, (height, width), frame=frame)
        if not isinstance(frame, np.ndarray):
            return self._estimate(width, height)

        visual_box, visual_score = self._visual_update(frame, width, height)
        candidates = [
            (detection, box)
            for detection in detections
            if (box := self._box_for(detection, width, height)) is not None
        ]
        persons = [item for item in candidates if str(item[0].source).lower() in {"person", "both"}]
        swimmers = [item for item in candidates if str(item[0].source).lower() in {"swimmer", "both"}]

        if self._state == "person":
            return self._update_person(frame, persons, swimmers, visual_box, visual_score, width, height)
        return self._update_water(frame, swimmers, visual_box, visual_score, width, height)

    def _update_person(
        self,
        frame: np.ndarray,
        persons: list[tuple[DetectionLike, BBox]],
        swimmers: list[tuple[DetectionLike, BBox]],
        visual_box: BBox | None,
        visual_score: float,
        width: int,
        height: int,
    ) -> TrackedSwimmer | None:
        assert self._visual_box is not None
        valid_visual = self._visual_is_sane(visual_box, visual_score, self._visual_box, width, height)
        if valid_visual:
            assert visual_box is not None
            self._visual_box = visual_box
            confirming = [box for _, box in persons if box.iou(visual_box) >= 0.5]
            if confirming:
                self._reference_box = max(confirming, key=visual_box.iou)

        person_gate = max(55.0, math.hypot(self._visual_box.width, self._visual_box.height) * 0.38)
        near_person = any(
            box.iou(self._visual_box) >= 0.04
            or math.dist(box.center, self._visual_box.center) <= person_gate
            for _, box in persons
        )
        self._person_missed = 0 if near_person else self._person_missed + 1
        self._entry_seen = self._entry_seen or any(
            box.y2 >= height * 0.70
            and box.area >= self._visual_box.area * 0.25
            and (
                box.iou(self._visual_box) >= 0.02
                or math.dist(box.center, self._visual_box.center)
                <= math.hypot(self._visual_box.width, self._visual_box.height) * 0.75
            )
            for _, box in swimmers
        )

        if self._person_missed >= 5 and self._entry_seen:
            gate = max(110.0, math.hypot(self._visual_box.width, self._visual_box.height) * 1.20)
            eligible = [
                (detection, box)
                for detection, box in swimmers
                if math.dist(box.center, self._visual_box.center) <= gate
                and box.center[1] >= max(height * 0.52, self._visual_box.center[1] - 20.0)
            ]
            if eligible:
                detection, box = min(
                    eligible,
                    key=lambda item: math.dist(item[1].center, self._visual_box.center),
                )
                self._state = "water"
                self._reference_box = box
                self._reset_visual(frame, box)
                return self._accept(box, self._confidence(detection), "swimmer", detection)

        if valid_visual:
            assert visual_box is not None
            return self._accept(visual_box, visual_score, "visual-person")

        # A tight YOLO correction can recover a one-frame visual dropout, but
        # the gate is intentionally too small to jump to the next block.
        assert self._box is not None
        correction_gate = max(35.0, math.hypot(self._box.width, self._box.height) * 0.30)
        corrections = [
            (detection, box)
            for detection, box in persons
            if math.dist(box.center, self._box.center) <= correction_gate
        ]
        if corrections:
            detection, box = min(corrections, key=lambda item: math.dist(item[1].center, self._box.center))
            self._visual_box = box
            self._reset_visual(frame, box)
            return self._accept(box, self._confidence(detection), "person", detection)
        return self._estimate(width, height)

    def _update_water(
        self,
        frame: np.ndarray,
        swimmers: list[tuple[DetectionLike, BBox]],
        visual_box: BBox | None,
        visual_score: float,
        width: int,
        height: int,
    ) -> TrackedSwimmer | None:
        assert self._box is not None
        gate = max(
            55.0,
            math.hypot(self._box.width, self._box.height) * 0.70 + self._missed_frames * 2.0,
        )
        gate = min(gate, math.hypot(width, height) * 0.20)
        visual_gate = gate
        if self._missed_frames > 0:
            # ViTTrack can lock onto a static look-alike (a lane sign or splash)
            # after a gap. Without a detector confirmation the visual box may
            # only move as far as the swimmer plausibly could have.
            speed = math.hypot(*self._velocity)
            diagonal = math.hypot(self._box.width, self._box.height)
            visual_gate = min(gate, max(30.0, diagonal * 0.8 + speed * self._missed_frames * 1.5))
        valid_visual = (
            self._visual_is_sane(visual_box, visual_score, self._box, width, height, gate=visual_gate)
            and visual_box is not None
            and self._consistent_size(visual_box)
        )
        eligible = [
            (detection, box)
            for detection, box in swimmers
            if math.dist(box.center, self._box.center) <= gate and self._consistent_size(box)
        ]
        if eligible:
            def rank(item: tuple[DetectionLike, BBox]) -> float:
                detection, box = item
                from_last = math.dist(box.center, self._box.center) / gate
                from_visual = (
                    math.dist(box.center, visual_box.center) / gate
                    if valid_visual and visual_box is not None else 1.0
                )
                return 0.55 * from_last + 0.30 * from_visual - 0.15 * self._confidence(detection)

            detection, box = min(eligible, key=rank)
            correction = (
                math.dist(box.center, visual_box.center)
                if valid_visual and visual_box is not None else math.inf
            )
            if not valid_visual or correction > max(25.0, math.hypot(box.width, box.height) * 0.45):
                self._reset_visual(frame, box)
            return self._accept(box, self._confidence(detection), "swimmer", detection)
        if valid_visual:
            assert visual_box is not None
            return self._accept(visual_box, visual_score, "visual-swimmer")
        return self._estimate(width, height)

    def _accept(
        self,
        box: BBox,
        confidence: float,
        source: str,
        detection: DetectionLike | None = None,
    ) -> TrackedSwimmer:
        if self._observed_box is not None:
            steps = self._missed_frames + 1
            old_x, old_y = self._observed_box.center
            new_x, new_y = box.center
            measured = ((new_x - old_x) / steps, (new_y - old_y) / steps)
            self._velocity = (
                self._velocity[0] * 0.65 + measured[0] * 0.35,
                self._velocity[1] * 0.65 + measured[1] * 0.35,
            )
        self._box = box
        self._observed_box = box
        if detection is not None:
            self._update_reference(box)
        self._missed_frames = 0
        self._last_source = source
        return TrackedSwimmer(box, confidence, source, confidence, False, 0, detection)

    def _estimate(self, width: int, height: int) -> TrackedSwimmer | None:
        assert self._box is not None
        self._missed_frames += 1
        if self._missed_frames > self.max_estimated_frames:
            return None
        move_x, move_y = self._velocity
        # Never turn an estimate into the next identity anchor.  Doing that
        # compounds small velocity errors and eventually pulls reacquisition
        # toward another lane.  All matching remains anchored to ``_box``, the
        # last verified visual/detector observation; this returned box is only
        # an explicitly labelled display estimate.
        steps = min(self._missed_frames, 12)
        decay = 0.82 ** max(0, steps - 1)
        predicted = BBox(
            self._box.x1 + move_x * steps * decay,
            self._box.y1 + move_y * steps * decay,
            self._box.x2 + move_x * steps * decay,
            self._box.y2 + move_y * steps * decay,
        ).clamp(width, height)
        confidence = max(0.0, 1.0 - self._missed_frames / (self.max_estimated_frames + 1))
        return TrackedSwimmer(
            predicted,
            confidence,
            f"estimated:{self._last_source}",
            0.0,
            True,
            self._missed_frames,
        )

    def _visual_update(self, frame: np.ndarray, width: int, height: int) -> tuple[BBox | None, float]:
        if self._visual is None:
            return None, 0.0
        try:
            box, score = self._visual.update(frame)
        except Exception:
            return None, 0.0
        if box is None:
            return None, score
        box = box.clamp(width, height)
        if box.width <= 0 or box.height <= 0:
            return None, score
        if self._inflated(box, width, height):
            if self._box is not None:
                self._reset_visual(frame, self._box)
            return None, 0.0
        return box, score

    def _update_reference(self, box: BBox) -> None:
        reference = self._reference_box
        if reference is None:
            self._reference_box = box
            return
        alpha = 0.4
        ref_w = reference.width + alpha * (box.width - reference.width)
        ref_h = reference.height + alpha * (box.height - reference.height)
        cx, cy = box.center
        self._reference_box = BBox(cx - ref_w / 2, cy - ref_h / 2, cx + ref_w / 2, cy + ref_h / 2)

    def _consistent_size(self, box: BBox) -> bool:
        """Reject merged adjacent-lane blobs and fragments as identity anchors."""
        reference = self._reference_box
        if reference is None or reference.area <= 0:
            return True
        # Butterfly splash legitimately changes the heatmap box a lot, so only
        # a blob that suddenly spans another lane is refused.
        relax = 1.0 if self._missed_frames <= 15 else 1.5
        return (
            box.width <= reference.width * 2.2 * relax
            and box.height <= reference.height * 2.6 * relax
            and reference.area * 0.12 / relax <= box.area <= reference.area * 3.5 * relax
        )

    def _inflated(self, box: BBox, width: int, height: int) -> bool:
        # Area only: a dive legitimately turns a tall standing box into a long,
        # flat one, while a merged neighbour in the water is refused by
        # ``_consistent_size``. Growth is measured from the box ViTTrack was
        # started from, because YOLO's person box is much tighter than
        # ViTTrack's; the looser detector bound stops growth compounding
        # over repeated restarts.
        seed = self._visual_seed or self._box
        if seed is None:
            return False
        growth_limit = 2.6 if self._state == "person" else 3.5
        if box.area > seed.area * growth_limit or box.area > width * height * 0.30:
            return True
        reference = self._reference_box
        return reference is not None and box.area > reference.area * growth_limit * 2.5

    def _reset_visual(self, frame: np.ndarray, box: BBox) -> None:
        self._visual = self._visual_factory(frame, box)
        self._visual_box = box
        self._visual_seed = box

    def _visual_is_sane(
        self,
        box: BBox | None,
        score: float,
        anchor: BBox,
        width: int,
        height: int,
        *,
        gate: float | None = None,
    ) -> bool:
        if box is None or score < self.visual_threshold:
            return False
        ratio = box.area / max(anchor.area, 1.0)
        maximum_shift = gate or max(70.0, math.hypot(anchor.width, anchor.height) * 0.55)
        return (
            0.25 <= ratio <= 3.0
            and math.dist(box.center, anchor.center) <= maximum_shift
            and box.is_inside(width, height)
        )

    @staticmethod
    def _box_for(detection: DetectionLike, width: int, height: int) -> BBox | None:
        return PersistentSwimmerTracker._box_for(detection, width, height)

    @staticmethod
    def _confidence(detection: DetectionLike) -> float:
        return PersistentSwimmerTracker._confidence(detection)
