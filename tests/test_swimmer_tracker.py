from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from core.geometry import BBox
from core.hybrid_detector import HybridDetection
from core.swimmer_tracker import LaneConstraint, PersistentSwimmerTracker


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 0.8
    source: str = "swimmer"


FRAME_SHAPE = (600, 1000)


def test_selects_candidate_near_user_target_not_highest_confidence() -> None:
    tracker = PersistentSwimmerTracker()
    target = Detection(100, 200, 160, 260, 0.55, "person")
    spectator = Detection(700, 50, 780, 180, 0.99, "person")

    result = tracker.start([spectator, target], FRAME_SHAPE, target_point=(130, 230))

    assert result.detection is target
    assert result.box == BBox(100, 200, 160, 260)


def test_predicts_through_gap_and_reacquires_swimmer_source() -> None:
    tracker = PersistentSwimmerTracker(max_missed_frames=3)
    tracker.start_from_box(BBox(100, 200, 160, 250), FRAME_SHAPE)
    first = tracker.update([Detection(110, 200, 170, 250, source="person")], FRAME_SHAPE)
    second = tracker.update([Detection(120, 200, 180, 250, source="person")], FRAME_SHAPE)

    predicted = tracker.update([], FRAME_SHAPE)
    reacquired = tracker.update([Detection(140, 202, 205, 252, source="swimmer")], FRAME_SHAPE)

    assert first is not None and second is not None
    assert predicted is not None and predicted.predicted
    assert predicted.center[0] > second.center[0]
    assert reacquired is not None and not reacquired.predicted
    assert reacquired.source == "swimmer"
    assert tracker.missed_frames == 0


def test_rejects_impossible_high_confidence_jump() -> None:
    tracker = PersistentSwimmerTracker(max_jump_fraction=0.05, max_missed_frames=2)
    tracker.start_from_box(BBox(100, 200, 160, 250), FRAME_SHAPE)

    result = tracker.update([Detection(800, 200, 860, 250, 1.0, "both")], FRAME_SHAPE)

    assert result is not None
    assert result.predicted
    assert result.box.center[0] == pytest.approx(130.0)


def test_lane_constraint_rejects_adjacent_lane() -> None:
    tracker = PersistentSwimmerTracker(lane=LaneConstraint(180, 280, "y"), max_missed_frames=2)
    tracker.start_from_box(BBox(100, 200, 160, 250), FRAME_SHAPE)
    adjacent = Detection(110, 330, 170, 380, 1.0, "both")

    result = tracker.update([adjacent], FRAME_SHAPE)

    assert result is not None and result.predicted


def test_source_score_breaks_an_otherwise_equal_tie() -> None:
    tracker = PersistentSwimmerTracker()
    tracker.start_from_box(BBox(100, 200, 160, 250), FRAME_SHAPE)
    person = Detection(110, 200, 170, 250, 0.8, "person")
    both = Detection(110, 200, 170, 250, 0.8, "both")

    result = tracker.update([person, both], FRAME_SHAPE)

    assert result is not None
    assert result.detection is both


def test_detector_integration_uses_public_detect_contract() -> None:
    class FakeHybridDetector:
        def detect(self, frame):
            return [Detection(110, 200, 170, 250, source="both")]

    tracker = PersistentSwimmerTracker(FakeHybridDetector())
    tracker.start_from_box(BBox(100, 200, 160, 250), FRAME_SHAPE)

    result = tracker.track(np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8))

    assert result is not None
    assert result.source == "both"


def test_accepts_real_hybrid_detection_type() -> None:
    tracker = PersistentSwimmerTracker()
    tracker.start_from_box(BBox(100, 200, 160, 250), FRAME_SHAPE)
    candidate = HybridDetection(110, 200, 170, 250, 0.8, "both", 0.7, 0.8)

    result = tracker.update([candidate], FRAME_SHAPE)

    assert result is not None
    assert result.detection is candidate


def test_stops_displaying_stale_prediction_but_can_reacquire() -> None:
    tracker = PersistentSwimmerTracker(max_missed_frames=1, max_reacquire_frames=5)
    tracker.start_from_box(BBox(100, 200, 160, 250), FRAME_SHAPE)

    assert tracker.update([], FRAME_SHAPE) is not None
    assert tracker.update([], FRAME_SHAPE) is None
    candidate = Detection(105, 200, 165, 250)
    pending = tracker.update([candidate], FRAME_SHAPE)
    reacquired = tracker.update([candidate], FRAME_SHAPE)

    assert pending is None
    assert reacquired is not None
    assert not reacquired.predicted
    assert tracker.missed_frames == 0


def test_requires_explicit_reselection_after_identity_memory_expires() -> None:
    tracker = PersistentSwimmerTracker(max_missed_frames=1, max_reacquire_frames=2)
    tracker.start_from_box(BBox(100, 200, 160, 250), FRAME_SHAPE)

    tracker.update([], FRAME_SHAPE)
    tracker.update([], FRAME_SHAPE)
    result = tracker.update([Detection(105, 200, 165, 250, 1.0, "both")], FRAME_SHAPE)

    assert result is None
    assert tracker.initialized


def _coloured_frame(boxes: list[tuple[BBox, tuple[int, int, int]]]) -> np.ndarray:
    frame = np.full((*FRAME_SHAPE, 3), 35, dtype=np.uint8)
    for box, colour in boxes:
        x1, y1, x2, y2 = box.to_int_list()
        frame[y1:y2, x1:x2] = colour
    return frame


def test_selected_appearance_beats_a_closer_different_swimmer() -> None:
    initial = BBox(100, 200, 160, 250)
    tracker = PersistentSwimmerTracker()
    tracker.start_from_box(initial, FRAME_SHAPE, frame=_coloured_frame([(initial, (0, 0, 240))]))

    wrong = Detection(105, 200, 155, 250, 0.99, "both")
    target = Detection(165, 200, 225, 250, 0.70, "swimmer")
    frame = _coloured_frame([
        (BBox(wrong.x1, wrong.y1, wrong.x2, wrong.y2), (240, 0, 0)),
        (BBox(target.x1, target.y1, target.x2, target.y2), (0, 0, 240)),
    ])

    result = tracker.update([wrong, target], FRAME_SHAPE, frame=frame)

    assert result is not None and not result.predicted
    assert result.detection is target


def test_inferred_lane_rejects_an_adjacent_lane_without_user_input() -> None:
    tracker = PersistentSwimmerTracker()
    tracker.start_from_box(BBox(100, 200, 160, 250), FRAME_SHAPE)

    result = tracker.update([Detection(110, 275, 170, 325, 1.0, "both")], FRAME_SHAPE)

    assert result is not None and result.predicted


def test_reacquisition_after_a_longer_gap_requires_two_consistent_frames() -> None:
    tracker = PersistentSwimmerTracker(max_missed_frames=6, reacquire_confirmations=2)
    tracker.start_from_box(BBox(100, 200, 160, 250), FRAME_SHAPE)
    tracker.update([], FRAME_SHAPE)
    tracker.update([], FRAME_SHAPE)
    candidate = Detection(110, 200, 170, 250)

    first = tracker.update([candidate], FRAME_SHAPE)
    second = tracker.update([candidate], FRAME_SHAPE)

    assert first is not None and first.predicted
    assert second is not None and not second.predicted


def test_gap_prediction_slows_instead_of_running_past_finish_wall() -> None:
    tracker = PersistentSwimmerTracker(max_missed_frames=10)
    tracker.start_from_box(BBox(100, 200, 160, 250), FRAME_SHAPE)
    tracker.update([Detection(110, 200, 170, 250)], FRAME_SHAPE)
    last_observed = tracker.update([Detection(120, 200, 180, 250)], FRAME_SHAPE)
    assert last_observed is not None

    predicted = None
    for _ in range(8):
        predicted = tracker.update([], FRAME_SHAPE)

    assert predicted is not None and predicted.predicted
    assert predicted.center[0] - last_observed.center[0] < 55
