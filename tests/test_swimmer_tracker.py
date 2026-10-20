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
    reacquired = tracker.update([Detection(105, 200, 165, 250)], FRAME_SHAPE)

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
