from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from core.geometry import BBox
from core.hybrid_detector import HybridDetection, HybridSwimmerDetector
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
BLANK_FRAME = np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8)


def test_selection_locks_the_matching_botsort_id() -> None:
    tracker = PersistentSwimmerTracker()
    target = Detection(100, 200, 160, 250, 0.55, "person")
    spectator = Detection(700, 50, 780, 180, 0.99, "person")

    result = tracker.start(
        [spectator, target], FRAME_SHAPE, target_box=BBox(95, 195, 165, 255), frame=BLANK_FRAME
    )

    assert result.detection is target
    assert tracker.target_id is not None


def test_minimum_confidence_selected_person_is_seeded_not_replaced() -> None:
    tracker = PersistentSwimmerTracker()
    target = Detection(100, 200, 160, 250, 0.25, "person")
    distractor = Detection(350, 200, 410, 250, 0.99, "person")

    result = tracker.start(
        [distractor, target],
        FRAME_SHAPE,
        target_box=BBox(95, 195, 165, 255),
        frame=BLANK_FRAME,
    )

    assert result.detection is target
    assert result.center[0] < 200
    assert tracker.target_id is not None


def test_never_substitutes_adjacent_lane_id_when_target_disappears() -> None:
    tracker = PersistentSwimmerTracker(max_missed_frames=3)
    target = Detection(100, 200, 160, 250, source="swimmer")
    adjacent = Detection(100, 300, 160, 350, source="swimmer")
    tracker.start([target, adjacent], FRAME_SHAPE, target_box=BBox(100, 200, 160, 250), frame=BLANK_FRAME)
    selected_id = tracker.target_id

    observed = tracker.update(
        [Detection(110, 200, 170, 250), Detection(110, 300, 170, 350)],
        FRAME_SHAPE,
        frame=BLANK_FRAME,
    )
    missing = tracker.update([Detection(120, 300, 180, 350, 1.0, "both")], FRAME_SHAPE, frame=BLANK_FRAME)

    assert observed is not None and not observed.predicted
    assert missing is not None and missing.predicted
    assert missing.center[1] < 280
    assert tracker.target_id == selected_id


def test_hands_selected_person_to_nearest_swimmer_once_after_dive_gap() -> None:
    tracker = PersistentSwimmerTracker(max_missed_frames=45)
    person = Detection(610, 120, 690, 300, 0.7, "person")
    tracker.start([person], FRAME_SHAPE, target_box=BBox(620, 140, 700, 310), frame=BLANK_FRAME)

    tracker.update([], FRAME_SHAPE, frame=BLANK_FRAME)
    tracker.update([], FRAME_SHAPE, frame=BLANK_FRAME)
    target = Detection(520, 320, 710, 420, 0.12, "swimmer")
    adjacent_lane = Detection(300, 260, 490, 350, 0.9, "swimmer")
    result = tracker.update([adjacent_lane, target], FRAME_SHAPE, frame=BLANK_FRAME)

    assert result is not None
    assert result.detection is target
    assert result.source == "swimmer"
    assert tracker.water_locked


def test_water_relink_rejects_a_detection_one_lane_away() -> None:
    tracker = PersistentSwimmerTracker(max_missed_frames=45)
    target = Detection(520, 330, 700, 420, 0.8, "swimmer")
    tracker.start([target], FRAME_SHAPE, target_box=BBox(520, 330, 700, 420), frame=BLANK_FRAME)
    for _ in range(6):
        result = tracker.update([], FRAME_SHAPE, frame=BLANK_FRAME)

    adjacent = Detection(500, 220, 690, 300, 0.9, "swimmer")
    result = tracker.update([adjacent], FRAME_SHAPE, frame=BLANK_FRAME)

    assert result is not None and result.predicted
    assert result.center[1] > 330


def test_water_relink_rejects_new_border_clipped_heatmap_blob() -> None:
    tracker = PersistentSwimmerTracker(max_missed_frames=45)
    target = Detection(520, 330, 700, 420, 0.8, "swimmer")
    tracker.start([target], FRAME_SHAPE, target_box=BBox(520, 330, 700, 420), frame=BLANK_FRAME)
    for _ in range(6):
        result = tracker.update([], FRAME_SHAPE, frame=BLANK_FRAME)

    border_false_positive = Detection(560, 540, 710, 600, 0.9, "swimmer")
    result = tracker.update([border_false_positive], FRAME_SHAPE, frame=BLANK_FRAME)

    assert result is not None and result.predicted
    assert result.box.y2 < FRAME_SHAPE[0]


def test_water_relink_never_uses_person_heatmap_overlap() -> None:
    tracker = PersistentSwimmerTracker(max_missed_frames=45)
    target = Detection(520, 330, 700, 420, 0.8, "swimmer")
    tracker.start([target], FRAME_SHAPE, target_box=BBox(520, 330, 700, 420), frame=BLANK_FRAME)
    for _ in range(6):
        tracker.update([], FRAME_SHAPE, frame=BLANK_FRAME)

    spectator_overlap = Detection(525, 330, 705, 420, 0.99, "both")
    result = tracker.update([spectator_overlap], FRAME_SHAPE, frame=BLANK_FRAME)

    assert result is not None and result.predicted
    assert result.detection is None


def test_low_confidence_heatmap_observation_recovers_existing_track() -> None:
    tracker = PersistentSwimmerTracker(max_missed_frames=3)
    tracker.start_from_box(BBox(100, 200, 160, 250), FRAME_SHAPE, frame=BLANK_FRAME)
    tracker.update([Detection(108, 200, 168, 250, 0.8)], FRAME_SHAPE, frame=BLANK_FRAME)
    tracker.update([], FRAME_SHAPE, frame=BLANK_FRAME)

    recovered = tracker.update(
        [Detection(116, 200, 176, 250, 0.22, "swimmer")], FRAME_SHAPE, frame=BLANK_FRAME
    )

    assert recovered is not None
    assert not recovered.predicted
    assert recovered.source == "swimmer"


def test_water_handoff_permanently_excludes_person_only_detections() -> None:
    tracker = PersistentSwimmerTracker(max_missed_frames=3)
    tracker.start_from_box(BBox(100, 200, 160, 250), FRAME_SHAPE, frame=BLANK_FRAME)
    handoff = tracker.update(
        [Detection(108, 200, 168, 250, 0.5, "swimmer")], FRAME_SHAPE, frame=BLANK_FRAME
    )
    assert handoff is not None and tracker.water_locked

    result = tracker.update(
        [
            Detection(116, 200, 176, 250, 0.99, "person"),
            Detection(116, 200, 176, 250, 0.35, "swimmer"),
        ],
        FRAME_SHAPE,
        frame=BLANK_FRAME,
    )

    assert result is not None and not result.predicted
    assert result.source == "swimmer"


def test_streamlit_pipeline_uses_the_same_threshold_as_heatmap_script() -> None:
    assert config.SWIMMER_THRESHOLD == 0.35


def test_impossible_jump_creates_another_track_but_not_an_id_switch() -> None:
    tracker = PersistentSwimmerTracker(max_missed_frames=2)
    tracker.start_from_box(BBox(100, 200, 160, 250), FRAME_SHAPE, frame=BLANK_FRAME)
    selected_id = tracker.target_id

    result = tracker.update([Detection(800, 200, 860, 250, 1.0, "both")], FRAME_SHAPE, frame=BLANK_FRAME)

    assert result is not None and result.predicted
    assert result.center[0] < 300
    assert tracker.target_id == selected_id


def test_prediction_stops_but_botsort_can_still_reacquire() -> None:
    tracker = PersistentSwimmerTracker(max_missed_frames=1, max_reacquire_frames=6)
    tracker.start_from_box(BBox(100, 200, 160, 250), FRAME_SHAPE, frame=BLANK_FRAME)

    assert tracker.update([], FRAME_SHAPE, frame=BLANK_FRAME) is not None
    assert tracker.update([], FRAME_SHAPE, frame=BLANK_FRAME) is None
    reacquired = tracker.update([Detection(105, 200, 165, 250)], FRAME_SHAPE, frame=BLANK_FRAME)

    assert reacquired is not None and not reacquired.predicted
    assert tracker.missed_frames == 0


def test_removed_id_is_never_replaced_by_a_new_nearby_id() -> None:
    tracker = PersistentSwimmerTracker(max_missed_frames=1, max_reacquire_frames=2)
    tracker.start_from_box(BBox(100, 200, 160, 250), FRAME_SHAPE, frame=BLANK_FRAME)
    selected_id = tracker.target_id
    for _ in range(4):
        tracker.update([], FRAME_SHAPE, frame=BLANK_FRAME)

    result = tracker.update([Detection(105, 200, 165, 250, 1.0, "both")], FRAME_SHAPE, frame=BLANK_FRAME)

    assert result is None
    assert tracker.target_id == selected_id


def test_explicit_lane_constraint_remains_available_for_fixed_cameras() -> None:
    tracker = PersistentSwimmerTracker(lane=LaneConstraint(180, 280, "y"), max_missed_frames=2)
    tracker.start_from_box(BBox(100, 200, 160, 250), FRAME_SHAPE, frame=BLANK_FRAME)

    result = tracker.update([Detection(110, 330, 170, 380, 1.0, "both")], FRAME_SHAPE, frame=BLANK_FRAME)

    assert result is not None and result.predicted


def test_detector_integration_uses_public_detect_contract() -> None:
    class FakeHybridDetector:
        def detect(self, frame):
            return [Detection(110, 200, 170, 250, source="both")]

    tracker = PersistentSwimmerTracker(FakeHybridDetector())
    tracker.start_from_box(BBox(100, 200, 160, 250), FRAME_SHAPE, frame=BLANK_FRAME)

    result = tracker.track(BLANK_FRAME)

    assert result is not None
    assert result.source == "both"


def test_accepts_real_hybrid_detection_type() -> None:
    tracker = PersistentSwimmerTracker()
    tracker.start_from_box(BBox(100, 200, 160, 250), FRAME_SHAPE, frame=BLANK_FRAME)
    candidate = HybridDetection(110, 200, 170, 250, 0.8, "both", 0.7, 0.8)

    result = tracker.update([candidate], FRAME_SHAPE, frame=BLANK_FRAME)

    assert result is not None
    assert result.detection is candidate


def test_hybrid_merge_preserves_heatmap_box_instead_of_union() -> None:
    detector = HybridSwimmerDetector.__new__(HybridSwimmerDetector)
    person = HybridDetection(80, 120, 200, 320, 0.9, "person", 0.9, 0.0)
    swimmer = HybridDetection(110, 210, 180, 260, 0.45, "swimmer", 0.0, 0.45)

    merged = detector._merge_pair(person, swimmer)

    assert (merged.x1, merged.y1, merged.x2, merged.y2) == (110, 210, 180, 260)
    assert merged.source == "both"


def test_reset_forgets_the_selected_track_id() -> None:
    tracker = PersistentSwimmerTracker()
    tracker.start_from_box(BBox(100, 200, 160, 250), FRAME_SHAPE, frame=BLANK_FRAME)
    assert tracker.target_id is not None

    tracker.reset()

    assert not tracker.initialized
    assert tracker.target_id is None
