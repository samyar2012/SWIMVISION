"""End-to-end regression against an uploaded SwimVision race clip."""

from __future__ import annotations

from pathlib import Path

import cv2

import config
from core.geometry import BBox
from core.hybrid_detector import HybridSwimmerDetector
from core.swimmer_tracker import PersistentSwimmerTracker


FIXTURE = Path(__file__).parent / "fixtures" / "swimmer_start_dive.mp4"
SELECTION = BBox(620, 150, 710, 340)
EXPECTED_FRAMES = 181


def test_uploaded_race_never_loses_or_switches_selected_swimmer() -> None:
    """The selected swimmer must survive blocks, dive, splash, and water entry."""
    detector = HybridSwimmerDetector(
        person_model_path=str(config.PERSON_MODEL_PATH),
        swimmer_model_path=str(config.SWIMMER_MODEL_PATH),
        person_confidence=config.PERSON_CONFIDENCE,
        swimmer_threshold=config.SWIMMER_THRESHOLD,
        merge_iou_threshold=config.MERGE_IOU_THRESHOLD,
    )
    tracker = PersistentSwimmerTracker(detector)
    video = cv2.VideoCapture(str(FIXTURE))
    assert video.isOpened(), f"could not open regression fixture: {FIXTURE}"

    lost_frames: list[int] = []
    water_observations = 0
    longest_prediction_gap = prediction_gap = 0
    final_result = None
    frame_index = 0
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            candidates = detector.detect(frame)
            final_result = (
                tracker.start(candidates, frame.shape[:2], target_box=SELECTION, frame=frame)
                if not tracker.initialized
                else tracker.update(candidates, frame.shape[:2], frame=frame)
            )
            if final_result is None:
                lost_frames.append(frame_index)
                prediction_gap = 0
            elif final_result.predicted:
                prediction_gap += 1
                longest_prediction_gap = max(longest_prediction_gap, prediction_gap)
            else:
                prediction_gap = 0
                if final_result.source in {"swimmer", "both"}:
                    water_observations += 1
            frame_index += 1
    finally:
        video.release()

    assert frame_index == EXPECTED_FRAMES
    assert lost_frames == [], f"selected swimmer was lost at fixture frames {lost_frames}"
    assert tracker.water_locked, "person track never handed off to the swimmer heatmap"
    assert water_observations >= 20, "water handoff did not produce enough real observations"
    assert longest_prediction_gap <= 30, "tracker predicted too long without reacquiring the swimmer"
    assert final_result is not None and not final_result.predicted

    # Frame 180 contains the selected swimmer in the nearest visible lane.
    # This rejects a fake pass that relinks to an adjacent lane or the yellow
    # caution tape at the bottom edge.
    final_x, final_y = final_result.center
    assert 520 <= final_x <= 780
    assert 260 <= final_y <= 390
