"""End-to-end regressions against clips from an uploaded SwimVision race."""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

import config
from core.geometry import BBox
from core.hybrid_detector import HybridSwimmerDetector
from core.swimmer_tracker import PersistentSwimmerTracker, TrackedSwimmer


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def detector() -> HybridSwimmerDetector:
    return HybridSwimmerDetector(
        person_model_path=str(config.PERSON_MODEL_PATH),
        swimmer_model_path=str(config.SWIMMER_MODEL_PATH),
        person_confidence=config.PERSON_CONFIDENCE,
        swimmer_threshold=config.SWIMMER_THRESHOLD,
        merge_iou_threshold=config.MERGE_IOU_THRESHOLD,
    )


def _run_clip(
    detector: HybridSwimmerDetector,
    name: str,
    selection: BBox,
    expected_frames: int,
    checkpoint_frames: set[int] | None = None,
) -> tuple[PersistentSwimmerTracker, TrackedSwimmer, int, int, dict[int, TrackedSwimmer]]:
    tracker = PersistentSwimmerTracker(detector)
    path = FIXTURES / name
    video = cv2.VideoCapture(str(path))
    assert video.isOpened(), f"could not open regression fixture: {path}"

    lost_frames: list[int] = []
    water_observations = 0
    longest_prediction_gap = prediction_gap = 0
    checkpoints: dict[int, TrackedSwimmer] = {}
    final_result = None
    frame_index = 0
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            candidates = detector.detect(frame)
            final_result = (
                tracker.start(candidates, frame.shape[:2], target_box=selection, frame=frame)
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
            if checkpoint_frames and frame_index in checkpoint_frames and final_result is not None:
                checkpoints[frame_index] = final_result
            frame_index += 1
    finally:
        video.release()

    assert frame_index == expected_frames
    assert lost_frames == [], f"selected swimmer was lost at fixture frames {lost_frames}"
    assert final_result is not None
    return tracker, final_result, water_observations, longest_prediction_gap, checkpoints


def test_uploaded_start_and_dive_never_loses_selected_swimmer(
    detector: HybridSwimmerDetector,
) -> None:
    tracker, final_result, water_observations, longest_gap, _ = _run_clip(
        detector,
        "swimmer_start_dive.mp4",
        BBox(620, 150, 710, 340),
        181,
    )

    assert tracker.water_locked, "person track never handed off to the swimmer heatmap"
    assert water_observations >= 20, "water handoff did not produce enough real observations"
    assert longest_gap <= 30, "tracker predicted too long without reacquiring the swimmer"
    assert not final_result.predicted
    final_x, final_y = final_result.center
    assert 520 <= final_x <= 780
    assert 260 <= final_y <= 390


def test_uploaded_camera_pan_never_relinks_to_spectators(
    detector: HybridSwimmerDetector,
) -> None:
    tracker, final_result, water_observations, longest_gap, checkpoints = _run_clip(
        detector,
        "swimmer_camera_pan.mp4",
        BBox(278, 53, 407, 125),
        341,
        {80},
    )

    assert tracker.water_locked
    assert water_observations >= 250
    assert longest_gap <= 60

    # Source frame 2400: the selected swimmer is left/center in the pool.
    # The rejected spectator is at the lower-right edge of the frame.
    pan_x, pan_y = checkpoints[80].center
    assert pan_x < 500
    assert pan_y < 250

    # At source frame 2660 the swimmer is partially off the lower-left edge.
    # This rejects a fake pass on spectators standing above the pool.
    final_x, final_y = final_result.center
    assert final_x < 200
    assert final_y > 330
