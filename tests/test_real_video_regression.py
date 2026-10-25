"""End-to-end regressions against clips from an uploaded SwimVision race."""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

import config
from core.geometry import BBox
from core.hybrid_detector import HybridSwimmerDetector
from core.models import RaceInfo
from core.race_analysis import analyze_race
from core.sam2_tracking import FrameResult, results_table
from core.swimmer_tracker import PersistentSwimmerTracker, PersistentVisualSwimmerTracker, TrackedSwimmer
from core.video import read_video_metadata


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
    start_frame: int = 0,
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
    processed_frames = 0
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            if frame_index < start_frame:
                frame_index += 1
                continue
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
            processed_frames += 1
    finally:
        video.release()

    assert processed_frames == expected_frames
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


def test_video2_full_race_keeps_selected_lane_identity(
    detector: HybridSwimmerDetector,
) -> None:
    # This is the complete user-supplied Video(2).mp4, not a synthetic clip.
    # The regions were checked against the selected lane at the dive, far-wall
    # approach, turn, return length, and finish. This prevents a false pass
    # where the tracker merely returns *some* box on every frame.
    identity_regions = {
        600: BBox(300, 250, 430, 400),
        693: BBox(300, 120, 420, 260),
        703: BBox(330, 150, 440, 270),
        718: BBox(300, 180, 430, 330),
        760: BBox(340, 210, 500, 340),
        800: BBox(350, 70, 520, 210),
        870: BBox(350, 0, 550, 140),
        890: BBox(350, 20, 500, 140),
        924: BBox(360, 140, 520, 270),
        995: BBox(580, 120, 760, 260),
        1066: BBox(730, 200, 848, 330),
        1137: BBox(760, 230, 848, 330),
        1208: BBox(650, 220, 848, 360),
        1279: BBox(400, 190, 600, 330),
        1350: BBox(350, 130, 490, 250),
        1421: BBox(380, 130, 530, 250),
        1490: BBox(400, 160, 530, 260),
        1529: BBox(380, 180, 520, 290),
        1634: BBox(380, 180, 520, 290),
    }
    tracker = PersistentVisualSwimmerTracker(config.VIT_TRACKER_MODEL_PATH)
    path = FIXTURES / "swimmer_video2_full.mp4"
    video = cv2.VideoCapture(str(path))
    assert video.isOpened(), f"could not open regression fixture: {path}"
    checkpoints: dict[int, TrackedSwimmer] = {}
    lost_frames: list[int] = []
    observed = estimated = 0
    longest_gap = prediction_gap = 0
    final_result = None
    analysis_frames: list[FrameResult] = []
    frame_index = 0
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            candidates = detector.detect(frame)
            final_result = (
                tracker.start_from_box(BBox(390, 40, 500, 300), frame.shape[:2], frame=frame)
                if not tracker.initialized
                else tracker.update(candidates, frame.shape[:2], frame=frame)
            )
            if final_result is None:
                lost_frames.append(frame_index)
                prediction_gap = 0
            elif final_result.predicted:
                estimated += 1
                prediction_gap += 1
                longest_gap = max(longest_gap, prediction_gap)
            else:
                observed += 1
                prediction_gap = 0
            if frame_index in identity_regions and final_result is not None:
                checkpoints[frame_index] = final_result
            analysis_frames.append(FrameResult(
                frame_index=frame_index, timestamp_s=frame_index / 29.98166259168704,
                box=final_result.box if final_result else None, mask_pixels=0,
                source=final_result.source if final_result else "",
                predicted=final_result.predicted if final_result else False,
            ))
            frame_index += 1
    finally:
        video.release()

    assert frame_index == 1635, "the regression must process every frame of Video(2).mp4"
    assert lost_frames == [], f"selected swimmer was lost at full-video frames {lost_frames}"
    assert tracker.water_locked
    assert observed >= 1400
    assert estimated <= 200
    assert longest_gap <= 90
    assert set(checkpoints) == set(identity_regions)
    for frame_index, region in identity_regions.items():
        assert region.x1 <= checkpoints[frame_index].center[0] <= region.x2
        assert region.y1 <= checkpoints[frame_index].center[1] <= region.y2

    assert final_result is not None

    # The same complete real video is also the analysis regression. A false
    # pass that merely produces boxes is not enough: the 50 SCY report must
    # find the race movement, one wall turn and two complete lengths.
    report = analyze_race(path, results_table(analysis_frames), RaceInfo("Freestyle", 50, "SCY"),
                          read_video_metadata(path))
    assert 500 <= report.start.frame_index <= 800, report.to_dict()
    assert len([event for event in report.events if event.kind == "turn"]) == 1, report.to_dict()
    water_entries = [event for event in report.events if event.kind == "water_entry"]
    assert len(water_entries) == 1 and 600 <= water_entries[0].frame_index <= 850, report.to_dict()
    assert [event.frame_index for event in report.events] == sorted(event.frame_index for event in report.events)
    assert len(report.lengths) == 2
    assert sum(length.distance for length in report.lengths) == 50
    assert report.finish.confidence in {"medium", "high"}, report.to_dict()
    assert report.finish.frame_index < frame_index - 1, report.to_dict()
    assert {"stroke_rhythm", "vertical_path", "cycle_economy", "finish_execution"}.issubset(
        {metric.key for metric in report.technique.metrics}
    )
