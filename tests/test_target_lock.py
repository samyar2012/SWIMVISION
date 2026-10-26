"""Target-lock behaviour of the persistent swimmer tracker on synthetic races.

Swimmers are 100 x 30 px boxes swimming along horizontal lanes 50 px apart,
the shape the U-Net heatmap produces for a side-on pool camera.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.geometry import BBox
from core.swimmer_tracker import PersistentSwimmerTracker


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
LENGTH, DEPTH = 100, 30
TARGET_LANE, ADJACENT_LANE = 200, 250
TARGET_LANE_CENTER = TARGET_LANE + DEPTH / 2


def swimmer(x: float, top: float, confidence: float = 0.8) -> Detection:
    return Detection(x, top, x + LENGTH, top + DEPTH, confidence)


def start_on(tracker: PersistentSwimmerTracker, detections: list[Detection], frame=BLANK_FRAME) -> int:
    target = detections[0]
    tracker.start(detections, FRAME_SHAPE, target_box=BBox(target.x1, target.y1, target.x2, target.y2),
                  frame=frame)
    assert tracker.target_id is not None
    return tracker.target_id


def test_botsort_uses_the_configured_starting_values() -> None:
    args = PersistentSwimmerTracker()._backend.args

    assert args.track_high_thresh == 0.20
    assert args.track_low_thresh == 0.08
    assert args.new_track_thresh == 0.30
    assert args.track_buffer == 120
    assert args.match_thresh == 0.70
    assert args.fuse_score is False
    assert args.gmc_method == "sparseOptFlow"
    assert args.with_reid is False


def test_adjacent_lane_merge_is_rejected_and_the_target_is_reacquired() -> None:
    tracker = PersistentSwimmerTracker()
    selected = start_on(tracker, [swimmer(100, TARGET_LANE), swimmer(100, ADJACENT_LANE)])
    for i in range(1, 16):
        x = 100 + 4 * i
        result = tracker.update([swimmer(x, TARGET_LANE), swimmer(x, ADJACENT_LANE)], FRAME_SHAPE,
                                frame=BLANK_FRAME)
        assert result is not None and not result.predicted

    # Both swimmers merge into one blob reaching into the adjacent lane. It
    # overlaps the target's predicted box (IoU 0.45), so plain BoT-SORT would
    # hand it to the selected ID and drag the box towards the other lane.
    for i in range(16, 19):
        x = 100 + 4 * i
        merged = Detection(x, 195, x + LENGTH, 262, 0.9)
        result = tracker.update([merged], FRAME_SHAPE, frame=BLANK_FRAME)
        assert result is not None and result.predicted
        assert abs(result.center[1] - TARGET_LANE_CENTER) < 5

    for i in range(19, 25):
        x = 100 + 4 * i
        result = tracker.update([swimmer(x, TARGET_LANE), swimmer(x, ADJACENT_LANE)], FRAME_SHAPE,
                                frame=BLANK_FRAME)
        assert result is not None and not result.predicted
        assert abs(result.center[1] - TARGET_LANE_CENTER) < 2
    assert tracker.target_id == selected


def test_temporary_detection_loss_predicts_inside_the_lane_and_reacquires_the_same_id() -> None:
    tracker = PersistentSwimmerTracker()
    selected = start_on(tracker, [swimmer(100, TARGET_LANE), swimmer(100, ADJACENT_LANE)])
    # The swimmer dips just before going under, so BoT-SORT's Kalman filter
    # learns a downward velocity that would carry it into the adjacent lane.
    tops = [TARGET_LANE] * 12 + [204, 208, 212, 216]
    for i, top in enumerate(tops, start=1):
        x = 100 + 4 * i
        result = tracker.update([swimmer(x, top), swimmer(x, ADJACENT_LANE)], FRAME_SHAPE, frame=BLANK_FRAME)
        assert result is not None and not result.predicted

    last_x = result.center[0]
    for i in range(len(tops) + 1, len(tops) + 9):
        x = 100 + 4 * i
        result = tracker.update([swimmer(x, ADJACENT_LANE)], FRAME_SHAPE, frame=BLANK_FRAME)
        assert result is not None and result.predicted
        assert result.center[1] < ADJACENT_LANE - 10
        assert result.center[0] >= last_x
        last_x = result.center[0]

    for i in range(len(tops) + 9, len(tops) + 13):
        x = 100 + 4 * i
        result = tracker.update([swimmer(x, TARGET_LANE), swimmer(x, ADJACENT_LANE)], FRAME_SHAPE,
                                frame=BLANK_FRAME)
        assert result is not None and not result.predicted
        assert abs(result.center[1] - TARGET_LANE_CENTER) < 5
    assert tracker.target_id == selected


def _pool_texture() -> np.ndarray:
    rng = np.random.default_rng(7)
    noise = (rng.random((FRAME_SHAPE[0], 1700)) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.GaussianBlur(noise, (0, 0), 2.0), cv2.COLOR_GRAY2BGR)


def test_camera_jump_and_pan_are_compensated_without_losing_the_target() -> None:
    texture = _pool_texture()
    # Static camera, a sudden 60 px pan (more than half a body length in one
    # frame), then a steady 15 px/frame pan following the race.
    offsets = [0] * 8 + [60] + [60 + 15 * k for k in range(1, 7)]
    tracker = PersistentSwimmerTracker()
    selected = None
    for i, offset in enumerate(offsets):
        frame = np.ascontiguousarray(texture[:, offset:offset + FRAME_SHAPE[1]])
        world_x = 300 + 4 * i
        detections = [swimmer(world_x - offset, TARGET_LANE), swimmer(world_x + 20 - offset, ADJACENT_LANE)]
        if selected is None:
            selected = start_on(tracker, detections, frame)
            continue
        result = tracker.update(detections, FRAME_SHAPE, frame=frame)
        assert result is not None and not result.predicted, f"lost the swimmer at frame {i}"
        assert abs(result.center[0] - (world_x - offset + LENGTH / 2)) < 10
        assert abs(result.center[1] - TARGET_LANE_CENTER) < 3
    assert tracker.target_id == selected


def test_finish_wall_damps_prediction_and_refuses_an_adjacent_swimmer() -> None:
    tracker = PersistentSwimmerTracker()
    selected = start_on(tracker, [swimmer(600, TARGET_LANE)])
    zone_mid_pool = None
    for i in range(1, 44):  # 6 px/frame towards the wall at x = 1000
        result = tracker.update([swimmer(600 + 6 * i, TARGET_LANE)], FRAME_SHAPE, frame=BLANK_FRAME)
        assert result is not None and not result.predicted
        if i == 20:
            zone_mid_pool = tracker.finish_zone
    assert zone_mid_pool is False
    assert tracker.finish_zone
    touch_x = result.center[0]

    # Splash at the touch hides the target. A swimmer from the next lane is
    # first detected reaching 14 px into the target's corridor: enough IoU for
    # BoT-SORT and inside the mid-pool corridor, but not at the wall.
    intruder = Detection(858, TARGET_LANE + 14, 958, TARGET_LANE + 14 + DEPTH, 0.9)
    for missing in range(1, 7):
        detections = [intruder] if 2 <= missing <= 4 else []
        result = tracker.update(detections, FRAME_SHAPE, frame=BLANK_FRAME)
        assert result is not None and result.predicted
        assert result.center[0] - touch_x <= 0.25 * 6 * missing + 1
        assert abs(result.center[1] - TARGET_LANE_CENTER) < 3

    # The target turns at the wall and swims back out.
    for step in range(1, 21):
        x = 858 - 4 * step
        result = tracker.update([swimmer(x, TARGET_LANE)], FRAME_SHAPE, frame=BLANK_FRAME)
        assert result is not None and not result.predicted
        assert abs(result.center[0] - (x + LENGTH / 2)) < 10
    assert tracker.target_id == selected
    assert not tracker.finish_zone
