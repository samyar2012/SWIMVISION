from __future__ import annotations

import cv2
import numpy as np

from core.pool_geometry import LaneBoundary, PoolGeometry, PoolLaneDetector
from core.swimmer_detector import SwimmerDetector


def test_detects_lane_ropes_and_brackets_selected_swimmer() -> None:
    frame = np.full((400, 800, 3), (180, 115, 55), dtype=np.uint8)
    for intercept, colour in ((65, (0, 0, 255)), (165, (0, 255, 255)), (265, (255, 40, 20))):
        cv2.line(frame, (0, intercept), (799, intercept + 80), colour, 7, cv2.LINE_AA)

    geometry = PoolLaneDetector().detect(frame)
    corridor = geometry.corridor_for((400, 255))

    assert len(geometry.boundaries) >= 3
    assert corridor is not None
    assert corridor.contains((400, 255))
    assert not corridor.contains((400, 355))


def test_lane_rope_mask_splits_a_merged_heatmap_blob() -> None:
    detector = SwimmerDetector.__new__(SwimmerDetector)
    detector.input_size = 64
    detector.threshold = 0.35
    detector.min_blob_area = 4
    heatmap = np.zeros((64, 64), dtype=np.float32)
    heatmap[14:51, 24:41] = 0.80
    geometry = PoolGeometry((LaneBoundary(0.0, 32.0),), 64, 64)

    detections = detector._heatmap_to_detections(heatmap, 64, 64, geometry)

    assert len(detections) == 2
    assert all(abs(detection.confidence - 0.80) < 1e-5 for detection in detections)


def test_heatmap_confidence_ignores_inactive_pixels_inside_box() -> None:
    detector = SwimmerDetector.__new__(SwimmerDetector)
    detector.input_size = 32
    detector.threshold = 0.35
    detector.min_blob_area = 2
    heatmap = np.zeros((32, 32), dtype=np.float32)
    heatmap[10:15, 10:15] = 0.60
    heatmap[11:14, 11:14] = 0.90

    detections = detector._heatmap_to_detections(heatmap, 32, 32)

    assert len(detections) == 1
    expected = float(heatmap[heatmap >= detector.threshold].mean())
    assert abs(detections[0].confidence - expected) < 1e-6
