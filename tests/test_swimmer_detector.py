from __future__ import annotations

import numpy as np
import pytest

from core.swimmer_detector import SwimmerDetector


def heatmap_detector(threshold: float = 0.35) -> SwimmerDetector:
    detector = SwimmerDetector.__new__(SwimmerDetector)
    detector.input_size = 256
    detector.threshold = threshold
    detector.min_blob_area = 10
    return detector


def test_swimmers_in_neighbouring_lanes_stay_separate_blobs() -> None:
    heatmap = np.zeros((256, 256), dtype=np.float32)
    heatmap[100:110, 50:90] = 0.9
    heatmap[118:128, 50:90] = 0.9  # 8 px of water between the two swimmers

    detections = heatmap_detector()._heatmap_to_detections(heatmap, 256, 256)

    assert len(detections) == 2


def test_confidence_uses_only_segmented_heatmap_pixels() -> None:
    heatmap = np.zeros((256, 256), dtype=np.float32)
    heatmap[100:130, 60:70] = 0.8   # L-shaped swimmer: its bounding rectangle
    heatmap[120:130, 70:120] = 0.8  # is mostly water ...
    heatmap[100:118, 72:120] = 0.3  # ... with sub-threshold heat in it.

    detections = heatmap_detector()._heatmap_to_detections(heatmap, 256, 256)

    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.8)
