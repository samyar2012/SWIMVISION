"""Tests for turning a mouse drag into a real-pixel box, and for the zoomed preview."""

from __future__ import annotations

import numpy as np

from core.geometry import BBox, box_from_drag
from core.overlay import zoomed_crop


def drag(x1, y1, x2, y2, width=424, height=232):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": width, "height": height}


def test_drag_is_scaled_to_real_frame_pixels():
    # Image shown at half size (424x232) of an 848x464 frame.
    box = box_from_drag(drag(100, 50, 150, 80), 848, 464)
    assert box == BBox(200, 100, 300, 160)


def test_drag_in_any_direction_gives_the_same_box():
    assert box_from_drag(drag(150, 80, 100, 50), 848, 464) == box_from_drag(drag(100, 50, 150, 80), 848, 464)


def test_release_outside_the_image_is_clamped():
    box = box_from_drag(drag(400, 200, 500, 300), 848, 464)
    assert box.x2 == 848 and box.y2 == 464


def test_plain_click_or_missing_data_gives_no_box():
    assert box_from_drag(drag(100, 50, 101, 51), 848, 464) is None
    assert box_from_drag({"x": 5, "y": 5}, 848, 464) is None
    assert box_from_drag(None, 848, 464) is None


def test_zoomed_crop_is_enlarged_and_stays_inside_frame():
    frame = np.zeros((464, 848, 3), np.uint8)
    crop = zoomed_crop(frame, BBox(2, 2, 40, 22))  # box at the frame corner
    assert crop.shape[0] >= 100 and crop.shape[1] > 0
    big = zoomed_crop(frame, BBox(300, 200, 500, 300))  # already big: no shrinking below 1x
    assert big.shape[0] >= 100
