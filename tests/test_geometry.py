"""Coordinate conversions: display pixels <-> original pixels <-> analysis pixels."""

from __future__ import annotations

import numpy as np
import pytest

from core.geometry import BBox, box_to_cropper_coords, cropper_rect_to_box, display_size, fit_size
from core.overlay import zoomed_crop


def test_fit_size_downscales_to_even_dimensions():
    assert fit_size(1920, 1080, 640) == (640, 360)
    assert fit_size(848, 464, 640) == (640, 350)  # 464 * 640/848 = 350.2 -> 350


def test_fit_size_never_upscales_and_none_means_original():
    assert fit_size(848, 464, 960) == (848, 464)
    assert fit_size(848, 464, None) == (848, 464)
    w, h = fit_size(849, 465, None)  # odd sizes become even (H.264), off by at most 1 px
    assert w % 2 == 0 and h % 2 == 0 and abs(w - 849) <= 1 and abs(h - 465) <= 1


def test_display_size_keeps_aspect_and_may_upscale():
    assert display_size(848, 464, 1200) == (1200, 657)
    assert display_size(1920, 1080, 900) == (900, 506)


def test_cropper_rect_maps_to_original_pixels():
    # 1920x1080 video shown at 960x540: every display pixel is 2 original pixels.
    rect = {"left": 100, "top": 50, "width": 40, "height": 80}
    assert cropper_rect_to_box(rect, 960, 540, 1920, 1080) == BBox(200, 100, 280, 260)


def test_cropper_rect_upscaled_display():
    # 848x464 video shown at 1200x657 (upscaled): coordinates shrink back.
    sx, sy = 848 / 1200, 464 / 657
    box = cropper_rect_to_box({"left": 600, "top": 300, "width": 120, "height": 90}, 1200, 657, 848, 464)
    assert box.x1 == pytest.approx(600 * sx) and box.y1 == pytest.approx(300 * sy)
    assert box.x2 == pytest.approx(720 * sx) and box.y2 == pytest.approx(390 * sy)


def test_cropper_round_trip_within_one_display_pixel():
    original = BBox(330, 133, 399, 278)
    for dw in (700, 900, 1200):
        dh = display_size(848, 464, dw)[1]
        left, right, top, bottom = box_to_cropper_coords(original, dw, dh, 848, 464)
        back = cropper_rect_to_box({"left": left, "top": top, "width": right - left, "height": bottom - top},
                                   dw, dh, 848, 464)
        tolerance = 848 / dw  # one display pixel in original pixels
        for a, b in zip(back.to_list(), original.to_list()):
            assert abs(a - b) <= tolerance


def test_cropper_rect_is_clamped_and_tiny_rejected():
    box = cropper_rect_to_box({"left": 900, "top": 500, "width": 200, "height": 200}, 960, 540, 1920, 1080)
    assert box == BBox(1800, 1000, 1920, 1080)
    assert cropper_rect_to_box({"left": 10, "top": 10, "width": 1, "height": 30}, 960, 540, 1920, 1080) is None


@pytest.mark.parametrize("bad", [None, {}, {"left": "x", "top": 0, "width": 5, "height": 5}])
def test_cropper_rect_malformed_returns_none(bad):
    assert cropper_rect_to_box(bad, 960, 540, 1920, 1080) is None


def test_analysis_scaling_round_trip():
    box = BBox(425, 115, 500, 160)
    aw, ah = fit_size(848, 464, 640)
    sx, sy = aw / 848, ah / 464
    back = box.scaled(sx, sy).scaled(1 / sx, 1 / sy)
    for a, b in zip(back.to_list(), box.to_list()):
        assert a == pytest.approx(b)


def test_is_inside():
    assert BBox(0, 0, 848, 464).is_inside(848, 464)
    assert not BBox(-1, 0, 10, 10).is_inside(848, 464)
    assert not BBox(10, 10, 10, 20).is_inside(848, 464)  # zero width


def test_zoomed_crop_enlarges_small_box():
    frame = np.zeros((464, 848, 3), dtype=np.uint8)
    crop = zoomed_crop(frame, BBox(400, 200, 420, 240))
    assert crop.shape[0] >= 120  # 3 x 40 px tall region, enlarged
