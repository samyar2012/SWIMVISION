"""Tests for camera-cut detection (core.scene)."""

from __future__ import annotations

import numpy as np

from core.scene import frame_signature, histogram_similarity, is_scene_cut, thumbnail_difference


def _noise_frame(seed: int, mean: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(mean, 30, (240, 320, 3)), 0, 255).astype(np.uint8)


def test_identical_frames_are_not_a_cut():
    s = frame_signature(_noise_frame(0, 120))
    assert histogram_similarity(s, s) > 0.99
    assert thumbnail_difference(s, s) == 0.0
    assert not is_scene_cut(s, s)


def test_small_motion_is_not_a_cut():
    f = _noise_frame(0, 120)
    shifted = np.roll(f, 12, axis=1)               # camera pan: same content, moved
    assert not is_scene_cut(frame_signature(f), frame_signature(shifted))


def test_very_different_brightness_is_a_cut():
    assert is_scene_cut(frame_signature(_noise_frame(1, 40)), frame_signature(_noise_frame(2, 210)))


def test_same_histogram_but_different_layout_is_a_cut():
    """Two shots with identical brightness distributions but different content
    (the case a histogram alone misses): left half dark vs right half dark."""
    a = np.zeros((240, 320, 3), np.uint8)
    a[:, 160:] = 200
    b = a[:, ::-1].copy()
    sa, sb = frame_signature(a), frame_signature(b)
    assert histogram_similarity(sa, sb) > 0.95      # histogram cannot tell them apart
    assert is_scene_cut(sa, sb)                     # the thumbnail difference can
