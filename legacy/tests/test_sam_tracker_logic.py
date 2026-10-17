"""Tests for the model-free parts of SAM 2 tracking and the 'is it swimming?' check."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.camera_motion import apply_affine, not_moving_flags, relative_steps, touches_edge
from core.geometry import BBox
from core.sam_tracker import mask_to_box, plan_segments, reacquired_flags
from core.tracker import STATUS_LOST, STATUS_NOT_MOVING, STATUS_TRACKED, Observation, stretch_notes


def shift(dx: float, dy: float, zoom: float = 1.0) -> np.ndarray:
    return np.array([[zoom, 0.0, dx], [0.0, zoom, dy]])


# --- plan_segments ----------------------------------------------------------
def test_single_checkpoint_tracks_forward_and_backward():
    plans = plan_segments([3], [False] * 8, 8)
    assert plans[0].positions == [3, 4, 5, 6, 7]
    assert plans[1].positions == [3, 2, 1, 0]


def test_each_checkpoint_runs_until_the_next_one():
    plans = plan_segments([0, 4], [False] * 8, 8)
    assert [p.positions for p in plans] == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_segments_stop_at_camera_cuts():
    cuts = [False, False, False, False, True, False]  # sample 4 starts a new shot
    plans = plan_segments([2], cuts, 6)
    assert plans[0].positions == [2, 3]
    assert plans[1].positions == [2, 1, 0]


def test_backward_stops_at_cut_before_checkpoint():
    cuts = [False, False, True, False, False]  # sample 2 starts the checkpoint's shot
    plans = plan_segments([3], cuts, 5)
    assert plans[1].positions == [3, 2]


def test_unsorted_checkpoints_are_rejected():
    with pytest.raises(ValueError):
        plan_segments([4, 1], [False] * 6, 6)


# --- mask_to_box --------------------------------------------------------------
def test_mask_box_uses_largest_blob_only():
    mask = np.zeros((100, 200), bool)
    mask[40:60, 50:90] = True     # swimmer
    mask[5:8, 180:183] = True     # stray splash pixels
    assert mask_to_box(mask) == BBox(50, 40, 90, 60)


def test_empty_or_huge_mask_gives_no_box():
    assert mask_to_box(np.zeros((50, 50), bool)) is None
    assert mask_to_box(np.ones((50, 50), bool)) is None  # covers the whole picture = clutter


# --- reacquired_flags -----------------------------------------------------------
def test_flag_after_a_long_gap_but_not_after_a_flicker():
    b = BBox(0, 0, 10, 10)
    boxes = [b, None, b, b, None, None, None, b, b, b, b]
    flags = reacquired_flags(boxes, [False] * len(boxes), gap=3, flag_len=2)
    assert flags == [False, False, False, False, False, False, False, True, True, False, False]


def test_checkpoint_frame_is_trusted_and_resets_gap():
    b = BBox(0, 0, 10, 10)
    boxes = [None, None, None, b, b]
    flags = reacquired_flags(boxes, [False, False, False, True, False], gap=3, flag_len=5)
    assert flags == [False] * 5


# --- camera-compensated motion --------------------------------------------------
def test_apply_affine():
    assert apply_affine(shift(5, -2), (10, 10)) == (15, 8)


def test_spectator_moving_only_with_the_camera_is_not_moving():
    # The camera pans 4 px per frame; the box moves exactly with it.
    boxes = [BBox(100 + 4 * t, 50, 130 + 4 * t, 110) for t in range(60)]
    motions = [None] + [shift(4, 0)] * 59
    flags = not_moving_flags(boxes, relative_steps(boxes, motions), window=60)
    assert all(flags[5:-5])


def test_swimmer_followed_by_a_panning_camera_is_moving():
    # The camera follows the swimmer, so the box barely moves in the image,
    # while the background moves 4 px per frame: relative to the pool the swimmer moves.
    boxes = [BBox(100, 50, 160, 70) for _ in range(60)]
    motions = [None] + [shift(-4, 0)] * 59
    flags = not_moving_flags(boxes, relative_steps(boxes, motions), window=60)
    assert not any(flags)


def test_swimmer_heading_to_the_camera_is_moving_because_box_grows():
    boxes = [BBox(100, 50, 100 + 30 * 1.02 ** t, 50 + 15 * 1.02 ** t) for t in range(60)]
    boxes = [BBox(b.x1 - b.width / 2 + 15, b.y1, b.x2 - b.width / 2 + 15, b.y2) for b in boxes]
    motions = [None] + [shift(0, 0)] * 59
    steps = relative_steps(boxes, motions)
    flags = not_moving_flags(boxes, steps, window=60)
    assert not any(flags[5:-5])


def test_camera_zoom_alone_does_not_count_as_movement():
    boxes = [BBox(0, 0, 40 * 1.01 ** t, 40 * 1.01 ** t) for t in range(40)]
    motions = [None] + [shift(0, 0, zoom=1.01)] * 39
    steps = relative_steps(boxes, motions)
    assert all(abs(s[2]) < 1e-9 for s in steps[1:])


def test_unknown_camera_motion_raises_no_flag():
    boxes = [BBox(100, 50, 130, 110)] * 40
    assert not any(not_moving_flags(boxes, relative_steps(boxes, [None] * 40), window=30))


def test_box_cut_off_at_the_border_is_not_measured():
    assert touches_edge(BBox(0, 10, 30, 50), 200, 100)
    boxes = [BBox(0, 10, 30, 50), BBox(0, 10, 20, 50)]
    assert relative_steps(boxes, [None, shift(0, 0)], (200, 100)) == [None, None]


# --- plain-language notes --------------------------------------------------------
def _obs(statuses: str, fps: float = 10.0) -> list[Observation]:
    code = {"#": STATUS_TRACKED, ".": STATUS_LOST, "S": STATUS_NOT_MOVING}
    return [Observation(i, i, i / fps, code[c], bbox=None if c == "." else BBox(0, 0, 5, 5))
            for i, c in enumerate(statuses)]


def test_notes_for_long_stretches_only():
    notes = stretch_notes(_obs("##" + "." * 15 + "##########" + "S" * 25 + "#" + ".." + "#"))
    assert len(notes) == 2
    assert "lost" in notes[0] and "0.2-1.6 s" in notes[0]
    assert "not moving" in notes[1] and "spectator" in notes[1]


def test_short_interruptions_do_not_split_a_stretch():
    # 1.2 s not moving, a 0.2 s blip of tracking, then 1.2 s not moving => ONE note
    notes = stretch_notes(_obs("S" * 12 + "##" + "S" * 12))
    assert len(notes) == 1 and "0.0-2.5 s" in notes[0]
