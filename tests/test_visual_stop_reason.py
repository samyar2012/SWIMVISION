"""Tests for the human-readable reasons the visual tracker gives for stopping."""

from __future__ import annotations

from core.geometry import BBox
from core.tracker import FollowConfig, visual_stop_reason

CFG = FollowConfig()
PREV = BBox(100, 100, 190, 144)


def test_no_reason_when_everything_is_fine():
    assert visual_stop_reason(True, BBox(105, 100, 195, 144), 0.5, PREV, 848, 464, CFG) is None


def test_reason_when_tracker_fails():
    assert "could not find" in visual_stop_reason(False, None, 0.0, PREV, 848, 464, CFG)


def test_reason_mentions_score_numbers():
    reason = visual_stop_reason(True, BBox(105, 100, 195, 144), 0.25, PREV, 848, 464, CFG)
    assert "0.25" in reason and "0.30" in reason


def test_reason_when_box_is_implausible():
    reason = visual_stop_reason(True, BBox(0, 0, 848, 464), 0.9, PREV, 848, 464, CFG)
    assert "implausible" in reason
