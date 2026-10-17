"""Tests for the pure tracking logic: IoU, target following, segments, summaries.

The most important property tested here: the tracker must NEVER silently jump
to a different swimmer. When unsure, it must report the swimmer as lost.
"""

from __future__ import annotations

import pytest

from core.errors import TrackingError
from core.geometry import BBox
from core.tracker import (
    FollowConfig,
    Observation,
    STATUS_LOST,
    STATUS_LOW_CONFIDENCE,
    STATUS_REACQUIRED,
    STATUS_TRACKED,
    SwimmerSelection,
    TrackedBox,
    TrackingResult,
    build_observations,
    cut_free_range,
    find_unreliable_segments,
    FollowResult,
    _pick_best,
    follow_target,
    located_fraction,
    match_anchor_track,
    observations_to_dataframe,
    snap_to_sample_grid,
    summarize_tracking,
    TrackingSummary,
    visual_box_is_plausible,
)
from core.video import analysis_stride

CFG = FollowConfig(max_gap_samples=3)


def box_at(x: float, y: float = 100, w: float = 100, h: float = 40) -> BBox:
    return BBox(x, y, x + w, y + h)


def tb(track_id: int, x: float, y: float = 100, conf: float = 0.9) -> TrackedBox:
    return TrackedBox(track_id, box_at(x, y), conf)


# --- geometry ----------------------------------------------------------------
def test_iou_identical_and_disjoint():
    assert box_at(0).iou(box_at(0)) == pytest.approx(1.0)
    assert box_at(0).iou(box_at(500)) == 0.0


def test_iou_half_overlap():
    # two 100-wide boxes shifted by 50 -> intersection 50x40, union 150x40 -> 1/3
    assert box_at(0).iou(box_at(50)) == pytest.approx(1 / 3)


def test_bbox_center_and_clamp():
    b = BBox(-10, 5, 110, 45)
    assert b.center == (50.0, 25.0)
    assert b.clamp(100, 40) == BBox(0, 5, 100, 40)


def test_xywh_round_trip():
    assert BBox.from_xywh(10, 20, 30, 40).to_xywh_int() == (10, 20, 30, 40)


# --- anchor matching -----------------------------------------------------------
def test_anchor_picks_best_overlap():
    cands = [tb(1, 0), tb(2, 400)]
    assert match_anchor_track(cands, box_at(10), 0.3).track_id == 1


def test_anchor_none_when_no_overlap():
    assert match_anchor_track([tb(1, 0)], box_at(900), 0.3) is None


# --- follow_target -------------------------------------------------------------
def test_simple_follow_keeps_same_id():
    frames = [[tb(7, 10 * i)] for i in range(6)]
    results = follow_target(frames, 0, box_at(0), CFG)
    assert [r.track_id for r in results] == [7] * 6
    assert not any(r.reacquired for r in results)


def test_ignores_other_swimmers_with_other_ids():
    # Swimmer 1 in lane at y=100, swimmer 2 in another lane at y=300.
    frames = [[tb(1, 10 * i, 100), tb(2, 10 * i, 300)] for i in range(5)]
    results = follow_target(frames, 2, box_at(20, 100), CFG)
    assert all(r.track_id == 1 for r in results)


def test_short_gap_same_id_is_flagged_reacquired():
    frames = [[tb(1, 0)], [tb(1, 10)], [], [tb(1, 30)], [tb(1, 40)]]
    results = follow_target(frames, 0, box_at(0), CFG)
    assert results[2] is None                     # lost frame has no coordinates
    assert results[3].reacquired is True          # first frame back is flagged
    assert results[4].reacquired is False


def test_id_switch_with_clear_overlap_is_reacquired_and_flagged():
    # ByteTrack gives the same swimmer a new ID (9) on frame 2.
    frames = [[tb(1, 0)], [tb(1, 10)], [tb(9, 20)], [tb(9, 30)]]
    results = follow_target(frames, 0, box_at(0), CFG)
    assert results[2].track_id == 9 and results[2].reacquired is True
    assert results[3].track_id == 9 and results[3].reacquired is False


def test_does_not_switch_to_distant_swimmer():
    # Our swimmer vanishes; a different swimmer (new ID) appears far away.
    frames = [[tb(1, 0)], [tb(1, 10)], [tb(5, 600, 300)], [tb(5, 610, 300)]]
    results = follow_target(frames, 0, box_at(0), CFG)
    assert results[2] is None and results[3] is None


def test_ambiguous_takeover_is_refused():
    # After our swimmer disappears, two new people overlap the last box equally.
    frames = [[tb(1, 0)], [tb(1, 10)], [tb(8, 15), tb(9, 15)]]
    results = follow_target(frames, 0, box_at(0), CFG)
    assert results[2] is None


def test_gap_longer_than_limit_stays_lost_for_new_ids():
    gap = [[] for _ in range(5)]                       # > max_gap_samples (3)
    frames = [[tb(1, 0)]] + gap + [[tb(4, 5)]]         # new ID right where the swimmer was
    results = follow_target(frames, 0, box_at(0), CFG)
    assert results[-1] is None


def test_original_id_returning_after_long_gap_is_trusted_but_flagged():
    gap = [[] for _ in range(5)]
    frames = [[tb(1, 0)]] + gap + [[tb(1, 50)]]
    results = follow_target(frames, 0, box_at(0), CFG)
    assert results[-1].track_id == 1 and results[-1].reacquired is True


def test_follows_backward_from_anchor_in_the_middle():
    frames = [[tb(3, 10 * i)] for i in range(7)]
    results = follow_target(frames, 3, box_at(30), CFG)
    assert len(results) == 7 and all(r.track_id == 3 for r in results)


def test_no_matching_track_raises():
    with pytest.raises(TrackingError):
        follow_target([[tb(1, 0)], [tb(1, 5)]], 0, box_at(900), CFG)


# --- visual tracker sanity checks --------------------------------------------------------------
def test_plausible_box_accepted():
    assert visual_box_is_plausible(box_at(20), box_at(10), 640, 480, CFG)


@pytest.mark.parametrize("bad", [
    BBox(100, 100, 102, 102),       # collapsed to a dot
    BBox(0, 0, 640, 480),           # ballooned to the whole frame
    BBox(600, 100, 760, 140),       # mostly outside the picture
    BBox(100, 100, 400, 220),       # jumped to ~9x the previous area in one step
])
def test_implausible_box_rejected(bad):
    assert not visual_box_is_plausible(bad, box_at(100), 640, 480, CFG)


def test_swimmer_approaching_camera_may_grow_a_lot_gradually():
    """Regression test from real footage: a swimmer coming toward the camera grew
    ~4x in area over two seconds. That is legitimate and must not be flagged."""
    box = BBox(300, 200, 390, 244)               # 90 x 44 px
    for _ in range(40):                           # +3.5% area per analysed frame
        grown = BBox(box.x1 - 1, box.y1 - 0.5, box.x2 + 1, box.y2 + 0.5)
        assert visual_box_is_plausible(grown, box, 848, 464, CFG)
        box = grown
    assert box.area > 3 * 90 * 44                 # grew more than 3x in total, step by step


# --- camera cuts ---------------------------------------------------------------------------
def test_cut_free_range():
    cuts = [False, False, True, False, False, True, False]   # shots: [0-1] [2-4] [5-6]
    assert cut_free_range(cuts, 3, 7) == (2, 4)
    assert cut_free_range(cuts, 0, 7) == (0, 1)
    assert cut_free_range(cuts, 6, 7) == (5, 6)
    assert cut_free_range(None, 3, 7) == (0, 6)


def test_tracking_never_crosses_a_camera_cut():
    # Same track ID continues after the cut (e.g. a different swimmer in the new shot).
    frames = [[tb(1, 10 * i)] for i in range(7)]
    cuts = [False, False, False, False, True, False, False]   # cut between sample 3 and 4
    results = follow_target(frames, 1, box_at(10), CFG, cuts)
    assert [r is not None for r in results] == [True] * 4 + [False] * 3


def test_cut_before_anchor_also_stops_backward_tracking():
    frames = [[tb(1, 10 * i)] for i in range(6)]
    cuts = [False, False, True, False, False, False]          # cut between sample 1 and 2
    results = follow_target(frames, 3, box_at(30), CFG, cuts)
    assert [r is not None for r in results] == [False, False, True, True, True, True]


# --- observations / segments / summary ------------------------------------------------
def make_observations() -> list[Observation]:
    results = [
        follow_result for follow_result in follow_target(
            [[tb(1, 0)], [tb(1, 0, conf=0.1)], [], [], [tb(1, 20)], [tb(1, 30)]], 0, box_at(0), CFG)
    ]
    return build_observations(range(6), 30.0, results, CFG)


def test_build_observations_status_and_timestamps():
    obs = make_observations()
    assert [o.status for o in obs] == [STATUS_TRACKED, STATUS_LOW_CONFIDENCE, STATUS_LOST,
                                       STATUS_LOST, STATUS_REACQUIRED, STATUS_TRACKED]
    assert obs[3].timestamp_s == pytest.approx(3 / 30)
    assert obs[2].bbox is None            # lost => no invented coordinates


def test_timestamps_respect_stride():
    results = follow_target([[tb(1, 0)], [tb(1, 5)], [tb(1, 10)]], 0, box_at(0), CFG)
    obs = build_observations([0, 2, 4], 60.0, results, CFG)   # every 2nd frame of 60 fps
    assert [o.timestamp_s for o in obs] == pytest.approx([0.0, 2 / 60, 4 / 60])


def test_unreliable_segments_group_consecutive_frames():
    segs = find_unreliable_segments(make_observations())
    assert len(segs) == 1
    seg = segs[0]
    assert (seg.start_frame, seg.end_frame) == (1, 4)
    assert set(seg.reasons) == {STATUS_LOW_CONFIDENCE, STATUS_LOST, STATUS_REACQUIRED}


def test_dataframe_has_nan_for_lost_frames():
    df = observations_to_dataframe(make_observations())
    assert len(df) == 6
    assert df.loc[2, "cx"] != df.loc[2, "cx"]          # NaN != NaN
    assert df.loc[0, "cx"] == pytest.approx(50.0)


def test_summary_counts_and_round_trip():
    result = TrackingResult(make_observations(), "yolo_tracker", 1, ["note"], 1.5, {"model": "x"})
    s = summarize_tracking(result)
    assert (s.n_samples, s.n_tracked, s.n_low_confidence, s.n_reacquired, s.n_lost) == (6, 2, 1, 1, 2)
    assert s.tracked_fraction == pytest.approx(4 / 6)
    assert TrackingSummary.from_dict(s.to_dict()) == s


def test_selection_round_trip():
    sel = SwimmerSelection(12, box_at(5), "manual", None)
    assert SwimmerSelection.from_dict(sel.to_dict()) == sel


# --- choosing between YOLO and visual results ----------------------------------------------------
def _fr(n_found: int, n_total: int) -> list:
    return [FollowResult(1, box_at(0), 0.9)] * n_found + [None] * (n_total - n_found)


def test_located_fraction():
    assert located_fraction(_fr(3, 4)) == pytest.approx(0.75)
    assert located_fraction([]) == 0.0


def test_pick_best_prefers_higher_coverage_and_explains():
    notes: list[str] = []
    results, mode = _pick_best(_fr(2, 10), _fr(8, 10), notes)
    assert mode == "visual_tracker" and located_fraction(results) == 0.8
    assert "20%" in notes[0] and "80%" in notes[0]


def test_pick_best_keeps_yolo_when_it_is_better():
    notes: list[str] = []
    _, mode = _pick_best(_fr(6, 10), _fr(3, 10), notes)
    assert mode == "yolo_tracker" and notes


def test_pick_best_handles_missing_candidates():
    assert _pick_best(_fr(5, 10), None, [])[1] == "yolo_tracker"
    assert _pick_best(None, _fr(5, 10), [])[1] == "visual_tracker"
    with pytest.raises(TrackingError):
        _pick_best(None, None, [])


# --- sampling helpers ---------------------------------------------------------------------
@pytest.mark.parametrize("fps,expected", [(24, 1), (29.97, 1), (30, 1), (50, 2), (59.94, 2), (120, 4)])
def test_analysis_stride(fps, expected):
    assert analysis_stride(fps) == expected


def test_snap_to_sample_grid():
    assert snap_to_sample_grid(7, 2, 100) == (4, 8)      # nearest multiple of 2
    assert snap_to_sample_grid(99, 2, 100) == (49, 98)   # never past the last sample
