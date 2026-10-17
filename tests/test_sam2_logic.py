"""Non-model parts of the SAM 2 pipeline: masks -> boxes, statistics, validation, clip + video I/O.

These tests use synthetic data only. They check the plumbing, NOT how well
SAM 2 follows a real swimmer (that needs scripts/test_sam_tracking.py on real footage).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.errors import TrackingError
from core.geometry import BBox
from core.sam2_tracking import (
    FrameResult,
    Pass,
    Selection,
    analysis_indices,
    choose_box,
    compute_stats,
    plan_passes,
    mask_to_box,
    results_table,
    validate_request,
    verify_output_video,
    write_analysis_clip,
)
from core.video import VideoMetadata, open_h264_writer, read_frame, read_video_metadata


def test_mask_to_box_single_pixel_and_region():
    mask = np.zeros((10, 20), dtype=bool)
    assert mask_to_box(mask) is None
    mask[5, 3] = True
    assert mask_to_box(mask) == BBox(3, 5, 4, 6)
    mask[2:8, 10:15] = True
    assert mask_to_box(mask) == BBox(3, 2, 15, 8)


def test_mask_to_box_none_and_wrong_dims():
    assert mask_to_box(None) is None
    assert mask_to_box(np.ones((2, 2, 2), dtype=bool)) is None


def _results(pattern: str, fps: float = 30.0) -> list[FrameResult]:
    """'T' = tracked, '.' = lost, one character per analysed frame starting at frame 100."""
    return [FrameResult(100 + i, (100 + i) / fps, BBox(0, 0, 5, 5) if c == "T" else None, 25 if c == "T" else 0)
            for i, c in enumerate(pattern)]


def test_compute_stats_counts_and_intervals():
    s = compute_stats(_results("TT...T..TT"), 1 / 30)
    assert (s.analyzed_frames, s.tracked_frames, s.lost_frames) == (10, 5, 5)
    assert s.tracked_fraction == pytest.approx(0.5)
    assert [(i.start_frame, i.end_frame, i.n_frames) for i in s.lost_intervals] == [(102, 104, 3), (106, 107, 2)]
    assert s.longest_lost_frames == 3
    assert s.longest_lost_s == pytest.approx(3 / 30)


def test_compute_stats_trailing_loss_and_all_tracked():
    s = compute_stats(_results("TT..."), 1 / 30)
    assert s.lost_intervals[-1].end_frame == 104
    clean = compute_stats(_results("TTTT"), 1 / 30)
    assert clean.lost_frames == 0 and clean.lost_intervals == () and clean.longest_lost_s == 0.0


def test_compute_stats_respects_stride():
    s = compute_stats(_results("T.."), 2 / 60)  # 60 fps analysed every 2nd frame
    assert s.longest_lost_s == pytest.approx(2 * 2 / 60)


def test_results_table_marks_lost_rows_empty():
    table = results_table(_results("T."))
    assert list(table["status"]) == ["tracked", "lost"]
    assert table.loc[1, ["x1", "cx"]].isna().all()


META = VideoMetadata(fps=30.0, frame_count=300, duration_s=10.0, width=848, height=464, codec="avc1")


GOOD = BBox(100, 100, 150, 200)


def test_validate_request_accepts_good_input():
    validate_request(META, [Selection(10, GOOD)], 160)
    validate_request(META, [Selection(10, GOOD), Selection(60, GOOD)], 160)


@pytest.mark.parametrize("marks, end", [
    ([], 160),                                          # no mark
    ([Selection(300, GOOD)], 310),                      # frame outside the video
    ([Selection(10, BBox(800, 100, 900, 200))], 160),   # box leaves the frame
    ([Selection(10, BBox(100, 100, 102, 200))], 160),   # box too thin
    ([Selection(10, GOOD)], 10),                        # end not after the mark
    ([Selection(10, GOOD), Selection(200, GOOD)], 160), # end before the last mark
    ([Selection(10, GOOD)], 300),                       # end beyond last frame
    ([Selection(10, GOOD), Selection(10, GOOD)], 160),  # two marks on one frame
])
def test_validate_request_rejects_bad_input(marks, end):
    with pytest.raises(TrackingError):
        validate_request(META, marks, end)


def test_analysis_indices_include_marks_off_the_stride():
    assert analysis_indices([10], 16, 2) == [10, 12, 14, 16]
    assert analysis_indices([10, 13], 16, 2) == [10, 12, 13, 14, 16]


def test_plan_passes_single_mark_is_one_forward_pass():
    assert plan_passes([0], 5) == [Pass(0, "forward", (0, 1, 2, 3, 4))]


def test_plan_passes_two_marks():
    passes = plan_passes([0, 4], 8)
    assert passes == [
        Pass(0, "forward", (0, 1, 2, 3)),
        Pass(1, "forward", (4, 5, 6, 7)),
        Pass(1, "backward", (4, 3, 2, 1)),   # from mark 2 back to just after mark 1
    ]
    covered = {pos for p in passes for pos in p.positions}
    assert covered == set(range(8))


A, B = BBox(0, 0, 10, 10), BBox(50, 50, 60, 60)


@pytest.mark.parametrize("fwd, back, d_fwd, d_back, expected", [
    (None, None, 3, 3, "lost"),
    (A, None, 3, 3, "forward"),
    (None, A, 3, 3, "backward"),             # the dive gap: only the backward pass sees the swimmer
    (A, BBox(1, 1, 11, 11), 2, 9, "forward"),  # both agree: nearer mark wins
    (A, BBox(1, 1, 11, 11), 9, 2, "backward"),
    (A, B, 3, 3, "conflict"),                # different objects: never guess
])
def test_choose_box(fwd, back, d_fwd, d_back, expected):
    assert choose_box(fwd, back, d_fwd, d_back) == expected


def test_conflict_frames_count_as_lost():
    results = [FrameResult(1, 0.0, A, 5), FrameResult(2, 0.1, None, 0, conflict=True)]
    s = compute_stats(results, 0.1)
    assert (s.lost_frames, s.conflict_frames) == (1, 1)
    assert list(results_table(results)["status"]) == ["tracked", "conflict"]


def test_write_analysis_clip_stride_and_resize(mp4_30fps, tmp_path):
    clip = tmp_path / "clip.avi"
    indices = write_analysis_clip(mp4_30fps, [50, 55, 60, 65, 70], (160, 120), 6.0, clip)
    assert indices == [50, 55, 60, 65, 70]
    cap = cv2.VideoCapture(str(clip))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    assert len(frames) == 5 and frames[0].shape == (120, 160, 3)
    # Brightness encodes the source frame number (2 per frame): clip frame k = source frame 50 + 5k.
    for k, frame in enumerate(frames):
        source = read_frame(mp4_30fps, 50 + 5 * k)
        assert abs(float(frame.mean()) - float(source.mean())) < 3


def test_h264_writer_output_is_verified(tmp_path):
    path = tmp_path / "out.mp4"
    writer = open_h264_writer(path, 64, 48, 30.0)
    for i in range(12):
        writer.send(np.full((48, 64, 3), i * 10, dtype=np.uint8))
    writer.close()
    meta = read_video_metadata(path)
    assert (meta.frame_count, meta.width, meta.height) == (12, 64, 48)
    assert verify_output_video(path, 12, (64, 48))["codec"] == meta.codec
    with pytest.raises(TrackingError):
        verify_output_video(path, 13, (64, 48))


def test_h264_writer_rejects_odd_size(tmp_path):
    with pytest.raises(ValueError):
        open_h264_writer(tmp_path / "x.mp4", 63, 48, 30.0)
