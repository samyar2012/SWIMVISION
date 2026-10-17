"""Integration tests for the tracking pipeline using videos with KNOWN ground truth.

A textured patch moves at a known speed over a textured background, so we can
measure real tracking error in pixels. These are mechanical checks of the
pipeline, not claims about swimming accuracy - that needs real, manually
labelled race footage (Phase 10 validation).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.errors import ModelLoadError
from core.geometry import BBox
from core.overlay import write_annotated_video
from core.tracker import STATUS_LOST, SwimmerSelection, track_swimmer
from core.video import is_browser_playable, read_video_metadata

W, H, N, SPEED = 320, 240, 60, 3          # frame size, frame count, patch speed (px/frame)
PATCH_W, PATCH_H, START_X, Y = 60, 40, 40, 100


def _make_moving_patch_video(path: Path, fps: float = 30.0, cut_at: int | None = None) -> Path:
    """Static noisy background + a noisy patch moving right by SPEED px per frame.

    If ``cut_at`` is given the background changes completely at that frame
    (simulating a camera cut).
    """
    rng = np.random.default_rng(1)
    background = rng.integers(60, 120, (H, W, 3), dtype=np.uint8)
    other_background = rng.integers(130, 250, (H, W, 3), dtype=np.uint8)
    patch = rng.integers(0, 255, (PATCH_H, PATCH_W, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i in range(N):
        frame = (other_background if cut_at is not None and i >= cut_at else background).copy()
        x = START_X + SPEED * i
        frame[Y:Y + PATCH_H, x:x + PATCH_W] = patch
        writer.write(frame)
    writer.release()
    return path


def _truth_center(frame_index: int) -> tuple[float, float]:
    return START_X + SPEED * frame_index + PATCH_W / 2, Y + PATCH_H / 2


def _selection() -> SwimmerSelection:
    return SwimmerSelection(0, BBox(START_X, Y, START_X + PATCH_W, Y + PATCH_H), "manual")


def _run(path: Path, **kwargs):
    meta = read_video_metadata(path)
    try:
        result = track_swimmer(path, meta, _selection(), stride=1, method="visual", **kwargs)
    except ModelLoadError:
        pytest.skip("visual tracking model could not be downloaded (offline)")
    return meta, result


def test_visual_tracker_follows_known_motion(tmp_path):
    meta, result = _run(_make_moving_patch_video(tmp_path / "move.mp4"))
    table = result.to_dataframe()
    found = table[table["status"] != STATUS_LOST]
    assert len(found) / len(table) >= 0.9
    errors = [np.hypot(r.cx - _truth_center(int(r.frame_index))[0], r.cy - _truth_center(int(r.frame_index))[1])
              for r in found.itertuples()]
    assert np.mean(errors) < 8.0          # mean centre error under 8 px on a 320 px-wide frame


def test_tracking_stops_at_camera_cut_and_says_so(tmp_path):
    meta, result = _run(_make_moving_patch_video(tmp_path / "cut.mp4", cut_at=30))
    table = result.to_dataframe()
    assert (table.loc[table["frame_index"] >= 30, "status"] == STATUS_LOST).all()
    assert (table.loc[table["frame_index"] < 30, "status"] != STATUS_LOST).sum() >= 25
    assert any("camera cut" in note.lower() for note in result.notes)


def test_timestamps_use_real_fps_with_stride(tmp_path):
    path = _make_moving_patch_video(tmp_path / "sixty.mp4", fps=60.0)
    meta = read_video_metadata(path)
    try:
        result = track_swimmer(path, meta, _selection(), stride=2, method="visual")
    except ModelLoadError:
        pytest.skip("offline")
    table = result.to_dataframe()
    assert len(table) == N // 2
    assert table["timestamp_s"].iloc[1] == pytest.approx(2 / 60)
    assert table["frame_index"].tolist() == list(range(0, N, 2))


def test_annotated_video_is_browser_playable_and_real_time(tmp_path):
    path = _make_moving_patch_video(tmp_path / "src.mp4", fps=60.0)
    meta = read_video_metadata(path)
    try:
        table = track_swimmer(path, meta, _selection(), stride=2, method="visual").to_dataframe()
    except ModelLoadError:
        pytest.skip("offline")
    out = write_annotated_video(path, meta, table, tmp_path / "annotated.mp4", stride=2)
    out_meta = read_video_metadata(out)
    assert is_browser_playable(out, out_meta)
    assert out_meta.frame_count == len(table)
    assert out_meta.fps == pytest.approx(30.0)                      # 60 fps / stride 2
    assert out_meta.duration_s == pytest.approx(meta.duration_s, abs=0.1)   # real-time playback
