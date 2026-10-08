"""Tests for core.video: timing math, metadata reading and error handling."""

from __future__ import annotations

import pytest

from core.errors import UnsupportedVideoError, VideoDecodeError
from core.video import (
    compute_duration,
    frame_to_timestamp,
    is_browser_playable,
    make_browser_preview,
    read_frame,
    read_video_metadata,
    timestamp_to_frame,
)


# --- pure timing math -------------------------------------------------------
def test_frame_to_timestamp_basic():
    assert frame_to_timestamp(0, 30) == 0.0
    assert frame_to_timestamp(45, 30) == pytest.approx(1.5)


def test_frame_to_timestamp_fractional_fps():
    assert frame_to_timestamp(30, 30000 / 1001) == pytest.approx(1.001)


def test_timestamp_round_trip():
    for fps in (24.0, 25.0, 29.97, 60.0):
        for frame in (0, 1, 17, 301):
            assert timestamp_to_frame(frame_to_timestamp(frame, fps), fps) == frame


def test_compute_duration():
    assert compute_duration(90, 30) == pytest.approx(3.0)
    assert compute_duration(0, 30) == 0.0


@pytest.mark.parametrize("bad_call", [
    lambda: frame_to_timestamp(1, 0),
    lambda: frame_to_timestamp(-1, 30),
    lambda: timestamp_to_frame(1.0, -5),
    lambda: timestamp_to_frame(-0.1, 30),
    lambda: compute_duration(10, 0),
])
def test_invalid_timing_inputs_raise(bad_call):
    with pytest.raises(ValueError):
        bad_call()


# --- metadata from real files ----------------------------------------------
def test_metadata_mp4(mp4_30fps):
    meta = read_video_metadata(mp4_30fps)
    assert meta.fps == pytest.approx(30.0)
    assert meta.frame_count == 90
    assert meta.duration_s == pytest.approx(3.0)
    assert (meta.width, meta.height) == (320, 240)


def test_metadata_avi(avi_25fps):
    meta = read_video_metadata(avi_25fps)
    assert meta.fps == pytest.approx(25.0)
    assert meta.frame_count == 50
    assert meta.duration_s == pytest.approx(2.0)


def test_metadata_fractional_fps_mov(mov_2997fps):
    meta = read_video_metadata(mov_2997fps)
    assert meta.fps == pytest.approx(30000 / 1001, rel=1e-3)
    assert meta.frame_count == 60
    assert meta.duration_s == pytest.approx(60 / (30000 / 1001), rel=1e-3)


def test_metadata_serialization_round_trip(mp4_30fps):
    from core.video import VideoMetadata
    meta = read_video_metadata(mp4_30fps)
    assert VideoMetadata.from_dict(meta.to_dict()) == meta


def test_read_frame_returns_correct_size(mp4_30fps):
    frame = read_frame(mp4_30fps, 10)
    assert frame.shape == (240, 320, 3)


def test_read_frame_out_of_range_raises(mp4_30fps):
    with pytest.raises(VideoDecodeError):
        read_frame(mp4_30fps, 10_000)


# --- error handling ---------------------------------------------------------
def test_unsupported_extension(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    with pytest.raises(UnsupportedVideoError):
        read_video_metadata(bad)


def test_missing_file(tmp_path):
    with pytest.raises(UnsupportedVideoError):
        read_video_metadata(tmp_path / "nope.mp4")


def test_corrupt_video_raises_decode_error(tmp_path):
    fake = tmp_path / "broken.mp4"
    fake.write_bytes(b"this is not really a video" * 100)
    with pytest.raises(VideoDecodeError):
        read_video_metadata(fake)


# --- browser preview --------------------------------------------------------
def test_avi_is_not_browser_playable_but_preview_is(avi_25fps, tmp_path):
    meta = read_video_metadata(avi_25fps)
    assert not is_browser_playable(avi_25fps, meta)
    preview = make_browser_preview(avi_25fps, tmp_path / "preview.mp4")
    preview_meta = read_video_metadata(preview)
    assert is_browser_playable(preview, preview_meta)
    # Re-encoding must not change the number of frames or the frame rate.
    assert preview_meta.frame_count == meta.frame_count
    assert preview_meta.fps == pytest.approx(meta.fps)
