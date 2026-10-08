"""Shared test helpers: build tiny videos whose true properties we control."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pytest


def write_test_video(path: Path, fps: float, n_frames: int, size=(320, 240), fourcc="mp4v") -> Path:
    """Write ``n_frames`` frames whose brightness encodes the frame number."""
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
    assert writer.isOpened(), f"OpenCV could not open a writer for {fourcc}"
    for i in range(n_frames):
        frame = np.full((height, width, 3), i * 2 % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def write_ffmpeg_video(path: Path, rate: str, n_frames: int, size="320x240") -> Path:
    """Use the bundled FFmpeg to make an H.264 clip at an exact (fractional) rate."""
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "lavfi",
        "-i", f"testsrc=size={size}:rate={rate}", "-frames:v", str(n_frames),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return path


@pytest.fixture
def mp4_30fps(tmp_path: Path) -> Path:
    return write_test_video(tmp_path / "clip.mp4", fps=30.0, n_frames=90)


@pytest.fixture
def avi_25fps(tmp_path: Path) -> Path:
    return write_test_video(tmp_path / "clip.avi", fps=25.0, n_frames=50, fourcc="MJPG")


@pytest.fixture
def mov_2997fps(tmp_path: Path) -> Path:
    return write_ffmpeg_video(tmp_path / "clip.mov", rate="30000/1001", n_frames=60)
