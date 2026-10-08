"""Video input: validation, metadata extraction, frame access, browser previews.

Design rule for the whole project: **all timing comes from frame index and the
video's real FPS**, never from wall-clock timers. The two tiny helpers
``frame_to_timestamp`` and ``timestamp_to_frame`` are the single source of
truth for converting between frames and seconds.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config
from core.errors import UnsupportedVideoError, VideoDecodeError

# Codecs (OpenCV FOURCC strings, lower-case) that browsers play natively.
_BROWSER_SAFE_CODECS = {"avc1", "h264", "x264"}
_BROWSER_SAFE_EXTENSIONS = {".mp4", ".m4v"}


# ---------------------------------------------------------------------------
# Pure timing math (unit-tested, no I/O)
# ---------------------------------------------------------------------------
def frame_to_timestamp(frame_index: int, fps: float) -> float:
    """Return the time in seconds at which ``frame_index`` is shown.

    Frame 0 is at t = 0.0 s. Example: at 30 fps, frame 45 -> 1.5 s.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if frame_index < 0:
        raise ValueError("frame_index must be >= 0")
    return frame_index / fps


def timestamp_to_frame(seconds: float, fps: float) -> int:
    """Return the frame index nearest to ``seconds`` (inverse of the above)."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if seconds < 0:
        raise ValueError("seconds must be >= 0")
    return int(round(seconds * fps))


def compute_duration(frame_count: int, fps: float) -> float:
    """Total video duration in seconds = frame_count / fps."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if frame_count < 0:
        raise ValueError("frame_count must be >= 0")
    return frame_count / fps


def analysis_stride(fps: float, target_fps: float | None = None) -> int:
    """How many source frames to step between analysed frames.

    We analyse at most ``target_fps`` (default 30) frames per second: a 60 fps
    video is analysed every 2nd frame, a 30 fps video every frame. Real
    timestamps are always kept (``frame_to_timestamp``), so skipping frames
    never changes the timing.
    """
    target = target_fps or config.TARGET_ANALYSIS_FPS
    if fps <= 0:
        raise ValueError("fps must be positive")
    return max(1, int(round(fps / target)))


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VideoMetadata:
    """Facts about a video file, all measured from the file itself."""

    fps: float
    frame_count: int
    duration_s: float
    width: int
    height: int
    codec: str  # FOURCC code reported by the decoder, e.g. "avc1" or "unknown"

    @property
    def resolution_label(self) -> str:
        return f"{self.width} × {self.height}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoMetadata":
        return cls(
            fps=float(data["fps"]),
            frame_count=int(data["frame_count"]),
            duration_s=float(data["duration_s"]),
            width=int(data["width"]),
            height=int(data["height"]),
            codec=str(data.get("codec", "unknown")),
        )


def validate_extension(path: str | Path) -> None:
    """Raise ``UnsupportedVideoError`` if the file extension is not accepted."""
    ext = Path(path).suffix.lower().lstrip(".")
    if ext not in config.SUPPORTED_VIDEO_EXTENSIONS:
        allowed = ", ".join(e.upper() for e in config.SUPPORTED_VIDEO_EXTENSIONS)
        raise UnsupportedVideoError(
            f"'.{ext}' files are not supported. Please upload one of: {allowed}."
        )


def _fourcc_to_str(code: float) -> str:
    """Turn OpenCV's numeric FOURCC property into text such as ``'avc1'``."""
    value = int(code)
    if value <= 0:
        return "unknown"
    chars = "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))
    # Keep only printable characters; otherwise report unknown.
    return chars.strip().lower() if chars.isprintable() and chars.strip() else "unknown"


def _count_frames_by_decoding(cap: cv2.VideoCapture) -> int:
    """Count frames by stepping through the whole file (slow but exact).

    Used only when the container's frame-count header is missing or wrong,
    which happens with some AVI files and variable-frame-rate phone videos.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    count = 0
    while cap.grab():
        count += 1
    return count


def _last_frame_is_readable(cap: cv2.VideoCapture, frame_count: int) -> bool:
    """Check that the header's frame count is real by reading its last frame."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    ok, _ = cap.read()
    return bool(ok)


def read_video_metadata(path: str | Path) -> VideoMetadata:
    """Open ``path`` and measure FPS, frame count, duration and resolution.

    Raises:
        UnsupportedVideoError: wrong file type or missing file.
        VideoDecodeError: OpenCV cannot open / decode the video.
    """
    path = Path(path)
    if not path.is_file():
        raise UnsupportedVideoError("The video file could not be found.")
    validate_extension(path)

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoDecodeError(
                "This video could not be opened. It may be corrupted or use a codec "
                "that OpenCV cannot decode. Try re-exporting it as H.264 MP4."
            )

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise VideoDecodeError(
                "Could not determine this video's frame rate (FPS). SwimVision needs "
                "a real FPS to compute accurate times. Try re-exporting the video."
            )

        # Read the first frame: it proves the video decodes, and its shape gives
        # the *displayed* width/height (correct even if the phone rotated it).
        ok, first_frame = cap.read()
        if not ok or first_frame is None:
            raise VideoDecodeError(
                "The video opened, but no frames could be decoded. "
                "Try re-exporting it as H.264 MP4."
            )
        height, width = first_frame.shape[:2]

        # Frame count: trust the header only if its last frame can really be read.
        header_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if header_count > 0 and _last_frame_is_readable(cap, header_count):
            frame_count = header_count
        else:
            frame_count = _count_frames_by_decoding(cap)
        if frame_count <= 0:
            raise VideoDecodeError("The video contains no readable frames.")

        return VideoMetadata(
            fps=fps,
            frame_count=frame_count,
            duration_s=compute_duration(frame_count, fps),
            width=int(width),
            height=int(height),
            codec=_fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)),
        )
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Frame access
# ---------------------------------------------------------------------------
def read_frame(path: str | Path, frame_index: int) -> np.ndarray:
    """Return one frame (BGR ``numpy`` array) by index.

    Raises ``VideoDecodeError`` if the frame cannot be read.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoDecodeError("The video could not be opened to read a frame.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoDecodeError(f"Frame {frame_index} could not be read from the video.")
        return frame
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Browser-friendly preview
# ---------------------------------------------------------------------------
def is_browser_playable(path: str | Path, metadata: VideoMetadata) -> bool:
    """True if the original file can probably be played by a web browser.

    Browsers reliably play H.264 in an MP4 container. MOV (often HEVC from
    iPhones) and AVI usually do not play, so we make a preview copy for those.
    """
    return (
        Path(path).suffix.lower() in _BROWSER_SAFE_EXTENSIONS
        and metadata.codec in _BROWSER_SAFE_CODECS
    )


def build_preview_command(ffmpeg_exe: str, src: Path, dst: Path) -> list[str]:
    """Build the FFmpeg command that makes a small H.264 MP4 preview copy."""
    max_h = config.PREVIEW_MAX_HEIGHT
    return [
        ffmpeg_exe, "-y", "-i", str(src),
        # Downscale only if taller than max_h; "-2" keeps width even (H.264 needs it).
        "-vf", f"scale=-2:'min({max_h},ih)'",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-an",  # analysis does not need audio
        str(dst),
    ]


def make_browser_preview(src: str | Path, dst: str | Path, timeout_s: int = 600) -> Path:
    """Re-encode ``src`` to an H.264 MP4 at ``dst`` using a bundled FFmpeg.

    The *original* file is always what gets analysed; this copy is only for
    on-screen playback. Uses the FFmpeg binary shipped by ``imageio-ffmpeg`` so
    no system-wide FFmpeg install is required.
    """
    import imageio_ffmpeg  # imported lazily: only needed for non-MP4 uploads

    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    command = build_preview_command(imageio_ffmpeg.get_ffmpeg_exe(), src, dst)
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise VideoDecodeError("Creating the preview video took too long.") from exc
    if result.returncode != 0 or not dst.is_file():
        tail = (result.stderr or "").strip().splitlines()[-1:]  # last log line only
        raise VideoDecodeError(
            "Could not convert this video into a playable preview. "
            f"FFmpeg said: {tail[0] if tail else 'unknown error'}"
        )
    return dst
