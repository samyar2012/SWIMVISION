"""Drawing on frames: candidate boxes, tracking overlay, annotated video export.

Drawing helpers take and return BGR ``numpy`` images (OpenCV's format) and never
modify the caller's array. Everything drawn comes from real data - lost frames
show a "TRACKING LOST" banner, not a guessed box.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import pandas as pd

import config
from core.detector import PersonDetection
from core.errors import VideoDecodeError
from core.geometry import BBox
from core.tracker import (
    STATUS_LOST,
    STATUS_LOW_CONFIDENCE,
    STATUS_NOT_MOVING,
    STATUS_REACQUIRED,
    STATUS_TRACKED,
)
from core.video import VideoMetadata

logger = logging.getLogger(__name__)

# Colours in BGR order (OpenCV), chosen to match the app's cyan accent.
COLOR_ACCENT = (238, 211, 34)
COLOR_OTHER = (170, 150, 120)
COLOR_WARN = (36, 191, 251)
COLOR_REACQUIRED = (200, 120, 255)
COLOR_LOST = (68, 68, 239)
COLOR_TEXT_BG = (20, 14, 8)
TRAIL_SECONDS = 3.0   # how much recent movement the drawn path shows
STATUS_COLORS = {STATUS_TRACKED: COLOR_ACCENT, STATUS_LOW_CONFIDENCE: COLOR_WARN,
                 STATUS_REACQUIRED: COLOR_REACQUIRED, STATUS_LOST: COLOR_LOST,
                 STATUS_NOT_MOVING: COLOR_WARN}
# Short banners burned into the video (the app explains each one in full).
STATUS_LABELS = {STATUS_TRACKED: "FOLLOWING", STATUS_LOW_CONFIDENCE: "WEAK MATCH",
                 STATUS_REACQUIRED: "FOUND AGAIN - CHECK IT IS YOUR SWIMMER",
                 STATUS_LOST: "SWIMMER LOST",
                 STATUS_NOT_MOVING: "NOT MOVING - ON THE BLOCK, OR WRONG PERSON?"}


def _scale(frame: np.ndarray) -> float:
    """Drawing scale so lines/text look the same on 480p and 4K video."""
    return max(0.5, frame.shape[0] / 720.0)


def _put_label(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int],
               scale: float) -> None:
    """Draw text on a dark box so it is readable on any background."""
    font, thickness = cv2.FONT_HERSHEY_SIMPLEX, max(1, int(round(scale * 1.5)))
    (tw, th), base = cv2.getTextSize(text, font, 0.55 * scale, thickness)
    x, y = int(org[0]), int(org[1])
    x = max(0, min(x, img.shape[1] - tw - 6))
    y = max(th + 6, min(y, img.shape[0] - 4))
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + base), COLOR_TEXT_BG, -1)
    cv2.putText(img, text, (x + 3, y - 3), font, 0.55 * scale, color, thickness, cv2.LINE_AA)


def _rect(img: np.ndarray, box: BBox, color: tuple[int, int, int], thickness: int) -> None:
    cv2.rectangle(img, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Swimmer selection screen
# ---------------------------------------------------------------------------
def draw_person_candidates(frame: np.ndarray, detections: Sequence[PersonDetection],
                           selected_number: int | None = None) -> np.ndarray:
    """Draw numbered boxes over every detected person; highlight the chosen one."""
    out, s = frame.copy(), _scale(frame)
    for det in detections:
        chosen = det.number == selected_number
        color = COLOR_ACCENT if chosen else COLOR_OTHER
        _rect(out, det.bbox, color, max(2, int(round(s * (4 if chosen else 2)))))
        _put_label(out, f"{det.number}  {det.confidence:.0%}", (det.bbox.x1, det.bbox.y1), color, s)
    return out


def draw_manual_box(frame: np.ndarray, box: BBox) -> np.ndarray:
    """Preview of the box the user is drawing by hand."""
    out, s = frame.copy(), _scale(frame)
    _rect(out, box, COLOR_ACCENT, max(2, int(round(s * 3))))
    _put_label(out, "manual box", (box.x1, box.y1), COLOR_ACCENT, s)
    return out


# ---------------------------------------------------------------------------
# Tracking overlay
# ---------------------------------------------------------------------------
def draw_tracking_frame(frame: np.ndarray, row: pd.Series, trail: Sequence[Sequence[tuple[int, int]]] = (),
                        ) -> np.ndarray:
    """Draw the tracked swimmer (box, ID, time, confidence, path) on one frame.

    ``row`` is one line of the tracking table. ``trail`` is a list of point
    lists (one list per unbroken stretch of tracking) forming the swimmer's path.
    """
    out, s = frame.copy(), _scale(frame)
    status = str(row["status"])
    color = STATUS_COLORS.get(status, COLOR_OTHER)

    for stretch in trail:  # path of the swimmer's centre, broken wherever tracking was lost
        if len(stretch) > 1:
            cv2.polylines(out, [np.array(stretch, dtype=np.int32)], False, COLOR_ACCENT,
                          max(1, int(round(s * 2))), cv2.LINE_AA)

    if status != STATUS_LOST:
        box = BBox(row["x1"], row["y1"], row["x2"], row["y2"])
        _rect(out, box, color, max(2, int(round(s * 3))))
        if pd.notna(row["confidence"]):  # only the YOLO / visual trackers report a score
            _put_label(out, f"match {row['confidence']:.0%}", (box.x1, box.y1), color, s)

    _put_label(out, f"t = {row['timestamp_s']:.2f} s   frame {int(row['frame_index'])}", (10, 10 + 26 * s),
               (255, 255, 255), s)
    if status != STATUS_TRACKED:
        _put_label(out, STATUS_LABELS.get(status, status.upper()), (10, 10 + 56 * s), color, s)
    return out


def _output_size(width: int, height: int) -> tuple[int, int]:
    """Downscale (if needed) to the annotated-video height cap; dimensions stay even."""
    factor = min(1.0, config.ANNOTATED_MAX_HEIGHT / height)
    return (int(round(width * factor / 2)) * 2, int(round(height * factor / 2)) * 2)


def write_annotated_video(video_path: str | Path, meta: VideoMetadata, tracking: pd.DataFrame,
                          out_path: str | Path, stride: int, show_trail: bool = True,
                          progress: Callable[[float, str], None] | None = None) -> Path:
    """Render the tracking overlay into an H.264 MP4 playable in the browser.

    Only analysed frames are written (every ``stride``-th source frame). The
    output frame rate is ``fps / stride``, so playback speed matches real time.
    """
    import imageio_ffmpeg

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_w, out_h = _output_size(meta.width, meta.height)
    rows = {int(r["frame_index"]): r for _, r in tracking.iterrows()}
    writer = imageio_ffmpeg.write_frames(
        str(out_path), (out_w, out_h), pix_fmt_in="bgr24", fps=meta.fps / stride, codec="libx264",
        quality=None, macro_block_size=2,
        output_params=["-crf", "22", "-preset", "veryfast", "-movflags", "+faststart"])
    writer.send(None)  # start the FFmpeg process

    cap = cv2.VideoCapture(str(video_path))
    trail: list[list[tuple[int, int]]] = [[]]
    max_trail_points = max(2, int(round(TRAIL_SECONDS * meta.fps / stride)))
    try:
        idx, written = 0, 0
        while cap.grab():
            if idx in rows:
                ok, frame = cap.retrieve()
                if not ok:
                    raise VideoDecodeError(f"Frame {idx} could not be decoded while rendering.")
                row = rows[idx]
                _update_trail(trail, row, show_trail, max_trail_points)
                drawn = draw_tracking_frame(frame, row, trail if show_trail else ())
                if (out_w, out_h) != (drawn.shape[1], drawn.shape[0]):
                    drawn = cv2.resize(drawn, (out_w, out_h), interpolation=cv2.INTER_AREA)
                writer.send(np.ascontiguousarray(drawn))
                written += 1
                if progress and written % 10 == 0:
                    progress(written / len(rows), "Rendering annotated video…")
            idx += 1
    finally:
        cap.release()
        writer.close()
    if not out_path.is_file():
        raise VideoDecodeError("The annotated video could not be created.")
    return out_path


def _update_trail(trail: list[list[tuple[int, int]]], row: pd.Series, enabled: bool,
                  max_points: int) -> None:
    """Append the swimmer's centre to the path; start a new stretch after a loss.

    Only the newest ``max_points`` points are kept so the path shows recent
    movement instead of an unreadable scribble over the whole race.
    """
    if not enabled:
        return
    if str(row["status"]) == STATUS_LOST:
        if trail[-1]:
            trail.append([])
        return
    trail[-1].append((int(row["cx"]), int(row["cy"])))
    del trail[-1][:-max_points]
    del trail[:-2]  # an older, finished stretch is no longer 'recent'


def zoomed_crop(frame: np.ndarray, box: BBox, box_color: tuple[int, int, int] = (0, 200, 255),
                margin: float = 1.0, target_height: int = 260) -> np.ndarray:
    """A magnified view around ``box`` so the user can check a tiny selection.

    ``margin`` is the extra context added on every side, as a multiple of the
    box size. The box outline is drawn on the enlarged crop.
    """
    h, w = frame.shape[:2]
    pad_x, pad_y = max(box.width * margin, 20), max(box.height * margin, 20)
    region = BBox(box.x1 - pad_x, box.y1 - pad_y, box.x2 + pad_x, box.y2 + pad_y).clamp(w, h)
    x1, y1, x2, y2 = (int(round(v)) for v in (region.x1, region.y1, region.x2, region.y2))
    crop = frame[y1:y2, x1:x2]
    # Enlarge to roughly target_height, but never beyond 700 px wide (and never shrink).
    zoom = min(target_height / max(crop.shape[0], 1), 700 / max(crop.shape[1], 1))
    zoom = max(zoom, 1.0)
    big = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    p1 = (int(round((box.x1 - x1) * zoom)), int(round((box.y1 - y1) * zoom)))
    p2 = (int(round((box.x2 - x1) * zoom)), int(round((box.y2 - y1) * zoom)))
    cv2.rectangle(big, p1, p2, box_color, 2, cv2.LINE_AA)
    return big
