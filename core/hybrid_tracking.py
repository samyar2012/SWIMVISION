"""Run the hybrid detector and persistent tracker over one selected swimmer.

The user supplies one visual box on one frame.  Everything else is automatic:
the hybrid detector produces candidates, ``PersistentVisualSwimmerTracker`` keeps
the selected visual identity through source handoffs and short gaps, and this module writes a
browser-playable annotated video plus the normal per-frame result table.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import cv2
import numpy as np

import config
from core.errors import ModelLoadError, SwimVisionError, TrackingError, VideoDecodeError
from core.geometry import BBox, fit_size
from core.sam2_tracking import (
    FrameResult,
    Selection,
    TrackingRun,
    compute_stats,
    verify_output_video,
)
from core.swimmer_tracker import HybridDetectorLike, PersistentVisualSwimmerTracker, TrackedSwimmer
from core.video import VideoMetadata, analysis_stride, frame_to_timestamp, open_h264_writer

if TYPE_CHECKING:
    from core.hybrid_detector import HybridSwimmerDetector

logger = logging.getLogger(__name__)
ProgressFn = Callable[[int, int], None]
PreviewFn = Callable[[np.ndarray], None]

_HIGHLIGHT = (40, 225, 255)  # BGR: warm yellow stands out against pool water


def load_hybrid_detector() -> HybridSwimmerDetector:
    """Load the two configured detection models with clear user-facing errors."""
    from core.hybrid_detector import HybridSwimmerDetector

    if not config.SWIMMER_MODEL_PATH.is_file():
        raise ModelLoadError(f"Swimmer model is missing: {config.SWIMMER_MODEL_PATH}")
    try:
        return HybridSwimmerDetector(
            person_model_path=str(config.PERSON_MODEL_PATH),
            swimmer_model_path=str(config.SWIMMER_MODEL_PATH),
            person_confidence=config.PERSON_CONFIDENCE,
            swimmer_threshold=config.SWIMMER_THRESHOLD,
            merge_iou_threshold=config.MERGE_IOU_THRESHOLD,
        )
    except SwimVisionError:
        raise
    except Exception as exc:
        raise ModelLoadError(f"Could not load the tracking models: {type(exc).__name__}: {exc}") from exc


class SwimmerHighlight:
    """Draw a steady highlight around the followed swimmer, with no text.

    The box is smoothed over a few frames so the highlight glides with the
    swimmer instead of jittering, and nothing is drawn while the swimmer is
    not being followed.
    """

    def __init__(self, fps: float) -> None:
        self._box: np.ndarray | None = None
        self._missing = 0
        self._reset_after = max(1, int(round(0.5 * fps)))

    def draw(self, frame: np.ndarray, result: TrackedSwimmer | None) -> np.ndarray:
        out = frame.copy()
        if result is None:
            self._missing += 1
            if self._missing >= self._reset_after:
                self._box = None
            return out
        self._missing = 0
        target = np.array(result.box.to_list(), dtype=float)
        if self._box is None or _jumped(self._box, target):
            self._box = target
        else:
            self._box = 0.55 * target + 0.45 * self._box
        _draw_highlight(out, self._box, faded=result.predicted)
        return out


def _jumped(previous: np.ndarray, target: np.ndarray) -> bool:
    size = max(previous[2] - previous[0], previous[3] - previous[1], 1.0)
    centre_shift = np.hypot((target[0] + target[2] - previous[0] - previous[2]) / 2,
                            (target[1] + target[3] - previous[1] - previous[3]) / 2)
    return centre_shift > 1.5 * size


def _draw_highlight(out: np.ndarray, box: np.ndarray, faded: bool) -> None:
    height, width = out.shape[:2]
    scale = max(0.6, height / 720.0)
    pad = 6 * scale
    x1, y1 = int(max(0, box[0] - pad)), int(max(0, box[1] - pad))
    x2, y2 = int(min(width - 1, box[2] + pad)), int(min(height - 1, box[3] + pad))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return
    radius = int(min(14 * scale, (x2 - x1) / 4, (y2 - y1) / 4))
    thickness = max(3, int(round(4 * scale)))
    strength = 0.55 if faded else 1.0

    tint = out.copy()
    cv2.rectangle(tint, (x1, y1), (x2, y2), _HIGHLIGHT, -1)
    region = (slice(y1, y2 + 1), slice(x1, x2 + 1))
    out[region] = cv2.addWeighted(tint[region], 0.22 * strength, out[region], 1 - 0.22 * strength, 0)

    margin = int(16 * scale)
    gx1, gy1 = max(0, x1 - margin), max(0, y1 - margin)
    gx2, gy2 = min(width, x2 + margin + 1), min(height, y2 + margin + 1)
    glow = np.zeros((gy2 - gy1, gx2 - gx1, 3), dtype=out.dtype)
    _rounded_rectangle(glow, x1 - gx1, y1 - gy1, x2 - gx1, y2 - gy1, radius, _HIGHLIGHT, thickness * 3)
    glow = cv2.GaussianBlur(glow, (0, 0), 4 * scale)
    area = out[gy1:gy2, gx1:gx2]
    cv2.addWeighted(area, 1.0, glow, 0.75 * strength, 0, dst=area)
    line = tuple(int(c * strength + 255 * (1 - strength) * 0.4) for c in _HIGHLIGHT)
    _rounded_rectangle(out, x1, y1, x2, y2, radius, line, thickness)


def _rounded_rectangle(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, radius: int,
                       colour: tuple[int, ...], thickness: int) -> None:
    if radius < 2:
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)
        return
    cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), colour, thickness, cv2.LINE_AA)
    cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), colour, thickness, cv2.LINE_AA)
    cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), colour, thickness, cv2.LINE_AA)
    cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), colour, thickness, cv2.LINE_AA)
    for centre, start in (((x1 + radius, y1 + radius), 180), ((x2 - radius, y1 + radius), 270),
                          ((x2 - radius, y2 - radius), 0), ((x1 + radius, y2 - radius), 90)):
        cv2.ellipse(image, centre, (radius, radius), start, 0, 90, colour, thickness, cv2.LINE_AA)


def _validate_request(meta: VideoMetadata, selection: Selection, end_frame: int) -> None:
    if not 0 <= selection.frame_index < meta.frame_count:
        raise TrackingError("The selected moment is outside the video.")
    if not selection.box.is_inside(meta.width, meta.height):
        raise TrackingError("The swimmer selection is outside the video frame.")
    if selection.box.width < 4 or selection.box.height < 4:
        raise TrackingError("Draw a larger box around the swimmer.")
    if not selection.frame_index < end_frame < meta.frame_count:
        raise TrackingError("The video must continue after the selected moment.")


def track_swimmer(
    video_path: str | Path,
    meta: VideoMetadata,
    selection: Selection,
    end_frame: int,
    annotated_path: str | Path,
    *,
    analysis_width: int | None = None,
    progress: ProgressFn | None = None,
    detector: HybridDetectorLike | None = None,
    preview: PreviewFn | None = None,
) -> tuple[list[FrameResult], TrackingRun]:
    """Track automatically from one visual selection to ``end_frame``.

    ``preview`` receives every highlighted frame as it is written, so a client
    can watch the swimmer being followed while tracking runs.
    """
    _validate_request(meta, selection, end_frame)
    video_path, annotated_path = Path(video_path), Path(annotated_path)
    aw, ah = fit_size(meta.width, meta.height, analysis_width)
    sx, sy = aw / meta.width, ah / meta.height
    stride = analysis_stride(meta.fps)
    indices = list(range(selection.frame_index, end_frame + 1, stride))
    detector = detector or load_hybrid_detector()
    if not config.VIT_TRACKER_MODEL_PATH.is_file():
        raise ModelLoadError(f"Visual tracker model is missing: {config.VIT_TRACKER_MODEL_PATH}")
    tracker = PersistentVisualSwimmerTracker(config.VIT_TRACKER_MODEL_PATH)
    selected_box = selection.box.scaled(sx, sy)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoDecodeError("The video could not be opened for tracking.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, selection.frame_index)
    writer = open_h264_writer(annotated_path, aw, ah, meta.fps / stride)
    results: list[FrameResult] = []
    wanted = set(indices)
    highlight = SwimmerHighlight(meta.fps / stride)
    started = time.perf_counter()
    try:
        for frame_index in range(selection.frame_index, end_frame + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                raise VideoDecodeError(f"The video stopped unexpectedly at frame {frame_index}.")
            if frame_index not in wanted:
                continue
            if (frame.shape[1], frame.shape[0]) != (aw, ah):
                frame = cv2.resize(frame, (aw, ah), interpolation=cv2.INTER_AREA)
            try:
                candidates = detector.detect(frame)
                if not tracker.initialized:
                    tracked = (
                        tracker.start(candidates, (ah, aw), target_box=selected_box, frame=frame)
                        if candidates
                        else tracker.start_from_box(selected_box, (ah, aw), frame=frame)
                    )
                else:
                    tracked = tracker.update(candidates, (ah, aw), frame=frame)
            except Exception as exc:
                raise TrackingError(
                    f"Tracking failed at {frame_to_timestamp(frame_index, meta.fps):.2f} seconds: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            timestamp = frame_to_timestamp(frame_index, meta.fps)
            annotated = np.ascontiguousarray(highlight.draw(frame, tracked))
            writer.send(annotated)
            if preview:
                preview(annotated)
            if tracked is None:
                result = FrameResult(frame_index, timestamp, None, 0, "")
            else:
                original_box = tracked.box.scaled(1 / sx, 1 / sy).clamp(meta.width, meta.height)
                result = FrameResult(
                    frame_index, timestamp, original_box, 0, tracked.source,
                    predicted=tracked.predicted,
                )
            results.append(result)
            if progress:
                progress(len(results), len(indices))
    finally:
        cap.release()
        writer.close()

    if len(results) != len(indices):
        raise TrackingError(f"Only {len(results)} of {len(indices)} frames were processed.")
    output = verify_output_video(annotated_path, len(results), (aw, ah))
    stats = compute_stats(results, stride / meta.fps)
    try:
        import torch

        device = f"cuda:0 ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    run = TrackingRun(
        selections=(selection,),
        start_frame=indices[0],
        end_frame=indices[-1],
        stride=stride,
        analysis_width=aw,
        analysis_height=ah,
        model=(f"{config.VIT_TRACKER_MODEL_PATH.name} + {config.PERSON_MODEL_PATH.name} + "
               f"{config.SWIMMER_MODEL_PATH.name}"),
        device=device,
        half_precision=False,
        runtime_s=round(time.perf_counter() - started, 2),
        stats=stats,
        output_video=output,
        debug_dir=None,
    )
    logger.info("Hybrid tracker observed %d/%d frames, estimated %d and lost %d in %.1f s",
                stats.tracked_frames, stats.analyzed_frames, stats.estimated_frames,
                stats.lost_frames, run.runtime_s)
    return results, run
