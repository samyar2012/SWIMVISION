"""Run the hybrid detector and persistent tracker over one selected swimmer.

The user supplies one visual box on one frame.  Everything else is automatic:
the hybrid detector produces candidates, ``PersistentSwimmerTracker`` keeps
identity through source handoffs and short gaps, and this module writes a
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
from core.swimmer_tracker import HybridDetectorLike, PersistentSwimmerTracker, TrackedSwimmer
from core.video import VideoMetadata, analysis_stride, frame_to_timestamp, open_h264_writer

if TYPE_CHECKING:
    from core.hybrid_detector import HybridSwimmerDetector

logger = logging.getLogger(__name__)
ProgressFn = Callable[[int, int], None]

_TRACKED = (70, 220, 70)
_PREDICTED = (0, 210, 255)
_LOST = (68, 68, 239)


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


def _draw_result(
    frame: np.ndarray,
    result: TrackedSwimmer | None,
    timestamp_s: float,
) -> np.ndarray:
    """Draw a simple observed/predicted/lost overlay without exposing internals."""
    out = frame.copy()
    scale = max(0.6, frame.shape[0] / 720.0)
    thickness = max(2, int(round(3 * scale)))
    if result is None:
        label, colour = "SEARCHING FOR SWIMMER", _LOST
    else:
        label = "PREDICTED THROUGH GAP" if result.predicted else "TRACKED"
        colour = _PREDICTED if result.predicted else _TRACKED
        x1, y1, x2, y2 = result.box.to_int_list()
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)

    cv2.rectangle(out, (10, 10), (420, int(82 * scale)), (18, 18, 18), -1)
    cv2.putText(out, label, (22, int(42 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                0.72 * scale, colour, max(1, int(round(2 * scale))), cv2.LINE_AA)
    cv2.putText(out, f"{timestamp_s:.2f} s", (22, int(70 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55 * scale, (245, 245, 245), max(1, int(round(scale))), cv2.LINE_AA)
    return out


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
) -> tuple[list[FrameResult], TrackingRun]:
    """Track automatically from one visual selection to ``end_frame``."""
    _validate_request(meta, selection, end_frame)
    video_path, annotated_path = Path(video_path), Path(annotated_path)
    aw, ah = fit_size(meta.width, meta.height, analysis_width)
    sx, sy = aw / meta.width, ah / meta.height
    stride = analysis_stride(meta.fps)
    indices = list(range(selection.frame_index, end_frame + 1, stride))
    detector = detector or load_hybrid_detector()
    tracker = PersistentSwimmerTracker(detector)
    selected_box = selection.box.scaled(sx, sy)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoDecodeError("The video could not be opened for tracking.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, selection.frame_index)
    writer = open_h264_writer(annotated_path, aw, ah, meta.fps / stride)
    results: list[FrameResult] = []
    wanted = set(indices)
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
            writer.send(np.ascontiguousarray(_draw_result(frame, tracked, timestamp)))
            if tracked is None:
                result = FrameResult(frame_index, timestamp, None, 0, "")
            else:
                original_box = tracked.box.scaled(1 / sx, 1 / sy).clamp(meta.width, meta.height)
                result = FrameResult(frame_index, timestamp, original_box, 0, tracked.source)
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
        model=f"{config.PERSON_MODEL_PATH.name} + {config.SWIMMER_MODEL_PATH.name}",
        device=device,
        half_precision=False,
        runtime_s=round(time.perf_counter() - started, 2),
        stats=stats,
        output_video=output,
        debug_dir=None,
    )
    logger.info("Hybrid tracker followed %d/%d frames in %.1f s",
                stats.tracked_frames, stats.analyzed_frames, run.runtime_s)
    return results, run
