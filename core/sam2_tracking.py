"""Track ONE user-selected swimmer with SAM 2 (Ultralytics ``SAM2VideoPredictor``).

This is the whole Phase-2 tracking architecture:

1. The user marks the swimmer with a box on one or more frames (``Selection``,
   in original video pixels). One mark on the block is enough until the dive;
   a second mark just after the swimmer surfaces lets tracking continue.
2. ``write_analysis_clip`` copies the frames from the first mark up to the end
   frame into a temporary video, resized to the analysis width and keeping
   every ``stride``-th frame (so at most 30 frames per second are analysed).
3. ``plan_passes``: SAM 2 runs FORWARD from every mark (until the next mark)
   and BACKWARD from every later mark (back to the previous one). Ultralytics'
   ``SAM2VideoPredictor`` applies the box prompt to the first frame it
   processes, so each pass gets its own clip that starts at its mark.
   Why: measured on real race footage, SAM 2 started on the block loses the
   swimmer at water entry and never finds them again (the bigger SAM 2.1-L
   model did the same), while a pass running backward from a mark in the
   water followed them back through the dive to the block.
4. ``choose_box`` merges the two passes covering a frame: if only one has a
   mask it is used; if both do and they agree (overlap), the pass whose mark
   is nearer in time is used; if they disagree the frame is reported as a
   CONFLICT (counted as lost) - we never guess which one is right.
5. Each mask becomes a box (``mask_to_box``) in original video pixels. Frames
   with no mask are LOST - nothing is interpolated. An annotated H.264 video
   and debug JPEGs are written, and the video is reopened to verify it.

There is no fallback tracker, no detector and no automatic re-selection.
"""

from __future__ import annotations

import gc
import logging
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd

import config
from core.errors import ModelLoadError, SwimVisionError, TrackingError, VideoDecodeError
from core.geometry import BBox, fit_size
from core.overlay import draw_tracking_overlay
from core.video import (
    VideoMetadata,
    analysis_stride,
    frame_to_timestamp,
    is_browser_playable,
    open_h264_writer,
    read_video_metadata,
)

logger = logging.getLogger(__name__)

ProgressFn = Callable[[int, int], None]  # (steps done, steps total)
MIN_BOX_SIDE_PX = 4
AGREE_IOU = 0.3  # forward/backward boxes overlapping at least this much = same swimmer
TRACKING_FORMAT_VERSION = 4  # versions 2 and 3 are still readable


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Selection:
    """One mark: a box (original video pixels) around the swimmer on one frame."""

    frame_index: int
    box: BBox

    def to_dict(self) -> dict[str, Any]:
        return {"frame_index": self.frame_index, "box": self.box.to_list()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Selection":
        if "anchor_frame" in data:  # format written by the old Phase-2 code
            return cls(int(data["anchor_frame"]), BBox.from_list(data["bbox"]))
        return cls(int(data["frame_index"]), BBox.from_list(data["box"]))


@dataclass(frozen=True)
class FrameResult:
    """The merged answer for one analysed frame."""

    frame_index: int
    timestamp_s: float
    box: BBox | None   # original video pixels; None = lost
    mask_pixels: int   # mask area in analysis pixels; 0 when lost
    source: str = ""   # e.g. "forward from mark 1", "backward from mark 2"
    conflict: bool = False  # forward and backward passes disagreed -> reported as lost
    predicted: bool = False  # a motion estimate, not a detector/visual observation

    @property
    def tracked(self) -> bool:
        return self.box is not None

    @property
    def observed(self) -> bool:
        return self.box is not None and not self.predicted


@dataclass(frozen=True)
class LostInterval:
    """A run of consecutive analysed frames without a (trusted) mask."""

    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    n_frames: int
    duration_s: float


@dataclass(frozen=True)
class TrackingStats:
    analyzed_frames: int
    tracked_frames: int
    lost_frames: int
    tracked_fraction: float
    longest_lost_frames: int
    longest_lost_s: float
    lost_intervals: tuple[LostInterval, ...]
    conflict_frames: int = 0  # included in lost_frames
    estimated_frames: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["lost_intervals"] = [asdict(i) for i in self.lost_intervals]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrackingStats":
        fields = {k: v for k, v in data.items() if k != "lost_intervals"}
        return cls(**fields, lost_intervals=tuple(LostInterval(**i) for i in data["lost_intervals"]))


@dataclass(frozen=True)
class TrackingRun:
    """Summary of one tracking run (saved with the analysis)."""

    selections: tuple[Selection, ...]
    start_frame: int
    end_frame: int
    stride: int
    analysis_width: int
    analysis_height: int
    model: str
    device: str
    half_precision: bool
    runtime_s: float
    stats: TrackingStats
    output_video: dict[str, Any]  # verified properties of the annotated video
    debug_dir: str | None

    def to_dict(self) -> dict[str, Any]:
        data = {k: getattr(self, k) for k in (
            "start_frame", "end_frame", "stride", "analysis_width", "analysis_height", "model",
            "device", "half_precision", "runtime_s", "output_video", "debug_dir")}
        data.update(version=TRACKING_FORMAT_VERSION, selections=[s.to_dict() for s in self.selections],
                    stats=self.stats.to_dict())
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrackingRun | None":
        """Load a saved run; ``None`` for records written by older tracking code."""
        version = data.get("version")
        if version not in (2, 3, TRACKING_FORMAT_VERSION):
            return None
        marks = data["selections"] if version in (3, TRACKING_FORMAT_VERSION) else [data["selection"]]
        plain = {k: v for k, v in data.items() if k not in ("version", "selection", "selections", "stats")}
        return cls(**plain, selections=tuple(Selection.from_dict(m) for m in marks),
                   stats=TrackingStats.from_dict(data["stats"]))


@dataclass(frozen=True)
class Pass:
    """One SAM 2 run: starts at a mark and walks the clip in one direction."""

    mark: int                    # 0-based mark number
    direction: str               # "forward" or "backward"
    positions: tuple[int, ...]   # positions in the analysis clip, in processing order


@dataclass(frozen=True)
class _PassFrame:
    """A mask found by one pass, in analysis pixels (mask cropped to its box to save memory)."""

    box: BBox
    mask_crop: np.ndarray


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------
def validate_request(meta: VideoMetadata, marks: list[Selection], end_frame: int) -> None:
    """Raise ``TrackingError`` with a clear message if the request cannot work."""
    if not marks:
        raise TrackingError("Mark the swimmer on at least one frame first.")
    frames = [m.frame_index for m in marks]
    if len(set(frames)) != len(frames):
        raise TrackingError("Each frame can only have one mark.")
    for n, mark in enumerate(sorted(marks, key=lambda m: m.frame_index), start=1):
        if not 0 <= mark.frame_index < meta.frame_count:
            raise TrackingError(f"Mark {n}: frame {mark.frame_index} is outside the video "
                                f"(0 to {meta.frame_count - 1}).")
        box = mark.box
        if not box.is_inside(meta.width, meta.height):
            raise TrackingError(f"Mark {n}: the box {box.to_int_list()} is not inside the "
                                f"{meta.width}x{meta.height} frame.")
        if box.width < MIN_BOX_SIDE_PX or box.height < MIN_BOX_SIDE_PX:
            raise TrackingError(f"Mark {n}: the box is too small. Draw it around the whole swimmer.")
    if not max(frames) < end_frame < meta.frame_count:
        raise TrackingError("The end time must be after the last mark and inside the video.")


def analysis_indices(mark_frames: list[int], end_frame: int, stride: int) -> list[int]:
    """Frames to analyse: every ``stride``-th frame from the first mark, plus every mark frame."""
    start = min(mark_frames)
    return sorted(set(range(start, end_frame + 1, stride)) | set(mark_frames))


def plan_passes(mark_positions: list[int], n_positions: int) -> list[Pass]:
    """Forward pass from each mark to just before the next; backward pass from each later mark.

    ``mark_positions`` are the (sorted) positions of the marks in the analysis
    clip; the first must be 0. Every position is covered by one forward pass,
    and positions between two marks also by the later mark's backward pass.
    """
    passes: list[Pass] = []
    for k, pos in enumerate(mark_positions):
        stop = mark_positions[k + 1] - 1 if k + 1 < len(mark_positions) else n_positions - 1
        passes.append(Pass(k, "forward", tuple(range(pos, stop + 1))))
        if k > 0:
            passes.append(Pass(k, "backward", tuple(range(pos, mark_positions[k - 1], -1))))
    return passes


def choose_box(forward: BBox | None, backward: BBox | None, frames_from_forward_mark: int,
               frames_from_backward_mark: int, min_iou: float = AGREE_IOU) -> str:
    """Decide which pass to trust on one frame: ``"forward"``, ``"backward"``, ``"lost"`` or ``"conflict"``."""
    if forward is None and backward is None:
        return "lost"
    if backward is None:
        return "forward"
    if forward is None:
        return "backward"
    if forward.iou(backward) < min_iou:
        return "conflict"
    return "forward" if frames_from_forward_mark <= frames_from_backward_mark else "backward"


def mask_to_box(mask: np.ndarray | None) -> BBox | None:
    """Tight box around all ``True`` pixels of a 2-D mask; ``None`` if the mask is empty.

    The box uses pixel-edge coordinates: a mask covering only pixel (x=3, y=5)
    gives ``BBox(3, 5, 4, 6)``.
    """
    if mask is None or mask.ndim != 2:
        return None
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return BBox(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def compute_stats(results: list[FrameResult], sample_period_s: float) -> TrackingStats:
    """Count tracked / lost frames and find every interval of consecutive lost frames.

    ``sample_period_s`` is the time between analysed frames (stride / fps); a
    lost interval of ``n`` frames lasts ``n * sample_period_s`` seconds.
    """
    intervals: list[LostInterval] = []
    run: list[FrameResult] = []
    for result in [*results, None]:
        if result is not None and not result.tracked:
            run.append(result)
            continue
        if run:
            intervals.append(LostInterval(run[0].frame_index, run[-1].frame_index, run[0].timestamp_s,
                                          run[-1].timestamp_s, len(run), len(run) * sample_period_s))
            run = []
    tracked = sum(r.observed for r in results)
    estimated = sum(r.tracked and r.predicted for r in results)
    lost = sum(not r.tracked for r in results)
    longest = max(intervals, key=lambda i: i.n_frames, default=None)
    return TrackingStats(
        analyzed_frames=len(results), tracked_frames=tracked, lost_frames=lost,
        tracked_fraction=tracked / len(results) if results else 0.0,
        longest_lost_frames=longest.n_frames if longest else 0,
        longest_lost_s=longest.duration_s if longest else 0.0,
        lost_intervals=tuple(intervals), conflict_frames=sum(r.conflict for r in results),
        estimated_frames=estimated,
    )


def results_table(results: list[FrameResult]) -> pd.DataFrame:
    """Per-frame table (original video pixels) for CSV export."""
    rows = []
    for r in results:
        box = r.box
        rows.append({
            "frame_index": r.frame_index, "timestamp_s": round(r.timestamp_s, 4),
            "status": ("estimated" if r.predicted and r.tracked else
                       "tracked" if r.tracked else "conflict" if r.conflict else "lost"),
            "source": r.source,
            "x1": box.x1 if box else None, "y1": box.y1 if box else None,
            "x2": box.x2 if box else None, "y2": box.y2 if box else None,
            "cx": box.center[0] if box else None, "cy": box.center[1] if box else None,
            "mask_pixels": r.mask_pixels,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def ensure_sam2_weights(path: Path | None = None) -> Path:
    """Return the SAM 2 weights file, downloading it first if it is missing."""
    path = Path(path or config.SAM2_MODEL_PATH)
    if path.is_file():
        logger.info("SAM 2 weights: %s (%.0f MB)", path, path.stat().st_size / 1e6)
        return path
    logger.info("SAM 2 weights not found at %s - downloading from Ultralytics…", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from ultralytics.utils.downloads import attempt_download_asset

        got = Path(attempt_download_asset(path))
    except Exception as exc:  # network / GitHub errors: report the exact cause
        raise ModelLoadError(f"Could not download SAM 2 weights '{path.name}': "
                             f"{type(exc).__name__}: {exc}") from exc
    if not got.is_file():
        raise ModelLoadError(f"SAM 2 weights '{path.name}' were not downloaded (expected at {path}).")
    logger.info("Downloaded SAM 2 weights to %s (%.0f MB)", got, got.stat().st_size / 1e6)
    return got


def describe_device() -> tuple[str, bool]:
    """``(description, is_gpu)``, e.g. ``("cuda:0 (NVIDIA GeForce RTX 4070)", True)``."""
    import torch

    if torch.cuda.is_available():
        return f"cuda:0 ({torch.cuda.get_device_name(0)})", True
    return "cpu", False


def _load_predictor(weights: Path, half: bool) -> Any:
    try:
        from ultralytics.models.sam import SAM2VideoPredictor
    except ImportError as exc:
        raise ModelLoadError("Ultralytics is not installed. Run: pip install -r requirements.txt") from exc
    # Ultralytics 8.4 replaced the deprecated `half=True` with `quantize=16` (fp16, GPU only).
    overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=config.SAM2_IMGSZ,
                     model=str(weights), save=False, verbose=False, quantize=16 if half else None)
    return SAM2VideoPredictor(overrides=overrides)


def _first_mask(result: Any) -> np.ndarray | None:
    """Boolean mask of our (single) object from one Ultralytics ``Results``; ``None`` if lost."""
    if result.masks is None or len(result.masks.data) == 0:
        return None
    if len(result.masks.data) > 1:
        logger.warning("SAM 2 returned %d masks for one prompt; using the first", len(result.masks.data))
    mask = result.masks.data[0].cpu().numpy().astype(bool)
    return mask if mask.any() else None


def _empty_cuda_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Clips
# ---------------------------------------------------------------------------
def write_analysis_clip(video_path: Path, indices: list[int], size: tuple[int, int], fps_out: float,
                        clip_path: Path) -> list[int]:
    """Copy the source frames ``indices`` (ascending) into ``clip_path``, resized to ``size``.

    Returns the indices actually written (shorter if the video ends early).
    The clip is Motion-JPEG, so every frame can be read back by position.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoDecodeError("The video could not be opened for tracking.")
    writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"MJPG"), fps_out, size)
    if not writer.isOpened():
        cap.release()
        raise VideoDecodeError("Could not create the temporary analysis clip.")
    wanted, written = set(indices), []
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
        for idx in range(indices[0], indices[-1] + 1):
            ok, frame = cap.read()
            if not ok:
                logger.warning("Video ended at frame %d (expected up to %d)", idx, indices[-1])
                break
            if idx not in wanted:
                continue
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            writer.write(frame)
            written.append(idx)
    finally:
        cap.release()
        writer.release()
    if not written:
        raise VideoDecodeError(f"No frames could be read starting at frame {indices[0]}.")
    return written


def _copy_positions(src: Path, positions: tuple[int, ...], size: tuple[int, int], fps_out: float,
                    dst: Path) -> None:
    """Write the frames at ``positions`` of the clip ``src`` into ``dst``, in that order."""
    cap = cv2.VideoCapture(str(src))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"MJPG"), fps_out, size)
    try:
        previous = None
        for pos in positions:
            if previous is None or pos != previous + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if not ok:
                raise VideoDecodeError(f"Could not read position {pos} of the analysis clip.")
            writer.write(frame)
            previous = pos
    finally:
        cap.release()
        writer.release()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def track_swimmer(video_path: str | Path, meta: VideoMetadata, marks: list[Selection], end_frame: int,
                  analysis_width: int | None, annotated_path: str | Path,
                  debug_dir: str | Path | None = None, progress: ProgressFn | None = None,
                  weights_path: Path | None = None) -> tuple[list[FrameResult], TrackingRun]:
    """Track from the first mark to ``end_frame``; return per-frame results + summary.

    Raises ``TrackingError`` (with SAM 2's exact error message) if anything
    fails, and if SAM 2 finds nothing inside a mark's box on its own frame.
    """
    video_path, annotated_path = Path(video_path), Path(annotated_path)
    marks = sorted(marks, key=lambda m: m.frame_index)
    validate_request(meta, marks, end_frame)
    stride = analysis_stride(meta.fps)
    aw, ah = fit_size(meta.width, meta.height, analysis_width)
    sx, sy = aw / meta.width, ah / meta.height
    weights = ensure_sam2_weights(weights_path)
    device, gpu = describe_device()
    fps_out = meta.fps / stride
    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        for old in debug_dir.glob("frame_*.jpg"):
            old.unlink()

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="swimvision_", ignore_cleanup_errors=True) as tmp:
        main_clip = Path(tmp) / "analysis_clip.avi"
        wanted = analysis_indices([m.frame_index for m in marks], end_frame, stride)
        indices = write_analysis_clip(video_path, wanted, (aw, ah), fps_out, main_clip)
        try:
            mark_positions = [indices.index(m.frame_index) for m in marks]
        except ValueError as exc:
            raise TrackingError("A marked frame could not be read from the video.") from exc
        passes = plan_passes(mark_positions, len(indices))
        logger.info("Tracking frames %d-%d (%d analysed, stride %d) at %dx%d on %s; %d mark(s), %d pass(es)",
                    indices[0], indices[-1], len(indices), stride, aw, ah, device, len(marks), len(passes))

        total = sum(len(p.positions) for p in passes) + len(indices)
        done = [0]

        def tick() -> None:
            done[0] += 1
            if progress:
                progress(done[0], total)

        found: dict[tuple[int, str], dict[int, _PassFrame | None]] = {}
        for p in passes:
            clip = Path(tmp) / f"pass_{p.mark}_{p.direction}.avi"
            _copy_positions(main_clip, p.positions, (aw, ah), fps_out, clip)
            prompt = marks[p.mark].box.scaled(sx, sy)
            found[(p.mark, p.direction)] = _run_pass(clip, p, prompt, (aw, ah), weights, gpu, tick)
            logger.info("Pass %s from mark %d: mask on %d/%d frames", p.direction, p.mark + 1,
                        sum(v is not None for v in found[(p.mark, p.direction)].values()), len(p.positions))

        chosen = _merge(found, mark_positions, len(indices))
        results = [_to_result(indices[pos], meta, chosen[pos], (sx, sy)) for pos in range(len(indices))]
        _render(main_clip, indices, chosen, meta, annotated_path, (aw, ah), fps_out, debug_dir, tick)
    runtime = time.perf_counter() - started

    output = verify_output_video(annotated_path, len(results), (aw, ah))
    stats = compute_stats(results, stride / meta.fps)
    run = TrackingRun(
        selections=tuple(marks), start_frame=indices[0], end_frame=indices[-1], stride=stride,
        analysis_width=aw, analysis_height=ah, model=weights.name, device=device, half_precision=gpu,
        runtime_s=round(runtime, 2), stats=stats, output_video=output,
        debug_dir=str(debug_dir) if debug_dir is not None else None,
    )
    logger.info("Tracked %d/%d frames (%.1f%%), %d conflicts, longest loss %d frames, %.1f s runtime",
                stats.tracked_frames, stats.analyzed_frames, 100 * stats.tracked_fraction,
                stats.conflict_frames, stats.longest_lost_frames, runtime)
    return results, run


def _run_pass(clip: Path, p: Pass, prompt: BBox, size: tuple[int, int], weights: Path, half: bool,
              tick: Callable[[], None]) -> dict[int, _PassFrame | None]:
    """Run SAM 2 over one pass clip; return {clip position: mask or None}."""
    aw, ah = size
    out: dict[int, _PassFrame | None] = {}
    predictor = None
    try:
        predictor = _load_predictor(weights, half)
        stream = predictor(source=str(clip), bboxes=[prompt.to_list()], stream=True)
        for i, res in enumerate(stream):
            if i >= len(p.positions):
                raise TrackingError(f"SAM 2 returned more frames than the clip has ({len(p.positions)}).")
            frame = res.orig_img
            if frame.shape[:2] != (ah, aw):
                raise TrackingError(f"SAM 2 frame is {frame.shape[1]}x{frame.shape[0]}, expected {aw}x{ah}.")
            mask = _first_mask(res)
            if mask is not None and mask.shape != frame.shape[:2]:
                raise TrackingError(f"SAM 2 mask is {mask.shape[1]}x{mask.shape[0]} but the frame is "
                                    f"{frame.shape[1]}x{frame.shape[0]}.")
            if i == 0 and mask is None:
                raise TrackingError(f"SAM 2 found nothing inside the box of mark {p.mark + 1}. "
                                    "Draw the box tightly around the swimmer and try again.")
            box = mask_to_box(mask)
            if box is None:
                out[p.positions[i]] = None
            else:
                x1, y1, x2, y2 = box.to_int_list()
                out[p.positions[i]] = _PassFrame(box, mask[y1:y2, x1:x2].copy())
            tick()
    except SwimVisionError:
        raise
    except Exception as exc:  # report SAM 2 / PyTorch errors with their exact message
        raise TrackingError(f"SAM 2 failed: {type(exc).__name__}: {exc}") from exc
    finally:
        del predictor
        gc.collect()
        _empty_cuda_cache()
    if len(out) != len(p.positions):
        raise TrackingError(f"SAM 2 stopped after {len(out)} of {len(p.positions)} frames.")
    return out


def _merge(found: dict[tuple[int, str], dict[int, _PassFrame | None]], mark_positions: list[int],
           n_positions: int) -> list[tuple[_PassFrame | None, str, bool]]:
    """For every clip position: (chosen mask or None, source text, conflict flag)."""
    chosen: list[tuple[_PassFrame | None, str, bool]] = []
    k = 0
    for pos in range(n_positions):
        while k + 1 < len(mark_positions) and pos >= mark_positions[k + 1]:
            k += 1
        fwd = found[(k, "forward")].get(pos)
        has_next = k + 1 < len(mark_positions) and pos > mark_positions[k]
        back = found[(k + 1, "backward")].get(pos) if has_next else None
        decision = choose_box(fwd.box if fwd else None, back.box if back else None,
                              pos - mark_positions[k],
                              mark_positions[k + 1] - pos if has_next else n_positions)
        if decision == "forward":
            chosen.append((fwd, f"forward from mark {k + 1}", False))
        elif decision == "backward":
            chosen.append((back, f"backward from mark {k + 2}", False))
        else:
            chosen.append((None, "", decision == "conflict"))
    return chosen


def _to_result(frame_index: int, meta: VideoMetadata, choice: tuple[_PassFrame | None, str, bool],
               scale: tuple[float, float]) -> FrameResult:
    found, source, conflict = choice
    t = frame_to_timestamp(frame_index, meta.fps)
    if found is None:
        return FrameResult(frame_index, t, None, 0, "", conflict)
    sx, sy = scale
    box = found.box.scaled(1 / sx, 1 / sy).clamp(meta.width, meta.height)
    return FrameResult(frame_index, t, box, int(found.mask_crop.sum()), source, False)


def _render(main_clip: Path, indices: list[int], chosen: list[tuple[_PassFrame | None, str, bool]],
            meta: VideoMetadata, annotated_path: Path, size: tuple[int, int], fps_out: float,
            debug_dir: Path | None, tick: Callable[[], None]) -> None:
    """Draw the merged result on every analysis frame into the annotated H.264 video."""
    aw, ah = size
    writer = open_h264_writer(annotated_path, aw, ah, fps_out)
    cap = cv2.VideoCapture(str(main_clip))
    try:
        for pos, frame_index in enumerate(indices):
            ok, frame = cap.read()
            if not ok:
                raise VideoDecodeError(f"Could not re-read analysis frame {frame_index} for the video.")
            found, _, conflict = chosen[pos]
            mask = None
            if found is not None:
                mask = np.zeros((ah, aw), dtype=bool)
                x1, y1, x2, y2 = found.box.to_int_list()
                mask[y1:y2, x1:x2] = found.mask_crop
            t = frame_to_timestamp(frame_index, meta.fps)
            drawn = draw_tracking_overlay(frame, mask, found.box if found else None, t, frame_index,
                                          lost_label="LOST (passes disagree)" if conflict else "LOST")
            writer.send(np.ascontiguousarray(drawn))
            if debug_dir is not None and pos % config.DEBUG_FRAME_EVERY == 0:
                cv2.imwrite(str(debug_dir / f"frame_{pos:03d}.jpg"), drawn)
            tick()
    finally:
        cap.release()
        writer.close()


def verify_output_video(path: Path, expected_frames: int, size: tuple[int, int]) -> dict[str, Any]:
    """Reopen the annotated video and check its frame count, size and codec."""
    meta = read_video_metadata(path)
    if meta.frame_count != expected_frames or (meta.width, meta.height) != size:
        raise TrackingError(f"Annotated video check failed: {meta.frame_count} frames at "
                            f"{meta.width}x{meta.height}, expected {expected_frames} at {size[0]}x{size[1]}.")
    if not is_browser_playable(path, meta):
        raise TrackingError(f"Annotated video codec '{meta.codec}' is not browser-playable H.264.")
    return {"frames": meta.frame_count, "fps": round(meta.fps, 3), "duration_s": round(meta.duration_s, 3),
            "width": meta.width, "height": meta.height, "codec": meta.codec}
