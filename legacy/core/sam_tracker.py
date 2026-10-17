"""Swimmer tracking with SAM 2 (Segment Anything 2) from user checkpoints.

Why SAM 2
---------
Generic box trackers match one fixed template, so they fail when the swimmer
changes shape (standing -> diving -> swimming) or the handheld camera zooms.
SAM 2 *segments* the swimmer (the outline of their body, not just a box) and
keeps a memory of how the swimmer looked in many earlier frames, so it copes
with much bigger changes in appearance. Ultralytics ships it
(``SAM2VideoPredictor``); weights are downloaded once into ``models/``.

Why checkpoints
---------------
After the dive and after every turn the swimmer is underwater - nothing in
the picture shows where they are. No tracker can follow what cannot be seen,
and while the swimmer is hidden a tracker may latch onto a spectator who looks
similar to the swimmer on the block (we observed exactly that on real race
footage). So the user may add *checkpoints*: a box on the swimmer at extra
frames (e.g. after the breakout, after each turn). Each checkpoint starts a
fresh SAM 2 track that runs until the next checkpoint or a camera cut.

Safety checks on SAM 2's output
-------------------------------
* No mask => "lost" (no coordinates are invented).
* The first frames after a gap are flagged "reacquired" - check it is the same swimmer.
* Camera-motion-compensated movement check (``core.camera_motion``): a target
  that does not move through the scene is flagged "not moving".
"""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

import config
from core.camera_motion import estimate_camera_motion, not_moving_flags, relative_steps, to_motion_gray
from core.errors import ModelLoadError, TrackingError
from core.geometry import BBox
from core.scene import frame_signature, is_scene_cut

logger = logging.getLogger(__name__)

MAX_MASK_FRAME_FRACTION = 0.35  # a "swimmer" covering more of the picture than this is clutter
REACQUIRE_GAP = 3               # frames missing before a re-found target is flagged
REACQUIRE_FLAG_S = 1.0          # how long after re-finding the target the flag lasts
NOT_MOVING_WINDOW_S = 2.0       # time window for the "is it swimming?" check


@dataclass(frozen=True)
class SegmentPlan:
    """One SAM 2 run: starts at a checkpoint and walks through ``positions`` in order."""

    checkpoint_index: int
    positions: list[int]   # analysed-frame positions; positions[0] is the checkpoint itself


@dataclass(frozen=True)
class SamFrame:
    """SAM 2's answer for one analysed frame."""

    bbox: BBox | None
    from_checkpoint: bool = False   # True on the checkpoint frame itself (the user's own box)
    segment: int = -1               # which SegmentPlan produced it


# ---------------------------------------------------------------------------
# Pure logic (unit-tested)
# ---------------------------------------------------------------------------
def plan_segments(checkpoint_positions: Sequence[int], cuts: Sequence[bool], n: int) -> list[SegmentPlan]:
    """Decide which analysed frames each checkpoint is responsible for.

    * Each checkpoint tracks **forward** until the next checkpoint or the next
      camera cut (``cuts[i]`` = sample ``i`` starts a new shot).
    * The **earliest** checkpoint also tracks **backward** to the start of its
      shot, so frames before it are covered too.
    Checkpoints must be distinct, sorted positions.
    """
    plans: list[SegmentPlan] = []
    ordered = sorted(checkpoint_positions)
    if list(ordered) != list(checkpoint_positions) or len(set(ordered)) != len(ordered):
        raise ValueError("checkpoint positions must be distinct and sorted")
    for k, pos in enumerate(ordered):
        stop = ordered[k + 1] if k + 1 < len(ordered) else n
        forward = [pos]
        nxt = pos + 1
        while nxt < stop and not cuts[nxt]:
            forward.append(nxt)
            nxt += 1
        plans.append(SegmentPlan(k, forward))
        if k == 0 and pos > 0 and not cuts[pos]:
            backward = [pos]
            prev = pos - 1
            while prev >= 0:
                backward.append(prev)
                if cuts[prev]:
                    break
                prev -= 1
            if len(backward) > 1:
                plans.append(SegmentPlan(k, backward))
    return plans


def mask_to_box(mask: np.ndarray) -> BBox | None:
    """Box around the largest connected blob of ``mask`` (``None`` if empty or implausible).

    Using only the largest blob stops stray splash pixels elsewhere in the
    picture from stretching the box.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    if mask_u8.sum() == 0:
        return None
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if count <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[biggest]
    if area > MAX_MASK_FRAME_FRACTION * mask.shape[0] * mask.shape[1] or min(w, h) < 3:
        return None
    return BBox(float(x), float(y), float(x + w), float(y + h))


def reacquired_flags(boxes: Sequence[BBox | None], from_checkpoint: Sequence[bool],
                     gap: int, flag_len: int) -> list[bool]:
    """Flag the first ``flag_len`` frames after the target reappears from a gap of ``>= gap`` frames.

    A checkpoint frame (the user's own box) is never flagged and resets the gap.
    """
    flags = [False] * len(boxes)
    missing, remaining = 0, 0
    for t, box in enumerate(boxes):
        if from_checkpoint[t]:
            missing, remaining = 0, 0
            continue
        if box is None:
            missing += 1  # a short flicker does not end an active flag
            continue
        if missing >= gap:
            remaining = flag_len
        missing = 0
        if remaining > 0:
            flags[t] = True
            remaining -= 1
    return flags


# ---------------------------------------------------------------------------
# Video passes
# ---------------------------------------------------------------------------
def scan_video(video_path: str | Path, frame_indices: Sequence[int], progress=None
               ) -> tuple[list[bool], list[np.ndarray | None], tuple[int, int]]:
    """One cheap pass over the analysed frames: camera cuts, camera motion and frame size."""
    wanted = set(frame_indices)
    last = max(frame_indices)
    cap = cv2.VideoCapture(str(video_path))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[0])
    cuts: list[bool] = []
    motions: list[np.ndarray | None] = []
    prev_sig = prev_gray = None
    try:
        idx = frame_indices[0]
        while idx <= last and cap.grab():
            if idx in wanted:
                ok, frame = cap.retrieve()
                if not ok:
                    raise TrackingError(f"Frame {idx} could not be decoded.")
                sig = frame_signature(frame)
                gray, scale = to_motion_gray(frame)
                cut = prev_sig is not None and is_scene_cut(prev_sig, sig)
                cuts.append(cut)
                motions.append(None if prev_gray is None or cut else estimate_camera_motion(prev_gray, gray, scale))
                prev_sig, prev_gray = sig, gray
                if progress and len(cuts) % 30 == 0:
                    progress(len(cuts) / len(frame_indices), f"Measuring camera movement… frame {idx}")
            idx += 1
    finally:
        cap.release()
    if len(cuts) != len(frame_indices):
        raise TrackingError("The video ended before all frames could be analysed.")
    return cuts, motions, frame_size


def ensure_sam2_model() -> Path:
    """Download the SAM 2 weights into ``models/`` on first use."""
    path = config.SAM2_MODEL_PATH
    if path.is_file():
        return path
    try:
        from ultralytics.utils.downloads import attempt_download_asset
        downloaded = Path(attempt_download_asset(str(path)))
    except Exception as exc:  # network problems, GitHub down...
        raise ModelLoadError("The SAM 2 tracking model could not be downloaded. Connect to the internet "
                             "once so it can be saved into the models/ folder.") from exc
    if not downloaded.is_file():
        raise ModelLoadError("The SAM 2 tracking model is missing from the models/ folder.")
    return downloaded


def _write_clip(video_path: str | Path, frame_indices: Sequence[int], positions: Sequence[int],
                out_path: Path, fps: float) -> None:
    """Write the analysed frames at ``positions`` (in that order) to a temporary clip."""
    cap = cv2.VideoCapture(str(video_path))
    writer = None
    try:
        ascending = list(positions) == sorted(positions)
        # Read in chunks of 60 frames so memory stays small. Backward segments
        # read each chunk forward (fast) and write it reversed.
        for start in range(0, len(positions), 60):
            frames = _read_positions(cap, frame_indices, sorted(positions[start:start + 60]))
            for frame in (frames if ascending else reversed(frames)):
                writer = writer or _open_writer(out_path, frame, fps)
                writer.write(frame)
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def _read_positions(cap: cv2.VideoCapture, frame_indices: Sequence[int], positions: Sequence[int]
                    ) -> list[np.ndarray]:
    """Read the analysed frames at ascending ``positions`` with one seek and sequential reads."""
    wanted = {frame_indices[p] for p in positions}
    idx, last = frame_indices[positions[0]], frame_indices[positions[-1]]
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    frames: list[np.ndarray] = []
    while idx <= last and cap.grab():
        if idx in wanted:
            ok, frame = cap.retrieve()
            if not ok:
                raise TrackingError(f"Frame {idx} could not be decoded.")
            frames.append(frame)
        idx += 1
    return frames


def _open_writer(path: Path, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    h, w = frame.shape[:2]
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


def _run_sam2_on_clip(clip: Path, box: BBox, imgsz: int, on_frame=None) -> list[BBox | None]:
    """Track ``box`` (given on the clip's first frame) through the clip with SAM 2.

    ``on_frame()`` is called after every processed frame (used for the progress bar).
    """
    import torch
    from ultralytics.models.sam import SAM2VideoPredictor

    predictor = SAM2VideoPredictor(overrides=dict(conf=0.25, task="segment", mode="predict", imgsz=imgsz,
                                                  model=str(ensure_sam2_model()), save=False, verbose=False,
                                                  half=torch.cuda.is_available()))
    boxes: list[BBox | None] = []
    for result in predictor(source=str(clip), bboxes=[box.to_list()], stream=True):
        mask = None
        if result.masks is not None and len(result.masks.data):
            mask = result.masks.data[0].cpu().numpy() > 0.5
        boxes.append(mask_to_box(mask) if mask is not None else None)
        if on_frame is not None:
            on_frame()
    return boxes


def _eta_text(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"about {seconds // 60} min {seconds % 60:02d} s left" if seconds >= 60 else f"about {seconds} s left"


def track_with_sam2(video_path: str | Path, frame_indices: Sequence[int], fps: float,
                    checkpoints: Sequence[tuple[int, BBox]], imgsz: int = 1024, progress=None
                    ) -> tuple[list[SamFrame], list[bool], list[bool], list[bool]]:
    """Track from the checkpoints ``[(position, box), ...]`` through all analysed frames.

    Returns ``(frames, reacquired, not_moving, cuts)``, one entry per analysed frame.
    """
    n = len(frame_indices)
    checkpoints = sorted(checkpoints, key=lambda c: c[0])
    cuts, motions, frame_size = scan_video(video_path, frame_indices,
                               progress=lambda f, m: progress(0.15 * f, m) if progress else None)
    plans = plan_segments([p for p, _ in checkpoints], cuts, n)
    out: list[SamFrame] = [SamFrame(None) for _ in range(n)]
    total = sum(len(p.positions) for p in plans)
    done = 0
    sample_fps = fps / (frame_indices[1] - frame_indices[0]) if len(frame_indices) > 1 else fps
    started = time.time()

    def frame_done(part: int) -> None:
        nonlocal done
        done += 1
        if progress and (done % 5 == 0 or done == total):
            rate = done / max(time.time() - started, 1e-6)
            progress(0.15 + 0.85 * done / max(total, 1),
                     f"Following the swimmer: frame {done} of {total} (part {part} of {len(plans)}) - "
                     f"{_eta_text((total - done) / rate)}")

    with tempfile.TemporaryDirectory(prefix="swimvision_") as tmp:
        for s, plan in enumerate(plans):
            pos0, box0 = checkpoints[plan.checkpoint_index]
            if progress:
                progress(0.15 + 0.85 * done / max(total, 1),
                         f"Preparing part {s + 1} of {len(plans)} and loading the AI model…")
            clip = Path(tmp) / f"segment_{s}.mp4"
            _write_clip(video_path, frame_indices, plan.positions, clip, sample_fps)
            boxes = _run_sam2_on_clip(clip, box0, imgsz, on_frame=lambda: frame_done(s + 1))
            for i, pos in enumerate(plan.positions):
                if i == 0:
                    out[pos] = SamFrame(box0, from_checkpoint=True, segment=s)
                elif i < len(boxes):
                    out[pos] = SamFrame(boxes[i], segment=s)
    boxes_only = [f.bbox for f in out]
    reacq = reacquired_flags(boxes_only, [f.from_checkpoint for f in out], REACQUIRE_GAP,
                             max(1, int(round(REACQUIRE_FLAG_S * sample_fps))))
    still = not_moving_flags(boxes_only, relative_steps(boxes_only, motions, frame_size),
                             window=max(3, int(round(NOT_MOVING_WINDOW_S * sample_fps))))
    return out, reacq, still, cuts
