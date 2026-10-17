"""Tracking the ONE swimmer the user selected.

How it works (read this first)
------------------------------
1. A pretrained YOLO model detects people in each analysed frame and
   **ByteTrack** (or BoT-SORT) links detections over time into *tracks*, each
   with an ID number. We call this the "candidate" layer.
2. The user's selection (a box on one frame) is matched to the candidate track
   that overlaps it most. That track is our **target**.
3. ``follow_target`` then walks through time and decides, frame by frame, which
   candidate is still our swimmer. Trackers sometimes give a person a *new* ID
   after a splash or occlusion; ``follow_target`` only accepts a new ID when it
   clearly overlaps where the swimmer just was, and it **flags** that frame as
   "reacquired" so the user can check it. If the situation is ambiguous it
   reports the swimmer as LOST rather than guessing. We never silently switch
   swimmers.
4. If YOLO cannot see the swimmer at all (common when partly submerged), the
   user's hand-drawn box starts an OpenCV **ViT visual tracker** instead.

Every frame ends up as an ``Observation`` with a status:
``tracked`` / ``low_confidence`` / ``reacquired`` / ``lost``. Lost frames have
NO coordinates - we never invent positions.

The pure logic (``follow_target``, ``find_unreliable_segments``, summaries) has
no video or model dependency and is unit-tested.
"""

from __future__ import annotations

import logging
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import cv2
import numpy as np
import pandas as pd

import config
from core.detector import PERSON_CLASS_ID, detect_people, load_yolo
from core.errors import ModelLoadError, TrackingError
from core.geometry import BBox
from core.scene import frame_signature, is_scene_cut
from core.video import VideoMetadata, frame_to_timestamp, read_frame

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]  # (fraction 0..1, message)

# Observation statuses --------------------------------------------------------
STATUS_TRACKED = "tracked"
STATUS_LOW_CONFIDENCE = "low_confidence"
STATUS_REACQUIRED = "reacquired"
STATUS_LOST = "lost"
STATUS_NOT_MOVING = "not_moving"
UNRELIABLE_STATUSES = {STATUS_LOW_CONFIDENCE, STATUS_REACQUIRED, STATUS_LOST, STATUS_NOT_MOVING}

# Plain-language meaning of each status, shown in the app and on the video.
STATUS_TEXT = {
    STATUS_TRACKED: "Following the swimmer",
    STATUS_LOST: "Swimmer lost (underwater, in splash, too small, off-screen, or the tracker lost them)",
    STATUS_REACQUIRED: "Found again after a gap - check it is still your swimmer",
    STATUS_NOT_MOVING: "Box is not moving through the pool - fine on the starting block, "
                       "otherwise it is probably on a spectator or official",
    STATUS_LOW_CONFIDENCE: "Weak match - the tracker is unsure",
}

MODE_YOLO = "yolo_tracker"
MODE_VISUAL = "visual_tracker"
MODE_SAM2 = "sam2_segmentation"


# ===========================================================================
# 1. Data classes
# ===========================================================================
@dataclass(frozen=True)
class TrackedBox:
    """One candidate from the detector+tracker layer in one frame."""

    track_id: int
    bbox: BBox
    confidence: float | None


@dataclass(frozen=True)
class SwimmerSelection:
    """What the user picked: a box on one frame."""

    anchor_frame: int       # frame index where the box was chosen
    bbox: BBox
    source: str             # "detected" (picked a YOLO box) or "manual" (drew it)
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"anchor_frame": self.anchor_frame, "bbox": self.bbox.to_list(),
                "source": self.source, "confidence": self.confidence}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SwimmerSelection":
        return cls(int(d["anchor_frame"]), BBox.from_list(d["bbox"]), d["source"], d.get("confidence"))


@dataclass(frozen=True)
class FollowConfig:
    """Tunable rules for deciding that a track is still our swimmer."""

    max_gap_samples: int = 15         # how long we try to re-acquire after losing the swimmer
    reacquire_iou: float = 0.30       # a new ID must overlap the last box at least this much
    ambiguity_ratio: float = 0.80     # runner-up overlap this close to the best => ambiguous => lost
    low_confidence: float = 0.35      # below this score a frame is flagged "low_confidence"
    anchor_min_iou: float = 0.30      # user's box must overlap a track at least this much
    visual_min_score: float = 0.30    # ViT score below this => visual tracker declares "lost"
    visual_step_area_ratio: float = 2.0   # box area may not change more than this factor per step
    visual_min_side_px: float = 6.0       # a box thinner than this is a collapsed "dot"
    visual_max_frame_fraction: float = 0.35  # a box covering more of the frame than this is clutter
    visual_min_visible: float = 0.6       # fraction of the box that must lie inside the frame
    min_good_coverage: float = 0.5        # "auto": below this YOLO coverage, also try the visual tracker


@dataclass(frozen=True)
class FollowResult:
    """The tracker's answer for one frame (``None`` in a list means 'lost')."""

    track_id: int
    bbox: BBox
    confidence: float | None
    reacquired: bool = False


@dataclass(frozen=True)
class Observation:
    """One analysed frame: where the swimmer was (or that we lost them)."""

    sample_index: int
    frame_index: int
    timestamp_s: float
    status: str
    track_id: int | None = None
    bbox: BBox | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class UnreliableSegment:
    """A run of consecutive frames the user should treat with caution."""

    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"start_frame": self.start_frame, "end_frame": self.end_frame,
                "start_time_s": self.start_time_s, "end_time_s": self.end_time_s,
                "reasons": list(self.reasons)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "UnreliableSegment":
        return cls(d["start_frame"], d["end_frame"], d["start_time_s"], d["end_time_s"], tuple(d["reasons"]))


@dataclass
class TrackingResult:
    """Everything produced by one tracking run."""

    observations: list[Observation]
    mode: str
    stride: int
    notes: list[str] = field(default_factory=list)
    runtime_s: float = 0.0
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        return observations_to_dataframe(self.observations)


@dataclass(frozen=True)
class TrackingSummary:
    """Small, JSON-friendly digest of a tracking run (kept in the analysis record)."""

    mode: str
    n_samples: int
    n_tracked: int
    n_low_confidence: int
    n_reacquired: int
    n_lost: int
    tracked_fraction: float
    stride: int
    runtime_s: float
    notes: list[str]
    settings: dict[str, Any]
    segments: list[UnreliableSegment]
    n_not_moving: int = 0

    @property
    def followed_fraction(self) -> float:
        """Share of analysed frames where the swimmer was followed with no warning at all."""
        return self.n_tracked / self.n_samples if self.n_samples else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in (
            "mode", "n_samples", "n_tracked", "n_low_confidence", "n_reacquired", "n_lost",
            "n_not_moving", "tracked_fraction", "stride", "runtime_s", "notes", "settings")}
        d["segments"] = [s.to_dict() for s in self.segments]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrackingSummary":
        return cls(
            mode=d["mode"], n_samples=d["n_samples"], n_tracked=d["n_tracked"],
            n_low_confidence=d["n_low_confidence"], n_reacquired=d["n_reacquired"],
            n_lost=d["n_lost"], tracked_fraction=d["tracked_fraction"], stride=d["stride"],
            runtime_s=d["runtime_s"], notes=list(d["notes"]), settings=dict(d["settings"]),
            segments=[UnreliableSegment.from_dict(s) for s in d["segments"]],
            n_not_moving=d.get("n_not_moving", 0),  # older records have no such key
        )


# ===========================================================================
# 2. Pure logic: which candidate is our swimmer?
# ===========================================================================
def match_anchor_track(candidates: Sequence[TrackedBox], anchor_box: BBox,
                       min_iou: float) -> TrackedBox | None:
    """Return the candidate that overlaps the user's box most (or ``None``)."""
    best = max(candidates, key=lambda c: c.bbox.iou(anchor_box), default=None)
    if best is not None and best.bbox.iou(anchor_box) >= min_iou:
        return best
    return None


def _reacquire(candidates: Sequence[TrackedBox], last_box: BBox, cfg: FollowConfig) -> TrackedBox | None:
    """Accept a *different* track ID only if it clearly took over the swimmer's spot.

    Rules (this is what prevents silently switching swimmers):
      * its box must overlap the swimmer's last known box by ``reacquire_iou``;
      * no other candidate may overlap almost as much (ambiguity => give up).
    """
    scored = sorted(((c.bbox.iou(last_box), c) for c in candidates), key=lambda t: t[0], reverse=True)
    if not scored or scored[0][0] < cfg.reacquire_iou:
        return None
    if len(scored) > 1 and scored[1][0] >= cfg.ambiguity_ratio * scored[0][0]:
        return None  # two people could both be the swimmer -> refuse to guess
    return scored[0][1]


def _follow_one_direction(sequence: Sequence[Sequence[TrackedBox]], start: TrackedBox,
                          cfg: FollowConfig) -> list[FollowResult | None]:
    """Follow the target through ``sequence`` (frames after the anchor, in order)."""
    current_id, last_box, gap = start.track_id, start.bbox, 0
    results: list[FollowResult | None] = []
    for candidates in sequence:
        same = next((c for c in candidates if c.track_id == current_id), None)
        if same is not None:
            results.append(FollowResult(same.track_id, same.bbox, same.confidence, reacquired=gap > 0))
            last_box, gap = same.bbox, 0
            continue

        gap += 1
        if gap <= cfg.max_gap_samples:
            taker = _reacquire(candidates, last_box, cfg)
            if taker is not None:
                current_id, last_box, gap = taker.track_id, taker.bbox, 0
                results.append(FollowResult(taker.track_id, taker.bbox, taker.confidence, reacquired=True))
                continue
        results.append(None)  # lost - no coordinates are invented
    return results


def cut_free_range(cuts: Sequence[bool] | None, anchor_pos: int, n: int) -> tuple[int, int]:
    """Positions ``(lo, hi)`` (inclusive) reachable from the anchor without crossing a camera cut.

    ``cuts[i]`` is True when the picture at sample ``i`` is a different shot
    from sample ``i - 1``. Tracking may not continue across such a boundary.
    """
    if cuts is None:
        return 0, n - 1
    lo = anchor_pos
    while lo > 0 and not cuts[lo]:
        lo -= 1
    hi = anchor_pos
    while hi + 1 < n and not cuts[hi + 1]:
        hi += 1
    return lo, hi


def follow_target(candidates_per_sample: Sequence[Sequence[TrackedBox]], anchor_pos: int,
                  anchor_box: BBox, cfg: FollowConfig = FollowConfig(),
                  cuts: Sequence[bool] | None = None) -> list[FollowResult | None]:
    """Decide, for every analysed frame, where the selected swimmer is.

    Args:
        candidates_per_sample: for each analysed frame, the tracker's boxes.
        anchor_pos: position (in that list) of the frame where the user chose.
        anchor_box: the user's box.
        cuts: optional per-frame camera-cut flags (see ``cut_free_range``).

    Returns:
        A list the same length as ``candidates_per_sample``; ``None`` = lost.
        Frames separated from the anchor by a camera cut are always ``None``.

    Raises:
        TrackingError: if no tracker box overlaps the user's selection.
    """
    n = len(candidates_per_sample)
    anchor = match_anchor_track(candidates_per_sample[anchor_pos], anchor_box, cfg.anchor_min_iou)
    if anchor is None:
        raise TrackingError("No tracked person overlaps the selected box on the chosen frame.")
    lo, hi = cut_free_range(cuts, anchor_pos, n)
    forward = _follow_one_direction(candidates_per_sample[anchor_pos + 1:hi + 1], anchor, cfg)
    backward = _follow_one_direction(candidates_per_sample[lo:anchor_pos][::-1], anchor, cfg)[::-1]
    return ([None] * lo + backward + [FollowResult(anchor.track_id, anchor.bbox, anchor.confidence)]
            + forward + [None] * (n - 1 - hi))


# ===========================================================================
# 3. Results -> observations, tables, summaries
# ===========================================================================
def build_observations(frame_indices: Sequence[int], fps: float,
                       results: Sequence[FollowResult | None],
                       cfg: FollowConfig = FollowConfig()) -> list[Observation]:
    """Attach timestamps and a status label to each follow result."""
    observations: list[Observation] = []
    for pos, (frame_idx, res) in enumerate(zip(frame_indices, results)):
        ts = frame_to_timestamp(frame_idx, fps)
        if res is None:
            observations.append(Observation(pos, frame_idx, ts, STATUS_LOST))
            continue
        if res.reacquired:
            status = STATUS_REACQUIRED
        elif res.confidence is not None and res.confidence < cfg.low_confidence:
            status = STATUS_LOW_CONFIDENCE
        else:
            status = STATUS_TRACKED
        observations.append(Observation(pos, frame_idx, ts, status, res.track_id, res.bbox, res.confidence))
    return observations


TRACKING_COLUMNS = ["sample_index", "frame_index", "timestamp_s", "status", "track_id",
                    "x1", "y1", "x2", "y2", "cx", "cy", "box_w", "box_h", "confidence"]


def observations_to_dataframe(observations: Sequence[Observation]) -> pd.DataFrame:
    """One row per analysed frame. Lost frames have empty (NaN) coordinates."""
    rows = []
    for o in observations:
        row: dict[str, Any] = {"sample_index": o.sample_index, "frame_index": o.frame_index,
                               "timestamp_s": o.timestamp_s, "status": o.status,
                               "track_id": o.track_id, "confidence": o.confidence}
        if o.bbox is not None:
            cx, cy = o.bbox.center
            row.update(x1=o.bbox.x1, y1=o.bbox.y1, x2=o.bbox.x2, y2=o.bbox.y2,
                       cx=cx, cy=cy, box_w=o.bbox.width, box_h=o.bbox.height)
        rows.append(row)
    return pd.DataFrame(rows, columns=TRACKING_COLUMNS)


def find_unreliable_segments(observations: Sequence[Observation]) -> list[UnreliableSegment]:
    """Group consecutive non-``tracked`` frames into segments the user should check."""
    segments: list[UnreliableSegment] = []
    run: list[Observation] = []

    def close_run() -> None:
        if run:
            reasons = tuple(sorted({o.status for o in run}))
            segments.append(UnreliableSegment(run[0].frame_index, run[-1].frame_index,
                                              run[0].timestamp_s, run[-1].timestamp_s, reasons))
            run.clear()

    for o in observations:
        if o.status in UNRELIABLE_STATUSES:
            run.append(o)
        else:
            close_run()
    close_run()
    return segments


def summarize_tracking(result: TrackingResult) -> TrackingSummary:
    """Count statuses and list unreliable segments."""
    counts = {s: 0 for s in STATUS_TEXT}
    for o in result.observations:
        counts[o.status] += 1
    n = len(result.observations)
    return TrackingSummary(
        mode=result.mode, n_samples=n, n_tracked=counts[STATUS_TRACKED],
        n_low_confidence=counts[STATUS_LOW_CONFIDENCE], n_reacquired=counts[STATUS_REACQUIRED],
        n_lost=counts[STATUS_LOST], n_not_moving=counts[STATUS_NOT_MOVING],
        tracked_fraction=(n - counts[STATUS_LOST]) / n if n else 0.0,
        stride=result.stride, runtime_s=result.runtime_s, notes=list(result.notes),
        settings=dict(result.settings), segments=find_unreliable_segments(result.observations),
    )


# ===========================================================================
# 4. Running the models over the video
# ===========================================================================
def _report(progress: ProgressCallback | None, fraction: float, message: str) -> None:
    if progress is not None:
        progress(min(max(fraction, 0.0), 1.0), message)


def run_yolo_tracking(video_path: str | Path, frame_indices: Sequence[int], *, model_name: str,
                      conf: float, imgsz: int, tracker_name: str,
                      progress: ProgressCallback | None = None
                      ) -> tuple[list[list[TrackedBox]], list[bool]]:
    """Run YOLO + ByteTrack/BoT-SORT over the sampled frames.

    Returns:
        ``(candidates, cuts)``: for every sampled frame, the tracked people and
        whether the camera cut to a new shot since the previous sampled frame.
    """
    model = load_yolo(model_name)  # fresh object => fresh tracker state
    wanted = set(frame_indices)
    last = max(frame_indices)
    cap = cv2.VideoCapture(str(video_path))
    per_sample: list[list[TrackedBox]] = []
    cuts: list[bool] = []
    previous_sig = None
    try:
        idx = 0
        while idx <= last and cap.grab():
            if idx in wanted:
                ok, frame = cap.retrieve()
                if not ok:
                    raise TrackingError(f"Frame {idx} could not be decoded during tracking.")
                sig = frame_signature(frame)
                cuts.append(previous_sig is not None and is_scene_cut(previous_sig, sig))
                previous_sig = sig
                per_sample.append(_track_one_frame(model, frame, conf, imgsz, tracker_name))
                _report(progress, len(per_sample) / len(frame_indices),
                        f"Detecting and tracking people… frame {idx}")
            idx += 1
    finally:
        cap.release()
    if len(per_sample) != len(frame_indices):
        raise TrackingError("The video ended before all frames could be analysed.")
    return per_sample, cuts


def _track_one_frame(model, frame: np.ndarray, conf: float, imgsz: int, tracker_name: str) -> list[TrackedBox]:
    """Run the tracker on one frame and convert the result to ``TrackedBox`` objects."""
    results = model.track(frame, persist=True, classes=[PERSON_CLASS_ID], conf=conf, imgsz=imgsz,
                          tracker=tracker_name, verbose=False)
    boxes = results[0].boxes if results else None
    if boxes is None or boxes.id is None:  # no confirmed tracks in this frame
        return []
    ids = boxes.id.int().cpu().tolist()
    xyxy = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    return [TrackedBox(int(i), BBox(*map(float, b)), float(s)) for i, b, s in zip(ids, xyxy, scores)]


def ensure_vit_model() -> Path:
    """Download the small (0.7 MB) OpenCV ViT tracking model on first use."""
    path = config.VIT_TRACKER_PATH
    if path.is_file() and path.stat().st_size > 100_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(config.VIT_TRACKER_URL, path)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise ModelLoadError(
            "The visual tracking model could not be downloaded (internet needed once). "
            f"Details: {type(exc).__name__}") from exc
    return path


def _make_vit_tracker():
    # OpenCV 5 prints a harmless "Targets are not supported by the new graph engine" warning.
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    params = cv2.TrackerVit_Params()
    params.net = str(ensure_vit_model())
    return cv2.TrackerVit_create(params)


def visual_box_is_plausible(box: BBox, previous: BBox, frame_w: int, frame_h: int,
                            cfg: FollowConfig) -> bool:
    """Catch DEGENERATE boxes from the visual tracker (and nothing more).

    Visual trackers sometimes lock onto background clutter while still reporting
    a decent-looking score. The tell-tale signs are a box that collapses to a
    dot, covers a large part of the picture, is mostly outside the frame, or
    jumps in size between two consecutive analysed frames.

    Deliberately NOT checked: total growth/shrinkage since the start. A swimmer
    heading toward (or away from) the camera really does change apparent size
    by a large factor, and an earlier version of this check wrongly cut off a
    correct track for that reason (found on real race footage).
    """
    if min(box.width, box.height) < cfg.visual_min_side_px or previous.area <= 1:
        return False
    if box.area > cfg.visual_max_frame_fraction * frame_w * frame_h:
        return False
    step = box.area / previous.area
    if not (1 / cfg.visual_step_area_ratio <= step <= cfg.visual_step_area_ratio):
        return False
    return box.clamp(frame_w, frame_h).area / box.area >= cfg.visual_min_visible


def visual_stop_reason(ok: bool, box: BBox | None, score: float, last_box: BBox, frame_w: int,
                       frame_h: int, cfg: FollowConfig) -> str | None:
    """Why (if at all) the visual tracker should stop at this frame. ``None`` = keep going.

    Returning a human-readable reason (instead of a bare True/False) lets the
    app tell the user exactly why tracking ended.
    """
    if not ok or box is None:
        return "the tracker could not find the target any more"
    if score < cfg.visual_min_score:
        return (f"the match score fell to {score:.2f} (minimum {cfg.visual_min_score:.2f}) - the swimmer "
                "may be hidden by splash, too small, or leaving the picture")
    if not visual_box_is_plausible(box, last_box, frame_w, frame_h, cfg):
        return "the tracking box became implausible (collapsed, too large, mostly off-screen, or jumped)"
    return None


def _visual_follow(frames: Iterable[tuple[int, np.ndarray]], start_frame: np.ndarray,
                   start_box: BBox, cfg: FollowConfig, track_id: int
                   ) -> tuple[list[FollowResult | None], tuple[int, str] | None]:
    """Follow ``start_box`` through ``frames`` with the ViT tracker.

    Stops (everything after stays 'lost') when the tracker fails, its score is
    too low, its box is implausible, or the camera cuts to a different shot.
    A visual tracker that lost its target may jump to a different swimmer, so
    we never let it continue.

    Returns ``(results, stop)`` where ``stop`` is ``(frame_index, reason)`` for
    the frame at which tracking ended (``None`` if it never stopped).
    """
    tracker = _make_vit_tracker()
    tracker.init(start_frame, start_box.to_xywh_int())
    previous_sig = frame_signature(start_frame)
    last_box = start_box
    results: list[FollowResult | None] = []
    stop: tuple[int, str] | None = None
    for frame_idx, frame in frames:
        if stop is not None:
            results.append(None)
            continue
        sig = frame_signature(frame)
        if is_scene_cut(previous_sig, sig):
            stop = (frame_idx, "the camera cut to a different shot (a swimmer cannot be followed into a new shot)")
            results.append(None)
            continue
        previous_sig = sig
        ok, xywh = tracker.update(frame)
        score = float(tracker.getTrackingScore())
        box = BBox.from_xywh(*xywh) if ok else None
        h, w = frame.shape[:2]
        reason = visual_stop_reason(ok, box, score, last_box, w, h, cfg)
        if reason is not None:
            stop = (frame_idx, reason)
            results.append(None)
            continue
        last_box = box
        results.append(FollowResult(track_id, box, score))
    return results, stop


def run_visual_tracking(video_path: str | Path, frame_indices: Sequence[int], anchor_pos: int,
                        selection: SwimmerSelection, cfg: FollowConfig,
                        progress: ProgressCallback | None = None
                        ) -> tuple[list[FollowResult | None], list[tuple[int, str, str]]]:
    """Track the user's box with the OpenCV ViT tracker, forward and backward.

    Returns ``(results, stops)``; each stop is ``(frame_index, reason, "before"|"after")``
    describing why tracking ended before / after the user's chosen frame.
    """
    cap = cv2.VideoCapture(str(video_path))
    total = max(1, len(frame_indices))
    try:
        anchor_frame = _seek_and_read(cap, frame_indices[anchor_pos])

        def forward_frames() -> Iterator[tuple[int, np.ndarray]]:
            """Sequential reads (fast): decode on, skipping frames between samples."""
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[anchor_pos])
            cap.grab()  # consume the anchor frame itself
            current = frame_indices[anchor_pos]
            for pos in range(anchor_pos + 1, len(frame_indices)):
                target = frame_indices[pos]
                while current < target and cap.grab():
                    current += 1
                ok, frame = cap.retrieve()
                if not ok:
                    raise TrackingError(f"Frame {target} could not be read for visual tracking.")
                _report(progress, pos / total, "Visual tracking…")
                yield target, frame

        def backward_frames() -> Iterator[tuple[int, np.ndarray]]:
            """Seeking reads (slower) - normally only a short stretch before the anchor."""
            for pos in range(anchor_pos - 1, -1, -1):
                yield frame_indices[pos], _seek_and_read(cap, frame_indices[pos])

        fwd, fwd_stop = _visual_follow(forward_frames(), anchor_frame, selection.bbox, cfg, track_id=0)
        bwd, bwd_stop = _visual_follow(backward_frames(), anchor_frame, selection.bbox, cfg, track_id=0)
    finally:
        cap.release()
    stops = [(frame, reason, direction) for direction, stop in (("before", bwd_stop), ("after", fwd_stop))
             if stop is not None for frame, reason in [stop]]
    return bwd[::-1] + [FollowResult(0, selection.bbox, None)] + fwd, stops


def _seek_and_read(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise TrackingError(f"Frame {frame_index} could not be read for visual tracking.")
    return frame


# ===========================================================================
# 5. Orchestration
# ===========================================================================
def snap_to_sample_grid(frame_index: int, stride: int, frame_count: int) -> tuple[int, int]:
    """Return ``(anchor_pos, snapped_frame_index)`` on the analysed-frame grid."""
    pos = int(round(frame_index / stride))
    pos = min(pos, (frame_count - 1) // stride)
    return pos, pos * stride


def _yolo_sees_selection(video_path: str | Path, frame_index: int, selection: SwimmerSelection,
                         model_name: str, conf: float, imgsz: int, cfg: FollowConfig) -> bool:
    """Quick check on the anchor frame: does any YOLO detection overlap the user's box?"""
    frame = read_frame(video_path, frame_index)
    detections = detect_people(load_yolo(model_name), frame, conf, imgsz)
    return any(d.bbox.iou(selection.bbox) >= cfg.anchor_min_iou for d in detections)


def _cut_notes(cuts: Sequence[bool], anchor_pos: int, frame_indices: Sequence[int], fps: float) -> list[str]:
    """Explain where camera cuts limited the tracked stretch."""
    lo, hi = cut_free_range(cuts, anchor_pos, len(cuts))
    boundaries: list[int] = []
    if lo > 0:                      # a cut before the anchor's shot begins
        boundaries.append(frame_indices[lo])
    if hi + 1 < len(cuts):          # a cut after the anchor's shot ends
        boundaries.append(frame_indices[hi + 1])
    return [f"Camera cut detected near t = {frame_to_timestamp(f, fps):.2f} s - tracking stops at a cut "
            "because the swimmer cannot be followed into a different shot." for f in boundaries]


def located_fraction(results: Sequence[FollowResult | None]) -> float:
    """Share of analysed frames in which the swimmer was located (not lost)."""
    return sum(r is not None for r in results) / len(results) if results else 0.0


def _try_yolo_mode(video_path, frame_indices, anchor_pos, selection, fps, model_name, conf, imgsz,
                   tracker_name, cfg, progress) -> tuple[list[FollowResult | None] | None, list[str]]:
    """Attempt YOLO tracking. Returns ``(results or None, notes)``."""
    anchor_frame = frame_indices[anchor_pos]
    if not _yolo_sees_selection(video_path, anchor_frame, selection, model_name, conf, imgsz, cfg):
        return None, ["YOLO could not find a person at your selection."]
    candidates, cuts = run_yolo_tracking(video_path, frame_indices, model_name=model_name, conf=conf,
                                         imgsz=imgsz, tracker_name=tracker_name, progress=progress)
    try:
        results = follow_target(candidates, anchor_pos, selection.bbox, cfg, cuts)
    except TrackingError:
        return None, ["YOLO detected a person there, but the tracker never confirmed a track for it."]
    return results, _cut_notes(cuts, anchor_pos, frame_indices, fps)


def _run_visual_mode(video_path, frame_indices, anchor_pos, selection, fps, cfg,
                     progress) -> tuple[list[FollowResult | None], list[str]]:
    results, stops = run_visual_tracking(video_path, frame_indices, anchor_pos, selection, cfg, progress)
    # Tell the user exactly why tracking ended on each side of the frame they chose.
    notes = [f"Visual tracking {'before' if side == 'before' else 'after'} your chosen frame ended at "
             f"t = {frame_to_timestamp(frame, fps):.2f} s because {reason}." for frame, reason, side in stops]
    notes.append("The visual tracker stops for good once the swimmer is lost "
                 "(it cannot verify that it is still following the same person).")
    return results, notes


def _pick_best(yolo: list[FollowResult | None] | None, visual: list[FollowResult | None] | None,
               notes: list[str]) -> tuple[list[FollowResult | None], str]:
    """Choose between the two candidate results and explain the choice in ``notes``."""
    if yolo is None and visual is None:
        raise TrackingError("No tracking method could be run.")
    if visual is None:
        return yolo, MODE_YOLO
    if yolo is None:
        notes.append("The visual tracker was used.")
        return visual, MODE_VISUAL
    y, v = located_fraction(yolo), located_fraction(visual)
    if v > y:
        notes.append(f"YOLO tracking located the swimmer in only {y:.0%} of frames; the visual tracker "
                     f"located them in {v:.0%}, so its result is shown.")
        return visual, MODE_VISUAL
    notes.append(f"The visual tracker was also tried ({v:.0%} of frames) but YOLO tracking ({y:.0%}) was better.")
    return yolo, MODE_YOLO


def problem_stretches(observations: Sequence[Observation], status: str,
                      merge_gap_s: float = 0.5) -> list[tuple[float, float]]:
    """``(start_s, end_s)`` stretches with ``status``; stretches less than ``merge_gap_s`` apart are joined."""
    stretches: list[list[float]] = []
    for o in observations:
        if o.status != status:
            continue
        if stretches and o.timestamp_s - stretches[-1][1] <= merge_gap_s:
            stretches[-1][1] = o.timestamp_s
        else:
            stretches.append([o.timestamp_s, o.timestamp_s])
    return [(a, b) for a, b in stretches]


def stretch_notes(observations: Sequence[Observation], min_duration_s: float = 1.0) -> list[str]:
    """Plain-language advice for every long 'lost' or 'not moving' stretch, in time order."""
    notes: list[tuple[float, str]] = []
    for a, b in problem_stretches(observations, STATUS_LOST):
        if b - a >= min_duration_s:
            notes.append((a, f"{a:.1f}-{b:.1f} s: swimmer lost (underwater, in splash, too small, or the "
                             "tracker lost them). If you can see your swimmer in this stretch, add a "
                             "checkpoint on the first frame where they reappear."))
    for a, b in problem_stretches(observations, STATUS_NOT_MOVING):
        if b - a >= min_duration_s:
            notes.append((a, f"{a:.1f}-{b:.1f} s: the box is not moving through the pool. That is fine while "
                             "the swimmer is on the block; during the race it means the box is probably on "
                             "a spectator or official - add a checkpoint on your swimmer there."))
    return [text for _, text in sorted(notes)]


def _track_sam2(video_path: str | Path, meta: VideoMetadata, frame_indices: list[int],
                checkpoints: Sequence[SwimmerSelection], imgsz: int,
                progress: ProgressCallback | None) -> tuple[list[Observation], list[str]]:
    """SAM 2 segmentation tracking from one or more user checkpoints."""
    from core.sam_tracker import track_with_sam2

    stride = frame_indices[1] - frame_indices[0] if len(frame_indices) > 1 else 1
    by_pos: dict[int, BBox] = {}
    for cp in checkpoints:  # a later checkpoint on the same analysed frame replaces an earlier one
        by_pos[snap_to_sample_grid(cp.anchor_frame, stride, meta.frame_count)[0]] = cp.bbox
    # Analysis starts at the earliest checkpoint: footage before it (warm-up, walking to the
    # block) is not part of the race and would only cost minutes of processing.
    first = min(by_pos)
    analysed = frame_indices[first:]
    frames, reacquired, not_moving, cuts = track_with_sam2(
        video_path, analysed, meta.fps, sorted((p - first, b) for p, b in by_pos.items()),
        imgsz=imgsz, progress=progress)
    observations: list[Observation] = []
    for i, (frame_idx, f) in enumerate(zip(analysed, frames)):
        pos, ts = first + i, frame_to_timestamp(frame_idx, meta.fps)
        if f.bbox is None:
            observations.append(Observation(pos, frame_idx, ts, STATUS_LOST))
            continue
        status = (STATUS_TRACKED if f.from_checkpoint else STATUS_REACQUIRED if reacquired[i]
                  else STATUS_NOT_MOVING if not_moving[i] else STATUS_TRACKED)
        observations.append(Observation(pos, frame_idx, ts, status, None, f.bbox, None))
    notes = [f"Segment Anything 2 followed the swimmer's outline from {len(by_pos)} checkpoint(s), "
             f"starting at your first checkpoint (t = {frame_to_timestamp(analysed[0], meta.fps):.2f} s)."]
    notes += [f"Camera cut near t = {frame_to_timestamp(analysed[i], meta.fps):.2f} s - tracking does "
              "not continue into a new shot. Add a checkpoint after the cut to keep going."
              for i, c in enumerate(cuts) if c]
    notes += stretch_notes(observations)
    return observations, notes


def track_swimmer(video_path: str | Path, meta: VideoMetadata, selection: SwimmerSelection, *,
                  stride: int, method: str = "auto", model_name: str = config.DEFAULT_YOLO_MODEL,
                  conf: float = 0.25, imgsz: int = 960, tracker_name: str = "bytetrack.yaml",
                  cfg: FollowConfig = FollowConfig(),
                  checkpoints: Sequence[SwimmerSelection] | None = None,
                  progress: ProgressCallback | None = None) -> TrackingResult:
    """Track the selected swimmer through the whole video.

    Args:
        method: ``"sam2"`` = Segment Anything 2 from the checkpoints (recommended);
            ``"auto"`` = YOLO + tracker, falling back to the visual tracker
            when no YOLO track matches the selection; ``"visual"`` = always use
            the visual tracker.
        checkpoints: extra user boxes on other frames (``"sam2"`` only).
            ``selection`` is always used as a checkpoint too.
    """
    started = time.time()
    frame_indices = list(range(0, meta.frame_count, stride))
    anchor_pos, _ = snap_to_sample_grid(selection.anchor_frame, stride, meta.frame_count)
    settings = {"method": method, "model": model_name, "conf": conf, "imgsz": imgsz,
                "tracker": tracker_name, "anchor_frame": frame_indices[anchor_pos]}
    notes: list[str] = []

    if method == "sam2":
        all_checkpoints = [selection, *(checkpoints or [])]
        settings = {"method": method, "checkpoints": [c.to_dict() for c in all_checkpoints]}
        observations, notes = _track_sam2(video_path, meta, frame_indices, all_checkpoints,
                                          imgsz=1024, progress=progress)
        _report(progress, 1.0, "Tracking finished")
        return TrackingResult(observations, MODE_SAM2, stride, notes, time.time() - started, settings)

    # --- candidate 1: YOLO + ByteTrack/BoT-SORT (only in "auto" mode) ---
    yolo_results = None
    if method == "auto":
        yolo_results, yolo_notes = _try_yolo_mode(video_path, frame_indices, anchor_pos, selection, meta.fps,
                                                  model_name, conf, imgsz, tracker_name, cfg, progress)
        notes += yolo_notes

    # --- candidate 2: visual tracker (forced, YOLO impossible, or YOLO coverage poor) ---
    visual_results = None
    if yolo_results is None or located_fraction(yolo_results) < cfg.min_good_coverage:
        visual_results, visual_notes = _run_visual_mode(video_path, frame_indices, anchor_pos, selection,
                                                        meta.fps, cfg, progress)
        notes += visual_notes

    # --- keep whichever located the swimmer in more frames, and say so ---
    results, mode = _pick_best(yolo_results, visual_results, notes)

    observations = build_observations(frame_indices, meta.fps, results, cfg)
    if all(o.status == STATUS_LOST for o in observations):
        raise TrackingError("The swimmer could not be tracked in any frame.")
    _report(progress, 1.0, "Tracking finished")
    return TrackingResult(observations, mode, stride, notes, time.time() - started, settings)
