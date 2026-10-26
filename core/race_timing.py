"""Race timing for one tracked swimmer: start signal, wall turns and finish touch.

Three independent measurements are combined:

* **Audio.** Electronic starting systems sound a short tone made of a stack of
  perfectly flat harmonics. Speech, whistles and splashes do not look like
  that, so when the tone is audible it is used as the start.
* **Camera-compensated path.** The background motion between consecutive
  frames is estimated (the swimmer is masked out) and removed, so a handheld,
  panning or zooming camera does not create fake turns. A lane is a straight
  line in 3D and therefore also in the image, so the swimmer's compensated
  path is projected onto its main direction. Wall turns are the alternating
  extremes of that one-dimensional progress, whatever the camera angle.
* **Splash activity.** Camera-compensated pixel change around the swimmer.
  Swimming creates splash; the touch at the finish ends it. Perspective makes
  the last metres at the far wall almost motionless in the image, but the
  splash still stops at the touch.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from core.errors import SwimVisionError

logger = logging.getLogger(__name__)
ProgressFn = Callable[[int, int], None]

_WORK_WIDTH = 640           # resolution used for camera motion and splash activity
_SPLASH_THRESHOLD = 18      # grey-level change counted as splash after camera compensation
_PROPAGATE_S = 3.0          # follow the last swimmer region with the camera for this long
# Start signal to the first visible movement on the block. Measured on a
# real race with an audible tone (0.38 s) and confirmed by a second race's
# official time.
_REACTION_TO_MOVEMENT_S = 0.38


@dataclass(frozen=True)
class VideoScan:
    """Per-track-row measurements from one pass over the video."""

    crop_motion: np.ndarray   # recentred swimmer-crop difference (stroke rhythm)
    activity: np.ndarray      # camera-compensated splash fraction around the swimmer
    world_x: np.ndarray       # swimmer centre with camera motion removed
    world_y: np.ndarray
    observed: np.ndarray      # direct observation (not an estimate or a gap)
    roi_observed: np.ndarray  # activity measured on an observed box (not a propagated region)


@dataclass(frozen=True)
class StartTiming:
    pos: int
    time_s: float
    method: str               # "beep", "movement" or "clip"
    confidence: str
    evidence: str
    movement_pos: int | None  # first movement on the block, when visible


@dataclass(frozen=True)
class FinishTiming:
    pos: int
    confidence: str
    evidence: str


def scan_video(video_path: Path, track: pd.DataFrame, progress: ProgressFn | None = None) -> VideoScan:
    """Read the video once and measure motion, splash and the compensated path."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SwimVisionError("The original video could not be opened for race analysis.")
    n = len(track)
    crop_motion = np.zeros(n)
    activity = np.full(n, np.nan)
    world_x = np.full(n, np.nan)
    world_y = np.full(n, np.nan)
    roi_observed = np.zeros(n, dtype=bool)
    observed = (track.status.astype(str) == "tracked").to_numpy()
    timestamps = track.timestamp_s.to_numpy(float)

    accumulated = np.eye(3)           # current frame -> first analysed frame
    previous_small: np.ndarray | None = None
    previous_crop: np.ndarray | None = None
    previous_frame_index: int | None = None
    region: tuple[float, float, float, float] | None = None  # full-resolution x1, y1, x2, y2
    last_observed_time = -np.inf
    try:
        for pos, row in enumerate(track.itertuples(index=False)):
            wanted = int(row.frame_index)
            if previous_frame_index is None or wanted <= previous_frame_index:
                cap.set(cv2.CAP_PROP_POS_FRAMES, wanted)
            else:
                for _ in range(previous_frame_index + 1, wanted):
                    if not cap.grab():
                        raise SwimVisionError(f"Could not advance to frame {wanted} during analysis.")
            ok, frame = cap.read()
            if not ok or frame is None:
                raise SwimVisionError(f"Could not read frame {wanted} during analysis.")
            previous_frame_index = wanted
            h, w = frame.shape[:2]
            scale = min(1.0, _WORK_WIDTH / w)
            small = cv2.GaussianBlur(
                cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                             if scale < 1.0 else frame, cv2.COLOR_BGR2GRAY),
                (3, 3), 0,
            )
            box = (float(row.x1), float(row.y1), float(row.x2), float(row.y2))

            affine_small: np.ndarray | None = None
            if previous_small is not None:
                affine_small = _background_motion(previous_small, small, region or box, scale)
            if affine_small is not None:
                full = affine_small.copy()
                full[:, 2] /= scale
                accumulated = accumulated @ np.linalg.inv(np.vstack([full, [0.0, 0.0, 1.0]]))

            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            world = accumulated @ np.array([cx, cy, 1.0])
            world_x[pos], world_y[pos] = world[0], world[1]

            if observed[pos]:
                region = box
                last_observed_time = timestamps[pos]
            elif region is not None and affine_small is not None:
                region = _move_region(region, affine_small, scale)
            if region is not None and timestamps[pos] - last_observed_time > _PROPAGATE_S:
                region = None

            if previous_small is not None and region is not None:
                warped = (cv2.warpAffine(previous_small, affine_small, (small.shape[1], small.shape[0]))
                          if affine_small is not None else previous_small)
                pad = 0.10 if observed[pos] else 0.60
                activity[pos] = _splash_fraction(small, warped, region, scale, pad)
                roi_observed[pos] = bool(observed[pos])

            bw, bh = box[2] - box[0], box[3] - box[1]
            x1 = max(0, int(round(box[0] - 0.20 * bw)))
            x2 = min(w, int(round(box[2] + 0.20 * bw)))
            y1 = max(0, int(round(box[1] - 0.20 * bh)))
            y2 = min(h, int(round(box[3] + 0.20 * bh)))
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                gray = cv2.GaussianBlur(
                    cv2.cvtColor(cv2.resize(crop, (96, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY),
                    (5, 5), 0,
                )
                if previous_crop is not None:
                    crop_motion[pos] = float(cv2.absdiff(gray, previous_crop).mean())
                previous_crop = gray
            previous_small = small
            if progress and (pos % 10 == 0 or pos + 1 == n):
                progress(pos + 1, n)
    finally:
        cap.release()
    return VideoScan(crop_motion, activity, world_x, world_y, observed, roi_observed)


def _background_motion(previous: np.ndarray, current: np.ndarray,
                       box: tuple[float, float, float, float], scale: float) -> np.ndarray | None:
    """Similarity transform previous -> current of the static background (work resolution)."""
    mask = np.full(previous.shape, 255, np.uint8)
    pad = 20
    x1, y1, x2, y2 = (int(round(v * scale)) for v in box)
    mask[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad] = 0
    points = cv2.goodFeaturesToTrack(previous, 400, 0.01, 7, mask=mask)
    if points is None or len(points) < 10:
        return None
    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None,
                                                winSize=(21, 21), maxLevel=4)
    good = status.ravel() == 1
    if good.sum() < 10:
        return None
    affine, _ = cv2.estimateAffinePartial2D(points[good], moved[good], method=cv2.RANSAC,
                                            ransacReprojThreshold=1.5)
    return affine


def _move_region(region: tuple[float, float, float, float], affine_small: np.ndarray,
                 scale: float) -> tuple[float, float, float, float]:
    full = affine_small.copy()
    full[:, 2] /= scale
    corners = full @ np.array([[region[0], region[2]], [region[1], region[3]], [1.0, 1.0]])
    return float(corners[0, 0]), float(corners[1, 0]), float(corners[0, 1]), float(corners[1, 1])


def _splash_fraction(current: np.ndarray, warped_previous: np.ndarray,
                     region: tuple[float, float, float, float], scale: float, pad: float) -> float:
    height, width = current.shape
    x1, y1, x2, y2 = (v * scale for v in region)
    bw, bh = x2 - x1, y2 - y1
    left, top = max(0, int(x1 - pad * bw)), max(0, int(y1 - pad * bh))
    right, bottom = min(width, int(x2 + pad * bw)), min(height, int(y2 + pad * bh))
    if right - left < 3 or bottom - top < 3:
        return float("nan")
    difference = cv2.absdiff(current[top:bottom, left:right], warped_previous[top:bottom, left:right])
    return float((difference > _SPLASH_THRESHOLD).mean())


# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------
def find_start_beep(video_path: Path, window: tuple[float, float]) -> tuple[float, int] | None:
    """Onset time and strength of an electronic start tone inside ``window``.

    Returns ``None`` when the video has no audio or no tone-like event. The
    strength is the number of simultaneous flat harmonic lines (a start tone
    shows eight or more; voices and splashes rarely sustain more than two).
    """
    begin, end = max(0.0, window[0]), window[1]
    if end - begin < 0.2:
        return None
    padding = 3.0  # context for the per-frequency background estimate
    samples, sample_rate = _read_audio(video_path, max(0.0, begin - padding), end - begin + 2 * padding)
    if samples is None or samples.size < sample_rate:
        return None
    offset = max(0.0, begin - padding)
    from scipy.ndimage import median_filter, uniform_filter1d

    n_fft, hop = 1024, 160
    window_fn = np.hanning(n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(samples, n_fft)[::hop] * window_fn
    spectrum = 20 * np.log10(np.abs(np.fft.rfft(frames, axis=1)) + 1e-9)
    freqs = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    times = offset + (np.arange(spectrum.shape[0]) * hop + n_fft / 2) / sample_rate
    whitened = spectrum - np.median(spectrum, axis=0, keepdims=True)
    narrow = whitened - median_filter(whitened, size=(1, 15))
    band = (freqs > 300) & (freqs < 7000)
    narrow[:, ~band] = 0.0
    persistence = max(1, int(round(0.15 * sample_rate / hop)))
    flat = uniform_filter1d(narrow, size=persistence, axis=0)
    lines = (flat > 10.0).sum(axis=1)

    best: tuple[float, int] | None = None
    i = 0
    while i < len(lines):
        if lines[i] < 4:
            i += 1
            continue
        j = i
        while j + 1 < len(lines) and lines[j + 1] >= 4:
            j += 1
        peak = int(lines[i:j + 1].max())
        duration = times[j] - times[i]
        onset_index = i + int(np.argmax(lines[i:j + 1] >= max(4, peak * 0.5)))
        onset = float(times[onset_index])
        if peak >= 8 and duration >= 0.18 and begin <= onset <= end:
            if best is None or peak > best[1]:
                best = (onset, peak)
        i = j + 1
    return best


def _read_audio(video_path: Path, start: float, duration: float) -> tuple[np.ndarray | None, int]:
    sample_rate = 16000
    try:
        import imageio_ffmpeg

        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
            "-i", str(video_path), "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-",
        ]
        result = subprocess.run(command, capture_output=True, timeout=60, check=False)
    except Exception:
        logger.exception("Audio could not be read for start detection")
        return None, sample_rate
    if result.returncode != 0 or not result.stdout:
        return None, sample_rate
    return np.frombuffer(result.stdout, np.int16).astype(np.float32) / 32768.0, sample_rate


def first_water_position(track: pd.DataFrame) -> int | None:
    """First row where the persistent tracker has locked onto the in-water swimmer."""
    if "source" not in track:
        return None
    sources = track.source.fillna("").astype(str).str.lower()
    in_water = sources.str.contains("swimmer") & ~sources.str.contains("visual-person")
    positions = np.flatnonzero(in_water.to_numpy())
    return int(positions[0]) if positions.size else None


def detect_block_movement(track: pd.DataFrame, scan: VideoScan, fps: float) -> int | None:
    """First movement out of the still 'take your marks' position on the block."""
    water = first_water_position(track)
    if water is None or water < int(fps):
        return None
    sources = track.source.fillna("").astype(str).str.lower()
    on_deck = (scan.observed & ~sources.str.contains("swimmer").to_numpy())
    # The dive usually hides the swimmer briefly. Anything after the first
    # real gap is in the water even if the person detector still fires.
    hidden = np.flatnonzero(~scan.observed[:water])
    if hidden.size:
        runs = np.split(hidden, np.flatnonzero(np.diff(hidden) > 1) + 1)
        for run in runs:
            if len(run) >= max(2, int(round(0.4 * fps))):
                water = min(water, int(run[0]))
                break
    on_deck[water:] = False
    deck_rows = np.flatnonzero(on_deck)
    if deck_rows.size < int(fps):
        return None
    last_deck = int(deck_rows[-1])
    heights = (track.y2 - track.y1).to_numpy(float)
    body = float(np.median(heights[deck_rows[-int(fps * 2):]]))
    if body <= 0:
        return None
    x = pd.Series(np.where(on_deck, scan.world_x, np.nan)).interpolate(limit_area="inside").to_numpy()
    y = pd.Series(np.where(on_deck, scan.world_y, np.nan)).interpolate(limit_area="inside").to_numpy()
    smooth = max(1, int(round(0.10 * fps)))
    x = pd.Series(x).rolling(smooth, center=True, min_periods=1).mean().to_numpy()
    y = pd.Series(y).rolling(smooth, center=True, min_periods=1).mean().to_numpy()
    lag = max(1, int(round(0.20 * fps)))
    speed = np.full(len(x), np.nan)
    speed[lag:] = np.hypot(x[lag:] - x[:-lag], y[lag:] - y[:-lag]) / (lag / fps) / body
    still = speed < 0.30
    need = max(2, int(round(0.40 * fps)))
    search_from = max(0, last_deck - int(round(8 * fps)))
    # Walk back from the end of the deck phase to the last still stretch.
    end_of_still: int | None = None
    run = 0
    for pos in range(last_deck, search_from - 1, -1):
        if still[pos]:
            run += 1
            if run >= need:
                end_of_still = pos + run - 1
                break
        else:
            run = 0
    if end_of_still is None:
        return None
    anchor = slice(max(0, end_of_still - need + 1), end_of_still + 1)
    ax, ay = float(np.nanmedian(x[anchor])), float(np.nanmedian(y[anchor]))
    for pos in range(end_of_still, last_deck + 1):
        if np.isfinite(x[pos]) and np.hypot(x[pos] - ax, y[pos] - ay) / body > 0.06:
            return pos
    return last_deck


def detect_start(video_path: Path, track: pd.DataFrame, scan: VideoScan, fps: float) -> StartTiming:
    """Start signal from the beep, else from the first movement on the block."""
    times = track.timestamp_s.to_numpy(float)
    movement = detect_block_movement(track, scan, fps)
    water = first_water_position(track)
    if movement is not None:
        window = (times[movement] - 1.5, times[movement] + 0.1)
    elif water is not None:
        window = (times[water] - 3.5, times[water] - 0.3)
    else:
        window = (times[0], min(times[-1], times[0] + 5.0))
    beep = find_start_beep(video_path, window)
    if beep is not None:
        onset, _ = beep
        pos = int(np.clip(np.searchsorted(times, onset), 0, len(times) - 1))
        evidence = "the electronic start tone is audible in the video"
        if movement is not None:
            evidence += f"; first movement {times[movement] - onset:.2f} s later"
        return StartTiming(pos, round(onset, 3), "beep", "high", evidence, movement)
    if movement is not None:
        start_time = max(times[0], times[movement] - _REACTION_TO_MOVEMENT_S)
        pos = int(np.clip(np.searchsorted(times, start_time), 0, len(times) - 1))
        return StartTiming(pos, round(start_time, 3), "movement", "medium",
                           "no start tone was audible; estimated from the first movement on the block", movement)
    if water is not None and water > 0:
        start_time = max(times[0], times[water] - 1.0)
        pos = int(np.clip(np.searchsorted(times, start_time), 0, len(times) - 1))
        return StartTiming(pos, round(start_time, 3), "movement", "low",
                           "estimated from the swimmer's first appearance in the water", None)
    return StartTiming(0, round(float(times[0]), 3), "clip", "low",
                       "the video begins after the start", None)


# ---------------------------------------------------------------------------
# Turns
# ---------------------------------------------------------------------------
def progress_signal(scan: VideoScan, fps: float, begin: int, end: int) -> np.ndarray:
    """Swimmer progress along the lane, in pixels, for rows ``begin..end`` (NaN elsewhere)."""
    n = len(scan.world_x)
    result = np.full(n, np.nan)
    window = slice(begin, end + 1)
    seen = scan.observed[window]
    if seen.sum() < max(10, int(fps)):
        return result
    x = pd.Series(np.where(seen, scan.world_x[window], np.nan)).interpolate(limit_direction="both").to_numpy()
    y = pd.Series(np.where(seen, scan.world_y[window], np.nan)).interpolate(limit_direction="both").to_numpy()
    # A fast pan can make the background estimate jump for a frame or two.
    # Those jumps are short velocity spikes; a running median removes them
    # while keeping the swimmer's sustained movement.
    width = max(3, int(round(0.30 * fps)) | 1)
    vx = pd.Series(np.diff(x, prepend=x[0])).rolling(width, center=True, min_periods=1).median().to_numpy()
    vy = pd.Series(np.diff(y, prepend=y[0])).rolling(width, center=True, min_periods=1).median().to_numpy()
    px, py = np.cumsum(vx), np.cumsum(vy)
    points = np.c_[px - px.mean(), py - py.mean()]
    _, _, axes = np.linalg.svd(points, full_matrices=False)
    result[window] = points @ axes[0]
    return result


@dataclass(frozen=True)
class _TurnCandidate:
    pos: int
    kind: int          # +1 progress maximum, -1 minimum
    prominence: float  # fraction of the race's progress range


def _turn_candidates(progress: np.ndarray, observed: np.ndarray, times: np.ndarray, fps: float,
                     begin: int, end: int) -> list[_TurnCandidate]:
    from scipy.signal import find_peaks

    segment = progress[begin:end + 1]
    if np.isnan(segment).all():
        return []
    smooth = pd.Series(segment).rolling(max(3, int(round(0.5 * fps))), center=True, min_periods=1).mean().to_numpy()
    span = float(np.nanmax(smooth) - np.nanmin(smooth))
    if span <= 0:
        return []
    candidates: list[_TurnCandidate] = []
    edge = int(round(1.5 * fps))
    for kind in (1, -1):
        peaks, info = find_peaks(kind * smooth, prominence=0.10 * span, distance=max(1, int(round(3.0 * fps))))
        for peak, prominence in zip(peaks, info["prominences"]):
            if peak < edge or peak > len(smooth) - 1 - edge:
                continue
            candidates.append(_TurnCandidate(begin + int(peak), kind, float(prominence / span)))
    return sorted(candidates, key=lambda c: c.pos)


def _place_turn(progress: np.ndarray, scan: VideoScan, fps: float, pos: int, kind: int,
                flip_turn: bool) -> int:
    """Time the wall contact near a progress extreme.

    Flip turns (freestyle, backstroke): the feet meet the wall at the push-off,
    which is when the turn's splash gives way to the quiet underwater glide.
    Close to the camera the swimmer is often clipped by the frame edge, so
    the position extreme alone runs late.

    Open turns (butterfly, breaststroke): the hands touch where the swimmer
    is closest to the wall, the last visible extreme of the progress.
    """
    n = len(progress)
    reach = int(round(1.2 * fps))
    lo, hi = max(0, pos - reach), min(n - 1, pos + reach)
    if flip_turn:
        approach = scan.activity[max(0, pos - int(round(4.0 * fps))):max(0, pos - int(round(1.0 * fps)))]
        approach = approach[np.isfinite(approach)]
        if approach.size >= int(fps):
            active = float(np.percentile(approach, 60))
            quiet = _quiet_onset(scan.activity, active, lo, hi, fps)
            if quiet is not None:
                return quiet
    seen = [i for i in range(lo, hi + 1) if scan.observed[i] and np.isfinite(progress[i])]
    if not seen:
        return pos
    values = pd.Series(progress[seen]).rolling(max(1, int(round(0.2 * fps))), center=True,
                                               min_periods=1).mean().to_numpy()
    return int(seen[int(np.argmax(kind * values))])


def _quiet_onset(activity: np.ndarray, active: float, lo: int, hi: int, fps: float) -> int | None:
    """First frame of an underwater glide: busy splash, then a stretch of calm."""
    if active < 0.02:
        return None
    hold = max(2, int(round(0.35 * fps)))
    before = max(1, int(round(0.5 * fps)))
    for i in range(lo, hi + 1):
        calm = activity[i:i + hold]
        busy = activity[max(0, i - before):i]
        if calm.size < hold or not np.isfinite(calm).all() or not np.isfinite(busy).any():
            continue
        if float(np.mean(calm)) < 0.3 * active and float(np.nanmax(busy)) >= 0.5 * active:
            return i
    return None


def _split_penalty(splits: list[float]) -> float:
    """How implausible a set of length times is for one race (0 = typical)."""
    if not splits or min(splits) <= 0:
        return 10.0
    if len(splits) == 1:
        return 0.0
    later = np.array(splits[1:])
    spread = float(np.std(later) / np.mean(later)) if len(later) > 1 else 0.0
    ratio = splits[0] / float(np.median(later))
    # The first length includes the dive, so it is normally 5-25% faster.
    return 2.0 * spread + 3.0 * max(0.0, 0.60 - ratio) + 3.0 * max(0.0, ratio - 1.10)


def choose_turns(progress: np.ndarray, scan: VideoScan, times: np.ndarray, fps: float,
                 expected: int, start: int, end: int, finish: int | None, flip_turn: bool) -> list[int]:
    """Pick and time the expected number of wall turns between ``start`` and ``end``."""
    if expected <= 0:
        return []
    candidates = _turn_candidates(progress, scan.observed, times, fps, start, end)
    if finish is not None:
        candidates = [c for c in candidates if c.pos < finish - int(round(2.0 * fps))]
    if not candidates:
        return []
    race_span = times[finish if finish is not None else end] - times[start]
    minimum_gap = max(4.0, 0.35 * race_span / (expected + 1))
    if len(candidates) > 16:
        candidates = sorted(sorted(candidates, key=lambda c: c.prominence, reverse=True)[:16], key=lambda c: c.pos)
    best: tuple[float, tuple[_TurnCandidate, ...]] | None = None
    for count in range(min(expected, len(candidates)), 0, -1):
        for choice in combinations(candidates, count):
            if times[choice[0].pos] - times[start] < minimum_gap:
                continue
            if any(times[b.pos] - times[a.pos] < minimum_gap for a, b in zip(choice, choice[1:])):
                continue
            if any(a.kind == b.kind for a, b in zip(choice, choice[1:])):
                continue
            boundaries = [times[start], *(times[c.pos] for c in choice)]
            if finish is not None and count == expected:
                boundaries.append(times[finish])
            splits = list(np.diff(boundaries))
            score = sum(c.prominence for c in choice) - 0.6 * _split_penalty(splits)
            if best is None or score > best[0]:
                best = score, choice
        if best is not None:
            break
    if best is None:
        return []
    placed = [_place_turn(progress, scan, fps, c.pos, c.kind, flip_turn) for c in best[1]]
    return sorted(placed)


# ---------------------------------------------------------------------------
# Finish
# ---------------------------------------------------------------------------
def detect_finish(scan: VideoScan, times: np.ndarray, fps: float, after: int,
                  earliest_s: float) -> FinishTiming | None:
    """Find the touch as the moment splash stops for good in the final length.

    ``after`` is the last turn (or the start); ``earliest_s`` is the earliest
    plausible touch time, which rejects the quiet underwater phase after a
    wall and an implausibly short final length.
    """
    n = len(times)
    activity = pd.Series(scan.activity).rolling(max(3, int(round(0.30 * fps))), center=True,
                                                min_periods=2).mean().to_numpy()
    active_from = after + int(round(2.5 * fps))
    active_values = activity[active_from:]
    active_values = active_values[np.isfinite(active_values)]
    if active_values.size < int(fps):
        return None
    active = float(np.percentile(active_values, 60))
    if active < 0.02:
        return None
    quiet_threshold = max(0.012, 0.30 * active)
    hold = max(3, int(round(0.80 * fps)))
    look_back = max(2, int(round(0.60 * fps)))
    first = max(after + look_back, int(np.searchsorted(times, earliest_s)))
    for pos in range(first, n - hold + 1):
        window = activity[pos:pos + hold]
        finite = np.isfinite(window)
        if finite.mean() < 0.6 or (window[finite] < quiet_threshold).mean() < 0.85:
            continue
        before = activity[pos - look_back:pos]
        if not np.isfinite(before).any() or np.nanmax(before) < 0.5 * active:
            continue
        quiet = float(np.nanmedian(window))
        midpoint = quiet + 0.5 * (active - quiet)
        # The smoothed curve finds the stop; the raw signal places it. The
        # touch is the last frame that still shows stroke splash.
        touch = pos
        for candidate in range(min(n - 1, pos + hold // 2), pos - look_back - 1, -1):
            if np.isfinite(scan.activity[candidate]) and scan.activity[candidate] >= midpoint:
                touch = candidate
                break
        near = slice(max(0, touch - int(round(0.3 * fps))), min(n, touch + int(round(0.3 * fps)) + 1))
        if scan.roi_observed[near].mean() >= 0.5:
            return FinishTiming(touch, "high", "the swimmer's splash stopped at the wall and stayed still")
        return FinishTiming(touch, "low",
                            "splash stopped where the swimmer was last seen, but the swimmer was not visible at the touch")
    return None
