"""Distance-aware race analysis built on the persistent swimmer track.

Measurements are kept separate from coaching words. This module times the
race (start, turns, finish), splits it into pool lengths and measures stroke
rhythm; ``core.race_report`` turns those numbers into plain-language feedback.
Nothing is invented: a split is only reported when both of its walls were
found, and the finish is only reported when it was seen or the official time
was entered.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from core.errors import SwimVisionError
from core.models import RaceInfo
from core.race_timing import (
    FinishTiming,
    VideoScan,
    choose_turns,
    detect_finish,
    detect_start,
    progress_signal,
    scan_video,
)
from core.video import VideoMetadata

ANALYSIS_FORMAT_VERSION = 4
SUPPORTED_ANALYSIS_DISTANCES = (25, 50, 100, 200)
ProgressFn = Callable[[int, int], None]


@dataclass(frozen=True)
class RaceEvent:
    kind: str
    label: str
    frame_index: int
    timestamp_s: float
    confidence: str
    evidence: str


@dataclass(frozen=True)
class LengthMetrics:
    number: int
    stroke: str
    distance: int
    unit: str
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    split_s: float
    cumulative_s: float
    average_speed: float              # pool units per second
    pace_per_100_s: float             # this length's speed expressed as time per 100
    stroke_rate_spm: float | None     # arm strokes (free/back) or strokes (fly/breast) per minute
    confidence: str


@dataclass(frozen=True)
class TechniqueNote:
    topic: str
    tone: str          # "strength", "work_on" or "neutral" (styling only, never shown as a label)
    observation: str   # what the video showed, with numbers
    meaning: str       # what that means for the swimmer


@dataclass(frozen=True)
class FocusArea:
    title: str
    why: str
    how: str
    look_up: tuple[str, ...]


@dataclass(frozen=True)
class RaceAnalysis:
    race_label: str
    stroke: str
    distance: int
    unit: str
    pool_length: int
    status: str                        # "complete" or "partial"
    start: RaceEvent
    finish: RaceEvent
    events: tuple[RaceEvent, ...]
    lengths: tuple[LengthMetrics, ...]
    expected_lengths: int
    race_time_s: float | None
    race_time_source: str              # "official", "video" or "unknown"
    video_race_time_s: float | None
    official_time_s: float | None
    reaction_s: float | None           # start tone to first movement on the block
    start_method: str                  # "beep", "movement" or "clip"
    average_stroke_rate_spm: float | None
    track_observed_fraction: float
    summary: str
    technique: tuple[TechniqueNote, ...]
    focus_areas: tuple[FocusArea, ...]
    notes: tuple[str, ...]
    format_version: int = ANALYSIS_FORMAT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RaceAnalysis | None":
        """Rebuild a saved analysis; ``None`` for reports saved by older versions."""
        if data.get("format_version") != ANALYSIS_FORMAT_VERSION:
            return None
        values = dict(data)
        try:
            values["start"] = RaceEvent(**data["start"])
            values["finish"] = RaceEvent(**data["finish"])
            values["events"] = tuple(RaceEvent(**e) for e in data["events"])
            values["lengths"] = tuple(LengthMetrics(**m) for m in data["lengths"])
            values["technique"] = tuple(TechniqueNote(**n) for n in data["technique"])
            values["focus_areas"] = tuple(
                FocusArea(**{**f, "look_up": tuple(f["look_up"])}) for f in data["focus_areas"]
            )
            values["notes"] = tuple(data["notes"])
            return cls(**values)
        except (KeyError, TypeError):
            return None


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, min(int(window), len(values)))
    if window == 1:
        return values.astype(float, copy=True)
    left, right = window // 2, window - 1 - window // 2
    padded = np.pad(values.astype(float), (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _filled_track(table: pd.DataFrame) -> pd.DataFrame:
    required = {"frame_index", "timestamp_s", "status", "x1", "y1", "x2", "y2", "cx", "cy"}
    missing = required - set(table.columns)
    if missing:
        raise SwimVisionError(f"Tracking data is missing: {', '.join(sorted(missing))}.")
    if len(table) < 10:
        raise SwimVisionError("There are not enough tracked frames to analyse this race.")
    result = table.sort_values("frame_index").drop_duplicates("frame_index").copy()
    for column in ("x1", "y1", "x2", "y2", "cx", "cy"):
        result[column] = pd.to_numeric(result[column], errors="coerce").interpolate(limit_direction="both")
    if result[["cx", "cy"]].isna().any().any():
        raise SwimVisionError("The swimmer path contains no usable positions.")
    if "source" not in result:
        result["source"] = ""
    return result.reset_index(drop=True)


# Plausible race stroke frequencies (Hz). Freestyle and backstroke are measured
# per arm stroke (each arm counts), butterfly and breaststroke per full stroke.
_STROKE_BANDS = {
    "Freestyle": (1.1, 2.6), "Backstroke": (1.1, 2.6),
    "Butterfly": (0.55, 1.4), "Breaststroke": (0.55, 1.4),
}
_SPECTRUM_SIZE = 8192


def _normalised_rhythm(values: np.ndarray, fps: float) -> np.ndarray:
    """Remove slow trends and scale to unit local amplitude so signals can be combined."""
    values = np.nan_to_num(np.asarray(values, dtype=float))
    residual = values - _moving_average(values, max(3, int(round(fps))))
    scale = _moving_average(np.abs(residual), max(3, int(round(3 * fps))))
    return residual / (scale + 1e-6)


def measure_stroke_rates(scan: VideoScan, fps: float, windows: list[tuple[int, int]],
                         stroke: str) -> tuple[float | None, list[float | None]]:
    """Race stroke rate and the rate in each window, in strokes per minute.

    Splash and body movement repeat once per stroke as sharp bursts, which put
    energy at the stroke frequency and its multiples. Summing the spectrum at
    ``f``, ``2f`` and ``3f`` picks the true stroke frequency even when camera
    pans add stray peaks. The race rate pools every window; each window's own
    rate is then searched within 15% of it, so one noisy length cannot jump
    to an unrelated rhythm.
    """
    combined = _normalised_rhythm(scan.crop_motion, fps) + _normalised_rhythm(scan.activity, fps)
    low, high = _STROKE_BANDS[stroke]
    step = fps / _SPECTRUM_SIZE
    candidates = np.arange(int(np.ceil(low / step)), int(high / step) + 1) * step
    usable = [(a, b) for a, b in windows if b - a >= int(3 * fps)]
    if not usable:
        return None, [None] * len(windows)
    scores = {window: _rhythm_scores(combined[window[0]:window[1]], fps, candidates) for window in usable}
    pooled = sum(score / score.max() for score in scores.values())
    race_frequency = float(candidates[int(np.argmax(pooled))])
    nearby = np.abs(candidates - race_frequency) <= 0.15 * race_frequency
    rates: list[float | None] = []
    for window in windows:
        if window not in scores:
            rates.append(None)
            continue
        local = np.where(nearby, scores[window], -np.inf)
        rates.append(float(candidates[int(np.argmax(local))]) * 60.0)
    return race_frequency * 60.0, rates


def _rhythm_scores(segment: np.ndarray, fps: float, candidates: np.ndarray) -> np.ndarray:
    segment = (segment - segment.mean()) * np.hanning(len(segment))
    amplitude = np.abs(np.fft.rfft(segment, n=_SPECTRUM_SIZE))
    step = fps / _SPECTRUM_SIZE
    reach = max(1, int(round(0.03 / step)))

    def strength(frequency: float) -> float:
        k = int(round(frequency / step))
        return float(amplitude[max(0, k - reach):k + reach + 1].max()) if k < len(amplitude) else 0.0

    return np.array([sum(strength(f * h) / np.sqrt(h) for h in (1, 2, 3)) for f in candidates])


def _position(times: np.ndarray, t: float) -> int:
    return int(np.clip(np.searchsorted(times, t), 0, len(times) - 1))


def _swimming_window(times: np.ndarray, begin_s: float, end_s: float) -> tuple[int, int]:
    """Part of a length that is certainly surface swimming.

    Underwater phases are limited to 15 m (60% of a 25), and in practice end
    well before 45% of the length's time; the last half second is the touch
    or turn approach.
    """
    return _position(times, begin_s + 0.45 * (end_s - begin_s)), _position(times, end_s - 0.5)


def analyze_race(video_path: str | Path, table: pd.DataFrame, race: RaceInfo,
                 meta: VideoMetadata, progress: ProgressFn | None = None,
                 official_time_s: float | None = None) -> RaceAnalysis:
    """Time the race, split it into lengths and describe it in plain language."""
    from core.race_report import describe_race

    if race.distance not in SUPPORTED_ANALYSIS_DISTANCES:
        raise SwimVisionError("Race analysis currently supports 25, 50, 100 and 200 yard/metre events.")
    if official_time_s is not None and (not np.isfinite(official_time_s) or official_time_s <= 0):
        raise SwimVisionError("The entered official race time must be greater than zero seconds.")
    video_path = Path(video_path)
    track = _filled_track(table)
    times = track.timestamp_s.to_numpy(float)
    fps = 1.0 / max(1e-6, float(np.median(np.diff(times))))
    last = len(track) - 1

    scan = scan_video(video_path, track, progress)
    start = detect_start(video_path, track, scan, fps)
    start_t = start.time_s

    route = progress_signal(scan, fps, start.pos, last)
    provisional_turns = choose_turns(route, scan, times, fps, race.turns, start.pos, last, None, race.flip_turns)
    finish = _find_finish(scan, times, fps, start.pos, start_t, provisional_turns, race)
    if finish is not None:
        turns = choose_turns(route, scan, times, fps, race.turns, start.pos, finish.pos, finish.pos,
                             race.flip_turns)
        if len(turns) == race.turns:
            refined = _find_finish(scan, times, fps, start.pos, start_t, turns, race)
            finish = refined or finish
    else:
        turns = provisional_turns
    turn_times = [float(times[pos]) for pos in turns]
    all_turns_found = len(turns) == race.turns

    # --- Race time and finish ------------------------------------------------
    video_time = round(float(times[finish.pos]) - start_t, 3) if finish and finish.confidence == "high" else None
    finish_t: float | None
    notes: list[str] = []
    if official_time_s is not None:
        if start.method != "beep" and video_time is not None:
            # The touch was seen clearly but the start was estimated from movement:
            # anchor the start to the official time instead.
            start_t = round(float(times[finish.pos]) - official_time_s, 3)
            finish_t = float(times[finish.pos])
        else:
            finish_t = start_t + official_time_s
        finish_evidence = "your official time"
        finish_confidence = "high"
        if video_time is not None and abs(video_time - official_time_s) > max(1.5, 0.05 * official_time_s):
            notes.append(
                f"The time you entered ({official_time_s:.2f} s) is {abs(video_time - official_time_s):.1f} s "
                f"different from what the video shows ({video_time:.2f} s). The splits use your official time "
                "for the total; check that the time belongs to this swim."
            )
        race_time, race_time_source = official_time_s, "official"
    elif finish is not None and finish.confidence == "high":
        finish_t = float(times[finish.pos])
        finish_evidence, finish_confidence = finish.evidence, "high"
        race_time, race_time_source = finish_t - start_t, "video"
    else:
        finish_t = None
        finish_evidence = ("the swimmer could not be seen at the final touch"
                           if finish is None else finish.evidence)
        finish_confidence = "low"
        race_time, race_time_source = None, "unknown"
        notes.append("The finish touch could not be seen clearly. Enter your official time to complete the "
                     "last split and the total.")
    if finish_t is not None and turn_times and finish_t <= turn_times[-1] + 2.0:
        finish_t, race_time, race_time_source = None, None, "unknown"
        finish_confidence = "low"
        notes.append("The finish could not be matched to the last turn. Check the official time you entered.")

    # --- Lengths ---------------------------------------------------------------
    boundaries = [start_t, *turn_times, finish_t if finish_t is not None else float(times[last])]
    windows = [_swimming_window(times, a, b) for a, b in zip(boundaries, boundaries[1:])]
    race_rate, window_rates = measure_stroke_rates(scan, fps, windows, race.stroke)
    lengths: list[LengthMetrics] = []
    if all_turns_found:
        if finish_t is None:
            boundaries = boundaries[:-1]
        for number, (a, b) in enumerate(zip(boundaries, boundaries[1:]), start=1):
            a_pos, b_pos = _position(times, a), _position(times, b)
            rate = window_rates[number - 1]
            lengths.append(LengthMetrics(
                number=number, stroke=race.stroke, distance=race.pool_length, unit=race.unit,
                start_frame=int(track.frame_index.iloc[a_pos]), end_frame=int(track.frame_index.iloc[b_pos]),
                start_s=round(a, 3), end_s=round(b, 3), split_s=round(b - a, 2),
                cumulative_s=round(b - start_t, 2), average_speed=round(race.pool_length / (b - a), 3),
                pace_per_100_s=round((b - a) * 100.0 / race.pool_length, 2),
                stroke_rate_spm=round(rate, 1) if rate else None,
                confidence="high",
            ))
    elif race.turns:
        notes.append(f"Only {len(turns)} of the {race.turns} turns were visible, so the time for each length "
                     "is not shown. A video that keeps both walls in view gives full splits.")

    rates = [length.stroke_rate_spm for length in lengths if length.stroke_rate_spm]
    if rates:
        average_rate = round(float(np.mean(rates)), 1)
    else:
        average_rate = round(race_rate, 1) if race_rate else None

    reaction = None
    if start.method == "beep" and start.movement_pos is not None:
        reaction = round(float(times[start.movement_pos]) - start.time_s, 2)
    elif start.method != "beep":
        notes.append("No start beep could be heard in the video, so your reaction time is not measured.")

    # --- Events ----------------------------------------------------------------
    def event(kind: str, label: str, t: float, confidence: str, evidence: str) -> RaceEvent:
        pos = _position(times, t)
        return RaceEvent(kind, label, int(track.frame_index.iloc[pos]), round(float(t), 3), confidence, evidence)

    start_label = "Start beep" if start.method == "beep" else "Start"
    start_event = event("start", start_label, start_t, start.confidence, start.evidence)
    finish_event = event("finish", "Finish", finish_t if finish_t is not None else float(times[last]),
                         finish_confidence, finish_evidence)
    events = [start_event]
    if start.movement_pos is not None:
        events.append(event("takeoff", "Left the block", float(times[start.movement_pos]), "medium",
                            "first movement from the still start position"))
    for number, t in enumerate(turn_times, start=1):
        events.append(event("turn", f"Turn {number}", t, "high" if all_turns_found else "medium",
                            "the swimmer reversed direction at the wall"))
    if finish_t is not None:
        events.append(finish_event)
    events.sort(key=lambda item: item.timestamp_s)

    observed_fraction = float(scan.observed.mean())
    status = "complete" if all_turns_found and finish_t is not None else "partial"
    analysis = RaceAnalysis(
        race_label=race.label, stroke=race.stroke, distance=race.distance, unit=race.unit,
        pool_length=race.pool_length, status=status, start=start_event, finish=finish_event,
        events=tuple(events), lengths=tuple(lengths), expected_lengths=race.lengths,
        race_time_s=round(race_time, 2) if race_time is not None else None,
        race_time_source=race_time_source,
        video_race_time_s=video_time, official_time_s=official_time_s,
        reaction_s=reaction, start_method=start.method,
        average_stroke_rate_spm=average_rate,
        track_observed_fraction=round(observed_fraction, 4),
        summary="", technique=(), focus_areas=(), notes=tuple(notes),
    )
    return describe_race(analysis)


def _find_finish(scan: VideoScan, times: np.ndarray, fps: float, start_pos: int, start_t: float,
                 turns: list[int], race: RaceInfo) -> FinishTiming | None:
    """Look for the touch only where the final length could plausibly end."""
    if turns:
        after = turns[-1]
        lengths_done = len(turns)
        average = (float(times[after]) - start_t) / lengths_done
        earliest = float(times[after]) + 0.85 * average
    else:
        after = start_pos
        # A one-length race: no length can be swum in under ~9 seconds.
        earliest = start_t + max(8.0, 0.30 * race.pool_length)
    return detect_finish(scan, times, fps, after, earliest)
