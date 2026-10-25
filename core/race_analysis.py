"""Distance-aware race analysis built on the persistent swimmer track.

The module deliberately separates *measurements* from *coaching*.  It never
invents missing turns or evenly divides a race when the video does not support
that conclusion.  The first version uses the tracked path and a swimmer-crop
motion signal; future pose models can replace the signal without changing the
saved report format or UI.

Research structure followed here:

* SwimTrack: explicit race events and stroke definitions.
* DeepDASH: analyse a temporal sequence of swimmer-only crops.
* Woinoski & Bajic: smooth a per-frame stroke signal, then find periodic peaks.
* Hall et al.: convert detections into a temporal race report and retain the
  evidence used for each event.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd

from core.errors import SwimVisionError
from core.models import IM_ORDER, RaceInfo
from core.video import VideoMetadata

ANALYSIS_FORMAT_VERSION = 1
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
    split_s: float
    average_speed: float
    stroke_cycles: int | None
    stroke_rate_spm: float | None
    distance_per_cycle: float | None
    confidence: str


@dataclass(frozen=True)
class Finding:
    tone: str  # "good", "improve", or "check"
    title: str
    explanation: str
    action: str


@dataclass(frozen=True)
class TrainingSet:
    name: str
    purpose: str
    set_text: str
    success_check: str


@dataclass(frozen=True)
class RaceAnalysis:
    race_label: str
    status: str
    start: RaceEvent
    finish: RaceEvent
    events: tuple[RaceEvent, ...]
    lengths: tuple[LengthMetrics, ...]
    race_time_s: float
    stroke_cycles: int | None
    arm_strokes: int | None
    average_stroke_rate_spm: float | None
    track_observed_fraction: float
    findings: tuple[Finding, ...]
    training_sets: tuple[TrainingSet, ...]
    nutrition_tips: tuple[str, ...]
    limitations: tuple[str, ...]
    format_version: int = ANALYSIS_FORMAT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RaceAnalysis | None":
        if data.get("format_version") != ANALYSIS_FORMAT_VERSION:
            return None
        return cls(
            race_label=data["race_label"], status=data["status"],
            start=RaceEvent(**data["start"]), finish=RaceEvent(**data["finish"]),
            events=tuple(RaceEvent(**e) for e in data["events"]),
            lengths=tuple(LengthMetrics(**m) for m in data["lengths"]),
            race_time_s=float(data["race_time_s"]),
            stroke_cycles=data.get("stroke_cycles"), arm_strokes=data.get("arm_strokes"),
            average_stroke_rate_spm=data.get("average_stroke_rate_spm"),
            track_observed_fraction=float(data["track_observed_fraction"]),
            findings=tuple(Finding(**f) for f in data["findings"]),
            training_sets=tuple(TrainingSet(**s) for s in data["training_sets"]),
            nutrition_tips=tuple(data["nutrition_tips"]),
            limitations=tuple(data["limitations"]),
            format_version=int(data["format_version"]),
        )


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
    return result.reset_index(drop=True)


def detect_race_start(table: pd.DataFrame, fps: float, width: int, height: int) -> tuple[int, str, str]:
    """Return row position, confidence and readable evidence for movement onset."""
    x, y = table.cx.to_numpy(float), table.cy.to_numpy(float)
    smooth_window = max(3, int(round(0.25 * fps)))
    x, y = _moving_average(x, smooth_window), _moving_average(y, smooth_window)
    lag = max(1, int(round(0.20 * fps)))
    speed = np.zeros(len(table), dtype=float)
    scale = max(1.0, float(np.hypot(width, height)))
    speed[lag:] = np.hypot(x[lag:] - x[:-lag], y[lag:] - y[:-lag]) / (lag / fps) / scale
    baseline_n = min(max(int(2.0 * fps), 10), max(10, len(speed) // 4))
    baseline = speed[:baseline_n]
    median = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - median)))
    threshold = max(0.025, median + 8.0 * max(mad, 0.001))
    hold = max(2, int(round(0.30 * fps)))
    moving = speed > threshold
    first_allowed = min(len(table) - 1, max(1, int(round(0.5 * fps))))

    # The persistent tracker explicitly changes from its on-deck person state
    # to its in-water swimmer state. This is a much stronger race anchor than
    # raw screen coordinates when the camera pans or zooms before the start.
    # The take-off precedes water entry by roughly one dive phase, so mark a
    # conservative 0.7 seconds before the first locked swimmer observation.
    # Audio-buzzer timing remains a future, separately-labelled metric.
    if "source" in table:
        sources = table.source.fillna("").astype(str).str.lower()
        in_water = sources.str.contains("swimmer") & ~sources.str.contains("visual-person")
        water_rows = np.flatnonzero(in_water.to_numpy())
        if water_rows.size and water_rows[0] >= first_allowed:
            inferred = max(first_allowed, int(water_rows[0]) - int(round(0.7 * fps)))
            return inferred, "medium", "take-off inferred immediately before the track changed to an in-water swimmer"

    speed_candidate: int | None = None
    for i in range(first_allowed, max(first_allowed + 1, len(table) - hold + 1)):
        if moving[i:i + hold].mean() >= 0.8:
            speed_candidate = i
            break

    # A panning camera can keep the swimmer horizontally centred and suppress
    # apparent speed. The dive still creates a sustained departure from the
    # swimmer's pre-race screen position, especially vertically. Compare with
    # a robust baseline rather than one first frame so small on-block motion is
    # not mistaken for the start.
    base_x, base_y = float(np.median(x[:baseline_n])), float(np.median(y[:baseline_n]))
    displacement = np.hypot(x - base_x, y - base_y) / scale
    base_displacement = displacement[:baseline_n]
    # Require a large change (roughly a body-length at normal framing). The
    # selected box can drift on a waiting swimmer for several seconds; a lower
    # threshold produced a false start in the full Video(2) regression.
    displacement_threshold = max(0.11, float(np.percentile(base_displacement, 98)) + 0.06)
    away = displacement > displacement_threshold
    displacement_candidate: int | None = None
    for i in range(first_allowed, max(first_allowed + 1, len(table) - hold + 1)):
        if away[i:i + hold].mean() >= 0.8:
            displacement_candidate = i
            break

    candidates = [i for i in (speed_candidate, displacement_candidate) if i is not None]
    if candidates:
        found = min(candidates)
        if found == displacement_candidate and found != speed_candidate:
            return found, "high", "sustained dive movement away from the pre-race starting position"
        confidence = "high" if speed[found:found + hold].mean() >= threshold * 1.5 else "medium"
        return found, confidence, "first sustained movement away from the starting position"
    # Honest fallback: the tracked clip may begin after the race started.
    return 0, "low", "the tracked clip begins with movement; start may be before this frame"


def _turn_candidates(x: np.ndarray, fps: float, width: int, start: int, finish: int) -> list[tuple[int, float, int]]:
    smooth = _moving_average(x, max(5, int(round(0.65 * fps))))
    span = max(2, int(round(0.65 * fps)))
    radius = max(span + 1, int(round(1.8 * fps)))
    minimum_prominence = max(12.0, width * 0.035)
    candidates: list[tuple[int, float, int]] = []
    for i in range(max(start + radius, span), min(finish - radius, len(x) - span)):
        before = smooth[i] - smooth[i - span]
        after = smooth[i + span] - smooth[i]
        if before == 0 or after == 0 or np.sign(before) == np.sign(after):
            continue
        local = smooth[i - radius:i + radius + 1]
        prominence = float(max(smooth[i] - local.min(), local.max() - smooth[i]))
        # A real wall turn has movement into and out of an extremum, not one noisy derivative.
        travel = float(abs(before) + abs(after))
        if prominence >= minimum_prominence and travel >= minimum_prominence * 0.35:
            candidates.append((i, prominence + travel, 1 if before > 0 else -1))

    # Collapse the cluster of extrema around one physical turn.
    merged: list[tuple[int, float, int]] = []
    merge_gap = max(2, int(round(1.5 * fps)))
    for candidate in candidates:
        if merged and candidate[0] - merged[-1][0] <= merge_gap:
            if candidate[1] > merged[-1][1]:
                merged[-1] = candidate
        else:
            merged.append(candidate)
    return merged


def detect_turns(table: pd.DataFrame, expected: int, fps: float, width: int,
                 start_pos: int, finish_pos: int) -> list[int]:
    """Choose the strongest time-ordered, alternating direction changes."""
    if expected == 0:
        return []
    candidates = _turn_candidates(table.cx.to_numpy(float), fps, width, start_pos, finish_pos)
    if len(candidates) < expected:
        return []
    minimum_gap = max(2, int(round(3.0 * fps)))
    best: tuple[float, tuple[tuple[int, float, int], ...]] | None = None
    # Races through 200 have at most seven turns, keeping this bounded. Trim noisy
    # candidate lists before combinations so analysis stays responsive.
    if len(candidates) > 18:
        candidates = sorted(sorted(candidates, key=lambda c: c[1], reverse=True)[:18])
    for choice in combinations(candidates, expected):
        if any(b[0] - a[0] < minimum_gap for a, b in zip(choice, choice[1:])):
            continue
        if any(a[2] == b[2] for a, b in zip(choice, choice[1:])):
            continue
        score = sum(c[1] for c in choice)
        if best is None or score > best[0]:
            best = score, choice
    return [c[0] for c in best[1]] if best else []


def _crop_motion_signal(video_path: Path, table: pd.DataFrame,
                        progress: ProgressFn | None = None) -> np.ndarray:
    """DeepDASH-style temporal input reduced to a deterministic motion signal.

    Each tracked swimmer crop is resized to the same dimensions.  Frame-to-frame
    absolute difference measures body/arm activity while mostly cancelling the
    swimmer's translation, because the persistent track recentres every crop.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SwimVisionError("The original video could not be opened for race analysis.")
    motion = np.zeros(len(table), dtype=float)
    previous: np.ndarray | None = None
    try:
        previous_frame_index: int | None = None
        for row_pos, row in table.iterrows():
            wanted_frame = int(row.frame_index)
            if previous_frame_index is None or wanted_frame <= previous_frame_index:
                cap.set(cv2.CAP_PROP_POS_FRAMES, wanted_frame)
            else:
                for _ in range(previous_frame_index + 1, wanted_frame):
                    if not cap.grab():
                        raise SwimVisionError(f"Could not advance to frame {wanted_frame} during analysis.")
            ok, frame = cap.read()
            if not ok or frame is None:
                raise SwimVisionError(f"Could not read frame {wanted_frame} during analysis.")
            previous_frame_index = wanted_frame
            h, w = frame.shape[:2]
            bw, bh = row.x2 - row.x1, row.y2 - row.y1
            x1 = max(0, int(round(row.x1 - 0.20 * bw)))
            x2 = min(w, int(round(row.x2 + 0.20 * bw)))
            y1 = max(0, int(round(row.y1 - 0.20 * bh)))
            y2 = min(h, int(round(row.y2 + 0.20 * bh)))
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                gray = cv2.cvtColor(cv2.resize(crop, (96, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                if previous is not None:
                    motion[row_pos] = float(cv2.absdiff(gray, previous).mean())
                previous = gray
            if progress and (row_pos % 10 == 0 or row_pos + 1 == len(table)):
                progress(row_pos + 1, len(table))
    finally:
        cap.release()
    return motion


def detect_stroke_cycles(signal: np.ndarray, fps: float, start: int, finish: int,
                         excluded: list[int], stroke: str) -> list[int]:
    """Find periodic stroke-cycle peaks after temporal smoothing.

    Freestyle/backstroke peaks represent a same-arm cycle (two arm strokes).
    Breaststroke/butterfly peaks represent one complete stroke cycle.
    """
    fast = _moving_average(signal, max(1, int(round(0.12 * fps))))
    baseline = _moving_average(fast, max(3, int(round(0.9 * fps))))
    activity = fast - baseline
    valid = np.ones(len(activity), dtype=bool)
    valid[:start] = False
    valid[finish + 1:] = False
    turn_radius = int(round(1.25 * fps))
    for turn in excluded:
        valid[max(0, turn - turn_radius):min(len(valid), turn + turn_radius + 1)] = False
    values = activity[valid]
    if values.size < max(10, int(fps * 2)) or float(np.std(values)) < 0.15:
        return []

    # Estimate the cycle duration from autocorrelation, then peak-pick with a
    # minimum spacing. This mirrors the filtered temporal-signal approach in
    # published race-video stroke-rate work without claiming its trained model.
    centered = values - float(np.mean(values))
    min_period = 0.55 if stroke in {"Freestyle", "Backstroke"} else 0.65
    max_period = 2.2
    lo, hi = max(2, int(min_period * fps)), min(len(centered) // 2, int(max_period * fps))
    if hi <= lo:
        return []
    correlations = np.correlate(centered, centered, mode="full")[len(centered) - 1:]
    lag = lo + int(np.argmax(correlations[lo:hi + 1]))
    min_gap = max(2, int(round(lag * 0.65)))
    threshold = float(np.percentile(values, 58))
    candidates = [i for i in range(start + 1, finish) if valid[i] and activity[i] >= threshold
                  and activity[i] >= activity[i - 1] and activity[i] > activity[i + 1]]
    peaks: list[int] = []
    for candidate in candidates:
        if not peaks or candidate - peaks[-1] >= min_gap:
            peaks.append(candidate)
        elif activity[candidate] > activity[peaks[-1]]:
            peaks[-1] = candidate
    return peaks


def _stroke_for_length(race: RaceInfo, number: int) -> str:
    if race.stroke != "Individual Medley":
        return race.stroke
    lengths_per_leg = race.lengths // 4
    return IM_ORDER[min(3, (number - 1) // lengths_per_leg)]


def _nutrition_tips() -> tuple[str, ...]:
    return (
        "Fuel consistently throughout the day; do not try a new food or drink for the first time on race day.",
        "Before practice or racing, choose familiar carbohydrate-rich food that sits comfortably for you.",
        "Simple pre-practice examples include fruit with toast, oatmeal, or another familiar carbohydrate-rich snack.",
        "After a hard session, pair carbohydrate with a protein food—for example yogurt and fruit, cereal and milk, or a sandwich—then eat a normal balanced meal.",
        "Build regular meals from grains or other carbohydrate foods, a protein food, colorful produce, and a calcium-rich food where appropriate.",
        "Use water as the everyday default. Longer or intense sessions may also need carbohydrate and electrolytes.",
        "Food and hydration needs depend on age, body size, health and training load. A parent/guardian, coach and sports dietitian should personalize them.",
    )


def _coach(race: RaceInfo, lengths: list[LengthMetrics], observed_fraction: float,
           strokes_found: bool) -> tuple[tuple[Finding, ...], tuple[TrainingSet, ...]]:
    findings: list[Finding] = []
    sets: list[TrainingSet] = []
    repeat_distance = race.pool_length  # 25 in short course, 50 in long course
    if lengths:
        splits = np.array([m.split_s for m in lengths], dtype=float)
        spread = float((splits.max() - splits.min()) / max(0.001, splits.mean()))
        if spread <= 0.04:
            findings.append(Finding("good", "Consistent pacing",
                                    "Your length times stayed within 4% of each other.",
                                    "Keep rehearsing the same controlled speed under fatigue."))
        elif len(splits) > 1 and splits[-1] > splits[0] * 1.06:
            loss = splits[-1] - splits[0]
            findings.append(Finding("improve", "Speed faded late",
                                    f"The final length was {loss:.2f} seconds slower than the first.",
                                    "Practice holding race pace after short rest, rather than starting harder."))
            repeat = 8 if race.distance >= 100 else 6
            sets.append(TrainingSet("Race-pace holding", "Keep speed when tired",
                                    f"{repeat} × {repeat_distance} {race.unit} at goal race pace, 20–30 seconds rest.",
                                    "Keep every repeat within 0.3 seconds; stop the set if technique breaks down."))
        else:
            findings.append(Finding("check", "Uneven pacing",
                                    "Length times changed more than 4% across the race.",
                                    "Review the split chart with your coach and choose a steadier opening pace."))
    else:
        findings.append(Finding("check", "Turns need verification",
                                "The video did not show every expected direction change clearly enough for trustworthy splits.",
                                "Use a full side view with both walls visible; SwimVision will rerun automatically."))

    if observed_fraction >= 0.95:
        findings.append(Finding("good", "Strong video evidence",
                                f"The swimmer was directly observed in {observed_fraction:.0%} of analysed frames.",
                                "The path is reliable enough for timing analysis; still review low-confidence events."))
    else:
        findings.append(Finding("check", "Some timing uses estimated positions",
                                f"Only {observed_fraction:.0%} of frames were direct observations.",
                                "Treat metrics around yellow tracking sections as provisional."))

    if not strokes_found:
        findings.append(Finding("check", "Stroke count not reported yet",
                                "The swimmer-crop motion was not periodic enough for a defensible automatic count.",
                                "Do not change technique from a guessed number; use a clearer side view and rerun."))
    else:
        sets.append(TrainingSet("Count-and-hold", "Connect stroke rhythm to speed",
                                f"6 × {repeat_distance} {race.unit}: count cycles and hold the same time with the same or fewer cycles.",
                                "Stop adding effort if the stroke becomes shorter or rushed."))

    if race.turns:
        sets.append(TrainingSet("Wall speed", "Carry speed through every turn",
                                f"6–8 repetitions: swim 5 {race.unit} into the wall, turn, streamline and race 5 {race.unit} out.",
                                "Use one clear cue: tight turn, solid push, then hold the streamline."))
    if not sets:
        sets.append(TrainingSet("Quality speed", "Maintain the strengths shown in this race",
                                f"6 × {repeat_distance} {race.unit} smooth-fast with full technique, 30 seconds rest.",
                                "Finish each repeat with the same stroke shape as the first."))
    return tuple(findings), tuple(sets[:3])


def analyze_race(video_path: str | Path, table: pd.DataFrame, race: RaceInfo,
                 meta: VideoMetadata, progress: ProgressFn | None = None) -> RaceAnalysis:
    """Create an evidence-backed report for a tracked 25/50/100/200 race."""
    if race.distance not in SUPPORTED_ANALYSIS_DISTANCES:
        raise SwimVisionError("Race analysis currently supports 25, 50, 100 and 200 yard/metre events.")
    track = _filled_track(table)
    sample_fps = 1.0 / max(1e-6, float(np.median(np.diff(track.timestamp_s.to_numpy(float)))))
    start_pos, start_confidence, start_evidence = detect_race_start(track, sample_fps, meta.width, meta.height)
    finish_pos = len(track) - 1
    turn_positions = detect_turns(track, race.turns, sample_fps, meta.width, start_pos, finish_pos)

    def event(kind: str, label: str, pos: int, confidence: str, evidence: str) -> RaceEvent:
        row = track.iloc[pos]
        return RaceEvent(kind, label, int(row.frame_index), round(float(row.timestamp_s), 4), confidence, evidence)

    start = event("start", "Estimated takeoff", start_pos, start_confidence, start_evidence)
    finish = event("finish", "Finish", finish_pos, "low",
                   "last tracked frame; wall-contact verification is not available yet")
    turns = tuple(event("turn", f"Turn {n}", pos, "medium",
                        "sustained approach followed by a direction reversal")
                  for n, pos in enumerate(turn_positions, start=1))

    signal = _crop_motion_signal(Path(video_path), track, progress)
    stroke_name = race.stroke if race.stroke != "Individual Medley" else "Freestyle"
    peaks = detect_stroke_cycles(signal, sample_fps, start_pos, finish_pos, turn_positions, stroke_name)
    boundaries = [start_pos, *turn_positions, finish_pos]
    complete_lengths = len(turn_positions) == race.turns
    lengths: list[LengthMetrics] = []
    if complete_lengths:
        for number, (a, b) in enumerate(zip(boundaries, boundaries[1:]), start=1):
            begin, end = track.iloc[a], track.iloc[b]
            split = float(end.timestamp_s - begin.timestamp_s)
            cycles = sum(a <= p < b for p in peaks) if peaks else None
            rate = (cycles * 60.0 / split) if cycles and split > 0 else None
            lengths.append(LengthMetrics(
                number=number, stroke=_stroke_for_length(race, number), distance=race.pool_length,
                unit=race.unit, start_frame=int(begin.frame_index), end_frame=int(end.frame_index),
                split_s=round(split, 3), average_speed=round(race.pool_length / split, 3),
                stroke_cycles=cycles, stroke_rate_spm=round(rate, 1) if rate else None,
                distance_per_cycle=round(race.pool_length / cycles, 2) if cycles else None,
                confidence="medium" if cycles else "low",
            ))

    observed_fraction = float((track.status == "tracked").mean())
    race_time = max(0.0, finish.timestamp_s - start.timestamp_s)
    total_cycles = len(peaks) or None
    average_rate = total_cycles * 60 / race_time if total_cycles and race_time else None
    arm_strokes = total_cycles * 2 if total_cycles and race.stroke in {"Freestyle", "Backstroke"} else total_cycles
    findings, sets = _coach(race, lengths, observed_fraction, bool(peaks))
    limitations = [
        "Start is detected from sustained movement; audio-buzzer timing is not implemented yet.",
        "Finish currently uses the final tracked frame and must be treated as provisional.",
        "Stroke cycles come from swimmer-crop motion, not a swim-specific pose model; review them before acting.",
    ]
    if not complete_lengths:
        limitations.append(f"Detected {len(turn_positions)} of {race.turns} expected turns, so per-length splits are withheld.")
    if race.stroke == "Individual Medley":
        limitations.append("IM stroke transitions are assigned from race order; visual stroke classification is not implemented yet.")
    status = "review_needed" if limitations or not complete_lengths else "complete"
    return RaceAnalysis(
        race_label=race.label, status=status, start=start, finish=finish, events=(start, *turns, finish),
        lengths=tuple(lengths), race_time_s=round(race_time, 3), stroke_cycles=total_cycles,
        arm_strokes=arm_strokes, average_stroke_rate_spm=round(average_rate, 1) if average_rate else None,
        track_observed_fraction=round(observed_fraction, 4), findings=findings, training_sets=sets,
        nutrition_tips=_nutrition_tips(), limitations=tuple(limitations),
    )
