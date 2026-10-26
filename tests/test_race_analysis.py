"""Race timing, stroke rate and report wording on synthetic races with known answers.

``scan_video`` and ``detect_start`` read real footage; here they are replaced
by a synthetic scan so every split, rate and touch has an exact expected value.
The real-footage checks live in ``test_real_video_regression.py``.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import core.race_analysis as race_analysis
from core.models import RaceInfo
from core.race_analysis import (
    ANALYSIS_FORMAT_VERSION,
    LengthMetrics,
    RaceAnalysis,
    RaceEvent,
    analyze_race,
    measure_stroke_rates,
)
from core.race_report import describe_race, format_time, strokes_word
from core.race_timing import StartTiming, VideoScan, _split_penalty, detect_finish, find_start_beep
from core.video import VideoMetadata
from storage.analyses import AnalysisRecord, load_analysis, save_analysis
from ui.analysis import _parse_race_time, plain_report

FPS = 30.0
STROKE_HZ = {"Freestyle": 1.9, "Backstroke": 1.6, "Butterfly": 0.9, "Breaststroke": 0.95}


def _race_scan(start: float, walls: list[float], finish: float | None, end: float, stroke_hz: float,
               fps: float = FPS, seed: int = 0) -> tuple[VideoScan, np.ndarray]:
    """A swimmer swimming lengths between ``walls`` with splash bursts at ``stroke_hz``.

    Before the start the swimmer is still on the block. Each length begins
    with a quiet underwater glide and the race ends with a still touch.
    """
    rng = np.random.default_rng(seed)
    times = np.arange(int(round(end * fps)) + 1) / fps
    boundaries = [start, *walls, finish if finish is not None else end]
    x = np.zeros(len(times))
    activity = np.full(len(times), 0.004)
    crop = np.full(len(times), 0.01)
    bursts = (0.5 + 0.5 * np.sin(2 * np.pi * stroke_hz * times)) ** 4
    for number, (a, b) in enumerate(zip(boundaries, boundaries[1:])):
        inside = (times >= a) & (times <= b)
        fraction = (times[inside] - a) / (b - a)
        x[inside] = 1000.0 * (fraction if number % 2 == 0 else 1.0 - fraction)
        swimming = inside & (times >= a + 1.5)
        activity[swimming] = 0.08 + 0.06 * bursts[swimming]
        crop[swimming] = 0.05 + 0.08 * bursts[swimming]
        if number == 0:
            dive = (times >= a) & (times < a + 0.8)
            activity[dive] = 0.15
    if finish is not None:
        x[times > finish] = x[np.searchsorted(times, finish) - 1]
    activity = activity + rng.normal(0, 0.003, len(times))
    crop = crop + rng.normal(0, 0.004, len(times))
    seen = np.ones(len(times), dtype=bool)
    return VideoScan(crop, activity, x, np.full(len(times), 200.0), seen, seen.copy()), times


def _track(times: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "frame_index": np.arange(len(times)), "timestamp_s": times, "status": "tracked",
        "source": "swimmer", "x1": 100.0, "y1": 100.0, "x2": 160.0, "y2": 140.0, "cx": 130.0, "cy": 120.0,
    })


def _meta(times: np.ndarray) -> VideoMetadata:
    return VideoMetadata(fps=FPS, frame_count=len(times), duration_s=float(times[-1]),
                         width=848, height=464, codec="avc1")


def _run(monkeypatch, race: RaceInfo, start: float, walls: list[float], finish: float | None,
         end: float, *, beep: bool = True, start_guess: float | None = None,
         official_time_s: float | None = None) -> RaceAnalysis:
    scan, times = _race_scan(start, walls, finish, end, STROKE_HZ[race.stroke])
    monkeypatch.setattr(race_analysis, "scan_video", lambda *_args, **_kwargs: scan)
    movement = int(np.searchsorted(times, start + 0.36))

    def fake_start(_video, _track, _scan, _fps) -> StartTiming:
        if beep:
            return StartTiming(int(np.searchsorted(times, start)), start, "beep", "high", "tone", movement)
        guess = start_guess if start_guess is not None else start
        return StartTiming(int(np.searchsorted(times, guess)), guess, "movement", "medium", "movement", movement)

    monkeypatch.setattr(race_analysis, "detect_start", fake_start)
    return analyze_race("unused.mp4", _track(times), race, _meta(times), official_time_s=official_time_s)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def test_50_freestyle_splits_total_reaction_and_stroke_rate(monkeypatch):
    report = _run(monkeypatch, RaceInfo("Freestyle", 50, "SCY"), 2.0, [16.0], 32.0, 35.0)

    assert report.status == "complete"
    assert [length.number for length in report.lengths] == [1, 2]
    assert report.lengths[0].split_s == pytest.approx(14.0, abs=0.1)
    assert report.lengths[1].split_s == pytest.approx(16.0, abs=0.1)
    assert report.lengths[1].cumulative_s == pytest.approx(30.0, abs=0.1)
    assert report.race_time_source == "video" and report.race_time_s == pytest.approx(30.0, abs=0.1)
    assert report.lengths[0].pace_per_100_s == pytest.approx(report.lengths[0].split_s * 4, abs=0.02)
    assert report.reaction_s == pytest.approx(0.36, abs=0.04)
    for length in report.lengths:
        assert length.stroke_rate_spm == pytest.approx(114.0, abs=3.0)
    assert [event.kind for event in report.events] == ["start", "takeoff", "turn", "finish"]
    assert report.finish.confidence == "high"


@pytest.mark.parametrize(("stroke", "distance", "splits"), [
    ("Freestyle", 25, [13.0]),
    ("Backstroke", 100, [15.0, 17.0, 17.4, 17.8]),
    ("Butterfly", 100, [14.5, 17.0, 17.6, 18.2]),
    ("Breaststroke", 200, [17.0, 19.5, 19.8, 20.0, 20.1, 20.3, 20.4, 20.0]),
    ("Freestyle", 200, [13.0, 15.0, 15.4, 15.6, 15.7, 15.8, 15.6, 15.2]),
])
def test_every_turn_and_split_is_found(monkeypatch, stroke, distance, splits):
    race = RaceInfo(stroke, distance, "SCY")
    assert race.lengths == len(splits)
    walls = list(2.0 + np.cumsum(splits))
    report = _run(monkeypatch, race, 2.0, walls[:-1], walls[-1], walls[-1] + 3.0)

    assert report.status == "complete"
    assert [length.split_s for length in report.lengths] == pytest.approx(splits, abs=0.25)
    assert report.race_time_s == pytest.approx(sum(splits), abs=0.15)
    expected_rate = STROKE_HZ[stroke] * 60
    assert report.average_stroke_rate_spm == pytest.approx(expected_rate, abs=0.05 * expected_rate)


def test_official_time_completes_the_race_from_the_beep(monkeypatch):
    report = _run(monkeypatch, RaceInfo("Freestyle", 50, "SCY"), 2.0, [16.0], 32.0, 35.0, official_time_s=30.5)

    assert report.race_time_source == "official" and report.race_time_s == 30.5
    assert report.video_race_time_s == pytest.approx(30.0, abs=0.1)
    assert report.lengths[1].split_s == pytest.approx(16.5, abs=0.1)
    assert not any("different" in note for note in report.notes)
    assert "your official time" in report.summary


def test_official_time_anchors_an_estimated_start_to_the_seen_touch(monkeypatch):
    report = _run(monkeypatch, RaceInfo("Freestyle", 50, "SCY"), 2.0, [16.0], 32.0, 35.0,
                  beep=False, start_guess=2.4, official_time_s=30.2)

    assert report.start.timestamp_s == pytest.approx(1.8, abs=0.05)
    assert report.lengths[0].split_s == pytest.approx(14.2, abs=0.1)
    assert report.lengths[1].cumulative_s == pytest.approx(30.2, abs=0.05)
    assert report.reaction_s is None
    assert any("No start beep" in note for note in report.notes)


def test_official_time_that_does_not_match_the_video_is_flagged(monkeypatch):
    report = _run(monkeypatch, RaceInfo("Freestyle", 50, "SCY"), 2.0, [16.0], 32.0, 35.0, official_time_s=35.0)

    assert report.race_time_s == 35.0
    assert any("different from what the video shows" in note for note in report.notes)


def test_unseen_finish_withholds_the_total_and_last_split(monkeypatch):
    # The video stops while the swimmer is still racing.
    report = _run(monkeypatch, RaceInfo("Freestyle", 50, "SCY"), 2.0, [16.0], None, 29.0)

    assert report.race_time_s is None and report.race_time_source == "unknown"
    assert report.status == "partial"
    assert [length.number for length in report.lengths] == [1]
    assert report.lengths[0].split_s == pytest.approx(14.0, abs=0.1)
    assert all(event.kind != "finish" for event in report.events)
    assert "not visible" in report.summary
    assert any("Enter your official time" in note for note in report.notes)


def test_missing_turns_withhold_splits_instead_of_guessing(monkeypatch):
    # A 100 was entered, but the footage only contains two lengths.
    report = _run(monkeypatch, RaceInfo("Freestyle", 100, "SCY"), 2.0, [16.0], 32.0, 35.0)

    assert report.lengths == ()
    assert report.status == "partial"
    assert any("of the 3 turns were visible" in note for note in report.notes)


def test_rejects_an_invalid_official_time(monkeypatch):
    with pytest.raises(Exception, match="greater than zero"):
        _run(monkeypatch, RaceInfo("Freestyle", 50, "SCY"), 2.0, [16.0], 32.0, 35.0, official_time_s=-1.0)


def test_finish_is_low_confidence_when_the_swimmer_is_not_visible_at_the_touch():
    scan, times = _race_scan(2.0, [], 16.0, 19.0, 1.9)
    hidden = replace(scan, roi_observed=np.where(times > 14.5, False, True))

    seen = detect_finish(scan, times, FPS, 0, 10.0)
    unseen = detect_finish(hidden, times, FPS, 0, 10.0)

    assert seen is not None and seen.confidence == "high"
    assert abs(times[seen.pos] - 16.0) <= 0.1
    assert unseen is not None and unseen.confidence == "low"


def test_split_plausibility_prefers_a_normal_race_shape():
    assert _split_penalty([14.0]) == 0.0
    assert _split_penalty([]) == 10.0
    assert _split_penalty([14.0, 16.0]) < _split_penalty([7.0, 23.0])
    assert _split_penalty([14.0, 16.0, 16.2, 16.4]) < _split_penalty([14.0, 9.0, 23.0, 16.4])


# ---------------------------------------------------------------------------
# Stroke rate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("stroke", "hz"), [
    ("Freestyle", 1.9), ("Backstroke", 1.5), ("Butterfly", 0.9), ("Breaststroke", 1.0),
])
def test_stroke_rate_matches_the_stroke_rhythm(stroke, hz):
    scan, times = _race_scan(0.0, [], None, 20.0, hz)
    window = (int(4 * FPS), int(19 * FPS))

    race_rate, (rate,) = measure_stroke_rates(scan, FPS, [window], stroke)

    assert race_rate == pytest.approx(hz * 60, abs=0.03 * hz * 60)
    assert rate == pytest.approx(hz * 60, abs=0.03 * hz * 60)


def test_stroke_rate_ignores_a_strong_unrelated_wobble():
    scan, times = _race_scan(0.0, [], None, 20.0, 1.9)
    window = slice(int(4 * FPS), int(19 * FPS))
    # A camera sway with no harmonics, stronger than the stroke's own
    # fundamental: the single tallest spectral peak belongs to the sway.
    wobble = 0.04 * np.sin(2 * np.pi * 1.45 * times)
    swaying = replace(scan, crop_motion=scan.crop_motion + wobble, activity=scan.activity + 0.5 * wobble)
    spectrum = np.abs(np.fft.rfft(swaying.crop_motion[window] - swaying.crop_motion[window].mean(), n=8192))
    frequencies = np.fft.rfftfreq(8192, 1 / FPS)
    band = (frequencies > 1.1) & (frequencies < 2.6)
    assert frequencies[band][np.argmax(spectrum[band])] == pytest.approx(1.45, abs=0.05)

    race_rate, _ = measure_stroke_rates(swaying, FPS, [(window.start, window.stop)], "Freestyle")

    assert race_rate == pytest.approx(114.0, abs=3.0)


def test_stroke_rate_is_not_reported_for_a_too_short_window():
    scan, _ = _race_scan(0.0, [], None, 20.0, 1.9)
    race_rate, rates = measure_stroke_rates(scan, FPS, [(150, 180), (120, 570)], "Freestyle")

    assert rates[0] is None
    assert rates[1] == pytest.approx(114.0, abs=3.0) and race_rate == pytest.approx(114.0, abs=3.0)


# ---------------------------------------------------------------------------
# Start beep (real audio decoding through ffmpeg)
# ---------------------------------------------------------------------------
def _video_with_audio(path, audio: np.ndarray, rate: int = 16000) -> None:
    import imageio_ffmpeg
    from scipy.io import wavfile

    wav = path.with_suffix(".wav")
    wavfile.write(wav, rate, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
    duration = len(audio) / rate
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error",
        "-f", "lavfi", "-i", f"color=c=blue:s=160x120:r=10:d={duration:.2f}", "-i", str(wav),
        "-shortest", "-c:v", "mpeg4", "-c:a", "aac", "-b:a", "128k", str(path),
    ], check=True, capture_output=True, timeout=60)


def _crowd_noise(seconds: float, rate: int = 16000, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 0.02, int(seconds * rate))


def test_start_beep_onset_is_found_in_the_audio(tmp_path):
    rate = 16000
    audio = _crowd_noise(6.0)
    t = np.arange(int(0.5 * rate)) / rate
    tone = sum(np.sin(2 * np.pi * 600 * k * t) for k in range(1, 11)) * 0.02
    begin = int(2.5 * rate)
    audio[begin:begin + len(tone)] += tone
    video = tmp_path / "beep.mp4"
    _video_with_audio(video, audio)

    found = find_start_beep(video, (1.5, 3.5))

    assert found is not None
    onset, lines = found
    assert onset == pytest.approx(2.5, abs=0.08)
    assert lines >= 8


@pytest.mark.parametrize("kind", ["noise", "whistle"])
def test_noise_or_a_single_whistle_is_not_a_start_beep(tmp_path, kind):
    rate = 16000
    audio = _crowd_noise(6.0)
    if kind == "whistle":
        t = np.arange(int(0.8 * rate)) / rate
        start = int(2.5 * rate)
        audio[start:start + len(t)] += 0.2 * np.sin(2 * np.pi * 2800 * t)
    video = tmp_path / f"{kind}.mp4"
    _video_with_audio(video, audio)

    assert find_start_beep(video, (1.5, 3.5)) is None


# ---------------------------------------------------------------------------
# Report wording
# ---------------------------------------------------------------------------
def _analysis(splits: list[float], rates: list[float | None], reaction: float | None = 0.4,
              stroke: str = "Freestyle", distance: int = 50) -> RaceAnalysis:
    lengths, clock = [], 0.0
    for number, (split, rate) in enumerate(zip(splits, rates), start=1):
        clock += split
        lengths.append(LengthMetrics(
            number=number, stroke=stroke, distance=25, unit="yd", start_frame=0, end_frame=0,
            start_s=clock - split, end_s=clock, split_s=split, cumulative_s=round(clock, 2),
            average_speed=round(25 / split, 3), pace_per_100_s=round(split * 4, 2), stroke_rate_spm=rate,
            confidence="high",
        ))
    event = RaceEvent("start", "Start beep", 0, 0.0, "high", "tone")
    known = [rate for rate in rates if rate]
    return describe_race(RaceAnalysis(
        race_label=f"{distance} SCY {stroke}", stroke=stroke, distance=distance, unit="yd", pool_length=25,
        status="complete", start=event, finish=replace(event, kind="finish", label="Finish"),
        events=(event,), lengths=tuple(lengths), expected_lengths=len(splits),
        race_time_s=round(clock, 2), race_time_source="video", video_race_time_s=round(clock, 2),
        official_time_s=None, reaction_s=reaction, start_method="beep",
        average_stroke_rate_spm=round(float(np.mean(known)), 1) if known else None,
        track_observed_fraction=0.97, summary="", technique=(), focus_areas=(), notes=(),
    ))


_DEVELOPER_WORDS = re.compile(r"\b(cycles?|priority|working well|confidence|detector|frames?|"
                              r"tracker|signal|heatmap|estimate[sd]?)\b", re.IGNORECASE)


def _all_text(report: RaceAnalysis) -> str:
    parts = [report.summary, *report.notes]
    for note in report.technique:
        parts += [note.topic, note.observation, note.meaning]
    for area in report.focus_areas:
        parts += [area.title, area.why, area.how, *area.look_up]
    return "\n".join(parts)


@pytest.mark.parametrize("report", [
    _analysis([14.0, 15.0], [112.0, 113.0], reaction=0.25),
    _analysis([14.4, 16.6], [113.0, 115.0]),
    _analysis([14.5, 17.5], [56.0, 48.0], reaction=0.7, stroke="Butterfly"),
    _analysis([13.0, 15.0, 15.2, 16.8], [118.0, 116.0, 114.0, 100.0], distance=100),
    _analysis([14.0, 16.0], [None, None], reaction=None),
])
def test_report_reads_like_a_coach_not_a_debug_log(report):
    text = _all_text(report)

    assert not _DEVELOPER_WORDS.search(text), _DEVELOPER_WORDS.search(text)
    assert report.summary and report.focus_areas
    assert 1 <= len(report.focus_areas) <= 4
    assert all(area.look_up and area.why and area.how for area in report.focus_areas)
    assert {note.tone for note in report.technique} <= {"strength", "work_on", "neutral"}


def test_report_summary_states_time_pace_and_splits():
    report = _analysis([14.42, 16.58], [113.0, 115.0])

    assert "31.00" in report.summary
    assert "per 100 yards" in report.summary
    assert "14.42" in report.summary and "16.58" in report.summary
    assert "114 arm strokes per minute" in report.summary


def test_fading_second_length_is_the_first_thing_to_work_on():
    report = _analysis([14.5, 17.5], [56.0, 54.0], stroke="Butterfly")

    pacing = next(note for note in report.technique if note.topic == "Pacing")
    assert pacing.tone == "work_on"
    assert "1.8 seconds" in pacing.meaning  # 3.0 s slower minus the 8% dive allowance
    assert report.focus_areas[0].title == "Holding your speed in the second half"
    assert "butterfly race pace sets" in report.focus_areas[0].look_up


def test_dropping_tempo_and_slow_reaction_become_focus_areas():
    report = _analysis([13.0, 15.0, 15.2, 16.8], [118.0, 116.0, 114.0, 100.0], reaction=0.62, distance=100)
    titles = [area.title for area in report.focus_areas]

    assert "Keeping your tempo when tired" in titles
    assert "Reacting to the start" in titles
    tempo = next(note for note in report.technique if note.topic == "Stroke rate")
    assert "116 to 100" in tempo.meaning  # compared with length 2, which has no dive


def test_well_paced_race_is_praised_and_still_gets_a_next_step():
    report = _analysis([14.0, 15.0], [112.0, 113.0], reaction=0.25)

    assert {note.topic: note.tone for note in report.technique} == {
        "The race": "strength", "Start": "strength", "Pacing": "strength", "Stroke rate": "strength",
    }
    assert [area.title for area in report.focus_areas] == ["Building more speed"]


def test_missing_measurements_are_left_out_not_guessed():
    report = _analysis([14.0, 16.0], [None, None], reaction=None)

    assert {note.topic for note in report.technique} == {"The race", "Pacing"}
    assert "per minute" not in report.summary


@pytest.mark.parametrize(("seconds", "text"), [(31.0, "31.00"), (59.996, "1:00.00"), (62.45, "1:02.45"),
                                               (125.0, "2:05.00"), (None, "—")])
def test_format_time(seconds, text):
    assert format_time(seconds) == text


def test_strokes_word():
    assert strokes_word("Freestyle") == strokes_word("Backstroke") == "arm strokes"
    assert strokes_word("Butterfly") == strokes_word("Breaststroke") == "strokes"


def test_length_label_is_ordinary_english():
    from core.race_report import length_label
    assert length_label(1, 25, "yd") == "1st 25 yards"
    assert length_label(2, 25, "m") == "2nd 25 meters"
    assert length_label(8, 50, "m") == "8th 50 meters"


def test_race_story_names_each_split_in_plain_words():
    report = _analysis([14.42, 16.58], [113.0, 115.0])
    story = next(note for note in report.technique if note.topic == "The race")
    assert "1st 25 yards in 14.42" in story.observation
    assert "2nd 25 yards in 16.58" in story.observation
    assert "2.16 seconds slower" in story.observation


# ---------------------------------------------------------------------------
# Storage and export
# ---------------------------------------------------------------------------
def test_report_round_trips_through_storage_with_the_official_time(tmp_path):
    report = _analysis([14.42, 16.58], [113.0, 115.0])
    race = RaceInfo("Freestyle", 50, "SCY")
    meta = VideoMetadata(fps=FPS, frame_count=1635, duration_s=54.5, width=848, height=464, codec="avc1")
    record = AnalysisRecord("analysis123", "race.mp4", "uploads/race.mp4", race, meta,
                            performance=report, official_time_s=31.0)

    save_analysis(record, tmp_path)
    loaded = load_analysis("analysis123", tmp_path)

    assert loaded.performance == report
    assert loaded.official_time_s == 31.0
    assert RaceAnalysis.from_dict(report.to_dict()) == report


def test_reports_from_older_versions_are_not_loaded():
    data = _analysis([14.0, 16.0], [110.0, 112.0]).to_dict()

    assert RaceAnalysis.from_dict({**data, "format_version": ANALYSIS_FORMAT_VERSION - 1}) is None
    broken = dict(data)
    del broken["lengths"]
    assert RaceAnalysis.from_dict(broken) is None


def test_plain_text_export_has_splits_feedback_and_look_ups():
    exported = plain_report(_analysis([14.5, 17.5], [56.0, 54.0], stroke="Butterfly"))

    assert "SPLITS" in exported and "Length 2: 17.50" in exported
    assert "HOW THE RACE WENT" in exported and "WHAT TO WORK ON" in exported
    assert "You can look up sets for:" in exported
    assert not _DEVELOPER_WORDS.search(exported)


@pytest.mark.parametrize(("entered", "seconds"), [
    ("54.32", 54.32),
    ("1:54.32", 114.32),
    (" 2:03.5 ", 123.5),
])
def test_official_time_accepts_stopwatch_formats(entered, seconds):
    assert _parse_race_time(entered) == seconds


@pytest.mark.parametrize("entered", ["", "1:75", "1:2:3", "words", "0"])
def test_official_time_rejects_invalid_values(entered):
    with pytest.raises(ValueError):
        _parse_race_time(entered)
