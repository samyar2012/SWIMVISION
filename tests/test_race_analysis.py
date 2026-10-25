"""Race-analysis tests, including every supported distance through 200."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest

from core.models import RaceInfo
from core.race_analysis import (
    RaceAnalysis,
    analyze_race,
    detect_race_start,
    detect_turns,
)
from core.video import VideoMetadata
from storage.analyses import AnalysisRecord, load_analysis, save_analysis


def _synthetic_track(lengths: int, fps: float = 10.0, seconds_per_length: float = 8.0,
                     width: int = 320) -> pd.DataFrame:
    pre_frames = int(2 * fps)
    length_frames = int(seconds_per_length * fps)
    rows = []
    for frame in range(pre_frames + lengths * length_frames + 1):
        if frame < pre_frames:
            x = 40.0
        else:
            race_frame = frame - pre_frames
            length_number = min(lengths - 1, race_frame // length_frames)
            progress = (race_frame % length_frames) / length_frames
            if race_frame == lengths * length_frames:
                length_number, progress = lengths - 1, 1.0
            x = 40 + 240 * (progress if length_number % 2 == 0 else 1 - progress)
        rows.append({
            "frame_index": frame, "timestamp_s": frame / fps, "status": "tracked", "source": "test",
            "x1": x - 15, "y1": 70, "x2": x + 15, "y2": 110, "cx": x, "cy": 90,
            "mask_pixels": 0,
        })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("pool,distance,expected_lengths", [
    ("SCY", 25, 1), ("SCY", 50, 2), ("SCY", 100, 4), ("SCY", 200, 8),
    ("SCM", 25, 1), ("SCM", 50, 2), ("SCM", 100, 4), ("SCM", 200, 8),
    ("LCM", 50, 1), ("LCM", 100, 2), ("LCM", 200, 4),
])
@pytest.mark.parametrize("stroke", ["Freestyle", "Backstroke", "Breaststroke", "Butterfly"])
def test_path_events_support_every_individual_event_through_200(pool, distance, expected_lengths, stroke):
    race = RaceInfo(stroke, distance, pool)
    table = _synthetic_track(expected_lengths)
    start, confidence, _ = detect_race_start(table, 10.0, 320, 180)
    turns = detect_turns(table, race.turns, 10.0, 320, start, len(table) - 1)
    assert confidence in {"high", "medium"}
    assert len(turns) == race.turns


@pytest.mark.parametrize("pool,distance,lengths", [
    ("SCY", 100, 4), ("SCY", 200, 8), ("SCM", 100, 4), ("SCM", 200, 8), ("LCM", 200, 4),
])
def test_im_path_support_through_200(pool, distance, lengths):
    race = RaceInfo("Individual Medley", distance, pool)
    table = _synthetic_track(lengths)
    start, _, _ = detect_race_start(table, 10.0, 320, 180)
    assert len(detect_turns(table, race.turns, 10.0, 320, start, len(table) - 1)) == race.turns


def test_turn_detector_withholds_splits_when_expected_turns_are_missing():
    race = RaceInfo("Freestyle", 200, "SCY")
    table = _synthetic_track(2)  # footage only contains a 50, not a 200
    start, _, _ = detect_race_start(table, 10.0, 320, 180)
    assert detect_turns(table, race.turns, 10.0, 320, start, len(table) - 1) == []


def _write_motion_video(path, table: pd.DataFrame, fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 180))
    assert writer.isOpened()
    for i, row in table.iterrows():
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        x = int(row.cx)
        # Alternating arm marker creates a repeatable swimmer-crop motion signal.
        offset = 13 if (i // 5) % 2 else -13
        cv2.rectangle(frame, (x - 12, 76), (x + 12, 105), (80, 120, 190), -1)
        cv2.circle(frame, (x + offset, 72), 6, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def test_full_report_round_trip_and_length_metrics(tmp_path):
    race = RaceInfo("Freestyle", 50, "SCY")
    table = _synthetic_track(race.lengths)
    video = tmp_path / "race.avi"
    _write_motion_video(video, table)
    meta = VideoMetadata(fps=10.0, frame_count=len(table), duration_s=len(table) / 10,
                         width=320, height=180, codec="MJPG")
    report = analyze_race(video, table, race, meta)
    assert len(report.lengths) == 2
    assert [length.number for length in report.lengths] == [1, 2]
    assert all(length.distance == 25 and length.unit == "yd" for length in report.lengths)
    assert report.events[0].kind == "start" and report.events[-1].kind == "finish"
    assert RaceAnalysis.from_dict(report.to_dict()) == report
    record = AnalysisRecord("analysis123", "race.avi", "uploads/race.avi", race, meta,
                            performance=report)
    records = tmp_path / "records"
    save_analysis(record, records)
    assert load_analysis(record.analysis_id, records).performance == report


def test_report_never_calls_estimated_tracking_direct_evidence(tmp_path):
    race = RaceInfo("Freestyle", 25, "SCY")
    table = _synthetic_track(1)
    table.loc[::2, "status"] = "estimated"
    video = tmp_path / "estimated.avi"
    _write_motion_video(video, table)
    meta = VideoMetadata(fps=10.0, frame_count=len(table), duration_s=len(table) / 10,
                         width=320, height=180, codec="MJPG")
    report = analyze_race(video, table, race, meta)
    assert 0.49 <= report.track_observed_fraction <= 0.51
    assert any("estimated positions" in finding.title.lower() for finding in report.findings)
