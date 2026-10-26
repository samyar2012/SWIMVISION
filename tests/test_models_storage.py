"""Tests for race models and JSON storage."""

from __future__ import annotations

import pytest

from core.models import RaceInfo
from core.video import VideoMetadata
from storage.analyses import AnalysisRecord, list_analyses, load_analysis, new_analysis_id, save_analysis


def test_race_label_uses_pool_units():
    assert RaceInfo("Freestyle", 50, "SCY").label == "50 yd Freestyle (SCY)"
    assert RaceInfo("Freestyle", 50, "LCM").label == "50 m Freestyle (LCM)"


def test_all_strokes_accepted():
    from core.models import STROKES
    for stroke in STROKES:
        assert RaceInfo(stroke, 200, "SCY").stroke == stroke


@pytest.mark.parametrize("args", [
    ("Dolphin", 50, "SCY"), ("Freestyle", 75, "SCY"), ("Freestyle", 50, "XYZ"),
    ("Freestyle", 25, "LCM"),            # a 50 m pool has no 25 events
    ("Individual Medley", 200, "SCY"),   # only the four strokes are supported
    ("Freestyle", 1650, "SCM"),          # 1650 is a yards event
    ("Freestyle", 1500, "SCY"),
    ("Butterfly", 400, "SCY"),
])
def test_invalid_race_values_rejected(args):
    with pytest.raises(ValueError):
        RaceInfo(*args)


def test_every_listed_event_is_valid():
    from core.models import EVENTS
    count = 0
    for pool, strokes in EVENTS.items():
        for stroke, distances in strokes.items():
            for d in distances:
                race = RaceInfo(stroke, d, pool)
                assert race.lengths * race.pool_length == d
                count += 1
    assert count > 50


@pytest.mark.parametrize("stroke, distance, pool, lengths, turns", [
    ("Freestyle", 50, "SCY", 2, 1), ("Freestyle", 1650, "SCY", 66, 65),
    ("Freestyle", 1500, "LCM", 30, 29), ("Freestyle", 50, "LCM", 1, 0),
    ("Butterfly", 200, "SCM", 8, 7),
])
def test_lengths_and_turns(stroke, distance, pool, lengths, turns):
    race = RaceInfo(stroke, distance, pool)
    assert (race.lengths, race.turns) == (lengths, turns)


def test_only_the_four_strokes_are_offered():
    from core.models import STROKES
    assert STROKES == ("Freestyle", "Backstroke", "Breaststroke", "Butterfly")


@pytest.mark.parametrize("stroke, flip", [
    ("Freestyle", True), ("Backstroke", True), ("Breaststroke", False), ("Butterfly", False),
])
def test_turn_style_follows_the_stroke(stroke, flip):
    assert RaceInfo(stroke, 100, "SCY").flip_turns is flip


def _record() -> AnalysisRecord:
    return AnalysisRecord(
        analysis_id=new_analysis_id(),
        original_filename="race.mp4",
        video_path="uploads/abc.mp4",
        race=RaceInfo("Freestyle", 50, "SCY"),
        video=VideoMetadata(fps=29.97, frame_count=900, duration_s=30.03, width=1920, height=1080, codec="avc1"),
    )


def test_save_and_load_round_trip(tmp_path):
    rec = _record()
    save_analysis(rec, tmp_path)
    assert load_analysis(rec.analysis_id, tmp_path) == rec


def test_phase1_records_without_tracking_still_load(tmp_path):
    """Analyses saved before Phase 2 have no selection/tracking keys."""
    import json
    rec = _record()
    data = rec.to_dict()
    for key in ("selections", "tracking", "tracking_csv", "annotated_video"):
        data.pop(key)
    (tmp_path / f"{rec.analysis_id}.json").write_text(json.dumps(data))
    loaded = load_analysis(rec.analysis_id, tmp_path)
    assert loaded.selections == [] and loaded.tracking is None


def test_old_tracking_format_is_dropped_but_old_box_is_kept(tmp_path):
    """Records from the removed tracking code keep the user's box; their old results are ignored."""
    import json
    rec = _record()
    data = rec.to_dict()
    data["selection"] = {"anchor_frame": 430, "bbox": [330, 133, 399, 278], "source": "manual", "confidence": None}
    data["tracking"] = {"mode": "sam2", "n_samples": 10}
    data["tracking_csv"] = "data/tracking/old.csv"
    data["annotated_video"] = "outputs/annotated/old.mp4"
    (tmp_path / f"{rec.analysis_id}.json").write_text(json.dumps(data))
    loaded = load_analysis(rec.analysis_id, tmp_path)
    assert [m.frame_index for m in loaded.selections] == [430]
    assert loaded.selections[0].box.to_list() == [330, 133, 399, 278]
    assert loaded.tracking is None and loaded.tracking_csv is None and loaded.annotated_video is None


def test_single_mark_tracking_record_still_loads(tmp_path):
    """Version-2 records (one 'selection') from earlier today become one mark."""
    import json
    from core.geometry import BBox
    from core.sam2_tracking import FrameResult, compute_stats
    rec = _record()
    data = rec.to_dict()
    data.pop("selections")
    data["selection"] = {"frame_index": 5, "box": [1, 2, 30, 40]}
    stats = compute_stats([FrameResult(5, 0.1, BBox(1, 2, 30, 40), 9)], 1 / 30).to_dict()
    stats.pop("conflict_frames")
    data["tracking"] = {"version": 2, "selection": data["selection"], "start_frame": 5, "end_frame": 5,
                        "stride": 1, "analysis_width": 640, "analysis_height": 360, "model": "m", "device": "cpu",
                        "half_precision": False, "runtime_s": 1.0, "stats": stats, "output_video": {},
                        "debug_dir": None}
    (tmp_path / f"{rec.analysis_id}.json").write_text(json.dumps(data))
    loaded = load_analysis(rec.analysis_id, tmp_path)
    assert [m.frame_index for m in loaded.selections] == [5]
    assert loaded.tracking.selections[0].frame_index == 5 and loaded.tracking.stats.conflict_frames == 0


def test_record_with_tracking_round_trips(tmp_path):
    from core.geometry import BBox
    from core.sam2_tracking import FrameResult, Selection, TrackingRun, compute_stats
    rec = _record()
    rec.selections = [Selection(10, BBox(1, 2, 30, 40)), Selection(20, BBox(5, 6, 30, 40))]
    results = [FrameResult(10, 0.33, BBox(1, 2, 30, 40), 500), FrameResult(11, 0.37, None, 0)]
    rec.tracking = TrackingRun(
        selections=tuple(rec.selections), start_frame=10, end_frame=11, stride=1, analysis_width=640,
        analysis_height=360, model="sam2.1_b.pt", device="cpu", half_precision=False, runtime_s=1.5,
        stats=compute_stats(results, 1 / 30), output_video={"frames": 2, "codec": "h264"}, debug_dir=None)
    rec.tracking_csv = "data/tracking/x.csv"
    rec.annotated_video = "outputs/annotated/x.mp4"
    save_analysis(rec, tmp_path)
    assert load_analysis(rec.analysis_id, tmp_path) == rec


def test_cleanup_removes_only_old_unsaved_uploads(tmp_path):
    import os, time
    from storage.analyses import cleanup_orphan_uploads

    uploads, previews, analyses = tmp_path / "up", tmp_path / "prev", tmp_path / "an"
    for d in (uploads, previews, analyses):
        d.mkdir()
    old = time.time() - 7200

    kept_saved = uploads / "saved1.mp4"      # referenced by an analysis -> keep
    orphan_old = uploads / "orphan1.mp4"     # unsaved + old -> delete
    orphan_new = uploads / "orphan2.mp4"     # unsaved but recent -> keep
    orphan_prev = previews / "orphan1_preview.mp4"
    for f in (kept_saved, orphan_old, orphan_prev):
        f.write_bytes(b"x")
        os.utime(f, (old, old))
    orphan_new.write_bytes(b"x")

    rec = _record()
    rec.video_path = "uploads/saved1.mp4"
    save_analysis(rec, analyses)

    deleted = cleanup_orphan_uploads(3600, uploads, previews, analyses)
    assert {p.name for p in deleted} == {"orphan1.mp4", "orphan1_preview.mp4"}
    assert kept_saved.exists() and orphan_new.exists()


def test_list_analyses_skips_corrupt_files(tmp_path):
    rec = _record()
    save_analysis(rec, tmp_path)
    (tmp_path / "garbage.json").write_text("{not json")
    assert [r.analysis_id for r in list_analyses(tmp_path)] == [rec.analysis_id]
