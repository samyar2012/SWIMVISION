"""Tests for race models and JSON storage."""

from __future__ import annotations

import pytest

from core.models import RaceInfo
from core.video import VideoMetadata
from storage.analyses import AnalysisRecord, list_analyses, load_analysis, new_analysis_id, save_analysis


def test_race_label_uses_pool_units():
    assert RaceInfo("Freestyle", 50, "SCY").label == "50 yd Freestyle (SCY)"
    assert RaceInfo("Freestyle", 50, "LCM").label == "50 m Freestyle (LCM)"


def test_race_support_level():
    assert RaceInfo("Freestyle", 25, "SCY").support_level == "supported"
    assert RaceInfo("Butterfly", 25, "SCY").support_level == "experimental"


@pytest.mark.parametrize("args", [("Dolphin", 50, "SCY"), ("Freestyle", 75, "SCY"), ("Freestyle", 50, "XYZ")])
def test_invalid_race_values_rejected(args):
    with pytest.raises(ValueError):
        RaceInfo(*args)


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
    for key in ("selection", "checkpoints", "tracking", "tracking_csv", "annotated_video"):
        data.pop(key)
    (tmp_path / f"{rec.analysis_id}.json").write_text(json.dumps(data))
    loaded = load_analysis(rec.analysis_id, tmp_path)
    assert loaded.selection is None and loaded.tracking is None and loaded.checkpoints == []


def test_record_with_tracking_round_trips(tmp_path):
    from core.geometry import BBox
    from core.tracker import (STATUS_TRACKED, SwimmerSelection, TrackingResult, build_observations,
                              FollowResult, summarize_tracking)
    rec = _record()
    rec.selection = SwimmerSelection(10, BBox(1, 2, 3, 4), "manual")
    rec.checkpoints = [rec.selection, SwimmerSelection(40, BBox(5, 6, 9, 9), "manual")]
    obs = build_observations([0, 1], 30.0, [FollowResult(1, BBox(0, 0, 5, 5), 0.9), None])
    assert obs[0].status == STATUS_TRACKED
    rec.tracking = summarize_tracking(TrackingResult(obs, "visual_tracker", 1, ["n"], 0.5, {"a": 1}))
    rec.tracking_csv = "data/tracking/x.csv"
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
