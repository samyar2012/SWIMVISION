from __future__ import annotations

from dataclasses import dataclass

from core.geometry import BBox
from core.hybrid_tracking import track_swimmer
from core.sam2_tracking import Selection
from core.video import read_video_metadata


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    source: str


class FakeHybridDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if self.calls == 3:  # splash: force one motion-predicted frame
            return []
        x1 = 40 + self.calls * 3
        return [Detection(x1, 80, x1 + 40, 120, 0.9, "swimmer")]


def test_tracks_from_one_visual_selection_without_lane_or_frame_inputs(mp4_30fps, tmp_path) -> None:
    meta = read_video_metadata(mp4_30fps)
    selection = Selection(0, BBox(40, 80, 80, 120))
    output = tmp_path / "hybrid.mp4"

    results, run = track_swimmer(
        mp4_30fps,
        meta,
        selection,
        end_frame=5,
        annotated_path=output,
        detector=FakeHybridDetector(),
    )

    assert len(results) == 6
    assert all(result.tracked for result in results)
    assert results[2].source.startswith("predicted:")
    assert run.selections == (selection,)
    assert run.stats.tracked_fraction == 1.0
    assert run.output_video["frames"] == 6
    assert output.is_file()
