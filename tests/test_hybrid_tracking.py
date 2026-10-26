from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.geometry import BBox
from core.hybrid_tracking import SwimmerHighlight, track_swimmer
from core.sam2_tracking import Selection
from core.swimmer_tracker import TrackedSwimmer
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
        return [Detection(x1, 80, x1 + 40, 120, 0.9, "person")]


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
    assert results[2].source.startswith("estimated:")
    assert results[2].predicted
    assert run.selections == (selection,)
    assert run.stats.tracked_frames == 5
    assert run.stats.estimated_frames == 1
    assert run.stats.lost_frames == 0
    assert run.output_video["frames"] == 6
    assert output.is_file()


def test_live_preview_receives_every_highlighted_frame(mp4_30fps, tmp_path) -> None:
    meta = read_video_metadata(mp4_30fps)
    shown: list[np.ndarray] = []

    track_swimmer(
        mp4_30fps, meta, Selection(0, BBox(40, 80, 80, 120)), end_frame=5,
        annotated_path=tmp_path / "hybrid.mp4", detector=FakeHybridDetector(), preview=shown.append,
    )

    assert len(shown) == 6
    assert all(frame.shape[:2] == (meta.height, meta.width) for frame in shown)


def _changed(before: np.ndarray, after: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(np.any(before != after, axis=2))
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if xs.size else None


def _followed(box: BBox, predicted: bool = False) -> TrackedSwimmer:
    return TrackedSwimmer(box, 0.9, "swimmer", 0.9, predicted, 0, None)


def test_highlight_marks_only_the_swimmer_with_no_text() -> None:
    frame = np.full((360, 640, 3), (120, 90, 30), np.uint8)
    highlight = SwimmerHighlight(30.0)

    drawn = highlight.draw(frame, _followed(BBox(300, 150, 360, 200)))

    x1, y1, x2, y2 = _changed(frame, drawn)
    # Outline, tint and soft glow stay right around the swimmer; there is no
    # status label or banner anywhere else in the frame.
    assert 270 <= x1 < 300 and 120 <= y1 < 150 and 360 < x2 <= 390 and 200 < y2 <= 230
    assert np.array_equal(drawn[:100], frame[:100])


def test_highlight_disappears_when_the_swimmer_is_lost_and_glides_when_followed() -> None:
    frame = np.full((360, 640, 3), (120, 90, 30), np.uint8)
    highlight = SwimmerHighlight(30.0)
    highlight.draw(frame, _followed(BBox(300, 150, 360, 200)))

    assert np.array_equal(highlight.draw(frame, None), frame)

    moved = highlight.draw(frame, _followed(BBox(320, 150, 380, 200)))
    old_edge = _changed(frame, SwimmerHighlight(30.0).draw(frame, _followed(BBox(300, 150, 360, 200))))[0]
    new_edge = _changed(frame, SwimmerHighlight(30.0).draw(frame, _followed(BBox(320, 150, 380, 200))))[0]
    # Between the old and new positions: it glides instead of snapping.
    assert old_edge < _changed(frame, moved)[0] < new_edge


def test_estimated_position_is_drawn_more_softly() -> None:
    frame = np.full((360, 640, 3), (120, 90, 30), np.uint8)
    box = BBox(300, 150, 360, 200)

    seen = SwimmerHighlight(30.0).draw(frame, _followed(box))
    estimated = SwimmerHighlight(30.0).draw(frame, _followed(box, predicted=True))

    strength = lambda image: int(np.abs(image.astype(int) - frame.astype(int)).sum())
    assert 0 < strength(estimated) < strength(seen)
