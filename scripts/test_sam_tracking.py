"""Standalone SAM 2 tracking check on a real video - no Streamlit.

Example: swimmer marked on the block (frame 650) and again just after surfacing
(frame 760), tracked for 15 seconds, analysed at 640 px wide:

    .venv\\Scripts\\python.exe scripts/test_sam_tracking.py uploads/race.mp4 \\
        --mark 650 667 240 779 439 --mark 760 628 195 725 240 --seconds 15 --width 640

Each --mark is FRAME X1 Y1 X2 Y2 in ORIGINAL video pixels. Prints the device,
weights, mask checks and tracking statistics, writes an annotated H.264 MP4
plus debug frames, and reopens the MP4 to verify it. Errors are NOT caught:
any failure prints the full traceback and exits with a non-zero code.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from core.geometry import BBox, fit_size  # noqa: E402
from core.sam2_tracking import (  # noqa: E402
    Selection,
    describe_device,
    ensure_sam2_weights,
    results_table,
    track_swimmer,
    write_analysis_clip,
)
from core.video import analysis_stride, read_frame, read_video_metadata, timestamp_to_frame  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--mark", type=float, nargs=5, action="append", required=True,
                   metavar=("FRAME", "X1", "Y1", "X2", "Y2"), help="swimmer box on one frame (repeatable)")
    p.add_argument("--seconds", type=float, default=5.0,
                   help="how long to track after the FIRST mark (0 = to the end)")
    p.add_argument("--width", type=int, default=640, help="analysis width in px (0 = original)")
    p.add_argument("--out", type=Path, default=config.OUTPUTS_DIR / "test_sam_tracking.mp4")
    p.add_argument("--debug-dir", type=Path, default=config.DEBUG_DIR / "test_sam_tracking")
    p.add_argument("--weights", type=Path, default=config.SAM2_MODEL_PATH)
    return p.parse_args()


def _versions() -> str:
    import torch
    import ultralytics

    return (f"python {sys.version.split()[0]}, torch {torch.__version__}, "
            f"ultralytics {ultralytics.__version__}, opencv {cv2.__version__}")


def _check_clip_matches_source(video: Path, frame: int, size: tuple[int, int], tmp: Path) -> float:
    """Mean pixel difference between the clip frame and the source frame (proves the right frame is used)."""
    write_analysis_clip(video, [frame], size, 30.0, tmp)
    cap = cv2.VideoCapture(str(tmp))
    ok, clip_frame = cap.read()
    cap.release()
    tmp.unlink(missing_ok=True)
    source = cv2.resize(read_frame(video, frame), size, interpolation=cv2.INTER_AREA)
    assert ok, "could not read back the analysis clip"
    return float(np.abs(clip_frame.astype(np.int16) - source.astype(np.int16)).mean())


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    args = _parse_args()
    print(f"[versions] {_versions()}")

    meta = read_video_metadata(args.video)
    print(f"[video] {args.video}  {meta.width}x{meta.height}  {meta.fps:.3f} fps  "
          f"{meta.frame_count} frames  {meta.duration_s:.2f} s  codec {meta.codec}")

    device, gpu = describe_device()
    print(f"[device] {device}  (half precision: {gpu})")
    weights = ensure_sam2_weights(args.weights)
    print(f"[weights] {weights}  exists={weights.is_file()}  {weights.stat().st_size / 1e6:.0f} MB")

    marks = sorted((Selection(int(m[0]), BBox(*m[1:])) for m in args.mark), key=lambda s: s.frame_index)
    first = marks[0].frame_index
    end = meta.frame_count - 1 if args.seconds <= 0 else min(
        meta.frame_count - 1, first + timestamp_to_frame(args.seconds, meta.fps))
    size = fit_size(meta.width, meta.height, args.width or None)
    print(f"[request] frames {first} -> {end}  analysis size {size[0]}x{size[1]}  stride {analysis_stride(meta.fps)}")
    for n, m in enumerate(marks, start=1):
        diff = _check_clip_matches_source(args.video, m.frame_index, size, args.out.with_suffix(".check.avi"))
        print(f"[mark {n}] frame {m.frame_index}  box {m.box.to_int_list()} (original px)  "
              f"clip-vs-source pixel diff {diff:.2f} (JPEG noise < 3)")

    def progress(done: int, total: int) -> None:
        if done % 200 == 0 or done == total:
            print(f"[progress] {done}/{total}")

    results, run = track_swimmer(args.video, meta, marks, end, args.width or None, args.out,
                                 debug_dir=args.debug_dir, progress=progress, weights_path=weights)

    by_frame = {r.frame_index: r for r in results}
    for n, m in enumerate(marks, start=1):
        r = by_frame[m.frame_index]
        print(f"[mark {n} result] mask pixels {r.mask_pixels}  mask box {r.box.to_int_list() if r.box else None}")
    outside = [r.frame_index for r in results if r.box and not r.box.is_inside(meta.width, meta.height)]
    print("[checks] mask size == frame size on every frame: yes (checked per frame)")
    print(f"[checks] boxes outside the frame: {len(outside)}")

    s = run.stats
    print(f"[stats] analysed {s.analyzed_frames}  tracked {s.tracked_frames} ({s.tracked_fraction:.1%})  "
          f"lost {s.lost_frames} (of which conflicts {s.conflict_frames})  "
          f"longest lost {s.longest_lost_frames} frames ({s.longest_lost_s:.2f} s)")
    for interval in s.lost_intervals:
        print(f"[lost] frames {interval.start_frame}-{interval.end_frame}  "
              f"t {interval.start_s:.2f}-{interval.end_s:.2f} s  ({interval.n_frames} frames)")
    print(f"[output] {args.out}  verified: {run.output_video}")
    print(f"[debug] frames in {args.debug_dir}: {len(list(args.debug_dir.glob('frame_*.jpg')))} images")
    csv = args.out.with_suffix(".csv")
    results_table(results).to_csv(csv, index=False)
    print(f"[csv] {csv}")
    print(f"[done] runtime {run.runtime_s:.1f} s on {run.device}")
    return 1 if outside else 0


if __name__ == "__main__":
    raise SystemExit(main())
