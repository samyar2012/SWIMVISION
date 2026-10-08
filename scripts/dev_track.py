"""Developer utility: run detection + tracking on a video from the command line.

Examples (from the project root, with the venv active):

    # 1. See which people are detected on frame 60 (numbered like in the app)
    python -m scripts.dev_track my_race.mp4 --frame 60

    # 2. Track detected person #2, write an annotated video + CSV to outputs/dev/
    python -m scripts.dev_track my_race.mp4 --frame 60 --select 2

    # 3. Track a hand-drawn box (x1 y1 x2 y2 in pixels) with the visual tracker
    python -m scripts.dev_track my_race.mp4 --frame 60 --box 100 200 400 300 --method visual

    # 4. Segment Anything 2 with extra checkpoints (frame x1 y1 x2 y2), e.g. after the dive
    python -m scripts.dev_track my_race.mp4 --frame 60 --box 100 200 140 300 --method sam2 \\
        --checkpoint 900 425 115 500 160
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

import config
from core.detector import detect_people, load_yolo
from core.geometry import BBox
from core.overlay import draw_person_candidates, write_annotated_video
from core.tracker import SwimmerSelection, summarize_tracking, track_swimmer
from core.video import analysis_stride, read_frame, read_video_metadata


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--frame", type=int, default=0, help="frame where the swimmer is selected")
    ap.add_argument("--select", type=int, help="number of the detected person to track")
    ap.add_argument("--box", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    ap.add_argument("--method", choices=["sam2", "auto", "visual"], default="auto")
    ap.add_argument("--checkpoint", type=float, nargs=5, action="append", default=[],
                    metavar=("FRAME", "X1", "Y1", "X2", "Y2"), help="extra checkpoint (sam2 only)")
    ap.add_argument("--model", default=config.DEFAULT_YOLO_MODEL, choices=config.YOLO_MODEL_CHOICES)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--tracker", default="bytetrack.yaml", choices=["bytetrack.yaml", "botsort.yaml"])
    ap.add_argument("--out", default=str(config.OUTPUTS_DIR / "dev"))
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = read_video_metadata(args.video)
    stride = analysis_stride(meta.fps)
    print(f"Video: {meta.resolution_label} @ {meta.fps:.3f} fps, {meta.frame_count} frames, "
          f"{meta.duration_s:.2f} s  ->  analysing every {stride} frame(s)")

    frame = read_frame(args.video, args.frame)
    detections = detect_people(load_yolo(args.model), frame, args.conf, args.imgsz)
    print(f"Frame {args.frame}: {len(detections)} person(s) detected")
    for d in detections:
        print(f"  #{d.number}: conf={d.confidence:.2f} box={[round(v) for v in d.bbox.to_list()]}")
    preview = out_dir / "detections.jpg"
    cv2.imwrite(str(preview), draw_person_candidates(frame, detections, args.select))
    print(f"Saved detection preview: {preview}")

    if args.select is None and args.box is None:
        return 0
    if args.box:
        selection = SwimmerSelection(args.frame, BBox(*args.box), "manual")
    else:
        chosen = next((d for d in detections if d.number == args.select), None)
        if chosen is None:
            print(f"No detected person #{args.select}.")
            return 1
        selection = SwimmerSelection(args.frame, chosen.bbox, "detected", chosen.confidence)

    last_pct = [-1]

    def progress(fraction: float, message: str) -> None:
        pct = int(fraction * 10) * 10
        if pct != last_pct[0]:
            last_pct[0] = pct
            print(f"  {pct:3d}%  {message}")

    result = track_swimmer(args.video, meta, selection, stride=stride, method=args.method,
                           model_name=args.model, conf=args.conf, imgsz=args.imgsz,
                           tracker_name=args.tracker, progress=progress,
                           checkpoints=[SwimmerSelection(int(c[0]), BBox(*c[1:]), "manual")
                                        for c in args.checkpoint])
    summary = summarize_tracking(result)
    print(f"\nMode: {summary.mode}   runtime: {summary.runtime_s:.1f} s")
    print(f"Analysed {summary.n_samples} frames: tracked={summary.n_tracked} "
          f"low_conf={summary.n_low_confidence} reacquired={summary.n_reacquired} "
          f"not_moving={summary.n_not_moving} lost={summary.n_lost}")
    for note in summary.notes:
        print("NOTE:", note)
    for seg in summary.segments:
        print(f"  unreliable {seg.start_time_s:.2f}-{seg.end_time_s:.2f} s: {', '.join(seg.reasons)}")

    table = result.to_dataframe()
    table.to_csv(out_dir / "tracking.csv", index=False)
    video_out = write_annotated_video(args.video, meta, table, out_dir / "annotated.mp4", stride)
    print(f"Saved: {out_dir / 'tracking.csv'}\nSaved: {video_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
