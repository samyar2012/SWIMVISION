"""Run the persistent swimmer tracker over a video and inspect the handoff.

Example (all coordinates are original video pixels):

    python -m scripts.test_tracker --video scripts/test_videos/video.mp4 \
        --output outputs/tracker_test.mp4 --start-frame 650 \
        --target-box 667 240 779 439 --lane-min 180 --lane-max 520

The target box is the swimmer selection before the race.  Green boxes are
detector observations, yellow boxes are short motion predictions through a
splash/underwater gap, and LOST means the tracker is retaining identity for
reacquisition but will not display a stale prediction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

from core.geometry import BBox  # noqa: E402
from core.swimmer_tracker import LaneConstraint, PersistentSwimmerTracker, TrackedSwimmer  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True, help="input race video")
    parser.add_argument("--output", type=Path, required=True, help="annotated MP4 to write")
    parser.add_argument("--person-model", default="yolo26n.pt", help="YOLO person-model weights")
    parser.add_argument(
        "--swimmer-model",
        default="models/colorShifts_deeper_zoomedOut_200epochs.pth",
        help="swimmer U-Net weights",
    )
    parser.add_argument("--person-confidence", type=float, default=0.25)
    parser.add_argument("--swimmer-threshold", type=float, default=0.35)
    parser.add_argument("--merge-iou-threshold", type=float, default=0.15)
    parser.add_argument("--start-frame", type=int, default=0, help="frame containing the target selection")
    parser.add_argument("--end-frame", type=int, help="last frame to process (defaults to video end)")
    parser.add_argument(
        "--target-box",
        type=float,
        nargs=4,
        required=True,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="box around the chosen swimmer on --start-frame",
    )
    parser.add_argument("--lane-axis", choices=("x", "y"), default="y")
    parser.add_argument("--lane-min", type=float, help="first image-coordinate boundary of the lane")
    parser.add_argument("--lane-max", type=float, help="second image-coordinate boundary of the lane")
    parser.add_argument("--max-missed", type=int, default=45, help="frames to show motion prediction")
    parser.add_argument(
        "--max-reacquire", type=int, default=90,
        help="frames to retain identity after detections stop",
    )
    parser.add_argument("--show", action="store_true", help="show a live preview; press q to stop")
    parser.add_argument("--draw-candidates", action="store_true", help="also draw all raw hybrid candidates")
    return parser.parse_args()


def _load_hybrid_detector(args: argparse.Namespace):
    """Import lazily so tracker logic can be tested without loading model code."""
    try:
        from core.hybrid_detector import HybridSwimmerDetector
    except ImportError as exc:
        raise RuntimeError(
            "core.hybrid_detector.HybridSwimmerDetector is required for the video test. "
            "Restore the hybrid detector files from the detector experiment before running this script."
        ) from exc

    return HybridSwimmerDetector(
        person_model_path=args.person_model,
        swimmer_model_path=args.swimmer_model,
        person_confidence=args.person_confidence,
        swimmer_threshold=args.swimmer_threshold,
        merge_iou_threshold=args.merge_iou_threshold,
    )


def _draw_candidate(frame, detection) -> None:
    colours = {"person": (180, 80, 40), "swimmer": (80, 80, 180), "both": (80, 150, 80)}
    colour = colours.get(str(detection.source).lower(), (120, 120, 120))
    p1 = (int(round(detection.x1)), int(round(detection.y1)))
    p2 = (int(round(detection.x2)), int(round(detection.y2)))
    cv2.rectangle(frame, p1, p2, colour, 1)


def _draw_track(frame, result: TrackedSwimmer | None, missed_frames: int) -> None:
    if result is None:
        label = f"LOST - identity held for reacquisition ({missed_frames} missed)"
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return

    colour = (0, 215, 255) if result.predicted else (0, 255, 0)
    status = "PREDICTED" if result.predicted else "TRACKED"
    x1, y1, x2, y2 = result.box.to_int_list()
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
    label = (
        f"{status} {result.source} conf={result.confidence:.2f} "
        f"score={result.score:.2f} missed={result.missed_frames}"
    )
    cv2.putText(frame, label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)


def main() -> int:
    args = _parse_args()
    if (args.lane_min is None) != (args.lane_max is None):
        raise ValueError("--lane-min and --lane-max must be supplied together")
    if args.start_frame < 0:
        raise ValueError("--start-frame cannot be negative")

    detector = _load_hybrid_detector(args)
    lane = None
    if args.lane_min is not None:
        lane = LaneConstraint(args.lane_min, args.lane_max, args.lane_axis)
    tracker = PersistentSwimmerTracker(
        detector,
        lane=lane,
        max_missed_frames=args.max_missed,
        max_reacquire_frames=args.max_reacquire,
    )

    video = cv2.VideoCapture(str(args.video))
    if not video.isOpened():
        raise RuntimeError(f"could not open video: {args.video}")
    fps = video.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.start_frame >= frame_count:
        video.release()
        raise ValueError(f"--start-frame {args.start_frame} is outside the {frame_count}-frame video")
    end_frame = frame_count - 1 if args.end_frame is None else args.end_frame
    if not args.start_frame <= end_frame < frame_count:
        video.release()
        raise ValueError(f"--end-frame must be between {args.start_frame} and {frame_count - 1}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        video.release()
        raise RuntimeError(f"could not create output video: {args.output}")

    tracked = predicted = lost = processed = 0
    target = BBox(*args.target_box)
    try:
        for frame_index in range(end_frame + 1):
            ok, frame = video.read()
            if not ok:
                break
            if frame_index < args.start_frame:
                cv2.putText(frame, "WAITING FOR TARGET FRAME", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
                writer.write(frame)
                continue

            candidates = detector.detect(frame)
            if not tracker.initialized:
                result = tracker.start(candidates, frame.shape[:2], target_box=target, frame=frame)
            else:
                result = tracker.update(candidates, frame.shape[:2], frame=frame)
            processed += 1
            if result is None:
                lost += 1
            elif result.predicted:
                predicted += 1
            else:
                tracked += 1

            if args.draw_candidates:
                for candidate in candidates:
                    _draw_candidate(frame, candidate)
            _draw_track(frame, result, tracker.missed_frames)
            cv2.putText(frame, f"frame {frame_index}", (20, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            writer.write(frame)

            if args.show:
                cv2.imshow("SwimVision persistent tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        video.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Processed: {processed} frames")
    print(f"Detector observations: {tracked}")
    print(f"Motion predictions: {predicted}")
    print(f"Lost (identity retained): {lost}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
