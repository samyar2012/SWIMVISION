import argparse
import sys
from pathlib import Path

import cv2

from core.hybrid_detector import (
    HybridSwimmerDetector
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test SwimVision hybrid "
            "swimmer detector."
        )
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Input swimming video."
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output annotated video."
    )

    return parser.parse_args()


def draw_detection(
    frame,
    detection
):
    #
    # Color by detector source.
    #
    # YOLO only    = blue
    # U-Net only   = red
    # Both agree   = green
    #

    if detection.source == "person":
        color = (255, 0, 0)

    elif detection.source == "swimmer":
        color = (0, 0, 255)

    else:
        color = (0, 255, 0)

    x1 = int(detection.x1)
    y1 = int(detection.y1)

    x2 = int(detection.x2)
    y2 = int(detection.y2)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    if detection.source == "both":

        label = (
            f"BOTH "
            f"P:{detection.person_confidence:.2f} "
            f"S:{detection.swimmer_confidence:.2f}"
        )

    elif detection.source == "person":

        label = (
            f"PERSON "
            f"{detection.person_confidence:.2f}"
        )

    else:

        label = (
            f"SWIMMER "
            f"{detection.swimmer_confidence:.2f}"
        )

    cv2.putText(
        frame,
        label,
        (
            x1,
            max(
                20,
                y1 - 10
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2
    )


def main() -> int:
    args = parse_args()

    detector = HybridSwimmerDetector(
        person_model_path="yolo26n.pt",

        swimmer_model_path=(
            "models/"
            "colorShifts_deeper_zoomedOut_200epochs.pth"
        ),

        person_confidence=0.25,

        swimmer_threshold=0.35,

        merge_iou_threshold=0.15,
    )

    video = cv2.VideoCapture(
        str(args.video)
    )

    if not video.isOpened():
        print(
            f"Could not open video: "
            f"{args.video}"
        )

        return 1

    fps = video.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        video.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        video.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    if fps <= 0:
        fps = 30.0

    writer = cv2.VideoWriter(
        str(args.output),

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        fps,

        (
            width,
            height
        )
    )

    if not writer.isOpened():
        print(
            "Could not create "
            "output video."
        )

        video.release()

        return 1

    frame_number = 0

    while True:
        ret, frame = video.read()

        if not ret:
            break

        frame_number += 1

        detections = detector.detect(
            frame
        )

        person_count = 0
        swimmer_count = 0
        both_count = 0

        for detection in detections:

            if detection.source == "person":
                person_count += 1

            elif detection.source == "swimmer":
                swimmer_count += 1

            else:
                both_count += 1

            draw_detection(
                frame,
                detection
            )

        status = (
            f"Frame: {frame_number} | "
            f"Person: {person_count} | "
            f"Swimmer: {swimmer_count} | "
            f"Both: {both_count}"
        )

        cv2.putText(
            frame,
            status,
            (20, 35),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.75,

            (255, 255, 255),

            2
        )

        cv2.putText(
            frame,
            "BLUE=YOLO  RED=SWIMMER  GREEN=BOTH",
            (20, 65),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.60,

            (255, 255, 255),

            2
        )

        writer.write(
            frame
        )

        cv2.imshow(
            "SwimVision Hybrid Detector",
            frame
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):
            break

    video.release()

    writer.release()

    cv2.destroyAllWindows()

    print()
    print("Finished.")
    print(
        f"Output saved to: {args.output}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )