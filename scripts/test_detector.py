import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core.swimmer_detector import SwimmerDetector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    detector = SwimmerDetector(
        "models/colorShifts_deeper_zoomedOut_200epochs.pth",
        threshold=0.35
    )

    video = cv2.VideoCapture(
        str(args.video)
    )

    if not video.isOpened():
        print(
            f"Could not open video: {args.video}"
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
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        print("Could not create output video.")
        video.release()
        return 1

    frame_number = 0

    while True:
        ret, frame = video.read()

        if not ret:
            break

        frame_number += 1

        heatmap = detector.get_heatmap(
            frame
        )

        detections = detector._heatmap_to_detections(
            heatmap,
            width,
            height
        )

        heatmap_display = cv2.resize(
            heatmap,
            (width, height)
        )

        heatmap_display = np.clip(
            heatmap_display,
            0.0,
            1.0
        )

        heatmap_display = (
            heatmap_display * 255
        ).astype(np.uint8)

        heatmap_display = cv2.applyColorMap(
            heatmap_display,
            cv2.COLORMAP_JET
        )

        combined = cv2.addWeighted(
            frame,
            0.65,
            heatmap_display,
            0.35,
            0
        )

        for detection in detections:
            x1 = int(detection.x1)
            y1 = int(detection.y1)
            x2 = int(detection.x2)
            y2 = int(detection.y2)

            cv2.rectangle(
                combined,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = (
                f"Swimmer "
                f"{detection.confidence:.2f}"
            )

            cv2.putText(
                combined,
                label,
                (
                    x1,
                    max(y1 - 10, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.putText(
            combined,
            f"Frame {frame_number}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        writer.write(
            combined
        )

        cv2.imshow(
            "SwimVision Heatmap Test",
            combined
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    writer.release()

    cv2.destroyAllWindows()

    print(
        f"Finished. Output: {args.output}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())