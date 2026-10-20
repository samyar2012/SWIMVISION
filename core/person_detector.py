from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class PersonDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class PersonDetector:
    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        confidence: float = 0.25,
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(
        self,
        frame: np.ndarray
    ) -> list[PersonDetection]:

        results = self.model.predict(
            frame,
            conf=self.confidence,
            classes=[0],  # COCO class 0 = person
            verbose=False
        )

        if not results:
            return []

        result = results[0]

        if result.boxes is None:
            return []

        boxes = (
            result.boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )

        scores = (
            result.boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

        detections = []

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box

            detections.append(
                PersonDetection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=float(score)
                )
            )

        return detections