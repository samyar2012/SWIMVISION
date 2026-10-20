from dataclasses import dataclass

import numpy as np

from core.person_detector import PersonDetector
from core.swimmer_detector import SwimmerDetector


@dataclass
class HybridDetection:
    x1: float
    y1: float
    x2: float
    y2: float

    confidence: float

    source: str

    person_confidence: float = 0.0
    swimmer_confidence: float = 0.0


def calculate_iou(
    a: HybridDetection,
    b: HybridDetection
) -> float:

    intersection_x1 = max(a.x1, b.x1)
    intersection_y1 = max(a.y1, b.y1)

    intersection_x2 = min(a.x2, b.x2)
    intersection_y2 = min(a.y2, b.y2)

    intersection_width = max(
        0.0,
        intersection_x2 - intersection_x1
    )

    intersection_height = max(
        0.0,
        intersection_y2 - intersection_y1
    )

    intersection_area = (
        intersection_width *
        intersection_height
    )

    area_a = max(
        0.0,
        a.x2 - a.x1
    ) * max(
        0.0,
        a.y2 - a.y1
    )

    area_b = max(
        0.0,
        b.x2 - b.x1
    ) * max(
        0.0,
        b.y2 - b.y1
    )

    union_area = (
        area_a +
        area_b -
        intersection_area
    )

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


class HybridSwimmerDetector:
    def __init__(
        self,
        person_model_path: str,
        swimmer_model_path: str,
        person_confidence: float = 0.25,
        swimmer_threshold: float = 0.35,
        merge_iou_threshold: float = 0.15,
    ):
        self.person_detector = PersonDetector(
            model_path=person_model_path,
            confidence=person_confidence,
        )

        self.swimmer_detector = SwimmerDetector(
            model_path=swimmer_model_path,
            threshold=swimmer_threshold,
        )

        self.merge_iou_threshold = (
            merge_iou_threshold
        )

    def _get_person_candidates(
        self,
        frame: np.ndarray
    ) -> list[HybridDetection]:

        detections = (
            self.person_detector.detect(frame)
        )

        candidates = []

        for detection in detections:
            candidates.append(
                HybridDetection(
                    x1=detection.x1,
                    y1=detection.y1,
                    x2=detection.x2,
                    y2=detection.y2,

                    confidence=detection.confidence,

                    source="person",

                    person_confidence=(
                        detection.confidence
                    ),

                    swimmer_confidence=0.0,
                )
            )

        return candidates

    def _get_swimmer_candidates(
        self,
        frame: np.ndarray
    ) -> list[HybridDetection]:

        detections = (
            self.swimmer_detector
            .detect_swimmers(frame)
        )

        candidates = []

        for detection in detections:
            candidates.append(
                HybridDetection(
                    x1=detection.x1,
                    y1=detection.y1,
                    x2=detection.x2,
                    y2=detection.y2,

                    confidence=detection.confidence,

                    source="swimmer",

                    person_confidence=0.0,

                    swimmer_confidence=(
                        detection.confidence
                    ),
                )
            )

        return candidates

    def _merge_pair(
        self,
        person: HybridDetection,
        swimmer: HybridDetection
    ) -> HybridDetection:

        #
        # Use the union of the two boxes.
        #
        # YOLO often sees more of the body outside
        # the water, while the swimmer network
        # frequently sees the stroke/body region.
        #

        x1 = min(
            person.x1,
            swimmer.x1
        )

        y1 = min(
            person.y1,
            swimmer.y1
        )

        x2 = max(
            person.x2,
            swimmer.x2
        )

        y2 = max(
            person.y2,
            swimmer.y2
        )

        #
        # Don't average the confidence directly
        # because these models produce confidence
        # values with different meanings.
        #
        # Instead, keep the strongest evidence.
        #

        confidence = max(
            person.person_confidence,
            swimmer.swimmer_confidence
        )

        return HybridDetection(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,

            confidence=confidence,

            source="both",

            person_confidence=(
                person.person_confidence
            ),

            swimmer_confidence=(
                swimmer.swimmer_confidence
            ),
        )

    def _merge_candidates(
        self,
        person_candidates: list[HybridDetection],
        swimmer_candidates: list[HybridDetection],
    ) -> list[HybridDetection]:

        results = []

        used_swimmers = set()

        for person in person_candidates:

            best_index = None
            best_iou = 0.0

            for index, swimmer in enumerate(
                swimmer_candidates
            ):
                if index in used_swimmers:
                    continue

                iou = calculate_iou(
                    person,
                    swimmer
                )

                if iou > best_iou:
                    best_iou = iou
                    best_index = index

            if (
                best_index is not None
                and
                best_iou >= self.merge_iou_threshold
            ):
                swimmer = swimmer_candidates[
                    best_index
                ]

                merged = self._merge_pair(
                    person,
                    swimmer
                )

                results.append(
                    merged
                )

                used_swimmers.add(
                    best_index
                )

            else:
                results.append(
                    person
                )

        #
        # Add swimmer detections that were not
        # matched with YOLO detections.
        #

        for index, swimmer in enumerate(
            swimmer_candidates
        ):
            if index not in used_swimmers:
                results.append(
                    swimmer
                )

        return results

    def detect(
        self,
        frame: np.ndarray
    ) -> list[HybridDetection]:

        person_candidates = (
            self._get_person_candidates(frame)
        )

        swimmer_candidates = (
            self._get_swimmer_candidates(frame)
        )

        detections = self._merge_candidates(
            person_candidates,
            swimmer_candidates
        )

        return detections

    def get_heatmap(
        self,
        frame: np.ndarray
    ) -> np.ndarray:

        return (
            self.swimmer_detector
            .get_heatmap(frame)
        )