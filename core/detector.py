"""Person detection with a pretrained YOLO model (Ultralytics).

We do NOT train anything. YOLO models are pretrained on the COCO dataset, where
class 0 is "person". We only ask it for that class.

Limitation to be aware of: COCO has few images of mostly-submerged swimmers, so
detection can miss a swimmer who is low in the water or in splash. That is why
SwimVision also lets the user draw a box by hand (see ``core/tracker.py``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

import config
from core.errors import ModelLoadError
from core.geometry import BBox

logger = logging.getLogger(__name__)

PERSON_CLASS_ID = 0  # COCO class index for "person"


@dataclass(frozen=True)
class PersonDetection:
    """One detected person in one frame."""

    number: int          # 1, 2, 3... label shown on screen for the user to pick
    bbox: BBox
    confidence: float    # YOLO's own score in [0, 1]


def load_yolo(model_name: str = config.DEFAULT_YOLO_MODEL):
    """Load a YOLO model, downloading the weights into ``models/`` on first use.

    Returns a **new** model object every call. Ultralytics keeps tracker state
    inside the object, so detection and tracking each use their own copy.
    """
    from ultralytics import YOLO  # imported lazily: heavy import (PyTorch)

    if model_name not in config.YOLO_MODEL_CHOICES:
        raise ModelLoadError(f"Unknown detection model '{model_name}'.")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    weights = config.MODELS_DIR / f"{model_name}.pt"
    try:
        return YOLO(str(weights))  # downloads automatically if the file is missing
    except Exception as exc:  # network error, corrupt download, ...
        logger.exception("Could not load YOLO weights %s", weights)
        raise ModelLoadError(
            f"Could not load the '{model_name}' detection model. The first run needs an "
            "internet connection to download it (about 20-50 MB). "
            f"Details: {type(exc).__name__}"
        ) from exc


def compute_device_name() -> str:
    """Human-readable compute device ("NVIDIA ..." or "CPU") for the UI."""
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"


def _number_detections(boxes: list[tuple[BBox, float]], frame_height: int) -> list[PersonDetection]:
    """Number detections top-to-bottom, then left-to-right.

    Sorting makes the numbers stable and meaningful: in a side-view pool shot
    "1" is the lane nearest the top of the picture. Boxes whose centres are
    within ~5% of the frame height count as the same "row" and are ordered
    left-to-right.
    """
    row_height = max(1.0, 0.05 * frame_height)
    ordered = sorted(boxes, key=lambda item: (round(item[0].center[1] / row_height), item[0].center[0]))
    return [PersonDetection(i + 1, box, conf) for i, (box, conf) in enumerate(ordered)]


def detect_people(
    model,
    frame_bgr: np.ndarray,
    conf: float = 0.25,
    imgsz: int = 960,
) -> list[PersonDetection]:
    """Find all people in one frame.

    Args:
        model: object returned by ``load_yolo``.
        frame_bgr: the frame as read by OpenCV.
        conf: minimum confidence to keep a detection.
        imgsz: size YOLO resizes to internally. Bigger finds smaller/farther
            swimmers but is slower.
    """
    results = model.predict(frame_bgr, classes=[PERSON_CLASS_ID], conf=conf,
                            imgsz=imgsz, verbose=False)
    if not results or results[0].boxes is None:
        return []
    xyxy = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    found = [(BBox(*map(float, b)), float(s)) for b, s in zip(xyxy, scores)]
    logger.info("Detected %d people (conf>=%.2f, imgsz=%d)", len(found), conf, imgsz)
    return _number_detections(found, frame_bgr.shape[0])
