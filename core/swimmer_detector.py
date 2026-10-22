from dataclasses import dataclass

import cv2
import numpy as np
import torch

from core.pool_geometry import PoolGeometry, PoolLaneDetector
from core.swimmer_model import DeeperUNetLike


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0


class SwimmerDetector:
    def __init__(
        self,
        model_path: str,
        input_size: int = 256,
        threshold: float = 0.35,
        min_blob_area: int = 10,
        separate_lanes: bool = True,
    ):
        self.input_size = input_size
        self.threshold = threshold
        self.min_blob_area = min_blob_area
        self.separate_lanes = separate_lanes
        self.lane_detector = PoolLaneDetector() if separate_lanes else None
        self.last_lane_geometry: PoolGeometry | None = None

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Using device: {self.device}")

        self.model = DeeperUNetLike()

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

    def _prepare_frame(
        self,
        frame: np.ndarray
    ) -> torch.Tensor:
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        resized = cv2.resize(
            rgb,
            (self.input_size, self.input_size)
        )

        image = resized.astype(np.float32) / 255.0

        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

        image = (image - mean) / std

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        tensor = torch.from_numpy(image).float()

        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def get_heatmap(
        self,
        frame: np.ndarray
    ) -> np.ndarray:
        tensor = self._prepare_frame(frame)

        with torch.no_grad():
            output = self.model(tensor)

        heatmap = (
            output[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        return heatmap

    def _heatmap_to_detections(
        self,
        heatmap: np.ndarray,
        original_width: int,
        original_height: int,
        lane_geometry: PoolGeometry | None = None,
    ) -> list[Detection]:

        binary = (
            heatmap >= self.threshold
        ).astype(np.uint8) * 255

        close_kernel = np.ones(
            (5, 5),
            dtype=np.uint8
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=1
        )

        dilation_kernel = np.ones(
            (3, 3),
            dtype=np.uint8
        )

        binary = cv2.dilate(
            binary,
            dilation_kernel,
            iterations=1
        )

        # The U-Net paper this model comes from separates blobs at detected
        # lane ropes before extracting boxes.  That prevents two adjacent
        # swimmers from becoming one contour and subsequently one track.
        if lane_geometry is not None:
            binary = cv2.bitwise_and(
                binary,
                lane_geometry.separator_mask(self.input_size, self.input_size),
            )

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []

        scale_x = original_width / self.input_size
        scale_y = original_height / self.input_size

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.min_blob_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            padding_x = int(w * 0.20)
            padding_y = int(h * 0.30)

            x1 = max(
                0,
                x - padding_x
            )

            y1 = max(
                0,
                y - padding_y
            )

            x2 = min(
                self.input_size,
                x + w + padding_x
            )

            y2 = min(
                self.input_size,
                y + h + padding_y
            )

            region = heatmap[
                y:y + h,
                x:x + w
            ]

            if region.size == 0:
                continue

            # Background inside the enclosing rectangle is not evidence
            # against the swimmer.  Score only pixels that actually activated
            # the segmentation model so BoT-SORT receives calibrated values.
            active_pixels = region[region >= self.threshold]
            if active_pixels.size == 0:
                continue
            confidence = float(active_pixels.mean())

            detections.append(
                Detection(
                    x1=float(x1 * scale_x),
                    y1=float(y1 * scale_y),
                    x2=float(x2 * scale_x),
                    y2=float(y2 * scale_y),
                    confidence=confidence,
                    class_id=0,
                )
            )

        return detections

    def detect_swimmers(
        self,
        frame: np.ndarray
    ) -> list[Detection]:

        if frame is None:
            return []

        original_height, original_width = frame.shape[:2]

        heatmap = self.get_heatmap(frame)

        self.last_lane_geometry = (
            self.lane_detector.detect(frame)
            if self.lane_detector is not None
            else None
        )

        return self._heatmap_to_detections(
            heatmap,
            original_width,
            original_height,
            self.last_lane_geometry,
        )
