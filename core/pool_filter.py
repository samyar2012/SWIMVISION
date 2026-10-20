from dataclasses import dataclass

from core.swimmer_detector import Detection


@dataclass
class PoolRegion:
    x1: int
    y1: int
    x2: int
    y2: int


def detection_center(
    detection: Detection
) -> tuple[float, float]:
    center_x = (detection.x1 + detection.x2) / 2
    center_y = (detection.y1 + detection.y2) / 2

    return center_x, center_y


def is_inside_pool(
    detection: Detection,
    pool: PoolRegion
) -> bool:
    center_x, center_y = detection_center(detection)

    return (
        pool.x1 <= center_x <= pool.x2
        and pool.y1 <= center_y <= pool.y2
    )


def filter_pool_detections(
    detections: list[Detection],
    pool: PoolRegion
) -> list[Detection]:
    filtered = []

    for detection in detections:
        if is_inside_pool(detection, pool):
            filtered.append(detection)

    return filtered