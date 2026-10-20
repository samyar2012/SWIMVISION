from core.pool_filter import PoolRegion, detection_center
import numpy as np
import cv2
import argparse
import logging
import sys

pool = PoolRegion(x1=0, x2=100, y1=0, y2=100)

detection = detector.detect swimmers(
    frame,
)

detections = filter_pool_detections(
    detections,
    pool.
    
)

for detection in detections:
    center_x, center_y = detection_center(detection)
    print(f"Detection center: ({center_x}, {center_y})")

return 0

if __name__ == "__main__":
    sys.exit(main())
