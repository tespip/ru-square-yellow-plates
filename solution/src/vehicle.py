from __future__ import annotations

import numpy as np

from .config import PipelineConfig
from .detect import Detection, UltralyticsOnnxDetector

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]
VEHICLE_LABELS = {"car", "motorcycle", "bus", "truck"}


class VehicleGate:
    def __init__(self, cfg: PipelineConfig) -> None:
        v = cfg.vehicle_check
        self.detector = UltralyticsOnnxDetector(cfg.resolve(v.model), COCO_CLASSES, v.conf_thresh, 0.5, None)
        self.min_overlap = v.min_overlap
        self._cache: list[Detection] | None = None

    def begin_frame(self, frame: np.ndarray) -> None:
        self._cache = [d for d in self.detector.predict(frame) if d.label in VEHICLE_LABELS]

    def is_vehicle(self, det: Detection) -> bool:
        if self._cache is None:
            return True
        area = max(1, det.width * det.height)
        for v in self._cache:
            ix = max(0, min(det.x2, v.x2) - max(det.x1, v.x1))
            iy = max(0, min(det.y2, v.y2) - max(det.y1, v.y1))
            if ix * iy / area >= self.min_overlap:
                return True
        return False
