from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import ClassifyConfig


@dataclass
class TypeDecision:
    plate_type: str
    two_line: bool
    yellow_fraction: float
    aspect: float
    source: str  


def yellow_fraction(plate_bgr: np.ndarray) -> float:
    if plate_bgr.size == 0:
        return 0.0
    h, w = plate_bgr.shape[:2]
    inner = plate_bgr[int(h * 0.1):int(h * 0.9), int(w * 0.05):int(w * 0.95)]
    if inner.size == 0:
        inner = plate_bgr
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (15, 70, 90), (40, 255, 255))
    return float(mask.mean() / 255.0)


def heuristic_type(aspect: float, yellow: float, cfg: ClassifyConfig) -> TypeDecision:
    two_line = aspect < cfg.ambiguous_split_aspect
    if two_line:
        ptype = "type1a"
    else:
        ptype = "type1b" if yellow >= cfg.yellow_min_fraction else "type1"
    return TypeDecision(ptype, two_line, yellow, aspect, "heuristic")


def decide_type(det_label: str | None, aspect: float, plate_bgr: np.ndarray, cfg: ClassifyConfig) -> TypeDecision:
    yellow = yellow_fraction(plate_bgr)
    heur = heuristic_type(aspect, yellow, cfg)
    if cfg.mode != "heuristic" and det_label == "other":
        return TypeDecision("other", False, yellow, aspect, "detector")
    label_ok = det_label in ("type1", "type1a", "type1b")
    if cfg.mode == "heuristic" or not label_ok:
        return heur
    assert det_label is not None
    if cfg.mode == "detector":
        return TypeDecision(det_label, det_label == "type1a", yellow, aspect, "detector")

    two_line = det_label == "type1a"
    if aspect <= cfg.two_line_max_aspect:
        two_line = True
    elif aspect >= cfg.single_line_min_aspect:
        two_line = False
    if two_line:
        return TypeDecision("type1a", True, yellow, aspect, "fused")
    if yellow >= cfg.yellow_min_fraction:
        ptype = "type1b"
    elif yellow <= cfg.white_max_yellow_fraction:
        ptype = "type1"
    else:
        ptype = det_label if det_label != "type1a" else "type1"
    return TypeDecision(ptype, False, yellow, aspect, "fused")
