from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .classify import decide_type
from .config import PipelineConfig
from .detect import Detection, build_detector, is_multiclass
from .ocr import PlateOCR, build_ocr, two_line_to_single
from .postprocess import PlateRow, deduplicate, finalize
from .rectify import RectifyResult, rectify, warp_quad
from .vehicle import VehicleGate


@dataclass
class PlateDebug:
    detection: Detection
    rectified: RectifyResult
    plate_type: str
    type_source: str
    ocr_text: str
    ocr_raw: str
    row: PlateRow | None


@dataclass
class FrameResult:
    rows: list[PlateRow]
    timings_ms: dict[str, float] = field(default_factory=dict)
    debug: list[PlateDebug] = field(default_factory=list)


class Pipeline:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.detector = build_detector(cfg)
        self.multiclass = is_multiclass(self.detector)
        self.ocr: PlateOCR = build_ocr(cfg)
        self.vehicle_gate = VehicleGate(cfg) if cfg.vehicle_check.enabled else None

    def warmup(self, n: int = 3) -> None:
        dummy = np.full((720, 1280, 3), 114, dtype=np.uint8)
        for _ in range(n):
            self.detector.predict(dummy)
            self.ocr.probabilities(np.full((64, 256, 3), 200, dtype=np.uint8))
            if self.vehicle_gate:
                self.vehicle_gate.begin_frame(dummy)

    def process(self, frame: np.ndarray, keep_debug: bool = False) -> FrameResult:
        t = {"detect": 0.0, "rectify": 0.0, "classify": 0.0, "ocr": 0.0, "post": 0.0, "vehicle": 0.0}
        t0 = time.perf_counter()
        detections = self.detector.predict(frame)
        t["detect"] = (time.perf_counter() - t0) * 1000

        if self.vehicle_gate is not None:
            t0 = time.perf_counter()
            self.vehicle_gate.begin_frame(frame)
            t["vehicle"] = (time.perf_counter() - t0) * 1000

        rows: list[PlateRow] = []
        debug: list[PlateDebug] = []
        for det in detections:
            if det.width < 8 or det.height < 4:
                continue
            det_label = det.label if self.multiclass else None

            t0 = time.perf_counter()
            two_line_hint = (det_label == "type1a") if det_label else None
            rect = rectify(frame, det, self.cfg.rectify, two_line_hint)
            t["rectify"] += (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            decision = decide_type(det_label, rect.aspect, rect.image, self.cfg.classify)
            plate_img = rect.image
            if rect.quad is not None and decision.two_line != rect.two_line:
                size = self.cfg.rectify.two_line_size if decision.two_line else self.cfg.rectify.single_line_size
                plate_img = warp_quad(frame, rect.quad, size)
            t["classify"] += (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            decoded = None
            if decision.plate_type in ("type1", "type1a", "type1b"):
                ocr_in = two_line_to_single(plate_img, self.cfg.two_line) if decision.two_line else plate_img
                decoded = self.ocr.read(ocr_in, decision.plate_type, self.cfg.postprocess.min_char_prob)
            t["ocr"] += (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            is_vehicle = self.vehicle_gate.is_vehicle(det) if self.vehicle_gate else True
            row = finalize(decision.plate_type, decoded, det.confidence, self.cfg.postprocess,
                           (det.x1, det.y1, det.x2, det.y2), is_vehicle)
            if row is not None:
                rows.append(row)
            t["post"] += (time.perf_counter() - t0) * 1000
            if keep_debug:
                debug.append(PlateDebug(det, rect, decision.plate_type, decision.source,
                                        decoded.text if decoded else "", decoded.raw_text if decoded else "", row))
        t0 = time.perf_counter()
        rows = deduplicate(rows)
        t["post"] += (time.perf_counter() - t0) * 1000
        t["total"] = sum(v for k, v in t.items() if k != "total")
        return FrameResult(rows, t, debug)


def draw_predictions(frame: np.ndarray, result: FrameResult) -> np.ndarray:
    img = frame.copy()
    for d in result.debug:
        color = (36, 255, 12) if d.row and d.row.plate_type != "other" else (0, 165, 255)
        det = d.detection
        cv2.rectangle(img, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        if d.rectified.quad is not None:
            cv2.polylines(img, [d.rectified.quad.astype(np.int32).reshape(-1, 1, 2)], True, (255, 80, 0), 2)
        label = f"{d.row.plate_num if d.row else ''} {d.plate_type} {d.row.confidence:.2f}" if d.row else d.plate_type
        y = det.y1 - 8 if det.y1 > 20 else det.y2 + 18
        cv2.putText(img, label, (det.x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (det.x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img
