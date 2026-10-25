from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort
from open_image_models.detection.core.yolo_v9.preprocess import preprocess

from .config import DetectorConfig, PipelineConfig


@dataclass
class Detection:
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height > 0 else 0.0

    def clamp(self, w: int, h: int) -> "Detection":
        return Detection(
            self.label, self.confidence,
            max(0, min(self.x1, w - 1)), max(0, min(self.y1, h - 1)),
            max(1, min(self.x2, w)), max(1, min(self.y2, h)),
        )


def _session(model_path: Path, providers: Sequence[str] | None) -> ort.InferenceSession:
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX-модель не найдена: {model_path}")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options=opts,
                                providers=list(providers) if providers else ort.get_available_providers())


class End2EndDetector:
    def __init__(self, model_path: Path, class_labels: Sequence[str], conf_thresh: float,
                 providers: Sequence[str] | None = None) -> None:
        self.session = _session(model_path, providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        _, _, h, w = self.session.get_inputs()[0].shape
        self.img_size = (int(h), int(w))
        self.labels = list(class_labels)
        self.conf_thresh = conf_thresh

    def predict(self, frame: np.ndarray) -> list[Detection]:
        inputs, ratio, (dw, dh) = preprocess(frame, self.img_size)
        try:
            preds = self.session.run([self.output_name], {self.input_name: inputs})[0]
        except Exception:  
            return []
        out: list[Detection] = []
        fh, fw = frame.shape[:2]
        for row in preds:
            score = float(row[6])
            if score < self.conf_thresh:
                continue
            x1 = (row[1] - dw) / ratio[0]
            y1 = (row[2] - dh) / ratio[1]
            x2 = (row[3] - dw) / ratio[0]
            y2 = (row[4] - dh) / ratio[1]
            cls = int(row[5])
            label = self.labels[cls] if cls < len(self.labels) else str(cls)
            out.append(Detection(label, score, int(x1), int(y1), int(x2), int(y2)).clamp(fw, fh))
        return out


class UltralyticsOnnxDetector:
    def __init__(self, model_path: Path, class_labels: Sequence[str], conf_thresh: float,
                 iou_thresh: float = 0.45, providers: Sequence[str] | None = None) -> None:
        self.session = _session(model_path, providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        _, _, h, w = self.session.get_inputs()[0].shape
        self.img_size = (int(h), int(w))
        self.labels = list(class_labels)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def predict(self, frame: np.ndarray) -> list[Detection]:
        inputs, ratio, (dw, dh) = preprocess(frame, self.img_size)
        preds = self.session.run([self.output_name], {self.input_name: inputs})[0]
        preds = np.squeeze(preds, axis=0)
        if preds.shape[0] < preds.shape[1]:  
            preds = preds.T
        boxes_xywh = preds[:, :4]
        cls_scores = preds[:, 4:]
        cls_ids = cls_scores.argmax(axis=1)
        scores = cls_scores[np.arange(len(cls_ids)), cls_ids]
        keep = scores >= self.conf_thresh
        if not keep.any():
            return []
        boxes_xywh, scores, cls_ids = boxes_xywh[keep], scores[keep], cls_ids[keep]
        xyxy = np.empty_like(boxes_xywh)
        xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
        idx = cv2.dnn.NMSBoxes(
            [(float(x), float(y), float(w), float(h)) for x, y, w, h in
             np.column_stack([xyxy[:, 0], xyxy[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]])],
            scores.astype(float).tolist(), self.conf_thresh, self.iou_thresh,
        )
        if len(idx) == 0:
            return []
        fh, fw = frame.shape[:2]
        out: list[Detection] = []
        for i in np.array(idx).reshape(-1):
            x1 = (xyxy[i, 0] - dw) / ratio[0]
            y1 = (xyxy[i, 1] - dh) / ratio[1]
            x2 = (xyxy[i, 2] - dw) / ratio[0]
            y2 = (xyxy[i, 3] - dh) / ratio[1]
            cls = int(cls_ids[i])
            label = self.labels[cls] if cls < len(self.labels) else str(cls)
            out.append(Detection(label, float(scores[i]), int(x1), int(y1), int(x2), int(y2)).clamp(fw, fh))
        return out


def _overlap(a: Detection, b: Detection) -> float:
    iw = min(a.x2, b.x2) - max(a.x1, b.x1)
    ih = min(a.y2, b.y2) - max(a.y1, b.y1)
    if iw <= 0 or ih <= 0:
        return 0.0
    small = min(a.width * a.height, b.width * b.height)
    return iw * ih / small if small > 0 else 0.0


class TiledDetector:
    def __init__(self, base, mode: str, grid: tuple[int, int], overlap: float, merge_thresh: float = 0.5) -> None:
        if mode not in ("off", "fallback", "always"):
            raise ValueError(f"tiles: off | fallback | always, а не {mode!r}")
        self.base, self.mode, self.grid, self.overlap, self.merge_thresh = base, mode, grid, overlap, merge_thresh

    @property
    def labels(self) -> list[str]:
        return self.base.labels

    def _tiles(self, w: int, h: int) -> list[tuple[int, int, int, int]]:
        cols, rows = self.grid
        tw = int(w / (cols - (cols - 1) * self.overlap)) if cols > 1 else w
        th = int(h / (rows - (rows - 1) * self.overlap)) if rows > 1 else h
        xs = [round(i * (w - tw) / (cols - 1)) for i in range(cols)] if cols > 1 else [0]
        ys = [round(j * (h - th) / (rows - 1)) for j in range(rows)] if rows > 1 else [0]
        return [(x, y, min(w, x + tw), min(h, y + th)) for y in ys for x in xs]

    def predict(self, frame: np.ndarray) -> list[Detection]:
        dets = self.base.predict(frame)
        if self.mode == "off" or (self.mode == "fallback" and dets):
            return dets
        h, w = frame.shape[:2]
        edge = 2
        for x0, y0, x1, y1 in self._tiles(w, h):
            for d in self.base.predict(frame[y0:y1, x0:x1]):
                cut = ((d.x1 <= edge and x0 > 0) or (d.x2 >= x1 - x0 - edge and x1 < w)
                       or (d.y1 <= edge and y0 > 0) or (d.y2 >= y1 - y0 - edge and y1 < h))
                if not cut:
                    dets.append(Detection(d.label, d.confidence, d.x1 + x0, d.y1 + y0, d.x2 + x0, d.y2 + y0))
        kept: list[Detection] = []
        for d in sorted(dets, key=lambda d: -d.confidence):
            if all(_overlap(d, k) < self.merge_thresh for k in kept):
                kept.append(d)
        return kept


def build_detector(cfg: PipelineConfig, det_cfg: DetectorConfig | None = None):
    d = det_cfg or cfg.detector
    path = cfg.resolve(d.model)
    if d.backend == "end2end":
        base = End2EndDetector(path, d.class_labels, d.conf_thresh, d.providers)
    elif d.backend == "ultralytics":
        base = UltralyticsOnnxDetector(path, d.class_labels, d.conf_thresh, d.iou_thresh, d.providers)
    else:
        raise ValueError(f"Неизвестный backend детектора: {d.backend}")
    mode = "off" if d.tiles in (False, None) else str(d.tiles)   
    return base if mode == "off" else TiledDetector(base, mode, d.tile_grid, d.tile_overlap)


def is_multiclass(detector) -> bool:
    return len(getattr(detector, "labels", [])) > 1
