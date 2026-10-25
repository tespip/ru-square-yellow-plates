from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import OTHER, TARGET_TYPES, PostprocessConfig
from .ocr import OcrDecoded

CSV_HEADER = ("image", "plate_num", "plate_type", "confidence")


@dataclass
class PlateRow:
    plate_num: str
    plate_type: str
    confidence: float
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    note: str = ""

    def iou(self, other: "PlateRow") -> float:
        ix = max(0, min(self.x2, other.x2) - max(self.x1, other.x1))
        iy = max(0, min(self.y2, other.y2) - max(self.y1, other.y1))
        inter = ix * iy
        a = (self.x2 - self.x1) * (self.y2 - self.y1) + (other.x2 - other.x1) * (other.y2 - other.y1) - inter
        return inter / a if a > 0 else 0.0


def combine_confidence(det_conf: float, ocr_conf: float) -> float:
    return float(min(1.0, max(0.0, math.sqrt(max(det_conf, 0.0) * max(ocr_conf, 0.0)))))


def finalize(plate_type: str, decoded: OcrDecoded | None, det_conf: float, cfg: PostprocessConfig,
             box: tuple[int, int, int, int], is_vehicle: bool = True) -> PlateRow | None:
    x1, y1, x2, y2 = box
    if plate_type not in TARGET_TYPES or decoded is None:
        return PlateRow("", OTHER, round(det_conf, 3), x1, y1, x2, y2, "not-target") if cfg.emit_other else None
    conf = combine_confidence(det_conf, decoded.confidence)
    text = decoded.text
    if not is_vehicle:
        return PlateRow(text, OTHER, conf, x1, y1, x2, y2, "not-vehicle") if cfg.emit_other else None
    if not decoded.valid:
        return PlateRow(text, OTHER, conf, x1, y1, x2, y2, f"invalid:{decoded.reason}") if cfg.emit_other else None
    if conf < cfg.min_confidence:
        return PlateRow(text, OTHER, conf, x1, y1, x2, y2, "low-confidence") if cfg.emit_other else None
    return PlateRow(text, plate_type, conf, x1, y1, x2, y2)


def deduplicate(rows: list[PlateRow], iou_thresh: float = 0.5) -> list[PlateRow]:
    rows = sorted(rows, key=lambda r: -r.confidence)
    kept: list[PlateRow] = []
    for r in rows:
        dup = False
        for k in kept:
            same_text = r.plate_num and r.plate_num == k.plate_num and r.plate_type == k.plate_type
            if same_text or r.iou(k) >= iou_thresh:
                dup = True
                break
        if not dup:
            kept.append(r)
    return sorted(kept, key=lambda r: (r.x1, r.y1))


def write_csv(path: Path, records: Iterable[tuple[str, PlateRow]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";", lineterminator="\n")
        w.writerow(CSV_HEADER)
        for image, r in records:
            w.writerow([image, r.plate_num, r.plate_type, f"{r.confidence:.3f}"])
            n += 1
    return n
