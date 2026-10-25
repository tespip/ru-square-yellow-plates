from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SOLUTION_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = SOLUTION_ROOT / "config.yaml"

TARGET_TYPES = ("type1", "type1a", "type1b")
OTHER = "other"
ALLOWED_LETTERS = "ABEKMHOPCTYX"
DIGITS = "0123456789"


@dataclass
class DetectorConfig:
    backend: str = "ultralytics" 
    model: str = "weights/detector_416.onnx"
    class_labels: list[str] = field(default_factory=lambda: ["type1", "type1a", "type1b"])
    conf_thresh: float = 0.30
    iou_thresh: float = 0.45
    providers: list[str] | None = None
    tiles: str = "off"       
    tile_grid: tuple[int, int] = (2, 2)  
    tile_overlap: float = 0.2            


@dataclass
class OcrConfig:
    model: str = "weights/ocr_ru.onnx"
    plate_config: str = "weights/ocr_ru_plate_config.yaml"
    providers: list[str] | None = None


@dataclass
class RectifyConfig:
    enabled: bool = True
    margin: float = 0.15
    single_line_size: tuple[int, int] = (300, 65)
    two_line_size: tuple[int, int] = (232, 136)


@dataclass
class ClassifyConfig:
    mode: str = "auto" 
    yellow_min_fraction: float = 0.30
    white_max_yellow_fraction: float = 0.10
    two_line_max_aspect: float = 2.3
    single_line_min_aspect: float = 3.2
    ambiguous_split_aspect: float = 2.6


@dataclass
class TwoLineConfig:
    split: float = 0.49     
    overlap: float = 0.02
    flag_cut: float = 0.0     
    row_height: int = 64


@dataclass
class PostprocessConfig:
    min_confidence: float = 0.50
    min_char_prob: float = 0.0
    emit_other: bool = True


@dataclass
class VehicleCheckConfig:
    enabled: bool = False
    model: str = "weights/yolo11n_coco.onnx"
    conf_thresh: float = 0.25
    min_overlap: float = 0.70


@dataclass
class PipelineConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    rectify: RectifyConfig = field(default_factory=RectifyConfig)
    classify: ClassifyConfig = field(default_factory=ClassifyConfig)
    two_line: TwoLineConfig = field(default_factory=TwoLineConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    vehicle_check: VehicleCheckConfig = field(default_factory=VehicleCheckConfig)
    root: Path = SOLUTION_ROOT

    def resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.root / p


_SECTIONS = {
    "detector": DetectorConfig,
    "ocr": OcrConfig,
    "rectify": RectifyConfig,
    "classify": ClassifyConfig,
    "two_line": TwoLineConfig,
    "postprocess": PostprocessConfig,
    "vehicle_check": VehicleCheckConfig,
}


def _build(cls, data: dict[str, Any] | None):
    data = dict(data or {})
    for key in ("single_line_size", "two_line_size", "tile_grid"):
        if key in data and data[key] is not None:
            data[key] = tuple(int(v) for v in data[key])
    return cls(**data)


def load_config(path: str | os.PathLike | None = None) -> PipelineConfig:
    cfg_path = Path(path) if path else Path(os.environ.get("ALPR_CONFIG", DEFAULT_CONFIG))
    raw: dict[str, Any] = {}
    if cfg_path.is_file():
        with open(cfg_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    sections = {name: _build(cls, raw.get(name)) for name, cls in _SECTIONS.items()}
    root = cfg_path.resolve().parent if cfg_path.is_file() else SOLUTION_ROOT
    return PipelineConfig(root=root, **sections)
