from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort
from fast_plate_ocr.core.process import preprocess_image, resize_image
from fast_plate_ocr.inference.config import PlateConfig

from .config import ALLOWED_LETTERS, DIGITS, OcrConfig, PipelineConfig, TwoLineConfig

L, D, P = "L", "D", "P"
TEMPLATES: dict[str, list[str]] = {
    "type1": [L, D, D, D, L, L, D, D, "D?"],
    "type1a": [L, D, D, D, L, L, D, D, "D?"],
    "type1b": [L, L, D, D, D, D, D, "D?", P],
}
BODY_LEN = {"type1": 6, "type1a": 6, "type1b": 5}


@dataclass
class OcrDecoded:
    text: str
    char_probs: list[float]
    valid: bool
    reason: str = ""
    raw_text: str = ""

    @property
    def confidence(self) -> float:
        return float(np.mean(self.char_probs)) if self.char_probs else 0.0


@dataclass
class SlotMask:
    allowed: list[np.ndarray] = field(default_factory=list)


class PlateOCR:
    def __init__(self, model_path: Path, plate_config_path: Path, providers: Sequence[str] | None = None) -> None:
        if not model_path.is_file() or not plate_config_path.is_file():
            raise FileNotFoundError(f"нет модели OCR или ее конфига: {model_path}, {plate_config_path}")
        self.cfg = PlateConfig.from_yaml(plate_config_path)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_path), sess_options=opts,
                                            providers=list(providers) if providers else ort.get_available_providers())
        self.input_name = self.session.get_inputs()[0].name
        names = [o.name for o in self.session.get_outputs()]
        self.output_name = "plate" if "plate" in names else names[0]
        self.alphabet = list(self.cfg.alphabet)
        self.pad_idx = self.cfg.pad_idx
        self.slots = self.cfg.max_plate_slots
        if self.slots < 9:
            raise ValueError("max_plate_slots должен быть >= 9 (A123BC456)")
        idx = {ch: i for i, ch in enumerate(self.alphabet)}
        letters = np.array([idx[c] for c in ALLOWED_LETTERS if c in idx], dtype=int)
        digits = np.array([idx[c] for c in DIGITS if c in idx], dtype=int)
        pad = np.array([self.pad_idx], dtype=int)
        if len(letters) == 0 or len(digits) == 0:
            raise ValueError("в алфавите OCR нет нужных букв или цифр")
        self._sets = {L: letters, D: digits, "D?": np.concatenate([digits, pad]), P: pad}
        self._masks = {t: self._build_mask(tpl) for t, tpl in TEMPLATES.items()}

    def _build_mask(self, template: list[str]) -> SlotMask:
        allowed = [self._sets[t] for t in template]
        allowed += [self._sets[P]] * (self.slots - len(template))
        return SlotMask(allowed)

    def _to_model_input(self, plate_bgr: np.ndarray) -> np.ndarray:
        if self.cfg.image_color_mode == "grayscale":
            img = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB)
        img = resize_image(img, self.cfg.img_height, self.cfg.img_width,
                           image_color_mode=self.cfg.image_color_mode,
                           keep_aspect_ratio=self.cfg.keep_aspect_ratio,
                           interpolation_method=self.cfg.interpolation,
                           padding_color=self.cfg.padding_color)
        return preprocess_image(img)

    def probabilities(self, plate_bgr: np.ndarray) -> np.ndarray:
        x = self._to_model_input(plate_bgr)
        out = self.session.run([self.output_name], {self.input_name: x})[0]
        probs = out.reshape(-1, self.slots, len(self.alphabet))[0].astype(np.float64)
        if probs.min() < 0 or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-2):
            e = np.exp(probs - probs.max(axis=1, keepdims=True))
            probs = e / e.sum(axis=1, keepdims=True)
        return probs

    def decode(self, probs: np.ndarray, plate_type: str, min_char_prob: float = 0.0) -> OcrDecoded:
        mask = self._masks[plate_type]
        raw = "".join(self.alphabet[int(i)] for i in probs.argmax(axis=1)).rstrip(self.cfg.pad_char)
        chars: list[str] = []
        confs: list[float] = []
        for slot, allowed in enumerate(mask.allowed):
            sub = probs[slot, allowed]
            j = int(allowed[int(sub.argmax())])
            p = float(probs[slot, j])
            if j == self.pad_idx:
                break
            chars.append(self.alphabet[j])
            confs.append(p)
        text = "".join(chars)
        if min_char_prob > 0:
            text = "".join("#" if p < min_char_prob else c for c, p in zip(text, confs))
        valid, reason = self.validate(text, plate_type)
        return OcrDecoded(text, confs, valid, reason, raw)

    @staticmethod
    def validate(text: str, plate_type: str) -> tuple[bool, str]:
        body = BODY_LEN[plate_type]
        if len(text) not in (body + 2, body + 3):
            return False, f"длина {len(text)}"
        region = text[body:]
        if "#" in region:
            return True, ""
        if set(region) == {"0"}:
            return False, f"регион {region}"
        if len(region) == 3 and region[0] == "0":
            return False, f"регион {region}"
        return True, ""

    def read(self, plate_bgr: np.ndarray, plate_type: str, min_char_prob: float = 0.0) -> OcrDecoded:
        return self.decode(self.probabilities(plate_bgr), plate_type, min_char_prob)


def two_line_to_single(plate_bgr: np.ndarray, cfg: TwoLineConfig) -> np.ndarray:
    h, w = plate_bgr.shape[:2]
    split = int(h * cfg.split)
    ov = int(h * cfg.overlap)
    top = plate_bgr[max(0, int(h * 0.03)):min(h, split + ov)]
    bottom = plate_bgr[max(0, split - ov):int(h * 0.97), :max(1, int(w * (1.0 - cfg.flag_cut)))]
    rh = cfg.row_height

    def fit(img: np.ndarray) -> np.ndarray:
        ih, iw = img.shape[:2]
        return cv2.resize(img, (max(1, int(iw * rh / ih)), rh), interpolation=cv2.INTER_LINEAR)

    top, bottom = fit(top), fit(bottom)
    gap = np.full((rh, max(2, rh // 8), 3), np.median(top.reshape(-1, 3), axis=0), dtype=np.uint8)
    return np.concatenate([top, gap, bottom], axis=1)


def build_ocr(cfg: PipelineConfig, ocr_cfg: OcrConfig | None = None) -> PlateOCR:
    o = ocr_cfg or cfg.ocr
    return PlateOCR(cfg.resolve(o.model), cfg.resolve(o.plate_config), o.providers)
