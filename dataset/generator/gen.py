#!/usr/bin/env python3
#
# Portions of this file are derived from deep-anpr
# (https://github.com/matthewearl/deep-anpr, gen.py):
#
# Copyright (c) 2016 Matthew Earl
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Изменения: Python 3, цветной вывод в размере фона, перспектива через гомографию,
# ретушь старого номера, manifest фонов, meta.csv / labels в формате конкурса.
"""Генератор синтетического датасета российских ГРЗ (Волга-IT 2026).

    python generator/gen.py --n 5000 --seed 42 --bg-dir generator/backgrounds \
        --fonts generator/fonts --out . [--merge] [--preview]

Пути по умолчанию считаются от папки скрипта: --bg-dir generator/backgrounds,
--fonts generator/fonts, --out — папка датасета (родитель generator/).
Работает офлайн, зависимости — numpy, opencv-python, Pillow.
"""

from __future__ import annotations

import argparse
import codecs
import csv
import math
import random
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import effects  # noqa: E402
import plates  # noqa: E402

GENERATOR_VERSION = "v1"

META_HEADER = "image;plate_num;plate_type;bbox;quad;is_vehicle;is_synthetic;source;license;conditions"
META_COLUMNS = META_HEADER.split(";")
CONDITIONS = ("day", "night", "rain", "snow", "dirt", "glare", "motion_blur", "angle")
PLATE_TYPES = ("type1", "type1a", "type1b", "other")
OTHER_KINDS = ("moto", "eu", "us")
LABEL_CLASSES = {"type1": 0, "type1a": 1, "type1b": 2, "other": 3}

DEFAULT_MIX = "type1=0.2,type1a=0.35,type1b=0.35,other=0.1"
DEFAULT_OTHER_MIX = "moto=0.4,eu=0.35,us=0.25"
JPEG_QUALITY = 92                    # фиксированное качество итогового JPEG
MIN_PLATE_H_PX = 12                  # ниже — пример отбрасывается
MAX_ATTEMPTS = 40                    # попыток разместить знак на одном изображении
NONVEHICLE_FRAC = 0.05               # доля примеров на фонах is_vehicle=0
MIN_BG_SIDE = 240                    # фоны меньше — отбрасываются (это кропы, а не кадры)
MAX_OLD_PLATE_FRAC = 0.25            # старый номер больше этой доли кадра — это кроп знака
SINGLE_LINE_MIN_BBOX_ASPECT = 2.1    # type1/type1b уже — валидатор ругается, перегенерировать
INPAINT_MARGIN = (0.03, 0.08)        # отступ ретуши старого номера (доля ширины, высоты), только чтобы захватить край
INPAINT_SMOOTH = 0.2                 # сглаживание результата inpaint (доля высоты маски), убирает полосы
PLATE_FRAME_PROB = 0.6               # доля знаков в пластиковой рамке (закрывает шов, как на реальных авто)
PLATE_FRAME_SIDE_MM = (6.0, 14.0)    # ширина рамки сверху и по бокам
PLATE_FRAME_BOTTOM_MM = (10.0, 26.0) # снизу рамка обычно шире
PLATE_FRAME_GRAY = (12, 55)          # яркость пластика
OLD_PLATE_STD_WIDTH_MM = 520.0       # старый номер считается стандартной ширины
FOREIGN_SCALE = (0.6, 1.4)           # масштаб при old_plate_std=0 (к ширине старого номера)
FREE_PLATE_H_FRAC = (0.02, 0.22)     # высота знака без plate_bbox, доля высоты кадра

# Знак на месте старого номера ставится в ту же плоскость и под тем же углом, что и старый:
# углы старого номера ищутся порогом Оцу + контуром в окрестности plate_bbox.
OLD_PLATE_SIZE_MM = (520.0, 112.0)   # старый номер считается type1 (сверить с ГОСТ Р 50577-2018)
OLD_QUAD_SEARCH_MARGIN = (0.25, 0.6) # запас поиска вокруг plate_bbox (доля ширины, высоты)
OLD_QUAD_AREA = (0.35, 1.6)          # площадь контура относительно plate_bbox
OLD_QUAD_ASPECT = (2.0, 7.5)         # допустимое отношение сторон найденного четырёхугольника
OLD_QUAD_MIN_IOU = 0.45              # совпадение описанного прямоугольника с plate_bbox
OLD_QUAD_THRESH_SHIFTS = (0, 20, 40, -20, 60)  # сдвиги порога Оцу: знак не сливается со светлым бампером
OLD_QUAD_MAX_SKEW_DEG = (4.0, 6.0)   # макс. расхождение углов верх/низ и лево/право; больше — берём повёрнутый прямоугольник
OLD_QUAD_MAX_OUTSIDE = (0.10, 0.35)  # насколько четырёхугольник может выходить за plate_bbox (доля ширины, высоты)
# Тег angle для знака на месте старого номера: по форме найденного четырёхугольника.
ANGLE_ROLL_DEG = 8.0                 # наклон верхней кромки больше этого
ANGLE_MIN_ASPECT = 3.6               # сильное сжатие по ширине (поворот > ~40° для 520x112)
ANGLE_EDGE_RATIO = 1.15              # разница длин противоположных сторон (перспектива)
# Если углы старого номера не нашлись: почти фронтально относительно машины.
ALIGNED_YAW_SIGMA, ALIGNED_YAW_MAX = 0.03, 0.06
ALIGNED_PITCH_SIGMA, ALIGNED_PITCH_MAX = 0.02, 0.05
ALIGNED_ROLL_SIGMA, ALIGNED_ROLL_MAX = 0.008, 0.02
# Знак в случайном месте кадра (нет plate_bbox): любые ракурсы.
YAW_SIGMA, YAW_MAX = 0.25, 0.60
PITCH_SIGMA, PITCH_MAX = 0.12, 0.30
ROLL_SIGMA, ROLL_MAX = 0.06, 0.15
ANGLE_TAG_THRESHOLD = 0.30           # |yaw| или |pitch| выше -> тег angle
DIST_FACTOR = (2.5, 6.0)             # расстояние до камеры в размерах плашки (меньше — сильнее перспектива)

# Имена как у реальных фото: <тип>_<номер 4 цифры>, нумерация продолжает images/real
# (первый синтетический type1a — следующий после последнего реального type1a_NNNN).
NAME_DIGITS = 4
REAL_NAME_RE = re.compile(r"^(type1|type1a|type1b|other)_(\d+)\.(jpg|jpeg|png|bmp)$", re.IGNORECASE)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MANIFEST_REQUIRED = ("file", "source", "license", "author", "plate_bbox", "is_vehicle", "faces_ok")


class GenError(RuntimeError):
    pass


# ============================================================================
# Шрифты и фоны
# ============================================================================
def load_fonts(font_dir: Path) -> list[plates.FontGlyphs]:
    if not font_dir.is_dir():
        raise GenError(f"папка шрифтов не найдена: {font_dir}")
    ttfs = sorted(p for p in font_dir.iterdir() if p.suffix.lower() == ".ttf")
    if not ttfs:
        raise GenError(f"в {font_dir} нет ни одного .ttf")
    licenses = [p for p in font_dir.iterdir()
                if p.is_file() and (p.name.upper().startswith(("OFL", "LICENSE")) or "LICENSE" in p.name.upper())]
    if not licenses:
        raise GenError(f"в {font_dir} нет файла лицензии шрифтов (OFL.txt / LICENSE*) — положите его рядом со шрифтами")
    fonts = []
    for path in ttfs:
        if "uknumberplate" in path.name.lower():
            raise GenError(f"{path.name}: шрифт UKNumberPlate не используется")
        font = plates.FontGlyphs(path)
        try:
            font.check()
        except plates.FontError as exc:
            raise GenError(str(exc)) from exc
        fonts.append(font)
    return fonts


@dataclass
class Background:
    path: Path
    rel: str
    source: str
    license: str
    author: str
    bbox: tuple[int, int, int, int] | None
    is_vehicle: int
    old_plate_std: int
    size: tuple[int, int]


def _clean(value: str) -> str:
    """Внутри значений meta.csv не должно быть ';' и переводов строк."""
    return " ".join(value.replace(";", " | ").split())


def load_backgrounds(bg_dir: Path) -> tuple[list[Background], Counter]:
    manifest = bg_dir / "manifest.csv"
    if not bg_dir.is_dir():
        raise GenError(f"папка фонов не найдена: {bg_dir}")
    if not manifest.is_file():
        raise GenError(f"нет {manifest} — без него генератор не знает, какие фоны брать")
    rejected: Counter = Counter()
    bgs: list[Background] = []
    listed: set[str] = set()
    with open(manifest, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        missing = [c for c in MANIFEST_REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise GenError(f"{manifest}: нет колонок {missing} (нужны {';'.join(MANIFEST_REQUIRED)})")
        for row in reader:
            rel = (row.get("file") or "").strip().replace("\\", "/")
            if not rel:
                rejected["пустое поле file"] += 1
                continue
            listed.add(rel)
            path = bg_dir / rel
            source, lic = (row.get("source") or "").strip(), (row.get("license") or "").strip()
            is_vehicle, faces_ok = (row.get("is_vehicle") or "").strip(), (row.get("faces_ok") or "").strip()
            old_std = (row.get("old_plate_std") or "1").strip() or "1"
            if not path.is_file():
                rejected["файла нет на диске"] += 1
                continue
            if not lic:
                rejected["пустая license"] += 1
                continue
            if not source:
                rejected["пустой source"] += 1
                continue
            if faces_ok != "1":
                rejected["faces_ok != 1"] += 1
                continue
            if is_vehicle not in ("0", "1") or old_std not in ("0", "1"):
                rejected["is_vehicle / old_plate_std не 0/1"] += 1
                continue
            try:
                with Image.open(path) as im:
                    size = im.size
            except OSError:
                rejected["не читается как изображение"] += 1
                continue
            if min(size) < MIN_BG_SIDE:
                rejected[f"кадр меньше {MIN_BG_SIDE}px"] += 1
                continue
            bbox = None
            raw_bbox = (row.get("plate_bbox") or "").strip()
            if raw_bbox:
                try:
                    x, y, w, h = (int(round(float(v))) for v in raw_bbox.split(","))
                except ValueError:
                    rejected["plate_bbox не x,y,w,h"] += 1
                    continue
                if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > size[0] or y + h > size[1]:
                    rejected["plate_bbox вне кадра"] += 1
                    continue
                if w * h > MAX_OLD_PLATE_FRAC * size[0] * size[1]:
                    rejected["кроп знака, а не кадр"] += 1
                    continue
                bbox = (x, y, w, h)
            bgs.append(Background(path, rel, _clean(source), _clean(lic), _clean(row.get("author") or ""),
                                  bbox, int(is_vehicle), int(old_std), size))
    for p in bg_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.relative_to(bg_dir).as_posix() not in listed:
            rejected["нет в manifest"] += 1
    return bgs, rejected


def read_image(path: Path) -> np.ndarray:
    # Без поворота по EXIF: координаты plate_bbox в manifest считаются по «сырому» кадру.
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise GenError(f"не удалось прочитать фон {path}")
    return img


# ============================================================================
# Геометрия
# ============================================================================
def euler_to_mat(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Матрица поворота (из deep-anpr gen.py): рысканье, тангаж, крен."""
    c, s = math.cos(yaw), math.sin(yaw)
    m = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    c, s = math.cos(pitch), math.sin(pitch)
    m = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]) @ m
    c, s = math.cos(roll), math.sin(roll)
    m = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ m
    return m


def project_corners(w_px: float, h_px: float, cx: float, cy: float,
                    yaw: float, pitch: float, roll: float, dist_factor: float) -> np.ndarray:
    """4 угла плашки (TL, TR, BR, BL шаблона) после поворота и перспективной проекции."""
    corners = np.array([[-w_px / 2, -h_px / 2, 0.0], [w_px / 2, -h_px / 2, 0.0],
                        [w_px / 2, h_px / 2, 0.0], [-w_px / 2, h_px / 2, 0.0]])
    p = corners @ euler_to_mat(yaw, pitch, roll).T
    d = dist_factor * max(w_px, h_px)
    z = d + p[:, 2]
    return np.stack([cx + p[:, 0] * d / z, cy + p[:, 1] * d / z], axis=1)


def signed_area(q: np.ndarray) -> float:
    """В экранных координатах (y вниз) положительная площадь = обход по часовой стрелке."""
    x, y = q[:, 0].astype(float), q[:, 1].astype(float)
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def quad_ok(q: np.ndarray, W: int, H: int) -> bool:
    """Внутри кадра, по часовой, первая точка — левая верхняя и по правилу валидатора (min x+y)."""
    if (q[:, 0] < 0).any() or (q[:, 1] < 0).any() or (q[:, 0] > W - 1).any() or (q[:, 1] > H - 1).any():
        return False
    return signed_area(q) > 1.0 and int(np.argmin(q.sum(axis=1))) == 0


def inpaint_old(frame: np.ndarray, bbox: tuple[int, int, int, int], rng: np.random.Generator) -> np.ndarray:
    """Ретушь старого номера: cv2.inpaint по маске с отступом, затем сглаживание и шум фактуры.

    Голый inpaint на широкой области тянет полосы от краёв старого номера; если новая плашка
    уже старой, они остаются видны по бокам. Сглаживание внутри маски и шум с дисперсией
    окружения дают ровный «пластик/кузов» без мазков.
    """
    H, W = frame.shape[:2]
    x, y, w, h = bbox
    mx, my = max(2, int(w * INPAINT_MARGIN[0])), max(2, int(h * INPAINT_MARGIN[1]))
    x0, y0, x1, y1 = max(0, x - mx), max(0, y - my), min(W, x + w + mx), min(H, y + h + my)
    rx0, ry0, rx1, ry1 = max(0, x0 - 2 * mx), max(0, y0 - 2 * my), min(W, x1 + 2 * mx), min(H, y1 + 2 * my)
    roi = frame[ry0:ry1, rx0:rx1]
    mask = np.zeros(roi.shape[:2], np.uint8)
    mask[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0] = 255
    filled = cv2.inpaint(roi, mask, 5, cv2.INPAINT_TELEA).astype(np.float32)
    sigma = max(1.0, INPAINT_SMOOTH * (y1 - y0))
    smooth = cv2.GaussianBlur(filled, (0, 0), sigma)
    ring = roi[(cv2.dilate(mask, np.ones((max(3, my), max(3, mx)), np.uint8)) > 0) & (mask == 0)]
    std = float(ring.std(axis=0).mean()) if ring.size else 4.0
    smooth += rng.normal(0.0, min(std, 12.0) * 0.5, smooth.shape).astype(np.float32)
    # Растушёвка только внутрь маски: за границу ретуши ничего не размывается.
    feather = max(1, min(mx, my) // 2)
    inner = cv2.erode(mask, np.ones((2 * feather + 1, 2 * feather + 1), np.uint8))
    m = cv2.GaussianBlur(inner.astype(np.float32) / 255.0, (0, 0), feather / 2 + 0.5)
    m = np.where(mask > 0, np.maximum(m, (inner > 0).astype(np.float32)), 0.0)[..., None]
    out = frame.copy()
    out[ry0:ry1, rx0:rx1] = np.clip(smooth * m + roi.astype(np.float32) * (1 - m), 0, 255).astype(np.uint8)
    return out


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Порядок: по часовой стрелке (экранные координаты) начиная с левого верхнего."""
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    c = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    return np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)


def _quad_wh(q: np.ndarray) -> tuple[float, float]:
    w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    return float(w), float(h)


_OLD_QUAD_CACHE: dict[str, np.ndarray | None] = {}


def _edge_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _quad_from_contour(cnt: np.ndarray) -> np.ndarray:
    """Четырёхугольник по выпуклой оболочке; если он кривой — повёрнутый прямоугольник (сохраняет наклон)."""
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    q = None
    for eps in (0.02, 0.03, 0.04, 0.05, 0.07):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            q = order_quad(approx.reshape(4, 2))
            break
        if len(approx) < 4:
            break
    if q is not None:
        top, bottom = _edge_deg(q[0], q[1]), _edge_deg(q[3], q[2])
        left, right = _edge_deg(q[0], q[3]), _edge_deg(q[1], q[2])
        if _angle_diff(top, bottom) <= OLD_QUAD_MAX_SKEW_DEG[0] and _angle_diff(left, right) <= OLD_QUAD_MAX_SKEW_DEG[1]:
            return q
    return order_quad(cv2.boxPoints(cv2.minAreaRect(hull)))


def find_old_quad(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    """4 угла старого номера внутри plate_bbox (светлая плашка) или None.

    Перебираются несколько порогов бинаризации; из подходящих контуров (накрывают центр рамки,
    площадь и пропорции как у номера) берётся самый «прямоугольный» с лучшим совпадением с рамкой.
    """
    H, W = frame.shape[:2]
    x, y, w, h = bbox
    mx, my = int(w * OLD_QUAD_SEARCH_MARGIN[0]), int(h * OLD_QUAD_SEARCH_MARGIN[1])
    x0, y0, x1, y1 = max(0, x - mx), max(0, y - my), min(W, x + w + mx), min(H, y + h + my)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0 or min(crop.shape[:2]) < 8:
        return None
    gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    center = (float(x + w / 2 - x0), float(y + h / 2 - y0))
    offset = np.array([x0, y0], dtype=np.float64)
    best, best_score = None, -1.0
    for shift in OLD_QUAD_THRESH_SHIFTS:
        t = float(np.clip(otsu + shift, 10, 245))
        _, thr = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not OLD_QUAD_AREA[0] * w * h <= area <= OLD_QUAD_AREA[1] * w * h:
                continue
            if cv2.pointPolygonTest(cnt, center, False) < 0:
                continue
            q = _quad_from_contour(cnt) + offset
            qw, qh = _quad_wh(q)
            if qh < 4 or not OLD_QUAD_ASPECT[0] <= qw / qh <= OLD_QUAD_ASPECT[1]:
                continue
            qx0, qy0 = q.min(axis=0)
            qx1, qy1 = q.max(axis=0)
            ox, oy = OLD_QUAD_MAX_OUTSIDE[0] * w, OLD_QUAD_MAX_OUTSIDE[1] * h
            if qx0 < x - ox or qx1 > x + w + ox or qy0 < y - oy or qy1 > y + h + oy:
                continue
            iw = max(0.0, min(qx1, x + w) - max(qx0, x))
            ih = max(0.0, min(qy1, y + h) - max(qy0, y))
            inter = iw * ih
            iou = inter / ((qx1 - qx0) * (qy1 - qy0) + w * h - inter)
            if iou < OLD_QUAD_MIN_IOU:
                continue
            rectangularity = area / max(1.0, abs(signed_area(q)))
            score = iou * min(1.0, rectangularity)
            if score > best_score:
                best, best_score = q, score
    return best


def old_quad_for(bg: "Background") -> np.ndarray | None:
    """Углы старого номера для фона (вычисляется один раз; детерминировано, на seed не влияет)."""
    if bg.rel not in _OLD_QUAD_CACHE:
        assert bg.bbox is not None
        _OLD_QUAD_CACHE[bg.rel] = find_old_quad(read_image(bg.path), bg.bbox)
    return _OLD_QUAD_CACHE[bg.rel]


def quad_is_angled(q: np.ndarray) -> bool:
    qw, qh = _quad_wh(q)
    roll = abs(math.degrees(math.atan2(q[1][1] - q[0][1], q[1][0] - q[0][0])))
    top, bottom = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])
    left, right = np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])
    edge = max(top, bottom) / max(1e-6, min(top, bottom)), max(left, right) / max(1e-6, min(left, right))
    return roll > ANGLE_ROLL_DEG or qw / qh < ANGLE_MIN_ASPECT or max(edge) > ANGLE_EDGE_RATIO


def place_like_old(old_q: np.ndarray, w_mm: float, h_mm: float, scale: float) -> np.ndarray:
    """Новая плашка в плоскости старого номера, по его центру: та же гомография, тот же угол."""
    ow, oh = OLD_PLATE_SIZE_MM
    src = np.float32([[0, 0], [ow, 0], [ow, oh], [0, oh]])
    hom = cv2.getPerspectiveTransform(src, old_q.astype(np.float32))
    nw, nh = w_mm * scale, h_mm * scale
    cx, cy = ow / 2, oh / 2
    new = np.float32([[cx - nw / 2, cy - nh / 2], [cx + nw / 2, cy - nh / 2],
                      [cx + nw / 2, cy + nh / 2], [cx - nw / 2, cy + nh / 2]])
    return cv2.perspectiveTransform(new.reshape(-1, 1, 2), hom).reshape(-1, 2).astype(np.float64)


def plate_frame_poly(m_full: np.ndarray, tw: int, th: int, w_mm: float, rng: np.random.Generator) -> np.ndarray:
    """Пластиковая рамка вокруг знака: прямоугольник шире шаблона, прогнанный через ту же гомографию."""
    k = tw / w_mm
    side = rng.uniform(*PLATE_FRAME_SIDE_MM) * k
    bottom = rng.uniform(*PLATE_FRAME_BOTTOM_MM) * k
    poly = np.float32([[-side, -side], [tw + side, -side], [tw + side, th + bottom], [-side, th + bottom]])
    return cv2.perspectiveTransform(poly.reshape(-1, 1, 2), m_full).reshape(-1, 2)


# ============================================================================
# Один пример
# ============================================================================
@dataclass(frozen=True)
class PlateInfo:
    plate_num: str
    plate_type: str
    kind: str


@dataclass
class Sample:
    index: int
    name: str
    plate: "PlateInfo"      # без изображений плашки: Sample хранится для всех кадров до конца запуска
    quad: np.ndarray
    bbox: tuple[int, int, int, int]
    is_vehicle: int
    conditions: list[str]
    bg: Background
    width: int
    height: int
    attempts: int


@dataclass
class Context:
    fonts: list[plates.FontGlyphs]
    vehicle_pool: list[Background]
    nonvehicle_pool: list[Background]
    mix: dict[str, float]
    other_mix: dict[str, float]
    nonvehicle_frac: float


def _choice(rng: np.random.Generator, weights: dict[str, float]) -> str:
    keys = list(weights)
    p = np.array([weights[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def _angle(rng: np.random.Generator, sigma: float, limit: float) -> float:
    return float(np.clip(rng.normal(0.0, sigma), -limit, limit))


def try_place(plate: plates.Plate, bg: Background, use_dirt: bool,
              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int, bool] | None:
    """Одна попытка: (кадр, quad float 4x2, is_vehicle, angle) или None, если пример не подходит."""
    W, H = bg.size
    w_mm, h_mm = plate.size_mm
    old_q = None
    if bg.bbox is not None:
        x, y, w, h = bg.bbox
        ratio = 1.0 if bg.old_plate_std else rng.uniform(*FOREIGN_SCALE)
        is_vehicle = bg.is_vehicle
        old_q = old_quad_for(bg)
    if old_q is not None:
        # Та же плоскость и тот же угол, что у старого номера.
        dst = place_like_old(old_q, w_mm, h_mm, ratio)
        angle = quad_is_angled(old_q)
    else:
        if bg.bbox is not None:
            px_per_mm = w * ratio / OLD_PLATE_STD_WIDTH_MM
            cx, cy = x + w / 2, y + h / 2
            yaw = _angle(rng, ALIGNED_YAW_SIGMA, ALIGNED_YAW_MAX)
            pitch = _angle(rng, ALIGNED_PITCH_SIGMA, ALIGNED_PITCH_MAX)
            roll = _angle(rng, ALIGNED_ROLL_SIGMA, ALIGNED_ROLL_MAX)
        else:
            lo = max(MIN_PLATE_H_PX + 2.0, FREE_PLATE_H_FRAC[0] * H)
            hi = max(lo + 1.0, FREE_PLATE_H_FRAC[1] * H)
            px_per_mm = math.exp(rng.uniform(math.log(lo), math.log(hi))) / h_mm
            cx, cy = rng.uniform(0, W), rng.uniform(H / 3, H)
            # Случайное место на фоне: знак не обязательно на ТС, поэтому is_vehicle=0.
            is_vehicle = 0
            yaw, pitch, roll = (_angle(rng, YAW_SIGMA, YAW_MAX), _angle(rng, PITCH_SIGMA, PITCH_MAX),
                                _angle(rng, ROLL_SIGMA, ROLL_MAX))
        dst = project_corners(w_mm * px_per_mm, h_mm * px_per_mm, cx, cy, yaw, pitch, roll, rng.uniform(*DIST_FACTOR))
        angle = abs(yaw) > ANGLE_TAG_THRESHOLD or abs(pitch) > ANGLE_TAG_THRESHOLD
    q = np.round(dst).astype(np.int64)
    if not quad_ok(q, W, H):
        return None
    plate_h = (np.linalg.norm(dst[3] - dst[0]) + np.linalg.norm(dst[2] - dst[1])) / 2
    if plate_h < MIN_PLATE_H_PX:
        return None
    bw, bh = int(q[:, 0].max() - q[:, 0].min()), int(q[:, 1].max() - q[:, 1].min())
    if bh <= 0 or bw <= 0:
        return None
    if plate.plate_type in ("type1", "type1b") and bw / bh < SINGLE_LINE_MIN_BBOX_ASPECT:
        return None

    frame = read_image(bg.path)
    if (frame.shape[1], frame.shape[0]) != (W, H):
        raise GenError(f"размер {bg.rel} при чтении не совпал с заголовком")
    if bg.bbox is not None:
        frame = inpaint_old(frame, bg.bbox, rng)

    # Уменьшаем шаблон с запасом x2 до перспективы (INTER_AREA), иначе будет ступенчатость.
    target_w = max(np.linalg.norm(dst[1] - dst[0]), np.linalg.norm(dst[2] - dst[3]))
    ph_src, pw_src = plate.image.shape[:2]
    tw = int(min(pw_src, max(8, round(2 * target_w))))
    th = max(4, int(round(tw * ph_src / pw_src)))
    tpl = cv2.resize(plate.image, (tw, th), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(plate.alpha, (tw, th), interpolation=cv2.INTER_AREA)
    if use_dirt:
        tpl = effects.dirt(tpl, alpha, rng)
    x0, y0 = dst.min(axis=0)
    x1, y1 = dst.max(axis=0)
    tpl = effects.match_tone(tpl, frame, (x0, y0, x1, y1), rng)

    # Гомография шаблон -> кадр, варп только в окрестность знака (и рамки, если она есть).
    src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    frame_poly = None
    extent = dst
    if rng.random() < PLATE_FRAME_PROB:
        frame_poly = plate_frame_poly(cv2.getPerspectiveTransform(src, dst.astype(np.float32)), tw, th, w_mm, rng)
        extent = np.vstack([dst, frame_poly])
    ex0, ey0 = extent.min(axis=0)
    ex1, ey1 = extent.max(axis=0)
    rx0, ry0 = max(0, int(ex0) - 2), max(0, int(ey0) - 2)
    rx1, ry1 = min(W, int(math.ceil(ex1)) + 3), min(H, int(math.ceil(ey1)) + 3)
    shift = np.array([rx0, ry0], dtype=np.float64)
    m = cv2.getPerspectiveTransform(src, (dst - shift).astype(np.float32))
    size = (rx1 - rx0, ry1 - ry0)
    if frame_poly is not None:
        gray = int(rng.integers(PLATE_FRAME_GRAY[0], PLATE_FRAME_GRAY[1] + 1))
        cv2.fillConvexPoly(frame[ry0:ry1, rx0:rx1], np.round((frame_poly - shift) * 16).astype(np.int32),
                           (gray, gray, gray), cv2.LINE_AA, 4)
    warped = cv2.warpPerspective(tpl, m, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    wa = cv2.warpPerspective(alpha, m, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    wa = cv2.GaussianBlur(wa, (0, 0), 0.6)[..., None]
    roi = frame[ry0:ry1, rx0:rx1].astype(np.float32)
    frame[ry0:ry1, rx0:rx1] = np.clip(warped.astype(np.float32) * wa + roi * (1 - wa), 0, 255).astype(np.uint8)
    return frame, dst, is_vehicle, angle


def make_sample(index: int, base_seed: int, ctx: Context) -> tuple[Sample, np.ndarray]:
    seed = base_seed + index
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    cv2.setRNGSeed(seed % (2 ** 31))
    rng = np.random.default_rng(seed)

    plate_type = _choice(rng, ctx.mix)
    kind = plate_type if plate_type != "other" else _choice(rng, ctx.other_mix)
    font = ctx.fonts[int(rng.integers(len(ctx.fonts)))]
    plate = plates.make_plate(kind, font, rng)
    use_dirt = rng.random() < effects.PROB["dirt"]
    want_nonvehicle = bool(ctx.nonvehicle_pool) and rng.random() < ctx.nonvehicle_frac
    pool = ctx.nonvehicle_pool if want_nonvehicle or not ctx.vehicle_pool else ctx.vehicle_pool

    for attempt in range(1, MAX_ATTEMPTS + 1):
        bg = pool[int(rng.integers(len(pool)))]
        placed = try_place(plate, bg, use_dirt, rng)
        if placed is not None:
            break
    else:
        raise GenError(f"изображение {index}: не удалось разместить знак за {MAX_ATTEMPTS} попыток")
    frame, dst, is_vehicle, angle = placed

    conditions: set[str] = set()
    if use_dirt:
        conditions.add("dirt")
    if angle:
        conditions.add("angle")
    plate_h = (np.linalg.norm(dst[3] - dst[0]) + np.linalg.norm(dst[2] - dst[1])) / 2
    frame = effects.apply_frame_effects(frame, dst, plate_h, rng, conditions)

    H, W = frame.shape[:2]
    quad = np.round(dst).astype(np.int64)
    quad[:, 0] = np.clip(quad[:, 0], 0, W)
    quad[:, 1] = np.clip(quad[:, 1], 0, H)
    x0, y0 = max(0, int(quad[:, 0].min())), max(0, int(quad[:, 1].min()))
    x1, y1 = min(W, int(quad[:, 0].max())), min(H, int(quad[:, 1].max()))
    info = PlateInfo(plate.plate_num, plate.plate_type, plate.kind)
    # имя назначает run(): нумерация продолжает реальные фото
    sample = Sample(index, "", info, quad, (x0, y0, x1 - x0, y1 - y0), is_vehicle,
                    [c for c in CONDITIONS if c in conditions], bg, W, H, attempt)
    return sample, frame


# ============================================================================
# Запись
# ============================================================================
def write_label(path: Path, plate_type: str, bbox: tuple[int, int, int, int], quad: np.ndarray,
                width: int, height: int, fmt: str = "yolo") -> None:
    """labels/<имя>.txt. Формат labels в задании не описан (уточнить у организаторов).

    yolo:      "<class_id> <cx> <cy> <w> <h>", нормировано 0..1
    yolo_quad: то же + 8 чисел quad (x1 y1 ... x4 y4), нормировано 0..1
    """
    x, y, w, h = bbox
    parts = [str(LABEL_CLASSES[plate_type])]
    parts += [f"{v:.6f}" for v in ((x + w / 2) / width, (y + h / 2) / height, w / width, h / height)]
    if fmt == "yolo_quad":
        parts += [f"{v:.6f}" for px, py in quad for v in (px / width, py / height)]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(" ".join(parts) + "\n")


def meta_fields(s: Sample, base_seed: int) -> list[str]:
    source = f"synthetic generator {GENERATOR_VERSION} seed={base_seed} | bg={s.bg.source}"
    lic = f"CC BY 4.0 (generated) | bg: {s.bg.license}"
    if s.bg.author:
        lic += f" | author: {s.bg.author}"
    x, y, w, h = s.bbox
    return [
        f"images/synthetic/{s.name}.jpg",
        s.plate.plate_num,
        s.plate.plate_type,
        f"{x},{y},{w},{h}",
        ",".join(str(int(v)) for v in s.quad.flatten()),
        str(s.is_vehicle),
        "1",
        _clean(source),
        _clean(lic),
        ",".join(s.conditions),
    ]


def row_to_line(fields: list[str]) -> str:
    """Текстовые поля с запятыми (source, license) — в кавычках по правилам CSV.

    bbox, quad и conditions не кавычим: так же записаны существующие строки meta.csv,
    а при разделителе ';' запятая кавычек не требует.
    """
    out = []
    for name, value in zip(META_COLUMNS, fields):
        if ";" in value or "\n" in value or "\r" in value:
            raise GenError(f"недопустимый символ в поле {name}: {value!r}")
        if name in ("source", "license") and ("," in value or '"' in value):
            value = '"' + value.replace('"', '""') + '"'
        out.append(value)
    return ";".join(out)


def parse_line(line: str) -> list[str]:
    return next(csv.reader([line], delimiter=";"))


def is_generator_row(fields: list[str]) -> bool:
    return len(fields) == len(META_COLUMNS) and fields[6].strip() == "1" and fields[0].strip().startswith("images/synthetic/")


@dataclass
class MetaFile:
    path: Path
    bom: bool
    newline: str
    header: str
    lines: list[str] = field(default_factory=list)   # строки без переводов строк (кроме заголовка)


def read_meta(path: Path) -> MetaFile:
    raw = path.read_bytes()
    bom = raw.startswith(codecs.BOM_UTF8)
    try:
        text = raw[3:].decode("utf-8") if bom else raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GenError(f"{path} не в UTF-8 — ничего не записано") from exc
    first = text.split("\n", 1)[0]
    newline = "\r\n" if first.endswith("\r") else "\n"
    header = first.rstrip("\r")
    if header != META_HEADER:
        raise GenError(f"заголовок {path} не совпадает с ожидаемым — ничего не записано.\n"
                       f"  ожидается: {META_HEADER}\n  в файле:   {header}")
    body = text[len(first) + 1:] if "\n" in text else ""
    lines = body.split(newline) if newline == "\r\n" else body.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return MetaFile(path, bom, newline, header, lines)


def write_meta(meta: MetaFile, keep_lines: list[str], new_lines: list[str]) -> None:
    text = meta.header + meta.newline + "".join(line + meta.newline for line in keep_lines + new_lines)
    data = (codecs.BOM_UTF8 if meta.bom else b"") + text.encode("utf-8")
    meta.path.write_bytes(data)


def other_image_stems(out: Path, synth_dir: Path) -> set[str]:
    """Имена (без расширения) всех изображений датасета вне images/synthetic."""
    stems: set[str] = set()
    images = out / "images"
    if images.is_dir():
        for p in images.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and synth_dir not in p.parents:
                stems.add(p.stem)
    return stems


def start_indices(out: Path, override: dict[str, int]) -> dict[str, int]:
    """Первый номер синтетики по каждому типу: после последнего images/real/<тип>_NNNN (или из --start-index)."""
    start = {t: 1 for t in PLATE_TYPES}
    real = out / "images" / "real"
    if real.is_dir():
        for p in real.iterdir():
            m = REAL_NAME_RE.match(p.name)
            if m:
                t = m.group(1).lower()
                start[t] = max(start[t], int(m.group(2)) + 1)
    start.update(override)
    return start


def cleanup_old_outputs(images_dir: Path, labels_dir: Path, protected: set[str]) -> int:
    """Удаляет прежние картинки генератора (всё в images/synthetic) и их метки.

    Метки ищутся по именам картинок из images/synthetic, а не по шаблону: в labels/ лежат и метки
    реальных фото с такими же именами вида <тип>_NNNN. Метка, чьё имя совпадает с реальным фото, не трогается.
    """
    removed = 0
    if not images_dir.is_dir():
        return 0
    for p in images_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() in IMAGE_EXTS and p.stem not in protected:
            label = labels_dir / f"{p.stem}.txt"
            if label.is_file():
                label.unlink()
                removed += 1
        p.unlink()
        removed += 1
    return removed


# ============================================================================
# Самопроверка, превью
# ============================================================================
def self_check(out: Path, rows: list[list[str]]) -> list[str]:
    problems: list[str] = []
    for fields in rows:
        if len(fields) != len(META_COLUMNS):
            problems.append(f"{fields[:1]}: {len(fields)} полей вместо {len(META_COLUMNS)} (лишние ';'?)")
            continue
        rec = dict(zip(META_COLUMNS, fields))
        where = rec["image"]
        path = out / rec["image"]
        if not path.is_file():
            problems.append(f"{where}: файла нет")
            continue
        with Image.open(path) as im:
            W, H = im.size
        try:
            x, y, w, h = (int(v) for v in rec["bbox"].split(","))
            q = np.array([int(v) for v in rec["quad"].split(",")]).reshape(4, 2)
        except ValueError:
            problems.append(f"{where}: bbox/quad не разбираются")
            continue
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > W or y + h > H:
            problems.append(f"{where}: bbox вне кадра")
        if (q < 0).any() or (q[:, 0] > W).any() or (q[:, 1] > H).any():
            problems.append(f"{where}: quad вне кадра")
        if signed_area(q) <= 0:
            problems.append(f"{where}: quad не по часовой стрелке")
        if int(np.argmin(q.sum(axis=1))) != 0:
            problems.append(f"{where}: первая точка quad не левая верхняя")
        t, num = rec["plate_type"], rec["plate_num"]
        regex = {"type1": plates.TYPE1_RE, "type1a": plates.TYPE1_RE, "type1b": plates.TYPE1B_RE}.get(t, plates.OTHER_RE)
        if t not in PLATE_TYPES:
            problems.append(f"{where}: plate_type {t}")
        elif not regex.match(num):
            problems.append(f"{where}: plate_num {num} не по маске {t}")
        conds = rec["conditions"].split(",") if rec["conditions"] else []
        if not conds or any(c not in CONDITIONS for c in conds) or " " in rec["conditions"]:
            problems.append(f"{where}: conditions {rec['conditions']!r}")
        if rec["is_synthetic"] != "1" or rec["is_vehicle"] not in ("0", "1"):
            problems.append(f"{where}: is_synthetic/is_vehicle")
        label = out / "labels" / (Path(rec["image"]).stem + ".txt")
        if not label.is_file():
            problems.append(f"{where}: нет {label.name}")
    return problems


def make_preview(out: Path, rows: list[list[str]], path: Path, count: int = 24) -> None:
    tile_w, tile_h, cols = 260, 210, 6
    tiles = []
    for fields in rows[:count]:
        rec = dict(zip(META_COLUMNS, fields))
        img = cv2.imdecode(np.fromfile(str(out / rec["image"]), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        x, y, w, h = (int(v) for v in rec["bbox"].split(","))
        q = np.array([int(v) for v in rec["quad"].split(",")]).reshape(4, 2)
        cv2.polylines(img, [q.reshape(-1, 1, 2).astype(np.int32)], True, (0, 0, 255), max(1, w // 80))
        cv2.circle(img, (int(q[0][0]), int(q[0][1])), max(2, w // 40), (0, 255, 0), -1)
        m = int(max(w, h) * 1.2)
        crop = img[max(0, y - m):y + h + m, max(0, x - m):x + w + m]
        s = min(tile_w / crop.shape[1], (tile_h - 34) / crop.shape[0])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))))
        tile = np.full((tile_h, tile_w, 3), 255, np.uint8)
        tile[34:34 + crop.shape[0], :crop.shape[1]] = crop
        cv2.putText(tile, f"{rec['plate_type']} {rec['plate_num']} v={rec['is_vehicle']}", (3, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 180), 1, cv2.LINE_AA)
        cv2.putText(tile, rec["conditions"][:40], (3, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return
    rows_n = (len(tiles) + cols - 1) // cols
    grid = np.full((rows_n * tile_h, cols * tile_w, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = t
    ok, buf = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        path.write_bytes(buf.tobytes())


# ============================================================================
# CLI
# ============================================================================
def parse_mix(text: str, allowed: tuple[str, ...]) -> dict[str, float]:
    mix: dict[str, float] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        key, _, value = part.partition("=")
        key = key.strip()
        if key not in allowed:
            raise GenError(f"неизвестный ключ {key!r} в доле, допустимо {allowed}")
        try:
            mix[key] = float(value)
        except ValueError as exc:
            raise GenError(f"доля {part!r} не число") from exc
        if mix[key] < 0:
            raise GenError(f"отрицательная доля {part!r}")
    if sum(mix.values()) <= 0:
        raise GenError(f"сумма долей {text!r} должна быть > 0")
    return {k: v for k, v in mix.items() if v > 0}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Генератор синтетических ГРЗ РФ (Волга-IT 2026)")
    p.add_argument("--n", type=int, default=5000, help="сколько изображений сгенерировать")
    p.add_argument("--seed", type=int, default=42, help="базовый seed; seed изображения = seed + индекс")
    p.add_argument("--bg-dir", default=str(SCRIPT_DIR / "backgrounds"), help="папка фонов с manifest.csv")
    p.add_argument("--fonts", default=str(SCRIPT_DIR / "fonts"), help="папка .ttf + файл лицензии")
    p.add_argument("--out", default=str(SCRIPT_DIR.parent), help="корень датасета (images/, labels/, meta.csv)")
    p.add_argument("--mix", default=DEFAULT_MIX, help=f"доли типов, по умолчанию {DEFAULT_MIX}")
    p.add_argument("--other-mix", default=DEFAULT_OTHER_MIX, help=f"доли вариантов other, по умолчанию {DEFAULT_OTHER_MIX}")
    p.add_argument("--label-format", choices=("yolo", "yolo_quad"), default="yolo")
    p.add_argument("--start-index", default="",
                   help="первые номера имён, напр. type1=3,type1a=155,type1b=306,other=55; по умолчанию — "
                        "сразу после последнего images/real/<тип>_NNNN")
    p.add_argument("--nonvehicle-frac", type=float, default=NONVEHICLE_FRAC,
                   help="доля примеров на фонах is_vehicle=0 (если такие фоны есть в manifest)")
    p.add_argument("--merge", action="store_true",
                   help="дописать строки в meta.csv (с бэкапом); иначе — в meta_synthetic.csv")
    p.add_argument("--preview", nargs="?", const="synthetic_preview.jpg", default=None,
                   help="сохранить сетку ~24 примеров (по умолчанию synthetic_preview.jpg в текущей папке)")
    args = p.parse_args(argv)
    if args.n <= 0:
        p.error("--n должно быть > 0")
    if not 0.0 <= args.nonvehicle_frac <= 1.0:
        p.error("--nonvehicle-frac в диапазоне 0..1")
    return args


def run(args: argparse.Namespace) -> int:
    t_start = time.time()
    mix = parse_mix(args.mix, PLATE_TYPES)
    other_mix = parse_mix(args.other_mix, OTHER_KINDS)
    fonts = load_fonts(Path(args.fonts))
    bgs, rejected = load_backgrounds(Path(args.bg_dir))
    if not bgs:
        raise GenError(f"нет пригодных фонов. Отброшено: {dict(rejected)}")
    vehicle_pool = [b for b in bgs if b.is_vehicle == 1]
    nonvehicle_pool = [b for b in bgs if b.is_vehicle == 0]
    if args.nonvehicle_frac > 0 and not nonvehicle_pool:
        print(f"ВНИМАНИЕ: в manifest нет фонов с is_vehicle=0 — доля {args.nonvehicle_frac:.0%} "
              "примеров не на ТС не будет сгенерирована")

    out = Path(args.out)
    if not out.is_dir():
        raise GenError(f"папка датасета не найдена: {out}")
    meta_path, synth_meta_path = out / "meta.csv", out / "meta_synthetic.csv"
    meta = read_meta(meta_path) if meta_path.is_file() else None   # проверка заголовка до любых записей
    if meta is not None:
        n_old = sum(1 for line in meta.lines if line and is_generator_row(parse_line(line)))
        if n_old and not args.merge:
            raise GenError(f"в meta.csv уже {n_old} синтетических строк: запустите с --merge, "
                           "чтобы пересоздать их, иначе картинки разойдутся с meta.csv. Ничего не записано.")
    elif args.merge:
        print(f"meta.csv не найден в {out} — создаю только meta_synthetic.csv")
    merge = bool(args.merge and meta is not None)

    images_dir, labels_dir = out / "images" / "synthetic", out / "labels"
    override: dict[str, int] = {}
    for part in filter(None, (s.strip() for s in args.start_index.split(","))):
        key, _, value = part.partition("=")
        if key.strip() not in PLATE_TYPES or not value.strip().isdigit() or int(value) < 1:
            raise GenError(f"--start-index: неверный элемент {part!r} (нужно тип=номер, номер >= 1)")
        override[key.strip()] = int(value)
    start = start_indices(out, override)
    protected = other_image_stems(out, images_dir)
    print("первые номера имён: " + ", ".join(f"{t}_{start[t]:0{NAME_DIGITS}d}" for t in PLATE_TYPES))
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    keep_lines: list[str] = []
    if merge:
        assert meta is not None
        backup = out / f"meta_backup_{datetime.now():%Y%m%d_%H%M%S}.csv"
        shutil.copy2(meta_path, backup)
        print(f"бэкап: {backup}")
        keep_lines = [line for line in meta.lines if not (line and is_generator_row(parse_line(line)))]
        write_meta(meta, keep_lines, [])   # сначала убираем старые синтетические строки
    removed = cleanup_old_outputs(images_dir, labels_dir, protected)
    if removed:
        print(f"удалено старых файлов генератора: {removed}")

    ctx = Context(fonts, vehicle_pool, nonvehicle_pool, mix, other_mix, args.nonvehicle_frac)
    samples: list[Sample] = []
    new_lines: list[str] = []
    next_num = dict(start)
    for index in range(args.n):
        sample, frame = make_sample(index, args.seed, ctx)
        t = sample.plate.plate_type
        sample.name = f"{t}_{next_num[t]:0{NAME_DIGITS}d}"
        next_num[t] += 1
        if sample.name in protected:
            raise GenError(f"имя {sample.name} уже занято реальным фото — задайте --start-index больше")
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise GenError(f"не удалось закодировать {sample.name}")
        (images_dir / f"{sample.name}.jpg").write_bytes(buf.tobytes())
        write_label(labels_dir / f"{sample.name}.txt", sample.plate.plate_type, sample.bbox, sample.quad,
                    sample.width, sample.height, args.label_format)
        new_lines.append(row_to_line(meta_fields(sample, args.seed)))
        samples.append(sample)
        if (index + 1) % 250 == 0 or index + 1 == args.n:
            print(f"  {index + 1}/{args.n}  ({time.time() - t_start:.0f} с)")

    if merge:
        assert meta is not None
        write_meta(meta, keep_lines, new_lines)
        target = meta_path
    else:
        newline = meta.newline if meta is not None else "\n"
        text = META_HEADER + newline + "".join(line + newline for line in new_lines)
        synth_meta_path.write_bytes(text.encode("utf-8"))
        target = synth_meta_path

    # Самопроверка: перечитываем только свои строки из записанного файла.
    written = read_meta(target)
    names = {f"images/synthetic/{s.name}.jpg" for s in samples}
    own = [f for f in (parse_line(line) for line in written.lines if line) if f and f[0] in names]
    problems = self_check(out, own)
    if len(own) != len(samples):
        problems.append(f"в {target.name} найдено {len(own)} своих строк из {len(samples)}")

    if args.preview:
        preview_path = Path(args.preview)
        make_preview(out, own, preview_path)
        print(f"превью: {preview_path.resolve()}")

    by_type = Counter(s.plate.plate_type for s in samples)
    uniq = {t: len({s.plate.plate_num for s in samples if s.plate.plate_type == t}) for t in by_type}
    kinds = Counter(s.plate.kind for s in samples if s.plate.plate_type == "other")
    conds = Counter(c for s in samples for c in s.conditions)
    print("\n== Сводка ==")
    print(f"строк по типам: {dict(by_type)}" + (f"  (other: {dict(kinds)})" if kinds else ""))
    print(f"уникальных plate_num по типам: {uniq}")
    print(f"is_vehicle=0: {sum(1 for s in samples if s.is_vehicle == 0)}")
    print(f"условия: {dict(conds)}")
    print(f"фонов: пригодных {len(bgs)}, использовано {len({s.bg.rel for s in samples})}, "
          f"отброшено {sum(rejected.values())} {dict(rejected)}")
    print(f"перегенерировано (неудачных попыток размещения): {sum(s.attempts - 1 for s in samples)}")
    used = {t: (start[t], next_num[t] - 1) for t in PLATE_TYPES if next_num[t] > start[t]}
    print("имена: " + ", ".join(f"{t}_{a:0{NAME_DIGITS}d}..{t}_{b:0{NAME_DIGITS}d}" for t, (a, b) in used.items()))
    print("для точного повтора: --start-index " + ",".join(f"{t}={start[t]}" for t in PLATE_TYPES))
    print(f"записано строк: {len(new_lines)} -> {target}")
    print(f"время: {time.time() - t_start:.1f} с")
    if problems:
        print(f"\nСАМОПРОВЕРКА: {len(problems)} проблем")
        for pr in problems[:30]:
            print("  " + pr)
        return 1
    print("\nСАМОПРОВЕРКА: ок")
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)
    try:
        return run(args)
    except GenError as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
