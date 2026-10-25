#!/usr/bin/env python3
#
# Portions of this file are derived from deep-anpr
# (https://github.com/matthewearl/deep-anpr, gen.py and common.py):
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
# Изменения: порт на Python 3, цветные плашки российских ГРЗ (type1 / type1a /
# type1b) и негативные примеры other (мотоцикл, европейский, американский стиль).
"""Генерация текста номера и рисование плашек в высоком разрешении (Pillow + OpenCV)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================================
# Размеры и геометрия плашек, мм. Это правдоподобное приближение, а не точная
# геометрия: сверить с ГОСТ Р 50577-2018 и при необходимости поправить здесь.
# Доли (0..1) — от ширины или высоты плашки.
# ============================================================================
PX_PER_MM = 8  # разрешение рендера; при наложении плашка уменьшается

SIZE_MM: dict[str, tuple[float, float]] = {
    "type1": (520.0, 112.0),
    "type1a": (290.0, 170.0),
    "type1b": (520.0, 112.0),
    "moto": (245.0, 160.0),   # other: мотоциклетный двухстрочный
    "eu": (520.0, 110.0),     # other: европейский стиль
    "us": (305.0, 152.0),     # other: американский стиль
}
CORNER_RADIUS_MM = 5.0
FRAME_INSET_MM = 2.5      # отступ чёрной рамки от края плашки
FRAME_WIDTH_MM = 2.0

# type1 / type1b (одна строка): основной номер слева, поле региона справа
DIGIT_H_MM = 76.0         # высота основных цифр
LETTER_SCALE = 0.87       # буквы немного меньше цифр (0.85–0.9)
CHAR_GAP_MM = 6.0
GROUP_GAP_MM = 14.0
MAIN_ZONE = (0.035, 0.735)
MAIN_BASELINE = 0.84
SEPARATOR_X = 0.748
REGION_ZONE = (0.765, 0.975)
REGION_DIGIT_H_MM = 54.0
REGION_BASELINE = 0.56
RUS_H_MM = 13.0
RUS_BASELINE = 0.90
FLAG_MM = (24.0, 14.0)
RUS_FLAG_GAP_MM = 4.0

# type1a (две строки): сверху буква + 3 цифры, снизу 2 буквы и рамка с регионом
T1A_DIGIT_H_MM = 62.0
T1A_TOP_ZONE = (0.06, 0.94)
T1A_TOP_BASELINE = 0.47
T1A_LETTERS_ZONE = (0.05, 0.50)
T1A_BOTTOM_BASELINE = 0.90
T1A_REGION_BOX = (0.53, 0.54, 0.955, 0.94)   # x0, y0, x1, y1
T1A_REGION_DIGIT_H_MM = 38.0
T1A_REGION_BASELINE = 0.62                    # доля высоты рамки региона
T1A_RUS_H_MM = 10.0
T1A_RUS_BASELINE = 0.90
T1A_FLAG_MM = (16.0, 10.0)

# other: мотоцикл (4 цифры сверху, 2 буквы и регион снизу)
MOTO_DIGIT_H_MM = 58.0
MOTO_TOP_ZONE = (0.07, 0.93)
MOTO_TOP_BASELINE = 0.45
MOTO_LETTER_H_MM = 50.0
MOTO_LETTERS_ZONE = (0.06, 0.47)
MOTO_BOTTOM_BASELINE = 0.90
MOTO_REGION_ZONE = (0.52, 0.94)
MOTO_REGION_DIGIT_H_MM = 42.0
MOTO_REGION_BASELINE = 0.80
MOTO_RUS_H_MM = 9.0
MOTO_RUS_BASELINE = 0.955
MOTO_FLAG_MM = (14.0, 8.5)

# other: европейский стиль (синяя полоса слева)
EU_STRIP_MM = 44.0
EU_TEXT_H_MM = 74.0
EU_TEXT_ZONE = (0.12, 0.97)
EU_BASELINE = 0.84
EU_CODE_H_MM = 17.0
EU_CODE_BASELINE = 0.90
EU_STARS_Y = 0.36
EU_STARS_RADIUS_MM = 12.0
EU_STAR_MM = 2.0
EU_COUNTRY_CODES = ("D", "F", "PL", "LT", "LV", "EST", "FIN", "CZ", "SK", "BG", "I", "NL", "B")

# other: американский стиль (цветная полоса сверху)
US_HEADER_H = 0.24
US_HEADER_TEXT_H_MM = 20.0
US_HEADER_WORDS = ("USA",)
US_TEXT_H_MM = 70.0
US_TEXT_ZONE = (0.06, 0.94)
US_BASELINE = 0.84

# Цвета (RGB)
EU_BLUE = (0, 51, 153)
EU_STAR = (255, 204, 0)
FLAG_COLORS = ((245, 245, 245), (0, 57, 166), (213, 43, 30))
US_COLORS = ((24, 58, 140), (150, 28, 36), (30, 100, 50), (25, 25, 25))

# ============================================================================
# Алфавит и маски номера
# ============================================================================
LETTERS = "ABEKMHOPCTYX"   # совпадают по начертанию в кириллице и латинице
DIGITS = "0123456789"
LATIN = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
REQUIRED_GLYPHS = LATIN + DIGITS   # включает RUS и латиницу для иностранных other

TYPE1_RE = re.compile(r"^[ABEKMHOPCTYX][0-9]{3}[ABEKMHOPCTYX]{2}[0-9]{2,3}$")
TYPE1B_RE = re.compile(r"^[ABEKMHOPCTYX]{2}[0-9]{3}[0-9]{2,3}$")
OTHER_RE = re.compile(r"^[ABEKMHOPCTYX0-9#]{1,12}$")

# Реальные коды регионов выбираются чаще, но любые другие 2–3-значные тоже допустимы.
# Первая цифра трёхзначного кода не ограничивается.
REAL_REGION_CODES = tuple(f"{i:02d}" for i in range(1, 100) if i not in (80, 81, 84, 85, 88)) + (
    "102", "113", "116", "121", "122", "123", "124", "125", "126", "134", "136", "138", "142",
    "147", "150", "152", "154", "159", "161", "163", "164", "173", "174", "177", "178", "186",
    "190", "193", "196", "197", "199", "250", "252", "254", "277", "299", "323", "550", "702",
    "716", "725", "750", "761", "763", "774", "777", "790", "797", "799", "977",
)
REAL_REGION_PROB = 0.85


class FontError(RuntimeError):
    pass


@dataclass
class Plate:
    image: np.ndarray          # BGR uint8, высокое разрешение
    alpha: np.ndarray          # float32 0..1, маска формы плашки
    plate_num: str
    plate_type: str            # type1 / type1a / type1b / other
    kind: str                  # type1 / type1a / type1b / moto / eu / us
    size_mm: tuple[float, float]


def mm(value: float) -> int:
    return int(round(value * PX_PER_MM))


# ============================================================================
# Шрифты: каждый символ рисуется Pillow, обрезается по чернилам и масштабируется
# до нужной высоты (идея make_char_ims из deep-anpr).
# ============================================================================
class FontGlyphs:
    RENDER_SIZE = 900

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.name = self.path.name
        self.font = ImageFont.truetype(str(self.path), self.RENDER_SIZE)
        self._base: dict[str, np.ndarray | None] = {}
        self._cache: dict[tuple[str, int], np.ndarray] = {}

    def _render(self, ch: str) -> np.ndarray | None:
        if ch not in self._base:
            left, top, right, bottom = self.font.getbbox(ch)
            w, h = right - left + 8, bottom - top + 8
            if w <= 8 or h <= 8:
                self._base[ch] = None
            else:
                img = Image.new("L", (w, h), 0)
                ImageDraw.Draw(img).text((4 - left, 4 - top), ch, fill=255, font=self.font)
                bb = img.getbbox()
                self._base[ch] = None if bb is None else np.asarray(img.crop(bb), dtype=np.uint8)
        return self._base[ch]

    def check(self) -> None:
        try:
            axes = self.font.get_variation_axes()
        except (OSError, AttributeError, NotImplementedError):
            axes = None
        if axes:
            raise FontError(f"{self.name}: variable-шрифт, нужен статичный TTF")
        notdef = self._render("")
        missing = []
        for ch in REQUIRED_GLYPHS:
            m = self._render(ch)
            if m is None or (notdef is not None and m.shape == notdef.shape and np.array_equal(m, notdef)):
                missing.append(ch)
        if missing:
            raise FontError(f"{self.name}: нет глифов {''.join(missing)}")

    def glyph(self, ch: str, height_px: int) -> np.ndarray:
        key = (ch, height_px)
        if key not in self._cache:
            base = self._render(ch)
            if base is None:
                raise FontError(f"{self.name}: нет глифа {ch!r}")
            h, w = base.shape
            new_w = max(1, int(round(w * height_px / h)))
            interp = cv2.INTER_AREA if height_px < h else cv2.INTER_CUBIC
            g = cv2.resize(base, (new_w, max(1, height_px)), interpolation=interp)
            self._cache[key] = np.clip(g.astype(np.float32) / 255.0, 0.0, 1.0)
        return self._cache[key]


# ============================================================================
# Генерация текста
# ============================================================================
def _pick(rng: np.random.Generator, chars: str) -> str:
    return chars[int(rng.integers(len(chars)))]


def gen_region(rng: np.random.Generator) -> str:
    if rng.random() < REAL_REGION_PROB:
        return REAL_REGION_CODES[int(rng.integers(len(REAL_REGION_CODES)))]
    while True:
        n = 2 if rng.random() < 0.6 else 3
        code = "".join(_pick(rng, DIGITS) for _ in range(n))
        if set(code) != {"0"}:
            return code


def gen_digits(rng: np.random.Generator, n: int) -> str:
    while True:
        s = "".join(_pick(rng, DIGITS) for _ in range(n))
        if set(s) != {"0"}:
            return s


def gen_type1_num(rng: np.random.Generator) -> str:
    return _pick(rng, LETTERS) + gen_digits(rng, 3) + _pick(rng, LETTERS) + _pick(rng, LETTERS) + gen_region(rng)


def gen_type1b_num(rng: np.random.Generator) -> str:
    return _pick(rng, LETTERS) + _pick(rng, LETTERS) + gen_digits(rng, 3) + gen_region(rng)


def mask_other(text: str) -> str:
    """Текст other-плашки -> plate_num: символы вне допустимого набора заменяются на #.

    TODO: уточнить у организаторов, как правильно заполнять plate_num для other.
    """
    return "".join(c if c in LETTERS + DIGITS else "#" for c in text.upper() if c not in " -")


def _looks_target(plate_num: str) -> bool:
    return bool(TYPE1_RE.match(plate_num) or TYPE1B_RE.match(plate_num))


def gen_moto(rng: np.random.Generator) -> tuple[str, list[str]]:
    # Формат мотоциклетного знака РФ: 4 цифры сверху, 2 буквы и регион снизу.
    digits, letters, region = gen_digits(rng, 4), _pick(rng, LETTERS) + _pick(rng, LETTERS), gen_region(rng)
    return digits + letters + region, [digits, letters, region]


def gen_eu(rng: np.random.Generator) -> tuple[str, list[str], str]:
    patterns = ("LL DDDD", "L LL DDDD", "LLL DDD", "LL DDD LL", "D LLL DDD", "LL DDDD L")
    while True:
        pattern = patterns[int(rng.integers(len(patterns)))]
        groups = ["".join(_pick(rng, LATIN) if c == "L" else _pick(rng, DIGITS) for c in g)
                  for g in pattern.split()]
        plate_num = mask_other("".join(groups))
        if not _looks_target(plate_num):
            code = EU_COUNTRY_CODES[int(rng.integers(len(EU_COUNTRY_CODES)))]
            return plate_num, groups, code


def gen_us(rng: np.random.Generator) -> tuple[str, list[str]]:
    patterns = ("DLLLDDD", "LLL DDDD", "LLL DDD", "DDD LLL", "LLLDDDD")
    while True:
        pattern = patterns[int(rng.integers(len(patterns)))]
        groups = ["".join(_pick(rng, LATIN) if c == "L" else _pick(rng, DIGITS) for c in g)
                  for g in pattern.split()]
        plate_num = mask_other("".join(groups))
        if not _looks_target(plate_num):
            return plate_num, groups


# ============================================================================
# Рисование
# ============================================================================
def rounded_rect(shape: tuple[int, int], radius: int) -> np.ndarray:
    """Маска плашки со скруглёнными углами (из deep-anpr gen.py)."""
    out = np.ones(shape, dtype=np.float32)
    out[:radius, :radius] = 0.0
    out[-radius:, :radius] = 0.0
    out[:radius, -radius:] = 0.0
    out[-radius:, -radius:] = 0.0
    cv2.circle(out, (radius, radius), radius, 1.0, -1)
    cv2.circle(out, (radius, shape[0] - radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, shape[0] - radius), radius, 1.0, -1)
    return out


def _white(rng: np.random.Generator) -> tuple[int, int, int]:
    v = rng.uniform(226, 250)
    return tuple(int(np.clip(v + rng.uniform(-4, 4), 0, 255)) for _ in range(3))


def _yellow(rng: np.random.Generator) -> tuple[int, int, int]:
    return (int(np.clip(248 + rng.uniform(-10, 6), 0, 255)),
            int(np.clip(198 + rng.uniform(-18, 14), 0, 255)),
            int(np.clip(28 + rng.uniform(-20, 25), 0, 255)))


def _ink(rng: np.random.Generator) -> tuple[int, int, int]:
    v = int(rng.uniform(12, 45))
    return (v, v, v)


def _paste_max(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    h, w = src.shape
    H, W = dst.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    dst[y0:y1, x0:x1] = np.maximum(dst[y0:y1, x0:x1], src[y0 - y:y1 - y, x0 - x:x1 - x])


def _seq(font: FontGlyphs, groups: list[str], digit_h: int, letter_h: int,
         char_gap: float, group_gap: float) -> tuple[list[np.ndarray], list[float]]:
    glyphs: list[np.ndarray] = []
    gaps: list[float] = []
    for gi, group in enumerate(groups):
        for ci, ch in enumerate(group):
            if glyphs:
                gaps.append(group_gap if ci == 0 and gi > 0 else char_gap)
            glyphs.append(font.glyph(ch, digit_h if ch in DIGITS else letter_h))
    return glyphs, gaps


def _row(mask: np.ndarray, glyphs: list[np.ndarray], gaps: list[float],
         x0: float, x1: float, baseline: float) -> list[tuple[int, int]]:
    """Ставит глифы в строку по базовой линии, по центру зоны. Если не влезают — сжимает по ширине."""
    zone = x1 - x0
    total = sum(g.shape[1] for g in glyphs) + sum(gaps)
    if total > zone and total > 0:
        s = zone / total
        glyphs = [cv2.resize(g, (max(1, int(round(g.shape[1] * s))), g.shape[0]), interpolation=cv2.INTER_AREA)
                  for g in glyphs]
        gaps = [gp * s for gp in gaps]
        total = sum(g.shape[1] for g in glyphs) + sum(gaps)
    x = x0 + (zone - total) / 2
    boxes = []
    for i, g in enumerate(glyphs):
        ix, iy = int(round(x)), int(round(baseline - g.shape[0]))
        _paste_max(mask, g, ix, iy)
        boxes.append((ix, g.shape[1]))
        x += g.shape[1] + (gaps[i] if i < len(gaps) else 0)
    return boxes


def _frame(draw: ImageDraw.ImageDraw, W: int, H: int, color: tuple[int, int, int]) -> None:
    inset = mm(FRAME_INSET_MM)
    draw.rounded_rectangle([inset, inset, W - 1 - inset, H - 1 - inset],
                           radius=max(1, mm(CORNER_RADIUS_MM) - inset), outline=color, width=mm(FRAME_WIDTH_MM))


def _flag(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, outline: tuple[int, int, int]) -> None:
    stripe = h / 3
    for i, color in enumerate(FLAG_COLORS):
        draw.rectangle([x, int(y + i * stripe), x + w, int(y + (i + 1) * stripe)], fill=color)
    draw.rectangle([x, y, x + w, y + h], outline=outline, width=max(1, mm(0.4)))


def _rus_flag(draw: ImageDraw.ImageDraw, mask: np.ndarray, font: FontGlyphs, ink: tuple[int, int, int],
              x0: float, x1: float, baseline: float, rus_h: int, flag_wh: tuple[int, int]) -> None:
    glyphs, gaps = _seq(font, ["RUS"], rus_h, rus_h, rus_h * 0.12, 0)
    flag_w, flag_h = flag_wh
    glyphs.append(np.zeros((flag_h, flag_w), np.float32))   # место под флаг
    gaps.append(mm(RUS_FLAG_GAP_MM))
    boxes = _row(mask, glyphs, gaps, x0, x1, baseline)
    fx, fw = boxes[-1]
    _flag(draw, fx, int(round(baseline - flag_h)), fw, flag_h, ink)


def _finish(img: Image.Image, inks: list[tuple[np.ndarray, tuple[int, int, int]]], plate_num: str,
            plate_type: str, kind: str) -> Plate:
    arr = np.asarray(img, dtype=np.float32).copy()
    for mask, color in inks:
        m = mask[..., None]
        arr = arr * (1.0 - m) + m * np.asarray(color, dtype=np.float32)
    bgr = cv2.cvtColor(np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    alpha = rounded_rect(bgr.shape[:2], mm(CORNER_RADIUS_MM))
    return Plate(bgr, alpha, plate_num, plate_type, kind, SIZE_MM[kind])


def _canvas(kind: str, bg: tuple[int, int, int]) -> tuple[Image.Image, ImageDraw.ImageDraw, int, int]:
    w_mm, h_mm = SIZE_MM[kind]
    W, H = mm(w_mm), mm(h_mm)
    img = Image.new("RGB", (W, H), bg)
    return img, ImageDraw.Draw(img), W, H


def render_single_line(plate_num: str, plate_type: str, font: FontGlyphs, rng: np.random.Generator) -> Plate:
    """type1 (белый) или type1b (жёлтый): 520x112 мм, поле региона с RUS и флагом справа."""
    yellow = plate_type == "type1b"
    ink = _ink(rng)
    img, draw, W, H = _canvas(plate_type, _yellow(rng) if yellow else _white(rng))
    _frame(draw, W, H, ink)
    sx, fi = int(SEPARATOR_X * W), mm(FRAME_INSET_MM)
    draw.line([(sx, fi), (sx, H - 1 - fi)], fill=ink, width=mm(FRAME_WIDTH_MM))
    mask = np.zeros((H, W), np.float32)

    body_len = 5 if yellow else 6
    body, region = plate_num[:body_len], plate_num[body_len:]
    groups = [body[:2], body[2:]] if yellow else [body[0], body[1:4], body[4:]]
    dh = mm(DIGIT_H_MM)
    glyphs, gaps = _seq(font, groups, dh, int(round(dh * LETTER_SCALE)), mm(CHAR_GAP_MM), mm(GROUP_GAP_MM))
    _row(mask, glyphs, gaps, MAIN_ZONE[0] * W, MAIN_ZONE[1] * W, MAIN_BASELINE * H)

    rh = mm(REGION_DIGIT_H_MM)
    glyphs, gaps = _seq(font, [region], rh, rh, mm(CHAR_GAP_MM * 0.6), 0)
    _row(mask, glyphs, gaps, REGION_ZONE[0] * W, REGION_ZONE[1] * W, REGION_BASELINE * H)
    _rus_flag(draw, mask, font, ink, REGION_ZONE[0] * W, REGION_ZONE[1] * W, RUS_BASELINE * H,
              mm(RUS_H_MM), (mm(FLAG_MM[0]), mm(FLAG_MM[1])))
    return _finish(img, [(mask, ink)], plate_num, plate_type, plate_type)


def render_type1a(plate_num: str, font: FontGlyphs, rng: np.random.Generator) -> Plate:
    """type1a: 290x170 мм, сверху буква + 3 цифры, снизу 2 буквы серии и регион с RUS и флагом."""
    ink = _ink(rng)
    img, draw, W, H = _canvas("type1a", _white(rng))
    _frame(draw, W, H, ink)
    mask = np.zeros((H, W), np.float32)
    dh = mm(T1A_DIGIT_H_MM)
    lh = int(round(dh * LETTER_SCALE))

    glyphs, gaps = _seq(font, [plate_num[0], plate_num[1:4]], dh, lh, mm(CHAR_GAP_MM * 0.8), mm(GROUP_GAP_MM * 0.8))
    _row(mask, glyphs, gaps, T1A_TOP_ZONE[0] * W, T1A_TOP_ZONE[1] * W, T1A_TOP_BASELINE * H)
    glyphs, gaps = _seq(font, [plate_num[4:6]], dh, lh, mm(CHAR_GAP_MM * 0.8), 0)
    _row(mask, glyphs, gaps, T1A_LETTERS_ZONE[0] * W, T1A_LETTERS_ZONE[1] * W, T1A_BOTTOM_BASELINE * H)

    bx0, by0, bx1, by1 = (T1A_REGION_BOX[0] * W, T1A_REGION_BOX[1] * H, T1A_REGION_BOX[2] * W, T1A_REGION_BOX[3] * H)
    draw.rounded_rectangle([int(bx0), int(by0), int(bx1), int(by1)], radius=mm(3), outline=ink, width=mm(1.6))
    pad = mm(3)
    rh = mm(T1A_REGION_DIGIT_H_MM)
    glyphs, gaps = _seq(font, [plate_num[6:]], rh, rh, mm(CHAR_GAP_MM * 0.5), 0)
    _row(mask, glyphs, gaps, bx0 + pad, bx1 - pad, by0 + (by1 - by0) * T1A_REGION_BASELINE)
    _rus_flag(draw, mask, font, ink, bx0 + pad, bx1 - pad, by0 + (by1 - by0) * T1A_RUS_BASELINE,
              mm(T1A_RUS_H_MM), (mm(T1A_FLAG_MM[0]), mm(T1A_FLAG_MM[1])))
    return _finish(img, [(mask, ink)], plate_num, "type1a", "type1a")


def render_moto(plate_num: str, groups: list[str], font: FontGlyphs, rng: np.random.Generator) -> Plate:
    ink = _ink(rng)
    img, draw, W, H = _canvas("moto", _white(rng))
    _frame(draw, W, H, ink)
    mask = np.zeros((H, W), np.float32)
    digits, letters, region = groups
    dh = mm(MOTO_DIGIT_H_MM)
    glyphs, gaps = _seq(font, [digits], dh, dh, mm(CHAR_GAP_MM * 0.8), 0)
    _row(mask, glyphs, gaps, MOTO_TOP_ZONE[0] * W, MOTO_TOP_ZONE[1] * W, MOTO_TOP_BASELINE * H)
    lh = mm(MOTO_LETTER_H_MM)
    glyphs, gaps = _seq(font, [letters], lh, lh, mm(CHAR_GAP_MM * 0.8), 0)
    _row(mask, glyphs, gaps, MOTO_LETTERS_ZONE[0] * W, MOTO_LETTERS_ZONE[1] * W, MOTO_BOTTOM_BASELINE * H)
    rh = mm(MOTO_REGION_DIGIT_H_MM)
    glyphs, gaps = _seq(font, [region], rh, rh, mm(CHAR_GAP_MM * 0.5), 0)
    _row(mask, glyphs, gaps, MOTO_REGION_ZONE[0] * W, MOTO_REGION_ZONE[1] * W, MOTO_REGION_BASELINE * H)
    _rus_flag(draw, mask, font, ink, MOTO_REGION_ZONE[0] * W, MOTO_REGION_ZONE[1] * W, MOTO_RUS_BASELINE * H,
              mm(MOTO_RUS_H_MM), (mm(MOTO_FLAG_MM[0]), mm(MOTO_FLAG_MM[1])))
    return _finish(img, [(mask, ink)], plate_num, "other", "moto")


def render_eu(plate_num: str, groups: list[str], code: str, font: FontGlyphs, rng: np.random.Generator) -> Plate:
    ink = _ink(rng)
    img, draw, W, H = _canvas("eu", _white(rng))
    fi, strip = mm(FRAME_INSET_MM), mm(EU_STRIP_MM)
    draw.rectangle([0, 0, fi + strip, H - 1], fill=EU_BLUE)
    cx, cy, r, sr = (fi + strip) / 2, EU_STARS_Y * H, mm(EU_STARS_RADIUS_MM), mm(EU_STAR_MM)
    for k in range(12):
        a = 2 * math.pi * k / 12
        x, y = cx + r * math.cos(a), cy + r * math.sin(a)
        draw.ellipse([x - sr, y - sr, x + sr, y + sr], fill=EU_STAR)
    _frame(draw, W, H, ink)
    text_mask = np.zeros((H, W), np.float32)
    code_mask = np.zeros((H, W), np.float32)
    ch = mm(EU_CODE_H_MM)
    glyphs, gaps = _seq(font, [code], ch, ch, ch * 0.1, 0)
    _row(code_mask, glyphs, gaps, fi + mm(2), fi + strip - mm(2), EU_CODE_BASELINE * H)
    th = mm(EU_TEXT_H_MM)
    glyphs, gaps = _seq(font, groups, th, th, mm(CHAR_GAP_MM), mm(GROUP_GAP_MM * 1.6))
    _row(text_mask, glyphs, gaps, EU_TEXT_ZONE[0] * W, EU_TEXT_ZONE[1] * W, EU_BASELINE * H)
    return _finish(img, [(text_mask, ink), (code_mask, (245, 245, 245))], plate_num, "other", "eu")


def render_us(plate_num: str, groups: list[str], font: FontGlyphs, rng: np.random.Generator) -> Plate:
    header_color = US_COLORS[int(rng.integers(len(US_COLORS)))]
    text_color = US_COLORS[int(rng.integers(len(US_COLORS)))]
    img, draw, W, H = _canvas("us", _white(rng))
    fi = mm(FRAME_INSET_MM)
    draw.rectangle([0, 0, W - 1, int(US_HEADER_H * H)], fill=header_color)
    _frame(draw, W, H, (40, 40, 40))
    text_mask = np.zeros((H, W), np.float32)
    header_mask = np.zeros((H, W), np.float32)
    hh = mm(US_HEADER_TEXT_H_MM)
    word = US_HEADER_WORDS[int(rng.integers(len(US_HEADER_WORDS)))]
    glyphs, gaps = _seq(font, [word], hh, hh, hh * 0.15, 0)
    _row(header_mask, glyphs, gaps, fi, W - fi, (US_HEADER_H * H + hh) / 2)
    th = mm(US_TEXT_H_MM)
    glyphs, gaps = _seq(font, groups, th, th, mm(CHAR_GAP_MM * 0.8), mm(GROUP_GAP_MM * 1.4))
    _row(text_mask, glyphs, gaps, US_TEXT_ZONE[0] * W, US_TEXT_ZONE[1] * W, US_BASELINE * H)
    return _finish(img, [(text_mask, text_color), (header_mask, (245, 245, 245))], plate_num, "other", "us")


def make_plate(kind: str, font: FontGlyphs, rng: np.random.Generator) -> Plate:
    """kind: type1 / type1a / type1b / moto / eu / us."""
    if kind in ("type1", "type1b"):
        num = gen_type1_num(rng) if kind == "type1" else gen_type1b_num(rng)
        return render_single_line(num, kind, font, rng)
    if kind == "type1a":
        return render_type1a(gen_type1_num(rng), font, rng)
    if kind == "moto":
        num, groups = gen_moto(rng)
        return render_moto(num, groups, font, rng)
    if kind == "eu":
        num, groups, code = gen_eu(rng)
        return render_eu(num, groups, code, font, rng)
    if kind == "us":
        num, groups = gen_us(rng)
        return render_us(num, groups, font, rng)
    raise ValueError(f"неизвестный вид плашки: {kind}")
