#!/usr/bin/env python3
#
# Portions of this file are derived from deep-anpr
# (https://github.com/matthewearl/deep-anpr, gen.py: additive gaussian noise):
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
# Изменения: цветные эффекты «камеры» (ночь, дождь, снег, грязь, блик, смаз),
# подгонка тона плашки под фон, JPEG-артефакты.
"""Эффекты для синтетических кадров. Все функции принимают и возвращают BGR uint8.

Случайность — только из переданного np.random.Generator (воспроизводимость по seed).
"""

from __future__ import annotations

import cv2
import numpy as np

# Вероятности эффектов; каждый применённый эффект ставит свой тег в conditions.
PROB = {
    "night": 0.20,
    "rain": 0.08,
    "snow": 0.06,          # rain и snow взаимоисключающие
    "dirt": 0.20,
    "glare": 0.18,
    "motion_blur": 0.15,
}
GLARE_PLATE_RADIUS = (0.35, 0.85)    # блик на плашке: радиус в долях ширины знака
GLARE_PLATE_STRENGTH = (0.6, 0.95)
GLARE_LAMP_RADIUS = (0.6, 1.5)       # засветка от фары рядом со знаком
GLARE_LAMP_STRENGTH = (0.7, 1.0)
MOTION_BLUR_REL = (0.05, 0.18)       # длина смаза в долях высоты знака (сильнее — знак не читается)
MOTION_BLUR_MAX_PX = 15
DOWN_UP_PROB = 0.30                  # иногда уменьшить и увеличить обратно (тега нет)
DOWN_UP_FACTOR = (0.45, 0.80)
NOISE_SIGMA = (1.5, 4.5)             # всегда: лёгкий гауссов шум
JPEG_ARTIFACT_QUALITY = (35, 90)     # всегда: промежуточное пережатие JPEG


def _u8(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0, 255).astype(np.uint8)


def dirt(plate: np.ndarray, alpha: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Пятна грязи и пыль на плашке (до перспективы, в координатах шаблона)."""
    h, w = plate.shape[:2]
    spots = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(3, 12))):
        center = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        axes = (max(1, int(rng.uniform(0.02, 0.12) * w)), max(1, int(rng.uniform(0.05, 0.35) * h)))
        cv2.ellipse(spots, center, axes, float(rng.uniform(0, 180)), 0, 360, float(rng.uniform(0.3, 0.8)), -1)
    k = max(3, int(0.02 * w)) | 1
    spots = cv2.GaussianBlur(spots, (k, k), 0)
    dust = rng.random((max(2, h // 40), max(2, w // 40))).astype(np.float32)
    dust = np.clip(cv2.resize(dust, (w, h), interpolation=cv2.INTER_CUBIC), 0, 1) * rng.uniform(0.05, 0.2)
    color = np.array([rng.uniform(30, 70), rng.uniform(50, 90), rng.uniform(70, 110)], np.float32)
    m = (np.clip(spots + dust, 0, 0.85) * alpha)[..., None]
    return _u8(plate.astype(np.float32) * (1 - m) + color * m)


def match_tone(plate: np.ndarray, frame: np.ndarray, box: tuple[float, float, float, float],
               rng: np.random.Generator) -> np.ndarray:
    """Подгонка яркости и оттенка плашки под окружение, чтобы не было заметного шва."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    rx0, ry0 = max(0, int(x0 - bw)), max(0, int(y0 - bh))
    rx1, ry1 = min(W, int(x1 + bw)), min(H, int(y1 + bh))
    region = frame[ry0:ry1, rx0:rx1].reshape(-1, 3).astype(np.float32)
    if region.size == 0:
        return plate
    mean = region.mean(axis=0) + 1.0
    luma = 0.114 * mean[0] + 0.587 * mean[1] + 0.299 * mean[2]
    gain = float(np.clip((luma / 120.0) ** 0.5 * rng.uniform(0.9, 1.1), 0.45, 1.15))
    tint = np.clip(1.0 + 0.35 * (mean / mean.mean() - 1.0), 0.85, 1.15)
    return _u8(plate.astype(np.float32) * gain * tint)


NIGHT_GAMMA = (2.0, 3.0)             # чем больше, тем темнее тени
NIGHT_GAIN = (0.35, 0.6)             # общая яркость ночного кадра
# Номер светоотражающий: ночью в свете фар / ИК-подсветки он остаётся заметно светлее кадра.
NIGHT_PLATE_GAMMA = (1.1, 1.5)
NIGHT_PLATE_GAIN = (0.55, 0.9)


def night(frame: np.ndarray, rng: np.random.Generator, quad: np.ndarray | None = None) -> np.ndarray:
    src = frame.astype(np.float32) / 255.0
    tints = (np.array([0.65, 0.9, 1.1]), np.array([1.1, 0.95, 0.8]), np.array([1.0, 1.0, 1.0]))  # натрий / холод / нейтр.
    tint = tints[int(rng.integers(len(tints)))].astype(np.float32)
    f = np.power(src, rng.uniform(*NIGHT_GAMMA)) * rng.uniform(*NIGHT_GAIN) * tint
    plate_gamma, plate_gain = rng.uniform(*NIGHT_PLATE_GAMMA), rng.uniform(*NIGHT_PLATE_GAIN)
    if quad is not None:
        H, W = frame.shape[:2]
        x0, y0 = np.floor(quad.min(axis=0)).astype(int) - 3
        x1, y1 = np.ceil(quad.max(axis=0)).astype(int) + 4
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 > x0 and y1 > y0:
            mask = np.zeros((y1 - y0, x1 - x0), np.float32)
            cv2.fillConvexPoly(mask, np.round((quad - [x0, y0]) * 16).astype(np.int32), 1.0, cv2.LINE_AA, 4)
            m = cv2.GaussianBlur(mask, (0, 0), 1.0)[..., None]
            lit = np.power(src[y0:y1, x0:x1], plate_gamma) * plate_gain * tint
            f[y0:y1, x0:x1] = lit * m + f[y0:y1, x0:x1] * (1 - m)
    f += rng.normal(0.0, rng.uniform(0.01, 0.035), f.shape).astype(np.float32)
    return _u8(f * 255.0)


def glare(frame: np.ndarray, quad: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Блик на плашке или засветка от фары рядом с ней."""
    H, W = frame.shape[:2]
    x0, y0 = quad.min(axis=0)
    x1, y1 = quad.max(axis=0)
    pw, ph = max(4.0, float(x1 - x0)), max(2.0, float(y1 - y0))
    if rng.random() < 0.6:
        cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
        radius, strength = rng.uniform(*GLARE_PLATE_RADIUS) * pw, rng.uniform(*GLARE_PLATE_STRENGTH)
    else:
        side = -1.0 if rng.random() < 0.5 else 1.0
        cx = (x0 if side < 0 else x1) + side * rng.uniform(0.2, 0.9) * pw
        cy = (y0 + y1) / 2 + rng.uniform(-1.0, 0.5) * ph
        radius, strength = rng.uniform(*GLARE_LAMP_RADIUS) * pw, rng.uniform(*GLARE_LAMP_STRENGTH)
    sigma = max(1.0, radius / 2)
    wx0, wy0 = max(0, int(cx - 3 * sigma)), max(0, int(cy - 3 * sigma))
    wx1, wy1 = min(W, int(cx + 3 * sigma) + 1), min(H, int(cy + 3 * sigma) + 1)
    if wx1 <= wx0 or wy1 <= wy0:
        return frame
    ys, xs = np.mgrid[wy0:wy1, wx0:wx1].astype(np.float32)
    g = (np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma * sigma)) * strength)[..., None]
    white = np.array([235.0, 247.0, 255.0], np.float32)
    out = frame.copy()
    patch = out[wy0:wy1, wx0:wx1].astype(np.float32)
    out[wy0:wy1, wx0:wx1] = _u8(patch + (white - patch) * g)
    return out


def rain(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    H, W = frame.shape[:2]
    n = max(1, int(H * W / 1800 * rng.uniform(0.5, 1.5)))
    angle = np.deg2rad(rng.uniform(70, 110))
    length = rng.uniform(0.012, 0.03) * H * rng.uniform(0.6, 1.4, n)
    xs, ys = rng.uniform(0, W, n), rng.uniform(0, H, n)
    pts = np.stack([xs, ys, xs + length * np.cos(angle), ys + length * np.sin(angle)], axis=1)
    pts = np.round(pts).astype(np.int32).reshape(-1, 2, 2)
    overlay = frame.copy()
    c = int(rng.uniform(180, 230))
    cv2.polylines(overlay, list(pts), False, (c, c, c), 1, cv2.LINE_AA)
    a = rng.uniform(0.2, 0.4)
    out = cv2.addWeighted(overlay, a, frame, 1 - a, 0)
    out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.5, 1.0))
    return cv2.convertScaleAbs(out, alpha=rng.uniform(0.8, 0.95), beta=rng.uniform(0, 15))


def snow(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    H, W = frame.shape[:2]
    flakes = np.zeros((H, W), np.float32)
    rmax = max(2, int(H / 250))
    for _ in range(max(1, int(H * W / 1400 * rng.uniform(0.5, 1.5)))):
        cv2.circle(flakes, (int(rng.uniform(0, W)), int(rng.uniform(0, H))), int(rng.integers(1, rmax + 1)), 1.0, -1)
    flakes = cv2.GaussianBlur(flakes, (0, 0), 0.8)[..., None] * rng.uniform(0.6, 0.95)
    f = frame.astype(np.float32)
    f = f + (245.0 - f) * flakes
    c = rng.uniform(0.7, 0.9)
    f = f * c + f.mean() * (1 - c) + rng.uniform(5, 20)
    return _u8(f)


def motion_blur(frame: np.ndarray, rng: np.random.Generator, plate_h: float) -> np.ndarray:
    size = int(max(3, min(MOTION_BLUR_MAX_PX, plate_h * rng.uniform(*MOTION_BLUR_REL))))
    kernel = np.zeros((size, size), np.float32)
    a = np.deg2rad(rng.uniform(0, 180))
    c = (size - 1) / 2
    dx, dy = np.cos(a) * c, np.sin(a) * c
    cv2.line(kernel, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1)
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(frame, -1, kernel)


def down_up(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    H, W = frame.shape[:2]
    f = rng.uniform(*DOWN_UP_FACTOR)
    small = cv2.resize(frame, (max(1, int(W * f)), max(1, int(H * f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


def sensor_noise(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = rng.uniform(*NOISE_SIGMA)
    return _u8(frame.astype(np.float32) + rng.normal(0.0, sigma, frame.shape).astype(np.float32))


def jpeg_artifacts(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    q = int(rng.integers(JPEG_ARTIFACT_QUALITY[0], JPEG_ARTIFACT_QUALITY[1] + 1))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        return frame
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def apply_frame_effects(frame: np.ndarray, quad: np.ndarray, plate_h: float, rng: np.random.Generator,
                        conditions: set[str]) -> np.ndarray:
    """Эффекты всего кадра после наложения плашки. Дописывает теги в conditions."""
    is_night = rng.random() < PROB["night"]
    if is_night:
        frame = night(frame, rng, quad)
        conditions.add("night")
    if rng.random() < PROB["glare"]:
        frame = glare(frame, quad, rng)
        conditions.add("glare")
    r = rng.random()
    if r < PROB["rain"]:
        frame = rain(frame, rng)
        conditions.add("rain")
    elif r < PROB["rain"] + PROB["snow"]:
        frame = snow(frame, rng)
        conditions.add("snow")
    # Ночь + смаз делают номер нечитаемым при полной разметке — такие сочетания не генерируем.
    if rng.random() < PROB["motion_blur"] and not is_night:
        frame = motion_blur(frame, rng, plate_h)
        conditions.add("motion_blur")
    if rng.random() < DOWN_UP_PROB:
        frame = down_up(frame, rng)
    frame = sensor_noise(frame, rng)
    frame = jpeg_artifacts(frame, rng)
    if not is_night:
        conditions.add("day")
    return frame
