from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import RectifyConfig
from .detect import Detection


@dataclass
class RectifyResult:
    image: np.ndarray           
    quad: np.ndarray | None     
    used_fallback: bool
    two_line: bool = False     
    aspect: float = 0.0      


def order_quad(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def crop_with_margin(frame: np.ndarray, det: Detection, margin: float) -> tuple[np.ndarray, int, int]:
    fh, fw = frame.shape[:2]
    mx, my = int(det.width * margin), int(det.height * margin)
    x1, y1 = max(0, det.x1 - mx), max(0, det.y1 - my)
    x2, y2 = min(fw, det.x2 + mx), min(fh, det.y2 + my)
    return frame[y1:y2, x1:x2], x1, y1


def find_plate_quad(crop: np.ndarray, inner_box: tuple[int, int, int, int]) -> np.ndarray | None:
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    bx1, by1, bx2, by2 = inner_box
    box_area = max(1, (bx2 - bx1) * (by2 - by1))
    cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
    best, best_area = None, 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.35 * box_area or area > 1.8 * box_area:
            continue
        if cv2.pointPolygonTest(cnt, (cx, cy), False) < 0:
            continue
        if area > best_area:
            best, best_area = cnt, area
    if best is None:
        return None
    peri = cv2.arcLength(best, True)
    approx = cv2.approxPolyDP(best, 0.03 * peri, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        quad = approx.reshape(4, 2).astype(np.float32)
    else:
        quad = cv2.boxPoints(cv2.minAreaRect(best)).astype(np.float32)
    quad = order_quad(quad)
    w = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    h = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
    if h < 4 or w < 8 or not 0.9 <= w / h <= 7.5:
        return None
    return quad


def warp_quad(frame: np.ndarray, quad: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    dst_w, dst_h = out_size
    dst = np.float32([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]])
    m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(frame, m, (dst_w, dst_h), flags=cv2.INTER_LINEAR)


def rectify(frame: np.ndarray, det: Detection, cfg: RectifyConfig, two_line: bool | None = None) -> RectifyResult:
    crop, ox, oy = crop_with_margin(frame, det, cfg.margin)
    fallback = frame[det.y1:det.y2, det.x1:det.x2]
    if not cfg.enabled or crop.size == 0:
        return RectifyResult(fallback, None, True, bool(two_line), det.aspect)
    inner = (det.x1 - ox, det.y1 - oy, det.x2 - ox, det.y2 - oy)
    quad = find_plate_quad(crop, inner)
    if quad is None:
        return RectifyResult(fallback, None, True, bool(two_line), det.aspect)
    quad_abs = quad + np.float32([ox, oy])
    aspect = quad_aspect(quad)
    if two_line is None:
        two_line = aspect < 2.6
    size = cfg.two_line_size if two_line else cfg.single_line_size
    return RectifyResult(warp_quad(frame, quad_abs, size), quad_abs, False, two_line, aspect)


def quad_aspect(quad: np.ndarray) -> float:
    w = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    h = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
    return float(w / h) if h > 0 else 0.0


def rectify_by_quad(frame: np.ndarray, quad: np.ndarray, two_line: bool, cfg: RectifyConfig) -> np.ndarray:
    size = cfg.two_line_size if two_line else cfg.single_line_size
    return warp_quad(frame, order_quad(quad), size)
