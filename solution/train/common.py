from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SOLUTION_ROOT = Path(__file__).resolve().parent.parent
if str(SOLUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLUTION_ROOT))


@dataclass
class PlateRecord:
    image: Path
    plate_num: str
    plate_type: str
    bbox: tuple[int, int, int, int]     
    quad: np.ndarray                    
    is_vehicle: bool
    is_synthetic: bool


def read_meta(dataset_root: Path) -> list[PlateRecord]:
    meta = dataset_root / "meta.csv"
    if not meta.is_file():
        raise FileNotFoundError(meta)
    out: list[PlateRecord] = []
    with open(meta, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            try:
                x, y, w, h = (int(float(v)) for v in row["bbox"].split(","))
                q = np.array([float(v) for v in row["quad"].split(",")], dtype=np.float32).reshape(4, 2)
            except (ValueError, KeyError):
                continue
            out.append(PlateRecord(
                image=dataset_root / row["image"].strip().replace("\\", "/"),
                plate_num=row["plate_num"].strip().upper(),
                plate_type=row["plate_type"].strip(),
                bbox=(x, y, w, h), quad=q,
                is_vehicle=row.get("is_vehicle", "1").strip() != "0",
                is_synthetic=row.get("is_synthetic", "0").strip() == "1",
            ))
    return out


def group_split(records: list[PlateRecord], val_fraction: float, seed: int) -> tuple[set[Path], set[Path]]:
    real = [r for r in records if not r.is_synthetic]
    if not real:
        raise ValueError("в датасете нет реальных фото — валидацию не на чем строить")
    all_images = {r.image for r in records}
    parent: dict[str, str] = {}

    def find(a: str) -> str:
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    for r in real:
        key_img = f"img:{r.image}"
        find(key_img)
        if r.plate_num and "#" not in r.plate_num:
            union(key_img, f"plate:{r.plate_num}")
    groups: dict[str, list[Path]] = defaultdict(list)
    seen: set[Path] = set()
    for r in real:
        if r.image in seen:
            continue
        seen.add(r.image)
        groups[find(f"img:{r.image}")].append(r.image)
    keys = sorted(groups, key=lambda k: str(groups[k][0]))
    random.Random(seed).shuffle(keys)
    total = len(seen)
    val: set[Path] = set()
    for k in keys:
        if len(val) >= total * val_fraction:
            break
        val.update(groups[k])
    train = all_images - val
    return train, val


def imread(path: Path) -> np.ndarray | None:
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
