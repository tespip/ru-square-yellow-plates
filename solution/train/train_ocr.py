from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import yaml

from common import SOLUTION_ROOT, PlateRecord, group_split, imread, read_meta

from src.config import ALLOWED_LETTERS, DIGITS, RectifyConfig, TwoLineConfig, load_config
from src.ocr import two_line_to_single
from src.detect import Detection
from src.rectify import rectify, rectify_by_quad

BASE_URL = "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates"
BASE_FILES = ("cct_xs_v2_global.keras", "cct_xs_v2_global_model_config.yaml", "cct_xs_v2_global_plate_config.yaml")
GOST_CHARS = DIGITS + ALLOWED_LETTERS
REGION = "Unknown"
PATIENCE = 25  

def download_base_model(base_dir: Path) -> tuple[Path, Path, Path]:
    base_dir.mkdir(parents=True, exist_ok=True)
    for name in BASE_FILES:
        path = base_dir / name
        if not path.is_file():
            print(f"скачиваю {name} ...")
            tmp = path.with_name(name + ".part")
            urllib.request.urlretrieve(f"{BASE_URL}/{name}", tmp)
            tmp.replace(path)
    return tuple(base_dir / name for name in BASE_FILES)  


def jitter_bbox(bbox, frame_shape, rng: np.random.Generator, amount: float = 0.06) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    fh, fw = frame_shape[:2]
    dx = rng.uniform(-amount, amount, 2) * w
    dy = rng.uniform(-amount, amount, 2) * h
    x1, x2 = int(max(0, x + dx[0])), int(min(fw, x + w + dx[1]))
    y1, y2 = int(max(0, y + dy[0])), int(min(fh, y + h + dy[1]))
    return x1, y1, max(x1 + 4, x2), max(y1 + 4, y2)


def make_crops(records: list[PlateRecord], crops_dir: Path, rectify_cfg: RectifyConfig, two_line_cfg: TwoLineConfig,
               variants_real: int = 0, variants_synth: int = 0,
               seed: int = 42, exact_quad: bool = True) -> list[tuple[Path, str]]:
    crops_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    out: list[tuple[Path, str]] = []
    last_path, last_frame = None, None
    for i, r in enumerate(records):
        if r.plate_type not in ("type1", "type1a", "type1b"):
            continue
        if not r.plate_num or len(r.plate_num) > 9 or any(c not in GOST_CHARS for c in r.plate_num):
            continue   
        if r.image != last_path:  
            last_path, last_frame = r.image, imread(r.image)
        if last_frame is None:
            continue
        two_line = r.plate_type == "type1a"
        plates = []
        if exact_quad:
            plates.append(rectify_by_quad(last_frame, r.quad, two_line, rectify_cfg))
        n_var = variants_synth if r.is_synthetic else variants_real
        for k in range(n_var):
            bx, by, bw, bh = r.bbox
            box = (bx, by, bx + bw, by + bh) if (not exact_quad and k == 0) else jitter_bbox(r.bbox, last_frame.shape, rng)
            det = Detection(r.plate_type, 1.0, *box)
            plates.append(rectify(last_frame, det, rectify_cfg, two_line).image)
        for k, plate in enumerate(plates):
            if plate is None or plate.size == 0:
                continue
            if two_line:
                plate = two_line_to_single(plate, two_line_cfg)
            path = crops_dir / f"{i:06d}_{k}_{r.plate_num}.png"
            cv2.imwrite(str(path), plate)
            out.append((path, r.plate_num))
    return out


def write_csv(path: Path, items: list[tuple[Path, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "plate_text", "plate_region"])
        for crop, text in items:
            w.writerow([crop.relative_to(path.parent).as_posix(), text, REGION])


def fast_plate_ocr(*args: str) -> None:
    cmd = [sys.executable, "-c", "from fast_plate_ocr.cli.cli import main_cli; main_cli()", *args]
    print(">> fast-plate-ocr", " ".join(args))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="дообучение OCR-модели FastALPR ")
    p.add_argument("--dataset", required=True, nargs="+")
    p.add_argument("--out", default="runs/ocr")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variants", type=int, default=4)
    p.add_argument("--config", default=str(SOLUTION_ROOT / "config.yaml"))
    p.add_argument("--weights-dir", default=str(SOLUTION_ROOT / "weights"))
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base_weights, base_model_cfg, base_plate_cfg = download_base_model(out / "base")
    model_cfg, plate_cfg = out / "model_config.yaml", out / "plate_config.yaml"
    shutil.copy2(base_model_cfg, model_cfg)
    shutil.copy2(base_plate_cfg, plate_cfg)
    regions = yaml.safe_load(plate_cfg.read_text(encoding="utf-8")).get("plate_regions") or []
    if REGION not in regions:
        raise SystemExit(f"в конфиге модели нет региона {REGION!r}")

    records: list[PlateRecord] = []
    for d in args.dataset:
        records += read_meta(Path(d))
    train_imgs, val_imgs = group_split(records, args.val_fraction, args.seed)
    cfg = load_config(args.config)
    print(f"склейка 1А: {cfg.two_line}")
    train_items = make_crops([r for r in records if r.image in train_imgs], out / "crops" / "train", cfg.rectify, cfg.two_line,
                             variants_real=args.variants, variants_synth=1, seed=args.seed)
    val_items = make_crops([r for r in records if r.image in val_imgs], out / "crops" / "val", cfg.rectify, cfg.two_line,
                           variants_real=1, variants_synth=1, seed=args.seed, exact_quad=False)
    write_csv(out / "train.csv", train_items)
    write_csv(out / "val.csv", val_items)
    print(f"кропов: train {len(train_items)}, val {len(val_items)} (val — только реальные фото)")
    if args.skip_train:
        return

    fast_plate_ocr(
        "train", "--model-config-file", str(model_cfg), "--plate-config-file", str(plate_cfg),
        "--annotations", str(out / "train.csv"), "--val-annotations", str(out / "val.csv"),
        "--weights-path", str(base_weights),
        "--epochs", str(args.epochs), "--lr", str(args.lr), "--batch-size", str(args.batch_size),
        "--early-stopping-metric", "val_plate_acc", "--early-stopping-patience", str(PATIENCE),
        "--output-dir", str(out / "trained"), "--seed", str(args.seed), "--validate-dataset", "warn",
    )

    runs = sorted((out / "trained").glob("*/best.keras"))
    if not runs:
        raise SystemExit("best.keras не найден")
    best = runs[-1]

    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name("export_ocr_onnx.py")), "--model", str(best),
                    "--plate-config", str(plate_cfg), "--out", str(weights_dir / "ocr_ru.onnx")], check=True)
    shutil.copy2(plate_cfg, weights_dir / "ocr_ru_plate_config.yaml")


if __name__ == "__main__":
    main()
