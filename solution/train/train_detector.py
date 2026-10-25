from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

from common import SOLUTION_ROOT, PlateRecord, group_split, read_meta

DEFAULT_CLASSES = ["type1", "type1a", "type1b"]


def image_size(path: Path) -> tuple[int, int]:
    import cv2

    from common import imread

    img = imread(path)
    if img is None:
        raise ValueError(f"не читается {path}")
    h, w = img.shape[:2]
    return w, h


def export_split(records: list[PlateRecord], images: set[Path], dst: Path, split: str, classes: list[str],
                 real_repeat: int = 1) -> int:
    img_dir, lbl_dir = dst / "images" / split, dst / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    by_img: dict[Path, list[PlateRecord]] = {}
    for r in records:
        by_img.setdefault(r.image, []).append(r)
    n = 0
    for img in sorted(images):
        rs = by_img.get(img, [])
        w, h = image_size(img)
        lines = []
        for r in rs:
            if r.plate_type not in classes or not r.is_vehicle:
                continue
            x, y, bw, bh = r.bbox
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            lines.append(f"{classes.index(r.plate_type)} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
        is_real = not any(r.is_synthetic for r in rs)
        copies = real_repeat if is_real else 1
        for k in range(copies):
            stem = img.stem if k == 0 else f"{img.stem}_rep{k}"
            target = img_dir / f"{stem}{img.suffix}"
            if not target.exists():
                try:
                    target.hardlink_to(img)
                except OSError:
                    shutil.copy2(img, target)
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="обучение многоклассового детектора yolo ГРЗ")
    p.add_argument("--dataset", required=True, nargs="+")
    p.add_argument("--extra", nargs="*", default=[])
    p.add_argument("--out", default="runs/detector")
    p.add_argument("--model", default="yolov9t.pt", help="стартовые веса ultralytics")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--other-as-class", action="store_true")
    p.add_argument("--no-synthetic", action="store_true")
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--real-repeat", type=int, default=1, help="сколько раз класть каждое реальное фото в train")
    p.add_argument("--device", default=None)
    p.add_argument("--export-only", default=None)
    p.add_argument("--weights-dir", default=str(SOLUTION_ROOT / "weights"))
    args = p.parse_args()

    classes = DEFAULT_CLASSES + (["other"] if args.other_as_class else [])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    if args.export_only:
        best = Path(args.export_only)
    else:
        records: list[PlateRecord] = []
        for d in args.dataset:
            records += read_meta(Path(d))
        if args.no_synthetic:
            records = [r for r in records if not r.is_synthetic]
        train_imgs, val_imgs = group_split(records, args.val_fraction, args.seed)
        extra: list[PlateRecord] = []
        for d in args.extra:
            extra += read_meta(Path(d))
        if extra:   
            records += extra
            train_imgs |= {r.image for r in extra}
            print(f"--extra: +{len({r.image for r in extra})} изображений в train")
        data_dir = out / "data"
        n_tr = export_split(records, train_imgs, data_dir, "train", classes, real_repeat=args.real_repeat)
        n_va = export_split(records, val_imgs, data_dir, "val", classes)
        data_yaml = out / "data.yaml"
        data_yaml.write_text(yaml.safe_dump({
            "path": str(data_dir.resolve()), "train": "images/train", "val": "images/val",
            "names": {i: c for i, c in enumerate(classes)},
        }, allow_unicode=True), encoding="utf-8")
        print(f"train: {n_tr} изображений (реальные + синтетика), val: {n_va} (только реальные); классы {classes}")

        model = YOLO(args.model)
        model.train(data=str(data_yaml), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
                    seed=args.seed, device=args.device, project=str(out.resolve()), name="train", exist_ok=True,
                    patience=30, close_mosaic=10, deterministic=True, scale=args.scale)
        best = out / "train" / "weights" / "best.pt"

    model = YOLO(str(best))
    onnx_path = Path(model.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True, dynamic=False, nms=False))
    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    dst = weights_dir / f"detector_{args.imgsz}.onnx"
    shutil.copy2(onnx_path, dst)
    (weights_dir / "detector_classes.yaml").write_text(
        yaml.safe_dump({"backend": "ultralytics", "model": f"weights/{dst.name}", "class_labels": classes,
                        "imgsz": args.imgsz}, allow_unicode=True), encoding="utf-8")



if __name__ == "__main__":
    main()
