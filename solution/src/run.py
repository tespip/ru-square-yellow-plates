from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .config import load_config
from .pipeline import Pipeline, draw_predictions
from .postprocess import write_csv

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
LOG = logging.getLogger("alpr")


def imread_unicode(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="распознавание ГРЗ: каталог изображений -> CSV")
    p.add_argument("--input", "-i", default=os.environ.get("ALPR_INPUT_DIR"))
    p.add_argument("--output", "-o", default=os.environ.get("ALPR_OUTPUT_CSV", "result.csv"))
    p.add_argument("--config", "-c", default=os.environ.get("ALPR_CONFIG"))
    p.add_argument("--debug-dir", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    if not args.input:
        p.error("укажите --input или переменную окружения ALPR_INPUT_DIR")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(levelname)s %(message)s")
    in_dir = Path(args.input)
    if not in_dir.is_dir():
        LOG.error("каталог не найден: %s", in_dir)
        return 2
    cfg = load_config(args.config)
    pipe = Pipeline(cfg)
    pipe.warmup()

    images = list_images(in_dir)
    if args.limit:
        images = images[: args.limit]
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    records = []
    times: list[float] = []
    for i, path in enumerate(images, 1):
        frame = imread_unicode(path)
        if frame is None:
            LOG.warning("не удалось прочитать %s - пропуск", path.name)
            continue
        t0 = time.perf_counter()
        res = pipe.process(frame, keep_debug=debug_dir is not None)
        times.append((time.perf_counter() - t0) * 1000)
        for row in res.rows:
            records.append((path.name, row))
        if debug_dir:
            ok, buf = cv2.imencode(".jpg", draw_predictions(frame, res))
            if ok:
                buf.tofile(str(debug_dir / path.name))
        if not args.quiet and (i % 25 == 0 or i == len(images)):
            LOG.info("%d/%d изображений, среднее %.1f мс", i, len(images), sum(times) / len(times))

    n = write_csv(Path(args.output), records)
    if times:
        LOG.info("готово: %d строк для %d изображений - %s; среднее время %.1f мс/изобр.",
                 n, len(times), args.output, sum(times) / len(times))
    else:
        LOG.warning("в каталоге нет изображений .jpg/.png; записан пустой CSV - %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
