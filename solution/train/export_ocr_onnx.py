from __future__ import annotations

import os

os.environ["KERAS_BACKEND"] = "tensorflow"   
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import contextlib
import tempfile
from pathlib import Path
from types import SimpleNamespace


@contextlib.contextmanager
def _temp_file(suffix: str = ""):
    with tempfile.TemporaryDirectory() as d:
        yield SimpleNamespace(name=os.path.join(d, "model" + suffix))


def main() -> None:
    p = argparse.ArgumentParser(description="экспорт OCR-модели .keras в ONNX")
    p.add_argument("--model", required=True)
    p.add_argument("--plate-config", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from fast_plate_ocr.cli import export as export_mod
    from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
    from fast_plate_ocr.train.utilities.utils import load_keras_model

    export_mod.NamedTemporaryFile = _temp_file   
    plate_config = load_plate_config_from_yaml(Path(args.plate_config))
    model = load_keras_model(Path(args.model), plate_config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    export_mod.export_onnx(model=model, plate_config=plate_config, out_file=out, simplify=True,
                           dynamic_batch=False, skip_validation=False)
    print(f"ONNX сохранён: {out} ({out.stat().st_size / 2**20:.1f} МБ)")


if __name__ == "__main__":
    main()
