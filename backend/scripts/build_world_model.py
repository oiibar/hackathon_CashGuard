from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BASE_WEIGHTS = "yolov8s-worldv2.pt"
VOCAB = ["banknote", "stack of banknotes", "money", "smartphone"]


def main() -> int:
    os.environ.pop("YOLO_OFFLINE", None)
    from ultralytics import YOLOWorld

    MODELS_DIR.mkdir(exist_ok=True)
    dest = MODELS_DIR / "cash_world.pt"
    cwd = os.getcwd()
    os.chdir(MODELS_DIR)
    try:
        model = YOLOWorld(BASE_WEIGHTS)
        model.set_classes(VOCAB)
        model.save(str(dest))
    finally:
        os.chdir(cwd)
    base = MODELS_DIR / BASE_WEIGHTS
    if base.exists():
        base.unlink()
    shutil.rmtree(MODELS_DIR / "runs", ignore_errors=True)
    print(f"ok   {dest} ({dest.stat().st_size / 1e6:.1f} MB), classes: {VOCAB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
