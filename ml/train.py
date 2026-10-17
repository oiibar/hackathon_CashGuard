from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml from the Roboflow YOLOv8 export")
    ap.add_argument("--model", default=str(ROOT / "backend" / "models" / "yolov8n.pt"),
                    help="start weights: yolov8n.pt (fast) or yolov8s.pt (more accurate)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="")
    ap.add_argument("--name", default="cash_custom")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        name=args.name,
        project=str(ROOT / "ml" / "runs"),
        patience=20,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        plots=True,
    )
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map:.3f}  mAP50: {metrics.box.map50:.3f}")

    best = Path(model.trainer.best)
    dest = ROOT / "backend" / "models" / f"{args.name}.pt"
    shutil.copy(best, dest)
    print(f"copied {best} -> {dest}")


if __name__ == "__main__":
    main()
