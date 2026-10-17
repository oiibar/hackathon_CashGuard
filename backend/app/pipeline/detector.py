"""YOLOv8 wrapper. Loads one or more local .pt files and merges detections.

Stock `yolov8n.pt` (COCO) already gives: person, cell phone, handbag,
backpack, suitcase, laptop. Bank-specific classes (cash, cash_drawer_open,
door_open, ...) come from your own fine-tuned model, see ml/train.py.

No face detection or recognition is done anywhere: `person` is a whole-body
box with no identity attached.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Keep Ultralytics fully offline: no hub sync, no update checks, no
# auto-install of missing packages at runtime.
os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("YOLO_VERBOSE", "False")


@dataclass(frozen=True)
class Detection:
    cls: str
    conf: float
    xyxy: tuple[float, float, float, float]  # pixels in the analyzed frame

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def bottom(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.xyxy
        return (x1 + x2) / 2, y2


class Detector:
    def __init__(self, model_paths: list[str], device: str = "", imgsz: int = 640, conf: float = 0.35,
                 wanted_classes: set[str] | None = None):
        from ultralytics import YOLO, settings as yolo_settings

        try:
            yolo_settings.update({"sync": False})
        except Exception:
            pass

        self.models = []
        for p in model_paths:
            if not Path(p).exists():
                raise FileNotFoundError(f"Model weights not found: {p}. Run scripts/download_weights.py")
            model = YOLO(p)
            # Only ask each model for classes that some rule needs: fewer boxes,
            # less NMS work, less noise.
            names: dict[int, str] = model.names
            if wanted_classes:
                ids = [i for i, n in names.items() if n in wanted_classes]
                if not ids:
                    continue
            else:
                ids = None
            self.models.append((model, ids, names))
        if not self.models:
            raise RuntimeError("No loaded model provides any class used in rules.yaml")
        self.device = device or None
        self.imgsz = imgsz
        self.conf = conf

    @property
    def class_names(self) -> set[str]:
        return {n for _, _, names in self.models for n in names.values()}

    def warmup(self) -> None:
        self(np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8))

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        out: list[Detection] = []
        for model, ids, names in self.models:
            results = model.predict(
                frame,
                imgsz=self.imgsz,
                conf=self.conf,
                classes=ids,
                device=self.device,
                verbose=False,
            )
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for b, c, k in zip(xyxy, confs, clss):
                out.append(Detection(cls=names[int(k)], conf=float(c), xyxy=tuple(float(v) for v in b)))
        return out
