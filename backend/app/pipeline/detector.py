from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("YOLO_VERBOSE", "False")


@dataclass(frozen=True)
class Detection:
    cls: str
    conf: float
    xyxy: tuple[float, float, float, float]

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
                 wanted_classes: set[str] | None = None, aliases: dict[str, str] | None = None,
                 class_conf: dict[str, float] | None = None):
        from ultralytics import YOLO, settings as yolo_settings

        try:
            yolo_settings.update({"sync": False})
        except Exception:
            pass

        self.aliases = aliases or {}
        self.class_conf = class_conf or {}
        self.default_conf = conf
        self.conf = min([conf, *self.class_conf.values()])

        self.models = []
        for p in model_paths:
            if not Path(p).exists():
                raise FileNotFoundError(f"Model weights not found: {p}. Run scripts/download_weights.py")
            model = YOLO(p)
            names: dict[int, str] = {i: self.aliases.get(n, n) for i, n in model.names.items()}
            if wanted_classes:
                ids = [i for i, n in names.items() if n in wanted_classes]
                if not ids:
                    continue
            else:
                ids = None
            self.models.append((model, ids, names, Path(p).name))
        if not self.models:
            raise RuntimeError("No loaded model provides any class used in rules.yaml")
        self.device = device or None
        self.imgsz = imgsz

    @property
    def class_names(self) -> set[str]:
        return {n for _, ids, names, _ in self.models for i, n in names.items() if ids is None or i in ids}

    @property
    def model_names(self) -> list[str]:
        return [name for *_, name in self.models]

    def warmup(self) -> None:
        self(np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8))

    def __call__(self, frame: np.ndarray, only: set[str] | None = None) -> list[Detection]:
        out: list[Detection] = []
        for model, ids, names, _ in self.models:
            if only is not None:
                ids = [i for i in (ids if ids is not None else names) if names[i] in only]
                if not ids:
                    continue
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
                name = names[int(k)]
                if c < self.class_conf.get(name, self.default_conf):
                    continue
                out.append(Detection(cls=name, conf=float(c), xyxy=tuple(float(v) for v in b)))
        return out
