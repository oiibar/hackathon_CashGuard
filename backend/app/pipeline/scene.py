from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .detector import Detection

GRID = 24


@dataclass
class Observation:
    t: float
    dets: list[Detection]
    jpeg: bytes
    brightness: float
    sharpness: float
    size: tuple[int, int]

    def frame(self) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(self.jpeg, np.uint8), cv2.IMREAD_COLOR)


def frame_stats(frame: np.ndarray) -> tuple[float, float]:
    small = cv2.resize(frame, (320, max(int(320 * frame.shape[0] / frame.shape[1]), 1)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _cell(x: float, y: float) -> tuple[int, int]:
    return min(int(y * GRID), GRID - 1), min(int(x * GRID), GRID - 1)


@dataclass
class ZoneInference:
    zones: dict[str, list[list[float]]]
    localized: bool
    info: dict[str, Any] = field(default_factory=dict)


FULL_FRAME = [[0, 0], [1, 0], [1, 1], [0, 1]]


def infer_zones(obs: list[Observation], min_occupancy: float = 0.3, min_conf: float = 0.35,
                expand: float = 0.05, desk_expand: float = 0.75) -> ZoneInference:
    persons_per_frame = []
    occ = np.zeros((GRID, GRID), np.float32)
    weight = np.zeros((GRID, GRID), np.float32)
    for o in obs:
        w, h = o.size
        people = [d for d in o.dets if d.cls == "person" and d.conf >= min_conf]
        persons_per_frame.append(people)
        seen: dict[tuple[int, int], float] = {}
        for d in people:
            cx, cy = d.center
            cell = _cell(cx / w, cy / h)
            seen[cell] = max(seen.get(cell, 0.0), (d.xyxy[3] - d.xyxy[1]) / h)
        for (r, c), size in seen.items():
            occ[r, c] += 1
            weight[r, c] += size
    if not obs:
        return ZoneInference({"cash_zone": FULL_FRAME}, False, {"reason": "no frames"})

    occ = np.minimum(cv2.blur(occ, (3, 3), borderType=cv2.BORDER_CONSTANT) * 9 / len(obs), 1.0)
    weight = cv2.blur(weight, (3, 3), borderType=cv2.BORDER_CONSTANT)
    r, c = np.unravel_index(int(np.argmax(weight * (occ >= min_occupancy))), occ.shape)
    peak = float(occ[r, c])
    if peak < min_occupancy:
        return ZoneInference({"cash_zone": FULL_FRAME}, False,
                             {"reason": f"no stationary person (peak occupancy {peak:.2f})"})

    px, py = (c + 0.5) / GRID, (r + 0.5) / GRID
    boxes = []
    for o, people in zip(obs, persons_per_frame):
        w, h = o.size
        for d in people:
            cx, cy = d.center
            if abs(cx / w - px) <= 1.5 / GRID and abs(cy / h - py) <= 1.5 / GRID:
                x1, y1, x2, y2 = d.xyxy
                boxes.append((x1 / w, y1 / h, x2 / w, y2 / h))
    box = np.median(np.array(boxes), axis=0)
    return ZoneInference({"cash_zone": _rect(box, expand), "desk_zone": _rect(box, desk_expand),
                          "client_zone": FULL_FRAME}, True,
                         {"method": "auto", "workplace_occupancy": round(peak, 3)})


def _rect(box: np.ndarray, expand: float) -> list[list[float]]:
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = round(max(x1 - expand * bw, 0.0), 4), round(min(x2 + expand * bw, 1.0), 4)
    y1, y2 = round(max(y1 - expand * bh, 0.0), 4), round(min(y2 + expand * bh, 1.0), 4)
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def static_fixtures(obs: list[Observation], classes: set[str], min_share: float = 0.2, min_span: float = 0.7,
                    iou_thr: float = 0.5) -> dict[str, list[tuple[float, float, float, float]]]:
    if len(obs) < 20:
        return {}
    out: dict[str, list[tuple[float, float, float, float]]] = {}
    span = max(obs[-1].t - obs[0].t, 1e-6)
    for cls in classes:
        clusters: list[list[Any]] = []
        for o in obs:
            w, h = o.size
            matched = set()
            for d in o.dets:
                if d.cls != cls:
                    continue
                b = (d.xyxy[0] / w, d.xyxy[1] / h, d.xyxy[2] / w, d.xyxy[3] / h)
                for i, cl in enumerate(clusters):
                    if i not in matched and iou(cl[0], b) >= iou_thr:
                        cl[1] += 1
                        cl[3] = o.t
                        matched.add(i)
                        break
                else:
                    clusters.append([b, 1, o.t, o.t])
                    matched.add(len(clusters) - 1)
        fixed = [cl[0] for cl in clusters
                 if cl[1] >= min_share * len(obs) and (cl[3] - cl[2]) >= min_span * span]
        if fixed:
            out[cls] = fixed
    return out


def drop_background_people(obs: list[Observation], min_rel: float = 0.45, min_conf: float = 0.35) -> float:
    heights = sorted((d.xyxy[3] - d.xyxy[1]) / o.size[1] for o in obs for d in o.dets
                     if d.cls == "person" and d.conf >= min_conf)
    if not heights:
        return 0.0
    ref = heights[int(0.9 * (len(heights) - 1))]
    for o in obs:
        o.dets = [d for d in o.dets if d.cls != "person" or (d.xyxy[3] - d.xyxy[1]) / o.size[1] >= min_rel * ref]
    return ref


def dedupe_people(obs: list[Observation], thr: float = 0.7) -> None:
    for o in obs:
        people = sorted((d for d in o.dets if d.cls == "person"), key=_area, reverse=True)
        keep: list[Detection] = []
        for d in people:
            if all(_inter(d.xyxy, k.xyxy) < thr * _area(d) for k in keep):
                keep.append(d)
        o.dets = [d for d in o.dets if d.cls != "person"] + keep


def _area(d: Detection) -> float:
    x1, y1, x2, y2 = d.xyxy
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def _inter(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def zone_bbox(poly: list[list[float]]) -> tuple[float, float, float, float]:
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return max(min(xs), 0.0), max(min(ys), 0.0), min(max(xs), 1.0), min(max(ys), 1.0)


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def drop_fixtures(obs: list[Observation], fixtures: dict[str, list[tuple[float, ...]]],
                  iou_thr: float = 0.5) -> None:
    if not fixtures:
        return
    for o in obs:
        w, h = o.size
        keep = []
        for d in o.dets:
            boxes = fixtures.get(d.cls)
            if boxes:
                b = (d.xyxy[0] / w, d.xyxy[1] / h, d.xyxy[2] / w, d.xyxy[3] / h)
                if any(iou(b, f) >= iou_thr for f in boxes):
                    continue
            keep.append(d)
        o.dets = keep
