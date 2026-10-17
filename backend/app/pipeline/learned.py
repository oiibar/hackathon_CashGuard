from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .scene import Observation, zone_bbox

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "violation_clf.npz"
STEP_SEC = 0.5


def sample_indices(obs: list[Observation], step: float = STEP_SEC) -> list[int]:
    out, next_t = [], -1e9
    for i, o in enumerate(obs):
        if o.t >= next_t:
            out.append(i)
            next_t = o.t + step - 1e-6
    return out


def _crop(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    return frame[int(y1 * h):max(int(y2 * h), int(y1 * h) + 2), int(x1 * w):max(int(x2 * w), int(x1 * w) + 2)]


def embed_moments(embedder, obs: list[Observation], idx: list[int], zones: dict[str, Any] | None,
                  batch: int = 16) -> tuple[np.ndarray, np.ndarray]:
    box = zone_bbox(zones["desk_zone"]) if zones and "desk_zone" in zones else (0.1, 0.2, 0.9, 0.9)
    full, desk = [], []
    for i in range(0, len(idx), batch):
        frames = [obs[j].frame() for j in idx[i:i + batch]]
        full.append(embedder(frames))
        desk.append(embedder([_crop(f, box) for f in frames]))
    if not full:
        return np.zeros((0, 512), np.float32), np.zeros((0, 512), np.float32)
    return np.concatenate(full), np.concatenate(desk)


def _window_mean(x: np.ndarray, t: np.ndarray, half: float) -> np.ndarray:
    out = np.empty_like(x)
    for i, ti in enumerate(t):
        out[i] = x[np.abs(t - ti) <= half + 1e-6].mean(axis=0)
    return out


def _lag(x: np.ndarray, t: np.ndarray, lag: float) -> np.ndarray:
    j = np.clip(np.searchsorted(t, t - lag), 0, len(t) - 1)
    return x - x[j]


def features(t: np.ndarray, e_full: np.ndarray, e_desk: np.ndarray) -> np.ndarray:
    d_full = e_full - np.median(e_full, axis=0, keepdims=True)
    d_desk = e_desk - np.median(e_desk, axis=0, keepdims=True)
    s_full = _window_mean(d_full, t, 1.0)
    s_desk = _window_mean(d_desk, t, 1.0)
    return np.hstack([s_full, s_desk, _lag(s_full, t, 2.0), _lag(s_desk, t, 2.0)]).astype(np.float32)


def smooth(score: np.ndarray, t: np.ndarray, half: float = 0.75) -> np.ndarray:
    return np.array([score[np.abs(t - ti) <= half + 1e-6].mean() for ti in t])


@dataclass
class Model:
    mean: np.ndarray
    scale: np.ndarray
    pca_mean: np.ndarray
    pca: np.ndarray
    coef: np.ndarray
    intercept: float
    threshold: float
    proto: np.ndarray
    proto_types: list[str]
    meta: dict[str, Any] = field(default_factory=dict)
    calib: tuple[float, float] | None = None

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "Model | None":
        if not path.exists():
            return None
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        calib = tuple(float(v) for v in z["calib"]) if "calib" in z.files else None
        return cls(z["mean"], z["scale"], z["pca_mean"], z["pca"], z["coef"], float(z["intercept"]),
                   float(z["threshold"]), z["proto"], [str(s) for s in z["proto_types"]], meta, calib)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, mean=self.mean, scale=self.scale, pca_mean=self.pca_mean, pca=self.pca, coef=self.coef,
                 intercept=np.float32(self.intercept), threshold=np.float32(self.threshold),
                 proto=self.proto.astype(np.float32), proto_types=np.array(self.proto_types, dtype=str),
                 meta=np.array(json.dumps(self.meta, ensure_ascii=False)),
                 **({"calib": np.array(self.calib, np.float32)} if self.calib else {}))
        tmp.replace(path)

    def project(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale - self.pca_mean) @ self.pca.T

    def score(self, x: np.ndarray) -> np.ndarray:
        z = self.project(x) @ self.coef + self.intercept
        return 1 / (1 + np.exp(-z))

    def confidence(self, score: float) -> float:
        if self.calib is None:
            return float(score)
        a, b = self.calib
        return float(1 / (1 + np.exp(-(a * score + b))))

    def type_of(self, x: np.ndarray) -> str:
        p = self.project(x[None])[0]
        sims = (self.proto @ p) / (np.linalg.norm(self.proto, axis=1) * np.linalg.norm(p) + 1e-9)
        return self.proto_types[int(np.argmax(sims))]


@dataclass
class Episode:
    start: float
    peak: float
    end: float
    score: float
    vtype: str
    peak_index: int


def episodes(model: Model, t: np.ndarray, feats: np.ndarray, gap: float = 1.5, min_len: float = 1.0) -> list[Episode]:
    s = smooth(model.score(feats), t)
    spans, cur = [], None
    for i, (ti, si) in enumerate(zip(t, s)):
        if si < model.threshold:
            continue
        if cur and ti - cur[1] <= gap:
            cur[1] = ti
            if si > s[cur[2]]:
                cur[2] = i
        else:
            if cur:
                spans.append(cur)
            cur = [ti, ti, i]
    if cur:
        spans.append(cur)
    out = []
    for start, end, pk in spans:
        if end - start + STEP_SEC < min_len:
            continue
        vtype = model.type_of(feats[pk]) if model.proto_types else "learned_anomaly"
        out.append(Episode(float(start), float(t[pk]), float(end + STEP_SEC), float(s[pk]), vtype, pk))
    return out


def save_embeddings(path: Path, t: np.ndarray, full: np.ndarray, desk: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, t=t.astype(np.float32), full=full.astype(np.float16), desk=desk.astype(np.float16))
    tmp.replace(path)
