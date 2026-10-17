from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from .pipeline.learned import Model, features, smooth

POS_BEFORE, POS_AFTER, NEG_GAP = 1.0, 2.0, 5.0
GRID = [(k, c) for k in (8, 16, 32, 64) for c in (0.01, 0.1, 1.0)]
OVERFIT_GAP = 0.15
UNDERFIT_AUC = 0.65


@dataclass
class Video:
    name: str
    t: np.ndarray
    full: np.ndarray
    desk: np.ndarray
    label: str
    label_t: float | None = None
    vtype: str | None = None
    source: str = "base"
    _x: np.ndarray | None = field(default=None, repr=False)

    @property
    def positive(self) -> bool:
        return self.label == "violation"

    @property
    def x(self) -> np.ndarray:
        if self._x is None:
            self._x = features(self.t, self.full.astype(np.float32), self.desk.astype(np.float32))
        return self._x

    def window(self) -> np.ndarray:
        if not self.positive or self.label_t is None:
            return np.zeros(len(self.t), bool)
        return (self.t >= self.label_t - POS_BEFORE) & (self.t <= self.label_t + POS_AFTER)

    def moment_labels(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.positive:
            return np.ones(len(self.t), bool), np.zeros(len(self.t), int)
        pos = self.window()
        keep = pos | (self.t < self.label_t - NEG_GAP)
        return keep, pos.astype(int)


def load_base(path: Path) -> list[Video]:
    if not path.exists():
        return []
    z = np.load(path, allow_pickle=False)
    out = []
    for i, name in enumerate(z["names"]):
        m = z["video"] == i
        ts = float(z["label_t"][i])
        out.append(Video(str(name), z["t"][m], z["full"][m], z["desk"][m], str(z["labels"][i]),
                         None if np.isnan(ts) else ts, str(z["types"][i]) or None, "base"))
    return out


def save_base(path: Path, videos: list[Video]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        names=np.array([v.name for v in videos]),
        labels=np.array([v.label for v in videos]),
        label_t=np.array([np.nan if v.label_t is None else v.label_t for v in videos], np.float32),
        types=np.array([v.vtype or "" for v in videos]),
        video=np.concatenate([np.full(len(v.t), i, np.int16) for i, v in enumerate(videos)]),
        t=np.concatenate([v.t for v in videos]).astype(np.float32),
        full=np.concatenate([v.full for v in videos]).astype(np.float16),
        desk=np.concatenate([v.desk for v in videos]).astype(np.float16),
    )


def training_set(videos: Iterable[Video]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, ws = [], [], []
    for v in videos:
        keep, y = v.moment_labels()
        if not keep.any():
            continue
        xs.append(v.x[keep])
        ys.append(y[keep])
        ws.append(np.full(keep.sum(), 1.0 / keep.sum()))
    w = np.concatenate(ws)
    return np.vstack(xs), np.concatenate(ys), w * len(w) / w.sum()


def fit(videos: list[Video], k: int, c: float) -> Model:
    x, y, w = training_set(videos)
    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)
    pca = PCA(n_components=min(k, *xs.shape), random_state=0, svd_solver="randomized").fit(xs)
    z = pca.transform(xs)
    clf = LogisticRegression(C=c, class_weight="balanced", max_iter=3000).fit(z, y, sample_weight=w)
    model = Model(scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32),
                  pca.mean_.astype(np.float32), pca.components_.astype(np.float32),
                  clf.coef_[0].astype(np.float32), float(clf.intercept_[0]), 0.5,
                  np.zeros((0, pca.n_components_), np.float32), [])
    proto, types = [], []
    for v in videos:
        win = v.window()
        if v.vtype and win.any():
            proto.append(model.project(v.x[win]).mean(axis=0))
            types.append(v.vtype)
    if proto:
        model.proto, model.proto_types = np.array(proto, np.float32), types
    return model


def folds(videos: list[Video], n: int, seed: int = 0) -> list[list[int]]:
    if n <= 1 or n >= len(videos):
        return [[i] for i in range(len(videos))]
    rng = np.random.default_rng(seed)
    out: list[list[int]] = [[] for _ in range(n)]
    pos = [i for i, v in enumerate(videos) if v.positive]
    neg = [i for i, v in enumerate(videos) if not v.positive]
    for j, i in enumerate(list(rng.permutation(pos)) + list(rng.permutation(neg))):
        out[j % n].append(int(i))
    return [f for f in out if f]


@dataclass
class Scored:
    video: Video
    t: np.ndarray
    s: np.ndarray
    raw: np.ndarray
    predicted_type: str | None = None

    @property
    def peak(self) -> float:
        return float(self.s.max()) if len(self.s) else 0.0

    def first_above(self, threshold: float) -> float | None:
        hit = np.flatnonzero(self.s >= threshold)
        return float(self.t[hit[0]]) if len(hit) else None


def score_video(model: Model, v: Video) -> Scored:
    raw = model.score(v.x)
    sc = Scored(v, v.t, smooth(raw, v.t), raw)
    win = v.window()
    if v.vtype and win.any() and model.proto_types:
        sc.predicted_type = model.type_of(v.x[win].mean(axis=0))
    return sc


def cross_validate(videos: list[Video], k: int, c: float, n_folds: int) -> list[Scored]:
    held: list[Scored | None] = [None] * len(videos)
    for fold in folds(videos, n_folds):
        test = set(fold)
        model = fit([v for i, v in enumerate(videos) if i not in test], k, c)
        for i in fold:
            held[i] = score_video(model, videos[i])
    return [h for h in held if h is not None]


def pick_threshold(held: list[Scored]) -> float:
    best = None
    for thr in sorted({round(h.peak, 4) for h in held}):
        m = verdict_counts(held, thr)
        key = (mcc(m), m["tp"], -thr)
        if best is None or key > best[0]:
            best = (key, thr)
    return best[1] if best else 0.5


def verdict_counts(held: list[Scored], thr: float) -> dict[str, int]:
    m = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for h in held:
        pred = h.peak >= thr
        m[("t" if pred == h.video.positive else "f") + ("p" if pred else "n")] += 1
    return m


def f_beta(m: dict[str, int], beta: float) -> float:
    b2 = beta * beta
    denom = (1 + b2) * m["tp"] + b2 * m["fn"] + m["fp"]
    return (1 + b2) * m["tp"] / denom if denom else 0.0


def mcc(m: dict[str, int]) -> float:
    tp, fp, tn, fn = m["tp"], m["fp"], m["tn"], m["fn"]
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return (tp * tn - fp * fn) / denom if denom else 0.0


def _ratio(a: float, b: float) -> float | None:
    return round(a / b, 4) if b else None


def _safe(fn, *args) -> float | None:
    try:
        return round(float(fn(*args)), 4)
    except ValueError:
        return None


def metrics(held: list[Scored], thr: float) -> dict[str, Any]:
    m = verdict_counts(held, thr)
    y = np.array([h.video.positive for h in held], int)
    p = np.array([h.peak for h in held])
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    out: dict[str, Any] = {
        "videos": len(held), "positives": pos, "negatives": neg, "threshold": round(thr, 4), **m,
        "accuracy": _ratio(m["tp"] + m["tn"], len(held)),
        "precision": _ratio(m["tp"], m["tp"] + m["fp"]),
        "recall": _ratio(m["tp"], pos),
        "specificity": _ratio(m["tn"], neg),
        "false_positive_rate": _ratio(m["fp"], neg),
        "f1": round(f_beta(m, 1.0), 4),
        "f2": round(f_beta(m, 2.0), 4),
        "mcc": round(mcc(m), 4),
        "balanced_accuracy": None,
        "roc_auc": _safe(roc_auc_score, y, p),
        "pr_auc": _safe(average_precision_score, y, p),
        "brier": round(float(np.mean((p - y) ** 2)), 4) if len(y) else None,
    }
    if out["recall"] is not None and out["specificity"] is not None:
        out["balanced_accuracy"] = round((out["recall"] + out["specificity"]) / 2, 4)
    if pos and neg:
        fpr, tpr, _ = roc_curve(y, p)
        out["roc_curve"] = [[round(float(a), 4), round(float(b), 4)] for a, b in zip(fpr, tpr)]

    my, ms = [], []
    for h in held:
        keep, lab = h.video.moment_labels()
        my.append(lab[keep])
        ms.append(h.raw[keep])
    if my:
        out["moment_roc_auc"] = _safe(roc_auc_score, np.concatenate(my), np.concatenate(ms))

    dts = []
    for h in held:
        first = h.first_above(thr)
        if h.video.positive and h.video.label_t is not None and first is not None:
            dts.append(abs(first - h.video.label_t))
    out["timecode"] = {
        "n": len(dts),
        "mae_sec": round(float(np.mean(dts)), 2) if dts else None,
        "median_sec": round(float(np.median(dts)), 2) if dts else None,
        "within_3s": _ratio(sum(d <= 3 for d in dts), len(dts)),
        "within_5s": _ratio(sum(d <= 5 for d in dts), len(dts)),
    }

    typed = [h for h in held if h.video.vtype and h.predicted_type]
    out["type"] = {"n": len(typed), "correct": sum(h.predicted_type == h.video.vtype for h in typed),
                   "accuracy": _ratio(sum(h.predicted_type == h.video.vtype for h in typed), len(typed))}
    return out


def fit_state(train_auc: float | None, cv_auc: float | None) -> str:
    if train_auc is None or cv_auc is None:
        return "unknown"
    if cv_auc < UNDERFIT_AUC and train_auc < UNDERFIT_AUC + 0.1:
        return "underfit"
    if train_auc - cv_auc > OVERFIT_GAP:
        return "overfit"
    return "good"


def _moment_auc(model: Model, videos: list[Video]) -> float | None:
    x, y, _ = training_set(videos)
    return _safe(roc_auc_score, y, model.score(x))


@dataclass
class TrainResult:
    model: Model
    params: dict[str, Any]
    metrics: dict[str, Any]
    held: list[Scored]


def train(videos: list[Video], n_folds: int = 10, grid: list[tuple[int, float]] | None = None,
          log=print) -> TrainResult:
    if sum(v.positive for v in videos) < 2 or sum(not v.positive for v in videos) < 2:
        raise ValueError("need at least two videos of each class")
    started = time.time()
    search = []
    best = None
    for k, c in grid or GRID:
        held = cross_validate(videos, k, c, n_folds)
        thr = pick_threshold(held)
        m = verdict_counts(held, thr)
        cv_auc = _safe(roc_auc_score, [h.video.positive for h in held], [h.peak for h in held])
        row = {"k": k, "c": c, "threshold": round(thr, 4), "mcc": round(mcc(m), 4), "f1": round(f_beta(m, 1.0), 4),
               "roc_auc": cv_auc, **m}
        search.append(row)
        log(f"k={k:<3} C={c:<5} thr={thr:.3f} MCC={row['mcc']:.3f} F1={row['f1']:.3f} AUC={cv_auc} "
            f"tp={m['tp']} fp={m['fp']} fn={m['fn']}")
        key = (row["mcc"], cv_auc or 0.0, -k, -c)
        if best is None or key > best[0]:
            best = (key, k, c, thr, held)

    _, k, c, thr, held = best
    cv = metrics(held, thr)
    model = fit(videos, k, c)
    model.threshold = thr
    train_m = metrics([score_video(model, v) for v in videos], thr)
    train_auc = _moment_auc(model, videos)
    cv["fit"] = {
        "train_moment_roc_auc": train_auc,
        "cv_moment_roc_auc": cv.get("moment_roc_auc"),
        "gap": round(train_auc - cv["moment_roc_auc"], 4) if train_auc and cv.get("moment_roc_auc") else None,
        "train_f1": train_m["f1"],
        "cv_f1": cv["f1"],
        "state": fit_state(train_auc, cv.get("moment_roc_auc")),
    }
    for row in search:
        row["selected"] = row["k"] == k and row["c"] == c
    params = {"k": k, "c": c, "threshold": round(thr, 4), "folds": len(folds(videos, n_folds)),
              "videos": len(videos), "reviewed": sum(v.source == "review" for v in videos),
              "train_sec": round(time.time() - started, 1)}
    cv["search"] = search
    cv["per_video"] = [
        {"name": h.video.name, "label": h.video.label, "label_t": h.video.label_t, "score": round(h.peak, 4),
         "predicted": h.peak >= thr, "first_t": h.first_above(thr), "type": h.video.vtype,
         "predicted_type": h.predicted_type, "source": h.video.source}
        for h in held
    ]
    model.meta = {"params": params, "metrics": cv}
    return TrainResult(model, params, cv, held)


def heldout_verdicts(model: Model, videos: list[Video]) -> dict[str, bool]:
    return {v.name: score_video(model, v).peak >= model.threshold for v in videos}


def dump_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
