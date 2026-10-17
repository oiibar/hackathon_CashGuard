from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.pipeline.embedder import Embedder  # noqa: E402
from app.pipeline.learned import MODEL_PATH, embed_moments, sample_indices  # noqa: E402
from app.training import Video, dump_json, save_base, train  # noqa: E402

BASE_SET = ROOT / "backend" / "models" / "base_set.npz"


def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [{k: v.strip() for k, v in r.items()} for r in csv.DictReader(f)]


def embeddings(cache: Path, name: str, embedder: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = cache / f"emb_{name}.npz"
    if not path.exists():
        if not embedder:
            embedder.append(Embedder())
        scene = pickle.loads((cache / f"{name}.pkl").read_bytes())
        idx = sample_indices(scene.obs)
        full, desk = embed_moments(embedder[0], scene.obs, idx, scene.zones if scene.localized else None)
        t = np.array([scene.obs[i].t for i in idx], np.float32)
        np.savez(path, t=t, full=full, desk=desk)
        print(f"embedded {name}: {len(idx)} moments")
    z = np.load(path)
    return z["t"], z["full"], z["desk"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=ROOT / "ml" / "labels.csv")
    ap.add_argument("--types", type=Path, default=ROOT / "ml" / "violation_types.csv")
    ap.add_argument("--cache", type=Path, default=ROOT / "ml" / "cache")
    ap.add_argument("--folds", type=int, default=10, help="grouped CV folds; 0 = leave-one-video-out")
    ap.add_argument("--cv-only", action="store_true")
    args = ap.parse_args()

    types = {r["video"]: r["type"] for r in load_csv(args.types)}
    videos, embedder = [], []
    for row in load_csv(args.labels):
        name = row["video"]
        if not (args.cache / f"{name}.pkl").exists() and not (args.cache / f"emb_{name}.npz").exists():
            print(f"skip {name}: no cache, run ml/evaluate.py first")
            continue
        t, full, desk = embeddings(args.cache, name, embedder)
        ts = float(row["timestamp_sec"]) if row["timestamp_sec"] else None
        videos.append(Video(name, t, full, desk, row["label"], ts, types.get(name)))

    result = train(videos, n_folds=args.folds)
    m = result.metrics
    print(f"\nselected k={result.params['k']} C={result.params['c']} threshold={result.params['threshold']}")
    print(f"accuracy {m['tp'] + m['tn']}/{m['videos']}  recall {m['tp']}/{m['positives']}  "
          f"false alarms {m['fp']}/{m['negatives']}  F1 {m['f1']}  ROC-AUC {m['roc_auc']}  PR-AUC {m['pr_auc']}")
    print(f"timecode median {m['timecode']['median_sec']} s, within 3 s {m['timecode']['within_3s']}")
    print(f"fit: {m['fit']}")

    dump_json(ROOT / "ml" / "cv_heldout.json", {
        r["name"]: {"max": r["score"], "violation": r["predicted"], "first": r["first_t"], "type": r["predicted_type"]}
        for r in m["per_video"]})
    if args.cv_only:
        return 0
    save_base(BASE_SET, videos)
    result.model.save(MODEL_PATH)
    print(f"saved {MODEL_PATH.name} and {BASE_SET.name} ({BASE_SET.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
