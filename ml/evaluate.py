from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.config import settings  # noqa: E402
from app.pipeline.analyzer import Analyzer  # noqa: E402
from app.pipeline.detector import Detector  # noqa: E402
from app.pipeline.rules import load_yaml, wanted_classes  # noqa: E402
from app.training import mcc  # noqa: E402

VIDEO_EXT = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
NO_DEADLINE = 1e12
OUT = ROOT / "backend" / "models" / "system_metrics.json"


def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [{k: v.strip() for k, v in r.items()} for r in csv.DictReader(f)]


def summarize(rows: list[dict], key: str) -> dict:
    m = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    dts, typed, typed_ok = [], 0, 0
    for r in rows:
        pred = r[key]["violation"]
        truth = r["label"] == "violation"
        m[("t" if pred == truth else "f") + ("p" if pred else "n")] += 1
        if pred and truth and r["label_t"] is not None and r[key]["times"]:
            dts.append(min(abs(t - r["label_t"]) for t in r[key]["times"]))
        if pred and truth and r["type"]:
            typed += 1
            typed_ok += r["type"] in r[key]["types"]
    pos, neg = m["tp"] + m["fn"], m["tn"] + m["fp"]
    prec = m["tp"] / (m["tp"] + m["fp"]) if m["tp"] + m["fp"] else 0.0
    rec = m["tp"] / pos if pos else 0.0
    return {
        **m, "videos": len(rows), "positives": pos, "negatives": neg,
        "accuracy": round((m["tp"] + m["tn"]) / len(rows), 4),
        "precision": round(prec, 4), "recall": round(rec, 4),
        "specificity": round(m["tn"] / neg, 4) if neg else None,
        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "mcc": round(mcc(m), 4),
        "timecode": {"n": len(dts), "mae_sec": round(float(np.mean(dts)), 2) if dts else None,
                     "median_sec": round(float(np.median(dts)), 2) if dts else None,
                     "within_3s": round(sum(d <= 3 for d in dts) / len(dts), 4) if dts else None},
        "type": {"n": typed, "correct": typed_ok, "accuracy": round(typed_ok / typed, 4) if typed else None},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", type=Path)
    ap.add_argument("labels", type=Path)
    ap.add_argument("--types", type=Path, default=ROOT / "ml" / "violation_types.csv")
    ap.add_argument("--cache", type=Path, default=ROOT / "ml" / "cache")
    ap.add_argument("--heldout", type=Path, default=ROOT / "ml" / "cv_heldout.json")
    ap.add_argument("--fresh", action="store_true", help="ignore the detection cache")
    args = ap.parse_args()

    labels = {r["video"]: (r["label"], float(r["timestamp_sec"]) if r["timestamp_sec"] else None)
              for r in load_csv(args.labels)}
    types = {r["video"]: r["type"] for r in load_csv(args.types)}
    held = json.loads(args.heldout.read_text(encoding="utf-8"))
    rules = load_yaml(settings.rules_path)
    det = Detector(settings.model_paths, settings.device, settings.imgsz, settings.conf, wanted_classes(rules),
                   aliases=rules.get("class_aliases"), class_conf=rules.get("class_conf"))
    analyzer = Analyzer(det, rules, settings)
    analyzer.learned = analyzer.embedder = None
    args.cache.mkdir(parents=True, exist_ok=True)

    rows = []
    for video in sorted(p for p in args.videos.iterdir() if p.suffix.lower() in VIDEO_EXT):
        if video.stem not in labels:
            continue
        cache = args.cache / f"{video.stem}.pkl"
        started = time.time()
        if cache.exists() and not args.fresh:
            scene = pickle.loads(cache.read_bytes())
        else:
            info, obs, stopped = analyzer.detect(video, NO_DEADLINE)
            scene = analyzer.prepare(info, obs, None, NO_DEADLINE, stopped)
            cache.write_bytes(pickle.dumps(scene))
        report = analyzer.judge(scene, started)
        found = report["violations"]
        h = held.get(video.stem, {})
        model = {"violation": bool(h.get("violation")), "times": [h["first"]] if h.get("first") is not None else [],
                 "types": [h.get("type")]}
        rule = {"violation": bool(found), "times": [v["timecode_sec"] for v in found],
                "types": [v["type_id"] for v in found]}
        both = {"violation": rule["violation"] or model["violation"], "times": rule["times"] or model["times"],
                "types": rule["types"] if rule["violation"] else model["types"]}
        label, label_t = labels[video.stem]
        rows.append({"name": video.stem, "label": label, "label_t": label_t, "type": types.get(video.stem),
                     "rules": rule, "model": model, "system": both,
                     "always": {"violation": True, "times": [], "types": []}})
        mark = "  " if both["violation"] == (label == "violation") else "XX"
        print(f"{mark}{video.stem:<6} {label:<13} {label_t if label_t is not None else '':>5}  "
              f"rules={'+' if rule['violation'] else '-'} model={'+' if model['violation'] else '-'}  "
              + ", ".join(f"{v['timecode_sec']:.1f}s {v['type_id']}" for v in found))

    out = {name: summarize(rows, name) for name in ("always", "rules", "model", "system")}
    out["generated_at"] = time.time()
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for name in ("always", "rules", "model", "system"):
        s = out[name]
        print(f"{name:<7} accuracy {s['tp'] + s['tn']}/{s['videos']}  recall {s['tp']}/{s['positives']}  "
              f"false alarms {s['fp']}/{s['negatives']}  F1 {s['f1']}  MCC {s['mcc']}  "
              f"timecode median {s['timecode']['median_sec']}")
    print(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
