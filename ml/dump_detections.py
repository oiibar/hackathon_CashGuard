from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.config import settings  # noqa: E402
from app.pipeline.detector import Detector  # noqa: E402
from app.pipeline.rules import load_yaml, wanted_classes  # noqa: E402
from app.pipeline.scene import frame_stats  # noqa: E402
from app.pipeline.video import iter_frames, probe  # noqa: E402

VIDEO_EXT = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--fps", type=float, default=settings.target_fps)
    args = ap.parse_args()

    rules = load_yaml(settings.rules_path)
    wanted = wanted_classes(rules) | {"laptop", "tv", "book", "keyboard", "chair"}
    det = Detector(settings.model_paths, settings.device, settings.imgsz, settings.conf, wanted,
                   aliases=rules.get("class_aliases"), class_conf=rules.get("class_conf"))
    args.dst.mkdir(parents=True, exist_ok=True)
    for video in sorted(p for p in args.src.iterdir() if p.suffix.lower() in VIDEO_EXT):
        info = probe(video)
        frames = []
        for t, frame in iter_frames(video, args.fps, settings.frame_width, info):
            h, w = frame.shape[:2]
            b, s = frame_stats(frame)
            frames.append({
                "t": round(t, 3), "w": w, "h": h, "bright": round(b, 1), "sharp": round(s, 1),
                "dets": [[d.cls, round(d.conf, 3), *(round(v, 1) for v in d.xyxy)] for d in det(frame)],
            })
        out = args.dst / f"{video.stem}.json"
        out.write_text(json.dumps({"video": video.name, "duration": info.duration_sec, "frames": frames},
                                  ensure_ascii=False), encoding="utf-8")
        print(f"{video.name}: {len(frames)} frames -> {out}")


if __name__ == "__main__":
    main()
