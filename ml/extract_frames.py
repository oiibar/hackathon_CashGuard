from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.pipeline.video import iter_frames, probe  # noqa: E402

VIDEO_EXT = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--every", type=float, default=1.0, help="seconds between frames")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--min-diff", type=float, default=4.0, help="skip frames this similar to the last kept one")
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    videos = [p for p in sorted(args.src.rglob("*")) if p.suffix.lower() in VIDEO_EXT]
    total = 0
    for video in videos:
        info = probe(video)
        last = None
        kept = 0
        for t, frame in iter_frames(video, 1.0 / args.every, args.width, info):
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32)
            if last is not None and float(np.abs(small - last).mean()) < args.min_diff:
                continue
            last = small
            cv2.imwrite(str(args.dst / f"{video.stem}_{t:07.1f}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            kept += 1
        print(f"{video.name}: {kept} frames")
        total += kept
    print(f"done: {total} frames in {args.dst}")


if __name__ == "__main__":
    main()
