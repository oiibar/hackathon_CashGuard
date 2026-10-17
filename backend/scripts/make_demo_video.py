"""Build a synthetic demo .mov from the images bundled with Ultralytics.

Timeline (30 s, 1280x720):
   0-8 s   two people            -> no violation (default profile allows 2)
   8-16 s  four+ people          -> unauthorized_person_in_cash_zone
  16-21 s  black frames          -> camera_obstructed
  21-30 s  two people            -> no violation

Use it to test the pipeline end to end before you have real footage:
    python scripts/make_demo_video.py data/demo.mov
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipeline.video import ffmpeg_exe  # noqa: E402

W, H, FPS = 1280, 720, 25


def letterbox(img: np.ndarray) -> np.ndarray:
    scale = min(W / img.shape[1], H / img.shape[0])
    resized = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    canvas = np.full((H, W, 3), 40, dtype=np.uint8)
    y, x = (H - resized.shape[0]) // 2, (W - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/demo.mov")
    out.parent.mkdir(parents=True, exist_ok=True)
    import ultralytics

    assets = Path(ultralytics.__file__).parent / "assets"
    two = letterbox(cv2.imread(str(assets / "zidane.jpg")))
    many = letterbox(cv2.imread(str(assets / "bus.jpg")))
    black = np.zeros((H, W, 3), dtype=np.uint8)
    timeline = [(two, 8), (many, 8), (black, 5), (two, 9)]

    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    rng = np.random.default_rng(0)
    for img, seconds in timeline:
        for _ in range(seconds * FPS):
            noise = rng.integers(-6, 7, img.shape, dtype=np.int16)  # sensor-like noise
            frame = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
