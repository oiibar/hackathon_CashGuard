"""Video decoding through a local ffmpeg process.

Why ffmpeg and not cv2.VideoCapture for reading:
- .mov from iPhones is often HEVC with a rotation tag; ffmpeg handles both.
- `fps=` and `scale=` filters run inside ffmpeg with its own threads, so a
  4K 60 fps clip is reduced to a few small frames per second before Python
  sees it. This is the main trick that keeps a job under the 5-minute limit.

The binary comes from `imageio-ffmpeg` (bundled, works offline), or from
the system PATH, or from the FFMPEG_BINARY env var.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def ffmpeg_exe() -> str:
    env = os.getenv("FFMPEG_BINARY")
    if env:
        return env
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - fallback path
        found = shutil.which("ffmpeg")
        if not found:
            raise RuntimeError("ffmpeg not found: pip install imageio-ffmpeg or install ffmpeg")
        return found


@dataclass
class VideoInfo:
    duration_sec: float
    fps: float
    width: int
    height: int


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_STREAM_RE = re.compile(r"Video:.*?(\d{2,5})x(\d{2,5})")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*fps")
_ROTATE_RE = re.compile(r"(?:rotate\s*:\s*|rotation of\s*)(-?\d+(?:\.\d+)?)")


def probe(path: Path) -> VideoInfo:
    """Read duration, fps and display size. ffmpeg first, OpenCV as fallback."""
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        errors="replace",
    )
    err = proc.stderr
    duration = fps = 0.0
    width = height = 0
    if m := _DURATION_RE.search(err):
        h, mnt, s = m.groups()
        duration = int(h) * 3600 + int(mnt) * 60 + float(s)
    if m := _STREAM_RE.search(err):
        width, height = int(m.group(1)), int(m.group(2))
    if m := _FPS_RE.search(err):
        fps = float(m.group(1))
    if m := _ROTATE_RE.search(err):
        if abs(float(m.group(1))) % 180 == 90:
            width, height = height, width

    if duration <= 0 or width == 0:
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise ValueError("Cannot open video: unsupported or corrupted file")
            fps = fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            duration = duration or (frames / fps if fps else 0.0)
            width = width or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = height or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()

    if duration <= 0 or width == 0 or height == 0:
        raise ValueError("Cannot read video stream (no duration or frame size)")
    return VideoInfo(duration_sec=duration, fps=fps or 25.0, width=width, height=height)


def iter_frames(path: Path, sample_fps: float, out_width: int, info: VideoInfo) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timecode_sec, BGR frame) at `sample_fps`, scaled to `out_width`."""
    out_w = min(out_width, info.width)
    out_w -= out_w % 2
    out_h = int(round(info.height * out_w / info.width))
    out_h -= out_h % 2
    frame_bytes = out_w * out_h * 3

    cmd = [
        ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", str(path),
        "-an", "-sn",
        "-vf", f"fps={sample_fps:.4f},scale={out_w}:{out_h}",
        "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
    ]
    # stderr goes to a temp file: a PIPE could fill up on a damaged file and
    # deadlock the reader.
    with tempfile.TemporaryFile() as err_file:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err_file, bufsize=frame_bytes * 4)
        try:
            idx = 0
            assert proc.stdout is not None
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(out_h, out_w, 3)
                yield idx / sample_fps, frame
                idx += 1
            if idx == 0:
                proc.wait()
                err_file.seek(0)
                err = err_file.read().decode(errors="replace")
                raise ValueError(f"ffmpeg produced no frames: {err.strip()[:300]}")
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
