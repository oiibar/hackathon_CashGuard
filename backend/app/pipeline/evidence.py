"""Evidence frames: draw zones and boxes, stamp the timecode, encode to base64 JPEG."""
from __future__ import annotations

import base64

import cv2
import numpy as np

from .detector import Detection

ZONE_COLORS = {
    "cash_zone": (0, 165, 255),    # orange (BGR)
    "client_zone": (255, 200, 0),  # cyan
}
HIT_COLOR = (40, 40, 230)          # red: object that triggered the rule
CTX_COLOR = (200, 200, 200)        # grey: other context objects


def fmt_tc(sec: float) -> str:
    m, s = divmod(max(sec, 0.0), 60)
    return f"{int(m):02d}:{s:05.2f}"


def render(frame: np.ndarray, t: float, title: str, zones_px: dict[str, np.ndarray],
           hits: list[Detection], context: list[Detection]) -> np.ndarray:
    img = frame.copy()
    overlay = img.copy()
    for name, poly in zones_px.items():
        color = ZONE_COLORS.get(name, (180, 180, 180))
        cv2.fillPoly(overlay, [poly], color)
        cv2.polylines(img, [poly], True, color, 2)
    img = cv2.addWeighted(overlay, 0.12, img, 0.88, 0)

    for det, color, thick in [(d, CTX_COLOR, 1) for d in context] + [(d, HIT_COLOR, 3) for d in hits]:
        x1, y1, x2, y2 = (int(v) for v in det.xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        label = f"{det.cls} {det.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, max(y1, th + 6)), color, -1)
        cv2.putText(img, label, (x1 + 2, max(y1 - 4, th + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255) if color == HIT_COLOR else (20, 20, 20), 1, cv2.LINE_AA)

    # Header bar. OpenCV fonts are ASCII-only, so the bar holds the rule type
    # code and timecode; the Russian text lives in the JSON description.
    bar = f"{fmt_tc(t)}  |  {title}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(img, bar, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def to_base64_jpeg(img: np.ndarray, max_width: int, quality: int) -> str:
    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(img, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")
