from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.pipeline.rules import in_zone, zones_to_px  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--from", dest="t0", type=float, default=0)
    ap.add_argument("--to", dest="t1", type=float, default=1e9)
    ap.add_argument("--classes", default="cell phone,cash,laptop,tv,handbag")
    ap.add_argument("--cache", type=Path, default=ROOT / "ml" / "cache")
    args = ap.parse_args()
    small = set(args.classes.split(","))

    scene = pickle.loads((args.cache / f"{args.video}.pkl").read_bytes())
    print("zones:", {k: v for k, v in scene.zones.items() if k != "client_zone"}, scene.zone_info)
    w, h = scene.obs[0].size
    poly = zones_to_px(scene.zones, w, h)["cash_zone"]

    def box(d):
        x1, y1, x2, y2 = d.xyxy
        return f"({x1 / w:.2f},{y1 / h:.2f})-({x2 / w:.2f},{y2 / h:.2f})"

    for o in scene.obs:
        if not args.t0 <= o.t <= args.t1:
            continue
        people = [d for d in o.dets if d.cls == "person"]
        tags = []
        for d in people:
            tags.append(f"{'S' if in_zone(d, poly) else 'V'}{d.conf:.2f}{box(d)}")
        objs = []
        for d in o.dets:
            if d.cls in small:
                cx, cy = d.center
                owner = "".join(str(i) for i, p in enumerate(people)
                                if p.xyxy[0] <= cx <= p.xyxy[2] and p.xyxy[1] <= cy <= p.xyxy[3])
                objs.append(f"{d.cls}:{d.conf:.2f}@{cx / w:.2f},{cy / h:.2f}[{owner or '-'}]")
        print(f"{o.t:6.2f} br{o.brightness:4.0f} | " + " ".join(f"{i}:{t}" for i, t in enumerate(tags))
              + " || " + " ".join(objs))


if __name__ == "__main__":
    main()
