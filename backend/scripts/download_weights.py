from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    names = sys.argv[1:] or ["yolov8n.pt"]
    MODELS_DIR.mkdir(exist_ok=True)
    for name in names:
        dest = MODELS_DIR / name
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"ok   {dest} (already present)")
            continue
        print(f"get  {BASE + name}")
        urllib.request.urlretrieve(BASE + name, dest)
        print(f"ok   {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
