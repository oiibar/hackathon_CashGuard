from __future__ import annotations

import sys
from pathlib import Path

CLIP_DIR = Path(__file__).resolve().parent.parent / "models" / "clip"


def main() -> int:
    import clip

    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    clip.load("ViT-B/32", device="cpu", download_root=str(CLIP_DIR))
    print(f"ok   {CLIP_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
