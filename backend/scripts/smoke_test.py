from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--key", default="change-me")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--save", type=Path, default=None, help="write full JSON response here")
    args = ap.parse_args()

    headers = {"X-API-Key": args.key}
    t0 = time.time()
    with open(args.video, "rb") as f:
        data = {"profile": args.profile} if args.profile else None
        r = httpx.post(f"{args.url}/api/v1/analyze", headers=headers, data=data,
                       files={"file": (args.video.name, f, "video/quicktime")}, timeout=600)
    r.raise_for_status()
    job_id = r.json()["job_id"]
    print(f"job_id={job_id} (upload {time.time() - t0:.1f}s)")

    while True:
        body = httpx.get(f"{args.url}/api/v1/jobs/{job_id}", headers=headers, timeout=30).json()
        if body["status"] != "processing":
            break
        print(f"  processing {body.get('progress', 0) * 100:.0f}%", end="\r")
        time.sleep(2)

    elapsed = time.time() - t0
    print(f"\nstatus={body['status']} total={elapsed:.1f}s {'OK' if elapsed <= 300 else 'OVER 5 MIN'}")
    if body["status"] == "failed":
        print("error:", body.get("error"))
        return 1
    print("verdict:", body["verdict"])
    for v in body["violations"]:
        print(f"  {v['timecode_sec']:>8.2f}s  {v['type']:<34} frames={len(v['evidence_frames'])}")
    print("meta:", json.dumps(body.get("meta", {}), ensure_ascii=False))
    if args.save:
        args.save.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        print("saved", args.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
