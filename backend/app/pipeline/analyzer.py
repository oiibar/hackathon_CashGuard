"""One video in, one report out. Runs in a worker thread (blocking code).

Time budget logic (5-minute hackathon limit):
1. Deadline is fixed at upload time (queue wait counts).
2. Sampling FPS is chosen up front from video duration and the measured
   speed of the detector on this machine.
3. While running, if the projection says we will miss the deadline, frames
   are skipped (process 1 of k). If the deadline is still hit, analysis stops
   and the report says which share of the video was covered.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Callable

from .detector import Detector
from .evidence import render, to_base64_jpeg
from .rules import RuleEngine, build_description
from .video import iter_frames, probe

log = logging.getLogger("cashguard.analyzer")

FINALIZE_RESERVE_SEC = 8.0  # time kept for evidence rendering + JSON


class Analyzer:
    def __init__(self, detector: Detector, rules_cfg: dict[str, Any], settings):
        self.detector = detector
        self.rules_cfg = rules_cfg
        self.s = settings
        self.sec_per_frame = self._benchmark()

    def _benchmark(self) -> float:
        import numpy as np

        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        self.detector(frame)  # warm-up (lazy init, allocations)
        t0 = time.perf_counter()
        for _ in range(3):
            self.detector(frame)
        spf = (time.perf_counter() - t0) / 3
        log.info("detector speed: %.3f s/frame (%.1f fps)", spf, 1 / spf)
        return spf

    def run(self, video_path: Path, zones: dict[str, Any], deadline: float,
            progress: Callable[[float], None] | None = None) -> dict[str, Any]:
        started = time.time()
        info = probe(video_path)

        engine = RuleEngine(self.rules_cfg, zones)
        engine.disable_missing_classes(self.detector.class_names)

        # 1) choose sampling rate that fits the remaining time
        budget = max(deadline - time.time() - FINALIZE_RESERVE_SEC, 5.0)
        affordable_frames = 0.8 * budget / max(self.sec_per_frame, 1e-3)
        sample_fps = min(self.s.target_fps, affordable_frames / max(info.duration_sec, 0.1))
        sample_fps = max(sample_fps, self.s.min_fps)
        dt = 1.0 / sample_fps

        processed = 0
        skip_every = 1
        last_t = 0.0
        stopped_early = False
        window_start = time.time()
        expected_total = max(int(info.duration_sec * sample_fps), 1)

        for idx, (t, frame) in enumerate(iter_frames(video_path, sample_fps, self.s.frame_width, info)):
            if idx % skip_every:
                continue
            now = time.time()
            if now > deadline - FINALIZE_RESERVE_SEC:
                stopped_early = True
                log.warning("deadline reached at t=%.1fs of %.1fs", t, info.duration_sec)
                break

            dets = self.detector(frame)
            engine.step(t, frame, dets, dt * skip_every)
            processed += 1
            last_t = t

            # 2) adaptive skipping, both ways: skip more when behind schedule,
            #    go back to every frame when ahead. Speed is measured on the
            #    last window only (includes waiting for the decoder).
            if processed % 10 == 0:
                now = time.time()
                per_frame = (now - window_start) / 10
                window_start = now
                sampled_left = max(expected_total - idx, 0)
                time_left = deadline - FINALIZE_RESERVE_SEC - now
                needed = math.ceil(1.1 * per_frame * sampled_left / time_left) if time_left > 0 else 50
                new_skip = min(max(needed, 1), 50)
                if new_skip != skip_every:
                    log.info("schedule: processing 1 of %d sampled frames", new_skip)
                    skip_every = new_skip
                if progress:
                    progress(min(t / info.duration_sec, 0.99))

        violations = self._build_violations(engine, dt * skip_every)
        covered = min((last_t + dt) / info.duration_sec, 1.0) if processed else 0.0
        return {
            "verdict": "violation" if violations else "no_violation",
            "violations": violations,
            "meta": {
                "duration_sec": round(info.duration_sec, 2),
                "resolution": f"{info.width}x{info.height}",
                "sample_fps": round(processed / max(last_t + dt, dt), 3) if processed else 0.0,
                "frames_analyzed": processed,
                "coverage": round(covered, 3),
                "stopped_early": stopped_early,
                "processing_time_sec": round(time.time() - started, 2),
                "models": [Path(p).name for p in self.s.model_paths],
                "skipped_rules": engine.skipped,
                "zones": list(zones.keys()),
            },
        }

    def _build_violations(self, engine: RuleEngine, dt: float) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        per_type: dict[str, int] = {}
        norms = self.rules_cfg.get("norms", {})
        for rule, ep in engine.finish(dt):
            vtype = rule["type"]
            per_type[vtype] = per_type.get(vtype, 0) + 1
            if per_type[vtype] > self.s.max_violations_per_type:
                continue
            description, basis = build_description(rule, norms)
            frames = []
            seen = set()
            for cand in (ep.first, ep.best, ep.latest):
                key = round(cand.t, 2)
                if key in seen:
                    continue
                seen.add(key)
                img = render(cand.frame, cand.t, vtype, engine.zones_px, cand.hits, cand.context)
                frames.append({
                    "timecode_sec": key,
                    "image_base64": to_base64_jpeg(img, self.s.evidence_max_width, self.s.evidence_jpeg_quality),
                })
            out.append({
                "type": vtype,
                "timecode_sec": round(ep.start, 2),
                "end_sec": round(ep.last + dt, 2),
                "duration_sec": round(ep.last - ep.start + dt, 2),
                "confidence": round(ep.max_conf, 3),
                "severity": rule.get("severity", "medium"),
                "title": rule.get("title", vtype),
                "description": description,
                "legal_basis": basis,
                "evidence_frames": frames,
            })
        return out
