from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .detector import Detection, Detector
from .evidence import render, to_base64_jpeg
from .rules import RuleEngine, build_description, observed_text
from .scene import (Observation, dedupe_people, drop_background_people, drop_fixtures, frame_stats,
                    infer_zones, static_fixtures, zone_bbox)
from . import learned
from .video import VideoInfo, iter_frames, probe

log = logging.getLogger("cashguard.analyzer")

FINALIZE_RESERVE_SEC = 12.0
ZOOM_SHARE = 0.45


@dataclass
class Scene:
    info: VideoInfo
    obs: list[Observation]
    zones: dict[str, Any]
    localized: bool
    zone_info: dict[str, Any]
    stopped_early: bool = False
    fixtures: dict[str, int] = field(default_factory=dict)
    zoom_frames: int = 0
    emb_t: Any = None
    emb_full: Any = None
    emb_desk: Any = None


class Analyzer:
    def __init__(self, detector: Detector, rules_cfg: dict[str, Any], settings):
        self.detector = detector
        self.rules_cfg = rules_cfg
        self.s = settings
        self.zoom_classes = set(rules_cfg.get("zoom_classes", [])) & detector.class_names
        self.sec_per_frame = self._benchmark()
        self.learned = learned.Model.load() if getattr(settings, "learned_enabled", True) else None
        self.embedder = None
        if self.learned is not None:
            from .embedder import Embedder, available
            if available():
                self.embedder = Embedder(settings.device or "cpu")
            else:
                log.warning("learned scorer disabled: CLIP weights not found (scripts/download_clip.py)")
                self.learned = None

    def _benchmark(self) -> float:
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        self.detector(frame)
        t0 = time.perf_counter()
        for _ in range(3):
            self.detector(frame)
        spf = (time.perf_counter() - t0) / 3
        log.info("detector speed: %.3f s/frame (%.1f fps)", spf, 1 / spf)
        return spf


    def detect(self, video_path: Path, deadline: float,
               progress: Callable[[float], None] | None = None) -> tuple[VideoInfo, list[Observation], bool]:
        info = probe(video_path)
        budget = max(deadline - time.time() - FINALIZE_RESERVE_SEC, 5.0)
        share = (1 - ZOOM_SHARE) if self.zoom_classes else 1.0
        affordable_frames = 0.85 * share * budget / max(self.sec_per_frame, 1e-3)
        sample_fps = min(self.s.target_fps, affordable_frames / max(info.duration_sec, 0.1))
        sample_fps = max(sample_fps, self.s.min_fps)
        stage_deadline = time.time() + share * budget

        obs: list[Observation] = []
        skip_every = 1
        stopped_early = False
        window_start = time.time()
        expected_total = max(int(info.duration_sec * sample_fps), 1)
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), self.s.store_jpeg_quality]

        for idx, (t, frame) in enumerate(iter_frames(video_path, sample_fps, self.s.frame_width, info)):
            if idx % skip_every:
                continue
            if time.time() > deadline - FINALIZE_RESERVE_SEC:
                stopped_early = True
                log.warning("deadline reached at t=%.1fs of %.1fs", t, info.duration_sec)
                break

            dets = self.detector(frame)
            bright, sharp = frame_stats(frame)
            ok, buf = cv2.imencode(".jpg", frame, enc)
            obs.append(Observation(t, dets, buf.tobytes() if ok else b"", bright, sharp,
                                   (frame.shape[1], frame.shape[0])))

            if len(obs) % 10 == 0:
                now = time.time()
                per_frame = (now - window_start) / 10
                window_start = now
                sampled_left = max(expected_total - idx, 0)
                time_left = stage_deadline - now
                needed = math.ceil(1.1 * per_frame * sampled_left / time_left) if time_left > 0 else 50
                new_skip = min(max(needed, 1), 50)
                if new_skip != skip_every:
                    log.info("schedule: processing 1 of %d sampled frames", new_skip)
                    skip_every = new_skip
                if progress:
                    progress(min(0.7 * t / info.duration_sec, 0.7))
        return info, obs, stopped_early


    def prepare(self, info: VideoInfo, obs: list[Observation], zones: dict[str, Any] | None, deadline: float,
                stopped_early: bool = False, progress: Callable[[float], None] | None = None) -> Scene:
        drop_background_people(obs)
        dedupe_people(obs)

        if zones:
            zone_info: dict[str, Any] = {"method": "manual"}
            localized = True
            if "desk_zone" not in zones and "cash_zone" in zones:
                zones = {**zones, "desk_zone": zones["cash_zone"]}
        else:
            inferred = infer_zones(obs)
            zones, localized, zone_info = inferred.zones, inferred.localized, inferred.info
            zone_info.setdefault("method", "auto")

        zoomed = self._zoom_pass(obs, zones, localized, deadline, progress)
        scene = Scene(info, obs, zones, localized, zone_info, stopped_early, zoom_frames=zoomed)
        self.embed(scene, deadline)
        return scene

    def embed(self, scene: Scene, deadline: float) -> None:
        if self.embedder is None or not scene.obs:
            return
        time_left = deadline - FINALIZE_RESERVE_SEC - time.time()
        per_moment = 2 * 0.06
        step = learned.STEP_SEC
        span = scene.obs[-1].t - scene.obs[0].t
        if time_left < (span / step + 1) * per_moment:
            step = max(span * per_moment / max(time_left, 1.0), step)
            log.info("learned scorer: embedding every %.1f s to fit the deadline", step)
        if time_left <= 2:
            return
        idx = learned.sample_indices(scene.obs, step)
        scene.emb_full, scene.emb_desk = learned.embed_moments(
            self.embedder, scene.obs, idx, scene.zones if scene.localized else None)
        scene.emb_t = np.array([scene.obs[i].t for i in idx], np.float32)

    def _zoom_pass(self, obs: list[Observation], zones: dict[str, Any], localized: bool, deadline: float,
                   progress: Callable[[float], None] | None) -> int:
        if not (self.zoom_classes and localized and obs and "desk_zone" in zones):
            return 0
        x1, y1, x2, y2 = zone_bbox(zones["desk_zone"])
        time_left = deadline - FINALIZE_RESERVE_SEC - time.time()
        affordable = int(0.9 * time_left / max(self.sec_per_frame, 1e-3))
        if affordable < 1:
            return 0
        step = max(math.ceil(len(obs) / affordable), 1)
        done = 0
        for i, o in enumerate(obs):
            o.dets = [d for d in o.dets if d.cls not in self.zoom_classes]
            if i % step:
                continue
            frame = o.frame()
            h, w = frame.shape[:2]
            ox, oy = int(x1 * w), int(y1 * h)
            crop = frame[oy:int(y2 * h), ox:int(x2 * w)]
            if crop.size == 0:
                continue
            for d in self.detector(crop, only=self.zoom_classes):
                a, b, c, e = d.xyxy
                o.dets.append(Detection(d.cls, d.conf, (a + ox, b + oy, c + ox, e + oy)))
            done += 1
            if progress and done % 10 == 0:
                progress(0.7 + 0.25 * i / len(obs))
        return done


    def judge(self, scene: Scene, started: float) -> dict[str, Any]:
        obs = scene.obs
        fixtures = static_fixtures(obs, set(self.rules_cfg.get("fixture_classes", [])))
        drop_fixtures(obs, fixtures)
        scene.fixtures = {k: len(v) for k, v in fixtures.items()}
        dt = (obs[-1].t - obs[0].t) / (len(obs) - 1) if len(obs) > 1 else 1.0 / max(self.s.target_fps, 0.1)
        engine = RuleEngine(self.rules_cfg, scene.zones, localized=scene.localized)
        engine.disable_missing_classes(self.detector.class_names)
        for o in obs:
            engine.step(o, dt)

        violations = self._build_violations(engine, dt)
        violations = self._merge_learned(scene, violations, engine.zones_px)
        info = scene.info
        last_t = obs[-1].t if obs else 0.0
        covered = min((last_t + dt) / info.duration_sec, 1.0) if obs else 0.0
        return {
            "verdict": "violation" if violations else "no_violation",
            "violations": violations,
            "meta": {
                "duration_sec": round(info.duration_sec, 2),
                "resolution": f"{info.width}x{info.height}",
                "sample_fps": round(1 / dt, 3) if obs else 0.0,
                "frames_analyzed": len(obs),
                "zoom_frames": scene.zoom_frames,
                "learned_scorer": self.learned is not None and scene.emb_t is not None,
                "coverage": round(covered, 3),
                "stopped_early": scene.stopped_early,
                "processing_time_sec": round(time.time() - started, 2),
                "models": self.detector.model_names,
                "skipped_rules": engine.skipped,
                "zones": scene.zones,
                "zone_detection": {**scene.zone_info, "localized": scene.localized},
                "ignored_fixtures": scene.fixtures,
            },
        }

    def run(self, video_path: Path, zones: dict[str, Any] | None, deadline: float,
            progress: Callable[[float], None] | None = None, embeddings_out: Path | None = None) -> dict[str, Any]:
        started = time.time()
        info, obs, stopped = self.detect(video_path, deadline, progress)
        scene = self.prepare(info, obs, zones, deadline, stopped, progress)
        if embeddings_out is not None and scene.emb_t is not None and len(scene.emb_t):
            try:
                learned.save_embeddings(embeddings_out, scene.emb_t, scene.emb_full, scene.emb_desk)
            except OSError:
                log.exception("cannot save embeddings to %s", embeddings_out)
        report = self.judge(scene, started)
        report["meta"]["model_version"] = self.learned.meta.get("version") if self.learned else None
        return report

    def _merge_learned(self, scene: Scene, violations: list[dict[str, Any]],
                       zones_px: dict[str, Any]) -> list[dict[str, Any]]:
        if self.learned is None or scene.emb_t is None or len(scene.emb_t) < 3:
            return violations
        defs = self.rules_cfg.get("learned_types", {})
        by_type = {r["type"]: r for r in self.rules_cfg.get("rules", [])}
        norms = self.rules_cfg.get("norms", {})
        feats = learned.features(scene.emb_t, scene.emb_full, scene.emb_desk)
        out = list(violations)
        times = [o.t for o in scene.obs]
        for ep in learned.episodes(self.learned, scene.emb_t, feats):
            if any(abs(v["timecode_sec"] - ep.start) <= 4 for v in out):
                continue
            rule = {"type": ep.vtype, **by_type.get(ep.vtype, {}), **defs.get(ep.vtype, {})}
            rule["type"] = ep.vtype
            facts = {"start": ep.start, "end": ep.end, "duration": f"{ep.end - ep.start:.1f}", "count": 0}
            description, basis = build_description(rule, norms, facts)
            frames, seen = [], set()
            for t in (ep.start, ep.peak, ep.end - learned.STEP_SEC):
                i = int(np.argmin(np.abs(np.array(times) - t)))
                if i in seen:
                    continue
                seen.add(i)
                o = scene.obs[i]
                img = render(o.frame(), o.t, ep.vtype, zones_px, [], [d for d in o.dets if d.cls == "person"])
                frames.append({"timestamp_sec": round(o.t, 2),
                               "image_base64": to_base64_jpeg(img, self.s.evidence_max_width, self.s.evidence_jpeg_quality)})
            out.append({
                "type": rule.get("title", ep.vtype),
                "type_id": ep.vtype,
                "timecode_sec": round(ep.start, 2),
                "end_sec": round(ep.end, 2),
                "duration_sec": round(ep.end - ep.start, 2),
                "confidence": round(ep.score, 3),
                "severity": rule.get("severity", "medium"),
                "source": "learned",
                "description": description,
                "observed": observed_text(rule, facts),
                "why": " ".join((rule.get("why") or "").split()),
                "legal_basis": basis,
                "evidence_frames": frames,
            })
        out.sort(key=lambda v: v["timecode_sec"])
        return out

    def _build_violations(self, engine: RuleEngine, dt: float) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        per_type: dict[str, int] = {}
        norms = self.rules_cfg.get("norms", {})
        for rule, ep in engine.finish(dt):
            vtype = rule["type"]
            per_type[vtype] = per_type.get(vtype, 0) + 1
            if per_type[vtype] > self.s.max_violations_per_type:
                continue
            end = ep.last + dt
            facts = {"start": ep.start, "end": end, "duration": f"{end - ep.start:.1f}",
                     "count": int(ep.peak_value)}
            description, basis = build_description(rule, norms, facts)
            frames = []
            seen = set()
            for cand in (ep.first, ep.best, ep.latest):
                key = round(cand.t, 2)
                if key in seen:
                    continue
                seen.add(key)
                img = render(cand.obs.frame(), cand.t, vtype, engine.zones_px, cand.hits, cand.context)
                frames.append({
                    "timestamp_sec": key,
                    "image_base64": to_base64_jpeg(img, self.s.evidence_max_width, self.s.evidence_jpeg_quality),
                })
            out.append({
                "type": rule.get("title", vtype),
                "type_id": vtype,
                "timecode_sec": round(ep.start, 2),
                "end_sec": round(end, 2),
                "duration_sec": round(end - ep.start, 2),
                "confidence": round(ep.max_conf, 3),
                "severity": rule.get("severity", "medium"),
                "source": "rule",
                "description": description,
                "observed": observed_text(rule, facts),
                "why": " ".join((rule.get("why") or "").split()),
                "legal_basis": basis,
                "evidence_frames": frames,
            })
        return out
