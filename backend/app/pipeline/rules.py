"""Rule engine: turns per-frame YOLO detections into timed violations.

YOLO only says "there is a cell phone at (x, y)". Whether that is a
violation depends on WHERE (zone), for HOW LONG (min duration) and WHICH
norm it breaks. All of that lives in config/rules.yaml, so adding a new
violation for a new custom class needs no code change.

Rule kinds:
  object_in_zone        classes seen inside a zone for >= min_duration_sec
  person_count_exceeds  more than max_count persons inside a zone
  zone_unattended       watched objects present in watch_zone while
                        staff_zone has no person (cash left alone, cashier
                        left a waiting client)
  camera_obstructed     frame suddenly dark or blurred vs. the video's own
                        baseline (camera covered or turned away)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np
import yaml

from .detector import Detection

# ---------------------------------------------------------------- config


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def wanted_classes(rules_cfg: dict[str, Any]) -> set[str]:
    """Every class name any rule refers to (person is always needed)."""
    out = {"person"}
    for r in rules_cfg.get("rules", []):
        out.update(r.get("classes", []))
        out.update(r.get("watch_classes", []))
    return out


def build_description(rule: dict[str, Any], norms: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    """Russian description with the RK legal basis, plus structured links."""
    basis = []
    parts = []
    for ref in rule.get("norms", []):
        norm = norms[ref["ref"]]
        item = {
            "act": norm["act"],
            "short": norm.get("short", norm["act"]),
            "clause": ref.get("clause", ""),
            "summary": ref.get("summary", ""),
            "url": norm["url"],
        }
        basis.append(item)
        parts.append(f"{item['short']}, {item['clause']}: {item['summary']} ({item['url']})")
    text = rule["description"].strip()
    if parts:
        text += " Нормативное основание: " + "; ".join(parts) + "."
    return text, basis


# ----------------------------------------------------------------- zones


def zones_to_px(zones_norm: dict[str, list[list[float]]], w: int, h: int) -> dict[str, np.ndarray]:
    out = {}
    for name, pts in (zones_norm or {}).items():
        if not pts or len(pts) < 3:
            continue
        arr = np.array([[min(max(x, 0.0), 1.0) * w, min(max(y, 0.0), 1.0) * h] for x, y in pts], dtype=np.int32)
        out[name] = arr
    return out


def in_zone(det: Detection, poly: np.ndarray, anchor: str = "center") -> bool:
    pt = det.bottom if anchor == "bottom" else det.center
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0


# -------------------------------------------------------------- episodes


@dataclass
class Candidate:
    t: float
    frame: np.ndarray
    hits: list[Detection]
    context: list[Detection]
    score: float


@dataclass
class Episode:
    start: float
    last: float
    first: Candidate
    best: Candidate
    latest: Candidate
    max_conf: float = 0.0

    def add(self, c: Candidate, conf: float) -> None:
        self.last = c.t
        self.latest = c
        if c.score > self.best.score:
            self.best = c
        self.max_conf = max(self.max_conf, conf)


@dataclass
class RuleState:
    rule: dict[str, Any]
    current: Episode | None = None
    closed: list[Episode] = field(default_factory=list)

    @property
    def min_duration(self) -> float:
        return float(self.rule.get("min_duration_sec", 2.0))

    @property
    def gap_tolerance(self) -> float:
        return float(self.rule.get("gap_tolerance_sec", 1.5))

    def update(self, active: bool, cand: Candidate | None, conf: float, dt: float) -> None:
        tol = max(self.gap_tolerance, 1.5 * dt)  # sampling may get sparser under time pressure
        if active and cand is not None:
            if self.current and cand.t - self.current.last <= tol:
                self.current.add(cand, conf)
            else:
                self._close(dt)
                self.current = Episode(start=cand.t, last=cand.t, first=cand, best=cand, latest=cand, max_conf=conf)
        elif self.current and cand is not None and cand.t - self.current.last > tol:
            self._close(dt)

    def _close(self, dt: float) -> None:
        ep = self.current
        self.current = None
        if ep and (ep.last - ep.start + dt) >= self.min_duration:
            self.closed.append(ep)

    def finish(self, dt: float) -> list[Episode]:
        self._close(dt)
        return self.closed


# ---------------------------------------------------------------- engine


class RuleEngine:
    def __init__(self, rules_cfg: dict[str, Any], zones_norm: dict[str, list[list[float]]]):
        self.norms = rules_cfg.get("norms", {})
        self.zones_norm = zones_norm or {}
        self.states: list[RuleState] = []
        self.skipped: list[dict[str, str]] = []
        for rule in rules_cfg.get("rules", []):
            if not rule.get("enabled", True):
                continue
            missing = [z for z in self._zones_needed(rule) if z not in self.zones_norm]
            if missing:
                self.skipped.append({"type": rule["type"], "reason": f"zone not configured: {', '.join(missing)}"})
                continue
            self.states.append(RuleState(rule))
        self._zones_px: dict[str, np.ndarray] = {}
        self._frame_size: tuple[int, int] | None = None
        self._cam_ref: list[tuple[float, float]] = []

    @staticmethod
    def _zones_needed(rule: dict[str, Any]) -> list[str]:
        return [rule[k] for k in ("zone", "watch_zone", "staff_zone") if rule.get(k)]

    def disable_missing_classes(self, available: set[str]) -> None:
        """Skip rules whose classes no loaded model can detect (e.g. 'cash'
        before the custom model is trained)."""
        keep = []
        for st in self.states:
            need = set(st.rule.get("classes", [])) | set(st.rule.get("watch_classes", []))
            if need and not (need & available):
                self.skipped.append({"type": st.rule["type"], "reason": f"no model detects: {', '.join(sorted(need))}"})
            else:
                keep.append(st)
        self.states = keep

    @property
    def zones_px(self) -> dict[str, np.ndarray]:
        return self._zones_px

    def _ensure_zones(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if self._frame_size != (w, h):
            self._frame_size = (w, h)
            self._zones_px = zones_to_px(self.zones_norm, w, h)

    # -- per-frame ----------------------------------------------------------

    def step(self, t: float, frame: np.ndarray, dets: list[Detection], dt: float) -> None:
        self._ensure_zones(frame)
        for st in self.states:
            kind = st.rule["kind"]
            if kind == "object_in_zone":
                self._object_in_zone(st, t, frame, dets, dt)
            elif kind == "person_count_exceeds":
                self._person_count(st, t, frame, dets, dt)
            elif kind == "zone_unattended":
                self._unattended(st, t, frame, dets, dt)
            elif kind == "camera_obstructed":
                self._obstructed(st, t, frame, dt)

    def _members(self, dets: list[Detection], classes: list[str], zone: str, anchor: str,
                 min_conf: float) -> list[Detection]:
        poly = self._zones_px[zone]
        return [d for d in dets if d.cls in classes and d.conf >= min_conf and in_zone(d, poly, anchor)]

    def _object_in_zone(self, st, t, frame, dets, dt):
        r = st.rule
        hits = self._members(dets, r["classes"], r["zone"], r.get("anchor", "center"), r.get("min_conf", 0.0))
        active = len(hits) >= int(r.get("min_count", 1))
        ctx = [d for d in dets if d.cls == "person"]
        cand = Candidate(t, frame, hits, ctx, sum(d.conf for d in hits))
        st.update(active, cand, max((d.conf for d in hits), default=0.0), dt)

    def _person_count(self, st, t, frame, dets, dt):
        r = st.rule
        people = self._members(dets, ["person"], r["zone"], r.get("anchor", "center"), r.get("min_conf", 0.0))
        active = len(people) > int(r["max_count"])
        cand = Candidate(t, frame, people, [], float(len(people)))
        st.update(active, cand, max((d.conf for d in people), default=0.0), dt)

    def _unattended(self, st, t, frame, dets, dt):
        r = st.rule
        anchor = r.get("anchor", "center")
        watched = self._members(dets, r["watch_classes"], r["watch_zone"], anchor, r.get("min_conf", 0.0))
        staff = self._members(dets, ["person"], r["staff_zone"], anchor, r.get("staff_min_conf", 0.25))
        # A person who is both "watched" (client) and "staff" (zones overlap)
        # must not count as staff.
        staff = [s for s in staff if s not in watched]
        active = bool(watched) and not staff
        cand = Candidate(t, frame, watched, [d for d in dets if d.cls == "person" and d not in watched],
                         sum(d.conf for d in watched))
        st.update(active, cand, max((d.conf for d in watched), default=0.0), dt)

    def _obstructed(self, st, t, frame, dt):
        r = st.rule
        small = cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1])))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        bright = float(gray.mean())
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        warm = int(r.get("baseline_frames", 8))
        if len(self._cam_ref) < warm:
            self._cam_ref.append((bright, sharp))
            st.update(False, Candidate(t, frame, [], [], 0.0), 0.0, dt)
            return
        ref_b = median(b for b, _ in self._cam_ref)
        ref_s = median(s for _, s in self._cam_ref)
        dark = bright < max(ref_b * float(r.get("dark_ratio", 0.3)), 8.0)
        blurred = sharp < ref_s * float(r.get("blur_ratio", 0.1))
        active = dark or blurred
        if not active:  # slow drift of the baseline (lighting changes)
            self._cam_ref.append((bright, sharp))
            self._cam_ref = self._cam_ref[-50:]
        score = (1 - bright / max(ref_b, 1e-6)) + (1 - sharp / max(ref_s, 1e-6))
        st.update(active, Candidate(t, frame, [], [], score), min(max(score / 2, 0.0), 1.0), dt)

    # -- results ------------------------------------------------------------

    def finish(self, dt: float) -> list[tuple[dict[str, Any], Episode]]:
        out = []
        for st in self.states:
            for ep in st.finish(dt):
                out.append((st.rule, ep))
        out.sort(key=lambda x: x[1].start)
        return out
