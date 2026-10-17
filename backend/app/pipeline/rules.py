from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np
import yaml

from .detector import Detection
from .scene import Observation


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def wanted_classes(rules_cfg: dict[str, Any]) -> set[str]:
    out = {"person"}
    for r in rules_cfg.get("rules", []):
        out.update(r.get("classes", []))
        out.update(r.get("watch_classes", []))
    return out


def fmt_tc(sec: float) -> str:
    m, s = divmod(max(sec, 0.0), 60)
    return f"{int(m):02d}:{s:04.1f}"


def legal_basis(rule: dict[str, Any], norms: dict[str, Any]) -> list[dict[str, str]]:
    out = []
    for ref in rule.get("norms", []):
        norm = norms[ref["ref"]]
        out.append({
            "act": " ".join(norm["act"].split()),
            "short": norm.get("short", norm["act"]),
            "clause": ref.get("clause", ""),
            "quote": " ".join(ref.get("quote", "").split()),
            "url": norm["url"],
        })
    return out


def observed_text(rule: dict[str, Any], facts: dict[str, Any] | None = None) -> str:
    text = rule.get("observed", rule.get("title", rule["type"]))
    try:
        return text.format(**(facts or {}))
    except (KeyError, IndexError, ValueError):
        return text


def build_description(rule: dict[str, Any], norms: dict[str, Any],
                      facts: dict[str, Any] | None = None) -> tuple[str, list[dict[str, str]]]:
    facts = facts or {}
    basis = legal_basis(rule, norms)
    observed = observed_text(rule, facts)
    parts = []
    if "start" in facts:
        parts.append(f"{fmt_tc(facts['start'])}–{fmt_tc(facts['end'])}: {observed}.")
    else:
        parts.append(f"{observed[:1].upper()}{observed[1:]}.")
    if rule.get("why"):
        parts.append(" ".join(rule["why"].split()))
    if basis:
        refs = "; ".join(f"{b['short']}, {b['clause']}: «{b['quote']}»" for b in basis)
        parts.append(f"Основание: {refs}.")
    parts.append("Требуется проверка человеком.")
    return " ".join(parts), basis


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


def _inside(obj: Detection, box: Detection, grow: float) -> bool:
    x1, y1, x2, y2 = box.xyxy
    gw, gh = grow * (x2 - x1), grow * (y2 - y1)
    cx, cy = obj.center
    return x1 - gw <= cx <= x2 + gw and y1 - gh <= cy <= y2 + gh


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Candidate:
    t: float
    obs: Observation
    hits: list[Detection]
    context: list[Detection]
    score: float
    value: float = 0.0


@dataclass
class Episode:
    start: float
    last: float
    first: Candidate
    best: Candidate
    latest: Candidate
    max_conf: float = 0.0
    peak_value: float = 0.0

    def add(self, c: Candidate, conf: float) -> None:
        self.last = c.t
        self.latest = c
        if c.score > self.best.score:
            self.best = c
        self.max_conf = max(self.max_conf, conf)
        self.peak_value = max(self.peak_value, c.value)


@dataclass
class RuleState:
    rule: dict[str, Any]
    current: Episode | None = None
    closed: list[Episode] = field(default_factory=list)
    memo: Any = None

    @property
    def min_duration(self) -> float:
        return float(self.rule.get("min_duration_sec", 2.0))

    @property
    def gap_tolerance(self) -> float:
        return float(self.rule.get("gap_tolerance_sec", 1.5))

    def update(self, active: bool, cand: Candidate | None, conf: float, dt: float) -> None:
        tol = max(self.gap_tolerance, 1.5 * dt)
        if active and cand is not None:
            if self.current and cand.t - self.current.last <= tol:
                self.current.add(cand, conf)
            else:
                self._close(dt)
                self.current = Episode(start=cand.t, last=cand.t, first=cand, best=cand, latest=cand,
                                       max_conf=conf, peak_value=cand.value)
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


class RuleEngine:
    def __init__(self, rules_cfg: dict[str, Any], zones_norm: dict[str, list[list[float]]],
                 localized: bool = True):
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
            if rule.get("requires_localized_zone") and not localized:
                self.skipped.append({"type": rule["type"], "reason": "cashier workplace not localized in this video"})
                continue
            self.states.append(RuleState(rule))
        self._zones_px: dict[str, np.ndarray] = {}
        self._frame_size: tuple[int, int] | None = None
        self._cam_ref: list[tuple[float, float]] = []

    @staticmethod
    def _zones_needed(rule: dict[str, Any]) -> list[str]:
        return [rule[k] for k in ("zone", "watch_zone", "staff_zone") if rule.get(k)]

    def disable_missing_classes(self, available: set[str]) -> None:
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

    def _ensure_zones(self, size: tuple[int, int]) -> None:
        if self._frame_size != size:
            self._frame_size = size
            self._zones_px = zones_to_px(self.zones_norm, *size)


    def step(self, obs: Observation, dt: float) -> None:
        self._ensure_zones(obs.size)
        for st in self.states:
            kind = st.rule["kind"]
            if kind == "object_in_zone":
                self._object_in_zone(st, obs, dt)
            elif kind == "person_count_exceeds":
                self._person_count(st, obs, dt)
            elif kind == "zone_unattended":
                self._unattended(st, obs, dt)
            elif kind == "camera_obstructed":
                self._obstructed(st, obs, dt)
            elif kind == "held_by_staff":
                self._held_by_staff(st, obs, dt)
            elif kind == "visitor_count_exceeds":
                self._visitors(st, obs, dt)
            elif kind == "static_object":
                self._static_object(st, obs, dt)

    def _split_people(self, dets: list[Detection], staff_zone: str,
                      min_conf: float = 0.35) -> tuple[list[Detection], list[Detection]]:
        poly = self._zones_px[staff_zone]
        people = [d for d in dets if d.cls == "person" and d.conf >= min_conf]
        staff = [d for d in people if in_zone(d, poly)]
        return staff, [d for d in people if d not in staff]

    def _held_by_staff(self, st, obs, dt):
        r = st.rule
        staff, visitors = self._split_people(obs.dets, r["staff_zone"], r.get("person_min_conf", 0.35))
        grow = float(r.get("box_expand", 0.1))
        hits = [d for d in obs.dets if d.cls in r["classes"] and d.conf >= r.get("min_conf", 0.0)
                and any(_inside(d, p, grow) for p in staff) and not any(_inside(d, v, 0.0) for v in visitors)]
        cand = Candidate(obs.t, obs, hits, staff + visitors, sum(d.conf for d in hits), float(len(hits)))
        st.update(bool(hits), cand, max((d.conf for d in hits), default=0.0), dt)

    def _visitors(self, st, obs, dt):
        r = st.rule
        staff, visitors = self._split_people(obs.dets, r["staff_zone"], r.get("min_conf", 0.45))
        if r.get("near_desk"):
            top = float(self._zones_px[r["staff_zone"]][:, 1].min())
            visitors = [v for v in visitors if v.xyxy[3] >= top]
        active = len(visitors) > int(r["max_count"]) and (bool(staff) or not r.get("require_staff", True))
        cand = Candidate(obs.t, obs, visitors, staff, float(len(visitors)), float(len(visitors)))
        st.update(active, cand, max((d.conf for d in visitors), default=0.0), dt)

    def _static_object(self, st, obs, dt):
        r = st.rule
        people = [d for d in obs.dets if d.cls == "person"]
        grow = float(r.get("hand_margin", 0.0))
        cands = [d for d in self._members(obs.dets, r["classes"], r["zone"], "center", r.get("min_conf", 0.0))
                 if not any(_inside(d, p, grow) for p in people)]
        if r.get("require_staff_absent"):
            staff, _ = self._split_people(obs.dets, r.get("staff_zone", "cash_zone"), 0.25)
            if staff:
                cands = []
        memo = st.memo
        hit = None
        if memo:
            near = [d for d in cands if _iou(d.xyxy, memo[0]) >= 0.2]
            if near:
                hit = max(near, key=lambda d: d.conf)
                st.memo = (hit.xyxy, memo[1], obs.t)
            elif obs.t - memo[2] > float(r.get("gap_tolerance_sec", 1.5)):
                st.memo = None
        if st.memo is None and cands:
            hit = max(cands, key=lambda d: d.conf)
            st.memo = (hit.xyxy, obs.t, obs.t)
        settled = bool(st.memo) and hit is not None and obs.t - st.memo[1] >= float(r.get("settle_sec", 2.0))
        hits = [hit] if settled else []
        cand = Candidate(obs.t, obs, hits, people, hit.conf if hit else 0.0, float(len(hits)))
        st.update(settled, cand, hit.conf if settled else 0.0, dt)

    def _members(self, dets: list[Detection], classes: list[str], zone: str, anchor: str,
                 min_conf: float) -> list[Detection]:
        poly = self._zones_px[zone]
        return [d for d in dets if d.cls in classes and d.conf >= min_conf and in_zone(d, poly, anchor)]

    def _object_in_zone(self, st, obs, dt):
        r = st.rule
        hits = self._members(obs.dets, r["classes"], r["zone"], r.get("anchor", "center"), r.get("min_conf", 0.0))
        active = len(hits) >= int(r.get("min_count", 1))
        ctx = [d for d in obs.dets if d.cls == "person"]
        cand = Candidate(obs.t, obs, hits, ctx, sum(d.conf for d in hits), float(len(hits)))
        st.update(active, cand, max((d.conf for d in hits), default=0.0), dt)

    def _person_count(self, st, obs, dt):
        r = st.rule
        people = self._members(obs.dets, ["person"], r["zone"], r.get("anchor", "center"), r.get("min_conf", 0.0))
        active = len(people) > int(r["max_count"])
        ctx = [d for d in obs.dets if d.cls == "person" and d not in people]
        cand = Candidate(obs.t, obs, people, ctx, float(len(people)), float(len(people)))
        st.update(active, cand, max((d.conf for d in people), default=0.0), dt)

    def _unattended(self, st, obs, dt):
        r = st.rule
        anchor = r.get("anchor", "center")
        watched = self._members(obs.dets, r["watch_classes"], r["watch_zone"], anchor, r.get("min_conf", 0.0))
        staff_poly = self._zones_px[r["staff_zone"]]
        if r.get("watch_outside_staff"):
            watched = [d for d in watched if not in_zone(d, staff_poly, anchor)]
        staff = self._members(obs.dets, ["person"], r["staff_zone"], anchor, r.get("staff_min_conf", 0.25))
        staff = [s for s in staff if s not in watched]
        if r.get("require_prior_client"):
            if staff:
                st.memo = [d.xyxy for d in watched]
            watched = [d for d in watched if any(_iou(d.xyxy, b) >= 0.3 for b in (st.memo or []))]
        active = bool(watched) and not staff
        cand = Candidate(obs.t, obs, watched, [d for d in obs.dets if d.cls == "person" and d not in watched],
                         sum(d.conf for d in watched), float(len(watched)))
        st.update(active, cand, max((d.conf for d in watched), default=0.0), dt)

    def _obstructed(self, st, obs, dt):
        r = st.rule
        bright, sharp = obs.brightness, obs.sharpness
        warm = int(r.get("baseline_frames", 8))
        if len(self._cam_ref) < warm:
            self._cam_ref.append((bright, sharp))
            st.update(False, Candidate(obs.t, obs, [], [], 0.0), 0.0, dt)
            return
        ref_b = median(b for b, _ in self._cam_ref)
        ref_s = median(s for _, s in self._cam_ref)
        dark = bright < max(ref_b * float(r.get("dark_ratio", 0.3)), 8.0)
        blurred = sharp < ref_s * float(r.get("blur_ratio", 0.1))
        active = dark or blurred
        if not active:
            self._cam_ref.append((bright, sharp))
            self._cam_ref = self._cam_ref[-50:]
        score = (1 - bright / max(ref_b, 1e-6)) + (1 - sharp / max(ref_s, 1e-6))
        st.update(active, Candidate(obs.t, obs, [], [], score), min(max(score / 2, 0.0), 1.0), dt)


    def finish(self, dt: float) -> list[tuple[dict[str, Any], Episode]]:
        out = []
        for st in self.states:
            for ep in st.finish(dt):
                out.append((st.rule, ep))
        out.sort(key=lambda x: x[1].start)
        return out
