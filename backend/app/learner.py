from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import settings
from .pipeline.learned import MODEL_PATH, Model

log = logging.getLogger("cashguard.learner")

BASE_SET = MODEL_PATH.parent / "base_set.npz"
MCC_TOLERANCE = 0.03


def embeddings_path(job_id: str) -> Path:
    return settings.data_dir / "embeddings" / f"{job_id}.npz"


class Learner:
    def __init__(self) -> None:
        self.store = None
        self.analyzers: list[Any] = []
        self.base_model: Model | None = None
        self.model: Model | None = None
        self.version = 0
        self.state: dict[str, Any] = {"running": False}
        self._lock = threading.Lock()
        self._task: asyncio.Task | None = None

    def start(self, store: Any, analyzers: list[Any]) -> None:
        self.store = store
        self.analyzers = analyzers
        self.base_model = Model.load(MODEL_PATH)
        self.model, self.version = self.base_model, 0
        active = next((v for v in store.versions() if v["status"] == "active"), None)
        if active:
            model = Model.load(Path(active["path"]))
            if model is not None:
                self.model, self.version = model, active["id"]
            else:
                log.warning("model v%s is missing at %s, using the base model", active["id"], active["path"])
        self._install(self.model)

    def _install(self, model: Model | None) -> None:
        if model is not None:
            model.meta["version"] = self.version
        for a in self.analyzers:
            if getattr(a, "embedder", None) is not None and model is not None:
                a.learned = model

    @property
    def enabled(self) -> bool:
        return settings.learning_enabled and BASE_SET.exists()

    def pending(self) -> list[dict[str, Any]]:
        return [r for r in self.store.reviews()
                if r["model_version"] is None and embeddings_path(r["job_id"]).exists()]

    def review(self, job: Any, label: str, timecode: float | None, type_id: str | None) -> dict[str, Any]:
        review = {"job_id": job.id, "label": label, "timecode_sec": timecode if label == "violation" else None,
                  "type_id": type_id if label == "violation" else None}
        self.store.save_review(review)
        trainable = embeddings_path(job.id).exists()
        return {**review, "trainable": trainable, "pending": len(self.pending())}

    def should_retrain(self) -> bool:
        return (self.enabled and settings.auto_retrain and not self.state["running"]
                and len(self.pending()) >= settings.retrain_min_reviews)

    def schedule(self, trigger: str, busy: Callable[[], bool]) -> bool:
        if self.state["running"] or not self.enabled:
            return False
        self.state = {"running": True, "trigger": trigger, "started_at": time.time()}
        self._task = asyncio.create_task(self._run(trigger, busy))
        return True

    async def _run(self, trigger: str, busy: Callable[[], bool]) -> None:
        try:
            while busy():
                await asyncio.sleep(5)
            result = await asyncio.to_thread(self.retrain, trigger)
            self.state = {"running": False, "last": result}
        except Exception as exc:
            log.exception("retraining failed")
            self.state = {"running": False, "last": {"error": str(exc)}}

    def _videos(self) -> tuple[list, list]:
        from .training import Video, load_base

        base = load_base(BASE_SET)
        reviewed, fresh = [], []
        for r in self.store.reviews():
            path = embeddings_path(r["job_id"])
            if not path.exists():
                continue
            z = np.load(path)
            if len(z["t"]) < 4:
                continue
            v = Video(f"job:{r['job_id'][:8]}", z["t"], z["full"], z["desk"], r["label"], r["timecode_sec"],
                      r["type_id"], "review")
            reviewed.append(v)
            if r["model_version"] is None:
                fresh.append((r["job_id"], v))
        return base + reviewed, fresh

    def retrain(self, trigger: str = "manual") -> dict[str, Any]:
        from .training import train

        with self._lock:
            videos, fresh = self._videos()
            log.info("retraining on %d videos (%d new reviews)", len(videos), len(fresh))
            result = train(videos, n_folds=settings.cv_folds, log=log.info)
            accepted, decision = self._gate(result, [v for _, v in fresh])
            n = max((v["id"] for v in self.store.versions()), default=0) + 1
            path = settings.data_dir / "models" / f"violation_clf_v{n}.npz"
            result.model.meta["decision"] = decision
            result.model.save(path)
            vid = self.store.add_version({
                "status": "active" if accepted else "rejected", "trigger": trigger, "path": str(path),
                "decision": decision, "params": result.params, "metrics": result.metrics})
            if accepted:
                self.store.mark_reviews([jid for jid, _ in fresh], vid)
                self.model, self.version = result.model, vid
                self._install(result.model)
            log.info("model v%d %s: %s", vid, "promoted" if accepted else "rejected", decision)
            return {"version": vid, "accepted": accepted, "decision": decision}

    def _gate(self, result: Any, fresh: list) -> tuple[bool, str]:
        from .training import heldout_verdicts

        champion = self.model
        new = result.metrics
        if champion is None or not champion.meta.get("metrics"):
            return True, "no previous model with metrics"
        old = champion.meta["metrics"]
        if new["mcc"] + MCC_TOLERANCE < old.get("mcc", -1):
            return False, f"cross-validated MCC dropped {old['mcc']:.3f} -> {new['mcc']:.3f}"
        if new["recall"] is not None and old.get("recall") is not None and new["recall"] + 0.05 < old["recall"]:
            return False, f"recall dropped {old['recall']:.3f} -> {new['recall']:.3f}"
        if fresh:
            truth = {v.name: v.positive for v in fresh}
            champ = heldout_verdicts(champion, fresh)
            chall = {r["name"]: r["predicted"] for r in new["per_video"] if r["name"] in truth}
            old_ok = sum(champ[n] == truth[n] for n in truth)
            new_ok = sum(chall.get(n) == truth[n] for n in truth)
            if new_ok < old_ok:
                return False, f"worse on the {len(truth)} newly reviewed videos: {new_ok} vs {old_ok} correct"
            return True, (f"MCC {old.get('mcc', 0):.3f} -> {new['mcc']:.3f}, newly reviewed videos {old_ok} -> {new_ok} "
                          f"of {len(truth)} correct")
        return True, f"MCC {old.get('mcc', 0):.3f} -> {new['mcc']:.3f}"

    def activate(self, version_id: int) -> bool:
        if version_id == 0:
            target = self.base_model
        else:
            row = next((v for v in self.store.versions() if v["id"] == version_id), None)
            target = Model.load(Path(row["path"])) if row else None
        if target is None or not self.store.activate_version(version_id):
            return False
        self.model, self.version = target, version_id
        self._install(target)
        return True

    def summary(self) -> dict[str, Any]:
        versions = self.store.versions() if self.store else []
        base = self.base_model.meta if self.base_model else {}
        rows = [{"id": v["id"], "created_at": v["created_at"], "status": v["status"], "trigger": v["trigger"],
                 "decision": v["decision"], "params": v["params"], "metrics": _short(v["metrics"])}
                for v in versions]
        rows.append({"id": 0, "created_at": None, "status": "active" if self.version == 0 else "retired",
                     "trigger": "dataset", "decision": "trained offline on the labeled dataset",
                     "params": base.get("params", {}), "metrics": _short(base.get("metrics", {}))})
        current = self.model.meta if self.model else {}
        reviews = self.store.reviews() if self.store else []
        system = MODEL_PATH.parent / "system_metrics.json"
        return {
            "system": json.loads(system.read_text(encoding="utf-8")) if system.exists() else None,
            "active_version": self.version,
            "params": current.get("params", {}),
            "metrics": {k: v for k, v in current.get("metrics", {}).items() if k != "per_video"},
            "versions": rows,
            "learning": {
                "enabled": self.enabled,
                "auto_retrain": settings.auto_retrain,
                "min_reviews": settings.retrain_min_reviews,
                "cv_folds": settings.cv_folds,
                "reviews": len(reviews),
                "pending": len(self.pending()) if self.store else 0,
                **self.state,
            },
        }


def _short(m: dict[str, Any]) -> dict[str, Any]:
    keys = ("videos", "tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc")
    out = {k: m.get(k) for k in keys}
    out["fit_state"] = (m.get("fit") or {}).get("state")
    return out


learner = Learner()
