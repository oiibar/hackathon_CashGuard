import time

import numpy as np

from app import learner as learner_mod
from app.jobs import Job
from app.learner import Learner, embeddings_path
from app.pipeline.learned import save_embeddings
from app.store import FileStore
from app.training import Video, folds, load_base, metrics, pick_threshold, save_base, train, verdict_counts

SIGNAL = np.random.default_rng(1).normal(size=512).astype(np.float32)


def synthetic(name, positive, rng, length=24.0, at=12.0):
    t = np.arange(0, length, 0.5, dtype=np.float32)
    room = rng.normal(size=512).astype(np.float32)
    full = room + 0.05 * rng.normal(size=(len(t), 512)).astype(np.float32)
    desk = room[::-1] + 0.05 * rng.normal(size=(len(t), 512)).astype(np.float32)
    if positive:
        on = (t >= at) & (t <= at + 5)
        full[on] += 0.6 * SIGNAL
        desk[on] += 0.6 * SIGNAL
    return Video(name, t, full, desk, "violation" if positive else "no_violation", at if positive else None,
                 "cash_left_open" if positive else None)


def dataset(n=12, seed=0):
    rng = np.random.default_rng(seed)
    return [synthetic(f"v{i}", i % 2 == 0, rng) for i in range(n)]


def test_folds_keep_videos_whole_and_stratified():
    videos = dataset(12)
    parts = folds(videos, 4)
    assert sorted(i for f in parts for i in f) == list(range(12))
    assert all(sum(videos[i].positive for i in f) >= 1 for f in parts)
    assert len(folds(videos, 0)) == 12


def test_training_separates_a_clear_signal_and_reports_metrics():
    result = train(dataset(12), n_folds=4, grid=[(4, 0.1), (8, 1.0)], log=lambda *_: None)
    m = result.metrics
    assert m["recall"] == 1.0 and m["fp"] == 0
    assert m["roc_auc"] == 1.0 and m["f1"] == 1.0 and m["mcc"] == 1.0 and m["cohen_kappa"] == 1.0
    assert m["false_negative_rate"] == 0.0 and m["cost"]["total"] == 0.0
    cal = m["calibration"]
    assert cal["applied"] == (cal["calibrated_ece"] < cal["raw_ece"])
    assert (result.model.calib is not None) == cal["applied"]
    assert 0.5 < result.model.confidence(0.99) <= 1.0
    assert m["timecode"]["median_sec"] <= 2.0
    assert m["fit"]["state"] == "good"
    assert len(m["search"]) == 2 and sum(r["selected"] for r in m["search"]) == 1
    assert result.model.proto_types and result.model.meta["params"]["videos"] == 12


def test_metrics_on_known_confusion():
    held = train(dataset(8), n_folds=4, grid=[(4, 0.1)], log=lambda *_: None).held
    m = metrics(held, 2.0)
    assert m["tp"] == 0 and m["fn"] == 4 and m["tn"] == 4 and m["recall"] == 0.0 and m["threshold"] == 2.0


def test_base_set_roundtrip(tmp_path):
    videos = dataset(4)
    save_base(tmp_path / "base.npz", videos)
    back = load_base(tmp_path / "base.npz")
    assert [v.name for v in back] == [v.name for v in videos]
    assert back[0].label_t == 12.0 and back[1].label_t is None and back[0].vtype == "cash_left_open"
    assert np.allclose(back[2].full, videos[2].full, atol=1e-2)


def test_reviews_retrain_promote_and_rollback(tmp_path, monkeypatch):
    base_path = tmp_path / "base.npz"
    save_base(base_path, dataset(10))
    monkeypatch.setattr(learner_mod, "BASE_SET", base_path)
    monkeypatch.setattr("app.training.GRID", [(4, 0.1)])
    base_model = train(dataset(10), n_folds=4, log=lambda *_: None).model
    base_model.save(tmp_path / "clf.npz")
    monkeypatch.setattr(learner_mod, "MODEL_PATH", tmp_path / "clf.npz")
    store = FileStore(tmp_path / "jobs", Job)
    store.open()
    lr = Learner()
    lr.start(store, [])
    assert lr.version == 0 and lr.model.meta["metrics"]["mcc"] == 1.0

    rng = np.random.default_rng(7)
    for i, positive in enumerate([True, False, True, False]):
        job = Job(id=f"job{i}", filename=f"{i}.mov", video_path="", zones=None, created_at=time.time(),
                  deadline=time.time())
        v = synthetic(job.id, positive, rng)
        save_embeddings(embeddings_path(job.id), v.t, v.full, v.desk)
        out = lr.review(job, v.label, v.label_t, v.vtype)
        assert out["trainable"]
    assert len(lr.pending()) == 4

    first = lr.retrain("test")
    assert first["accepted"] and lr.version == first["version"]
    assert lr.pending() == []
    assert store.versions()[0]["metrics"]["videos"] == 14

    summary = lr.summary()
    assert summary["active_version"] == first["version"]
    assert {v["id"] for v in summary["versions"]} == {0, first["version"]}

    assert lr.activate(0) and lr.version == 0
    assert all(v["status"] != "active" for v in store.versions())


def test_cost_policy_never_trades_a_miss_for_a_false_alarm():
    held = train(dataset(8), n_folds=4, grid=[(4, 0.1)], log=lambda *_: None).held
    for h, peak in zip(held, [0.9, 0.2, 0.7, 0.6, 0.65, 0.3, 0.8, 0.1]):
        h.s = np.full(len(h.t), peak)
    thr = pick_threshold(held, "cost")
    m = verdict_counts(held, thr)
    assert m["fn"] == 0
    assert pick_threshold(held, "mcc") >= thr
