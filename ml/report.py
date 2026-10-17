from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.pipeline.learned import MODEL_PATH, Model, _crop, features, sample_indices, smooth  # noqa: E402
from app.pipeline.rules import load_yaml  # noqa: E402
from app.pipeline.scene import zone_bbox  # noqa: E402
from app.training import (COST_FN, COST_FP, Video, f_beta, fit, folds, load_base, mcc, metrics,  # noqa: E402
                          score_video, verdict_counts)

OUT = ROOT / "ml" / "reports"
FIG = OUT / "figures"
ROBUST_STEP = 1.0
BRAND, GREY, INK = "#A3134A", "#9C8E96", "#1B1420"


def perturb(name: str, frame: np.ndarray, seed: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if name == "dark":
        return np.clip(frame.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)
    if name == "compressed":
        small = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 12])
        return cv2.resize(cv2.imdecode(buf, cv2.IMREAD_COLOR), (w, h))
    if name == "occluded":
        rng = np.random.default_rng(seed)
        bw, bh = int(w * 0.35), int(h * 0.35)
        x, y = int(rng.integers(0, w - bw)), int(rng.integers(0, h - bh))
        out = frame.copy()
        out[y:y + bh, x:x + bw] = 0
        return out
    if name == "angle":
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[0.08 * w, 0.04 * h], [0.97 * w, 0], [w, h], [0.02 * w, 0.93 * h]])
        warped = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(src, dst), (w, h))
        rot = cv2.getRotationMatrix2D((w / 2, h / 2), 7, 1.08)
        return cv2.warpAffine(warped, rot, (w, h), borderMode=cv2.BORDER_REFLECT)
    raise ValueError(name)


CONDITIONS = {
    "dark": "освещение ×0,35",
    "compressed": "½ разрешения, JPEG q=12",
    "occluded": "чёрный блок 35% кадра",
    "angle": "перспектива + поворот 7° (имитация другого ракурса)",
}


def perturbed_embeddings(cache: Path, name: str, cond: str, embedder: list) -> tuple | None:
    path = cache / "robust" / cond / f"emb_{name}.npz"
    if not path.exists():
        pkl = cache / f"{name}.pkl"
        if not pkl.exists():
            return None
        if not embedder:
            from app.pipeline.embedder import Embedder
            embedder.append(Embedder())
        scene = pickle.loads(pkl.read_bytes())
        idx = sample_indices(scene.obs, ROBUST_STEP)
        box = zone_bbox(scene.zones["desk_zone"]) if scene.localized and "desk_zone" in scene.zones \
            else (0.1, 0.2, 0.9, 0.9)
        seed = zlib.crc32(name.encode())
        full, desk = [], []
        for i in range(0, len(idx), 16):
            frames = [perturb(cond, scene.obs[j].frame(), seed) for j in idx[i:i + 16]]
            full.append(embedder[0](frames))
            desk.append(embedder[0]([_crop(f, box) for f in frames]))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, t=np.array([scene.obs[i].t for i in idx], np.float32),
                 full=np.concatenate(full), desk=np.concatenate(desk))
        print(f"  {cond:<10} {name}: {len(idx)} moments", flush=True)
    z = np.load(path)
    return z["t"], z["full"], z["desk"]


def thin(v: Video, step: float = ROBUST_STEP) -> Video:
    keep, next_t = [], -1e9
    for i, t in enumerate(v.t):
        if t >= next_t:
            keep.append(i)
            next_t = t + step - 1e-6
    return Video(v.name, v.t[keep], v.full[keep], v.desk[keep], v.label, v.label_t, v.vtype)


def robustness(videos: list[Video], model: Model, cache: Path, n_folds: int) -> dict:
    k, c, thr = model.meta["params"]["k"], model.meta["params"]["c"], model.threshold
    parts = folds(videos, n_folds)
    fold_models = {}
    for fold in parts:
        m = fit([v for i, v in enumerate(videos) if i not in set(fold)], k, c)
        for i in fold:
            fold_models[i] = m
    out = {"clean": metrics([score_video(fold_models[i], thin(v)) for i, v in enumerate(videos)], thr)}
    embedder: list = []
    for cond in CONDITIONS:
        held = []
        for i, v in enumerate(videos):
            emb = perturbed_embeddings(cache, v.name, cond, embedder)
            if emb is None:
                continue
            t, full, desk = emb
            pv = Video(v.name, t, full, desk, v.label, v.label_t, v.vtype)
            held.append(score_video(fold_models[i], pv))
        out[cond] = metrics(held, thr)
    return out


def ensure_platt(model: Model, held_metrics: dict) -> None:
    cal = held_metrics.get("calibration", {})
    if model.calib is None and cal.get("applied"):
        model.calib = (cal["platt_a"], cal["platt_b"])


def subgroups(rows: list[dict]) -> dict:
    def group(key: str, labels: tuple[str, str, str]) -> list[dict]:
        vals = np.array([r[key] for r in rows])
        cuts = np.quantile(vals, [1 / 3, 2 / 3])
        out = []
        for gi, label in enumerate(labels):
            sel = [r for r, v in zip(rows, vals) if (gi == 0 and v <= cuts[0]) or
                   (gi == 1 and cuts[0] < v <= cuts[1]) or (gi == 2 and v > cuts[1])]
            out.append({"group": label, "range": [round(float(min(r[key] for r in sel)), 1),
                                                  round(float(max(r[key] for r in sel)), 1)], **verdicts(sel)})
        return out

    return {
        "brightness": group("brightness", ("тёмные", "средние", "светлые")),
        "duration_sec": group("duration_sec", ("короткие", "средние", "длинные")),
        "mean_people": group("mean_people", ("мало людей", "средне", "людно")),
    }


def verdicts(rows: list[dict], key: str = "system") -> dict:
    m = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for r in rows:
        pred, truth = r[key]["violation"], r["label"] == "violation"
        m[("t" if pred == truth else "f") + ("p" if pred else "n")] += 1
    pos, neg = m["tp"] + m["fn"], m["tn"] + m["fp"]
    return {"videos": len(rows), **m, "accuracy": round((m["tp"] + m["tn"]) / max(len(rows), 1), 3),
            "recall": round(m["tp"] / pos, 3) if pos else None, "fpr": round(m["fp"] / neg, 3) if neg else None,
            "mcc": round(mcc(m), 3)}


def per_type(rows: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for r in rows:
        if r["label"] == "violation":
            by[r["type"] or "unknown"].append(r)
    out = []
    for vtype, rs in sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        dts = [min(abs(t - r["label_t"]) for t in r["system"]["times"]) for r in rs
               if r["system"]["violation"] and r["system"]["times"] and r["label_t"] is not None]
        out.append({"type": vtype, "videos": len(rs),
                    "system": sum(r["system"]["violation"] for r in rs),
                    "model": sum(r["model"]["violation"] for r in rs),
                    "rules": sum(r["rules"]["violation"] for r in rs),
                    "exact_type": sum(vtype in r["system"]["types"] for r in rs if r["system"]["violation"]),
                    "median_dt": round(float(np.median(dts)), 1) if dts else None})
    return out


def production(dsn: str) -> dict:
    import psycopg

    with psycopg.connect(dsn, connect_timeout=5) as conn:
        jobs = conn.execute(
            "SELECT extract(epoch FROM created_at), status, verdict, (meta->>'duration_sec')::float, "
            "(meta->>'processing_time_sec')::float, extract(epoch FROM finished_at - created_at), "
            "(meta->>'coverage')::float, meta->>'model_version' FROM jobs").fetchall()
        reviews = conn.execute(
            "SELECT extract(epoch FROM r.created_at), r.label, j.verdict FROM reviews r JOIN jobs j ON j.id = r.job_id"
        ).fetchall()
        versions = conn.execute(
            "SELECT id, extract(epoch FROM created_at), status, trigger, decision, (metrics->>'mcc')::float, "
            "(metrics->>'recall')::float, (params->>'videos')::int FROM model_versions ORDER BY id").fetchall()
    done = [j for j in jobs if j[1] == "done" and j[3] and j[4]]
    weeks = defaultdict(lambda: {"reviews": 0, "agree": 0})
    for ts, label, verdict in reviews:
        wk = time.strftime("%G-W%V", time.gmtime(ts))
        weeks[wk]["reviews"] += 1
        weeks[wk]["agree"] += label == verdict
    kappa = None
    if len(reviews) >= 2 and len({r[1] for r in reviews}) > 1:
        from sklearn.metrics import cohen_kappa_score
        kappa = round(float(cohen_kappa_score([r[1] for r in reviews], [r[2] for r in reviews])), 3)
    return {
        "jobs": len(jobs), "done": sum(j[1] == "done" for j in jobs), "failed": sum(j[1] == "failed" for j in jobs),
        "flagged": sum(j[2] == "violation" for j in jobs if j[1] == "done"),
        "reviewed": len(reviews),
        "realtime_factor": round(float(np.median([j[4] / j[3] for j in done])), 2) if done else None,
        "processing_sec_p50": round(float(np.median([j[4] for j in done])), 1) if done else None,
        "processing_sec_max": round(float(max(j[4] for j in done)), 1) if done else None,
        "turnaround_sec_max": round(float(max(j[5] for j in done)), 1) if done else None,
        "full_coverage": sum((j[6] or 0) >= 0.999 for j in done),
        "review_agreement_by_week": dict(sorted(weeks.items())),
        "review_kappa": kappa,
        "versions": [{"id": v[0], "created_at": float(v[1]), "status": v[2], "trigger": v[3], "decision": v[4],
                      "mcc": v[5], "recall": v[6], "videos": v[7]} for v in versions],
    }


def legal(rules_cfg: dict) -> list[dict]:
    norms = rules_cfg.get("norms", {})
    defs = {r["type"]: r for r in rules_cfg.get("rules", [])}
    out = []
    for vtype in list(defs) + [t for t in rules_cfg.get("learned_types", {}) if t not in defs]:
        learned = rules_cfg.get("learned_types", {})
        d = {**defs.get(vtype, {}), **learned.get(vtype, {})}
        if not d.get("enabled", True) and vtype not in learned:
            continue
        for n in d.get("norms", []):
            act = norms.get(n.get("ref"), {})
            out.append({"type": vtype, "title": d.get("title", vtype), "act": act.get("short", n.get("ref")),
                        "clause": n.get("clause"), "url": act.get("url"), "quote": bool(n.get("quote")),
                        "enabled": d.get("enabled", True)})
    return out


def figures(m: dict, sysm: dict, robust: dict | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    roc = np.array(m["roc_curve"])
    ax[0].plot(roc[:, 0], roc[:, 1], color=BRAND, lw=2, label=f"модель, AUC {m['roc_auc']:.2f}")
    ax[0].plot([0, 1], [0, 1], "--", color=GREY, lw=1, label="случайно")
    ax[0].set(xlabel="FPR", ylabel="TPR", title="ROC (по видео)")
    ax[0].legend(frameon=False, loc="lower right")
    ops = m["operating_points"]
    rec = [o["recall"] for o in ops]
    prec = [o["precision"] or 0 for o in ops]
    ax[1].plot(rec, prec, color=BRAND, lw=2, marker="o", ms=3, label=f"PR-AUC {m['pr_auc']:.2f}")
    ax[1].axhline(m["positives"] / m["videos"], color=GREY, ls="--", lw=1, label="доля нарушений")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision–Recall", ylim=(0, 1.05))
    ax[1].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "roc_pr.png", dpi=150)
    plt.close(fig)

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    thr = [o["threshold"] for o in ops]
    ax.plot(thr, rec, color=BRAND, lw=2)
    ax.plot(thr, prec, color=INK, lw=2)
    ax.text(thr[0], rec[0] + 0.03, "recall", color=BRAND)
    ax.text(thr[0], prec[0] - 0.09, "precision", color=INK)
    ax.set(ylabel="доля", ylim=(0, 1.08), title="Рабочая точка")
    bx.plot(thr, [o["cost"] for o in ops], color=GREY, lw=2)
    bx.set(xlabel="порог на пиковую оценку видео", ylabel=f"стоимость (FN={COST_FN:g}, FP={COST_FP:g})")
    for axis in (ax, bx):
        axis.axvline(m["threshold"], color=BRAND, ls="--", lw=1)
    ax.text(m["threshold"] + 0.005, 0.05, "выбранный порог", color=BRAND, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "operating_point.png", dpi=150)
    plt.close(fig)

    cal = m["calibration"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1)
    for key, color, label in (("raw_bins", GREY, f"сырая, ECE {cal['raw_ece']:.2f}"),
                              ("calibrated_bins", BRAND, f"Platt, ECE {cal.get('calibrated_ece', float('nan')):.2f}")):
        if key in cal:
            b = np.array(cal[key])
            ax.plot(b[:, 0], b[:, 1], marker="o", color=color, lw=2, label=label)
    ax.set(xlabel="предсказанная вероятность", ylabel="доля нарушений", title="Диаграмма надёжности",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "reliability.png", dpi=150)
    plt.close(fig)

    if robust:
        conds = list(robust)
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(conds))
        ax.bar(x - 0.2, [robust[c]["recall"] for c in conds], 0.38, color=BRAND, label="recall")
        ax.bar(x + 0.2, [robust[c]["false_positive_rate"] for c in conds], 0.38, color=GREY, label="доля ложных тревог")
        names = {"clean": "без искажений", "dark": "темно", "compressed": "сжатие", "occluded": "перекрытие",
                 "angle": "другой ракурс"}
        ax.set_xticks(x, [names.get(c, c) for c in conds])
        ax.set(ylim=(0, 1.0), title="Устойчивость модели (отложенные видео)")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
        fig.tight_layout()
        fig.savefig(FIG / "robustness.png", dpi=150)
        plt.close(fig)


def fmt(x, d=2) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{d}f}"
    return str(x)


def table(head: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def write_markdown(r: dict) -> str:
    m, sysm, p = r["model"], r["system"], r["params"]
    s = sysm["system"]
    lines = [
        "# CashGuard — отчёт о качестве модели",
        "",
        f"Сформирован {time.strftime('%Y-%m-%d %H:%M', time.localtime(r['generated_at']))} командой "
        "`python ml/report.py`. Для разработчиков: в интерфейс не выводится.",
        "",
        f"Модель: PCA k={p['k']}, C={p['c']}, порог {p['threshold']}, {p['videos']} видео, "
        f"{p['folds']} фолдов по роликам. Все цифры ниже — на видео, не участвовавших в обучении.",
        "",
        "## 1. Классификация (вердикт по видео)",
        "",
        table(["Метрика", "Модель", "Сервис (правила + модель)", "Пояснение"], [
            ["Precision", m["precision"], s["precision"], "доля флагов, которые оказались нарушениями"],
            ["Recall (чувствительность)", m["recall"], s["recall"], "доля нарушений, которые поймали"],
            ["F1", m["f1"], s["f1"], "гармоническое среднее precision и recall"],
            ["F2", m["f2"], r["system_extra"]["f2"], "recall весит вдвое больше: пропуск дороже"],
            ["Accuracy", m["accuracy"], s["accuracy"], "при 22/20 классах слабая метрика"],
            ["Specificity", m["specificity"], r["system_extra"]["specificity"], "доля чистых роликов, пропущенных верно"],
            ["FPR", m["false_positive_rate"], r["system_extra"]["fpr"], "ложные тревоги / чистые"],
            ["FNR", m["false_negative_rate"], r["system_extra"]["fnr"], "пропуски / нарушения"],
            ["ROC-AUC", m["roc_auc"], None, "ранжирование без порога"],
            ["PR-AUC (AP)", m["pr_auc"], None, "то же для класса «нарушение»"],
            ["MCC", m["mcc"], s["mcc"], "главный критерий отбора модели"],
            ["Cohen's kappa", m["cohen_kappa"], r["system_extra"]["kappa"], "согласие с разметкой сверх случайного"],
            ["Brier", m["brier"], None, "квадрат ошибки оценки"],
        ]),
        "",
        "Матрицы ошибок:",
        "",
        table(["", "TP", "FN", "FP", "TN"], [
            ["Модель", m["tp"], m["fn"], m["fp"], m["tn"]],
            ["Сервис", s["tp"], s["fn"], s["fp"], s["tn"]],
            ["Только правила", sysm["rules"]["tp"], sysm["rules"]["fn"], sysm["rules"]["fp"], sysm["rules"]["tn"]],
            ["Всегда «нарушение»", sysm["always"]["tp"], sysm["always"]["fn"], sysm["always"]["fp"],
             sysm["always"]["tn"]],
        ]),
        "",
        "![ROC и PR](figures/roc_pr.png)",
        "",
        "### По типам нарушений",
        "",
        "Тип — по нашей разметке `ml/violation_types.csv`. «Найдено» — вердикт по видео; «точный тип» — "
        "сервис назвал именно этот тип.",
        "",
        table(["Тип", "Видео", "Найдено сервисом", "моделью", "правилами", "Точный тип", "Медиана |Δt|, с"],
              [[t["type"], t["videos"], t["system"], t["model"], t["rules"], t["exact_type"], t["median_dt"]]
               for t in r["per_type"]]),
        "",
        "## 2. Калибровка и порог",
        "",
        f"ECE сырой оценки {fmt(m['calibration']['raw_ece'])}, после калибровки Platt "
        f"{fmt(m['calibration'].get('calibrated_ece'))} (калибровка оценена leave-one-out). Brier после калибровки "
        f"{fmt(m['calibration'].get('calibrated_brier'))}. "
        + ("Калибровка улучшила ECE и применяется: поле `confidence` в отчёте — калиброванная вероятность."
           if m["calibration"].get("applied") else
           "Калибровка ECE не улучшила и не применяется: `confidence` — сырая оценка модели."),
        "",
        "![Диаграмма надёжности](figures/reliability.png)",
        "",
        f"Рабочая точка: порог {p['threshold']} выбран по максимуму MCC на отложенных оценках. Стоимость ошибок "
        f"при FN={COST_FN:g}, FP={COST_FP:g}: {fmt(m['cost']['total'], 0)} "
        f"({fmt(m['cost']['per_video'])} на видео). Порог с минимальной стоимостью: "
        f"{r['cost_optimal']['threshold']} (TP {r['cost_optimal']['tp']}, FP {r['cost_optimal']['fp']}, "
        f"стоимость {fmt(r['cost_optimal']['cost'], 0)}). Веса стоимости задаёт банк: "
        "`COST_FN`, `COST_FP` в `backend/app/training.py`.",
        "",
        "![Рабочая точка](figures/operating_point.png)",
        "",
        table(["Порог", "TP", "FP", "FN", "Precision", "Recall", "F2", "MCC", "Стоимость"],
              [[o["threshold"], o["tp"], o["fp"], o["fn"], o["precision"], o["recall"], o["f2"], o["mcc"], o["cost"]]
               for o in r["operating_subset"]]),
        "",
        "## 3. Переобучение и недообучение",
        "",
        f"ROC-AUC по моментам: обучение {fmt(m['fit']['train_moment_roc_auc'])}, отложенные "
        f"{fmt(m['fit']['cv_moment_roc_auc'])}, разрыв {fmt(m['fit']['gap'])} → **{m['fit']['state']}** "
        "(порог переобучения 0,15).",
        "",
        table(["k", "C", "MCC", "F1", "ROC-AUC", "TP", "FP", "FN", ""],
              [[x["k"], x["c"], x["mcc"], x["f1"], x["roc_auc"], x["tp"], x["fp"], x["fn"],
                "выбрано" if x["selected"] else ""] for x in m["search"]]),
        "",
        "## 4. Справедливость и устойчивость",
        "",
        "**Защищённые признаки.** Пол, возраст, этническая принадлежность в данных отсутствуют и не "
        "извлекаются: распознавание лиц запрещено условиями кейса, сервис не идентифицирует людей. "
        "Поэтому подгрупповой анализ по защищённым признакам и метрики demographic parity / equalized odds "
        "посчитать нельзя. Для аудита нужна отдельная разметка видео по этим признакам с согласия "
        "сотрудников. Вместо этого ниже — разрез по условиям съёмки (сервис целиком).",
        "",
    ]
    for key, title in (("brightness", "Яркость кадра"), ("duration_sec", "Длина видео, с"),
                       ("mean_people", "Людей в кадре (среднее)")):
        lines += [f"**{title}**", "", table(["Группа", "Диапазон", "Видео", "Accuracy", "Recall", "FPR", "MCC"], [
            [g["group"], f"{g['range'][0]}–{g['range'][1]}", g["videos"], g["accuracy"], g["recall"], g["fpr"],
             g["mcc"]] for g in r["subgroups"][key]]), ""]
    gaps = r["subgroup_gaps"]
    lines += [f"Наибольший разрыв recall между группами: {fmt(gaps['recall'])}, FPR: {fmt(gaps['fpr'])} "
              "(аналог equalized odds gap по условиям съёмки).", ""]
    if r.get("robustness"):
        rb = r["robustness"]
        lines += [
            "**Искажения кадра.** Каждый ролик перекодирован с искажением и заново пропущен через CLIP "
            "(1 кадр в секунду, чистые ролики для сравнения прорежены так же); "
            "оценивает модель фолда, которая этот ролик не видела. Порог тот же, что в сервисе.",
            "",
            table(["Условие", "Что сделано", "Recall", "FPR", "MCC", "ROC-AUC"],
                  [["чистые", "без искажений", rb["clean"]["recall"], rb["clean"]["false_positive_rate"],
                    rb["clean"]["mcc"], rb["clean"]["roc_auc"]]] +
                  [[c, CONDITIONS[c], rb[c]["recall"], rb[c]["false_positive_rate"], rb[c]["mcc"], rb[c]["roc_auc"]]
                   for c in CONDITIONS if c in rb]),
            "",
            "![Устойчивость](figures/robustness.png)",
            "",
            "Проверяли обучение с аугментацией (те же искажённые копии обучающих роликов). При k=16 recall на "
            "чистых видео упал с 1,00 до 0,64, при k=32 — до 0,86; темнота и ракурс стали лучше, сжатие хуже "
            "(ROC-AUC 0,35). Для политики «пропуск недопустим» это ухудшение, поэтому аугментация не включена.",
            "",
        ]
    prod = r.get("production")
    lines += ["## 5. Дрейф и работа в эксплуатации", ""]
    if prod:
        lines += [
            f"Задач в базе: {prod['jobs']} (готово {prod['done']}, ошибок {prod['failed']}). Флагов: {prod['flagged']}, "
            f"подтверждено проверяющими: {prod['reviewed']}. Согласие ответов с вердиктом (kappa): "
            f"{fmt(prod['review_kappa'])}.",
            "",
        ]
        if prod["review_agreement_by_week"]:
            lines += [table(["Неделя", "Ответов", "Совпало с вердиктом"],
                            [[w, v["reviews"], v["agree"]] for w, v in prod["review_agreement_by_week"].items()]), ""]
        else:
            lines += ["Подтверждений пока нет: дрейф начнёт считаться по неделям с первых ответов проверяющих.", ""]
        if prod["versions"]:
            lines += [table(["Версия", "Статус", "Причина", "Видео", "MCC", "Recall"],
                            [[v["id"], v["status"], v["decision"], v["videos"], v["mcc"], v["recall"]]
                             for v in prod["versions"]]), ""]
    else:
        lines += ["База недоступна (нет `DATABASE_URL`): разделы о дрейфе и скорости в эксплуатации пропущены.", ""]

    lg = r["legal"]
    ev = r["evidence"]
    lines += [
        "## 6. Юридические и операционные критерии",
        "",
        f"**Объяснимость.** У {r['explain']['types_with_norm']} из {r['explain']['types']} типов есть пункт акта с "
        f"дословной цитатой и ссылкой на adilet.zan.kz. В прогоне по датасету {ev['with_basis']} из {ev['flags']} "
        "флагов сопровождаются нормой.",
        "",
        f"**Доказательства.** Кадров на нарушение: от {ev['frames_min']} до {ev['frames_max']} (в среднем "
        f"{fmt(ev['frames_mean'])}), у каждого кадра таймкод. Кадры хранятся JPEG-файлами, в базе — пути "
        "(`evidence_frames`), что позволяет поднять их при аудите.",
        "",
        "**Персональные данные.** Сервис хранит минимум: исходное видео удаляется сразу после обработки "
        "(`KEEP_VIDEOS=false`), лица не распознаются, звук не используется. Срок хранения ограничен "
        "`RETENTION_DAYS`: отчёты и кадры старше срока удаляются, у подтверждённых видео остаются только "
        "эмбеддинги и ответ проверяющего. Основание обработки и согласие сотрудников — вопрос к DPO банка "
        "(Закон РК «О персональных данных и их защите»); юридическую оценку сервис не даёт.",
        "",
        f"**Человек в контуре.** На датасете сервис отправляет человеку {r['hitl']['flagged']} из "
        f"{r['hitl']['videos']} видео ({fmt(r['hitl']['rate'] * 100, 0)}%): каждый флаг — подсказка для проверки."
        + (f" В эксплуатации проверено {prod['reviewed']} из {prod['flagged']} флагов." if prod else ""),
        "",
        "**Скорость.** " + (
            f"Медиана обработки {fmt(prod['processing_sec_p50'], 1)} с, максимум {fmt(prod['processing_sec_max'], 1)} с, "
            f"от загрузки до ответа не дольше {fmt(prod['turnaround_sec_max'], 1)} с; медианный коэффициент "
            f"реального времени {fmt(prod['realtime_factor'])} (время обработки / длина видео). "
            if prod and prod["processing_sec_p50"] else "") +
        "Самый длинный ролик датасета (142 с) — 171 с на CPU; лимит 300 с соблюдается за счёт адаптивной частоты "
        "кадров.",
        "",
        "**Юридическая проверка соответствия.** Сопоставление правил и норм ниже требует подписи юриста или "
        "комплаенса банка. Статус у всех строк — «ожидает проверки».",
        "",
        table(["Тип", "Акт", "Пункт", "Цитата", "Ссылка", "Статус"],
              [[x["type"], x["act"], x["clause"], "есть" if x["quote"] else "нет", x["url"], "ожидает проверки"]
               for x in lg]),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=ROOT / "ml" / "cache")
    ap.add_argument("--skip-robustness", action="store_true")
    args = ap.parse_args()

    model = Model.load(MODEL_PATH)
    videos = load_base(MODEL_PATH.parent / "base_set.npz")
    sysm = json.loads((MODEL_PATH.parent / "system_metrics.json").read_text(encoding="utf-8"))
    m = model.meta["metrics"]
    ensure_platt(model, m)
    rows = sysm["per_video"]

    s = sysm["system"]
    pos, neg = s["tp"] + s["fn"], s["tn"] + s["fp"]
    from sklearn.metrics import cohen_kappa_score
    system_extra = {
        "f2": round(f_beta(s, 2.0), 4), "specificity": round(s["tn"] / neg, 4), "fpr": round(s["fp"] / neg, 4),
        "fnr": round(s["fn"] / pos, 4),
        "kappa": round(float(cohen_kappa_score([r["label"] == "violation" for r in rows],
                                               [r["system"]["violation"] for r in rows])), 4),
    }
    ops = m["operating_points"]
    step = max(len(ops) // 10, 1)
    subset = ops[::step]
    if not any(o["threshold"] == m["threshold"] for o in subset):
        subset = sorted(subset + [o for o in ops if o["threshold"] == m["threshold"]], key=lambda o: o["threshold"])
    groups = subgroups(rows)
    flat = [g for gs in groups.values() for g in gs]
    flags = [e for r in rows for e in r["evidence"]]
    rules_cfg = load_yaml(ROOT / "backend" / "config" / "rules.yaml")
    lg = legal(rules_cfg)

    report = {
        "generated_at": time.time(),
        "params": model.meta["params"],
        "model": {k: v for k, v in m.items() if k != "per_video"},
        "system": {k: v for k, v in sysm.items() if k != "per_video"},
        "system_extra": system_extra,
        "per_type": per_type(rows),
        "cost_optimal": min(ops, key=lambda o: (o["cost"], -o["threshold"])),
        "operating_subset": subset,
        "subgroups": groups,
        "subgroup_gaps": {
            "recall": round(max(g["recall"] for g in flat if g["recall"] is not None)
                            - min(g["recall"] for g in flat if g["recall"] is not None), 3),
            "fpr": round(max(g["fpr"] for g in flat if g["fpr"] is not None)
                         - min(g["fpr"] for g in flat if g["fpr"] is not None), 3),
        },
        "legal": lg,
        "explain": {"types": len({x["type"] for x in lg}),
                    "types_with_norm": len({x["type"] for x in lg if x["quote"] and x["url"]})},
        "evidence": {"flags": len(flags), "with_basis": sum(e["legal_basis"] > 0 and e["quoted"] for e in flags),
                     "frames_min": min((e["frames"] for e in flags), default=0),
                     "frames_max": max((e["frames"] for e in flags), default=0),
                     "frames_mean": float(np.mean([e["frames"] for e in flags])) if flags else None},
        "hitl": {"videos": len(rows), "flagged": sum(r["system"]["violation"] for r in rows),
                 "rate": sum(r["system"]["violation"] for r in rows) / len(rows)},
    }
    if not args.skip_robustness:
        print("robustness: embedding perturbed videos (cached in ml/cache/robust)")
        report["robustness"] = robustness(videos, model, args.cache, model.meta["params"]["folds"])
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        try:
            report["production"] = production(dsn)
        except Exception as exc:
            print("production metrics skipped:", exc)

    OUT.mkdir(parents=True, exist_ok=True)
    figures(m, sysm, report.get("robustness"))
    (OUT / "model_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float),
                                           encoding="utf-8")
    (OUT / "model_report.md").write_text(write_markdown(report), encoding="utf-8")
    print(f"saved {OUT / 'model_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
