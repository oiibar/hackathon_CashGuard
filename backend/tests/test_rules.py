import numpy as np

from app.config import settings
from app.pipeline.detector import Detection
from app.pipeline.rules import RuleEngine, build_description, load_yaml

RULES = load_yaml(settings.rules_path)
FULL = {"cash_zone": [[0, 0], [1, 0], [1, 1], [0, 1]]}
FRAME = np.zeros((100, 200, 3), dtype=np.uint8)


def person(x, conf=0.9):
    return Detection("person", conf, (x, 10, x + 20, 90))


def run(engine, timeline, fps=4.0):
    dt = 1 / fps
    for i, dets in enumerate(timeline):
        engine.step(i * dt, FRAME, dets, dt)
    return engine.finish(dt)


def test_person_count_violation_needs_min_duration():
    eng = RuleEngine(RULES, FULL)
    three = [person(10), person(60), person(110)]
    # 2 s of 3 people: shorter than min_duration_sec=3 -> nothing
    assert not [r for r, _ in run(eng, [three] * 8) if r["type"] == "unauthorized_person_in_cash_zone"]
    eng = RuleEngine(RULES, FULL)
    hits = [ep for r, ep in run(eng, [[]] * 4 + [three] * 16) if r["type"] == "unauthorized_person_in_cash_zone"]
    assert len(hits) == 1 and hits[0].start == 1.0


def test_short_detection_gap_does_not_split_episode():
    eng = RuleEngine(RULES, FULL)
    phone = Detection("cell phone", 0.8, (50, 50, 60, 60))
    timeline = [[phone]] * 6 + [[]] * 2 + [[phone]] * 6  # 0.5 s dropout
    eps = [ep for r, ep in run(eng, timeline) if r["type"] == "mobile_phone_in_cash_zone"]
    assert len(eps) == 1


def test_rule_skipped_without_zone_or_class():
    eng = RuleEngine(RULES, FULL)
    eng.disable_missing_classes({"person", "cell phone"})
    skipped = {s["type"] for s in eng.skipped}
    assert "cashier_left_during_service" in skipped   # no client_zone
    assert "cash_left_unattended" in skipped          # no 'cash' class


def test_cashier_left_during_service():
    zones = {"cash_zone": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]],
             "client_zone": [[0.5, 0], [1, 0], [1, 1], [0.5, 1]]}
    eng = RuleEngine(RULES, zones)
    client = person(150)
    cashier = person(20)
    timeline = [[client, cashier]] * 8 + [[client]] * 48  # 12 s alone
    eps = [ep for r, ep in run(eng, timeline) if r["type"] == "cashier_left_during_service"]
    assert len(eps) == 1 and eps[0].start == 2.0


def test_description_cites_rk_act():
    rule = next(r for r in RULES["rules"] if r["type"] == "mobile_phone_in_cash_zone")
    text, basis = build_description(rule, RULES["norms"])
    assert "№ 231" in text and "adilet.zan.kz" in text
    assert basis[0]["clause"] == "п. 12"
