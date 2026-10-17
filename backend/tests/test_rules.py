from app.config import settings
from app.pipeline.detector import Detection
from app.pipeline.rules import RuleEngine, build_description, load_yaml
from app.pipeline.scene import Observation

RULES = load_yaml(settings.rules_path)
SIZE = (200, 100)
HALF = [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]
WORKPLACE = {"cash_zone": HALF, "desk_zone": HALF, "client_zone": [[0, 0], [1, 0], [1, 1], [0, 1]]}


def person(x, conf=0.9):
    return Detection("person", conf, (x, 10, x + 20, 90))


def run(engine, timeline, fps=4.0, brightness=None):
    dt = 1 / fps
    for i, dets in enumerate(timeline):
        b = brightness[i] if brightness else 120.0
        engine.step(Observation(i * dt, dets, b"", b, 500.0, SIZE), dt)
    return engine.finish(dt)


def episodes(result, vtype):
    return [ep for r, ep in result if r["type"] == vtype]


def test_second_person_at_workplace_needs_min_duration():
    two = [person(10), person(60)]
    assert not episodes(run(RuleEngine(RULES, WORKPLACE), [two] * 8), "unauthorized_person_in_cash_zone")
    eps = episodes(run(RuleEngine(RULES, WORKPLACE), [[person(10)]] * 4 + [two] * 16),
                   "unauthorized_person_in_cash_zone")
    assert len(eps) == 1 and eps[0].start == 1.0 and eps[0].peak_value == 2


def test_client_across_counter_is_not_an_outsider():
    timeline = [[person(10), person(150)]] * 20
    assert not episodes(run(RuleEngine(RULES, WORKPLACE), timeline), "unauthorized_person_in_cash_zone")


def test_short_detection_gap_does_not_split_episode():
    cashier, phone = person(20), Detection("cell phone", 0.8, (26, 40, 34, 50))
    timeline = [[cashier, phone]] * 6 + [[cashier]] * 2 + [[cashier, phone]] * 6
    assert len(episodes(run(RuleEngine(RULES, WORKPLACE), timeline), "cashier_uses_phone")) == 1


def test_rule_skipped_without_zone_class_or_localization():
    eng = RuleEngine(RULES, {"cash_zone": [[0, 0], [1, 0], [1, 1], [0, 1]]}, localized=False)
    eng.disable_missing_classes({"person", "cell phone"})
    skipped = {s["type"] for s in eng.skipped}
    assert "cashier_left_during_service" in skipped
    assert "unauthorized_person_in_cash_zone" in skipped
    assert "cash_zone_door_open" in skipped


def test_cashier_left_during_service():
    client, cashier = person(150), person(20)
    timeline = [[client, cashier]] * 8 + [[client]] * 48
    eps = episodes(run(RuleEngine(RULES, WORKPLACE), timeline), "cashier_left_during_service")
    assert len(eps) == 1 and eps[0].start == 2.0


def test_cash_left_unattended():
    cash = Detection("cash", 0.4, (45, 60, 55, 70))
    timeline = [[person(20), cash]] * 8 + [[cash]] * 24
    eps = episodes(run(RuleEngine(RULES, WORKPLACE), timeline), "cash_left_unattended")
    assert len(eps) == 1 and eps[0].start == 4.0


def test_phone_in_cashier_hands_not_client_hands():
    cashier, client = person(20), person(150)
    in_cashier = Detection("cell phone", 0.4, (26, 40, 34, 50))
    in_client = Detection("cell phone", 0.6, (156, 40, 164, 50))
    eps = episodes(run(RuleEngine(RULES, WORKPLACE), [[cashier, client, in_client]] * 12), "cashier_uses_phone")
    assert not eps
    eps = episodes(run(RuleEngine(RULES, WORKPLACE), [[cashier, client, in_cashier]] * 12), "cashier_uses_phone")
    assert len(eps) == 1


def test_second_visitor_at_counter():
    cashier, client, other = person(20), person(120), person(160)
    timeline = [[cashier, client]] * 8 + [[cashier, client, other]] * 16
    eps = episodes(run(RuleEngine(RULES, WORKPLACE), timeline), "second_visitor_during_service")
    assert len(eps) == 1 and eps[0].start == 2.0 and eps[0].peak_value == 2


def test_workplace_left_with_screen_on():
    rules = {**RULES, "rules": [{**r, "enabled": True} for r in RULES["rules"]]}
    assert all(st.rule["type"] != "workplace_left_open" for st in RuleEngine(RULES, WORKPLACE).states)
    screen = Detection("laptop", 0.85, (60, 30, 90, 60))
    timeline = [[person(20), screen]] * 8 + [[screen]] * 16
    eps = episodes(run(RuleEngine(rules, WORKPLACE), timeline), "workplace_left_open")
    assert len(eps) == 1 and eps[0].start == 2.0
    closed = [[person(20), screen]] * 8 + [[]] * 16
    assert not episodes(run(RuleEngine(rules, WORKPLACE), closed), "workplace_left_open")


def test_cashier_walking_away_alone_is_not_leaving_a_client():
    timeline = [[person(20)]] * 8 + [[person(150)]] * 48
    assert not episodes(run(RuleEngine(RULES, WORKPLACE), timeline), "cashier_left_during_service")


def test_camera_obstructed():
    bright = [120.0] * 12 + [3.0] * 16 + [120.0] * 8
    eps = episodes(run(RuleEngine(RULES, WORKPLACE), [[]] * len(bright), brightness=bright), "camera_obstructed")
    assert len(eps) == 1 and eps[0].start == 3.0


def test_description_cites_rk_act_with_facts():
    rule = next(r for r in RULES["rules"] if r["type"] == "unauthorized_person_in_cash_zone")
    text, basis = build_description(rule, RULES["norms"], {"start": 8.0, "end": 16.0, "count": 2, "duration": "8.0"})
    assert "00:08.0–00:16.0" in text and "2 человека" in text
    assert "№ 14" in text and basis[0]["clause"] == "п. 8" and basis[0]["url"].startswith("https://adilet.zan.kz/")


def test_every_rule_has_quoted_norm():
    for rule in RULES["rules"]:
        assert rule.get("title") and rule.get("observed"), rule["type"]
        for ref in rule["norms"]:
            assert ref["ref"] in RULES["norms"] and ref["clause"] and ref["quote"].strip(), rule["type"]
