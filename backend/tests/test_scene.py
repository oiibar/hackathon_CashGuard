from app.pipeline.detector import Detection
from app.pipeline.scene import Observation, drop_fixtures, infer_zones, static_fixtures

W, H = 1280, 720


def obs(t, dets):
    return Observation(t, dets, b"", 120.0, 500.0, (W, H))


def person(x, y, w=160, h=300):
    return Detection("person", 0.9, (x, y, x + w, y + h))


def test_workplace_is_where_a_person_stays():
    cashier = person(200, 300)
    frames = []
    for i in range(40):
        dets = [cashier]
        if 10 <= i < 25:
            dets.append(person(800 + i * 5, 100))
        frames.append(obs(i * 0.25, dets))
    res = infer_zones(frames)
    assert res.localized
    (x1, y1), _, (x2, y2), _ = res.zones["cash_zone"]
    cx, cy = (200 + 80) / W, (300 + 150) / H
    assert x1 < cx < x2 and y1 < cy < y2
    assert x2 < 800 / W


def test_no_stationary_person_falls_back_to_full_frame():
    frames = [obs(i * 0.25, [person(40 * i, 100)]) for i in range(30)]
    res = infer_zones(frames)
    assert not res.localized and res.zones["cash_zone"] == [[0, 0], [1, 0], [1, 1], [0, 1]]


def test_static_terminal_is_not_a_phone():
    terminal = Detection("cell phone", 0.6, (600, 500, 640, 560))
    real_phone = Detection("cell phone", 0.7, (300, 300, 330, 350))
    frames = [obs(i * 0.25, [terminal] + ([real_phone] if 10 <= i < 16 else [])) for i in range(40)]
    fixtures = static_fixtures(frames, {"cell phone"})
    assert len(fixtures["cell phone"]) == 1
    drop_fixtures(frames, fixtures)
    assert all(terminal not in o.dets for o in frames)
    assert sum(real_phone in o.dets for o in frames) == 6


def test_background_people_and_duplicates_are_dropped():
    from app.pipeline.scene import dedupe_people, drop_background_people

    cashier = person(200, 300)
    behind_glass = Detection("person", 0.45, (10, 100, 60, 190))
    upper_body = Detection("person", 0.6, (210, 300, 350, 450))
    frames = [obs(i * 0.25, [cashier, behind_glass, upper_body]) for i in range(10)]
    drop_background_people(frames)
    dedupe_people(frames)
    assert all(o.dets == [cashier] for o in frames)
