import numpy as np

from app.pipeline.learned import Model, episodes, features, sample_indices
from app.pipeline.scene import Observation


def toy_model(dim=2048, threshold=0.5):
    pca = np.zeros((1, dim), np.float32)
    pca[0, 0] = 1.0
    return Model(mean=np.zeros(dim, np.float32), scale=np.ones(dim, np.float32), pca_mean=np.zeros(1, np.float32),
                 pca=pca, coef=np.array([10.0], np.float32), intercept=-5.0, threshold=threshold,
                 proto=np.array([[1.0], [-1.0]], np.float32), proto_types=["cash_left_open", "outsider_entered"])


def test_sampling_is_two_per_second():
    obs = [Observation(i * 0.25, [], b"", 100.0, 100.0, (10, 10)) for i in range(40)]
    idx = sample_indices(obs)
    assert len(idx) == 20 and obs[idx[1]].t == 0.5


def test_features_remove_the_videos_normal_state():
    t = np.arange(20, dtype=np.float32) * 0.5
    room = np.random.default_rng(0).normal(size=512).astype(np.float32)
    e = np.tile(room, (20, 1))
    f = features(t, e, e)
    assert f.shape == (20, 2048)
    assert np.allclose(f, 0, atol=1e-5)


def test_episode_found_where_score_is_high_and_typed_by_prototype():
    t = np.arange(40, dtype=np.float32) * 0.5
    x = np.zeros((40, 2048), np.float32)
    x[20:28, 0] = 1.0
    eps = episodes(toy_model(), t, x)
    assert len(eps) == 1
    ep = eps[0]
    assert 9.5 <= ep.start <= 10.5 and ep.vtype == "cash_left_open" and ep.score > 0.5


def test_no_episode_when_nothing_stands_out():
    t = np.arange(40, dtype=np.float32) * 0.5
    assert episodes(toy_model(), t, np.zeros((40, 2048), np.float32)) == []
