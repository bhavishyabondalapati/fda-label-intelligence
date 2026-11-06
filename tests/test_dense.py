import numpy as np

from src.retrieve.dense import DenseIndex


def unit(v):
    v = np.array(v, dtype="float32")
    return v / np.linalg.norm(v)


def test_search_returns_best_first():
    vecs = np.stack([unit([1, 0]), unit([0.9, 0.1]), unit([0, 1]), unit([-1, 0])])
    idx = DenseIndex(vecs)
    top = idx.search(np.stack([unit([1, 0])]), k=3)
    assert top.tolist() == [[0, 1, 2]]


def test_search_handles_k_larger_than_index_and_many_queries():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(10, 8)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    idx = DenseIndex(vecs)
    top = idx.search(vecs, k=50, block=3)           # small block forces multiple batches
    assert top.shape == (10, 10)
    assert top[:, 0].tolist() == list(range(10))    # each vector's nearest neighbour is itself
