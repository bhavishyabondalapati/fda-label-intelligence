"""BM25 keyword search over all chunks, built with scipy sparse matrices (fast for 300K+ chunks).

BM25 scores a chunk by how many query words it contains, weighting rare words more (idf) and
discounting very long chunks. It catches exact words (drug names, "boxed warning") that dense
embeddings sometimes blur.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

K1, B = 1.5, 0.75


class BM25:
    def __init__(self, texts):
        self.vec = CountVectorizer(lowercase=True, stop_words="english", token_pattern=r"(?u)\b[a-z0-9][a-z0-9\-]+\b")
        tf = self.vec.fit_transform(texts).tocsr().astype("float32")        # docs x terms
        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype("float32")
        dl = np.asarray(tf.sum(axis=1)).ravel()
        norm = K1 * (1 - B + B * dl / dl.mean())                              # per-doc length factor
        tf = tf.tocoo()
        data = tf.data * (K1 + 1) / (tf.data + norm[tf.row]) * self.idf[tf.col]
        self.w = sp.csc_matrix((data.astype("float32"), (tf.row, tf.col)), shape=tf.shape)   # column slices are fast
        self.n = n_docs

    def scores(self, query, allowed=None):
        """Score of every chunk for one query (zeros for chunks with no query word)."""
        terms = [self.vec.vocabulary_[t] for t in self.vec.build_analyzer()(query) if t in self.vec.vocabulary_]
        if not terms:
            return np.zeros(self.n, dtype="float32")
        return np.asarray(self.w[:, terms].sum(axis=1)).ravel()

    def top(self, query, k=100, allowed=None):
        s = self.scores(query)
        if allowed is not None:
            m = np.full(self.n, -np.inf, dtype="float32"); m[allowed] = s[allowed]; s = m
        k = min(k, self.n)
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.argsort(-s[idx])]
        return [int(i) for i in idx if s[i] > 0]
