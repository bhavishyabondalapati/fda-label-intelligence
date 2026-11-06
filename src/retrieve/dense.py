"""A tiny exact (brute-force) dense index: one numpy matrix, cosine similarity.

This is the local stand-in for Pinecone in Phase 0. Exact search means the
baseline has no approximation error, so any later gain comes from better
chunking/retrieval, not from a quirk of the index.
"""
import numpy as np


class DenseIndex:
    def __init__(self, vectors):
        # vectors must already be L2-normalised, shape (n, dim)
        self.vectors = np.asarray(vectors, dtype="float32")

    def search(self, queries, k=5, block=64):
        """Return the indices of the top-k vectors for each query, best first.

        Queries are processed in small blocks so the score matrix stays small.
        """
        queries = np.asarray(queries, dtype="float32")
        k = min(k, len(self.vectors))
        out = []
        for i in range(0, len(queries), block):
            scores = queries[i:i + block] @ self.vectors.T          # (block, n) cosine scores
            top = np.argpartition(-scores, k - 1, axis=1)[:, :k]    # k best, unordered
            order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
            out.append(np.take_along_axis(top, order, axis=1))      # now sorted best first
        return np.vstack(out)
