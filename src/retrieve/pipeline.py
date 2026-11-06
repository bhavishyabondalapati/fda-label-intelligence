"""The Phase 2 search pipeline, used by the query service. Each stage is a switch:

    question --embed--> [drug detect -> Pinecone metadata filter]  --dense top 100--+
                        [BM25 over the same chunks]                --keyword top 100-+--> RRF
    --> [cross-encoder rerank of the top 50] --> top_k chunks (text from the local SQLite store)

The same stage code runs in the ablation (scripts/ablate.py uses numpy instead of Pinecone for the
dense step); here the dense step is Pinecone, as in production.
"""
import time

import numpy as np

from src.chunk.variants import embed_text
from src.eval.gold import norm_name
from src.retrieve.fusion import rrf

POOL = 100
RERANK_TOP = 50


class Pipeline:
    def __init__(self, embed_fn, index, namespace, store, header=True, use_filter=True, use_hybrid=True,
                 use_rerank=True, detector=None, bm25=None, reranker=None, rows=None):
        self.embed_fn, self.index, self.namespace, self.store = embed_fn, index, namespace, store
        self.header, self.use_filter, self.use_hybrid, self.use_rerank = header, use_filter, use_hybrid, use_rerank
        self.detector, self.bm25, self.reranker = detector, bm25, reranker
        self.ids = [r["chunk_id"] for r in rows] if rows is not None else []
        self.by_drug = {}                                    # drug_norm -> {route_lower: np.array of positions}
        if rows is not None and use_filter:
            tmp = {}
            for i, r in enumerate(rows):
                tmp.setdefault(norm_name(r["drug"]), {}).setdefault(r["route"].strip().lower(), []).append(i)
            self.by_drug = {d: {rt: np.array(v) for rt, v in routes.items()} for d, routes in tmp.items()}

    def _allowed(self, det):
        routes = self.by_drug.get(det[0])
        if not routes:
            return None
        if det[1] and det[1] in routes:
            return routes[det[1]]
        return np.concatenate(list(routes.values()))

    def search(self, question, top_k=10):
        """-> (results, timings_ms). results: list of row dicts (chunk_id, set_id, ..., text) + score."""
        t = {}
        t0 = time.perf_counter()
        vec = self.embed_fn(question)
        t["embed"] = time.perf_counter() - t0

        det = self.detector.detect(question) if (self.use_filter and self.detector) else None
        flt = None
        if det:
            flt = {"drug_norm": {"$eq": det[0]}}
            if det[1] and det[1] in self.by_drug.get(det[0], {}):
                flt["route_norm"] = {"$eq": det[1]}
        t0 = time.perf_counter()
        need = POOL if (self.use_hybrid or self.use_rerank) else top_k
        kw = {"vector": vec, "top_k": need, "include_metadata": False, "namespace": self.namespace}
        if flt:
            kw["filter"] = flt
        res = self.index.query(**kw)
        dense = [m.id for m in res.matches]
        scores = {m.id: float(m.score) for m in res.matches}
        t["pinecone"] = time.perf_counter() - t0

        ranked = dense
        if self.use_hybrid and self.bm25 is not None:
            t0 = time.perf_counter()
            allowed = self._allowed(det) if det else None
            hits = [self.ids[i] for i in self.bm25.top(question, POOL, allowed)]
            ranked = rrf([dense, hits], top=POOL)
            t["bm25"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        n_text = max(top_k, RERANK_TOP) if self.use_rerank else top_k
        rows = self.store.get_many(ranked[:n_text])
        t["lookup"] = time.perf_counter() - t0
        ranked = [c for c in ranked[:n_text] if c in rows]

        if self.use_rerank and self.reranker is not None and ranked:
            t0 = time.perf_counter()
            texts = [embed_text(rows[c], self.header) for c in ranked[:RERANK_TOP]]
            from src.retrieve.rerank import rerank
            ranked = rerank(self.reranker, question, ranked[:RERANK_TOP], texts) + ranked[RERANK_TOP:]
            t["rerank"] = time.perf_counter() - t0

        out = [{"chunk_id": c, "score": scores.get(c), **{k: v for k, v in rows[c].items() if k != "chunk_id"}}
               for c in ranked[:top_k]]
        t["total"] = sum(t.values())
        return out, {k: round(v * 1000, 3) for k, v in t.items()}, det
