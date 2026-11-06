"""PHASE 2 ABLATION on the DEV set (never eval-v1). Everything is local and exact (free).

Five switches, each tried alone and in every combination (2^5 = 32 configs):
  H  contextual header  "<drug> (<route>) - <section>:" prepended before embedding
  S  section-aware chunking (split on section boundaries first)
  F  query-side drug/route detection -> restrict search to that drug's chunks (a metadata filter)
  B  hybrid: BM25 + dense merged with reciprocal rank fusion
  R  cross-encoder reranker over the top 50

Selection rule (fixed BEFORE looking at results): highest dev recall@5; configs within 0.005 of the
best count as tied, and among tied configs we take the one with the fewest switches, then higher MRR.

Writes results/ablation_dev.json.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.chunk.variants import embed_text  # noqa: E402
from src.embed.local import embed_texts, load_model  # noqa: E402
from src.eval.score import load_context, read_jsonl, score_rankings  # noqa: E402
from src.eval.variants_data import load_chunks  # noqa: E402
from src.retrieve.bm25 import BM25  # noqa: E402
from src.retrieve.drug_detect import DrugDetector  # noqa: E402
from src.retrieve.fusion import rrf  # noqa: E402
from src.retrieve.rerank import load_reranker, rerank  # noqa: E402
from src.services import common  # noqa: E402
from src.eval.gold import norm_name  # noqa: E402

POOL = 100       # candidates taken from each first-stage ranking
RERANK_TOP = 50
NAMES = ["H", "S", "F", "B", "R"]


class Corpus:
    """One chunker variant: chunks + vectors (with/without header) + BM25 + filter index."""

    def __init__(self, chunker):
        self.chunker = chunker
        self.chunks = load_chunks(common.DATA_DIR, chunker)
        self.n = len(self.chunks)
        self.set_ids = [c["set_id"] for c in self.chunks]
        self.dnorm = np.array([norm_name(c["drug"]) for c in self.chunks])
        self.route = np.array([c["route"].strip().lower() for c in self.chunks])
        self.vecs, self.bm25, self.rr_cache = {}, {}, {}
        order = {}
        for i, d in enumerate(self.dnorm):
            order.setdefault(d, []).append(i)
        self.by_drug = {d: np.array(v) for d, v in order.items()}

    def vectors(self, header):
        if header not in self.vecs:
            if self.chunker == "naive" and not header:
                self.vecs[header] = np.array(np.memmap(common.DATA_DIR / "phase1_vectors.f32", dtype="float32",
                                                       mode="r", shape=(self.n, 384)))
            else:
                name = f"ablate_{self.chunker}_{'hdr' if header else 'nohdr'}.npy"
                self.vecs[header] = np.load(common.DATA_DIR / name)
        return self.vecs[header]

    def bm25_index(self, header):
        if header not in self.bm25:
            t0 = time.time()
            self.bm25[header] = BM25([embed_text(c, header) for c in self.chunks])
            print(f"  built BM25 {self.chunker}/{'hdr' if header else 'nohdr'} in {time.time() - t0:.0f}s", flush=True)
        return self.bm25[header]

    def allowed(self, det):
        if det is None:
            return None
        name, route = det
        idx = self.by_drug.get(name)
        if idx is None:
            return None
        if route:
            idx = idx[self.route[idx] == route]
            if len(idx) == 0:
                return None
        return idx


def search(corpus, cfg, question, qvec, detector, reranker, timing):
    h, s, f, b, r = cfg
    allowed = corpus.allowed(detector.detect(question)) if f else None
    V = corpus.vectors(h)
    t0 = time.perf_counter()
    if allowed is None:
        sc = V @ qvec
        top = np.argpartition(-sc, POOL - 1)[:POOL]
        dense = [int(i) for i in top[np.argsort(-sc[top])]]
    else:
        sc = V[allowed] @ qvec
        k = min(POOL, len(allowed))
        top = np.argpartition(-sc, k - 1)[:k]
        dense = [int(allowed[i]) for i in top[np.argsort(-sc[top])]]
    timing["dense"] += time.perf_counter() - t0
    ranked = dense
    if b:
        t0 = time.perf_counter()
        kw = corpus.bm25_index(h).top(question, POOL, allowed)
        ranked = rrf([dense, kw], top=POOL)
        timing["bm25"] += time.perf_counter() - t0
    if r:
        cand = ranked[:RERANK_TOP]
        ranked = rerank(reranker, question, cand, [embed_text(corpus.chunks[i], h) for i in cand]) + ranked[RERANK_TOP:]
    return ranked[:10]


def rerank_cached(corpus_h, qi_holder):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=ROOT / "data" / "eval" / "dev_questions.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "ablation_dev.json")
    args = ap.parse_args()
    assert "gold_questions" not in args.gold.name, "ablation must never run on eval-v1"

    gold = list(read_jsonl(args.gold))
    ctx = load_context(common.DATA_DIR)
    detector = DrugDetector(ctx.labels.values())
    model = load_model()
    qvecs = embed_texts(model, [g["question"] for g in gold], show_progress=False)
    reranker = load_reranker()
    corpora = {c: Corpus(c) for c in ("naive", "section")}

    # cache cross-encoder scores: many configs share candidates
    import src.retrieve.rerank as rr
    cache = {}
    real_predict = reranker.predict

    def cached_predict(pairs, **kw):
        miss = [p for p in pairs if p not in cache]
        if miss:
            for p, v in zip(miss, real_predict(miss, **kw)):
                cache[p] = float(v)
        return [cache[p] for p in pairs]
    reranker.predict = cached_predict

    results = []
    for cfg in itertools.product([0, 1], repeat=5):
        h, s, f, b, r = cfg
        corpus = corpora["section" if s else "naive"]
        timing = {"dense": 0.0, "bm25": 0.0}
        rankings = []
        for g, qv in zip(gold, qvecs):
            top = search(corpus, cfg, g["question"], qv, detector, reranker, timing)
            rankings.append([(corpus.set_ids[i], corpus.chunks[i]["start"], corpus.chunks[i]["end"]) for i in top])
        sc = score_rankings(gold, rankings, ctx)
        row = {"config": "".join(n for n, on in zip(NAMES, cfg) if on) or "baseline", "switches": list(cfg),
               **sc["relaxed"], "strict_recall_at_5": sc["strict"]["recall_at_5"],
               "hit5": [bool(any(x[:5])) for x in sc["rel_flags"]],
               "bm25_ms_per_query": round(timing["bm25"] / len(gold) * 1000, 2)}
        results.append(row)
        print(f"{row['config']:10s} R@1 {row['recall_at_1']:.4f}  R@5 {row['recall_at_5']:.4f}  MRR {row['mrr_at_10']:.4f}", flush=True)

    # micro-benchmark of the two extra stages, uncached (ms per query)
    corpus = corpora["naive"]
    bm = corpus.bm25_index(False)
    t0 = time.perf_counter()
    for g in gold[:100]:
        bm.top(g["question"], POOL)
    bm_ms = (time.perf_counter() - t0) / 100 * 1000
    pairs_q = [(g["question"], corpus.chunks[i]["text"]) for g in gold[:30] for i in range(RERANK_TOP)]
    real_predict(pairs_q[:50], batch_size=50, show_progress_bar=False)       # warm up
    t0 = time.perf_counter()
    for j in range(0, len(pairs_q), 50):
        real_predict(pairs_q[j:j + 50], batch_size=50, show_progress_bar=False)
    rr_ms = (time.perf_counter() - t0) / 30 * 1000
    t0 = time.perf_counter()
    for g in gold:
        detector.detect(g["question"])
    det_ms = (time.perf_counter() - t0) / len(gold) * 1000

    json.dump({"set": args.gold.name, "n_questions": len(gold), "results": results,
               "stage_ms_per_query": {"bm25": round(bm_ms, 1), "rerank_top50": round(rr_ms, 1),
                                      "drug_detect": round(det_ms, 3)},
               "run_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
