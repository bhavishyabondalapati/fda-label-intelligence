"""PARITY CHECK: run the frozen eval-v1 questions through the HTTP query service (Pinecone) and
compare with the Phase 0 local exact-search baseline.

Why they can differ:
  * Pinecone uses approximate nearest-neighbour (ANN) search; the local baseline checks every
    vector exactly. ANN can miss a few true neighbours.
  * Vectors were embedded twice (Phase 0 in-process, Phase 1 through the embed service), so
    floating-point noise could reorder near-ties.
To tell these apart we also run EXACT search on the Phase 1 vectors and report all three.

Needs: services running (make up) and the index built (scripts/build_index.py).
Writes results/parity_phase1.json
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval.score import load_context, read_jsonl, score_rankings  # noqa: E402
from src.retrieve.dense import DenseIndex  # noqa: E402
from src.services import common  # noqa: E402
from src.store.chunk_store import ChunkStore  # noqa: E402

K = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-url", default="http://127.0.0.1:8004")
    ap.add_argument("--gold", type=Path, default=ROOT / "data" / "eval" / "gold_questions.jsonl")
    ap.add_argument("--baseline", type=Path, default=ROOT / "results" / "baseline.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "parity_phase1.json")
    args = ap.parse_args()

    gold = list(read_jsonl(args.gold))
    ctx = load_context(common.DATA_DIR)
    store = ChunkStore(common.CHUNK_DB)
    baseline = json.loads(args.baseline.read_text())

    # 1. Ask the service (embed -> Pinecone ANN -> text lookup) for every gold question.
    ann_ids, ann_rank = [], []
    with httpx.Client(base_url=args.query_url, timeout=60) as http:
        for g in gold:
            r = http.post("/query", json={"question": g["question"], "top_k": K})
            r.raise_for_status()
            res = r.json()["results"]
            ann_ids.append([x["chunk_id"] for x in res])
            ann_rank.append([(x["set_id"], x["start"], x["end"]) for x in res])
    service = score_rankings(gold, ann_rank, ctx)

    # 2. Exact search with the SAME question vectors, on (a) the Phase 0 vectors and (b) the Phase 1 vectors.
    chunk_ids = [cid for cid, _ in store.iter_in_order()]
    n = len(chunk_ids)
    with httpx.Client(base_url=common.EMBED_URL, timeout=120) as http:
        q = np.asarray(http.post("/embed", json={"texts": [g["question"] for g in gold]}).json()["vectors"], dtype="float32")
    v0 = np.load(sorted(glob.glob(str(common.DATA_DIR / "baseline_embeddings_*.npy")))[-1])
    v1 = np.asarray(np.memmap(common.DATA_DIR / "phase1_vectors.f32", dtype="float32", mode="r", shape=(n, v0.shape[1])))
    assert v0.shape == v1.shape, "chunk counts differ between Phase 0 and Phase 1"

    exact = {}
    for name, vecs in (("exact_phase0_vectors", v0), ("exact_phase1_vectors", v1)):
        top = DenseIndex(vecs).search(q, k=K)
        exact[name] = [[chunk_ids[i] for i in row] for row in top]
    need = {cid for rows in exact.values() for row in rows for cid in row}
    rows = store.get_many(sorted(need))      # chunk positions (start, end) come from the store
    scored_exact = {}
    for name, id_rows in exact.items():
        ranking = [[(rows[c]["set_id"], rows[c]["start"], rows[c]["end"]) for c in row] for row in id_rows]
        scored_exact[name] = score_rankings(gold, ranking, ctx)

    def overlap(a_rows, b_rows):   # mean share of the top-10 that the two systems agree on
        return float(np.mean([len(set(a) & set(b)) / K for a, b in zip(a_rows, b_rows)]))

    def top5_overlap(a_rows, b_rows):
        return float(np.mean([len(set(a[:5]) & set(b[:5])) / 5 for a, b in zip(a_rows, b_rows)]))

    # 3. How different are the two embedding runs?
    diff = np.abs(v0 - v1)
    cos = (v0 * v1).sum(axis=1)       # both are unit vectors, so the dot product is the cosine

    base = {m: baseline[m] for m in ("recall_at_1", "recall_at_5", "mrr_at_10", "n_hits_at_5")}
    svc = service["relaxed"]
    differing = [g["qid"] for g, a, b in zip(gold, service["rel_flags"], scored_exact["exact_phase0_vectors"]["rel_flags"])
                 if any(a[:5]) != any(b[:5])]
    out = {
        "what": "eval-v1 through the HTTP query service (Pinecone ANN) vs the Phase 0 local exact baseline",
        "n_questions": len(gold),
        "relaxed_primary": {
            "local_exact_baseline (results/baseline.json)": base,
            "pinecone_service": svc,
            "exact_search_on_phase1_vectors": scored_exact["exact_phase1_vectors"]["relaxed"],
            "delta_service_minus_baseline": {m: round(svc[m] - base[m], 4) for m in base},
        },
        "strict_crosscheck": {
            "local_exact_baseline": baseline["strict_crosscheck"],
            "pinecone_service": service["strict"],
        },
        "questions_where_hit_at_5_differs_service_vs_exact": differing,
        "ann_vs_exact": {
            "mean_top10_overlap_pinecone_vs_exact_phase1_vectors": round(overlap(ann_ids, exact["exact_phase1_vectors"]), 4),
            "mean_top5_overlap_pinecone_vs_exact_phase1_vectors": round(top5_overlap(ann_ids, exact["exact_phase1_vectors"]), 4),
            "mean_top10_overlap_pinecone_vs_exact_phase0_vectors": round(overlap(ann_ids, exact["exact_phase0_vectors"]), 4),
            "mean_top10_overlap_exact_phase0_vs_exact_phase1": round(
                overlap(exact["exact_phase0_vectors"], exact["exact_phase1_vectors"]), 4),
        },
        "embedding_runs": {
            "n_vectors": n,
            "max_abs_difference_phase0_vs_phase1": float(diff.max()),
            "min_cosine_phase0_vs_phase1": float(cos.min()),
            "mean_cosine_phase0_vs_phase1": float(cos.mean()),
        },
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("relaxed_primary", "ann_vs_exact", "embedding_runs")}, indent=2))
    print("questions whose hit@5 differs (service vs exact):", differing)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
