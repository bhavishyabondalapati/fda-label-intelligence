"""LATENCY BENCHMARK (PLAN.md metric M3): 500 queries through the HTTP query service.

Method (stated so the number can be trusted):
  * Fresh start: `make restart` right before this script, so the first queries are really cold
    (embed model's first GPU call, first Pinecone connection, cold file cache).
  * 520 DISTINCT queries: the 200 frozen eval-v1 questions + 320 more built from the same
    templates over other labels (fixed seed), shuffled. The first 20 are the COLD set; the
    remaining 500 are the timed WARM set. (Latency does not depend on whether answers are right.)
  * One client, one request at a time (concurrency 1), a persistent HTTP connection.
  * End-to-end = the client's wall-clock time from sending the HTTP request to having parsed the
    JSON reply. It includes query embedding (a second HTTP hop to the embed service), the Pinecone
    call over the internet, and the SQLite text lookup. The service reports its own breakdown.

Writes results/latency_phase1.json
"""
import argparse
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval.gold import TEMPLATES  # noqa: E402
from src.eval.score import read_jsonl  # noqa: E402
from src.services import common  # noqa: E402

THRESHOLD_MS = 3000


def build_queries(n_total, seed):
    """200 frozen gold questions + enough templated ones to reach n_total distinct queries."""
    gold = [g["question"] for g in read_jsonl(ROOT / "data" / "eval" / "gold_questions.jsonl")]
    labels = [lab for lab in read_jsonl(common.DATA_DIR / "labels.jsonl") if lab["split"] != "retrieval_eval"]
    rng = random.Random(seed)
    seen = set(gold)
    extra = []
    while len(gold) + len(extra) < n_total:
        lab = rng.choice(labels)
        q = rng.choice(TEMPLATES[rng.choice(sorted(TEMPLATES))]).format(d=lab["drug_name"])
        if q not in seen:
            seen.add(q)
            extra.append(q)
    queries = gold + extra
    rng.shuffle(queries)
    return queries


def stats(values):
    a = np.asarray(values, dtype=float)
    return {"p50": round(float(np.percentile(a, 50)), 1), "p95": round(float(np.percentile(a, 95)), 1),
            "p99": round(float(np.percentile(a, 99)), 1), "mean": round(float(a.mean()), 1),
            "max": round(float(a.max()), 1)}


def run(http, queries):
    rows = []
    for q in queries:
        t0 = time.perf_counter()
        r = http.post("/query", json={"question": q, "top_k": 10})
        r.raise_for_status()
        body = r.json()
        wall = (time.perf_counter() - t0) * 1000
        rows.append({"end_to_end": wall, **body["timings_ms"]})
    return rows


def summarize(rows):
    comp = {"end_to_end_client": stats([r["end_to_end"] for r in rows])}
    comp["service_total"] = stats([r["total"] for r in rows])
    for name, key in (("query_embedding", "embed"), ("pinecone_call", "pinecone"), ("text_lookup", "lookup"),
                      ("bm25", "bm25"), ("rerank", "rerank")):        # bm25 / rerank exist only in the Phase 2 pipeline
        if key in rows[0]:
            comp[name] = stats([r[key] for r in rows])
    return comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8004")
    ap.add_argument("--warm", type=int, default=500)
    ap.add_argument("--cold", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "latency_phase1.json")
    ap.add_argument("--namespace", default=None, help="Pinecone namespace to count vectors in (default: Phase 1's)")
    args = ap.parse_args()

    queries = build_queries(args.cold + args.warm, args.seed)
    assert len(set(queries)) == len(queries) == args.cold + args.warm

    # vector count in the index (the claim needs >= 100,000)
    from src.retrieve.pinecone_store import get_client
    index = get_client().Index(common.PINECONE_INDEX)
    namespace = args.namespace or common.PINECONE_NAMESPACE
    n_vectors = index.describe_index_stats().namespaces.get(namespace, {}).get("vector_count", 0)

    with httpx.Client(base_url=args.url, timeout=60) as http:
        cold_rows = run(http, queries[:args.cold])
        warm_rows = run(http, queries[args.cold:])

    warm = summarize(warm_rows)
    p95 = warm["end_to_end_client"]["p95"]
    out = {
        "metric": "M3: p95 end-to-end latency over 500 queries (PLAN.md)",
        "threshold_ms": THRESHOLD_MS,
        "p95_ms": p95,
        "passes_threshold": bool(p95 < THRESHOLD_MS and n_vectors >= 100_000),
        "vectors_in_index": n_vectors,
        "index": common.PINECONE_INDEX,
        "namespace": namespace,
        "pinecone_region": "aws us-east-1 (Starter plan only allows this region)",
        "n_timed_queries": len(warm_rows),
        "n_cold_queries": len(cold_rows),
        "concurrency": 1,
        "top_k": 10,
        "warm": warm,
        "cold": {
            "first_query_ms": round(cold_rows[0]["end_to_end"], 1),
            "first_query_breakdown_ms": {k: round(cold_rows[0][k], 1) for k in ("embed", "pinecone", "lookup")},
            "first_20_end_to_end": stats([r["end_to_end"] for r in cold_rows]),
            **{k: v for k, v in summarize(cold_rows).items() if k != "end_to_end_client"},
        },
        "path": "client -> query service -> embed service (HTTP) -> Pinecone (internet) [-> BM25 -> rerank in Phase 2] -> SQLite lookup",
        "health": httpx.get(f"{args.url}/health", timeout=5).json(),
        "machine": f"{platform.platform()}, {platform.machine()}",
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("p95_ms", "passes_threshold", "vectors_in_index", "warm", "cold")}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
