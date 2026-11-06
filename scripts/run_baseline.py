"""BASELINE retrieval: naive fixed-size chunks + all-MiniLM-L6-v2 + dense-only + local index.

Reads  data/processed/{labels,sections}.jsonl and data/eval/gold_questions.jsonl
Writes results/baseline.json  (recall@5, per-section breakdown, config, chunk count)

Chunk embeddings are cached in data/processed/ (gitignored) so a re-run takes seconds.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.chunk.naive import CHUNK_SIZE, OVERLAP, join_label, naive_chunks  # noqa: E402
from src.eval.metrics import chunk_hits_gold, recall_at_k  # noqa: E402
from src.retrieve.dense import DenseIndex  # noqa: E402

K = 5


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--gold", type=Path, default=ROOT / "data" / "eval" / "gold_questions.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "baseline.json")
    args = ap.parse_args()

    # 1. Naive chunking over ALL labels (the index holds the whole corpus).
    sections = defaultdict(list)
    for s in read_jsonl(args.data_dir / "sections.jsonl"):
        sections[s["set_id"]].append(s)

    chunk_texts, chunk_meta = [], []        # meta: (set_id, start, end)
    label_spans = {}                        # set_id -> {section: (start, end)}
    for set_id, secs in sections.items():
        full_text, spans = join_label(secs)
        label_spans[set_id] = spans
        for start, end, text in naive_chunks(full_text, CHUNK_SIZE, OVERLAP):
            chunk_texts.append(text)
            chunk_meta.append((set_id, start, end))
    print(f"{len(sections)} labels -> {len(chunk_texts)} chunks")

    # 2. Embed chunks (cached on disk, keyed by the chunk contents + settings).
    from src.embed.local import MODEL_NAME, embed_texts, load_model, pick_device
    fingerprint = hashlib.sha256(
        (MODEL_NAME + f"{CHUNK_SIZE}:{OVERLAP}:{len(chunk_texts)}:" + "".join(chunk_texts[:50])
         + "".join(chunk_texts[-50:])).encode()).hexdigest()[:16]
    cache = args.data_dir / f"baseline_embeddings_{fingerprint}.npy"
    device = pick_device()
    model = load_model(device)
    t0 = time.time()
    if cache.exists():
        vectors = np.load(cache)
        print(f"loaded cached embeddings {cache.name}")
    else:
        vectors = embed_texts(model, chunk_texts)
        np.save(cache, vectors)
    embed_seconds = round(time.time() - t0, 1)

    # 3. Answer the gold questions and score recall@5.
    gold = list(read_jsonl(args.gold))
    q_vecs = embed_texts(model, [g["question"] for g in gold], show_progress=False)
    index = DenseIndex(vectors)
    t0 = time.time()
    top = index.search(q_vecs, k=K)
    search_ms_per_query = round((time.time() - t0) / len(gold) * 1000, 2)

    hit_flags, per_section = [], defaultdict(list)
    for g, row in zip(gold, top):
        span = label_spans[g["set_id"]][g["section"]]
        flags = []
        for idx in row:
            set_id, start, end = chunk_meta[idx]
            flags.append(chunk_hits_gold(set_id, start, end, g["set_id"], span))
        hit_flags.append(flags)
        per_section[g["section"]].append(any(flags))

    recall = recall_at_k(hit_flags)
    result = {
        "system": "baseline",
        "description": "naive fixed-size chunks, all-MiniLM-L6-v2, dense only, local exact numpy index",
        "metric": f"recall@{K}",
        "recall_at_5": round(recall, 4),
        "n_questions": len(gold),
        "n_hits": int(sum(any(f) for f in hit_flags)),
        "per_section_recall": {s: round(sum(v) / len(v), 3) for s, v in sorted(per_section.items())},
        "per_section_n": {s: len(v) for s, v in sorted(per_section.items())},
        "n_labels": len(sections),
        "n_chunks": len(chunk_texts),
        "config": {
            "chunk_size_chars": CHUNK_SIZE,
            "overlap_chars": OVERLAP,
            "embedding_model": MODEL_NAME,
            "embedding_dim": int(vectors.shape[1]),
            "k": K,
            "hit_rule": "same set_id and chunk overlaps gold section by >= min(200, section length) chars",
        },
        "timing": {"embed_seconds": embed_seconds, "search_ms_per_query": search_ms_per_query, "device": device},
        "manifest": json.loads((args.data_dir / "manifest.json").read_text()),
        "gold_file_sha256": hashlib.sha256(args.gold.read_bytes()).hexdigest(),
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": platform.platform(),
    }
    result["manifest"].pop("raw_file_sha256", None)   # keep the results file short
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nrecall@5 = {recall:.4f}  ({result['n_hits']}/{len(gold)})   chunks = {len(chunk_texts)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
