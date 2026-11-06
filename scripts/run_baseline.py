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
from src.eval.gold import drug_key  # noqa: E402
from src.eval.metrics import chunk_hits_gold, chunk_hits_relevant, mrr, recall_at_k  # noqa: E402
from src.retrieve.dense import DenseIndex  # noqa: E402

K = 10          # we retrieve 10 so MRR@10 can be computed; recall@5 uses the first 5


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--gold", type=Path, default=ROOT / "data" / "eval" / "gold_questions.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "baseline.json")
    ap.add_argument("--inspect", help="also print failing examples for this section (e.g. boxed_warning)")
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

    # 3. Answer the gold questions.
    gold = list(read_jsonl(args.gold))
    labels = {l["set_id"]: l for l in read_jsonl(args.data_dir / "labels.jsonl")}
    key_of = {sid: drug_key(l) for sid, l in labels.items()}     # (normalized name, route)
    q_vecs = embed_texts(model, [g["question"] for g in gold], show_progress=False)
    index = DenseIndex(vectors)
    t0 = time.time()
    top = index.search(q_vecs, k=K)
    search_ms_per_query = round((time.time() - t0) / len(gold) * 1000, 2)

    # 4. Score. Two rules (see PLAN.md, 2026-11-05 note):
    #    RELAXED (primary): chunk is from ANY label with the same drug name + route as the
    #                       gold label and covers that label's copy of the gold section.
    #    STRICT (cross-check): chunk is from the exact gold label and covers the gold section.
    gold_key = {g["qid"]: key_of[g["set_id"]] for g in gold}
    rel_flags, strict_flags, per_section = [], [], defaultdict(list)
    for g, row in zip(gold, top):
        gspan = label_spans[g["set_id"]][g["section"]]
        r_row, s_row = [], []
        for idx in row:
            set_id, start, end = chunk_meta[idx]
            own_span = label_spans[set_id].get(g["section"])
            r_row.append(chunk_hits_relevant(key_of[set_id], start, end, gold_key[g["qid"]], own_span))
            s_row.append(chunk_hits_gold(set_id, start, end, g["set_id"], gspan))
        rel_flags.append(r_row)
        strict_flags.append(s_row)
        per_section[g["section"]].append(any(r_row[:5]))

    def summary(flags):
        return {"recall_at_1": round(recall_at_k([f[:1] for f in flags]), 4),
                "recall_at_5": round(recall_at_k([f[:5] for f in flags]), 4),
                "mrr_at_10": round(mrr(flags), 4),
                "n_hits_at_5": int(sum(any(f[:5]) for f in flags))}

    # How many labels in the corpus count as relevant for a typical question (context only).
    key_counts = defaultdict(int)
    for sid, k in key_of.items():
        key_counts[k] += 1
    result = {
        "system": "baseline",
        "description": "naive fixed-size chunks, all-MiniLM-L6-v2, dense only, local exact numpy index",
        "relevance_rule": "relaxed (primary): same normalized drug name + route as the gold label, chunk covers "
                          "that label's copy of the gold section",
        **summary(rel_flags),
        "strict_crosscheck": {"rule": "exact gold set_id, chunk covers gold section", **summary(strict_flags)},
        "n_questions": len(gold),
        "mean_relevant_labels_per_question": round(sum(key_counts[gold_key[g["qid"]]] for g in gold) / len(gold), 2),
        "per_section_recall_at_5": {s: round(sum(v) / len(v), 3) for s, v in sorted(per_section.items())},
        "per_section_n": {s: len(v) for s, v in sorted(per_section.items())},
        "n_labels": len(sections),
        "n_chunks": len(chunk_texts),
        "config": {
            "chunk_size_chars": CHUNK_SIZE,
            "overlap_chars": OVERLAP,
            "embedding_model": MODEL_NAME,
            "embedding_dim": int(vectors.shape[1]),
            "k_retrieved": K,
            "min_overlap_chars": "min(200, section length)",
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
    r, st = result, result["strict_crosscheck"]
    print(f"\nRELAXED  recall@1 = {r['recall_at_1']}  recall@5 = {r['recall_at_5']} "
          f"({r['n_hits_at_5']}/{len(gold)})  MRR@10 = {r['mrr_at_10']}")
    print(f"STRICT   recall@1 = {st['recall_at_1']}  recall@5 = {st['recall_at_5']} "
          f"({st['n_hits_at_5']}/{len(gold)})  MRR@10 = {st['mrr_at_10']}   chunks = {len(chunk_texts)}")
    print(f"wrote {args.out}")

    # Sanity check: top-3 chunks for a few questions (hit = relaxed rule).
    import random
    sec_of = lambda set_id, start, end: [  # which sections does this chunk overlap?
        n for n, (a, b) in label_spans[set_id].items() if min(end, b) - max(start, a) > 0]

    def show(i):
        g = gold[i]
        print(f"\n=== {g['qid']}: {g['question']}")
        print(f"    GOLD: {g['drug_name']} [{g['route']}] / {g['section']}   hit@5={any(rel_flags[i][:5])}")
        for rank, idx in enumerate(top[i][:3], start=1):
            set_id, start, end = chunk_meta[idx]
            snippet = chunk_texts[idx][:150].replace("\n", " ")
            print(f"    #{rank} {'HIT ' if rel_flags[i][rank-1] else 'miss'} {labels[set_id]['drug_name']} "
                  f"[{labels[set_id]['route']}] [{','.join(sec_of(set_id, start, end))}]: {snippet}")

    for i in random.Random(0).sample(range(len(gold)), 5):
        show(i)
    if args.inspect:
        fails = [i for i, g in enumerate(gold) if g["section"] == args.inspect and not any(rel_flags[i][:5])]
        n_sec = sum(g["section"] == args.inspect for g in gold)
        print(f"\n##### {args.inspect}: {len(fails)} of {n_sec} questions fail; showing 2 failing examples")
        for i in fails[:2]:
            g = gold[i]
            gs, ge = label_spans[g["set_id"]][g["section"]]
            gold_text = next(x["text"] for x in sections[g["set_id"]] if x["section"] == g["section"])
            print(f"\n    gold section is chars {gs}-{ge} of its label (label total {len(join_label(sections[g['set_id']])[0])}); "
                  f"its text starts: {gold_text[:150]!r}")
            show(i)


if __name__ == "__main__":
    main()
