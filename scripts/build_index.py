"""Build the Phase 1 index THROUGH the running services (start them first: make up).

  stage 1  chunk   labels.jsonl + sections.jsonl --(chunk service)--> data/processed/chunks.sqlite
  stage 2  embed   chunk text --(embed service)--> data/processed/phase1_vectors.f32  (resumable)
  stage 3  upsert  vectors + small metadata --> Pinecone serverless index   (resumable, batched, retries)

Pinecone keeps only the vector and tiny metadata (set_id, drug, route, section, chunk_id).
The chunk text stays in SQLite and is looked up by chunk_id at query time.

Usage:
  python scripts/build_index.py                    # all three stages
  python scripts/build_index.py --smoke            # 100 vectors into a throwaway namespace, query, delete
  python scripts/build_index.py --stages upsert
"""
import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval.score import read_jsonl  # noqa: E402
from src.retrieve.pinecone_store import DIM, ensure_index, get_client, to_metadata, upsert_with_retries  # noqa: E402
from src.services import common  # noqa: E402
from src.store.chunk_store import ChunkStore  # noqa: E402

VECTORS = common.DATA_DIR / "phase1_vectors.f32"
EMBED_PROGRESS = common.DATA_DIR / "phase1_embed_progress.json"
UPSERT_PROGRESS = common.DATA_DIR / "phase1_upsert_progress.json"
LABELS_PER_CHUNK_CALL = 50
TEXTS_PER_EMBED_CALL = 256
RECORDS_PER_UPSERT = 200      # ~0.4 MB per request, well under Pinecone's 1000-record / 2 MB limit
UPSERT_THREADS = 4


def read_progress(path):
    return json.loads(path.read_text())["done"] if path.exists() else 0


def write_progress(path, done):
    path.write_text(json.dumps({"done": done}))


# ---------- stage 1 ----------
def stage_chunk(store):
    labels = {lab["set_id"]: lab for lab in read_jsonl(common.DATA_DIR / "labels.jsonl")}
    by_label = {}
    for s in read_jsonl(common.DATA_DIR / "sections.jsonl"):
        by_label.setdefault(s["set_id"], []).append({"section": s["section"], "text": s["text"]})
    store.create()
    ids = list(by_label)                                  # same label order as the Phase 0 baseline
    with httpx.Client(base_url=common.CHUNK_URL, timeout=120) as http:
        for i in tqdm(range(0, len(ids), LABELS_PER_CHUNK_CALL), desc="chunk"):
            batch = [{"set_id": sid, "drug_name": labels[sid]["drug_name"], "route": labels[sid]["route"],
                      "sections": by_label[sid]} for sid in ids[i:i + LABELS_PER_CHUNK_CALL]]
            r = http.post("/chunk", json={"labels": batch})
            r.raise_for_status()
            store.add_many(r.json()["chunks"])
    print(f"chunk store: {store.count()} chunks")


# ---------- stage 2 ----------
def stage_embed(store):
    n = store.count()
    done = read_progress(EMBED_PROGRESS)
    mode = "r+" if VECTORS.exists() and done else "w+"
    vecs = np.memmap(VECTORS, dtype="float32", mode=mode, shape=(n, DIM))
    texts = (t for _, t in store.iter_in_order())
    for _ in itertools.islice(texts, done):                   # skip texts that are already embedded
        pass
    with httpx.Client(base_url=common.EMBED_URL, timeout=300) as http, tqdm(total=n, initial=done, desc="embed") as bar:
        pos = done
        while pos < n:
            batch = list(itertools.islice(texts, TEXTS_PER_EMBED_CALL))
            r = http.post("/embed", json={"texts": batch})
            r.raise_for_status()
            vecs[pos:pos + len(batch)] = np.asarray(r.json()["vectors"], dtype="float32")
            pos += len(batch)
            bar.update(len(batch))
            if (pos // TEXTS_PER_EMBED_CALL) % 40 == 0 or pos == n:
                vecs.flush()
                write_progress(EMBED_PROGRESS, pos)
    vecs.flush()
    write_progress(EMBED_PROGRESS, n)


# ---------- stage 3 ----------
def stage_upsert(store, namespace, limit=None):
    n = limit or store.count()
    vecs = np.memmap(VECTORS, dtype="float32", mode="r", shape=(store.count(), DIM))
    pc = get_client()
    index = ensure_index(pc, common.PINECONE_INDEX)
    done = 0 if limit else read_progress(UPSERT_PROGRESS)        # smoke runs never touch real progress
    print(f"index {common.PINECONE_INDEX!r}, namespace {namespace!r}: upserting {n - done} of {n} vectors "
          f"(~{(n - done) * 2 / 1000:.0f}K write units of the 2,000K monthly allowance)")

    def make_batches(start):
        meta = store.metadata_in_order()
        for _ in itertools.islice(meta, start):               # skip rows already upserted
            pass
        pos = start
        while pos < n:
            rows = list(itertools.islice(meta, min(RECORDS_PER_UPSERT, n - pos)))
            if not rows:
                break
            yield pos, [{"id": row[0], "values": vecs[pos + k].tolist(), "metadata": to_metadata(*row)}
                        for k, row in enumerate(rows)]
            pos += len(rows)

    t0 = time.time()
    with ThreadPoolExecutor(UPSERT_THREADS) as pool, tqdm(total=n, initial=done, desc="upsert") as bar:
        window = []
        for pos, records in make_batches(done):
            window.append((pos, len(records), pool.submit(upsert_with_retries, index, records, namespace)))
            if len(window) == UPSERT_THREADS:
                done = flush(window, bar, limit)
        if window:
            done = flush(window, bar, limit)
    print(f"upsert finished in {time.time() - t0:.0f}s")
    return index, n


def flush(window, bar, limit):
    """Wait for the in-flight batches; only then advance the saved progress (so it is always safe)."""
    end = 0
    for pos, size, fut in window:
        fut.result()                                      # raises if a batch failed after all retries
        bar.update(size)
        end = pos + size
    window.clear()
    if not limit:
        write_progress(UPSERT_PROGRESS, end)
    return end


def wait_for_count(index, namespace, expected, timeout=180):
    """Pinecone stats are eventually consistent; wait until the namespace shows all vectors."""
    deadline = time.time() + timeout
    while True:
        count = index.describe_index_stats().namespaces.get(namespace, {}).get("vector_count", 0)
        if count >= expected or time.time() > deadline:
            return count
        time.sleep(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="chunk,embed,upsert")
    ap.add_argument("--smoke", action="store_true", help="upsert 100 vectors into namespace 'smoke', query, delete")
    args = ap.parse_args()
    store = ChunkStore(common.CHUNK_DB)
    stages = args.stages.split(",")

    if args.smoke:
        index, n = stage_upsert(store, "smoke", limit=100)
        got = wait_for_count(index, "smoke", n)
        first = np.asarray(np.memmap(VECTORS, dtype="float32", mode="r", shape=(store.count(), DIM))[0])
        res = index.query(vector=first.tolist(), top_k=3, include_metadata=True, namespace="smoke")
        print(f"smoke: {got} vectors stored; top-1 for vector 0 -> {res.matches[0].id} (score {res.matches[0].score:.4f})")
        print("smoke metadata:", dict(res.matches[0].metadata))
        index.delete(delete_all=True, namespace="smoke")
        print("smoke namespace deleted")
        return

    if "chunk" in stages:
        stage_chunk(store)
    if "embed" in stages:
        stage_embed(store)
    if "upsert" in stages:
        index, n = stage_upsert(store, common.PINECONE_NAMESPACE)
        got = wait_for_count(index, common.PINECONE_NAMESPACE, n)
        print(f"Pinecone namespace {common.PINECONE_NAMESPACE!r}: {got} vectors (expected {n})")
        print("index stats:", index.describe_index_stats().to_dict())


if __name__ == "__main__":
    main()
