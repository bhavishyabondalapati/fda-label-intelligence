"""Build the Phase 2 index for the CHOSEN config (from results/ablation_dev.json selection).

  1. writes data/processed/pipeline_v2.json (which switches the query service turns on)
  2. writes the chosen chunker's chunks into data/processed/chunks_v2.sqlite (text stays local)
  3. uploads the vectors (already embedded locally during the ablation, so no re-embedding) to a NEW
     Pinecone namespace "v2" with metadata incl. drug_norm / route_norm for filtering.
     Resumable. Prints the write-unit estimate first. Add --upsert to really upload.

Usage: python scripts/build_index_v2.py --config HSFBR [--upsert]    (letters: H S F B R, see ablate.py)
"""
import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval.variants_data import load_chunks  # noqa: E402
from src.retrieve.pinecone_store import ensure_index, get_client, to_metadata_v2, upsert_with_retries  # noqa: E402
from src.services import common  # noqa: E402
from src.store.chunk_store import ChunkStore  # noqa: E402

BATCH, THREADS = 200, 4
WU_PER_VECTOR = 2            # measured in Phase 1: 647K write units for 323K vectors
MONTHLY_WU = 2_000_000
PROGRESS = common.DATA_DIR / "phase2_upsert_progress.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="letters among H S F B R")
    ap.add_argument("--upsert", action="store_true")
    ap.add_argument("--used-wu", type=int, default=650_000, help="write units already used this month")
    args = ap.parse_args()
    on = set(args.config)
    cfg = {"header": "H" in on, "section_chunks": "S" in on, "filter": "F" in on, "hybrid": "B" in on, "rerank": "R" in on}
    chunker = "section" if cfg["section_chunks"] else "naive"
    vec_path = (common.DATA_DIR / "phase1_vectors.f32") if (chunker == "naive" and not cfg["header"]) else \
        common.DATA_DIR / f"ablate_{chunker}_{'hdr' if cfg['header'] else 'nohdr'}.npy"

    chunks = load_chunks(common.DATA_DIR, chunker)
    n = len(chunks)
    est = n * WU_PER_VECTOR
    print(f"config {args.config}: chunker={chunker}, {n} chunks -> ~{est/1000:.0f}K write units "
          f"({est/MONTHLY_WU:.0%} of the monthly 2,000K); already used ~{args.used_wu/1000:.0f}K; "
          f"left after: ~{(MONTHLY_WU - args.used_wu - est)/1000:.0f}K")

    common.PIPELINE_CONFIG.write_text(json.dumps(cfg, indent=1))
    store = ChunkStore(common.CHUNK_DB_V2)
    if not (store.path.exists() and store.count() == n):
        store.create()
        for i in range(0, n, 5000):
            store.add_many(chunks[i:i + 5000])
    print(f"chunk store v2: {store.count()} chunks")

    if not args.upsert:
        print("dry run: add --upsert to upload")
        return
    vecs = np.load(vec_path, mmap_mode="r") if vec_path.suffix == ".npy" else \
        np.memmap(vec_path, dtype="float32", mode="r", shape=(n, 384))
    assert vecs.shape == (n, 384), (vecs.shape, n)
    index = ensure_index(get_client(), common.PINECONE_INDEX)
    done = json.loads(PROGRESS.read_text())["done"] if PROGRESS.exists() else 0

    def batches(start):
        for pos in range(start, n, BATCH):
            recs = [{"id": c["chunk_id"], "values": vecs[pos + k].tolist(),
                     "metadata": to_metadata_v2(c["chunk_id"], c["set_id"], c["drug"], c["route"], c["section"])}
                    for k, c in enumerate(chunks[pos:pos + BATCH])]
            yield pos, recs

    t0 = time.time()
    with ThreadPoolExecutor(THREADS) as pool, tqdm(total=n, initial=done, desc="upsert v2") as bar:
        it = batches(done)
        while True:
            window = [(pos, len(r), pool.submit(upsert_with_retries, index, r, common.PINECONE_NAMESPACE_V2))
                      for pos, r in itertools.islice(it, THREADS)]
            if not window:
                break
            for pos, size, fut in window:
                fut.result()
                bar.update(size)
            PROGRESS.write_text(json.dumps({"done": window[-1][0] + window[-1][1]}))
    print(f"upsert finished in {time.time() - t0:.0f}s")
    deadline = time.time() + 180
    while time.time() < deadline:
        got = index.describe_index_stats().namespaces.get(common.PINECONE_NAMESPACE_V2, {}).get("vector_count", 0)
        if got >= n:
            break
        time.sleep(3)
    print(f"namespace {common.PINECONE_NAMESPACE_V2!r}: {got} vectors (expected {n})")


if __name__ == "__main__":
    main()
