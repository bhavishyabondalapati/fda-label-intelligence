"""Embed every chunking variant locally (free) and cache vectors for the ablation.

Writes data/processed/ablate_<chunker>_<hdr|nohdr>.npy (git-ignored). Resumable per file.
naive+nohdr already exists from Phase 1 (phase1_vectors.f32) and is reused.
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.chunk.variants import embed_text  # noqa: E402
from src.embed.local import embed_texts, load_model  # noqa: E402
from src.eval.variants_data import load_chunks  # noqa: E402
from src.services import common  # noqa: E402

VARIANTS = [("naive", True), ("section", False), ("section", True)]


def main():
    model = load_model()
    for chunker, header in VARIANTS:
        path = common.DATA_DIR / f"ablate_{chunker}_{'hdr' if header else 'nohdr'}.npy"
        if path.exists():
            print("have", path.name)
            continue
        chunks = load_chunks(common.DATA_DIR, chunker)
        texts = [embed_text(c, header) for c in chunks]
        # sort by length so batches pad little (much faster); restore order afterwards
        order = np.argsort([len(t) for t in texts])
        t0 = time.time()
        vecs = np.empty((len(texts), 384), dtype="float32")
        vecs[order] = embed_texts(model, [texts[i] for i in order], batch_size=128)
        np.save(path, vecs)
        print(f"{path.name}: {len(texts)} chunks in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
