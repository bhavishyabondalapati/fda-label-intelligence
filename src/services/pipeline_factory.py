"""Builds the Phase 2 Pipeline for the query service from the config that the ablation chose
(data/processed/pipeline_v2.json). Slow parts (BM25 build ~1 min) happen once at startup."""
import json
import pickle
import time

from src.chunk.variants import embed_text
from src.eval.score import read_jsonl
from src.services import common


def build_pipeline_v2(embed_fn, namespace):
    from src.retrieve.bm25 import BM25
    from src.retrieve.drug_detect import DrugDetector
    from src.retrieve.pinecone_store import get_client, open_index
    from src.retrieve.pipeline import Pipeline
    from src.retrieve.rerank import load_reranker
    from src.store.chunk_store import ChunkStore

    cfg = json.loads(common.PIPELINE_CONFIG.read_text())
    store = ChunkStore(common.CHUNK_DB_V2)
    need_rows = cfg["filter"] or cfg["hybrid"]
    rows = list(store.iter_rows_in_order()) if need_rows else None
    bm25 = None
    if cfg["hybrid"]:
        cache = common.DATA_DIR / "bm25_v2.pkl"
        if cache.exists():
            bm25 = pickle.loads(cache.read_bytes())
        else:
            t0 = time.time()
            bm25 = BM25([embed_text(r, cfg["header"]) for r in rows])
            cache.write_bytes(pickle.dumps(bm25))
            print(f"BM25 built in {time.time() - t0:.0f}s")
    detector = DrugDetector(read_jsonl(common.DATA_DIR / "labels.jsonl")) if cfg["filter"] else None
    reranker = load_reranker() if cfg["rerank"] else None
    index = open_index(get_client(), common.PINECONE_INDEX)
    return Pipeline(embed_fn, index, namespace, store, header=cfg["header"], use_filter=cfg["filter"],
                    use_hybrid=cfg["hybrid"], use_rerank=cfg["rerank"], detector=detector, bm25=bm25,
                    reranker=reranker, rows=rows)
